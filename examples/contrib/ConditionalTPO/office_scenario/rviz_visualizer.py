"""
RViz2 Tour Visualizer for Office Scenario

Writes tour data to JSON, then spawns rviz_publisher.py (using the ros2
env Python) as a subprocess. The two envs never share a process.
"""

import copy
import json
import os
import subprocess
import sys

import networkx as nx

ROS2_PYTHON = os.path.expanduser("~/miniconda3/envs/ros2/bin/python")
ROS2_LIB    = os.path.expanduser("~/miniconda3/envs/ros2/lib")
ROS2_PREFIX = os.path.expanduser("~/miniconda3/envs/ros2")
ROS2_PYPATH = os.path.expanduser("~/miniconda3/envs/ros2/lib/python3.11/site-packages")


def _extract_dts_path(tour_nodes, mapping, transition_system):
    skipped_states = set()
    for node_id, states in mapping.node_to_states.items():
        if node_id not in tour_nodes:
            skipped_states.update(states)

    G = copy.deepcopy(transition_system)
    G.remove_nodes_from(skipped_states)
    for src, tgt in G.edges():
        G[src][tgt][0]["weight"] = 1

    full_path = []
    current_dts_state = None

    for seg_idx in range(len(tour_nodes)):
        current_tsp_node = tour_nodes[seg_idx]

        if seg_idx < len(tour_nodes) - 1:
            next_tsp_node = tour_nodes[seg_idx + 1]
            current_states = (
                [current_dts_state] if current_dts_state is not None
                else mapping.node_to_states[current_tsp_node]
            )
            next_states = mapping.node_to_states[next_tsp_node]

            best_path, best_dist, best_next = None, float("inf"), None
            for cs in current_states:
                for ns in next_states:
                    if cs == ns:
                        continue
                    try:
                        d, p = nx.single_source_dijkstra(G, source=cs, target=ns, weight="weight")
                        if d < best_dist:
                            best_dist, best_path, best_next = d, p, ns
                    except nx.NetworkXNoPath:
                        pass

            if best_path is None:
                src0 = current_dts_state or mapping.node_to_states[current_tsp_node][0]
                best_path = nx.shortest_path(transition_system, source=src0, target=next_states[0])
                best_next = next_states[0]

            full_path.extend(best_path)
            current_dts_state = best_next
        else:
            state = current_dts_state or mapping.node_to_states[current_tsp_node][0]
            if not full_path or full_path[-1] != state:
                full_path.append(state)

    return full_path


def visualize_tour_rviz(
    tour_nodes,
    mapping,
    transition_system,
    event_labels=None,
    step_delay=0.15,
    json_path="visualization/tour_data.json",
    publisher_script=None,
):
    """Write tour JSON and spawn rviz_publisher.py (or a custom script) using the ros2 Python."""

    print("\n[RViz2] Extracting DTS path...")
    dts_path = _extract_dts_path(tour_nodes, mapping, transition_system)
    print(f"[RViz2] Path has {len(dts_path)} states.")

    # Build event node list for markers — always include all known locations,
    # even if not visited in this tour (e.g. carpet in scenario 1)
    all_obs_in_tour = {mapping.node_to_obs.get(n) for n in tour_nodes}
    event_nodes = []
    for node_id in tour_nodes:
        obs = mapping.node_to_obs.get(node_id, "")
        states = mapping.node_to_states.get(node_id, [])
        if not states:
            continue
        x, y = float(states[0][0]), float(states[0][1])
        label = (event_labels or {}).get(node_id, obs)
        event_nodes.append({"obs": obs, "x": x, "y": y, "label": label})

    # Add carpet marker even when not in the tour
    if "floor_grey" not in all_obs_in_tour:
        carpet_node = mapping.obs_to_node.get("floor_grey")
        if carpet_node is not None:
            states = mapping.node_to_states.get(carpet_node, [])
            if states:
                x, y = float(states[0][0]), float(states[0][1])
                label = (event_labels or {}).get(carpet_node, "floor_grey")
                event_nodes.append({"obs": "floor_grey", "x": x, "y": y, "label": label})

    # Collect puddle tile positions from the transition system for environment rendering
    puddle_positions = []
    for node_id, data in transition_system.nodes(data=True):
        obs = data.get("observation", "")
        if obs == "floor_blue":
            if hasattr(node_id, '__iter__') and not isinstance(node_id, str):
                puddle_positions.append([float(node_id[0]), float(node_id[1])])

    tour_data = {
        "path": [[float(s[0]), float(s[1])] for s in dts_path],
        "event_nodes": event_nodes,
        "puddle_positions": puddle_positions,
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(tour_data, f)
    print(f"[RViz2] Tour data written to {json_path}")

    # Locate publisher script next to this file (default: rviz_publisher.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if publisher_script is None:
        publisher_script = os.path.join(script_dir, "rviz_publisher.py")
    elif not os.path.isabs(publisher_script):
        publisher_script = os.path.join(script_dir, publisher_script)

    cyclone_cfg = os.path.join(script_dir, "cyclonedds.xml")

    env = os.environ.copy()
    env["AMENT_PREFIX_PATH"]          = ROS2_PREFIX
    env["PYTHONPATH"]                 = ROS2_PYPATH
    env["RMW_IMPLEMENTATION"]         = "rmw_cyclonedds_cpp"
    env["CYCLONEDDS_URI"]             = f"file://{cyclone_cfg}"
    env["DYLD_FALLBACK_LIBRARY_PATH"] = ROS2_LIB + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")

    rviz_config = os.path.join(script_dir, "visualization", "office.rviz")
    print(f"[RViz2] Launching (ros2 Python + rviz2). Close RViz2 window or Ctrl-C to stop.\n")
    try:
        subprocess.run(
            [ROS2_PYTHON, publisher_script, json_path, rviz_config, str(step_delay)],
            env=env,
        )
    except KeyboardInterrupt:
        pass
