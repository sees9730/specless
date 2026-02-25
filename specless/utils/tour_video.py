"""
Tour Video Generation

Generates a video of a robot executing a planned tour through a MiniGrid
environment, rendering the actual DTS path between consecutive TSP nodes.
"""

import copy
from typing import List

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from specless.utils.conditional_tsp_mapper import ConditionalTSPMapper


def generate_video_from_tour(
    env,
    tour_nodes: List[int],
    mapping: ConditionalTSPMapper,
    transition_system,
    filename: str = "tour_video.mp4",
    fps: int = 2,
):
    """
    Generate a video of the agent executing a planned tour.

    Renders the actual DTS state path between consecutive TSP nodes,
    avoiding intermediate event-labeled states.

    Args:
        env: MiniGrid environment with render_mode="rgb_array"
        tour_nodes: Ordered list of TSP node IDs from the planner
        mapping: ConditionalTSPMapper linking TSP nodes to DTS states
        transition_system: The TransitionSystem used for path computation
        filename: Output video file path
        fps: Frames per second
    """
    print(f"\n Generating video: {filename}")

    frames = []
    total_dts_steps = 0
    current_dts_state = None

    env.reset()

    # Uniform-weight copy for shortest-path queries.
    # Remove states belonging to TSP nodes not in the tour so the rendered
    # path cannot accidentally pass through skipped event tiles.
    skipped_states: set = set()
    for node_id, states in mapping.node_to_states.items():
        if node_id not in tour_nodes:
            skipped_states.update(states)

    G = copy.deepcopy(transition_system)
    G.remove_nodes_from(skipped_states)
    for src, tgt in G.edges():
        G[src][tgt][0]['weight'] = 1

    for seg_idx in range(len(tour_nodes)):
        current_tsp_node = tour_nodes[seg_idx]

        if seg_idx < len(tour_nodes) - 1:
            next_tsp_node = tour_nodes[seg_idx + 1]

            current_states = (
                [current_dts_state] if current_dts_state is not None
                else mapping.node_to_states[current_tsp_node]
            )
            next_states = mapping.node_to_states[next_tsp_node]

            best_path = None
            best_distance = float('inf')
            best_next_state = None

            for current_state in current_states:
                for next_state in next_states:
                    if current_state == next_state:
                        continue
                    try:
                        distance, path = nx.single_source_dijkstra(
                            G, source=current_state, target=next_state, weight='weight'
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_path = path
                            best_next_state = next_state
                    except nx.NetworkXNoPath:
                        pass

            if best_path is None:
                fallback_src = current_dts_state or mapping.node_to_states[current_tsp_node][0]
                fallback_tgt = next_states[0]
                best_path = nx.shortest_path(transition_system, source=fallback_src, target=fallback_tgt)
                best_next_state = fallback_tgt

            for dts_state in best_path:
                base_env = env.unwrapped
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                base_env.agent_pos = np.array(dts_state[:2])
                base_env.agent_dir = 0
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    frames.append(frame)
                    total_dts_steps += 1

            current_dts_state = best_next_state

        else:
            if current_dts_state is not None:
                base_env = env.unwrapped
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                base_env.agent_pos = np.array(current_dts_state[:2])
                base_env.agent_dir = 0
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    frames.append(frame)
                    total_dts_steps += 1

    if frames:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"  Video saved: {filename} ({total_dts_steps} frames, {total_dts_steps / fps:.1f}s)")
    else:
        print(f"  No frames captured (render_mode={env.render_mode})")


def generate_path_image(
    env,
    tour_nodes: List[int],
    mapping: ConditionalTSPMapper,
    transition_system,
    filename: str = "tour_path.pdf",
    event_labels: dict = None,
):
    """
    Generate a static PDF image of the agent's planned path overlaid on the grid.

    Renders the environment once, then draws the full DTS path as a colored
    line with markers for each visited state and stars for TSP event nodes.

    Args:
        env: MiniGrid environment with render_mode="rgb_array"
        tour_nodes: Ordered list of TSP node IDs from the planner
        mapping: ConditionalTSPMapper linking TSP nodes to DTS states
        transition_system: The TransitionSystem used for path computation
        filename: Output PDF file path
    """
    print(f"\n Generating path image: {filename}")

    # Build the same restricted graph as generate_video_from_tour
    skipped_states: set = set()
    for node_id, states in mapping.node_to_states.items():
        if node_id not in tour_nodes:
            skipped_states.update(states)

    G = copy.deepcopy(transition_system)
    G.remove_nodes_from(skipped_states)
    for src, tgt in G.edges():
        G[src][tgt][0]['weight'] = 1

    # Collect the full DTS path
    full_path = []
    event_positions = []  # (x, y, label) for TSP nodes
    current_dts_state = None

    env.reset()

    for seg_idx in range(len(tour_nodes)):
        current_tsp_node = tour_nodes[seg_idx]

        if seg_idx < len(tour_nodes) - 1:
            next_tsp_node = tour_nodes[seg_idx + 1]

            current_states = (
                [current_dts_state] if current_dts_state is not None
                else mapping.node_to_states[current_tsp_node]
            )
            next_states = mapping.node_to_states[next_tsp_node]

            best_path = None
            best_distance = float('inf')
            best_next_state = None

            for cs in current_states:
                for ns in next_states:
                    if cs == ns:
                        continue
                    try:
                        distance, path = nx.single_source_dijkstra(
                            G, source=cs, target=ns, weight='weight'
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_path = path
                            best_next_state = ns
                    except nx.NetworkXNoPath:
                        pass

            if best_path is None:
                fallback_src = current_dts_state or mapping.node_to_states[current_tsp_node][0]
                fallback_tgt = next_states[0]
                best_path = nx.shortest_path(transition_system, source=fallback_src, target=fallback_tgt)
                best_next_state = fallback_tgt

            # Record event node position at the start of this segment
            src_state = (current_dts_state or mapping.node_to_states[current_tsp_node][0])
            if event_labels and current_tsp_node in event_labels:
                label = event_labels[current_tsp_node]
            else:
                label = mapping.node_to_obs.get(current_tsp_node, str(current_tsp_node))
            event_positions.append((src_state[0], src_state[1], label, current_tsp_node))

            full_path.extend(best_path)
            current_dts_state = best_next_state
        else:
            # Last node in the tour
            state = current_dts_state or mapping.node_to_states[current_tsp_node][0]
            if event_labels and current_tsp_node in event_labels:
                label = event_labels[current_tsp_node]
            else:
                label = mapping.node_to_obs.get(current_tsp_node, str(current_tsp_node))
            event_positions.append((state[0], state[1], label, current_tsp_node))
            if current_dts_state is not None:
                full_path.append(current_dts_state)

    # Render base grid image (agent parked off-screen via reset)
    frame = env.render()

    # Compute pixel coordinates from grid coordinates
    # MiniGrid renders each cell as TILE_PIXELS x TILE_PIXELS pixels
    try:
        from minigrid.core.constants import TILE_PIXELS
    except ImportError:
        TILE_PIXELS = 32

    def grid_to_pixel(gx, gy):
        px = (gx + 0.5) * TILE_PIXELS
        py = (gy + 0.5) * TILE_PIXELS
        return px, py

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame)
    ax.axis("off")

    # Draw path line
    if full_path:
        xs = [(s[0] + 0.5) * TILE_PIXELS for s in full_path]
        ys = [(s[1] + 0.5) * TILE_PIXELS for s in full_path]
        ax.plot(xs, ys, color="white", linewidth=2, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, color="white", s=10, zorder=3, alpha=0.6)

    # Colors matching the actual grid tile colors
    _obs_color_map = {
        "initial_state0": "#888888",
        "floor_green":    "#00aa00",
        "floor_red":      "#cc0000",
        "floor_purple":   "#8800cc",
        "floor_blue":     "#0055cc",
        "floor_yellow":   "#ccaa00",
        "floor_grey":     "#666666",
        "goal_green":     "#00cc44",
    }
    _fallback_colors = plt.get_cmap("tab10")
    _fallback_idx = 0

    for idx, (gx, gy, label, node_id) in enumerate(event_positions):
        px, py = grid_to_pixel(gx, gy)
        obs_name = mapping.node_to_obs.get(node_id, "")
        if obs_name in _obs_color_map:
            color = _obs_color_map[obs_name]
        else:
            color = _fallback_colors(_fallback_idx % 10)
            _fallback_idx += 1
        ax.scatter(px, py, color=color, s=120, zorder=4, marker="*",
                   edgecolors="black", linewidths=0.5)
        ax.annotate(f"{label}", xy=(px, py),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.7))

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Path image saved: {filename}")
