"""
Mars Rover Atmospheric Science Mission — Conditional TPO Scenario

Setting:
    A Mars rover must collect atmospheric measurements at two science sites
    before returning to its lander.  Two gridworld variants are provided:

    MiniGrid-MarsRover-Direct-v0:
        No outcrop on any navigable path.  The rover completes only the base
        mission (e1–e5) regardless of route chosen.

    MiniGrid-MarsRover-Outcrop-v0:
        The geological outcrop (e6, e7) sits in the lower corridor between
        Site B and the Lander.  Any rover routing from Site B to the Lander
        via the lower path passes through the outcrop, activating the
        conditional TPO automatically — no cost manipulation required.

Observation → Event mapping:
    initial_state0  depot   Starting position
    floor_green     e1      Arrive at science zone
    floor_red       e2      Soil heating at site A
    floor_purple    e3      Outgassing measurement at site A
    floor_blue      e4      Atmospheric reading at site B
    floor_yellow    e5      Return to lander
    floor_grey      e6      Navigate to geological outcrop  [conditional]
    goal_green      e7      Collect rock sample at outcrop  [conditional]

Base TPO (always active):
    e1 → e2 → e3,  e1 → e4,  e3 → e5,  e4 → e5
    e1 → e2: [0, 15],  e2 → e3: [0, 10]
    e1 → e4: [0, 25],  e3 → e5: [0, 30],  e4 → e5: [0, 30]
    e5 global: [0, 50]  (nightfall deadline)

Conditional TPO (IF rover enters outcrop_region):
    e6 → e7: [0, 30]
"""

import networkx as nx
import numpy as np
import gymnasium as gym
import specless as sl
from IPython.utils.io import capture_output
from minigrid.core.world_object import fill_coords, point_in_circle
from minigrid.utils.rendering import downsample
from minigrid.core.grid import Grid
from minigrid.core.constants import TILE_PIXELS
from gym_minigrid.minigrid import Grid as GymMinigridGrid

# Register both environments
from specless.minigrid.marsroverenv import MarsRoverEnvDirect, MarsRoverEnvOutcrop  # noqa: F401
from specless.utils.conditional_tsp_mapper import ConditionalTSPMapper
from specless.utils.state_regions import StateRegions
from specless.utils.tpo_utils import create_precedence_edges, has_precedence_path
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.specification.conditional_tpo import ConditionalTPO


# ---------------------------------------------------------------------------
# Monkey-patch: render rover as a circle instead of a triangle
# ---------------------------------------------------------------------------
@classmethod
def _patched_render_tile(cls, obj, agent_dir=None, highlight=False,
                         tile_size=TILE_PIXELS, subdivs=3):
    key = (agent_dir, highlight, tile_size)
    key = obj.encode() + key if obj else key
    if key in cls.tile_cache:
        return cls.tile_cache[key]
    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
    fill_coords(img, point_in_circle(0.031, 0.031, 0.031), (100, 100, 100))
    if obj is not None:
        obj.render(img)
    if agent_dir is not None:
        fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.31), (255, 0, 0))
    img = downsample(img, subdivs)
    cls.tile_cache[key] = img
    return img


_original_render_tile = Grid.render_tile
Grid.render_tile = _patched_render_tile
GymMinigridGrid.render_tile = _patched_render_tile


# ---------------------------------------------------------------------------
# 1. Environments
# ---------------------------------------------------------------------------
def create_mars_environment(env_id, render_mode=None):
    """Create and wrap a Mars Rover environment by ID.

    Args:
        env_id: Either 'MiniGrid-MarsRover-Direct-v0' or
                'MiniGrid-MarsRover-Outcrop-v0'
        render_mode: e.g. 'rgb_array' for video generation

    Returns:
        (env, transition_system)
    """
    env = gym.make(env_id, render_mode=render_mode, agent_start_pos=(4, 5))
    env = sl.LabelMiniGridWrapper(env, labelkey="color", skiplist=["empty", "wall"])
    env = sl.MiniGridTransitionSystemWrapper(env, ignore_direction=True)

    tsbuilder = sl.TSBuilder()
    transition_system = tsbuilder(env)

    print(f" Transition System built ({env_id}):")
    print(f"   States: {len(transition_system.nodes)}")
    print(f"   Edges:  {len(transition_system.edges)}")

    return env, transition_system


# ---------------------------------------------------------------------------
# 2. Events & mapping
# ---------------------------------------------------------------------------
def define_events_and_mapping(transition_system):
    """Map transition-system states to TSP nodes.

    Mandatory events (e1–e5) are always in the tour.
    Optional events (e6, e7) appear only when the rover enters outcrop_region.
    Missing optional observations (Direct env has no outcrop tiles) are
    silently ignored — the mapper only registers what is present in the TS.

    Returns:
        ConditionalTSPMapper
    """
    mandatory_events = [
        "initial_state0",
        "floor_green",     # e1
        "floor_red",       # e2
        "floor_purple",    # e3
        "floor_blue",      # e4
        "floor_yellow",    # e5
    ]

    # Only register optional events that actually exist in the transition system
    all_obs = set(
        d.get("observation", "")
        for _, d in transition_system.nodes(data=True)
    )
    optional_events = [o for o in ["floor_grey", "goal_green"] if o in all_obs]

    print(f" Mandatory events: {mandatory_events}")
    print(f" Optional events:  {optional_events}")

    # lava_red tiles mark the outcrop boundary (AP label p_o); they get a TSP
    # node for edge cost/region tracking but no visit-count constraint (d_i=0).
    ap_events = [o for o in ["lava_red"] if o in all_obs]
    print(f" AP waypoint events: {ap_events}")

    mapping = ConditionalTSPMapper(
        transition_system,
        mandatory_events=mandatory_events,
        optional_events=optional_events,
        ap_events=ap_events,
        ignoring_obs_keys=["empty", "wall"],
    )

    mapping.print_summary()
    return mapping


# ---------------------------------------------------------------------------
# 3. Regions
# ---------------------------------------------------------------------------
def define_regions(mapping):
    """Define R_o (geological outcrop) if its tiles exist in the mapping.

    Returns:
        StateRegions
    """
    regions = StateRegions()

    present = set(mapping.obs_to_states.keys())

    if "lava_red" in present:
        # lava_red is an AP waypoint node → to_tsp_regions() finds it directly.
        # floor_grey/goal_green are optional event nodes that already appear in
        # the tour graph; including them here would add their edges to the
        # touching set but is not needed for correct E_k construction.
        regions.add_region_by_observation("outcrop_region", "lava_red", mapping)
        regions.print_summary(mapping)
    else:
        print(" No lava tiles in this environment — outcrop_region not defined.")

    return regions


# ---------------------------------------------------------------------------
# 4. Conditional TPOs
# ---------------------------------------------------------------------------
def create_conditional_tpos(mapping, regions):
    """Build the base TPO and (if applicable) the conditional outcrop TPO.

    Returns:
        ConditionalTPO
    """
    n = mapping.obs_to_node
    depot = n["initial_state0"]
    e1    = n["floor_green"]
    e2    = n["floor_red"]
    e3    = n["floor_purple"]
    e4    = n["floor_blue"]
    e5    = n["floor_yellow"]

    base_tpo = TimedPartialOrder.from_constraints(
        global_constraints={},
        local_constraints={
            (e1, e2): (0, np.inf),
            (e2, e3): (0, np.inf),
            (e1, e4): (0, np.inf),
            (e3, e5): (0, np.inf),
            (e4, e5): (0, np.inf),
            (e1, e5): (0, 40)
        },
    )
    # |t[e3] - t[e4]| <= 4: both samples must reflect the same atmospheric state
    base_tpo.add_difference_constraint(e3, e4, 10)

    print(" Base TPO:")
    print(f"   {e1}(e1) → {e2}(e2) [0,15],  {e2}(e2) → {e3}(e3) [0,10]")
    print(f"   {e1}(e1) → {e4}(e4) [0,25]")
    print(f"   (e5 enforced only via cTPO branches)")

    ctpo = ConditionalTPO(base_tpo, regions)

    # Add conditional TPO only if outcrop tiles exist in this environment
    if "outcrop_region" in regions.regions and \
       "floor_grey" in n and "goal_green" in n:
        e6 = n["floor_grey"]
        e7 = n["goal_green"]

        # cTPO 1: IF outcrop visited → e6 → e7 → e5
        outcrop_tpo = TimedPartialOrder.from_constraints(
            global_constraints={},
            local_constraints={
                (e1, e6): (0, np.inf),
                (e6, e7): (0, 10),  # navigate before sampling
                (e7, e5): (0, np.inf),  # must return to lander after sampling
                (e1, e5): (0, 40)
            },
        )
        print(f"\n cTPO 1 — 'outcrop_region' (IF outcrop visited):")
        print(f"   {e6}(e6) → {e7}(e7) [0,30]")
        print(f"   {e7}(e7) → {e5}(e5) [0,30]")
        ctpo.add_conditional_tpo("outcrop_region", outcrop_tpo)

        # cTPO 2: IF outcrop NOT visited → e5 deadline still required
        no_outcrop_tpo = TimedPartialOrder.from_constraints(
            global_constraints={},
            local_constraints={
                (e1, e5): (0, 40)
            },
        )
        print(f"\n cTPO 2 — 'outcrop_region' negated (IF outcrop NOT visited):")
        print(f"   {e5}(e5) global [0,50]")
        ctpo.add_conditional_tpo("outcrop_region", no_outcrop_tpo, negate=True)
    else:
        print("\n No outcrop region — conditional TPOs not added.")

    with capture_output():
        sl.draw_graph(base_tpo, "visualization/base_TPO")
        if "outcrop_region" in regions.regions and \
           "floor_grey" in n and "goal_green" in n:
            sl.draw_graph(outcrop_tpo, "visualization/conditional_TPO_outcrop")
            sl.draw_graph(no_outcrop_tpo, "visualization/conditional_TPO_no_outcrop")

    print(f"\n Saved TPO visualizations to 'visualization/' directory.")
    ctpo.print_summary()
    return ctpo


# ---------------------------------------------------------------------------
# 5. Unified TPO + TSP graph
# ---------------------------------------------------------------------------
def build_tpo_and_tsp(mapping, ctpo):
    """Merge all conditional TPOs and build the TSP precedence graph.

    Returns:
        (unified_tpo, base_tpo, edges, tsp_nodes)
    """
    unified_tpo = ctpo.build_unified_tpo()
    tsp_nodes   = mapping.get_tsp_nodes()

    precedence_edges = create_precedence_edges(tsp_nodes, unified_tpo)

    initial_nodes = [
        nd for nd in tsp_nodes
        if not any(has_precedence_path(unified_tpo, o, nd)
                   for o in tsp_nodes if o != nd)
    ]
    final_nodes = [
        nd for nd in tsp_nodes
        if not any(has_precedence_path(unified_tpo, nd, o)
                   for o in tsp_nodes if o != nd)
    ]

    edges = list(precedence_edges)
    for fn in final_nodes:
        for in_ in initial_nodes:
            if fn != in_ and (fn, in_) not in edges:
                edges.append((fn, in_))

    # AP waypoint nodes have no precedence constraints, so they don't appear in
    # precedence_edges.  Add bidirectional edges between every AP node and every
    # other TSP node so the solver can route through them freely.
    ap_nodes = mapping.get_ap_nodes()
    for ap in ap_nodes:
        for nd in tsp_nodes:
            if nd != ap:
                if (ap, nd) not in edges:
                    edges.append((ap, nd))
                if (nd, ap) not in edges:
                    edges.append((nd, ap))

    print(f" Unified TPO nodes: {tsp_nodes}")
    print(f" Total TSP edges:   {len(edges)}")

    tsp_graph = nx.DiGraph()
    tsp_graph.add_edges_from(edges)
    with capture_output():
        sl.draw_graph(unified_tpo, "visualization/unified_TPO")
        sl.draw_graph(tsp_graph,   "visualization/TSP_graph")

    return unified_tpo, ctpo.base_tpo, edges, tsp_nodes


# ---------------------------------------------------------------------------
# 6. Solve
# ---------------------------------------------------------------------------
def _sp_cost(transition_system, states_a, states_b):
    """Shortest-path distance (TS steps) between two sets of TS states."""
    best = float('inf')
    for sa in states_a:
        for sb in states_b:
            if sa == sb:
                continue
            try:
                d = nx.shortest_path_length(transition_system, source=sa, target=sb)
                if d < best:
                    best = d
            except nx.NetworkXNoPath:
                pass
    return float(best) if best < float('inf') else 1e6


def build_cost_matrix(mapping, transition_system, tsp_nodes):
    """Cost matrix from real grid shortest-path distances.

    costs[i][j] = minimum TS steps from any state in node i to any state
    in node j.  The solver minimises total time (sum of costs along the
    chosen tour), so it naturally avoids long detours and takes short ones.
    """
    max_node = max(tsp_nodes) + 1
    costs = [[0.0] * max_node for _ in range(max_node)]
    for i in tsp_nodes:
        for j in tsp_nodes:
            if i != j:
                costs[i][j] = _sp_cost(
                    transition_system,
                    mapping.node_to_states[i],
                    mapping.node_to_states[j],
                )
    return costs


def solve(mapping, transition_system, unified_tpo, base_tpo, edges, tsp_nodes, ctpo):
    """Solve the TSP-with-TPO using real grid shortest-path distances as costs.

    Costs come from actual grid distances so the solver naturally prefers
    shorter routes.  The two DTS environments encode the outcrop placement:
    - Direct env: outcrop is far away → high cost → solver skips it (z=0)
    - Outcrop env: outcrop is nearby  → low cost  → solver visits it (z=1)

    No manual cost manipulation is needed; the conditional TPO structure
    (both positive and negated branches) drives the binary visit/skip choice.

    Returns:
        (solver, tours, cost, timestamps)
    """
    n = mapping.obs_to_node
    depot = n["initial_state0"]

    # Human-readable labels: event obs → short name, AP synthetic obs → lava_red_#
    _event_labels = {
        "initial_state0": "depot",
        "floor_green":    "e1",
        "floor_red":      "e2",
        "floor_purple":   "e3",
        "floor_blue":     "e4",
        "floor_yellow":   "e5",
        "floor_grey":     "e6",
        "goal_green":     "e7",
    }
    # Count per AP observation prefix for numbering
    _ap_counters = {}
    _node_label = {}
    for nd in tsp_nodes:
        obs = mapping.node_to_obs[nd]
        if mapping.is_node_ap_waypoint(nd):
            # synthetic obs is e.g. "lava_red_0" — strip the index and re-number
            prefix = "_".join(obs.split("_")[:-1])  # e.g. "lava_red"
            idx = _ap_counters.get(prefix, 0)
            _ap_counters[prefix] = idx + 1
            _node_label[nd] = f"{prefix}_{idx}"
        else:
            _node_label[nd] = _event_labels.get(obs, obs)

    def lbl(nd):
        return _node_label[nd]

    print(" Computing shortest-path cost matrix...")
    costs = build_cost_matrix(mapping, transition_system, tsp_nodes)

    print("  Full cost matrix:")
    for i in tsp_nodes:
        for j in tsp_nodes:
            if i != j and costs[i][j] > 0:
                print(f"    {lbl(i)} → {lbl(j)}: {costs[i][j]:.0f}")

    if "floor_grey" in n and "goal_green" in n:
        e6, e7 = n["floor_grey"], n["goal_green"]
        e5 = n["floor_yellow"]
        print(f"  cost(e6→e7)={costs[e6][e7]:.0f}  cost(e4→e5)={costs[n['floor_blue']][e5]:.0f}  cost(e7→e5)={costs[e7][e5]:.0f}")

    tsp = sl.TSPWithTPO(tsp_nodes, costs, base_tpo)
    tsp.edges    = edges

    # print(f"  TSP edges ({len(edges)}):")
    # for (i, j) in edges:
    #     print(f"    {lbl(i)} → {lbl(j)}: {costs[i][j]:.0f}")
    # AP waypoint nodes must NOT be in nodesets (no visit-count == K constraint).
    non_ap_nodes = mapping.get_mandatory_nodes() + mapping.get_optional_nodes()
    tsp.nodesets = [[nd] for nd in non_ap_nodes]

    solver = sl.MILPTSPWithRegionTracking(mapping)
    tours, cost, timestamps = solver.solve(
        tsp,
        ctpo=ctpo,
        num_agent=1,
        init_nodes=[depot],
        come_back_home=False,
        export_filename='debug.lp'
    )

    print(f" Tour:   {tours[0]}")
    print(f" Cost:   {cost:.2f}")
    print(f" Events: {[mapping.node_to_obs[nd] for nd in tours[0]]}")
    print(f" Timestamps: {[f'{v:.1f}' for v in timestamps[0]]}")

    solver.print_visited_regions()
    return solver, tours, cost, timestamps
