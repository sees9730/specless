import networkx as nx
import specless as sl
from IPython.utils.io import capture_output

from specless.specification.conditional_tpo import ConditionalTPO
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.utils.state_regions import StateRegions
from specless.utils.tpo_utils import create_precedence_edges, get_tpo_nodes, has_precedence_path


def build_tpo_and_tsp(nodes, costs, local_constraints):
    tpo = TimedPartialOrder.from_constraints(
        global_constraints={},
        local_constraints=local_constraints,
    )

    tpo_nodes = get_tpo_nodes(tpo)

    precedence_edges = create_precedence_edges(nodes, tpo)

    initial_nodes = [
        n for n in nodes
        if not any(has_precedence_path(tpo, o, n) for o in nodes if o != n)
    ]
    final_nodes = [
        n for n in nodes
        if not any(has_precedence_path(tpo, n, o) for o in nodes if o != n)
    ]

    allowed_edges = list(precedence_edges)
    for fn in final_nodes:
        for in_ in initial_nodes:
            if fn != in_ and (fn, in_) not in allowed_edges:
                allowed_edges.append((fn, in_))

    tsp = sl.TSPWithTPO(nodes, costs, tpo)
    tsp.edges = allowed_edges
    tsp.nodesets = [[n] for n in tpo_nodes]

    print(f"  TPO nodes: {tpo_nodes}")
    print(f"  TSP edges: {len(tsp.edges)}")
    print(f"  Nodesets:  {tsp.nodesets}")

    return tpo, tsp


def define_regions():
    regions = StateRegions()
    regions.add_region("kitchen", {1, 2, 3})
    regions.add_region("hallway", {4, 5, 6})
    regions.add_region("bedroom", {7, 8, 9})

    regions.print_summary()

    return regions


def set_edge_cost(costs, src, tgt, value):
    costs[src][tgt] = value


def solve_with_region_tracking(tpo, tsp, regions):
    ctpo = ConditionalTPO(base_tpo=tpo, regions=regions)

    tsp_graph = nx.DiGraph()
    tsp_graph.add_edges_from(tsp.edges)
    with capture_output():
        sl.draw_graph(tpo, "visualization/region_tracking_tpo")
        sl.draw_graph(tsp_graph, "visualization/region_tracking_tsp")

    solver = sl.MILPTSPWithRegionTracking()
    tours, cost, timestamps = solver.solve(
        tsp,
        num_agent=1,
        init_nodes=[0],
        come_back_home=False,
        regions=regions,
    )

    print(f"  Tour:       {tours[0]}")
    print(f"  Cost:       {cost:.2f}")
    print(f"  Timestamps: {[f'{t:.2f}' for t in timestamps[0]]}")

    solver.print_visited_regions()

    return solver, tours, cost, timestamps


def validate_region_tracking(solver, tours, regions):
    tour_nodes = set(tours[0])
    visited = solver.get_visited_regions()
    passed = True

    for name, node_set in regions.regions.items():
        expected = bool(tour_nodes & node_set)
        actual = visited[name]
        marker = "" if expected == actual else "FAIL"
        print(f"  {marker} {name}: expected={expected}, actual={actual}")
        if expected != actual:
            passed = False

    return passed
