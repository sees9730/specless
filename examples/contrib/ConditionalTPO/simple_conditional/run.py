"""
Conditional TPO — all simple examples in one file.

Usage
-----
  python run.py --condition if-visited          # e3 only if e2 visited
  python run.py --condition if-not-visited      # e3 only if e2 NOT visited
  python run.py --condition and                 # e4 only if e2 AND e3 visited
  python run.py --condition or                  # e4 if e2 OR e3 visited
  python run.py --condition and-e2-optional     # AND, but e2 is optional (skipped)
  python run.py --condition or-e2-optional      # OR,  but e2 is optional (still visits e4)
  python run.py --condition ap-waypoint         # e4 triggered by passing thru AP node
"""

import argparse
import gurobipy as gp
from gurobipy import GRB
from viz import plot_ctpo, plot_tsp

# ---------------------------------------------------------------------------
# MILP helpers
# ---------------------------------------------------------------------------

def _new_model(nodes, depot, costs, edges, mandatory, optional):
    """Base TSP model: flow conservation + time propagation."""
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    m = gp.Model(env=env)

    x  = m.addVars(edges, vtype=GRB.BINARY, name="x")
    t  = m.addVars(nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
    tT = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tT")
    tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tf")

    for nd in mandatory:
        m.addConstr(gp.quicksum(x[i, nd] for i in nodes if i != nd) == 1, f"in_{nd}")
        m.addConstr(gp.quicksum(x[nd, j] for j in nodes if j != nd) == 1, f"out_{nd}")

    for nd in optional:
        m.addConstr(gp.quicksum(x[i, nd] for i in nodes if i != nd) <= 1, f"in_{nd}_opt")
        m.addConstr(gp.quicksum(x[nd, j] for j in nodes if j != nd) <= 1, f"out_{nd}_opt")

    for nd in nodes:
        m.addConstr(
            gp.quicksum(x[i, nd] for i in nodes if i != nd) ==
            gp.quicksum(x[nd, j] for j in nodes if j != nd),
            f"flow_{nd}"
        )

    m.addConstr(t[depot] == 0, "t_depot")
    m.addConstrs(
        (x[i, j] == 1) >> (t[j] - t[i] >= costs[i][j])
        for (i, j) in edges if j != depot
    )
    non_depot = [nd for nd in nodes if nd != depot]
    m.addConstrs(
        (x[i, depot] == 1) >> (tT - t[i] >= 0)
        for i in non_depot
    )
    m.addGenConstrMax(tf, [tT], name="tf_max")
    m.setObjective(tf, GRB.MINIMIZE)
    return m, x, t, tf


def _region_z(m, x, nodes, tracked_node):
    """Binary z = 1 iff tracked_node is visited."""
    z = m.addVar(vtype=GRB.BINARY, name=f"z{tracked_node}")
    incoming = [(i, tracked_node) for i in nodes if i != tracked_node]
    for (i, j) in incoming:
        m.addConstr(z >= x[i, j], f"region_lb_{tracked_node}_{i}")
    m.addConstr(z <= gp.quicksum(x[i, j] for (i, j) in incoming), f"region_ub_{tracked_node}")
    return z


def _combine_z(m, z_vars, logic):
    """
    Combine a list of z vars into a single activation variable ze.

    logic : "single"   — ze = z_vars[0]  (no new var needed)
            "single_neg" — ze = 1 - z_vars[0]  (negated)
            "and"      — ze = AND(z_vars)
            "or"       — ze = OR(z_vars)
    """
    if logic in ("single", "single_neg"):
        return z_vars[0]   # caller handles negation via zval
    ze = m.addVar(vtype=GRB.BINARY, name="ze")
    if logic == "and":
        for i, z in enumerate(z_vars):
            m.addConstr(ze <= z, f"and_{i}")
        m.addConstr(ze >= gp.quicksum(z_vars) - (len(z_vars) - 1), "and_lb")
    elif logic == "or":
        for i, z in enumerate(z_vars):
            m.addConstr(ze >= z, f"or_{i}")
        m.addConstr(ze <= gp.quicksum(z_vars), "or_ub")
    return ze


def _add_conditional(m, x, t, nodes, optional_node, trigger_node, ze, zval,
                     cond_src, cond_tgt, cond_lb, cond_ub):
    """
    Add optional-visit + timing constraints for one conditional TPO edge.

      optional_node : TSP node that is visited iff ze == zval
      trigger_node  : not used here (already encoded in ze)
      ze            : activation variable
      zval          : 1 = visit when ze=1, 0 = visit when ze=0 (negated)
      cond_src/tgt  : timing constraint t[tgt] - t[src] in [cond_lb, cond_ub]
    """
    in_opt = gp.quicksum(x[i, optional_node] for i in nodes if i != optional_node)
    m.addConstr((ze == zval)     >> (in_opt == 1), f"ze{zval}_req_{optional_node}")
    m.addConstr((ze == 1-zval)   >> (in_opt == 0), f"ze{1-zval}_skip_{optional_node}")
    m.addConstr((ze == zval) >> (t[cond_tgt] - t[cond_src] >= cond_lb), "cond_lb")
    if cond_ub < float("inf"):
        m.addConstr((ze == zval) >> (t[cond_tgt] - t[cond_src] <= cond_ub), "cond_ub")


def _extract_tour(m, x, nodes, edges, depot):
    if m.status != GRB.OPTIMAL:
        print(f"  No optimal solution (status={m.status})")
        return None
    x_vals = m.getAttr("X", x)
    active = [(i, j) for (i, j) in edges if x_vals[i, j] > 0.5]
    tour, current = [depot], depot
    for _ in range(len(nodes)):
        nxt = next((j for (i, j) in active if i == current), None)
        if nxt is None or nxt == depot:
            break
        tour.append(nxt)
        current = nxt
    tour.append(depot)
    return tour


def _print_tour(tour, labels, t, tf, extra=""):
    print(f"  Tour:       {' -> '.join(labels[n] for n in tour)}")
    print(f"  Cost (tf):  {tf.X:.1f}")
    print(f"  Timestamps: { {labels[n]: round(t[n].X, 1) for n in tour[:-1]} }")
    if extra:
        print(f"  {extra}")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_if_visited():
    """e3 required only if e2 visited. Since e2 mandatory, e3 always visited."""
    nodes  = [0,1,2,3]
    depot  = 0
    costs  = [[0,2,5,8],[2,0,3,6],[5,3,0,2],[8,6,2,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,2}, optional={3})
    ze = _region_z(m, x, nodes, 2)
    _add_conditional(m, x, t, nodes, optional_node=3, trigger_node=2,
                     ze=ze, zval=1, cond_src=2, cond_tgt=3, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z_e2={int(ze.X)}  e3 {'visited' if ze.X>0.5 else 'skipped'}")
        plot_ctpo("visualization/ctpo_if_visited.png", {1:"e_1",2:"e_2",3:"e_3"},
                  mandatory={1,2}, optional={3}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf"))],
                  cond_edges=[(1,3,0,float("inf"),"if z_{e2}","#d73027")],
                  title="cTPO — e_3 if e_2 visited")
        plot_tsp("visualization/tsp_if_visited.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3"},
                 mandatory={0,1,2}, optional={3}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — if visited")


def case_if_not_visited():
    """e3 required only if e2 NOT visited. Since e2 mandatory, e3 never triggered."""
    nodes  = [0,1,2,3]
    depot  = 0
    costs  = [[0,2,5,8],[2,0,3,6],[5,3,0,2],[8,6,2,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,2}, optional={3})
    ze = _region_z(m, x, nodes, 2)
    _add_conditional(m, x, t, nodes, optional_node=3, trigger_node=2,
                     ze=ze, zval=0, cond_src=2, cond_tgt=3, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        x_vals = m.getAttr("X", x)
        e3_visited = any(x_vals[i,3] > 0.5 for i in nodes if i != 3)
        _print_tour(tour, labels, t, tf, f"z_e2={int(ze.X)}  e3 {'visited' if e3_visited else 'skipped'}")
        plot_ctpo("visualization/ctpo_if_not_visited.png", {1:"e_1",2:"e_2",3:"e_3"},
                  mandatory={1,2}, optional={3}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf"))],
                  cond_edges=[(1,3,0,float("inf"),"if NOT z_{e2}","#7b2d8b")],
                  title="cTPO — e_3 if e_2 NOT visited")
        plot_tsp("visualization/tsp_if_not_visited.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3"},
                 mandatory={0,1,2}, optional={3}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — if not visited")


def case_and():
    """e4 only if BOTH e2 AND e3 visited. Both mandatory → e4 always visited."""
    nodes  = [0,1,2,3,4]
    depot  = 0
    costs  = [[0,2,6,7,9],[2,0,3,4,8],[6,3,0,2,3],[7,4,2,0,3],[9,8,3,3,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3",4:"e4"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,2,3}, optional={4})
    z2 = _region_z(m, x, nodes, 2)
    z3 = _region_z(m, x, nodes, 3)
    ze = _combine_z(m, [z2, z3], "and")
    _add_conditional(m, x, t, nodes, optional_node=4, trigger_node=None,
                     ze=ze, zval=1, cond_src=3, cond_tgt=4, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z2={int(z2.X)} z3={int(z3.X)} ze={int(ze.X)}  e4 {'visited' if ze.X>0.5 else 'skipped'}")
        plot_ctpo("visualization/ctpo_and.png", {1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                  mandatory={1,2,3}, optional={4}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf")),(1,3,0,float("inf"))],
                  cond_edges=[(1,4,0,float("inf"),"if z_{e2} AND z_{e3}","#d73027")],
                  title="cTPO — AND: e_4 if e_2 AND e_3")
        plot_tsp("visualization/tsp_and.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                 mandatory={0,1,2,3}, optional={4}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — AND condition")


def case_or():
    """e4 if e2 OR e3 visited. Both mandatory → e4 always visited."""
    nodes  = [0,1,2,3,4]
    depot  = 0
    costs  = [[0,2,6,7,9],[2,0,3,4,8],[6,3,0,2,3],[7,4,2,0,3],[9,8,3,3,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3",4:"e4"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,2,3}, optional={4})
    z2 = _region_z(m, x, nodes, 2)
    z3 = _region_z(m, x, nodes, 3)
    ze = _combine_z(m, [z2, z3], "or")
    _add_conditional(m, x, t, nodes, optional_node=4, trigger_node=None,
                     ze=ze, zval=1, cond_src=3, cond_tgt=4, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z2={int(z2.X)} z3={int(z3.X)} ze={int(ze.X)}  e4 {'visited' if ze.X>0.5 else 'skipped'}")
        plot_ctpo("visualization/ctpo_or.png", {1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                  mandatory={1,2,3}, optional={4}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf")),(1,3,0,float("inf"))],
                  cond_edges=[(1,4,0,float("inf"),"if z_{e2} OR z_{e3}","#1a9641")],
                  title="cTPO — OR: e_4 if e_2 OR e_3")
        plot_tsp("visualization/tsp_or.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                 mandatory={0,1,2,3}, optional={4}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — OR condition")


def case_and_e2_optional():
    """AND with e2 optional. Solver skips e2 (costly) → ze=0 → e4 skipped."""
    nodes  = [0,1,2,3,4]
    depot  = 0
    costs  = [[0,2,6,7,9],[2,0,3,4,8],[6,3,0,2,3],[7,4,2,0,3],[9,8,3,3,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3",4:"e4"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,3}, optional={2,4})
    z2 = _region_z(m, x, nodes, 2)
    z3 = _region_z(m, x, nodes, 3)
    ze = _combine_z(m, [z2, z3], "and")
    _add_conditional(m, x, t, nodes, optional_node=4, trigger_node=None,
                     ze=ze, zval=1, cond_src=3, cond_tgt=4, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z2={int(z2.X)} z3={int(z3.X)} ze={int(ze.X)}  e4 {'visited' if ze.X>0.5 else 'skipped'}")
        plot_ctpo("visualization/ctpo_and_e2opt.png", {1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                  mandatory={1,3}, optional={2,4}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf")),(1,3,0,float("inf"))],
                  cond_edges=[(1,4,0,float("inf"),"if z_{e2} AND z_{e3}","#d73027")],
                  title="cTPO — AND (e_2 optional): e_4 if e_2 AND e_3")
        plot_tsp("visualization/tsp_and_e2opt.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                 mandatory={0,1,3}, optional={2,4}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — AND, e_2 optional")


def case_or_e2_optional():
    """OR with e2 optional. z3=1 (e3 mandatory) → ze=1 → e4 visited even without e2."""
    nodes  = [0,1,2,3,4]
    depot  = 0
    costs  = [[0,2,6,7,9],[2,0,3,4,8],[6,3,0,2,3],[7,4,2,0,3],[9,8,3,3,0]]
    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3",4:"e4"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,3}, optional={2,4})
    z2 = _region_z(m, x, nodes, 2)
    z3 = _region_z(m, x, nodes, 3)
    ze = _combine_z(m, [z2, z3], "or")
    _add_conditional(m, x, t, nodes, optional_node=4, trigger_node=None,
                     ze=ze, zval=1, cond_src=3, cond_tgt=4, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z2={int(z2.X)} z3={int(z3.X)} ze={int(ze.X)}  e4 {'visited' if ze.X>0.5 else 'skipped'}")
        plot_ctpo("visualization/ctpo_or_e2opt.png", {1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                  mandatory={1,3}, optional={2,4}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf")),(1,3,0,float("inf"))],
                  cond_edges=[(1,4,0,float("inf"),"if z_{e2} OR z_{e3}","#1a9641")],
                  title="cTPO — OR (e_2 optional): e_4 if e_2 OR e_3")
        plot_tsp("visualization/tsp_or_e2opt.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                 mandatory={0,1,3}, optional={2,4}, ap_nodes=set(),
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title="TSP — OR, e_2 optional")


def case_ap_waypoint(via_ap=False):
    """e4 triggered by traversing AP waypoint node. Runs both skip and via variants."""
    nodes = [0,1,2,3,4,5]
    depot = 0
    ap    = 5
    if via_ap:
        costs = [
            [ 0,  2, 99, 99, 99,  3],
            [ 2,  0, 99, 99, 99,  2],
            [99, 99,  0,  1,  2,  1],
            [99, 99,  1,  0,  2,  1],
            [99, 99,  2,  2,  0,  2],
            [ 3,  2,  1,  1,  2,  0],
        ]
        print("  Mode: via-ap  (AP is cheap gateway → z_ap=1 expected → e4 visited)")
    else:
        costs = [
            [0,  2,  4,  5,  9,  99],
            [2,  0,  3,  4,  8,  99],
            [4,  3,  0,  1,  2,  99],
            [5,  4,  1,  0,  2,  99],
            [9,  8,  2,  2,  0,  99],
            [99, 99, 99, 99, 99,   0],
        ]
        print("  Mode: skip-ap (direct edges cheap → z_ap=0 expected → e4 skipped)")

    edges  = [(i,j) for i in nodes for j in nodes if i!=j]
    labels = {0:"depot",1:"e1",2:"e2",3:"e3",4:"e4",5:"ap"}

    m, x, t, tf = _new_model(nodes, depot, costs, edges, mandatory={0,1,2,3}, optional={4})
    ze = _region_z(m, x, nodes, ap)
    _add_conditional(m, x, t, nodes, optional_node=4, trigger_node=ap,
                     ze=ze, zval=1, cond_src=3, cond_tgt=4, cond_lb=0, cond_ub=5)
    m.optimize()

    tour = _extract_tour(m, x, nodes, edges, depot)
    if tour:
        _print_tour(tour, labels, t, tf, f"z_ap={int(ze.X)}  e4 {'visited' if ze.X>0.5 else 'skipped'}")
        suffix = "via_ap" if via_ap else "skip_ap"
        plot_ctpo(f"visualization/ctpo_ap_{suffix}.png", {1:"e_1",2:"e_2",3:"e_3",4:"e_4"},
                  mandatory={1,2,3}, optional={4}, ap_nodes=set(),
                  base_edges=[(1,2,0,float("inf")),(1,3,0,float("inf"))],
                  cond_edges=[(1,4,0,float("inf"),"if p_{lava}","#d73027")],
                  title=f"cTPO — AP waypoint ({suffix})")
        plot_tsp(f"visualization/tsp_ap_{suffix}.png",
                 {0:"depot",1:"e_1",2:"e_2",3:"e_3",4:"e_4",5:"lava"},
                 mandatory={0,1,2,3}, optional={4}, ap_nodes={5},
                 nodes=nodes, edges=edges, costs=costs, tour=tour,
                 title=f"TSP — AP waypoint ({suffix})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CASES = {
    "if-visited":      case_if_visited,
    "if-not-visited":  case_if_not_visited,
    "and":             case_and,
    "or":              case_or,
    "and-e2-optional": case_and_e2_optional,
    "or-e2-optional":  case_or_e2_optional,
    "ap-waypoint":     lambda: (case_ap_waypoint(via_ap=False), case_ap_waypoint(via_ap=True)),
}

parser = argparse.ArgumentParser(description="Conditional TPO simple examples")
parser.add_argument(
    "--condition",
    choices=list(CASES.keys()),
    required=True,
    help="Which conditional TPO case to run",
)
args = parser.parse_args()

print(f"\n=== {args.condition} ===")
CASES[args.condition]()
