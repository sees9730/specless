"""
Simple Conditional TPO Example
===============================

4 nodes: depot(0), e1(1), e2(2), e3(3)

- depot, e1, e2 are mandatory; e3 is optional
- z = 1 iff e2 is visited (always 1 since e2 is mandatory)

Case A: IF e2 visited (z=1)  → must also visit e3, e2->e3 in [0,5]
Case B: IF e2 NOT visited (z=0) → must visit e3  (never fires since e2 mandatory → e3 skipped)

Cost matrix:
       depot  e1   e2   e3
depot [  0,   2,   5,   8 ]
e1    [  2,   0,   3,   6 ]
e2    [  5,   3,   0,   2 ]
e3    [  8,   6,   2,   0 ]
"""

import gurobipy as gp
from gurobipy import GRB
from viz import plot_ctpo, plot_tsp

VIZ_LABELS = {0: "depot", 1: "e_1", 2: "e_2", 3: "e_3"}

nodes = [0, 1, 2, 3]   # depot=0, e1=1, e2=2, e3=3
depot = 0
costs = [
    [0, 2, 5, 8],
    [2, 0, 3, 6],
    [5, 3, 0, 2],
    [8, 6, 2, 0],
]
edges = [(i, j) for i in nodes for j in nodes if i != j]
labels = {0: "depot", 1: "e1", 2: "e2", 3: "e3"}


def build_base_model():
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    m = gp.Model(env=env)

    x  = m.addVars(edges, vtype=GRB.BINARY, name="x")
    t  = m.addVars(nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
    tT = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tT")
    tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tf")
    z  = m.addVar(vtype=GRB.BINARY, name="z")

    # mandatory: depot, e1, e2
    for nd in [depot, 1, 2]:
        m.addConstr(gp.quicksum(x[i, nd] for i in nodes if i != nd) == 1, f"in_{nd}")
        m.addConstr(gp.quicksum(x[nd, j] for j in nodes if j != nd) == 1, f"out_{nd}")

    # flow conservation for all nodes
    for nd in nodes:
        m.addConstr(
            gp.quicksum(x[i, nd] for i in nodes if i != nd) ==
            gp.quicksum(x[nd, j] for j in nodes if j != nd),
            f"flow_{nd}"
        )

    # region tracking: z=1 iff e2 visited
    incoming_e2 = [(i, 2) for i in nodes if i != 2]
    for (i, j) in incoming_e2:
        m.addConstr(z >= x[i, j], f"region_lb_{i}_{j}")
    m.addConstr(z <= gp.quicksum(x[i, j] for (i, j) in incoming_e2), "region_ub")

    # time propagation
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

    return m, x, t, tf, z


def print_result(m, x, t, tf, z, case_name=None):
    if m.status != GRB.OPTIMAL:
        print(f"  No optimal solution (status={m.status})")
        return None
    x_vals = m.getAttr("X", x)
    t_vals = m.getAttr("X", t)
    active = [(i, j) for (i, j) in edges if x_vals[i, j] > 0.5]
    tour, current = [depot], depot
    for _ in range(len(nodes)):
        nxt = next((j for (i, j) in active if i == current), None)
        if nxt is None or nxt == depot:
            break
        tour.append(nxt); current = nxt
    tour.append(depot)
    print(f"  Tour:       {' -> '.join(labels[n] for n in tour)}")
    print(f"  Cost (tf):  {tf.X:.1f}")
    print(f"  Timestamps: { {labels[n]: round(t_vals[n], 1) for n in tour[:-1]} }")
    print(f"  z = {int(z.X)}  (e2 {'visited' if z.X > 0.5 else 'avoided'}, e3 {'visited' if any(x_vals[i,3] > 0.5 for i in nodes if i != 3) else 'skipped'})")
    return tour


# ---------------------------------------------------------------------------
# Case A: visit e3 IF e2 visited (z=1)
# ---------------------------------------------------------------------------
print("\n--- Case A: e3 required when z=1 (e2 visited) ---")
m, x, t, tf, z = build_base_model()
incoming_e3 = gp.quicksum(x[i, 3] for i in nodes if i != 3)
m.addConstr((z == 1) >> (incoming_e3 == 1), "z1_req_e3")
m.addConstr((z == 0) >> (incoming_e3 == 0), "z0_skip_e3")
m.addConstr((z == 1) >> (t[3] - t[2] >= 0), "cond_lb")
m.addConstr((z == 1) >> (t[3] - t[2] <= 5), "cond_ub")
m.optimize()
tour_a = print_result(m, x, t, tf, z)
if tour_a:
    plot_ctpo(
        "visualization/ctpo_case_a.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3"},
        mandatory={0, 1, 2},
        optional={3},
        ap_nodes=set(),
        base_edges=[],
        cond_edges=[(2, 3, 0, 5, "if z_{e2}", "#d73027")],
        title="cTPO — Case A: e_3 if e_2 visited",
    )
    plot_tsp(
        "visualization/tsp_case_a.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3"},
        mandatory={0, 1, 2},
        optional={3},
        ap_nodes=set(),
        nodes=nodes,
        edges=edges,
        costs=costs,
        tour=tour_a,
        title="TSP tour — Case A",
    )

# ---------------------------------------------------------------------------
# Case B: visit e3 IF e2 NOT visited (z=0)
# Since e2 is mandatory, z is always 1 → e3 is never triggered → skipped.
# ---------------------------------------------------------------------------
print("\n--- Case B: e3 required when z=0 (e2 NOT visited) ---")
m, x, t, tf, z = build_base_model()
incoming_e3 = gp.quicksum(x[i, 3] for i in nodes if i != 3)
m.addConstr((z == 0) >> (incoming_e3 == 1), "z0_req_e3")
m.addConstr((z == 1) >> (incoming_e3 == 0), "z1_skip_e3")
m.addConstr((z == 0) >> (t[3] - t[2] >= 0), "cond_lb")
m.addConstr((z == 0) >> (t[3] - t[2] <= 5), "cond_ub")
m.optimize()
tour_b = print_result(m, x, t, tf, z)
if tour_b:
    plot_ctpo(
        "visualization/ctpo_case_b.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3"},
        mandatory={0, 1, 2},
        optional={3},
        ap_nodes=set(),
        base_edges=[],
        cond_edges=[(2, 3, 0, 5, "if NOT z_{e2}", "#7b2d8b")],
        title="cTPO — Case B: e_3 if e_2 NOT visited",
    )
    plot_tsp(
        "visualization/tsp_case_b.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3"},
        mandatory={0, 1, 2},
        optional={3},
        ap_nodes=set(),
        nodes=nodes,
        edges=edges,
        costs=costs,
        tour=tour_b,
        title="TSP tour — Case B",
    )
