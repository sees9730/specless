"""
Conditional TPO — OR condition example (e2 optional)
=====================================================

5 nodes: depot(0), e1(1), e2(2), e3(3), e4(4)

Mandatory:  depot, e1, e3
Optional:   e2, e4
e4 visited if e2 OR e3 visited.
Since e3 is mandatory (z3 always 1), ze4=1 always → e4 always visited.
Even though e2 is optional and may be skipped, OR fires from z3 alone.

Cost matrix:
       depot  e1  e2  e3  e4
depot [  0,   2,  6,  7,  9 ]
e1    [  2,   0,  3,  4,  8 ]
e2    [  6,   3,  0,  2,  3 ]
e3    [  7,   4,  2,  0,  3 ]
e4    [  9,   8,  3,  3,  0 ]
"""

import gurobipy as gp
from gurobipy import GRB

nodes  = [0, 1, 2, 3, 4]
depot  = 0
costs  = [
    [0, 2, 6, 7, 9],
    [2, 0, 3, 4, 8],
    [6, 3, 0, 2, 3],
    [7, 4, 2, 0, 3],
    [9, 8, 3, 3, 0],
]
edges  = [(i, j) for i in nodes for j in nodes if i != j]
labels = {0: "depot", 1: "e1", 2: "e2", 3: "e3", 4: "e4"}

env = gp.Env()
env.setParam("OutputFlag", 0)
m = gp.Model(env=env)

x   = m.addVars(edges, vtype=GRB.BINARY,     name="x")
t   = m.addVars(nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
tT  = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tT")
tf  = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tf")
z2  = m.addVar(vtype=GRB.BINARY, name="z2")
z3  = m.addVar(vtype=GRB.BINARY, name="z3")
ze4 = m.addVar(vtype=GRB.BINARY, name="ze4")

# --- Mandatory: depot, e1, e3 ---
for nd in [depot, 1, 3]:
    m.addConstr(gp.quicksum(x[i, nd] for i in nodes if i != nd) == 1, f"in_{nd}")
    m.addConstr(gp.quicksum(x[nd, j] for j in nodes if j != nd) == 1, f"out_{nd}")

# --- Optional: e2 ---
m.addConstr(gp.quicksum(x[i, 2] for i in nodes if i != 2) <= 1, "in_2_opt")
m.addConstr(gp.quicksum(x[2, j] for j in nodes if j != 2) <= 1, "out_2_opt")

# --- Flow conservation ---
for nd in nodes:
    m.addConstr(
        gp.quicksum(x[i, nd] for i in nodes if i != nd) ==
        gp.quicksum(x[nd, j] for j in nodes if j != nd),
        f"flow_{nd}"
    )

# --- Region tracking: z2=1 iff e2 visited, z3=1 iff e3 visited ---
for nd, z in [(2, z2), (3, z3)]:
    incoming = [(i, nd) for i in nodes if i != nd]
    for (i, j) in incoming:
        m.addConstr(z >= x[i, j], f"region_lb_{nd}_{i}")
    m.addConstr(z <= gp.quicksum(x[i, j] for (i, j) in incoming), f"region_ub_{nd}")

# --- OR linearization: ze4 = z2 OR z3 ---
m.addConstr(ze4 >= z2,       "or_lb1")
m.addConstr(ze4 >= z3,       "or_lb2")
m.addConstr(ze4 <= z2 + z3,  "or_ub")

# --- Optional e4: visit iff ze4=1 ---
in_e4 = gp.quicksum(x[i, 4] for i in nodes if i != 4)
m.addConstr((ze4 == 1) >> (in_e4 == 1), "ze4_req_e4")
m.addConstr((ze4 == 0) >> (in_e4 == 0), "ze4_skip_e4")

# --- Time propagation ---
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

# --- Conditional TPO: if ze4=1 → e3->e4 in [0,5] ---
m.addConstr((ze4 == 1) >> (t[4] - t[3] >= 0), "cond_lb")
m.addConstr((ze4 == 1) >> (t[4] - t[3] <= 5), "cond_ub")

m.setObjective(tf, GRB.MINIMIZE)
m.optimize()

# --- Results ---
if m.status != GRB.OPTIMAL:
    print(f"No optimal solution (status={m.status})")
else:
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

    print(f"Tour:       {' -> '.join(labels[n] for n in tour)}")
    print(f"Cost (tf):  {tf.X:.1f}")
    print(f"Timestamps: { {labels[n]: round(t_vals[n], 1) for n in tour[:-1]} }")
    print(f"z2={int(z2.X)} z3={int(z3.X)} ze4={int(ze4.X)}")
    print(f"e4 {'visited' if any(x_vals[i,4]>0.5 for i in nodes if i!=4) else 'skipped'}")
