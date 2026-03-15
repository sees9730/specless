"""
Conditional TPO — AP waypoint node example
==========================================

6 nodes: depot(0), e1(1), e2(2), e3(3), e4(4), ap(5)

Mandatory:  depot, e1, e2, e3
Optional:   e4 — visit only if ap is traversed (z_ap=1)
AP:         ap (node 5) — physical waypoint, flow conservation only

Flag --via-ap:
  costs favor routing through ap (e1->ap->e2 cheaper than e1->e2 direct)
  → z_ap=1 → e4 required and visited

Flag --skip-ap (default):
  direct edges are cheap, ap detour not worth it
  → z_ap=0 → e4 skipped
"""

import argparse
import gurobipy as gp
from gurobipy import GRB
from viz import plot_ctpo, plot_tsp

parser = argparse.ArgumentParser()
parser.add_argument("--via-ap", action="store_true", help="Use costs that favor routing through ap")
args = parser.parse_args()

nodes  = [0, 1, 2, 3, 4, 5]
depot  = 0
ap     = 5
labels = {0: "depot", 1: "e1", 2: "e2", 3: "e3", 4: "e4", 5: "ap"}

if args.via_ap:
    # ap is a cheap gateway between e1 and e2/e3 — solver routes through it
    costs = [
        [ 0,  2, 99, 99, 99,  3],
        [ 2,  0, 99, 99, 99,  2],
        [99, 99,  0,  1,  2,  1],
        [99, 99,  1,  0,  2,  1],
        [99, 99,  2,  2,  0,  2],
        [ 3,  2,  1,  1,  2,  0],
    ]
    print("Mode: via-ap  (ap is cheap gateway → z_ap=1 expected → e4 visited)")
else:
    # direct edges are cheap — ap detour not worthwhile
    costs = [
        [0,  2,  4,  5,  9,  99],
        [2,  0,  3,  4,  8,  99],
        [4,  3,  0,  1,  2,  99],
        [5,  4,  1,  0,  2,  99],
        [9,  8,  2,  2,  0,  99],
        [99, 99, 99, 99, 99,   0],
    ]
    print("Mode: skip-ap (direct edges cheap → z_ap=0 expected → e4 skipped)")

edges = [(i, j) for i in nodes for j in nodes if i != j]

env = gp.Env()
env.setParam("OutputFlag", 0)
m = gp.Model(env=env)

x    = m.addVars(edges, vtype=GRB.BINARY,          name="x")
t    = m.addVars(nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
tT   = m.addVar(lb=0.0,  vtype=GRB.CONTINUOUS,     name="tT")
tf   = m.addVar(lb=0.0,  vtype=GRB.CONTINUOUS,     name="tf")
z_ap = m.addVar(vtype=GRB.BINARY, name="z_ap")

# --- Mandatory: depot, e1, e2, e3 ---
for nd in [depot, 1, 2, 3]:
    m.addConstr(gp.quicksum(x[i, nd] for i in nodes if i != nd) == 1, f"in_{nd}")
    m.addConstr(gp.quicksum(x[nd, j] for j in nodes if j != nd) == 1, f"out_{nd}")

# --- AP node: flow conservation only (no visit-count constraint) ---
for nd in [ap]:
    m.addConstr(
        gp.quicksum(x[i, nd] for i in nodes if i != nd) ==
        gp.quicksum(x[nd, j] for j in nodes if j != nd),
        f"flow_{nd}"
    )

# --- Optional e4: flow <= 1 ---
m.addConstr(gp.quicksum(x[i, 4] for i in nodes if i != 4) <= 1, "in_4_opt")
m.addConstr(gp.quicksum(x[4, j] for j in nodes if j != 4) <= 1, "out_4_opt")

# --- Flow conservation for all nodes ---
for nd in nodes:
    m.addConstr(
        gp.quicksum(x[i, nd] for i in nodes if i != nd) ==
        gp.quicksum(x[nd, j] for j in nodes if j != nd),
        f"flow_{nd}"
    )

# --- Region tracking: z_ap=1 iff ap traversed ---
incoming_ap = [(i, ap) for i in nodes if i != ap]
for (i, j) in incoming_ap:
    m.addConstr(z_ap >= x[i, j], f"zap_lb_{i}")
m.addConstr(z_ap <= gp.quicksum(x[i, j] for (i, j) in incoming_ap), "zap_ub")

# --- Optional e4: visit iff z_ap=1 ---
in_e4 = gp.quicksum(x[i, 4] for i in nodes if i != 4)
m.addConstr((z_ap == 1) >> (in_e4 == 1), "zap_req_e4")
m.addConstr((z_ap == 0) >> (in_e4 == 0), "zap_skip_e4")

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

# --- Conditional TPO: if z_ap=1 → e3->e4 in [0,5] ---
m.addConstr((z_ap == 1) >> (t[4] - t[3] >= 0), "cond_lb")
m.addConstr((z_ap == 1) >> (t[4] - t[3] <= 5), "cond_ub")

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
    print(f"z_ap={int(z_ap.X)}  ({'traversed' if z_ap.X > 0.5 else 'avoided'})")
    print(f"e4 {'visited' if any(x_vals[i,4]>0.5 for i in nodes if i!=4) else 'skipped'}")

    suffix = "via_ap" if args.via_ap else "skip_ap"

    plot_ctpo(
        f"visualization/ctpo_{suffix}.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3", 4: "e_4", 5: "lava"},
        mandatory={0, 1, 2, 3},
        optional={4},
        ap_nodes={5},
        base_edges=[
            (1, 2, 0, float("inf")),
            (1, 3, 0, float("inf")),
        ],
        cond_edges=[
            (1, 4, 0, float("inf"), "if p_{lava}", "#d73027"),
        ],
        title=f"cTPO — AP waypoint ({suffix})",
    )

    plot_tsp(
        f"visualization/tsp_{suffix}.png",
        labels={0: "depot", 1: "e_1", 2: "e_2", 3: "e_3", 4: "e_4", 5: "lava"},
        mandatory={0, 1, 2, 3},
        optional={4},
        ap_nodes={5},
        nodes=nodes,
        edges=edges,
        costs=costs,
        tour=tour,
        title=f"TSP tour — AP waypoint ({suffix})",
    )
