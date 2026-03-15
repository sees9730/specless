"""
Visualization helpers for simple conditional TPO examples.

Produces two plots per scenario:
  1. cTPO graph  — mandatory events (solid/green), optional events (dashed/blue),
                   conditional TPO edges (colored by condition),
                   AP nodes shown as diamonds labeled p_{name}
  2. TSP graph   — all nodes with edge costs, active tour highlighted
"""

import os
import networkx as nx
from specless.io import draw_graph


def _node_attrs(node_id, labels, mandatory, optional, ap_nodes):
    label = labels[node_id]
    if node_id in ap_nodes:
        return dict(
            label=f"p_{{{label}}}",
            shape="diamond",
            style="filled",
            fillcolor="#f0a500",
            fontcolor="black",
            fontsize="11",
            penwidth="2",
        )
    if node_id in optional:
        return dict(
            label=label,
            shape="circle",
            style="filled,dashed",
            fillcolor="#6baed6",
            fontcolor="black",
            fontsize="11",
            penwidth="2",
        )
    return dict(
        label=label,
        shape="circle",
        style="filled",
        fillcolor="#74c476",
        fontcolor="black",
        fontsize="11",
        penwidth="2",
    )


def plot_ctpo(
    filename,
    labels,          # {node_id: "e1" / "ap" / ...}
    mandatory,       # set of mandatory node ids
    optional,        # set of optional node ids
    ap_nodes,        # set of AP node ids
    base_edges,      # [(src, tgt, lb, ub)]  — unconditional precedences
    cond_edges,      # [(src, tgt, lb, ub, condition_str, color)]
    title="cTPO",
):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    G = nx.DiGraph()
    G.graph["label"] = title
    G.graph["labelloc"] = "t"
    G.graph["fontsize"] = "14"
    G.graph["rankdir"] = "LR"

    for nd in sorted(labels.keys()):
        G.add_node(nd, **_node_attrs(nd, labels, mandatory, optional, ap_nodes))

    for (s, t, lb, ub) in base_edges:
        ub_str = f"{ub}" if ub < 1e8 else "∞"
        G.add_edge(s, t,
                   label=f"[{lb},{ub_str}]",
                   color="#2c7bb6",
                   fontcolor="#2c7bb6",
                   style="solid",
                   penwidth="2",
                   fontsize="10")

    for (s, t, lb, ub, cond, color) in cond_edges:
        ub_str = f"{ub}" if ub < 1e8 else "∞"
        G.add_edge(s, t,
                   label=f"[{lb},{ub_str}]\\n{cond}",
                   color=color,
                   fontcolor=color,
                   style="dashed",
                   penwidth="2",
                   fontsize="10")

    # strip .png — draw_graph appends the format itself
    base = filename[:-4] if filename.endswith(".png") else filename
    draw_graph(G, base, should_display=False, img_format="png")
    print(f"  Saved {filename}")


def plot_tsp(
    filename,
    labels,        # {node_id: "e1" / ...}
    mandatory,
    optional,
    ap_nodes,
    nodes,
    edges,         # all TSP edges (i, j)
    costs,         # 2D list
    tour=None,     # ordered list of node ids for active tour
    title="TSP graph",
):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    tour_edges = set(zip(tour[:-1], tour[1:])) if tour else set()

    G = nx.DiGraph()
    G.graph["label"] = title
    G.graph["labelloc"] = "t"
    G.graph["fontsize"] = "14"

    for nd in nodes:
        G.add_node(nd, **_node_attrs(nd, labels, mandatory, optional, ap_nodes))

    for (i, j) in edges:
        c = costs[i][j]
        if (i, j) in tour_edges:
            G.add_edge(i, j,
                       label=str(c),
                       color="#d73027",
                       fontcolor="#d73027",
                       style="solid",
                       penwidth="3",
                       fontsize="11")
        elif c >= 99:
            G.add_edge(i, j,
                       label="∞",
                       color="#dddddd",
                       fontcolor="#cccccc",
                       style="dashed",
                       penwidth="1",
                       fontsize="8")
        else:
            G.add_edge(i, j,
                       label=str(c),
                       color="#cccccc",
                       fontcolor="#999999",
                       style="solid",
                       penwidth="1",
                       fontsize="9")

    base = filename[:-4] if filename.endswith(".png") else filename
    draw_graph(G, base, should_display=False, img_format="png")
    print(f"  Saved {filename}")
