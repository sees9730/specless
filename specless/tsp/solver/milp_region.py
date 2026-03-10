import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from specless.tsp.tsp import Node

from .milp import MILPTSPWithTPOSolver

if TYPE_CHECKING:
    from specless.utils.state_regions import StateRegions
    from specless.utils.conditional_tsp_mapper import ConditionalTSPMapper
    from specless.specification.conditional_tpo import ConditionalTPO


class MILPTSPWithRegionTracking(MILPTSPWithTPOSolver):
    """
    Extends MILPTSPWithTPOSolver with conditional event support.

    Supports:
    - Region tracking: binary z[region] variables linked to edge usage
    - Optional events: nodes that can be skipped unless their region condition is met
    - Conditional TPO: precedence constraints activated by region visit variables
    """

    def __init__(self, mapping: "ConditionalTSPMapper"):
        super().__init__()
        self.mapping = mapping
        self.z_vars: Dict[str, gp.Var] = {}
        self.region_values: Dict[str, bool] = {}
        self.tsp_regions: Dict[str, Set[int]] = {}
        self.crossing_edges: Dict[str, Set[Tuple[int, int]]] = {}
        self.ctpo: Optional["ConditionalTPO"] = None

    def solve(
        self,
        tsp,
        ctpo: "ConditionalTPO",
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = False,
        export_filename: Optional[str] = None,
        transition_system=None,
    ) -> Tuple[List, float, List]:
        self.ctpo = ctpo
        self.tsp_regions = self._convert_to_tsp_regions(ctpo.regions)
        self.crossing_edges = self._compute_crossing_edges(ctpo.regions, transition_system)

        tsp = copy.deepcopy(tsp)

        if len(tsp.nodes) < num_agent:
            num_agent = len(tsp.nodes) - 1

        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        self.agents = list(range(num_agent))
        self.init_nodes = init_nodes
        self.come_back_home = come_back_home

        if not self.come_back_home:
            for init_node in init_nodes:
                for i in set(tsp.nodes) - {init_node}:
                    tsp.costs[i][init_node] = 0

        m, variables = self.initialize_problem(tsp)

        self._add_region_variables(m)
        self._add_constraints(m, variables, tsp, init_nodes)
        self._add_optional_event_constraints(m, variables, tsp)

        tf = variables["tf"]
        m.setObjective(tf, GRB.MINIMIZE)
        m.optimize()

        if export_filename:
            m.write(export_filename)

        if m.status != GRB.OPTIMAL:
            print("Tour: [], Cost: n/a")
            return [], -1, []

        self._extract_region_values()

        tours = self.get_tours(m, variables)
        timestamps = [self.get_timestamps(m, variables, tour) for tour in tours]
        cost = self.get_cost(m)

        return tours, cost, timestamps

    # -------------------------------------------------------------------------

    def _convert_to_tsp_regions(self, regions: "StateRegions") -> Dict[str, Set[int]]:
        return regions.to_tsp_regions(self.mapping)

    def _compute_crossing_edges(self, regions: "StateRegions", transition_system) -> Dict[str, Set[Tuple[int, int]]]:
        """Find TSP edge pairs (i,j) whose shortest TS path crosses a region.

        This implements the same region-touching logic as the standard node-set
        approach, but handles the case where a region has no TSP node of its own.

        Normally a region is defined by observation tiles that are also TSP nodes
        (mandatory or optional events), so to_tsp_regions() finds them directly
        and the touching edges are populated via node membership.  When the
        observation color palette is exhausted (e.g. gym_minigrid only has 6
        floor colors, all taken by event tiles), a background object like Lava is
        used to mark the region instead.  Lava tiles are stored in obs_to_states
        but carry no TSP node, so tsp_regions[region] is empty and the standard
        path is a no-op.

        Passing transition_system here bridges that gap: we walk every pair of
        TSP nodes and check whether j is reachable from i in the region-free
        graph (all region states removed).  If not, every path between i and j
        must pass through the region, so the edge (i,j) is marked as crossing.
        Those edges are then merged into the touching set in _add_constraints,
        giving identical MILP semantics.
        """
        if transition_system is None:
            return {}

        crossing: Dict[str, Set[Tuple[int, int]]] = {r: set() for r in regions.regions}
        tsp_nodes = self.mapping.get_tsp_nodes()

        # Build a subgraph with region nodes removed for each region, to test
        # whether the region is truly unavoidable on the shortest path.
        region_free_graphs = {}
        for region_name, region_states in regions.regions.items():
            g = nx.DiGraph(transition_system)
            g.remove_nodes_from(region_states)
            region_free_graphs[region_name] = g

        for i in tsp_nodes:
            for j in tsp_nodes:
                if i == j:
                    continue
                states_i = self.mapping.node_to_states[i]
                states_j = self.mapping.node_to_states[j]

                # Find shortest path length through full graph
                sp_len = None
                for si in states_i:
                    for sj in states_j:
                        try:
                            d = nx.shortest_path_length(transition_system, source=si, target=sj)
                            if sp_len is None or d < sp_len:
                                sp_len = d
                        except nx.NetworkXNoPath:
                            continue

                if sp_len is None:
                    continue

                # Mark crossing only if every path between i and j passes through
                # the region — i.e. j is unreachable from i in the region-free graph.
                for region_name in regions.regions:
                    g_free = region_free_graphs[region_name]
                    region_states = regions.regions[region_name]
                    # skip if start/end are themselves in the region
                    if set(states_i) & region_states or set(states_j) & region_states:
                        continue
                    reachable = any(
                        nx.has_path(g_free, si, sj)
                        for si in states_i
                        for sj in states_j
                    )
                    if not reachable:
                        crossing[region_name].add((i, j))

        return crossing

    def _add_region_variables(self, m: gp.Model):
        self.z_vars = {}
        needed = set()
        for key in self.ctpo.conditional_tpos:
            needed.add(key.replace("NOT_", "") if key.startswith("NOT_") else key)
        for region_name in needed:
            if region_name in self.tsp_regions:
                self.z_vars[region_name] = m.addVar(vtype=GRB.BINARY, name=f"z_{region_name}")
        m.update()

    def _add_constraints(self, m: gp.Model, variables: Dict, tsp, init_nodes: List[int]):
        x = variables["x"]
        t = variables["t"]
        tT = variables["tT"]
        tf = variables["tf"]
        non_init_nodes = list(set(tsp.nodes) - set(init_nodes))
        mandatory_nodes = set(self.mapping.get_mandatory_nodes())

        # GTSP: mandatory nodesets use ==, optional use <=
        for nodes in tsp.nodesets:
            is_mandatory = any(n in mandatory_nodes for n in nodes)
            K = sum(n in init_nodes for n in nodes) or 1
            op = m.addConstr if is_mandatory else None
            if is_mandatory:
                m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) == K, "incoming")
                m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) == K, "outgoing")
            else:
                m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) <= K, "incoming_opt")
                m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) <= K, "outgoing_opt")

        m.addConstrs((x.sum("*", n) == x.sum(n, "*") for n in tsp.nodes), "flow")

        m.addConstrs(
            (t[n] >= tsp.tpo.global_constraints[n]["lb"] for n in tsp.tpo.global_constraints),
            "nodeLB",
        )
        m.addConstrs(
            (t[n] <= tsp.tpo.global_constraints[n]["ub"] for n in tsp.tpo.global_constraints),
            "nodeUB",
        )

        local_const_edges = [
            (src, tgt)
            for src, d in tsp.tpo.local_constraints.items()
            for tgt in d
        ]
        m.addConstrs(
            (t[tgt] - t[src] >= tsp.tpo.local_constraints[src][tgt]["lb"]
             for src, tgt in local_const_edges),
            "localLB",
        )
        m.addConstrs(
            (t[tgt] - t[src] <= tsp.tpo.local_constraints[src][tgt]["ub"]
             for src, tgt in local_const_edges),
            "localUB",
        )

        for idx, (node_a, node_b, max_diff) in enumerate(tsp.tpo.difference_constraints):
            m.addConstr(t[node_a] - t[node_b] <=  max_diff, name=f"diff_{idx}_ab")
            m.addConstr(t[node_b] - t[node_a] <=  max_diff, name=f"diff_{idx}_ba")

        M = 100000
        for (i, j) in tsp.edges:
            if i in init_nodes:
                m.addConstr(t[i] == 0, name=f"init_time_{i}")
            if j not in init_nodes:
                m.addConstr(
                    t[j] >= t[i] + tsp.costs[i][j] - M * (1 - x[i, j]),
                    name=f"time_prop_{i}_{j}",
                )

        for ii, I in enumerate(init_nodes):
            edges_to_init = [(i, I) for i in non_init_nodes if (i, I) in tsp.edges]
            for (i, _) in edges_to_init:
                m.addConstr(
                    tT[ii, I] >= t[i] + tsp.costs[i][I] * x[i, I] - M * (1 - x[i, I]),
                    name=f"terminal_time_{ii}_{I}_{i}",
                )

        for ii, I in enumerate(init_nodes):
            m.addConstr(tf >= tT[ii, I], name=f"final_time_{ii}_{I}")

        # Region tracking: z[r] linked to edge variables
        for region_name, node_set in self.tsp_regions.items():
            if region_name not in self.z_vars:
                continue
            z = self.z_vars[region_name]
            crossing = self.crossing_edges.get(region_name, set())
            touching = list({
                (i, j) for (i, j) in tsp.edges
                if i in node_set or j in node_set or (i, j) in crossing
            })
            for (i, j) in touching:
                m.addConstr(z >= x[i, j], name=f"region_{region_name}_lb_{i}_{j}")
            if touching:
                m.addConstr(
                    z <= gp.quicksum(x[i, j] for (i, j) in touching),
                    name=f"region_{region_name}_ub",
                )

        m.update()

    def _add_optional_event_constraints(self, m: gp.Model, variables: Dict, tsp):
        if not self.ctpo or not self.ctpo.conditional_tpos:
            return

        x = variables["x"]
        t = variables["t"]
        M = 100000

        # Map each event node to the cTPO keys it appears in
        event_to_keys: Dict[int, Set[str]] = defaultdict(set)
        for key, cond_tpo in self.ctpo.conditional_tpos.items():
            for event_node in self.ctpo.get_conditional_events_for_region(key):
                event_to_keys[event_node].add(key)

        for event_node, keys in event_to_keys.items():
            incoming = [(i, j) for (i, j) in tsp.edges if j == event_node]
            if not incoming:
                continue
            incoming_sum = gp.quicksum(x[i, j] for (i, j) in incoming)

            # Detect if this event is required in both normal and negated forms
            region_groups: Dict[str, Dict[str, bool]] = {}
            for key in keys:
                is_neg = self.ctpo.negated_regions.get(key, False)
                rname = key.replace("NOT_", "") if is_neg else key
                if rname not in region_groups:
                    region_groups[rname] = {"normal": False, "negated": False}
                region_groups[rname]["negated" if is_neg else "normal"] = True

            always_required = any(
                g["normal"] and g["negated"] for g in region_groups.values()
            )

            if always_required:
                m.addConstr(incoming_sum == 1, name=f"mandatory_event_{event_node}")
            else:
                for key in sorted(keys):
                    is_neg = self.ctpo.negated_regions.get(key, False)
                    rname = key.replace("NOT_", "") if is_neg else key
                    z = self.z_vars.get(rname)
                    if z is None:
                        continue
                    if is_neg:
                        m.addConstr(incoming_sum <= 1 - z, name=f"opt_{event_node}_req_NOT_{rname}")
                        m.addConstr(incoming_sum >= 1 - z, name=f"NOT_{rname}_req_{event_node}")
                    else:
                        m.addConstr(incoming_sum <= z, name=f"opt_{event_node}_req_{rname}")
                        m.addConstr(incoming_sum >= z, name=f"_{rname}_req_{event_node}")

        # Conditional precedence constraints (Big-M)
        for key, cond_tpo in self.ctpo.conditional_tpos.items():
            is_neg = self.ctpo.negated_regions.get(key, False)
            rname = key.replace("NOT_", "") if is_neg else key
            z = self.z_vars.get(rname)
            if z is None:
                continue
            for src, targets in cond_tpo.local_constraints.items():
                for tgt, bounds in targets.items():
                    lb, ub = bounds["lb"], bounds["ub"]
                    if is_neg:
                        m.addConstr(t[tgt] - t[src] >= lb - M * z,
                                    name=f"cond_prec_NOT_{rname}_{src}_{tgt}_lb")
                        m.addConstr(t[tgt] - t[src] <= ub + M * z,
                                    name=f"cond_prec_NOT_{rname}_{src}_{tgt}_ub")
                    else:
                        m.addConstr(t[tgt] - t[src] >= lb - M * (1 - z),
                                    name=f"cond_prec_{rname}_{src}_{tgt}_lb")
                        m.addConstr(t[tgt] - t[src] <= ub + M * (1 - z),
                                    name=f"cond_prec_{rname}_{src}_{tgt}_ub")

        m.update()

    def _extract_region_values(self):
        self.region_values = {name: var.X > 0.5 for name, var in self.z_vars.items()}

    def get_visited_regions(self) -> Dict[str, bool]:
        return self.region_values.copy()

    def print_visited_regions(self):
        print("\n=== Regions Visited ===")
        for name in sorted(self.region_values):
            status = "visited" if self.region_values[name] else "avoided"
            print(f"  {name}: {status}")
