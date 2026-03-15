import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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
        self.ctpo: Optional["ConditionalTPO"] = None
        self._M: float = 1000

    def solve(
        self,
        tsp,
        ctpo: "ConditionalTPO",
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = False,
        export_filename: Optional[str] = None,
    ) -> Tuple[List, float, List]:
        self.ctpo = ctpo
        self.tsp_regions = self._convert_to_tsp_regions(ctpo.regions)

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
        print('Optimizing model...')
        m.optimize()
        print('Done')

        if export_filename:
            m.write(export_filename)
            m.write(export_filename.replace(".lp", ".mps"))

        if m.status != GRB.OPTIMAL:
            print(f"Model status: {m.status}")
            if m.status == GRB.INFEASIBLE:
                m.computeIIS()
                m.write("iis.ilp")
                print("IIS written to iis.ilp")
            print("Tour: [], Cost: n/a")
            return [], -1, []

        self._extract_region_values()

        tours = self.get_tours(m, variables)
        timestamps = [self.get_timestamps(m, variables, tour) for tour in tours]
        cost = self.get_cost(m)

        # Debug: print active edges and time propagation
        x_vals = m.getAttr("X", variables["x"])
        t_vals = m.getAttr("X", variables["t"])
        print(f"  [debug] M={self._M:.0f}  tf={variables['tf'].X:.3f}")
        for (i, j) in sorted(x_vals):
            if x_vals[i, j] > 0.5:
                need = t_vals[i] + tsp.costs[i][j]
                ok = "OK" if t_vals[j] >= need - 1e-4 or j in self.init_nodes else "VIOLATED"
                print(f"  [debug] x[{i},{j}]=1  t[{i}]={t_vals[i]:.1f}+{tsp.costs[i][j]:.0f} -> need t[{j}]>={need:.1f}, got {t_vals[j]:.1f}  {ok}")

        return tours, cost, timestamps

    # -------------------------------------------------------------------------

    # def initialize_problem(self, tsp):
    #     """Override to set tight upper bounds on time variables."""
    #     m, variables = super().initialize_problem(tsp)
    #     # Tight UB on t: max tour time = n_nodes * max_single_edge_cost.
    #     # This makes M = UB sufficient and keeps the LP relaxation tight.
    #     finite_costs = [
    #         tsp.costs[i][j]
    #         for (i, j) in tsp.edges
    #         if 0 < tsp.costs[i][j] < 1e5
    #     ]
    #     max_edge = max(finite_costs) if finite_costs else 100
    #     self._M = len(tsp.nodes) * max_edge
    #     t = variables["t"]
    #     for n in tsp.nodes:
    #         t[n].ub = self._M
    #     variables["tf"].ub = self._M
    #     variables["tT"].values()  # iterate to set ub
    #     for v in variables["tT"].values():
    #         v.ub = self._M
    #     m.update()
    #     return m, variables

    def _convert_to_tsp_regions(self, regions: "StateRegions") -> Dict[str, Set[int]]:
        return regions.to_tsp_regions(self.mapping)

    def _add_region_variables(self, m: gp.Model):
        self.z_vars = {}
        needed = set()
        for key in self.ctpo.conditional_tpos:
            # needed.add(key.replace("NOT_", "") if key.startswith("NOT_") else key)
            needed.add(key)
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
            print(f"  [debug] nodeset {nodes}  mandatory={is_mandatory}  K={K}")
            op = m.addConstr if is_mandatory else None
            if is_mandatory:
                m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) == K, "incoming")
                m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) == K, "outgoing")
            else:
                m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) <= K, "incoming_opt")
                m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) <= K, "outgoing_opt")

        # Flow conservation: in-degree == out-degree for all nodes 
        m.addConstrs((x.sum("*", n) == x.sum(n, "*") for n in tsp.nodes), "flow")

        m.addConstrs(
            (t[n] >= tsp.tpo.global_constraints[n]["lb"] for n in tsp.tpo.global_constraints.keys()),
            "nodeLB",
        )
        m.addConstrs(
            (t[n] <= tsp.tpo.global_constraints[n]["ub"] for n in tsp.tpo.global_constraints.keys()),
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

        for i in set(init_nodes):
            m.addConstr(t[i] == 0, name=f"init_time_{i}")

        # Indicator constraints: if edge (i,j) is taken, propagate time.
        m.addConstrs(
            (
                (x[(i, j)] == 1) >> (t[j] - t[i] >= tsp.costs[i][j])
                for (i, j) in tsp.edges
                if j not in self.init_nodes
            ),
            "delay",
        )

        # for ii, I in enumerate(init_nodes):
        #     m.addConstrs(
        #         (x[i, I] == 1) >> (tT[ii, I] - t[i] >= 0)
        #         for i in non_init_nodes
        #         if (i, I) in tsp.edges
        #     )
        for ii, I in enumerate(init_nodes):
            m.addConstrs(
                (
                    (x[(i, I)] == 1) >> (tT[(ii, I)] - t[i] >= tsp.costs[i][I])
                    for i in non_init_nodes
                ),
                "delayTerm",
            )

        # m.addGenConstrMax(tf, list(tT.values()), name="tf_max")
        m.addGenConstrMax(tf, tT)

        # Region tracking: z[r]=1 iff any AP node in the region is visited.
        # Use all incoming edges to region nodes: z >= x[i,j] for each, z <= sum.
        for region_name, node_set in self.tsp_regions.items():
            if region_name not in self.z_vars:
                continue
            z = self.z_vars[region_name]
            incoming = [(i, j) for (i, j) in tsp.edges if j in node_set]
            for (i, j) in incoming:
                m.addConstr(z >= x[i, j], name=f"region_{region_name}_lb_{i}_{j}")
            if incoming:
                m.addConstr(
                    z <= gp.quicksum(x[i, j] for (i, j) in incoming),
                    name=f"region_{region_name}_ub",
                )

        m.update()

    def _add_optional_event_constraints(self, m: gp.Model, variables: Dict, tsp):
        if not self.ctpo or not self.ctpo.conditional_tpos:
            return

        x = variables["x"]
        t = variables["t"]

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

            print(f"Event {event_node} is always required: {always_required}")

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
                        m.addConstr((z == 0) >> (incoming_sum == 1), name=f"NOT_{rname}_req_{event_node}")
                        m.addConstr((z == 1) >> (incoming_sum == 0), name=f"opt_{event_node}_skip_if_{rname}")
                    else:
                        m.addConstr((z == 1) >> (incoming_sum == 1), name=f"{rname}_req_{event_node}")
                        m.addConstr((z == 0) >> (incoming_sum == 0), name=f"opt_{event_node}_skip_if_NOT_{rname}")

        # Conditional precedence constraints via indicator constraints on z.
        # z=1 → region visited; z=0 → not visited.
        # is_neg=True  → constraint active when z=0 (NOT visited)
        # is_neg=False → constraint active when z=1 (visited)
        for key, cond_tpo in self.ctpo.conditional_tpos.items():
            is_neg = self.ctpo.negated_regions.get(key, False)
            rname = key.replace("NOT_", "") if is_neg else key
            z = self.z_vars.get(rname)
            if z is None:
                continue
            zval = 0 if is_neg else 1
            for src, targets in cond_tpo.local_constraints.items():
                for tgt, bounds in targets.items():
                    lb, ub = bounds["lb"], bounds["ub"]
                    m.addConstr(
                        (z == zval) >> (t[tgt] - t[src] >= lb),
                        name=f"cond_prec_{key}_{src}_{tgt}_lb",
                    )
                    # if ub < 1e9:
                    m.addConstr(
                        (z == zval) >> (t[tgt] - t[src] <= ub),
                        name=f"cond_prec_{key}_{src}_{tgt}_ub",
                    )

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
