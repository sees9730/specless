"""
Conditional TPO Planner - Level 1: Region Tracking

This implements the foundational level of conditional TPO planning,
which tracks which regions the robot visits without changing planning behavior yet.

Based on cTPOPlanner.md Level 1 specification.
"""

import sys
from typing import List, Set, Dict, Tuple, Optional
from contextlib import contextmanager
import os

import networkx as nx
import specless as sl
from specless.specification.timed_partial_order import TimedPartialOrder
import gurobipy as gp
from gurobipy import GRB


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ============================================================================
# TPO-based Edge Creation (from final_testing_detection.py)
# ============================================================================
# These functions create smart edges based on TPO structure instead of all O(n²) edges.
# This is essential because in DTS-based planning, TSP nodes represent events (not states),
# and we only want edges that make sense given the partial order constraints.

def has_precedence_path(tpo: TimedPartialOrder, src: int, tgt: int, visited: Set[int] = None) -> bool:
    """Check if there's a precedence path from src to tgt in TPO"""
    if visited is None:
        visited = set()

    if src == tgt:
        return True

    if src in visited:
        return False

    visited.add(src)

    # Direct edge
    if src in tpo.local_constraints and tgt in tpo.local_constraints[src]:
        return True

    # Transitive path
    if src in tpo.local_constraints:
        for next_node in tpo.local_constraints[src].keys():
            if has_precedence_path(tpo, next_node, tgt, visited):
                return True

    return False


def create_precedence_edges(nodes: List[int], tpo: TimedPartialOrder) -> List[Tuple[int, int]]:
    """
    Create edges based on precedence structure:
    - Edge (i, j) exists if there's a DIRECT TPO edge L(x_i) → L(x_j)
    - OR if no precedence exists in either direction (parallel)

    NOTE: We only create DIRECT TPO edges, not transitive ones.
    This prevents redundant edges like (0,7) when path 0→1→3→7 exists.

    This is crucial for DTS-based planning where TSP nodes represent events,
    and we only want edges that make sense given the partial order.
    """
    edges = []

    for i in nodes:
        for j in nodes:
            if i == j:
                continue

            # Check if DIRECT TPO edge exists i → j
            has_direct_edge = (i in tpo.local_constraints and
                              j in tpo.local_constraints[i])

            # Check if precedence exists in EITHER direction (for parallel detection)
            has_i_to_j = has_precedence_path(tpo, i, j)
            has_j_to_i = has_precedence_path(tpo, j, i)

            # Edge exists if:
            # 1. Direct TPO edge exists
            if has_direct_edge:
                edges.append((i, j))
            # 2. OR events are parallel (no precedence in either direction)
            elif not has_i_to_j and not has_j_to_i:
                edges.append((i, j))

    return edges


def get_all_nodes(tpo: TimedPartialOrder) -> List[int]:
    """Extract all nodes from TPO"""
    nodes = set()
    for src, targets in tpo.local_constraints.items():
        nodes.add(src)
        nodes.update(targets.keys())
    return sorted(nodes)


# ============================================================================
# Level 1: Region Tracking
# ============================================================================  

class RegionDefinition:
    """
    Container for region definitions.
    A region is a named set of nodes in the planning problem.
    """
    def __init__(self):
        self.regions: Dict[str, Set[int]] = {}

    def add_region(self, region_name: str, node_set: Set[int]):
        """
        Define a named region as a set of nodes.

        Args:
            region_name: string identifier (e.g., "kitchen", "hallway")
            node_set: set of node indices in this region
        """
        self.regions[region_name] = set(node_set)

    def get_region(self, region_name: str) -> Set[int]:
        """Get the set of nodes for a given region"""
        return self.regions.get(region_name, set())

    def get_region_for_node(self, node: int) -> List[str]:
        """Get all regions that contain a given node"""
        return [name for name, nodes in self.regions.items() if node in nodes]


class ConditionalTPO:
    """
    Level 1: Basic Conditional TPO with region tracking.

    At this level, we only track which regions are visited.
    Future levels will add conditional events and constraints.
    """
    def __init__(self, base_tpo: TimedPartialOrder, regions: RegionDefinition):
        """
        Initialize conditional TPO.

        Args:
            base_tpo: The mandatory TPO that must be satisfied
            regions: Region definitions for tracking
        """
        self.base_tpo = base_tpo
        self.regions = regions
        # Future levels will add:
        # self.conditional_events = {}
        # self.conditional_constraints = {}


class MILPTSPWithTPOAndRegionTracking(sl.MILPTSPWithTPOSolver):
    """
    Extended MILP solver that adds region tracking variables.

    This solver extends the base MILPTSPWithTPOSolver to add:
    - Binary variables z[region_name] indicating if a region is visited
    - Constraints linking z variables to edge variables
    """

    def __init__(self):
        super().__init__()
        self.z_vars = {}  # Will store region tracking variables
        self.region_values = {}  # Will store solution values
        self.model = None  # Will store the Gurobi model
        self.variables = None  # Will store the variables dict

    def solve(
        self,
        tsp: sl.TSPWithTPO,
        num_agent: int = 1,
        init_nodes: Optional[List[int]] = None,
        come_back_home: bool = False,
        export_filename: Optional[str] = None,
        regions: Optional[RegionDefinition] = None
    ) -> Tuple[List[List[int]], float, List[List[float]]]:
        """
        Solve TSP with TPO constraints and region tracking.

        Args:
            tsp: TSP with TPO problem instance
            num_agent: Number of agents
            init_nodes: Initial nodes for each agent
            come_back_home: Whether agents return to start
            export_filename: Optional filename to export model
            regions: Optional region definitions for tracking

        Returns:
            Tuple of (tours, cost, timestamps)
        """
        if regions is None:
            # No regions defined, use base solver
            return super().solve(tsp, num_agent, init_nodes, come_back_home, export_filename)

        # Setup agent and init_nodes (following parent class pattern)
        import copy
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

        # If not coming back home, zero out the return costs
        if not self.come_back_home:
            for init_node in init_nodes:
                for i in set(tsp.nodes) - set([init_node]):
                    tsp.costs[i][init_node] = 0

        # Create model and variables using parent class logic
        m, variables = self.initialize_problem(tsp)

        # Add region tracking variables
        self._add_region_variables(m, regions)

        # Add region tracking constraints
        self._add_region_tracking_constraints(m, variables, tsp, regions, num_agent)

        # Add all standard TSP+TPO constraints from parent class
        m = self._add_all_standard_constraints(m, variables, tsp, num_agent, init_nodes, come_back_home)

        # Set objective (minimize total travel time)
        objective = gp.quicksum(
            tsp.costs[i][j] * variables["x"][i, j]
            for (i, j) in tsp.edges
        )

        # Optimize
        self.optimize(m, variables, objective)

        # Store model and variables for debugging
        self.model = m
        self.variables = variables

        # Extract region values from solution
        self._extract_region_values(m)

        # Export if requested
        if export_filename:
            m.write(export_filename)

        # Get results
        tours = self.get_tours(m, variables)
        timestamps = [self.get_timestamps(m, variables, tour) for tour in tours]
        cost = self.get_cost(m)

        return tours, cost, timestamps

    def _add_region_variables(self, m: gp.Model, regions: RegionDefinition):
        """Add binary variables for each region"""
        self.z_vars = {}
        for region_name in regions.regions.keys():
            print(f"Adding z[{region_name}]")
            self.z_vars[region_name] = m.addVar(
                vtype=GRB.BINARY,
                name=f"z_{region_name}"
            )
        m.update()

    def _add_region_tracking_constraints(
        self,
        m: gp.Model,
        variables: Dict,
        tsp: sl.TSPWithTPO,
        regions: RegionDefinition,
        num_agent: int
    ):
        """
        Link region binary variables to edge variables.

        If any edge touches a region (either endpoint in the region),
        that region's z variable must be 1.
        """
        x = variables["x"]  # Edge variables: x[i, j] for edges in tsp.edges

        print(f"All x variables: {list(x.keys())}")

        for region_name, node_set in regions.regions.items():
            z = self.z_vars[region_name]

            # Track edges touching this region
            touching_edges = []

            # For each edge in the problem
            for (i, j) in tsp.edges:
                # If either endpoint is in the region
                if i in node_set or j in node_set:
                    touching_edges.append((i, j))
                    # z[region] must be 1 if this edge is used
                    # z >= x[i, j]
                    m.addConstr(
                        z >= x[i, j],
                        name=f"region_{region_name}_edge_i{i}_j{j}"
                    )

            # CRITICAL: If region is activated, at least one touching edge must be used
            # This prevents z from being arbitrarily set to 1 when no edges are used
            # z <= sum of all edge variables touching this region
            if touching_edges:
                m.addConstr(
                    z <= gp.quicksum(x[i, j] for (i, j) in touching_edges),
                    name=f"region_{region_name}_requires_edge"
                )

    def _add_all_standard_constraints(
        self,
        m: gp.Model,
        variables: Dict,
        tsp: sl.TSPWithTPO,
        num_agent: int,
        init_nodes: Optional[List[int]],
        come_back_home: bool
    ) -> gp.Model:
        """
        Add all standard TSP+TPO constraints.
        This replicates the logic from the parent class solve() method.
        """
        x = variables["x"]  # Edge variables
        t = variables["t"]  # Time variables
        tT = variables["tT"]  # Terminal time variables
        tf = variables["tf"]  # Final time variable

        non_init_nodes = list(set(tsp.nodes) - set(self.init_nodes))

        # 1. GTSP constraints: visit one node from each nodeset
        for nodes in tsp.nodesets:
            K = sum(init_node in nodes for init_node in self.init_nodes)
            if K == 0:
                K = 1
            # Incoming and outgoing flow must be equal
            m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) == K, "incoming")
            m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) == K, "outgoing")

        # 2. Flow conservation
        m.addConstrs((x.sum("*", n) == x.sum(n, "*") for n in tsp.nodes), "flow")

        # 3. Time window constraints (global)
        m.addConstrs(
            (
                t[n] >= tsp.tpo.global_constraints[n]["lb"]
                for n in tsp.tpo.global_constraints.keys()
            ),
            "nodeLB",
        )
        m.addConstrs(
            (
                t[n] <= tsp.tpo.global_constraints[n]["ub"]
                for n in tsp.tpo.global_constraints.keys()
            ),
            "nodeUB",
        )

        # 4. Precedence constraints (local)
        local_const_edges = [
            (src, tgt)
            for src, d in tsp.tpo.local_constraints.items()
            for tgt, b in d.items()
        ]
        m.addConstrs(
            (
                t[tgt] - t[src] >= tsp.tpo.local_constraints[src][tgt]["lb"]
                for (src, tgt) in local_const_edges
            ),
            "localLB",
        )
        m.addConstrs(
            (
                t[tgt] - t[src] <= tsp.tpo.local_constraints[src][tgt]["ub"]
                for (src, tgt) in local_const_edges
            ),
            "localUB",
        )

        # 5. Time progression constraints
        M = 100000
        for (i, j) in tsp.edges:
            if i in self.init_nodes:
                # Initial nodes start at time 0
                m.addConstr(t[i] == 0, name=f"init_time_{i}")
            if j not in self.init_nodes:
                # If edge used, propagate time with travel cost
                m.addConstr(
                    t[j] >= t[i] + tsp.costs[i][j] - M * (1 - x[i, j]),
                    name=f"time_prop_{i}_{j}"
                )

        # 6. Terminal time tracking
        for idx, init_node in enumerate(self.init_nodes):
            # Track terminal time for each agent's end
            m.addConstr(
                tT[idx, init_node] >= gp.quicksum(
                    t[j] * x[j, init_node] for j in non_init_nodes if (j, init_node) in tsp.edges
                ),
                name=f"terminal_time_{idx}_{init_node}"
            )

        # 7. Final time is max of all terminal times
        for idx, init_node in enumerate(self.init_nodes):
            m.addConstr(tf >= tT[idx, init_node], name=f"final_time_{idx}_{init_node}")

        m.update()
        return m

    def _extract_region_values(self, m: gp.Model):
        """Extract region tracking variable values from solution"""
        self.region_values = {}
        for region_name, z_var in self.z_vars.items():
            self.region_values[region_name] = z_var.X

    def get_visited_regions(self) -> Dict[str, bool]:
        """
        Get which regions were visited in the solution.

        Returns:
            Dict mapping region names to whether they were visited
        """
        return {name: value > 0.5 for name, value in self.region_values.items()}

    def print_visited_regions(self):
        """Print which regions were visited"""
        print("\n=== Regions Visited ===")
        for region_name in sorted(self.region_values.keys()):
            if self.region_values[region_name] > 0.5:
                print(f"! Robot passed through: {region_name}")
            else:
                print(f"! Robot avoided: {region_name}")


# ============================================================================
# Level 1 Test: Region Tracking
# ============================================================================

def test_level1_region_tracking():
    """
    Test Level 1: Region tracking without conditional behavior.

    Creates a simple grid-like problem with defined regions and verifies
    that the solver correctly tracks which regions are visited.
    """
    print("="*70)
    print("Level 1 Test: Region Tracking")
    print("="*70)

    # Define a simple problem with 10 nodes
    # Layout (conceptual):
    #   0 (start) -> 1,2,3 (kitchen) -> 4,5,6 (hallway) -> 7,8,9 (bedroom)

    nodes = list(range(10))

    # Define regions
    regions = RegionDefinition()
    regions.add_region("kitchen", {1, 2, 3})
    regions.add_region("hallway", {4, 5, 6})
    regions.add_region("bedroom", {7, 8, 9})

    print(f"\nDefined regions:")
    for name, node_set in regions.regions.items():
        print(f"  {name}: {sorted(node_set)}")

    # Create simple cost matrix (euclidean-like distances)
    # Make it so that going through kitchen is optimal for reaching bedroom
    costs = [[0.0] * 10 for _ in range(10)]
    for i in range(10):
        for j in range(10):
            if i == j:
                costs[i][j] = 0
            else:
                # Sequential layout: cost is roughly based on distance
                costs[i][j] = abs(j - i) * 1.0

    # Make a path that avoids hallway attractive by adding shortcuts
    costs[3][7] = 2.0  # Shortcut from kitchen to bedroom (avoiding hallway)

    print("\nCost matrix:")
    for i in range(10):
        print(f"  {costs[i]}")

    # Define TPO: must visit node 0 (start), node 2 (kitchen), and node 8 (bedroom)
    local_constraints = {
        (0, 2): (0, 100),  # Visit kitchen after start
        (2, 8): (0, 100),  # Visit bedroom after kitchen
    }

    tpo = TimedPartialOrder.from_constraints(
        global_constraints={},
        local_constraints=local_constraints
    )

    sl.draw_graph(tpo, 'Level 1 TPO Problem')

    print(f"\nTPO specification:")
    print(f"  Must visit: 0 (start) -> 2 (kitchen) -> 8 (bedroom)")

    # ============================================================================
    # IMPORTANT: Build smart edges based on TPO structure
    # Instead of all O(n²) edges, only create edges that make sense
    # This is how it works in real DTS-based planning where nodes are events
    # ============================================================================

    print(f"\nBuilding smart edges based on TPO structure...")

    # Get all nodes from TPO (events that must be visited)
    tpo_nodes = get_all_nodes(tpo)

    # But we also need nodes that aren't in TPO constraints but exist in the problem
    # For this test, all nodes 0-9 are valid events
    all_nodes = nodes

    # Create precedence-based edges (direct TPO edges + parallel edges)
    precedence_edges = create_precedence_edges(all_nodes, tpo)

    # Identify initial and final nodes for flow constraints
    initial_nodes = [n for n in all_nodes
                    if not any(has_precedence_path(tpo, other, n) for other in all_nodes if other != n)]
    final_nodes = [n for n in all_nodes
                  if not any(has_precedence_path(tpo, n, other) for other in all_nodes if other != n)]

    print(f"  Initial nodes (no predecessors): {initial_nodes}")
    print(f"  Final nodes (no successors): {final_nodes}")
    print(f"  Precedence edges created: {len(precedence_edges)}")

    # Add return edges for flow constraints (from final to initial)
    allowed_edges = precedence_edges.copy()
    print(f"  Adding return edges for flow constraints...")
    for final_node in final_nodes:
        for init_node in initial_nodes:
            # IMPORTANT: Skip self-loops! A node can be both initial and final
            # (e.g., isolated nodes with no predecessors or successors in TPO)
            if final_node != init_node and (final_node, init_node) not in allowed_edges:
                allowed_edges.append((final_node, init_node))

    print(f"  Total edges (with returns): {len(allowed_edges)}")
    print(f"  Edge reduction: {len(nodes)**2 - len(nodes) - len(allowed_edges)} edges eliminated")
    print(f"  (Full graph would have {len(nodes)**2 - len(nodes)} edges)")

    # Create TSP problem with filtered edges
    tsp = sl.TSPWithTPO(nodes, costs, tpo)
    tsp.edges = allowed_edges  # Set the filtered edge list

    # IMPORTANT: Only include TPO nodes in nodesets
    # Nodes not in nodesets can be used as waypoints but don't need to be visited
    # In DTS-based planning, only events mentioned in the TPO have TSP nodes
    tsp.nodesets = [[n] for n in tpo_nodes]  # Only visit TPO-required nodes

    print(f"  Nodes that must be visited (in nodesets): {tpo_nodes}")
    print(f"  Nodes that can be used as waypoints: {sorted(set(nodes) - set(tpo_nodes))}")

    tsp_graph = nx.DiGraph()
    tsp_graph.add_edges_from(allowed_edges)
    sl.draw_graph(tsp_graph, 'Level 1 TSP problem')

    # Create conditional TPO (just for structure, no conditionals yet)
    ctpo = ConditionalTPO(base_tpo=tpo, regions=regions)

    # Solve with region tracking
    print(f"\nSolving with region tracking...")
    solver = MILPTSPWithTPOAndRegionTracking()

    # with suppress_output():
    tours, cost, timestamps = solver.solve(
        tsp,
        num_agent=1,
        init_nodes=[0],
        come_back_home=False,
        regions=regions
    )

    print(f"\nSolution:")
    print(f"  Tour: {tours[0]}")
    print(f"  Cost: {cost:.2f}")
    print(f"  Timestamps: {[f'{t:.2f}' for t in timestamps[0]]}")

    # Debug: show which edges are active (uncomment for debugging)
    # debug_active_edges(solver, regions)

    # Print which regions were visited
    solver.print_visited_regions()

    # Validation
    visited_regions = solver.get_visited_regions()

    print(f"\n{'='*70}")
    print("Validation")
    print(f"{'='*70}")

    # Verify that regions containing nodes in the tour are marked as visited
    tour_nodes = set(tours[0])

    for region_name, node_set in regions.regions.items():
        expected_visited = bool(tour_nodes & node_set)  # Intersection
        actual_visited = visited_regions[region_name]

        match = "" if expected_visited == actual_visited else ""
        print(f"{match} {region_name}: expected={expected_visited}, actual={actual_visited}")

        if expected_visited == actual_visited:
            if expected_visited:
                overlapping_nodes = sorted(tour_nodes & node_set)
                print(f"     Tour visits nodes {overlapping_nodes} in {region_name}")
        else:
            print(f"     ERROR: Mismatch for region {region_name}")
            return False

    print(f"\n{'='*70}")
    print(" Level 1 Test Passed: Region tracking works correctly!")
    print(f"{'='*70}")
    return True


def debug_active_edges(solver, regions):
    """Debug function to print which edges are active"""
    if solver.model is None or solver.variables is None:
        print("ERROR: Model not available. Run solve() first.")
        return

    x = solver.variables["x"]
    vals = solver.model.getAttr("X", x)

    print("\n=== DEBUG: Active Edges ===")
    for (i, j), val in vals.items():
        if val > 0.5:
            # This edge is used
            touching_regions = []
            for region_name, node_set in regions.regions.items():
                if i in node_set or j in node_set:
                    touching_regions.append(region_name)

            if touching_regions:
                print(f"  Edge ({i:2d}, {j:2d}): ACTIVE, touches {touching_regions}")
            else:
                print(f"  Edge ({i:2d}, {j:2d}): ACTIVE, no regions")

    print("\n=== DEBUG: Region Variable Values ===")
    for region_name, z_val in solver.region_values.items():
        print(f"  z[{region_name}] = {z_val:.3f}")

if __name__ == "__main__":
    success = test_level1_region_tracking()
    sys.exit(0 if success else 1)
