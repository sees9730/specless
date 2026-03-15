"""
Conditional TSP Mapper

This module provides mapping utilities for TransitionSystem to TSP with conditional events.
Supports:
- Mandatory events (must visit)
- Optional events (visit only if triggered by region)

Integrates with TransitionSystem from specless.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from bidict import bidict
import copy
import networkx as nx

from specless.automaton.transition_system import TransitionSystem


class ConditionalTSPMapper:
    """
    Mapping between TransitionSystem and TSP with conditional events.

    This class works with TransitionSystem and extends the TSPBuilder concept
    to distinguish between:
    - Mandatory events: Must be visited
    - Optional events: Only visited if triggered

    Key Concept:
        TransitionSystem State → Observation (Event) → TSP Node

        Multiple TS states can share the same observation/event.
        Each unique observation becomes a TSP node (or not, if it's just a waypoint).

    Example:
        TS States: kitchen_sink, kitchen_counter (both have obs="kitchen")
        - Single TSP node for "kitchen" event
        - Nodeset contains both state IDs
    """

    def __init__(
        self,
        transition_system: TransitionSystem,
        mandatory_events: List[str],
        optional_events: Optional[List[str]] = None,
        ap_events: Optional[List[str]] = None,
        ignoring_obs_keys: List[str] = []
    ):
        """
        Initialize mapping from TransitionSystem to TSP with optional events.

        Args:
            transition_system: The TS
            mandatory_events: Event observations that must be visited
            optional_events: Event observations that are conditionally required
            ap_events: Atomic proposition waypoint observations (e.g. lava tiles).
                       These get TSP node IDs for edge cost computation and region
                       tracking, but carry no visit-count constraint (d_i = 0).
            ignoring_obs_keys: Observation substrings to ignore (e.g., ["empty", "wall"])
        """
        self.ts = transition_system
        self.mandatory_events = set(mandatory_events)
        self.optional_events = set(optional_events) if optional_events else set()
        self.ap_events: Set[str] = set(ap_events) if ap_events else set()
        self.ignoring_obs_keys = ignoring_obs_keys

        # Core mappings
        self.state_to_obs: Dict[Tuple, str] = {}  # TS state → observation
        self.obs_to_states: Dict[str, List[Tuple]] = defaultdict(list)  # Obs → TS states

        # TSP node mappings (mandatory + optional share obs; AP nodes are per-state)
        self.obs_to_node: Dict[str, int] = {}  # Observation → TSP node ID
        self.node_to_obs: Dict[int, str] = {}  # TSP node ID → Observation
        self.node_to_states: Dict[int, List[Tuple]] = {}  # TSP node → TS states
        self.state_to_node: Dict[Tuple, Optional[int]] = {}  # TS state → TSP node (or None)

        # Set of TSP node IDs that are AP waypoints (one per DTS state)
        self._ap_node_ids: Set[int] = set()

        # Build all mappings
        self._build_mappings()

    def _build_mappings(self):
        """
        Build mappings from TransitionSystem to TSP nodes.

        Step 0: Add initial state 
        Step 1: Map all TS states to their observations
        Step 2: Create TSP nodes only for mandatory + optional events
        Step 3: Mark other states as waypoints (no TSP node)
        """
        # Add initial state
        initial_state = self.ts.start_state
        initial_obs = "initial_state0"

        self.state_to_obs[initial_state] = initial_obs
        self.obs_to_states[initial_obs].append(initial_state)

        # Map all TS states to observations
        for state in self.ts.nodes:
            obs = self.ts.observe(state)

            # Skip ignored observations
            if any(key in obs for key in self.ignoring_obs_keys):
                continue

            if obs == "":
                continue

            # Record this state's observation
            self.state_to_obs[state] = obs
            self.obs_to_states[obs].append(state)

        # Create TSP nodes for mandatory + optional events (one node per observation)
        non_ap_events = self.mandatory_events | self.optional_events

        tsp_node_id = 0
        for obs in sorted(non_ap_events):  # Sort for consistent ordering
            if obs not in self.obs_to_states:
                raise ValueError(
                    f"Event '{obs}' is required but no TS state has this observation!\n"
                    f"Available observations: {sorted(self.obs_to_states.keys())}"
                )

            states = self.obs_to_states[obs]

            self.obs_to_node[obs] = tsp_node_id
            self.node_to_obs[tsp_node_id] = obs
            self.node_to_states[tsp_node_id] = states

            for state in states:
                self.state_to_node[state] = tsp_node_id

            tsp_node_id += 1

        # Create one TSP node per individual DTS state for AP events.
        # Each lava tile (etc.) becomes its own node so the solver can
        # route through specific tiles and region detection fires correctly.
        for obs in sorted(self.ap_events):
            if obs not in self.obs_to_states:
                continue  # AP events absent in this environment are silently skipped

            for idx, state in enumerate(sorted(self.obs_to_states[obs])):
                synthetic_obs = f"{obs}_{idx}"

                self.obs_to_node[synthetic_obs] = tsp_node_id
                self.node_to_obs[tsp_node_id] = synthetic_obs
                self.node_to_states[tsp_node_id] = [state]
                self.state_to_node[state] = tsp_node_id
                self._ap_node_ids.add(tsp_node_id)

                tsp_node_id += 1

        # Mark non-TSP states as waypoints
        for state in self.state_to_obs.keys():
            if state not in self.state_to_node:
                self.state_to_node[state] = None  # Waypoint only

    def get_tsp_nodes(self) -> List[int]:
        """Get all TSP node IDs (mandatory + optional)."""
        return sorted(self.node_to_obs.keys())

    def get_nodesets(self) -> List[List[Tuple]]:
        """
        Get GTSP nodesets for the TSP problem.

        Each nodeset contains TS state IDs that satisfy an event.
        This is used by GTSP to allow visiting any state in the set.

        Returns:
            List of nodesets, where each nodeset is a list of TS states
        """
        nodesets = []
        for tsp_node in sorted(self.node_to_states.keys()):
            state_list = self.node_to_states[tsp_node]
            nodesets.append(state_list)
        return nodesets

    def is_event_mandatory(self, event_name: str) -> bool:
        """Check if an event is mandatory (must visit)."""
        return event_name in self.mandatory_events

    def is_event_optional(self, event_name: str) -> bool:
        """Check if an event is optional (conditional)."""
        return event_name in self.optional_events

    def is_node_mandatory(self, tsp_node: int) -> bool:
        """Check if a TSP node represents a mandatory event."""
        obs = self.node_to_obs.get(tsp_node)
        return obs in self.mandatory_events if obs else False

    def is_node_optional(self, tsp_node: int) -> bool:
        """Check if a TSP node represents an optional event."""
        obs = self.node_to_obs.get(tsp_node)
        return obs in self.optional_events if obs else False

    def get_mandatory_nodes(self) -> List[int]:
        """Get TSP nodes for mandatory events."""
        return [node for node in self.get_tsp_nodes()
                if self.is_node_mandatory(node)]

    def get_optional_nodes(self) -> List[int]:
        """Get TSP nodes for optional events."""
        return [node for node in self.get_tsp_nodes()
                if self.is_node_optional(node)]

    def get_ap_nodes(self) -> List[int]:
        """Get TSP nodes for AP waypoint events (one per DTS state)."""
        return sorted(self._ap_node_ids)

    def is_node_ap_waypoint(self, tsp_node: int) -> bool:
        """Check if a TSP node represents an AP waypoint (not mandatory/optional)."""
        return tsp_node in self._ap_node_ids

    def print_summary(self):
        """Print a summary of the mapping."""
        print(" Conditional TSP Mapping Summary:")

        print(f"   Total TS states: {len(self.state_to_obs)}")
        print(f"   Unique observations: {len(self.obs_to_states)}")
        print(f"   TSP nodes created: {len(self.get_tsp_nodes())}")

        print(f" Mandatory events ({len(self.mandatory_events)}):")
        for event in sorted(self.mandatory_events):
            if event in self.obs_to_node:
                node_id = self.obs_to_node[event]
                num_states = len(self.node_to_states[node_id])
                print(f"   Node {node_id}: '{event}' ({num_states} TS states)")

        if self.optional_events:
            print(f" Optional events ({len(self.optional_events)}):")
            for event in sorted(self.optional_events):
                if event in self.obs_to_node:
                    node_id = self.obs_to_node[event]
                    num_states = len(self.node_to_states[node_id])
                    print(f"   Node {node_id}: '{event}' ({num_states} TS states)")

        if self.ap_events:
            ap_nodes = self.get_ap_nodes()
            print(f" AP waypoint events ({len(self.ap_events)} obs, {len(ap_nodes)} nodes):")
            for event in sorted(self.ap_events):
                ap_for_obs = [n for n in ap_nodes
                              if self.node_to_obs[n].startswith(event + "_")]
                if ap_for_obs:
                    print(f"   '{event}': {len(ap_for_obs)} nodes → {ap_for_obs}")
                else:
                    print(f"   '{event}' (not present in this environment)")

        # Show waypoint observations (no TSP node at all)
        waypoint_obs = set(self.obs_to_states.keys()) - self.mandatory_events - self.optional_events - self.ap_events
        if waypoint_obs:
            print(f" Waypoint observations (no TSP node): {sorted(waypoint_obs)}")
