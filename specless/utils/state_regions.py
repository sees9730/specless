from typing import Dict, List, Set, Tuple, Optional

from specless.utils.conditional_tsp_mapper import ConditionalTSPMapper


class StateRegions:
    """
    Define regions over TransitionSystem states.

    A region is a named set of TS states representing a spatial area,
    logical zone, or any meaningful grouping.

    Example:
        kitchen_region = {state1, state2, state3}  # TS states in kitchen
        puddle_region = {state4, state5}  # TS states near puddle

    Usage:
        regions = StateRegions()
        regions.add_region("kitchen", {kitchen_state1, kitchen_state2})
        regions.add_region("bedroom", {bedroom_state1, bedroom_state2})
    """

    def __init__(self):
        """Initialize empty region definition."""
        self.regions: Dict[str, Set[Tuple]] = {}  # region_name → set of TS states

    def add_region(self, region_name: str, states: Set[Tuple]):
        """
        Define a region as a set of TransitionSystem states.

        Args:
            region_name: Name of the region (e.g., "kitchen", "puddle_area")
            states: Set of TS state tuples belonging to this region
        """
        self.regions[region_name] = set(states)

    def add_region_by_observation(
        self,
        region_name: str,
        observation: str,
        mapping: ConditionalTSPMapper
    ):
        """
        Define a region by observation label.

        All TS states with this observation become the region.

        Args:
            region_name: Name of the region
            observation: Observation label (e.g., "floor_blue" for puddle)
            mapping: ConditionalTSPMapper to lookup states by observation

        Example:
            # All states with "floor_blue" observation become "puddle_area" region
            regions.add_region_by_observation("puddle_area", "floor_blue", mapping)
        """
        if observation not in mapping.obs_to_states:
            raise ValueError(
                f"Observation '{observation}' not found in mapping.\n"
                f"Available: {sorted(mapping.obs_to_states.keys())}"
            )

        states = set(mapping.obs_to_states[observation])
        self.add_region(region_name, states)

    def add_region_by_observations(
        self,
        region_name: str,
        observations: List[str],
        mapping: ConditionalTSPMapper
    ):
        """
        Define a region by multiple observation labels.

        Union of all TS states with these observations becomes the region.

        Args:
            region_name: Name of the region
            observations: List of observation labels
            mapping: ConditionalTSPMapper to lookup states

        Example:
            # Kitchen has multiple observation types
            regions.add_region_by_observations(
                "kitchen",
                ["floor_grey", "counter_blue", "sink_white"],
                mapping
            )
        """
        states = set()
        for obs in observations:
            if obs not in mapping.obs_to_states:
                raise ValueError(
                    f"Observation '{obs}' not found in mapping.\n"
                    f"Available: {sorted(mapping.obs_to_states.keys())}"
                )
            states.update(mapping.obs_to_states[obs])

        self.add_region(region_name, states)

    def get_region_names(self) -> List[str]:
        """Get all region names."""
        return sorted(self.regions.keys())

    def to_tsp_regions(self, mapping: ConditionalTSPMapper) -> Dict[str, Set[int]]:
        """
        Convert DTS-level regions to TSP node-level regions.

        A TSP node touches a region if any of its TS states belongs to that region.

        Args:
            mapping: ConditionalTSPMapper to lookup TSP nodes by TS state

        Returns:
            Dict mapping region names to sets of TSP node IDs
        """
        tsp_regions: Dict[str, Set[int]] = {}
        for region_name, ts_states in self.regions.items():
            tsp_nodes: Set[int] = set()
            for tsp_node, node_states in mapping.node_to_states.items():
                if any(state in ts_states for state in node_states):
                    tsp_nodes.add(tsp_node)
            tsp_regions[region_name] = tsp_nodes
        return tsp_regions

    def print_summary(self, mapping: Optional[ConditionalTSPMapper] = None):
        """
        Print a summary of all regions.

        Args:
            mapping: Optional mapping to show TSP nodes in each region
        """
        for region_name, states in sorted(self.regions.items()):
            print(f" Region '{region_name}':")
            print(f"  TS states: {len(states)}")

            # Show sample states
            sample_states = list(states)[:3]
            for state in sample_states:
                if mapping and state in mapping.state_to_obs:
                    obs = mapping.state_to_obs[state]
                    print(f"    {state} - obs: '{obs}'")
                else:
                    print(f"    {state}")

            if len(states) > 3:
                print(f"    ... and {len(states) - 3} more states")

