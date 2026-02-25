"""
Conditional Timed Partial Order

This module provides support for conditional temporal specifications:
- Base TPO: Always enforced (mandatory events)
- Conditional TPOs: Only enforced if certain regions are visited

Key Concept:
    "IF robot visits region R, THEN it must satisfy TPO_R"
"""

from typing import Dict, Set, Optional, TYPE_CHECKING

from specless.specification.timed_partial_order import TimedPartialOrder

if TYPE_CHECKING:
    from specless.utils.state_regions import StateRegions


class ConditionalTPO:
    """
    Manages conditional TPO specifications based on region visits.

    Architecture:
        Base TPO: Mandatory constraints (always enforced)
        Conditional TPOs: Region-triggered constraints (conditionally enforced)

    Usage:
        base_tpo = TimedPartialOrder.from_constraints(
            global_constraints={start_node: (0, 15)}
        )
        kitchen_tpo = TimedPartialOrder.from_constraints(
            local_constraints={(enter_node, wash_node): (0, 10)}
        )
        ctpo = ConditionalTPO(base_tpo, regions)
        ctpo.add_conditional_tpo("kitchen", kitchen_tpo)
    """

    def __init__(
        self,
        base_tpo: TimedPartialOrder,
        regions: "StateRegions"
    ):
        """
        Initialize conditional TPO.

        Args:
            base_tpo: Mandatory TPO (always enforced)
            regions: Region definitions for validation
        """
        self.base_tpo = base_tpo
        self.regions = regions
        self.conditional_tpos: Dict[str, TimedPartialOrder] = {}
        self.negated_regions: Dict[str, bool] = {}

    def add_conditional_tpo(
        self,
        region_name: str,
        tpo: TimedPartialOrder,
        negate: bool = False
    ):
        """
        Add a conditional TPO for a specific region.

        Args:
            region_name: Name of the triggering region
            tpo: TPO specification to enforce when condition is met
            negate: If True, enforce TPO when region is NOT visited
        """
        if region_name not in self.regions.regions:
            if not negate:
                print(f"WARNING: Region '{region_name}' not defined.")

        key = f"NOT_{region_name}" if negate else region_name
        self.conditional_tpos[key] = tpo
        self.negated_regions[key] = negate

    def get_all_mandatory_events(self) -> Set:
        """Get all events from the base TPO."""
        events = set()
        events.update(self.base_tpo.global_constraints.keys())
        for src, targets in self.base_tpo.local_constraints.items():
            events.add(src)
            events.update(targets.keys())
        return events

    def get_all_conditional_events(self) -> Set:
        """Get union of all events from conditional TPOs."""
        events = set()
        for tpo in self.conditional_tpos.values():
            events.update(tpo.global_constraints.keys())
            for src, targets in tpo.local_constraints.items():
                events.add(src)
                events.update(targets.keys())
        return events

    def get_conditional_local_edges(self) -> Set:
        """Return (src, tgt) pairs that appear only in conditional TPOs, not in base."""
        base_edges = {
            (src, tgt)
            for src, targets in self.base_tpo.local_constraints.items()
            for tgt in targets
        }
        cond_edges = set()
        for tpo in self.conditional_tpos.values():
            for src, targets in tpo.local_constraints.items():
                for tgt in targets:
                    if (src, tgt) not in base_edges:
                        cond_edges.add((src, tgt))
        return cond_edges

    def get_conditional_events_for_region(self, region_name: str) -> Set:
        """Get events from a specific region's conditional TPO."""
        if region_name not in self.conditional_tpos:
            return set()

        tpo = self.conditional_tpos[region_name]
        events = set()
        events.update(tpo.global_constraints.keys())
        for src, targets in tpo.local_constraints.items():
            events.add(src)
            events.update(targets.keys())
        return events

    def build_unified_tpo(self) -> TimedPartialOrder:
        """Build a unified TPO containing all constraints."""
        unified_local = {}
        for src, targets in self.base_tpo.local_constraints.items():
            unified_local[src] = dict(targets)
        unified_global = dict(self.base_tpo.global_constraints)

        print(f" Local constraints: {unified_local}")
        print(f" Global constraints: {unified_global}")

        for region_name, cond_tpo in self.conditional_tpos.items():
            for src, targets in cond_tpo.local_constraints.items():
                if src not in unified_local:
                    unified_local[src] = {}

                for tgt, bounds in targets.items():
                    if tgt in unified_local[src]:
                        existing = unified_local[src][tgt]
                        unified_local[src][tgt] = {
                            "lb": max(existing["lb"], bounds["lb"]),
                            "ub": min(existing["ub"], bounds["ub"])
                        }
                    else:
                        unified_local[src][tgt] = bounds

            for node, bounds in cond_tpo.global_constraints.items():
                if node in unified_global:
                    existing = unified_global[node]
                    unified_global[node] = {
                        "lb": max(existing["lb"], bounds["lb"]),
                        "ub": min(existing["ub"], bounds["ub"])
                    }
                else:
                    unified_global[node] = bounds

        unified_local_edges = {}
        for src, targets in unified_local.items():
            for tgt, bounds in targets.items():
                unified_local_edges[(src, tgt)] = (bounds["lb"], bounds["ub"])

        unified_global_nodes = {}
        for node, bounds in unified_global.items():
            unified_global_nodes[node] = (bounds["lb"], bounds["ub"])

        unified_tpo = TimedPartialOrder.from_constraints(
            global_constraints=unified_global_nodes,
            local_constraints=unified_local_edges
        )
        for dc in self.base_tpo.difference_constraints:
            unified_tpo.difference_constraints.append(dc)
        return unified_tpo

    def print_summary(self):
        """Print a summary of the conditional TPO structure."""
        print(f"\n{'='*20}")
        print("Conditional TPO Summary")
        print(f"{'='*20}")

        mandatory_events = self.get_all_mandatory_events()
        print(f"\nBase TPO (mandatory):")
        print(f"  Events: {sorted(mandatory_events)}")
        print(f"  Global constraints: {len(self.base_tpo.global_constraints)}")
        print(f"  Local constraints: {len(self.base_tpo.local_constraints)}")

        if self.conditional_tpos:
            print(f"\nConditional TPOs:")
            for region_name, tpo in sorted(self.conditional_tpos.items()):
                events = self.get_conditional_events_for_region(region_name)
                print(f"  Region: '{region_name}'")
                print(f"    Events: {sorted(events)}")
                print(f"    Global constraints: {len(tpo.global_constraints)}")
                print(f"    Local constraints: {len(tpo.local_constraints)}")

        all_events = self.get_all_mandatory_events() | self.get_all_conditional_events()
        print(f"\nUnified TPO:")
        print(f"  Total unique events: {len(all_events)}")
        print(f"  Events: {sorted(all_events)}")
