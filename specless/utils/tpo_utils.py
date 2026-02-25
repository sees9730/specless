"""
TPO Graph Utilities

Helper functions for working with TimedPartialOrder graph structure:
- Precedence path queries
- Node extraction
- Edge generation based on precedence structure
"""

from typing import List, Optional, Set, Tuple, FrozenSet

from specless.specification.timed_partial_order import TimedPartialOrder


def has_precedence_path(
    tpo: TimedPartialOrder,
    src: int,
    tgt: int,
    visited: Optional[Set[int]] = None,
    skip_edges: Optional[Set[Tuple[int, int]]] = None,
) -> bool:
    """Return True if there is a (transitive) precedence path from src to tgt.

    Args:
        skip_edges: Optional set of (src, tgt) pairs to treat as non-existent.
                    Used to exclude conditional-only edges from topology queries.
    """
    if visited is None:
        visited = set()
    if skip_edges is None:
        skip_edges = set()
    if src == tgt:
        return True
    if src in visited:
        return False
    visited.add(src)
    if src in tpo.local_constraints and tgt in tpo.local_constraints[src]:
        if (src, tgt) not in skip_edges:
            return True
    for next_node in tpo.local_constraints.get(src, {}):
        if (src, next_node) not in skip_edges:
            if has_precedence_path(tpo, next_node, tgt, visited, skip_edges):
                return True
    return False


def get_tpo_nodes(tpo: TimedPartialOrder) -> List[int]:
    """Return all node IDs referenced in the TPO, sorted."""
    nodes: Set[int] = set(tpo.global_constraints.keys())
    for src, targets in tpo.local_constraints.items():
        nodes.add(src)
        nodes.update(targets.keys())
    return sorted(nodes)


def create_precedence_edges(
    nodes: List[int],
    tpo: TimedPartialOrder,
    conditional_edges: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Return edges compatible with the TPO precedence structure.

    An edge (i, j) is included when:
    - A direct TPO edge i → j exists, OR
    - i and j are unordered (no precedence in either direction)

    Transitive edges are excluded to avoid redundancy.

    Args:
        conditional_edges: (src, tgt) pairs from conditional TPOs only.
                           These are skipped when computing topology so that
                           conditional constraints don't restrict TSP ordering.
    """
    if conditional_edges is None:
        conditional_edges = set()
    edges = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            has_direct = (
                i in tpo.local_constraints
                and j in tpo.local_constraints[i]
                and (i, j) not in conditional_edges
            )
            has_i_to_j = has_precedence_path(tpo, i, j, skip_edges=conditional_edges)
            has_j_to_i = has_precedence_path(tpo, j, i, skip_edges=conditional_edges)
            if has_direct or (not has_i_to_j and not has_j_to_i):
                edges.append((i, j))
    return edges
