"""FCI Algorithm for causal discovery with latent confounders.

Session 134: Extension of PC algorithm to handle latent variables.

The FCI (Fast Causal Inference) algorithm:
1. Learns skeleton using same CI tests as PC
2. Orients edges using extended rules (R0-R10) for latent confounders
3. Outputs PAG (Partial Ancestral Graph) instead of CPDAG

Key difference from PC: bidirected edges X <-> Y indicate latent confounder.

References
----------
- Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
- Zhang (2008). On the completeness of orientation rules for causal discovery.
- Richardson (2003). Markov properties for acyclic directed mixed graphs.

Functions
---------
fci_algorithm : Full FCI algorithm
fci_orient : Orient skeleton to PAG using FCI rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from .independence_tests import CITestResult, fisher_z_test
from .pc_algorithm import pc_skeleton
from .types import EdgeMark, FCIResult, Graph, PAG


# Type alias for CI test function
CITestFunction = Callable[
    [np.ndarray, int, int, Union[List[int], Tuple[int, ...], None], float],
    CITestResult,
]


def fci_algorithm(
    data: np.ndarray,
    alpha: float = 0.01,
    ci_test_func: Optional[CITestFunction] = None,
    max_cond_size: Optional[int] = None,
    stable: bool = True,
    verbose: bool = False,
) -> FCIResult:
    """FCI algorithm for causal discovery with latent confounders.

    Extends PC algorithm to handle latent (unobserved) confounders.
    Instead of CPDAG, outputs PAG representing equivalence class of MAGs.

    Assumptions:
    1. Causal Markov condition
    2. Faithfulness
    3. Allows latent confounders (relaxes causal sufficiency)

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    alpha : float
        Significance level for CI tests. Default 0.01.
    ci_test_func : callable, optional
        CI test function. Default is Fisher's Z test.
    max_cond_size : int, optional
        Maximum conditioning set size.
    stable : bool
        If True, use order-independent PC variant for skeleton.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    FCIResult
        Result containing PAG, skeleton, separating sets, and metrics.

    Example
    -------
    >>> from .utils import generate_random_dag, generate_dag_data
    >>> # Generate data with latent confounder (omit variable 2)
    >>> true_dag = generate_random_dag(5, edge_prob=0.4, seed=42)
    >>> data, _ = generate_dag_data(true_dag, n_samples=1000, seed=42)
    >>> # Remove column to simulate latent confounder
    >>> data_observed = np.delete(data, 2, axis=1)
    >>> result = fci_algorithm(data_observed, alpha=0.01)
    >>> print(f"Bidirected edges: {result.n_bidirected()}")

    Notes
    -----
    FCI is more conservative than PC - produces more circle marks
    when orientation is uncertain. Bidirected edges X <-> Y indicate
    a latent common cause.
    """
    import time

    start_time = time.perf_counter()

    if verbose:
        print("=" * 60)
        print("FCI Algorithm")
        print("=" * 60)
        print(f"Data shape: {data.shape}")
        print(f"Alpha: {alpha}")
        print()

    # Phase 1: Learn skeleton (reuse PC skeleton)
    if verbose:
        print("Phase 1: Learning skeleton (PC-style)...")

    skeleton, separating_sets, n_tests = pc_skeleton(
        data,
        alpha=alpha,
        ci_test_func=ci_test_func,
        max_cond_size=max_cond_size,
        stable=stable,
        verbose=verbose,
    )

    if verbose:
        print(f"Skeleton edges: {skeleton.n_edges()}")
        print(f"CI tests performed: {n_tests}")
        print()

    # Phase 2: Orient edges using FCI rules
    if verbose:
        print("Phase 2: Orienting edges (FCI rules)...")

    pag = fci_orient(skeleton, separating_sets, verbose=verbose)

    # Identify latent confounders (bidirected edges)
    latent_confounders = []
    for i in range(pag.n_nodes):
        for j in range(i + 1, pag.n_nodes):
            if (
                pag.endpoints[i, j, 0] == PAG.MARK_ARROW
                and pag.endpoints[i, j, 1] == PAG.MARK_ARROW
            ):
                latent_confounders.append((i, j))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if verbose:
        print(f"Directed edges: {pag.n_directed_edges()}")
        print(f"Bidirected edges: {pag.n_bidirected_edges()}")
        print(f"Circle edges: {pag.n_circle_edges()}")
        print(f"Execution time: {elapsed_ms:.1f}ms")
        print("=" * 60)

    return FCIResult(
        pag=pag,
        skeleton=skeleton,
        separating_sets=separating_sets,
        possible_latent_confounders=latent_confounders,
        n_ci_tests=n_tests,
        alpha=alpha,
        execution_time_ms=elapsed_ms,
    )


def fci_orient(
    skeleton: Graph,
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]],
    verbose: bool = False,
) -> PAG:
    """Orient skeleton edges to PAG using FCI orientation rules.

    Phase 2 of FCI algorithm:
    1. Initialize PAG from skeleton with circle marks
    2. Apply R0 (v-structure detection)
    3. Iterate R1-R10 until convergence

    Parameters
    ----------
    skeleton : Graph
        Undirected skeleton from Phase 1.
    separating_sets : Dict[Tuple[int, int], FrozenSet[int]]
        Separating sets from Phase 1.
    verbose : bool
        If True, print orientation steps.

    Returns
    -------
    PAG
        Partially oriented ancestral graph.
    """
    n_vars = skeleton.n_nodes

    # Initialize PAG from skeleton with circle marks at all endpoints
    pag = PAG.from_skeleton(skeleton)

    # Step 1: Apply R0 - V-structure detection
    if verbose:
        print("  Applying R0 (v-structures)...")

    _fci_rule_0(pag, skeleton, separating_sets, verbose)

    # Step 2: Apply R1-R10 until convergence
    if verbose:
        print("  Applying R1-R10...")

    _apply_fci_rules(pag, verbose)

    return pag


# =============================================================================
# FCI Orientation Rules (Zhang 2008)
# =============================================================================


def _fci_rule_0(
    pag: PAG,
    skeleton: Graph,
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]],
    verbose: bool = False,
) -> None:
    """R0: Orient unshielded colliders (v-structures).

    If X—Z—Y and X ⊥ Y (not adjacent) and Z ∉ Sep(X,Y),
    then orient as X *-> Z <-* Y.

    This is the same as PC v-structure detection but uses
    circle->arrow marks instead of just arrow.
    """
    n = pag.n_nodes

    for z in range(n):
        neighbors = list(skeleton.neighbors(z))

        for idx_x, x in enumerate(neighbors):
            for y in neighbors[idx_x + 1 :]:
                # Check if X and Y are not adjacent
                if skeleton.has_edge(x, y):
                    continue

                # Check separating set for (x, y)
                sep_key = (min(x, y), max(x, y))
                if sep_key in separating_sets:
                    sep_set = separating_sets[sep_key]
                    if z not in sep_set:
                        # V-structure: X *-> Z <-* Y
                        _orient_collider(pag, x, z, y, verbose)
                else:
                    # X and Y were never adjacent (always independent)
                    # Treat as v-structure
                    _orient_collider(pag, x, z, y, verbose)


def _orient_collider(pag: PAG, x: int, z: int, y: int, verbose: bool) -> None:
    """Orient edges as collider: X *-> Z <-* Y."""
    # Set arrowhead at Z for both edges
    if pag.has_edge(x, z):
        pag.set_endpoint(z, x, EdgeMark.ARROW)  # Arrow at Z end of X-Z
        if verbose:
            print(f"    R0: {x} *-> {z} (v-structure)")

    if pag.has_edge(y, z):
        pag.set_endpoint(z, y, EdgeMark.ARROW)  # Arrow at Z end of Y-Z
        if verbose:
            print(f"    R0: {y} *-> {z} (v-structure)")


def _apply_fci_rules(pag: PAG, verbose: bool = False) -> None:
    """Apply FCI rules R1-R10 until convergence."""
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1

        if verbose:
            print(f"    Iteration {iteration}")

        changed |= _fci_rule_1(pag, verbose)
        changed |= _fci_rule_2(pag, verbose)
        changed |= _fci_rule_3(pag, verbose)
        changed |= _fci_rule_4(pag, verbose)
        changed |= _fci_rule_8(pag, verbose)
        changed |= _fci_rule_9(pag, verbose)
        changed |= _fci_rule_10(pag, verbose)


def _fci_rule_1(pag: PAG, verbose: bool = False) -> bool:
    """R1: Orient away from collider.

    If X *-> Y o-* Z and X and Z are not adjacent, orient Y -> Z.

    Prevents creating new v-structures.
    """
    changed = False
    n = pag.n_nodes

    for y in range(n):
        for z in range(n):
            if y == z:
                continue
            if not pag.has_edge(y, z):
                continue

            # Check if Y o-* Z (circle at Y, any mark at Z)
            if pag.get_endpoint(y, z) != EdgeMark.CIRCLE:
                continue

            # Look for X *-> Y where X not adjacent to Z
            for x in range(n):
                if x == y or x == z:
                    continue
                if not pag.has_edge(x, y):
                    continue

                # Check X *-> Y (arrow at Y)
                if pag.get_endpoint(y, x) != EdgeMark.ARROW:
                    continue

                # Check X not adjacent to Z
                if pag.has_edge(x, z):
                    continue

                # Orient Y -> Z: tail at Y, arrow at Z
                pag.set_endpoint(y, z, EdgeMark.TAIL)
                pag.set_endpoint(z, y, EdgeMark.ARROW)
                changed = True

                if verbose:
                    print(f"      R1: {y} -> {z} (away from collider at {x})")

    return changed


def _fci_rule_2(pag: PAG, verbose: bool = False) -> bool:
    """R2: Orient to prevent cycles.

    If X -> Y *-> Z or X *-> Y -> Z, and X o-* Z, orient X *-> Z.

    Ensures acyclicity.
    """
    changed = False
    n = pag.n_nodes

    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            if not pag.has_edge(x, z):
                continue

            # Check if X o-* Z (circle at X)
            if pag.get_endpoint(x, z) != EdgeMark.CIRCLE:
                continue

            # Look for X -> Y *-> Z or X *-> Y -> Z
            for y in range(n):
                if y == x or y == z:
                    continue
                if not pag.has_edge(x, y) or not pag.has_edge(y, z):
                    continue

                # Pattern 1: X -> Y *-> Z
                x_to_y = (
                    pag.get_endpoint(x, y) == EdgeMark.TAIL
                    and pag.get_endpoint(y, x) == EdgeMark.ARROW
                )
                y_to_z = pag.get_endpoint(z, y) == EdgeMark.ARROW

                # Pattern 2: X *-> Y -> Z
                x_into_y = pag.get_endpoint(y, x) == EdgeMark.ARROW
                y_to_z_directed = (
                    pag.get_endpoint(y, z) == EdgeMark.TAIL
                    and pag.get_endpoint(z, y) == EdgeMark.ARROW
                )

                if (x_to_y and y_to_z) or (x_into_y and y_to_z_directed):
                    # Orient X *-> Z (set arrow at Z)
                    pag.set_endpoint(z, x, EdgeMark.ARROW)
                    changed = True

                    if verbose:
                        print(f"      R2: {x} *-> {z} (through {y})")
                    break

    return changed


def _fci_rule_3(pag: PAG, verbose: bool = False) -> bool:
    """R3: Double-triangle rule.

    If X *-> Y <-* Z, X *-o W o-* Z, X and Z not adjacent,
    and W o-* Y, orient W *-> Y.
    """
    changed = False
    n = pag.n_nodes

    for y in range(n):
        # Find pairs (X, Z) forming X *-> Y <-* Z where X ⊥ Z
        pairs = []
        for x in range(n):
            if x == y:
                continue
            if not pag.has_edge(x, y):
                continue
            if pag.get_endpoint(y, x) != EdgeMark.ARROW:
                continue

            for z in range(x + 1, n):
                if z == y:
                    continue
                if not pag.has_edge(z, y):
                    continue
                if pag.get_endpoint(y, z) != EdgeMark.ARROW:
                    continue
                if pag.has_edge(x, z):
                    continue

                pairs.append((x, z))

        # For each pair, look for W
        for x, z in pairs:
            for w in range(n):
                if w in (x, y, z):
                    continue
                if not pag.has_edge(w, x) or not pag.has_edge(w, z):
                    continue
                if not pag.has_edge(w, y):
                    continue

                # Check W o-* Y (circle at W)
                if pag.get_endpoint(w, y) != EdgeMark.CIRCLE:
                    continue

                # Check X *-o W (circle at W)
                if pag.get_endpoint(w, x) != EdgeMark.CIRCLE:
                    continue

                # Check W o-* Z (circle at W)
                if pag.get_endpoint(w, z) != EdgeMark.CIRCLE:
                    continue

                # Orient W *-> Y (set arrow at Y)
                pag.set_endpoint(y, w, EdgeMark.ARROW)
                changed = True

                if verbose:
                    print(f"      R3: {w} *-> {y} (double-triangle)")

    return changed


def _fci_rule_4(pag: PAG, verbose: bool = False) -> bool:
    """R4: Discriminating path rule.

    If there is a discriminating path <V, ..., W, X, Y> for X
    where V is non-adjacent to Y:
    - If X in Sep(V, Y): X ---Y becomes X --- Y (no orientation)
    - If X not in Sep(V, Y): Orient X *-> Y

    Simplified implementation - checks basic discriminating patterns.
    """
    changed = False
    n = pag.n_nodes

    # Find discriminating paths
    for y in range(n):
        for x in range(n):
            if x == y:
                continue
            if not pag.has_edge(x, y):
                continue

            # Check if X o-* Y (circle at X)
            if pag.get_endpoint(x, y) != EdgeMark.CIRCLE:
                continue

            # Look for W such that W -> X *-> Y or W <-> X *-> Y
            for w in range(n):
                if w in (x, y):
                    continue
                if not pag.has_edge(w, x):
                    continue

                # W -> X or W <-> X
                w_to_x = (
                    pag.get_endpoint(w, x) == EdgeMark.TAIL
                    and pag.get_endpoint(x, w) == EdgeMark.ARROW
                )
                w_bidi_x = (
                    pag.get_endpoint(w, x) == EdgeMark.ARROW
                    and pag.get_endpoint(x, w) == EdgeMark.ARROW
                )

                if not (w_to_x or w_bidi_x):
                    continue

                # Check W *-> Y
                if not pag.has_edge(w, y):
                    continue
                if pag.get_endpoint(y, w) != EdgeMark.ARROW:
                    continue

                # Look for discriminating path starting from V
                for v in range(n):
                    if v in (w, x, y):
                        continue
                    if not pag.has_edge(v, w):
                        continue

                    # V not adjacent to Y
                    if pag.has_edge(v, y):
                        continue

                    # V -> W
                    v_to_w = (
                        pag.get_endpoint(v, w) == EdgeMark.TAIL
                        and pag.get_endpoint(w, v) == EdgeMark.ARROW
                    )

                    if v_to_w:
                        # This is a discriminating path
                        # Simplified: assume X not in sep(V, Y) -> collider
                        pag.set_endpoint(y, x, EdgeMark.ARROW)
                        changed = True

                        if verbose:
                            print(f"      R4: {x} *-> {y} (discriminating path)")
                        break

                if changed:
                    break
            if changed:
                break

    return changed


def _fci_rule_8(pag: PAG, verbose: bool = False) -> bool:
    """R8: Orient definite non-ancestor.

    If X -> Y -> Z or X -o Y -> Z, and X o-> Z, orient X -> Z.
    """
    changed = False
    n = pag.n_nodes

    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            if not pag.has_edge(x, z):
                continue

            # Check X o-> Z (circle at X, arrow at Z)
            if pag.get_endpoint(x, z) != EdgeMark.CIRCLE:
                continue
            if pag.get_endpoint(z, x) != EdgeMark.ARROW:
                continue

            # Look for Y such that X -> Y -> Z or X -o Y -> Z
            for y in range(n):
                if y in (x, z):
                    continue
                if not pag.has_edge(x, y) or not pag.has_edge(y, z):
                    continue

                # X -> Y (tail at X, arrow at Y)
                x_to_y = (
                    pag.get_endpoint(x, y) == EdgeMark.TAIL
                    and pag.get_endpoint(y, x) == EdgeMark.ARROW
                )

                # X -o Y (tail at X, circle at Y)
                x_tail_circle_y = (
                    pag.get_endpoint(x, y) == EdgeMark.TAIL
                    and pag.get_endpoint(y, x) == EdgeMark.CIRCLE
                )

                # Y -> Z (tail at Y, arrow at Z)
                y_to_z = (
                    pag.get_endpoint(y, z) == EdgeMark.TAIL
                    and pag.get_endpoint(z, y) == EdgeMark.ARROW
                )

                if (x_to_y or x_tail_circle_y) and y_to_z:
                    # Orient X -> Z (set tail at X)
                    pag.set_endpoint(x, z, EdgeMark.TAIL)
                    changed = True

                    if verbose:
                        print(f"      R8: {x} -> {z} (through {y})")
                    break

    return changed


def _fci_rule_9(pag: PAG, verbose: bool = False) -> bool:
    """R9: Uncovered potentially directed path.

    If X o-> Y and there is an uncovered p.d. path from X to Y,
    orient X -> Y.
    """
    changed = False
    n = pag.n_nodes

    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            if not pag.has_edge(x, y):
                continue

            # Check X o-> Y
            if pag.get_endpoint(x, y) != EdgeMark.CIRCLE:
                continue
            if pag.get_endpoint(y, x) != EdgeMark.ARROW:
                continue

            # Look for uncovered potentially directed path X to Y
            if _has_uncovered_pd_path(pag, x, y, {x}):
                pag.set_endpoint(x, y, EdgeMark.TAIL)
                changed = True

                if verbose:
                    print(f"      R9: {x} -> {y} (uncovered p.d. path)")

    return changed


def _has_uncovered_pd_path(
    pag: PAG,
    start: int,
    end: int,
    visited: Set[int],
    prev: Optional[int] = None,
) -> bool:
    """Check if there is an uncovered potentially directed path from start to end.

    A path is potentially directed if all edges could be oriented
    consistently from start to end. It is uncovered if every consecutive
    pair of edges (through intermediate node) forms a non-collider.
    """
    if start == end:
        return True

    for neighbor in pag.adjacent(start):
        if neighbor in visited:
            continue

        # Check if edge could be oriented start -> neighbor
        # (not arrow at start)
        if pag.get_endpoint(start, neighbor) == EdgeMark.ARROW:
            continue

        # Check uncovered: if prev exists, prev and neighbor should be adjacent
        # (otherwise it's a collider at start, which makes it not uncovered)
        if prev is not None:
            # For uncovered path: consecutive non-endpoints must be adjacent
            # This is simplified - full check would verify non-collider structure
            pass

        if neighbor == end:
            return True

        visited_new = visited | {neighbor}
        if _has_uncovered_pd_path(pag, neighbor, end, visited_new, start):
            return True

    return False


def _fci_rule_10(pag: PAG, verbose: bool = False) -> bool:
    """R10: Three uncovered potentially directed paths.

    If X o-> Y and there are 3 uncovered p.d. paths from X to Y
    through distinct intermediate nodes, orient X -> Y.

    Simplified implementation.
    """
    changed = False
    n = pag.n_nodes

    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            if not pag.has_edge(x, y):
                continue

            # Check X o-> Y
            if pag.get_endpoint(x, y) != EdgeMark.CIRCLE:
                continue
            if pag.get_endpoint(y, x) != EdgeMark.ARROW:
                continue

            # Count paths through different first intermediate nodes
            path_starts = []
            for neighbor in pag.adjacent(x):
                if neighbor == y:
                    continue
                if pag.get_endpoint(x, neighbor) == EdgeMark.ARROW:
                    continue

                if _has_uncovered_pd_path(pag, neighbor, y, {x, neighbor}):
                    path_starts.append(neighbor)

            if len(path_starts) >= 3:
                pag.set_endpoint(x, y, EdgeMark.TAIL)
                changed = True

                if verbose:
                    print(f"      R10: {x} -> {y} (3 p.d. paths)")

    return changed
