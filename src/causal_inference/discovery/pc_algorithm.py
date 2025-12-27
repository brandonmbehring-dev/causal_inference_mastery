"""PC Algorithm for causal discovery.

Session 133: Constraint-based causal structure learning.

The PC algorithm learns causal structure from observational data by:
1. Starting with a complete undirected graph
2. Removing edges based on conditional independence tests
3. Orienting edges using v-structures and Meek rules

Output is a CPDAG representing the Markov equivalence class.

References
----------
- Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
- Meek (1995). Causal inference and causal explanation with background knowledge.
- Kalisch & Buhlmann (2007). Estimating high-dimensional DAGs with the PC algorithm.

Functions
---------
pc_skeleton : Learn undirected skeleton via CI tests
pc_orient : Orient skeleton edges to CPDAG
pc_algorithm : Full PC algorithm (skeleton + orientation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from .independence_tests import CITestResult, ci_test, fisher_z_test
from .types import CPDAG, DAG, Graph, PCResult


# Type alias for CI test function
CITestFunction = Callable[
    [np.ndarray, int, int, Union[List[int], Tuple[int, ...], None], float],
    CITestResult,
]


def pc_skeleton(
    data: np.ndarray,
    alpha: float = 0.01,
    ci_test_func: Optional[CITestFunction] = None,
    max_cond_size: Optional[int] = None,
    stable: bool = True,
    verbose: bool = False,
) -> Tuple[Graph, Dict[Tuple[int, int], FrozenSet[int]], int]:
    """Learn undirected skeleton using conditional independence tests.

    Phase 1 of PC algorithm. Iteratively tests conditional independence
    for increasing conditioning set sizes and removes edges where
    independence is found.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    alpha : float
        Significance level for CI tests. Lower values are more conservative
        (keep more edges).
    ci_test_func : callable, optional
        CI test function. Default is Fisher's Z test.
    max_cond_size : int, optional
        Maximum conditioning set size. If None, uses n_variables - 2.
    stable : bool
        If True, use stable PC variant that doesn't depend on variable order.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    skeleton : Graph
        Undirected skeleton graph.
    separating_sets : Dict[Tuple[int, int], FrozenSet[int]]
        For each removed edge (i, j), the separating set that rendered
        i and j conditionally independent.
    n_tests : int
        Total number of CI tests performed.

    Example
    -------
    >>> from .utils import generate_random_dag, generate_dag_data
    >>> dag = generate_random_dag(5, edge_prob=0.3, seed=42)
    >>> data, _ = generate_dag_data(dag, n_samples=1000, seed=42)
    >>> skeleton, sep_sets, n_tests = pc_skeleton(data, alpha=0.01)
    >>> print(f"Skeleton edges: {skeleton.n_edges()}")
    """
    if ci_test_func is None:
        ci_test_func = fisher_z_test

    n_vars = data.shape[1]
    if max_cond_size is None:
        max_cond_size = n_vars - 2

    # Initialize complete undirected graph
    skeleton = Graph.complete(n_vars)
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]] = {}
    n_tests = 0

    # Iterate over conditioning set sizes
    for cond_size in range(max_cond_size + 1):
        if verbose:
            print(f"Testing conditioning set size: {cond_size}")

        # For stable PC, collect all edges first, then test
        if stable:
            edges_to_test = list(skeleton.edges())
        else:
            edges_to_test = None

        # Track edges to remove (for stable PC)
        edges_to_remove: List[Tuple[int, int, FrozenSet[int]]] = []

        # Check each edge
        edge_iter = edges_to_test if stable else list(skeleton.edges())

        for i, j in edge_iter:
            if not skeleton.has_edge(i, j):
                continue  # Already removed

            # Get adjacent nodes (potential conditioning sets)
            adj_i = skeleton.neighbors(i) - {j}
            adj_j = skeleton.neighbors(j) - {i}

            # Use larger adjacency set
            if len(adj_i) >= len(adj_j):
                adjacent = adj_i
            else:
                adjacent = adj_j

            # Skip if not enough neighbors for this conditioning size
            if len(adjacent) < cond_size:
                continue

            # Test all conditioning sets of current size
            found_independent = False
            for cond_set in _subsets(list(adjacent), cond_size):
                n_tests += 1

                result = ci_test_func(data, i, j, list(cond_set), alpha)

                if result.independent:
                    # Found separating set
                    sep_key = (min(i, j), max(i, j))
                    separating_sets[sep_key] = frozenset(cond_set)
                    found_independent = True

                    if stable:
                        edges_to_remove.append((i, j, frozenset(cond_set)))
                    else:
                        skeleton.remove_edge(i, j)

                    if verbose:
                        print(f"  Removed edge {i}--{j} | {cond_set} (p={result.pvalue:.4f})")
                    break

            if found_independent and not stable:
                break

        # For stable PC, remove edges after testing all
        if stable:
            for i, j, _ in edges_to_remove:
                skeleton.remove_edge(i, j)

        # Check termination: if max degree < cond_size + 1, we're done
        max_degree = max(len(skeleton.neighbors(v)) for v in range(n_vars))
        if max_degree < cond_size + 1:
            break

    return skeleton, separating_sets, n_tests


def _subsets(elements: List[int], size: int):
    """Generate all subsets of given size."""
    from itertools import combinations

    return combinations(elements, size)


def pc_orient(
    skeleton: Graph,
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]],
    verbose: bool = False,
) -> CPDAG:
    """Orient skeleton edges to CPDAG using v-structures and Meek rules.

    Phase 2 of PC algorithm:
    1. Identify and orient v-structures (immoralities)
    2. Apply Meek rules to orient additional edges

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
    CPDAG
        Completed partially directed acyclic graph.

    Example
    -------
    >>> cpdag = pc_orient(skeleton, sep_sets)
    >>> n_directed = np.sum(cpdag.directed > 0)
    >>> n_undirected = np.sum(cpdag.undirected > 0) // 2
    >>> print(f"Directed: {n_directed}, Undirected: {n_undirected}")
    """
    n_vars = skeleton.n_nodes

    # Initialize CPDAG with skeleton as undirected
    cpdag = CPDAG(n_nodes=n_vars, node_names=skeleton.node_names.copy())

    for i, j in skeleton.edges():
        cpdag.add_undirected_edge(i, j)

    # Step 1: Orient v-structures (immoralities)
    # X -> Z <- Y where X and Y are not adjacent
    if verbose:
        print("Orienting v-structures...")

    for z in range(n_vars):
        neighbors = list(skeleton.neighbors(z))

        for idx_x, x in enumerate(neighbors):
            for y in neighbors[idx_x + 1 :]:
                # Check if x and y are NOT adjacent
                if skeleton.has_edge(x, y):
                    continue

                # Check separating set for (x, y) edge
                sep_key = (min(x, y), max(x, y))
                if sep_key not in separating_sets:
                    # x and y were never adjacent, so no sep set
                    # This means x -> z <- y is a v-structure if z not in sep(x,y)
                    # Since they were never tested together, treat as v-structure
                    _orient_v_structure(cpdag, x, z, y, verbose)
                else:
                    sep_set = separating_sets[sep_key]
                    if z not in sep_set:
                        # z is NOT in separating set -> v-structure
                        _orient_v_structure(cpdag, x, z, y, verbose)

    # Step 2: Apply Meek rules until no changes
    if verbose:
        print("Applying Meek rules...")

    _apply_meek_rules(cpdag, verbose)

    return cpdag


def _orient_v_structure(cpdag: CPDAG, x: int, z: int, y: int, verbose: bool) -> None:
    """Orient edges as v-structure: X -> Z <- Y."""
    if cpdag.has_undirected_edge(x, z):
        cpdag.add_directed_edge(x, z)
        if verbose:
            print(f"  V-structure: {x} -> {z}")

    if cpdag.has_undirected_edge(y, z):
        cpdag.add_directed_edge(y, z)
        if verbose:
            print(f"  V-structure: {y} -> {z}")


def _apply_meek_rules(cpdag: CPDAG, verbose: bool = False) -> None:
    """Apply Meek's rules R1-R4 until convergence."""
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1

        if verbose:
            print(f"  Meek iteration {iteration}")

        changed |= _meek_r1(cpdag, verbose)
        changed |= _meek_r2(cpdag, verbose)
        changed |= _meek_r3(cpdag, verbose)
        changed |= _meek_r4(cpdag, verbose)


def _meek_r1(cpdag: CPDAG, verbose: bool = False) -> bool:
    """Meek Rule 1: Orient i --- j as i -> j if k -> i and k not adj j.

    If there is a directed edge into i, and an undirected edge from i to j,
    and k and j are not adjacent, then orient i -> j.

    This prevents creating a new v-structure.
    """
    changed = False
    n = cpdag.n_nodes

    for i in range(n):
        for j in range(n):
            if not cpdag.has_undirected_edge(i, j):
                continue

            # Look for k -> i where k not adjacent to j
            for k in range(n):
                if cpdag.has_directed_edge(k, i) and not cpdag.has_any_edge(k, j):
                    cpdag.add_directed_edge(i, j)
                    changed = True
                    if verbose:
                        print(f"    R1: {i} -> {j} (due to {k} -> {i})")
                    break

    return changed


def _meek_r2(cpdag: CPDAG, verbose: bool = False) -> bool:
    """Meek Rule 2: Orient i --- j as i -> j if i -> k -> j.

    If there is a directed path from i to j, orient the undirected
    edge to avoid a cycle.
    """
    changed = False
    n = cpdag.n_nodes

    for i in range(n):
        for j in range(n):
            if not cpdag.has_undirected_edge(i, j):
                continue

            # Look for i -> k -> j
            for k in range(n):
                if cpdag.has_directed_edge(i, k) and cpdag.has_directed_edge(k, j):
                    cpdag.add_directed_edge(i, j)
                    changed = True
                    if verbose:
                        print(f"    R2: {i} -> {j} (chain through {k})")
                    break

    return changed


def _meek_r3(cpdag: CPDAG, verbose: bool = False) -> bool:
    """Meek Rule 3: Orient i --- j as i -> j if i --- k1 -> j, i --- k2 -> j,
    and k1 not adj k2.

    Two uncolliders pointing to j force i -> j.
    """
    changed = False
    n = cpdag.n_nodes

    for i in range(n):
        for j in range(n):
            if not cpdag.has_undirected_edge(i, j):
                continue

            # Find candidates: k such that i --- k -> j
            candidates = []
            for k in range(n):
                if cpdag.has_undirected_edge(i, k) and cpdag.has_directed_edge(k, j):
                    candidates.append(k)

            # Check if any pair of candidates is non-adjacent
            for idx1, k1 in enumerate(candidates):
                for k2 in candidates[idx1 + 1 :]:
                    if not cpdag.has_any_edge(k1, k2):
                        cpdag.add_directed_edge(i, j)
                        changed = True
                        if verbose:
                            print(f"    R3: {i} -> {j} (via {k1}, {k2})")
                        break
                if changed:
                    break

    return changed


def _meek_r4(cpdag: CPDAG, verbose: bool = False) -> bool:
    """Meek Rule 4: Orient i --- j as i -> j if i --- k -> l -> j, k not adj j.

    A directed path from an undirected neighbor forces orientation.
    """
    changed = False
    n = cpdag.n_nodes

    for i in range(n):
        for j in range(n):
            if not cpdag.has_undirected_edge(i, j):
                continue

            # Look for i --- k -> l -> j where k not adj j
            for k in range(n):
                if not cpdag.has_undirected_edge(i, k):
                    continue
                if cpdag.has_any_edge(k, j):
                    continue

                for l in range(n):
                    if cpdag.has_directed_edge(k, l) and cpdag.has_directed_edge(l, j):
                        cpdag.add_directed_edge(i, j)
                        changed = True
                        if verbose:
                            print(f"    R4: {i} -> {j} (path {k} -> {l} -> {j})")
                        break
                if changed:
                    break

    return changed


def pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.01,
    ci_test_func: Optional[CITestFunction] = None,
    max_cond_size: Optional[int] = None,
    stable: bool = True,
    verbose: bool = False,
) -> PCResult:
    """Full PC algorithm for causal discovery.

    Learns causal structure from observational data assuming:
    1. Causal Markov condition
    2. Faithfulness (no cancellation of paths)
    3. Causal sufficiency (no hidden confounders)

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    alpha : float
        Significance level for CI tests. Default 0.01.
        - Lower alpha -> more conservative -> more edges retained
        - Higher alpha -> more aggressive -> fewer edges
    ci_test_func : callable, optional
        CI test function. Default is Fisher's Z test.
    max_cond_size : int, optional
        Maximum conditioning set size to consider.
    stable : bool
        If True, use order-independent (stable) PC variant.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    PCResult
        Result containing CPDAG, skeleton, separating sets, and metrics.

    Example
    -------
    >>> from .utils import generate_random_dag, generate_dag_data
    >>> # Generate data from known DAG
    >>> true_dag = generate_random_dag(5, edge_prob=0.4, seed=42)
    >>> data, _ = generate_dag_data(true_dag, n_samples=1000, seed=42)
    >>> # Run PC algorithm
    >>> result = pc_algorithm(data, alpha=0.01)
    >>> print(f"Skeleton F1: {result.skeleton_f1(true_dag)[2]:.3f}")
    >>> print(f"SHD: {result.structural_hamming_distance(true_dag)}")

    Notes
    -----
    The PC algorithm has complexity O(p^d) where p is number of variables
    and d is maximum degree. For high-dimensional data with sparse graphs,
    this is often manageable. For dense graphs, consider PC-stable or FGES.

    Faithfulness violations can cause the algorithm to:
    - Miss edges (if paths cancel perfectly)
    - Add spurious edges (if non-causal correlations exist)

    For latent confounders, use FCI algorithm instead.
    """
    if verbose:
        print("=" * 60)
        print("PC Algorithm")
        print("=" * 60)
        print(f"Data shape: {data.shape}")
        print(f"Alpha: {alpha}")
        print(f"Stable: {stable}")
        print()

    # Phase 1: Learn skeleton
    if verbose:
        print("Phase 1: Learning skeleton...")

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

    # Phase 2: Orient edges
    if verbose:
        print("Phase 2: Orienting edges...")

    cpdag = pc_orient(skeleton, separating_sets, verbose=verbose)

    if verbose:
        n_directed = np.sum(cpdag.directed > 0)
        n_undirected = np.sum(cpdag.undirected > 0) // 2
        print(f"Directed edges: {n_directed}")
        print(f"Undirected edges: {n_undirected}")
        print("=" * 60)

    return PCResult(
        cpdag=cpdag,
        skeleton=skeleton,
        separating_sets=separating_sets,
        n_ci_tests=n_tests,
        alpha=alpha,
    )


# =============================================================================
# Variants and Extensions
# =============================================================================


def pc_conservative(
    data: np.ndarray,
    alpha: float = 0.01,
    ci_test_func: Optional[CITestFunction] = None,
    verbose: bool = False,
) -> PCResult:
    """Conservative PC algorithm (PC-CPC).

    More conservative in orienting v-structures. Only orients X -> Z <- Y
    if ALL separating sets for X---Y do not contain Z.

    This reduces false positive v-structures at the cost of leaving
    more edges undirected.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    alpha : float
        Significance level.
    ci_test_func : callable, optional
        CI test function.
    verbose : bool
        Print progress.

    Returns
    -------
    PCResult
        Result with conservatively oriented CPDAG.
    """
    if ci_test_func is None:
        ci_test_func = fisher_z_test

    n_vars = data.shape[1]

    # Phase 1: Standard skeleton learning
    skeleton, _, n_tests = pc_skeleton(
        data, alpha=alpha, ci_test_func=ci_test_func, stable=True, verbose=verbose
    )

    # Phase 1b: Find ALL separating sets (conservative extension)
    all_separating_sets: Dict[Tuple[int, int], List[FrozenSet[int]]] = {}

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton.has_edge(i, j):
                continue  # Still adjacent, no separating set

            # Find all conditioning sets that make i ⊥ j
            sep_key = (i, j)
            all_separating_sets[sep_key] = []

            adj_i = skeleton.neighbors(i)
            adj_j = skeleton.neighbors(j)
            adjacent = adj_i | adj_j

            for size in range(len(adjacent) + 1):
                for cond_set in _subsets(list(adjacent), size):
                    result = ci_test_func(data, i, j, list(cond_set), alpha)
                    n_tests += 1
                    if result.independent:
                        all_separating_sets[sep_key].append(frozenset(cond_set))

    # Phase 2: Conservative orientation
    cpdag = CPDAG(n_nodes=n_vars, node_names=skeleton.node_names.copy())

    for i, j in skeleton.edges():
        cpdag.add_undirected_edge(i, j)

    # Orient v-structures conservatively
    for z in range(n_vars):
        neighbors = list(skeleton.neighbors(z))

        for idx_x, x in enumerate(neighbors):
            for y in neighbors[idx_x + 1 :]:
                if skeleton.has_edge(x, y):
                    continue

                sep_key = (min(x, y), max(x, y))
                if sep_key not in all_separating_sets:
                    continue

                # Conservative: z must be absent from ALL separating sets
                all_sep_sets = all_separating_sets[sep_key]
                if all_sep_sets and all(z not in s for s in all_sep_sets):
                    _orient_v_structure(cpdag, x, z, y, verbose)

    # Apply Meek rules
    _apply_meek_rules(cpdag, verbose)

    return PCResult(
        cpdag=cpdag,
        skeleton=skeleton,
        separating_sets={k: v[0] if v else frozenset() for k, v in all_separating_sets.items()},
        n_ci_tests=n_tests,
        alpha=alpha,
    )


def pc_majority(
    data: np.ndarray,
    alpha: float = 0.01,
    ci_test_func: Optional[CITestFunction] = None,
    verbose: bool = False,
) -> PCResult:
    """Majority rule PC algorithm (PC-MPC).

    Orients v-structure X -> Z <- Y if MAJORITY of separating sets
    for X---Y do not contain Z.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    alpha : float
        Significance level.
    ci_test_func : callable, optional
        CI test function.
    verbose : bool
        Print progress.

    Returns
    -------
    PCResult
        Result with majority-rule oriented CPDAG.
    """
    if ci_test_func is None:
        ci_test_func = fisher_z_test

    n_vars = data.shape[1]

    # Phase 1: Standard skeleton
    skeleton, _, n_tests = pc_skeleton(
        data, alpha=alpha, ci_test_func=ci_test_func, stable=True, verbose=verbose
    )

    # Find all separating sets
    all_separating_sets: Dict[Tuple[int, int], List[FrozenSet[int]]] = {}

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton.has_edge(i, j):
                continue

            sep_key = (i, j)
            all_separating_sets[sep_key] = []

            adj_i = skeleton.neighbors(i)
            adj_j = skeleton.neighbors(j)
            adjacent = adj_i | adj_j

            for size in range(len(adjacent) + 1):
                for cond_set in _subsets(list(adjacent), size):
                    result = ci_test_func(data, i, j, list(cond_set), alpha)
                    n_tests += 1
                    if result.independent:
                        all_separating_sets[sep_key].append(frozenset(cond_set))

    # Phase 2: Majority-rule orientation
    cpdag = CPDAG(n_nodes=n_vars, node_names=skeleton.node_names.copy())

    for i, j in skeleton.edges():
        cpdag.add_undirected_edge(i, j)

    # Orient v-structures by majority vote
    for z in range(n_vars):
        neighbors = list(skeleton.neighbors(z))

        for idx_x, x in enumerate(neighbors):
            for y in neighbors[idx_x + 1 :]:
                if skeleton.has_edge(x, y):
                    continue

                sep_key = (min(x, y), max(x, y))
                if sep_key not in all_separating_sets:
                    continue

                all_sep_sets = all_separating_sets[sep_key]
                if not all_sep_sets:
                    continue

                # Majority: z absent from more than half of separating sets
                n_without_z = sum(1 for s in all_sep_sets if z not in s)
                if n_without_z > len(all_sep_sets) / 2:
                    _orient_v_structure(cpdag, x, z, y, verbose)

    # Apply Meek rules
    _apply_meek_rules(cpdag, verbose)

    return PCResult(
        cpdag=cpdag,
        skeleton=skeleton,
        separating_sets={k: v[0] if v else frozenset() for k, v in all_separating_sets.items()},
        n_ci_tests=n_tests,
        alpha=alpha,
    )
