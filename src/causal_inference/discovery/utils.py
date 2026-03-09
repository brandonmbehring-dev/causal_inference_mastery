"""Utility functions for causal discovery algorithms.

Session 133: Graph manipulation, DAG generation, and evaluation metrics.

Functions
---------
generate_random_dag : Generate random DAG with given sparsity
generate_dag_data : Generate data from linear SCM
dag_to_cpdag : Convert DAG to its CPDAG (equivalence class)
skeleton_f1 : Compute F1 score for skeleton recovery
orientation_accuracy : Compute accuracy of edge orientations
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from .types import CPDAG, DAG, Graph


def generate_random_dag(
    n_nodes: int,
    edge_prob: float = 0.3,
    seed: Optional[int] = None,
    node_names: Optional[List[str]] = None,
) -> DAG:
    """Generate random DAG using Erdos-Renyi model with topological ordering.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Probability of edge between any two nodes (in valid direction).
    seed : Optional[int]
        Random seed.
    node_names : Optional[List[str]]
        Names for nodes.

    Returns
    -------
    DAG
        Random DAG with expected density edge_prob.

    Example
    -------
    >>> dag = generate_random_dag(5, edge_prob=0.3, seed=42)
    >>> dag.is_acyclic()
    True
    """
    rng = np.random.default_rng(seed)
    names = node_names or [f"X{i}" for i in range(n_nodes)]

    # Generate random order (implicit topological ordering)
    order = rng.permutation(n_nodes)

    # Create adjacency matrix respecting ordering
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                # Edge goes from earlier in order to later
                parent, child = order[i], order[j]
                adj[parent, child] = 1

    return DAG(n_nodes=n_nodes, node_names=names, adjacency=adj)


def generate_dag_data(
    dag: DAG,
    n_samples: int,
    noise_scale: float = 1.0,
    coefficient_range: Tuple[float, float] = (0.5, 1.5),
    noise_type: str = "gaussian",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from linear structural causal model.

    X_j = sum_{i in Pa(j)} B[i,j] * X_i + epsilon_j

    Parameters
    ----------
    dag : DAG
        Causal DAG structure.
    n_samples : int
        Number of samples to generate.
    noise_scale : float
        Standard deviation of noise terms.
    coefficient_range : Tuple[float, float]
        Range for random edge coefficients (absolute value).
    noise_type : str
        Type of noise: "gaussian", "uniform", "laplace", "exponential".
    seed : Optional[int]
        Random seed.

    Returns
    -------
    data : np.ndarray
        Data matrix of shape (n_samples, n_nodes).
    coefficients : np.ndarray
        Weighted adjacency matrix B.

    Example
    -------
    >>> dag = generate_random_dag(5, edge_prob=0.3, seed=42)
    >>> data, B = generate_dag_data(dag, n_samples=1000, seed=42)
    >>> data.shape
    (1000, 5)
    """
    rng = np.random.default_rng(seed)
    n_nodes = dag.n_nodes

    # Generate random coefficients for edges
    B = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if dag.adjacency[i, j]:
                sign = rng.choice([-1, 1])
                magnitude = rng.uniform(*coefficient_range)
                B[i, j] = sign * magnitude

    # Get topological order
    order = dag.topological_order()

    # Generate data following topological order
    data = np.zeros((n_samples, n_nodes))

    for node in order:
        # Generate noise
        if noise_type == "gaussian":
            noise = rng.normal(0, noise_scale, n_samples)
        elif noise_type == "uniform":
            noise = rng.uniform(-noise_scale * np.sqrt(3), noise_scale * np.sqrt(3), n_samples)
        elif noise_type == "laplace":
            noise = rng.laplace(0, noise_scale / np.sqrt(2), n_samples)
        elif noise_type == "exponential":
            # Centered exponential
            noise = rng.exponential(noise_scale, n_samples) - noise_scale
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Compute value: X_j = sum B[i,j] * X_i + noise
        parents = dag.parents(node)
        value = noise.copy()
        for parent in parents:
            value += B[parent, node] * data[:, parent]

        data[:, node] = value

    return data, B


def dag_to_cpdag(dag: DAG) -> CPDAG:
    """Convert DAG to its CPDAG (Markov equivalence class).

    The CPDAG contains:
    - Compelled (directed) edges that have the same orientation in all DAGs
      in the equivalence class
    - Reversible (undirected) edges that can be oriented either way

    Uses Meek's rules to identify compelled edges.

    Parameters
    ----------
    dag : DAG
        Input DAG.

    Returns
    -------
    CPDAG
        CPDAG representing the Markov equivalence class.

    Example
    -------
    >>> dag = DAG(n_nodes=3)
    >>> dag.add_edge(0, 1)  # X -> Y
    >>> dag.add_edge(1, 2)  # Y -> Z
    >>> cpdag = dag_to_cpdag(dag)
    >>> # X -> Y -> Z forms a chain, all edges reversible
    """
    n_nodes = dag.n_nodes
    cpdag = CPDAG(
        n_nodes=n_nodes,
        node_names=dag.node_names.copy(),
    )

    # Step 1: Find v-structures (immoralities)
    # A v-structure is X -> Z <- Y where X and Y are not adjacent
    v_structures = set()
    for z in range(n_nodes):
        parents_z = list(dag.parents(z))
        for i, x in enumerate(parents_z):
            for y in parents_z[i + 1 :]:
                # Check if x and y are not adjacent
                if not dag.has_edge(x, y) and not dag.has_edge(y, x):
                    v_structures.add((x, z))
                    v_structures.add((y, z))

    # Step 2: Initialize CPDAG
    # Start with skeleton as undirected
    for i in range(n_nodes):
        for j in range(n_nodes):
            if dag.adjacency[i, j]:
                if (i, j) in v_structures:
                    cpdag.add_directed_edge(i, j)
                else:
                    cpdag.add_undirected_edge(i, j)

    # Step 3: Apply Meek's rules until no changes
    changed = True
    while changed:
        changed = False
        changed |= _meek_rule_1(cpdag)
        changed |= _meek_rule_2(cpdag)
        changed |= _meek_rule_3(cpdag)
        changed |= _meek_rule_4(cpdag)

    return cpdag


def _meek_rule_1(cpdag: CPDAG) -> bool:
    """Meek Rule 1: Orient i --- j as i -> j if exists k -> i and k not adj j."""
    changed = False
    n = cpdag.n_nodes
    for i in range(n):
        for j in range(n):
            if cpdag.has_undirected_edge(i, j):
                # Look for k -> i where k not adjacent to j
                for k in range(n):
                    if cpdag.has_directed_edge(k, i) and not cpdag.has_any_edge(k, j):
                        cpdag.add_directed_edge(i, j)
                        changed = True
                        break
    return changed


def _meek_rule_2(cpdag: CPDAG) -> bool:
    """Meek Rule 2: Orient i --- j as i -> j if i -> k -> j."""
    changed = False
    n = cpdag.n_nodes
    for i in range(n):
        for j in range(n):
            if cpdag.has_undirected_edge(i, j):
                # Look for i -> k -> j
                for k in range(n):
                    if cpdag.has_directed_edge(i, k) and cpdag.has_directed_edge(k, j):
                        cpdag.add_directed_edge(i, j)
                        changed = True
                        break
    return changed


def _meek_rule_3(cpdag: CPDAG) -> bool:
    """Meek Rule 3: Orient i --- j as i -> j if i --- k1 -> j and i --- k2 -> j
    and k1 not adj k2."""
    changed = False
    n = cpdag.n_nodes
    for i in range(n):
        for j in range(n):
            if cpdag.has_undirected_edge(i, j):
                # Find k1, k2 such that i --- k1 -> j, i --- k2 -> j, k1 not adj k2
                candidates = []
                for k in range(n):
                    if cpdag.has_undirected_edge(i, k) and cpdag.has_directed_edge(k, j):
                        candidates.append(k)

                for idx1, k1 in enumerate(candidates):
                    for k2 in candidates[idx1 + 1 :]:
                        if not cpdag.has_any_edge(k1, k2):
                            cpdag.add_directed_edge(i, j)
                            changed = True
                            break
                    if changed:
                        break
    return changed


def _meek_rule_4(cpdag: CPDAG) -> bool:
    """Meek Rule 4: Orient i --- j as i -> j if i --- k -> l -> j and k not adj j."""
    changed = False
    n = cpdag.n_nodes
    for i in range(n):
        for j in range(n):
            if cpdag.has_undirected_edge(i, j):
                # Look for i --- k -> l -> j where k not adj j
                for k in range(n):
                    if cpdag.has_undirected_edge(i, k) and not cpdag.has_any_edge(k, j):
                        for l in range(n):
                            if cpdag.has_directed_edge(k, l) and cpdag.has_directed_edge(l, j):
                                cpdag.add_directed_edge(i, j)
                                changed = True
                                break
                        if changed:
                            break
    return changed


def skeleton_f1(
    estimated: Graph,
    true_dag: DAG,
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for skeleton recovery.

    Parameters
    ----------
    estimated : Graph
        Estimated skeleton (undirected).
    true_dag : DAG
        Ground truth DAG.

    Returns
    -------
    precision : float
        Fraction of estimated edges that are true.
    recall : float
        Fraction of true edges that are estimated.
    f1 : float
        F1 score = 2 * precision * recall / (precision + recall).
    """
    n = estimated.n_nodes
    true_edges = set()
    est_edges = set()

    # Get true skeleton edges
    for i in range(n):
        for j in range(i + 1, n):
            if true_dag.has_edge(i, j) or true_dag.has_edge(j, i):
                true_edges.add((i, j))
            if estimated.has_edge(i, j):
                est_edges.add((i, j))

    true_positives = len(true_edges & est_edges)
    false_positives = len(est_edges - true_edges)
    false_negatives = len(true_edges - est_edges)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def orientation_accuracy(
    estimated: CPDAG,
    true_dag: DAG,
) -> float:
    """Compute accuracy of edge orientations.

    For edges that are correctly identified in skeleton:
    - Count directed edges with correct orientation
    - Undirected edges are counted as 0.5 (could be either)

    Parameters
    ----------
    estimated : CPDAG
        Estimated CPDAG.
    true_dag : DAG
        Ground truth DAG.

    Returns
    -------
    float
        Orientation accuracy (0 to 1).
    """
    n = estimated.n_nodes
    correct = 0.0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if true edge exists
            true_ij = true_dag.has_edge(i, j)
            true_ji = true_dag.has_edge(j, i)

            if not (true_ij or true_ji):
                continue

            # Check estimated
            est_ij = estimated.has_directed_edge(i, j)
            est_ji = estimated.has_directed_edge(j, i)
            est_undir = estimated.has_undirected_edge(i, j)

            if not (est_ij or est_ji or est_undir):
                continue  # Missing edge

            total += 1
            if est_undir:
                correct += 0.5  # Undirected is ambiguous
            elif (true_ij and est_ij) or (true_ji and est_ji):
                correct += 1.0  # Correct orientation

    return correct / total if total > 0 else 1.0


def compute_shd(estimated: CPDAG, true_dag: DAG) -> int:
    """Compute Structural Hamming Distance.

    SHD counts:
    - Missing edges
    - Extra edges
    - Wrongly oriented edges (compelled wrong direction)

    Parameters
    ----------
    estimated : CPDAG
        Estimated CPDAG.
    true_dag : DAG
        Ground truth DAG.

    Returns
    -------
    int
        Structural Hamming distance.
    """
    n = estimated.n_nodes
    shd = 0

    for i in range(n):
        for j in range(i + 1, n):
            # True structure
            true_ij = true_dag.has_edge(i, j)
            true_ji = true_dag.has_edge(j, i)
            true_edge = true_ij or true_ji

            # Estimated structure
            est_ij = estimated.has_directed_edge(i, j)
            est_ji = estimated.has_directed_edge(j, i)
            est_undir = estimated.has_undirected_edge(i, j)
            est_edge = est_ij or est_ji or est_undir

            if true_edge and not est_edge:
                shd += 1  # Missing edge
            elif not true_edge and est_edge:
                shd += 1  # Extra edge
            elif true_edge and est_edge:
                # Check orientation for directed edges
                if est_ij and true_ji:
                    shd += 1  # Wrong direction
                elif est_ji and true_ij:
                    shd += 1  # Wrong direction
                # Undirected edges don't count as wrong

    return shd


def is_markov_equivalent(dag1: DAG, dag2: DAG) -> bool:
    """Check if two DAGs are Markov equivalent.

    Two DAGs are Markov equivalent iff they have:
    1. Same skeleton
    2. Same v-structures (immoralities)

    Parameters
    ----------
    dag1, dag2 : DAG
        DAGs to compare.

    Returns
    -------
    bool
        True if DAGs are Markov equivalent.
    """
    if dag1.n_nodes != dag2.n_nodes:
        return False

    n = dag1.n_nodes

    # Check same skeleton
    for i in range(n):
        for j in range(i + 1, n):
            edge1 = dag1.has_edge(i, j) or dag1.has_edge(j, i)
            edge2 = dag2.has_edge(i, j) or dag2.has_edge(j, i)
            if edge1 != edge2:
                return False

    # Check same v-structures
    def get_v_structures(dag: DAG) -> set:
        v_structs = set()
        for z in range(n):
            parents = list(dag.parents(z))
            for i, x in enumerate(parents):
                for y in parents[i + 1 :]:
                    if not dag.has_edge(x, y) and not dag.has_edge(y, x):
                        # Canonicalize: smaller index first
                        v_structs.add((min(x, y), z, max(x, y)))
        return v_structs

    return get_v_structures(dag1) == get_v_structures(dag2)
