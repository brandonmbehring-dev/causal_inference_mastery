"""Greedy Equivalence Search (GES) Algorithm.

Session 138: Score-based causal discovery.

GES (Chickering, 2002) searches for the DAG that maximizes a
decomposable score (e.g., BIC). It proceeds in two phases:

1. **Forward Phase**: Start with empty graph, greedily add edges
   that improve the score until no improvement possible.

2. **Backward Phase**: Greedily remove edges that improve the score
   until no improvement possible.

The output is a CPDAG representing the Markov equivalence class.

Properties
----------
- **Consistency**: GES recovers the true CPDAG asymptotically
  (under faithfulness and correct model specification)
- **Score-equivalent**: All DAGs in the equivalence class have
  the same score
- **Polynomial complexity**: O(n^3 * p^4) worst case

Comparison with PC
------------------
| Aspect | PC | GES |
|--------|-------|-----|
| Approach | Constraint-based | Score-based |
| Tests | CI tests | Score optimization |
| Tuning | α (significance) | Score penalty |
| Speed | O(p^2) CI tests | O(n^3 p^4) |

References
----------
- Chickering (2002). Optimal structure identification with greedy search.
- Meek (1997). Graphical models: Selecting causal and statistical models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .score_functions import LocalScore, ScoreType, local_score, total_score
from .types import CPDAG, DAG


@dataclass
class GESResult:
    """Result from GES algorithm.

    Attributes
    ----------
    cpdag : CPDAG
        Estimated CPDAG (Markov equivalence class)
    score : float
        Final score
    n_forward_steps : int
        Number of edges added in forward phase
    n_backward_steps : int
        Number of edges removed in backward phase
    forward_scores : List[float]
        Score after each forward step
    backward_scores : List[float]
        Score after each backward step
    n_vars : int
        Number of variables
    n_samples : int
        Number of samples
    score_type : str
        Score function used
    """

    cpdag: CPDAG
    score: float
    n_forward_steps: int
    n_backward_steps: int
    forward_scores: List[float] = field(default_factory=list)
    backward_scores: List[float] = field(default_factory=list)
    n_vars: int = 0
    n_samples: int = 0
    score_type: str = "bic"

    def n_edges(self) -> int:
        """Return total number of edges in CPDAG."""
        return self.cpdag.n_directed_edges() + self.cpdag.n_undirected_edges()

    def structural_hamming_distance(self, true_dag: DAG) -> int:
        """Compute structural Hamming distance to true DAG.

        SHD counts: missing edges + extra edges + wrongly oriented edges.
        """
        from .utils import compute_shd

        return compute_shd(self.cpdag, true_dag)

    def skeleton_f1(self, true_dag: DAG) -> float:
        """Compute skeleton F1 score."""
        from .utils import skeleton_f1 as compute_skeleton_f1

        # Get estimated skeleton
        estimated_skeleton = self.cpdag.to_skeleton()

        # skeleton_f1 returns (precision, recall, f1) tuple
        _, _, f1 = compute_skeleton_f1(estimated_skeleton, true_dag)
        return f1


def _get_neighbors(adjacency: np.ndarray, node: int) -> Set[int]:
    """Get neighbors (connected by undirected edge) of a node."""
    n = adjacency.shape[0]
    neighbors = set()
    for j in range(n):
        if j != node:
            # Undirected edge: both directions present or neither
            if adjacency[node, j] == 1 and adjacency[j, node] == 1:
                neighbors.add(j)
    return neighbors


def _get_parents(adjacency: np.ndarray, node: int) -> Set[int]:
    """Get parents (directed edges into node) of a node."""
    n = adjacency.shape[0]
    parents = set()
    for j in range(n):
        if j != node:
            # Directed edge j -> node: adj[j, node] = 1, adj[node, j] = 0
            if adjacency[j, node] == 1 and adjacency[node, j] == 0:
                parents.add(j)
    return parents


def _get_children(adjacency: np.ndarray, node: int) -> Set[int]:
    """Get children (directed edges out of node) of a node."""
    n = adjacency.shape[0]
    children = set()
    for j in range(n):
        if j != node:
            # Directed edge node -> j: adj[node, j] = 1, adj[j, node] = 0
            if adjacency[node, j] == 1 and adjacency[j, node] == 0:
                children.add(j)
    return children


def _is_clique(adjacency: np.ndarray, nodes: Set[int]) -> bool:
    """Check if a set of nodes forms a clique (all connected)."""
    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u, v = node_list[i], node_list[j]
            # Must be connected (either direction or undirected)
            if adjacency[u, v] == 0 and adjacency[v, u] == 0:
                return False
    return True


def _valid_insert(adjacency: np.ndarray, x: int, y: int, T: Set[int]) -> bool:
    """Check if insert(x, y, T) is valid.

    Insert adds edge x -> y and orients all edges from T to y.

    Valid if:
    1. x and y are not adjacent
    2. T ⊆ neighbors(y)
    3. Na(y) ∪ {x} forms a clique (Na = neighbors + parents)
    4. No new cycles
    """
    n = adjacency.shape[0]

    # 1. x and y not adjacent
    if adjacency[x, y] == 1 or adjacency[y, x] == 1:
        return False

    # 2. T subset of neighbors(y)
    neighbors_y = _get_neighbors(adjacency, y)
    if not T.issubset(neighbors_y):
        return False

    # 3. Na(y) ∪ {x} is clique
    Na_y = neighbors_y | _get_parents(adjacency, y)
    clique_check = Na_y | {x}
    if not _is_clique(adjacency, clique_check):
        return False

    # 4. No cycle (simplified check: x not reachable from y)
    # BFS from y to check if x is reachable
    visited = set()
    queue = [y]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == x:
            return False
        # Follow directed edges and undirected edges
        for j in range(n):
            if adjacency[node, j] == 1:  # outgoing edge
                if j not in visited:
                    queue.append(j)

    return True


def _valid_delete(adjacency: np.ndarray, x: int, y: int, H: Set[int]) -> bool:
    """Check if delete(x, y, H) is valid.

    Delete removes edge x - y and orients edges from Na(y) \\ H to y.

    Valid if:
    1. x and y are adjacent
    2. H ⊆ neighbors(y)
    3. Na(y) \\ H is a clique
    """
    # 1. x and y adjacent
    if adjacency[x, y] == 0 and adjacency[y, x] == 0:
        return False

    # 2. H subset of neighbors(y)
    neighbors_y = _get_neighbors(adjacency, y)
    if not H.issubset(neighbors_y):
        return False

    # 3. Na(y) \ H is clique
    Na_y = neighbors_y | _get_parents(adjacency, y)
    remaining = Na_y - H - {x}
    if not _is_clique(adjacency, remaining):
        return False

    return True


def _apply_insert(adjacency: np.ndarray, x: int, y: int, T: Set[int]) -> np.ndarray:
    """Apply insert(x, y, T) operation.

    1. Add edge x -> y
    2. For each t in T, orient t - y as t -> y
    """
    adj = adjacency.copy()
    adj[x, y] = 1  # x -> y
    adj[y, x] = 0

    for t in T:
        adj[t, y] = 1  # t -> y
        adj[y, t] = 0

    return adj


def _apply_delete(adjacency: np.ndarray, x: int, y: int, H: Set[int]) -> np.ndarray:
    """Apply delete(x, y, H) operation.

    1. Remove edge x - y
    2. For each h in H, orient edges as h -> y
    """
    adj = adjacency.copy()
    adj[x, y] = 0
    adj[y, x] = 0

    for h in H:
        adj[h, y] = 1
        adj[y, h] = 0

    return adj


def _score_insert(
    data: np.ndarray,
    adjacency: np.ndarray,
    x: int,
    y: int,
    T: Set[int],
    score_type: ScoreType,
    cache: Dict,
) -> float:
    """Compute score change from insert(x, y, T)."""
    # Current parents of y
    old_parents = _get_parents(adjacency, y)

    # New parents of y: old + x + T (T becomes parents from neighbors)
    new_parents = old_parents | {x} | T

    # Score change for y
    old_score = local_score(data, y, old_parents, score_type, cache)
    new_score = local_score(data, y, new_parents, score_type, cache)

    return new_score.score - old_score.score


def _score_delete(
    data: np.ndarray,
    adjacency: np.ndarray,
    x: int,
    y: int,
    H: Set[int],
    score_type: ScoreType,
    cache: Dict,
) -> float:
    """Compute score change from delete(x, y, H)."""
    # Current parents of y
    old_parents = _get_parents(adjacency, y)
    neighbors_y = _get_neighbors(adjacency, y)

    # If x -> y (directed), remove x from parents
    # If x - y (undirected), no change to parents
    if x in old_parents:
        base_parents = old_parents - {x}
    else:
        base_parents = old_parents

    # New parents: base + (Na_y \ H) oriented as parents
    # Na_y \ H - x becomes parents
    Na_y = neighbors_y | old_parents
    new_directed = (Na_y - H - {x}) & neighbors_y  # Only neighbors become parents
    new_parents = base_parents | new_directed

    old_score = local_score(data, y, old_parents, score_type, cache)
    new_score = local_score(data, y, new_parents, score_type, cache)

    return new_score.score - old_score.score


def ges_forward(
    data: np.ndarray,
    adjacency: np.ndarray,
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
    max_parents: int = 10,
) -> Tuple[np.ndarray, int, List[float]]:
    """GES forward phase: greedily add edges.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    adjacency : np.ndarray
        Current adjacency matrix
    score_type : ScoreType
        Score function to use
    cache : Optional[Dict]
        Cache for local scores
    max_parents : int
        Maximum number of parents per node

    Returns
    -------
    Tuple[np.ndarray, int, List[float]]
        (final adjacency, n_steps, scores)
    """
    if cache is None:
        cache = {}

    n_vars = data.shape[1]
    adj = adjacency.copy()
    n_steps = 0
    scores = []

    while True:
        best_delta = 0.0
        best_op = None

        # Try all possible insert(x, y, T) operations
        for x in range(n_vars):
            for y in range(n_vars):
                if x == y:
                    continue

                # Check if already adjacent
                if adj[x, y] == 1 or adj[y, x] == 1:
                    continue

                # Check parent limit
                if len(_get_parents(adj, y)) >= max_parents:
                    continue

                # Try all subsets T of neighbors(y)
                neighbors_y = _get_neighbors(adj, y)
                for T_size in range(len(neighbors_y) + 1):
                    for T_tuple in _subsets_of_size(neighbors_y, T_size):
                        T = set(T_tuple)
                        if _valid_insert(adj, x, y, T):
                            delta = _score_insert(data, adj, x, y, T, score_type, cache)
                            if delta > best_delta:
                                best_delta = delta
                                best_op = ("insert", x, y, T)

        if best_op is None:
            break

        # Apply best operation
        _, x, y, T = best_op
        adj = _apply_insert(adj, x, y, T)
        n_steps += 1
        scores.append(total_score(data, adj, score_type, cache))

    return adj, n_steps, scores


def ges_backward(
    data: np.ndarray,
    adjacency: np.ndarray,
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
) -> Tuple[np.ndarray, int, List[float]]:
    """GES backward phase: greedily remove edges.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    adjacency : np.ndarray
        Current adjacency matrix
    score_type : ScoreType
        Score function to use
    cache : Optional[Dict]
        Cache for local scores

    Returns
    -------
    Tuple[np.ndarray, int, List[float]]
        (final adjacency, n_steps, scores)
    """
    if cache is None:
        cache = {}

    n_vars = data.shape[1]
    adj = adjacency.copy()
    n_steps = 0
    scores = []

    while True:
        best_delta = 0.0
        best_op = None

        # Try all possible delete(x, y, H) operations
        for x in range(n_vars):
            for y in range(n_vars):
                if x == y:
                    continue

                # Check if adjacent
                if adj[x, y] == 0 and adj[y, x] == 0:
                    continue

                # Try all subsets H of neighbors(y)
                neighbors_y = _get_neighbors(adj, y)
                for H_size in range(len(neighbors_y) + 1):
                    for H_tuple in _subsets_of_size(neighbors_y, H_size):
                        H = set(H_tuple)
                        if _valid_delete(adj, x, y, H):
                            delta = _score_delete(data, adj, x, y, H, score_type, cache)
                            if delta > best_delta:
                                best_delta = delta
                                best_op = ("delete", x, y, H)

        if best_op is None:
            break

        # Apply best operation
        _, x, y, H = best_op
        adj = _apply_delete(adj, x, y, H)
        n_steps += 1
        scores.append(total_score(data, adj, score_type, cache))

    return adj, n_steps, scores


def _subsets_of_size(s: Set[int], k: int):
    """Generate all subsets of size k from set s."""
    from itertools import combinations

    return combinations(s, k)


def _adjacency_to_cpdag(adjacency: np.ndarray, var_names: Optional[List[str]] = None) -> CPDAG:
    """Convert adjacency matrix to CPDAG object."""
    n = adjacency.shape[0]
    if var_names is None:
        var_names = [f"X{i}" for i in range(n)]

    cpdag = CPDAG(n, var_names)

    for i in range(n):
        for j in range(n):
            if i < j:
                if adjacency[i, j] == 1 and adjacency[j, i] == 1:
                    # Undirected edge
                    cpdag.add_undirected_edge(i, j)
                elif adjacency[i, j] == 1 and adjacency[j, i] == 0:
                    # Directed edge i -> j
                    cpdag.add_directed_edge(i, j)
                elif adjacency[i, j] == 0 and adjacency[j, i] == 1:
                    # Directed edge j -> i
                    cpdag.add_directed_edge(j, i)

    return cpdag


def ges_algorithm(
    data: np.ndarray,
    score_type: str = "bic",
    var_names: Optional[List[str]] = None,
    max_parents: int = 10,
    verbose: bool = False,
) -> GESResult:
    """Greedy Equivalence Search for causal discovery.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    score_type : str
        Score function: "bic" (default), "aic"
    var_names : Optional[List[str]]
        Variable names
    max_parents : int
        Maximum number of parents per node
    verbose : bool
        Print progress

    Returns
    -------
    GESResult
        GES algorithm result

    Example
    -------
    >>> import numpy as np
    >>> from causal_inference.discovery import ges_algorithm
    >>> np.random.seed(42)
    >>> n = 500
    >>> X1 = np.random.randn(n)
    >>> X2 = 0.8 * X1 + np.random.randn(n) * 0.5
    >>> X3 = 0.6 * X2 + np.random.randn(n) * 0.5
    >>> data = np.column_stack([X1, X2, X3])
    >>> result = ges_algorithm(data, score_type="bic")
    >>> print(f"Edges found: {result.n_edges()}")
    """
    # Validate inputs
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    n_samples, n_vars = data.shape

    if n_samples < 10:
        raise ValueError(f"Too few samples: {n_samples} < 10")

    if n_vars < 2:
        raise ValueError(f"Need at least 2 variables, got {n_vars}")

    # Parse score type
    if score_type.lower() == "bic":
        st = ScoreType.BIC
    elif score_type.lower() == "aic":
        st = ScoreType.AIC
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    if var_names is None:
        var_names = [f"X{i}" for i in range(n_vars)]

    # Initialize empty graph
    adjacency = np.zeros((n_vars, n_vars), dtype=float)
    cache: Dict = {}

    # Forward phase
    if verbose:
        print("GES Forward Phase...")
    adj_fwd, n_fwd, scores_fwd = ges_forward(data, adjacency, st, cache, max_parents)

    if verbose:
        print(f"  Added {n_fwd} edges")

    # Backward phase
    if verbose:
        print("GES Backward Phase...")
    adj_final, n_bwd, scores_bwd = ges_backward(data, adj_fwd, st, cache)

    if verbose:
        print(f"  Removed {n_bwd} edges")

    # Compute final score
    final_score = total_score(data, adj_final, st, cache)

    # Convert to CPDAG
    cpdag = _adjacency_to_cpdag(adj_final, var_names)

    return GESResult(
        cpdag=cpdag,
        score=final_score,
        n_forward_steps=n_fwd,
        n_backward_steps=n_bwd,
        forward_scores=scores_fwd,
        backward_scores=scores_bwd,
        n_vars=n_vars,
        n_samples=n_samples,
        score_type=score_type,
    )
