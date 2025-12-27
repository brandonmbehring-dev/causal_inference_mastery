"""Type definitions for causal discovery algorithms.

Session 133: Causal Discovery (PC Algorithm + LiNGAM)

This module provides graph data structures and result types for causal
structure learning from observational data.

Key Types
---------
Graph : Undirected graph for skeleton representation
DAG : Directed acyclic graph for causal structure
CPDAG : Completed partially directed acyclic graph (Markov equivalence class)
PCResult : Result from PC algorithm
LiNGAMResult : Result from LiNGAM algorithm

References
----------
- Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
- Shimizu et al. (2006). A Linear Non-Gaussian Acyclic Model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TypedDict

import numpy as np


class EdgeType(Enum):
    """Edge types in mixed graphs."""

    UNDIRECTED = "---"  # X --- Y (unknown direction)
    DIRECTED = "-->"  # X --> Y (X causes Y)
    BIDIRECTED = "<->"  # X <-> Y (latent confounder)
    PARTIALLY_DIRECTED = "o->"  # X o-> Y (partially oriented)


@dataclass
class Graph:
    """Undirected graph for skeleton representation.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the graph.
    node_names : List[str]
        Names of nodes (variables).
    adjacency : np.ndarray
        Symmetric adjacency matrix (1 = edge, 0 = no edge).

    Example
    -------
    >>> g = Graph(n_nodes=3, node_names=["X", "Y", "Z"])
    >>> g.add_edge(0, 1)  # X --- Y
    >>> g.has_edge(0, 1)
    True
    """

    n_nodes: int
    node_names: List[str] = field(default_factory=list)
    adjacency: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        """Initialize adjacency matrix and node names."""
        if self.adjacency is None:
            self.adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        if not self.node_names:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

    def add_edge(self, i: int, j: int) -> None:
        """Add undirected edge between nodes i and j."""
        self.adjacency[i, j] = 1
        self.adjacency[j, i] = 1

    def remove_edge(self, i: int, j: int) -> None:
        """Remove edge between nodes i and j."""
        self.adjacency[i, j] = 0
        self.adjacency[j, i] = 0

    def has_edge(self, i: int, j: int) -> bool:
        """Check if edge exists between nodes i and j."""
        return bool(self.adjacency[i, j])

    def neighbors(self, i: int) -> Set[int]:
        """Get neighbors of node i."""
        return set(np.where(self.adjacency[i, :] == 1)[0])

    def edges(self) -> List[Tuple[int, int]]:
        """Get list of edges (i, j) where i < j."""
        edge_list = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.adjacency[i, j]:
                    edge_list.append((i, j))
        return edge_list

    def n_edges(self) -> int:
        """Count number of edges."""
        return int(np.sum(self.adjacency) // 2)

    def copy(self) -> "Graph":
        """Create a copy of the graph."""
        return Graph(
            n_nodes=self.n_nodes,
            node_names=self.node_names.copy(),
            adjacency=self.adjacency.copy(),
        )

    @classmethod
    def complete(cls, n_nodes: int, node_names: Optional[List[str]] = None) -> "Graph":
        """Create complete graph (all edges present)."""
        adj = np.ones((n_nodes, n_nodes), dtype=np.int8)
        np.fill_diagonal(adj, 0)
        names = node_names or [f"X{i}" for i in range(n_nodes)]
        return cls(n_nodes=n_nodes, node_names=names, adjacency=adj)


@dataclass
class DAG:
    """Directed acyclic graph for causal structure.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.
    node_names : List[str]
        Names of nodes.
    adjacency : np.ndarray
        Adjacency matrix where adjacency[i,j] = 1 means i -> j.

    Example
    -------
    >>> dag = DAG(n_nodes=3, node_names=["X", "Y", "Z"])
    >>> dag.add_edge(0, 1)  # X -> Y
    >>> dag.add_edge(1, 2)  # Y -> Z
    >>> dag.parents(2)
    {1}
    """

    n_nodes: int
    node_names: List[str] = field(default_factory=list)
    adjacency: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        """Initialize adjacency matrix and node names."""
        if self.adjacency is None:
            self.adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        if not self.node_names:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

    def add_edge(self, parent: int, child: int) -> None:
        """Add directed edge parent -> child."""
        self.adjacency[parent, child] = 1

    def remove_edge(self, parent: int, child: int) -> None:
        """Remove directed edge parent -> child."""
        self.adjacency[parent, child] = 0

    def has_edge(self, parent: int, child: int) -> bool:
        """Check if directed edge parent -> child exists."""
        return bool(self.adjacency[parent, child])

    def parents(self, i: int) -> Set[int]:
        """Get parents of node i."""
        return set(np.where(self.adjacency[:, i] == 1)[0])

    def children(self, i: int) -> Set[int]:
        """Get children of node i."""
        return set(np.where(self.adjacency[i, :] == 1)[0])

    def ancestors(self, i: int) -> Set[int]:
        """Get all ancestors of node i."""
        anc = set()
        to_visit = list(self.parents(i))
        while to_visit:
            node = to_visit.pop()
            if node not in anc:
                anc.add(node)
                to_visit.extend(self.parents(node))
        return anc

    def descendants(self, i: int) -> Set[int]:
        """Get all descendants of node i."""
        desc = set()
        to_visit = list(self.children(i))
        while to_visit:
            node = to_visit.pop()
            if node not in desc:
                desc.add(node)
                to_visit.extend(self.children(node))
        return desc

    def edges(self) -> List[Tuple[int, int]]:
        """Get list of directed edges (parent, child)."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency[i, j]:
                    edges.append((i, j))
        return edges

    def n_edges(self) -> int:
        """Count number of directed edges."""
        return int(np.sum(self.adjacency))

    def is_acyclic(self) -> bool:
        """Check if graph is acyclic using topological sort."""
        try:
            _ = self.topological_order()
            return True
        except ValueError:
            return False

    def topological_order(self) -> List[int]:
        """Return topological ordering of nodes.

        Raises
        ------
        ValueError
            If graph contains a cycle.
        """
        in_degree = np.sum(self.adjacency, axis=0)
        queue = [i for i in range(self.n_nodes) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != self.n_nodes:
            raise ValueError("Graph contains a cycle")

        return order

    def copy(self) -> "DAG":
        """Create a copy of the DAG."""
        return DAG(
            n_nodes=self.n_nodes,
            node_names=self.node_names.copy(),
            adjacency=self.adjacency.copy(),
        )

    @classmethod
    def from_adjacency(
        cls, adjacency: np.ndarray, node_names: Optional[List[str]] = None
    ) -> "DAG":
        """Create DAG from adjacency matrix."""
        n_nodes = adjacency.shape[0]
        names = node_names or [f"X{i}" for i in range(n_nodes)]
        return cls(n_nodes=n_nodes, node_names=names, adjacency=adjacency.copy())


@dataclass
class CPDAG:
    """Completed partially directed acyclic graph (Markov equivalence class).

    A CPDAG represents the equivalence class of DAGs that encode the same
    conditional independencies. It contains:
    - Directed edges: Edges with known orientation (compelled)
    - Undirected edges: Edges with unknown orientation (reversible)

    Attributes
    ----------
    n_nodes : int
        Number of nodes.
    node_names : List[str]
        Names of nodes.
    directed : np.ndarray
        Directed edges (directed[i,j] = 1 means i -> j is compelled).
    undirected : np.ndarray
        Undirected edges (symmetric, undirected[i,j] = 1 means i --- j).
    """

    n_nodes: int
    node_names: List[str] = field(default_factory=list)
    directed: np.ndarray = field(default=None)
    undirected: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        """Initialize matrices and node names."""
        if self.directed is None:
            self.directed = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        if self.undirected is None:
            self.undirected = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        if not self.node_names:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

    def add_directed_edge(self, parent: int, child: int) -> None:
        """Add compelled directed edge parent -> child."""
        self.directed[parent, child] = 1
        # Remove from undirected if present
        self.undirected[parent, child] = 0
        self.undirected[child, parent] = 0

    def add_undirected_edge(self, i: int, j: int) -> None:
        """Add reversible undirected edge i --- j."""
        self.undirected[i, j] = 1
        self.undirected[j, i] = 1
        # Remove from directed if present
        self.directed[i, j] = 0
        self.directed[j, i] = 0

    def has_directed_edge(self, parent: int, child: int) -> bool:
        """Check if compelled edge parent -> child exists."""
        return bool(self.directed[parent, child])

    def has_undirected_edge(self, i: int, j: int) -> bool:
        """Check if reversible edge i --- j exists."""
        return bool(self.undirected[i, j])

    def has_any_edge(self, i: int, j: int) -> bool:
        """Check if any edge exists between i and j."""
        return bool(
            self.directed[i, j]
            or self.directed[j, i]
            or self.undirected[i, j]
        )

    def adjacent(self, i: int) -> Set[int]:
        """Get all nodes adjacent to i (any edge type)."""
        dir_out = set(np.where(self.directed[i, :] == 1)[0])
        dir_in = set(np.where(self.directed[:, i] == 1)[0])
        undir = set(np.where(self.undirected[i, :] == 1)[0])
        return dir_out | dir_in | undir

    def n_directed_edges(self) -> int:
        """Count compelled directed edges."""
        return int(np.sum(self.directed))

    def n_undirected_edges(self) -> int:
        """Count reversible undirected edges."""
        return int(np.sum(self.undirected) // 2)

    def to_skeleton(self) -> Graph:
        """Convert to undirected skeleton."""
        adj = (
            self.undirected
            | self.directed
            | self.directed.T
        ).astype(np.int8)
        return Graph(
            n_nodes=self.n_nodes,
            node_names=self.node_names.copy(),
            adjacency=adj,
        )

    def copy(self) -> "CPDAG":
        """Create a copy of the CPDAG."""
        return CPDAG(
            n_nodes=self.n_nodes,
            node_names=self.node_names.copy(),
            directed=self.directed.copy(),
            undirected=self.undirected.copy(),
        )

    @classmethod
    def from_skeleton(cls, skeleton: Graph) -> "CPDAG":
        """Create CPDAG from skeleton (all edges undirected)."""
        return cls(
            n_nodes=skeleton.n_nodes,
            node_names=skeleton.node_names.copy(),
            directed=np.zeros_like(skeleton.adjacency),
            undirected=skeleton.adjacency.copy(),
        )


@dataclass
class PCResult:
    """Result from PC algorithm.

    Attributes
    ----------
    cpdag : CPDAG
        Estimated CPDAG (Markov equivalence class).
    skeleton : Graph
        Undirected skeleton.
    separating_sets : Dict[Tuple[int, int], FrozenSet[int]]
        Separating sets for removed edges.
    n_ci_tests : int
        Number of conditional independence tests performed.
    alpha : float
        Significance level used for CI tests.
    execution_time_ms : float
        Execution time in milliseconds.
    """

    cpdag: CPDAG
    skeleton: Graph
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]]
    n_ci_tests: int
    alpha: float
    execution_time_ms: float = 0.0

    def structural_hamming_distance(self, true_dag: DAG) -> int:
        """Compute structural Hamming distance to true DAG.

        SHD counts:
        - Missing edges
        - Extra edges
        - Wrongly oriented edges

        Parameters
        ----------
        true_dag : DAG
            Ground truth DAG.

        Returns
        -------
        int
            Structural Hamming distance.
        """
        shd = 0
        cpdag_skeleton = self.cpdag.to_skeleton()

        for i in range(self.cpdag.n_nodes):
            for j in range(i + 1, self.cpdag.n_nodes):
                true_edge = true_dag.has_edge(i, j) or true_dag.has_edge(j, i)
                est_edge = cpdag_skeleton.has_edge(i, j)

                if true_edge and not est_edge:
                    shd += 1  # Missing edge
                elif not true_edge and est_edge:
                    shd += 1  # Extra edge
                elif true_edge and est_edge:
                    # Check orientation
                    true_ij = true_dag.has_edge(i, j)
                    true_ji = true_dag.has_edge(j, i)
                    est_ij = self.cpdag.has_directed_edge(i, j)
                    est_ji = self.cpdag.has_directed_edge(j, i)
                    est_undir = self.cpdag.has_undirected_edge(i, j)

                    if est_undir:
                        # Undirected in CPDAG is ok (could be either direction)
                        pass
                    elif (true_ij and est_ji) or (true_ji and est_ij):
                        shd += 1  # Wrong orientation

        return shd

    def skeleton_f1(self, true_dag: DAG) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 for skeleton recovery.

        Parameters
        ----------
        true_dag : DAG
            Ground truth DAG.

        Returns
        -------
        precision : float
            Fraction of estimated edges that are true.
        recall : float
            Fraction of true edges that are estimated.
        f1 : float
            F1 score.
        """
        n = self.skeleton.n_nodes
        true_edges = set()
        est_edges = set()

        for i in range(n):
            for j in range(i + 1, n):
                if true_dag.has_edge(i, j) or true_dag.has_edge(j, i):
                    true_edges.add((i, j))
                if self.skeleton.has_edge(i, j):
                    est_edges.add((i, j))

        true_positives = len(true_edges & est_edges)
        false_positives = len(est_edges - true_edges)
        false_negatives = len(true_edges - est_edges)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1


@dataclass
class LiNGAMResult:
    """Result from LiNGAM algorithm.

    Attributes
    ----------
    dag : DAG
        Estimated unique DAG.
    causal_order : List[int]
        Topological ordering of nodes (causal order).
    adjacency_matrix : np.ndarray
        Weighted adjacency matrix B where B[i,j] is effect i -> j.
    pruned_adjacency : np.ndarray
        Adjacency after pruning small coefficients.
    ica_mixing_matrix : np.ndarray
        ICA mixing matrix A.
    execution_time_ms : float
        Execution time in milliseconds.
    """

    dag: DAG
    causal_order: List[int]
    adjacency_matrix: np.ndarray
    pruned_adjacency: Optional[np.ndarray] = None
    ica_mixing_matrix: Optional[np.ndarray] = None
    execution_time_ms: float = 0.0

    def structural_hamming_distance(self, true_dag: DAG) -> int:
        """Compute structural Hamming distance to true DAG.

        Parameters
        ----------
        true_dag : DAG
            Ground truth DAG.

        Returns
        -------
        int
            Structural Hamming distance.
        """
        shd = 0
        for i in range(self.dag.n_nodes):
            for j in range(self.dag.n_nodes):
                if i != j:
                    true_edge = true_dag.has_edge(i, j)
                    est_edge = self.dag.has_edge(i, j)
                    if true_edge != est_edge:
                        shd += 1
        return shd

    def causal_order_accuracy(self, true_order: List[int]) -> float:
        """Compute accuracy of causal ordering.

        Parameters
        ----------
        true_order : List[int]
            Ground truth topological order.

        Returns
        -------
        float
            Fraction of correctly ordered pairs.
        """
        n = len(true_order)
        true_pairs = set()
        est_pairs = set()

        for i in range(n):
            for j in range(i + 1, n):
                true_pairs.add((true_order[i], true_order[j]))
                est_pairs.add((self.causal_order[i], self.causal_order[j]))

        correct = len(true_pairs & est_pairs)
        total = len(true_pairs)
        return correct / total if total > 0 else 1.0


class CITestType(Enum):
    """Types of conditional independence tests."""

    FISHER_Z = "fisher_z"  # Fisher's Z-transform (Gaussian)
    PARTIAL_CORRELATION = "partial_correlation"  # Partial correlation
    G_SQUARED = "g_squared"  # G² test (categorical)
    KCI = "kernel_ci"  # Kernel conditional independence


@dataclass
class CITestResult:
    """Result from a conditional independence test.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        P-value for the test.
    is_independent : bool
        Whether null hypothesis (independence) is accepted.
    alpha : float
        Significance level used.
    """

    statistic: float
    p_value: float
    is_independent: bool
    alpha: float


# =============================================================================
# FCI Types (Session 134)
# =============================================================================


class EdgeMark(Enum):
    """Edge marks for PAG (Partial Ancestral Graph) edges.

    FCI algorithm outputs PAGs with three types of edge marks:
    - TAIL (-): Definite non-arrowhead
    - ARROW (>): Definite arrowhead
    - CIRCLE (o): Uncertain (could be either)

    Examples
    --------
    >>> mark = EdgeMark.ARROW
    >>> mark.value
    '>'
    >>> EdgeMark.from_string('o')
    EdgeMark.CIRCLE
    """

    TAIL = "-"  # Definite tail (non-arrowhead)
    ARROW = ">"  # Definite arrowhead
    CIRCLE = "o"  # Unknown (could be tail or arrow)

    @classmethod
    def from_string(cls, s: str) -> "EdgeMark":
        """Create EdgeMark from string representation."""
        mapping = {"-": cls.TAIL, ">": cls.ARROW, "o": cls.CIRCLE}
        if s not in mapping:
            raise ValueError(f"Unknown edge mark: {s}")
        return mapping[s]

    def __str__(self) -> str:
        return self.value


@dataclass
class PAGEdge:
    """An edge in a PAG with endpoint marks.

    Attributes
    ----------
    i : int
        First endpoint node.
    j : int
        Second endpoint node.
    mark_i : EdgeMark
        Mark at node i endpoint.
    mark_j : EdgeMark
        Mark at node j endpoint.

    Examples
    --------
    >>> edge = PAGEdge(0, 1, EdgeMark.TAIL, EdgeMark.ARROW)  # 0 -> 1
    >>> edge.is_directed()
    True
    >>> edge.to_string()
    'X0 -> X1'
    """

    i: int
    j: int
    mark_i: EdgeMark
    mark_j: EdgeMark

    def is_directed(self) -> bool:
        """Check if edge is definitely directed (i -> j)."""
        return self.mark_i == EdgeMark.TAIL and self.mark_j == EdgeMark.ARROW

    def is_bidirected(self) -> bool:
        """Check if edge is definitely bidirected (i <-> j)."""
        return self.mark_i == EdgeMark.ARROW and self.mark_j == EdgeMark.ARROW

    def is_undirected(self) -> bool:
        """Check if edge is definitely undirected (i - j)."""
        return self.mark_i == EdgeMark.TAIL and self.mark_j == EdgeMark.TAIL

    def is_partially_directed(self) -> bool:
        """Check if edge has circle mark (uncertain)."""
        return self.mark_i == EdgeMark.CIRCLE or self.mark_j == EdgeMark.CIRCLE

    def is_into(self, node: int) -> bool:
        """Check if there is definitely an arrowhead into node."""
        if node == self.j:
            return self.mark_j == EdgeMark.ARROW
        elif node == self.i:
            return self.mark_i == EdgeMark.ARROW
        return False

    def is_out_of(self, node: int) -> bool:
        """Check if there is definitely a tail at node."""
        if node == self.i:
            return self.mark_i == EdgeMark.TAIL
        elif node == self.j:
            return self.mark_j == EdgeMark.TAIL
        return False

    def to_string(self, node_names: Optional[List[str]] = None) -> str:
        """Convert to human-readable string."""
        name_i = node_names[self.i] if node_names else f"X{self.i}"
        name_j = node_names[self.j] if node_names else f"X{self.j}"

        mark_to_left = {"<": EdgeMark.ARROW, "-": EdgeMark.TAIL, "o": EdgeMark.CIRCLE}
        left = "<" if self.mark_i == EdgeMark.ARROW else self.mark_i.value
        right = self.mark_j.value

        return f"{name_i} {left}-{right} {name_j}"


@dataclass
class PAG:
    """Partial Ancestral Graph for FCI algorithm output.

    A PAG represents the Markov equivalence class of MAGs (Maximal Ancestral Graphs)
    that encode the same conditional independencies in the presence of latent confounders.

    Edge endpoints are represented by a 3D matrix:
    - endpoints[i, j, 0] = mark at i for edge i-j
    - endpoints[i, j, 1] = mark at j for edge i-j

    Marks are: 0 = no edge, 1 = tail, 2 = arrow, 3 = circle

    Attributes
    ----------
    n_nodes : int
        Number of observed variables.
    node_names : List[str]
        Names of nodes.
    endpoints : np.ndarray
        Edge endpoint matrix (n_nodes, n_nodes, 2).

    Example
    -------
    >>> pag = PAG(n_nodes=3)
    >>> pag.add_edge(0, 1, EdgeMark.TAIL, EdgeMark.ARROW)  # 0 -> 1
    >>> pag.add_edge(1, 2, EdgeMark.CIRCLE, EdgeMark.ARROW)  # 1 o-> 2
    >>> pag.has_edge(0, 1)
    True
    >>> pag.is_definitely_ancestor(0, 1)
    True
    """

    n_nodes: int
    node_names: List[str] = field(default_factory=list)
    endpoints: np.ndarray = field(default=None)

    # Mark encoding in matrix
    NO_EDGE = 0
    MARK_TAIL = 1
    MARK_ARROW = 2
    MARK_CIRCLE = 3

    def __post_init__(self) -> None:
        """Initialize endpoint matrix and node names."""
        if self.endpoints is None:
            self.endpoints = np.zeros((self.n_nodes, self.n_nodes, 2), dtype=np.int8)
        if not self.node_names:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]

    def _mark_to_code(self, mark: EdgeMark) -> int:
        """Convert EdgeMark to internal code."""
        return {EdgeMark.TAIL: self.MARK_TAIL, EdgeMark.ARROW: self.MARK_ARROW, EdgeMark.CIRCLE: self.MARK_CIRCLE}[mark]

    def _code_to_mark(self, code: int) -> Optional[EdgeMark]:
        """Convert internal code to EdgeMark."""
        if code == self.NO_EDGE:
            return None
        return {self.MARK_TAIL: EdgeMark.TAIL, self.MARK_ARROW: EdgeMark.ARROW, self.MARK_CIRCLE: EdgeMark.CIRCLE}[code]

    def add_edge(self, i: int, j: int, mark_i: EdgeMark, mark_j: EdgeMark) -> None:
        """Add edge between i and j with specified marks."""
        self.endpoints[i, j, 0] = self._mark_to_code(mark_i)
        self.endpoints[i, j, 1] = self._mark_to_code(mark_j)
        self.endpoints[j, i, 0] = self._mark_to_code(mark_j)
        self.endpoints[j, i, 1] = self._mark_to_code(mark_i)

    def remove_edge(self, i: int, j: int) -> None:
        """Remove edge between i and j."""
        self.endpoints[i, j, :] = 0
        self.endpoints[j, i, :] = 0

    def has_edge(self, i: int, j: int) -> bool:
        """Check if any edge exists between i and j."""
        return bool(np.any(self.endpoints[i, j, :] > 0))

    def get_edge(self, i: int, j: int) -> Optional[PAGEdge]:
        """Get edge between i and j if it exists."""
        if not self.has_edge(i, j):
            return None
        mark_i = self._code_to_mark(self.endpoints[i, j, 0])
        mark_j = self._code_to_mark(self.endpoints[i, j, 1])
        return PAGEdge(i, j, mark_i, mark_j)

    def set_endpoint(self, i: int, j: int, mark: EdgeMark) -> None:
        """Set the endpoint mark at i for edge i-j."""
        self.endpoints[i, j, 0] = self._mark_to_code(mark)
        self.endpoints[j, i, 1] = self._mark_to_code(mark)

    def get_endpoint(self, i: int, j: int) -> Optional[EdgeMark]:
        """Get the endpoint mark at i for edge i-j."""
        return self._code_to_mark(self.endpoints[i, j, 0])

    def is_definitely_directed(self, i: int, j: int) -> bool:
        """Check if edge is definitely i -> j."""
        return (self.endpoints[i, j, 0] == self.MARK_TAIL and
                self.endpoints[i, j, 1] == self.MARK_ARROW)

    def is_definitely_ancestor(self, i: int, j: int) -> bool:
        """Check if i is definitely an ancestor of j.

        Uses graph reachability following only definite directed edges.
        """
        if i == j:
            return False
        if self.is_definitely_directed(i, j):
            return True

        visited = set()
        queue = [i]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for k in range(self.n_nodes):
                if k != node and self.is_definitely_directed(node, k):
                    if k == j:
                        return True
                    queue.append(k)
        return False

    def possible_ancestors(self, j: int) -> Set[int]:
        """Get nodes that could possibly be ancestors of j."""
        poss_anc = set()
        for i in range(self.n_nodes):
            if i != j and self.has_edge(i, j):
                # Check if i could be an ancestor (has tail/circle at i, arrow/circle at j)
                edge = self.get_edge(i, j)
                if edge and edge.mark_i in {EdgeMark.TAIL, EdgeMark.CIRCLE}:
                    poss_anc.add(i)
        return poss_anc

    def definitely_non_adjacent(self, i: int, j: int) -> bool:
        """Check if i and j are definitely not adjacent."""
        return not self.has_edge(i, j)

    def adjacent(self, i: int) -> Set[int]:
        """Get all nodes adjacent to i."""
        return {j for j in range(self.n_nodes) if j != i and self.has_edge(i, j)}

    def edges(self) -> List[PAGEdge]:
        """Get list of all edges."""
        edge_list = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge = self.get_edge(i, j)
                if edge:
                    edge_list.append(edge)
        return edge_list

    def n_edges(self) -> int:
        """Count number of edges."""
        count = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.has_edge(i, j):
                    count += 1
        return count

    def n_directed_edges(self) -> int:
        """Count definitely directed edges."""
        count = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and self.is_definitely_directed(i, j):
                    count += 1
        return count

    def n_bidirected_edges(self) -> int:
        """Count definitely bidirected edges (latent confounders)."""
        count = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if (self.endpoints[i, j, 0] == self.MARK_ARROW and
                        self.endpoints[i, j, 1] == self.MARK_ARROW):
                    count += 1
        return count

    def n_circle_edges(self) -> int:
        """Count edges with at least one circle mark."""
        count = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.has_edge(i, j):
                    if (self.endpoints[i, j, 0] == self.MARK_CIRCLE or
                            self.endpoints[i, j, 1] == self.MARK_CIRCLE):
                        count += 1
        return count

    def copy(self) -> "PAG":
        """Create a copy of the PAG."""
        return PAG(
            n_nodes=self.n_nodes,
            node_names=self.node_names.copy(),
            endpoints=self.endpoints.copy(),
        )

    @classmethod
    def from_skeleton(cls, skeleton: Graph) -> "PAG":
        """Create PAG from skeleton with all circle marks."""
        pag = cls(n_nodes=skeleton.n_nodes, node_names=skeleton.node_names.copy())
        for i, j in skeleton.edges():
            pag.add_edge(i, j, EdgeMark.CIRCLE, EdgeMark.CIRCLE)
        return pag


@dataclass
class FCIResult:
    """Result from FCI algorithm.

    FCI (Fast Causal Inference) is an extension of PC that handles
    latent confounders by outputting a PAG instead of a CPDAG.

    Attributes
    ----------
    pag : PAG
        Estimated Partial Ancestral Graph.
    skeleton : Graph
        Undirected skeleton.
    separating_sets : Dict[Tuple[int, int], FrozenSet[int]]
        Separating sets for removed edges.
    possible_latent_confounders : List[Tuple[int, int]]
        Pairs of nodes with possible latent confounder (bidirected edges).
    n_ci_tests : int
        Number of conditional independence tests performed.
    alpha : float
        Significance level used for CI tests.
    execution_time_ms : float
        Execution time in milliseconds.

    Example
    -------
    >>> result = fci_algorithm(data, alpha=0.01)
    >>> result.pag.n_bidirected_edges()  # Latent confounders detected
    2
    >>> result.possible_latent_confounders
    [(0, 2), (1, 3)]
    """

    pag: PAG
    skeleton: Graph
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]]
    possible_latent_confounders: List[Tuple[int, int]]
    n_ci_tests: int
    alpha: float
    execution_time_ms: float = 0.0

    def n_definite_directed(self) -> int:
        """Count definitely directed edges."""
        return self.pag.n_directed_edges()

    def n_bidirected(self) -> int:
        """Count bidirected edges (latent confounders)."""
        return self.pag.n_bidirected_edges()

    def n_uncertain(self) -> int:
        """Count edges with uncertain orientation (circle marks)."""
        return self.pag.n_circle_edges()

    def has_latent_confounder(self, i: int, j: int) -> bool:
        """Check if there's a bidirected edge between i and j."""
        return (i, j) in self.possible_latent_confounders or (j, i) in self.possible_latent_confounders
