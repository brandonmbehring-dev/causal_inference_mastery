"""
PCMCI Types for Time Series Causal Discovery.

Session 136: Data structures for PCMCI algorithm (Runge et al., 2019).

PCMCI combines PC algorithm structure learning with time-series conditioning
to discover causal relationships in multivariate temporal data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np


class LinkType(Enum):
    """Type of causal link in time-lagged graph."""

    DIRECTED = "-->"  # Definite causal direction
    UNDIRECTED = "---"  # Undetermined direction (contemporaneous)
    BIDIRECTED = "<->"  # Confounded (latent common cause)
    NONE = ""  # No link


@dataclass
class TimeSeriesLink:
    """
    A directed link in a time-lagged causal graph.

    Represents a causal relationship X_{t-τ} → Y_t where X at time t-τ
    causes Y at time t.

    Attributes
    ----------
    source_var : int
        Index of source variable (cause)
    target_var : int
        Index of target variable (effect)
    lag : int
        Time lag τ (positive integer, 0 for contemporaneous)
    strength : float
        Effect strength (partial correlation or mutual information)
    p_value : float
        P-value from conditional independence test
    link_type : LinkType
        Type of causal link

    Examples
    --------
    >>> link = TimeSeriesLink(source_var=0, target_var=1, lag=2,
    ...                       strength=0.45, p_value=0.001)
    >>> print(link)
    TimeSeriesLink(X0_{t-2} → X1_t, val=0.450, p=0.0010)
    """

    source_var: int
    target_var: int
    lag: int
    strength: float
    p_value: float
    link_type: LinkType = LinkType.DIRECTED

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError(f"Lag must be >= 0, got {self.lag}")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"P-value must be in [0, 1], got {self.p_value}")

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if link is significant at given alpha level."""
        return self.p_value < alpha

    def is_lagged(self) -> bool:
        """Check if this is a lagged (not contemporaneous) link."""
        return self.lag > 0

    def __repr__(self) -> str:
        lag_str = f"t-{self.lag}" if self.lag > 0 else "t"
        arrow = self.link_type.value if self.link_type != LinkType.NONE else "x"
        return (
            f"TimeSeriesLink(X{self.source_var}_{{{lag_str}}} {arrow} "
            f"X{self.target_var}_t, val={self.strength:.3f}, p={self.p_value:.4f})"
        )


@dataclass
class LaggedDAG:
    """
    Time-lagged directed acyclic graph.

    Represents causal structure over time with separate edges for each lag.
    The graph is acyclic within each time slice but allows lagged effects.

    Attributes
    ----------
    n_vars : int
        Number of variables
    max_lag : int
        Maximum lag considered
    adjacency : np.ndarray
        Shape (n_vars, n_vars, max_lag + 1) binary adjacency tensor.
        adjacency[i, j, τ] = 1 means X_i at time t-τ causes X_j at time t.
    weights : np.ndarray
        Shape (n_vars, n_vars, max_lag + 1) edge weights (effect strengths).
    var_names : List[str]
        Variable names

    Examples
    --------
    >>> dag = LaggedDAG(n_vars=3, max_lag=2)
    >>> dag.add_edge(0, 1, lag=1, weight=0.5)  # X0_{t-1} → X1_t
    >>> dag.has_edge(0, 1, lag=1)
    True
    """

    n_vars: int
    max_lag: int
    adjacency: np.ndarray = field(default=None, repr=False)
    weights: np.ndarray = field(default=None, repr=False)
    var_names: List[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.adjacency is None:
            self.adjacency = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1), dtype=np.int8)
        if self.weights is None:
            self.weights = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        if self.var_names is None:
            self.var_names = [f"X{i}" for i in range(self.n_vars)]

    def add_edge(self, source: int, target: int, lag: int, weight: float = 1.0) -> None:
        """Add a directed edge from source at lag to target at time t."""
        if lag < 0 or lag > self.max_lag:
            raise ValueError(f"Lag must be in [0, {self.max_lag}], got {lag}")
        self.adjacency[source, target, lag] = 1
        self.weights[source, target, lag] = weight

    def remove_edge(self, source: int, target: int, lag: int) -> None:
        """Remove edge from source at lag to target."""
        self.adjacency[source, target, lag] = 0
        self.weights[source, target, lag] = 0.0

    def has_edge(self, source: int, target: int, lag: int) -> bool:
        """Check if edge exists."""
        return bool(self.adjacency[source, target, lag])

    def get_parents(self, target: int) -> List[Tuple[int, int]]:
        """
        Get all parents of target variable.

        Returns
        -------
        List[Tuple[int, int]]
            List of (source_var, lag) tuples for all parents
        """
        parents = []
        for source in range(self.n_vars):
            for lag in range(self.max_lag + 1):
                if self.adjacency[source, target, lag]:
                    parents.append((source, lag))
        return parents

    def get_children(self, source: int, lag: int = None) -> List[Tuple[int, int]]:
        """
        Get all children of source variable.

        Parameters
        ----------
        source : int
            Source variable index
        lag : int, optional
            If specified, only get children at this lag

        Returns
        -------
        List[Tuple[int, int]]
            List of (target_var, lag) tuples
        """
        children = []
        lags = [lag] if lag is not None else range(self.max_lag + 1)
        for target in range(self.n_vars):
            for l in lags:
                if self.adjacency[source, target, l]:
                    children.append((target, l))
        return children

    def n_edges(self) -> int:
        """Total number of edges in the graph."""
        return int(np.sum(self.adjacency))

    def to_links(self, alpha: float = 0.05) -> List[TimeSeriesLink]:
        """Convert adjacency matrix to list of TimeSeriesLink objects."""
        links = []
        for source in range(self.n_vars):
            for target in range(self.n_vars):
                for lag in range(self.max_lag + 1):
                    if self.adjacency[source, target, lag]:
                        links.append(
                            TimeSeriesLink(
                                source_var=source,
                                target_var=target,
                                lag=lag,
                                strength=self.weights[source, target, lag],
                                p_value=0.0,  # Unknown from adjacency alone
                            )
                        )
        return links

    def __repr__(self) -> str:
        return f"LaggedDAG(n_vars={self.n_vars}, max_lag={self.max_lag}, n_edges={self.n_edges()})"


@dataclass
class PCMCIResult:
    """
    Result from PCMCI algorithm.

    Contains the discovered time-lagged causal graph, test statistics,
    and parent sets for each variable.

    Attributes
    ----------
    links : List[TimeSeriesLink]
        List of significant causal links discovered
    p_matrix : np.ndarray
        Shape (n_vars, n_vars, max_lag + 1) p-values from MCI tests.
        p_matrix[i, j, τ] is p-value for X_i_{t-τ} → X_j_t
    val_matrix : np.ndarray
        Shape (n_vars, n_vars, max_lag + 1) test statistic values
    graph : np.ndarray
        Shape (n_vars, n_vars, max_lag + 1) binary adjacency.
        graph[i, j, τ] = 1 if X_i_{t-τ} → X_j_t is significant
    parents : Dict[int, List[Tuple[int, int]]]
        Mapping from target variable to list of (parent_var, lag) tuples
    n_vars : int
        Number of variables
    n_obs : int
        Number of time observations
    max_lag : int
        Maximum lag tested
    alpha : float
        Significance level used
    ci_test : str
        Conditional independence test used ("parcorr", "cmi", etc.)

    Examples
    --------
    >>> result = pcmci(data, max_lag=3, alpha=0.05)
    >>> print(f"Found {len(result.links)} significant links")
    >>> for var, parents in result.parents.items():
    ...     print(f"X{var} has parents: {parents}")
    """

    links: List[TimeSeriesLink]
    p_matrix: np.ndarray
    val_matrix: np.ndarray
    graph: np.ndarray
    parents: Dict[int, List[Tuple[int, int]]]
    n_vars: int
    n_obs: int
    max_lag: int
    alpha: float = 0.05
    ci_test: str = "parcorr"

    def get_lagged_dag(self) -> LaggedDAG:
        """Convert result to LaggedDAG object."""
        dag = LaggedDAG(n_vars=self.n_vars, max_lag=self.max_lag)
        dag.adjacency = self.graph.astype(np.int8)
        dag.weights = self.val_matrix.copy()
        return dag

    def get_significant_links(self, alpha: Optional[float] = None) -> List[TimeSeriesLink]:
        """Get links significant at given alpha (default: self.alpha)."""
        threshold = alpha if alpha is not None else self.alpha
        return [link for link in self.links if link.p_value < threshold]

    def get_parents_of(self, target: int) -> List[Tuple[int, int]]:
        """Get parents of specific target variable."""
        return self.parents.get(target, [])

    def summary(self) -> str:
        """Generate summary string of discovered causal structure."""
        lines = [
            f"PCMCI Result Summary",
            f"=" * 40,
            f"Variables: {self.n_vars}",
            f"Observations: {self.n_obs}",
            f"Max lag: {self.max_lag}",
            f"Alpha: {self.alpha}",
            f"CI test: {self.ci_test}",
            f"",
            f"Discovered {len(self.links)} significant links:",
        ]

        for link in self.links:
            lines.append(f"  {link}")

        lines.append("")
        lines.append("Parent sets:")
        for var in range(self.n_vars):
            parents = self.get_parents_of(var)
            if parents:
                parent_str = ", ".join(f"X{p}(t-{l})" for p, l in parents)
                lines.append(f"  X{var}: [{parent_str}]")
            else:
                lines.append(f"  X{var}: []")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PCMCIResult(n_vars={self.n_vars}, n_obs={self.n_obs}, "
            f"max_lag={self.max_lag}, n_links={len(self.links)}, alpha={self.alpha})"
        )


@dataclass
class ConditionSelectionResult:
    """
    Result from PC-stable condition selection phase.

    The first phase of PCMCI that identifies potential parents for each variable.

    Attributes
    ----------
    parents : Dict[int, List[Tuple[int, int]]]
        Mapping from target variable to candidate parents (var, lag)
    separating_sets : Dict[Tuple[int, int, int], Set[Tuple[int, int]]]
        For each removed link (source, target, lag), the separating set
        that made them conditionally independent
    n_vars : int
        Number of variables
    max_lag : int
        Maximum lag considered
    """

    parents: Dict[int, List[Tuple[int, int]]]
    separating_sets: Dict[Tuple[int, int, int], Set[Tuple[int, int]]]
    n_vars: int
    max_lag: int

    def get_candidates(self, target: int) -> List[Tuple[int, int]]:
        """Get candidate parents for target variable."""
        return self.parents.get(target, [])

    def __repr__(self) -> str:
        total_parents = sum(len(p) for p in self.parents.values())
        return (
            f"ConditionSelectionResult(n_vars={self.n_vars}, "
            f"max_lag={self.max_lag}, total_candidates={total_parents})"
        )


@dataclass
class CITestResult:
    """
    Result from a conditional independence test.

    Attributes
    ----------
    statistic : float
        Test statistic value
    p_value : float
        P-value
    is_independent : bool
        True if null hypothesis (independence) not rejected
    dof : int
        Degrees of freedom (if applicable)
    """

    statistic: float
    p_value: float
    is_independent: bool
    dof: int = 0

    def __repr__(self) -> str:
        ind = "⊥" if self.is_independent else "⊥̸"
        return f"CITestResult({ind}, stat={self.statistic:.4f}, p={self.p_value:.4f})"
