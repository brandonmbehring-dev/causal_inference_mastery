"""
PCMCI Algorithm for Time Series Causal Discovery.

Session 136: Implementation of PCMCI (Runge et al., 2019).

PCMCI discovers causal relationships in multivariate time series by:
1. PC-stable condition selection: Find candidate parents for each variable
2. MCI testing: Apply momentary conditional independence tests

The key insight is that standard PC fails on time series due to autocorrelation;
PCMCI conditions on lagged parents to remove spurious correlations.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations

from .pcmci_types import (
    PCMCIResult,
    TimeSeriesLink,
    LinkType,
    ConditionSelectionResult,
    CITestResult,
    LaggedDAG,
)
from .ci_tests_timeseries import run_ci_test, get_ci_test


def pcmci(
    data: np.ndarray,
    max_lag: int = 3,
    alpha: float = 0.05,
    ci_test: str = "parcorr",
    pc_alpha: Optional[float] = None,
    min_lag: int = 1,
    max_cond_size: Optional[int] = None,
    verbosity: int = 0,
) -> PCMCIResult:
    """
    PCMCI algorithm for time series causal discovery.

    Discovers causal relationships X_{t-τ} → Y_t in multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data. Rows are time points,
        columns are variables.
    max_lag : int
        Maximum time lag to consider
    alpha : float
        Significance level for MCI phase
    ci_test : str
        Conditional independence test: "parcorr" (Gaussian) or "cmi" (non-Gaussian)
    pc_alpha : float, optional
        Significance level for PC-stable phase (default: 2 * alpha)
    min_lag : int
        Minimum lag to consider (default 1, set to 0 for contemporaneous)
    max_cond_size : int, optional
        Maximum conditioning set size (default: no limit)
    verbosity : int
        Verbosity level (0=silent, 1=summary, 2=detailed)

    Returns
    -------
    PCMCIResult
        Discovered causal structure with links, p-values, and parent sets

    Notes
    -----
    The algorithm proceeds in two phases:

    **Phase 1: PC-stable condition selection**
    For each variable Y, identify candidate parents by iteratively testing
    conditional independence and removing non-significant links.

    **Phase 2: MCI (Momentary Conditional Independence)**
    For each candidate link (X, τ) → Y, test:
        X_{t-τ} ⊥ Y_t | Parents(X) ∪ Parents(Y) \\ {X_{t-τ}}

    This conditioning removes spurious correlations due to autocorrelation.

    References
    ----------
    Runge et al. (2019). Detecting and quantifying causal associations in
    large nonlinear time series datasets. Science Advances.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate X → Y with lag 1
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + np.random.randn() * 0.5
    >>> data = np.column_stack([x, y])
    >>> result = pcmci(data, max_lag=2, alpha=0.05)
    >>> print(f"Found {len(result.links)} causal links")
    """
    n_obs, n_vars = data.shape

    # Input validation
    if n_obs <= max_lag + 1:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for max_lag={max_lag}. "
            f"Need at least {max_lag + 2}."
        )

    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    if pc_alpha is None:
        pc_alpha = min(2 * alpha, 0.2)  # More liberal for condition selection

    if verbosity > 0:
        print(f"PCMCI: n_obs={n_obs}, n_vars={n_vars}, max_lag={max_lag}")
        print(f"       alpha={alpha}, pc_alpha={pc_alpha}, ci_test={ci_test}")

    # Phase 1: PC-stable condition selection
    if verbosity > 0:
        print("\nPhase 1: PC-stable condition selection...")

    condition_result = pc_stable_condition_selection(
        data=data,
        max_lag=max_lag,
        alpha=pc_alpha,
        ci_test=ci_test,
        min_lag=min_lag,
        max_cond_size=max_cond_size,
        verbosity=verbosity,
    )

    if verbosity > 0:
        total_candidates = sum(len(p) for p in condition_result.parents.values())
        print(f"       Found {total_candidates} candidate links")

    # Phase 2: MCI testing
    if verbosity > 0:
        print("\nPhase 2: MCI testing...")

    mci_result = mci_test_all(
        data=data,
        parents=condition_result.parents,
        max_lag=max_lag,
        alpha=alpha,
        ci_test=ci_test,
        min_lag=min_lag,
        verbosity=verbosity,
    )

    if verbosity > 0:
        n_significant = np.sum(mci_result["graph"])
        print(f"       Found {n_significant} significant links")

    # Build result
    links = _build_links(
        mci_result["p_matrix"],
        mci_result["val_matrix"],
        mci_result["graph"],
        n_vars,
        max_lag,
        min_lag,
    )

    # Extract final parent sets from significant links
    final_parents: Dict[int, List[Tuple[int, int]]] = {j: [] for j in range(n_vars)}
    for link in links:
        final_parents[link.target_var].append((link.source_var, link.lag))

    result = PCMCIResult(
        links=links,
        p_matrix=mci_result["p_matrix"],
        val_matrix=mci_result["val_matrix"],
        graph=mci_result["graph"],
        parents=final_parents,
        n_vars=n_vars,
        n_obs=n_obs,
        max_lag=max_lag,
        alpha=alpha,
        ci_test=ci_test,
    )

    if verbosity > 0:
        print("\n" + result.summary())

    return result


def pc_stable_condition_selection(
    data: np.ndarray,
    max_lag: int,
    alpha: float,
    ci_test: str = "parcorr",
    min_lag: int = 1,
    max_cond_size: Optional[int] = None,
    verbosity: int = 0,
) -> ConditionSelectionResult:
    """
    PC-stable algorithm for condition selection.

    For each target variable, find candidate parents by iteratively
    testing conditional independence.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    max_lag : int
        Maximum lag to consider
    alpha : float
        Significance level for independence tests
    ci_test : str
        CI test to use
    min_lag : int
        Minimum lag (1 excludes contemporaneous effects)
    max_cond_size : int, optional
        Maximum conditioning set size
    verbosity : int
        Verbosity level

    Returns
    -------
    ConditionSelectionResult
        Candidate parents and separating sets for each variable
    """
    n_obs, n_vars = data.shape

    # Initialize: all possible (var, lag) pairs are candidates for each target
    parents: Dict[int, List[Tuple[int, int]]] = {}
    separating_sets: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = {}

    for target in range(n_vars):
        # Initialize candidates: all (var, lag) pairs
        candidates: List[Tuple[int, int]] = []
        for source in range(n_vars):
            for lag in range(min_lag, max_lag + 1):
                # Skip self-contemporaneous (if min_lag=0)
                if source == target and lag == 0:
                    continue
                candidates.append((source, lag))

        parents[target] = candidates

    # Iteratively test and remove independent links
    cond_size = 0
    max_possible_cond = max_lag * n_vars if max_cond_size is None else max_cond_size

    while cond_size <= max_possible_cond:
        if verbosity > 1:
            print(f"  Condition set size: {cond_size}")

        any_removed = False

        for target in range(n_vars):
            current_parents = parents[target].copy()

            if len(current_parents) <= cond_size:
                continue

            for source, lag in current_parents:
                if (source, lag) not in parents[target]:
                    continue  # Already removed

                # Get conditioning candidates (other parents of target)
                other_parents = [p for p in parents[target] if p != (source, lag)]

                if len(other_parents) < cond_size:
                    continue

                # Test all conditioning sets of size cond_size
                removed = False
                for cond_set in combinations(other_parents, cond_size):
                    result = run_ci_test(
                        data=data,
                        source=source,
                        target=target,
                        source_lag=lag,
                        conditioning_set=list(cond_set),
                        ci_test=ci_test,
                        alpha=alpha,
                    )

                    if result.is_independent:
                        # Remove link and store separator
                        parents[target].remove((source, lag))
                        separating_sets[(source, target, lag)] = set(cond_set)
                        removed = True
                        any_removed = True

                        if verbosity > 1:
                            cond_str = ", ".join(f"X{v}(t-{l})" for v, l in cond_set)
                            print(f"    Removed: X{source}(t-{lag}) → X{target} | {{{cond_str}}}")
                        break

        cond_size += 1

        if not any_removed and cond_size > 0:
            break  # No more links can be removed

    return ConditionSelectionResult(
        parents=parents,
        separating_sets=separating_sets,
        n_vars=n_vars,
        max_lag=max_lag,
    )


def mci_test_all(
    data: np.ndarray,
    parents: Dict[int, List[Tuple[int, int]]],
    max_lag: int,
    alpha: float,
    ci_test: str = "parcorr",
    min_lag: int = 1,
    verbosity: int = 0,
) -> Dict:
    """
    MCI (Momentary Conditional Independence) tests for all candidate links.

    For each candidate link X_{t-τ} → Y, test:
        X_{t-τ} ⊥ Y_t | Parents(X) ∪ Parents(Y) \\ {X_{t-τ}}

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    parents : Dict[int, List[Tuple[int, int]]]
        Candidate parents from PC-stable phase
    max_lag : int
        Maximum lag
    alpha : float
        Significance level
    ci_test : str
        CI test to use
    min_lag : int
        Minimum lag
    verbosity : int
        Verbosity level

    Returns
    -------
    Dict
        Contains p_matrix, val_matrix, and graph arrays
    """
    n_obs, n_vars = data.shape

    # Initialize output matrices
    p_matrix = np.ones((n_vars, n_vars, max_lag + 1))
    val_matrix = np.zeros((n_vars, n_vars, max_lag + 1))
    graph = np.zeros((n_vars, n_vars, max_lag + 1), dtype=np.int8)

    for target in range(n_vars):
        target_parents = parents.get(target, [])

        for source, lag in target_parents:
            # Build conditioning set: Parents(source) ∪ Parents(target) \ {(source, lag)}
            # For source, we need lagged parents
            source_parents = _get_lagged_parents(parents.get(source, []), lag)
            cond_set = list(set(source_parents + target_parents) - {(source, lag)})

            # Run MCI test
            result = run_ci_test(
                data=data,
                source=source,
                target=target,
                source_lag=lag,
                conditioning_set=cond_set,
                ci_test=ci_test,
                alpha=alpha,
            )

            p_matrix[source, target, lag] = result.p_value
            val_matrix[source, target, lag] = result.statistic

            if not result.is_independent:
                graph[source, target, lag] = 1

                if verbosity > 1:
                    print(f"  Significant: X{source}(t-{lag}) → X{target}, p={result.p_value:.4f}")

    return {
        "p_matrix": p_matrix,
        "val_matrix": val_matrix,
        "graph": graph,
    }


def _get_lagged_parents(
    parents: List[Tuple[int, int]], additional_lag: int
) -> List[Tuple[int, int]]:
    """
    Get parents with additional lag offset.

    For MCI conditioning, parents of X_{t-τ} are the same as parents of X_t
    but with τ added to their lags.
    """
    lagged = []
    for var, lag in parents:
        new_lag = lag + additional_lag
        lagged.append((var, new_lag))
    return lagged


def _build_links(
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    graph: np.ndarray,
    n_vars: int,
    max_lag: int,
    min_lag: int,
) -> List[TimeSeriesLink]:
    """Build list of TimeSeriesLink from result matrices."""
    links = []

    for source in range(n_vars):
        for target in range(n_vars):
            for lag in range(min_lag, max_lag + 1):
                if graph[source, target, lag]:
                    link = TimeSeriesLink(
                        source_var=source,
                        target_var=target,
                        lag=lag,
                        strength=val_matrix[source, target, lag],
                        p_value=p_matrix[source, target, lag],
                        link_type=LinkType.DIRECTED,
                    )
                    links.append(link)

    # Sort by p-value
    links.sort(key=lambda x: x.p_value)
    return links


def pcmci_plus(
    data: np.ndarray,
    max_lag: int = 3,
    alpha: float = 0.05,
    ci_test: str = "parcorr",
    verbosity: int = 0,
) -> PCMCIResult:
    """
    PCMCI+ algorithm that also handles contemporaneous causal effects.

    Extension of PCMCI that can discover causal relationships at lag 0
    (X_t → Y_t) in addition to lagged effects.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    max_lag : int
        Maximum lag
    alpha : float
        Significance level
    ci_test : str
        CI test to use
    verbosity : int
        Verbosity level

    Returns
    -------
    PCMCIResult
        Discovered causal structure including contemporaneous effects

    Notes
    -----
    PCMCI+ uses additional orientation rules to determine the direction
    of contemporaneous links where possible. Links that cannot be oriented
    are marked as undirected.
    """
    # Run PCMCI with min_lag=0 to include contemporaneous
    result = pcmci(
        data=data,
        max_lag=max_lag,
        alpha=alpha,
        ci_test=ci_test,
        min_lag=0,
        verbosity=verbosity,
    )

    # TODO: Add orientation rules for contemporaneous links
    # For now, contemporaneous links are treated as directed
    # based on the test result alone

    return result


def run_granger_style_pcmci(
    data: np.ndarray,
    max_lag: int = 3,
    alpha: float = 0.05,
) -> PCMCIResult:
    """
    Simplified PCMCI that mimics Granger causality testing.

    Skips the PC-stable phase and directly tests all lagged links.
    Useful for comparison with traditional Granger causality.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    max_lag : int
        Maximum lag
    alpha : float
        Significance level

    Returns
    -------
    PCMCIResult
        Discovered causal structure
    """
    n_obs, n_vars = data.shape

    # All possible lagged links are candidates
    parents: Dict[int, List[Tuple[int, int]]] = {}
    for target in range(n_vars):
        candidates = []
        for source in range(n_vars):
            for lag in range(1, max_lag + 1):
                candidates.append((source, lag))
        parents[target] = candidates

    # Run MCI tests directly
    mci_result = mci_test_all(
        data=data,
        parents=parents,
        max_lag=max_lag,
        alpha=alpha,
        ci_test="parcorr",
        min_lag=1,
    )

    links = _build_links(
        mci_result["p_matrix"],
        mci_result["val_matrix"],
        mci_result["graph"],
        n_vars,
        max_lag,
        min_lag=1,
    )

    final_parents: Dict[int, List[Tuple[int, int]]] = {j: [] for j in range(n_vars)}
    for link in links:
        final_parents[link.target_var].append((link.source_var, link.lag))

    return PCMCIResult(
        links=links,
        p_matrix=mci_result["p_matrix"],
        val_matrix=mci_result["val_matrix"],
        graph=mci_result["graph"],
        parents=final_parents,
        n_vars=n_vars,
        n_obs=n_obs,
        max_lag=max_lag,
        alpha=alpha,
        ci_test="parcorr",
    )
