"""
Conditional Independence Tests for Time Series.

Session 136: CI tests for PCMCI algorithm.

Implements partial correlation (Gaussian) and conditional mutual information
(non-Gaussian) tests adapted for time-lagged data.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List, Union
from .pcmci_types import CITestResult


def parcorr_test(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    z_indices: List[Tuple[int, int]],
    x_lag: int = 0,
    y_lag: int = 0,
    alpha: float = 0.05,
) -> CITestResult:
    """
    Partial correlation conditional independence test for time series.

    Tests X_{t-x_lag} ⊥ Y_{t-y_lag} | Z where Z is a set of lagged variables.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    x_idx : int
        Index of X variable
    y_idx : int
        Index of Y variable
    z_indices : List[Tuple[int, int]]
        List of (variable_index, lag) tuples for conditioning set Z
    x_lag : int
        Lag for X variable (default 0 = current time)
    y_lag : int
        Lag for Y variable (default 0 = current time)
    alpha : float
        Significance level

    Returns
    -------
    CITestResult
        Test result with statistic, p-value, and independence decision

    Notes
    -----
    The partial correlation is computed as:
        ρ(X, Y | Z) = Cov(X - E[X|Z], Y - E[Y|Z]) / sqrt(Var(X|Z) * Var(Y|Z))

    Under H0 (X ⊥ Y | Z), the test statistic:
        t = ρ * sqrt((n - |Z| - 2) / (1 - ρ²))

    follows a t-distribution with (n - |Z| - 2) degrees of freedom.

    Examples
    --------
    >>> data = np.random.randn(200, 3)
    >>> result = parcorr_test(data, x_idx=0, y_idx=1, z_indices=[(2, 1)],
    ...                       x_lag=1, y_lag=0)
    >>> print(f"Independent: {result.is_independent}")
    """
    n_obs, n_vars = data.shape

    # Determine effective sample size after accounting for lags
    all_lags = [x_lag, y_lag] + [lag for _, lag in z_indices]
    max_lag = max(all_lags) if all_lags else 0
    n_effective = n_obs - max_lag

    if n_effective <= len(z_indices) + 2:
        raise ValueError(
            f"Insufficient observations ({n_effective}) for conditioning set "
            f"of size {len(z_indices)}. Need at least {len(z_indices) + 3}."
        )

    # Extract time-aligned data
    # All variables are aligned to time t, with appropriate lags applied
    x_series = data[max_lag - x_lag : n_obs - x_lag if x_lag > 0 else None, x_idx]
    y_series = data[max_lag - y_lag : n_obs - y_lag if y_lag > 0 else None, y_idx]

    if len(z_indices) == 0:
        # Unconditional correlation
        rho = np.corrcoef(x_series, y_series)[0, 1]
        dof = n_effective - 2
    else:
        # Build conditioning matrix Z
        z_matrix = np.zeros((n_effective, len(z_indices)))
        for i, (var_idx, lag) in enumerate(z_indices):
            z_matrix[:, i] = data[
                max_lag - lag : n_obs - lag if lag > 0 else None, var_idx
            ]

        # Compute partial correlation via regression residuals
        rho = _partial_correlation(x_series, y_series, z_matrix)
        dof = n_effective - len(z_indices) - 2

    if dof <= 0:
        return CITestResult(
            statistic=0.0, p_value=1.0, is_independent=True, dof=0
        )

    # Compute t-statistic
    # Handle edge cases
    if np.abs(rho) > 1 - 1e-10:
        # Perfect correlation
        t_stat = np.sign(rho) * np.inf
        p_value = 0.0
    elif np.isnan(rho):
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = rho * np.sqrt(dof / (1 - rho ** 2 + 1e-10))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

    is_independent = p_value >= alpha

    return CITestResult(
        statistic=rho, p_value=p_value, is_independent=is_independent, dof=dof
    )


def _partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    """
    Compute partial correlation between x and y given z.

    Uses residualization: regress x and y on z, then correlate residuals.

    Parameters
    ----------
    x : np.ndarray
        Shape (n,) first variable
    y : np.ndarray
        Shape (n,) second variable
    z : np.ndarray
        Shape (n, k) conditioning variables

    Returns
    -------
    float
        Partial correlation coefficient
    """
    # Add intercept to Z
    n = len(x)
    z_with_const = np.column_stack([np.ones(n), z])

    try:
        # Compute residuals from regressing x and y on z
        # x_resid = x - Z @ (Z'Z)^{-1} Z'x
        q, r = np.linalg.qr(z_with_const)
        x_resid = x - q @ (q.T @ x)
        y_resid = y - q @ (q.T @ y)

        # Correlation of residuals
        x_centered = x_resid - np.mean(x_resid)
        y_centered = y_resid - np.mean(y_resid)

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    except np.linalg.LinAlgError:
        return 0.0


def cmi_knn_test(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    z_indices: List[Tuple[int, int]],
    x_lag: int = 0,
    y_lag: int = 0,
    alpha: float = 0.05,
    knn: int = 7,
) -> CITestResult:
    """
    Conditional mutual information test using k-nearest neighbors.

    Tests X_{t-x_lag} ⊥ Y_{t-y_lag} | Z using the Kraskov-Stögbauer-Grassberger
    (KSG) estimator for mutual information.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    x_idx : int
        Index of X variable
    y_idx : int
        Index of Y variable
    z_indices : List[Tuple[int, int]]
        List of (variable_index, lag) tuples for conditioning set Z
    x_lag : int
        Lag for X variable
    y_lag : int
        Lag for Y variable
    alpha : float
        Significance level
    knn : int
        Number of nearest neighbors (default 7)

    Returns
    -------
    CITestResult
        Test result with CMI estimate as statistic

    Notes
    -----
    CMI is computed as:
        I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z) - H(Z)
                    = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    We use the KSG estimator (Kraskov et al., 2004) with a permutation
    test for significance.

    This is more robust to non-Gaussian distributions than partial correlation.
    """
    n_obs, n_vars = data.shape

    # Determine effective sample size
    all_lags = [x_lag, y_lag] + [lag for _, lag in z_indices]
    max_lag = max(all_lags) if all_lags else 0
    n_effective = n_obs - max_lag

    if n_effective <= knn + 1:
        raise ValueError(
            f"Insufficient observations ({n_effective}) for knn={knn}. "
            f"Need at least {knn + 2}."
        )

    # Extract time-aligned data
    x_series = data[max_lag - x_lag : n_obs - x_lag if x_lag > 0 else None, x_idx]
    y_series = data[max_lag - y_lag : n_obs - y_lag if y_lag > 0 else None, y_idx]

    if len(z_indices) == 0:
        # Unconditional mutual information
        cmi_val = _mutual_info_ksg(x_series.reshape(-1, 1), y_series.reshape(-1, 1), knn)
    else:
        # Build conditioning matrix Z
        z_matrix = np.zeros((n_effective, len(z_indices)))
        for i, (var_idx, lag) in enumerate(z_indices):
            z_matrix[:, i] = data[
                max_lag - lag : n_obs - lag if lag > 0 else None, var_idx
            ]

        cmi_val = _conditional_mutual_info_ksg(
            x_series.reshape(-1, 1), y_series.reshape(-1, 1), z_matrix, knn
        )

    # Permutation test for significance
    p_value = _cmi_permutation_test(
        x_series, y_series,
        z_matrix if len(z_indices) > 0 else None,
        cmi_val, knn, n_permutations=100
    )

    is_independent = p_value >= alpha

    return CITestResult(
        statistic=cmi_val, p_value=p_value, is_independent=is_independent, dof=0
    )


def _mutual_info_ksg(
    x: np.ndarray, y: np.ndarray, k: int = 7
) -> float:
    """
    Estimate mutual information using KSG estimator.

    Parameters
    ----------
    x : np.ndarray
        Shape (n, dx) first variable
    y : np.ndarray
        Shape (n, dy) second variable
    k : int
        Number of nearest neighbors

    Returns
    -------
    float
        Mutual information estimate in nats
    """
    from scipy.spatial import cKDTree

    n = len(x)
    xy = np.hstack([x, y])

    # Build KD-trees
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find k-th nearest neighbor distance in joint space
    # Using k+1 because query includes the point itself
    dists_xy, _ = tree_xy.query(xy, k=k + 1)
    eps = dists_xy[:, -1]  # Distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    nx = np.array([tree_x.query_ball_point(x[i], eps[i] - 1e-10, return_length=True) - 1
                   for i in range(n)])
    ny = np.array([tree_y.query_ball_point(y[i], eps[i] - 1e-10, return_length=True) - 1
                   for i in range(n)])

    # Handle edge cases
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    # KSG estimator: I(X;Y) ≈ ψ(k) - <ψ(nx) + ψ(ny)> + ψ(n)
    from scipy.special import digamma
    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)

    return max(0.0, mi)


def _conditional_mutual_info_ksg(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 7
) -> float:
    """
    Estimate conditional mutual information I(X; Y | Z) using KSG.

    Uses the chain rule: I(X; Y | Z) = I(X; Y, Z) - I(X; Z)

    Parameters
    ----------
    x : np.ndarray
        Shape (n, dx) first variable
    y : np.ndarray
        Shape (n, dy) second variable
    z : np.ndarray
        Shape (n, dz) conditioning variable
    k : int
        Number of nearest neighbors

    Returns
    -------
    float
        Conditional mutual information estimate
    """
    # I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    yz = np.hstack([y, z])
    mi_xyz = _mutual_info_ksg(x, yz, k)
    mi_xz = _mutual_info_ksg(x, z, k)

    return max(0.0, mi_xyz - mi_xz)


def _cmi_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray],
    observed_cmi: float,
    knn: int,
    n_permutations: int = 100,
) -> float:
    """
    Permutation test for CMI significance.

    Permutes Y while keeping X and Z fixed to generate null distribution.
    """
    n = len(x)
    count_greater = 0

    for _ in range(n_permutations):
        # Permute y
        perm_idx = np.random.permutation(n)
        y_perm = y[perm_idx]

        if z is None:
            perm_cmi = _mutual_info_ksg(
                x.reshape(-1, 1), y_perm.reshape(-1, 1), knn
            )
        else:
            perm_cmi = _conditional_mutual_info_ksg(
                x.reshape(-1, 1), y_perm.reshape(-1, 1), z, knn
            )

        if perm_cmi >= observed_cmi:
            count_greater += 1

    return (count_greater + 1) / (n_permutations + 1)


def get_ci_test(name: str):
    """
    Get conditional independence test function by name.

    Parameters
    ----------
    name : str
        Test name: "parcorr" or "cmi"

    Returns
    -------
    Callable
        CI test function
    """
    tests = {
        "parcorr": parcorr_test,
        "cmi": cmi_knn_test,
    }

    if name not in tests:
        raise ValueError(f"Unknown CI test: {name}. Available: {list(tests.keys())}")

    return tests[name]


def run_ci_test(
    data: np.ndarray,
    source: int,
    target: int,
    source_lag: int,
    conditioning_set: List[Tuple[int, int]],
    ci_test: str = "parcorr",
    alpha: float = 0.05,
    **kwargs,
) -> CITestResult:
    """
    Run conditional independence test for time series.

    Convenience wrapper that handles the common case of testing
    X_{t-τ} ⊥ Y_t | Z.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    source : int
        Source variable index
    target : int
        Target variable index
    source_lag : int
        Lag of source variable
    conditioning_set : List[Tuple[int, int]]
        List of (var_idx, lag) for conditioning variables
    ci_test : str
        CI test to use: "parcorr" or "cmi"
    alpha : float
        Significance level
    **kwargs
        Additional arguments passed to CI test

    Returns
    -------
    CITestResult
        Test result
    """
    test_func = get_ci_test(ci_test)

    return test_func(
        data=data,
        x_idx=source,
        y_idx=target,
        z_indices=conditioning_set,
        x_lag=source_lag,
        y_lag=0,  # Target is always at time t
        alpha=alpha,
        **kwargs,
    )
