"""Conditional independence tests for causal discovery.

Session 133: Statistical tests for X ⊥ Y | Z (conditional independence).

Functions
---------
fisher_z_test : Fisher's Z-transform test for Gaussian data
partial_correlation : Compute partial correlation coefficient
partial_correlation_test : CI test based on partial correlation
g_squared_test : G² (likelihood ratio) test for categorical data
kernel_ci_test : Kernel-based CI test for nonlinear dependencies

Notes
-----
The choice of CI test affects PC algorithm performance:
- Fisher Z: Fast, assumes linear Gaussian relationships
- Partial correlation: Same assumptions, equivalent to Fisher Z
- G²: For discrete/categorical data
- Kernel: Non-parametric, handles nonlinear but slower
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.linalg import pinv


@dataclass
class CITestResult:
    """Result of a conditional independence test.

    Attributes
    ----------
    independent : bool
        True if X ⊥ Y | Z at significance level alpha.
    pvalue : float
        P-value of the test.
    statistic : float
        Test statistic value.
    alpha : float
        Significance level used.
    conditioning_set : Tuple[int, ...]
        Indices of conditioning variables Z.
    """

    independent: bool
    pvalue: float
    statistic: float
    alpha: float
    conditioning_set: Tuple[int, ...]

    def __repr__(self) -> str:
        status = "independent" if self.independent else "dependent"
        return f"CITestResult({status}, p={self.pvalue:.4f}, stat={self.statistic:.4f})"


def partial_correlation(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
) -> float:
    """Compute partial correlation between X and Y given Z.

    Uses the recursive formula or matrix inversion approach.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    x : int
        Index of first variable.
    y : int
        Index of second variable.
    z : List[int] or Tuple[int, ...] or None
        Indices of conditioning variables. If None or empty, returns
        simple correlation.

    Returns
    -------
    float
        Partial correlation coefficient rho(X, Y | Z).

    Example
    -------
    >>> data = np.random.randn(1000, 4)
    >>> # Partial correlation of X0 and X1 given X2, X3
    >>> rho = partial_correlation(data, 0, 1, [2, 3])
    """
    if z is None or len(z) == 0:
        # Simple correlation
        return np.corrcoef(data[:, x], data[:, y])[0, 1]

    # Use precision matrix approach (more numerically stable)
    z_list = list(z)
    indices = [x, y] + z_list
    sub_data = data[:, indices]

    # Compute correlation matrix of subset
    C = np.corrcoef(sub_data.T)

    # Handle numerical issues
    if np.any(np.isnan(C)) or np.any(np.isinf(C)):
        return 0.0

    # Partial correlation via precision matrix
    try:
        P = pinv(C)
        # rho(X,Y|Z) = -P[0,1] / sqrt(P[0,0] * P[1,1])
        denom = np.sqrt(np.abs(P[0, 0] * P[1, 1]))
        if denom < 1e-10:
            return 0.0
        rho = -P[0, 1] / denom
        # Clamp to valid range
        return np.clip(rho, -1.0, 1.0)
    except np.linalg.LinAlgError:
        return 0.0


def fisher_z_test(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
    alpha: float = 0.01,
) -> CITestResult:
    """Fisher's Z-transform test for conditional independence.

    Tests H0: X ⊥ Y | Z (conditional independence) assuming linear
    Gaussian relationships.

    The test statistic is:
        Z = 0.5 * sqrt(n - |Z| - 3) * log((1+r)/(1-r))

    where r is the partial correlation and n is sample size.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    x : int
        Index of first variable.
    y : int
        Index of second variable.
    z : List[int] or Tuple[int, ...] or None
        Indices of conditioning variables.
    alpha : float
        Significance level (default 0.01).

    Returns
    -------
    CITestResult
        Test result with independence decision and p-value.

    Example
    -------
    >>> # Test if X0 ⊥ X1 | X2
    >>> result = fisher_z_test(data, 0, 1, [2], alpha=0.05)
    >>> if result.independent:
    ...     print("X0 and X1 are conditionally independent given X2")
    """
    n = data.shape[0]
    z_tuple = tuple(z) if z is not None else ()
    k = len(z_tuple)

    # Compute partial correlation
    rho = partial_correlation(data, x, y, z_tuple)

    # Degrees of freedom
    dof = n - k - 3

    if dof < 1:
        # Not enough samples for test
        return CITestResult(
            independent=True,
            pvalue=1.0,
            statistic=0.0,
            alpha=alpha,
            conditioning_set=z_tuple,
        )

    # Fisher's Z-transform
    # Avoid numerical issues at boundaries
    rho_clipped = np.clip(rho, -0.9999, 0.9999)
    z_stat = 0.5 * np.sqrt(dof) * np.log((1 + rho_clipped) / (1 - rho_clipped))

    # Two-tailed p-value (Z ~ N(0,1) under H0)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return CITestResult(
        independent=pvalue > alpha,
        pvalue=pvalue,
        statistic=z_stat,
        alpha=alpha,
        conditioning_set=z_tuple,
    )


def partial_correlation_test(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
    alpha: float = 0.01,
) -> CITestResult:
    """CI test based on partial correlation with t-test.

    Equivalent to Fisher Z for large samples, but uses t-distribution
    which may be more accurate for smaller samples.

    Test statistic: t = r * sqrt((n-k-2) / (1-r²))

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    x : int
        Index of first variable.
    y : int
        Index of second variable.
    z : List[int] or Tuple[int, ...] or None
        Indices of conditioning variables.
    alpha : float
        Significance level.

    Returns
    -------
    CITestResult
        Test result.
    """
    n = data.shape[0]
    z_tuple = tuple(z) if z is not None else ()
    k = len(z_tuple)

    # Compute partial correlation
    rho = partial_correlation(data, x, y, z_tuple)

    # Degrees of freedom for t-test
    dof = n - k - 2

    if dof < 1:
        return CITestResult(
            independent=True,
            pvalue=1.0,
            statistic=0.0,
            alpha=alpha,
            conditioning_set=z_tuple,
        )

    # t-statistic
    rho_sq = rho**2
    if rho_sq >= 1.0:
        rho_sq = 0.9999

    t_stat = rho * np.sqrt(dof / (1 - rho_sq))

    # Two-tailed p-value
    pvalue = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

    return CITestResult(
        independent=pvalue > alpha,
        pvalue=pvalue,
        statistic=t_stat,
        alpha=alpha,
        conditioning_set=z_tuple,
    )


def g_squared_test(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
    alpha: float = 0.01,
    n_bins: int = 3,
) -> CITestResult:
    """G² (likelihood ratio) test for categorical data.

    Discretizes continuous data into bins if needed, then performs
    G² test for conditional independence.

    G² = 2 * sum(O * log(O/E))

    where O is observed frequency and E is expected under independence.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
        Can be continuous (will be discretized) or integer-coded categories.
    x : int
        Index of first variable.
    y : int
        Index of second variable.
    z : List[int] or Tuple[int, ...] or None
        Indices of conditioning variables.
    alpha : float
        Significance level.
    n_bins : int
        Number of bins for discretizing continuous variables.

    Returns
    -------
    CITestResult
        Test result.

    Notes
    -----
    G² test requires sufficient counts in each cell. With continuous
    data and many conditioning variables, cells become sparse and
    the test loses power. Consider Fisher Z for continuous data.
    """
    n = data.shape[0]
    z_tuple = tuple(z) if z is not None else ()

    # Discretize data if continuous
    def discretize(col: np.ndarray, bins: int) -> np.ndarray:
        if np.issubdtype(col.dtype, np.integer):
            return col
        return np.digitize(col, np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]))

    x_disc = discretize(data[:, x], n_bins)
    y_disc = discretize(data[:, y], n_bins)

    if len(z_tuple) == 0:
        # Unconditional G² test (2D contingency table)
        contingency = _contingency_table_2d(x_disc, y_disc)
        g2, dof = _g_squared_2d(contingency)
    else:
        # Conditional G² test
        z_disc = np.column_stack([discretize(data[:, zi], n_bins) for zi in z_tuple])
        g2, dof = _g_squared_conditional(x_disc, y_disc, z_disc)

    if dof < 1:
        return CITestResult(
            independent=True,
            pvalue=1.0,
            statistic=0.0,
            alpha=alpha,
            conditioning_set=z_tuple,
        )

    # Chi-squared p-value
    pvalue = 1 - stats.chi2.cdf(g2, dof)

    return CITestResult(
        independent=pvalue > alpha,
        pvalue=pvalue,
        statistic=g2,
        alpha=alpha,
        conditioning_set=z_tuple,
    )


def _contingency_table_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build 2D contingency table."""
    x_cats = np.unique(x)
    y_cats = np.unique(y)

    table = np.zeros((len(x_cats), len(y_cats)))
    for i, xi in enumerate(x_cats):
        for j, yj in enumerate(y_cats):
            table[i, j] = np.sum((x == xi) & (y == yj))

    return table


def _g_squared_2d(contingency: np.ndarray) -> Tuple[float, int]:
    """Compute G² statistic for 2D table."""
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()

    if total == 0:
        return 0.0, 0

    # Expected frequencies under independence
    expected = row_sums * col_sums / total

    # G² = 2 * sum(O * log(O/E))
    # Handle zeros
    mask = (contingency > 0) & (expected > 0)
    g2 = 2 * np.sum(contingency[mask] * np.log(contingency[mask] / expected[mask]))

    # Degrees of freedom
    n_rows, n_cols = contingency.shape
    dof = (n_rows - 1) * (n_cols - 1)

    return g2, dof


def _g_squared_conditional(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[float, int]:
    """Compute conditional G² statistic.

    Sum G² over strata defined by Z values.
    """
    # Get unique Z configurations
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Use tuple representation for Z values
    z_tuples = [tuple(row) for row in z]
    z_unique = list(set(z_tuples))

    g2_total = 0.0
    dof_total = 0

    for z_val in z_unique:
        # Subset to this Z stratum
        mask = np.array([zt == z_val for zt in z_tuples])
        if np.sum(mask) < 5:  # Skip sparse strata
            continue

        x_sub = x[mask]
        y_sub = y[mask]

        contingency = _contingency_table_2d(x_sub, y_sub)
        g2, dof = _g_squared_2d(contingency)

        g2_total += g2
        dof_total += dof

    return g2_total, dof_total


def kernel_ci_test(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
    alpha: float = 0.01,
    kernel: str = "rbf",
    n_bootstrap: int = 100,
) -> CITestResult:
    """Kernel-based conditional independence test.

    Uses HSIC (Hilbert-Schmidt Independence Criterion) for testing
    independence. Can detect nonlinear dependencies.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    x : int
        Index of first variable.
    y : int
        Index of second variable.
    z : List[int] or Tuple[int, ...] or None
        Indices of conditioning variables.
    alpha : float
        Significance level.
    kernel : str
        Kernel type: "rbf" (Gaussian) or "linear".
    n_bootstrap : int
        Number of bootstrap samples for p-value.

    Returns
    -------
    CITestResult
        Test result.

    Notes
    -----
    This is a simplified implementation. For production use, consider
    the KCIT implementation from the causal-learn library which uses
    more sophisticated conditional HSIC.
    """
    n = data.shape[0]
    z_tuple = tuple(z) if z is not None else ()

    x_data = data[:, x].reshape(-1, 1)
    y_data = data[:, y].reshape(-1, 1)

    if len(z_tuple) == 0:
        # Unconditional test: HSIC(X, Y)
        stat = _hsic(x_data, y_data, kernel)
        # Bootstrap p-value
        null_stats = []
        for _ in range(n_bootstrap):
            perm = np.random.permutation(n)
            null_stats.append(_hsic(x_data, y_data[perm], kernel))
        pvalue = np.mean(np.array(null_stats) >= stat)
    else:
        # Conditional test: residual-based approach
        # Regress X, Y on Z and test independence of residuals
        z_data = data[:, list(z_tuple)]
        x_resid = _residualize(x_data, z_data)
        y_resid = _residualize(y_data, z_data)

        stat = _hsic(x_resid, y_resid, kernel)
        # Bootstrap p-value
        null_stats = []
        for _ in range(n_bootstrap):
            perm = np.random.permutation(n)
            null_stats.append(_hsic(x_resid, y_resid[perm], kernel))
        pvalue = np.mean(np.array(null_stats) >= stat)

    return CITestResult(
        independent=pvalue > alpha,
        pvalue=float(pvalue),
        statistic=float(stat),
        alpha=alpha,
        conditioning_set=z_tuple,
    )


def _hsic(x: np.ndarray, y: np.ndarray, kernel: str = "rbf") -> float:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion).

    HSIC measures dependence between X and Y in kernel space.
    HSIC = 0 iff X ⊥ Y (for characteristic kernels).
    """
    n = x.shape[0]
    if n < 4:
        return 0.0

    # Compute kernel matrices
    Kx = _kernel_matrix(x, kernel)
    Ky = _kernel_matrix(y, kernel)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # HSIC = (1/n²) * tr(Kx H Ky H)
    # Simplified: trace(H @ Kx @ H @ Ky) / n²
    HKx = H @ Kx
    HKy = H @ Ky
    hsic = np.trace(HKx @ HKy) / (n * n)

    return hsic


def _kernel_matrix(x: np.ndarray, kernel: str) -> np.ndarray:
    """Compute kernel matrix."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n = x.shape[0]

    if kernel == "linear":
        return x @ x.T
    elif kernel == "rbf":
        # Median heuristic for bandwidth
        dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
        median_dist = np.median(dists[dists > 0])
        if median_dist < 1e-10:
            median_dist = 1.0
        sigma = np.sqrt(median_dist / 2)
        return np.exp(-dists / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute residuals from regressing y on x."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Add intercept
    x_with_intercept = np.column_stack([np.ones(x.shape[0]), x])

    # OLS: y = Xb + e
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_with_intercept, y, rcond=None)
        residuals = y - x_with_intercept @ coeffs
    except np.linalg.LinAlgError:
        residuals = y - y.mean()

    return residuals


# =============================================================================
# Convenience Functions
# =============================================================================


def ci_test(
    data: np.ndarray,
    x: int,
    y: int,
    z: Union[List[int], Tuple[int, ...], None] = None,
    alpha: float = 0.01,
    method: str = "fisher_z",
    **kwargs,
) -> CITestResult:
    """Unified interface for conditional independence tests.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    x, y : int
        Variable indices to test.
    z : indices or None
        Conditioning set.
    alpha : float
        Significance level.
    method : str
        Test method: "fisher_z", "partial_correlation", "g_squared", "kernel".
    **kwargs
        Additional arguments passed to specific test.

    Returns
    -------
    CITestResult
        Test result.

    Example
    -------
    >>> result = ci_test(data, 0, 1, [2, 3], method="fisher_z")
    >>> print(f"Independent: {result.independent}, p={result.pvalue:.4f}")
    """
    methods = {
        "fisher_z": fisher_z_test,
        "partial_correlation": partial_correlation_test,
        "g_squared": g_squared_test,
        "kernel": kernel_ci_test,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    return methods[method](data, x, y, z, alpha, **kwargs)
