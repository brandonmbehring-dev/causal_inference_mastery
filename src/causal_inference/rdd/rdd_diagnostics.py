"""
RDD Diagnostic Tools.

This module provides diagnostic tools for validating RDD designs:
- McCrary density test for manipulation detection
- Covariate balance tests (falsification)
- Bandwidth sensitivity analysis
- Polynomial order sensitivity
- Donut-hole RDD (robustness to exclusion near cutoff)

References
----------
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. Journal of Econometrics, 142(2), 698-714.
- Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the
  regression discontinuity estimator. Review of Economic Studies, 79(3), 933-959.
"""

from typing import Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .sharp_rdd import SharpRDD


def _estimate_density_at_cutoff(
    bin_centers: np.ndarray,
    log_density: np.ndarray,
    cutoff: float,
    bandwidth: float,
) -> float:
    """
    Estimate density at cutoff using weighted quadratic regression on log densities.

    Fits quadratic polynomial to log density values weighted by distance from cutoff,
    then extrapolates to cutoff and exponentiates to get density estimate.

    Parameters
    ----------
    bin_centers : np.ndarray
        Bin center points
    log_density : np.ndarray
        Log of bin densities
    cutoff : float
        RDD cutoff value
    bandwidth : float
        Bandwidth for triangular kernel weights

    Returns
    -------
    float
        Estimated density at cutoff

    Notes
    -----
    Uses triangular kernel: w = max(1 - |x-c|/h, 0)
    Fallback to mean density if polynomial fit fails
    """
    # Compute triangular kernel weights
    dist = np.abs(bin_centers - cutoff)
    weights = np.maximum(1 - dist / bandwidth, 0)
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    # Fit weighted quadratic polynomial
    try:
        poly = np.polyfit(bin_centers - cutoff, log_density, deg=2, w=weights)
        density_at_cutoff = np.exp(np.polyval(poly, 0.0))
    except (np.linalg.LinAlgError, RuntimeWarning):
        # Fallback to mean if fit fails
        density_at_cutoff = np.exp(np.mean(log_density))

    return float(density_at_cutoff)


def _local_polynomial_regression(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    bandwidth: float,
    order: int,
    kernel: str = "triangular",
) -> tuple[float, float]:
    """
    Fit local polynomial regression on each side of cutoff.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    bandwidth : float
        Bandwidth for kernel weighting
    order : int
        Polynomial order (0=constant, 1=linear, 2=quadratic, etc.)
    kernel : str, default='triangular'
        Kernel function ('triangular' or 'rectangular')

    Returns
    -------
    estimate : float
        Treatment effect at cutoff (jump in fitted polynomial)
    se : float
        Robust standard error of the estimate
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    def kernel_weight(u: np.ndarray, kernel: str) -> np.ndarray:
        """Compute kernel weights."""
        if kernel == "triangular":
            return np.maximum(1 - np.abs(u), 0)
        else:  # rectangular
            return (np.abs(u) <= 1).astype(float)

    def fit_side(Y_side: np.ndarray, X_side: np.ndarray, bw: float) -> tuple[float, float]:
        """Fit polynomial on one side, return intercept and its variance."""
        if len(X_side) == 0:
            return np.nan, np.nan

        X_centered = X_side - cutoff
        u = X_centered / bw
        weights = kernel_weight(u, kernel)

        # Only use observations with positive weight
        pos_weight = weights > 0
        if np.sum(pos_weight) < order + 1:
            return np.nan, np.nan

        X_c = X_centered[pos_weight]
        Y_s = Y_side[pos_weight]
        W = weights[pos_weight]

        # Build design matrix for polynomial of given order
        # [1, X, X^2, ..., X^order]
        design = np.column_stack([X_c**p for p in range(order + 1)])

        # Weighted least squares
        W_diag = np.diag(W)
        XtWX = design.T @ W_diag @ design
        XtWY = design.T @ W_diag @ Y_s

        try:
            coefs = np.linalg.solve(XtWX, XtWY)
        except np.linalg.LinAlgError:
            return np.nan, np.nan

        # Intercept is the value at cutoff (X_centered = 0)
        alpha = coefs[0]

        # Robust variance (sandwich estimator)
        residuals = Y_s - design @ coefs
        meat = design.T @ W_diag @ np.diag(residuals**2) @ W_diag @ design
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)

        var_matrix = XtWX_inv @ meat @ XtWX_inv
        variance = var_matrix[0, 0]

        return alpha, variance

    # Fit left side (X < cutoff)
    left_mask = X < cutoff
    alpha_left, var_left = fit_side(Y[left_mask], X[left_mask], bandwidth)

    # Fit right side (X >= cutoff)
    right_mask = X >= cutoff
    alpha_right, var_right = fit_side(Y[right_mask], X[right_mask], bandwidth)

    # Treatment effect: jump at cutoff
    estimate = alpha_right - alpha_left
    se = np.sqrt(var_left + var_right)

    return estimate, se


def mccrary_density_test(
    X: np.ndarray,
    cutoff: float,
    bandwidth: float | None = None,
    n_bins: int = 20,
) -> Tuple[float, float, str]:
    """
    McCrary (2008) density test for manipulation.

    Tests null hypothesis: f(X|X=c⁺) = f(X|X=c⁻) (no density discontinuity)
    Alternative: Density jump at cutoff (manipulation detected)

    If units can precisely manipulate the running variable to cross the cutoff,
    this creates bunching and a discontinuity in the density. McCrary test
    detects this by estimating density on each side and testing for a jump.

    Parameters
    ----------
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    bandwidth : float, optional
        Bandwidth for density estimation. If None, uses Silverman's rule:
        h = 0.9 * min(sd, IQR/1.34) * n^(-1/5)
    n_bins : int, default=20
        Number of bins on each side of cutoff for density estimation

    Returns
    -------
    theta : float
        Log difference in densities: θ = log(f_right / f_left)
        θ > 0: More density on right (bunching above cutoff)
        θ < 0: More density on left (bunching below cutoff)
        θ ≈ 0: No manipulation
    p_value : float
        P-value from normal test (H0: θ = 0)
    interpretation : str
        "No evidence of manipulation (p=X.XX)" or
        "Potential manipulation detected (p=X.XX, p<0.05)"

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.uniform(-5, 5, 1000)  # Uniform density
    >>> theta, p_value, interp = mccrary_density_test(X, cutoff=0.0)
    >>> print(f"θ = {theta:.3f}, p = {p_value:.3f}")
    θ = 0.012, p = 0.856
    >>> print(interp)
    No evidence of manipulation (p=0.856)

    Notes
    -----
    McCrary test implementation:
    1. Bin observations on each side of cutoff
    2. Compute bin frequencies (counts)
    3. Fit local polynomial (quadratic) to log frequencies on each side
    4. Extrapolate to cutoff: f_left(c⁻), f_right(c⁺)
    5. Test statistic: θ = log(f_right / f_left)
    6. SE via binomial approximation
    7. P-value from Z-test: Z = θ / SE(θ)

    Interpretation:
    - p > 0.05: No evidence of manipulation (RDD likely valid)
    - p < 0.05: Potential manipulation (RDD may be invalid)
    - p < 0.01: Strong evidence of manipulation (RDD invalid)

    Limitations:
    - Test has low power with small samples (n < 500)
    - Discrete running variables can cause false positives
    - Natural bunching (e.g., age heaping) can trigger test
    """
    X = np.asarray(X).flatten()

    # Silverman's rule of thumb for bandwidth
    if bandwidth is None:
        sd = np.std(X)
        iqr = np.percentile(X, 75) - np.percentile(X, 25)
        bandwidth = 0.9 * min(sd, iqr / 1.34) * len(X) ** (-1 / 5)

    # Split into left and right
    X_left = X[X < cutoff]
    X_right = X[X >= cutoff]

    if len(X_left) < 10 or len(X_right) < 10:
        # Too few observations for reliable test
        return 0.0, 1.0, "Insufficient data for McCrary test"

    # Create bins on each side
    left_range = cutoff - X_left.min()
    right_range = X_right.max() - cutoff

    # Bin edges (must be increasing for np.histogram)
    left_bins = np.linspace(X_left.min(), cutoff, n_bins + 1)
    right_bins = np.linspace(cutoff, X_right.max(), n_bins + 1)

    # Compute frequencies
    left_counts, _ = np.histogram(X_left, bins=left_bins)
    right_counts, _ = np.histogram(X_right, bins=right_bins)

    # Bin centers
    left_centers = (left_bins[:-1] + left_bins[1:]) / 2
    right_centers = (right_bins[:-1] + right_bins[1:]) / 2

    # Normalize to get densities (frequency / bin width)
    left_widths = np.diff(left_bins)
    right_widths = np.diff(right_bins)
    left_density = left_counts / (left_widths * len(X_left))
    right_density = right_counts / (right_widths * len(X_right))

    # Fit local polynomial (quadratic) to log densities
    # Avoid log(0) by adding small constant
    eps = 1e-10
    log_left_density = np.log(left_density + eps)
    log_right_density = np.log(right_density + eps)

    # Estimate density at cutoff from both sides
    f_left_at_cutoff = _estimate_density_at_cutoff(left_centers, log_left_density, cutoff, bandwidth)
    f_right_at_cutoff = _estimate_density_at_cutoff(right_centers, log_right_density, cutoff, bandwidth)

    # Test statistic: log difference
    theta = np.log(f_right_at_cutoff / f_left_at_cutoff)

    # Standard error (binomial approximation)
    # SE(θ) ≈ sqrt(1/n_left + 1/n_right)
    se_theta = np.sqrt(1 / len(X_left) + 1 / len(X_right))

    # Z-test
    z_stat = theta / se_theta
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Interpretation
    if p_value < 0.05:
        interpretation = f"Potential manipulation detected (p={p_value:.3f}, p<0.05)"
    else:
        interpretation = f"No evidence of manipulation (p={p_value:.3f})"

    return float(theta), float(p_value), interpretation


def covariate_balance_test(
    X: np.ndarray,
    W: np.ndarray,
    cutoff: float,
    bandwidth: float | Literal["ik", "cct"] = "ik",
    covariate_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Test covariate balance at cutoff (placebo RDD).

    Pre-treatment covariates should NOT have discontinuity if RDD is valid.
    Discontinuity in covariate → sorting on observables → RDD invalid.

    This is a falsification test: we run RDD with each pre-treatment covariate
    as the outcome. If we find significant "effects", this suggests units are
    sorting across the cutoff based on observables, violating the RDD assumption.

    Parameters
    ----------
    X : array-like, shape (n,)
        Running variable
    W : array-like, shape (n, p)
        Pre-treatment covariates matrix (p covariates)
        Can also be 1D array for single covariate
    cutoff : float
        RDD cutoff value
    bandwidth : float or str, default='ik'
        Bandwidth or method ('ik', 'cct')
    covariate_names : list of str, optional
        Names of covariates. If None, uses ['W1', 'W2', ...]

    Returns
    -------
    results : DataFrame
        Columns: covariate, estimate, se, t_stat, p_value, significant
        One row per covariate
        significant=True if p < 0.05 (balance violation)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> # Balanced covariates (no discontinuity)
    >>> W1 = np.random.normal(0, 1, n)
    >>> W2 = np.random.uniform(0, 1, n)
    >>> W = np.column_stack([W1, W2])
    >>>
    >>> results = covariate_balance_test(X, W, cutoff=0.0)
    >>> print(results)
      covariate  estimate       se  t_stat  p_value  significant
    0        W1    -0.045    0.089  -0.505    0.614        False
    1        W2     0.012    0.032   0.375    0.708        False

    Notes
    -----
    Interpretation:
    - All p > 0.05: Covariates balanced, RDD assumption plausible
    - Any p < 0.05: Covariate imbalance, potential sorting
    - Multiple p < 0.05: Strong evidence of sorting, RDD likely invalid

    What to do if balance fails:
    - Check if imbalance is economically significant (not just statistically)
    - Consider controlling for imbalanced covariates in RDD
    - If severe: RDD may not be valid identification strategy
    """
    X = np.asarray(X).flatten()
    W = np.asarray(W)

    # Handle 1D covariate
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    n, p = W.shape

    # Covariate names
    if covariate_names is None:
        covariate_names = [f"W{i+1}" for i in range(p)]

    # Run RDD for each covariate
    results = []
    for j in range(p):
        W_j = W[:, j]

        # Fit Sharp RDD with W_j as outcome
        rdd = SharpRDD(cutoff=cutoff, bandwidth=bandwidth, inference="robust")
        rdd.fit(W_j, X)

        results.append(
            {
                "covariate": covariate_names[j],
                "estimate": rdd.coef_,
                "se": rdd.se_,
                "t_stat": rdd.t_stat_,
                "p_value": rdd.p_value_,
                "significant": rdd.p_value_ < 0.05,
            }
        )

    return pd.DataFrame(results)


def bandwidth_sensitivity_analysis(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    h_optimal: float,
    bandwidth_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Bandwidth sensitivity analysis.

    Estimates treatment effect at h ∈ {0.5h, 0.75h, h, 1.5h, 2h}.
    Result should be stable if RDD is robust.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    h_optimal : float
        Optimal bandwidth (from IK or CCT)
    bandwidth_grid : array-like, optional
        Custom grid. Default: [0.5, 0.75, 1.0, 1.5, 2.0] * h_optimal

    Returns
    -------
    results : DataFrame
        Columns: bandwidth, estimate, se, ci_lower, ci_upper
        One row per bandwidth

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> Y = X + 2.0 * (X >= 0) + np.random.normal(0, 1, n)
    >>>
    >>> from causal_inference.rdd.bandwidth import imbens_kalyanaraman_bandwidth
    >>> h_opt = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)
    >>>
    >>> results = bandwidth_sensitivity_analysis(Y, X, 0.0, h_opt)
    >>> print(results)
       bandwidth  estimate    se  ci_lower  ci_upper
    0       0.62     2.105  0.21     1.690     2.520
    1       0.93     2.043  0.17     1.702     2.384
    2       1.24     2.018  0.15     1.720     2.316
    3       1.86     1.987  0.12     1.744     2.230
    4       2.48     1.965  0.10     1.764     2.166

    Notes
    -----
    Interpretation:
    - Stable estimates (±20%): RDD is robust to bandwidth choice
    - Large variation (>50%): Result sensitive to bandwidth, interpret with caution
    - Systematic trend: May indicate misspecification or curvature

    What to do if sensitive:
    - Check for nonlinearity (use higher-order polynomial)
    - Try CCT bandwidth (may be more robust)
    - Report range of estimates across bandwidths
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    # Default grid
    if bandwidth_grid is None:
        bandwidth_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0]) * h_optimal

    results = []
    for h in bandwidth_grid:
        # Fit Sharp RDD with fixed bandwidth
        rdd = SharpRDD(cutoff=cutoff, bandwidth=h, inference="robust")
        rdd.fit(Y, X)

        results.append(
            {
                "bandwidth": h,
                "estimate": rdd.coef_,
                "se": rdd.se_,
                "ci_lower": rdd.ci_[0],
                "ci_upper": rdd.ci_[1],
            }
        )

    return pd.DataFrame(results)


def polynomial_order_sensitivity(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    bandwidth: float,
    max_order: int = 3,
) -> pd.DataFrame:
    """
    Polynomial order sensitivity analysis.

    Tests local constant (p=0), local linear (p=1), local quadratic (p=2), etc.
    Result should be stable across orders if specification is correct.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    bandwidth : float
        Fixed bandwidth for all polynomial orders
    max_order : int, default=3
        Maximum polynomial order to test

    Returns
    -------
    results : DataFrame
        Columns: order, estimate, se, ci_lower, ci_upper
        One row per polynomial order (0, 1, 2, ..., max_order)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> Y = X**2 + 2.0 * (X >= 0) + np.random.normal(0, 1, n)
    >>>
    >>> results = polynomial_order_sensitivity(Y, X, 0.0, bandwidth=1.5)
    >>> print(results)
       order  estimate    se  ci_lower  ci_upper
    0      0     1.874  0.24     1.404     2.344
    1      1     2.021  0.19     1.647     2.395
    2      2     2.005  0.18     1.651     2.359
    3      3     2.012  0.18     1.660     2.364

    Notes
    -----
    Interpretation:
    - Similar estimates across orders: Specification robust
    - Large differences: May indicate nonlinearity or overfitting
    - Order 1 (local linear) recommended as default (Fan & Gijbels 1996)

    What to do if unstable:
    - If p=0 differs from p=1,2: Likely linear trend, use p≥1
    - If p=1 differs from p=2,3: Likely curvature, use p≥2
    - If p=2,3 differ: Possible overfitting, stick with p=1 or p=2
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    results = []
    for order in range(max_order + 1):
        # Fit local polynomial regression of given order
        estimate, se = _local_polynomial_regression(
            Y, X, cutoff=cutoff, bandwidth=bandwidth, order=order
        )

        # Compute CI using t-distribution
        # df approximation: effective n on each side minus parameters
        in_band = np.abs(X - cutoff) <= bandwidth
        n_eff = np.sum(in_band)
        df = max(n_eff - 2 * (order + 1), 1)
        t_crit = stats.t.ppf(0.975, df=df)

        results.append(
            {
                "order": order,
                "estimate": estimate,
                "se": se,
                "ci_lower": estimate - t_crit * se,
                "ci_upper": estimate + t_crit * se,
            }
        )

    return pd.DataFrame(results)


def donut_hole_rdd(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    bandwidth: float,
    hole_width: float | list[float] = 0.1,
) -> pd.DataFrame:
    """
    Donut-hole RDD: exclude |X - c| < δ.

    If manipulation only affects units at exact cutoff, estimate should
    be stable when excluding small window around cutoff.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    bandwidth : float
        Fixed bandwidth for all donut-hole widths
    hole_width : float or list of float, default=0.1
        Width of exclusion window. Can be single value or list.
        If single value, uses [0, hole_width/2, hole_width, 2*hole_width]

    Returns
    -------
    results : DataFrame
        Columns: hole_width, estimate, se, n_excluded
        One row per hole width

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> Y = X + 2.0 * (X >= 0) + np.random.normal(0, 1, n)
    >>>
    >>> results = donut_hole_rdd(Y, X, 0.0, bandwidth=1.5, hole_width=0.2)
    >>> print(results)
       hole_width  estimate    se  n_excluded
    0        0.00     2.018  0.15           0
    1        0.10     2.024  0.16          42
    2        0.20     2.011  0.17          87
    3        0.40     1.998  0.19         156

    Notes
    -----
    Interpretation:
    - Stable estimates: Manipulation (if any) doesn't affect result
    - Large changes (>30%): Manipulation near cutoff affecting estimate
    - Increasing SE: Normal (fewer observations)

    What to do if unstable:
    - If estimate decreases with hole width: Possible manipulation
    - If very unstable: Consider donut-hole RDD as primary specification
    - Report both full sample and donut-hole estimates

    Limitations:
    - Loses precision (fewer observations)
    - Ad-hoc choice of hole width
    - If effect varies near cutoff, donut-hole changes estimand
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    # Create hole width grid
    if isinstance(hole_width, (int, float)):
        hole_widths = [0, hole_width / 2, hole_width, 2 * hole_width]
    else:
        hole_widths = list(hole_width)

    results = []
    for delta in hole_widths:
        # Exclude observations with |X - cutoff| < delta
        mask = np.abs(X - cutoff) >= delta
        Y_donut = Y[mask]
        X_donut = X[mask]
        n_excluded = len(Y) - len(Y_donut)

        if len(Y_donut) < 50:
            # Too few observations
            results.append(
                {
                    "hole_width": delta,
                    "estimate": np.nan,
                    "se": np.nan,
                    "n_excluded": n_excluded,
                }
            )
            continue

        # Fit Sharp RDD on donut data
        rdd = SharpRDD(cutoff=cutoff, bandwidth=bandwidth, inference="robust")
        try:
            rdd.fit(Y_donut, X_donut)

            results.append(
                {
                    "hole_width": delta,
                    "estimate": rdd.coef_,
                    "se": rdd.se_,
                    "n_excluded": n_excluded,
                }
            )
        except ValueError:
            # Not enough observations on one side
            results.append(
                {
                    "hole_width": delta,
                    "estimate": np.nan,
                    "se": np.nan,
                    "n_excluded": n_excluded,
                }
            )

    return pd.DataFrame(results)
