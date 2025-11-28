"""
Sensitivity Analysis for Regression Discontinuity Designs.

This module provides tools for testing the robustness of RDD estimates to:
1. Bandwidth choice (h sensitivity)
2. Polynomial order specification (p sensitivity)

Stable estimates across specifications strengthen causal claims, while
sensitivity to specification choices warrants cautious interpretation.

References
----------
Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs:
A guide to practice. Journal of Econometrics, 142(2), 615-635.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .sharp_rdd import SharpRDD
from .mccrary import _local_polynomial_regression


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
