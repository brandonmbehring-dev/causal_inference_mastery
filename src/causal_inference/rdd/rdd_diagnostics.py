"""
RDD Diagnostic Tools.

This module provides diagnostic tools for validating RDD designs:
- Covariate balance tests (falsification)
- Donut-hole RDD (robustness to exclusion near cutoff)

For other diagnostics, see:
- mccrary.py: McCrary density test for manipulation detection
- sensitivity.py: Bandwidth and polynomial order sensitivity analysis

References
----------
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. Journal of Econometrics, 142(2), 698-714.
- Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the
  regression discontinuity estimator. Review of Economic Studies, 79(3), 933-959.
"""

from typing import Literal

import numpy as np
import pandas as pd

from .sharp_rdd import SharpRDD

# Re-export common diagnostics from specialized modules
from .mccrary import mccrary_density_test
from .sensitivity import bandwidth_sensitivity_analysis, polynomial_order_sensitivity

__all__ = [
    "covariate_balance_test",
    "donut_hole_rdd",
    "mccrary_density_test",
    "bandwidth_sensitivity_analysis",
    "polynomial_order_sensitivity",
]


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
