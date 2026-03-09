"""
Diagnostics for Regression Kink Design (RKD)

Implements validation tests for RKD assumptions:
1. Density smoothness - no bunching at the kink
2. Covariate smoothness - predetermined covariates continuous at kink
3. First stage strength - sufficient variation in treatment at kink

Key Assumption:
--------------
For RKD to be valid, units must not be able to precisely manipulate their
position around the kink. This is tested by checking for smoothness in:
- The density of the running variable
- Predetermined covariates

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal
    effects in a generalized regression kink design. Econometrica, 83(6).
McCrary, J. (2008). Manipulation of the running variable in the regression
    discontinuity design: A density test. Journal of Econometrics, 142(2).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
from scipy import stats


@dataclass
class DensitySmoothnessResult:
    """
    Result of density smoothness test at kink.

    Attributes
    ----------
    slope_left : float
        Estimated density slope on left of kink
    slope_right : float
        Estimated density slope on right of kink
    slope_difference : float
        Difference in slopes (test for discontinuity in density slope)
    se : float
        Standard error of the slope difference
    t_stat : float
        T-statistic
    p_value : float
        P-value (H0: slopes are equal)
    n_bins : int
        Number of bins used
    interpretation : str
        Human-readable interpretation
    """

    slope_left: float
    slope_right: float
    slope_difference: float
    se: float
    t_stat: float
    p_value: float
    n_bins: int
    interpretation: str


@dataclass
class CovariateSmoothnessResult:
    """
    Result of covariate smoothness test.

    Attributes
    ----------
    covariate_name : str
        Name of the covariate tested
    slope_left : float
        Estimated covariate slope on left of kink
    slope_right : float
        Estimated covariate slope on right of kink
    slope_difference : float
        Difference in slopes
    se : float
        Standard error
    t_stat : float
        T-statistic
    p_value : float
        P-value (H0: smooth at kink)
    is_smooth : bool
        Whether the covariate passes the smoothness test
    """

    covariate_name: str
    slope_left: float
    slope_right: float
    slope_difference: float
    se: float
    t_stat: float
    p_value: float
    is_smooth: bool


@dataclass
class FirstStageResult:
    """
    Result of first stage strength test.

    Attributes
    ----------
    kink_estimate : float
        Estimated kink in treatment
    se : float
        Standard error
    f_stat : float
        F-statistic (kink_estimate² / se²)
    p_value : float
        P-value for H0: no kink
    is_strong : bool
        Whether first stage is strong (F > 10)
    interpretation : str
        Human-readable interpretation
    """

    kink_estimate: float
    se: float
    f_stat: float
    p_value: float
    is_strong: bool
    interpretation: str


def density_smoothness_test(
    x: np.ndarray,
    cutoff: float,
    n_bins: int = 20,
    bandwidth: Optional[float] = None,
) -> DensitySmoothnessResult:
    """
    Test for smoothness in the density of the running variable at the kink.

    Unlike McCrary's test for RDD (which tests for a jump in density level),
    this tests for a kink in the density function - i.e., whether the
    derivative of the density changes at the cutoff.

    Parameters
    ----------
    x : array-like
        Running variable
    cutoff : float
        Kink point
    n_bins : int, default=20
        Number of bins on each side
    bandwidth : float, optional
        Bandwidth for density estimation. If None, uses Silverman's rule.

    Returns
    -------
    DensitySmoothnessResult
        Test results including slopes, difference, and p-value

    Notes
    -----
    H0: The density is smooth at the kink (no bunching)
    H1: There is a kink in the density (bunching/manipulation)

    A significant result (p < 0.05) suggests potential manipulation.
    """
    x = np.asarray(x).flatten()

    # Silverman's rule for bandwidth
    if bandwidth is None:
        sigma = np.std(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        bandwidth = 0.9 * min(sigma, iqr / 1.34) * len(x) ** (-0.2)

    # Create bins
    x_left = x[x < cutoff]
    x_right = x[x >= cutoff]

    if len(x_left) < 20 or len(x_right) < 20:
        return DensitySmoothnessResult(
            slope_left=np.nan,
            slope_right=np.nan,
            slope_difference=np.nan,
            se=np.nan,
            t_stat=np.nan,
            p_value=1.0,
            n_bins=n_bins,
            interpretation="Insufficient data for density test",
        )

    # Bin the data
    left_bins = np.linspace(x_left.min(), cutoff, n_bins + 1)
    right_bins = np.linspace(cutoff, x_right.max(), n_bins + 1)

    left_counts, _ = np.histogram(x_left, bins=left_bins)
    right_counts, _ = np.histogram(x_right, bins=right_bins)

    left_centers = (left_bins[:-1] + left_bins[1:]) / 2
    right_centers = (right_bins[:-1] + right_bins[1:]) / 2

    # Convert to density
    left_widths = np.diff(left_bins)
    right_widths = np.diff(right_bins)
    left_density = left_counts / (left_widths * len(x_left))
    right_density = right_counts / (right_widths * len(x_right))

    # Fit linear regression to log density on each side to get slope
    # (slope of log density = derivative of density / density)

    # Filter out zero counts
    left_valid = left_density > 0
    right_valid = right_density > 0

    if np.sum(left_valid) < 3 or np.sum(right_valid) < 3:
        return DensitySmoothnessResult(
            slope_left=np.nan,
            slope_right=np.nan,
            slope_difference=np.nan,
            se=np.nan,
            t_stat=np.nan,
            p_value=1.0,
            n_bins=n_bins,
            interpretation="Insufficient non-zero bins for density test",
        )

    # Fit slopes
    x_left_c = left_centers[left_valid] - cutoff
    y_left = np.log(left_density[left_valid])

    x_right_c = right_centers[right_valid] - cutoff
    y_right = np.log(right_density[right_valid])

    # Linear regression for slopes
    slope_left, intercept_left, _, _, se_left = stats.linregress(x_left_c, y_left)
    slope_right, intercept_right, _, _, se_right = stats.linregress(x_right_c, y_right)

    # Test for difference in slopes
    slope_diff = slope_right - slope_left
    se_diff = np.sqrt(se_left**2 + se_right**2)

    if se_diff > 0:
        t_stat = slope_diff / se_diff
        df = np.sum(left_valid) + np.sum(right_valid) - 4
        df = max(df, 1)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        t_stat = np.inf if slope_diff != 0 else 0
        p_value = 0.0 if slope_diff != 0 else 1.0

    # Interpretation
    if p_value < 0.01:
        interpretation = (
            f"Strong evidence of density kink (p={p_value:.4f}). Potential manipulation."
        )
    elif p_value < 0.05:
        interpretation = f"Evidence of density kink (p={p_value:.4f}). Possible manipulation."
    elif p_value < 0.10:
        interpretation = f"Weak evidence of density kink (p={p_value:.4f}). Caution advised."
    else:
        interpretation = (
            f"No evidence of density kink (p={p_value:.4f}). Smoothness assumption supported."
        )

    return DensitySmoothnessResult(
        slope_left=float(slope_left),
        slope_right=float(slope_right),
        slope_difference=float(slope_diff),
        se=float(se_diff),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_bins=n_bins,
        interpretation=interpretation,
    )


def covariate_smoothness_test(
    x: np.ndarray,
    covariates: np.ndarray,
    cutoff: float,
    covariate_names: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1,
) -> List[CovariateSmoothnessResult]:
    """
    Test for smoothness of predetermined covariates at the kink.

    If the RKD is valid, predetermined covariates should vary smoothly
    at the kink - they should not exhibit a kink in their relationship
    with the running variable.

    Parameters
    ----------
    x : array-like
        Running variable
    covariates : array-like, shape (n, k)
        Matrix of k covariates to test
    cutoff : float
        Kink point
    covariate_names : list of str, optional
        Names for each covariate
    bandwidth : float, optional
        Bandwidth for local regression
    polynomial_order : int, default=1
        Order of local polynomial

    Returns
    -------
    list of CovariateSmoothnessResult
        Test results for each covariate

    Notes
    -----
    H0: Covariate is smooth at kink
    H1: Covariate has a kink (discontinuity in slope)

    A significant result suggests the covariate is correlated with
    unobserved factors that change at the kink, violating RKD assumptions.
    """
    x = np.asarray(x).flatten()
    covariates = np.asarray(covariates)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, k = covariates.shape

    if covariate_names is None:
        covariate_names = [f"Covariate_{i + 1}" for i in range(k)]

    if bandwidth is None:
        sigma = np.std(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        bandwidth = 1.5 * min(sigma, iqr / 1.34) * n ** (-0.2)

    results = []

    for j in range(k):
        cov = covariates[:, j]
        name = covariate_names[j] if j < len(covariate_names) else f"Covariate_{j + 1}"

        # Filter to bandwidth region
        in_bw = np.abs(x - cutoff) <= bandwidth
        x_bw = x[in_bw]
        cov_bw = cov[in_bw]

        left_mask = x_bw < cutoff
        right_mask = x_bw >= cutoff

        if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
            results.append(
                CovariateSmoothnessResult(
                    covariate_name=name,
                    slope_left=np.nan,
                    slope_right=np.nan,
                    slope_difference=np.nan,
                    se=np.nan,
                    t_stat=np.nan,
                    p_value=1.0,
                    is_smooth=True,  # Can't reject smoothness with insufficient data
                )
            )
            continue

        # Fit local polynomial on each side
        x_left = x_bw[left_mask] - cutoff
        cov_left = cov_bw[left_mask]
        x_right = x_bw[right_mask] - cutoff
        cov_right = cov_bw[right_mask]

        # Simple OLS for slopes
        try:
            slope_left, _, _, _, se_left = stats.linregress(x_left, cov_left)
            slope_right, _, _, _, se_right = stats.linregress(x_right, cov_right)
        except Exception:
            results.append(
                CovariateSmoothnessResult(
                    covariate_name=name,
                    slope_left=np.nan,
                    slope_right=np.nan,
                    slope_difference=np.nan,
                    se=np.nan,
                    t_stat=np.nan,
                    p_value=1.0,
                    is_smooth=True,
                )
            )
            continue

        slope_diff = slope_right - slope_left
        se_diff = np.sqrt(se_left**2 + se_right**2)

        if se_diff > 0:
            t_stat = slope_diff / se_diff
            df = np.sum(left_mask) + np.sum(right_mask) - 4
            df = max(df, 1)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            t_stat = np.inf if slope_diff != 0 else 0
            p_value = 0.0 if slope_diff != 0 else 1.0

        is_smooth = p_value >= 0.05

        results.append(
            CovariateSmoothnessResult(
                covariate_name=name,
                slope_left=float(slope_left),
                slope_right=float(slope_right),
                slope_difference=float(slope_diff),
                se=float(se_diff),
                t_stat=float(t_stat),
                p_value=float(p_value),
                is_smooth=is_smooth,
            )
        )

    return results


def first_stage_test(
    d: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 2,
) -> FirstStageResult:
    """
    Test the strength of the first stage in Fuzzy RKD.

    Tests whether there is a significant kink in E[D|X] at the cutoff,
    which is required for identification in Fuzzy RKD.

    Parameters
    ----------
    d : array-like
        Treatment variable
    x : array-like
        Running variable
    cutoff : float
        Kink point
    bandwidth : float, optional
        Bandwidth for local regression
    polynomial_order : int, default=2
        Order of local polynomial

    Returns
    -------
    FirstStageResult
        Test results including F-statistic

    Notes
    -----
    Rule of thumb: F > 10 indicates a strong first stage.
    F < 10 suggests weak instrument concerns - the LATE may be biased.
    """
    d = np.asarray(d).flatten()
    x = np.asarray(x).flatten()

    if bandwidth is None:
        sigma = np.std(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        bandwidth = 1.5 * min(sigma, iqr / 1.34) * len(x) ** (-0.2)

    # Filter to bandwidth region
    in_bw = np.abs(x - cutoff) <= bandwidth
    x_bw = x[in_bw]
    d_bw = d[in_bw]

    left_mask = x_bw < cutoff
    right_mask = x_bw >= cutoff

    if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
        return FirstStageResult(
            kink_estimate=np.nan,
            se=np.nan,
            f_stat=np.nan,
            p_value=1.0,
            is_strong=False,
            interpretation="Insufficient data for first stage test",
        )

    # Fit local polynomial on each side
    x_left = x_bw[left_mask] - cutoff
    d_left = d_bw[left_mask]
    x_right = x_bw[right_mask] - cutoff
    d_right = d_bw[right_mask]

    try:
        slope_left, _, _, _, se_left = stats.linregress(x_left, d_left)
        slope_right, _, _, _, se_right = stats.linregress(x_right, d_right)
    except Exception:
        return FirstStageResult(
            kink_estimate=np.nan,
            se=np.nan,
            f_stat=np.nan,
            p_value=1.0,
            is_strong=False,
            interpretation="Regression failed in first stage test",
        )

    kink = slope_right - slope_left
    se = np.sqrt(se_left**2 + se_right**2)

    if se > 0:
        f_stat = (kink / se) ** 2
        df1 = 1
        df2 = np.sum(left_mask) + np.sum(right_mask) - 4
        df2 = max(df2, 1)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    else:
        f_stat = np.inf if kink != 0 else 0
        p_value = 0.0 if kink != 0 else 1.0

    is_strong = f_stat >= 10

    if f_stat >= 10:
        interpretation = f"Strong first stage (F={f_stat:.2f} ≥ 10). Identification is reliable."
    elif f_stat >= 5:
        interpretation = f"Moderate first stage (F={f_stat:.2f}). Some weak instrument concern."
    else:
        interpretation = f"Weak first stage (F={f_stat:.2f} < 5). LATE may be severely biased."

    return FirstStageResult(
        kink_estimate=float(kink),
        se=float(se),
        f_stat=float(f_stat),
        p_value=float(p_value),
        is_strong=is_strong,
        interpretation=interpretation,
    )


def rkd_diagnostics_summary(
    y: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    cutoff: float,
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
) -> Dict:
    """
    Run comprehensive RKD diagnostics and return summary.

    Parameters
    ----------
    y : array-like
        Outcome variable
    x : array-like
        Running variable
    d : array-like
        Treatment variable
    cutoff : float
        Kink point
    covariates : array-like, optional
        Predetermined covariates to test
    covariate_names : list of str, optional
        Names for covariates
    bandwidth : float, optional
        Bandwidth for tests

    Returns
    -------
    dict
        Dictionary containing all diagnostic results
    """
    results = {}

    # Density smoothness
    results["density_test"] = density_smoothness_test(x, cutoff, bandwidth=bandwidth)

    # First stage
    results["first_stage_test"] = first_stage_test(d, x, cutoff, bandwidth=bandwidth)

    # Covariate smoothness
    if covariates is not None:
        results["covariate_tests"] = covariate_smoothness_test(
            x, covariates, cutoff, covariate_names, bandwidth
        )
    else:
        results["covariate_tests"] = []

    # Summary
    density_ok = results["density_test"].p_value >= 0.05
    first_stage_ok = results["first_stage_test"].is_strong
    covariates_ok = all(c.is_smooth for c in results["covariate_tests"])

    results["summary"] = {
        "density_smooth": density_ok,
        "first_stage_strong": first_stage_ok,
        "covariates_smooth": covariates_ok,
        "all_pass": density_ok and first_stage_ok and covariates_ok,
    }

    return results
