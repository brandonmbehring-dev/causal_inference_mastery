"""
Diagnostics for MTE estimation.

Tools for checking identifying assumptions and estimation quality.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import stats

from .types import CommonSupportResult


def common_support_check(
    propensity: np.ndarray,
    treatment: np.ndarray,
    trim_fraction: float = 0.01,
) -> CommonSupportResult:
    """
    Check propensity score overlap between treatment groups.

    MTE identification requires propensity scores to vary sufficiently.
    Limited overlap restricts the region where MTE can be estimated.

    Parameters
    ----------
    propensity : np.ndarray
        Estimated propensity scores P(D=1|Z)
    treatment : np.ndarray
        Treatment indicator D
    trim_fraction : float, default=0.01
        Fraction to trim from tails

    Returns
    -------
    CommonSupportResult
        Diagnostics about support region

    Notes
    -----
    - MTE is only identified where propensity has positive density in both groups
    - Thin tails cause large SE in MTE
    """
    p = np.asarray(propensity)
    d = np.asarray(treatment)

    # Propensity by treatment status
    p_treated = p[d == 1]
    p_control = p[d == 0]

    # Overall bounds
    p_min_treated = np.percentile(p_treated, 100 * trim_fraction)
    p_max_treated = np.percentile(p_treated, 100 * (1 - trim_fraction))

    p_min_control = np.percentile(p_control, 100 * trim_fraction)
    p_max_control = np.percentile(p_control, 100 * (1 - trim_fraction))

    # Common support region
    support_min = max(p_min_treated, p_min_control)
    support_max = min(p_max_treated, p_max_control)

    has_support = support_max > support_min

    # Count units outside support
    outside_support = (p < support_min) | (p > support_max)
    n_outside = outside_support.sum()
    fraction_outside = n_outside / len(p)

    # Generate recommendation
    if not has_support:
        recommendation = (
            "NO COMMON SUPPORT: Propensity distributions do not overlap. "
            "MTE cannot be estimated. Check instrument strength and model."
        )
    elif fraction_outside > 0.2:
        recommendation = (
            f"LIMITED SUPPORT: {fraction_outside:.1%} of observations outside "
            f"common support [{support_min:.2f}, {support_max:.2f}]. "
            "Consider stronger instruments or different specification."
        )
    elif (support_max - support_min) < 0.3:
        recommendation = (
            f"NARROW SUPPORT: Only [{support_min:.2f}, {support_max:.2f}] is "
            "identified. MTE extrapolation beyond this range is unreliable."
        )
    else:
        recommendation = (
            f"ADEQUATE SUPPORT: [{support_min:.2f}, {support_max:.2f}] "
            f"with {1 - fraction_outside:.1%} of observations in support."
        )

    return CommonSupportResult(
        has_support=has_support,
        support_region=(support_min, support_max) if has_support else (np.nan, np.nan),
        n_outside_support=int(n_outside),
        fraction_outside=fraction_outside,
        recommendation=recommendation,
    )


def mte_sensitivity_to_trimming(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    trim_fractions: List[float] = [0.01, 0.025, 0.05, 0.10],
) -> dict:
    """
    Assess sensitivity of MTE to propensity score trimming.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator
    instrument : np.ndarray
        Instrument(s)
    covariates : np.ndarray, optional
        Covariates
    trim_fractions : list
        Trimming fractions to evaluate

    Returns
    -------
    dict
        Results for each trimming level:
        - ate_estimates: ATE at each trim level
        - support_widths: Support width at each trim level
        - n_trimmed: Units trimmed at each level

    Notes
    -----
    - Large sensitivity to trimming suggests thin tails / extrapolation issues
    - Stable estimates across trim levels indicate robust identification
    """
    from .local_iv import local_iv
    from .policy import ate_from_mte

    results = {
        "trim_fractions": trim_fractions,
        "ate_estimates": [],
        "ate_se": [],
        "support_widths": [],
        "n_trimmed": [],
    }

    for trim in trim_fractions:
        try:
            mte_result = local_iv(
                outcome, treatment, instrument, covariates,
                trim_fraction=trim, n_bootstrap=100
            )
            ate_result = ate_from_mte(mte_result)

            p_min, p_max = mte_result["propensity_support"]

            results["ate_estimates"].append(ate_result["estimate"])
            results["ate_se"].append(ate_result["se"])
            results["support_widths"].append(p_max - p_min)
            results["n_trimmed"].append(mte_result["n_trimmed"])

        except Exception:
            results["ate_estimates"].append(np.nan)
            results["ate_se"].append(np.nan)
            results["support_widths"].append(np.nan)
            results["n_trimmed"].append(np.nan)

    # Assess stability
    ate_estimates = np.array(results["ate_estimates"])
    valid = ~np.isnan(ate_estimates)

    if valid.sum() >= 2:
        ate_range = ate_estimates[valid].max() - ate_estimates[valid].min()
        ate_mean = np.nanmean(ate_estimates)
        relative_range = ate_range / abs(ate_mean) if abs(ate_mean) > 1e-10 else np.inf

        if relative_range < 0.1:
            results["sensitivity_assessment"] = "ROBUST: ATE stable across trimming"
        elif relative_range < 0.25:
            results["sensitivity_assessment"] = "MODERATE: Some sensitivity to trimming"
        else:
            results["sensitivity_assessment"] = "SENSITIVE: Large variation with trimming"
    else:
        results["sensitivity_assessment"] = "INSUFFICIENT: Could not compute at multiple levels"

    return results


def monotonicity_test(
    treatment: np.ndarray,
    instrument: np.ndarray,
    significance: float = 0.05,
) -> dict:
    """
    Test for monotonicity assumption in LATE/MTE framework.

    Monotonicity: D(z') ≥ D(z) for all individuals when z' > z.
    (No defiers: no one decreases treatment when instrument increases)

    Parameters
    ----------
    treatment : np.ndarray
        Treatment indicator D
    instrument : np.ndarray
        Instrument Z (binary for this test)
    significance : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Test results:
        - z0_d1_rate: P(D=1|Z=0) - always-taker rate
        - z1_d1_rate: P(D=1|Z=1) - treated rate with high Z
        - complier_share: Upper bound on complier share
        - monotonicity_plausible: Whether monotonicity is consistent with data
        - notes: Interpretation

    Notes
    -----
    - Monotonicity is NOT directly testable (involves counterfactuals)
    - This test checks for CONSISTENCY with monotonicity
    - Violations indicate either defiers OR model misspecification
    """
    D = np.asarray(treatment)
    Z = np.asarray(instrument)

    # Convert to binary if continuous
    if len(np.unique(Z)) > 2:
        Z = (Z > np.median(Z)).astype(float)

    z_values = np.unique(Z)
    if len(z_values) != 2:
        return {
            "error": "Instrument must be binary (or binarizable) for monotonicity test",
            "monotonicity_plausible": None,
        }

    z_low, z_high = sorted(z_values)

    # Treatment rates
    d_rate_z_low = D[Z == z_low].mean()
    d_rate_z_high = D[Z == z_high].mean()

    # First stage coefficient
    first_stage = d_rate_z_high - d_rate_z_low

    # Under monotonicity:
    # P(D=1|Z=0) = P(always-taker)
    # P(D=1|Z=1) = P(always-taker) + P(complier)
    # P(D=1|Z=1) - P(D=1|Z=0) = P(complier) ≥ 0

    always_taker_rate = d_rate_z_low
    never_taker_rate = 1 - d_rate_z_high
    complier_share = first_stage

    # Test: first stage should be non-negative under monotonicity
    # (or non-positive if instrument effect is negative)
    monotonicity_plausible = first_stage >= -1e-6

    # Standard error for first stage
    n_low = (Z == z_low).sum()
    n_high = (Z == z_high).sum()
    se_first_stage = np.sqrt(
        d_rate_z_low * (1 - d_rate_z_low) / n_low +
        d_rate_z_high * (1 - d_rate_z_high) / n_high
    )

    # Test statistic
    t_stat = first_stage / se_first_stage if se_first_stage > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    # Generate notes
    if first_stage < 0:
        notes = (
            "First stage is NEGATIVE - instrument decreases treatment. "
            "This could indicate: (1) defiers exist, (2) instrument coded backwards, "
            "or (3) model misspecification. Check instrument direction."
        )
    elif first_stage < 0.05:
        notes = (
            "First stage is WEAK (< 0.05). Complier share is small, "
            "making LATE poorly identified. Consider stronger instruments."
        )
    elif abs(t_stat) < 2:
        notes = (
            f"First stage not statistically significant (t={t_stat:.2f}). "
            "Cannot reject zero first stage. IV assumptions may be violated."
        )
    else:
        notes = (
            f"First stage is significant (t={t_stat:.2f}) and positive. "
            f"Data is CONSISTENT with monotonicity. Complier share ≈ {complier_share:.1%}."
        )

    return {
        "z0_d1_rate": d_rate_z_low,
        "z1_d1_rate": d_rate_z_high,
        "first_stage": first_stage,
        "first_stage_se": se_first_stage,
        "first_stage_t": t_stat,
        "first_stage_p": p_value,
        "always_taker_share": always_taker_rate,
        "never_taker_share": never_taker_rate,
        "complier_share": max(0, complier_share),
        "monotonicity_plausible": monotonicity_plausible,
        "notes": notes,
    }


def propensity_variation_test(
    propensity: np.ndarray,
    min_variation: float = 0.1,
) -> dict:
    """
    Test whether propensity scores vary enough for MTE identification.

    Parameters
    ----------
    propensity : np.ndarray
        Estimated propensity scores
    min_variation : float, default=0.1
        Minimum required variation (support width)

    Returns
    -------
    dict
        - sufficient_variation: Whether variation is adequate
        - support_width: Range of propensity scores
        - coefficient_of_variation: CV of propensity
        - recommendation: Action to take
    """
    p = np.asarray(propensity)

    p_min = p.min()
    p_max = p.max()
    support_width = p_max - p_min

    p_mean = p.mean()
    p_std = p.std()
    cv = p_std / p_mean if p_mean > 0 else 0

    sufficient = support_width >= min_variation

    if not sufficient:
        recommendation = (
            f"INSUFFICIENT VARIATION: Support width {support_width:.3f} < {min_variation}. "
            "Propensity scores are too concentrated. MTE not well identified. "
            "Consider: (1) stronger instruments, (2) additional instruments, "
            "(3) different functional form."
        )
    elif cv < 0.1:
        recommendation = (
            f"LOW VARIATION: CV = {cv:.3f}. Propensity scores cluster around mean. "
            "MTE identified but may have large SE."
        )
    else:
        recommendation = (
            f"ADEQUATE VARIATION: Support [{p_min:.2f}, {p_max:.2f}], CV = {cv:.2f}."
        )

    return {
        "sufficient_variation": sufficient,
        "p_min": p_min,
        "p_max": p_max,
        "support_width": support_width,
        "coefficient_of_variation": cv,
        "recommendation": recommendation,
    }


def mte_shape_test(
    mte_grid: np.ndarray,
    u_grid: np.ndarray,
    se_grid: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Test hypotheses about MTE shape.

    Parameters
    ----------
    mte_grid : np.ndarray
        MTE estimates at grid points
    u_grid : np.ndarray
        Grid points
    se_grid : np.ndarray
        Standard errors at grid points
    alpha : float
        Significance level

    Returns
    -------
    dict
        Tests for:
        - constant_mte: Test H0: MTE is constant (no heterogeneity)
        - monotone_mte: Whether MTE is monotonically increasing/decreasing
        - linear_mte: Test H0: MTE is linear in u
    """
    valid = ~np.isnan(mte_grid)
    mte = mte_grid[valid]
    u = u_grid[valid]
    se = se_grid[valid]

    results = {}

    if len(mte) < 5:
        return {"error": "Insufficient grid points for shape tests"}

    # 1. Test for constant MTE
    mte_mean = np.mean(mte)
    mte_var = np.var(mte, ddof=1)

    # Chi-squared test: deviations from mean
    if np.all(se > 0):
        chi_sq = np.sum(((mte - mte_mean) / se) ** 2)
        df = len(mte) - 1
        p_constant = 1 - stats.chi2.cdf(chi_sq, df)
        results["constant_mte_test"] = {
            "chi_square": chi_sq,
            "df": df,
            "p_value": p_constant,
            "reject_constant": p_constant < alpha,
            "interpretation": (
                "MTE varies significantly with u" if p_constant < alpha
                else "Cannot reject constant MTE"
            ),
        }
    else:
        results["constant_mte_test"] = {"error": "SE all zero"}

    # 2. Test for monotonicity
    diffs = np.diff(mte)
    monotone_increasing = np.all(diffs >= 0)
    monotone_decreasing = np.all(diffs <= 0)

    # Account for noise: check if mostly monotone
    frac_increasing = np.mean(diffs >= 0) if len(diffs) > 0 else 0
    frac_decreasing = np.mean(diffs <= 0) if len(diffs) > 0 else 0

    results["monotonicity_test"] = {
        "strictly_monotone": monotone_increasing or monotone_decreasing,
        "direction": (
            "increasing" if monotone_increasing
            else "decreasing" if monotone_decreasing
            else "non-monotone"
        ),
        "frac_increasing": frac_increasing,
        "frac_decreasing": frac_decreasing,
        "interpretation": (
            f"MTE is approximately {'increasing' if frac_increasing > 0.7 else 'decreasing' if frac_decreasing > 0.7 else 'non-monotone'}"
        ),
    }

    # 3. Test for linearity
    # Fit linear model: MTE = a + b*u
    X = np.column_stack([np.ones(len(u)), u])
    beta = np.linalg.lstsq(X, mte, rcond=None)[0]
    mte_linear = X @ beta
    residuals = mte - mte_linear

    # F-test for nonlinearity (quadratic component)
    X_quad = np.column_stack([X, u**2])
    beta_quad = np.linalg.lstsq(X_quad, mte, rcond=None)[0]
    mte_quad = X_quad @ beta_quad
    residuals_quad = mte - mte_quad

    ss_linear = np.sum(residuals**2)
    ss_quad = np.sum(residuals_quad**2)

    if ss_linear > 0:
        f_stat = ((ss_linear - ss_quad) / 1) / (ss_quad / (len(mte) - 3))
        p_linear = 1 - stats.f.cdf(f_stat, 1, len(mte) - 3)
    else:
        f_stat = 0
        p_linear = 1.0

    results["linearity_test"] = {
        "f_statistic": f_stat,
        "p_value": p_linear,
        "reject_linear": p_linear < alpha,
        "linear_slope": beta[1],
        "interpretation": (
            "MTE is significantly nonlinear" if p_linear < alpha
            else "Cannot reject linear MTE"
        ),
    }

    return results
