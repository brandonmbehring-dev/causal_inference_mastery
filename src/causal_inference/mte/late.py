"""
Local Average Treatment Effect (LATE) estimation.

Implements Imbens & Angrist (1994) LATE framework for binary instruments.

LATE = E[Y₁ - Y₀ | Complier] = Cov(Y, Z) / Cov(D, Z)
"""

from typing import Optional, Union, List
import numpy as np
from scipy import stats

from .types import LATEResult, ComplierResult


def late_estimator(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    instrument: Union[np.ndarray, List[int]],
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> LATEResult:
    """
    Estimate Local Average Treatment Effect via Wald estimator.

    LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])

    This is equivalent to 2SLS with a binary instrument.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y
    treatment : array-like
        Binary treatment indicator D ∈ {0, 1}
    instrument : array-like
        Binary instrument Z ∈ {0, 1}
    covariates : np.ndarray, optional
        Exogenous controls X (n x p). If provided, uses residualized outcomes.
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    LATEResult
        Dictionary containing:
        - late: LATE estimate
        - se: Standard error
        - ci_lower, ci_upper: Confidence interval
        - complier_share: P(D₁ > D₀)
        - first_stage_f: First-stage F-statistic

    Raises
    ------
    ValueError
        If instrument is not binary or has no variation

    Examples
    --------
    >>> # Angrist & Krueger style: Quarter of birth as instrument for education
    >>> late_result = late_estimator(wages, education, qob_instrument)
    >>> print(f"LATE: {late_result['late']:.3f} (SE: {late_result['se']:.3f})")

    References
    ----------
    - Imbens, G.W. & Angrist, J.D. (1994). Identification and Estimation of LATE.
    - Angrist, J.D., Imbens, G.W., & Rubin, D.B. (1996). Identification of Causal
      Effects Using Instrumental Variables.
    """
    # Convert to numpy arrays
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    n = len(Y)

    # Validate inputs
    _validate_late_inputs(Y, D, Z)

    # For LATE with binary instrument, we don't residualize Z
    # Instead, we condition on X by including it in the regression
    # For simplicity, we just residualize Y on X (partial out covariates from outcome)
    if covariates is not None:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Y = _residualize(Y, X)

    # Compute Wald estimator components
    z1_mask = Z == 1
    z0_mask = Z == 0

    # Reduced form: E[Y|Z=1] - E[Y|Z=0]
    mean_y_z1 = Y[z1_mask].mean()
    mean_y_z0 = Y[z0_mask].mean()
    reduced_form = mean_y_z1 - mean_y_z0

    # First stage: E[D|Z=1] - E[D|Z=0]
    mean_d_z1 = D[z1_mask].mean()
    mean_d_z0 = D[z0_mask].mean()
    first_stage = mean_d_z1 - mean_d_z0

    # Check first stage strength
    if abs(first_stage) < 1e-10:
        raise ValueError(
            f"First stage coefficient is essentially zero ({first_stage:.2e}). "
            "Instrument has no effect on treatment."
        )

    # LATE = reduced form / first stage
    late = reduced_form / first_stage

    # Complier share = first stage coefficient
    complier_share = first_stage

    # Compute always-takers and never-takers
    # P(D=1|Z=0) = always-takers
    # P(D=0|Z=1) = never-takers
    always_taker_share = mean_d_z0
    never_taker_share = 1 - mean_d_z1

    # Standard error via delta method
    # Var(LATE) ≈ (1/first_stage²) * [Var(reduced_form) + LATE² * Var(first_stage)
    #              - 2 * LATE * Cov(reduced_form, first_stage)]
    n_z1 = z1_mask.sum()
    n_z0 = z0_mask.sum()

    var_y_z1 = Y[z1_mask].var(ddof=1) if n_z1 > 1 else 0
    var_y_z0 = Y[z0_mask].var(ddof=1) if n_z0 > 1 else 0
    var_d_z1 = D[z1_mask].var(ddof=1) if n_z1 > 1 else 0
    var_d_z0 = D[z0_mask].var(ddof=1) if n_z0 > 1 else 0

    # Variance of reduced form
    var_reduced_form = var_y_z1 / n_z1 + var_y_z0 / n_z0

    # Variance of first stage
    var_first_stage = var_d_z1 / n_z1 + var_d_z0 / n_z0

    # Covariance (using within-group covariance)
    cov_yd_z1 = np.cov(Y[z1_mask], D[z1_mask])[0, 1] if n_z1 > 1 else 0
    cov_yd_z0 = np.cov(Y[z0_mask], D[z0_mask])[0, 1] if n_z0 > 1 else 0
    cov_rf_fs = cov_yd_z1 / n_z1 + cov_yd_z0 / n_z0

    # Delta method variance
    var_late = (1 / first_stage**2) * (
        var_reduced_form + late**2 * var_first_stage - 2 * late * cov_rf_fs
    )
    se = np.sqrt(max(var_late, 0))

    # First-stage F-statistic
    first_stage_f = _compute_first_stage_f(D, Z, covariates)

    # Confidence interval and p-value
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = late - z_crit * se
    ci_upper = late + z_crit * se

    z_stat = late / se if se > 0 else 0
    pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return LATEResult(
        late=late,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        complier_share=complier_share,
        always_taker_share=always_taker_share,
        never_taker_share=never_taker_share,
        first_stage_coef=first_stage,
        first_stage_f=first_stage_f,
        n_obs=n,
        method="wald",
    )


def late_bounds(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    instrument: Union[np.ndarray, List[int]],
    alpha: float = 0.05,
) -> dict:
    """
    Compute bounds on LATE when monotonicity may be violated.

    Under weaker assumptions, we can bound LATE without point identification.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y
    treatment : array-like
        Binary treatment D
    instrument : array-like
        Binary instrument Z
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with bounds_lower, bounds_upper, and diagnostics
    """
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    _validate_late_inputs(Y, D, Z)

    # Get outcome support
    y_min, y_max = Y.min(), Y.max()

    z1_mask = Z == 1
    z0_mask = Z == 0

    # Conditional means
    mean_y_z1 = Y[z1_mask].mean()
    mean_y_z0 = Y[z0_mask].mean()
    mean_d_z1 = D[z1_mask].mean()
    mean_d_z0 = D[z0_mask].mean()

    # First stage
    first_stage = mean_d_z1 - mean_d_z0

    if abs(first_stage) < 1e-10:
        return {
            "bounds_lower": -np.inf,
            "bounds_upper": np.inf,
            "first_stage": first_stage,
            "warning": "First stage too weak for meaningful bounds",
        }

    # Worst-case bounds (no monotonicity)
    # Lower bound: assume defiers have maximal positive treatment effect
    # Upper bound: assume defiers have maximal negative treatment effect
    reduced_form = mean_y_z1 - mean_y_z0

    # Under monotonicity, LATE = reduced_form / first_stage
    late_mono = reduced_form / first_stage

    # Manski-type bounds without monotonicity
    # These are conservative: Y ∈ [y_min, y_max]
    bounds_lower = (reduced_form - (y_max - y_min) * abs(first_stage)) / first_stage
    bounds_upper = (reduced_form + (y_max - y_min) * abs(first_stage)) / first_stage

    if bounds_lower > bounds_upper:
        bounds_lower, bounds_upper = bounds_upper, bounds_lower

    return {
        "bounds_lower": bounds_lower,
        "bounds_upper": bounds_upper,
        "late_under_monotonicity": late_mono,
        "first_stage": first_stage,
        "reduced_form": reduced_form,
        "outcome_support": (y_min, y_max),
        "bounds_width": bounds_upper - bounds_lower,
    }


def complier_characteristics(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    instrument: Union[np.ndarray, List[int]],
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
) -> ComplierResult:
    """
    Characterize the complier subpopulation.

    Uses Abadie (2003) kappa-weighting to estimate complier characteristics.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y
    treatment : array-like
        Binary treatment D
    instrument : array-like
        Binary instrument Z
    covariates : np.ndarray, optional
        Covariates X for which to compute complier means
    covariate_names : list, optional
        Names for covariates

    Returns
    -------
    ComplierResult
        Dictionary with complier characteristics

    References
    ----------
    - Abadie, A. (2003). Semiparametric Instrumental Variable Estimation of
      Treatment Response Models.
    """
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    _validate_late_inputs(Y, D, Z)

    z1_mask = Z == 1
    z0_mask = Z == 0

    # First stage for complier share
    mean_d_z1 = D[z1_mask].mean()
    mean_d_z0 = D[z0_mask].mean()
    complier_share = mean_d_z1 - mean_d_z0

    if complier_share <= 0:
        raise ValueError(
            f"Non-positive complier share ({complier_share:.4f}). "
            "Monotonicity may be violated or instrument is defective."
        )

    # Kappa weights for compliers (Abadie 2003)
    # κ = 1 - D(1-Z)/(1-P(Z=1)) - (1-D)Z/P(Z=1)
    # Simplified for binary Z:
    p_z1 = Z.mean()
    p_z0 = 1 - p_z1

    # For D=1, Z=1 group: κ = 1
    # For D=0, Z=0 group: κ = 1
    # For D=1, Z=0 group: κ = 1 - 1/p_z0 (negative for always-takers)
    # For D=0, Z=1 group: κ = 1 - 1/p_z1 (negative for never-takers)

    kappa = np.ones(len(Y))
    kappa[(D == 1) & (Z == 0)] = 1 - 1 / p_z0 if p_z0 > 0 else 0
    kappa[(D == 0) & (Z == 1)] = 1 - 1 / p_z1 if p_z1 > 0 else 0

    # Complier mean outcomes
    # E[Y|D=1, complier] via kappa-weighting
    kappa_d1 = kappa.copy()
    kappa_d1[D == 0] = 0
    kappa_d1_sum = kappa_d1.sum()

    if kappa_d1_sum > 0:
        complier_mean_y1 = (Y * kappa_d1).sum() / kappa_d1_sum
    else:
        complier_mean_y1 = np.nan

    # E[Y|D=0, complier] via kappa-weighting
    kappa_d0 = kappa.copy()
    kappa_d0[D == 1] = 0
    kappa_d0_sum = kappa_d0.sum()

    if kappa_d0_sum > 0:
        complier_mean_y0 = (Y * kappa_d0).sum() / kappa_d0_sum
    else:
        complier_mean_y0 = np.nan

    # Covariate means for compliers
    covariate_means = None
    if covariates is not None:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        kappa_pos = np.maximum(kappa, 0)  # Only positive weights
        kappa_sum = kappa_pos.sum()

        if kappa_sum > 0:
            covariate_means = (X.T @ kappa_pos) / kappa_sum
        else:
            covariate_means = np.full(X.shape[1], np.nan)

    return ComplierResult(
        complier_mean_outcome_treated=complier_mean_y1,
        complier_mean_outcome_control=complier_mean_y0,
        complier_share=complier_share,
        covariate_means=covariate_means,
        covariate_names=covariate_names,
        method="kappa_weights",
    )


def _validate_late_inputs(Y: np.ndarray, D: np.ndarray, Z: np.ndarray) -> None:
    """Validate inputs for LATE estimation."""
    n = len(Y)

    # Length checks
    if len(D) != n or len(Z) != n:
        raise ValueError(
            f"Input length mismatch: outcome ({n}), treatment ({len(D)}), instrument ({len(Z)})"
        )

    # Binary checks
    unique_z = np.unique(Z[~np.isnan(Z)])
    if not np.all(np.isin(unique_z, [0, 1])):
        raise ValueError(f"Instrument must be binary (0/1). Found values: {unique_z}")

    unique_d = np.unique(D[~np.isnan(D)])
    if not np.all(np.isin(unique_d, [0, 1])):
        raise ValueError(f"Treatment must be binary (0/1). Found values: {unique_d}")

    # Variation checks
    if len(unique_z) < 2:
        raise ValueError("Instrument has no variation")

    if len(unique_d) < 2:
        raise ValueError("Treatment has no variation")

    # Sample size in each instrument group
    n_z1 = (Z == 1).sum()
    n_z0 = (Z == 0).sum()

    if n_z1 < 2:
        raise ValueError(f"Only {n_z1} observations with Z=1 (need at least 2)")
    if n_z0 < 2:
        raise ValueError(f"Only {n_z0} observations with Z=0 (need at least 2)")


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residualize y on X via OLS."""
    n = len(y)
    X_with_const = np.column_stack([np.ones(n), X])

    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        return y - X_with_const @ beta
    except np.linalg.LinAlgError:
        # Fallback: return original
        return y


def _compute_first_stage_f(
    D: np.ndarray,
    Z: np.ndarray,
    covariates: Optional[np.ndarray],
) -> float:
    """Compute first-stage F-statistic for instrument strength."""
    n = len(D)

    # Build design matrix
    if covariates is not None:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        design = np.column_stack([np.ones(n), Z, X])
        k_restricted = X.shape[1] + 1  # Intercept + covariates
    else:
        design = np.column_stack([np.ones(n), Z])
        k_restricted = 1  # Just intercept

    k_full = design.shape[1]

    # Full model: D ~ 1 + Z + X
    try:
        beta_full = np.linalg.lstsq(design, D, rcond=None)[0]
        ssr_full = np.sum((D - design @ beta_full) ** 2)
    except np.linalg.LinAlgError:
        return 0.0

    # Restricted model: D ~ 1 + X (no Z)
    if covariates is not None:
        design_r = np.column_stack([np.ones(n), X])
    else:
        design_r = np.ones((n, 1))

    try:
        beta_r = np.linalg.lstsq(design_r, D, rcond=None)[0]
        ssr_restricted = np.sum((D - design_r @ beta_r) ** 2)
    except np.linalg.LinAlgError:
        return 0.0

    # F-statistic
    q = 1  # Number of restrictions (just Z)
    df_num = q
    df_denom = n - k_full

    if df_denom <= 0 or ssr_full <= 0:
        return 0.0

    f_stat = ((ssr_restricted - ssr_full) / df_num) / (ssr_full / df_denom)
    return max(f_stat, 0.0)
