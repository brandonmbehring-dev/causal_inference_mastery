"""
Synthetic Control Methods - Core Estimator

Implements the synthetic control method (Abadie et al. 2003, 2010, 2015) for
comparative case studies with few treated units.

The method constructs a synthetic control as a weighted combination of
untreated units that matches the treated unit's pre-treatment trajectory.

References:
    Abadie, A., & Gardeazabal, J. (2003). "The Economic Costs of Conflict"
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods"
    Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and
        the Synthetic Control Method"
"""

from typing import Optional

import numpy as np
from scipy import stats

from .types import SCMResult, validate_panel_data
from .weights import compute_scm_weights, compute_pre_treatment_fit


def synthetic_control(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    covariate_weight: float = 1.0,
    inference: str = "placebo",
    n_placebo: int = 100,
) -> SCMResult:
    """
    Estimate treatment effect using Synthetic Control Method.

    Constructs a synthetic control as a weighted average of donor units that
    matches the treated unit's pre-treatment outcomes. The treatment effect
    is the difference between the treated unit and synthetic control in the
    post-treatment period.

    Parameters
    ----------
    outcomes : np.ndarray
        Panel data with shape (n_units, n_periods). Each row is a unit,
        each column is a time period.
    treatment : np.ndarray
        Binary treatment indicator with shape (n_units,). 1 = treated, 0 = control.
    treatment_period : int
        Period when treatment starts (0-indexed). Periods 0 to treatment_period-1
        are pre-treatment; treatment_period onwards are post-treatment.
    covariates : np.ndarray, optional
        Pre-treatment covariates with shape (n_units, n_covariates).
        Used to improve matching if provided.
    alpha : float
        Significance level for confidence intervals (default: 0.05 for 95% CI).
    covariate_weight : float
        Relative weight on covariate matching vs outcome matching (default: 1.0).
    inference : str
        Inference method: "placebo" (in-space placebo tests), "bootstrap",
        or "none" (no inference).
    n_placebo : int
        Number of placebo iterations for inference.

    Returns
    -------
    SCMResult
        Dictionary containing:
        - estimate: Average treatment effect on treated
        - se: Standard error
        - ci_lower, ci_upper: Confidence interval bounds
        - p_value: P-value from placebo distribution
        - weights: Synthetic control weights
        - pre_rmse: Pre-treatment fit RMSE
        - pre_r_squared: Pre-treatment fit R²
        - n_treated, n_control: Unit counts
        - n_pre_periods, n_post_periods: Period counts
        - synthetic_control: Counterfactual series
        - treated_series: Observed treated series
        - gap: Period-by-period treatment effects

    Raises
    ------
    ValueError
        If inputs fail validation (dimensions, missing data, etc.)

    Examples
    --------
    >>> import numpy as np
    >>> # Panel: 5 units, 10 periods
    >>> outcomes = np.random.randn(5, 10)
    >>> treatment = np.array([1, 0, 0, 0, 0])  # First unit treated
    >>> treatment_period = 5  # Treatment starts at period 5
    >>> result = synthetic_control(outcomes, treatment, treatment_period)
    >>> print(f"ATT: {result['estimate']:.3f} (SE: {result['se']:.3f})")
    """
    # Validate inputs
    validate_panel_data(outcomes, treatment, treatment_period, covariates)

    n_units, n_periods = outcomes.shape
    n_pre_periods = treatment_period
    n_post_periods = n_periods - treatment_period

    # Identify treated and control units
    treated_mask = treatment == 1
    control_mask = treatment == 0
    n_treated = np.sum(treated_mask)
    n_control = np.sum(control_mask)

    # Extract treated and control data
    treated_outcomes = outcomes[treated_mask, :]  # (n_treated, n_periods)
    control_outcomes = outcomes[control_mask, :]  # (n_control, n_periods)

    # Split into pre/post periods
    treated_pre = treated_outcomes[:, :treatment_period]  # (n_treated, n_pre)
    control_pre = control_outcomes[:, :treatment_period]  # (n_control, n_pre)
    treated_post = treated_outcomes[:, treatment_period:]  # (n_treated, n_post)
    control_post = control_outcomes[:, treatment_period:]  # (n_control, n_post)

    # Handle covariates
    cov_treated = None
    cov_control = None
    if covariates is not None:
        cov_treated = covariates[treated_mask, :].mean(axis=0)  # Average if multiple treated
        cov_control = covariates[control_mask, :]

    # Compute optimal weights
    weights, opt_result = compute_scm_weights(
        treated_pre=treated_pre,
        control_pre=control_pre,
        covariates_treated=cov_treated,
        covariates_control=cov_control,
        covariate_weight=covariate_weight,
    )

    # Compute synthetic control series
    treated_series = treated_outcomes.mean(axis=0)  # Average if multiple treated
    synthetic_series = control_outcomes.T @ weights  # (n_periods,)

    # Compute gap (treatment effect by period)
    gap = treated_series - synthetic_series

    # Pre-treatment fit
    pre_rmse, pre_r_squared = compute_pre_treatment_fit(
        treated_pre.mean(axis=0), control_pre, weights
    )

    # Post-treatment effect (ATT)
    post_gap = gap[treatment_period:]
    estimate = np.mean(post_gap)

    # Inference
    if inference == "placebo":
        se, p_value = _placebo_inference(
            control_outcomes=control_outcomes,
            treatment_period=treatment_period,
            observed_effect=estimate,
            n_placebo=n_placebo,
        )
    elif inference == "bootstrap":
        se, p_value = _bootstrap_inference(
            treated_pre=treated_pre.mean(axis=0),
            control_pre=control_pre,
            treated_post=treated_post.mean(axis=0),
            control_post=control_post,
            n_bootstrap=n_placebo,
            observed_effect=estimate,
        )
    elif inference == "none":
        se = np.nan
        p_value = np.nan
    else:
        raise ValueError(f"Unknown inference method: {inference}. Use 'placebo', 'bootstrap', or 'none'")

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = estimate - z * se if not np.isnan(se) else np.nan
    ci_upper = estimate + z * se if not np.isnan(se) else np.nan

    return SCMResult(
        estimate=float(estimate),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        weights=weights,
        pre_rmse=float(pre_rmse),
        pre_r_squared=float(pre_r_squared),
        n_treated=int(n_treated),
        n_control=int(n_control),
        n_pre_periods=int(n_pre_periods),
        n_post_periods=int(n_post_periods),
        synthetic_control=synthetic_series,
        treated_series=treated_series,
        gap=gap,
    )


def _placebo_inference(
    control_outcomes: np.ndarray,
    treatment_period: int,
    observed_effect: float,
    n_placebo: int = 100,
) -> tuple:
    """
    Compute p-value and SE using in-space placebo tests.

    For each control unit, pretend it was treated and compute the
    "placebo effect" using remaining controls as donors.

    Parameters
    ----------
    control_outcomes : np.ndarray
        Control unit outcomes (n_control, n_periods)
    treatment_period : int
        Period when treatment starts
    observed_effect : float
        Observed ATT for actual treated unit
    n_placebo : int
        Number of placebo iterations (up to n_control)

    Returns
    -------
    se : float
        Standard error from placebo distribution
    p_value : float
        Two-sided p-value (proportion of placebos with |effect| >= |observed|)
    """
    n_control, n_periods = control_outcomes.shape
    n_pre = treatment_period
    n_post = n_periods - treatment_period

    # Limit placebo iterations to available control units
    n_placebo = min(n_placebo, n_control)

    placebo_effects = []

    for i in range(n_placebo):
        # Treat control unit i as "treated"
        pseudo_treated = control_outcomes[i, :]
        pseudo_control_mask = np.ones(n_control, dtype=bool)
        pseudo_control_mask[i] = False
        pseudo_control = control_outcomes[pseudo_control_mask, :]

        if pseudo_control.shape[0] < 2:
            continue  # Need at least 2 donors

        # Compute weights for pseudo-treated
        pseudo_treated_pre = pseudo_treated[:n_pre].reshape(1, -1)
        pseudo_control_pre = pseudo_control[:, :n_pre]

        try:
            weights, _ = compute_scm_weights(pseudo_treated_pre, pseudo_control_pre)

            # Compute pseudo effect
            pseudo_synthetic = pseudo_control.T @ weights
            pseudo_gap = pseudo_treated - pseudo_synthetic
            pseudo_effect = np.mean(pseudo_gap[n_pre:])
            placebo_effects.append(pseudo_effect)
        except Exception:
            # Skip if optimization fails
            continue

    if len(placebo_effects) < 2:
        return np.nan, np.nan

    placebo_effects = np.array(placebo_effects)

    # Standard error from placebo distribution
    se = np.std(placebo_effects, ddof=1)

    # Two-sided p-value
    n_extreme = np.sum(np.abs(placebo_effects) >= np.abs(observed_effect))
    p_value = (n_extreme + 1) / (len(placebo_effects) + 1)

    return se, p_value


def _bootstrap_inference(
    treated_pre: np.ndarray,
    control_pre: np.ndarray,
    treated_post: np.ndarray,
    control_post: np.ndarray,
    n_bootstrap: int = 200,
    observed_effect: float = 0.0,
) -> tuple:
    """
    Compute SE and p-value using bootstrap resampling.

    Resamples time periods (block bootstrap on pre-treatment) and
    recomputes weights to get bootstrap distribution.

    Parameters
    ----------
    treated_pre : np.ndarray
        Pre-treatment outcomes for treated (n_pre,)
    control_pre : np.ndarray
        Pre-treatment outcomes for controls (n_control, n_pre)
    treated_post : np.ndarray
        Post-treatment outcomes for treated (n_post,)
    control_post : np.ndarray
        Post-treatment outcomes for controls (n_control, n_post)
    n_bootstrap : int
        Number of bootstrap samples
    observed_effect : float
        Observed ATT

    Returns
    -------
    se : float
        Bootstrap standard error
    p_value : float
        Bootstrap p-value (two-sided)
    """
    n_pre = len(treated_pre)
    n_control = control_pre.shape[0]
    n_post = len(treated_post)

    bootstrap_effects = []

    for _ in range(n_bootstrap):
        # Resample time periods (pre-treatment)
        pre_idx = np.random.choice(n_pre, size=n_pre, replace=True)
        boot_treated_pre = treated_pre[pre_idx]
        boot_control_pre = control_pre[:, pre_idx]

        try:
            weights, _ = compute_scm_weights(
                boot_treated_pre.reshape(1, -1), boot_control_pre
            )

            # Apply weights to post-treatment (no resampling)
            boot_synthetic_post = control_post.T @ weights
            boot_gap_post = treated_post - boot_synthetic_post
            boot_effect = np.mean(boot_gap_post)
            bootstrap_effects.append(boot_effect)
        except Exception:
            continue

    if len(bootstrap_effects) < 2:
        return np.nan, np.nan

    bootstrap_effects = np.array(bootstrap_effects)

    se = np.std(bootstrap_effects, ddof=1)

    # Two-sided p-value
    n_extreme = np.sum(np.abs(bootstrap_effects - observed_effect) >= np.abs(observed_effect))
    p_value = (n_extreme + 1) / (len(bootstrap_effects) + 1)

    return se, p_value
