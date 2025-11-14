"""Inverse Probability Weighting (IPW) ATE estimator.

This module implements IPW for estimating average treatment effects. IPW reweights
units by inverse of propensity scores to create a pseudo-population where treatment
is independent of covariates.

Key benefit: Works for RCTs with varying assignment probabilities and observational studies.
"""

import numpy as np
from scipy import stats
from typing import Dict, Union


def ipw_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    propensity: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Calculate Average Treatment Effect using Inverse Probability Weighting.

    IPW reweights units by inverse of propensity scores (P(T=1|X)) to create
    balance. Under correct propensity model, IPW is consistent for ATE.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control). Also accepts boolean.
    propensity : np.ndarray or list
        Propensity scores P(T=1|X) for each unit. Must be in (0,1) exclusive.
        For simple RCT: constant value (e.g., 0.5).
        For blocked RCT: block-specific probabilities.
        For observational: estimated from logistic regression.
    alpha : float, default=0.05
        Significance level for confidence interval (must be in (0, 1)).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': IPW estimate of ATE
        - 'se': Standard error (robust to heteroskedasticity and weighting)
        - 'ci_lower': Lower bound of (1-alpha)% CI
        - 'ci_upper': Upper bound of (1-alpha)% CI
        - 'n_treated': Number of treated units
        - 'n_control': Number of control units
        - 'effective_n': Effective sample size (accounting for weights)

    Raises
    ------
    ValueError
        If inputs invalid (propensity out of range, mismatched lengths, NaN, etc.)

    Examples
    --------
    >>> # Simple RCT (constant propensity)
    >>> treatment = np.array([1, 1, 0, 0])
    >>> outcomes = np.array([7, 5, 3, 1])
    >>> propensity = np.array([0.5, 0.5, 0.5, 0.5])
    >>> result = ipw_ate(outcomes, treatment, propensity)
    >>> result['estimate']  # Same as simple difference-in-means
    4.0

    >>> # Varying propensity (blocked RCT)
    >>> propensity = np.array([0.8, 0.6, 0.6, 0.8])
    >>> result = ipw_ate(outcomes, treatment, propensity)
    >>> result['estimate']  # Adjusts for varying assignment probabilities

    Notes
    -----
    - Weights: w_i = 1/P(T=1|X) for treated, 1/(1-P(T=1|X)) for control
    - Estimator: Horvitz-Thompson weighted difference-in-means
    - Variance: Robust estimator accounting for weights
    - Assumes propensity scores are known or correctly estimated
    - Large weights (extreme propensity) increase variance
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    # Convert to numpy arrays
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    propensity = np.asarray(propensity, dtype=float)

    n = len(outcomes)

    # Check lengths match
    if len(treatment) != n or len(propensity) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: ipw_ate\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"len(propensity)={len(propensity)}"
        )

    # Check for empty
    if n == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: ipw_ate\n"
            f"Expected: Non-empty arrays"
        )

    # Check for NaN
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)) or np.any(np.isnan(propensity)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: ipw_ate\n"
            f"NaN indicates data quality issues that must be addressed.\n"
            f"Got: {np.sum(np.isnan(outcomes))} NaN in outcomes, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment, "
            f"{np.sum(np.isnan(propensity))} NaN in propensity"
        )

    # Check for infinite values
    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)) or np.any(np.isinf(propensity)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: ipw_ate\n"
            f"Got: {np.sum(np.isinf(outcomes))} inf in outcomes, "
            f"{np.sum(np.isinf(treatment))} inf in treatment, "
            f"{np.sum(np.isinf(propensity))} inf in propensity"
        )

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: ipw_ate\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # Check for treatment variation
    if len(unique_treatment) < 2:
        if unique_treatment[0] == 1:
            raise ValueError(
                f"CRITICAL ERROR: No control units in data.\n"
                f"Function: ipw_ate\n"
                f"Cannot estimate treatment effect without control group.\n"
                f"Got: All units have treatment=1"
            )
        else:
            raise ValueError(
                f"CRITICAL ERROR: No treated units in data.\n"
                f"Function: ipw_ate\n"
                f"Cannot estimate treatment effect without treated group.\n"
                f"Got: All units have treatment=0"
            )

    # Check propensity scores are in (0,1) exclusive
    if np.any(propensity <= 0) or np.any(propensity >= 1):
        raise ValueError(
            f"CRITICAL ERROR: Propensity scores must be in (0,1) exclusive.\n"
            f"Function: ipw_ate\n"
            f"Propensity scores represent probabilities and cannot be 0, 1, or outside this range.\n"
            f"Got: min={np.min(propensity)}, max={np.max(propensity)}\n"
            f"Valid range: (0, 1) exclusive"
        )

    # Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\n"
            f"Function: ipw_ate\n"
            f"Expected: alpha in (0, 1)\n"
            f"Got: alpha={alpha}"
        )

    # ============================================================================
    # IPW Estimation
    # ============================================================================

    # Compute IPW weights
    # Treated: w_i = 1 / P(T=1|X_i)
    # Control: w_i = 1 / P(T=0|X_i) = 1 / (1 - P(T=1|X_i))
    weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))

    # Weighted means
    # Treated mean = sum(w_i * Y_i * T_i) / sum(w_i * T_i)
    treated_mask = treatment == 1
    control_mask = treatment == 0

    weighted_sum_treated = np.sum(weights[treated_mask] * outcomes[treated_mask])
    sum_weights_treated = np.sum(weights[treated_mask])

    weighted_sum_control = np.sum(weights[control_mask] * outcomes[control_mask])
    sum_weights_control = np.sum(weights[control_mask])

    mean_treated = weighted_sum_treated / sum_weights_treated
    mean_control = weighted_sum_control / sum_weights_control

    # IPW ATE estimate
    ate = mean_treated - mean_control

    # ============================================================================
    # Variance Estimation (Robust)
    # ============================================================================

    # Variance of weighted mean: Var(weighted mean) = sum(w_i^2 * (Y_i - mean)^2) / (sum w_i)^2

    # Treated variance
    residuals_treated = outcomes[treated_mask] - mean_treated
    var_treated = (
        np.sum((weights[treated_mask] ** 2) * (residuals_treated ** 2))
        / (sum_weights_treated ** 2)
    )

    # Control variance
    residuals_control = outcomes[control_mask] - mean_control
    var_control = (
        np.sum((weights[control_mask] ** 2) * (residuals_control ** 2))
        / (sum_weights_control ** 2)
    )

    # Variance of ATE
    var_ate = var_treated + var_control

    # Standard error
    se = np.sqrt(var_ate)

    # Confidence interval
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_critical * se
    ci_upper = ate + z_critical * se

    # Effective sample size (sum of weights)^2 / sum of squared weights
    # This measures how much information we have accounting for variable weights
    effective_n_treated = (sum_weights_treated ** 2) / np.sum(weights[treated_mask] ** 2)
    effective_n_control = (sum_weights_control ** 2) / np.sum(weights[control_mask] ** 2)
    effective_n = effective_n_treated + effective_n_control

    # Total counts
    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    return {
        "estimate": float(ate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_treated": n_treated,
        "n_control": n_control,
        "effective_n": float(effective_n),
    }
