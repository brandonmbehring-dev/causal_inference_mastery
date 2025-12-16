"""
Synthetic Control Methods - Inference

Provides inference methods for SCM:
- In-space placebo tests (Abadie et al. 2010)
- In-time placebo tests
- Bootstrap inference (for multiple treated units)

References:
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
    for Comparative Case Studies"
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .weights import compute_scm_weights


def placebo_test_in_space(
    control_outcomes: np.ndarray,
    treatment_period: int,
    observed_effect: float,
    n_placebo: Optional[int] = None,
) -> dict:
    """
    Perform in-space placebo test for SCM inference.

    For each control unit, pretend it was treated and compute the "placebo effect"
    using remaining controls as donors. Compare observed effect to placebo
    distribution.

    Parameters
    ----------
    control_outcomes : np.ndarray
        Control unit outcomes (n_control, n_periods)
    treatment_period : int
        Period when treatment starts
    observed_effect : float
        Observed ATT for actual treated unit
    n_placebo : int, optional
        Number of placebo iterations (defaults to all controls)

    Returns
    -------
    dict
        Contains:
        - placebo_effects: array of placebo treatment effects
        - p_value: two-sided p-value
        - se: standard error from placebo distribution
        - ratio: observed/placebo RMSPE ratio (if possible)
        - pre_post_ratios: array of post/pre RMSPE ratios for each placebo
    """
    n_control, n_periods = control_outcomes.shape
    n_pre = treatment_period
    n_post = n_periods - treatment_period

    # Limit placebo iterations
    if n_placebo is None:
        n_placebo = n_control
    n_placebo = min(n_placebo, n_control)

    placebo_effects = []
    pre_post_ratios = []

    for i in range(n_placebo):
        # Treat control unit i as "treated"
        pseudo_treated = control_outcomes[i, :]
        pseudo_control_mask = np.ones(n_control, dtype=bool)
        pseudo_control_mask[i] = False
        pseudo_control = control_outcomes[pseudo_control_mask, :]

        if pseudo_control.shape[0] < 2:
            continue

        pseudo_treated_pre = pseudo_treated[:n_pre].reshape(1, -1)
        pseudo_control_pre = pseudo_control[:, :n_pre]

        try:
            weights, _ = compute_scm_weights(pseudo_treated_pre, pseudo_control_pre)

            # Compute synthetic for all periods
            pseudo_synthetic = pseudo_control.T @ weights
            pseudo_gap = pseudo_treated - pseudo_synthetic

            # Pre and post RMSPE
            pre_rmspe = np.sqrt(np.mean(pseudo_gap[:n_pre] ** 2))
            post_rmspe = np.sqrt(np.mean(pseudo_gap[n_pre:] ** 2))

            # Effect = mean post-treatment gap
            pseudo_effect = np.mean(pseudo_gap[n_pre:])
            placebo_effects.append(pseudo_effect)

            # Ratio (filter if pre-fit is poor)
            if pre_rmspe > 1e-6:
                pre_post_ratios.append(post_rmspe / pre_rmspe)

        except Exception:
            continue

    if len(placebo_effects) < 2:
        return {
            "placebo_effects": np.array([]),
            "p_value": np.nan,
            "se": np.nan,
            "ratio": np.nan,
            "pre_post_ratios": np.array([]),
        }

    placebo_effects = np.array(placebo_effects)
    pre_post_ratios = np.array(pre_post_ratios)

    # Standard error from placebo distribution
    se = np.std(placebo_effects, ddof=1)

    # Two-sided p-value
    n_extreme = np.sum(np.abs(placebo_effects) >= np.abs(observed_effect))
    p_value = (n_extreme + 1) / (len(placebo_effects) + 1)

    return {
        "placebo_effects": placebo_effects,
        "p_value": float(p_value),
        "se": float(se),
        "ratio": np.nan,  # Computed in main function if needed
        "pre_post_ratios": pre_post_ratios,
    }


def placebo_test_in_time(
    treated_series: np.ndarray,
    synthetic_series: np.ndarray,
    treatment_period: int,
    pseudo_treatment_period: int,
) -> dict:
    """
    Perform in-time placebo test.

    Apply SCM to a pseudo-treatment period before actual treatment to check
    if the method would detect a "false" effect.

    Parameters
    ----------
    treated_series : np.ndarray
        Observed treated series (n_periods,)
    synthetic_series : np.ndarray
        Synthetic control series (n_periods,)
    treatment_period : int
        Actual treatment period
    pseudo_treatment_period : int
        Period to use for pseudo-treatment (must be < treatment_period)

    Returns
    -------
    dict
        Contains:
        - pseudo_effect: estimated effect at pseudo-treatment
        - actual_effect: actual post-treatment effect
        - ratio: actual/pseudo effect ratio
    """
    if pseudo_treatment_period >= treatment_period:
        raise ValueError(
            f"pseudo_treatment_period ({pseudo_treatment_period}) must be "
            f"< treatment_period ({treatment_period})"
        )

    gap = treated_series - synthetic_series

    # Pseudo effect: gap from pseudo period to actual treatment
    pseudo_gap = gap[pseudo_treatment_period:treatment_period]
    pseudo_effect = np.mean(pseudo_gap)

    # Actual effect
    actual_gap = gap[treatment_period:]
    actual_effect = np.mean(actual_gap)

    # Ratio (higher is better - actual effect should dominate pseudo)
    if abs(pseudo_effect) > 1e-10:
        ratio = abs(actual_effect) / abs(pseudo_effect)
    else:
        ratio = np.inf if abs(actual_effect) > 1e-10 else 1.0

    return {
        "pseudo_effect": float(pseudo_effect),
        "actual_effect": float(actual_effect),
        "ratio": float(ratio),
    }


def bootstrap_se(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    treatment_period: int,
    n_bootstrap: int = 500,
    block_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute bootstrap standard error for SCM.

    Uses block bootstrap on time periods to preserve temporal dependence.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Treated unit outcomes (n_treated, n_periods) or (n_periods,)
    control_outcomes : np.ndarray
        Control unit outcomes (n_control, n_periods)
    treatment_period : int
        Period when treatment starts
    n_bootstrap : int
        Number of bootstrap iterations
    block_length : int, optional
        Block length for block bootstrap (default: sqrt(n_pre))
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    se : float
        Bootstrap standard error
    effects : np.ndarray
        Bootstrap distribution of effects
    """
    if seed is not None:
        np.random.seed(seed)

    # Handle 1D treated case
    if treated_outcomes.ndim == 1:
        treated_outcomes = treated_outcomes.reshape(1, -1)

    n_treated, n_periods = treated_outcomes.shape
    n_control = control_outcomes.shape[0]
    n_pre = treatment_period
    n_post = n_periods - treatment_period

    # Default block length
    if block_length is None:
        block_length = max(1, int(np.sqrt(n_pre)))

    treated_avg = treated_outcomes.mean(axis=0)

    bootstrap_effects = []

    for _ in range(n_bootstrap):
        # Block bootstrap on pre-treatment periods
        n_blocks = int(np.ceil(n_pre / block_length))
        start_indices = np.random.randint(0, n_pre - block_length + 1, size=n_blocks)

        boot_pre_idx = []
        for start in start_indices:
            boot_pre_idx.extend(range(start, min(start + block_length, n_pre)))
        boot_pre_idx = np.array(boot_pre_idx[:n_pre])

        # Resample pre-treatment periods
        boot_treated_pre = treated_avg[boot_pre_idx]
        boot_control_pre = control_outcomes[:, boot_pre_idx]

        try:
            weights, _ = compute_scm_weights(
                boot_treated_pre.reshape(1, -1), boot_control_pre
            )

            # Apply weights to original post-treatment (no resampling)
            boot_synthetic_post = control_outcomes[:, n_pre:].T @ weights
            boot_gap_post = treated_avg[n_pre:] - boot_synthetic_post
            boot_effect = np.mean(boot_gap_post)
            bootstrap_effects.append(boot_effect)

        except Exception:
            continue

    if len(bootstrap_effects) < 2:
        return np.nan, np.array([])

    bootstrap_effects = np.array(bootstrap_effects)
    se = np.std(bootstrap_effects, ddof=1)

    return se, bootstrap_effects


def compute_confidence_interval(
    estimate: float,
    se: float,
    alpha: float = 0.05,
    method: str = "normal",
    bootstrap_effects: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute confidence interval for SCM estimate.

    Parameters
    ----------
    estimate : float
        Point estimate (ATT)
    se : float
        Standard error
    alpha : float
        Significance level (default: 0.05 for 95% CI)
    method : str
        "normal" for normal approximation, "percentile" for bootstrap percentile
    bootstrap_effects : np.ndarray, optional
        Bootstrap distribution (required for percentile method)

    Returns
    -------
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    """
    if method == "normal":
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = estimate - z * se
        ci_upper = estimate + z * se

    elif method == "percentile":
        if bootstrap_effects is None or len(bootstrap_effects) < 2:
            return np.nan, np.nan
        ci_lower = np.percentile(bootstrap_effects, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha / 2))

    else:
        raise ValueError(f"Unknown CI method: {method}. Use 'normal' or 'percentile'.")

    return float(ci_lower), float(ci_upper)


def compute_p_value(
    observed_effect: float,
    placebo_effects: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    """
    Compute p-value from placebo distribution.

    Parameters
    ----------
    observed_effect : float
        Observed treatment effect
    placebo_effects : np.ndarray
        Placebo effect distribution
    alternative : str
        "two-sided", "greater", or "less"

    Returns
    -------
    p_value : float
        P-value from placebo distribution
    """
    if len(placebo_effects) == 0:
        return np.nan

    n = len(placebo_effects)

    if alternative == "two-sided":
        n_extreme = np.sum(np.abs(placebo_effects) >= np.abs(observed_effect))
    elif alternative == "greater":
        n_extreme = np.sum(placebo_effects >= observed_effect)
    elif alternative == "less":
        n_extreme = np.sum(placebo_effects <= observed_effect)
    else:
        raise ValueError(
            f"Unknown alternative: {alternative}. "
            "Use 'two-sided', 'greater', or 'less'."
        )

    # Add 1 for observed effect itself
    p_value = (n_extreme + 1) / (n + 1)

    return float(p_value)
