"""
Lee (2009) bounds for treatment effects under sample selection.

Implements sharp bounds when outcomes are missing for some units due to
attrition or sample selection, under a monotonicity assumption.

Key Assumption (Monotonicity)
-----------------------------
Treatment can only affect selection in one direction:
- Positive: Treatment weakly increases probability of being observed
- Negative: Treatment weakly decreases probability of being observed

Algorithm
---------
1. Compute attrition rates: r₁ = P(missing|T=1), r₀ = P(missing|T=0)
2. Identify always-observed (would be observed regardless of treatment)
3. Trim the group with lower attrition to match the higher attrition rate
4. Compute bounds by trimming from top (lower bound) or bottom (upper bound)
5. Bootstrap for confidence intervals

References
----------
- Lee, D. S. (2009). Training, Wages, and Sample Selection: Estimating
  Sharp Bounds on Treatment Effects. Review of Economic Studies, 76(3), 1071-1102.
"""

from typing import Optional, Literal, Tuple
import numpy as np
import warnings

from .types import LeeBoundsResult


def lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    monotonicity: Literal["positive", "negative"] = "positive",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> LeeBoundsResult:
    """
    Compute Lee (2009) sharp bounds under sample selection.

    When some outcomes are missing due to attrition, point identification
    fails. Under monotonicity, we can construct sharp (tight) bounds.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,). Use NaN or any value for unobserved.
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    observed : np.ndarray
        Observation indicator (n,), binary 0/1. 1 = outcome observed.
    monotonicity : {"positive", "negative"}, default="positive"
        Direction of monotonicity.
        - "positive": Treatment (weakly) increases P(observed)
        - "negative": Treatment (weakly) decreases P(observed)
    n_bootstrap : int, default=1000
        Number of bootstrap replications for CI.
    alpha : float, default=0.05
        Significance level for CI (yields (1-α)% CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    LeeBoundsResult
        Bounds with bootstrap CI and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> # Treatment increases observation probability (positive monotonicity)
    >>> p_observed = 0.7 + 0.2 * treatment
    >>> observed = np.random.binomial(1, p_observed)
    >>> outcome = 2.0 * treatment + np.random.randn(n)
    >>> result = lee_bounds(outcome, treatment, observed)
    >>> print(f"Bounds: [{result['bounds_lower']:.2f}, {result['bounds_upper']:.2f}]")

    Notes
    -----
    The bounds are sharp: under the maintained assumptions, no tighter
    bounds are possible without additional restrictions.

    Monotonicity is testable: if r₁ > r₀ but we assume negative monotonicity
    (or vice versa), the assumption is violated.
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)
    observed = np.asarray(observed)

    n = len(outcome)
    if len(treatment) != n or len(observed) != n:
        raise ValueError("All arrays must have the same length")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    if not np.isin(observed, [0, 1]).all():
        raise ValueError("Observed indicator must be binary (0 or 1)")

    if monotonicity not in ("positive", "negative"):
        raise ValueError(f"monotonicity must be 'positive' or 'negative', got '{monotonicity}'")

    # Set random state
    rng = np.random.default_rng(random_state)

    # Compute attrition rates
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0 or n_control == 0:
        raise ValueError("Need both treated and control observations")

    n_treated_observed = np.sum((treatment == 1) & (observed == 1))
    n_control_observed = np.sum((treatment == 0) & (observed == 1))

    if n_treated_observed == 0 or n_control_observed == 0:
        raise ValueError("Need observed outcomes in both treatment groups")

    # Observation rates (1 - attrition)
    obs_rate_treated = n_treated_observed / n_treated
    obs_rate_control = n_control_observed / n_control

    attrition_treated = 1 - obs_rate_treated
    attrition_control = 1 - obs_rate_control

    # Check monotonicity assumption
    if monotonicity == "positive" and obs_rate_treated < obs_rate_control:
        warnings.warn(
            f"Monotonicity violation: positive monotonicity assumed but "
            f"treatment decreases observation rate ({obs_rate_treated:.2%} < {obs_rate_control:.2%}). "
            f"Consider using monotonicity='negative'.",
            UserWarning,
        )
    elif monotonicity == "negative" and obs_rate_treated > obs_rate_control:
        warnings.warn(
            f"Monotonicity violation: negative monotonicity assumed but "
            f"treatment increases observation rate ({obs_rate_treated:.2%} > {obs_rate_control:.2%}). "
            f"Consider using monotonicity='positive'.",
            UserWarning,
        )

    # Compute bounds
    bounds_lower, bounds_upper, n_trimmed = _compute_lee_bounds(
        outcome, treatment, observed, monotonicity
    )

    # Bootstrap CI
    bootstrap_lowers = []
    bootstrap_uppers = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_outcome = outcome[indices]
        boot_treatment = treatment[indices]
        boot_observed = observed[indices]

        try:
            boot_lower, boot_upper, _ = _compute_lee_bounds(
                boot_outcome, boot_treatment, boot_observed, monotonicity
            )
            bootstrap_lowers.append(boot_lower)
            bootstrap_uppers.append(boot_upper)
        except (ValueError, ZeroDivisionError):
            # Skip failed bootstrap samples
            continue

    if len(bootstrap_lowers) < n_bootstrap * 0.5:
        warnings.warn(
            f"Only {len(bootstrap_lowers)}/{n_bootstrap} bootstrap samples succeeded",
            UserWarning,
        )

    # Confidence interval using percentile method
    if len(bootstrap_lowers) > 0:
        ci_lower = float(np.percentile(bootstrap_lowers, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_uppers, 100 * (1 - alpha / 2)))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10

    # Determine trimmed group
    if obs_rate_treated > obs_rate_control:
        trimmed_group = "treated"
    elif obs_rate_control > obs_rate_treated:
        trimmed_group = "control"
    else:
        trimmed_group = "none"

    # Trimming proportion
    if obs_rate_treated != obs_rate_control:
        p_trim = abs(obs_rate_treated - obs_rate_control) / max(obs_rate_treated, obs_rate_control)
    else:
        p_trim = 0.0

    interpretation = (
        f"Lee bounds under {monotonicity} monotonicity: "
        f"[{bounds_lower:.3f}, {bounds_upper:.3f}]. "
        f"Attrition: treated={attrition_treated:.1%}, control={attrition_control:.1%}. "
        f"Trimmed {n_trimmed} from {trimmed_group} group."
    )

    return LeeBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        point_identified=point_identified,
        trimming_proportion=p_trim,
        trimmed_group=trimmed_group,
        attrition_treated=attrition_treated,
        attrition_control=attrition_control,
        n_treated_observed=int(n_treated_observed),
        n_control_observed=int(n_control_observed),
        n_trimmed=int(n_trimmed),
        monotonicity_assumption=monotonicity,
        interpretation=interpretation,
    )


def _compute_lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    monotonicity: str,
) -> Tuple[float, float, int]:
    """
    Compute Lee bounds for a single sample.

    Returns
    -------
    tuple
        (lower_bound, upper_bound, n_trimmed)
    """
    # Get observed outcomes by group
    treated_mask = (treatment == 1) & (observed == 1)
    control_mask = (treatment == 0) & (observed == 1)

    y_treated = outcome[treated_mask]
    y_control = outcome[control_mask]

    n_treated_obs = len(y_treated)
    n_control_obs = len(y_control)

    if n_treated_obs == 0 or n_control_obs == 0:
        raise ValueError("Need observed outcomes in both groups")

    # Observation rates
    n_treated_total = np.sum(treatment == 1)
    n_control_total = np.sum(treatment == 0)

    obs_rate_treated = n_treated_obs / n_treated_total
    obs_rate_control = n_control_obs / n_control_total

    # If rates are equal, no trimming needed → point identified
    if np.isclose(obs_rate_treated, obs_rate_control, rtol=1e-10):
        mean_treated = np.mean(y_treated)
        mean_control = np.mean(y_control)
        ate = mean_treated - mean_control
        return ate, ate, 0

    # Determine which group to trim
    if monotonicity == "positive":
        # Treatment increases observation → treated group has "extra" observations
        # Trim treated group if obs_rate_treated > obs_rate_control
        if obs_rate_treated > obs_rate_control:
            trim_treated = True
            p_trim = (obs_rate_treated - obs_rate_control) / obs_rate_treated
        else:
            # Monotonicity may be violated, but proceed with control trimming
            trim_treated = False
            p_trim = (obs_rate_control - obs_rate_treated) / obs_rate_control
    else:  # negative monotonicity
        # Treatment decreases observation → control group has "extra" observations
        if obs_rate_control > obs_rate_treated:
            trim_treated = False
            p_trim = (obs_rate_control - obs_rate_treated) / obs_rate_control
        else:
            trim_treated = True
            p_trim = (obs_rate_treated - obs_rate_control) / obs_rate_treated

    # Number to trim
    if trim_treated:
        n_trim = int(np.floor(p_trim * n_treated_obs))
        y_to_trim = y_treated
        y_other = y_control
    else:
        n_trim = int(np.floor(p_trim * n_control_obs))
        y_to_trim = y_control
        y_other = y_treated

    if n_trim >= len(y_to_trim):
        n_trim = len(y_to_trim) - 1  # Keep at least one observation

    if n_trim <= 0:
        # No trimming needed
        mean_treated = np.mean(y_treated)
        mean_control = np.mean(y_control)
        ate = mean_treated - mean_control
        return ate, ate, 0

    # Sort for trimming
    y_sorted = np.sort(y_to_trim)

    # Upper bound: trim from bottom (keep high values)
    y_trimmed_upper = y_sorted[n_trim:]
    mean_trimmed_upper = np.mean(y_trimmed_upper)

    # Lower bound: trim from top (keep low values)
    y_trimmed_lower = y_sorted[:-n_trim] if n_trim > 0 else y_sorted
    mean_trimmed_lower = np.mean(y_trimmed_lower)

    mean_other = np.mean(y_other)

    if trim_treated:
        # Trimming treated group
        upper_bound = mean_trimmed_upper - mean_other
        lower_bound = mean_trimmed_lower - mean_other
    else:
        # Trimming control group
        upper_bound = np.mean(y_treated) - mean_trimmed_lower
        lower_bound = np.mean(y_treated) - mean_trimmed_upper

    return float(lower_bound), float(upper_bound), n_trim


def lee_bounds_tightened(
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    covariates: np.ndarray,
    monotonicity: Literal["positive", "negative"] = "positive",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> LeeBoundsResult:
    """
    Compute tightened Lee bounds using covariates.

    Covariates can tighten bounds by allowing trimming to be done
    conditional on X, reducing the variance from trimming.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment indicator (n,).
    observed : np.ndarray
        Observation indicator (n,).
    covariates : np.ndarray
        Covariates for tightening (n, k).
    monotonicity : {"positive", "negative"}, default="positive"
        Direction of monotonicity.
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    alpha : float, default=0.05
        Significance level.
    random_state : int, optional
        Random seed.

    Returns
    -------
    LeeBoundsResult
        Tightened bounds.

    Notes
    -----
    This implements a simplified version of covariate tightening.
    For full efficiency, use the conditional trimming approach from
    Lee (2009) Section 4.
    """
    # For now, implement basic version (stratified by covariate quantiles)
    # Full implementation would require conditional density estimation

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcome)
    if covariates.shape[0] != n:
        raise ValueError("Covariates must have same number of rows as outcome")

    # Create strata based on covariate quantiles
    # Use first covariate for stratification
    x = covariates[:, 0]
    n_strata = min(5, n // 50)  # At least 50 obs per stratum

    if n_strata < 2:
        # Not enough data for stratification, use basic Lee bounds
        return lee_bounds(
            outcome, treatment, observed, monotonicity, n_bootstrap, alpha, random_state
        )

    quantiles = np.linspace(0, 100, n_strata + 1)
    thresholds = np.percentile(x, quantiles)

    # Compute bounds within each stratum
    stratum_bounds = []
    stratum_weights = []

    for i in range(n_strata):
        if i < n_strata - 1:
            mask = (x >= thresholds[i]) & (x < thresholds[i + 1])
        else:
            mask = (x >= thresholds[i]) & (x <= thresholds[i + 1])

        if mask.sum() < 10:
            continue

        try:
            stratum_result = lee_bounds(
                outcome[mask],
                treatment[mask],
                observed[mask],
                monotonicity,
                n_bootstrap=0,  # Skip bootstrap for strata
                alpha=alpha,
                random_state=random_state,
            )
            stratum_bounds.append((stratum_result["bounds_lower"], stratum_result["bounds_upper"]))
            stratum_weights.append(mask.sum())
        except ValueError:
            continue

    if len(stratum_bounds) == 0:
        # Stratification failed, use basic bounds
        return lee_bounds(
            outcome, treatment, observed, monotonicity, n_bootstrap, alpha, random_state
        )

    # Aggregate bounds (weighted average)
    weights = np.array(stratum_weights) / sum(stratum_weights)
    bounds_lower = sum(w * b[0] for w, b in zip(weights, stratum_bounds))
    bounds_upper = sum(w * b[1] for w, b in zip(weights, stratum_bounds))

    # Bootstrap for CI
    rng = np.random.default_rng(random_state)
    bootstrap_lowers = []
    bootstrap_uppers = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)

        try:
            boot_result = lee_bounds(
                outcome[indices],
                treatment[indices],
                observed[indices],
                monotonicity,
                n_bootstrap=0,
                random_state=None,
            )
            bootstrap_lowers.append(boot_result["bounds_lower"])
            bootstrap_uppers.append(boot_result["bounds_upper"])
        except ValueError:
            continue

    if len(bootstrap_lowers) > 0:
        ci_lower = float(np.percentile(bootstrap_lowers, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_uppers, 100 * (1 - alpha / 2)))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    # Compute attrition rates
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)
    attrition_treated = 1 - np.sum((treatment == 1) & (observed == 1)) / n_treated
    attrition_control = 1 - np.sum((treatment == 0) & (observed == 1)) / n_control

    return LeeBoundsResult(
        bounds_lower=float(bounds_lower),
        bounds_upper=float(bounds_upper),
        bounds_width=float(bounds_upper - bounds_lower),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        point_identified=abs(bounds_upper - bounds_lower) < 1e-10,
        trimming_proportion=0.0,  # Not directly applicable
        trimmed_group="stratified",
        attrition_treated=attrition_treated,
        attrition_control=attrition_control,
        n_treated_observed=int(np.sum((treatment == 1) & (observed == 1))),
        n_control_observed=int(np.sum((treatment == 0) & (observed == 1))),
        n_trimmed=0,
        monotonicity_assumption=monotonicity,
        interpretation=f"Tightened Lee bounds using {n_strata} covariate strata.",
    )


def check_monotonicity(
    treatment: np.ndarray,
    observed: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Check whether monotonicity assumption is plausible.

    Computes observation rates by treatment status and tests
    whether the difference is statistically significant.

    Parameters
    ----------
    treatment : np.ndarray
        Treatment indicator (n,).
    observed : np.ndarray
        Observation indicator (n,).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Test results including suggested monotonicity direction.
    """
    treatment = np.asarray(treatment)
    observed = np.asarray(observed)

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    obs_rate_treated = np.mean(observed[treatment == 1])
    obs_rate_control = np.mean(observed[treatment == 0])

    diff = obs_rate_treated - obs_rate_control

    # Standard error for difference in proportions
    se = np.sqrt(
        obs_rate_treated * (1 - obs_rate_treated) / n_treated
        + obs_rate_control * (1 - obs_rate_control) / n_control
    )

    z_stat = diff / se if se > 0 else 0
    from scipy.stats import norm

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    if diff > 0:
        suggested = "positive"
        interpretation = "Treatment increases observation probability"
    elif diff < 0:
        suggested = "negative"
        interpretation = "Treatment decreases observation probability"
    else:
        suggested = "either"
        interpretation = "No differential attrition"

    return {
        "obs_rate_treated": obs_rate_treated,
        "obs_rate_control": obs_rate_control,
        "difference": diff,
        "se": se,
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "suggested_monotonicity": suggested,
        "interpretation": interpretation,
    }
