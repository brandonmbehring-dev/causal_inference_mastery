"""
Policy-relevant treatment effects from MTE.

Derives population parameters (ATE, ATT, ATU, PRTE) by integrating
the MTE curve with appropriate weights.

References:
- Heckman & Vytlacil (2005): Structural Equations, Treatment Effects
- Carneiro, Heckman & Vytlacil (2011): Estimating Marginal Returns to Education
"""

from typing import Optional, Union, Callable
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid

from .types import MTEResult, PolicyResult


def ate_from_mte(mte_result: MTEResult, n_bootstrap: int = 0) -> PolicyResult:
    """
    Compute Average Treatment Effect from MTE.

    ATE = ∫ MTE(u) du over the support [p_min, p_max]

    For full population ATE, assumes MTE is constant outside observed support.
    This is a strong assumption - interpret with caution.

    Parameters
    ----------
    mte_result : MTEResult
        Result from local_iv() or polynomial_mte()
    n_bootstrap : int, default=0
        If > 0, compute bootstrap SE by resampling MTE curve with SE

    Returns
    -------
    PolicyResult
        ATE estimate with SE and CI

    Notes
    -----
    - ATE uses uniform weights: ω(u) = 1
    - Requires MTE to be identified over sufficient support
    - Extrapolation beyond support is unreliable
    """
    mte_grid = mte_result["mte_grid"]
    u_grid = mte_result["u_grid"]
    se_grid = mte_result["se_grid"]
    p_min, p_max = mte_result["propensity_support"]

    # Remove NaN values
    valid = ~np.isnan(mte_grid)
    if valid.sum() < 3:
        return _empty_policy_result("ate", mte_result["n_obs"])

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Integrate MTE over support: ATE = ∫ MTE(u) du / (p_max - p_min)
    # Normalize by support width for average
    integral = trapezoid(mte_valid, u_valid)
    support_width = u_valid[-1] - u_valid[0]

    if support_width < 1e-10:
        return _empty_policy_result("ate", mte_result["n_obs"])

    ate = integral / support_width

    # Standard error via delta method or bootstrap
    if n_bootstrap > 0 and np.any(se_valid > 0):
        se, ci_lower, ci_upper = _bootstrap_integral(
            mte_valid, u_valid, se_valid, n_bootstrap, normalize=support_width
        )
    else:
        # Approximate SE using midpoint SE
        se = np.nanmedian(se_valid)
        z = stats.norm.ppf(0.975)
        ci_lower = ate - z * se
        ci_upper = ate + z * se

    return PolicyResult(
        estimate=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        parameter="ate",
        weights_used="uniform (∫ MTE(u) du)",
        n_obs=mte_result["n_obs"],
    )


def att_from_mte(
    mte_result: MTEResult,
    propensity: Optional[np.ndarray] = None,
    treatment: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
) -> PolicyResult:
    """
    Compute Average Treatment Effect on the Treated from MTE.

    ATT = ∫ MTE(u) · ω_ATT(u) du

    where ω_ATT(u) = P(U ≤ u | D=1) / E[D]

    For treated individuals, weight by their distribution of u.

    Parameters
    ----------
    mte_result : MTEResult
        Result from local_iv() or polynomial_mte()
    propensity : np.ndarray, optional
        Estimated propensity scores. If provided, computes weights.
    treatment : np.ndarray, optional
        Treatment indicator. Required if propensity provided.
    n_bootstrap : int, default=0
        Bootstrap replications for SE

    Returns
    -------
    PolicyResult
        ATT estimate

    Notes
    -----
    - Without propensity data, approximates ATT using linear weights
    - ATT weights lower u values more heavily (treated have lower U)
    """
    mte_grid = mte_result["mte_grid"]
    u_grid = mte_result["u_grid"]
    se_grid = mte_result["se_grid"]
    p_min, p_max = mte_result["propensity_support"]

    valid = ~np.isnan(mte_grid)
    if valid.sum() < 3:
        return _empty_policy_result("att", mte_result["n_obs"])

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Compute ATT weights
    if propensity is not None and treatment is not None:
        # Empirical weights from data
        weights = _compute_att_weights_empirical(
            u_valid, propensity[treatment == 1]
        )
    else:
        # Theoretical weights: ω_ATT(u) ∝ (P - u) for u < P
        # Approximation: linearly decreasing weights
        weights = _compute_att_weights_theoretical(u_valid, p_min, p_max)

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted integral: ATT = Σ MTE(u) · ω(u) · Δu
    du = np.diff(u_valid, prepend=u_valid[0])
    du[0] = du[1] if len(du) > 1 else 1.0
    att = np.sum(mte_valid * weights)

    # Standard error
    if n_bootstrap > 0 and np.any(se_valid > 0):
        se, ci_lower, ci_upper = _bootstrap_weighted_integral(
            mte_valid, se_valid, weights, n_bootstrap
        )
    else:
        se = np.sqrt(np.sum((weights * se_valid) ** 2))
        z = stats.norm.ppf(0.975)
        ci_lower = att - z * se
        ci_upper = att + z * se

    return PolicyResult(
        estimate=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        parameter="att",
        weights_used="ATT weights (∫ MTE(u) · P(U≤u|D=1) du)",
        n_obs=mte_result["n_obs"],
    )


def atu_from_mte(
    mte_result: MTEResult,
    propensity: Optional[np.ndarray] = None,
    treatment: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
) -> PolicyResult:
    """
    Compute Average Treatment Effect on the Untreated from MTE.

    ATU = ∫ MTE(u) · ω_ATU(u) du

    where ω_ATU(u) = P(U > u | D=0) / E[1-D]

    Parameters
    ----------
    mte_result : MTEResult
        Result from local_iv() or polynomial_mte()
    propensity : np.ndarray, optional
        Estimated propensity scores
    treatment : np.ndarray, optional
        Treatment indicator
    n_bootstrap : int, default=0
        Bootstrap replications for SE

    Returns
    -------
    PolicyResult
        ATU estimate

    Notes
    -----
    - ATU weights higher u values more heavily (untreated have higher U)
    """
    mte_grid = mte_result["mte_grid"]
    u_grid = mte_result["u_grid"]
    se_grid = mte_result["se_grid"]
    p_min, p_max = mte_result["propensity_support"]

    valid = ~np.isnan(mte_grid)
    if valid.sum() < 3:
        return _empty_policy_result("atu", mte_result["n_obs"])

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Compute ATU weights
    if propensity is not None and treatment is not None:
        weights = _compute_atu_weights_empirical(
            u_valid, propensity[treatment == 0]
        )
    else:
        # Theoretical: linearly increasing weights
        weights = _compute_atu_weights_theoretical(u_valid, p_min, p_max)

    weights = weights / weights.sum()

    atu = np.sum(mte_valid * weights)

    if n_bootstrap > 0 and np.any(se_valid > 0):
        se, ci_lower, ci_upper = _bootstrap_weighted_integral(
            mte_valid, se_valid, weights, n_bootstrap
        )
    else:
        se = np.sqrt(np.sum((weights * se_valid) ** 2))
        z = stats.norm.ppf(0.975)
        ci_lower = atu - z * se
        ci_upper = atu + z * se

    return PolicyResult(
        estimate=atu,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        parameter="atu",
        weights_used="ATU weights (∫ MTE(u) · P(U>u|D=0) du)",
        n_obs=mte_result["n_obs"],
    )


def prte(
    mte_result: MTEResult,
    policy_weights: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    n_bootstrap: int = 0,
) -> PolicyResult:
    """
    Compute Policy-Relevant Treatment Effect.

    PRTE = ∫ MTE(u) · ω_policy(u) du

    Evaluates treatment effect under a specific policy that changes
    the distribution of who is treated.

    Parameters
    ----------
    mte_result : MTEResult
        Result from local_iv() or polynomial_mte()
    policy_weights : array or callable
        Either:
        - np.ndarray: weights at each grid point
        - Callable: function that takes u_grid and returns weights
    n_bootstrap : int, default=0
        Bootstrap replications for SE

    Returns
    -------
    PolicyResult
        PRTE estimate

    Examples
    --------
    >>> # Uniform expansion of treatment by 10%
    >>> def uniform_expansion(u):
    ...     return np.ones_like(u) * 0.1
    >>> prte_result = prte(mte_result, uniform_expansion)

    >>> # Target bottom 20% of u distribution
    >>> def target_low_u(u):
    ...     return np.where(u < 0.2, 1.0, 0.0)
    >>> prte_result = prte(mte_result, target_low_u)

    Notes
    -----
    - PRTE answers: "What would be the effect of this policy change?"
    - Weights should integrate to treatment probability change
    """
    mte_grid = mte_result["mte_grid"]
    u_grid = mte_result["u_grid"]
    se_grid = mte_result["se_grid"]

    valid = ~np.isnan(mte_grid)
    if valid.sum() < 3:
        return _empty_policy_result("prte", mte_result["n_obs"])

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Get weights
    if callable(policy_weights):
        weights = policy_weights(u_valid)
    else:
        if len(policy_weights) != len(u_grid):
            raise ValueError(
                f"Policy weights length ({len(policy_weights)}) != "
                f"grid length ({len(u_grid)})"
            )
        weights = policy_weights[valid]

    weights = np.asarray(weights, dtype=float)

    # Normalize if weights sum to nonzero
    weight_sum = weights.sum()
    if abs(weight_sum) > 1e-10:
        weights = weights / weight_sum

    prte_estimate = np.sum(mte_valid * weights)

    if n_bootstrap > 0 and np.any(se_valid > 0):
        se, ci_lower, ci_upper = _bootstrap_weighted_integral(
            mte_valid, se_valid, weights, n_bootstrap
        )
    else:
        se = np.sqrt(np.sum((weights * se_valid) ** 2))
        z = stats.norm.ppf(0.975)
        ci_lower = prte_estimate - z * se
        ci_upper = prte_estimate + z * se

    return PolicyResult(
        estimate=prte_estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        parameter="prte",
        weights_used="custom policy weights",
        n_obs=mte_result["n_obs"],
    )


def late_from_mte(
    mte_result: MTEResult,
    p_old: float,
    p_new: float,
    n_bootstrap: int = 0,
) -> PolicyResult:
    """
    Compute LATE for a specific instrument shift from MTE.

    LATE(p_old → p_new) = ∫_{p_old}^{p_new} MTE(u) du / (p_new - p_old)

    This is the average treatment effect for compliers whose treatment
    status changes when the instrument shifts.

    Parameters
    ----------
    mte_result : MTEResult
        Result from local_iv() or polynomial_mte()
    p_old : float
        Old propensity score value
    p_new : float
        New propensity score value
    n_bootstrap : int, default=0
        Bootstrap replications for SE

    Returns
    -------
    PolicyResult
        LATE estimate for compliers in [p_old, p_new]

    Notes
    -----
    - This recovers standard IV LATE as integral of MTE
    - Shows how LATE varies with instrument strength
    """
    if p_new < p_old:
        p_old, p_new = p_new, p_old

    mte_grid = mte_result["mte_grid"]
    u_grid = mte_result["u_grid"]
    se_grid = mte_result["se_grid"]

    # Find grid points in [p_old, p_new]
    mask = (u_grid >= p_old) & (u_grid <= p_new)
    if mask.sum() < 2:
        return _empty_policy_result("late", mte_result["n_obs"])

    mte_range = mte_grid[mask]
    u_range = u_grid[mask]
    se_range = se_grid[mask]

    valid = ~np.isnan(mte_range)
    if valid.sum() < 2:
        return _empty_policy_result("late", mte_result["n_obs"])

    # Integrate MTE over complier range
    integral = trapezoid(mte_range[valid], u_range[valid])
    range_width = u_range[valid][-1] - u_range[valid][0]

    if range_width < 1e-10:
        return _empty_policy_result("late", mte_result["n_obs"])

    late_estimate = integral / range_width

    if n_bootstrap > 0 and np.any(se_range[valid] > 0):
        se, ci_lower, ci_upper = _bootstrap_integral(
            mte_range[valid], u_range[valid], se_range[valid],
            n_bootstrap, normalize=range_width
        )
    else:
        se = np.nanmedian(se_range[valid])
        z = stats.norm.ppf(0.975)
        ci_lower = late_estimate - z * se
        ci_upper = late_estimate + z * se

    return PolicyResult(
        estimate=late_estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        parameter="late",
        weights_used=f"LATE weights over [{p_old:.2f}, {p_new:.2f}]",
        n_obs=mte_result["n_obs"],
    )


# --- Helper Functions ---


def _compute_att_weights_theoretical(
    u_grid: np.ndarray, p_min: float, p_max: float
) -> np.ndarray:
    """
    Compute theoretical ATT weights.

    ATT weights ω(u) ∝ P(D=1 | U≤u) ∝ (p_mean - u) for u < p_mean
    """
    p_mean = (p_min + p_max) / 2

    # Linearly decreasing weights (treated have lower U)
    weights = np.maximum(0, p_mean - u_grid + 0.5 * (p_max - p_min))
    weights = weights / weights.max() if weights.max() > 0 else np.ones_like(u_grid)

    return weights


def _compute_att_weights_empirical(
    u_grid: np.ndarray, propensity_treated: np.ndarray
) -> np.ndarray:
    """Compute empirical ATT weights from treated propensity distribution."""
    # Histogram of propensity scores for treated
    weights = np.zeros(len(u_grid))

    for i, u in enumerate(u_grid):
        # Weight = proportion of treated with P > u
        weights[i] = np.mean(propensity_treated >= u)

    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(u_grid) / len(u_grid)

    return weights


def _compute_atu_weights_theoretical(
    u_grid: np.ndarray, p_min: float, p_max: float
) -> np.ndarray:
    """
    Compute theoretical ATU weights.

    ATU weights ω(u) ∝ P(D=0 | U>u) ∝ (u - p_mean) for u > p_mean
    """
    p_mean = (p_min + p_max) / 2

    # Linearly increasing weights (untreated have higher U)
    weights = np.maximum(0, u_grid - p_mean + 0.5 * (p_max - p_min))
    weights = weights / weights.max() if weights.max() > 0 else np.ones_like(u_grid)

    return weights


def _compute_atu_weights_empirical(
    u_grid: np.ndarray, propensity_untreated: np.ndarray
) -> np.ndarray:
    """Compute empirical ATU weights from untreated propensity distribution."""
    weights = np.zeros(len(u_grid))

    for i, u in enumerate(u_grid):
        # Weight = proportion of untreated with P < u
        weights[i] = np.mean(propensity_untreated <= u)

    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(u_grid) / len(u_grid)

    return weights


def _bootstrap_integral(
    mte: np.ndarray,
    u: np.ndarray,
    se: np.ndarray,
    n_bootstrap: int,
    normalize: float = 1.0,
) -> tuple:
    """Bootstrap standard error for integral estimate."""
    rng = np.random.default_rng()
    boot_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Sample from MTE distribution at each point
        mte_boot = mte + rng.normal(0, se)
        integral = trapezoid(mte_boot, u) / normalize
        boot_estimates[b] = integral

    se_boot = np.std(boot_estimates, ddof=1)
    ci_lower = np.percentile(boot_estimates, 2.5)
    ci_upper = np.percentile(boot_estimates, 97.5)

    return se_boot, ci_lower, ci_upper


def _bootstrap_weighted_integral(
    mte: np.ndarray,
    se: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int,
) -> tuple:
    """Bootstrap standard error for weighted integral."""
    rng = np.random.default_rng()
    boot_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        mte_boot = mte + rng.normal(0, se)
        boot_estimates[b] = np.sum(mte_boot * weights)

    se_boot = np.std(boot_estimates, ddof=1)
    ci_lower = np.percentile(boot_estimates, 2.5)
    ci_upper = np.percentile(boot_estimates, 97.5)

    return se_boot, ci_lower, ci_upper


def _empty_policy_result(parameter: str, n_obs: int) -> PolicyResult:
    """Return empty result when computation fails."""
    return PolicyResult(
        estimate=np.nan,
        se=np.nan,
        ci_lower=np.nan,
        ci_upper=np.nan,
        parameter=parameter,
        weights_used="N/A (insufficient data)",
        n_obs=n_obs,
    )
