"""
High-level PSM estimation function.

Integrates PropensityScoreEstimator + NearestNeighborMatcher + AbadieImbensVariance
into single psm_ate() function for easy use.

Author: Brandon Behring
Date: 2025-11-21
"""

from typing import Dict, Any, Optional
import warnings
import numpy as np
from scipy import stats

from .propensity import PropensityScoreEstimator
from .matching import NearestNeighborMatcher
from .variance import abadie_imbens_variance
from .balance import check_covariate_balance, balance_summary


def psm_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    M: int = 1,
    with_replacement: bool = False,
    caliper: float = np.inf,
    alpha: float = 0.05,
    variance_method: str = "abadie_imbens",
) -> Dict[str, Any]:
    """
    Estimate average treatment effect using propensity score matching.

    Pipeline:
    1. Estimate propensity scores P(T=1|X) via logistic regression
    2. Check common support (overlap in propensity distributions)
    3. Match treated units to M nearest controls by propensity score
    4. Compute ATE from matched sample
    5. Compute standard errors via Abadie-Imbens analytic variance
    6. Construct confidence intervals

    Args:
        outcomes: Observed outcomes Y (n,)
        treatment: Binary treatment indicator (n,) with {0, 1} or {False, True}
        covariates: Covariate matrix X (n, p) for propensity estimation
        M: Number of matches per treated unit (default: 1)
        with_replacement: Allow reusing control units (default: False)
        caliper: Maximum propensity distance allowed (default: np.inf, no restriction)
        alpha: Significance level for confidence intervals (default: 0.05)
        variance_method: Variance estimator ("abadie_imbens" or "paired", default: "abadie_imbens")

    Returns:
        Dictionary with:
        - estimate: ATE point estimate
        - se: Standard error
        - ci_lower: Lower confidence interval bound
        - ci_upper: Upper confidence interval bound
        - n_treated: Number of treated units
        - n_control: Number of control units
        - n_matched: Number of treated units successfully matched
        - propensity_scores: Estimated propensity scores for all units
        - matches: Match lists (matches[i] = control indices for treated unit i)
        - balance_metrics: Dictionary with balance diagnostics (TODO: Session 3)
        - convergence_status: Propensity estimation convergence status

    Raises:
        ValueError: If inputs invalid, no common support, or matching fails

    Example:
        >>> result = psm_ate(
        ...     outcomes=outcomes,
        ...     treatment=treatment,
        ...     covariates=covariates,
        ...     M=1,
        ...     with_replacement=False,
        ...     caliper=0.25,
        ... )
        >>> print(f"ATE = {result['estimate']:.3f} ± {result['se']:.3f}")
        >>> print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")

    Notes:
        - Uses sklearn LogisticRegression for propensity estimation
        - Greedy nearest neighbor matching algorithm
        - Abadie-Imbens (2006) variance accounts for matching uncertainty
        - Bootstrap FAILS for with_replacement (use abadie_imbens)
        - Balance diagnostics not yet implemented (Session 3)
    """
    # ====================================================================
    # Input Validation
    # ====================================================================

    outcomes = np.asarray(outcomes)
    treatment = np.asarray(treatment)
    covariates = np.asarray(covariates)

    n = len(outcomes)

    if len(treatment) != n or len(covariates) != n:
        raise ValueError(
            f"CRITICAL ERROR: Mismatched lengths.\n"
            f"Function: psm_ate\n"
            f"outcomes: {len(outcomes)}, treatment: {len(treatment)}, covariates: {len(covariates)}\n"
            f"All inputs must have same length."
        )

    if covariates.ndim != 2:
        raise ValueError(
            f"CRITICAL ERROR: Invalid covariate shape.\n"
            f"Function: psm_ate\n"
            f"covariates must be 2D array (n, p), got shape {covariates.shape}\n"
            f"Use covariates.reshape(-1, 1) for single covariate."
        )

    if not (0 < alpha < 1):
        raise ValueError(
            f"CRITICAL ERROR: Invalid significance level.\n"
            f"Function: psm_ate\n"
            f"alpha must be in (0, 1), got alpha = {alpha}\n"
            f"Common values: 0.01, 0.05, 0.10"
        )

    if variance_method not in ["abadie_imbens", "paired"]:
        raise ValueError(
            f"CRITICAL ERROR: Invalid variance_method.\n"
            f"Function: psm_ate\n"
            f"Must be 'abadie_imbens' or 'paired', got '{variance_method}'\n"
            f"Recommended: 'abadie_imbens' (accounts for matching uncertainty)"
        )

    # ====================================================================
    # Step 1: Estimate Propensity Scores
    # ====================================================================

    estimator = PropensityScoreEstimator()
    propensity_result = estimator.fit(treatment, covariates)

    propensity = propensity_result.propensity_scores

    # Check convergence
    if not propensity_result.converged:
        # Warning already issued by PropensityScoreEstimator, continue with results
        pass

    # ====================================================================
    # Step 2: Check Common Support (Diagnostic Only)
    # ====================================================================

    # Note: We don't fail here - limited overlap is just a warning
    # The real failure comes if NO matches found (Step 3)
    if not propensity_result.has_common_support:
        warnings.warn(
            f"Limited common support detected.\n"
            f"Support region: {propensity_result.support_region}\n"
            f"{propensity_result.n_outside_support} units outside support.\n"
            f"Matching may fail or produce unreliable estimates.\n"
            f"Consider:\n"
            f"  - Relaxing caliper restriction\n"
            f"  - Adding more data\n"
            f"  - Checking for perfect separation in covariates",
            category=RuntimeWarning,
        )

    # ====================================================================
    # Step 3: Nearest Neighbor Matching
    # ====================================================================

    matcher = NearestNeighborMatcher(
        M=M, with_replacement=with_replacement, caliper=caliper
    )
    matching_result = matcher.match(propensity, treatment)

    if matching_result.n_matched == 0:
        raise ValueError(
            f"CRITICAL ERROR: No matches found.\n"
            f"Function: psm_ate\n"
            f"No treated units matched within caliper = {caliper}.\n"
            f"Solutions:\n"
            f"  - Increase caliper (try 0.25 or 0.5)\n"
            f"  - Use caliper = np.inf for no restriction\n"
            f"  - Check common support region: {propensity_result.support_region}"
        )

    # ====================================================================
    # Step 4: Compute ATE
    # ====================================================================

    ate, treated_outcomes, control_outcomes = matcher.compute_ate_from_matches(
        outcomes, treatment, matching_result.matches
    )

    # ====================================================================
    # Step 5: Compute Variance
    # ====================================================================

    if variance_method == "abadie_imbens":
        variance, se = abadie_imbens_variance(
            outcomes, treatment, matching_result.matches, M=M
        )
    elif variance_method == "paired":
        # Only valid for 1:1 matching
        if M != 1:
            raise ValueError(
                f"CRITICAL ERROR: Invalid variance_method for M:1 matching.\n"
                f"Function: psm_ate\n"
                f"'paired' variance only valid for M=1 (1:1 matching).\n"
                f"Got M={M}. Use variance_method='abadie_imbens' for M:1 matching."
            )
        from .variance import compute_matched_pairs_variance

        variance, se = compute_matched_pairs_variance(
            outcomes, treatment, matching_result.matches
        )
    else:
        raise ValueError(f"Unknown variance_method: {variance_method}")

    # ====================================================================
    # Step 6: Confidence Intervals
    # ====================================================================

    # Normal approximation (large sample)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * se
    ci_upper = ate + z_crit * se

    # ====================================================================
    # Step 7: Balance Diagnostics
    # ====================================================================

    # Create matched pairs indices for balance checking
    matched_pairs = []
    indices_treated = np.where(treatment)[0]
    for i, match_list in enumerate(matching_result.matches):
        if len(match_list) > 0:
            treated_idx = indices_treated[i]
            # For M:1 matching, use first match for balance check (representative)
            control_idx = match_list[0]
            matched_pairs.append((treated_idx, control_idx))

    # Compute balance metrics
    balanced, smd_after, vr_after, smd_before, vr_before = check_covariate_balance(
        covariates, treatment, matched_pairs, threshold=0.1
    )

    # Summary statistics
    summary = balance_summary(smd_after, vr_after, smd_before, vr_before)

    balance_metrics = {
        "balanced": balanced,
        "smd_after": smd_after,
        "vr_after": vr_after,
        "smd_before": smd_before,
        "vr_before": vr_before,
        "summary": summary,
    }

    # ====================================================================
    # Return Result
    # ====================================================================

    n_treated = np.sum(treatment)
    n_control = n - n_treated

    return {
        "estimate": float(ate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "n_matched": int(matching_result.n_matched),
        "propensity_scores": propensity,
        "matches": matching_result.matches,
        "balance_metrics": balance_metrics,
        "convergence_status": {
            "propensity_converged": propensity_result.converged,
            "has_common_support": propensity_result.has_common_support,
            "support_region": propensity_result.support_region,
            "n_outside_support": propensity_result.n_outside_support,
        },
    }
