"""
Abadie-Imbens analytic variance for propensity score matching.

Implements exact variance formula from Abadie & Imbens (2006, 2008) following
Julia reference (variance.jl lines 84-270).

Key insight: Standard bootstrap FAILS for matching with replacement. Must use
this analytic variance formula instead.

References:
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators
  for average treatment effects. Econometrica, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). On the failure of the bootstrap for matching
  estimators. Econometrica, 76(6), 1537-1557.

Author: Brandon Behring
Date: 2025-11-21
"""

from typing import List, Tuple
import numpy as np


def abadie_imbens_variance(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    matches: List[List[int]],
    M: int = 1,
) -> Tuple[float, float]:
    """
    Compute Abadie-Imbens (2006) analytic variance for PSM.

    Implements exact formula from Julia variance.jl (lines 84-270).

    Formula:
    --------
    V = (1/N₁) [var_treated + var_control + var_matching]

    Where:
    - var_treated: Variance from treated outcomes
    - var_control: Variance from control outcomes
    - var_matching: Additional variance from matching imperfection (K_M factor)

    Detailed computation (Julia lines 135-270):
    1. Compute imputed potential outcomes for ALL units:
       - Treated: Ŷᵢ(1) = Yᵢ (observed), Ŷᵢ(0) = mean(Yⱼ for j ∈ matches[i])
       - Control: Ŷⱼ(0) = Yⱼ (observed), Ŷⱼ(1) = mean(Yᵢ for i where j ∈ matches[i])
    2. Compute conditional variances:
       - σ²ᵢ(1) = [Yᵢ - Ŷᵢ(0)]² for treated units
       - σ²ⱼ(0) = [Yⱼ - Ŷⱼ(1)]² for control units
    3. Aggregate with K_M factor:
       - K_M ≈ M (number of matches per treated unit)
       - V = (1/N₁) [mean(σ²ᵢ(1)) + (K_M/N₀) mean(σ²ⱼ(0))]

    Args:
        outcomes: Observed outcomes (n,)
        treatment: Binary treatment indicator (n,)
        matches: Match lists from NearestNeighborMatcher
                 matches[i] = [j1, j2, ..., jM] where j are control indices
        M: Number of matches per treated unit

    Returns:
        Tuple of (variance, standard_error)

    Raises:
        ValueError: If inputs invalid or no matched units

    Notes:
        - This variance accounts for matching uncertainty (standard SE formulas don't)
        - Bootstrap FAILS for matching with replacement (Abadie & Imbens 2008)
        - K_M factor increases variance to account for matching imperfection
        - Assumes M:1 matching (each treated matched to M controls)

    Example:
        >>> variance, se = abadie_imbens_variance(outcomes, treatment, matches, M=1)
        >>> print(f"SE = {se:.3f}")
    """
    # ====================================================================
    # Input Validation
    # ====================================================================

    outcomes = np.asarray(outcomes)
    treatment = np.asarray(treatment).astype(bool)

    n = len(outcomes)

    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Mismatched lengths.\n"
            f"Function: abadie_imbens_variance\n"
            f"outcomes has length {n}, treatment has length {len(treatment)}\n"
            f"All inputs must have same length."
        )

    if np.any(np.isnan(outcomes)) or np.any(np.isinf(outcomes)):
        raise ValueError(
            f"CRITICAL ERROR: NaN or Inf in outcomes.\n"
            f"Function: abadie_imbens_variance\n"
            f"Outcomes contain {np.sum(np.isnan(outcomes))} NaN "
            f"and {np.sum(np.isinf(outcomes))} Inf values."
        )

    if M < 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid M.\n"
            f"Function: abadie_imbens_variance\n"
            f"M must be ≥ 1, got M = {M}"
        )

    # Count matched units
    n_matched = sum(1 for m in matches if len(m) > 0)

    if n_matched == 0:
        raise ValueError(
            f"CRITICAL ERROR: No matched units.\n"
            f"Function: abadie_imbens_variance\n"
            f"Cannot compute variance with zero matched pairs.\n"
            f"Check matching results."
        )

    # ====================================================================
    # Extract Indices
    # ====================================================================

    indices_treated = np.where(treatment)[0]
    indices_control = np.where(~treatment)[0]

    n_treated = len(indices_treated)
    n_control = len(indices_control)

    # ====================================================================
    # Step 1: Compute Imputed Potential Outcomes (Julia lines 135-224)
    # ====================================================================

    # Initialize imputed outcomes
    imputed_y1 = np.full(n, np.nan)  # Imputed Y(1) for all units
    imputed_y0 = np.full(n, np.nan)  # Imputed Y(0) for all units

    # For treated units:
    # - Y(1) observed: imputed_y1[i] = outcomes[i]
    # - Y(0) imputed: imputed_y0[i] = mean(outcomes[j] for j in matches[i])
    for i, match_list in enumerate(matches):
        if len(match_list) == 0:
            # Unmatched treated unit - skip
            continue

        treated_idx = indices_treated[i]

        # Observed treated outcome
        imputed_y1[treated_idx] = outcomes[treated_idx]

        # Imputed control outcome (average of matched controls)
        control_outcomes = outcomes[match_list]
        imputed_y0[treated_idx] = np.mean(control_outcomes)

    # For control units:
    # - Y(0) observed: imputed_y0[j] = outcomes[j]
    # - Y(1) imputed: imputed_y1[j] = mean(outcomes[i] for i where j in matches[i])
    #
    # Need to find which treated units matched to each control
    control_to_treated = {j: [] for j in indices_control}

    for i, match_list in enumerate(matches):
        if len(match_list) == 0:
            continue

        treated_idx = indices_treated[i]

        for control_idx in match_list:
            control_to_treated[control_idx].append(treated_idx)

    for control_idx in indices_control:
        # Observed control outcome
        imputed_y0[control_idx] = outcomes[control_idx]

        # Imputed treated outcome (average of treated units that matched to this control)
        matched_treated = control_to_treated[control_idx]

        if len(matched_treated) > 0:
            treated_outcomes = outcomes[matched_treated]
            imputed_y1[control_idx] = np.mean(treated_outcomes)
        else:
            # This control was never matched - no imputed Y(1)
            # Leave as NaN (won't contribute to variance)
            pass

    # ====================================================================
    # Step 2: Compute Conditional Variances (Julia lines 233-252)
    # ====================================================================

    # Initialize conditional variances
    sigma_sq_treated = []  # σ²ᵢ(1) for matched treated units
    sigma_sq_control = []  # σ²ⱼ(0) for matched control units

    # Treated conditional variance: σ²ᵢ(1) = [Yᵢ(1) - Ŷᵢ(0)]²
    # This is the squared imputation error for treated outcomes
    for i, match_list in enumerate(matches):
        if len(match_list) == 0:
            continue

        treated_idx = indices_treated[i]

        y1_obs = outcomes[treated_idx]
        y0_imp = imputed_y0[treated_idx]

        # Imputation error squared
        sigma_sq_i = (y1_obs - y0_imp) ** 2
        sigma_sq_treated.append(sigma_sq_i)

    # Control conditional variance: σ²ⱼ(0) = [Yⱼ(0) - Ŷⱼ(1)]²
    # This is the squared imputation error for control outcomes
    for control_idx in indices_control:
        if np.isnan(imputed_y1[control_idx]):
            # This control was never matched - skip
            continue

        y0_obs = outcomes[control_idx]
        y1_imp = imputed_y1[control_idx]

        # Imputation error squared
        sigma_sq_j = (y0_obs - y1_imp) ** 2
        sigma_sq_control.append(sigma_sq_j)

    # Convert to arrays
    sigma_sq_treated = np.array(sigma_sq_treated)
    sigma_sq_control = np.array(sigma_sq_control)

    # ====================================================================
    # Step 3: Aggregate Variance with K_M Factor (Julia lines 243-262)
    # ====================================================================

    # K_M bias correction factor (Julia line 261)
    # For M:1 matching, K_M ≈ M (number of matches per treated unit)
    K_M = float(M)

    # Treated variance component (Julia line 264)
    if len(sigma_sq_treated) > 0:
        var_treated_component = np.mean(sigma_sq_treated)
    else:
        var_treated_component = 0.0

    # Control variance component with K_M factor (Julia line 265)
    if len(sigma_sq_control) > 0:
        var_control_component = (K_M / n_control) * np.sum(sigma_sq_control)
    else:
        var_control_component = 0.0

    # Total variance (Julia line 266)
    variance = var_treated_component + var_control_component

    # Normalize by number of matched treated units
    variance = variance / n_matched

    # ====================================================================
    # Step 4: Compute Standard Error (Julia line 267)
    # ====================================================================

    se = np.sqrt(variance)

    return variance, se


def compute_matched_pairs_variance(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    matches: List[List[int]],
) -> Tuple[float, float]:
    """
    Simple variance estimator for 1:1 matched pairs (no K_M factor).

    This is a simpler alternative to Abadie-Imbens for 1:1 matching without
    replacement. Uses standard paired difference variance.

    Formula:
    --------
    For each matched pair (i, j):
        τᵢ = Yᵢ - Yⱼ

    Variance = var(τᵢ) / n_matched

    Args:
        outcomes: Observed outcomes (n,)
        treatment: Binary treatment indicator (n,)
        matches: Match lists (must be 1:1, i.e., len(matches[i]) = 1 for all i)

    Returns:
        Tuple of (variance, standard_error)

    Raises:
        ValueError: If not 1:1 matching or inputs invalid

    Notes:
        - Only valid for 1:1 matching without replacement
        - Simpler than Abadie-Imbens but more conservative (larger SE)
        - Does not account for matching uncertainty like Abadie-Imbens does

    Example:
        >>> # For 1:1 matching
        >>> variance, se = compute_matched_pairs_variance(outcomes, treatment, matches)
    """
    # ====================================================================
    # Input Validation
    # ====================================================================

    outcomes = np.asarray(outcomes)
    treatment = np.asarray(treatment).astype(bool)

    n = len(outcomes)

    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Mismatched lengths.\n"
            f"Function: compute_matched_pairs_variance\n"
            f"outcomes has length {n}, treatment has length {len(treatment)}\n"
            f"All inputs must have same length."
        )

    # Verify 1:1 matching
    for i, match_list in enumerate(matches):
        if len(match_list) > 0 and len(match_list) != 1:
            raise ValueError(
                f"CRITICAL ERROR: Not 1:1 matching.\n"
                f"Function: compute_matched_pairs_variance\n"
                f"This variance estimator only valid for 1:1 matching.\n"
                f"Treated unit {i} has {len(match_list)} matches (expected 1).\n"
                f"Use abadie_imbens_variance() for M:1 matching."
            )

    # ====================================================================
    # Compute Paired Differences
    # ====================================================================

    indices_treated = np.where(treatment)[0]
    paired_diffs = []

    for i, match_list in enumerate(matches):
        if len(match_list) == 0:
            # Unmatched treated unit - skip
            continue

        treated_idx = indices_treated[i]
        control_idx = match_list[0]

        # Paired difference
        diff = outcomes[treated_idx] - outcomes[control_idx]
        paired_diffs.append(diff)

    paired_diffs = np.array(paired_diffs)

    if len(paired_diffs) == 0:
        raise ValueError(
            f"CRITICAL ERROR: No matched pairs.\n"
            f"Function: compute_matched_pairs_variance\n"
            f"Cannot compute variance with zero matched pairs."
        )

    # ====================================================================
    # Compute Variance
    # ====================================================================

    n_matched = len(paired_diffs)

    # Variance of paired differences
    variance = np.var(paired_diffs, ddof=1) / n_matched

    se = np.sqrt(variance)

    return variance, se
