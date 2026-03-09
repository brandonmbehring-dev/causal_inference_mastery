"""
Balance diagnostics for propensity score matching.

Implements Standardized Mean Difference (SMD) and Variance Ratio (VR) calculations
from Julia reference (balance.jl).

Key metrics:
- SMD: Scale-free measure of covariate balance (target: |SMD| < 0.1)
- VR: Ratio of variances between treated/control (target: 0.5 < VR < 2.0)

References:
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline
  covariates between treatment groups in propensity-score matched samples.
  Statistics in Medicine, 28(25), 3083-3107.
- Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward.
  Statistical Science, 25(1), 1-21.

Author: Brandon Behring
Date: 2025-11-21
"""

from typing import Tuple, Dict, Any, List
import numpy as np


def compute_smd(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Compute Standardized Mean Difference (SMD) for covariate balance.

    Implements formula from Julia balance.jl (lines 60-103).

    Formula:
    --------
    **Pooled SMD** (default, recommended):
    ```
    SMD = (mean(X_T) - mean(X_C)) / sqrt((var(X_T) + var(X_C)) / 2)
    ```

    **Unpooled SMD** (alternative):
    ```
    SMD = (mean(X_T) - mean(X_C)) / sqrt(var(X_T))
    ```

    Args:
        x_treated: Covariate values for treated group (n_treated,)
        x_control: Covariate values for control group (n_control,)
        pooled: Use pooled standard deviation (default: True, recommended)

    Returns:
        smd: Standardized mean difference

    Interpretation:
        - |SMD| < 0.1: Good balance (standard threshold)
        - 0.1 ≤ |SMD| < 0.2: Acceptable balance
        - |SMD| ≥ 0.2: Poor balance (problematic)

    Notes:
        - Independent of sample size (unlike t-test)
        - Scale-free measure (unlike raw mean difference)
        - Pooled version recommended for matching (Austin 2009)
        - Returns 0.0 if both groups have zero variance
        - Returns 1e6 (proxy for Inf) if different means but zero variance

    Example:
        >>> x_t = np.array([5.0, 6.0, 5.5])
        >>> x_c = np.array([4.5, 5.0, 4.8])
        >>> smd = compute_smd(x_t, x_c)
        >>> if abs(smd) < 0.1:
        ...     print("Good balance achieved")
    """
    x_treated = np.asarray(x_treated)
    x_control = np.asarray(x_control)

    mean_t = np.mean(x_treated)
    mean_c = np.mean(x_control)
    var_t = np.var(x_treated, ddof=1)
    var_c = np.var(x_control, ddof=1)

    if pooled:
        # Pooled standard deviation (recommended)
        pooled_std = np.sqrt((var_t + var_c) / 2)

        # Avoid division by zero (Julia balance.jl lines 74-83)
        if pooled_std < 1e-10:
            # Both have zero variance
            if abs(mean_t - mean_c) < 1e-10:
                return 0.0  # Identical distributions
            else:
                # Different means but no variance -> infinite SMD (perfect separation)
                # Return large value as proxy for Inf
                return np.sign(mean_t - mean_c) * 1e6

        smd = (mean_t - mean_c) / pooled_std
    else:
        # Unpooled (standardize by treated variance only)
        if var_t < 1e-10:
            # Treated has zero variance
            if abs(mean_t - mean_c) < 1e-10:
                return 0.0  # Identical values
            else:
                # Different means but no treated variance -> infinite SMD
                return np.sign(mean_t - mean_c) * 1e6

        smd = (mean_t - mean_c) / np.sqrt(var_t)

    return float(smd)


def compute_variance_ratio(
    x_treated: np.ndarray,
    x_control: np.ndarray,
) -> float:
    """
    Compute Variance Ratio (VR) for covariate balance.

    Implements formula from Julia balance.jl (lines 148-165).

    Formula:
    --------
    ```
    VR = var(X_T) / var(X_C)
    ```

    Args:
        x_treated: Covariate values for treated group (n_treated,)
        x_control: Covariate values for control group (n_control,)

    Returns:
        vr: Variance ratio (var_treated / var_control)

    Interpretation:
        - VR ≈ 1.0: Good balance
        - 0.5 < VR < 2.0: Acceptable balance (some recommend 0.8-1.25)
        - VR ≤ 0.5 or VR ≥ 2.0: Poor balance

    Notes:
        - Complements SMD (checks second moment)
        - Sensitive to outliers
        - Log scale often used for visualization
        - Returns 1.0 if both variances near zero
        - Returns np.inf if only control variance near zero

    Example:
        >>> vr = compute_variance_ratio(x_treated, x_control)
        >>> if 0.5 < vr < 2.0:
        ...     print("Good variance balance")
    """
    x_treated = np.asarray(x_treated)
    x_control = np.asarray(x_control)

    var_t = np.var(x_treated, ddof=1)
    var_c = np.var(x_control, ddof=1)

    # Avoid division by zero (Julia balance.jl lines 156-162)
    if var_c < 1e-10:
        if var_t < 1e-10:
            return 1.0  # Both variances near zero
        else:
            return np.inf  # Only control variance near zero

    return float(var_t / var_c)


def check_covariate_balance(
    covariates: np.ndarray,
    treatment: np.ndarray,
    matched_indices: List[Tuple[int, int]],
    threshold: float = 0.1,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Check covariate balance after matching for ALL covariates.

    Implements algorithm from Julia balance.jl (lines 233-316).

    **CRITICAL (MEDIUM-5)**: Must verify balance on ALL covariates, not subset.

    Args:
        covariates: Covariate matrix (n, p)
        treatment: Binary treatment indicator (n,)
        matched_indices: Matched pairs [(treated_idx, control_idx), ...]
        threshold: SMD threshold for good balance (default: 0.1)

    Returns:
        Tuple of:
        - balanced: True if ALL covariates have |SMD| < threshold
        - smd_after: SMD for each covariate after matching (p,)
        - vr_after: Variance ratios for each covariate after matching (p,)
        - smd_before: SMD before matching (p,)
        - vr_before: Variance ratios before matching (p,)

    Method:
    -------
    1. **Before matching**: Compute SMD and VR for all covariates using full sample
    2. **After matching**: Compute SMD and VR using only matched pairs
    3. **Balance check**: ALL covariates must have |SMD| < threshold

    Interpretation:
    ---------------
    **Good balance**:
    - ALL |SMD| < 0.1
    - Most VR in (0.5, 2.0)

    **Poor balance**:
    - ANY |SMD| ≥ 0.1
    - Many VR outside (0.5, 2.0)

    Example:
        >>> balanced, smd, vr, smd_before, vr_before = check_covariate_balance(
        ...     covariates, treatment, matched_indices, threshold=0.1
        ... )
        >>> if balanced:
        ...     print("✓ All covariates balanced (SMD < 0.1)")
        >>> else:
        ...     print("✗ Some covariates imbalanced")
        ...     for j, s in enumerate(smd):
        ...         if abs(s) >= 0.1:
        ...             print(f"  Covariate {j}: SMD = {s:.3f}")
    """
    # ====================================================================
    # Input Validation
    # ====================================================================

    covariates = np.asarray(covariates)
    treatment = np.asarray(treatment).astype(bool)

    n, p = covariates.shape

    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Mismatched lengths.\n"
            f"Function: check_covariate_balance\n"
            f"covariates has {n} rows, treatment has {len(treatment)} elements.\n"
            f"Must have same length."
        )

    if threshold <= 0:
        raise ValueError(
            f"CRITICAL ERROR: Invalid threshold.\n"
            f"Function: check_covariate_balance\n"
            f"threshold must be > 0, got threshold = {threshold}\n"
            f"Standard value: 0.1 (SMD < 0.1 = good balance)"
        )

    # ====================================================================
    # Balance BEFORE Matching
    # ====================================================================

    indices_treated = np.where(treatment)[0]
    indices_control = np.where(~treatment)[0]

    smd_before = np.zeros(p)
    vr_before = np.zeros(p)

    for j in range(p):
        x_t = covariates[indices_treated, j]
        x_c = covariates[indices_control, j]

        smd_before[j] = compute_smd(x_t, x_c)
        vr_before[j] = compute_variance_ratio(x_t, x_c)

    # ====================================================================
    # Balance AFTER Matching
    # ====================================================================

    if len(matched_indices) == 0:
        # No matches - return NaN for after-matching metrics
        return False, np.full(p, np.nan), np.full(p, np.nan), smd_before, vr_before

    # Extract matched treated and control indices
    matched_treated = np.array([pair[0] for pair in matched_indices])
    matched_control = np.array([pair[1] for pair in matched_indices])

    smd_after = np.zeros(p)
    vr_after = np.zeros(p)

    for j in range(p):
        x_t_matched = covariates[matched_treated, j]
        x_c_matched = covariates[matched_control, j]

        smd_after[j] = compute_smd(x_t_matched, x_c_matched)
        vr_after[j] = compute_variance_ratio(x_t_matched, x_c_matched)

    # ====================================================================
    # Check Balance: ALL Covariates
    # ====================================================================

    # CRITICAL (MEDIUM-5): Must check ALL covariates, not subset
    balanced = np.all(np.abs(smd_after) < threshold)

    return balanced, smd_after, vr_after, smd_before, vr_before


def balance_summary(
    smd_after: np.ndarray,
    vr_after: np.ndarray,
    smd_before: np.ndarray,
    vr_before: np.ndarray,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Create summary of covariate balance diagnostics.

    Implements summary from Julia balance.jl (lines 351-385).

    Args:
        smd_after: SMD values after matching (p,)
        vr_after: Variance ratios after matching (p,)
        smd_before: SMD values before matching (p,)
        vr_before: Variance ratios before matching (p,)
        threshold: SMD threshold for good balance (default: 0.1)

    Returns:
        Dictionary with balance summary:
        - n_covariates: Total number of covariates
        - n_balanced: Number with |SMD| < threshold (after matching)
        - n_imbalanced: Number with |SMD| ≥ threshold (after matching)
        - max_smd_before: Maximum |SMD| before matching
        - max_smd_after: Maximum |SMD| after matching
        - mean_smd_before: Mean |SMD| before matching
        - mean_smd_after: Mean |SMD| after matching
        - improvement: Reduction in mean |SMD| (percentage)
        - all_balanced: True if ALL covariates balanced

    Example:
        >>> summary = balance_summary(smd_after, vr_after, smd_before, vr_before)
        >>> print(f"Balance: {summary['n_balanced']}/{summary['n_covariates']} covariates")
        >>> print(f"Improvement: {summary['improvement']*100:.1f}%")
    """
    p = len(smd_after)

    n_balanced = int(np.sum(np.abs(smd_after) < threshold))
    n_imbalanced = p - n_balanced

    max_smd_before = float(np.max(np.abs(smd_before)))
    max_smd_after = float(np.max(np.abs(smd_after)))

    mean_smd_before = float(np.mean(np.abs(smd_before)))
    mean_smd_after = float(np.mean(np.abs(smd_after)))

    # Improvement: reduction in mean |SMD|
    if mean_smd_before > 0:
        improvement = (mean_smd_before - mean_smd_after) / mean_smd_before
    else:
        improvement = 0.0

    all_balanced = n_imbalanced == 0

    return {
        "n_covariates": p,
        "n_balanced": n_balanced,
        "n_imbalanced": n_imbalanced,
        "max_smd_before": max_smd_before,
        "max_smd_after": max_smd_after,
        "mean_smd_before": mean_smd_before,
        "mean_smd_after": mean_smd_after,
        "improvement": improvement,
        "all_balanced": all_balanced,
    }
