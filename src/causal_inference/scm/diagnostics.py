"""
Synthetic Control Methods - Diagnostics

Provides diagnostic tools for SCM:
- Pre-treatment fit assessment (RMSE, R², MAPE)
- Covariate balance checks
- Weight diagnostics
- Sparsity and donor concentration metrics

References:
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods"
    Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements,
        and Methodological Aspects"
"""

from typing import Optional, Dict, List, Tuple
import warnings

import numpy as np


def check_pre_treatment_fit(
    treated_pre: np.ndarray,
    synthetic_pre: np.ndarray,
    outcome_mean: Optional[float] = None,
) -> Dict[str, float]:
    """
    Assess quality of pre-treatment fit.

    Parameters
    ----------
    treated_pre : np.ndarray
        Pre-treatment outcomes for treated unit(s), shape (n_pre,)
    synthetic_pre : np.ndarray
        Pre-treatment synthetic control, shape (n_pre,)
    outcome_mean : float, optional
        Mean outcome for MAPE calculation. If None, uses treated mean.

    Returns
    -------
    dict
        Contains:
        - rmse: Root mean squared prediction error
        - r_squared: R-squared (1 = perfect fit)
        - mape: Mean absolute percentage error
        - max_gap: Maximum absolute gap
        - mean_gap: Mean signed gap (should be ~0)
        - fit_quality: "excellent", "good", "acceptable", "poor"
    """
    treated_1d = treated_pre.flatten()
    synthetic_1d = synthetic_pre.flatten()

    gap = treated_1d - synthetic_1d
    n = len(gap)

    # RMSE
    rmse = np.sqrt(np.mean(gap ** 2))

    # R-squared
    ss_res = np.sum(gap ** 2)
    ss_tot = np.sum((treated_1d - np.mean(treated_1d)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    # MAPE
    if outcome_mean is None:
        outcome_mean = np.mean(np.abs(treated_1d))
    if outcome_mean > 1e-10:
        mape = np.mean(np.abs(gap)) / outcome_mean * 100
    else:
        mape = np.nan

    # Max and mean gap
    max_gap = np.max(np.abs(gap))
    mean_gap = np.mean(gap)

    # Fit quality classification
    if r_squared > 0.99:
        fit_quality = "excellent"
    elif r_squared > 0.95:
        fit_quality = "good"
    elif r_squared > 0.85:
        fit_quality = "acceptable"
    else:
        fit_quality = "poor"

    return {
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "mape": float(mape),
        "max_gap": float(max_gap),
        "mean_gap": float(mean_gap),
        "fit_quality": fit_quality,
    }


def check_covariate_balance(
    treated_covariates: np.ndarray,
    synthetic_covariates: np.ndarray,
    covariate_names: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Check covariate balance between treated and synthetic control.

    Parameters
    ----------
    treated_covariates : np.ndarray
        Covariates for treated unit(s), shape (n_covariates,)
    synthetic_covariates : np.ndarray
        Covariates for synthetic control, shape (n_covariates,)
    covariate_names : list of str, optional
        Names for covariates

    Returns
    -------
    dict
        Per-covariate balance metrics:
        - difference: raw difference (treated - synthetic)
        - percent_diff: percentage difference
        - balanced: True if |percent_diff| < 10%
    """
    treated_1d = treated_covariates.flatten()
    synthetic_1d = synthetic_covariates.flatten()

    if len(treated_1d) != len(synthetic_1d):
        raise ValueError(
            f"Covariate dimension mismatch: treated {len(treated_1d)}, "
            f"synthetic {len(synthetic_1d)}"
        )

    n_covariates = len(treated_1d)

    if covariate_names is None:
        covariate_names = [f"X{i}" for i in range(n_covariates)]

    results = {}
    for i, name in enumerate(covariate_names):
        diff = treated_1d[i] - synthetic_1d[i]
        denom = abs(treated_1d[i]) if abs(treated_1d[i]) > 1e-10 else 1.0
        pct_diff = diff / denom * 100

        results[name] = {
            "treated": float(treated_1d[i]),
            "synthetic": float(synthetic_1d[i]),
            "difference": float(diff),
            "percent_diff": float(pct_diff),
            "balanced": abs(pct_diff) < 10.0,
        }

    return results


def check_weight_properties(
    weights: np.ndarray,
    control_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Analyze properties of synthetic control weights.

    Parameters
    ----------
    weights : np.ndarray
        Synthetic control weights, shape (n_control,)
    control_labels : list of str, optional
        Labels for control units

    Returns
    -------
    dict
        Contains:
        - n_nonzero: number of donors with non-zero weight
        - sparsity: fraction of weights that are zero
        - max_weight: maximum weight (concentration)
        - hhi: Herfindahl-Hirschman Index (concentration)
        - effective_n: effective number of donors (1/HHI)
        - top_donors: list of (label, weight) for top donors
    """
    n_control = len(weights)

    if control_labels is None:
        control_labels = [f"Control_{i}" for i in range(n_control)]

    # Non-zero weights (threshold for numerical precision)
    nonzero_mask = weights > 1e-6
    n_nonzero = np.sum(nonzero_mask)
    sparsity = 1.0 - n_nonzero / n_control

    # Concentration metrics
    max_weight = np.max(weights)
    hhi = np.sum(weights ** 2)  # Herfindahl-Hirschman Index
    effective_n = 1.0 / hhi if hhi > 1e-10 else n_control

    # Top donors
    sorted_idx = np.argsort(weights)[::-1]
    top_donors = []
    for idx in sorted_idx[:5]:  # Top 5
        if weights[idx] > 1e-6:
            top_donors.append((control_labels[idx], float(weights[idx])))

    return {
        "n_nonzero": int(n_nonzero),
        "sparsity": float(sparsity),
        "max_weight": float(max_weight),
        "hhi": float(hhi),
        "effective_n": float(effective_n),
        "top_donors": top_donors,
    }


def diagnose_scm_quality(
    pre_fit: Dict[str, float],
    weight_properties: Dict[str, float],
    n_pre_periods: int,
) -> Dict[str, str]:
    """
    Comprehensive SCM quality diagnostics with warnings.

    Parameters
    ----------
    pre_fit : dict
        Output from check_pre_treatment_fit()
    weight_properties : dict
        Output from check_weight_properties()
    n_pre_periods : int
        Number of pre-treatment periods

    Returns
    -------
    dict
        Contains:
        - overall_quality: "high", "medium", "low"
        - warnings: list of warning messages
        - recommendations: list of recommendations
    """
    warnings_list = []
    recommendations = []

    # Check pre-treatment fit
    if pre_fit["r_squared"] < 0.85:
        warnings_list.append(
            f"Poor pre-treatment fit (R² = {pre_fit['r_squared']:.2f}). "
            "Causal interpretation may be unreliable."
        )
        recommendations.append(
            "Consider using Augmented SCM or adding covariates to improve fit."
        )

    # Check weight concentration
    if weight_properties["max_weight"] > 0.9:
        warnings_list.append(
            f"High weight concentration: max weight = {weight_properties['max_weight']:.2f}. "
            "Synthetic control dominated by single donor."
        )

    if weight_properties["effective_n"] < 2:
        recommendations.append(
            "Consider if a DiD or matched comparison might be more appropriate."
        )

    # Check pre-treatment periods
    if n_pre_periods < 5:
        warnings_list.append(
            f"Only {n_pre_periods} pre-treatment periods. "
            "SCM typically requires >= 5 for reliable estimation."
        )
        recommendations.append(
            "Consider Augmented SCM (Ben-Michael et al. 2021) for better bias properties."
        )

    # Check sparsity
    if weight_properties["sparsity"] < 0.5:
        # Many non-zero weights might indicate overfitting
        warnings_list.append(
            f"Low sparsity ({weight_properties['sparsity']:.1%} zeros). "
            "Many donors have non-trivial weights."
        )

    # Overall quality
    if pre_fit["r_squared"] > 0.95 and weight_properties["effective_n"] >= 2:
        overall_quality = "high"
    elif pre_fit["r_squared"] > 0.85:
        overall_quality = "medium"
    else:
        overall_quality = "low"

    return {
        "overall_quality": overall_quality,
        "warnings": warnings_list,
        "recommendations": recommendations,
    }


def compute_rmspe_ratio(
    gap: np.ndarray,
    treatment_period: int,
) -> Tuple[float, float, float]:
    """
    Compute pre and post RMSPE and their ratio.

    The ratio of post/pre RMSPE is used in placebo tests.
    A high ratio indicates the treatment effect is large relative
    to pre-treatment prediction error.

    Parameters
    ----------
    gap : np.ndarray
        Gap between treated and synthetic, shape (n_periods,)
    treatment_period : int
        Period when treatment starts

    Returns
    -------
    pre_rmspe : float
        Pre-treatment RMSPE
    post_rmspe : float
        Post-treatment RMSPE
    ratio : float
        post_rmspe / pre_rmspe
    """
    pre_gap = gap[:treatment_period]
    post_gap = gap[treatment_period:]

    pre_rmspe = np.sqrt(np.mean(pre_gap ** 2))
    post_rmspe = np.sqrt(np.mean(post_gap ** 2))

    if pre_rmspe > 1e-10:
        ratio = post_rmspe / pre_rmspe
    else:
        ratio = np.inf if post_rmspe > 1e-10 else 1.0

    return float(pre_rmspe), float(post_rmspe), float(ratio)


def suggest_donor_pool_trim(
    weights: np.ndarray,
    pre_rmspe_by_donor: np.ndarray,
    threshold_multiple: float = 2.0,
) -> List[int]:
    """
    Suggest donors to exclude based on poor pre-treatment fit.

    In placebo tests, donors with much higher pre-treatment RMSPE
    than the treated unit should be excluded.

    Parameters
    ----------
    weights : np.ndarray
        Original SCM weights
    pre_rmspe_by_donor : np.ndarray
        Pre-treatment RMSPE for each placebo donor
    threshold_multiple : float
        Exclude donors with RMSPE > threshold * treated RMSPE

    Returns
    -------
    exclude_indices : list of int
        Indices of donors to exclude from placebo tests
    """
    # Treated unit's pre-RMSPE (weighted combination approximation)
    # Since we're comparing to donors, use median as reference
    reference_rmspe = np.median(pre_rmspe_by_donor)
    threshold = threshold_multiple * reference_rmspe

    exclude_indices = []
    for i, rmspe in enumerate(pre_rmspe_by_donor):
        if rmspe > threshold:
            exclude_indices.append(i)

    return exclude_indices
