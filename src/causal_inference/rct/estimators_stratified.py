"""Stratified ATE estimator using variance reduction through blocking.

This module implements stratified (blocked) randomized experiments where
treatment is assigned randomly WITHIN strata, not overall.

Key benefit: Removes between-stratum variation, reducing standard errors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, List


def stratified_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    strata: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate stratified Average Treatment Effect using variance reduction.

    In stratified randomization, treatment is assigned randomly WITHIN each stratum.
    This removes between-stratum variation, typically reducing standard errors compared
    to simple difference-in-means.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control). Also accepts boolean.
    strata : np.ndarray or list
        Stratum indicator for each unit. Can be int, string, or categorical.
    alpha : float, default=0.05
        Significance level for confidence interval (must be in (0, 1)).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': Overall stratified ATE (weighted average across strata)
        - 'se': Standard error accounting for stratification
        - 'ci_lower': Lower bound of (1-alpha)% CI
        - 'ci_upper': Upper bound of (1-alpha)% CI
        - 'n_strata': Number of strata
        - 'stratum_estimates': List of stratum-specific ATEs
        - 'stratum_weights': List of weights (proportional to stratum size)
        - 'stratum_ses': List of stratum-specific standard errors
        - 'n_treated': Total treated units across all strata
        - 'n_control': Total control units across all strata

    Raises
    ------
    ValueError
        If inputs invalid (mismatched lengths, no variation within strata, etc.)

    Examples
    --------
    >>> # Two strata with different baselines
    >>> outcomes = np.array([10, 8, 4, 2, 20, 18, 14, 12])
    >>> treatment = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    >>> strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    >>> result = stratified_ate(outcomes, treatment, strata)
    >>> result['estimate']  # Both strata have ATE=6, so overall ATE=6
    6.0

    Notes
    -----
    - Estimator: ATE = sum_h (n_h / n) * ATE_h where h indexes strata
    - Variance: Var(ATE) = sum_h (n_h / n)^2 * Var(ATE_h)
    - Each stratum uses Neyman variance: Var(ATE_h) = s^2_1h/n_1h + s^2_0h/n_0h
    - More efficient than simple_ate when outcomes vary substantially by stratum
    - Requires treatment randomization WITHIN each stratum
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    # Convert to numpy arrays
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    strata = np.asarray(strata)

    n = len(outcomes)

    # Check lengths match
    if len(treatment) != n or len(strata) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: stratified_ate\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"len(strata)={len(strata)}"
        )

    # Check for empty
    if n == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: stratified_ate\n"
            f"Expected: Non-empty arrays"
        )

    # Check for NaN
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: stratified_ate\n"
            f"NaN indicates data quality issues that must be addressed.\n"
            f"Got: {np.sum(np.isnan(outcomes))} NaN in outcomes, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment"
        )

    # Check for infinite values
    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: stratified_ate\n"
            f"Got: {np.sum(np.isinf(outcomes))} inf in outcomes, "
            f"{np.sum(np.isinf(treatment))} inf in treatment"
        )

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: stratified_ate\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\n"
            f"Function: stratified_ate\n"
            f"Expected: alpha in (0, 1)\n"
            f"Got: alpha={alpha}"
        )

    # ============================================================================
    # Stratified Estimation
    # ============================================================================

    # Create dataframe for grouping (easier than manual loops)
    df = pd.DataFrame({
        'outcome': outcomes,
        'treatment': treatment,
        'stratum': strata
    })

    unique_strata = df['stratum'].unique()
    n_strata = len(unique_strata)

    stratum_estimates = []
    stratum_ses = []
    stratum_weights = []
    stratum_ns = []

    for stratum_id in unique_strata:
        # Get data for this stratum
        stratum_data = df[df['stratum'] == stratum_id]
        y_stratum = stratum_data['outcome'].values
        t_stratum = stratum_data['treatment'].values

        n_stratum = len(y_stratum)

        # Check for variation within stratum
        if len(np.unique(t_stratum)) < 2:
            if np.all(t_stratum == 1):
                raise ValueError(
                    f"CRITICAL ERROR: No control units in stratum '{stratum_id}'.\n"
                    f"Function: stratified_ate\n"
                    f"Cannot estimate treatment effect without control group in each stratum.\n"
                    f"Stratum '{stratum_id}' has all units treated."
                )
            else:
                raise ValueError(
                    f"CRITICAL ERROR: No treated units in stratum '{stratum_id}'.\n"
                    f"Function: stratified_ate\n"
                    f"Cannot estimate treatment effect without treated group in each stratum.\n"
                    f"Stratum '{stratum_id}' has all units in control."
                )

        # Calculate ATE for this stratum
        y1 = y_stratum[t_stratum == 1]
        y0 = y_stratum[t_stratum == 0]

        n1 = len(y1)
        n0 = len(y0)

        mean1 = np.mean(y1)
        mean0 = np.mean(y0)

        ate_stratum = mean1 - mean0

        # Neyman variance for this stratum
        var1 = np.var(y1, ddof=1) if n1 > 1 else 0
        var0 = np.var(y0, ddof=1) if n0 > 1 else 0

        var_ate_stratum = var1 / n1 + var0 / n0
        se_stratum = np.sqrt(var_ate_stratum)

        stratum_estimates.append(ate_stratum)
        stratum_ses.append(se_stratum)
        stratum_weights.append(n_stratum / n)  # Weight by stratum size
        stratum_ns.append(n_stratum)

    # Overall stratified ATE (weighted average)
    ate = np.sum(np.array(stratum_estimates) * np.array(stratum_weights))

    # Overall variance (sum of weighted variances)
    var_ate = np.sum((np.array(stratum_weights)**2) * (np.array(stratum_ses)**2))
    se = np.sqrt(var_ate)

    # Confidence interval
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_critical * se
    ci_upper = ate + z_critical * se

    # Total counts
    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    return {
        "estimate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_strata": n_strata,
        "stratum_estimates": stratum_estimates,
        "stratum_weights": stratum_weights,
        "stratum_ses": stratum_ses,
        "n_treated": n_treated,
        "n_control": n_control,
    }
