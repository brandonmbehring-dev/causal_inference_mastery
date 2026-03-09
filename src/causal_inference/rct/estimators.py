"""RCT estimators for average treatment effect (ATE).

Following Brandon's principles:
1. NEVER FAIL SILENTLY - All errors explicit with diagnostic info
2. Fail fast - Stop immediately on invalid inputs
3. Type hints required
4. Comprehensive error messages
"""

import numpy as np
from scipy import stats
from typing import Dict, Union


def simple_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Calculate Average Treatment Effect (ATE) using difference-in-means.

    This is the gold-standard estimator for randomized controlled trials (RCTs).
    Under randomization, E[Y(1)] = E[Y|T=1] and E[Y(0)] = E[Y|T=0], so the
    difference-in-means is an unbiased estimate of the ATE.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control). Also accepts boolean.
    alpha : float, default=0.05
        Significance level for confidence interval (must be in (0, 1)).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': ATE point estimate
        - 'se': Standard error (using Neyman variance)
        - 'ci_lower': Lower bound of (1-alpha)% confidence interval
        - 'ci_upper': Upper bound of (1-alpha)% confidence interval
        - 'n_treated': Number of treated units
        - 'n_control': Number of control units

    Raises
    ------
    ValueError
        If inputs are invalid (empty, mismatched lengths, NaN, inf, no variation, etc.)

    Examples
    --------
    >>> outcomes = np.array([7.0, 5.0, 3.0, 1.0])
    >>> treatment = np.array([1, 1, 0, 0])
    >>> result = simple_ate(outcomes, treatment)
    >>> result['estimate']
    4.0

    Notes
    -----
    - Standard error uses Neyman's heteroskedasticity-robust variance:
      Var(ATE) = Var(Y|T=1)/n1 + Var(Y|T=0)/n0
    - Confidence intervals use t-distribution with Satterthwaite degrees of freedom
      (accounts for heteroskedasticity and small samples)
    - Treatment must be binary (0/1 or boolean)
    """
    # ============================================================================
    # Input Validation - Fail Fast with Diagnostic Info
    # ============================================================================

    # Convert to numpy arrays
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    # 1. Check for empty arrays
    if len(outcomes) == 0 or len(treatment) == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: simple_ate\n"
            f"Expected: Non-empty arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}"
        )

    # 2. Check for length mismatch
    if len(outcomes) != len(treatment):
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: simple_ate\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}"
        )

    # 3. Check for NaN values
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: simple_ate\n"
            f"NaN indicates data quality issues that must be addressed.\n"
            f"Got: {np.sum(np.isnan(outcomes))} NaN in outcomes, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment"
        )

    # 4. Check for infinite values
    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: simple_ate\n"
            f"Infinite values indicate numerical instability or data errors.\n"
            f"Got: {np.sum(np.isinf(outcomes))} inf in outcomes, "
            f"{np.sum(np.isinf(treatment))} inf in treatment"
        )

    # 5. Check for binary treatment
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: simple_ate\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # 6. Check for treatment variation
    if len(unique_treatment) < 2:
        if unique_treatment[0] == 1:
            raise ValueError(
                f"CRITICAL ERROR: No control units in data.\n"
                f"Function: simple_ate\n"
                f"Cannot estimate treatment effect without control group.\n"
                f"Got: All units have treatment=1"
            )
        else:
            raise ValueError(
                f"CRITICAL ERROR: No treated units in data.\n"
                f"Function: simple_ate\n"
                f"Cannot estimate treatment effect without treated group.\n"
                f"Got: All units have treatment=0"
            )

    # 7. Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\n"
            f"Function: simple_ate\n"
            f"Expected: alpha in (0, 1)\n"
            f"Got: alpha={alpha}"
        )

    # ============================================================================
    # Estimation
    # ============================================================================

    # Separate outcomes by treatment status
    y1 = outcomes[treatment == 1]  # Treated outcomes
    y0 = outcomes[treatment == 0]  # Control outcomes

    n1 = len(y1)
    n0 = len(y0)

    # Calculate means
    mean1 = np.mean(y1)
    mean0 = np.mean(y0)

    # ATE = difference in means
    ate = mean1 - mean0

    # Neyman heteroskedasticity-robust variance
    var1 = np.var(y1, ddof=1)  # Sample variance (unbiased)
    var0 = np.var(y0, ddof=1)

    # Variance of ATE
    var_ate = var1 / n1 + var0 / n0

    # Standard error
    se = np.sqrt(var_ate)

    # Degrees of freedom (Satterthwaite approximation for Welch's t-test)
    # df = (s1²/n1 + s0²/n0)² / [(s1²/n1)²/(n1-1) + (s0²/n0)²/(n0-1)]
    df = (var_ate**2) / ((var1 / n1) ** 2 / (n1 - 1) + (var0 / n0) ** 2 / (n0 - 1))

    # Confidence interval (t-distribution for small samples)
    t_critical = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = ate - t_critical * se
    ci_upper = ate + t_critical * se

    return {
        "estimate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_treated": n1,
        "n_control": n0,
    }
