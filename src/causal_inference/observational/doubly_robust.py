"""Doubly robust (DR) estimation for causal inference.

This module implements doubly robust ATE estimation that combines inverse probability
weighting (IPW) with outcome regression. The DR estimator is consistent if EITHER
the propensity model OR the outcome model is correctly specified.

Key property: Double robustness
- If propensity model correct → consistent (even if outcome model wrong)
- If outcome model correct → consistent (even if propensity model wrong)
- If both correct → consistent AND efficient (lowest variance)
- If both wrong → biased (no protection)
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple

from .propensity import estimate_propensity, trim_propensity
from .outcome_regression import fit_outcome_models


def dr_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    propensity: Optional[Union[np.ndarray, list]] = None,
    outcome_models: Optional[Dict[str, Any]] = None,
    trim_at: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Doubly robust ATE estimator combining IPW with outcome regression.

    The DR estimator has the form:

        ATE_DR = (1/n) * Σ[
            T/e(X) * (Y - μ₁(X)) + μ₁(X)           # Treated contribution
            - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)  # Control contribution
        ]

    Where:
        - e(X) = P(T=1|X) is the propensity score
        - μ₁(X) = E[Y|T=1, X] is the outcome model for treated
        - μ₀(X) = E[Y|T=0, X] is the outcome model for control

    Double robustness property: The estimator is consistent if EITHER the
    propensity model OR the outcome model is correctly specified.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units (n,).
    treatment : np.ndarray or list
        Binary treatment indicator (1=treated, 0=control) (n,).
    covariates : np.ndarray or list
        Covariate matrix (n, p). Can be 1D for single covariate.
    propensity : np.ndarray or list, optional
        Pre-computed propensity scores P(T=1|X) (n,). If None, estimated via logistic regression.
    outcome_models : dict, optional
        Pre-computed outcome models. If None, fitted via linear regression.
        Expected keys: 'mu0_predictions', 'mu1_predictions' (both shape (n,)).
    trim_at : tuple of float, optional
        Propensity trimming percentiles (lower, upper). E.g., (0.01, 0.99).
        If None, no trimming applied.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': DR ATE estimate (float)
        - 'se': Robust standard error via influence function (float)
        - 'ci_lower': Lower bound of (1-alpha)% CI (float)
        - 'ci_upper': Upper bound of (1-alpha)% CI (float)
        - 'n': Total sample size (int)
        - 'n_treated': Number of treated units (int)
        - 'n_control': Number of control units (int)
        - 'n_trimmed': Number of units trimmed (0 if no trimming) (int)
        - 'propensity_diagnostics': dict with AUC, pseudo-R², convergence
        - 'outcome_diagnostics': dict with R², RMSE for μ₁ and μ₀
        - 'propensity': Propensity scores used (n,)
        - 'mu0_predictions': E[Y|T=0, X] for all X (n,)
        - 'mu1_predictions': E[Y|T=1, X] for all X (n,)

    Raises
    ------
    ValueError
        If inputs invalid, mismatched lengths, or insufficient data.

    Examples
    --------
    >>> # Both models correct (ideal case)
    >>> X = np.random.normal(0, 1, 200)
    >>> e_X = 1 / (1 + np.exp(-0.8*X))
    >>> T = np.random.binomial(1, e_X)
    >>> Y = 3.0*T + 0.5*X + np.random.normal(0, 1, 200)
    >>> result = dr_ate(Y, T, X)
    >>> result['estimate']  # Should be near 3.0

    >>> # Propensity correct, outcome misspecified (still consistent)
    >>> Y_quad = 3.0*T + 0.5*X**2 + np.random.normal(0, 1, 200)
    >>> result = dr_ate(Y_quad, T, X)  # Fits linear μ(X) but DR protects via IPW

    Notes
    -----
    - Variance estimation uses influence function approach
    - Propensity scores clipped at [1e-6, 1-1e-6] for numerical stability
    - If both models misspecified, DR estimator may be biased
    - Trimming reduces variance but introduces bias (finite sample)
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    if len(treatment) != n or covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Input arrays have different lengths.\n"
            f"Function: dr_ate\n"
            f"Expected: All arrays length {n}\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"covariates.shape={covariates.shape}"
        )

    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)) or np.any(np.isnan(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: dr_ate\n"
            f"NaN indicates data quality issues that must be addressed."
        )

    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)) or np.any(np.isinf(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n" f"Function: dr_ate"
        )

    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: dr_ate\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # ============================================================================
    # Step 1: Estimate or Use Provided Propensity Scores
    # ============================================================================

    if propensity is None:
        prop_result = estimate_propensity(treatment, covariates)
        propensity_scores = prop_result["propensity"]
        propensity_diagnostics = prop_result["diagnostics"]
    else:
        propensity_scores = np.asarray(propensity, dtype=float)
        if len(propensity_scores) != n:
            raise ValueError(
                f"CRITICAL ERROR: Propensity length mismatch.\n"
                f"Function: dr_ate\n"
                f"Expected: {n}\n"
                f"Got: {len(propensity_scores)}"
            )
        propensity_diagnostics = {"provided": True}

    # ============================================================================
    # Step 2: Trim Propensity Scores (Optional)
    # ============================================================================

    n_trimmed = 0
    if trim_at is not None:
        trim_result = trim_propensity(
            propensity_scores, treatment, outcomes, covariates, trim_at=trim_at
        )
        propensity_scores = trim_result["propensity"]
        treatment = trim_result["treatment"]
        outcomes = trim_result["outcomes"]
        covariates = trim_result["covariates"]
        n_trimmed = n - len(propensity_scores)
        n = len(propensity_scores)

    # ============================================================================
    # Step 3: Clip Propensity Scores for Numerical Stability
    # ============================================================================

    epsilon = 1e-6
    propensity_clipped = np.clip(propensity_scores, epsilon, 1 - epsilon)

    # ============================================================================
    # Step 4: Fit or Use Provided Outcome Models
    # ============================================================================

    if outcome_models is None:
        outcome_result = fit_outcome_models(outcomes, treatment, covariates)
        mu0_predictions = outcome_result["mu0_predictions"]
        mu1_predictions = outcome_result["mu1_predictions"]
        outcome_diagnostics = outcome_result["diagnostics"]
    else:
        if "mu0_predictions" not in outcome_models or "mu1_predictions" not in outcome_models:
            raise ValueError(
                f"CRITICAL ERROR: outcome_models must contain 'mu0_predictions' and 'mu1_predictions'.\n"
                f"Function: dr_ate\n"
                f"Got keys: {list(outcome_models.keys())}"
            )
        mu0_predictions = np.asarray(outcome_models["mu0_predictions"], dtype=float)
        mu1_predictions = np.asarray(outcome_models["mu1_predictions"], dtype=float)
        if len(mu0_predictions) != n or len(mu1_predictions) != n:
            raise ValueError(
                f"CRITICAL ERROR: Outcome model predictions length mismatch.\n"
                f"Function: dr_ate\n"
                f"Expected: {n}\n"
                f"Got: mu0={len(mu0_predictions)}, mu1={len(mu1_predictions)}"
            )
        outcome_diagnostics = {"provided": True}

    # ============================================================================
    # Step 5: Compute Doubly Robust Estimator
    # ============================================================================

    # DR formula:
    # ATE_DR = (1/n) * Σ[T/e(X) * (Y - μ₁(X)) + μ₁(X)
    #                   - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)]

    treated_contribution = (
        treatment / propensity_clipped * (outcomes - mu1_predictions) + mu1_predictions
    )
    control_contribution = (
        (1 - treatment) / (1 - propensity_clipped) * (outcomes - mu0_predictions) + mu0_predictions
    )

    dr_estimate = np.mean(treated_contribution - control_contribution)

    # ============================================================================
    # Step 6: Compute Variance via Influence Function
    # ============================================================================

    # Influence function:
    # IF_i = T/e(X) * (Y - μ₁(X)) + μ₁(X)
    #       - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)
    #       - ATE_DR

    influence_function = treated_contribution - control_contribution - dr_estimate

    # Variance: Var(ATE_DR) = (1/n²) * Σ IF_i²
    variance = np.mean(influence_function**2) / n
    se = np.sqrt(variance)

    # ============================================================================
    # Step 7: Confidence Interval
    # ============================================================================

    # Use t-distribution for small samples (n < 50), z for large samples
    if n < 50:
        # Degrees of freedom: n - 2 (approximate for DR estimator)
        df = n - 2
        critical = stats.t.ppf(1 - alpha / 2, df=df)
    else:
        # For large samples, t ≈ z
        critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = dr_estimate - critical * se
    ci_upper = dr_estimate + critical * se

    # ============================================================================
    # Step 8: Sample Sizes
    # ============================================================================

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    return {
        "estimate": float(dr_estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": int(n),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "n_trimmed": int(n_trimmed),
        "propensity_diagnostics": propensity_diagnostics,
        "outcome_diagnostics": outcome_diagnostics,
        "propensity": propensity_clipped,
        "mu0_predictions": mu0_predictions,
        "mu1_predictions": mu1_predictions,
    }
