"""Helper functions for propensity score estimation.

This module provides internal utilities for propensity score modeling:
- Logistic regression fitting with convergence checking
- Separation detection (perfect and near)
- Model diagnostics (AUC, pseudo-R²)

These functions support the main estimate_propensity() function but are
separated to improve code organization and maintainability.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional


def fit_logistic_model(
    covariates: np.ndarray,
    treatment: np.ndarray,
    max_iter: int,
    random_state: Optional[int],
) -> LogisticRegression:
    """
    Fit logistic regression model for propensity scores.

    Uses scikit-learn's LogisticRegression with no regularization (penalty=None)
    for interpretability. Checks convergence and raises explicit errors on failure.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features)
    treatment : np.ndarray
        Binary treatment indicator (0 or 1)
    max_iter : int
        Maximum iterations for convergence
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    LogisticRegression
        Fitted logistic regression model

    Raises
    ------
    ValueError
        If model fails to fit (perfect separation, collinearity) or does not
        converge within max_iter iterations

    Notes
    -----
    Configuration:
    - penalty=None: No regularization (L1/L2) for interpretability
    - solver='lbfgs': Default solver, works well for most cases
    - fit_intercept=True: Include intercept term

    Convergence check:
    - Raises ValueError if n_iter >= max_iter
    - Suggests: increase max_iter, standardize covariates, check collinearity

    Examples
    --------
    >>> X = np.random.normal(0, 1, (100, 3))
    >>> T = (X[:, 0] > 0).astype(int)
    >>> model = fit_logistic_model(X, T, max_iter=1000, random_state=42)
    >>> propensity = model.predict_proba(X)[:, 1]
    """
    model = LogisticRegression(
        penalty=None,  # No regularization (for interpretability)
        fit_intercept=True,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",  # Default solver, works well for most cases
    )

    try:
        model.fit(covariates, treatment)
    except Exception as e:
        raise ValueError(
            f"Logistic regression failed to fit: {str(e)}. "
            f"Possible causes: Perfect separation, collinearity, singular matrix"
        )

    # Check convergence
    if not model.n_iter_ < max_iter:
        raise ValueError(
            f"Logistic regression did not converge after {model.n_iter_} iterations (max={max_iter}). "
            f"Options: Increase max_iter, standardize covariates, or check for collinearity."
        )

    return model


def check_separation(propensity: np.ndarray) -> None:
    """
    Check for perfect and near separation in propensity scores.

    Perfect separation (propensity exactly 0 or 1) violates the positivity
    assumption and makes IPW estimation impossible due to infinite weights.
    Near separation (extreme propensities) leads to unstable estimates.

    Parameters
    ----------
    propensity : np.ndarray
        Propensity scores P(T=1|X), must be in (0, 1)

    Raises
    ------
    ValueError
        If perfect separation detected (propensity ≈ 0 or ≈ 1 within epsilon=1e-10)

    Warnings
    --------
    UserWarning
        If near separation detected (propensity < 0.01 or > 0.99)

    Notes
    -----
    Perfect separation occurs when treatment is perfectly predicted by covariates:
    - Some units have P(T=1|X) ≈ 0 (never treated)
    - Some units have P(T=1|X) ≈ 1 (always treated)
    - IPW weights become infinite: w = 1/P(T|X)
    - Positivity assumption violated

    Near separation warnings:
    - Propensity < 0.01 or > 0.99 (1% threshold)
    - Estimation possible but unstable (high variance)
    - Consider trimming extreme weights

    Solutions for perfect separation:
    - Drop perfectly-predictive covariates
    - Use matching instead of IPW
    - Use overlap weights (less sensitive to extremes)

    Examples
    --------
    >>> # Valid propensity scores
    >>> p = np.array([0.2, 0.5, 0.8])
    >>> check_separation(p)  # No error, no warning

    >>> # Near separation (warns)
    >>> p = np.array([0.005, 0.5, 0.995])
    >>> check_separation(p)  # Warns about extreme scores

    >>> # Perfect separation (raises)
    >>> p = np.array([0.0, 0.5, 1.0])
    >>> check_separation(p)  # ValueError raised
    """
    # Perfect separation: propensity exactly 0 or 1 (leads to infinite IPW weights)
    epsilon = 1e-10
    has_perfect_separation = np.any((propensity < epsilon) | (propensity > 1 - epsilon))

    if has_perfect_separation:
        n_zero: int = int(np.sum(propensity < epsilon))
        n_one: int = int(np.sum(propensity > 1 - epsilon))
        raise ValueError(
            f"Perfect separation detected in propensity estimation. "
            f"Propensity scores of exactly 0 or 1 indicate treatment is perfectly "
            f"predicted by covariates, violating the positivity assumption. "
            f"Got: {n_zero} units with P(T=1|X)≈0, {n_one} units with P(T=1|X)≈1. "
            f"This makes IPW weights infinite and estimation impossible. "
            f"Options: Drop perfectly-predictive covariates, or use matching instead of IPW."
        )

    # Near separation: extreme propensities (< 0.01 or > 0.99)
    # This is a warning, not an error - estimation is possible but unstable
    extreme_threshold = 0.01
    n_extreme_low: int = int(np.sum(propensity < extreme_threshold))
    n_extreme_high: int = int(np.sum(propensity > 1 - extreme_threshold))

    if n_extreme_low > 0 or n_extreme_high > 0:
        warnings.warn(
            f"Extreme propensity scores detected (potential positivity violation). "
            f"{n_extreme_low} units with P(T=1|X)<{extreme_threshold}, "
            f"{n_extreme_high} units with P(T=1|X)>{1 - extreme_threshold}. "
            f"IPW estimates may be unstable. Consider trimming extreme weights.",
            UserWarning,
        )


def compute_propensity_diagnostics(
    treatment: np.ndarray, propensity: np.ndarray, model: LogisticRegression
) -> Dict[str, Any]:
    """
    Compute diagnostic statistics for propensity score model.

    Calculates model fit and discrimination metrics to assess quality of
    propensity score estimation.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (0 or 1)
    propensity : np.ndarray
        Estimated propensity scores P(T=1|X)
    model : LogisticRegression
        Fitted logistic regression model

    Returns
    -------
    dict
        Diagnostic statistics with keys:
        - 'auc': float, ROC AUC score (0.5=random, 1.0=perfect discrimination)
        - 'pseudo_r2': float, McFadden's pseudo-R² (0=null model, 1=perfect fit)
        - 'converged': bool, whether model converged (always True here)
        - 'n_iter': int, iterations to convergence
        - 'coef': np.ndarray, regression coefficients for each covariate
        - 'intercept': float, regression intercept

    Notes
    -----
    AUC (Area Under ROC Curve):
    - Measures discrimination: how well model separates treated vs control
    - AUC = 0.5: No discrimination (treatment independent of covariates)
    - AUC = 0.7: Good discrimination (moderate confounding)
    - AUC = 0.9: Excellent discrimination (strong confounding)
    - High AUC means propensity adjustment is important

    Pseudo-R² (McFadden):
    - R² = 1 - (log-likelihood model / log-likelihood null)
    - Null model: constant propensity = mean(T)
    - Range: [0, 1], but rarely exceeds 0.4 even for good models
    - Pseudo-R² = 0.2 considered decent fit for logistic regression

    Examples
    --------
    >>> X = np.random.normal(0, 1, (100, 2))
    >>> T = (X[:, 0] > 0).astype(int)
    >>> model = LogisticRegression(penalty=None).fit(X, T)
    >>> p = model.predict_proba(X)[:, 1]
    >>> diag = compute_propensity_diagnostics(T, p, model)
    >>> print(f"AUC: {diag['auc']:.3f}, Pseudo-R²: {diag['pseudo_r2']:.3f}")
    AUC: 0.874, Pseudo-R²: 0.412
    """
    # AUC (discrimination: how well model separates treated vs control)
    auc = roc_auc_score(treatment, propensity)

    # Pseudo-R² (McFadden): R² = 1 - (log-likelihood model / log-likelihood null)
    # Null model: constant propensity = mean(T)
    log_likelihood_null: float = float(
        np.sum(
            treatment * np.log(np.mean(treatment) + 1e-10)
            + (1 - treatment) * np.log(1 - np.mean(treatment) + 1e-10)
        )
    )
    log_likelihood_model: float = float(
        np.sum(
            treatment * np.log(propensity + 1e-10)
            + (1 - treatment) * np.log(1 - propensity + 1e-10)
        )
    )
    pseudo_r2 = 1 - (log_likelihood_model / log_likelihood_null)

    return {
        "auc": auc,
        "pseudo_r2": pseudo_r2,
        "converged": True,
        "n_iter": (model.n_iter_[0] if hasattr(model.n_iter_, "__getitem__") else model.n_iter_),
        "coef": model.coef_.flatten(),
        "intercept": model.intercept_[0],
    }
