"""Propensity score estimation and weight adjustments for observational studies.

This module implements propensity score estimation using logistic regression,
along with weight trimming and stabilization utilities.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Tuple, Union

from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
    validate_binary,
    validate_not_empty,
    validate_has_variation,
)


# ============================================================================
# Private Helper Functions
# ============================================================================


def _fit_logistic_model(
    covariates: np.ndarray,
    treatment: np.ndarray,
    max_iter: int,
    random_state: int | None,
) -> LogisticRegression:
    """
    Fit logistic regression model for propensity scores.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features)
    treatment : np.ndarray
        Binary treatment indicator
    max_iter : int
        Maximum iterations for convergence
    random_state : int, optional
        Random seed

    Returns
    -------
    LogisticRegression
        Fitted model

    Raises
    ------
    ValueError
        If model fails to fit or converge
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


def _check_separation(propensity: np.ndarray) -> None:
    """
    Check for perfect and near separation in propensity scores.

    Perfect separation (propensity exactly 0 or 1) violates positivity and makes
    IPW estimation impossible. Near separation (extreme propensities) leads to
    unstable estimates.

    Parameters
    ----------
    propensity : np.ndarray
        Propensity scores P(T=1|X)

    Raises
    ------
    ValueError
        If perfect separation detected (propensity ≈ 0 or ≈ 1)

    Warnings
    --------
    UserWarning
        If near separation detected (propensity < 0.01 or > 0.99)
    """
    # Perfect separation: propensity exactly 0 or 1 (leads to infinite IPW weights)
    epsilon = 1e-10
    has_perfect_separation = np.any((propensity < epsilon) | (propensity > 1 - epsilon))

    if has_perfect_separation:
        n_zero = np.sum(propensity < epsilon)
        n_one = np.sum(propensity > 1 - epsilon)
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
    n_extreme_low = np.sum(propensity < extreme_threshold)
    n_extreme_high = np.sum(propensity > 1 - extreme_threshold)

    if n_extreme_low > 0 or n_extreme_high > 0:
        warnings.warn(
            f"Extreme propensity scores detected (potential positivity violation). "
            f"{n_extreme_low} units with P(T=1|X)<{extreme_threshold}, "
            f"{n_extreme_high} units with P(T=1|X)>{1-extreme_threshold}. "
            f"IPW estimates may be unstable. Consider trimming extreme weights.",
            UserWarning,
        )


def _compute_propensity_diagnostics(
    treatment: np.ndarray, propensity: np.ndarray, model: LogisticRegression
) -> Dict[str, Any]:
    """
    Compute diagnostic statistics for propensity score model.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator
    propensity : np.ndarray
        Estimated propensity scores P(T=1|X)
    model : LogisticRegression
        Fitted logistic regression model

    Returns
    -------
    dict
        Diagnostic statistics:
        - 'auc': ROC AUC score (model discrimination)
        - 'pseudo_r2': McFadden's pseudo-R² (model fit)
        - 'converged': Whether model converged
        - 'n_iter': Iterations to convergence
        - 'coef': Regression coefficients
        - 'intercept': Regression intercept
    """
    # AUC (discrimination: how well model separates treated vs control)
    auc = roc_auc_score(treatment, propensity)

    # Pseudo-R² (McFadden): R² = 1 - (log-likelihood model / log-likelihood null)
    # Null model: constant propensity = mean(T)
    log_likelihood_null = np.sum(
        treatment * np.log(np.mean(treatment) + 1e-10)
        + (1 - treatment) * np.log(1 - np.mean(treatment) + 1e-10)
    )
    log_likelihood_model = np.sum(
        treatment * np.log(propensity + 1e-10)
        + (1 - treatment) * np.log(1 - propensity + 1e-10)
    )
    pseudo_r2 = 1 - (log_likelihood_model / log_likelihood_null)

    return {
        "auc": auc,
        "pseudo_r2": pseudo_r2,
        "converged": True,
        "n_iter": model.n_iter_[0] if hasattr(model.n_iter_, "__getitem__") else model.n_iter_,
        "coef": model.coef_.flatten(),
        "intercept": model.intercept_[0],
    }


def estimate_propensity(
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    method: str = "logistic",
    max_iter: int = 1000,
    random_state: int = None,
) -> Dict[str, Any]:
    """
    Estimate propensity scores P(T=1|X) using logistic regression.

    Parameters
    ----------
    treatment : np.ndarray or list
        Binary treatment indicator (1=treated, 0=control).
    covariates : np.ndarray or list
        Covariate matrix (n_samples, n_features). Can be 1D for single covariate.
    method : str, default="logistic"
        Estimation method. Currently only "logistic" supported.
    max_iter : int, default=1000
        Maximum iterations for logistic regression convergence.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'propensity': np.ndarray of shape (n,), P(T=1|X) for each unit
        - 'model': fitted LogisticRegression object
        - 'diagnostics': dict with:
            - 'auc': float, ROC AUC score (model discrimination)
            - 'pseudo_r2': float, McFadden's pseudo-R² (model fit)
            - 'converged': bool, whether model converged
            - 'n_iter': int, iterations to convergence
            - 'coef': np.ndarray, regression coefficients
            - 'intercept': float, regression intercept

    Raises
    ------
    ValueError
        If treatment not binary, covariates invalid, or model fails to converge.

    Examples
    --------
    >>> # Single covariate
    >>> X = np.random.normal(0, 1, 100)
    >>> T = (X > 0).astype(int)  # Treatment depends on X
    >>> result = estimate_propensity(T, X)
    >>> result['propensity']  # Estimated P(T=1|X)
    >>> result['diagnostics']['auc']  # Should be > 0.5 (better than random)

    >>> # Multiple covariates
    >>> X = np.random.normal(0, 1, (100, 3))
    >>> result = estimate_propensity(T, X)
    >>> result['diagnostics']['pseudo_r2']  # Model fit quality

    Notes
    -----
    - Propensity scores estimated via logistic regression: logit(P(T=1|X)) = β₀ + β'X
    - AUC > 0.7 indicates good discrimination (strong confounding)
    - AUC ≈ 0.5 indicates weak confounding (T nearly independent of X)
    - Pseudo-R² (McFadden): 1 - (log-likelihood / log-likelihood null model)
    - Non-convergence raises error (NEVER FAIL SILENTLY)
    """
    # ============================================================================
    # Input Validation (using shared utilities)
    # ============================================================================

    # Convert to numpy arrays
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Handle 1D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape

    # Shared validations
    validate_not_empty(treatment, "treatment")
    validate_finite(treatment, "treatment")
    validate_finite(covariates, "covariates")
    validate_binary(treatment, "treatment")
    validate_arrays_same_length(treatment=treatment, covariates=covariates[:, 0])
    validate_has_variation(covariates, "covariates", axis=0)

    # Propensity-specific: Check both treated and control groups present
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Treatment has no variation. "
            f"Need both treated and control units. "
            f"Got: n_treated={n_treated}, n_control={n_control}"
        )

    # ============================================================================
    # Propensity Score Estimation
    # ============================================================================

    if method != "logistic":
        raise ValueError(
            f"Unsupported estimation method. "
            f"Expected: method='logistic', got: method='{method}'"
        )

    # Fit logistic regression model
    model = _fit_logistic_model(covariates, treatment, max_iter, random_state)

    # Predict propensity scores
    propensity = model.predict_proba(covariates)[:, 1]  # P(T=1|X)

    # Check for perfect/near separation (raises ValueError if perfect, warns if near)
    _check_separation(propensity)

    # Compute diagnostics (AUC, pseudo-R², etc.)
    diagnostics = _compute_propensity_diagnostics(treatment, propensity, model)

    return {"propensity": propensity, "model": model, "diagnostics": diagnostics}


def trim_propensity(
    propensity: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    outcomes: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    trim_at: Tuple[float, float] = (0.01, 0.99),
) -> Dict[str, np.ndarray]:
    """
    Trim extreme propensity scores by removing units outside percentile range.

    Parameters
    ----------
    propensity : np.ndarray or list
        Propensity scores P(T=1|X).
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control).
    outcomes : np.ndarray or list
        Observed outcomes.
    covariates : np.ndarray or list
        Covariate matrix.
    trim_at : tuple of (float, float), default=(0.01, 0.99)
        Trim at (lower_percentile, upper_percentile).
        Units with propensity outside this range are removed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'propensity': Trimmed propensity scores
        - 'treatment': Trimmed treatment
        - 'outcomes': Trimmed outcomes
        - 'covariates': Trimmed covariates
        - 'keep_mask': Boolean mask of kept units
        - 'n_trimmed': Number of units removed
        - 'n_kept': Number of units kept

    Raises
    ------
    ValueError
        If trim_at invalid or inputs have mismatched lengths.

    Examples
    --------
    >>> propensity = np.array([0.001, 0.2, 0.5, 0.8, 0.999])
    >>> treatment = np.array([0, 0, 1, 1, 1])
    >>> outcomes = np.array([1, 2, 3, 4, 5])
    >>> X = np.random.normal(0, 1, (5, 2))
    >>> result = trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.01, 0.99))
    >>> result['n_trimmed']  # Should remove first and last unit
    2

    Notes
    -----
    - Trimming reduces variance from extreme weights but introduces bias
    - Common practice: trim at 1st/99th percentile (removes ~2% of sample)
    - Propensity near 0 or 1 indicates poor overlap (positivity violation)
    - Always report trimmed and untrimmed results for transparency
    """
    # Convert to numpy arrays
    propensity = np.asarray(propensity, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    n = len(propensity)

    # Check lengths match
    if not (len(treatment) == n and len(outcomes) == n):
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: trim_propensity\n"
            f"Got: len(propensity)={n}, len(treatment)={len(treatment)}, "
            f"len(outcomes)={len(outcomes)}, covariates.shape={covariates.shape}"
        )

    # Check trim_at valid
    if not (0 < trim_at[0] < trim_at[1] < 1):
        raise ValueError(
            f"CRITICAL ERROR: Invalid trim_at parameter.\n"
            f"Function: trim_propensity\n"
            f"Expected: 0 < lower < upper < 1\n"
            f"Got: trim_at={trim_at}"
        )

    # Compute trim thresholds
    lower_threshold = np.percentile(propensity, trim_at[0] * 100)
    upper_threshold = np.percentile(propensity, trim_at[1] * 100)

    # Create keep mask
    keep_mask = (propensity >= lower_threshold) & (propensity <= upper_threshold)

    # Trim arrays
    propensity_trimmed = propensity[keep_mask]
    treatment_trimmed = treatment[keep_mask]
    outcomes_trimmed = outcomes[keep_mask]

    # Handle 1D vs 2D covariates
    if covariates.ndim == 1:
        covariates_trimmed = covariates[keep_mask]
    else:
        covariates_trimmed = covariates[keep_mask, :]

    n_trimmed = n - np.sum(keep_mask)
    n_kept = np.sum(keep_mask)

    return {
        "propensity": propensity_trimmed,
        "treatment": treatment_trimmed,
        "outcomes": outcomes_trimmed,
        "covariates": covariates_trimmed,
        "keep_mask": keep_mask,
        "n_trimmed": int(n_trimmed),
        "n_kept": int(n_kept),
    }


def stabilize_weights(
    propensity: Union[np.ndarray, list], treatment: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Compute stabilized weights: SW = P(T) / P(T|X).

    Stabilized weights reduce variance while maintaining unbiasedness.
    They have mean ≈ 1 and lower variance than non-stabilized IPW weights.

    Parameters
    ----------
    propensity : np.ndarray or list
        Propensity scores P(T=1|X).
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control).

    Returns
    -------
    np.ndarray
        Stabilized weights for each unit.

    Raises
    ------
    ValueError
        If inputs invalid or propensity out of range.

    Examples
    --------
    >>> propensity = np.array([0.2, 0.5, 0.8, 0.9])
    >>> treatment = np.array([0, 1, 1, 1])
    >>> sw = stabilize_weights(propensity, treatment)
    >>> np.mean(sw)  # Should be ≈ 1.0
    1.0

    Notes
    -----
    - For treated: SW = P(T=1) / P(T=1|X)
    - For control: SW = P(T=0) / P(T=0|X) = (1-P(T=1)) / (1-P(T=1|X))
    - Stabilized weights have mean = 1 (by construction)
    - Variance reduction: typically 50-70% vs non-stabilized
    - Maintains unbiasedness if propensity model correct
    """
    # Convert to numpy arrays
    propensity = np.asarray(propensity, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n = len(propensity)

    # Check lengths match
    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: stabilize_weights\n"
            f"Got: len(propensity)={n}, len(treatment)={len(treatment)}"
        )

    # Check propensity in (0,1)
    if np.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError(
            f"CRITICAL ERROR: Propensity scores must be in (0,1) exclusive.\n"
            f"Function: stabilize_weights\n"
            f"Got: min={np.min(propensity)}, max={np.max(propensity)}"
        )

    # Marginal probability of treatment
    p_t = np.mean(treatment)

    # Stabilized weights
    # Treated: SW = P(T=1) / P(T=1|X)
    # Control: SW = P(T=0) / P(T=0|X)
    stabilized_weights = np.where(
        treatment == 1,
        p_t / propensity,  # Treated
        (1 - p_t) / (1 - propensity),  # Control
    )

    return stabilized_weights
