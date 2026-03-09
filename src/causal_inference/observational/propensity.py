"""Propensity score estimation and weight adjustments for observational studies.

This module implements propensity score estimation using logistic regression,
along with weight trimming and stabilization utilities.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Tuple, Union, Optional, TypedDict

from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
    validate_binary,
    validate_not_empty,
    validate_has_variation,
)

from .propensity_helpers import (
    fit_logistic_model,
    check_separation,
    compute_propensity_diagnostics,
)


class PropensityDiagnostics(TypedDict):
    """Diagnostic statistics for propensity score model."""

    auc: float
    pseudo_r2: float
    converged: bool
    n_iter: int
    coef: np.ndarray
    intercept: float


class PropensityResult(TypedDict):
    """Return type for estimate_propensity() function."""

    propensity: np.ndarray
    model: LogisticRegression
    diagnostics: PropensityDiagnostics


def estimate_propensity(
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    method: str = "logistic",
    max_iter: int = 1000,
    random_state: int = None,
) -> PropensityResult:
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
    n_treated: int = int(np.sum(treatment == 1))
    n_control: int = int(np.sum(treatment == 0))
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
            f"Unsupported estimation method. Expected: method='logistic', got: method='{method}'"
        )

    # Fit logistic regression model
    model = fit_logistic_model(covariates, treatment, max_iter, random_state)

    # Predict propensity scores
    propensity = model.predict_proba(covariates)[:, 1]  # P(T=1|X)

    # Check for perfect/near separation (raises ValueError if perfect, warns if near)
    check_separation(propensity)

    # Compute diagnostics (AUC, pseudo-R², etc.)
    diagnostics = compute_propensity_diagnostics(treatment, propensity, model)

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
    n_kept: int = int(np.sum(keep_mask))

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
