"""Outcome regression models for doubly robust estimation.

This module implements outcome regression modeling by fitting separate models
for treated (T=1) and control (T=0) units to estimate E[Y|T, X].
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, Union


def fit_outcome_models(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    method: str = "linear",
) -> Dict[str, Any]:
    """
    Fit separate outcome models for treated and control units.

    Fits two models:
    - μ₁(X) = E[Y|T=1, X] using treated units
    - μ₀(X) = E[Y|T=0, X] using control units

    Then predicts on all covariates (not just own treatment group).

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Binary treatment indicator (1=treated, 0=control).
    covariates : np.ndarray or list
        Covariate matrix (n_samples, n_features). Can be 1D for single covariate.
    method : str, default="linear"
        Modeling method. Currently only "linear" supported.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mu0_model': Fitted LinearRegression for control
        - 'mu1_model': Fitted LinearRegression for treated
        - 'mu0_predictions': E[Y|T=0, X] for all X (n,)
        - 'mu1_predictions': E[Y|T=1, X] for all X (n,)
        - 'diagnostics': dict with:
            - 'mu0_r2': R² for control model (in-sample)
            - 'mu1_r2': R² for treated model (in-sample)
            - 'mu0_rmse': RMSE for control model
            - 'mu1_rmse': RMSE for treated model
            - 'n_control': Number of control units
            - 'n_treated': Number of treated units

    Raises
    ------
    ValueError
        If inputs invalid, insufficient data, or model fails to fit.

    Examples
    --------
    >>> # Simple linear outcome model
    >>> X = np.random.normal(0, 1, 200)
    >>> T = np.array([1] * 100 + [0] * 100)
    >>> Y = 3.0*T + 0.5*X + np.random.normal(0, 1, 200)
    >>> result = fit_outcome_models(Y, T, X)
    >>> result['mu1_predictions'] - result['mu0_predictions']  # Should be near 3.0

    Notes
    -----
    - Fits separate models to allow different covariate effects by treatment group
    - Predicts on all covariates (needed for doubly robust estimation)
    - Returns in-sample diagnostics (R², RMSE) for model quality assessment
    - Linear regression assumes linear relationship Y = β₀ + β'X + ε
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Handle 1D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Check lengths match
    if len(treatment) != n or covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Input arrays have different lengths.\\n"
            f"Function: fit_outcome_models\\n"
            f"Expected: All arrays length {n}\\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"covariates.shape={covariates.shape}"
        )

    # Check for NaN/inf
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)) or np.any(np.isnan(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\\n"
            f"Function: fit_outcome_models\\n"
            f"NaN indicates data quality issues that must be addressed."
        )

    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)) or np.any(np.isinf(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\\nFunction: fit_outcome_models"
        )

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\\n"
            f"Function: fit_outcome_models\\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # Check sufficient data in each group
    n_control = np.sum(treatment == 0)
    n_treated = np.sum(treatment == 1)

    if n_control < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient control units for outcome modeling.\\n"
            f"Function: fit_outcome_models\\n"
            f"Expected: At least 2 control units\\n"
            f"Got: {n_control} control units"
        )

    if n_treated < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient treated units for outcome modeling.\\n"
            f"Function: fit_outcome_models\\n"
            f"Expected: At least 2 treated units\\n"
            f"Got: {n_treated} treated units"
        )

    # ============================================================================
    # Fit Outcome Models
    # ============================================================================

    if method != "linear":
        raise ValueError(
            f"CRITICAL ERROR: Unsupported modeling method.\\n"
            f"Function: fit_outcome_models\\n"
            f"Expected: method='linear'\\n"
            f"Got: method='{method}'"
        )

    # Separate data by treatment
    control_mask = treatment == 0
    treated_mask = treatment == 1

    Y_control = outcomes[control_mask]
    X_control = covariates[control_mask]

    Y_treated = outcomes[treated_mask]
    X_treated = covariates[treated_mask]

    # Fit control model: μ₀(X) = E[Y|T=0, X]
    mu0_model = LinearRegression()
    try:
        mu0_model.fit(X_control, Y_control)
    except Exception as e:
        raise ValueError(
            f"CRITICAL ERROR: Control outcome model failed to fit.\\n"
            f"Function: fit_outcome_models\\n"
            f"Error: {str(e)}"
        )

    # Fit treated model: μ₁(X) = E[Y|T=1, X]
    mu1_model = LinearRegression()
    try:
        mu1_model.fit(X_treated, Y_treated)
    except Exception as e:
        raise ValueError(
            f"CRITICAL ERROR: Treated outcome model failed to fit.\\n"
            f"Function: fit_outcome_models\\n"
            f"Error: {str(e)}"
        )

    # ============================================================================
    # Predict on All Covariates
    # ============================================================================

    # Predict on ALL covariates (needed for DR estimation)
    mu0_predictions = mu0_model.predict(covariates)
    mu1_predictions = mu1_model.predict(covariates)

    # ============================================================================
    # Compute Diagnostics
    # ============================================================================

    # In-sample R² and RMSE
    mu0_fitted = mu0_model.predict(X_control)
    mu1_fitted = mu1_model.predict(X_treated)

    mu0_r2 = r2_score(Y_control, mu0_fitted)
    mu1_r2 = r2_score(Y_treated, mu1_fitted)

    mu0_rmse = np.sqrt(mean_squared_error(Y_control, mu0_fitted))
    mu1_rmse = np.sqrt(mean_squared_error(Y_treated, mu1_fitted))

    diagnostics = {
        "mu0_r2": float(mu0_r2),
        "mu1_r2": float(mu1_r2),
        "mu0_rmse": float(mu0_rmse),
        "mu1_rmse": float(mu1_rmse),
        "n_control": int(n_control),
        "n_treated": int(n_treated),
    }

    return {
        "mu0_model": mu0_model,
        "mu1_model": mu1_model,
        "mu0_predictions": mu0_predictions,
        "mu1_predictions": mu1_predictions,
        "diagnostics": diagnostics,
    }
