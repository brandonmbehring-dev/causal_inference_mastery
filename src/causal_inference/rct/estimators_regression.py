"""Regression-adjusted ATE estimator using ANCOVA.

This module implements regression adjustment (ANCOVA) for RCTs, which improves
precision by controlling for pre-treatment covariates.

Key benefit: Reduces variance without biasing the estimate under randomization.
"""

import numpy as np
from scipy import stats
from typing import Dict, Union, List
import warnings


def regression_adjusted_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate regression-adjusted Average Treatment Effect (ANCOVA).

    Fits the regression: Y = intercept + tau*T + beta*X + epsilon
    where tau is the ATE. Under randomization, this is unbiased and more
    efficient than simple difference-in-means when covariates predict outcomes.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control). Also accepts boolean.
    covariates : np.ndarray or list
        Pre-treatment covariates. Can be 1D (single covariate) or 2D (multiple).
    alpha : float, default=0.05
        Significance level for confidence interval (must be in (0, 1)).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': Regression-adjusted ATE (treatment coefficient)
        - 'se': Standard error (heteroskedasticity-robust)
        - 'ci_lower': Lower bound of (1-alpha)% CI
        - 'ci_upper': Upper bound of (1-alpha)% CI
        - 'intercept': Regression intercept
        - 'covariate_coef': Coefficient(s) for covariate(s)
        - 'n_treated': Number of treated units
        - 'n_control': Number of control units
        - 'r_squared': R-squared of regression

    Raises
    ------
    ValueError
        If inputs invalid (mismatched lengths, NaN, inf, etc.)

    Examples
    --------
    >>> # Single covariate
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    >>> treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    >>> outcomes = 2 + 5*treatment + 3*X
    >>> result = regression_adjusted_ate(outcomes, treatment, X)
    >>> result['estimate']  # Should recover ATE=5
    5.0

    Notes
    -----
    - Model: Y = intercept + tau*T + beta*X + epsilon
    - tau is the ATE (treatment coefficient)
    - Standard errors are heteroskedasticity-robust (HC3)
    - More efficient than simple_ate when X predicts Y
    - Unbiased under randomization (even if X imbalanced)
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    # Convert to numpy arrays
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Ensure covariates is 2D (n x p)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)
    p = covariates.shape[1]

    # Check lengths match
    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: regression_adjusted_ate\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}"
        )

    if covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Covariate rows don't match outcome length.\n"
            f"Function: regression_adjusted_ate\n"
            f"Expected: covariates with {n} rows\n"
            f"Got: covariates with {covariates.shape[0]} rows"
        )

    # Check for empty
    if n == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: regression_adjusted_ate\n"
            f"Expected: Non-empty arrays"
        )

    # Check for NaN
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)) or np.any(np.isnan(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: regression_adjusted_ate\n"
            f"NaN indicates data quality issues that must be addressed.\n"
            f"Got: {np.sum(np.isnan(outcomes))} NaN in outcomes, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment, "
            f"{np.sum(np.isnan(covariates))} NaN in covariates"
        )

    # Check for infinite values
    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)) or np.any(np.isinf(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: regression_adjusted_ate\n"
            f"Got: {np.sum(np.isinf(outcomes))} inf in outcomes, "
            f"{np.sum(np.isinf(treatment))} inf in treatment, "
            f"{np.sum(np.isinf(covariates))} inf in covariates"
        )

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: regression_adjusted_ate\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # Check for treatment variation
    if len(unique_treatment) < 2:
        if unique_treatment[0] == 1:
            raise ValueError(
                f"CRITICAL ERROR: No control units in data.\n"
                f"Function: regression_adjusted_ate\n"
                f"Cannot estimate treatment effect without control group.\n"
                f"Got: All units have treatment=1"
            )
        else:
            raise ValueError(
                f"CRITICAL ERROR: No treated units in data.\n"
                f"Function: regression_adjusted_ate\n"
                f"Cannot estimate treatment effect without treated group.\n"
                f"Got: All units have treatment=0"
            )

    # Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\n"
            f"Function: regression_adjusted_ate\n"
            f"Expected: alpha in (0, 1)\n"
            f"Got: alpha={alpha}"
        )

    # ============================================================================
    # Regression Estimation
    # ============================================================================

    # Build design matrix: [1, T, X]
    # Intercept column
    intercept_col = np.ones((n, 1))
    treatment_col = treatment.reshape(-1, 1)

    # Full design matrix
    X_design = np.hstack([intercept_col, treatment_col, covariates])

    # Fit OLS: Y = X_design * beta + epsilon
    # beta = [intercept, tau, beta_1, ..., beta_p]
    # We want tau (index 1)

    try:
        # Solve normal equations: (X'X)^-1 X'Y
        XtX = X_design.T @ X_design
        XtY = X_design.T @ outcomes
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"CRITICAL ERROR: Singular design matrix.\n"
            f"Function: regression_adjusted_ate\n"
            f"This may be due to perfect collinearity in covariates.\n"
            f"Check for duplicate or linearly dependent columns."
        )

    # Extract coefficients
    intercept = beta[0]
    tau = beta[1]  # ATE (treatment coefficient)
    beta_covariates = beta[2:]  # Covariate coefficients

    # Residuals
    y_fitted = X_design @ beta
    residuals = outcomes - y_fitted

    # R-squared
    ss_total = np.sum((outcomes - np.mean(outcomes)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    # ============================================================================
    # Heteroskedasticity-Robust Standard Errors (HC3)
    # ============================================================================

    # HC3 variance: (X'X)^-1 X' diag(e_i^2 / (1-h_i)^2) X (X'X)^-1
    # where h_i is the leverage (diagonal of hat matrix)

    # Hat matrix diagonal (leverage)
    XtX_inv = np.linalg.inv(XtX)
    H_diag = np.sum((X_design @ XtX_inv) * X_design, axis=1)  # Diagonal of H = X(X'X)^-1X'

    # HC3 adjustment
    hc3_weights = residuals ** 2 / (1 - H_diag) ** 2

    # Variance-covariance matrix
    Sigma = XtX_inv @ (X_design.T @ np.diag(hc3_weights) @ X_design) @ XtX_inv

    # Standard error for tau (index 1)
    se_tau = np.sqrt(Sigma[1, 1])

    # Confidence interval
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = tau - z_critical * se_tau
    ci_upper = tau + z_critical * se_tau

    # Total counts
    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    return {
        "estimate": tau,
        "se": se_tau,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "intercept": intercept,
        "covariate_coef": beta_covariates.tolist() if p > 1 else beta_covariates[0],
        "n_treated": n_treated,
        "n_control": n_control,
        "r_squared": r_squared,
    }
