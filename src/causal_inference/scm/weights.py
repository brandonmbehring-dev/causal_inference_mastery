"""
Synthetic Control Methods - Weight Optimization

Implements constrained optimization to find synthetic control weights:
    minimize ||Y₁_pre - Y₀_pre @ W||²
    subject to: W >= 0, sum(W) = 1

References:
    Abadie, Diamond, & Hainmueller (2010). "Synthetic Control Methods
    for Comparative Case Studies"
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult


def compute_scm_weights(
    treated_pre: np.ndarray,
    control_pre: np.ndarray,
    covariates_treated: Optional[np.ndarray] = None,
    covariates_control: Optional[np.ndarray] = None,
    covariate_weight: float = 1.0,
    method: str = "SLSQP",
    max_iter: int = 1000,
) -> Tuple[np.ndarray, OptimizeResult]:
    """
    Compute optimal synthetic control weights via constrained optimization.

    Solves the quadratic program:
        minimize ||Y₁_pre - Y₀_pre @ W||² + λ||X₁ - X₀ @ W||²
        subject to: W >= 0, sum(W) = 1

    Parameters
    ----------
    treated_pre : np.ndarray
        Pre-treatment outcomes for treated unit(s), shape (n_pre_periods,)
        or (n_treated, n_pre_periods) for multiple treated units
    control_pre : np.ndarray
        Pre-treatment outcomes for control units, shape (n_control, n_pre_periods)
    covariates_treated : np.ndarray, optional
        Pre-treatment covariates for treated, shape (n_covariates,)
    covariates_control : np.ndarray, optional
        Pre-treatment covariates for controls, shape (n_control, n_covariates)
    covariate_weight : float
        Relative weight on covariate matching vs outcome matching
    method : str
        Optimization method (default: "SLSQP" for constrained optimization)
    max_iter : int
        Maximum optimization iterations

    Returns
    -------
    weights : np.ndarray
        Optimal weights for control units, shape (n_control,)
    result : OptimizeResult
        Full optimization result from scipy

    Raises
    ------
    ValueError
        If dimensions mismatch or optimization fails
    """
    # Handle 1D treated case (single treated unit)
    if treated_pre.ndim == 1:
        treated_pre = treated_pre.reshape(1, -1)

    n_treated, n_pre = treated_pre.shape
    n_control, n_pre_control = control_pre.shape

    if n_pre != n_pre_control:
        raise ValueError(
            f"Pre-period mismatch: treated has {n_pre}, control has {n_pre_control}"
        )

    # For multiple treated units, use average (standard approach)
    treated_avg = treated_pre.mean(axis=0) if n_treated > 1 else treated_pre.flatten()

    # Build objective function
    def objective(w: np.ndarray) -> float:
        """Squared prediction error."""
        synthetic = control_pre.T @ w  # (n_pre,)
        outcome_error = np.sum((treated_avg - synthetic) ** 2)

        # Add covariate matching if provided
        covariate_error = 0.0
        if covariates_treated is not None and covariates_control is not None:
            cov_treated = covariates_treated.flatten()
            cov_synthetic = covariates_control.T @ w
            covariate_error = covariate_weight * np.sum((cov_treated - cov_synthetic) ** 2)

        return outcome_error + covariate_error

    def gradient(w: np.ndarray) -> np.ndarray:
        """Gradient of objective."""
        synthetic = control_pre.T @ w
        residual = synthetic - treated_avg

        # Gradient: 2 * X₀ @ (X₀ᵀW - Y₁)
        grad = 2.0 * control_pre @ residual

        # Add covariate gradient if provided
        if covariates_treated is not None and covariates_control is not None:
            cov_treated = covariates_treated.flatten()
            cov_synthetic = covariates_control.T @ w
            cov_residual = cov_synthetic - cov_treated
            grad += 2.0 * covariate_weight * covariates_control @ cov_residual

        return grad

    # Constraints: simplex (non-negative, sum to 1)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n_control)}
    ]

    # Bounds: w_i >= 0
    bounds = [(0.0, 1.0) for _ in range(n_control)]

    # Initial weights: uniform
    w0 = np.ones(n_control) / n_control

    # Optimize
    result = minimize(
        objective,
        w0,
        method=method,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    if not result.success:
        # Try alternative optimization
        result = minimize(
            objective,
            w0,
            method="trust-constr",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter},
        )

    # Clean up weights: project small negatives to 0, renormalize
    weights = np.maximum(result.x, 0.0)
    weights = weights / weights.sum()

    return weights, result


def compute_scm_weights_ols(
    treated_pre: np.ndarray,
    control_pre: np.ndarray,
) -> np.ndarray:
    """
    Compute unconstrained OLS weights (for comparison/diagnostics).

    Solves: W = argmin ||Y₁_pre - Y₀_pre @ W||²

    This is the unconstrained version - weights may be negative or > 1.
    Useful for checking if simplex constraint is binding.

    Parameters
    ----------
    treated_pre : np.ndarray
        Pre-treatment outcomes for treated, shape (n_pre_periods,)
    control_pre : np.ndarray
        Pre-treatment outcomes for controls, shape (n_control, n_pre_periods)

    Returns
    -------
    weights : np.ndarray
        OLS weights (unconstrained)
    """
    # OLS: W = (X₀X₀ᵀ)⁻¹ X₀ Y₁
    # Where X₀ is (n_control, n_pre)
    # We want to solve X₀ᵀ @ W = Y₁

    treated_1d = treated_pre.flatten()

    # Solve normal equations
    XtX = control_pre @ control_pre.T  # (n_control, n_control)
    Xty = control_pre @ treated_1d  # (n_control,)

    try:
        weights = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    return weights


def compute_pre_treatment_fit(
    treated_pre: np.ndarray,
    control_pre: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute pre-treatment fit statistics.

    Parameters
    ----------
    treated_pre : np.ndarray
        Pre-treatment outcomes for treated, shape (n_pre_periods,)
    control_pre : np.ndarray
        Pre-treatment outcomes for controls, shape (n_control, n_pre_periods)
    weights : np.ndarray
        Synthetic control weights, shape (n_control,)

    Returns
    -------
    rmse : float
        Root mean squared error of pre-treatment fit
    r_squared : float
        R-squared of pre-treatment fit (1 = perfect, 0 = mean baseline)
    """
    treated_1d = treated_pre.flatten()
    synthetic = control_pre.T @ weights  # (n_pre,)

    residuals = treated_1d - synthetic
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((treated_1d - treated_1d.mean()) ** 2)

    rmse = np.sqrt(ss_res / len(treated_1d))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return rmse, r_squared
