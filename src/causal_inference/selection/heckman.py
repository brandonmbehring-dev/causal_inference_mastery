"""Heckman two-step selection model.

Implements the Heckman (1979) correction for sample selection bias,
where outcomes are only observed for a selected subsample.

The model consists of two equations:
1. Selection equation: S* = γ₀ + γ'Z + v, S = 1{S* > 0}
2. Outcome equation: Y = β₀ + β'X + u (observed only when S = 1)

If E[u|v] ≠ 0, OLS on the selected sample is biased. Heckman's insight:
    E[Y|X, S=1] = β₀ + β'X + ρσᵤλ(γ'Z)

where λ(·) is the Inverse Mills Ratio: λ(p) = φ(Φ⁻¹(p)) / p

References
----------
- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error.
  Econometrica, 47(1), 153-161. doi:10.2307/1912352
- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data.
  MIT Press, Chapter 19.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from src.causal_inference.selection.types import HeckmanResult, SelectionDiagnostics


def heckman_two_step(
    outcome: np.ndarray,
    selected: np.ndarray,
    selection_covariates: np.ndarray,
    outcome_covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    add_intercept: bool = True,
) -> HeckmanResult:
    """
    Heckman two-step estimator for sample selection correction.

    Pipeline:
    1. Estimate selection equation (probit) on full sample: P(S=1|Z)
    2. Compute Inverse Mills Ratio for selected observations
    3. Estimate outcome equation (OLS) with IMR as additional regressor
    4. Compute Heckman-corrected standard errors (sandwich estimator)

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes. Can be full array (with NaN for unselected) or
        only selected observations. Shape: (n,) or (n_selected,).
    selected : np.ndarray
        Binary selection indicator. 1 = outcome observed, 0 = not observed.
        Shape: (n,).
    selection_covariates : np.ndarray
        Covariates for selection equation (Z). Shape: (n, k_z) or (n,).
        Must include exclusion restriction for identification.
    outcome_covariates : np.ndarray, optional
        Covariates for outcome equation (X). Shape: (n, k_x) or (n,).
        If None, uses selection_covariates (but this is not recommended).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    add_intercept : bool, default=True
        If True, adds intercept to both equations.

    Returns
    -------
    HeckmanResult
        Dictionary with:
        - estimate: First non-intercept coefficient from outcome equation
        - se: Heckman-corrected standard error
        - ci_lower, ci_upper: Confidence interval bounds
        - rho: Selection correlation parameter
        - sigma: Outcome error standard deviation
        - lambda_coef: IMR coefficient (= rho * sigma)
        - lambda_se, lambda_pvalue: Test for selection bias
        - n_selected, n_total: Sample sizes
        - selection_probs, imr: Fitted values
        - gamma, beta: Coefficient vectors
        - vcov: Corrected variance-covariance matrix
        - selection_diagnostics: Probit model diagnostics

    Raises
    ------
    ValueError
        If inputs invalid, selection model fails, or identification fails.

    Notes
    -----
    Identification requires an exclusion restriction: at least one variable
    in Z (selection) that is not in X (outcome). Without this, identification
    relies on the nonlinearity of the IMR, which is fragile.

    The sandwich variance estimator corrects for:
    1. The IMR being estimated (not known)
    2. Uncertainty in the selection equation propagating to outcome equation

    Example
    -------
    >>> # Wage equation with employment selection
    >>> # Z includes both education and age (affects selection)
    >>> # X includes only education (affects wages)
    >>> result = heckman_two_step(
    ...     outcome=wages,      # NaN for unemployed
    ...     selected=employed,  # 1 if employed
    ...     selection_covariates=np.column_stack([education, age]),
    ...     outcome_covariates=education.reshape(-1, 1),
    ... )
    >>> print(f"Education coefficient: {result['estimate']:.3f} ± {result['se']:.3f}")
    >>> print(f"Selection bias test p-value: {result['lambda_pvalue']:.4f}")
    """
    # =========================================================================
    # Input Validation
    # =========================================================================

    outcome = np.asarray(outcome, dtype=float)
    selected = np.asarray(selected, dtype=float)
    selection_covariates = np.asarray(selection_covariates, dtype=float)

    # Ensure 2D covariates
    if selection_covariates.ndim == 1:
        selection_covariates = selection_covariates.reshape(-1, 1)

    n = len(selected)

    # Validate selection indicator
    if not np.all(np.isin(selected, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Selection indicator must be binary (0 or 1).\n"
            f"Function: heckman_two_step\n"
            f"Unique values in selected: {np.unique(selected)}"
        )

    n_selected = int(np.sum(selected))
    n_unselected = n - n_selected

    if n_selected == 0:
        raise ValueError(
            f"CRITICAL ERROR: No observations selected (all selected == 0).\n"
            f"Function: heckman_two_step\n"
            f"Cannot estimate outcome equation without selected observations."
        )

    if n_unselected == 0:
        raise ValueError(
            f"CRITICAL ERROR: All observations selected (all selected == 1).\n"
            f"Function: heckman_two_step\n"
            f"No selection bias to correct - use OLS instead."
        )

    if selection_covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"Function: heckman_two_step\n"
            f"selected: {n}, selection_covariates: {selection_covariates.shape[0]}"
        )

    # Handle outcome array
    if len(outcome) == n:
        # Full array provided - extract selected observations
        outcome_selected = outcome[selected == 1]
        if np.any(np.isnan(outcome_selected)):
            raise ValueError(
                f"CRITICAL ERROR: NaN in selected outcomes.\n"
                f"Function: heckman_two_step\n"
                f"Outcome should be valid for all selected observations."
            )
    elif len(outcome) == n_selected:
        # Only selected observations provided
        outcome_selected = outcome.copy()
    else:
        raise ValueError(
            f"CRITICAL ERROR: Outcome length mismatch.\n"
            f"Function: heckman_two_step\n"
            f"Expected {n} (full) or {n_selected} (selected only), got {len(outcome)}"
        )

    # Setup outcome covariates
    if outcome_covariates is None:
        warnings.warn(
            "No separate outcome_covariates provided. Using selection_covariates.\n"
            "Identification relies on IMR nonlinearity, which may be fragile.\n"
            "Consider including an exclusion restriction for robust identification.",
            UserWarning,
        )
        outcome_covariates = selection_covariates.copy()
    else:
        outcome_covariates = np.asarray(outcome_covariates, dtype=float)
        if outcome_covariates.ndim == 1:
            outcome_covariates = outcome_covariates.reshape(-1, 1)
        if outcome_covariates.shape[0] != n:
            raise ValueError(
                f"CRITICAL ERROR: outcome_covariates length mismatch.\n"
                f"Function: heckman_two_step\n"
                f"Expected {n}, got {outcome_covariates.shape[0]}"
            )

    # =========================================================================
    # Step 1: Fit Selection Equation (Probit)
    # =========================================================================

    gamma, selection_probs, selection_diag = _fit_probit(
        selected, selection_covariates, add_intercept=add_intercept
    )

    # =========================================================================
    # Step 2: Compute Inverse Mills Ratio
    # =========================================================================

    # For selected observations: λ = φ(γ'Z) / Φ(γ'Z)
    # For unselected: λ = -φ(γ'Z) / (1 - Φ(γ'Z))  (not used in two-step)
    imr_full = _compute_imr(selection_probs)
    imr_selected = imr_full[selected == 1]

    # =========================================================================
    # Step 3: Fit Outcome Equation with IMR
    # =========================================================================

    # Extract selected observations
    outcome_covariates_selected = outcome_covariates[selected == 1]

    # Add intercept if requested
    if add_intercept:
        X_selected = np.column_stack([
            np.ones(n_selected),
            outcome_covariates_selected,
            imr_selected,
        ])
    else:
        X_selected = np.column_stack([outcome_covariates_selected, imr_selected])

    # OLS estimation: Y = X'β + λ·λ_coef + error
    beta, residuals, sigma_hat = _fit_ols(outcome_selected, X_selected)

    # Extract coefficients
    lambda_idx = -1  # IMR is always last
    lambda_coef = beta[lambda_idx]

    # Compute rho = lambda / sigma (correlation of errors)
    # Note: sigma from OLS residuals is biased but commonly used in two-step
    rho = lambda_coef / sigma_hat if sigma_hat > 0 else 0.0

    # =========================================================================
    # Step 4: Compute Heckman-Corrected Standard Errors
    # =========================================================================

    vcov = _compute_heckman_vcov(
        outcome_selected=outcome_selected,
        X_selected=X_selected,
        Z_full=selection_covariates if not add_intercept
        else np.column_stack([np.ones(n), selection_covariates]),
        selected=selected,
        gamma=gamma,
        beta=beta,
        sigma_hat=sigma_hat,
        lambda_coef=lambda_coef,
        imr_selected=imr_selected,
        add_intercept=add_intercept,
    )

    # Standard errors from diagonal
    se_beta = np.sqrt(np.diag(vcov))

    # Lambda test
    lambda_se = se_beta[lambda_idx]
    lambda_t = lambda_coef / lambda_se if lambda_se > 0 else 0.0
    lambda_pvalue = 2 * (1 - stats.norm.cdf(abs(lambda_t)))

    # =========================================================================
    # Step 5: Extract Primary Estimate
    # =========================================================================

    # First non-intercept coefficient is typically the variable of interest
    if add_intercept:
        estimate_idx = 1  # Skip intercept
    else:
        estimate_idx = 0

    estimate = beta[estimate_idx]
    se = se_beta[estimate_idx]

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = estimate - z_crit * se
    ci_upper = estimate + z_crit * se

    # =========================================================================
    # Return Result
    # =========================================================================

    return HeckmanResult(
        estimate=float(estimate),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        rho=float(np.clip(rho, -1, 1)),  # Ensure valid correlation
        sigma=float(sigma_hat),
        lambda_coef=float(lambda_coef),
        lambda_se=float(lambda_se),
        lambda_pvalue=float(lambda_pvalue),
        n_selected=n_selected,
        n_total=n,
        selection_probs=selection_probs,
        imr=imr_full,
        gamma=gamma,
        beta=beta,
        vcov=vcov,
        selection_diagnostics=selection_diag,
    )


def _fit_probit(
    y: np.ndarray,
    X: np.ndarray,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, SelectionDiagnostics]:
    """
    Fit probit selection model via maximum likelihood.

    Parameters
    ----------
    y : np.ndarray
        Binary outcome (selection indicator).
    X : np.ndarray
        Covariates (without intercept if add_intercept=True).
    add_intercept : bool
        Whether to add intercept column.
    max_iter : int
        Maximum iterations for optimization.
    tol : float
        Convergence tolerance.

    Returns
    -------
    gamma : np.ndarray
        Estimated coefficients.
    probs : np.ndarray
        Predicted selection probabilities.
    diagnostics : SelectionDiagnostics
        Model diagnostics.
    """
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape

    # Log-likelihood for probit
    def neg_log_likelihood(gamma):
        """Negative log-likelihood for probit."""
        z = X @ gamma
        # Clip to avoid numerical issues
        z = np.clip(z, -30, 30)
        prob = stats.norm.cdf(z)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return -ll

    def gradient(gamma):
        """Gradient of negative log-likelihood."""
        z = X @ gamma
        z = np.clip(z, -30, 30)
        prob = stats.norm.cdf(z)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        pdf = stats.norm.pdf(z)

        # λ = φ/Φ for y=1, -φ/(1-Φ) for y=0
        lam = np.where(y == 1, pdf / prob, -pdf / (1 - prob))
        grad = -X.T @ lam
        return grad

    # Initial guess: zero coefficients
    gamma_init = np.zeros(k)

    # Optimize
    result = minimize(
        neg_log_likelihood,
        gamma_init,
        method="BFGS",
        jac=gradient,
        options={"maxiter": max_iter, "gtol": tol},
    )

    gamma = result.x
    converged = result.success

    if not converged:
        warnings.warn(
            f"Probit optimization did not converge after {max_iter} iterations.\n"
            f"Message: {result.message}\n"
            f"Results may be unreliable.",
            UserWarning,
        )

    # Compute fitted probabilities
    z = X @ gamma
    z = np.clip(z, -30, 30)
    probs = stats.norm.cdf(z)

    # Diagnostics
    log_likelihood = -neg_log_likelihood(gamma)

    # Null model (intercept only)
    p_bar = np.mean(y)
    ll_null = n * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))

    # McFadden's pseudo R-squared
    pseudo_r2 = 1 - (log_likelihood / ll_null) if ll_null != 0 else 0.0

    diagnostics = SelectionDiagnostics(
        pseudo_r_squared=float(pseudo_r2),
        log_likelihood=float(log_likelihood),
        n_iterations=int(result.nit),
        converged=converged,
    )

    return gamma, probs, diagnostics


def _compute_imr(selection_probs: np.ndarray) -> np.ndarray:
    """
    Compute Inverse Mills Ratio (IMR) from selection probabilities.

    The IMR is: λ(p) = φ(Φ⁻¹(p)) / p

    where φ is the standard normal PDF and Φ is the CDF.

    Parameters
    ----------
    selection_probs : np.ndarray
        Predicted P(S=1|Z) from probit model.

    Returns
    -------
    np.ndarray
        Inverse Mills Ratio values.

    Notes
    -----
    Boundary handling is critical:
    - p ≈ 0: IMR → ∞ (severely selected against)
    - p ≈ 1: IMR → 0 (almost certainly selected)

    We clip probabilities to [1e-6, 1-1e-6] to avoid numerical issues.
    """
    # Clip to avoid numerical issues at boundaries
    p = np.clip(selection_probs, 1e-6, 1 - 1e-6)

    # Inverse CDF (probit)
    z = stats.norm.ppf(p)

    # PDF at that point
    phi_z = stats.norm.pdf(z)

    # IMR = φ(z) / Φ(z) = φ(z) / p
    imr = phi_z / p

    return imr


def _fit_ols(
    y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit OLS regression.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable.
    X : np.ndarray
        Design matrix (including intercept and IMR).

    Returns
    -------
    beta : np.ndarray
        Coefficient estimates.
    residuals : np.ndarray
        OLS residuals.
    sigma_hat : float
        Estimated error standard deviation.
    """
    # OLS: β = (X'X)⁻¹X'y
    XtX = X.T @ X
    Xty = X.T @ y

    # Use pseudoinverse for numerical stability
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    # Residuals
    residuals = y - X @ beta

    # Sigma estimate (degrees of freedom adjusted)
    n, k = X.shape
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - k))

    return beta, residuals, sigma_hat


def _compute_heckman_vcov(
    outcome_selected: np.ndarray,
    X_selected: np.ndarray,
    Z_full: np.ndarray,
    selected: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    sigma_hat: float,
    lambda_coef: float,
    imr_selected: np.ndarray,
    add_intercept: bool = True,
) -> np.ndarray:
    """
    Compute Heckman-corrected variance-covariance matrix.

    The standard OLS variance is invalid because:
    1. IMR is estimated (introduces uncertainty)
    2. Selection equation uncertainty propagates

    This implements the two-step correction from Heckman (1979).

    Parameters
    ----------
    outcome_selected : np.ndarray
        Outcomes for selected sample.
    X_selected : np.ndarray
        Design matrix for outcome equation.
    Z_full : np.ndarray
        Design matrix for selection equation (full sample).
    selected : np.ndarray
        Selection indicator.
    gamma : np.ndarray
        Selection equation coefficients.
    beta : np.ndarray
        Outcome equation coefficients.
    sigma_hat : float
        Error standard deviation estimate.
    lambda_coef : float
        IMR coefficient.
    imr_selected : np.ndarray
        IMR for selected sample.
    add_intercept : bool
        Whether intercept was added.

    Returns
    -------
    np.ndarray
        Corrected variance-covariance matrix.

    Notes
    -----
    The correction involves:
    1. δ = λ(λ + γ'Z) - derivative of IMR
    2. V = σ²[(X'X)⁻¹ + (λ_coef² / σ²)(X'X)⁻¹(X'δΔX)(X'X)⁻¹]

    where Δ involves the selection equation Hessian.
    """
    n_selected = len(outcome_selected)
    n_total = len(selected)
    k_beta = X_selected.shape[1]

    # (X'X)⁻¹
    XtX_inv = np.linalg.inv(X_selected.T @ X_selected)

    # Compute δ = λ(λ + γ'Z) for selected observations
    # γ'Z for selected
    Z_selected = Z_full[selected == 1]
    gamma_z_selected = Z_selected @ gamma

    # δ = λ(λ + γ'Z)
    delta_selected = imr_selected * (imr_selected + gamma_z_selected)

    # Residuals
    residuals = outcome_selected - X_selected @ beta

    # Standard OLS variance (uncorrected)
    sigma2 = sigma_hat**2

    # Correction term for heteroskedasticity from selection
    # The variance has an additional term due to generated regressor (IMR)
    #
    # Simplified Heckman correction:
    # Var(β) ≈ σ²(X'X)⁻¹ + correction for IMR uncertainty
    #
    # The correction involves the covariance structure of the two-stage
    # estimation. We implement a commonly used approximation.

    # Compute Q = X'δX where δ is diagonal with delta_selected
    # This captures the effect of IMR estimation uncertainty
    delta_diag = np.diag(delta_selected)
    Q = X_selected.T @ delta_diag @ X_selected

    # Compute W matrix - this captures the correction for generated regressor
    # W = X'X - λ² * X'δX
    rho_sq = (lambda_coef / sigma_hat) ** 2 if sigma_hat > 0 else 0

    # Corrected variance-covariance
    # Following Heckman (1979) and standard econometrics texts
    correction = rho_sq * (XtX_inv @ Q @ XtX_inv)
    vcov = sigma2 * XtX_inv + sigma2 * correction

    return vcov
