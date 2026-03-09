"""
Marginal Treatment Effects via Local Instrumental Variables.

Implements Heckman & Vytlacil (1999, 2005) framework for estimating
treatment effect heterogeneity indexed by unobserved resistance.

MTE(u) = ∂E[Y|P(Z)=p]/∂p evaluated at u = p
"""

from typing import Optional, Union, List, Literal
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from .types import MTEResult


def local_iv(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    instrument: Union[np.ndarray, List[float]],
    covariates: Optional[np.ndarray] = None,
    n_grid: int = 50,
    bandwidth: Optional[float] = None,
    bandwidth_rule: Literal["silverman", "scott", "cv"] = "silverman",
    trim_fraction: float = 0.01,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> MTEResult:
    """
    Estimate Marginal Treatment Effects via local instrumental variables.

    Computes MTE(u) at grid points across the propensity score support,
    where u represents unobserved resistance to treatment.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y
    treatment : array-like
        Binary treatment D ∈ {0, 1}
    instrument : array-like
        Instrument(s) Z. Can be continuous or discrete.
    covariates : np.ndarray, optional
        Exogenous controls X
    n_grid : int, default=50
        Number of grid points for MTE evaluation
    bandwidth : float, optional
        Kernel bandwidth. If None, uses bandwidth_rule.
    bandwidth_rule : str, default="silverman"
        Rule for automatic bandwidth: "silverman", "scott", or "cv"
    trim_fraction : float, default=0.01
        Fraction of propensity scores to trim from each tail
    alpha : float, default=0.05
        Significance level for confidence intervals
    n_bootstrap : int, default=500
        Bootstrap replications for standard errors
    random_state : int, optional
        Random seed for bootstrap

    Returns
    -------
    MTEResult
        Dictionary containing:
        - mte_grid: MTE estimates at grid points
        - u_grid: Grid points (propensity values)
        - se_grid: Bootstrap standard errors
        - ci_lower, ci_upper: Pointwise confidence intervals

    Examples
    --------
    >>> # Estimate MTE using college proximity as instrument
    >>> mte_result = local_iv(wages, college_degree, distance_to_college)
    >>> # Plot MTE curve
    >>> plt.plot(mte_result['u_grid'], mte_result['mte_grid'])

    References
    ----------
    - Heckman, J.J. & Vytlacil, E. (1999). Local Instrumental Variables and
      Latent Variable Models for Identifying and Bounding Treatment Effects.
    - Heckman, J.J. & Vytlacil, E. (2005). Structural Equations, Treatment
      Effects, and Econometric Policy Evaluation.
    """
    # Convert inputs
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    n = len(Y)

    # Validate inputs
    _validate_mte_inputs(Y, D, Z)

    # Estimate propensity scores P(D=1|Z)
    propensity = _estimate_propensity(D, Z, covariates)

    # Determine support and trim
    p_min = np.quantile(propensity, trim_fraction)
    p_max = np.quantile(propensity, 1 - trim_fraction)

    # Mask for common support
    support_mask = (propensity >= p_min) & (propensity <= p_max)
    n_trimmed = n - support_mask.sum()

    Y_trim = Y[support_mask]
    D_trim = D[support_mask]
    P_trim = propensity[support_mask]

    if covariates is not None:
        X_trim = covariates[support_mask]
        # Residualize outcome on covariates
        Y_trim = _residualize(Y_trim, X_trim)
    else:
        X_trim = None

    # Create evaluation grid
    u_grid = np.linspace(p_min, p_max, n_grid)

    # Select bandwidth
    if bandwidth is None:
        bandwidth = _select_bandwidth(P_trim, bandwidth_rule)

    # Estimate MTE at each grid point
    mte_grid = _estimate_mte_grid(Y_trim, D_trim, P_trim, u_grid, bandwidth)

    # Bootstrap for standard errors
    rng = np.random.default_rng(random_state)
    bootstrap_mte = np.zeros((n_bootstrap, n_grid))

    n_trim = len(Y_trim)
    for b in range(n_bootstrap):
        idx = rng.choice(n_trim, size=n_trim, replace=True)
        Y_boot = Y_trim[idx]
        D_boot = D_trim[idx]
        P_boot = P_trim[idx]

        bootstrap_mte[b, :] = _estimate_mte_grid(Y_boot, D_boot, P_boot, u_grid, bandwidth)

    # Standard errors and CIs
    se_grid = np.nanstd(bootstrap_mte, axis=0, ddof=1)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = mte_grid - z_crit * se_grid
    ci_upper = mte_grid + z_crit * se_grid

    return MTEResult(
        mte_grid=mte_grid,
        u_grid=u_grid,
        se_grid=se_grid,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        propensity_support=(p_min, p_max),
        n_obs=n,
        n_trimmed=n_trimmed,
        bandwidth=bandwidth,
        method="local_iv",
    )


def polynomial_mte(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    instrument: Union[np.ndarray, List[float]],
    covariates: Optional[np.ndarray] = None,
    degree: int = 3,
    n_grid: int = 50,
    trim_fraction: float = 0.01,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> MTEResult:
    """
    Estimate MTE using polynomial approximation.

    Fits E[Y|P] and E[D|P] as polynomials in P, then computes
    MTE = ∂E[Y|P]/∂P / ∂E[D|P]/∂P.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y
    treatment : array-like
        Binary treatment D
    instrument : array-like
        Instrument(s) Z
    covariates : np.ndarray, optional
        Exogenous controls X
    degree : int, default=3
        Polynomial degree
    n_grid : int, default=50
        Number of grid points
    trim_fraction : float, default=0.01
        Trimming fraction
    alpha : float, default=0.05
        Significance level
    n_bootstrap : int, default=500
        Bootstrap replications
    random_state : int, optional
        Random seed

    Returns
    -------
    MTEResult
        MTE estimates with polynomial method
    """
    # Convert inputs
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    n = len(Y)
    _validate_mte_inputs(Y, D, Z)

    # Estimate propensity
    propensity = _estimate_propensity(D, Z, covariates)

    # Trim
    p_min = np.quantile(propensity, trim_fraction)
    p_max = np.quantile(propensity, 1 - trim_fraction)
    support_mask = (propensity >= p_min) & (propensity <= p_max)
    n_trimmed = n - support_mask.sum()

    Y_trim = Y[support_mask]
    D_trim = D[support_mask]
    P_trim = propensity[support_mask]

    if covariates is not None:
        Y_trim = _residualize(Y_trim, covariates[support_mask])

    # Fit polynomials
    u_grid = np.linspace(p_min, p_max, n_grid)
    mte_grid = _polynomial_mte_estimate(Y_trim, D_trim, P_trim, u_grid, degree)

    # Bootstrap
    rng = np.random.default_rng(random_state)
    bootstrap_mte = np.zeros((n_bootstrap, n_grid))

    n_trim = len(Y_trim)
    for b in range(n_bootstrap):
        idx = rng.choice(n_trim, size=n_trim, replace=True)
        bootstrap_mte[b, :] = _polynomial_mte_estimate(
            Y_trim[idx], D_trim[idx], P_trim[idx], u_grid, degree
        )

    se_grid = np.nanstd(bootstrap_mte, axis=0, ddof=1)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return MTEResult(
        mte_grid=mte_grid,
        u_grid=u_grid,
        se_grid=se_grid,
        ci_lower=mte_grid - z_crit * se_grid,
        ci_upper=mte_grid + z_crit * se_grid,
        propensity_support=(p_min, p_max),
        n_obs=n,
        n_trimmed=n_trimmed,
        bandwidth=float(degree),  # Store degree as "bandwidth"
        method="polynomial",
    )


def _estimate_propensity(
    D: np.ndarray,
    Z: np.ndarray,
    X: Optional[np.ndarray],
) -> np.ndarray:
    """Estimate propensity score P(D=1|Z, X) via logistic regression."""
    n = len(D)

    # Build design matrix
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if X is not None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        design = np.column_stack([np.ones(n), Z, X])
    else:
        design = np.column_stack([np.ones(n), Z])

    # Fit logistic regression via IRLS
    propensity = _logistic_regression(D, design)

    # Clamp for numerical stability
    eps = 1e-6
    propensity = np.clip(propensity, eps, 1 - eps)

    return propensity


def _logistic_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit logistic regression via IRLS, return predicted probabilities."""
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        # Predicted probabilities
        eta = X @ beta
        eta = np.clip(eta, -500, 500)  # Prevent overflow
        prob = 1 / (1 + np.exp(-eta))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        # Weights and working response
        W = prob * (1 - prob)
        W = np.clip(W, 1e-10, None)

        z = eta + (y - prob) / W

        # Weighted least squares step
        XtWX = X.T @ (W[:, None] * X)
        XtWz = X.T @ (W * z)

        try:
            beta_new = np.linalg.solve(XtWX + 1e-10 * np.eye(p), XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    # Final predictions
    eta = X @ beta
    eta = np.clip(eta, -500, 500)
    return 1 / (1 + np.exp(-eta))


def _estimate_mte_grid(
    Y: np.ndarray,
    D: np.ndarray,
    P: np.ndarray,
    u_grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Estimate MTE at grid points using local linear regression.

    MTE(p) = ∂E[Y|P=p]/∂P
    """
    n_grid = len(u_grid)
    mte = np.zeros(n_grid)

    for i, p in enumerate(u_grid):
        # Kernel weights (Epanechnikov)
        u = (P - p) / bandwidth
        weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

        if weights.sum() < 1e-10:
            mte[i] = np.nan
            continue

        # Local linear regression of Y on P
        # Y = a + b*(P-p) + error
        X_local = np.column_stack([np.ones(len(P)), P - p])
        W = np.diag(weights)

        try:
            XtWX = X_local.T @ W @ X_local
            XtWY = X_local.T @ W @ Y

            # Regularize
            XtWX += 1e-10 * np.eye(2)

            beta = np.linalg.solve(XtWX, XtWY)
            mte[i] = beta[1]  # Slope = ∂E[Y|P]/∂P

        except np.linalg.LinAlgError:
            mte[i] = np.nan

    # Smooth result
    mte = _smooth_mte(mte)

    return mte


def _polynomial_mte_estimate(
    Y: np.ndarray,
    D: np.ndarray,
    P: np.ndarray,
    u_grid: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Estimate MTE using polynomial approximation."""
    # Fit polynomial: E[Y|P] = Σ β_k P^k
    # MTE(p) = dE[Y|P]/dP = Σ k * β_k * p^(k-1)

    # Create polynomial features
    n = len(P)
    X_poly = np.column_stack([P**k for k in range(degree + 1)])

    # OLS fit
    try:
        beta = np.linalg.lstsq(X_poly, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full(len(u_grid), np.nan)

    # Compute derivative at grid points
    mte = np.zeros(len(u_grid))
    for i, p in enumerate(u_grid):
        # d/dp [Σ β_k p^k] = Σ k * β_k * p^(k-1)
        derivative = sum(k * beta[k] * p ** (k - 1) for k in range(1, degree + 1))
        mte[i] = derivative

    return mte


def _select_bandwidth(P: np.ndarray, rule: str) -> float:
    """Select bandwidth using specified rule."""
    sigma = np.std(P, ddof=1)
    n = len(P)

    if sigma < 1e-10:
        return 0.1

    if rule == "silverman":
        # Silverman's rule of thumb
        iqr = np.percentile(P, 75) - np.percentile(P, 25)
        sigma_iqr = min(sigma, iqr / 1.34)
        h = 1.06 * sigma_iqr * n ** (-0.2)
    elif rule == "scott":
        # Scott's rule
        h = 1.059 * sigma * n ** (-0.2)
    else:
        # Default to Silverman
        h = 1.06 * sigma * n ** (-0.2)

    return max(h, 0.01)


def _smooth_mte(mte: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to MTE curve."""
    # Handle NaN values
    valid = ~np.isnan(mte)
    if valid.sum() < 3:
        return mte

    # Interpolate NaN values
    mte_interp = mte.copy()
    if np.any(~valid):
        x = np.arange(len(mte))
        mte_interp[~valid] = np.interp(x[~valid], x[valid], mte[valid])

    # Apply Gaussian filter
    mte_smooth = gaussian_filter1d(mte_interp, sigma=sigma)

    # Restore NaN at boundaries if original was NaN
    mte_smooth[~valid] = np.nan

    return mte_smooth


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residualize y on X via OLS."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(y)
    X_const = np.column_stack([np.ones(n), X])

    try:
        beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
        return y - X_const @ beta
    except np.linalg.LinAlgError:
        return y


def _validate_mte_inputs(Y: np.ndarray, D: np.ndarray, Z: np.ndarray) -> None:
    """Validate inputs for MTE estimation."""
    n = len(Y)

    if len(D) != n:
        raise ValueError(f"Treatment length ({len(D)}) != outcome length ({n})")

    if Z.ndim == 1 and len(Z) != n:
        raise ValueError(f"Instrument length ({len(Z)}) != outcome length ({n})")
    elif Z.ndim == 2 and Z.shape[0] != n:
        raise ValueError(f"Instrument rows ({Z.shape[0]}) != outcome length ({n})")

    # Check treatment is binary
    unique_d = np.unique(D[~np.isnan(D)])
    if not np.all(np.isin(unique_d, [0, 1])):
        raise ValueError(f"Treatment must be binary. Found: {unique_d}")

    # Check for NaN/Inf
    if np.any(np.isnan(Y)):
        raise ValueError("Outcome contains NaN values")
    if np.any(np.isinf(Y)):
        raise ValueError("Outcome contains Inf values")

    # Minimum sample size
    if n < 50:
        import warnings

        warnings.warn(
            f"Small sample size ({n}). MTE estimation may be unreliable.",
            UserWarning,
        )
