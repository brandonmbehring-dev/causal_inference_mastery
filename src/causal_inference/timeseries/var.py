"""
Vector Autoregression (VAR) Estimation.

Session 135: VAR model estimation for time series analysis.

The VAR(p) model:
    Y_t = A_0 + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + epsilon_t

where:
    Y_t is a (k x 1) vector of k variables at time t
    A_0 is a (k x 1) vector of intercepts
    A_i is a (k x k) coefficient matrix for lag i
    epsilon_t is a (k x 1) vector of error terms
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VARResult


def var_estimate(
    data: np.ndarray,
    lags: int = 1,
    var_names: Optional[List[str]] = None,
    include_constant: bool = True,
) -> VARResult:
    """
    Estimate a VAR(p) model using OLS equation-by-equation.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data.
        Each column is a variable, rows are time points.
    lags : int
        Lag order p (number of lags to include)
    var_names : List[str], optional
        Variable names. If None, uses ["var_1", "var_2", ...]
    include_constant : bool
        Whether to include intercept term

    Returns
    -------
    VARResult
        Estimation results including coefficients, residuals, and diagnostics

    Raises
    ------
    ValueError
        If data has insufficient observations for the lag order

    Example
    -------
    >>> np.random.seed(42)
    >>> n, k = 200, 2
    >>> data = np.random.randn(n, k)
    >>> result = var_estimate(data, lags=2)
    >>> print(f"AIC: {result.aic:.2f}")
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if lags < 1:
        raise ValueError(f"Lags must be >= 1, got {lags}")

    if n_obs <= lags + 1:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for lag order {lags}. "
            f"Need at least {lags + 2} observations."
        )

    if var_names is None:
        var_names = [f"var_{i + 1}" for i in range(n_vars)]
    elif len(var_names) != n_vars:
        raise ValueError(f"var_names length ({len(var_names)}) must match n_vars ({n_vars})")

    # Build design matrix and dependent variable
    Y, X = _build_var_matrices(data, lags, include_constant)

    # OLS estimation: B = (X'X)^{-1} X'Y
    # Each row of B corresponds to one equation
    XtX = X.T @ X
    XtY = X.T @ Y

    try:
        coefficients = linalg.solve(XtX, XtY, assume_a="pos").T
    except linalg.LinAlgError:
        # Fallback to pseudo-inverse for near-singular matrices
        coefficients = (np.linalg.pinv(XtX) @ XtY).T

    # Compute residuals
    residuals = Y - X @ coefficients.T

    # Compute covariance matrix of residuals
    n_obs_effective = residuals.shape[0]
    n_params = X.shape[1]
    dof = n_obs_effective - n_params
    sigma = (residuals.T @ residuals) / max(dof, 1)

    # Compute log-likelihood
    log_likelihood = _compute_log_likelihood(residuals, sigma, n_obs_effective, n_vars)

    # Compute information criteria
    n_params_total = n_vars * n_params  # Total parameters in system
    aic = _compute_aic_var(log_likelihood, n_params_total, n_obs_effective)
    bic = _compute_bic_var(log_likelihood, n_params_total, n_obs_effective)
    hqc = _compute_hqc_var(log_likelihood, n_params_total, n_obs_effective)

    return VARResult(
        coefficients=coefficients,
        residuals=residuals,
        aic=aic,
        bic=bic,
        hqc=hqc,
        lags=lags,
        n_obs=n_obs,
        n_obs_effective=n_obs_effective,
        var_names=list(var_names),
        sigma=sigma,
        log_likelihood=log_likelihood,
    )


def _build_var_matrices(
    data: np.ndarray,
    lags: int,
    include_constant: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build design matrices for VAR estimation.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) data
    lags : int
        Number of lags
    include_constant : bool
        Whether to include intercept

    Returns
    -------
    Y : np.ndarray
        Shape (n_obs - lags, n_vars) dependent variable matrix
    X : np.ndarray
        Shape (n_obs - lags, n_vars * lags + constant) design matrix
    """
    n_obs, n_vars = data.shape
    n_effective = n_obs - lags

    # Dependent variable: Y_t for t = lags+1, ..., n_obs
    Y = data[lags:, :]

    # Build X matrix
    n_cols = n_vars * lags + (1 if include_constant else 0)
    X = np.zeros((n_effective, n_cols))

    col_idx = 0

    # Intercept column
    if include_constant:
        X[:, 0] = 1.0
        col_idx = 1

    # Lagged values
    for lag in range(1, lags + 1):
        for var in range(n_vars):
            X[:, col_idx] = data[lags - lag : n_obs - lag, var]
            col_idx += 1

    return Y, X


def var_forecast(
    result: VARResult,
    data: np.ndarray,
    steps: int = 1,
) -> np.ndarray:
    """
    Generate forecasts from estimated VAR model.

    Parameters
    ----------
    result : VARResult
        Estimated VAR model
    data : np.ndarray
        Shape (n_obs, n_vars) historical data
    steps : int
        Number of forecast steps ahead

    Returns
    -------
    np.ndarray
        Shape (steps, n_vars) forecasted values

    Example
    -------
    >>> np.random.seed(42)
    >>> data = np.random.randn(100, 2)
    >>> result = var_estimate(data, lags=2)
    >>> forecast = var_forecast(result, data, steps=5)
    >>> print(f"Forecast shape: {forecast.shape}")  # (5, 2)
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    data = np.asarray(data, dtype=np.float64)
    n_vars = result.n_vars
    lags = result.lags

    if data.shape[1] != n_vars:
        raise ValueError(f"Data has {data.shape[1]} variables, model expects {n_vars}")

    if data.shape[0] < lags:
        raise ValueError(f"Need at least {lags} observations for forecasting, got {data.shape[0]}")

    forecasts = np.zeros((steps, n_vars))
    intercepts = result.get_intercepts()

    # Use last `lags` observations + any new forecasts
    history = data[-lags:, :].copy()

    for h in range(steps):
        # Compute forecast for step h
        y_hat = intercepts.copy()

        for lag in range(1, lags + 1):
            if h < lag:
                # Use historical data
                y_past = history[lags - lag + h, :]
            else:
                # Use previous forecasts
                y_past = forecasts[h - lag, :]

            A_lag = result.get_lag_matrix(lag)
            y_hat += A_lag @ y_past

        forecasts[h, :] = y_hat

    return forecasts


def var_residuals(
    result: VARResult,
    data: np.ndarray,
) -> np.ndarray:
    """
    Compute residuals from VAR model for given data.

    Parameters
    ----------
    result : VARResult
        Estimated VAR model
    data : np.ndarray
        Shape (n_obs, n_vars) data to compute residuals for

    Returns
    -------
    np.ndarray
        Shape (n_obs - lags, n_vars) residual matrix
    """
    data = np.asarray(data, dtype=np.float64)

    if data.shape[1] != result.n_vars:
        raise ValueError(f"Data has {data.shape[1]} variables, model expects {result.n_vars}")

    Y, X = _build_var_matrices(data, result.lags, include_constant=True)
    fitted = X @ result.coefficients.T
    return Y - fitted


def _compute_log_likelihood(
    residuals: np.ndarray,
    sigma: np.ndarray,
    n_obs: int,
    n_vars: int,
) -> float:
    """Compute Gaussian log-likelihood for VAR model."""
    # Log-likelihood: -0.5 * n * (k * log(2*pi) + log|Sigma| + k)
    # where k is number of variables

    # Avoid log of zero/negative
    det_sigma = np.linalg.det(sigma)
    if det_sigma <= 0:
        # Use pseudo-determinant
        eigvals = np.linalg.eigvalsh(sigma)
        eigvals = eigvals[eigvals > 1e-10]
        log_det = np.sum(np.log(eigvals))
    else:
        log_det = np.log(det_sigma)

    log_lik = -0.5 * n_obs * (n_vars * np.log(2 * np.pi) + log_det + n_vars)
    return log_lik


def _compute_aic_var(log_lik: float, n_params: int, n_obs: int) -> float:
    """Compute AIC for VAR model."""
    return -2 * log_lik + 2 * n_params


def _compute_bic_var(log_lik: float, n_params: int, n_obs: int) -> float:
    """Compute BIC for VAR model."""
    return -2 * log_lik + n_params * np.log(n_obs)


def _compute_hqc_var(log_lik: float, n_params: int, n_obs: int) -> float:
    """Compute Hannan-Quinn Criterion for VAR model."""
    return -2 * log_lik + 2 * n_params * np.log(np.log(n_obs))


def granger_var_test(
    data: np.ndarray,
    cause_idx: int,
    effect_idx: int,
    lags: int = 1,
    var_names: Optional[List[str]] = None,
) -> Tuple[float, float, int, int]:
    """
    Granger causality test using VAR framework.

    Tests whether excluding lags of cause_idx improves prediction of effect_idx.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) data
    cause_idx : int
        Index of potential cause variable
    effect_idx : int
        Index of effect variable
    lags : int
        Number of lags
    var_names : List[str], optional
        Variable names

    Returns
    -------
    f_stat : float
        F-statistic for the test
    p_value : float
        P-value
    df_num : int
        Numerator degrees of freedom
    df_denom : int
        Denominator degrees of freedom
    """
    from scipy import stats

    data = np.asarray(data, dtype=np.float64)
    n_obs, n_vars = data.shape

    # Unrestricted model: full VAR
    Y, X_full = _build_var_matrices(data, lags, include_constant=True)
    y = Y[:, effect_idx]

    # Restricted model: exclude cause_idx lags
    # Find column indices to keep
    cols_to_keep = [0]  # Intercept
    for lag in range(1, lags + 1):
        for var in range(n_vars):
            if var != cause_idx:
                col_idx = 1 + (lag - 1) * n_vars + var
                cols_to_keep.append(col_idx)

    X_restricted = X_full[:, cols_to_keep]

    # OLS for both models
    try:
        beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
        beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 1.0, lags, n_obs - X_full.shape[1]

    resid_full = y - X_full @ beta_full
    resid_restricted = y - X_restricted @ beta_restricted

    rss_full = np.sum(resid_full**2)
    rss_restricted = np.sum(resid_restricted**2)

    # F-test
    df_num = lags  # Number of restrictions
    df_denom = len(y) - X_full.shape[1]

    if df_denom <= 0 or rss_full <= 0:
        return 0.0, 1.0, df_num, max(df_denom, 1)

    f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_denom)
    f_stat = max(0.0, f_stat)

    p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_denom)

    return f_stat, p_value, df_num, df_denom
