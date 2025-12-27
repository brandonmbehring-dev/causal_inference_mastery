"""
Stationarity Tests for Time Series.

Session 135: Augmented Dickey-Fuller test and related utilities.

The ADF test examines:
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary

The test regression:
    Δy_t = α + β*t + γ*y_{t-1} + Σδ_i*Δy_{t-i} + ε_t

where γ = ρ - 1, testing H0: γ = 0 (unit root) vs H1: γ < 0 (stationary).
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats

from causal_inference.timeseries.types import ADFResult


# ADF critical values (MacKinnon 1994, 2010)
# Format: {regression_type: {sample_size: {level: value}}}
# Asymptotic values
ADF_CRITICAL_VALUES = {
    "n": {  # No constant, no trend
        "1%": -2.566,
        "5%": -1.941,
        "10%": -1.617,
    },
    "c": {  # Constant only
        "1%": -3.430,
        "5%": -2.862,
        "10%": -2.567,
    },
    "ct": {  # Constant and trend
        "1%": -3.960,
        "5%": -3.410,
        "10%": -3.127,
    },
}


def adf_test(
    series: np.ndarray,
    max_lags: Optional[int] = None,
    regression: str = "c",
    alpha: float = 0.05,
    autolag: str = "aic",
) -> ADFResult:
    """
    Augmented Dickey-Fuller test for unit root.

    Tests H0: series has unit root (non-stationary)
    vs H1: series is stationary.

    Parameters
    ----------
    series : np.ndarray
        1D time series array
    max_lags : int, optional
        Maximum number of lags to include. If None, uses
        int(12 * (n/100)^(1/4)) as in Schwert (1989)
    regression : str
        Type of regression:
        - "c": constant only (default)
        - "ct": constant and trend
        - "n": no constant, no trend
    alpha : float
        Significance level for determining stationarity
    autolag : str
        Method for automatic lag selection:
        - "aic": minimize AIC
        - "bic": minimize BIC
        - "fixed": use max_lags (no selection)

    Returns
    -------
    ADFResult
        Test results including statistic, p-value, and stationarity decision

    Example
    -------
    >>> np.random.seed(42)
    >>> # Stationary series
    >>> y_stat = np.random.randn(200)
    >>> result = adf_test(y_stat)
    >>> print(f"Stationary: {result.is_stationary}")

    >>> # Non-stationary series (random walk)
    >>> y_nonstat = np.cumsum(np.random.randn(200))
    >>> result = adf_test(y_nonstat)
    >>> print(f"Stationary: {result.is_stationary}")
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)

    if n < 10:
        raise ValueError(f"Series too short for ADF test (n={n}, need >= 10)")

    if regression not in ["n", "c", "ct"]:
        raise ValueError(f"regression must be 'n', 'c', or 'ct', got '{regression}'")

    # Default max_lags (Schwert 1989)
    if max_lags is None:
        max_lags = int(np.ceil(12 * (n / 100) ** 0.25))
    max_lags = min(max_lags, n // 3 - 1)  # Ensure enough observations

    # Select optimal lag
    if autolag in ["aic", "bic"]:
        optimal_lag = _select_adf_lag(series, max_lags, regression, autolag)
    else:
        optimal_lag = max_lags

    # Run ADF test with selected lag
    adf_stat, n_used = _adf_statistic(series, optimal_lag, regression)

    # Get critical values
    critical_values = ADF_CRITICAL_VALUES.get(regression, ADF_CRITICAL_VALUES["c"])

    # Compute approximate p-value using MacKinnon (1994) method
    p_value = _adf_pvalue(adf_stat, regression, n_used)

    # Determine stationarity (reject unit root if stat < critical value)
    is_stationary = p_value < alpha

    return ADFResult(
        statistic=adf_stat,
        p_value=p_value,
        lags=optimal_lag,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
    )


def _adf_statistic(
    series: np.ndarray,
    lags: int,
    regression: str,
) -> Tuple[float, int]:
    """
    Compute ADF test statistic.

    Parameters
    ----------
    series : np.ndarray
        Time series
    lags : int
        Number of lags for augmentation
    regression : str
        Type of regression ("n", "c", "ct")

    Returns
    -------
    adf_stat : float
        ADF t-statistic for γ = 0
    n_used : int
        Number of observations used
    """
    n = len(series)

    # First difference
    dy = np.diff(series)

    # Lagged level
    y_lag = series[lags:-1]

    # Lagged differences
    n_obs = len(dy) - lags
    dy_trimmed = dy[lags:]

    # Build design matrix
    X_parts = []

    # Constant
    if regression in ["c", "ct"]:
        X_parts.append(np.ones(n_obs))

    # Time trend
    if regression == "ct":
        trend = np.arange(lags + 1, lags + 1 + n_obs)
        X_parts.append(trend)

    # Lagged level (y_{t-1})
    X_parts.append(y_lag[:n_obs])

    # Lagged differences
    for i in range(1, lags + 1):
        dy_lag = dy[lags - i : -i if i < lags else len(dy) - lags]
        if len(dy_lag) > n_obs:
            dy_lag = dy_lag[:n_obs]
        X_parts.append(dy_lag)

    X = np.column_stack(X_parts) if X_parts else np.ones((n_obs, 1))
    y = dy_trimmed[:n_obs]

    # OLS estimation
    try:
        beta, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, n_obs

    residuals = y - X @ beta

    # Standard error of gamma coefficient
    # gamma is at index: 0 (if n), 1 (if c), 2 (if ct)
    if regression == "n":
        gamma_idx = 0
    elif regression == "c":
        gamma_idx = 1
    else:  # ct
        gamma_idx = 2

    # Compute variance-covariance matrix
    mse = np.sum(residuals**2) / (n_obs - X.shape[1])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(mse * XtX_inv[gamma_idx, gamma_idx])
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_gamma = np.sqrt(mse * XtX_inv[gamma_idx, gamma_idx])

    # t-statistic for gamma
    gamma = beta[gamma_idx]
    if se_gamma > 0:
        adf_stat = gamma / se_gamma
    else:
        adf_stat = 0.0

    return adf_stat, n_obs


def _select_adf_lag(
    series: np.ndarray,
    max_lags: int,
    regression: str,
    criterion: str,
) -> int:
    """Select optimal lag for ADF test using information criterion."""
    n = len(series)
    best_lag = 0
    best_ic = np.inf

    for lag in range(0, max_lags + 1):
        try:
            _, n_used = _adf_statistic(series, lag, regression)

            # Compute ADF regression for IC
            dy = np.diff(series)
            n_obs = len(dy) - lag
            if n_obs < 5:
                continue

            y_lag = series[lag:-1][:n_obs]
            dy_trimmed = dy[lag:][:n_obs]

            # Build design matrix
            X_parts = []
            if regression in ["c", "ct"]:
                X_parts.append(np.ones(n_obs))
            if regression == "ct":
                X_parts.append(np.arange(lag + 1, lag + 1 + n_obs))
            X_parts.append(y_lag)
            for i in range(1, lag + 1):
                dy_lag = dy[lag - i : -i if i < lag else len(dy) - lag][:n_obs]
                X_parts.append(dy_lag)

            X = np.column_stack(X_parts) if X_parts else np.ones((n_obs, 1))
            beta = np.linalg.lstsq(X, dy_trimmed, rcond=None)[0]
            residuals = dy_trimmed - X @ beta
            rss = np.sum(residuals**2)
            sigma2 = rss / n_obs
            k = X.shape[1]

            if criterion == "aic":
                ic = n_obs * np.log(sigma2) + 2 * k
            else:  # bic
                ic = n_obs * np.log(sigma2) + k * np.log(n_obs)

            if ic < best_ic:
                best_ic = ic
                best_lag = lag

        except (np.linalg.LinAlgError, ValueError):
            continue

    return best_lag


def _adf_pvalue(tau: float, regression: str, n: int) -> float:
    """
    Compute approximate p-value for ADF statistic using MacKinnon (1994).

    This is a simplified approximation. For exact values, use statsmodels.
    """
    # Asymptotic approximation using normal distribution tail
    # The ADF statistic follows a non-standard distribution under H0
    # This approximation works reasonably for large samples

    # Critical value at 5% for comparison
    cv_5 = ADF_CRITICAL_VALUES.get(regression, ADF_CRITICAL_VALUES["c"])["5%"]

    if tau < -10:
        return 0.0001
    elif tau > 0:
        return 0.99

    # Simple linear interpolation based on critical values
    cv_1 = ADF_CRITICAL_VALUES.get(regression, ADF_CRITICAL_VALUES["c"])["1%"]
    cv_10 = ADF_CRITICAL_VALUES.get(regression, ADF_CRITICAL_VALUES["c"])["10%"]

    if tau <= cv_1:
        # More extreme than 1% critical value
        return 0.005
    elif tau <= cv_5:
        # Between 1% and 5%
        return 0.01 + 0.04 * (tau - cv_1) / (cv_5 - cv_1)
    elif tau <= cv_10:
        # Between 5% and 10%
        return 0.05 + 0.05 * (tau - cv_5) / (cv_10 - cv_5)
    else:
        # Greater than 10% critical value
        # Use exponential decay approximation
        return 0.10 + 0.45 * (1 - np.exp(-(tau - cv_10) * 0.5))


def difference_series(
    series: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Difference a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series
    order : int
        Order of differencing (1 = first difference, 2 = second difference, etc.)

    Returns
    -------
    np.ndarray
        Differenced series (length = n - order)

    Example
    -------
    >>> y = np.array([1, 3, 6, 10, 15])
    >>> difference_series(y, order=1)
    array([2., 3., 4., 5.])
    >>> difference_series(y, order=2)
    array([1., 1., 1.])
    """
    series = np.asarray(series, dtype=np.float64).ravel()

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    if order >= len(series):
        raise ValueError(
            f"order ({order}) must be less than series length ({len(series)})"
        )

    result = series.copy()
    for _ in range(order):
        result = np.diff(result)

    return result


def check_stationarity(
    data: np.ndarray,
    var_names: Optional[list] = None,
    alpha: float = 0.05,
    regression: str = "c",
) -> Dict[str, ADFResult]:
    """
    Check stationarity for multiple series.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) multivariate time series
    var_names : list, optional
        Variable names
    alpha : float
        Significance level
    regression : str
        Regression type for ADF test

    Returns
    -------
    Dict[str, ADFResult]
        Mapping from variable name to ADF result

    Example
    -------
    >>> np.random.seed(42)
    >>> data = np.column_stack([
    ...     np.random.randn(200),  # Stationary
    ...     np.cumsum(np.random.randn(200))  # Non-stationary
    ... ])
    >>> results = check_stationarity(data, var_names=["stat", "nonstat"])
    >>> for name, result in results.items():
    ...     print(f"{name}: {result.is_stationary}")
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_obs, n_vars = data.shape

    if var_names is None:
        var_names = [f"var_{i+1}" for i in range(n_vars)]

    results = {}
    for i, name in enumerate(var_names):
        results[name] = adf_test(data[:, i], regression=regression, alpha=alpha)

    return results
