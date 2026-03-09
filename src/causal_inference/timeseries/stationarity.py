"""
Stationarity Tests for Time Series.

Session 135: Augmented Dickey-Fuller test and related utilities.
Session 145: Added KPSS and Phillips-Perron tests.

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

from causal_inference.timeseries.types import ADFResult, KPSSResult, PPResult


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
        raise ValueError(f"order ({order}) must be less than series length ({len(series)})")

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
        var_names = [f"var_{i + 1}" for i in range(n_vars)]

    results = {}
    for i, name in enumerate(var_names):
        results[name] = adf_test(data[:, i], regression=regression, alpha=alpha)

    return results


# ============================================================================
# KPSS Test (Session 145)
# ============================================================================

# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
# Note: KPSS has opposite rejection rule - reject H0 (stationary) if stat > CV
KPSS_CRITICAL_VALUES = {
    "c": {  # Level stationary (constant only)
        "10%": 0.347,
        "5%": 0.463,
        "2.5%": 0.574,
        "1%": 0.739,
    },
    "ct": {  # Trend stationary (constant and trend)
        "10%": 0.119,
        "5%": 0.146,
        "2.5%": 0.176,
        "1%": 0.216,
    },
}


def kpss_test(
    series: np.ndarray,
    regression: str = "c",
    lags: Optional[int] = None,
    alpha: float = 0.05,
) -> KPSSResult:
    """
    KPSS test for stationarity.

    Tests H0: series is trend-stationary (stationary around deterministic trend)
    vs H1: series has unit root (non-stationary).

    IMPORTANT: Opposite null hypothesis from ADF test!
    - KPSS: H0 = stationary (low stat = stationary)
    - ADF: H0 = unit root (low stat = stationary)

    Parameters
    ----------
    series : np.ndarray
        1D time series array
    regression : str
        Type of regression:
        - "c": constant only (level stationarity)
        - "ct": constant and trend (trend stationarity)
    lags : int, optional
        Number of lags for Newey-West long-run variance estimation.
        If None, uses int(4 * (n/100)^(1/4)) (Schwert rule).
    alpha : float
        Significance level for determining stationarity

    Returns
    -------
    KPSSResult
        Test results including statistic, p-value, and stationarity decision

    Example
    -------
    >>> np.random.seed(42)
    >>> # Stationary series
    >>> y_stat = np.random.randn(200)
    >>> result = kpss_test(y_stat)
    >>> print(f"Stationary: {result.is_stationary}")  # Should be True

    >>> # Non-stationary series (random walk)
    >>> y_nonstat = np.cumsum(np.random.randn(200))
    >>> result = kpss_test(y_nonstat)
    >>> print(f"Stationary: {result.is_stationary}")  # Should be False

    Notes
    -----
    Use KPSS with ADF for confirmatory testing:
    - ADF rejects + KPSS fails to reject → stationary
    - ADF fails to reject + KPSS rejects → non-stationary
    - Both reject or both fail to reject → inconclusive

    References
    ----------
    Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
    against the alternative of a unit root." J. Econometrics 54: 159-178.
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)

    if n < 10:
        raise ValueError(f"Series too short for KPSS test (n={n}, need >= 10)")

    if regression not in ["c", "ct"]:
        raise ValueError(f"regression must be 'c' or 'ct', got '{regression}'")

    # Default lags (Schwert rule, commonly used)
    if lags is None:
        lags = int(np.ceil(4 * (n / 100) ** 0.25))

    # Step 1: Fit OLS regression to get residuals
    if regression == "c":
        # y_t = a + e_t
        X = np.ones((n, 1))
    else:  # ct
        # y_t = a + b*t + e_t
        t = np.arange(1, n + 1)
        X = np.column_stack([np.ones(n), t])

    beta = np.linalg.lstsq(X, series, rcond=None)[0]
    residuals = series - X @ beta

    # Step 2: Compute partial sums S_t = sum_{i=1}^{t} e_i
    partial_sums = np.cumsum(residuals)

    # Step 3: Compute Newey-West long-run variance with Bartlett kernel
    # sigma^2_l = (1/T) * sum_{t=1}^{T} e_t^2
    #           + (2/T) * sum_{s=1}^{l} w_s * sum_{t=s+1}^{T} e_t * e_{t-s}
    # where w_s = 1 - s/(l+1) (Bartlett weights)

    # Autocovariances
    gamma_0 = np.sum(residuals**2) / n

    gamma_sum = 0.0
    for s in range(1, lags + 1):
        weight = 1 - s / (lags + 1)  # Bartlett kernel
        gamma_s = np.sum(residuals[s:] * residuals[:-s]) / n
        gamma_sum += 2 * weight * gamma_s

    long_run_var = gamma_0 + gamma_sum

    # Ensure positive variance
    if long_run_var <= 0:
        long_run_var = gamma_0

    # Step 4: Compute KPSS statistic
    # KPSS = (1/T^2) * sum_{t=1}^{T} S_t^2 / sigma^2_l
    kpss_stat = np.sum(partial_sums**2) / (n**2 * long_run_var)

    # Step 5: Get critical values and compute p-value
    critical_values = KPSS_CRITICAL_VALUES[regression]

    # Approximate p-value using interpolation
    p_value = _kpss_pvalue(kpss_stat, regression)

    # Determine stationarity (fail to reject H0 if stat < critical value)
    # Note: Opposite of ADF - we WANT to fail to reject for stationarity
    is_stationary = p_value >= alpha

    return KPSSResult(
        statistic=kpss_stat,
        p_value=p_value,
        lags=lags,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
    )


def _kpss_pvalue(stat: float, regression: str) -> float:
    """
    Compute approximate p-value for KPSS statistic.

    Uses linear interpolation between critical values.
    """
    cv = KPSS_CRITICAL_VALUES[regression]

    # KPSS: reject stationarity if stat > CV
    # So p-value is probability of getting stat this large or larger under H0

    if stat <= cv["10%"]:
        # Below 10% CV, p > 0.10
        return 0.15  # Conservative upper bound
    elif stat <= cv["5%"]:
        # Between 10% and 5%
        return 0.10 - 0.05 * (stat - cv["10%"]) / (cv["5%"] - cv["10%"])
    elif stat <= cv["2.5%"]:
        # Between 5% and 2.5%
        return 0.05 - 0.025 * (stat - cv["5%"]) / (cv["2.5%"] - cv["5%"])
    elif stat <= cv["1%"]:
        # Between 2.5% and 1%
        return 0.025 - 0.015 * (stat - cv["2.5%"]) / (cv["1%"] - cv["2.5%"])
    else:
        # Above 1% CV
        return 0.005


# ============================================================================
# Phillips-Perron Test (Session 145)
# ============================================================================


def phillips_perron_test(
    series: np.ndarray,
    regression: str = "c",
    lags: Optional[int] = None,
    alpha: float = 0.05,
) -> PPResult:
    """
    Phillips-Perron test for unit root.

    Tests H0: series has unit root (non-stationary)
    vs H1: series is stationary.

    Like ADF but uses Newey-West HAC correction instead of augmented lags.
    Robust to heteroskedasticity and autocorrelation of unknown form.

    Parameters
    ----------
    series : np.ndarray
        1D time series array
    regression : str
        Type of regression:
        - "n": no constant, no trend
        - "c": constant only (default)
        - "ct": constant and trend
    lags : int, optional
        Number of lags for Newey-West correction.
        If None, uses int(4 * (n/100)^(1/4)).
    alpha : float
        Significance level for determining stationarity

    Returns
    -------
    PPResult
        Test results including Z_t statistic, p-value, and stationarity decision

    Example
    -------
    >>> np.random.seed(42)
    >>> # Stationary series
    >>> y_stat = np.random.randn(200)
    >>> result = phillips_perron_test(y_stat)
    >>> print(f"Stationary: {result.is_stationary}")

    >>> # Non-stationary series (random walk)
    >>> y_nonstat = np.cumsum(np.random.randn(200))
    >>> result = phillips_perron_test(y_nonstat)
    >>> print(f"Stationary: {result.is_stationary}")

    Notes
    -----
    PP test has same null hypothesis and critical values as ADF test.
    Advantage: robust to general heteroskedasticity without specifying lag structure.
    Disadvantage: may have worse size properties in small samples.

    References
    ----------
    Phillips & Perron (1988). "Testing for a unit root in time series
    regression." Biometrika 75(2): 335-346.
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)

    if n < 10:
        raise ValueError(f"Series too short for PP test (n={n}, need >= 10)")

    if regression not in ["n", "c", "ct"]:
        raise ValueError(f"regression must be 'n', 'c', or 'ct', got '{regression}'")

    # Default lags for Newey-West
    if lags is None:
        lags = int(np.ceil(4 * (n / 100) ** 0.25))

    # Step 1: Run simple Dickey-Fuller regression (no augmentation)
    # Δy_t = α + β*t + ρ*y_{t-1} + e_t

    dy = np.diff(series)  # First difference
    y_lag = series[:-1]  # y_{t-1}
    T = len(dy)

    # Build design matrix
    X_parts = []

    if regression in ["c", "ct"]:
        X_parts.append(np.ones(T))

    if regression == "ct":
        trend = np.arange(1, T + 1)
        X_parts.append(trend)

    X_parts.append(y_lag)

    X = np.column_stack(X_parts)

    # OLS estimation
    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    residuals = dy - X @ beta

    # Get rho coefficient and its position
    if regression == "n":
        rho_idx = 0
    elif regression == "c":
        rho_idx = 1
    else:  # ct
        rho_idx = 2

    rho_hat = beta[rho_idx]

    # Step 2: Compute short-run and long-run variance

    # Short-run variance (residual variance)
    s2 = np.sum(residuals**2) / (T - X.shape[1])

    # Newey-West long-run variance with Bartlett kernel
    gamma_0 = np.sum(residuals**2) / T

    gamma_sum = 0.0
    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)
        gamma_j = np.sum(residuals[j:] * residuals[:-j]) / T
        gamma_sum += 2 * weight * gamma_j

    lambda_sq = gamma_0 + gamma_sum

    # Ensure positive variance
    if lambda_sq <= 0:
        lambda_sq = gamma_0

    # Step 3: Compute PP Z_t statistic
    # Formula (matching arch package implementation):
    # Z_t = sqrt(γ₀/λ²) * t_ρ - 0.5 * (λ² - γ₀)/λ * (T * σ_ρ / s)
    #
    # Where:
    #   γ₀ = SSR/T (short-run variance, no DoF adjustment)
    #   λ² = Newey-West long-run variance
    #   t_ρ = ρ̂/σ_ρ (OLS t-statistic)
    #   σ_ρ = OLS standard error of ρ
    #   s = sqrt(SSR/(T-k)) (DoF-adjusted std)
    #
    # Reference: Phillips & Perron (1988), arch package unitroot.py

    # Compute OLS standard error of rho
    XtX_inv = np.linalg.inv(X.T @ X)
    se_rho = np.sqrt(s2 * XtX_inv[rho_idx, rho_idx])

    # OLS t-statistic
    t_rho = rho_hat / se_rho

    # DoF-adjusted standard deviation
    s = np.sqrt(s2)
    lambda_hat = np.sqrt(lambda_sq)

    # PP Z_t statistic (correct formula from arch)
    z_t = np.sqrt(gamma_0 / lambda_sq) * t_rho - 0.5 * ((lambda_sq - gamma_0) / lambda_hat) * (
        T * se_rho / s
    )

    # PP Z_rho statistic (alternative form)
    # Z_ρ = T * ρ̂ - 0.5 * (T² * σ²_ρ / s²) * (λ² - γ₀)
    se_rho_sq = se_rho**2
    z_rho = T * rho_hat - 0.5 * (T**2 * se_rho_sq / s2) * (lambda_sq - gamma_0)

    # Step 4: Get critical values (same as ADF) and p-value
    critical_values = ADF_CRITICAL_VALUES.get(regression, ADF_CRITICAL_VALUES["c"])

    # P-value using same approximation as ADF (same asymptotic distribution)
    p_value = _adf_pvalue(z_t, regression, T)

    # Determine stationarity
    is_stationary = p_value < alpha

    return PPResult(
        statistic=z_t,
        p_value=p_value,
        lags=lags,
        n_obs=n,
        critical_values=critical_values,
        is_stationary=is_stationary,
        regression=regression,
        alpha=alpha,
        rho_stat=z_rho,
    )


def confirmatory_stationarity_test(
    series: np.ndarray,
    regression: str = "c",
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Run ADF and KPSS tests together for confirmatory analysis.

    Combines opposite-null tests for stronger inference:
    - ADF: H0 = unit root
    - KPSS: H0 = stationary

    Parameters
    ----------
    series : np.ndarray
        1D time series
    regression : str
        Type of regression ("c" or "ct")
    alpha : float
        Significance level

    Returns
    -------
    dict
        Contains ADF result, KPSS result, and interpretation

    Example
    -------
    >>> np.random.seed(42)
    >>> y = np.random.randn(200)
    >>> result = confirmatory_stationarity_test(y)
    >>> print(result["interpretation"])
    """
    adf_result = adf_test(series, regression=regression, alpha=alpha)
    kpss_result = kpss_test(series, regression=regression, alpha=alpha)

    # Interpret combined results
    adf_rejects = adf_result.is_stationary  # Rejects unit root → stationary
    kpss_rejects = not kpss_result.is_stationary  # Rejects stationarity → non-stationary

    if adf_rejects and not kpss_rejects:
        interpretation = "Stationary (ADF rejects unit root, KPSS fails to reject stationarity)"
        conclusion = "stationary"
    elif not adf_rejects and kpss_rejects:
        interpretation = "Non-stationary (ADF fails to reject unit root, KPSS rejects stationarity)"
        conclusion = "non-stationary"
    elif adf_rejects and kpss_rejects:
        interpretation = "Inconclusive (both tests reject - possible fractional integration)"
        conclusion = "inconclusive"
    else:
        interpretation = "Inconclusive (neither test rejects - low power or near unit root)"
        conclusion = "inconclusive"

    return {
        "adf": adf_result,
        "kpss": kpss_result,
        "interpretation": interpretation,
        "conclusion": conclusion,
    }
