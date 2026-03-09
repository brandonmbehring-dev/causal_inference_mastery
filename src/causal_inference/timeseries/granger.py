"""
Granger Causality Tests.

Session 135: Pairwise and multivariate Granger causality analysis.

Granger causality (Granger 1969):
    X "Granger-causes" Y if past values of X help predict Y
    beyond Y's own past values.

Test framework:
    Unrestricted: Y_t = α + Σβ_i Y_{t-i} + Σγ_j X_{t-j} + ε_t
    Restricted:   Y_t = α + Σβ_i Y_{t-i} + ε_t

    H0: γ_1 = γ_2 = ... = γ_p = 0 (X does not Granger-cause Y)

    F-statistic: F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from causal_inference.timeseries.types import GrangerResult, MultiGrangerResult


def granger_causality(
    data: np.ndarray,
    lags: int = 1,
    alpha: float = 0.05,
    cause_idx: int = 1,
    effect_idx: int = 0,
    var_names: Optional[List[str]] = None,
) -> GrangerResult:
    """
    Test Granger causality between two time series.

    Tests H0: cause does not Granger-cause effect
    vs H1: cause Granger-causes effect

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, 2) or (n_obs, n_vars) time series data.
        For 2-column data: column 0 is effect (Y), column 1 is cause (X).
    lags : int
        Number of lags to include in the test
    alpha : float
        Significance level
    cause_idx : int
        Index of potential cause variable (default 1)
    effect_idx : int
        Index of effect variable (default 0)
    var_names : List[str], optional
        Variable names

    Returns
    -------
    GrangerResult
        Test result including F-statistic, p-value, and causality decision

    Example
    -------
    >>> np.random.seed(42)
    >>> n = 200
    >>> # X causes Y with lag 1
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + np.random.randn() * 0.5
    >>> result = granger_causality(np.column_stack([y, x]), lags=2)
    >>> print(f"X Granger-causes Y: {result.granger_causes}")

    Notes
    -----
    - Series should be stationary. Use ADF test to check.
    - F-test assumes Gaussian errors. For non-Gaussian, use bootstrap.
    - Granger causality is predictive, not true causality.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if n_vars < 2:
        raise ValueError(f"Need at least 2 variables, got {n_vars}")

    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    if n_obs <= 2 * lags + 1:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for {lags} lags. Need at least {2 * lags + 2}."
        )

    if cause_idx >= n_vars or effect_idx >= n_vars:
        raise ValueError(
            f"cause_idx ({cause_idx}) and effect_idx ({effect_idx}) must be < n_vars ({n_vars})"
        )

    if var_names is None:
        var_names = [f"var_{i + 1}" for i in range(n_vars)]

    # Extract the two series
    y = data[:, effect_idx]  # Effect variable
    x = data[:, cause_idx]  # Cause variable

    # Build design matrices
    n_effective = n_obs - lags

    # Unrestricted model: Y_t ~ const + Y_{t-1:t-p} + X_{t-1:t-p}
    X_unrestricted = _build_granger_design(y, x, lags, include_cause=True)

    # Restricted model: Y_t ~ const + Y_{t-1:t-p}
    X_restricted = _build_granger_design(y, x, lags, include_cause=False)

    # Dependent variable
    y_dep = y[lags:]

    # OLS estimation
    try:
        beta_u = np.linalg.lstsq(X_unrestricted, y_dep, rcond=None)[0]
        beta_r = np.linalg.lstsq(X_restricted, y_dep, rcond=None)[0]
    except np.linalg.LinAlgError:
        return GrangerResult(
            cause_var=var_names[cause_idx],
            effect_var=var_names[effect_idx],
            f_statistic=0.0,
            p_value=1.0,
            lags=lags,
            granger_causes=False,
            alpha=alpha,
        )

    # Compute residuals and RSS
    resid_u = y_dep - X_unrestricted @ beta_u
    resid_r = y_dep - X_restricted @ beta_r

    rss_u = np.sum(resid_u**2)
    rss_r = np.sum(resid_r**2)

    # Compute R-squared
    tss = np.sum((y_dep - np.mean(y_dep)) ** 2)
    r2_u = 1 - rss_u / tss if tss > 0 else 0.0
    r2_r = 1 - rss_r / tss if tss > 0 else 0.0

    # F-test
    df_num = lags  # Number of restrictions (gamma coefficients)
    df_denom = n_effective - X_unrestricted.shape[1]  # Residual df

    if df_denom <= 0 or rss_u <= 0:
        f_stat = 0.0
        p_value = 1.0
    else:
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
        f_stat = max(0.0, f_stat)  # Ensure non-negative
        p_value = 1.0 - stats.f.cdf(f_stat, df_num, df_denom)

    # Compute AIC for both models
    k_u = X_unrestricted.shape[1]
    k_r = X_restricted.shape[1]
    aic_u = n_effective * np.log(rss_u / n_effective) + 2 * k_u
    aic_r = n_effective * np.log(rss_r / n_effective) + 2 * k_r

    granger_causes = p_value < alpha

    return GrangerResult(
        cause_var=var_names[cause_idx],
        effect_var=var_names[effect_idx],
        f_statistic=f_stat,
        p_value=p_value,
        lags=lags,
        granger_causes=granger_causes,
        alpha=alpha,
        r2_unrestricted=r2_u,
        r2_restricted=r2_r,
        aic_unrestricted=aic_u,
        aic_restricted=aic_r,
        df_num=df_num,
        df_denom=df_denom,
        rss_unrestricted=rss_u,
        rss_restricted=rss_r,
    )


def _build_granger_design(
    y: np.ndarray,
    x: np.ndarray,
    lags: int,
    include_cause: bool,
) -> np.ndarray:
    """
    Build design matrix for Granger causality regression.

    Parameters
    ----------
    y : np.ndarray
        Effect (dependent) variable
    x : np.ndarray
        Cause variable
    lags : int
        Number of lags
    include_cause : bool
        Whether to include lagged cause variable

    Returns
    -------
    np.ndarray
        Design matrix
    """
    n = len(y)
    n_effective = n - lags

    # Start with constant
    cols = [np.ones(n_effective)]

    # Add lagged y values
    for lag in range(1, lags + 1):
        cols.append(y[lags - lag : n - lag])

    # Add lagged x values if unrestricted
    if include_cause:
        for lag in range(1, lags + 1):
            cols.append(x[lags - lag : n - lag])

    return np.column_stack(cols)


def granger_causality_matrix(
    data: np.ndarray,
    lags: int = 1,
    alpha: float = 0.05,
    var_names: Optional[List[str]] = None,
) -> MultiGrangerResult:
    """
    Compute pairwise Granger causality for all variable pairs.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) multivariate time series
    lags : int
        Number of lags
    alpha : float
        Significance level
    var_names : List[str], optional
        Variable names

    Returns
    -------
    MultiGrangerResult
        Matrix of pairwise Granger causality results

    Example
    -------
    >>> np.random.seed(42)
    >>> n = 200
    >>> # Three variables: X1 -> X2 -> X3
    >>> x1 = np.random.randn(n)
    >>> x2 = np.zeros(n)
    >>> x3 = np.zeros(n)
    >>> for t in range(1, n):
    ...     x2[t] = 0.5 * x1[t-1] + np.random.randn() * 0.5
    ...     x3[t] = 0.5 * x2[t-1] + np.random.randn() * 0.5
    >>> data = np.column_stack([x1, x2, x3])
    >>> result = granger_causality_matrix(data, lags=2)
    >>> print(f"Causal pairs: {result.causality_matrix.sum()}")
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if var_names is None:
        var_names = [f"var_{i + 1}" for i in range(n_vars)]

    pairwise_results: Dict[Tuple[str, str], GrangerResult] = {}
    causality_matrix = np.zeros((n_vars, n_vars), dtype=bool)

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue  # Skip self-causality

            result = granger_causality(
                data,
                lags=lags,
                alpha=alpha,
                cause_idx=i,
                effect_idx=j,
                var_names=var_names,
            )

            key = (var_names[i], var_names[j])
            pairwise_results[key] = result
            causality_matrix[i, j] = result.granger_causes

    return MultiGrangerResult(
        n_vars=n_vars,
        var_names=list(var_names),
        pairwise_results=pairwise_results,
        causality_matrix=causality_matrix,
        lags=lags,
        alpha=alpha,
    )


def bidirectional_granger(
    data: np.ndarray,
    lags: int = 1,
    alpha: float = 0.05,
    var_names: Optional[List[str]] = None,
) -> Tuple[GrangerResult, GrangerResult]:
    """
    Test Granger causality in both directions between two variables.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, 2) bivariate time series
    lags : int
        Number of lags
    alpha : float
        Significance level
    var_names : List[str], optional
        Variable names (length 2)

    Returns
    -------
    Tuple[GrangerResult, GrangerResult]
        (X -> Y result, Y -> X result)

    Example
    -------
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 * x[t-1] + np.random.randn() * 0.5
    >>> data = np.column_stack([x, y])
    >>> result_xy, result_yx = bidirectional_granger(data, lags=2)
    >>> print(f"X -> Y: {result_xy.granger_causes}")
    >>> print(f"Y -> X: {result_yx.granger_causes}")
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Data must have shape (n_obs, 2), got {data.shape}")

    if var_names is None:
        var_names = ["X", "Y"]

    # X -> Y: X causes Y (cause_idx=0, effect_idx=1)
    result_xy = granger_causality(
        data,
        lags=lags,
        alpha=alpha,
        cause_idx=0,
        effect_idx=1,
        var_names=var_names,
    )

    # Y -> X: Y causes X (cause_idx=1, effect_idx=0)
    result_yx = granger_causality(
        data,
        lags=lags,
        alpha=alpha,
        cause_idx=1,
        effect_idx=0,
        var_names=var_names,
    )

    return result_xy, result_yx


def granger_with_lag_selection(
    data: np.ndarray,
    max_lags: int = 10,
    alpha: float = 0.05,
    criterion: str = "aic",
    cause_idx: int = 1,
    effect_idx: int = 0,
    var_names: Optional[List[str]] = None,
) -> Tuple[GrangerResult, int]:
    """
    Granger causality test with automatic lag selection.

    Selects optimal lag order using information criterion, then
    performs Granger causality test.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series
    max_lags : int
        Maximum lag order to consider
    alpha : float
        Significance level
    criterion : str
        Selection criterion: "aic", "bic"
    cause_idx : int
        Index of cause variable
    effect_idx : int
        Index of effect variable
    var_names : List[str], optional
        Variable names

    Returns
    -------
    Tuple[GrangerResult, int]
        (Granger test result, selected lag order)

    Example
    -------
    >>> np.random.seed(42)
    >>> n = 200
    >>> x = np.random.randn(n)
    >>> y = np.zeros(n)
    >>> for t in range(2, n):
    ...     y[t] = 0.3 * x[t-1] + 0.2 * x[t-2] + np.random.randn() * 0.5
    >>> data = np.column_stack([y, x])
    >>> result, optimal_lag = granger_with_lag_selection(data, max_lags=5)
    >>> print(f"Optimal lag: {optimal_lag}, Granger-causes: {result.granger_causes}")
    """
    data = np.asarray(data, dtype=np.float64)
    n_obs = data.shape[0]

    best_ic = np.inf
    best_lag = 1

    for lag in range(1, max_lags + 1):
        if n_obs <= 2 * lag + 1:
            break

        try:
            result = granger_causality(
                data,
                lags=lag,
                alpha=alpha,
                cause_idx=cause_idx,
                effect_idx=effect_idx,
                var_names=var_names,
            )

            if criterion == "aic":
                ic = result.aic_unrestricted
            else:  # bic
                n_eff = n_obs - lag
                k = 1 + 2 * lag  # intercept + y lags + x lags
                ic = n_eff * np.log(result.rss_unrestricted / n_eff) + k * np.log(n_eff)

            if ic < best_ic:
                best_ic = ic
                best_lag = lag

        except ValueError:
            continue

    # Final test with optimal lag
    final_result = granger_causality(
        data,
        lags=best_lag,
        alpha=alpha,
        cause_idx=cause_idx,
        effect_idx=effect_idx,
        var_names=var_names,
    )

    return final_result, best_lag
