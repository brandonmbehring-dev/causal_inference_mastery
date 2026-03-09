"""
Local Projections (Jordà 2005) for Impulse Response Estimation.

Session 159: Alternative to VAR-based IRF. More robust to misspecification.

Local projections estimate impulse responses directly via horizon-specific
regressions, avoiding the need to specify and invert a VAR system.

Algorithm
---------
For each horizon h = 0, 1, ..., H:
    Y_{t+h} = α_h + β_h · Shock_t + γ_h · Controls_t + ε_{t+h}

    1. Run regression of Y_{t+h} on shock and controls
    2. β_h is the impulse response at horizon h
    3. HAC standard errors for Newey-West inference

Key advantages over VAR-based IRF:
- More robust to VAR misspecification (lag length, omitted variables)
- Allows for nonlinear dynamics and state dependence
- Direct estimation avoids compound error from VAR inversion

Key disadvantages:
- Less efficient when VAR is correctly specified
- Requires more data (separate regression for each horizon)
- Overlapping residuals require HAC correction

References
----------
Jordà (2005). "Estimation and Inference of Impulse Responses by Local
Projections." American Economic Review 95(1): 161-182.

Plagborg-Møller & Wolf (2021). "Local Projections and VARs Estimate the
Same Impulse Responses." Econometrica 89(2): 955-980.

Ramey (2016). "Macroeconomic Shocks and Their Propagation." Handbook of
Macroeconomics, Vol. 2.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Literal
import numpy as np
from scipy import stats
from scipy.linalg import cholesky

from causal_inference.timeseries.types import VARResult
from causal_inference.timeseries.svar_types import IRFResult


@dataclass
class LocalProjectionResult:
    """
    Result from Local Projection impulse response estimation.

    Attributes
    ----------
    irf : np.ndarray
        Shape (n_vars, n_vars, horizons+1) impulse response matrix.
        irf[i, j, h] = response of var i to shock in var j at horizon h.
    se : np.ndarray
        Shape (n_vars, n_vars, horizons+1) standard errors.
        HAC-corrected (Newey-West) for overlapping residuals.
    ci_lower : np.ndarray
        Lower confidence band (same shape as irf).
    ci_upper : np.ndarray
        Upper confidence band (same shape as irf).
    horizons : int
        Maximum horizon (0 to horizons inclusive).
    n_obs : int
        Number of observations used (varies by horizon).
    lags : int
        Number of control lags included.
    alpha : float
        Significance level for confidence bands.
    method : str
        Estimation method: "cholesky" or "external".
    var_names : List[str]
        Variable names.
    hac_kernel : str
        HAC kernel used ("bartlett" or "quadratic_spectral").
    hac_bandwidth : int
        Bandwidth used for HAC estimation.

    Notes
    -----
    Standard errors are computed using Newey-West HAC correction
    to account for serial correlation in overlapping h-step-ahead
    regression residuals.
    """

    irf: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    horizons: int
    n_obs: int
    lags: int
    alpha: float
    method: str
    var_names: List[str]
    hac_kernel: str = "bartlett"
    hac_bandwidth: int = 0

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.irf.shape[0]

    @property
    def has_confidence_bands(self) -> bool:
        """Whether confidence bands are available."""
        return self.ci_lower is not None and self.ci_upper is not None

    def get_response(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get impulse response for specific shock-response pair.

        Parameters
        ----------
        response_var : int or str
            Response variable (index or name)
        shock_var : int or str
            Shock variable (index or name)
        horizon : int, optional
            Specific horizon. If None, returns all horizons.

        Returns
        -------
        np.ndarray
            Response values (scalar if horizon specified, else 1D array)
        """
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        if horizon is not None:
            return self.irf[response_idx, shock_idx, horizon]
        return self.irf[response_idx, shock_idx, :]

    def get_response_with_ci(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
    ) -> dict:
        """
        Get impulse response with confidence bands.

        Returns
        -------
        dict
            Keys: 'irf', 'se', 'lower', 'upper', 'horizon'
        """
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        return {
            "irf": self.irf[response_idx, shock_idx, :],
            "se": self.se[response_idx, shock_idx, :],
            "lower": self.ci_lower[response_idx, shock_idx, :],
            "upper": self.ci_upper[response_idx, shock_idx, :],
            "horizon": np.arange(self.horizons + 1),
        }

    def is_significant(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
        horizon: int,
    ) -> bool:
        """Check if response is significantly different from zero."""
        data = self.get_response_with_ci(response_var, shock_var)
        lower = data["lower"][horizon]
        upper = data["upper"][horizon]
        return (lower > 0) or (upper < 0)

    def __repr__(self) -> str:
        return (
            f"LocalProjectionResult(n_vars={self.n_vars}, horizons={self.horizons}, "
            f"lags={self.lags}, method='{self.method}')"
        )


def local_projection_irf(
    data: np.ndarray,
    horizons: int = 20,
    lags: int = 4,
    shock_type: Literal["cholesky", "external"] = "cholesky",
    external_shock: Optional[np.ndarray] = None,
    shock_var: int = 0,
    alpha: float = 0.05,
    var_names: Optional[List[str]] = None,
    cumulative: bool = False,
    hac_kernel: Literal["bartlett", "quadratic_spectral"] = "bartlett",
    hac_bandwidth: Optional[int] = None,
) -> LocalProjectionResult:
    """
    Estimate impulse response functions using Local Projections.

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (n_obs, n_vars).
    horizons : int
        Maximum horizon (0 to horizons inclusive).
    lags : int
        Number of lagged controls to include in regression.
    shock_type : str
        "cholesky": Use Cholesky-orthogonalized shocks (recursive ordering).
        "external": Use external shock variable (must provide external_shock).
    external_shock : np.ndarray, optional
        External shock series, shape (n_obs,). Required if shock_type="external".
    shock_var : int
        For Cholesky: which variable's shock to trace.
        For external: which variable's response to external shock.
    alpha : float
        Significance level for confidence bands.
    var_names : List[str], optional
        Variable names. Default: ["var_0", "var_1", ...].
    cumulative : bool
        If True, compute cumulative IRF (sum up to horizon h).
    hac_kernel : str
        Kernel for Newey-West HAC: "bartlett" or "quadratic_spectral".
    hac_bandwidth : int, optional
        Bandwidth for HAC. Default: floor(4 * (T/100)^(2/9)) following Newey-West.

    Returns
    -------
    LocalProjectionResult
        Local projection impulse responses with HAC standard errors.

    Example
    -------
    >>> import numpy as np
    >>> from causal_inference.timeseries import local_projection_irf
    >>> np.random.seed(42)
    >>> n = 200
    >>> # Simple VAR(1) DGP
    >>> data = np.zeros((n, 2))
    >>> for t in range(1, n):
    ...     data[t, 0] = 0.5 * data[t-1, 0] + np.random.randn()
    ...     data[t, 1] = 0.3 * data[t-1, 0] + 0.4 * data[t-1, 1] + np.random.randn()
    >>> lp = local_projection_irf(data, horizons=10, lags=2)
    >>> print(f"IRF[1,0,5] = {lp.irf[1, 0, 5]:.4f}")  # Response of var1 to var0 shock

    Notes
    -----
    The Cholesky identification assumes recursive ordering: the first
    variable is not contemporaneously affected by shocks to other variables.

    For each horizon h, we estimate:
        Y_{t+h}[i] = α + β * shock_t + Σ γ_j Y_{t-j} + ε_{t+h}

    The coefficient β is the impulse response at horizon h.

    References
    ----------
    Jordà (2005). "Estimation and Inference of Impulse Responses by Local
    Projections." American Economic Review 95(1): 161-182.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")

    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    if n_obs <= lags + horizons:
        raise ValueError(
            f"Insufficient observations: n_obs={n_obs}, need > lags + horizons = {lags + horizons}"
        )

    if shock_type == "external" and external_shock is None:
        raise ValueError("external_shock must be provided when shock_type='external'")

    if shock_type not in ("cholesky", "external"):
        raise ValueError(f"shock_type must be 'cholesky' or 'external', got '{shock_type}'")

    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    if len(var_names) != n_vars:
        raise ValueError(f"var_names length ({len(var_names)}) != n_vars ({n_vars})")

    # Default HAC bandwidth: Newey-West rule
    if hac_bandwidth is None:
        hac_bandwidth = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
        hac_bandwidth = max(1, hac_bandwidth)

    # Storage for results
    irf = np.zeros((n_vars, n_vars, horizons + 1))
    se = np.zeros((n_vars, n_vars, horizons + 1))

    if shock_type == "cholesky":
        # Compute Cholesky shock using first-stage OLS residuals
        shocks = _compute_cholesky_shocks(data, lags)
        n_shock_obs = shocks.shape[0]

        # For each response variable and shock variable
        for shock_idx in range(n_vars):
            for response_idx in range(n_vars):
                for h in range(horizons + 1):
                    beta, se_beta = _lp_regression_single(
                        data=data,
                        response_idx=response_idx,
                        shock=shocks[:, shock_idx],
                        horizon=h,
                        lags=lags,
                        hac_kernel=hac_kernel,
                        hac_bandwidth=hac_bandwidth,
                    )
                    irf[response_idx, shock_idx, h] = beta
                    se[response_idx, shock_idx, h] = se_beta

    else:  # external shock
        # External shock identification
        if external_shock.shape[0] != n_obs:
            raise ValueError(
                f"external_shock length ({external_shock.shape[0]}) != n_obs ({n_obs})"
            )

        # All responses to the single external shock
        for response_idx in range(n_vars):
            for h in range(horizons + 1):
                beta, se_beta = _lp_regression_single(
                    data=data,
                    response_idx=response_idx,
                    shock=external_shock,
                    horizon=h,
                    lags=lags,
                    hac_kernel=hac_kernel,
                    hac_bandwidth=hac_bandwidth,
                    external=True,
                )
                irf[response_idx, 0, h] = beta
                se[response_idx, 0, h] = se_beta

    # Cumulative IRF
    if cumulative:
        irf = np.cumsum(irf, axis=2)
        # SEs for cumulative: approximate using delta method (sum of variances)
        se = np.sqrt(np.cumsum(se**2, axis=2))

    # Confidence bands
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = irf - z_crit * se
    ci_upper = irf + z_crit * se

    return LocalProjectionResult(
        irf=irf,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        horizons=horizons,
        n_obs=n_obs - lags - horizons,
        lags=lags,
        alpha=alpha,
        method=shock_type,
        var_names=var_names,
        hac_kernel=hac_kernel,
        hac_bandwidth=hac_bandwidth,
    )


def _compute_cholesky_shocks(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Compute Cholesky-orthogonalized shocks from data.

    Uses OLS to get VAR residuals, then Cholesky decomposition to
    orthogonalize them.

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (n_obs, n_vars).
    lags : int
        Number of lags for VAR estimation.

    Returns
    -------
    np.ndarray
        Orthogonalized shocks, shape (n_obs - lags, n_vars).
    """
    n_obs, n_vars = data.shape

    # Build VAR regression matrices
    # Y: (T-p) x k, X: (T-p) x (1 + k*p)
    T = n_obs - lags
    Y = data[lags:, :]  # (T, k)

    # Design matrix: constant + lagged values
    X = np.ones((T, 1 + n_vars * lags))
    for j in range(lags):
        X[:, 1 + j * n_vars : 1 + (j + 1) * n_vars] = data[lags - j - 1 : n_obs - j - 1, :]

    # OLS: β = (X'X)^{-1} X'Y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y  # (1 + k*p, k)

    # Residuals
    residuals = Y - X @ beta  # (T, k)

    # Covariance matrix
    Sigma = residuals.T @ residuals / T

    # Cholesky decomposition: Σ = PP'
    try:
        P = cholesky(Sigma, lower=True)
    except np.linalg.LinAlgError:
        # Add small regularization if not positive definite
        eps = 1e-6 * np.eye(n_vars)
        P = cholesky(Sigma + eps, lower=True)

    # Orthogonalized shocks: ε = P^{-1} u
    P_inv = np.linalg.inv(P)
    shocks = (P_inv @ residuals.T).T  # (T, k)

    return shocks


def _lp_regression_single(
    data: np.ndarray,
    response_idx: int,
    shock: np.ndarray,
    horizon: int,
    lags: int,
    hac_kernel: str,
    hac_bandwidth: int,
    external: bool = False,
) -> tuple:
    """
    Run single LP regression for one horizon.

    Y_{t+h} = α + β * shock_t + Σ γ_j Y_{t-j} + ε_{t+h}

    Returns
    -------
    tuple
        (beta, se_beta) impulse response and HAC standard error.
    """
    n_obs, n_vars = data.shape

    # Effective sample size depends on horizon and lags
    if external:
        # For external shock, shock series aligned with data
        T = n_obs - lags - horizon
        start = lags
        end = n_obs - horizon
    else:
        # For Cholesky shocks, they start at index 0 (after lags removed)
        shock_len = shock.shape[0]
        T = shock_len - horizon
        start = 0
        end = shock_len - horizon

    if T <= lags + 2:
        # Not enough observations
        return 0.0, np.inf

    # Response variable at t+h
    if external:
        Y = data[start + horizon : end + horizon, response_idx]
    else:
        # Shock is already post-lag, so align differently
        Y = data[lags + horizon : lags + horizon + T, response_idx]

    # Shock at time t
    shock_t = shock[start:end]

    # Build control matrix: constant + lagged Y values
    n_controls = 1 + n_vars * lags
    X = np.ones((T, 1 + n_controls))
    X[:, 1] = shock_t  # Shock in second column

    # Add lagged values as controls
    for j in range(lags):
        if external:
            lag_data = data[start - j - 1 : end - j - 1, :]
        else:
            lag_data = data[lags - j - 1 : lags - j - 1 + T, :]
        X[:, 2 + j * n_vars : 2 + (j + 1) * n_vars] = lag_data

    # OLS estimation
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Singular matrix - add regularization
        XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))

    beta_hat = XtX_inv @ X.T @ Y  # All coefficients
    beta = beta_hat[1]  # Shock coefficient

    # Residuals for HAC
    residuals = Y - X @ beta_hat

    # HAC standard error
    se = _newey_west_se(X, residuals, XtX_inv, hac_kernel, hac_bandwidth)
    se_beta = se[1]  # SE for shock coefficient

    return beta, se_beta


def _newey_west_se(
    X: np.ndarray,
    residuals: np.ndarray,
    XtX_inv: np.ndarray,
    kernel: str,
    bandwidth: int,
) -> np.ndarray:
    """
    Compute Newey-West HAC standard errors.

    The HAC covariance estimator is:
        V = (X'X)^{-1} Ω (X'X)^{-1}

    where Ω = Σ_{j=-M}^{M} w(j/M) Γ_j
    and Γ_j = (1/T) Σ_t (u_t u_{t-j}) x_t x_{t-j}'

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (T, k).
    residuals : np.ndarray
        OLS residuals, shape (T,).
    XtX_inv : np.ndarray
        Inverse of X'X, shape (k, k).
    kernel : str
        "bartlett" or "quadratic_spectral".
    bandwidth : int
        Truncation lag (M).

    Returns
    -------
    np.ndarray
        Standard errors for each coefficient, shape (k,).
    """
    T, k = X.shape
    bandwidth = min(bandwidth, T - 2)

    # Omega = sum of weighted autocovariances
    Omega = np.zeros((k, k))

    # Lag 0 (always weight 1)
    xu = X * residuals[:, np.newaxis]  # (T, k)
    Omega = xu.T @ xu / T

    # Add lagged terms
    for j in range(1, bandwidth + 1):
        if kernel == "bartlett":
            w = 1 - j / (bandwidth + 1)
        else:  # quadratic_spectral
            z = 6 * np.pi * j / (5 * bandwidth)
            w = 3 / z**2 * (np.sin(z) / z - np.cos(z))

        # Autocovariance at lag j
        Gamma_j = xu[j:, :].T @ xu[:-j, :] / T

        # Add both j and -j (symmetric)
        Omega += w * (Gamma_j + Gamma_j.T)

    # HAC variance: (X'X)^{-1} Ω (X'X)^{-1}
    V = XtX_inv @ Omega @ XtX_inv

    # Standard errors
    se = np.sqrt(np.diag(V))

    # Ensure positive
    se = np.maximum(se, 1e-10)

    return se


def compare_lp_var_irf(
    data: np.ndarray,
    horizons: int = 20,
    lags: int = 4,
    var_names: Optional[List[str]] = None,
) -> dict:
    """
    Compare Local Projection and VAR-based IRF.

    When the VAR is correctly specified, LP and VAR should produce
    similar impulse responses (Plagborg-Møller & Wolf 2021).

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (n_obs, n_vars).
    horizons : int
        Maximum horizon.
    lags : int
        Number of lags.
    var_names : List[str], optional
        Variable names.

    Returns
    -------
    dict
        Keys: 'lp_irf', 'var_irf', 'difference', 'max_diff'
    """
    from causal_inference.timeseries.var import var_estimate
    from causal_inference.timeseries.svar import cholesky_svar
    from causal_inference.timeseries.irf import compute_irf

    # LP-based IRF
    lp_result = local_projection_irf(data, horizons=horizons, lags=lags, var_names=var_names)

    # VAR-based IRF
    var_result = var_estimate(data, lags=lags, var_names=var_names)
    svar_result = cholesky_svar(var_result)
    var_irf = compute_irf(svar_result, horizons=horizons)

    # Difference
    diff = lp_result.irf - var_irf.irf
    max_diff = np.max(np.abs(diff))

    return {
        "lp_irf": lp_result,
        "var_irf": var_irf,
        "difference": diff,
        "max_diff": max_diff,
    }


def state_dependent_lp(
    data: np.ndarray,
    state_indicator: np.ndarray,
    horizons: int = 20,
    lags: int = 4,
    shock_type: Literal["cholesky", "external"] = "cholesky",
    external_shock: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    var_names: Optional[List[str]] = None,
) -> dict:
    """
    State-dependent Local Projections (Auerbach & Gorodnichenko 2012).

    Estimates separate impulse responses depending on a state indicator
    (e.g., recession vs expansion).

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (n_obs, n_vars).
    state_indicator : np.ndarray
        Binary state indicator, shape (n_obs,). 1 = high state, 0 = low state.
    horizons : int
        Maximum horizon.
    lags : int
        Number of lagged controls.
    shock_type : str
        "cholesky" or "external".
    external_shock : np.ndarray, optional
        External shock series.
    alpha : float
        Significance level.
    var_names : List[str], optional
        Variable names.

    Returns
    -------
    dict
        Keys: 'high_state_irf', 'low_state_irf', 'difference', 'diff_significant'

    Example
    -------
    >>> # Recession indicator
    >>> recession = (gdp_growth < 0).astype(int)
    >>> result = state_dependent_lp(data, recession, horizons=20)
    >>> # Check if responses differ across states
    >>> print(f"Significant difference: {result['diff_significant'].any()}")

    References
    ----------
    Auerbach & Gorodnichenko (2012). "Measuring the Output Responses to
    Fiscal Policy." AEJ: Economic Policy 4(2): 1-27.
    """
    n_obs, n_vars = data.shape

    if state_indicator.shape[0] != n_obs:
        raise ValueError(f"state_indicator length must equal n_obs ({n_obs})")

    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    # Split data by state
    high_mask = state_indicator.astype(bool)
    low_mask = ~high_mask

    # Need continuous segments for each state
    # For simplicity, use interaction approach instead of splitting

    # Compute Cholesky shocks on full sample
    if shock_type == "cholesky":
        shocks = _compute_cholesky_shocks(data, lags)
    else:
        if external_shock is None:
            raise ValueError("external_shock required for shock_type='external'")
        shocks = external_shock[lags:]

    # Storage
    irf_high = np.zeros((n_vars, n_vars, horizons + 1))
    irf_low = np.zeros((n_vars, n_vars, horizons + 1))
    se_high = np.zeros((n_vars, n_vars, horizons + 1))
    se_low = np.zeros((n_vars, n_vars, horizons + 1))

    # Interaction regression for each (response, shock, horizon)
    for shock_idx in range(n_vars):
        for response_idx in range(n_vars):
            for h in range(horizons + 1):
                betas, ses = _lp_state_regression(
                    data=data,
                    response_idx=response_idx,
                    shock=shocks[:, shock_idx] if shock_type == "cholesky" else shocks,
                    state=state_indicator,
                    horizon=h,
                    lags=lags,
                )
                irf_low[response_idx, shock_idx, h] = betas[0]
                irf_high[response_idx, shock_idx, h] = betas[1]
                se_low[response_idx, shock_idx, h] = ses[0]
                se_high[response_idx, shock_idx, h] = ses[1]

    # Confidence bands
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Difference and its significance
    diff = irf_high - irf_low
    se_diff = np.sqrt(se_high**2 + se_low**2)
    diff_significant = np.abs(diff) > z_crit * se_diff

    return {
        "high_state_irf": LocalProjectionResult(
            irf=irf_high,
            se=se_high,
            ci_lower=irf_high - z_crit * se_high,
            ci_upper=irf_high + z_crit * se_high,
            horizons=horizons,
            n_obs=n_obs - lags - horizons,
            lags=lags,
            alpha=alpha,
            method=shock_type,
            var_names=var_names,
        ),
        "low_state_irf": LocalProjectionResult(
            irf=irf_low,
            se=se_low,
            ci_lower=irf_low - z_crit * se_low,
            ci_upper=irf_low + z_crit * se_low,
            horizons=horizons,
            n_obs=n_obs - lags - horizons,
            lags=lags,
            alpha=alpha,
            method=shock_type,
            var_names=var_names,
        ),
        "difference": diff,
        "diff_significant": diff_significant,
    }


def _lp_state_regression(
    data: np.ndarray,
    response_idx: int,
    shock: np.ndarray,
    state: np.ndarray,
    horizon: int,
    lags: int,
) -> tuple:
    """
    Run state-dependent LP regression with interaction terms.

    Y_{t+h} = α₀ + α₁·S_t + β₀·shock·(1-S_t) + β₁·shock·S_t + controls + ε

    Returns (beta_low, beta_high) and (se_low, se_high).
    """
    n_obs, n_vars = data.shape

    # Align shock with state (shock starts after lags)
    shock_len = shock.shape[0]
    state_aligned = state[lags : lags + shock_len]

    # Effective sample
    T = shock_len - horizon
    if T <= lags + 4:
        return (0.0, 0.0), (np.inf, np.inf)

    # Response at t+h
    Y = data[lags + horizon : lags + horizon + T, response_idx]

    # Shock and state at t
    shock_t = shock[:T]
    state_t = state_aligned[:T]

    # Build design matrix with interactions
    # Columns: constant, state, shock*(1-state), shock*state, lagged controls
    n_controls = n_vars * lags
    X = np.zeros((T, 4 + n_controls))
    X[:, 0] = 1  # Constant
    X[:, 1] = state_t  # State indicator
    X[:, 2] = shock_t * (1 - state_t)  # Shock in low state
    X[:, 3] = shock_t * state_t  # Shock in high state

    # Lagged controls
    for j in range(lags):
        X[:, 4 + j * n_vars : 4 + (j + 1) * n_vars] = data[lags - j - 1 : lags - j - 1 + T, :]

    # OLS
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))

    beta_hat = XtX_inv @ X.T @ Y
    residuals = Y - X @ beta_hat

    # HAC SEs
    bandwidth = int(np.floor(4 * (T / 100) ** (2 / 9)))
    bandwidth = max(1, bandwidth)
    se = _newey_west_se(X, residuals, XtX_inv, "bartlett", bandwidth)

    # Low state: beta[2], High state: beta[3]
    return (beta_hat[2], beta_hat[3]), (se[2], se[3])


def lp_to_irf_result(lp_result: LocalProjectionResult) -> IRFResult:
    """
    Convert LocalProjectionResult to IRFResult for API compatibility.

    This allows LP results to be used with existing IRF visualization
    and analysis functions.

    Parameters
    ----------
    lp_result : LocalProjectionResult
        Local projection result.

    Returns
    -------
    IRFResult
        Compatible IRF result structure.
    """
    return IRFResult(
        irf=lp_result.irf,
        irf_lower=lp_result.ci_lower,
        irf_upper=lp_result.ci_upper,
        horizons=lp_result.horizons,
        cumulative=False,
        orthogonalized=True,
        var_names=lp_result.var_names,
        alpha=lp_result.alpha,
        n_bootstrap=0,
    )
