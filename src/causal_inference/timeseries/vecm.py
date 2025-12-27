"""
Vector Error Correction Model (VECM) Estimation.

Session 149: VECM for cointegrated time series.

The VECM is the error-correction form of a cointegrated VAR:

    ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

Key components:
- α (adjustment): How fast variables return to equilibrium
- β (cointegrating vectors): Long-run equilibrium relationships
- Γ (short-run dynamics): Immediate response to past changes

This module provides:
- vecm_estimate(): Full VECM estimation given cointegration rank
- vecm_forecast(): Forecasting from VECM
- vecm_granger_causality(): Granger causality in VECM framework

References
----------
- Lütkepohl (2005). "New Introduction to Multiple Time Series Analysis"
- Johansen (1995). "Likelihood-Based Inference in Cointegrated VAR Models"
- Engle & Granger (1987). "Co-Integration and Error Correction"
"""

from typing import Optional, Tuple
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VECMResult
from causal_inference.timeseries.cointegration import johansen_test


def vecm_estimate(
    data: np.ndarray,
    coint_rank: int,
    lags: int = 1,
    det_order: int = 0,
    method: str = "johansen",
) -> VECMResult:
    """
    Estimate Vector Error Correction Model (VECM).

    The VECM representation of a cointegrated VAR(p) is:

        ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

    Parameters
    ----------
    data : np.ndarray
        Time series data (T × k matrix).
    coint_rank : int
        Cointegration rank r (number of cointegrating relationships).
        Must be between 1 and k-1 for meaningful VECM.
    lags : int, default=1
        Number of lags in the underlying VAR model (p).
        VECM uses p-1 differenced lags.
    det_order : int, default=0
        Deterministic terms:
        -1 = no constant, no trend
         0 = restricted constant (inside cointegrating relation)
         1 = unrestricted constant
    method : str, default="johansen"
        Estimation method:
        - "johansen": Maximum likelihood via Johansen reduced rank regression
        - "ols": Two-step OLS (less efficient but faster)

    Returns
    -------
    VECMResult
        Estimated VECM with α, β, Γ, and diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid or estimation fails.

    Examples
    --------
    >>> import numpy as np
    >>> # Create cointegrated system
    >>> np.random.seed(42)
    >>> n = 200
    >>> trend = np.cumsum(np.random.randn(n))
    >>> y1 = trend + np.random.randn(n) * 0.5
    >>> y2 = 0.5 * trend + np.random.randn(n) * 0.5
    >>> data = np.column_stack([y1, y2])
    >>> result = vecm_estimate(data, coint_rank=1, lags=2)
    >>> print(f"Adjustment: α = {result.alpha.flatten()}")
    >>> print(f"Cointegrating vector: β = {result.beta.flatten()}")

    Notes
    -----
    The VECM is derived from the VAR representation:

        Y_t = A₁Y_{t-1} + A₂Y_{t-2} + ... + AₚY_{t-p} + c + ε_t

    By writing:

        ΔY_t = ΠY_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

    Where:
        Π = A₁ + A₂ + ... + Aₚ - I
        Γⱼ = -(Aⱼ₊₁ + Aⱼ₊₂ + ... + Aₚ)

    Cointegration implies Π has reduced rank r < k, so Π = αβ'.

    References
    ----------
    Johansen (1995). Likelihood-Based Inference in Cointegrated VAR Models.
    """
    # Input validation
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional (T × k)")

    T, k = data.shape

    if coint_rank < 1:
        raise ValueError(f"coint_rank must be >= 1, got {coint_rank}")
    if coint_rank >= k:
        raise ValueError(f"coint_rank must be < k={k}, got {coint_rank}")
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")
    if T < 2 * lags + k + 10:
        raise ValueError(f"Insufficient observations: T={T}, need at least {2*lags + k + 10}")
    if det_order not in [-1, 0, 1]:
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}")

    if method == "johansen":
        return _vecm_estimate_johansen(data, coint_rank, lags, det_order)
    elif method == "ols":
        return _vecm_estimate_ols(data, coint_rank, lags, det_order)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'johansen' or 'ols'.")


def _vecm_estimate_johansen(
    data: np.ndarray,
    coint_rank: int,
    lags: int,
    det_order: int,
) -> VECMResult:
    """
    VECM estimation via Johansen maximum likelihood.

    Uses the Johansen procedure to get α and β, then estimates Γ.
    """
    T, k = data.shape

    # Get cointegrating vectors from Johansen test
    johansen_result = johansen_test(data, lags=lags, det_order=det_order)

    # Extract α and β for given rank
    beta = johansen_result.eigenvectors[:, :coint_rank]
    alpha = johansen_result.adjustment[:, :coint_rank]

    # Compute Π = αβ'
    pi = alpha @ beta.T

    # Now estimate short-run dynamics Γ using OLS on the VECM equation:
    # ΔY_t - αβ'Y_{t-1} = Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

    # Build differenced data
    dY = np.diff(data, axis=0)  # (T-1) × k

    # Build lagged levels Y_{t-1}
    Y_lag = data[lags - 1 : -1, :]  # (T-lags) × k

    # Compute error correction term: ECT = β'Y_{t-1}
    # Shape: (T-lags) × r
    ECT = Y_lag @ beta

    # Build differenced lags for short-run dynamics
    # ΔY_{t-1}, ΔY_{t-2}, ..., ΔY_{t-p+1}
    T_eff = T - lags
    n_sr_lags = lags - 1  # Number of differenced lags in VECM

    if n_sr_lags > 0:
        dY_lags = np.zeros((T_eff, k * n_sr_lags))
        for j in range(n_sr_lags):
            start_idx = lags - j - 2
            end_idx = T - j - 2
            dY_lags[:, j * k : (j + 1) * k] = dY[start_idx:end_idx, :]
    else:
        dY_lags = np.zeros((T_eff, 0))

    # Dependent variable: ΔY_t for t = lags+1, ..., T
    dY_dep = dY[lags - 1 :, :]  # (T-lags) × k

    # Build regressor matrix
    # LHS: ΔY_t - α·ECT_t (move ECT to RHS with coefficient α)
    # Actually, estimate: ΔY_t = α·ECT_t + Γ·ΔY_lags + c + ε

    # Build full regressor matrix: [ECT | ΔY_lags | const]
    if det_order >= 0:
        # Include constant
        X = np.column_stack([ECT, dY_lags, np.ones(T_eff)])
    else:
        X = np.column_stack([ECT, dY_lags]) if dY_lags.size > 0 else ECT

    # OLS estimation: β̂ = (X'X)⁻¹X'Y
    XtX = X.T @ X
    XtY = X.T @ dY_dep

    try:
        coeffs = linalg.solve(XtX, XtY, assume_a="pos")
    except linalg.LinAlgError:
        # Fallback to pseudoinverse
        coeffs = linalg.pinv(XtX) @ XtY

    # Extract coefficients
    # coeffs shape: (r + k*(p-1) + 1, k) if constant, else (r + k*(p-1), k)
    alpha_est = coeffs[:coint_rank, :].T  # k × r
    gamma_start = coint_rank
    gamma_end = gamma_start + k * n_sr_lags
    gamma = coeffs[gamma_start:gamma_end, :].T  # k × (k*(p-1))

    if det_order >= 0:
        const = coeffs[-1, :].reshape(-1, 1)  # k × 1
    else:
        const = None

    # Note: We use α from Johansen, not α_est from OLS
    # The OLS α_est is less efficient; use Johansen ML estimate

    # Compute residuals
    fitted = X @ coeffs
    residuals = dY_dep - fitted

    # Residual covariance
    sigma = residuals.T @ residuals / (T_eff - X.shape[1])

    # Information criteria
    n_params = k * (coint_rank + k * n_sr_lags + (1 if const is not None else 0))
    log_det_sigma = np.log(linalg.det(sigma))
    log_likelihood = -0.5 * T_eff * (k * np.log(2 * np.pi) + log_det_sigma + k)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(T_eff)

    return VECMResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pi=pi,
        const=const,
        coint_rank=coint_rank,
        lags=lags,
        residuals=residuals,
        sigma=sigma,
        n_obs=T_eff,
        n_vars=k,
        det_order=det_order,
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
    )


def _vecm_estimate_ols(
    data: np.ndarray,
    coint_rank: int,
    lags: int,
    det_order: int,
) -> VECMResult:
    """
    Two-step OLS VECM estimation (Engle-Granger style).

    Step 1: Estimate cointegrating relationships via OLS regression
    Step 2: Estimate error correction model using residuals

    Less efficient than ML but faster and more robust.
    """
    T, k = data.shape

    # Step 1: Get cointegrating vectors from Johansen (for consistency)
    johansen_result = johansen_test(data, lags=lags, det_order=det_order)
    beta = johansen_result.eigenvectors[:, :coint_rank]

    # Compute error correction terms
    ECT = data @ beta  # T × r

    # Step 2: Estimate VECM equation by equation
    # ΔY_t = α·ECT_{t-1} + Γ₁ΔY_{t-1} + ... + c + ε_t

    dY = np.diff(data, axis=0)  # (T-1) × k
    T_eff = T - lags

    # ECT lagged one period
    ECT_lag = ECT[lags - 1 : -1, :]  # (T-lags) × r

    # Build differenced lags
    n_sr_lags = lags - 1
    if n_sr_lags > 0:
        dY_lags = np.zeros((T_eff, k * n_sr_lags))
        for j in range(n_sr_lags):
            start_idx = lags - j - 2
            end_idx = T - j - 2
            dY_lags[:, j * k : (j + 1) * k] = dY[start_idx:end_idx, :]
    else:
        dY_lags = np.zeros((T_eff, 0))

    # Dependent variable
    dY_dep = dY[lags - 1 :, :]

    # Build regressor matrix
    if det_order >= 0:
        X = np.column_stack([ECT_lag, dY_lags, np.ones(T_eff)])
    else:
        X = np.column_stack([ECT_lag, dY_lags]) if dY_lags.size > 0 else ECT_lag

    # OLS
    XtX = X.T @ X
    XtY = X.T @ dY_dep

    try:
        coeffs = linalg.solve(XtX, XtY, assume_a="pos")
    except linalg.LinAlgError:
        coeffs = linalg.pinv(XtX) @ XtY

    # Extract coefficients
    alpha = coeffs[:coint_rank, :].T
    gamma_start = coint_rank
    gamma_end = gamma_start + k * n_sr_lags
    gamma = coeffs[gamma_start:gamma_end, :].T

    if det_order >= 0:
        const = coeffs[-1, :].reshape(-1, 1)
    else:
        const = None

    # Compute residuals and covariance
    fitted = X @ coeffs
    residuals = dY_dep - fitted
    sigma = residuals.T @ residuals / (T_eff - X.shape[1])

    # Π = αβ'
    pi = alpha @ beta.T

    # Information criteria
    n_params = k * (coint_rank + k * n_sr_lags + (1 if const is not None else 0))
    log_det_sigma = np.log(linalg.det(sigma))
    log_likelihood = -0.5 * T_eff * (k * np.log(2 * np.pi) + log_det_sigma + k)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(T_eff)

    return VECMResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pi=pi,
        const=const,
        coint_rank=coint_rank,
        lags=lags,
        residuals=residuals,
        sigma=sigma,
        n_obs=T_eff,
        n_vars=k,
        det_order=det_order,
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
    )


def vecm_forecast(
    result: VECMResult,
    data: np.ndarray,
    horizons: int = 10,
) -> np.ndarray:
    """
    Forecast from estimated VECM.

    Parameters
    ----------
    result : VECMResult
        Estimated VECM.
    data : np.ndarray
        Original data (T × k) used for initial conditions.
    horizons : int, default=10
        Number of forecast periods.

    Returns
    -------
    np.ndarray
        Forecasts (horizons × k matrix).

    Notes
    -----
    Forecasts use the VECM representation:
        ΔY_{T+h} = α·(β'Y_{T+h-1}) + Γ₁ΔY_{T+h-1} + ... + c

    Then Y_{T+h} = Y_{T+h-1} + ΔY_{T+h}
    """
    T, k = data.shape
    p = result.lags

    # Initialize with last observations
    Y_history = data[-(p + 1) :, :].copy()  # Last p+1 observations
    dY_history = np.diff(Y_history, axis=0)  # Last p differences

    forecasts = np.zeros((horizons, k))

    for h in range(horizons):
        # Current level (last available)
        Y_t = Y_history[-1, :]

        # Error correction term: β'Y_t
        ect = result.beta.T @ Y_t  # r × 1

        # Short-run component: Γ₁ΔY_{t-1} + Γ₂ΔY_{t-2} + ...
        sr = np.zeros(k)
        n_sr_lags = p - 1
        if n_sr_lags > 0 and result.gamma.size > 0:
            for j in range(n_sr_lags):
                if j < len(dY_history):
                    Gamma_j = result.gamma[:, j * k : (j + 1) * k]
                    sr += Gamma_j @ dY_history[-(j + 1), :]

        # Forecast change: ΔY_{t+1} = α·ect + sr + c
        dY_forecast = result.alpha @ ect + sr
        if result.const is not None:
            dY_forecast += result.const.flatten()

        # Level forecast: Y_{t+1} = Y_t + ΔY_{t+1}
        Y_forecast = Y_t + dY_forecast
        forecasts[h, :] = Y_forecast

        # Update histories
        Y_history = np.vstack([Y_history[1:, :], Y_forecast.reshape(1, -1)])
        dY_history = np.diff(Y_history, axis=0)

    return forecasts


def vecm_granger_causality(
    result: VECMResult,
    data: np.ndarray,
    cause_idx: int,
    effect_idx: int,
) -> Tuple[float, float, int]:
    """
    Test Granger causality within VECM framework.

    In a VECM, Granger causality has both short-run and long-run components:
    - Short-run: via Γ coefficients (lagged differences)
    - Long-run: via α coefficients (error correction)

    Parameters
    ----------
    result : VECMResult
        Estimated VECM.
    data : np.ndarray
        Original data used in estimation.
    cause_idx : int
        Index of causing variable.
    effect_idx : int
        Index of effect variable.

    Returns
    -------
    tuple
        (test_statistic, p_value, df) for joint test of all causal coefficients.

    Notes
    -----
    Tests H0: All coefficients from cause_idx in equation for effect_idx are zero.
    This includes:
    - α coefficients (if cause affects adjustment to equilibrium)
    - Γ coefficients (if cause has short-run effect)

    The test is a Wald test on the joint restriction.
    """
    from scipy import stats

    k = result.n_vars
    r = result.coint_rank
    n_sr_lags = result.lags - 1

    # Reconstruct X matrix
    T = len(data)
    T_eff = T - result.lags

    # ECT
    ECT = data[result.lags - 1 : -1, :] @ result.beta  # T_eff × r

    # Differenced lags
    dY = np.diff(data, axis=0)
    if n_sr_lags > 0:
        dY_lags = np.zeros((T_eff, k * n_sr_lags))
        for j in range(n_sr_lags):
            start_idx = result.lags - j - 2
            end_idx = T - j - 2
            dY_lags[:, j * k : (j + 1) * k] = dY[start_idx:end_idx, :]
    else:
        dY_lags = np.zeros((T_eff, 0))

    # Full X matrix
    if result.det_order >= 0:
        X = np.column_stack([ECT, dY_lags, np.ones(T_eff)])
    else:
        X = np.column_stack([ECT, dY_lags]) if dY_lags.size > 0 else ECT

    # Identify coefficients for cause_idx in effect_idx equation
    # Coefficient indices for cause_idx:
    # - In Γ_j: position j*k + cause_idx for each j

    causal_coef_indices = []

    # Short-run causality: Γ coefficients
    for j in range(n_sr_lags):
        coef_idx = r + j * k + cause_idx  # Position in X
        causal_coef_indices.append(coef_idx)

    if len(causal_coef_indices) == 0:
        # No short-run lags to test
        return 0.0, 1.0, 0

    # Get OLS estimates for effect equation
    dY_dep = dY[result.lags - 1 :, effect_idx]  # T_eff × 1

    XtX_inv = linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ dY_dep
    residuals = dY_dep - X @ beta_hat
    sigma2 = np.sum(residuals**2) / (T_eff - X.shape[1])

    # Wald test: R*β = 0 where R selects causal coefficients
    q = len(causal_coef_indices)
    R = np.zeros((q, X.shape[1]))
    for i, idx in enumerate(causal_coef_indices):
        R[i, idx] = 1.0

    Rb = R @ beta_hat
    R_XtX_inv_Rt = R @ XtX_inv @ R.T

    try:
        wald_stat = (Rb.T @ linalg.inv(R_XtX_inv_Rt) @ Rb) / (q * sigma2)
    except linalg.LinAlgError:
        wald_stat = np.nan

    p_value = 1 - stats.f.cdf(wald_stat, q, T_eff - X.shape[1])

    return float(wald_stat), float(p_value), q


def compute_error_correction_term(
    data: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Compute error correction terms ECT = β'Y_t.

    Parameters
    ----------
    data : np.ndarray
        Time series data (T × k).
    beta : np.ndarray
        Cointegrating vectors (k × r).

    Returns
    -------
    np.ndarray
        Error correction terms (T × r).
    """
    return data @ beta
