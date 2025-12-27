"""
Vector Error Correction Model (VECM) Estimation

Session 149: VECM for cointegrated time series.

The VECM is the error-correction form of a cointegrated VAR:

    ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

Key components:
- α (adjustment): How fast variables return to equilibrium
- β (cointegrating vectors): Long-run equilibrium relationships
- Γ (short-run dynamics): Immediate response to past changes

References:
- Lütkepohl (2005). "New Introduction to Multiple Time Series Analysis"
- Johansen (1995). "Likelihood-Based Inference in Cointegrated VAR Models"
"""

using LinearAlgebra


"""
    vecm_estimate(data; coint_rank, lags=1, det_order=0, method=:johansen)

Estimate Vector Error Correction Model (VECM).

# Arguments
- `data::AbstractMatrix{<:Real}`: Time series data (T × k matrix)
- `coint_rank::Int`: Cointegration rank r (1 to k-1)
- `lags::Int=1`: Number of lags in underlying VAR
- `det_order::Int=0`: Deterministic terms (-1=none, 0=restricted const, 1=unrestricted)
- `method::Symbol=:johansen`: Estimation method (:johansen or :ols)

# Returns
VECMResult with α, β, Γ, and diagnostics.

# Example
```julia
using Random
Random.seed!(42)
n = 200
trend = cumsum(randn(n))
y1 = trend + randn(n) * 0.5
y2 = 0.5 * trend + randn(n) * 0.5
data = hcat(y1, y2)
result = vecm_estimate(data; coint_rank=1, lags=2)
```
"""
function vecm_estimate(
    data::AbstractMatrix{<:Real};
    coint_rank::Int,
    lags::Int=1,
    det_order::Int=0,
    method::Symbol=:johansen,
)
    # Input validation
    T, k = size(data)

    if coint_rank < 1
        error("coint_rank must be >= 1, got $coint_rank")
    end
    if coint_rank >= k
        error("coint_rank must be < k=$k, got $coint_rank")
    end
    if lags < 1
        error("lags must be >= 1, got $lags")
    end
    if T < 2 * lags + k + 10
        error("Insufficient observations: T=$T, need at least $(2*lags + k + 10)")
    end
    if det_order ∉ [-1, 0, 1]
        error("det_order must be -1, 0, or 1, got $det_order")
    end

    if method == :johansen
        return _vecm_estimate_johansen(data, coint_rank, lags, det_order)
    elseif method == :ols
        return _vecm_estimate_ols(data, coint_rank, lags, det_order)
    else
        error("Unknown method: $method. Use :johansen or :ols.")
    end
end


function _vecm_estimate_johansen(
    data::AbstractMatrix{<:Real},
    coint_rank::Int,
    lags::Int,
    det_order::Int,
)
    T, k = size(data)

    # Get cointegrating vectors from Johansen test
    johansen_result = johansen_test(data; lags=lags, det_order=det_order)

    # Extract α and β for given rank
    beta = johansen_result.eigenvectors[:, 1:coint_rank]
    alpha = johansen_result.adjustment[:, 1:coint_rank]

    # Compute Π = αβ'
    pi = alpha * beta'

    # Build differenced data
    dY = diff(data, dims=1)  # (T-1) × k

    # Build lagged levels Y_{t-1}
    Y_lag = data[lags:end-1, :]  # (T-lags) × k

    # Error correction term: ECT = Y_{t-1} * β
    # Shape: (T-lags) × r
    ECT = Y_lag * beta

    # Build differenced lags for short-run dynamics
    T_eff = T - lags
    n_sr_lags = lags - 1

    if n_sr_lags > 0
        dY_lags = zeros(T_eff, k * n_sr_lags)
        for j in 1:n_sr_lags
            start_idx = lags - j
            end_idx = T - 1 - j
            dY_lags[:, (j-1)*k+1:j*k] = dY[start_idx:end_idx, :]
        end
    else
        dY_lags = zeros(T_eff, 0)
    end

    # Dependent variable: ΔY_t for t = lags+1, ..., T
    dY_dep = dY[lags:end, :]  # (T-lags) × k

    # Build regressor matrix: [ECT | ΔY_lags | const]
    if det_order >= 0
        X = hcat(ECT, dY_lags, ones(T_eff))
    else
        X = size(dY_lags, 2) > 0 ? hcat(ECT, dY_lags) : ECT
    end

    # OLS estimation: β̂ = (X'X)⁻¹X'Y
    XtX = X' * X
    XtY = X' * dY_dep

    coeffs = try
        XtX \ XtY
    catch
        pinv(XtX) * XtY
    end

    # Extract coefficients
    alpha_est = coeffs[1:coint_rank, :]'  # k × r
    gamma_start = coint_rank + 1
    gamma_end = gamma_start + k * n_sr_lags - 1

    if n_sr_lags > 0
        gamma = coeffs[gamma_start:gamma_end, :]'  # k × (k*(p-1))
    else
        gamma = zeros(k, 0)
    end

    if det_order >= 0
        const_term = coeffs[end, :]  # k × 1
    else
        const_term = nothing
    end

    # Compute residuals
    fitted = X * coeffs
    residuals = dY_dep - fitted

    # Residual covariance
    sigma = (residuals' * residuals) / (T_eff - size(X, 2))

    # Information criteria
    n_params = k * (coint_rank + k * n_sr_lags + (const_term !== nothing ? 1 : 0))
    log_det_sigma = log(det(sigma))
    log_likelihood = -0.5 * T_eff * (k * log(2π) + log_det_sigma + k)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * log(T_eff)

    return VECMResult(
        alpha,
        beta,
        gamma,
        pi,
        const_term,
        coint_rank,
        lags,
        residuals,
        sigma,
        T_eff,
        k,
        det_order,
        aic,
        bic,
        log_likelihood,
    )
end


function _vecm_estimate_ols(
    data::AbstractMatrix{<:Real},
    coint_rank::Int,
    lags::Int,
    det_order::Int,
)
    T, k = size(data)

    # Get cointegrating vectors from Johansen
    johansen_result = johansen_test(data; lags=lags, det_order=det_order)
    beta = johansen_result.eigenvectors[:, 1:coint_rank]

    # Compute error correction terms
    ECT = data * beta  # T × r

    # Build differenced data
    dY = diff(data, dims=1)  # (T-1) × k
    T_eff = T - lags

    # ECT lagged one period
    ECT_lag = ECT[lags:end-1, :]  # (T-lags) × r

    # Build differenced lags
    n_sr_lags = lags - 1
    if n_sr_lags > 0
        dY_lags = zeros(T_eff, k * n_sr_lags)
        for j in 1:n_sr_lags
            start_idx = lags - j
            end_idx = T - 1 - j
            dY_lags[:, (j-1)*k+1:j*k] = dY[start_idx:end_idx, :]
        end
    else
        dY_lags = zeros(T_eff, 0)
    end

    # Dependent variable
    dY_dep = dY[lags:end, :]

    # Build regressor matrix
    if det_order >= 0
        X = hcat(ECT_lag, dY_lags, ones(T_eff))
    else
        X = size(dY_lags, 2) > 0 ? hcat(ECT_lag, dY_lags) : ECT_lag
    end

    # OLS
    XtX = X' * X
    XtY = X' * dY_dep

    coeffs = try
        XtX \ XtY
    catch
        pinv(XtX) * XtY
    end

    # Extract coefficients
    alpha = coeffs[1:coint_rank, :]'
    n_gamma_cols = k * n_sr_lags
    if n_sr_lags > 0
        gamma = coeffs[coint_rank+1:coint_rank+n_gamma_cols, :]'
    else
        gamma = zeros(k, 0)
    end

    if det_order >= 0
        const_term = coeffs[end, :]
    else
        const_term = nothing
    end

    # Compute residuals and covariance
    fitted = X * coeffs
    residuals = dY_dep - fitted
    sigma = (residuals' * residuals) / (T_eff - size(X, 2))

    # Π = αβ'
    pi = alpha * beta'

    # Information criteria
    n_params = k * (coint_rank + k * n_sr_lags + (const_term !== nothing ? 1 : 0))
    log_det_sigma = log(det(sigma))
    log_likelihood = -0.5 * T_eff * (k * log(2π) + log_det_sigma + k)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * log(T_eff)

    return VECMResult(
        alpha,
        beta,
        gamma,
        pi,
        const_term,
        coint_rank,
        lags,
        residuals,
        sigma,
        T_eff,
        k,
        det_order,
        aic,
        bic,
        log_likelihood,
    )
end


"""
    vecm_forecast(result, data; horizons=10)

Forecast from estimated VECM.

# Arguments
- `result::VECMResult`: Estimated VECM
- `data::AbstractMatrix{<:Real}`: Original data (T × k)
- `horizons::Int=10`: Number of forecast periods

# Returns
Matrix of forecasts (horizons × k).
"""
function vecm_forecast(
    result::VECMResult,
    data::AbstractMatrix{<:Real};
    horizons::Int=10,
)
    T, k = size(data)
    p = result.lags

    # Initialize with last observations
    Y_history = copy(data[end-p:end, :])  # Last p+1 observations
    dY_history = diff(Y_history, dims=1)  # Last p differences

    forecasts = zeros(horizons, k)

    for h in 1:horizons
        # Current level (last available)
        Y_t = Y_history[end, :]

        # Error correction term: β'Y_t
        ect = result.beta' * Y_t  # r × 1

        # Short-run component: Γ₁ΔY_{t-1} + Γ₂ΔY_{t-2} + ...
        sr = zeros(k)
        n_sr_lags = p - 1
        if n_sr_lags > 0 && size(result.gamma, 2) > 0
            for j in 1:n_sr_lags
                if j <= size(dY_history, 1)
                    Gamma_j = result.gamma[:, (j-1)*k+1:j*k]
                    sr .+= Gamma_j * dY_history[end-j+1, :]
                end
            end
        end

        # Forecast change: ΔY_{t+1} = α·ect + sr + c
        dY_forecast = result.alpha * ect .+ sr
        if result.const_term !== nothing
            dY_forecast .+= result.const_term
        end

        # Level forecast: Y_{t+1} = Y_t + ΔY_{t+1}
        Y_forecast = Y_t .+ dY_forecast
        forecasts[h, :] = Y_forecast

        # Update histories
        Y_history = vcat(Y_history[2:end, :], Y_forecast')
        dY_history = diff(Y_history, dims=1)
    end

    return forecasts
end


"""
    compute_error_correction_term(data, beta)

Compute error correction terms ECT = Y * β.

# Arguments
- `data::AbstractMatrix{<:Real}`: Time series data (T × k)
- `beta::AbstractMatrix{<:Real}`: Cointegrating vectors (k × r)

# Returns
Error correction terms (T × r matrix).
"""
function compute_error_correction_term(
    data::AbstractMatrix{<:Real},
    beta::AbstractMatrix{<:Real},
)
    return data * beta
end
