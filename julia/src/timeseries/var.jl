"""
Vector Autoregression (VAR) Estimation

Session 135: VAR model estimation for time series analysis.
"""
module VAR

using LinearAlgebra
using Statistics

using ..TimeSeriesTypes

export var_estimate, var_forecast, var_residuals


"""
    var_estimate(data; lags=1, var_names=nothing, include_constant=true)

Estimate a VAR(p) model using OLS equation-by-equation.

# Arguments
- `data::Matrix{Float64}`: Shape (n_obs, n_vars) time series data
- `lags::Int`: Lag order p
- `var_names::Union{Vector{String}, Nothing}`: Variable names
- `include_constant::Bool`: Whether to include intercept

# Returns
- `VARResult`: Estimation results

# Example
```julia
using Random
Random.seed!(42)
data = randn(200, 2)
result = var_estimate(data, lags=2)
println("AIC: ", round(result.aic, digits=2))
```
"""
function var_estimate(data::Matrix{Float64};
                      lags::Int=1,
                      var_names::Union{Vector{String}, Nothing}=nothing,
                      include_constant::Bool=true)
    n_obs, n_vars = size(data)

    if lags < 1
        error("Lags must be >= 1, got $lags")
    end

    if n_obs <= lags + 1
        error("Insufficient observations ($n_obs) for lag order $lags. " *
              "Need at least $(lags + 2) observations.")
    end

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    elseif length(var_names) != n_vars
        error("var_names length ($(length(var_names))) must match n_vars ($n_vars)")
    end

    # Build design matrix and dependent variable
    Y, X = build_var_matrices(data, lags, include_constant)

    # OLS estimation: B = (X'X)^{-1} X'Y
    XtX = X' * X
    XtY = X' * Y

    coefficients = try
        (XtX \ XtY)'
    catch
        (pinv(XtX) * XtY)'
    end

    # Compute residuals
    residuals = Y - X * coefficients'

    # Compute covariance matrix of residuals
    n_obs_effective = size(residuals, 1)
    n_params = size(X, 2)
    dof = max(n_obs_effective - n_params, 1)
    sigma = (residuals' * residuals) / dof

    # Compute log-likelihood
    log_likelihood = compute_log_likelihood(residuals, sigma, n_obs_effective, n_vars)

    # Compute information criteria
    n_params_total = n_vars * n_params
    aic = -2 * log_likelihood + 2 * n_params_total
    bic = -2 * log_likelihood + n_params_total * log(n_obs_effective)
    hqc = -2 * log_likelihood + 2 * n_params_total * log(log(n_obs_effective))

    VARResult(
        coefficients, residuals, aic, bic, hqc, lags, n_obs, n_obs_effective,
        var_names, sigma, log_likelihood
    )
end


"""Build design matrices for VAR estimation."""
function build_var_matrices(data::Matrix{Float64}, lags::Int, include_constant::Bool)
    n_obs, n_vars = size(data)
    n_effective = n_obs - lags

    # Dependent variable: Y_t for t = lags+1, ..., n_obs
    Y = data[lags+1:end, :]

    # Build X matrix
    n_cols = n_vars * lags + (include_constant ? 1 : 0)
    X = zeros(n_effective, n_cols)

    col_idx = 1

    # Intercept column
    if include_constant
        X[:, 1] .= 1.0
        col_idx = 2
    end

    # Lagged values
    for lag in 1:lags
        for var in 1:n_vars
            X[:, col_idx] = data[lags+1-lag:n_obs-lag, var]
            col_idx += 1
        end
    end

    return Y, X
end


"""
    var_forecast(result, data; steps=1)

Generate forecasts from estimated VAR model.

# Arguments
- `result::VARResult`: Estimated VAR model
- `data::Matrix{Float64}`: Historical data
- `steps::Int`: Number of forecast steps

# Returns
- `Matrix{Float64}`: Shape (steps, n_vars) forecasted values
"""
function var_forecast(result::VARResult, data::Matrix{Float64}; steps::Int=1)
    if steps < 1
        error("steps must be >= 1, got $steps")
    end

    n_vars_result = length(result.var_names)
    lags = result.lags

    if size(data, 2) != n_vars_result
        error("Data has $(size(data, 2)) variables, model expects $n_vars_result")
    end

    if size(data, 1) < lags
        error("Need at least $lags observations for forecasting, got $(size(data, 1))")
    end

    forecasts = zeros(steps, n_vars_result)
    intercepts = get_intercepts(result)

    # Use last `lags` observations
    history = copy(data[end-lags+1:end, :])

    for h in 1:steps
        y_hat = copy(intercepts)

        for lag in 1:lags
            if h <= lag
                # Use historical data
                y_past = history[lags - lag + h, :]
            else
                # Use previous forecasts
                y_past = forecasts[h - lag, :]
            end

            A_lag = get_lag_matrix(result, lag)
            y_hat += A_lag * y_past
        end

        forecasts[h, :] = y_hat
    end

    return forecasts
end


"""
    var_residuals(result, data)

Compute residuals from VAR model for given data.
"""
function var_residuals(result::VARResult, data::Matrix{Float64})
    if size(data, 2) != length(result.var_names)
        error("Data has $(size(data, 2)) variables, model expects $(length(result.var_names))")
    end

    Y, X = build_var_matrices(data, result.lags, true)
    fitted = X * result.coefficients'
    return Y - fitted
end


"""Compute Gaussian log-likelihood for VAR model."""
function compute_log_likelihood(residuals::Matrix{Float64}, sigma::Matrix{Float64},
                                 n_obs::Int, n_vars::Int)
    det_sigma = det(sigma)
    if det_sigma <= 0
        # Use pseudo-determinant
        eigvals = eigvals!(copy(sigma))
        eigvals = eigvals[eigvals .> 1e-10]
        log_det = sum(log.(eigvals))
    else
        log_det = log(det_sigma)
    end

    log_lik = -0.5 * n_obs * (n_vars * log(2π) + log_det + n_vars)
    return log_lik
end

end # module
