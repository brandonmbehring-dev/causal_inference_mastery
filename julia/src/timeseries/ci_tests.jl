"""
Conditional Independence Tests for Time Series

Session 136: CI tests for PCMCI algorithm.
"""
module CITests

using LinearAlgebra
using Statistics
using Distributions

using ..PCMCITypes

export parcorr_test, run_ci_test


"""
    parcorr_test(data, x_idx, y_idx, z_indices; x_lag=0, y_lag=0, alpha=0.05)

Partial correlation conditional independence test for time series.

Tests X_{t-x_lag} ⊥ Y_{t-y_lag} | Z where Z is a set of lagged variables.

# Arguments
- `data::Matrix{Float64}`: (n_obs, n_vars) time series data
- `x_idx::Int`: Index of X variable
- `y_idx::Int`: Index of Y variable
- `z_indices::Vector{Tuple{Int,Int}}`: (variable_index, lag) for conditioning set Z
- `x_lag::Int`: Lag for X variable
- `y_lag::Int`: Lag for Y variable
- `alpha::Float64`: Significance level

# Returns
- `CITestResult`: Test result
"""
function parcorr_test(data::Matrix{Float64}, x_idx::Int, y_idx::Int,
                      z_indices::Vector{Tuple{Int,Int}};
                      x_lag::Int=0, y_lag::Int=0, alpha::Float64=0.05)
    n_obs, n_vars = size(data)

    # Determine effective sample size
    all_lags = [x_lag, y_lag]
    for (_, lag) in z_indices
        push!(all_lags, lag)
    end
    max_lag = maximum(all_lags)
    n_effective = n_obs - max_lag

    if n_effective <= length(z_indices) + 2
        error("Insufficient observations ($n_effective) for conditioning set " *
              "of size $(length(z_indices)). Need at least $(length(z_indices) + 3).")
    end

    # Extract time-aligned data
    x_start = max_lag - x_lag + 1
    x_end = n_obs - (x_lag > 0 ? x_lag : 0)
    x_series = data[x_start:x_end, x_idx]

    y_start = max_lag - y_lag + 1
    y_end = n_obs - (y_lag > 0 ? y_lag : 0)
    y_series = data[y_start:y_end, y_idx]

    if isempty(z_indices)
        # Unconditional correlation
        rho = cor(x_series, y_series)
        dof = n_effective - 2
    else
        # Build conditioning matrix Z
        z_matrix = zeros(n_effective, length(z_indices))
        for (i, (var_idx, lag)) in enumerate(z_indices)
            z_start = max_lag - lag + 1
            z_end = n_obs - (lag > 0 ? lag : 0)
            z_matrix[:, i] = data[z_start:z_end, var_idx]
        end

        rho = partial_correlation(x_series, y_series, z_matrix)
        dof = n_effective - length(z_indices) - 2
    end

    if dof <= 0
        return CITestResult(0.0, 1.0, true, 0)
    end

    # Compute t-statistic
    if abs(rho) > 1 - 1e-10
        t_stat = sign(rho) * 1000.0
        p_value = 0.0
    elseif isnan(rho)
        t_stat = 0.0
        p_value = 1.0
    else
        t_stat = rho * sqrt(dof / (1 - rho^2 + 1e-10))
        p_value = 2 * (1 - cdf(TDist(dof), abs(t_stat)))
    end

    is_independent = p_value >= alpha

    CITestResult(rho, p_value, is_independent, dof)
end


"""Compute partial correlation via regression residuals."""
function partial_correlation(x::Vector{Float64}, y::Vector{Float64},
                             z::Matrix{Float64})
    n = length(x)

    # Add intercept
    z_with_const = hcat(ones(n), z)

    try
        # QR decomposition for numerical stability
        Q, R = qr(z_with_const)
        Q_mat = Matrix(Q)

        # Residuals from regressing x and y on z
        x_resid = x - Q_mat * (Q_mat' * x)
        y_resid = y - Q_mat * (Q_mat' * y)

        # Correlation of residuals
        x_centered = x_resid .- mean(x_resid)
        y_centered = y_resid .- mean(y_resid)

        numerator = sum(x_centered .* y_centered)
        denominator = sqrt(sum(x_centered.^2) * sum(y_centered.^2))

        if denominator < 1e-10
            return 0.0
        end

        return numerator / denominator
    catch
        return 0.0
    end
end


"""
    run_ci_test(data, source, target, source_lag, conditioning_set;
                ci_test="parcorr", alpha=0.05)

Run conditional independence test for time series.

Convenience wrapper that handles the common case of testing X_{t-τ} ⊥ Y_t | Z.
"""
function run_ci_test(data::Matrix{Float64}, source::Int, target::Int,
                     source_lag::Int, conditioning_set::Vector{Tuple{Int,Int}};
                     ci_test::String="parcorr", alpha::Float64=0.05)
    if ci_test != "parcorr"
        error("Only parcorr CI test is implemented in Julia. Got: $ci_test")
    end

    parcorr_test(data, source, target, conditioning_set;
                 x_lag=source_lag, y_lag=0, alpha=alpha)
end

end # module
