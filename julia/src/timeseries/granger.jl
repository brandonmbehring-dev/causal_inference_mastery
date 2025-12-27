"""
Granger Causality Tests

Session 135: Pairwise and multivariate Granger causality analysis.
"""
module Granger

using LinearAlgebra
using Statistics
using Distributions

using ..TimeSeriesTypes

export granger_causality, granger_causality_matrix, bidirectional_granger


"""
    granger_causality(data; lags=1, alpha=0.05, cause_idx=2, effect_idx=1, var_names=nothing)

Test Granger causality between two time series.

Tests H0: cause does not Granger-cause effect
vs H1: cause Granger-causes effect

# Arguments
- `data::Matrix{Float64}`: Shape (n_obs, n_vars) time series data
- `lags::Int`: Number of lags to include
- `alpha::Float64`: Significance level
- `cause_idx::Int`: Index of potential cause variable
- `effect_idx::Int`: Index of effect variable
- `var_names::Union{Vector{String}, Nothing}`: Variable names

# Returns
- `GrangerResult`: Test result

# Example
```julia
using Random
Random.seed!(42)
n = 200
x = randn(n)
y = zeros(n)
for t in 2:n
    y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + randn() * 0.5
end
data = hcat(y, x)
result = granger_causality(data, lags=2)
println("X Granger-causes Y: ", result.granger_causes)
```
"""
function granger_causality(data::Matrix{Float64};
                           lags::Int=1,
                           alpha::Float64=0.05,
                           cause_idx::Int=2,
                           effect_idx::Int=1,
                           var_names::Union{Vector{String}, Nothing}=nothing)
    n_obs, n_vars = size(data)

    if n_vars < 2
        error("Need at least 2 variables, got $n_vars")
    end

    if lags < 1
        error("lags must be >= 1, got $lags")
    end

    if n_obs <= 2 * lags + 1
        error("Insufficient observations ($n_obs) for $lags lags. " *
              "Need at least $(2 * lags + 2).")
    end

    if cause_idx > n_vars || effect_idx > n_vars
        error("cause_idx ($cause_idx) and effect_idx ($effect_idx) " *
              "must be <= n_vars ($n_vars)")
    end

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    end

    # Extract the two series
    y = data[:, effect_idx]  # Effect variable
    x = data[:, cause_idx]   # Cause variable

    # Build design matrices
    n_effective = n_obs - lags

    # Unrestricted model: Y_t ~ const + Y_{t-1:t-p} + X_{t-1:t-p}
    X_unrestricted = build_granger_design(y, x, lags, true)

    # Restricted model: Y_t ~ const + Y_{t-1:t-p}
    X_restricted = build_granger_design(y, x, lags, false)

    # Dependent variable
    y_dep = y[lags+1:end]

    # OLS estimation
    beta_u = X_unrestricted \ y_dep
    beta_r = X_restricted \ y_dep

    # Compute residuals and RSS
    resid_u = y_dep - X_unrestricted * beta_u
    resid_r = y_dep - X_restricted * beta_r

    rss_u = sum(resid_u.^2)
    rss_r = sum(resid_r.^2)

    # Compute R-squared
    tss = sum((y_dep .- mean(y_dep)).^2)
    r2_u = tss > 0 ? 1 - rss_u / tss : 0.0
    r2_r = tss > 0 ? 1 - rss_r / tss : 0.0

    # F-test
    df_num = lags
    df_denom = n_effective - size(X_unrestricted, 2)

    if df_denom <= 0 || rss_u <= 0
        f_stat = 0.0
        p_value = 1.0
    else
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
        f_stat = max(0.0, f_stat)
        p_value = 1.0 - cdf(FDist(df_num, df_denom), f_stat)
    end

    # Compute AIC for both models
    k_u = size(X_unrestricted, 2)
    k_r = size(X_restricted, 2)
    aic_u = n_effective * log(rss_u / n_effective) + 2 * k_u
    aic_r = n_effective * log(rss_r / n_effective) + 2 * k_r

    granger_causes = p_value < alpha

    GrangerResult(
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
end


"""Build design matrix for Granger causality regression."""
function build_granger_design(y::Vector{Float64}, x::Vector{Float64},
                               lags::Int, include_cause::Bool)
    n = length(y)
    n_effective = n - lags

    # Start with constant
    cols = [ones(n_effective)]

    # Add lagged y values
    for lag in 1:lags
        push!(cols, y[lags+1-lag:n-lag])
    end

    # Add lagged x values if unrestricted
    if include_cause
        for lag in 1:lags
            push!(cols, x[lags+1-lag:n-lag])
        end
    end

    return hcat(cols...)
end


"""
    granger_causality_matrix(data; lags=1, alpha=0.05, var_names=nothing)

Compute pairwise Granger causality for all variable pairs.

# Returns
- `Dict{Tuple{String,String}, GrangerResult}`: Pairwise results
- `BitMatrix`: (n_vars, n_vars) causality matrix
"""
function granger_causality_matrix(data::Matrix{Float64};
                                   lags::Int=1,
                                   alpha::Float64=0.05,
                                   var_names::Union{Vector{String}, Nothing}=nothing)
    n_obs, n_vars = size(data)

    if var_names === nothing
        var_names = ["var_$i" for i in 1:n_vars]
    end

    pairwise_results = Dict{Tuple{String,String}, GrangerResult}()
    causality_matrix = falses(n_vars, n_vars)

    for i in 1:n_vars
        for j in 1:n_vars
            i == j && continue

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
        end
    end

    return pairwise_results, causality_matrix
end


"""
    bidirectional_granger(data; lags=1, alpha=0.05, var_names=nothing)

Test Granger causality in both directions between two variables.

For data with columns [col1, col2]:
- First result: col2 -> col1 (second column causes first)
- Second result: col1 -> col2 (first column causes second)

# Returns
- `Tuple{GrangerResult, GrangerResult}`: (col2 -> col1 result, col1 -> col2 result)
"""
function bidirectional_granger(data::Matrix{Float64};
                                lags::Int=1,
                                alpha::Float64=0.05,
                                var_names::Union{Vector{String}, Nothing}=nothing)
    if size(data, 2) != 2
        error("Data must have shape (n_obs, 2), got $(size(data))")
    end

    if var_names === nothing
        var_names = ["X", "Y"]
    end

    # col2 -> col1: cause_idx=2, effect_idx=1 (matches Python's default)
    result_xy = granger_causality(
        data,
        lags=lags,
        alpha=alpha,
        cause_idx=2,
        effect_idx=1,
        var_names=var_names,
    )

    # col1 -> col2: cause_idx=1, effect_idx=2
    result_yx = granger_causality(
        data,
        lags=lags,
        alpha=alpha,
        cause_idx=1,
        effect_idx=2,
        var_names=var_names,
    )

    return result_xy, result_yx
end

end # module
