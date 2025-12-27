"""
Time Series Causal Inference Types

Session 135: Data structures for Granger causality and VAR analysis.
"""
module TimeSeriesTypes

using LinearAlgebra

export GrangerResult, VARResult, ADFResult, LagSelectionResult
export n_vars, n_params_per_eq, get_lag_matrix, get_intercepts
export get_optimal_by_criterion


"""
    GrangerResult

Result from pairwise Granger causality test.

Tests H0: X does not Granger-cause Y
(past values of X do not help predict Y given past values of Y)
"""
struct GrangerResult
    cause_var::String
    effect_var::String
    f_statistic::Float64
    p_value::Float64
    lags::Int
    granger_causes::Bool
    alpha::Float64
    r2_unrestricted::Float64
    r2_restricted::Float64
    aic_unrestricted::Float64
    aic_restricted::Float64
    df_num::Int
    df_denom::Int
    rss_unrestricted::Float64
    rss_restricted::Float64
end

function GrangerResult(;
    cause_var::String,
    effect_var::String,
    f_statistic::Float64,
    p_value::Float64,
    lags::Int,
    granger_causes::Bool,
    alpha::Float64=0.05,
    r2_unrestricted::Float64=0.0,
    r2_restricted::Float64=0.0,
    aic_unrestricted::Float64=0.0,
    aic_restricted::Float64=0.0,
    df_num::Int=0,
    df_denom::Int=0,
    rss_unrestricted::Float64=0.0,
    rss_restricted::Float64=0.0,
)
    GrangerResult(
        cause_var, effect_var, f_statistic, p_value, lags, granger_causes,
        alpha, r2_unrestricted, r2_restricted, aic_unrestricted, aic_restricted,
        df_num, df_denom, rss_unrestricted, rss_restricted
    )
end

function Base.show(io::IO, r::GrangerResult)
    direction = r.granger_causes ? "→" : "↛"
    sig = r.granger_causes ? "*" : ""
    print(io, "GrangerResult($(r.cause_var) $direction $(r.effect_var)$sig, ",
          "F=$(round(r.f_statistic, digits=3)), p=$(round(r.p_value, digits=4)), ",
          "lags=$(r.lags))")
end


"""
    VARResult

Result from VAR (Vector Autoregression) estimation.
"""
struct VARResult
    coefficients::Matrix{Float64}
    residuals::Matrix{Float64}
    aic::Float64
    bic::Float64
    hqc::Float64
    lags::Int
    n_obs::Int
    n_obs_effective::Int
    var_names::Vector{String}
    sigma::Matrix{Float64}
    log_likelihood::Float64
end

function n_vars(r::VARResult)
    length(r.var_names)
end

function n_params_per_eq(r::VARResult)
    n_vars(r) * r.lags + 1
end

"""Get coefficient matrix for specific lag."""
function get_lag_matrix(r::VARResult, lag::Int)
    if lag < 1 || lag > r.lags
        error("Lag must be between 1 and $(r.lags)")
    end
    k = n_vars(r)
    start_idx = 2 + (lag - 1) * k
    end_idx = start_idx + k - 1
    return r.coefficients[:, start_idx:end_idx]
end

"""Get intercept vector."""
function get_intercepts(r::VARResult)
    return r.coefficients[:, 1]
end

function Base.show(io::IO, r::VARResult)
    print(io, "VARResult(n_vars=$(n_vars(r)), lags=$(r.lags), ",
          "n_obs=$(r.n_obs_effective), AIC=$(round(r.aic, digits=2)))")
end


"""
    ADFResult

Augmented Dickey-Fuller test result.
"""
struct ADFResult
    statistic::Float64
    p_value::Float64
    lags::Int
    n_obs::Int
    critical_values::Dict{String, Float64}
    is_stationary::Bool
    regression::String
    alpha::Float64
end

function ADFResult(;
    statistic::Float64,
    p_value::Float64,
    lags::Int,
    n_obs::Int,
    critical_values::Dict{String, Float64},
    is_stationary::Bool,
    regression::String="c",
    alpha::Float64=0.05,
)
    ADFResult(statistic, p_value, lags, n_obs, critical_values, is_stationary,
              regression, alpha)
end

function Base.show(io::IO, r::ADFResult)
    status = r.is_stationary ? "Stationary" : "Non-stationary"
    print(io, "ADFResult($status, stat=$(round(r.statistic, digits=4)), ",
          "p=$(round(r.p_value, digits=4)), lags=$(r.lags))")
end


"""
    LagSelectionResult

Result from lag order selection.
"""
struct LagSelectionResult
    optimal_lag::Int
    criterion::String
    all_values::Dict{Int, Float64}
    all_lags::Vector{Int}
    aic_values::Dict{Int, Float64}
    bic_values::Dict{Int, Float64}
    hqc_values::Dict{Int, Float64}
end

function get_optimal_by_criterion(r::LagSelectionResult, criterion::String)
    values = if criterion == "aic"
        r.aic_values
    elseif criterion == "bic"
        r.bic_values
    elseif criterion == "hqc"
        r.hqc_values
    else
        error("Unknown criterion: $criterion")
    end

    if isempty(values)
        error("No values computed for $criterion")
    end

    return argmin(values)
end

function Base.show(io::IO, r::LagSelectionResult)
    print(io, "LagSelectionResult(optimal=$(r.optimal_lag), ",
          "criterion=$(r.criterion), tested=$(length(r.all_lags)) lags)")
end

end # module
