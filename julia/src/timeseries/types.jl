"""
Time Series Causal Inference Types

Session 135: Data structures for Granger causality and VAR analysis.
"""
module TimeSeriesTypes

using LinearAlgebra

export GrangerResult, VARResult, ADFResult, LagSelectionResult
export KPSSResult, PPResult, ConfirmatoryResult
export JohansenResult, EngleGrangerResult, VECMResult
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


"""
    KPSSResult

KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test result.

Tests H0: series is trend-stationary (stationary around deterministic trend)
vs H1: series has unit root (non-stationary).

IMPORTANT: Opposite null hypothesis from ADF test!
- KPSS: H0 = stationary (low stat = stationary)
- ADF: H0 = unit root (low stat = stationary)
"""
struct KPSSResult
    statistic::Float64
    p_value::Float64
    lags::Int
    n_obs::Int
    critical_values::Dict{String, Float64}
    is_stationary::Bool
    regression::String
    alpha::Float64
end

function KPSSResult(;
    statistic::Float64,
    p_value::Float64,
    lags::Int,
    n_obs::Int,
    critical_values::Dict{String, Float64},
    is_stationary::Bool,
    regression::String="c",
    alpha::Float64=0.05,
)
    KPSSResult(statistic, p_value, lags, n_obs, critical_values, is_stationary,
               regression, alpha)
end

function Base.show(io::IO, r::KPSSResult)
    status = r.is_stationary ? "Stationary" : "Non-stationary"
    print(io, "KPSSResult($status, stat=$(round(r.statistic, digits=4)), ",
          "p=$(round(r.p_value, digits=4)), lags=$(r.lags))")
end


"""
    PPResult

Phillips-Perron test result.

Tests H0: series has unit root (non-stationary)
vs H1: series is stationary.

Like ADF but uses Newey-West HAC correction instead of augmented lags.
Robust to heteroskedasticity and autocorrelation of unknown form.
"""
struct PPResult
    statistic::Float64
    p_value::Float64
    lags::Int
    n_obs::Int
    critical_values::Dict{String, Float64}
    is_stationary::Bool
    regression::String
    alpha::Float64
    rho_stat::Float64
end

function PPResult(;
    statistic::Float64,
    p_value::Float64,
    lags::Int,
    n_obs::Int,
    critical_values::Dict{String, Float64},
    is_stationary::Bool,
    regression::String="c",
    alpha::Float64=0.05,
    rho_stat::Float64=0.0,
)
    PPResult(statistic, p_value, lags, n_obs, critical_values, is_stationary,
             regression, alpha, rho_stat)
end

function Base.show(io::IO, r::PPResult)
    status = r.is_stationary ? "Stationary" : "Non-stationary"
    print(io, "PPResult($status, stat=$(round(r.statistic, digits=4)), ",
          "p=$(round(r.p_value, digits=4)), lags=$(r.lags))")
end


"""
    ConfirmatoryResult

Combined ADF + KPSS confirmatory stationarity test result.

Combines opposite-null tests for stronger inference:
- ADF: H0 = unit root
- KPSS: H0 = stationary
"""
struct ConfirmatoryResult
    adf::ADFResult
    kpss::KPSSResult
    interpretation::String
    conclusion::String
end

function Base.show(io::IO, r::ConfirmatoryResult)
    print(io, "ConfirmatoryResult($(r.conclusion))")
end


"""
    JohansenResult

Johansen cointegration test result.

Tests for cointegration rank r in a VAR system of n variables.
Uses both trace and maximum eigenvalue tests.

# Fields
- `rank::Int`: Estimated cointegration rank (0 to n_vars-1)
- `trace_stats::Vector{Float64}`: Trace statistics for each null hypothesis r=0,1,...,n-1
- `trace_crit::Vector{Float64}`: Critical values for trace statistics
- `trace_pvalues::Vector{Float64}`: P-values for trace statistics
- `max_eigen_stats::Vector{Float64}`: Max eigenvalue statistics for each null
- `max_eigen_crit::Vector{Float64}`: Critical values for max eigenvalue statistics
- `max_eigen_pvalues::Vector{Float64}`: P-values for max eigenvalue statistics
- `eigenvalues::Vector{Float64}`: Eigenvalues from reduced rank regression (sorted descending)
- `eigenvectors::Matrix{Float64}`: Eigenvectors (columns are cointegrating vectors β)
- `adjustment::Matrix{Float64}`: Adjustment coefficients α (loading matrix)
- `lags::Int`: Number of VAR lags used
- `n_obs::Int`: Number of effective observations
- `n_vars::Int`: Number of variables
- `det_order::Int`: Deterministic order (-1, 0, or 1)
- `alpha::Float64`: Significance level used
"""
struct JohansenResult
    rank::Int
    trace_stats::Vector{Float64}
    trace_crit::Vector{Float64}
    trace_pvalues::Vector{Float64}
    max_eigen_stats::Vector{Float64}
    max_eigen_crit::Vector{Float64}
    max_eigen_pvalues::Vector{Float64}
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}
    adjustment::Matrix{Float64}
    lags::Int
    n_obs::Int
    n_vars::Int
    det_order::Int
    alpha::Float64
end

function Base.show(io::IO, r::JohansenResult)
    print(io, "JohansenResult(rank=$(r.rank), n_vars=$(r.n_vars), lags=$(r.lags))")
end


"""
    EngleGrangerResult

Engle-Granger two-step cointegration test result.

Simpler alternative to Johansen for bivariate case.
"""
struct EngleGrangerResult
    beta::Vector{Float64}
    residuals::Vector{Float64}
    adf_result::ADFResult
    coint_critical_values::Dict{String, Float64}
    is_cointegrated::Bool
end

function Base.show(io::IO, r::EngleGrangerResult)
    status = r.is_cointegrated ? "Cointegrated" : "Not cointegrated"
    print(io, "EngleGrangerResult($status, ADF stat=$(round(r.adf_result.statistic, digits=4)))")
end


"""
    VECMResult

Vector Error Correction Model (VECM) estimation result.

The VECM representation of a cointegrated VAR(p) is:

    ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

# Fields
- `alpha`: Adjustment coefficients (k × r matrix)
- `beta`: Cointegrating vectors (k × r matrix)
- `gamma`: Short-run dynamics (k × k*(p-1) matrix)
- `pi`: Long-run matrix αβ' (k × k)
- `const_term`: Constant term (k × 1 vector) or nothing
- `coint_rank`: Cointegration rank r
- `lags`: VAR lags (VECM uses p-1 differenced lags)
- `residuals`: Model residuals (T × k)
- `sigma`: Residual covariance (k × k)
- `n_obs`: Number of observations
- `n_vars`: Number of variables
- `det_order`: Deterministic terms: -1=none, 0=restricted const, 1=unrestricted
- `aic`: Akaike Information Criterion
- `bic`: Bayesian Information Criterion
- `log_likelihood`: Log-likelihood value
"""
struct VECMResult
    alpha::Matrix{Float64}
    beta::Matrix{Float64}
    gamma::Matrix{Float64}
    pi::Matrix{Float64}
    const_term::Union{Vector{Float64}, Nothing}
    coint_rank::Int
    lags::Int
    residuals::Matrix{Float64}
    sigma::Matrix{Float64}
    n_obs::Int
    n_vars::Int
    det_order::Int
    aic::Float64
    bic::Float64
    log_likelihood::Float64
end

function Base.show(io::IO, r::VECMResult)
    print(io, "VECMResult(rank=$(r.coint_rank), lags=$(r.lags), n_vars=$(r.n_vars), n_obs=$(r.n_obs), AIC=$(round(r.aic, digits=2)))")
end

end # module
