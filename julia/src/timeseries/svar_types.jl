"""
Structural VAR Types

Session 137: Data structures for Structural VAR analysis.
"""
module SVARTypes

using LinearAlgebra

export IdentificationMethod, SVARResult, IRFResult, FEVDResult
export is_stable, is_just_identified, is_over_identified
export get_structural_coefficient, get_response, get_decomposition

"""SVAR identification methods."""
@enum IdentificationMethod begin
    CHOLESKY
    SHORT_RUN
    LONG_RUN
    SIGN
end


"""
    SVARResult

Result from Structural VAR estimation.

SVAR decomposes reduced-form errors u_t into structural shocks ε_t:
    u_t = B₀⁻¹ ε_t

where ε_t ~ N(0, I) are orthogonal structural shocks.

# Fields
- `var_coefficients`: VAR coefficient matrix
- `var_residuals`: Reduced-form residuals
- `var_sigma`: Reduced-form covariance matrix
- `B0_inv`: Impact matrix (n_vars × n_vars)
- `B0`: Structural matrix (inverse of B0_inv)
- `structural_shocks`: Structural shock series
- `identification`: Identification method used
- `n_restrictions`: Number of identifying restrictions
- `n_vars`: Number of variables
- `n_obs`: Number of observations
- `lags`: VAR lag order
- `var_names`: Variable names
- `ordering`: Variable ordering (for Cholesky)
- `log_likelihood`: Log-likelihood
"""
struct SVARResult
    var_coefficients::Matrix{Float64}
    var_residuals::Matrix{Float64}
    var_sigma::Matrix{Float64}
    B0_inv::Matrix{Float64}
    B0::Matrix{Float64}
    structural_shocks::Matrix{Float64}
    identification::IdentificationMethod
    n_restrictions::Int
    n_vars::Int
    n_obs::Int
    lags::Int
    var_names::Vector{String}
    ordering::Vector{String}
    log_likelihood::Float64
end

function SVARResult(;
    var_coefficients::Matrix{Float64},
    var_residuals::Matrix{Float64},
    var_sigma::Matrix{Float64},
    B0_inv::Matrix{Float64},
    B0::Matrix{Float64},
    structural_shocks::Matrix{Float64},
    identification::IdentificationMethod,
    n_restrictions::Int,
    n_vars::Int,
    n_obs::Int,
    lags::Int,
    var_names::Vector{String},
    ordering::Vector{String}=String[],
    log_likelihood::Float64=0.0
)
    SVARResult(
        var_coefficients, var_residuals, var_sigma,
        B0_inv, B0, structural_shocks,
        identification, n_restrictions,
        n_vars, n_obs, lags, var_names, ordering, log_likelihood
    )
end

is_just_identified(r::SVARResult) = r.n_restrictions == r.n_vars * (r.n_vars - 1) ÷ 2
is_over_identified(r::SVARResult) = r.n_restrictions > r.n_vars * (r.n_vars - 1) ÷ 2

function get_structural_coefficient(r::SVARResult, shock_var::Int, response_var::Int)
    r.B0_inv[response_var, shock_var]
end

function Base.show(io::IO, r::SVARResult)
    print(io, "SVARResult(n_vars=$(r.n_vars), lags=$(r.lags), ",
          "identification=$(r.identification), n_restrictions=$(r.n_restrictions))")
end


"""
    IRFResult

Impulse Response Function result.

# Fields
- `irf`: (n_vars, n_vars, horizons+1) impulse response matrix
- `irf_lower`: Lower confidence band (or nothing)
- `irf_upper`: Upper confidence band (or nothing)
- `horizons`: Maximum horizon
- `cumulative`: Whether IRF is cumulative
- `orthogonalized`: Whether shocks are orthogonalized
- `var_names`: Variable names
- `alpha`: Confidence level
- `n_bootstrap`: Number of bootstrap replications
"""
struct IRFResult
    irf::Array{Float64,3}
    irf_lower::Union{Array{Float64,3},Nothing}
    irf_upper::Union{Array{Float64,3},Nothing}
    horizons::Int
    cumulative::Bool
    orthogonalized::Bool
    var_names::Vector{String}
    alpha::Float64
    n_bootstrap::Int
end

function IRFResult(;
    irf::Array{Float64,3},
    irf_lower::Union{Array{Float64,3},Nothing}=nothing,
    irf_upper::Union{Array{Float64,3},Nothing}=nothing,
    horizons::Int,
    cumulative::Bool=false,
    orthogonalized::Bool=true,
    var_names::Vector{String},
    alpha::Float64=0.05,
    n_bootstrap::Int=0
)
    IRFResult(irf, irf_lower, irf_upper, horizons, cumulative,
              orthogonalized, var_names, alpha, n_bootstrap)
end

has_confidence_bands(r::IRFResult) = r.irf_lower !== nothing && r.irf_upper !== nothing

function get_response(r::IRFResult, response_var::Int, shock_var::Int)
    r.irf[response_var, shock_var, :]
end

function get_response(r::IRFResult, response_var::Int, shock_var::Int, horizon::Int)
    r.irf[response_var, shock_var, horizon+1]  # 1-indexed in Julia
end

function Base.show(io::IO, r::IRFResult)
    cum_str = r.cumulative ? ", cumulative" : ""
    ci_str = has_confidence_bands(r) ? ", CI=$(round(100*(1-r.alpha)))%" : ""
    print(io, "IRFResult(n_vars=$(size(r.irf, 1)), horizons=$(r.horizons)$(cum_str)$(ci_str))")
end


"""
    FEVDResult

Forecast Error Variance Decomposition result.

# Fields
- `fevd`: (n_vars, n_vars, horizons+1) decomposition matrix
- `horizons`: Maximum horizon
- `var_names`: Variable names
"""
struct FEVDResult
    fevd::Array{Float64,3}
    horizons::Int
    var_names::Vector{String}
end

function FEVDResult(;
    fevd::Array{Float64,3},
    horizons::Int,
    var_names::Vector{String}
)
    FEVDResult(fevd, horizons, var_names)
end

function get_decomposition(r::FEVDResult, response_var::Int, horizon::Int)
    r.fevd[response_var, :, horizon+1]  # 1-indexed
end

function get_contribution(r::FEVDResult, response_var::Int, shock_var::Int, horizon::Int)
    r.fevd[response_var, shock_var, horizon+1]
end

function validate_rows_sum_to_one(r::FEVDResult; tol::Float64=1e-10)
    for h in 1:(r.horizons+1)
        row_sums = sum(r.fevd[:, :, h], dims=2)
        if !all(isapprox.(row_sums, 1.0, atol=tol))
            return false
        end
    end
    return true
end

function Base.show(io::IO, r::FEVDResult)
    print(io, "FEVDResult(n_vars=$(size(r.fevd, 1)), horizons=$(r.horizons))")
end

end # module
