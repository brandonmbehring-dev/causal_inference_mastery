"""
Type definitions for Marginal Treatment Effects (MTE) estimation.

Implements Heckman & Vytlacil (2005) framework for treatment effect heterogeneity.
Follows SciML Problem-Estimator-Solution architecture.
"""

# ============================================================================
# MTE Problem Types
# ============================================================================

"""
    MTEProblem{T<:Real}

Problem specification for MTE estimation via local IV.

# Fields
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector{T}`: Binary treatment indicator D in {0, 1}
- `instrument::Union{Vector{T}, Matrix{T}}`: Instrument(s) Z
- `covariates::Union{Nothing, Matrix{T}}`: Optional covariates X
- `n_grid::Int`: Number of grid points for MTE evaluation
- `trim_fraction::T`: Fraction to trim from each propensity tail
"""
struct MTEProblem{T<:Real}
    outcome::Vector{T}
    treatment::Vector{T}
    instrument::Union{Vector{T}, Matrix{T}}
    covariates::Union{Nothing, Matrix{T}}
    n_grid::Int
    trim_fraction::T

    function MTEProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        instrument::Union{Vector{T}, Matrix{T}},
        covariates::Union{Nothing, Matrix{T}} = nothing,
        n_grid::Int = 50,
        trim_fraction::Union{T, Nothing} = nothing
    ) where T<:Real
        # Handle default for trim_fraction
        trim_frac = trim_fraction === nothing ? T(0.01) : trim_fraction
        n = length(outcome)

        # Validate lengths
        length(treatment) == n || error("Treatment length mismatch")
        if instrument isa Vector
            length(instrument) == n || error("Instrument length mismatch")
        else
            size(instrument, 1) == n || error("Instrument row count mismatch")
        end
        if covariates !== nothing
            size(covariates, 1) == n || error("Covariates row count mismatch")
        end

        # Validate treatment is binary
        all(t -> t == zero(T) || t == one(T), treatment) ||
            error("Treatment must be binary (0 or 1)")

        # Validate treatment variation
        length(unique(treatment)) >= 2 ||
            error("No treatment variation")

        # Validate no NaN/Inf
        !any(isnan, outcome) || error("NaN in outcome")
        !any(isinf, outcome) || error("Inf in outcome")

        # Validate parameters
        n_grid >= 2 || error("n_grid must be at least 2")
        zero(T) <= trim_frac < T(0.5) ||
            error("trim_fraction must be in [0, 0.5)")

        new{T}(outcome, treatment, instrument, covariates, n_grid, trim_frac)
    end
end


"""
    LATEProblem{T<:Real}

Problem specification for LATE estimation with binary instrument.

# Fields
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector{T}`: Binary treatment D in {0, 1}
- `instrument::Vector{T}`: Binary instrument Z in {0, 1}
- `covariates::Union{Nothing, Matrix{T}}`: Optional exogenous controls
"""
struct LATEProblem{T<:Real}
    outcome::Vector{T}
    treatment::Vector{T}
    instrument::Vector{T}
    covariates::Union{Nothing, Matrix{T}}

    function LATEProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        instrument::Vector{T},
        covariates::Union{Nothing, Matrix{T}} = nothing
    ) where T<:Real
        n = length(outcome)

        # Validate lengths
        length(treatment) == n || error("Treatment length mismatch")
        length(instrument) == n || error("Instrument length mismatch")
        if covariates !== nothing
            size(covariates, 1) == n || error("Covariates row count mismatch")
        end

        # Validate binary
        all(t -> t == zero(T) || t == one(T), treatment) ||
            error("Treatment must be binary (0 or 1)")
        all(z -> z == zero(T) || z == one(T), instrument) ||
            error("Instrument must be binary (0 or 1)")

        # Validate variation
        length(unique(treatment)) >= 2 || error("No treatment variation")
        length(unique(instrument)) >= 2 || error("No instrument variation")

        # Validate no NaN/Inf
        !any(isnan, outcome) || error("NaN in outcome")
        !any(isinf, outcome) || error("Inf in outcome")

        # Validate sufficient observations in each instrument group
        n_z1 = sum(instrument .== one(T))
        n_z0 = n - n_z1
        n_z1 >= 2 || error("Need at least 2 observations with Z=1")
        n_z0 >= 2 || error("Need at least 2 observations with Z=0")

        new{T}(outcome, treatment, instrument, covariates)
    end
end


# ============================================================================
# MTE Solution Types
# ============================================================================

"""
    MTESolution{T<:Real}

Result from MTE estimation via local IV or polynomial.

# Fields
- `mte_grid::Vector{T}`: MTE(u) at grid points
- `u_grid::Vector{T}`: Grid points u in [p_min, p_max]
- `se_grid::Vector{T}`: Bootstrap standard errors
- `ci_lower::Vector{T}`: Lower CI bound at each point
- `ci_upper::Vector{T}`: Upper CI bound at each point
- `propensity_support::Tuple{T, T}`: (min, max) propensity
- `n_obs::Int`: Total sample size
- `n_trimmed::Int`: Units trimmed for support
- `bandwidth::T`: Bandwidth used
- `method::Symbol`: :local_iv or :polynomial
"""
struct MTESolution{T<:Real}
    mte_grid::Vector{T}
    u_grid::Vector{T}
    se_grid::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    propensity_support::Tuple{T, T}
    n_obs::Int
    n_trimmed::Int
    bandwidth::T
    method::Symbol
end

# Keyword constructor
function MTESolution(;
    mte_grid::Vector{T},
    u_grid::Vector{T},
    se_grid::Vector{T},
    ci_lower::Vector{T},
    ci_upper::Vector{T},
    propensity_support::Tuple{T, T},
    n_obs::Int,
    n_trimmed::Int,
    bandwidth::T,
    method::Symbol
) where T<:Real
    MTESolution{T}(mte_grid, u_grid, se_grid, ci_lower, ci_upper,
                   propensity_support, n_obs, n_trimmed, bandwidth, method)
end


"""
    LATESolution{T<:Real}

Result from LATE estimation.

# Fields
- `late::T`: Local average treatment effect
- `se::T`: Standard error
- `ci_lower::T`: Lower 95% CI bound
- `ci_upper::T`: Upper 95% CI bound
- `pvalue::T`: Two-sided p-value
- `complier_share::T`: Proportion compliers
- `always_taker_share::T`: Proportion always-takers
- `never_taker_share::T`: Proportion never-takers
- `first_stage_coef::T`: First-stage coefficient
- `first_stage_f::T`: First-stage F-statistic
- `n_obs::Int`: Sample size
- `method::Symbol`: :wald or :twosls
"""
struct LATESolution{T<:Real}
    late::T
    se::T
    ci_lower::T
    ci_upper::T
    pvalue::T
    complier_share::T
    always_taker_share::T
    never_taker_share::T
    first_stage_coef::T
    first_stage_f::T
    n_obs::Int
    method::Symbol
end

function LATESolution(;
    late::T,
    se::T,
    ci_lower::T,
    ci_upper::T,
    pvalue::T,
    complier_share::T,
    always_taker_share::T,
    never_taker_share::T,
    first_stage_coef::T,
    first_stage_f::T,
    n_obs::Int,
    method::Symbol
) where T<:Real
    LATESolution{T}(late, se, ci_lower, ci_upper, pvalue,
                    complier_share, always_taker_share, never_taker_share,
                    first_stage_coef, first_stage_f, n_obs, method)
end


"""
    LATEBoundsResult{T<:Real}

Bounds on LATE when monotonicity may be violated.

# Fields
- `bounds_lower::T`: Lower bound
- `bounds_upper::T`: Upper bound
- `late_under_monotonicity::T`: Point estimate if monotonicity holds
- `first_stage::T`: First-stage coefficient
- `reduced_form::T`: Reduced-form coefficient
- `outcome_support::Tuple{T, T}`: (min, max) outcome
- `bounds_width::T`: Width of bounds
"""
struct LATEBoundsResult{T<:Real}
    bounds_lower::T
    bounds_upper::T
    late_under_monotonicity::T
    first_stage::T
    reduced_form::T
    outcome_support::Tuple{T, T}
    bounds_width::T
end

function LATEBoundsResult(;
    bounds_lower::T,
    bounds_upper::T,
    late_under_monotonicity::T,
    first_stage::T,
    reduced_form::T,
    outcome_support::Tuple{T, T},
    bounds_width::T
) where T<:Real
    LATEBoundsResult{T}(bounds_lower, bounds_upper, late_under_monotonicity,
                        first_stage, reduced_form, outcome_support, bounds_width)
end


"""
    ComplierResult{T<:Real}

Complier subpopulation characteristics.

# Fields
- `complier_mean_outcome_treated::T`: E[Y1 | Complier]
- `complier_mean_outcome_control::T`: E[Y0 | Complier]
- `complier_share::T`: Fraction of compliers
- `covariate_means::Union{Nothing, Vector{T}}`: Mean covariates for compliers
- `method::Symbol`: :kappa_weights or :bounds
"""
struct ComplierResult{T<:Real}
    complier_mean_outcome_treated::T
    complier_mean_outcome_control::T
    complier_share::T
    covariate_means::Union{Nothing, Vector{T}}
    method::Symbol
end

function ComplierResult(;
    complier_mean_outcome_treated::T,
    complier_mean_outcome_control::T,
    complier_share::T,
    covariate_means::Union{Nothing, Vector{T}} = nothing,
    method::Symbol = :kappa_weights
) where T<:Real
    ComplierResult{T}(complier_mean_outcome_treated, complier_mean_outcome_control,
                      complier_share, covariate_means, method)
end


"""
    PolicyResult{T<:Real}

Policy-relevant treatment effect from MTE integration.

# Fields
- `estimate::T`: Policy parameter estimate
- `se::T`: Standard error
- `ci_lower::T`: Lower CI bound
- `ci_upper::T`: Upper CI bound
- `parameter::Symbol`: :ate, :att, :atu, :prte, :late
- `weights_used::String`: Description of weighting scheme
- `n_obs::Int`: Sample size
"""
struct PolicyResult{T<:Real}
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    parameter::Symbol
    weights_used::String
    n_obs::Int
end

function PolicyResult(;
    estimate::T,
    se::T,
    ci_lower::T,
    ci_upper::T,
    parameter::Symbol,
    weights_used::String,
    n_obs::Int
) where T<:Real
    PolicyResult{T}(estimate, se, ci_lower, ci_upper, parameter, weights_used, n_obs)
end


"""
    CommonSupportResult{T<:Real}

Common support diagnostic result.

# Fields
- `has_support::Bool`: Whether common support exists
- `support_region::Tuple{T, T}`: (min, max) in common support
- `n_outside_support::Int`: Units outside support
- `fraction_outside::T`: Proportion outside
- `recommendation::String`: Trimming recommendation
"""
struct CommonSupportResult{T<:Real}
    has_support::Bool
    support_region::Tuple{T, T}
    n_outside_support::Int
    fraction_outside::T
    recommendation::String
end

function CommonSupportResult(;
    has_support::Bool,
    support_region::Tuple{T, T},
    n_outside_support::Int,
    fraction_outside::T,
    recommendation::String
) where T<:Real
    CommonSupportResult{T}(has_support, support_region, n_outside_support,
                           fraction_outside, recommendation)
end
