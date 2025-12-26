"""
Principal Stratification types and data structures.

Implements the SciML Problem-Estimator-Solution architecture for CACE estimation.

# Types
- `CACEProblem`: Immutable data specification for CACE estimation
- `CACESolution`: Results from CACE (Complier Average Causal Effect) estimation
- `StrataProportions`: Principal strata proportions

# Principal Strata
Under binary treatment and instrument:
- Compliers: D(0)=0, D(1)=1 (take treatment iff assigned)
- Always-takers: D(0)=1, D(1)=1 (always take treatment)
- Never-takers: D(0)=0, D(1)=0 (never take treatment)
- Defiers: D(0)=1, D(1)=0 (ruled out by monotonicity)

# Key Result
Under standard IV assumptions:
    CACE = LATE = (Reduced Form) / (First Stage)

# References
- Frangakis, C. E., & Rubin, D. B. (2002). Principal Stratification.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). LATE paper.
"""

using LinearAlgebra
using Statistics
using Distributions

# Import abstract types from problems module
# Note: These should be available from main CausalEstimators module
abstract type AbstractPSProblem{T,P} end
abstract type AbstractPSEstimator end
abstract type AbstractPSSolution end

"""
    StrataProportions{T<:Real}

Estimated proportions of principal strata.

Under monotonicity (no defiers), we identify:
- π_c (compliers) = P(D=1|Z=1) - P(D=1|Z=0)
- π_a (always-takers) = P(D=1|Z=0)
- π_n (never-takers) = P(D=0|Z=1) = 1 - P(D=1|Z=1)

# Fields
- `compliers::T`: Proportion who comply with assignment
- `always_takers::T`: Proportion who always take treatment
- `never_takers::T`: Proportion who never take treatment
- `compliers_se::T`: Standard error of complier proportion
"""
struct StrataProportions{T<:Real}
    compliers::T
    always_takers::T
    never_takers::T
    compliers_se::T

    function StrataProportions(
        compliers::T,
        always_takers::T,
        never_takers::T,
        compliers_se::T
    ) where {T<:Real}
        # Validate proportions
        if compliers < 0 || compliers > 1
            throw(ArgumentError("compliers must be in [0,1], got $compliers"))
        end
        if always_takers < 0 || always_takers > 1
            throw(ArgumentError("always_takers must be in [0,1], got $always_takers"))
        end
        if never_takers < 0 || never_takers > 1
            throw(ArgumentError("never_takers must be in [0,1], got $never_takers"))
        end
        if compliers_se < 0
            throw(ArgumentError("compliers_se must be non-negative, got $compliers_se"))
        end

        # Check they sum to ~1
        total = compliers + always_takers + never_takers
        if !isapprox(total, 1.0, atol=1e-6)
            throw(ArgumentError("Strata proportions must sum to 1, got $total"))
        end

        new{T}(compliers, always_takers, never_takers, compliers_se)
    end
end

"""
    CACEProblem{T<:Real, P<:NamedTuple}

Specification of a CACE estimation problem.

# Fields
- `outcome::Vector{T}`: Outcome variable Y (n×1)
- `treatment::Vector{Bool}`: Actual treatment received D (n×1)
- `instrument::Vector{Bool}`: Random assignment Z (n×1)
- `covariates::Union{Matrix{T}, Nothing}`: Optional covariates X
- `parameters::P`: Estimation parameters (alpha, inference)

# Identification Requirements
Under monotonicity (D(1) ≥ D(0)):
    CACE = E[Y(1) - Y(0) | Compliers] = LATE

# Examples
```julia
problem = CACEProblem(
    Y,                              # Outcome
    D .== 1,                        # Treatment (Bool)
    Z .== 1,                        # Instrument (Bool)
    nothing,                        # No covariates
    (alpha=0.05, inference=:robust)
)
```
"""
struct CACEProblem{T<:Real, P<:NamedTuple} <: AbstractPSProblem{T,P}
    outcome::Vector{T}
    treatment::Vector{Bool}
    instrument::Vector{Bool}
    covariates::Union{Matrix{T}, Nothing}
    parameters::P

    function CACEProblem(
        outcome::Vector{T},
        treatment::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
        instrument::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
        covariates::Union{Matrix{T}, Nothing},
        parameters::P
    ) where {T<:Real, P<:NamedTuple}
        n = length(outcome)

        # Convert to Bool
        treatment = convert(Vector{Bool}, treatment .!= 0)
        instrument = convert(Vector{Bool}, instrument .!= 0)

        # Validate dimensions
        if length(treatment) != n
            throw(ArgumentError(
                "treatment must have same length as outcome (got $(length(treatment)), expected $n)"
            ))
        end

        if length(instrument) != n
            throw(ArgumentError(
                "instrument must have same length as outcome (got $(length(instrument)), expected $n)"
            ))
        end

        if !isnothing(covariates) && size(covariates, 1) != n
            throw(ArgumentError(
                "covariates must have $n rows (got $(size(covariates, 1)))"
            ))
        end

        # Validate data
        if any(isnan, outcome) || any(isinf, outcome)
            throw(ArgumentError("outcome contains NaN or Inf values"))
        end

        # Check for instrument variation
        if all(instrument) || all(.!instrument)
            throw(ArgumentError("instrument must have variation (both 0 and 1 present)"))
        end

        # Check for compliers (first-stage > 0)
        D_z1 = treatment[instrument]
        D_z0 = treatment[.!instrument]
        first_stage = mean(D_z1) - mean(D_z0)

        if first_stage <= 0
            throw(ArgumentError(
                "No compliers detected: first-stage coefficient = $(round(first_stage, digits=4)) ≤ 0. " *
                "This violates the relevance assumption. Check that Z causes D."
            ))
        end

        # Validate parameters
        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        if !(0 < parameters.alpha < 1)
            throw(ArgumentError("alpha must be in (0,1), got $(parameters.alpha)"))
        end

        new{T,P}(outcome, treatment, instrument, covariates, parameters)
    end
end

# Convenience constructor for simple cases
function CACEProblem(
    outcome::Vector{T},
    treatment::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
    instrument::Union{Vector{Bool}, BitVector, Vector{<:Integer}};
    alpha::Float64=0.05,
    inference::Symbol=:robust
) where {T<:Real}
    CACEProblem(outcome, treatment, instrument, nothing, (alpha=alpha, inference=inference))
end

"""
    CACESolution{T<:Real}

Results from CACE (Complier Average Causal Effect) estimation.

# Fields
- `cace::T`: Complier Average Causal Effect estimate
- `se::T`: Standard error
- `ci_lower::T`: Lower CI bound
- `ci_upper::T`: Upper CI bound
- `z_stat::T`: Z-statistic
- `pvalue::T`: P-value for H₀: CACE = 0
- `strata_proportions::StrataProportions{T}`: Estimated strata proportions
- `first_stage_coef::T`: First-stage coefficient (= complier proportion)
- `first_stage_se::T`: SE of first-stage
- `first_stage_f::T`: First-stage F-statistic
- `reduced_form::T`: Reduced-form coefficient
- `reduced_form_se::T`: SE of reduced-form
- `n::Int`: Sample size
- `n_treated_assigned::Int`: Number assigned to treatment
- `n_control_assigned::Int`: Number assigned to control
- `method::Symbol`: Estimation method used

# Interpretation
- CACE is the average treatment effect for COMPLIERS ONLY
- If first_stage_f < 10, weak instrument concern
- strata_proportions gives the population breakdown
"""
struct CACESolution{T<:Real} <: AbstractPSSolution
    cace::T
    se::T
    ci_lower::T
    ci_upper::T
    z_stat::T
    pvalue::T
    strata_proportions::StrataProportions{T}
    first_stage_coef::T
    first_stage_se::T
    first_stage_f::T
    reduced_form::T
    reduced_form_se::T
    n::Int
    n_treated_assigned::Int
    n_control_assigned::Int
    method::Symbol
end

"""
    CACETwoSLS

Two-Stage Least Squares estimator for CACE.

Exploits the key result: CACE = LATE under standard IV assumptions.

# Options
- `add_intercept::Bool`: Whether to add intercept (default: true)

# Example
```julia
problem = CACEProblem(Y, D, Z)
solution = solve(problem, CACETwoSLS())
```
"""
struct CACETwoSLS <: AbstractPSEstimator
    add_intercept::Bool

    CACETwoSLS(; add_intercept::Bool=true) = new(add_intercept)
end

"""
    WaldEstimator

Simple Wald/ratio estimator for CACE.

CACE = [E(Y|Z=1) - E(Y|Z=0)] / [E(D|Z=1) - E(D|Z=0)]

No covariates supported. Uses delta method for SE.
"""
struct WaldEstimator <: AbstractPSEstimator end

# Display methods
function Base.show(io::IO, problem::CACEProblem{T}) where {T}
    n = length(problem.outcome)
    n1 = sum(problem.instrument)
    n0 = n - n1

    println(io, "CACEProblem{$T}")
    println(io, "  Sample size: $n")
    println(io, "  Assigned to treatment: $n1 ($(round(100*n1/n, digits=1))%)")
    println(io, "  Assigned to control: $n0")

    if !isnothing(problem.covariates)
        println(io, "  Covariates: $(size(problem.covariates, 2))")
    end

    println(io, "  Alpha: $(problem.parameters.alpha)")
end

function Base.show(io::IO, solution::CACESolution{T}) where {T}
    println(io, "CACESolution{$T}")
    println(io, "  CACE: $(round(solution.cace, digits=4)) ± $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  p-value: $(round(solution.pvalue, digits=4))")
    println(io, "  ")
    println(io, "  First-stage: $(round(solution.first_stage_coef, digits=4)) (F = $(round(solution.first_stage_f, digits=2)))")
    println(io, "  Reduced-form: $(round(solution.reduced_form, digits=4))")
    println(io, "  ")
    println(io, "  Strata proportions:")
    println(io, "    Compliers: $(round(100*solution.strata_proportions.compliers, digits=1))%")
    println(io, "    Always-takers: $(round(100*solution.strata_proportions.always_takers, digits=1))%")
    println(io, "    Never-takers: $(round(100*solution.strata_proportions.never_takers, digits=1))%")
    println(io, "  ")
    println(io, "  Sample: $(solution.n) ($(solution.n_treated_assigned) treated, $(solution.n_control_assigned) control)")

    if solution.first_stage_f < 10
        println(io, "  ⚠️  Weak instrument warning: F-stat < 10")
    end
end

# Export strata proportions as NamedTuple for interop
function Base.NamedTuple(sp::StrataProportions)
    (
        compliers = sp.compliers,
        always_takers = sp.always_takers,
        never_takers = sp.never_takers,
        compliers_se = sp.compliers_se
    )
end
