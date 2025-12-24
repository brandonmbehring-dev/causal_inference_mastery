"""
Type definitions for Shift-Share (Bartik) Instrumental Variables estimation.

The shift-share instrument (Bartik 1991) constructs an IV from:
- Shares: Local exposure to sectors (share_{i,s})
- Shifts: Aggregate shocks to sectors (shock_s)
- Instrument: Z_i = sum_s(share_{i,s} * shock_s)

Identification requires either:
- Exogeneity of shares (Goldsmith-Pinkham et al. 2020)
- Exogeneity of shocks (Borusyak et al. 2022)

References:
- Bartik (1991). Who Benefits from State and Local Economic Development Policies?
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik Instruments: What, When, Why, and How
- Borusyak, Hull, Jaravel (2022). Quasi-Experimental Shift-Share Research Designs
- Rotemberg (1983). Instrument Variable Estimation of Misspecified Models
"""

using Statistics
using LinearAlgebra
using Distributions

# ============================================================================
# Abstract Types
# ============================================================================

abstract type AbstractShiftShareProblem{T} end
abstract type AbstractShiftShareEstimator end
abstract type AbstractShiftShareSolution end

# ============================================================================
# Problem Types
# ============================================================================

"""
    ShiftShareProblem{T<:Real}

Problem specification for Shift-Share IV estimation.

# Fields
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector{T}`: Endogenous treatment D
- `shares::Matrix{T}`: Sector shares (n × S), rows should sum to ~1
- `shocks::Vector{T}`: Aggregate shocks to sectors (S)
- `covariates::Union{Nothing, Matrix{T}}`: Optional exogenous controls X
- `clusters::Union{Nothing, Vector}`: Optional cluster identifiers
- `alpha::T`: Significance level for inference (default 0.05)
"""
struct ShiftShareProblem{T<:Real} <: AbstractShiftShareProblem{T}
    outcome::Vector{T}
    treatment::Vector{T}
    shares::Matrix{T}
    shocks::Vector{T}
    covariates::Union{Nothing, Matrix{T}}
    clusters::Union{Nothing, Vector}
    alpha::T

    function ShiftShareProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        shares::Matrix{T},
        shocks::Vector{T},
        covariates::Union{Nothing, Matrix{T}} = nothing,
        clusters::Union{Nothing, Vector} = nothing,
        alpha::Real = 0.05
    ) where T<:Real
        alpha = T(alpha)
        n = length(outcome)
        n_sectors = length(shocks)

        # Validate lengths
        length(treatment) == n ||
            error("Treatment length mismatch: expected $n, got $(length(treatment))")
        size(shares, 1) == n ||
            error("Shares row count mismatch: expected $n, got $(size(shares, 1))")
        size(shares, 2) == n_sectors ||
            error("Shares columns ($(size(shares, 2))) != shocks length ($n_sectors)")

        if covariates !== nothing
            size(covariates, 1) == n ||
                error("Covariates row count mismatch: expected $n, got $(size(covariates, 1))")
        end

        if clusters !== nothing
            length(clusters) == n ||
                error("Clusters length mismatch: expected $n, got $(length(clusters))")
        end

        # Validate no NaN/Inf
        !any(isnan, outcome) || error("NaN values in outcome")
        !any(isinf, outcome) || error("Inf values in outcome")
        !any(isnan, treatment) || error("NaN values in treatment")
        !any(isinf, treatment) || error("Inf values in treatment")
        !any(isnan, shares) || error("NaN values in shares")
        !any(isinf, shares) || error("Inf values in shares")
        !any(isnan, shocks) || error("NaN values in shocks")
        !any(isinf, shocks) || error("Inf values in shocks")

        # Validate treatment variation
        std(treatment) > 1e-10 || error("No variation in treatment")

        # Validate sample size
        n >= 10 || error("Insufficient sample size (n=$n). Need at least 10.")

        # Validate alpha
        zero(T) < alpha < one(T) || error("alpha must be in (0, 1)")

        # Validate shares are 2D
        ndims(shares) == 2 || error("shares must be 2D matrix")

        new{T}(outcome, treatment, shares, shocks, covariates, clusters, alpha)
    end
end

# ============================================================================
# Solution Types
# ============================================================================

"""
    RotembergDiagnostics{T<:Real}

Rotemberg (1983) weight diagnostics for shift-share instruments.

The Rotemberg weights decompose the overall IV estimate into contributions
from each sector/shock. Negative weights indicate potential violations of
monotonicity.

# Fields
- `weights::Vector{T}`: Rotemberg weight for each sector (sum of abs = 1)
- `negative_weight_share::T`: Fraction of total weight that is negative (0-1)
- `top_5_sectors::Vector{Int}`: Indices of 5 sectors with largest absolute weights
- `top_5_weights::Vector{T}`: Weights for the top 5 sectors
- `herfindahl::T`: Herfindahl index of weights (concentration measure)
"""
struct RotembergDiagnostics{T<:Real}
    weights::Vector{T}
    negative_weight_share::T
    top_5_sectors::Vector{Int}
    top_5_weights::Vector{T}
    herfindahl::T
end


"""
    FirstStageSSResult{T<:Real}

First-stage regression results for Shift-Share IV.

# Fields
- `f_statistic::T`: F-statistic for instrument strength
- `f_pvalue::T`: P-value for F-test
- `partial_r2::T`: Partial R-squared (after partialing out controls)
- `coefficient::T`: First-stage coefficient on instrument
- `se::T`: Standard error of coefficient
- `t_stat::T`: T-statistic
- `weak_iv_warning::Bool`: True if F < 10 (Stock-Yogo threshold)
"""
struct FirstStageSSResult{T<:Real}
    f_statistic::T
    f_pvalue::T
    partial_r2::T
    coefficient::T
    se::T
    t_stat::T
    weak_iv_warning::Bool
end


"""
    ShiftShareSolution{T<:Real}

Solution from Shift-Share IV estimation.

# Fields
- `estimate::T`: Estimated causal effect (2SLS coefficient on D)
- `se::T`: Standard error (robust or clustered)
- `t_stat::T`: T-statistic for estimate
- `p_value::T`: Two-sided p-value
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `first_stage::FirstStageSSResult{T}`: First-stage diagnostics
- `rotemberg::RotembergDiagnostics{T}`: Rotemberg weight diagnostics
- `n_obs::Int`: Number of observations
- `n_sectors::Int`: Number of sectors/shocks
- `share_sum_mean::T`: Mean of share row sums (should be ~1 if normalized)
- `inference::Symbol`: Inference method (:robust or :clustered)
- `alpha::T`: Significance level used
- `instrument::Vector{T}`: Constructed Bartik instrument
"""
struct ShiftShareSolution{T<:Real} <: AbstractShiftShareSolution
    estimate::T
    se::T
    t_stat::T
    p_value::T
    ci_lower::T
    ci_upper::T
    first_stage::FirstStageSSResult{T}
    rotemberg::RotembergDiagnostics{T}
    n_obs::Int
    n_sectors::Int
    share_sum_mean::T
    inference::Symbol
    alpha::T
    instrument::Vector{T}
end


# ============================================================================
# Estimator Types
# ============================================================================

"""
    ShiftShareIV <: AbstractShiftShareEstimator

Shift-Share (Bartik) Instrumental Variables estimator.

Constructs an instrument from sector shares and aggregate shocks,
then runs 2SLS with the constructed instrument.

# Fields
- `inference::Symbol`: Standard error type, :robust or :clustered
"""
struct ShiftShareIV <: AbstractShiftShareEstimator
    inference::Symbol

    function ShiftShareIV(; inference::Symbol = :robust)
        inference in (:robust, :clustered) ||
            error("inference must be :robust or :clustered, got :$inference")
        new(inference)
    end
end
