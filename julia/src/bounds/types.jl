"""
Type definitions for partial identification bounds.

Implements Manski bounds and Lee (2009) attrition bounds.

References:
- Manski (1990). Nonparametric Bounds on Treatment Effects
- Lee (2009). Training, Wages, and Sample Selection
"""

using Statistics

# ============================================================================
# Abstract Types
# ============================================================================

abstract type AbstractBoundsProblem{T} end
abstract type AbstractBoundsEstimator end
abstract type AbstractBoundsResult end

# ============================================================================
# Manski Bounds Types
# ============================================================================

"""
    ManskiBoundsResult{T<:Real}

Result from Manski bounds estimation.

# Fields
- `bounds_lower::T`: Lower bound on ATE
- `bounds_upper::T`: Upper bound on ATE
- `bounds_width::T`: Width of identification region
- `point_identified::Bool`: True if bounds collapse to a point
- `assumptions::Symbol`: :worst_case, :mtr, :mts, :mtr_mts, or :iv
- `mtr_direction::Union{Nothing, Symbol}`: :positive or :negative for MTR
- `naive_ate::T`: Naive difference-in-means estimate
- `ate_in_bounds::Bool`: Whether naive estimate is within bounds
- `n_treated::Int`: Number of treated observations
- `n_control::Int`: Number of control observations
- `outcome_support::Tuple{T, T}`: (Y_min, Y_max) outcome bounds
"""
struct ManskiBoundsResult{T<:Real} <: AbstractBoundsResult
    bounds_lower::T
    bounds_upper::T
    bounds_width::T
    point_identified::Bool
    assumptions::Symbol
    mtr_direction::Union{Nothing, Symbol}
    naive_ate::T
    ate_in_bounds::Bool
    n_treated::Int
    n_control::Int
    outcome_support::Tuple{T, T}
end


"""
    ManskiIVBoundsResult{T<:Real}

Result from Manski IV bounds estimation.

# Fields
- `bounds_lower::T`: Lower bound on ATE
- `bounds_upper::T`: Upper bound on ATE
- `bounds_width::T`: Width of identification region
- `point_identified::Bool`: True if bounds collapse to a point
- `assumptions::Symbol`: Always :iv
- `iv_strength::T`: Instrument strength measure
- `complier_share::T`: Estimated complier share
- `n_treated::Int`: Number treated
- `n_control::Int`: Number control
- `n_iv_1::Int`: Number with Z=1
- `n_iv_0::Int`: Number with Z=0
- `outcome_support::Tuple{T, T}`: Outcome bounds
"""
struct ManskiIVBoundsResult{T<:Real} <: AbstractBoundsResult
    bounds_lower::T
    bounds_upper::T
    bounds_width::T
    point_identified::Bool
    assumptions::Symbol
    iv_strength::T
    complier_share::T
    n_treated::Int
    n_control::Int
    n_iv_1::Int
    n_iv_0::Int
    outcome_support::Tuple{T, T}
end


# ============================================================================
# Lee Bounds Types
# ============================================================================

"""
    LeeBoundsResult{T<:Real}

Result from Lee (2009) bounds for sample selection.

# Fields
- `bounds_lower::T`: Lower bound on ATE
- `bounds_upper::T`: Upper bound on ATE
- `bounds_width::T`: Width of identification region
- `ci_lower::T`: Bootstrap CI lower bound
- `ci_upper::T`: Bootstrap CI upper bound
- `point_identified::Bool`: True if no differential attrition
- `trimming_proportion::T`: Proportion trimmed
- `trimmed_group::Symbol`: :treated, :control, or :none
- `attrition_treated::T`: Attrition rate for treated
- `attrition_control::T`: Attrition rate for control
- `n_treated_observed::Int`: Observed treated count
- `n_control_observed::Int`: Observed control count
- `n_trimmed::Int`: Number of observations trimmed
- `monotonicity::Symbol`: :positive or :negative
"""
struct LeeBoundsResult{T<:Real} <: AbstractBoundsResult
    bounds_lower::T
    bounds_upper::T
    bounds_width::T
    ci_lower::T
    ci_upper::T
    point_identified::Bool
    trimming_proportion::T
    trimmed_group::Symbol
    attrition_treated::T
    attrition_control::T
    n_treated_observed::Int
    n_control_observed::Int
    n_trimmed::Int
    monotonicity::Symbol
end
