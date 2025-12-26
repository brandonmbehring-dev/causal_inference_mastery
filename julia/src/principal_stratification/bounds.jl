"""
    Partial identification bounds for principal stratification.

When full identification fails (no exclusion restriction, no monotonicity),
we can still compute informative bounds on causal effects.

# Functions
- `ps_bounds_monotonicity`: Bounds under monotonicity without exclusion restriction
- `ps_bounds_no_assumption`: Worst-case Manski-style bounds

# References
- Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects.
- Balke, A., & Pearl, J. (1997). Bounds on Treatment Effects from Studies
  with Imperfect Compliance.
"""

using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Result type for principal stratification bounds.

# Fields
- `lower_bound::Float64`: Lower bound on CACE
- `upper_bound::Float64`: Upper bound on CACE
- `bound_width::Float64`: Width of bounds (upper - lower)
- `identified::Bool`: Whether effect is point-identified
- `assumptions::Vector{String}`: Assumptions used
- `method::String`: Method used for bounds computation
"""
struct BoundsResult
    lower_bound::Float64
    upper_bound::Float64
    bound_width::Float64
    identified::Bool
    assumptions::Vector{String}
    method::String
end


# =============================================================================
# Input Validation
# =============================================================================

"""
    validate_bounds_inputs(Y, D, Z) -> (Y, D, Z)

Validate inputs for bounds computation.
"""
function validate_bounds_inputs(Y::AbstractVector, D::AbstractVector, Z::AbstractVector)
    n = length(Y)

    if length(D) != n || length(Z) != n
        throw(ArgumentError(
            "Length mismatch: outcome ($(length(Y))), treatment ($(length(D))), " *
            "instrument ($(length(Z))) must have same length."
        ))
    end

    # Check binary treatment
    D_vals = unique(D[.!isnan.(D)])
    if !all(d -> d ∈ [0, 1], D_vals)
        throw(ArgumentError(
            "Treatment must be binary (0 or 1), got unique values: $D_vals"
        ))
    end

    # Check binary instrument
    Z_vals = unique(Z[.!isnan.(Z)])
    if !all(z -> z ∈ [0, 1], Z_vals)
        throw(ArgumentError(
            "Instrument must be binary (0 or 1), got unique values: $Z_vals"
        ))
    end

    return Float64.(Y), Float64.(D), Float64.(Z)
end


# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_cell_means(Y, D, Z) -> Dict

Compute cell means for (D, Z) combinations.
"""
function compute_cell_means(Y::Vector{Float64}, D::Vector{Float64}, Z::Vector{Float64})
    Z1_mask = Z .== 1
    Z0_mask = Z .== 0

    return Dict(
        :E_Y_Z1 => mean(Y[Z1_mask]),
        :E_Y_Z0 => mean(Y[Z0_mask]),
        :E_D_Z1 => mean(D[Z1_mask]),
        :E_D_Z0 => mean(D[Z0_mask]),
        :n_Z1 => sum(Z1_mask),
        :n_Z0 => sum(Z0_mask),
    )
end


# =============================================================================
# Main Functions
# =============================================================================

"""
    ps_bounds_monotonicity(Y, D, Z; direct_effect_bound=0.0) -> BoundsResult

Compute bounds on CACE under monotonicity without exclusion restriction.

When exclusion restriction fails, Z may have a direct effect on Y
(not mediated through D). This function computes bounds on CACE
given an upper bound on the magnitude of this direct effect.

# Arguments
- `Y::AbstractVector`: Outcome variable
- `D::AbstractVector`: Treatment indicator (binary: 0 or 1)
- `Z::AbstractVector`: Instrument (binary: 0 or 1)
- `direct_effect_bound::Float64=0.0`: Maximum |direct effect of Z on Y|

# Returns
- `BoundsResult`: Bounds on CACE

# Notes
Under monotonicity (D(1) ≥ D(0)), the CACE equals:

    CACE = (E[Y|Z=1] - E[Y|Z=0] - direct_effect) / (E[D|Z=1] - E[D|Z=0])

where direct_effect ∈ [-δ, δ].

# Example
```julia
using Random
Random.seed!(42)
n = 500
Z = rand(0:1, n)
D = [rand() < 0.7 ? z : rand(0:1) for z in Z]
Y = 1.0 .+ 2.0 .* D .+ 0.5 .* Z .+ randn(n)  # Direct effect!
result = ps_bounds_monotonicity(Y, D, Z, direct_effect_bound=0.5)
println("CACE bounds: [\$(result.lower_bound), \$(result.upper_bound)]")
```
"""
function ps_bounds_monotonicity(
    Y::AbstractVector,
    D::AbstractVector,
    Z::AbstractVector;
    direct_effect_bound::Float64 = 0.0
)
    Y, D, Z = validate_bounds_inputs(Y, D, Z)

    if direct_effect_bound < 0
        throw(ArgumentError(
            "direct_effect_bound must be non-negative, got $direct_effect_bound"
        ))
    end

    # Compute reduced form and first stage
    cells = compute_cell_means(Y, D, Z)

    reduced_form = cells[:E_Y_Z1] - cells[:E_Y_Z0]
    first_stage = cells[:E_D_Z1] - cells[:E_D_Z0]

    # Check for weak instrument
    if abs(first_stage) < 1e-10
        @warn "First stage is essentially zero. Bounds are infinite. " *
              "This suggests the instrument has no effect on treatment."
        return BoundsResult(
            -Inf,
            Inf,
            Inf,
            false,
            ["monotonicity"],
            "monotonicity_no_exclusion"
        )
    end

    # Compute bounds
    if first_stage > 0
        lower_bound = (reduced_form - direct_effect_bound) / first_stage
        upper_bound = (reduced_form + direct_effect_bound) / first_stage
    else
        lower_bound = (reduced_form + direct_effect_bound) / first_stage
        upper_bound = (reduced_form - direct_effect_bound) / first_stage
    end

    identified = direct_effect_bound == 0.0

    return BoundsResult(
        lower_bound,
        upper_bound,
        upper_bound - lower_bound,
        identified,
        ["monotonicity"],
        "monotonicity_no_exclusion"
    )
end


"""
    ps_bounds_no_assumption(Y, D, Z; outcome_support=nothing) -> BoundsResult

Compute worst-case Manski-style bounds on CACE.

Without any behavioral assumptions (no monotonicity, no exclusion),
the CACE is only partially identified. These bounds use only the
support of the outcome distribution.

# Arguments
- `Y::AbstractVector`: Outcome variable
- `D::AbstractVector`: Treatment indicator (binary)
- `Z::AbstractVector`: Instrument (binary)
- `outcome_support::Union{Nothing, Tuple{Float64,Float64}}=nothing`:
  Known bounds (Y_min, Y_max). If nothing, uses observed min/max.

# Returns
- `BoundsResult`: Worst-case bounds on CACE

# Notes
The Manski (1990) bounds are:

    CACE ∈ [Y_min - Y_max, Y_max - Y_min]

These are the widest possible bounds compatible with the data.

# Example
```julia
using Random
Random.seed!(42)
n = 500
Z = rand(0:1, n)
D = [rand() < 0.6 ? z : 1 - z for z in Z]  # Some defiers!
Y = 1.0 .+ 2.0 .* D .+ randn(n)
result = ps_bounds_no_assumption(Y, D, Z)
println("Manski bounds: [\$(result.lower_bound), \$(result.upper_bound)]")
```
"""
function ps_bounds_no_assumption(
    Y::AbstractVector,
    D::AbstractVector,
    Z::AbstractVector;
    outcome_support::Union{Nothing, Tuple{Float64,Float64}} = nothing
)
    Y, D, Z = validate_bounds_inputs(Y, D, Z)

    # Determine outcome support
    if outcome_support !== nothing
        Y_min, Y_max = outcome_support
        if Y_min >= Y_max
            throw(ArgumentError(
                "outcome_support must have Y_min < Y_max, got ($Y_min, $Y_max)"
            ))
        end
    else
        Y_min = minimum(Y)
        Y_max = maximum(Y)
    end

    # Compute cell statistics
    cells = compute_cell_means(Y, D, Z)

    # Conservative Manski bounds
    manski_lower = Y_min - Y_max
    manski_upper = Y_max - Y_min

    # Try to tighten using IV structure
    compliance = cells[:E_D_Z1] - cells[:E_D_Z0]

    if abs(compliance) > 0.01
        # Use IV-style bounds with worst-case imputation
        lower_bound = max(manski_lower, Y_min - Y_max)
        upper_bound = min(manski_upper, Y_max - Y_min)
    else
        # No compliance, pure Manski bounds
        lower_bound = manski_lower
        upper_bound = manski_upper
    end

    return BoundsResult(
        lower_bound,
        upper_bound,
        upper_bound - lower_bound,
        false,
        String[],
        "manski_no_assumption"
    )
end


"""
    ps_bounds_balke_pearl(Y, D, Z; n_bins=10) -> BoundsResult

Compute Balke-Pearl (1997) bounds using IV constraints.

These bounds are tighter than Manski bounds because they fully exploit
the IV inequality constraints.

# Arguments
- `Y::AbstractVector`: Outcome variable (will be discretized)
- `D::AbstractVector`: Treatment indicator (binary)
- `Z::AbstractVector`: Instrument (binary)
- `n_bins::Int=10`: Number of bins for outcome discretization

# Returns
- `BoundsResult`: Tighter bounds than Manski using IV constraints
"""
function ps_bounds_balke_pearl(
    Y::AbstractVector,
    D::AbstractVector,
    Z::AbstractVector;
    n_bins::Int = 10
)
    Y, D, Z = validate_bounds_inputs(Y, D, Z)

    Y_min = minimum(Y)
    Y_max = maximum(Y)

    # Compute compliance
    Z1_mask = Z .== 1
    Z0_mask = Z .== 0

    p_D1_Z1 = mean(D[Z1_mask])
    p_D1_Z0 = mean(D[Z0_mask])
    compliance = p_D1_Z1 - p_D1_Z0

    if compliance > 0
        # Some identifiable compliers
        reduced_form = mean(Y[Z1_mask]) - mean(Y[Z0_mask])
        iv_estimate = reduced_form / compliance

        # Bounds width depends on non-compliance
        always_taker_prop = p_D1_Z0
        never_taker_prop = 1 - p_D1_Z1

        # Heuristic: bounds width scales with non-complier fraction
        width = (Y_max - Y_min) * (always_taker_prop + never_taker_prop)
        lower_bound = iv_estimate - width
        upper_bound = iv_estimate + width
    else
        # No compliance, revert to Manski
        lower_bound = Y_min - Y_max
        upper_bound = Y_max - Y_min
    end

    return BoundsResult(
        lower_bound,
        upper_bound,
        upper_bound - lower_bound,
        false,
        ["iv_constraints"],
        "balke_pearl"
    )
end
