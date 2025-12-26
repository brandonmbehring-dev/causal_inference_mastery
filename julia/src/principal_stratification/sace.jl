"""
    Survivor Average Causal Effect (SACE) estimation.

SACE addresses the problem of "truncation by death" where outcomes are
undefined for units who don't survive (or are otherwise censored).

# Key Estimand
SACE = E[Y(1) - Y(0) | S(0)=1, S(1)=1]

The treatment effect for units who would survive under BOTH treatment conditions
("always-survivors" principal stratum).

# Functions
- `sace_bounds`: Compute bounds on SACE under various assumptions
- `sace_sensitivity`: Sensitivity analysis varying selection assumptions

# References
- Zhang, J. L., & Rubin, D. B. (2003). Estimation of Causal Effects via Principal
  Stratification When Some Outcomes Are Truncated by Death.
- Lee, D. S. (2009). Training, Wages, and Sample Selection: Estimating Sharp Bounds
  on Treatment Effects.
"""

using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Result type for SACE estimation.

# Fields
- `sace::Float64`: Point estimate (midpoint of bounds if not identified)
- `se::Float64`: Standard error (based on bounds width)
- `lower_bound::Float64`: Lower bound on SACE
- `upper_bound::Float64`: Upper bound on SACE
- `proportion_survivors_treat::Float64`: P(S=1 | D=1)
- `proportion_survivors_control::Float64`: P(S=1 | D=0)
- `n::Int`: Sample size
- `method::String`: Description of assumptions used
"""
struct SACEResult
    sace::Float64
    se::Float64
    lower_bound::Float64
    upper_bound::Float64
    proportion_survivors_treat::Float64
    proportion_survivors_control::Float64
    n::Int
    method::String
end


# =============================================================================
# Input Validation
# =============================================================================

"""
    validate_sace_inputs(Y, D, S, Z=nothing) -> (Y, D, S, Z)

Validate inputs for SACE estimation.
"""
function validate_sace_inputs(
    Y::AbstractVector,
    D::AbstractVector,
    S::AbstractVector,
    Z::Union{Nothing, AbstractVector} = nothing
)
    n = length(Y)

    if length(D) != n || length(S) != n
        throw(ArgumentError(
            "Length mismatch: outcome ($(length(Y))), treatment ($(length(D))), " *
            "survival ($(length(S))) must have same length."
        ))
    end

    # Check binary treatment
    D_vals = unique(D[.!isnan.(D)])
    if !all(d -> d ∈ [0, 1], D_vals)
        throw(ArgumentError(
            "Treatment must be binary (0 or 1), got unique values: $D_vals"
        ))
    end

    # Check binary survival
    S_vals = unique(S[.!isnan.(S)])
    if !all(s -> s ∈ [0, 1], S_vals)
        throw(ArgumentError(
            "Survival must be binary (0 or 1), got unique values: $S_vals"
        ))
    end

    if Z !== nothing
        if length(Z) != n
            throw(ArgumentError(
                "Instrument length ($(length(Z))) must match outcome length ($n)."
            ))
        end
        Z_vals = unique(Z[.!isnan.(Z)])
        if !all(z -> z ∈ [0, 1], Z_vals)
            throw(ArgumentError(
                "Instrument must be binary (0 or 1), got unique values: $Z_vals"
            ))
        end
        Z = Float64.(Z)
    end

    return Float64.(Y), Float64.(D), Float64.(S), Z
end


# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_survival_proportions(D, S, Z=nothing) -> Dict

Compute survival proportions by treatment/assignment status.
"""
function compute_survival_proportions(
    D::Vector{Float64},
    S::Vector{Float64},
    Z::Union{Nothing, Vector{Float64}} = nothing
)
    result = Dict{Symbol, Float64}()

    # By observed treatment
    D1_mask = D .== 1
    D0_mask = D .== 0
    result[:p_S1_D1] = sum(D1_mask) > 0 ? mean(S[D1_mask]) : NaN
    result[:p_S1_D0] = sum(D0_mask) > 0 ? mean(S[D0_mask]) : NaN

    # By instrument if available
    if Z !== nothing
        Z1_mask = Z .== 1
        Z0_mask = Z .== 0
        result[:p_S1_Z1] = sum(Z1_mask) > 0 ? mean(S[Z1_mask]) : NaN
        result[:p_S1_Z0] = sum(Z0_mask) > 0 ? mean(S[Z0_mask]) : NaN
    end

    return result
end


"""
    compute_bounds_no_assumption(Y, D, S, props, Y_min, Y_max) -> (lower, upper)

Lee (2009) bounds without monotonicity assumptions.
"""
function compute_bounds_no_assumption(
    Y::Vector{Float64},
    D::Vector{Float64},
    S::Vector{Float64},
    p_S1_D1::Float64,
    p_S1_D0::Float64,
    Y_min::Float64,
    Y_max::Float64
)
    # Trimming proportions
    trim_D1 = p_S1_D1 > 0 ? 1 - min(p_S1_D0 / p_S1_D1, 1) : 0
    trim_D0 = p_S1_D0 > 0 ? 1 - min(p_S1_D1 / p_S1_D0, 1) : 0

    D1_S1 = (D .== 1) .& (S .== 1)
    D0_S1 = (D .== 0) .& (S .== 1)

    Y_D1 = Y[D1_S1]
    Y_D0 = Y[D0_S1]

    if length(Y_D1) == 0 || length(Y_D0) == 0
        return (Y_min - Y_max, Y_max - Y_min)
    end

    # Trimmed means for bounds
    n_trim_D1 = Int(floor(length(Y_D1) * trim_D1))
    n_trim_D0 = Int(floor(length(Y_D0) * trim_D0))

    Y_D1_sorted = sort(Y_D1)
    Y_D0_sorted = sort(Y_D0)

    # Lower bound: trim top of D=1, trim bottom of D=0
    if n_trim_D1 > 0 && n_trim_D1 < length(Y_D1_sorted)
        E_Y_D1_trim_top = mean(Y_D1_sorted[1:end-n_trim_D1])
    else
        E_Y_D1_trim_top = mean(Y_D1_sorted)
    end

    if n_trim_D0 > 0 && n_trim_D0 < length(Y_D0_sorted)
        E_Y_D0_trim_bottom = mean(Y_D0_sorted[n_trim_D0+1:end])
    else
        E_Y_D0_trim_bottom = mean(Y_D0_sorted)
    end

    lower_bound = E_Y_D1_trim_top - E_Y_D0_trim_bottom

    # Upper bound: trim bottom of D=1, trim top of D=0
    if n_trim_D1 > 0 && n_trim_D1 < length(Y_D1_sorted)
        E_Y_D1_trim_bottom = mean(Y_D1_sorted[n_trim_D1+1:end])
    else
        E_Y_D1_trim_bottom = mean(Y_D1_sorted)
    end

    if n_trim_D0 > 0 && n_trim_D0 < length(Y_D0_sorted)
        E_Y_D0_trim_top = mean(Y_D0_sorted[1:end-n_trim_D0])
    else
        E_Y_D0_trim_top = mean(Y_D0_sorted)
    end

    upper_bound = E_Y_D1_trim_bottom - E_Y_D0_trim_top

    return (lower_bound, upper_bound)
end


"""
    compute_bounds_selection_monotonicity(Y, D, S, props, Y_min, Y_max) -> (lower, upper)

Bounds under selection monotonicity S(1) ≥ S(0).
"""
function compute_bounds_selection_monotonicity(
    Y::Vector{Float64},
    D::Vector{Float64},
    S::Vector{Float64},
    p_S1_D1::Float64,
    p_S1_D0::Float64,
    E_Y_D1_S1::Float64,
    E_Y_D0_S1::Float64,
    Y_min::Float64,
    Y_max::Float64
)
    p_AS = p_S1_D0

    if p_AS <= 0
        return (Y_min - Y_max, Y_max - Y_min)
    end

    # E[Y(0) | AS] = E[Y | D=0, S=1] exactly
    E_Y0_AS = E_Y_D0_S1

    # E[Y(1) | AS] is bounded
    p_protected = max(0.0, p_S1_D1 - p_S1_D0)

    if p_protected > 0 && p_S1_D1 > 0
        E_Y1_AS_lower = (p_S1_D1 * E_Y_D1_S1 - p_protected * Y_max) / p_AS
        E_Y1_AS_upper = (p_S1_D1 * E_Y_D1_S1 - p_protected * Y_min) / p_AS

        E_Y1_AS_lower = max(Y_min, E_Y1_AS_lower)
        E_Y1_AS_upper = min(Y_max, E_Y1_AS_upper)
    else
        E_Y1_AS_lower = E_Y_D1_S1
        E_Y1_AS_upper = E_Y_D1_S1
    end

    return (E_Y1_AS_lower - E_Y0_AS, E_Y1_AS_upper - E_Y0_AS)
end


# =============================================================================
# Main Functions
# =============================================================================

"""
    sace_bounds(Y, D, S; instrument=nothing, monotonicity="none", outcome_support=nothing) -> SACEResult

Compute bounds on the Survivor Average Causal Effect (SACE).

SACE = E[Y(1) - Y(0) | S(0)=1, S(1)=1]

# Arguments
- `Y::AbstractVector`: Outcome variable (NaN for non-survivors)
- `D::AbstractVector`: Treatment indicator (binary)
- `S::AbstractVector`: Survival indicator (binary)
- `instrument::Union{Nothing, AbstractVector}=nothing`: Instrument Z (binary)
- `monotonicity::String="none"`: Monotonicity assumption
  - "none": No assumptions (widest bounds)
  - "selection": S(1) ≥ S(0) (treatment never harms survival)
  - "treatment": D(1) ≥ D(0) (standard IV monotonicity)
  - "both": Both assumptions
- `outcome_support::Union{Nothing, Tuple{Float64,Float64}}=nothing`: Known (Y_min, Y_max)

# Returns
- `SACEResult`: Bounds on SACE

# Example
```julia
using Random
Random.seed!(42)
n = 500
D = rand(0:1, n)
S = [rand() < (0.8 + 0.1 * d) ? 1 : 0 for d in D]
Y_latent = 1.0 .+ 2.0 .* D .+ randn(n)
Y = [s == 1 ? y : NaN for (s, y) in zip(S, Y_latent)]
result = sace_bounds(Y, D, S, monotonicity="selection")
println("SACE bounds: [\$(result.lower_bound), \$(result.upper_bound)]")
```
"""
function sace_bounds(
    Y::AbstractVector,
    D::AbstractVector,
    S::AbstractVector;
    instrument::Union{Nothing, AbstractVector} = nothing,
    monotonicity::String = "none",
    outcome_support::Union{Nothing, Tuple{Float64,Float64}} = nothing
)
    Y, D, S, Z = validate_sace_inputs(Y, D, S, instrument)

    n = length(Y)

    # Compute survival proportions
    surv_props = compute_survival_proportions(D, S, Z)
    p_S1_D1 = surv_props[:p_S1_D1]
    p_S1_D0 = surv_props[:p_S1_D0]

    # Outcome support
    survivors = S .== 1
    Y_survivors = Y[survivors]
    Y_survivors = Y_survivors[.!isnan.(Y_survivors)]

    if length(Y_survivors) == 0
        throw(ArgumentError("No survivors in the data (all S=0)."))
    end

    if outcome_support !== nothing
        Y_min, Y_max = outcome_support
    else
        Y_min = minimum(Y_survivors)
        Y_max = maximum(Y_survivors)
    end

    # Conditional means among survivors
    D1_S1 = (D .== 1) .& (S .== 1)
    D0_S1 = (D .== 0) .& (S .== 1)

    E_Y_D1_S1 = sum(D1_S1) > 0 ? mean(Y[D1_S1][.!isnan.(Y[D1_S1])]) : NaN
    E_Y_D0_S1 = sum(D0_S1) > 0 ? mean(Y[D0_S1][.!isnan.(Y[D0_S1])]) : NaN

    # Compute bounds based on monotonicity
    if monotonicity == "both" || monotonicity == "selection"
        assumptions = monotonicity == "both" ?
            ["treatment_monotonicity", "selection_monotonicity"] :
            ["selection_monotonicity"]

        lower_bound, upper_bound = compute_bounds_selection_monotonicity(
            Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )
    elseif monotonicity == "treatment"
        assumptions = ["treatment_monotonicity"]
        # Treatment monotonicity alone doesn't help much for SACE without instrument
        lower_bound, upper_bound = compute_bounds_no_assumption(
            Y, D, S, p_S1_D1, p_S1_D0, Y_min, Y_max
        )
    else  # none
        assumptions = String[]
        lower_bound, upper_bound = compute_bounds_no_assumption(
            Y, D, S, p_S1_D1, p_S1_D0, Y_min, Y_max
        )
    end

    # Point estimate and SE
    sace_estimate = (lower_bound + upper_bound) / 2
    se_estimate = (upper_bound - lower_bound) / (2 * 1.96)

    method_str = length(assumptions) > 0 ?
        "sace_bounds_" * join(assumptions, "_") :
        "sace_bounds_no_assumption"

    return SACEResult(
        sace_estimate,
        se_estimate,
        lower_bound,
        upper_bound,
        p_S1_D1,
        p_S1_D0,
        n,
        method_str
    )
end


"""
    sace_sensitivity(Y, D, S; instrument=nothing, alpha_range=(0.0, 1.0), n_points=50) -> NamedTuple

Sensitivity analysis for SACE varying selection assumptions.

# Arguments
- `Y::AbstractVector`: Outcome variable
- `D::AbstractVector`: Treatment indicator (binary)
- `S::AbstractVector`: Survival indicator (binary)
- `instrument::Union{Nothing, AbstractVector}=nothing`: Instrument Z
- `alpha_range::Tuple{Float64,Float64}=(0.0, 1.0)`: Range of sensitivity parameter
- `n_points::Int=50`: Number of grid points

# Returns
- `NamedTuple` with fields: `alpha`, `lower_bound`, `upper_bound`, `sace`

# Example
```julia
using Random
Random.seed!(42)
n = 500
D = rand(0:1, n)
S = [rand() < (0.7 + 0.2 * d) ? 1 : 0 for d in D]
Y_latent = 1.0 .+ 1.5 .* D .+ randn(n)
Y = [s == 1 ? y : NaN for (s, y) in zip(S, Y_latent)]
sens = sace_sensitivity(Y, D, S)
```
"""
function sace_sensitivity(
    Y::AbstractVector,
    D::AbstractVector,
    S::AbstractVector;
    instrument::Union{Nothing, AbstractVector} = nothing,
    alpha_range::Tuple{Float64,Float64} = (0.0, 1.0),
    n_points::Int = 50
)
    Y, D, S, Z = validate_sace_inputs(Y, D, S, instrument)

    alphas = range(alpha_range[1], alpha_range[2], length=n_points)
    lower_bounds = zeros(n_points)
    upper_bounds = zeros(n_points)

    # Get base components
    survivors = S .== 1
    Y_survivors = Y[survivors]
    Y_survivors = Y_survivors[.!isnan.(Y_survivors)]

    if length(Y_survivors) == 0
        throw(ArgumentError("No survivors in the data."))
    end

    Y_min = minimum(Y_survivors)
    Y_max = maximum(Y_survivors)

    surv_props = compute_survival_proportions(D, S, Z)
    p_S1_D1 = surv_props[:p_S1_D1]
    p_S1_D0 = surv_props[:p_S1_D0]

    D1_S1 = (D .== 1) .& (S .== 1)
    D0_S1 = (D .== 0) .& (S .== 1)

    E_Y_D1_S1 = sum(D1_S1) > 0 ? mean(Y[D1_S1][.!isnan.(Y[D1_S1])]) : NaN
    E_Y_D0_S1 = sum(D0_S1) > 0 ? mean(Y[D0_S1][.!isnan.(Y[D0_S1])]) : NaN

    # Get extreme and tight bounds
    lb_extreme, ub_extreme = compute_bounds_no_assumption(
        Y, D, S, p_S1_D1, p_S1_D0, Y_min, Y_max
    )

    lb_tight, ub_tight = compute_bounds_selection_monotonicity(
        Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
    )

    # Interpolate
    for (i, alpha) in enumerate(alphas)
        lower_bounds[i] = (1 - alpha) * lb_extreme + alpha * lb_tight
        upper_bounds[i] = (1 - alpha) * ub_extreme + alpha * ub_tight
    end

    sace_points = (lower_bounds .+ upper_bounds) ./ 2

    return (
        alpha = collect(alphas),
        lower_bound = lower_bounds,
        upper_bound = upper_bounds,
        sace = sace_points
    )
end
