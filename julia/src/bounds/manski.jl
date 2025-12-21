"""
Manski partial identification bounds for treatment effects.

Implements non-parametric bounds under various identifying assumptions:
1. Worst-case bounds (no assumptions)
2. Monotone Treatment Response (MTR)
3. Monotone Treatment Selection (MTS)
4. Combined MTR + MTS
5. Instrumental variable bounds

References:
- Manski (1990). Nonparametric Bounds on Treatment Effects
- Manski (2003). Partial Identification of Probability Distributions
- Manski & Pepper (2000). Monotone Instrumental Variables
"""

using Statistics


"""
    manski_worst_case(outcome, treatment; outcome_support=nothing)

Compute worst-case (no assumption) Manski bounds.

These are the widest possible bounds, assuming only that outcomes are bounded.

# Arguments
- `outcome::Vector{T}`: Observed outcomes
- `treatment::Vector`: Treatment indicator (0/1)
- `outcome_support::Union{Nothing, Tuple{T,T}}`: (Y_min, Y_max), defaults to observed range

# Returns
- `ManskiBoundsResult`: Bounds and diagnostic information
"""
function manski_worst_case(
    outcome::Vector{T},
    treatment::Vector;
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch: outcome ($n) != treatment ($(length(treatment)))")

    treatment = T.(treatment)
    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary (0 or 1)")

    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_treated > 0 || error("No treated observations")
    n_control > 0 || error("No control observations")

    # Outcome support
    if outcome_support !== nothing
        y_min, y_max = outcome_support
        y_min <= y_max || error("Invalid support: y_min ($y_min) > y_max ($y_max)")
    else
        y_min, y_max = minimum(outcome), maximum(outcome)
    end

    # Observed conditional means
    e_y1 = mean(outcome[treatment .== one(T)])  # E[Y|T=1]
    e_y0 = mean(outcome[treatment .== zero(T)])  # E[Y|T=0]

    naive_ate = e_y1 - e_y0

    # Treatment probabilities
    p_t1 = n_treated / n
    p_t0 = n_control / n

    # E[Y₁] bounds
    e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
    e_y1_upper = p_t1 * e_y1 + p_t0 * y_max

    # E[Y₀] bounds
    e_y0_lower = p_t1 * y_min + p_t0 * e_y0
    e_y0_upper = p_t1 * y_max + p_t0 * e_y0

    # ATE bounds
    bounds_lower = e_y1_lower - e_y0_upper
    bounds_upper = e_y1_upper - e_y0_lower

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    return ManskiBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        point_identified,
        :worst_case,
        nothing,
        naive_ate,
        ate_in_bounds,
        n_treated,
        n_control,
        (y_min, y_max)
    )
end


"""
    manski_mtr(outcome, treatment; direction=:positive, outcome_support=nothing)

Compute Manski bounds under Monotone Treatment Response (MTR).

MTR assumes treatment has monotone effect on outcomes:
- Positive MTR: Y₁ ≥ Y₀ for all units (treatment never hurts)
- Negative MTR: Y₁ ≤ Y₀ for all units (treatment never helps)

# Arguments
- `outcome::Vector{T}`: Observed outcomes
- `treatment::Vector`: Treatment indicator (0/1)
- `direction::Symbol`: :positive or :negative
- `outcome_support::Union{Nothing, Tuple{T,T}}`: Outcome bounds

# Returns
- `ManskiBoundsResult`: Bounds under MTR
"""
function manski_mtr(
    outcome::Vector{T},
    treatment::Vector;
    direction::Symbol = :positive,
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch")
    direction in (:positive, :negative) || error("direction must be :positive or :negative")

    treatment = T.(treatment)
    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary")

    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_treated > 0 || error("No treated observations")
    n_control > 0 || error("No control observations")

    # Outcome support
    if outcome_support !== nothing
        y_min, y_max = outcome_support
    else
        y_min, y_max = minimum(outcome), maximum(outcome)
    end

    # Observed conditional means
    e_y1 = mean(outcome[treatment .== one(T)])
    e_y0 = mean(outcome[treatment .== zero(T)])

    p_t1 = n_treated / n
    p_t0 = n_control / n

    naive_ate = e_y1 - e_y0

    if direction == :positive
        # Y₁ ≥ Y₀ for all units
        e_y1_lower = p_t1 * e_y1 + p_t0 * e_y0
        e_y1_upper = p_t1 * e_y1 + p_t0 * y_max

        e_y0_lower = p_t1 * y_min + p_t0 * e_y0
        e_y0_upper = p_t1 * e_y1 + p_t0 * e_y0

        bounds_lower = max(zero(T), e_y1_lower - e_y0_upper)
        bounds_upper = e_y1_upper - e_y0_lower
    else
        # Y₁ ≤ Y₀ for all units
        e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
        e_y1_upper = p_t1 * e_y1 + p_t0 * e_y0

        e_y0_lower = p_t1 * e_y1 + p_t0 * e_y0
        e_y0_upper = p_t1 * y_max + p_t0 * e_y0

        bounds_lower = e_y1_lower - e_y0_upper
        bounds_upper = min(zero(T), e_y1_upper - e_y0_lower)
    end

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    return ManskiBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        point_identified,
        :mtr,
        direction,
        naive_ate,
        ate_in_bounds,
        n_treated,
        n_control,
        (y_min, y_max)
    )
end


"""
    manski_mts(outcome, treatment; outcome_support=nothing)

Compute Manski bounds under Monotone Treatment Selection (MTS).

MTS assumes positive selection: units with higher potential outcomes
are more likely to select into treatment.

# Arguments
- `outcome::Vector{T}`: Observed outcomes
- `treatment::Vector`: Treatment indicator (0/1)
- `outcome_support::Union{Nothing, Tuple{T,T}}`: Outcome bounds

# Returns
- `ManskiBoundsResult`: Bounds under MTS
"""
function manski_mts(
    outcome::Vector{T},
    treatment::Vector;
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch")

    treatment = T.(treatment)
    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary")

    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_treated > 0 || error("No treated observations")
    n_control > 0 || error("No control observations")

    # Outcome support
    if outcome_support !== nothing
        y_min, y_max = outcome_support
    else
        y_min, y_max = minimum(outcome), maximum(outcome)
    end

    # Observed conditional means
    e_y1 = mean(outcome[treatment .== one(T)])
    e_y0 = mean(outcome[treatment .== zero(T)])

    p_t1 = n_treated / n
    p_t0 = n_control / n

    naive_ate = e_y1 - e_y0

    # Under MTS: E[Y₁|T=0] ≤ E[Y|T=1], E[Y₀|T=1] ≥ E[Y|T=0]
    e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
    e_y1_upper = p_t1 * e_y1 + p_t0 * e_y1  # = E[Y|T=1]

    e_y0_lower = p_t1 * e_y0 + p_t0 * e_y0  # = E[Y|T=0]
    e_y0_upper = p_t1 * y_max + p_t0 * e_y0

    bounds_lower = e_y1_lower - e_y0_upper
    bounds_upper = e_y1_upper - e_y0_lower  # = naive

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    return ManskiBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        point_identified,
        :mts,
        nothing,
        naive_ate,
        ate_in_bounds,
        n_treated,
        n_control,
        (y_min, y_max)
    )
end


"""
    manski_mtr_mts(outcome, treatment; mtr_direction=:positive, outcome_support=nothing)

Compute Manski bounds under combined MTR + MTS assumptions.

# Arguments
- `outcome::Vector{T}`: Observed outcomes
- `treatment::Vector`: Treatment indicator (0/1)
- `mtr_direction::Symbol`: :positive or :negative for MTR
- `outcome_support::Union{Nothing, Tuple{T,T}}`: Outcome bounds

# Returns
- `ManskiBoundsResult`: Bounds under combined assumptions
"""
function manski_mtr_mts(
    outcome::Vector{T},
    treatment::Vector;
    mtr_direction::Symbol = :positive,
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch")
    mtr_direction in (:positive, :negative) || error("mtr_direction must be :positive or :negative")

    treatment = T.(treatment)
    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary")

    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_treated > 0 || error("No treated observations")
    n_control > 0 || error("No control observations")

    # Outcome support
    if outcome_support !== nothing
        y_min, y_max = outcome_support
    else
        y_min, y_max = minimum(outcome), maximum(outcome)
    end

    # Observed conditional means
    e_y1 = mean(outcome[treatment .== one(T)])
    e_y0 = mean(outcome[treatment .== zero(T)])

    naive_ate = e_y1 - e_y0

    if mtr_direction == :positive
        # MTR (positive): ATE ≥ 0
        # MTS: ATE ≤ naive
        bounds_lower = zero(T)
        bounds_upper = naive_ate >= zero(T) ? naive_ate : zero(T)
    else
        # MTR (negative): ATE ≤ 0
        # MTS: ATE ≤ naive
        bounds_lower = min(naive_ate, zero(T))
        bounds_upper = zero(T)
    end

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    return ManskiBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        point_identified,
        :mtr_mts,
        mtr_direction,
        naive_ate,
        ate_in_bounds,
        n_treated,
        n_control,
        (y_min, y_max)
    )
end


"""
    manski_iv(outcome, treatment, instrument; outcome_support=nothing)

Compute Manski bounds with an instrumental variable.

# Arguments
- `outcome::Vector{T}`: Observed outcomes
- `treatment::Vector`: Treatment indicator (0/1)
- `instrument::Vector`: Instrumental variable (0/1)
- `outcome_support::Union{Nothing, Tuple{T,T}}`: Outcome bounds

# Returns
- `ManskiIVBoundsResult`: Bounds with IV diagnostics
"""
function manski_iv(
    outcome::Vector{T},
    treatment::Vector,
    instrument::Vector;
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch: treatment")
    length(instrument) == n || error("Length mismatch: instrument")

    treatment = T.(treatment)
    instrument = T.(instrument)

    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary")
    all(z -> z == zero(T) || z == one(T), instrument) || error("Instrument must be binary")

    n_iv_1 = sum(instrument .== one(T))
    n_iv_0 = sum(instrument .== zero(T))
    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_iv_1 > 0 || error("No observations with Z=1")
    n_iv_0 > 0 || error("No observations with Z=0")

    # Outcome support
    if outcome_support !== nothing
        y_min, y_max = outcome_support
    else
        y_min, y_max = minimum(outcome), maximum(outcome)
    end

    # Conditional means by instrument
    e_y_z1 = mean(outcome[instrument .== one(T)])
    e_y_z0 = mean(outcome[instrument .== zero(T)])

    # Treatment probabilities by instrument
    p_t1_z1 = mean(treatment[instrument .== one(T)])
    p_t1_z0 = mean(treatment[instrument .== zero(T)])

    # Complier share
    complier_share = abs(p_t1_z1 - p_t1_z0)
    iv_strength = complier_share

    # Reduced form effect
    rf_effect = e_y_z1 - e_y_z0

    if complier_share < T(1e-10)
        # No first stage variation
        return ManskiIVBoundsResult{T}(
            y_min - y_max,
            y_max - y_min,
            T(2) * (y_max - y_min),
            false,
            :iv,
            zero(T),
            zero(T),
            n_treated,
            n_control,
            n_iv_1,
            n_iv_0,
            (y_min, y_max)
        )
    end

    # IV bounds
    if p_t1_z1 > p_t1_z0
        bounds_lower = (rf_effect - (one(T) - complier_share) * (y_max - y_min)) / complier_share
        bounds_upper = (rf_effect + (one(T) - complier_share) * (y_max - y_min)) / complier_share
    else
        bounds_lower = -(rf_effect + (one(T) - complier_share) * (y_max - y_min)) / complier_share
        bounds_upper = -(rf_effect - (one(T) - complier_share) * (y_max - y_min)) / complier_share
    end

    # Tighten to reasonable range
    bounds_lower = max(bounds_lower, y_min - y_max)
    bounds_upper = min(bounds_upper, y_max - y_min)

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)

    return ManskiIVBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        point_identified,
        :iv,
        iv_strength,
        complier_share,
        n_treated,
        n_control,
        n_iv_1,
        n_iv_0,
        (y_min, y_max)
    )
end


"""
    compare_bounds(outcome, treatment; outcome_support=nothing, mtr_direction=:positive)

Compare bounds under different assumptions.

# Returns
Dict with results from each bounds method.
"""
function compare_bounds(
    outcome::Vector{T},
    treatment::Vector;
    outcome_support::Union{Nothing, Tuple{T,T}} = nothing,
    mtr_direction::Symbol = :positive
) where T<:Real
    return Dict(
        :worst_case => manski_worst_case(outcome, treatment; outcome_support=outcome_support),
        :mtr => manski_mtr(outcome, treatment; direction=mtr_direction, outcome_support=outcome_support),
        :mts => manski_mts(outcome, treatment; outcome_support=outcome_support),
        :mtr_mts => manski_mtr_mts(outcome, treatment; mtr_direction=mtr_direction, outcome_support=outcome_support)
    )
end
