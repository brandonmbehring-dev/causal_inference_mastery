"""
Lee (2009) bounds for treatment effects under sample selection.

Implements sharp bounds when outcomes are missing for some units due to
attrition, under a monotonicity assumption.

Key Assumption (Monotonicity):
- Positive: Treatment weakly increases probability of being observed
- Negative: Treatment weakly decreases probability of being observed

References:
- Lee, D. S. (2009). Training, Wages, and Sample Selection.
  Review of Economic Studies, 76(3), 1071-1102.
"""

using Statistics
using Random


"""
    lee_bounds(outcome, treatment, observed; monotonicity=:positive,
               n_bootstrap=1000, alpha=0.05, rng=nothing)

Compute Lee (2009) sharp bounds under sample selection.

# Arguments
- `outcome::Vector{T}`: Outcome variable (NaN for unobserved)
- `treatment::Vector`: Treatment indicator (0/1)
- `observed::Vector`: Observation indicator (0/1), 1 = outcome observed
- `monotonicity::Symbol`: :positive or :negative
- `n_bootstrap::Int`: Number of bootstrap replications for CI
- `alpha::Float64`: Significance level for CI
- `rng::Union{Nothing, AbstractRNG}`: Random number generator

# Returns
- `LeeBoundsResult`: Bounds with bootstrap CI and diagnostics
"""
function lee_bounds(
    outcome::Vector{T},
    treatment::Vector,
    observed::Vector;
    monotonicity::Symbol = :positive,
    n_bootstrap::Int = 1000,
    alpha::Float64 = 0.05,
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch: treatment")
    length(observed) == n || error("Length mismatch: observed")

    treatment = T.(treatment)
    observed_vec = T.(observed)

    all(t -> t == zero(T) || t == one(T), treatment) || error("Treatment must be binary")
    all(o -> o == zero(T) || o == one(T), observed_vec) || error("Observed must be binary")
    monotonicity in (:positive, :negative) || error("monotonicity must be :positive or :negative")

    if rng === nothing
        rng = Random.default_rng()
    end

    # Count observations
    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    n_treated > 0 || error("No treated observations")
    n_control > 0 || error("No control observations")

    n_treated_observed = sum((treatment .== one(T)) .& (observed_vec .== one(T)))
    n_control_observed = sum((treatment .== zero(T)) .& (observed_vec .== one(T)))

    n_treated_observed > 0 || error("No observed outcomes in treatment group")
    n_control_observed > 0 || error("No observed outcomes in control group")

    # Observation rates
    obs_rate_treated = n_treated_observed / n_treated
    obs_rate_control = n_control_observed / n_control

    attrition_treated = one(T) - obs_rate_treated
    attrition_control = one(T) - obs_rate_control

    # Check monotonicity
    if monotonicity == :positive && obs_rate_treated < obs_rate_control
        @warn "Monotonicity violation: positive assumed but treatment decreases observation rate"
    elseif monotonicity == :negative && obs_rate_treated > obs_rate_control
        @warn "Monotonicity violation: negative assumed but treatment increases observation rate"
    end

    # Compute bounds
    bounds_lower, bounds_upper, n_trimmed = _compute_lee_bounds(
        outcome, treatment, observed_vec, monotonicity
    )

    # Bootstrap CI
    bootstrap_lowers = T[]
    bootstrap_uppers = T[]

    for _ in 1:n_bootstrap
        indices = rand(rng, 1:n, n)
        boot_outcome = outcome[indices]
        boot_treatment = treatment[indices]
        boot_observed = observed_vec[indices]

        try
            boot_lower, boot_upper, _ = _compute_lee_bounds(
                boot_outcome, boot_treatment, boot_observed, monotonicity
            )
            push!(bootstrap_lowers, boot_lower)
            push!(bootstrap_uppers, boot_upper)
        catch
            continue
        end
    end

    # Confidence interval using percentile method
    if length(bootstrap_lowers) > 0
        ci_lower = T(quantile(bootstrap_lowers, alpha / 2))
        ci_upper = T(quantile(bootstrap_uppers, 1 - alpha / 2))
    else
        ci_lower = T(NaN)
        ci_upper = T(NaN)
    end

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < T(1e-10)

    # Determine trimmed group
    if obs_rate_treated > obs_rate_control
        trimmed_group = :treated
    elseif obs_rate_control > obs_rate_treated
        trimmed_group = :control
    else
        trimmed_group = :none
    end

    # Trimming proportion
    if obs_rate_treated != obs_rate_control
        p_trim = abs(obs_rate_treated - obs_rate_control) / max(obs_rate_treated, obs_rate_control)
    else
        p_trim = zero(T)
    end

    return LeeBoundsResult{T}(
        bounds_lower,
        bounds_upper,
        bounds_width,
        ci_lower,
        ci_upper,
        point_identified,
        p_trim,
        trimmed_group,
        attrition_treated,
        attrition_control,
        n_treated_observed,
        n_control_observed,
        n_trimmed,
        monotonicity
    )
end


"""
    _compute_lee_bounds(outcome, treatment, observed, monotonicity)

Compute Lee bounds for a single sample.

Returns (lower_bound, upper_bound, n_trimmed).
"""
function _compute_lee_bounds(
    outcome::Vector{T},
    treatment::Vector{T},
    observed::Vector{T},
    monotonicity::Symbol
) where T<:Real
    # Get observed outcomes by group
    treated_mask = (treatment .== one(T)) .& (observed .== one(T))
    control_mask = (treatment .== zero(T)) .& (observed .== one(T))

    y_treated = outcome[treated_mask]
    y_control = outcome[control_mask]

    n_treated_obs = length(y_treated)
    n_control_obs = length(y_control)

    n_treated_obs > 0 || error("No observed outcomes in treatment group")
    n_control_obs > 0 || error("No observed outcomes in control group")

    # Observation rates
    n_treated_total = sum(treatment .== one(T))
    n_control_total = sum(treatment .== zero(T))

    obs_rate_treated = n_treated_obs / n_treated_total
    obs_rate_control = n_control_obs / n_control_total

    # If rates are equal, no trimming needed
    if isapprox(obs_rate_treated, obs_rate_control, rtol=1e-10)
        ate = mean(y_treated) - mean(y_control)
        return (ate, ate, 0)
    end

    # Determine which group to trim
    if monotonicity == :positive
        if obs_rate_treated > obs_rate_control
            trim_treated = true
            p_trim = (obs_rate_treated - obs_rate_control) / obs_rate_treated
        else
            trim_treated = false
            p_trim = (obs_rate_control - obs_rate_treated) / obs_rate_control
        end
    else  # negative monotonicity
        if obs_rate_control > obs_rate_treated
            trim_treated = false
            p_trim = (obs_rate_control - obs_rate_treated) / obs_rate_control
        else
            trim_treated = true
            p_trim = (obs_rate_treated - obs_rate_control) / obs_rate_treated
        end
    end

    # Number to trim
    if trim_treated
        n_trim = floor(Int, p_trim * n_treated_obs)
        y_to_trim = y_treated
        y_other = y_control
    else
        n_trim = floor(Int, p_trim * n_control_obs)
        y_to_trim = y_control
        y_other = y_treated
    end

    if n_trim >= length(y_to_trim)
        n_trim = length(y_to_trim) - 1
    end

    if n_trim <= 0
        ate = mean(y_treated) - mean(y_control)
        return (ate, ate, 0)
    end

    # Sort for trimming
    y_sorted = sort(y_to_trim)

    # Upper bound: trim from bottom (keep high values)
    y_trimmed_upper = y_sorted[(n_trim + 1):end]
    mean_trimmed_upper = mean(y_trimmed_upper)

    # Lower bound: trim from top (keep low values)
    y_trimmed_lower = y_sorted[1:(end - n_trim)]
    mean_trimmed_lower = mean(y_trimmed_lower)

    mean_other = mean(y_other)

    if trim_treated
        upper_bound = mean_trimmed_upper - mean_other
        lower_bound = mean_trimmed_lower - mean_other
    else
        upper_bound = mean(y_treated) - mean_trimmed_lower
        lower_bound = mean(y_treated) - mean_trimmed_upper
    end

    return (lower_bound, upper_bound, n_trim)
end


"""
    check_monotonicity(treatment, observed; alpha=0.05)

Check whether monotonicity assumption is plausible.

# Returns
Dict with test results including suggested monotonicity direction.
"""
function check_monotonicity(
    treatment::Vector,
    observed::Vector;
    alpha::Float64 = 0.05
)
    treatment = Float64.(treatment)
    observed_vec = Float64.(observed)

    n_treated = sum(treatment .== 1.0)
    n_control = sum(treatment .== 0.0)

    obs_rate_treated = mean(observed_vec[treatment .== 1.0])
    obs_rate_control = mean(observed_vec[treatment .== 0.0])

    diff = obs_rate_treated - obs_rate_control

    # Standard error for difference in proportions
    se = sqrt(
        obs_rate_treated * (1 - obs_rate_treated) / n_treated +
        obs_rate_control * (1 - obs_rate_control) / n_control
    )

    z_stat = se > 0 ? diff / se : 0.0

    # Two-sided p-value
    p_value = 2 * (1 - cdf(Normal(), abs(z_stat)))

    if diff > 0
        suggested = :positive
        interpretation = "Treatment increases observation probability"
    elseif diff < 0
        suggested = :negative
        interpretation = "Treatment decreases observation probability"
    else
        suggested = :either
        interpretation = "No differential attrition"
    end

    return Dict(
        :obs_rate_treated => obs_rate_treated,
        :obs_rate_control => obs_rate_control,
        :difference => diff,
        :se => se,
        :z_statistic => z_stat,
        :p_value => p_value,
        :significant => p_value < alpha,
        :suggested_monotonicity => suggested,
        :interpretation => interpretation
    )
end
