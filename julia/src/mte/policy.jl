"""
Policy-relevant treatment effects from MTE.

Derives population parameters (ATE, ATT, ATU, PRTE) by integrating
the MTE curve with appropriate weights.

References:
- Heckman & Vytlacil (2005): Structural Equations, Treatment Effects
- Carneiro, Heckman & Vytlacil (2011): Estimating Marginal Returns to Education
"""

using Statistics
using Distributions

# Note: types.jl is included by CausalEstimators.jl before this file


"""
    ate_from_mte(mte_result::MTESolution; n_bootstrap=0)

Compute Average Treatment Effect from MTE.

ATE = integrate MTE(u) du over the support [p_min, p_max]

# Arguments
- `mte_result::MTESolution`: Result from local_iv() or polynomial_mte()
- `n_bootstrap::Int`: If > 0, compute bootstrap SE

# Returns
- `PolicyResult`: ATE estimate with SE and CI

# Notes
- ATE uses uniform weights
- Requires MTE to be identified over sufficient support
"""
function ate_from_mte(
    mte_result::MTESolution{T};
    n_bootstrap::Int = 0
) where T<:Real
    mte_grid = mte_result.mte_grid
    u_grid = mte_result.u_grid
    se_grid = mte_result.se_grid
    p_min, p_max = mte_result.propensity_support

    # Remove NaN values
    valid = .!isnan.(mte_grid)
    if sum(valid) < 3
        return empty_policy_result(:ate, mte_result.n_obs, T)
    end

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Integrate MTE over support using trapezoidal rule
    integral = trapezoid(mte_valid, u_valid)
    support_width = u_valid[end] - u_valid[1]

    if support_width < T(1e-10)
        return empty_policy_result(:ate, mte_result.n_obs, T)
    end

    ate = integral / support_width

    # Standard error
    se, ci_lower, ci_upper = if n_bootstrap > 0 && any(se_valid .> zero(T))
        bootstrap_integral(mte_valid, u_valid, se_valid, n_bootstrap, support_width)
    else
        se_approx = median(filter(!isnan, se_valid))
        z = quantile(Normal(), 0.975)
        (se_approx, ate - z * se_approx, ate + z * se_approx)
    end

    return PolicyResult(
        estimate = ate,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        parameter = :ate,
        weights_used = "uniform (integrate MTE(u) du)",
        n_obs = mte_result.n_obs
    )
end


"""
    att_from_mte(mte_result::MTESolution; propensity=nothing, treatment=nothing, n_bootstrap=0)

Compute Average Treatment Effect on the Treated from MTE.

ATT = integrate MTE(u) * w_ATT(u) du

# Arguments
- `mte_result::MTESolution`: Result from local_iv() or polynomial_mte()
- `propensity::Vector`: Optional propensity scores
- `treatment::Vector`: Optional treatment indicator
- `n_bootstrap::Int`: Bootstrap replications

# Returns
- `PolicyResult`: ATT estimate

# Notes
- ATT weights lower u values more heavily (treated have lower U)
"""
function att_from_mte(
    mte_result::MTESolution{T};
    propensity::Union{Nothing, Vector{T}} = nothing,
    treatment::Union{Nothing, Vector{T}} = nothing,
    n_bootstrap::Int = 0
) where T<:Real
    mte_grid = mte_result.mte_grid
    u_grid = mte_result.u_grid
    se_grid = mte_result.se_grid
    p_min, p_max = mte_result.propensity_support

    valid = .!isnan.(mte_grid)
    if sum(valid) < 3
        return empty_policy_result(:att, mte_result.n_obs, T)
    end

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Compute ATT weights
    weights = if propensity !== nothing && treatment !== nothing
        compute_att_weights_empirical(u_valid, propensity[treatment .== one(T)])
    else
        compute_att_weights_theoretical(u_valid, p_min, p_max)
    end

    # Normalize weights
    weights = weights ./ sum(weights)

    # Weighted sum
    att = sum(mte_valid .* weights)

    # Standard error
    se, ci_lower, ci_upper = if n_bootstrap > 0 && any(se_valid .> zero(T))
        bootstrap_weighted_integral(mte_valid, se_valid, weights, n_bootstrap)
    else
        se_approx = sqrt(sum((weights .* se_valid).^2))
        z = quantile(Normal(), 0.975)
        (se_approx, att - z * se_approx, att + z * se_approx)
    end

    return PolicyResult(
        estimate = att,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        parameter = :att,
        weights_used = "ATT weights (integrate MTE(u) * P(U<=u|D=1) du)",
        n_obs = mte_result.n_obs
    )
end


"""
    atu_from_mte(mte_result::MTESolution; propensity=nothing, treatment=nothing, n_bootstrap=0)

Compute Average Treatment Effect on the Untreated from MTE.

ATU = integrate MTE(u) * w_ATU(u) du

# Arguments
- `mte_result::MTESolution`: Result from local_iv() or polynomial_mte()
- `propensity::Vector`: Optional propensity scores
- `treatment::Vector`: Optional treatment indicator
- `n_bootstrap::Int`: Bootstrap replications

# Returns
- `PolicyResult`: ATU estimate

# Notes
- ATU weights higher u values more heavily (untreated have higher U)
"""
function atu_from_mte(
    mte_result::MTESolution{T};
    propensity::Union{Nothing, Vector{T}} = nothing,
    treatment::Union{Nothing, Vector{T}} = nothing,
    n_bootstrap::Int = 0
) where T<:Real
    mte_grid = mte_result.mte_grid
    u_grid = mte_result.u_grid
    se_grid = mte_result.se_grid
    p_min, p_max = mte_result.propensity_support

    valid = .!isnan.(mte_grid)
    if sum(valid) < 3
        return empty_policy_result(:atu, mte_result.n_obs, T)
    end

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Compute ATU weights
    weights = if propensity !== nothing && treatment !== nothing
        compute_atu_weights_empirical(u_valid, propensity[treatment .== zero(T)])
    else
        compute_atu_weights_theoretical(u_valid, p_min, p_max)
    end

    weights = weights ./ sum(weights)
    atu = sum(mte_valid .* weights)

    se, ci_lower, ci_upper = if n_bootstrap > 0 && any(se_valid .> zero(T))
        bootstrap_weighted_integral(mte_valid, se_valid, weights, n_bootstrap)
    else
        se_approx = sqrt(sum((weights .* se_valid).^2))
        z = quantile(Normal(), 0.975)
        (se_approx, atu - z * se_approx, atu + z * se_approx)
    end

    return PolicyResult(
        estimate = atu,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        parameter = :atu,
        weights_used = "ATU weights (integrate MTE(u) * P(U>u|D=0) du)",
        n_obs = mte_result.n_obs
    )
end


"""
    prte(mte_result::MTESolution, policy_weights; n_bootstrap=0)

Compute Policy-Relevant Treatment Effect.

PRTE = integrate MTE(u) * w_policy(u) du

# Arguments
- `mte_result::MTESolution`: Result from local_iv() or polynomial_mte()
- `policy_weights`: Either Vector or function u_grid -> weights
- `n_bootstrap::Int`: Bootstrap replications

# Returns
- `PolicyResult`: PRTE estimate
"""
function prte(
    mte_result::MTESolution{T},
    policy_weights::Union{Vector{T}, Function};
    n_bootstrap::Int = 0
) where T<:Real
    mte_grid = mte_result.mte_grid
    u_grid = mte_result.u_grid
    se_grid = mte_result.se_grid

    valid = .!isnan.(mte_grid)
    if sum(valid) < 3
        return empty_policy_result(:prte, mte_result.n_obs, T)
    end

    mte_valid = mte_grid[valid]
    u_valid = u_grid[valid]
    se_valid = se_grid[valid]

    # Get weights
    weights = if policy_weights isa Function
        policy_weights(u_valid)
    else
        if length(policy_weights) != length(u_grid)
            error("Policy weights length ($(length(policy_weights))) != " *
                  "grid length ($(length(u_grid)))")
        end
        policy_weights[valid]
    end

    # Normalize
    weight_sum = sum(weights)
    if abs(weight_sum) > T(1e-10)
        weights = weights ./ weight_sum
    end

    prte_estimate = sum(mte_valid .* weights)

    se, ci_lower, ci_upper = if n_bootstrap > 0 && any(se_valid .> zero(T))
        bootstrap_weighted_integral(mte_valid, se_valid, weights, n_bootstrap)
    else
        se_approx = sqrt(sum((weights .* se_valid).^2))
        z = quantile(Normal(), 0.975)
        (se_approx, prte_estimate - z * se_approx, prte_estimate + z * se_approx)
    end

    return PolicyResult(
        estimate = prte_estimate,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        parameter = :prte,
        weights_used = "custom policy weights",
        n_obs = mte_result.n_obs
    )
end


"""
    late_from_mte(mte_result::MTESolution, p_old, p_new; n_bootstrap=0)

Compute LATE for a specific instrument shift from MTE.

LATE(p_old -> p_new) = integrate_{p_old}^{p_new} MTE(u) du / (p_new - p_old)

# Arguments
- `mte_result::MTESolution`: Result from local_iv() or polynomial_mte()
- `p_old::T`: Old propensity score
- `p_new::T`: New propensity score
- `n_bootstrap::Int`: Bootstrap replications

# Returns
- `PolicyResult`: LATE estimate for compliers in [p_old, p_new]
"""
function late_from_mte(
    mte_result::MTESolution{T},
    p_old::T,
    p_new::T;
    n_bootstrap::Int = 0
) where T<:Real
    if p_new < p_old
        p_old, p_new = p_new, p_old
    end

    mte_grid = mte_result.mte_grid
    u_grid = mte_result.u_grid
    se_grid = mte_result.se_grid

    # Find grid points in [p_old, p_new]
    mask = (u_grid .>= p_old) .& (u_grid .<= p_new)
    if sum(mask) < 2
        return empty_policy_result(:late, mte_result.n_obs, T)
    end

    mte_range = mte_grid[mask]
    u_range = u_grid[mask]
    se_range = se_grid[mask]

    valid = .!isnan.(mte_range)
    if sum(valid) < 2
        return empty_policy_result(:late, mte_result.n_obs, T)
    end

    # Integrate MTE over complier range
    integral = trapezoid(mte_range[valid], u_range[valid])
    range_width = u_range[valid][end] - u_range[valid][1]

    if range_width < T(1e-10)
        return empty_policy_result(:late, mte_result.n_obs, T)
    end

    late_estimate = integral / range_width

    se, ci_lower, ci_upper = if n_bootstrap > 0 && any(se_range[valid] .> zero(T))
        bootstrap_integral(mte_range[valid], u_range[valid], se_range[valid],
                          n_bootstrap, range_width)
    else
        se_approx = median(filter(!isnan, se_range[valid]))
        z = quantile(Normal(), 0.975)
        (se_approx, late_estimate - z * se_approx, late_estimate + z * se_approx)
    end

    return PolicyResult(
        estimate = late_estimate,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        parameter = :late,
        weights_used = "LATE weights over [$(round(p_old, digits=2)), $(round(p_new, digits=2))]",
        n_obs = mte_result.n_obs
    )
end


# ============================================================================
# Helper Functions
# ============================================================================

"""
    trapezoid(y, x)

Trapezoidal integration.
"""
function trapezoid(y::Vector{T}, x::Vector{T}) where T<:Real
    n = length(y)
    if n < 2
        return zero(T)
    end

    integral = zero(T)
    for i in 2:n
        integral += (y[i] + y[i-1]) / T(2) * (x[i] - x[i-1])
    end
    return integral
end


"""
    compute_att_weights_theoretical(u_grid, p_min, p_max)

Compute theoretical ATT weights (linearly decreasing).
"""
function compute_att_weights_theoretical(
    u_grid::Vector{T},
    p_min::T,
    p_max::T
) where T<:Real
    p_mean = (p_min + p_max) / T(2)

    # Linearly decreasing weights
    weights = max.(zero(T), p_mean .- u_grid .+ T(0.5) * (p_max - p_min))
    w_max = maximum(weights)
    return w_max > zero(T) ? weights ./ w_max : ones(T, length(u_grid))
end


"""
    compute_att_weights_empirical(u_grid, propensity_treated)

Compute empirical ATT weights from treated propensity distribution.
"""
function compute_att_weights_empirical(
    u_grid::Vector{T},
    propensity_treated::Vector{T}
) where T<:Real
    weights = zeros(T, length(u_grid))

    for (i, u) in enumerate(u_grid)
        weights[i] = mean(propensity_treated .>= u)
    end

    w_sum = sum(weights)
    return w_sum > zero(T) ? weights ./ w_sum : ones(T, length(u_grid)) ./ length(u_grid)
end


"""
    compute_atu_weights_theoretical(u_grid, p_min, p_max)

Compute theoretical ATU weights (linearly increasing).
"""
function compute_atu_weights_theoretical(
    u_grid::Vector{T},
    p_min::T,
    p_max::T
) where T<:Real
    p_mean = (p_min + p_max) / T(2)

    # Linearly increasing weights
    weights = max.(zero(T), u_grid .- p_mean .+ T(0.5) * (p_max - p_min))
    w_max = maximum(weights)
    return w_max > zero(T) ? weights ./ w_max : ones(T, length(u_grid))
end


"""
    compute_atu_weights_empirical(u_grid, propensity_untreated)

Compute empirical ATU weights from untreated propensity distribution.
"""
function compute_atu_weights_empirical(
    u_grid::Vector{T},
    propensity_untreated::Vector{T}
) where T<:Real
    weights = zeros(T, length(u_grid))

    for (i, u) in enumerate(u_grid)
        weights[i] = mean(propensity_untreated .<= u)
    end

    w_sum = sum(weights)
    return w_sum > zero(T) ? weights ./ w_sum : ones(T, length(u_grid)) ./ length(u_grid)
end


"""
    bootstrap_integral(mte, u, se, n_bootstrap, normalize)

Bootstrap standard error for integral estimate.
"""
function bootstrap_integral(
    mte::Vector{T},
    u::Vector{T},
    se::Vector{T},
    n_bootstrap::Int,
    normalize::T
) where T<:Real
    rng = Random.default_rng()
    boot_estimates = zeros(T, n_bootstrap)

    for b in 1:n_bootstrap
        mte_boot = mte .+ randn(rng, length(mte)) .* se
        integral = trapezoid(mte_boot, u) / normalize
        boot_estimates[b] = integral
    end

    se_boot = std(boot_estimates; corrected=true)
    ci_lower = quantile(boot_estimates, 0.025)
    ci_upper = quantile(boot_estimates, 0.975)

    return (se_boot, ci_lower, ci_upper)
end


"""
    bootstrap_weighted_integral(mte, se, weights, n_bootstrap)

Bootstrap standard error for weighted integral.
"""
function bootstrap_weighted_integral(
    mte::Vector{T},
    se::Vector{T},
    weights::Vector{T},
    n_bootstrap::Int
) where T<:Real
    rng = Random.default_rng()
    boot_estimates = zeros(T, n_bootstrap)

    for b in 1:n_bootstrap
        mte_boot = mte .+ randn(rng, length(mte)) .* se
        boot_estimates[b] = sum(mte_boot .* weights)
    end

    se_boot = std(boot_estimates; corrected=true)
    ci_lower = quantile(boot_estimates, 0.025)
    ci_upper = quantile(boot_estimates, 0.975)

    return (se_boot, ci_lower, ci_upper)
end


"""
    empty_policy_result(parameter, n_obs, T)

Return empty result when computation fails.
"""
function empty_policy_result(parameter::Symbol, n_obs::Int, ::Type{T}) where T<:Real
    return PolicyResult(
        estimate = T(NaN),
        se = T(NaN),
        ci_lower = T(NaN),
        ci_upper = T(NaN),
        parameter = parameter,
        weights_used = "N/A (insufficient data)",
        n_obs = n_obs
    )
end
