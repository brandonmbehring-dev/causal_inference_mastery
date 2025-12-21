"""
Unconditional Quantile Treatment Effects estimation in Julia.

Mirrors the Python implementation for cross-language parity.
"""

using Statistics
using Random

# types.jl is included by CausalEstimators.jl


"""
    unconditional_qte(problem::QTEProblem; n_bootstrap=1000, alpha=0.05, rng=nothing)

Estimate unconditional QTE: Q_τ(Y|T=1) - Q_τ(Y|T=0).

# Arguments
- `problem::QTEProblem`: Problem specification with outcome, treatment, quantile
- `n_bootstrap::Int=1000`: Number of bootstrap replications
- `alpha::Float64=0.05`: Significance level for CI
- `rng::Union{Nothing, AbstractRNG}=nothing`: Random number generator

# Returns
- `QTESolution`: Solution with estimate, SE, CI, etc.

# Example
```julia
outcome = randn(200) .+ 2.0 .* treatment
treatment = rand([0.0, 1.0], 200)
problem = QTEProblem(outcome=outcome, treatment=treatment, quantile=0.5)
solution = unconditional_qte(problem)
```
"""
function unconditional_qte(
    problem::QTEProblem{T};
    n_bootstrap::Int = 1000,
    alpha::T = T(0.05),
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real

    if rng === nothing
        rng = Random.default_rng()
    end

    outcome = problem.outcome
    treatment = problem.treatment
    τ = problem.quantile

    # Split by treatment
    y_treated = outcome[treatment .== one(T)]
    y_control = outcome[treatment .== zero(T)]

    n_treated = length(y_treated)
    n_control = length(y_control)
    n_total = length(outcome)

    # Validate minimum sample sizes
    n_treated >= 2 || error("Insufficient treated observations: $n_treated")
    n_control >= 2 || error("Insufficient control observations: $n_control")

    # Point estimate
    q_treated = quantile(y_treated, τ)
    q_control = quantile(y_control, τ)
    tau_q = q_treated - q_control

    # Bootstrap inference
    bootstrap_estimates = bootstrap_qte(y_treated, y_control, τ, n_bootstrap, rng)

    se = std(bootstrap_estimates; corrected=true)
    ci_lower = quantile(bootstrap_estimates, alpha / 2)
    ci_upper = quantile(bootstrap_estimates, 1 - alpha / 2)

    return QTESolution(
        tau_q = tau_q,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        quantile = τ,
        method = :unconditional,
        n_treated = n_treated,
        n_control = n_control,
        n_total = n_total,
        outcome_support = (minimum(outcome), maximum(outcome)),
        inference = :bootstrap
    )
end


"""
    unconditional_qte(outcome, treatment; quantile=0.5, kwargs...)

Convenience method that creates QTEProblem and calls main function.
"""
function unconditional_qte(
    outcome::Vector{T},
    treatment::Vector{T};
    quantile::T = T(0.5),
    n_bootstrap::Int = 1000,
    alpha::T = T(0.05),
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real

    problem = QTEProblem(
        outcome = outcome,
        treatment = treatment,
        quantile = quantile
    )
    return unconditional_qte(problem; n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
end


"""
    unconditional_qte_band(problem::QTEBandProblem; n_bootstrap=1000, alpha=0.05, joint=false, rng=nothing)

Estimate QTE across multiple quantiles.
"""
function unconditional_qte_band(
    problem::QTEBandProblem{T};
    n_bootstrap::Int = 1000,
    alpha::T = T(0.05),
    joint::Bool = false,
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real

    if rng === nothing
        rng = Random.default_rng()
    end

    outcome = problem.outcome
    treatment = problem.treatment
    quantiles = problem.quantiles

    y_treated = outcome[treatment .== one(T)]
    y_control = outcome[treatment .== zero(T)]

    n_treated = length(y_treated)
    n_control = length(y_control)
    n_total = length(outcome)
    n_quantiles = length(quantiles)

    # Point estimates at each quantile
    qte_estimates = [quantile(y_treated, τ) - quantile(y_control, τ) for τ in quantiles]

    # Bootstrap matrix: (n_bootstrap, n_quantiles)
    bootstrap_matrix = zeros(T, n_bootstrap, n_quantiles)

    for b in 1:n_bootstrap
        idx_t = rand(rng, 1:n_treated, n_treated)
        idx_c = rand(rng, 1:n_control, n_control)

        y_t_boot = y_treated[idx_t]
        y_c_boot = y_control[idx_c]

        for (i, τ) in enumerate(quantiles)
            bootstrap_matrix[b, i] = quantile(y_t_boot, τ) - quantile(y_c_boot, τ)
        end
    end

    # Pointwise SE and CI
    se_estimates = [std(bootstrap_matrix[:, i]; corrected=true) for i in 1:n_quantiles]
    ci_lower = [quantile(bootstrap_matrix[:, i], alpha / 2) for i in 1:n_quantiles]
    ci_upper = [quantile(bootstrap_matrix[:, i], 1 - alpha / 2) for i in 1:n_quantiles]

    # Joint CI (optional)
    joint_ci_lower = nothing
    joint_ci_upper = nothing

    if joint
        # Compute supremum of t-statistics
        t_stats = abs.(bootstrap_matrix .- qte_estimates') ./ max.(se_estimates', T(1e-10))
        sup_t_stats = maximum(t_stats, dims=2)[:]

        critical_value = quantile(sup_t_stats, 1 - alpha)

        joint_ci_lower = qte_estimates .- critical_value .* se_estimates
        joint_ci_upper = qte_estimates .+ critical_value .* se_estimates
    end

    return QTEBandSolution(
        quantiles = quantiles,
        qte_estimates = qte_estimates,
        se_estimates = se_estimates,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        joint_ci_lower = joint_ci_lower,
        joint_ci_upper = joint_ci_upper,
        method = :unconditional,
        n_bootstrap = n_bootstrap,
        n_treated = n_treated,
        n_control = n_control,
        n_total = n_total,
        alpha = alpha
    )
end


"""
    bootstrap_qte(y_treated, y_control, τ, n_bootstrap, rng)

Compute bootstrap distribution of QTE estimate.
"""
function bootstrap_qte(
    y_treated::Vector{T},
    y_control::Vector{T},
    τ::T,
    n_bootstrap::Int,
    rng::AbstractRNG
) where T<:Real

    n_t = length(y_treated)
    n_c = length(y_control)

    bootstrap_estimates = zeros(T, n_bootstrap)

    for b in 1:n_bootstrap
        idx_t = rand(rng, 1:n_t, n_t)
        idx_c = rand(rng, 1:n_c, n_c)

        q_t = quantile(y_treated[idx_t], τ)
        q_c = quantile(y_control[idx_c], τ)

        bootstrap_estimates[b] = q_t - q_c
    end

    return bootstrap_estimates
end
