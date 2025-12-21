"""
Recentered Influence Function (RIF) regression for unconditional QTE.

Implements Firpo, Fortin, & Lemieux (2009) approach.
"""

using Statistics
using LinearAlgebra
using Random

# types.jl is included by CausalEstimators.jl


"""
    rif_qte(problem::QTEProblem; n_bootstrap=1000, alpha=0.05, rng=nothing)

Estimate unconditional QTE via RIF-OLS.

RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)

Regress RIF on treatment (and covariates) to get marginal effect
on unconditional quantile.

# Arguments
- `problem::QTEProblem`: Problem specification
- `n_bootstrap::Int=1000`: Bootstrap replications for inference
- `alpha::Float64=0.05`: Significance level
- `rng`: Random number generator

# Returns
- `QTESolution`: Solution with RIF-based QTE estimate
"""
function rif_qte(
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
    covariates = problem.covariates
    τ = problem.quantile

    n = length(outcome)
    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    # Compute RIF
    rif = compute_rif(outcome, τ)

    # Build design matrix
    if covariates !== nothing
        X = hcat(ones(T, n), treatment, covariates)
    else
        X = hcat(ones(T, n), treatment)
    end

    # Point estimate via OLS
    beta = X \ rif
    tau_q = beta[2]  # Treatment coefficient

    # Bootstrap inference
    bootstrap_estimates = zeros(T, n_bootstrap)

    for b in 1:n_bootstrap
        idx = rand(rng, 1:n, n)

        y_boot = outcome[idx]
        t_boot = treatment[idx]
        rif_boot = compute_rif(y_boot, τ)

        if covariates !== nothing
            X_boot = hcat(ones(T, n), t_boot, covariates[idx, :])
        else
            X_boot = hcat(ones(T, n), t_boot)
        end

        beta_boot = X_boot \ rif_boot
        bootstrap_estimates[b] = beta_boot[2]
    end

    se = std(bootstrap_estimates; corrected=true)
    ci_lower = quantile(bootstrap_estimates, alpha / 2)
    ci_upper = quantile(bootstrap_estimates, 1 - alpha / 2)

    return QTESolution(
        tau_q = tau_q,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        quantile = τ,
        method = :rif,
        n_treated = Int(n_treated),
        n_control = Int(n_control),
        n_total = n,
        outcome_support = (minimum(outcome), maximum(outcome)),
        inference = :bootstrap
    )
end


"""
    rif_qte(outcome, treatment; quantile=0.5, covariates=nothing, kwargs...)

Convenience method.
"""
function rif_qte(
    outcome::Vector{T},
    treatment::Vector{T};
    quantile::T = T(0.5),
    covariates::Union{Nothing, Matrix{T}} = nothing,
    n_bootstrap::Int = 1000,
    alpha::T = T(0.05),
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real

    problem = QTEProblem(
        outcome = outcome,
        treatment = treatment,
        covariates = covariates,
        quantile = quantile
    )
    return rif_qte(problem; n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
end


"""
    compute_rif(outcome, τ)

Compute Recentered Influence Function for quantile τ.

RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)
"""
function compute_rif(outcome::Vector{T}, τ::T) where T<:Real
    n = length(outcome)

    # Sample quantile
    q_tau = quantile(outcome, τ)

    # Kernel density estimate at quantile (Silverman's rule)
    h = T(1.06) * std(outcome) * n^(-T(0.2))
    h = max(h, T(1e-10))

    # Density: proportion within bandwidth / (2h)
    f_q = sum(abs.(outcome .- q_tau) .< h) / (2 * h * n)
    f_q = max(f_q, T(1e-10))

    # RIF = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)
    indicator = T.(outcome .<= q_tau)
    rif = q_tau .+ (τ .- indicator) ./ f_q

    return rif
end


"""
    rif_qte_band(problem::QTEBandProblem; n_bootstrap=1000, alpha=0.05, joint=false, rng=nothing)

Estimate RIF-based QTE across multiple quantiles.
"""
function rif_qte_band(
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
    covariates = problem.covariates
    quantiles = problem.quantiles

    n = length(outcome)
    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))
    n_quantiles = length(quantiles)

    # Point estimates
    qte_estimates = zeros(T, n_quantiles)
    for (i, τ) in enumerate(quantiles)
        rif = compute_rif(outcome, τ)
        if covariates !== nothing
            X = hcat(ones(T, n), treatment, covariates)
        else
            X = hcat(ones(T, n), treatment)
        end
        beta = X \ rif
        qte_estimates[i] = beta[2]
    end

    # Bootstrap
    bootstrap_matrix = zeros(T, n_bootstrap, n_quantiles)

    for b in 1:n_bootstrap
        idx = rand(rng, 1:n, n)
        y_boot = outcome[idx]
        t_boot = treatment[idx]

        for (i, τ) in enumerate(quantiles)
            rif_boot = compute_rif(y_boot, τ)
            if covariates !== nothing
                X_boot = hcat(ones(T, n), t_boot, covariates[idx, :])
            else
                X_boot = hcat(ones(T, n), t_boot)
            end
            beta_boot = X_boot \ rif_boot
            bootstrap_matrix[b, i] = beta_boot[2]
        end
    end

    se_estimates = [std(bootstrap_matrix[:, i]; corrected=true) for i in 1:n_quantiles]
    ci_lower = [quantile(bootstrap_matrix[:, i], alpha / 2) for i in 1:n_quantiles]
    ci_upper = [quantile(bootstrap_matrix[:, i], 1 - alpha / 2) for i in 1:n_quantiles]

    # Joint CI
    joint_ci_lower = nothing
    joint_ci_upper = nothing

    if joint
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
        method = :rif,
        n_bootstrap = n_bootstrap,
        n_treated = Int(n_treated),
        n_control = Int(n_control),
        n_total = n,
        alpha = alpha
    )
end
