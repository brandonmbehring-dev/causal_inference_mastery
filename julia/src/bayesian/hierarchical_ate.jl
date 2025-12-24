#=
Hierarchical Bayesian ATE Estimation with MCMC.

Implements partial pooling across groups/sites for multi-site studies.
Uses Turing.jl for MCMC sampling via the NUTS algorithm.

Session 104: Initial implementation.

References:
- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and
  Multilevel/Hierarchical Models. Cambridge University Press.
=#

using Statistics
using Distributions
using Random

# Import Turing at top level (for @model macro to work at parse time)
using Turing
using MCMCDiagnosticTools


"""
    hierarchical_bayesian_ate(outcomes, treatment, groups; kwargs...)

Hierarchical Bayesian ATE with partial pooling across groups.

Uses a hierarchical model with non-centered parameterization and
MCMC sampling via Turing.jl's NUTS algorithm.

# Arguments
- `outcomes::AbstractVector{<:Real}`: Observed outcomes, shape (n,)
- `treatment::AbstractVector{<:Real}`: Binary treatment indicator (0/1), shape (n,)
- `groups::AbstractVector`: Group identifiers for each observation, shape (n,)

# Keyword Arguments
- `mu_prior_sd::Real=10.0`: Prior SD for population-level ATE
- `tau_prior_sd::Real=5.0`: Prior SD for between-group heterogeneity
- `sigma_prior_sd::Real=10.0`: Prior SD for observation-level noise
- `n_samples::Int=2000`: Number of MCMC samples per chain (after warmup)
- `n_chains::Int=4`: Number of parallel MCMC chains
- `n_warmup::Int=1000`: Number of warmup samples per chain
- `credible_level::Real=0.95`: Credible interval level
- `seed::Int=nothing`: Random seed for reproducibility
- `progress::Bool=true`: Whether to show MCMC progress

# Returns
- `HierarchicalATEResult`: Contains population ATE, group-specific ATEs,
  heterogeneity (tau), full posterior samples, and MCMC diagnostics.

# Example
```julia
using Random
Random.seed!(42)
n = 300
groups = repeat(0:4, inner=60)  # 5 groups, 60 obs each
treatment = Float64.(rand(n) .< 0.5)
true_effects = [1.0, 1.5, 2.0, 2.5, 3.0]
outcomes = [true_effects[g+1] for g in groups] .* treatment .+ randn(n)

result = hierarchical_bayesian_ate(outcomes, treatment, groups; n_samples=500)
println("Population ATE: \$(result.population_ate)")
```
"""
function hierarchical_bayesian_ate(
    outcomes::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    groups::AbstractVector;
    mu_prior_sd::Real = 10.0,
    tau_prior_sd::Real = 5.0,
    sigma_prior_sd::Real = 10.0,
    n_samples::Int = 2000,
    n_chains::Int = 4,
    n_warmup::Int = 1000,
    credible_level::Real = 0.95,
    seed::Union{Int, Nothing} = nothing,
    progress::Bool = true,
)
    Y = Float64.(outcomes)
    T = Float64.(treatment)
    G = groups
    n = length(Y)

    # =========================================================================
    # Input Validation
    # =========================================================================

    if length(T) != n
        throw(ArgumentError(
            "Length mismatch: outcomes ($n) != treatment ($(length(T)))"
        ))
    end

    if length(G) != n
        throw(ArgumentError(
            "Length mismatch: outcomes ($n) != groups ($(length(G)))"
        ))
    end

    if !all(t -> t == 0 || t == 1, T)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if !(0 < credible_level < 1)
        throw(ArgumentError("credible_level must be in (0, 1)"))
    end

    if n_samples < 100
        throw(ArgumentError("n_samples must be >= 100"))
    end

    if n_chains < 1
        throw(ArgumentError("n_chains must be >= 1"))
    end

    # Encode groups as integers
    unique_groups = sort(unique(G))
    n_groups = length(unique_groups)
    group_to_idx = Dict(g => i for (i, g) in enumerate(unique_groups))
    group_idx = [group_to_idx[g] for g in G]

    if n_groups < 2
        throw(ArgumentError(
            "Need at least 2 groups for hierarchical model, got $n_groups. " *
            "For single-group analysis, use bayesian_ate() instead."
        ))
    end

    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # =========================================================================
    # Define Turing Model (Non-Centered Parameterization)
    # =========================================================================

    @model function _hierarchical_ate_model(Y, T, group_idx, n_groups,
        mu_prior_sd, tau_prior_sd, sigma_prior_sd)

        # Population-level priors
        μ ~ Normal(0, mu_prior_sd)
        τ ~ truncated(Normal(0, tau_prior_sd), 0, Inf)
        σ ~ truncated(Normal(0, sigma_prior_sd), 0, Inf)
        α ~ Normal(0, 10)

        # Group-level effects (non-centered parameterization)
        θ_raw ~ filldist(Normal(0, 1), n_groups)
        θ = μ .+ τ .* θ_raw

        # Likelihood
        for i in eachindex(Y)
            Y[i] ~ Normal(α + θ[group_idx[i]] * T[i], σ)
        end
    end

    # =========================================================================
    # Run MCMC Sampling
    # =========================================================================

    model = _hierarchical_ate_model(Y, T, group_idx, n_groups,
        mu_prior_sd, tau_prior_sd, sigma_prior_sd)

    # Sample with NUTS
    sampler = NUTS(n_warmup, 0.9)  # 0.9 = target_accept
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress=progress)

    # =========================================================================
    # Extract Results
    # =========================================================================

    # Get posterior samples
    μ_samples = vec(chain[:μ])
    τ_samples = vec(chain[:τ])

    # Get θ samples (group-specific effects)
    θ_samples = zeros(length(μ_samples), n_groups)
    for j in 1:n_groups
        θ_raw_j = vec(chain[Symbol("θ_raw[$j]")])
        θ_samples[:, j] = μ_samples .+ τ_samples .* θ_raw_j
    end

    # Compute credible intervals
    α_ci = (1 - credible_level) / 2

    # Population ATE
    population_ate = mean(μ_samples)
    population_ate_se = std(μ_samples)
    population_ate_ci_lower = quantile(μ_samples, α_ci)
    population_ate_ci_upper = quantile(μ_samples, 1 - α_ci)

    # Heterogeneity (tau)
    tau_mean = mean(τ_samples)
    tau_ci_lower = quantile(τ_samples, α_ci)
    tau_ci_upper = quantile(τ_samples, 1 - α_ci)

    # Group-specific ATEs
    group_ates = vec(mean(θ_samples, dims=1))
    group_ate_ses = vec(std(θ_samples, dims=1))

    # =========================================================================
    # MCMC Diagnostics
    # =========================================================================

    # R-hat (should be < 1.05)
    rhat_vals = Float64[]
    for param in [:μ, :τ, :σ, :α]
        r = MCMCDiagnosticTools.rhat(chain[param])
        push!(rhat_vals, r)
    end
    for j in 1:n_groups
        r = MCMCDiagnosticTools.rhat(chain[Symbol("θ_raw[$j]")])
        push!(rhat_vals, r)
    end
    rhat_max = maximum(rhat_vals)

    # Effective sample size (should be > 400)
    ess_vals = Float64[]
    for param in [:μ, :τ, :σ, :α]
        e = MCMCDiagnosticTools.ess(chain[param])
        push!(ess_vals, e)
    end
    for j in 1:n_groups
        e = MCMCDiagnosticTools.ess(chain[Symbol("θ_raw[$j]")])
        push!(ess_vals, e)
    end
    ess_min = minimum(ess_vals)

    # Divergences (count from chain info if available)
    divergences = 0
    try
        if hasproperty(chain.info, :numerical_error)
            divergences = sum(chain.info.numerical_error)
        end
    catch
        # If we can't access divergence info, leave as 0
    end

    # =========================================================================
    # Build Result
    # =========================================================================

    posterior_samples = Dict{Symbol, Array}(
        :μ => μ_samples,
        :τ => τ_samples,
        :θ => θ_samples,
    )

    return HierarchicalATEResult(
        population_ate,
        population_ate_se,
        population_ate_ci_lower,
        population_ate_ci_upper,
        group_ates,
        group_ate_ses,
        collect(unique_groups),
        tau_mean,
        tau_ci_lower,
        tau_ci_upper,
        posterior_samples,
        n_groups,
        n,
        credible_level,
        rhat_max,
        ess_min,
        divergences,
    )
end
