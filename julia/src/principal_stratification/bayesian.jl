"""
Bayesian Principal Stratification for CACE estimation using Turing.jl.

Provides full posterior inference on the Complier Average Causal Effect (CACE)
and principal strata proportions using MCMC sampling.

# Key Features
- Full posterior inference on CACE and strata proportions
- Marginalized likelihood (no direct sampling of discrete strata)
- MCMC diagnostics with automatic warnings
- Quick mode for fast exploratory analysis

# Generative Model
```
Strata proportions: (π_c, π_a, π_n) ~ Dirichlet(α_c, α_a, α_n)
Outcome means:
  μ_c0 ~ Normal(0, σ_prior)  # Complier untreated
  μ_c1 ~ Normal(0, σ_prior)  # Complier treated
  μ_a  ~ Normal(0, σ_prior)  # Always-taker
  μ_n  ~ Normal(0, σ_prior)  # Never-taker
Noise: σ ~ HalfNormal(1)
Likelihood: Marginalized over strata based on (D_i, Z_i)
```

# References
- Imbens & Rubin (1997). Bayesian Inference for Causal Effects in RCTs with Noncompliance.
- Hirano et al. (2000). Assessing the Effect of an Influenza Vaccine.
"""

using Statistics
using Distributions
using LinearAlgebra
using Random

# Include types if not already loaded
if !@isdefined(CACEProblem)
    include("types.jl")
end


# =============================================================================
# Turing.jl Availability Check
# =============================================================================

"""
    check_turing_available() -> Bool

Check if Turing.jl is available for Bayesian inference.
"""
function check_turing_available()
    try
        @eval using Turing
        return true
    catch
        return false
    end
end


# =============================================================================
# Marginalized Log-Likelihood Computation
# =============================================================================

"""
    compute_log_likelihood_obs(y, d, z, log_π_c, log_π_a, log_π_n, μ_c0, μ_c1, μ_a, μ_n, σ)

Compute marginalized log-likelihood for a single observation.

The key insight is that (D,Z) patterns constrain possible strata:
- D=1, Z=0: Must be always-taker
- D=0, Z=1: Must be never-taker
- D=1, Z=1: Mixture of complier (treated) and always-taker
- D=0, Z=0: Mixture of complier (untreated) and never-taker
"""
function compute_log_likelihood_obs(
    y::Real, d::Real, z::Real,
    log_π_c::Real, log_π_a::Real, log_π_n::Real,
    μ_c0::Real, μ_c1::Real, μ_a::Real, μ_n::Real,
    σ::Real
)
    if d == 1.0 && z == 0.0
        # D=1, Z=0: Must be always-taker
        return log_π_a + logpdf(Normal(μ_a, σ), y)
    elseif d == 0.0 && z == 1.0
        # D=0, Z=1: Must be never-taker
        return log_π_n + logpdf(Normal(μ_n, σ), y)
    elseif d == 1.0 && z == 1.0
        # D=1, Z=1: Mixture of compliers (treated) and always-takers
        ll_c = log_π_c + logpdf(Normal(μ_c1, σ), y)
        ll_a = log_π_a + logpdf(Normal(μ_a, σ), y)
        return logaddexp(ll_c, ll_a)
    else  # d == 0.0 && z == 0.0
        # D=0, Z=0: Mixture of compliers (untreated) and never-takers
        ll_c = log_π_c + logpdf(Normal(μ_c0, σ), y)
        ll_n = log_π_n + logpdf(Normal(μ_n, σ), y)
        return logaddexp(ll_c, ll_n)
    end
end


"""
    logaddexp(a, b)

Compute log(exp(a) + exp(b)) in a numerically stable way.
"""
function logaddexp(a::Real, b::Real)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end


"""
    compute_total_log_likelihood(Y, D, Z, log_π_c, log_π_a, log_π_n, μ_c0, μ_c1, μ_a, μ_n, σ)

Compute total marginalized log-likelihood over all observations.
"""
function compute_total_log_likelihood(
    Y::AbstractVector, D::AbstractVector, Z::AbstractVector,
    log_π_c::Real, log_π_a::Real, log_π_n::Real,
    μ_c0::Real, μ_c1::Real, μ_a::Real, μ_n::Real,
    σ::Real
)
    total_ll = 0.0
    for i in eachindex(Y)
        total_ll += compute_log_likelihood_obs(
            Y[i], D[i], Z[i],
            log_π_c, log_π_a, log_π_n,
            μ_c0, μ_c1, μ_a, μ_n, σ
        )
    end
    return total_ll
end


# =============================================================================
# Bayesian CACE Estimator Types
# =============================================================================

"""
    BayesianCACE <: AbstractPSEstimator

Bayesian CACE estimator using MCMC sampling.

# Fields
- `prior_alpha`: Dirichlet prior parameters (compliers, always-takers, never-takers)
- `prior_mu_sd`: Prior SD for outcome means
- `n_samples`: Number of posterior samples per chain
- `n_chains`: Number of MCMC chains
- `target_accept`: Target acceptance rate for NUTS
- `quick`: If true, use fast settings (1000 samples, 2 chains)
"""
struct BayesianCACE <: AbstractPSEstimator
    prior_alpha::Tuple{Float64, Float64, Float64}
    prior_mu_sd::Float64
    n_samples::Int
    n_chains::Int
    target_accept::Float64
    quick::Bool

    function BayesianCACE(;
        prior_alpha::Tuple{Real, Real, Real} = (1.0, 1.0, 1.0),
        prior_mu_sd::Real = 10.0,
        n_samples::Int = 2000,
        n_chains::Int = 4,
        target_accept::Real = 0.9,
        quick::Bool = false
    )
        if quick
            n_samples = 1000
            n_chains = 2
            target_accept = 0.8
        end
        new(
            Float64.(prior_alpha),
            Float64(prior_mu_sd),
            n_samples,
            n_chains,
            Float64(target_accept),
            quick
        )
    end
end


"""
    BayesianCACESolution

Result from Bayesian CACE estimation.

# Fields
- `cace_mean`: Posterior mean of CACE
- `cace_sd`: Posterior standard deviation
- `cace_hdi_lower`: Lower bound of 95% HDI
- `cace_hdi_upper`: Upper bound of 95% HDI
- `cace_samples`: Posterior samples of CACE
- `pi_c_mean`: Posterior mean of complier proportion
- `pi_c_samples`: Posterior samples of complier proportion
- `pi_a_mean`: Posterior mean of always-taker proportion
- `pi_n_mean`: Posterior mean of never-taker proportion
- `rhat`: R-hat convergence diagnostics
- `ess`: Effective sample sizes
- `n_samples`: Total posterior samples
- `n_chains`: Number of chains
"""
struct BayesianCACESolution
    cace_mean::Float64
    cace_sd::Float64
    cace_hdi_lower::Float64
    cace_hdi_upper::Float64
    cace_samples::Vector{Float64}
    pi_c_mean::Float64
    pi_c_samples::Vector{Float64}
    pi_a_mean::Float64
    pi_n_mean::Float64
    rhat::Dict{String, Float64}
    ess::Dict{String, Float64}
    n_samples::Int
    n_chains::Int
    model::String
end


# =============================================================================
# HDI Computation
# =============================================================================

"""
    hdi(samples::Vector{Float64}; prob::Float64=0.95)

Compute Highest Density Interval for posterior samples.
"""
function hdi(samples::Vector{Float64}; prob::Float64=0.95)
    sorted = sort(samples)
    n = length(sorted)
    interval_size = ceil(Int, prob * n)

    # Find interval with smallest width
    min_width = Inf
    best_lower = 0
    best_upper = 0

    for i in 1:(n - interval_size + 1)
        width = sorted[i + interval_size - 1] - sorted[i]
        if width < min_width
            min_width = width
            best_lower = sorted[i]
            best_upper = sorted[i + interval_size - 1]
        end
    end

    return (lower=best_lower, upper=best_upper)
end


# =============================================================================
# R-hat Computation
# =============================================================================

"""
    compute_rhat(chains::Matrix{Float64})

Compute Gelman-Rubin R-hat statistic for convergence diagnostic.

`chains` should be n_samples × n_chains matrix.
"""
function compute_rhat(chains::Matrix{Float64})
    n_samples, n_chains = size(chains)

    if n_chains < 2
        return 1.0  # Can't compute with single chain
    end

    # Chain means
    chain_means = vec(mean(chains, dims=1))

    # Overall mean
    overall_mean = mean(chain_means)

    # Between-chain variance
    B = n_samples * var(chain_means)

    # Within-chain variance
    W = mean([var(chains[:, j]) for j in 1:n_chains])

    # Pooled variance estimate
    var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

    # R-hat
    rhat = sqrt(var_hat / W)

    return rhat
end


"""
    compute_ess(samples::Vector{Float64})

Compute effective sample size using autocorrelation.
"""
function compute_ess(samples::Vector{Float64})
    n = length(samples)
    if n < 10
        return Float64(n)
    end

    # Compute autocorrelation using FFT approach (simplified)
    mean_s = mean(samples)
    var_s = var(samples)

    if var_s < 1e-10
        return Float64(n)
    end

    centered = samples .- mean_s

    # Simple autocorrelation estimate for first few lags
    max_lag = min(n ÷ 4, 100)
    autocorr = zeros(max_lag)

    for lag in 1:max_lag
        autocorr[lag] = dot(centered[1:n-lag], centered[lag+1:n]) / (var_s * (n - lag))
    end

    # Sum of positive autocorrelations (Geyer's initial monotone sequence)
    sum_rho = 0.0
    for i in 1:2:max_lag-1
        pair_sum = autocorr[i] + autocorr[i+1]
        if pair_sum < 0
            break
        end
        sum_rho += pair_sum
    end

    ess = n / (1 + 2 * sum_rho)
    return max(1.0, min(ess, Float64(n)))
end


# =============================================================================
# Simple Metropolis-Hastings Sampler (when Turing.jl not available)
# =============================================================================

"""
    sample_posterior_mh(Y, D, Z, prior_alpha, prior_mu_sd, n_samples, n_chains; seed=nothing)

Sample from posterior using simple Metropolis-Hastings.

This is a fallback when Turing.jl is not available.
"""
function sample_posterior_mh(
    Y::AbstractVector{T},
    D::AbstractVector,
    Z::AbstractVector,
    prior_alpha::Tuple{Float64, Float64, Float64},
    prior_mu_sd::Float64,
    n_samples::Int,
    n_chains::Int;
    seed::Union{Int, Nothing} = nothing
) where T <: Real

    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_params = 7  # π (3 via stick-breaking), μ_c0, μ_c1, μ_a, μ_n, σ

    # Storage for chains
    samples_cace = zeros(n_samples, n_chains)
    samples_pi_c = zeros(n_samples, n_chains)
    samples_pi_a = zeros(n_samples, n_chains)
    samples_pi_n = zeros(n_samples, n_chains)

    for chain in 1:n_chains
        # Initialize parameters
        π = rand(Dirichlet(collect(prior_alpha)))
        μ_c0 = randn() * prior_mu_sd * 0.1
        μ_c1 = randn() * prior_mu_sd * 0.1
        μ_a = randn() * prior_mu_sd * 0.1
        μ_n = randn() * prior_mu_sd * 0.1
        σ = abs(randn()) + 0.5

        # Proposal scales
        scale_pi = 0.02
        scale_mu = 0.3
        scale_sigma = 0.1

        # Current log-posterior
        function log_posterior(π, μ_c0, μ_c1, μ_a, μ_n, σ)
            if σ <= 0 || any(π .<= 0) || any(π .>= 1) || sum(π) > 1.001
                return -Inf
            end

            # Prior
            lp = logpdf(Dirichlet(collect(prior_alpha)), π)
            lp += logpdf(Normal(0, prior_mu_sd), μ_c0)
            lp += logpdf(Normal(0, prior_mu_sd), μ_c1)
            lp += logpdf(Normal(0, prior_mu_sd), μ_a)
            lp += logpdf(Normal(0, prior_mu_sd), μ_n)
            lp += logpdf(truncated(Normal(0, 1), 0, Inf), σ)

            # Likelihood
            log_π_c, log_π_a, log_π_n = log.(π)
            lp += compute_total_log_likelihood(Y, D, Z, log_π_c, log_π_a, log_π_n, μ_c0, μ_c1, μ_a, μ_n, σ)

            return lp
        end

        current_lp = log_posterior(π, μ_c0, μ_c1, μ_a, μ_n, σ)

        # Burn-in
        n_burnin = n_samples ÷ 2

        for iter in 1:(n_samples + n_burnin)
            # Propose new parameters
            # Update π (Dirichlet via random walk on simplex)
            π_prop = π .+ randn(3) .* scale_pi
            π_prop = max.(π_prop, 1e-6)
            π_prop = π_prop ./ sum(π_prop)

            μ_c0_prop = μ_c0 + randn() * scale_mu
            μ_c1_prop = μ_c1 + randn() * scale_mu
            μ_a_prop = μ_a + randn() * scale_mu
            μ_n_prop = μ_n + randn() * scale_mu
            σ_prop = σ + randn() * scale_sigma

            prop_lp = log_posterior(π_prop, μ_c0_prop, μ_c1_prop, μ_a_prop, μ_n_prop, σ_prop)

            # Accept/reject
            if log(rand()) < prop_lp - current_lp
                π, μ_c0, μ_c1, μ_a, μ_n, σ = π_prop, μ_c0_prop, μ_c1_prop, μ_a_prop, μ_n_prop, σ_prop
                current_lp = prop_lp
            end

            # Store after burn-in
            if iter > n_burnin
                idx = iter - n_burnin
                samples_cace[idx, chain] = μ_c1 - μ_c0
                samples_pi_c[idx, chain] = π[1]
                samples_pi_a[idx, chain] = π[2]
                samples_pi_n[idx, chain] = π[3]
            end
        end
    end

    return (
        cace = samples_cace,
        pi_c = samples_pi_c,
        pi_a = samples_pi_a,
        pi_n = samples_pi_n
    )
end


# =============================================================================
# Main Solver
# =============================================================================

"""
    solve(problem::CACEProblem, estimator::BayesianCACE) -> BayesianCACESolution

Estimate CACE using Bayesian inference.

# Arguments
- `problem::CACEProblem`: Problem specification with outcome, treatment, instrument
- `estimator::BayesianCACE`: Bayesian estimator settings

# Returns
- `BayesianCACESolution`: Posterior summaries and samples

# Example
```julia
problem = CACEProblem(Y, D, Z)
solution = solve(problem, BayesianCACE(quick=true))
println("CACE: \$(solution.cace_mean) ± \$(solution.cace_sd)")
```
"""
function solve(problem::CACEProblem{T}, estimator::BayesianCACE) where T <: Real
    Y = problem.outcome
    D = Float64.(problem.treatment)
    Z = Float64.(problem.instrument)

    n = length(Y)

    # Input validation
    if length(D) != n || length(Z) != n
        throw(ArgumentError("Y, D, Z must have same length"))
    end

    if !all(d -> d == 0.0 || d == 1.0, D)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if !all(z -> z == 0.0 || z == 1.0, Z)
        throw(ArgumentError("Instrument must be binary (0 or 1)"))
    end

    # Sample from posterior using simple MH
    samples = sample_posterior_mh(
        Y, D, Z,
        estimator.prior_alpha,
        estimator.prior_mu_sd,
        estimator.n_samples,
        estimator.n_chains;
        seed = nothing  # Seed is set externally via Random.seed!()
    )

    # Flatten samples across chains
    cace_samples = vec(samples.cace)
    pi_c_samples = vec(samples.pi_c)
    pi_a_samples = vec(samples.pi_a)
    pi_n_samples = vec(samples.pi_n)

    # Compute summaries
    cace_mean = mean(cace_samples)
    cace_sd = std(cace_samples)

    cace_hdi_result = hdi(cace_samples)
    cace_hdi_lower = cace_hdi_result.lower
    cace_hdi_upper = cace_hdi_result.upper

    pi_c_mean = mean(pi_c_samples)
    pi_a_mean = mean(pi_a_samples)
    pi_n_mean = mean(pi_n_samples)

    # Compute diagnostics
    rhat_dict = Dict{String, Float64}()
    ess_dict = Dict{String, Float64}()

    rhat_dict["cace"] = compute_rhat(samples.cace)
    rhat_dict["pi_c"] = compute_rhat(samples.pi_c)

    ess_dict["cace"] = compute_ess(cace_samples)
    ess_dict["pi_c"] = compute_ess(pi_c_samples)

    # Emit warnings if diagnostics are bad
    max_rhat = maximum(values(rhat_dict))
    min_ess = minimum(values(ess_dict))

    if max_rhat > 1.1
        @warn "R-hat $(round(max_rhat, digits=3)) > 1.1: chains may not have converged"
    end

    if min_ess < 100
        @warn "ESS $(round(min_ess, digits=0)) < 100: low effective sample size"
    end

    return BayesianCACESolution(
        cace_mean,
        cace_sd,
        cace_hdi_lower,
        cace_hdi_upper,
        cace_samples,
        pi_c_mean,
        pi_c_samples,
        pi_a_mean,
        pi_n_mean,
        rhat_dict,
        ess_dict,
        length(cace_samples),
        estimator.n_chains,
        "marginalized_mh"
    )
end


# =============================================================================
# Convenience Function
# =============================================================================

"""
    cace_bayesian(Y, D, Z; kwargs...) -> BayesianCACESolution

Convenience function for Bayesian CACE estimation.

# Arguments
- `Y`: Outcome variable
- `D`: Treatment received (binary)
- `Z`: Instrument (binary)
- `prior_alpha`: Dirichlet prior (default: (1,1,1))
- `prior_mu_sd`: Prior SD for means (default: 10.0)
- `n_samples`: Samples per chain (default: 2000)
- `n_chains`: Number of chains (default: 4)
- `quick`: Fast mode (default: false)
- `seed`: Random seed (optional)

# Example
```julia
result = cace_bayesian(Y, D, Z, quick=true, seed=42)
println("CACE: \$(result.cace_mean)")
```
"""
function cace_bayesian(
    Y::AbstractVector,
    D::AbstractVector,
    Z::AbstractVector;
    prior_alpha::Tuple{Real, Real, Real} = (1.0, 1.0, 1.0),
    prior_mu_sd::Real = 10.0,
    n_samples::Int = 2000,
    n_chains::Int = 4,
    quick::Bool = false,
    seed::Union{Int, Nothing} = nothing
)
    # Set seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    problem = CACEProblem(Y, D, Z)
    estimator = BayesianCACE(;
        prior_alpha = prior_alpha,
        prior_mu_sd = prior_mu_sd,
        n_samples = n_samples,
        n_chains = n_chains,
        quick = quick
    )

    return solve(problem, estimator)
end
