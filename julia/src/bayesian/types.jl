#=
Type definitions for Bayesian causal inference.

Session 101: Initial types for Bayesian ATE with conjugate priors.
Session 104: Added HierarchicalATEResult for MCMC-based hierarchical models.
=#

"""
    BayesianATEResult

Result from Bayesian Average Treatment Effect estimation.

# Fields
- `posterior_mean::Float64`: Posterior mean of treatment effect, E[τ | data]
- `posterior_sd::Float64`: Posterior standard deviation, SD[τ | data]
- `ci_lower::Float64`: Lower bound of credible interval (α/2 quantile)
- `ci_upper::Float64`: Upper bound of credible interval (1-α/2 quantile)
- `credible_level::Float64`: Credible interval level (e.g., 0.95)
- `prior_mean::Float64`: Prior mean for treatment effect
- `prior_sd::Float64`: Prior standard deviation
- `posterior_samples::Vector{Float64}`: Samples from posterior distribution
- `n::Int`: Total sample size
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `prior_to_posterior_shrinkage::Float64`: Measure of prior influence (0=data, 1=prior)
- `effective_sample_size::Float64`: Effective sample size
- `ols_estimate::Float64`: OLS/MLE estimate for comparison
- `ols_se::Float64`: Standard error of OLS estimate
- `sigma2_mle::Float64`: MLE estimate of residual variance

# Notes
**Credible vs Confidence Intervals**:

The credible interval has a direct probability interpretation:
"There is a 95% probability that the true treatment effect
lies within this interval, given the data and prior."

**Prior Shrinkage**:

    shrinkage = prior_precision / (prior_precision + likelihood_precision)

- shrinkage near 0: Data dominates
- shrinkage near 1: Prior dominates
"""
struct BayesianATEResult
    posterior_mean::Float64
    posterior_sd::Float64
    ci_lower::Float64
    ci_upper::Float64
    credible_level::Float64
    prior_mean::Float64
    prior_sd::Float64
    posterior_samples::Vector{Float64}
    n::Int
    n_treated::Int
    n_control::Int
    prior_to_posterior_shrinkage::Float64
    effective_sample_size::Float64
    ols_estimate::Float64
    ols_se::Float64
    sigma2_mle::Float64
end


"""Pretty-print BayesianATEResult."""
function Base.show(io::IO, r::BayesianATEResult)
    println(io, "BayesianATEResult")
    println(io, "═════════════════════════════════════════")
    println(io, "Posterior Estimate")
    println(io, "  Mean:   $(round(r.posterior_mean, digits=4))")
    println(io, "  SD:     $(round(r.posterior_sd, digits=4))")
    println(io, "  $(round(r.credible_level * 100, digits=0))% CI: [$(round(r.ci_lower, digits=4)), $(round(r.ci_upper, digits=4))]")
    println(io, "─────────────────────────────────────────")
    println(io, "Prior Specification")
    println(io, "  Mean:   $(round(r.prior_mean, digits=4))")
    println(io, "  SD:     $(round(r.prior_sd, digits=4))")
    println(io, "─────────────────────────────────────────")
    println(io, "Diagnostics")
    println(io, "  Shrinkage:      $(round(r.prior_to_posterior_shrinkage, digits=4)) (0=data, 1=prior)")
    println(io, "  Effective n:    $(round(r.effective_sample_size, digits=1))")
    println(io, "─────────────────────────────────────────")
    println(io, "Sample Information")
    println(io, "  n:         $(r.n)")
    println(io, "  n_treated: $(r.n_treated)")
    println(io, "  n_control: $(r.n_control)")
    println(io, "─────────────────────────────────────────")
    println(io, "OLS Comparison")
    println(io, "  OLS est:   $(round(r.ols_estimate, digits=4))")
    println(io, "  OLS SE:    $(round(r.ols_se, digits=4))")
end


"""
    HierarchicalATEResult

Result from Hierarchical Bayesian ATE estimation with MCMC.

Provides partial pooling across groups/sites, shrinking group-specific
estimates toward the population mean. Uses MCMC for full posterior inference.

# Fields
- `population_ate::Float64`: Posterior mean of population-level ATE
- `population_ate_se::Float64`: Posterior SD of population-level ATE
- `population_ate_ci_lower::Float64`: Lower bound of population ATE credible interval
- `population_ate_ci_upper::Float64`: Upper bound of population ATE credible interval
- `group_ates::Vector{Float64}`: Posterior mean of group-specific ATEs
- `group_ate_ses::Vector{Float64}`: Posterior SD of group-specific ATEs
- `group_ids::Vector`: Unique group identifiers
- `tau::Float64`: Posterior mean of between-group SD (heterogeneity)
- `tau_ci_lower::Float64`: Lower bound of tau credible interval
- `tau_ci_upper::Float64`: Upper bound of tau credible interval
- `posterior_samples::Dict{Symbol, Array}`: Full posterior samples
- `n_groups::Int`: Number of groups
- `n_obs::Int`: Total number of observations
- `credible_level::Float64`: Credible interval level
- `rhat_max::Float64`: Worst R-hat across parameters
- `ess_min::Float64`: Minimum effective sample size
- `divergences::Int`: Number of divergent transitions

# Notes
**Partial Pooling**:
- Small groups shrink toward population mean
- Large groups retain more group-specific signal
- τ (heterogeneity) estimated from data

**MCMC Diagnostics**:
- R-hat < 1.05: Chain convergence
- ESS > 400: Adequate effective samples
- Divergences = 0: NUTS sampler healthy
"""
struct HierarchicalATEResult
    population_ate::Float64
    population_ate_se::Float64
    population_ate_ci_lower::Float64
    population_ate_ci_upper::Float64
    group_ates::Vector{Float64}
    group_ate_ses::Vector{Float64}
    group_ids::Vector
    tau::Float64
    tau_ci_lower::Float64
    tau_ci_upper::Float64
    posterior_samples::Dict{Symbol, Array}
    n_groups::Int
    n_obs::Int
    credible_level::Float64
    rhat_max::Float64
    ess_min::Float64
    divergences::Int
end


"""Pretty-print HierarchicalATEResult."""
function Base.show(io::IO, r::HierarchicalATEResult)
    println(io, "HierarchicalATEResult")
    println(io, "═════════════════════════════════════════")
    println(io, "Population Estimate")
    println(io, "  ATE:      $(round(r.population_ate, digits=4))")
    println(io, "  SE:       $(round(r.population_ate_se, digits=4))")
    level_pct = Int(r.credible_level * 100)
    println(io, "  $(level_pct)% CI:   [$(round(r.population_ate_ci_lower, digits=4)), $(round(r.population_ate_ci_upper, digits=4))]")
    println(io, "─────────────────────────────────────────")
    println(io, "Heterogeneity")
    println(io, "  τ:        $(round(r.tau, digits=4))")
    println(io, "  τ CI:     [$(round(r.tau_ci_lower, digits=4)), $(round(r.tau_ci_upper, digits=4))]")
    println(io, "─────────────────────────────────────────")
    println(io, "Sample Information")
    println(io, "  n_obs:    $(r.n_obs)")
    println(io, "  n_groups: $(r.n_groups)")
    println(io, "─────────────────────────────────────────")
    println(io, "MCMC Diagnostics")
    println(io, "  R-hat max:    $(round(r.rhat_max, digits=4))")
    println(io, "  ESS min:      $(round(r.ess_min, digits=1))")
    println(io, "  Divergences:  $(r.divergences)")
end
