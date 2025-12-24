#=
Bayesian Doubly Robust (DR) Estimation for Causal Inference.

Combines Bayesian propensity score estimation with frequentist outcome
models, propagating propensity uncertainty through the AIPW formula.

Session 103: Initial implementation.

References:
- Robins, Rotnitzky & Zhao (1994). Estimation of regression coefficients.
- Bang & Robins (2005). Doubly robust estimation.
=#

using Statistics
using Distributions


"""
    BayesianDRResult

Result from Bayesian Doubly Robust ATE estimation.
"""
struct BayesianDRResult
    estimate::Float64
    se::Float64
    ci_lower::Float64
    ci_upper::Float64
    credible_level::Float64
    posterior_samples::Vector{Float64}
    n::Int
    n_treated::Int
    n_control::Int
    propensity_mean::Vector{Float64}
    propensity_mean_uncertainty::Float64
    outcome_r2::Float64
    frequentist_estimate::Float64
    frequentist_se::Float64
end


"""Pretty-print BayesianDRResult."""
function Base.show(io::IO, r::BayesianDRResult)
    println(io, "BayesianDRResult")
    println(io, "═════════════════════════════════════════")
    println(io, "Point Estimate")
    println(io, "  ATE:      $(round(r.estimate, digits=4))")
    println(io, "  SE:       $(round(r.se, digits=4))")
    println(io, "  $(Int(r.credible_level*100))% CI:   [$(round(r.ci_lower, digits=4)), $(round(r.ci_upper, digits=4))]")
    println(io, "─────────────────────────────────────────")
    println(io, "Sample Information")
    println(io, "  n:         $(r.n)")
    println(io, "  n_treated: $(r.n_treated)")
    println(io, "  n_control: $(r.n_control)")
    println(io, "─────────────────────────────────────────")
    println(io, "Model Diagnostics")
    println(io, "  Propensity uncertainty: $(round(r.propensity_mean_uncertainty, digits=4))")
    println(io, "  Outcome R²:            $(round(r.outcome_r2, digits=4))")
    println(io, "─────────────────────────────────────────")
    println(io, "Frequentist Comparison")
    println(io, "  Estimate:  $(round(r.frequentist_estimate, digits=4))")
    println(io, "  SE:        $(round(r.frequentist_se, digits=4))")
end


"""
    _bayesian_dr_core(outcomes, treatment, e_samples, mu0, mu1, trim_threshold)

Propagate propensity uncertainty through the DR formula.
"""
function _bayesian_dr_core(
    outcomes::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    e_samples::AbstractMatrix{<:Real},
    mu0::AbstractVector{<:Real},
    mu1::AbstractVector{<:Real},
    trim_threshold::Real,
)
    n_samples = size(e_samples, 1)
    ate_samples = zeros(n_samples)

    for k in 1:n_samples
        # Clip propensity sample
        e_k = clamp.(e_samples[k, :], trim_threshold, 1 - trim_threshold)

        # AIPW formula
        treated = treatment ./ e_k .* (outcomes .- mu1) .+ mu1
        control = (1 .- treatment) ./ (1 .- e_k) .* (outcomes .- mu0) .+ mu0

        ate_samples[k] = mean(treated .- control)
    end

    return ate_samples
end


"""
    bayesian_dr_ate(outcomes, treatment, covariates; kwargs...)

Bayesian Doubly Robust ATE estimation.

Combines Bayesian propensity score estimation with frequentist outcome
models, propagating propensity uncertainty through the AIPW formula.

# Arguments
- `outcomes`: Observed outcomes, shape (n,).
- `treatment`: Binary treatment indicator (0/1), shape (n,).
- `covariates`: Covariate matrix, shape (n, p).

# Keyword Arguments
- `propensity_method="auto"`: Method for Bayesian propensity.
- `propensity_prior_alpha=1.0`: Beta prior alpha (stratified method).
- `propensity_prior_beta=1.0`: Beta prior beta (stratified method).
- `propensity_prior_sd=10.0`: Prior SD for logistic coefficients.
- `n_posterior_samples=1000`: Number of posterior samples.
- `credible_level=0.95`: Credible interval level.
- `trim_threshold=0.01`: Propensity clipping threshold.

# Returns
- `BayesianDRResult`

# Example
```julia
using Random
Random.seed!(42)
n = 200
X = randn(n, 2)
logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
prob = 1 ./ (1 .+ exp.(-logit))
T = Float64.(rand(n) .< prob)
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

result = bayesian_dr_ate(Y, T, X)
println("ATE: \$(result.estimate) ± \$(result.se)")
```
"""
function bayesian_dr_ate(
    outcomes::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    propensity_method::String = "auto",
    propensity_prior_alpha::Real = 1.0,
    propensity_prior_beta::Real = 1.0,
    propensity_prior_sd::Real = 10.0,
    n_posterior_samples::Int = 1000,
    credible_level::Real = 0.95,
    trim_threshold::Real = 0.01,
)
    Y = Float64.(outcomes)
    T = Float64.(treatment)
    X = Float64.(covariates)
    n = length(Y)

    # =========================================================================
    # Input Validation
    # =========================================================================

    if size(X, 1) != n
        throw(ArgumentError(
            "Length mismatch: outcomes ($n) != covariates ($(size(X, 1)))"
        ))
    end

    if length(T) != n
        throw(ArgumentError(
            "Length mismatch: outcomes ($n) != treatment ($(length(T)))"
        ))
    end

    if !all(t -> t == 0 || t == 1, T)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if trim_threshold <= 0 || trim_threshold >= 0.5
        throw(ArgumentError("trim_threshold must be in (0, 0.5)"))
    end

    if credible_level <= 0 || credible_level >= 1
        throw(ArgumentError("credible_level must be in (0, 1)"))
    end

    # =========================================================================
    # Step 1: Bayesian Propensity Score Estimation
    # =========================================================================

    prop_result = bayesian_propensity(
        T, X;
        method=propensity_method,
        prior_alpha=propensity_prior_alpha,
        prior_beta=propensity_prior_beta,
        prior_sd=propensity_prior_sd,
        n_posterior_samples=n_posterior_samples,
    )

    e_samples = prop_result.posterior_samples  # (n_samples, n)
    propensity_mean = prop_result.posterior_mean  # (n,)
    propensity_mean_uncertainty = prop_result.mean_uncertainty

    # =========================================================================
    # Step 2: Fit Outcome Models (Frequentist)
    # =========================================================================

    # fit_outcome_models expects treatment as Bool
    T_bool = T .== 1.0
    outcome_result = fit_outcome_models(Y, T_bool, X)
    mu0 = outcome_result.mu0_predictions
    mu1 = outcome_result.mu1_predictions

    # Average R² across treated and control models
    outcome_r2 = (outcome_result.mu1_r2 + outcome_result.mu0_r2) / 2

    # =========================================================================
    # Step 3: Propagate Propensity Uncertainty Through DR Formula
    # =========================================================================

    ate_samples = _bayesian_dr_core(Y, T, e_samples, mu0, mu1, trim_threshold)

    # =========================================================================
    # Step 4: Add Outcome Model Uncertainty
    # =========================================================================

    # Compute influence function residuals
    propensity_mean_clipped = clamp.(propensity_mean, trim_threshold, 1 - trim_threshold)

    treated_contrib = T ./ propensity_mean_clipped .* (Y .- mu1) .+ mu1
    control_contrib = (1 .- T) ./ (1 .- propensity_mean_clipped) .* (Y .- mu0) .+ mu0

    dr_point_estimate = mean(treated_contrib .- control_contrib)
    influence_function = treated_contrib .- control_contrib .- dr_point_estimate

    # Outcome model variance contribution
    outcome_variance = mean(influence_function.^2) / n

    # Propensity uncertainty contribution
    propensity_variance = var(ate_samples)

    # Combine variances
    total_variance = max(outcome_variance, propensity_variance + outcome_variance * 0.5)

    # Inflate posterior samples
    propensity_sd = std(ate_samples)
    target_sd = sqrt(total_variance)
    if propensity_sd > 0
        inflation_factor = target_sd / propensity_sd
        ate_samples = mean(ate_samples) .+ (ate_samples .- mean(ate_samples)) .* inflation_factor
    end

    # =========================================================================
    # Step 5: Summarize Posterior Distribution
    # =========================================================================

    posterior_mean = mean(ate_samples)
    posterior_sd = std(ate_samples)

    alpha = 1 - credible_level
    ci_lower = quantile(ate_samples, alpha / 2)
    ci_upper = quantile(ate_samples, 1 - alpha / 2)

    # =========================================================================
    # Step 6: Frequentist DR for Comparison
    # =========================================================================

    # Compute frequentist DR estimate
    freq_estimate = dr_point_estimate
    freq_se = sqrt(outcome_variance)

    # Sample sizes
    n_treated = Int(sum(T .== 1))
    n_control = Int(sum(T .== 0))

    return BayesianDRResult(
        posterior_mean,
        posterior_sd,
        ci_lower,
        ci_upper,
        credible_level,
        ate_samples,
        n,
        n_treated,
        n_control,
        propensity_mean,
        propensity_mean_uncertainty,
        outcome_r2,
        freq_estimate,
        freq_se,
    )
end


# Handle vector covariates
function bayesian_dr_ate(
    outcomes::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    covariates::AbstractVector{<:Real};
    kwargs...
)
    return bayesian_dr_ate(outcomes, treatment, reshape(covariates, :, 1); kwargs...)
end
