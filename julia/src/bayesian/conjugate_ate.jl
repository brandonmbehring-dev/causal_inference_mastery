#=
Bayesian ATE Estimation with Conjugate Priors.

Implements closed-form posterior computation using the normal-normal
conjugate model. No MCMC required.

Session 101: Initial implementation.

Mathematical Foundation
-----------------------
The conjugate normal-normal model:

    Prior:      τ ~ N(μ₀, σ₀²)
    Likelihood: Y | T, X ~ N(α + τ*T + X*β, σ²)
    Posterior:  τ | Y, T, X ~ N(μₙ, σₙ²)  [closed form]

Where:
    μₙ = σₙ² * (μ₀/σ₀² + τ_ols/var_τ_ols)
    σₙ² = 1 / (1/σ₀² + 1/var_τ_ols)

The posterior is a precision-weighted average of prior and likelihood.
=#

# Uses: LinearAlgebra, Distributions, Statistics from main module


"""
    _compute_conjugate_posterior(Y, T, X, prior_mean, prior_var)

Compute posterior mean and variance using conjugate formulas.

# Returns
- `post_mean`: Posterior mean of treatment effect
- `post_var`: Posterior variance of treatment effect
- `tau_ols`: OLS estimate of treatment effect
- `var_tau_ols`: Variance of OLS estimate
- `sigma2_mle`: MLE estimate of residual variance
"""
function _compute_conjugate_posterior(
    Y::AbstractVector{<:Real},
    T::AbstractVector{<:Real},
    X::Union{Nothing, AbstractMatrix{<:Real}},
    prior_mean::Real,
    prior_var::Real
)
    n = length(Y)

    # Build design matrix: [intercept, treatment, covariates]
    if X !== nothing
        design = hcat(ones(n), T, X)
    else
        design = hcat(ones(n), T)
    end

    # OLS estimates (MLE)
    beta_ols = design \ Y

    # Residual variance (MLE with bias correction)
    fitted = design * beta_ols
    residuals_vec = Y .- fitted
    df = n - size(design, 2)
    sigma2_mle = sum(residuals_vec.^2) / df

    # Treatment coefficient is second element (after intercept)
    tau_ols = beta_ols[2]

    # Variance of OLS estimate for τ
    # Var(τ_ols) = σ² * (X'X)⁻¹[2,2]
    XtX = design' * design
    XtX_inv = inv(XtX)
    var_tau_ols = sigma2_mle * XtX_inv[2, 2]

    # Conjugate posterior update
    prior_prec = 1.0 / prior_var
    lik_prec = 1.0 / var_tau_ols
    post_prec = prior_prec + lik_prec
    post_var = 1.0 / post_prec

    # Posterior mean = weighted average of prior and likelihood
    post_mean = post_var * (prior_mean * prior_prec + tau_ols * lik_prec)

    return post_mean, post_var, tau_ols, var_tau_ols, sigma2_mle
end


"""
    bayesian_ate(outcomes, treatment; kwargs...)

Bayesian estimation of Average Treatment Effect using conjugate priors.

Uses normal-normal conjugate prior for closed-form posterior.
No MCMC required.

# Arguments
- `outcomes::AbstractVector`: Observed outcomes Y, shape (n,)
- `treatment::AbstractVector`: Binary treatment indicator (0/1), shape (n,)

# Keyword Arguments
- `covariates=nothing`: Covariate matrix X, shape (n, p). If nothing, simple difference.
- `prior_mean::Real=0.0`: Prior mean for treatment effect.
- `prior_sd::Real=10.0`: Prior standard deviation.
- `credible_level::Real=0.95`: Credible interval level.
- `n_posterior_samples::Int=5000`: Number of posterior samples to draw.

# Returns
- `BayesianATEResult`: Posterior mean, SD, credible interval, samples, and diagnostics.

# Examples
```julia
using Random
Random.seed!(42)
n = 200
treatment = rand(0:1, n)
outcomes = 2.0 .* treatment .+ randn(n)
result = bayesian_ate(outcomes, treatment)
println("Posterior mean: ", round(result.posterior_mean, digits=3))
```

# Notes
The credible interval has a direct probability interpretation:
P(τ ∈ [ci_lower, ci_upper] | data) = credible_level.
"""
function bayesian_ate(
    outcomes::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real};
    covariates::Union{Nothing, AbstractMatrix{<:Real}, AbstractVector{<:Real}} = nothing,
    prior_mean::Real = 0.0,
    prior_sd::Real = 10.0,
    credible_level::Real = 0.95,
    n_posterior_samples::Int = 5000
)
    # Input validation
    Y = Float64.(outcomes)
    T = Float64.(treatment)

    n = length(Y)

    if length(T) != n
        throw(ArgumentError(
            "Length mismatch: outcomes ($(n)) != treatment ($(length(T)))"
        ))
    end

    if !all(t -> t == 0 || t == 1, T)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if prior_sd <= 0
        throw(ArgumentError("prior_sd must be positive, got $prior_sd"))
    end

    if !(0 < credible_level < 1)
        throw(ArgumentError("credible_level must be in (0, 1), got $credible_level"))
    end

    if n_posterior_samples < 1
        throw(ArgumentError("n_posterior_samples must be >= 1, got $n_posterior_samples"))
    end

    # Handle covariates
    X = nothing
    if covariates !== nothing
        if isa(covariates, AbstractVector)
            X = reshape(Float64.(covariates), :, 1)
        else
            X = Float64.(covariates)
        end
        if size(X, 1) != n
            throw(ArgumentError(
                "Length mismatch: covariates ($(size(X, 1))) != outcomes ($n)"
            ))
        end
    end

    n_treated = Int(sum(T))
    n_control = n - n_treated

    if n_treated == 0 || n_control == 0
        throw(ArgumentError(
            "Both treatment groups must be non-empty. " *
            "Got n_treated=$n_treated, n_control=$n_control"
        ))
    end

    # Compute conjugate posterior
    prior_var = prior_sd^2
    post_mean, post_var, tau_ols, var_tau_ols, sigma2_mle = _compute_conjugate_posterior(
        Y, T, X, prior_mean, prior_var
    )
    post_sd = sqrt(post_var)

    # Credible interval
    alpha = 1 - credible_level
    z_alpha = quantile(Normal(), 1 - alpha / 2)
    ci_lower = post_mean - z_alpha * post_sd
    ci_upper = post_mean + z_alpha * post_sd

    # Draw posterior samples
    posterior_samples = rand(Normal(post_mean, post_sd), n_posterior_samples)

    # Compute diagnostics
    prior_prec = 1.0 / prior_var
    lik_prec = 1.0 / var_tau_ols
    total_prec = prior_prec + lik_prec
    prior_to_posterior_shrinkage = prior_prec / total_prec

    # Effective sample size
    effective_sample_size = n * (1 - prior_to_posterior_shrinkage)

    return BayesianATEResult(
        post_mean,
        post_sd,
        ci_lower,
        ci_upper,
        credible_level,
        prior_mean,
        prior_sd,
        posterior_samples,
        n,
        n_treated,
        n_control,
        prior_to_posterior_shrinkage,
        effective_sample_size,
        tau_ols,
        sqrt(var_tau_ols),
        sigma2_mle
    )
end
