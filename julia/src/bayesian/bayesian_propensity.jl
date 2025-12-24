#=
Bayesian Propensity Score Estimation.

Implements two approaches:
1. Stratified Beta-Binomial (conjugate, for discrete covariates)
2. Bayesian Logistic Regression (Laplace approximation, for continuous covariates)

Session 102: Initial implementation.
=#

# Uses: Distributions, Statistics, LinearAlgebra from main module


"""
    StratumInfo

Information about a propensity stratum in Beta-Binomial estimation.
"""
struct StratumInfo
    stratum_id::Int
    n_obs::Int
    n_treated::Int
    n_control::Int
    posterior_alpha::Float64
    posterior_beta::Float64
    posterior_mean::Float64
    posterior_sd::Float64
end


"""
    BayesianPropensityResult

Result from Bayesian propensity score estimation.
"""
struct BayesianPropensityResult
    posterior_samples::Matrix{Float64}   # (n_samples, n)
    posterior_mean::Vector{Float64}      # (n,)
    posterior_sd::Vector{Float64}        # (n,)
    strata::Union{Vector{Int}, Nothing}
    n_strata::Int
    stratum_info::Union{Vector{StratumInfo}, Nothing}
    prior_alpha::Float64
    prior_beta::Float64
    method::String
    n::Int
    n_treated::Int
    n_control::Int
    mean_uncertainty::Float64
    propensity_range::Float64
    # Logistic method fields
    coefficient_mean::Union{Vector{Float64}, Nothing}
    coefficient_sd::Union{Vector{Float64}, Nothing}
    coefficient_samples::Union{Matrix{Float64}, Nothing}
    prior_sd::Float64
end


"""Pretty-print BayesianPropensityResult."""
function Base.show(io::IO, r::BayesianPropensityResult)
    println(io, "BayesianPropensityResult")
    println(io, "═════════════════════════════════════════")
    println(io, "Method: $(r.method)")
    println(io, "─────────────────────────────────────────")
    println(io, "Sample Information")
    println(io, "  n:         $(r.n)")
    println(io, "  n_treated: $(r.n_treated)")
    println(io, "  n_control: $(r.n_control)")
    if r.n_strata > 0
        println(io, "  n_strata:  $(r.n_strata)")
    end
    println(io, "─────────────────────────────────────────")
    println(io, "Propensity Summary")
    println(io, "  Mean propensity:  $(round(mean(r.posterior_mean), digits=4))")
    println(io, "  Mean uncertainty: $(round(r.mean_uncertainty, digits=4))")
    println(io, "  Range:            $(round(r.propensity_range, digits=4))")
    if r.coefficient_mean !== nothing
        println(io, "─────────────────────────────────────────")
        println(io, "Coefficients")
        for (i, (m, s)) in enumerate(zip(r.coefficient_mean, r.coefficient_sd))
            println(io, "  β[$i]: $(round(m, digits=4)) ± $(round(s, digits=4))")
        end
    end
end


"""
    _create_strata(covariates, n_bins)

Create strata from covariates by discretizing continuous variables.
"""
function _create_strata(
    covariates::AbstractMatrix{<:Real},
    n_bins::Int = 5
)
    n, p = size(covariates)

    # Discretize each covariate
    discretized = zeros(Int, n, p)
    for j in 1:p
        col = covariates[:, j]
        unique_vals = unique(col)

        if length(unique_vals) <= n_bins
            # Already discrete, use as-is
            val_to_idx = Dict(v => i for (i, v) in enumerate(unique_vals))
            discretized[:, j] = [val_to_idx[v] for v in col]
        else
            # Discretize using quantile bins
            percentiles = range(0, 100, length=n_bins+1)
            bins = quantile(col, percentiles ./ 100)
            bins[1] = -Inf
            bins[end] = Inf
            for i in 1:n
                for k in 1:(length(bins)-1)
                    if bins[k] <= col[i] < bins[k+1]
                        discretized[i, j] = k
                        break
                    end
                end
                if col[i] >= bins[end-1]
                    discretized[i, j] = n_bins
                end
            end
        end
    end

    # Create unique stratum for each combination
    strata = zeros(Int, n)
    stratum_map = Dict{Tuple, Int}()
    for i in 1:n
        key = Tuple(discretized[i, :])
        if !haskey(stratum_map, key)
            stratum_map[key] = length(stratum_map) + 1
        end
        strata[i] = stratum_map[key]
    end

    return strata, length(stratum_map)
end


"""
    bayesian_propensity_stratified(treatment, covariates; kwargs...)

Bayesian propensity score estimation using stratified Beta-Binomial.

# Arguments
- `treatment`: Binary treatment indicator (0/1)
- `covariates`: Covariate matrix

# Keyword Arguments
- `prior_alpha=1.0`: Beta prior alpha parameter
- `prior_beta=1.0`: Beta prior beta parameter
- `n_bins=5`: Number of bins for discretization
- `n_posterior_samples=1000`: Number of posterior samples

# Returns
- `BayesianPropensityResult`
"""
function bayesian_propensity_stratified(
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    prior_alpha::Real = 1.0,
    prior_beta::Real = 1.0,
    n_bins::Int = 5,
    n_posterior_samples::Int = 1000
)
    T = Float64.(treatment)
    X = Float64.(covariates)
    n = length(T)

    if size(X, 1) != n
        throw(ArgumentError(
            "Length mismatch: treatment ($n) != covariates ($(size(X, 1)))"
        ))
    end

    if !all(t -> t == 0 || t == 1, T)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if prior_alpha <= 0 || prior_beta <= 0
        throw(ArgumentError(
            "prior_alpha and prior_beta must be positive, " *
            "got alpha=$prior_alpha, beta=$prior_beta"
        ))
    end

    # Create strata
    strata, n_strata = _create_strata(X, n_bins)

    # Compute posterior for each stratum
    stratum_info = StratumInfo[]
    posterior_samples = zeros(n_posterior_samples, n)

    for s in 1:n_strata
        mask = strata .== s
        n_in_stratum = sum(mask)
        n_treated = Int(sum(T[mask]))
        n_control = n_in_stratum - n_treated

        # Posterior parameters
        post_alpha = prior_alpha + n_treated
        post_beta = prior_beta + n_control

        # Posterior mean
        post_mean = post_alpha / (post_alpha + post_beta)
        post_sd = sqrt(
            (post_alpha * post_beta) /
            ((post_alpha + post_beta)^2 * (post_alpha + post_beta + 1))
        )

        # Draw samples from posterior
        samples = rand(Beta(post_alpha, post_beta), n_posterior_samples)

        # Assign to all observations in this stratum
        for i in 1:n
            if mask[i]
                posterior_samples[:, i] = samples
            end
        end

        push!(stratum_info, StratumInfo(
            s, n_in_stratum, n_treated, n_control,
            post_alpha, post_beta, post_mean, post_sd
        ))
    end

    # Compute summary statistics
    posterior_mean = vec(mean(posterior_samples, dims=1))
    posterior_sd = vec(std(posterior_samples, dims=1))

    mean_uncertainty = mean(posterior_sd)
    propensity_range = maximum(posterior_mean) - minimum(posterior_mean)

    return BayesianPropensityResult(
        posterior_samples,
        posterior_mean,
        posterior_sd,
        strata,
        n_strata,
        stratum_info,
        prior_alpha,
        prior_beta,
        "stratified_beta_binomial",
        n,
        Int(sum(T)),
        Int(n - sum(T)),
        mean_uncertainty,
        propensity_range,
        nothing,
        nothing,
        nothing,
        0.0
    )
end


"""
    _sigmoid(x)

Sigmoid function with numerical stability.
"""
function _sigmoid(x::Real)
    if x >= 0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else
        z = exp(x)
        return z / (1.0 + z)
    end
end

_sigmoid(x::AbstractArray) = _sigmoid.(x)


"""
    _neg_log_posterior_logistic(beta, X, T, prior_var)

Negative log-posterior for logistic regression.
"""
function _neg_log_posterior_logistic(
    beta::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    T::AbstractVector{<:Real},
    prior_var::Real
)
    linear = X * beta
    linear = clamp.(linear, -500, 500)
    prob = _sigmoid(linear)

    # Log-likelihood
    ll = sum(T .* log.(prob .+ 1e-10) .+ (1 .- T) .* log.(1 .- prob .+ 1e-10))

    # Log-prior (normal)
    log_prior = -0.5 * sum(beta.^2) / prior_var

    return -(ll + log_prior)
end


"""
    _hessian_logistic(beta, X, prior_var)

Hessian of negative log-posterior for logistic regression.
"""
function _hessian_logistic(
    beta::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior_var::Real
)
    linear = X * beta
    linear = clamp.(linear, -500, 500)
    prob = _sigmoid(linear)

    # Weight matrix
    W = prob .* (1 .- prob)

    # Hessian = X' diag(W) X + I/prior_var
    n_coef = size(X, 2)
    hessian = X' * Diagonal(W) * X + I(n_coef) / prior_var

    return hessian
end


"""
    bayesian_propensity_logistic(treatment, covariates; kwargs...)

Bayesian propensity score estimation using logistic regression with Laplace approximation.

# Arguments
- `treatment`: Binary treatment indicator (0/1)
- `covariates`: Covariate matrix

# Keyword Arguments
- `prior_sd=10.0`: Prior SD for coefficients
- `n_posterior_samples=1000`: Number of posterior samples
- `include_intercept=true`: Whether to include intercept

# Returns
- `BayesianPropensityResult`
"""
function bayesian_propensity_logistic(
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    prior_sd::Real = 10.0,
    n_posterior_samples::Int = 1000,
    include_intercept::Bool = true
)
    T = Float64.(treatment)
    n = length(T)

    if size(covariates, 1) != n
        throw(ArgumentError(
            "Length mismatch: treatment ($n) != covariates ($(size(covariates, 1)))"
        ))
    end

    if !all(t -> t == 0 || t == 1, T)
        throw(ArgumentError("Treatment must be binary (0 or 1)"))
    end

    if prior_sd <= 0
        throw(ArgumentError("prior_sd must be positive, got $prior_sd"))
    end

    # Add intercept if requested
    X = if include_intercept
        hcat(ones(n), Float64.(covariates))
    else
        Float64.(covariates)
    end

    n_coef = size(X, 2)
    prior_var = prior_sd^2

    # Find MAP estimate using simple gradient descent
    beta = zeros(n_coef)
    lr = 0.01
    for _ in 1:1000
        linear = X * beta
        linear = clamp.(linear, -500, 500)
        prob = _sigmoid(linear)

        # Gradient
        grad_ll = X' * (T .- prob)
        grad_prior = -beta / prior_var
        grad = grad_ll + grad_prior

        beta += lr * grad

        # Check convergence
        if maximum(abs.(grad)) < 1e-6
            break
        end
    end
    beta_map = beta

    # Compute Hessian at MAP for Laplace approximation
    hessian = _hessian_logistic(beta_map, X, prior_var)
    cov_matrix = try
        inv(hessian)
    catch
        pinv(hessian)
    end

    # Make sure covariance is positive definite
    cov_matrix = (cov_matrix + cov_matrix') / 2
    eigvals_cov = eigvals(cov_matrix)
    if minimum(eigvals_cov) < 0
        cov_matrix = cov_matrix - minimum(eigvals_cov) * I + 1e-6 * I
    end

    # Draw coefficient samples from approximate posterior
    beta_samples = try
        rand(MvNormal(beta_map, cov_matrix), n_posterior_samples)'
    catch
        stds = sqrt.(diag(cov_matrix))
        randn(n_posterior_samples, n_coef) .* stds' .+ beta_map'
    end

    # Compute propensity samples
    linear_samples = beta_samples * X'  # (n_samples, n)
    propensity_samples = _sigmoid.(linear_samples)

    # Summary statistics
    posterior_mean = vec(mean(propensity_samples, dims=1))
    posterior_sd = vec(std(propensity_samples, dims=1))

    coefficient_mean = vec(mean(beta_samples, dims=1))
    coefficient_sd = vec(std(beta_samples, dims=1))

    mean_uncertainty = mean(posterior_sd)
    propensity_range = maximum(posterior_mean) - minimum(posterior_mean)

    return BayesianPropensityResult(
        propensity_samples,
        posterior_mean,
        posterior_sd,
        nothing,
        0,
        nothing,
        0.0,
        0.0,
        "logistic_laplace",
        n,
        Int(sum(T)),
        Int(n - sum(T)),
        mean_uncertainty,
        propensity_range,
        coefficient_mean,
        coefficient_sd,
        beta_samples,
        prior_sd
    )
end


# Handle vector covariates
function bayesian_propensity_stratified(
    treatment::AbstractVector{<:Real},
    covariates::AbstractVector{<:Real};
    kwargs...
)
    return bayesian_propensity_stratified(treatment, reshape(covariates, :, 1); kwargs...)
end

function bayesian_propensity_logistic(
    treatment::AbstractVector{<:Real},
    covariates::AbstractVector{<:Real};
    kwargs...
)
    return bayesian_propensity_logistic(treatment, reshape(covariates, :, 1); kwargs...)
end


"""
    bayesian_propensity(treatment, covariates; method="auto", kwargs...)

Bayesian propensity score estimation with automatic method selection.

# Arguments
- `treatment`: Binary treatment indicator (0/1)
- `covariates`: Covariate matrix
- `method`: "auto", "stratified", or "logistic"
"""
function bayesian_propensity(
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    method::String = "auto",
    kwargs...
)
    if method == "auto"
        # Heuristic: use stratified if few unique values per covariate
        n_unique = [length(unique(covariates[:, j])) for j in 1:size(covariates, 2)]
        if all(nu -> nu <= 10, n_unique)
            method = "stratified"
        else
            method = "logistic"
        end
    end

    # Filter kwargs to only include parameters accepted by the selected method
    stratified_params = (:prior_alpha, :prior_beta, :n_bins, :n_posterior_samples)
    logistic_params = (:prior_sd, :n_posterior_samples, :include_intercept)

    if method == "stratified"
        filtered_kwargs = filter(kv -> kv.first in stratified_params, kwargs)
        return bayesian_propensity_stratified(treatment, covariates; filtered_kwargs...)
    elseif method == "logistic"
        filtered_kwargs = filter(kv -> kv.first in logistic_params, kwargs)
        return bayesian_propensity_logistic(treatment, covariates; filtered_kwargs...)
    else
        throw(ArgumentError("Unknown method: $method. Use 'auto', 'stratified', or 'logistic'."))
    end
end

function bayesian_propensity(
    treatment::AbstractVector{<:Real},
    covariates::AbstractVector{<:Real};
    kwargs...
)
    return bayesian_propensity(treatment, reshape(covariates, :, 1); kwargs...)
end
