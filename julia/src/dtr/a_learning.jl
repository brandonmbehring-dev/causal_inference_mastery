"""
    A-learning for Dynamic Treatment Regimes

Implements A-learning (Advantage Learning) with double robustness for estimating
optimal dynamic treatment regimes. A-learning is consistent if EITHER the
propensity score model OR the baseline outcome model is correctly specified.

Algorithm
---------
A-learning estimates the optimal treatment regime by solving the estimating
equation:

    E[(A - π(H)) * (Y - γ(H,A;ψ) - m(H)) * ∂γ/∂ψ] = 0

where:
- π(H) = P(A=1|H) is the propensity score
- γ(H,A;ψ) = A * H'ψ is the blip function (treatment contrast)
- m(H) = E[Y|H, A=0] is the baseline outcome model

For multi-stage settings, backward induction is used similar to Q-learning.

Double Robustness
-----------------
A-learning is consistent if EITHER:
1. π(H) is correctly specified, OR
2. m(H) is correctly specified

This makes it more robust to model misspecification than Q-learning.

References
----------
Robins, J. M. (2004). Optimal structural nested models for optimal sequential
    decisions. In Proceedings of the Second Seattle Symposium on Biostatistics.
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B, 65(2), 331-355.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating optimal
    dynamic treatment regimes. Statistical Science, 29(4), 640-661.
"""

using LinearAlgebra
using Statistics
using Distributions


"""
    a_learning(data::DTRData; kwargs...) -> ALearningResult

A-learning for optimal dynamic treatment regime estimation.

Estimates the optimal treatment regime using doubly robust A-learning
with backward induction for multi-stage settings.

# Arguments
- `data::DTRData`: Multi-stage treatment data with outcomes, treatments, and covariates.
- `propensity_model::Symbol=:logit`: Model for P(A=1|H): :logit or :probit.
- `outcome_model::Symbol=:ols`: Model for E[Y|H, A=0]: :ols or :ridge.
- `doubly_robust::Bool=true`: Use doubly robust estimator.
- `se_method::Symbol=:sandwich`: Standard error method (:sandwich or :bootstrap).
- `n_bootstrap::Int=500`: Number of bootstrap replicates if se_method=:bootstrap.
- `alpha::Float64=0.05`: Significance level for confidence intervals.
- `propensity_trim::Float64=0.01`: Trim propensity scores to [trim, 1-trim].

# Returns
- `ALearningResult`: Contains optimal value, blip coefficients, and regime.

# Notes
A-learning is doubly robust: consistent if either propensity or outcome model
is correct. This makes it more robust than Q-learning to model misspecification.

# Examples
```julia
using Random
Random.seed!(42)

n = 500
X = randn(n, 3)
A = Float64.(rand(n) .< 0.5)
Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
data = DTRData([Y], [A], [X])
result = a_learning(data)
println("Optimal value: \$(result.value_estimate)")
```
"""
function a_learning(
    data::DTRData{T};
    propensity_model::Symbol = :logit,
    outcome_model::Symbol = :ols,
    doubly_robust::Bool = true,
    se_method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
    alpha::Float64 = 0.05,
    propensity_trim::Float64 = 0.01,
) where {T<:Real}

    # Validate inputs
    propensity_model in (:logit, :probit) || error(
        "CRITICAL ERROR: Unknown propensity_model.\n" *
        "Function: a_learning\n" *
        "propensity_model must be :logit or :probit, got $propensity_model"
    )
    outcome_model in (:ols, :ridge) || error(
        "CRITICAL ERROR: Unknown outcome_model.\n" *
        "Function: a_learning\n" *
        "outcome_model must be :ols or :ridge, got $outcome_model"
    )
    se_method in (:sandwich, :bootstrap) || error(
        "CRITICAL ERROR: Unknown se_method.\n" *
        "Function: a_learning\n" *
        "se_method must be :sandwich or :bootstrap, got $se_method"
    )

    if data.n_obs < 50
        @warn "Small sample size (n=$(data.n_obs)). A-learning estimates may be unstable."
    end

    K = data.n_stages
    n = data.n_obs

    # Storage for results
    blip_coefficients = Vector{Vector{Float64}}(undef, K)
    blip_se = Vector{Vector{Float64}}(undef, K)

    # Backward induction: k = K, K-1, ..., 1
    future_value = zeros(T, n)

    for k in K:-1:1
        # Get data for this stage
        Y_k = data.outcomes[k]
        A_k = data.treatments[k]
        H_k = get_history(data, k)

        # Pseudo-outcome for this stage
        pseudo_outcome = Y_k .+ future_value

        # Fit propensity score
        propensity, _ = _fit_propensity(A_k, H_k, propensity_model)

        # Trim propensity scores
        propensity = clamp.(propensity, propensity_trim, 1 - propensity_trim)

        # Fit baseline outcome model (on controls)
        if doubly_robust
            baseline_pred, _ = _fit_baseline_outcome(
                pseudo_outcome, A_k, H_k, outcome_model
            )
        else
            baseline_pred = zeros(T, n)
        end

        # Solve A-learning estimating equation
        psi_k = _solve_a_learning_equation(
            pseudo_outcome, A_k, H_k, propensity, baseline_pred
        )

        # Compute standard errors
        psi_se = _compute_a_learning_se(
            pseudo_outcome, A_k, H_k, propensity, baseline_pred, psi_k,
            method=se_method, n_bootstrap=n_bootstrap
        )

        # Store results
        blip_coefficients[k] = psi_k
        blip_se[k] = psi_se

        # Compute value function for next iteration
        # Add intercept for blip computation
        H_k_aug = hcat(ones(T, n), H_k)
        blip_values = H_k_aug * psi_k
        # Value = baseline + max(0, blip)
        future_value = baseline_pred .+ max.(0, blip_values)
    end

    # Estimate value under optimal regime
    H_1 = get_history(data, 1)
    H_1_aug = hcat(ones(T, n), H_1)

    # Refit baseline for stage 1 for value estimation
    if doubly_robust
        baseline_1, _ = _fit_baseline_outcome(
            data.outcomes[1], data.treatments[1], H_1, outcome_model
        )
    else
        baseline_1 = zeros(T, n)
    end

    value_estimate = mean(baseline_1 .+ max.(0, H_1_aug * blip_coefficients[1]))

    # Compute value SE
    if se_method == :bootstrap
        value_se = _bootstrap_a_learning_value_se(
            data, propensity_model, outcome_model, doubly_robust, n_bootstrap
        )
    else
        V_1 = baseline_1 .+ max.(0, H_1_aug * blip_coefficients[1])
        value_se = std(V_1) / sqrt(n)
    end

    # Confidence intervals
    z_crit = quantile(Normal(), 1 - alpha / 2)
    value_ci_lower = value_estimate - z_crit * value_se
    value_ci_upper = value_estimate + z_crit * value_se

    return ALearningResult(
        value_estimate,
        value_se,
        value_ci_lower,
        value_ci_upper,
        blip_coefficients,
        blip_se,
        K,
        propensity_model,
        outcome_model,
        doubly_robust,
        se_method,
    )
end


"""
    a_learning_single_stage(outcome, treatment, covariates; kwargs...) -> ALearningResult

Convenience wrapper for single-stage A-learning.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y of length n.
- `treatment::Vector{T}`: Binary treatment A of length n.
- `covariates::Matrix{T}`: Covariates X of size (n, p).
- `propensity_model::Symbol=:logit`: Propensity score model.
- `outcome_model::Symbol=:ols`: Baseline outcome model.
- `doubly_robust::Bool=true`: Use doubly robust estimation.
- `se_method::Symbol=:sandwich`: Standard error method.
- `n_bootstrap::Int=500`: Bootstrap replicates if se_method=:bootstrap.
- `alpha::Float64=0.05`: Significance level.

# Examples
```julia
n = 500
X = randn(n, 3)
A = Float64.(rand(n) .< 0.5)
Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
result = a_learning_single_stage(Y, A, X)
println("Blip intercept: \$(result.blip_coefficients[1][1])")
```
"""
function a_learning_single_stage(
    outcome::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    propensity_model::Symbol = :logit,
    outcome_model::Symbol = :ols,
    doubly_robust::Bool = true,
    se_method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
    alpha::Float64 = 0.05,
    propensity_trim::Float64 = 0.01,
)
    data = DTRData([collect(outcome)], [collect(treatment)], [Matrix(covariates)])
    return a_learning(
        data;
        propensity_model=propensity_model,
        outcome_model=outcome_model,
        doubly_robust=doubly_robust,
        se_method=se_method,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        propensity_trim=propensity_trim,
    )
end


"""
    _fit_propensity(treatment, covariates, model)

Fit propensity score model P(A=1|H).

# Arguments
- `treatment::Vector{T}`: Binary treatment of length n.
- `covariates::Matrix{T}`: Covariates of size (n, p).
- `model::Symbol`: :logit or :probit.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (propensity_scores, coefficients)
"""
function _fit_propensity(
    treatment::AbstractVector{T},
    covariates::AbstractMatrix{T},
    model::Symbol,
) where {T<:Real}
    n = length(treatment)
    p = size(covariates, 2)

    # Add intercept
    X = hcat(ones(T, n), covariates)

    # Logistic function
    sigmoid(x) = 1 / (1 + exp(-x))

    if model == :logit
        # Iteratively reweighted least squares for logistic regression
        beta = zeros(Float64, size(X, 2))

        for iter in 1:25
            eta = X * beta
            mu = sigmoid.(eta)
            # Avoid numerical issues
            mu = clamp.(mu, 1e-10, 1 - 1e-10)
            W = Diagonal(mu .* (1 .- mu))
            z = eta .+ (treatment .- mu) ./ (mu .* (1 .- mu))

            # Weighted least squares update
            XtWX = X' * W * X
            XtWz = X' * W * z
            beta_new = pinv(XtWX) * XtWz

            if maximum(abs.(beta_new .- beta)) < 1e-8
                break
            end
            beta = beta_new
        end

        propensity = sigmoid.(X * beta)

    elseif model == :probit
        # Simplified probit via logit approximation (scale by 1.7)
        beta = zeros(Float64, size(X, 2))

        for iter in 1:25
            eta = X * beta * 1.7  # Probit-logit scaling
            mu = sigmoid.(eta)
            mu = clamp.(mu, 1e-10, 1 - 1e-10)
            W = Diagonal(mu .* (1 .- mu))
            z = eta .+ (treatment .- mu) ./ (mu .* (1 .- mu))

            XtWX = X' * W * X
            XtWz = X' * W * z
            beta_new = pinv(XtWX) * XtWz

            if maximum(abs.(beta_new .- beta)) < 1e-8
                break
            end
            beta = beta_new
        end

        propensity = sigmoid.(X * beta * 1.7)

    else
        error("Unknown propensity model: $model")
    end

    return propensity, beta
end


"""
    _fit_baseline_outcome(outcome, treatment, covariates, model; ridge_lambda=0.1)

Fit baseline outcome model E[Y|H, A=0] on control observations.

# Arguments
- `outcome::Vector{T}`: Outcome of length n.
- `treatment::Vector{T}`: Treatment of length n.
- `covariates::Matrix{T}`: Covariates of size (n, p).
- `model::Symbol`: :ols or :ridge.
- `ridge_lambda::Float64=0.1`: Ridge penalty.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (predictions_for_all, coefficients)
"""
function _fit_baseline_outcome(
    outcome::AbstractVector{T},
    treatment::AbstractVector{T},
    covariates::AbstractMatrix{T},
    model::Symbol;
    ridge_lambda::Float64 = 0.1,
) where {T<:Real}
    n = length(outcome)
    p = size(covariates, 2)

    # Add intercept
    X = hcat(ones(T, n), covariates)

    # Fit on control observations (A=0)
    control_mask = treatment .== 0
    X_control = X[control_mask, :]
    Y_control = outcome[control_mask]

    if length(Y_control) < 5
        # Not enough controls, return zeros
        return zeros(Float64, n), zeros(Float64, size(X, 2))
    end

    if model == :ols
        beta = X_control \ Y_control

    elseif model == :ridge
        # Ridge regression
        XtX = X_control' * X_control
        XtY = X_control' * Y_control
        penalty = ridge_lambda * I(size(XtX, 1))
        # Don't penalize intercept (set first diagonal to 0)
        penalty_mat = Matrix(penalty)
        penalty_mat[1, 1] = 0.0
        beta = (XtX + penalty_mat) \ XtY

    else
        error("Unknown outcome model: $model")
    end

    # Predict for all observations
    predictions = X * beta

    return predictions, beta
end


"""
    _solve_a_learning_equation(outcome, treatment, covariates, propensity, baseline_pred)

Solve A-learning estimating equation for blip coefficients.

The A-learning estimating equation is:
    E[(A - π(H)) * (Y - m(H) - A*H'ψ) * H] = 0

Solved via weighted least squares.

# Returns
- `Vector{Float64}`: Blip coefficients ψ (with intercept).
"""
function _solve_a_learning_equation(
    outcome::AbstractVector{T},
    treatment::AbstractVector{T},
    covariates::AbstractMatrix{T},
    propensity::AbstractVector,
    baseline_pred::AbstractVector,
) where {T<:Real}
    n = length(outcome)

    # Add intercept
    H = hcat(ones(T, n), covariates)

    # Residual after removing baseline
    residual = outcome .- baseline_pred

    # Weights: (A - π)²
    weights = (treatment .- propensity).^2

    # Design matrix for blip: A * H
    blip_design = treatment .* H

    # Weighted least squares
    W = Diagonal(weights)
    XtWX = blip_design' * W * blip_design
    XtWY = blip_design' * W * residual

    psi = pinv(XtWX) * XtWY

    return psi
end


"""
    _compute_a_learning_se(pseudo_outcome, treatment, history, propensity, baseline_pred, blip_coef; kwargs...)

Compute standard errors for A-learning blip coefficients.

# Arguments
- `method::Symbol`: :sandwich or :bootstrap.
- `n_bootstrap::Int`: Number of bootstrap replicates.

# Returns
- `Vector{Float64}`: Standard errors for blip coefficients.
"""
function _compute_a_learning_se(
    pseudo_outcome::AbstractVector{T},
    treatment::AbstractVector{T},
    history::AbstractMatrix{T},
    propensity::AbstractVector,
    baseline_pred::AbstractVector,
    blip_coef::AbstractVector;
    method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
) where {T<:Real}
    n = length(pseudo_outcome)
    p = size(history, 2)

    # Add intercept
    H = hcat(ones(T, n), history)

    if method == :sandwich
        # Influence function approach
        # Residual
        residual = pseudo_outcome .- baseline_pred .- treatment .* (H * blip_coef)

        # Weights
        weights = (treatment .- propensity).^2

        # Design
        blip_design = treatment .* H

        # Sandwich variance
        W = Diagonal(weights)
        XtWX = blip_design' * W * blip_design
        XtWX_inv = pinv(XtWX)

        # Meat matrix: Σᵢ wᵢ² * residualᵢ² * Hᵢ Hᵢ'
        meat = zeros(Float64, size(H, 2), size(H, 2))
        for i in 1:n
            Hi = blip_design[i, :]
            meat .+= weights[i]^2 * residual[i]^2 * (Hi * Hi')
        end

        # Sandwich variance
        var_mat = XtWX_inv * meat * XtWX_inv
        se = sqrt.(max.(diag(var_mat), 0.0))

    elseif method == :bootstrap
        bootstrap_psi = Vector{Vector{Float64}}()

        for _ in 1:n_bootstrap
            idx = rand(1:n, n)

            psi_boot = _solve_a_learning_equation(
                pseudo_outcome[idx],
                treatment[idx],
                history[idx, :],
                propensity[idx],
                baseline_pred[idx],
            )
            push!(bootstrap_psi, psi_boot)
        end

        # Stack into matrix and compute std
        psi_matrix = hcat(bootstrap_psi...)
        se = vec(std(psi_matrix, dims=2))

    else
        error("Unknown SE method: $method")
    end

    return se
end


"""
    _bootstrap_a_learning_value_se(data, propensity_model, outcome_model, doubly_robust, n_bootstrap)

Bootstrap standard error for value function estimate.
"""
function _bootstrap_a_learning_value_se(
    data::DTRData{T},
    propensity_model::Symbol,
    outcome_model::Symbol,
    doubly_robust::Bool,
    n_bootstrap::Int,
) where {T<:Real}
    n = data.n_obs
    K = data.n_stages
    bootstrap_values = Float64[]

    for _ in 1:n_bootstrap
        # Resample observations
        idx = rand(1:n, n)

        # Create bootstrap data
        boot_outcomes = [data.outcomes[k][idx] for k in 1:K]
        boot_treatments = [data.treatments[k][idx] for k in 1:K]
        boot_covariates = [data.covariates[k][idx, :] for k in 1:K]

        boot_data = DTRData(boot_outcomes, boot_treatments, boot_covariates)

        # Fit A-learning
        result = a_learning(
            boot_data;
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            doubly_robust=doubly_robust,
            se_method=:sandwich,
        )
        push!(bootstrap_values, result.value_estimate)
    end

    return std(bootstrap_values)
end
