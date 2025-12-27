"""
    Q-learning for Dynamic Treatment Regimes

Implements Q-learning with backward induction for estimating optimal
dynamic treatment regimes in multi-stage settings.

Algorithm
---------
Q-learning estimates the optimal treatment regime by fitting Q-functions
via backward induction from the final stage to the first:

    For k = K down to 1:
        1. Pseudo-outcome: Y_tilde_k = Y_k + V_hat_{k+1}(H_{k+1})
        2. Fit Q_k(H_k, a) = mu_k(H_k) + a * gamma_k(H_k)
        3. Blip: gamma_k(H_k) = Q_k(H_k, 1) - Q_k(H_k, 0) = H_k' * psi_k
        4. Optimal: d*_k(H_k) = I(gamma_k(H_k) > 0)
        5. Value: V_k(H_k) = max_a Q_k(H_k, a)

References
----------
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B, 65(2), 331-355.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating optimal
    dynamic treatment regimes. Statistical Science, 29(4), 640-661.
"""

using LinearAlgebra
using Statistics
using Distributions


"""
    q_learning(data::DTRData; kwargs...) -> QLearningResult

Q-learning for optimal dynamic treatment regime estimation.

Estimates the optimal treatment regime using backward induction,
fitting Q-functions from the final stage to the first.

# Arguments
- `data::DTRData`: Multi-stage treatment data with outcomes, treatments, and covariates.
- `se_method::Symbol=:sandwich`: Standard error method (:sandwich or :bootstrap).
- `n_bootstrap::Int=500`: Number of bootstrap replicates if se_method=:bootstrap.
- `alpha::Float64=0.05`: Significance level for confidence intervals.

# Returns
- `QLearningResult`: Contains optimal value, blip coefficients, and SE.

# Notes
The Q-function at each stage is parameterized as:
    Q_k(H_k, a) = H_k' * beta_k + a * (H_k' * psi_k)

The optimal treatment at stage k is:
    d*_k(H_k) = I(H_k' * psi_k > 0)

# Examples
```julia
using Random
Random.seed!(42)

n = 500
X = randn(n, 3)
A = Float64.(rand(n) .< 0.5)
Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
data = DTRData([Y], [A], [X])
result = q_learning(data)
println("Optimal value: \$(result.value_estimate)")
```
"""
function q_learning(
    data::DTRData{T};
    se_method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
    alpha::Float64 = 0.05,
) where {T<:Real}

    # Validate inputs
    se_method in (:sandwich, :bootstrap) || error(
        "CRITICAL ERROR: Unknown se_method.\n" *
        "Function: q_learning\n" *
        "se_method must be :sandwich or :bootstrap, got $se_method"
    )

    if data.n_obs < 50
        @warn "Small sample size (n=$(data.n_obs)). Q-learning estimates may be unstable."
    end

    K = data.n_stages
    n = data.n_obs

    # Storage for results
    blip_coefficients = Vector{Vector{Float64}}(undef, K)
    blip_se = Vector{Vector{Float64}}(undef, K)
    baseline_coefficients = Vector{Vector{Float64}}(undef, K)

    # Backward induction: k = K, K-1, ..., 1
    # Initialize value function for stage after K (i.e., V_{K+1} = 0)
    future_value = zeros(T, n)

    for k in K:-1:1
        # Get data for this stage
        Y_k = data.outcomes[k]
        A_k = data.treatments[k]
        H_k = get_history(data, k)

        # Pseudo-outcome: Y_tilde_k = Y_k + V_{k+1}
        pseudo_outcome = Y_k .+ future_value

        # Fit Q-function
        beta_k, psi_k = _fit_stage_q_function(pseudo_outcome, A_k, H_k)

        # Compute standard errors
        psi_se = _compute_blip_se(
            pseudo_outcome, A_k, H_k, psi_k,
            method=se_method, n_bootstrap=n_bootstrap
        )

        # Store results
        blip_coefficients[k] = psi_k
        blip_se[k] = psi_se
        baseline_coefficients[k] = beta_k

        # Compute value function for this stage (for next iteration)
        # V_k(H_k) = max_a Q_k(H_k, a) = Q_k(H_k, d*_k(H_k))
        # Note: beta_k and psi_k include intercept
        H_k_aug = hcat(ones(T, n), H_k)
        blip_values = H_k_aug * psi_k
        optimal_A_k = Float64.(blip_values .> 0)
        future_value = H_k_aug * beta_k .+ optimal_A_k .* blip_values
    end

    # Estimate value under optimal regime
    H_1 = get_history(data, 1)
    H_1_aug = hcat(ones(T, n), H_1)
    value_estimate = mean(H_1_aug * baseline_coefficients[1] .+
                          max.(0, H_1_aug * blip_coefficients[1]))

    # Compute value SE
    if se_method == :bootstrap
        value_se = _bootstrap_value_se(data, n_bootstrap)
    else
        # Simplified SE: variance of V_1(H_1)
        V_1 = H_1_aug * baseline_coefficients[1] .+ max.(0, H_1_aug * blip_coefficients[1])
        value_se = std(V_1) / sqrt(n)
    end

    # Confidence intervals
    z_crit = quantile(Normal(), 1 - alpha / 2)
    value_ci_lower = value_estimate - z_crit * value_se
    value_ci_upper = value_estimate + z_crit * value_se

    return QLearningResult(
        value_estimate,
        value_se,
        value_ci_lower,
        value_ci_upper,
        blip_coefficients,
        blip_se,
        K,
        se_method,
    )
end


"""
    q_learning_single_stage(outcome, treatment, covariates; kwargs...) -> QLearningResult

Convenience wrapper for single-stage Q-learning.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y of length n.
- `treatment::Vector{T}`: Binary treatment A of length n.
- `covariates::Matrix{T}`: Covariates X of size (n, p).
- `se_method::Symbol=:sandwich`: Standard error method.
- `n_bootstrap::Int=500`: Bootstrap replicates if se_method=:bootstrap.
- `alpha::Float64=0.05`: Significance level.

# Examples
```julia
n = 500
X = randn(n, 3)
A = Float64.(rand(n) .< 0.5)
Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
result = q_learning_single_stage(Y, A, X)
println("Blip intercept: \$(result.blip_coefficients[1][1])")
```
"""
function q_learning_single_stage(
    outcome::AbstractVector{<:Real},
    treatment::AbstractVector{<:Real},
    covariates::AbstractMatrix{<:Real};
    se_method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
    alpha::Float64 = 0.05,
)
    data = DTRData([collect(outcome)], [collect(treatment)], [Matrix(covariates)])
    return q_learning(data; se_method=se_method, n_bootstrap=n_bootstrap, alpha=alpha)
end


"""
    _fit_stage_q_function(pseudo_outcome, treatment, history)

Fit Q-function at a single stage.

Models Q(H, a) = H' * beta + a * (H' * psi) via OLS regression.

# Returns
- `beta::Vector{Float64}`: Baseline coefficients (with intercept).
- `psi::Vector{Float64}`: Blip coefficients (with intercept).
"""
function _fit_stage_q_function(
    pseudo_outcome::AbstractVector{T},
    treatment::AbstractVector{T},
    history::AbstractMatrix{T},
) where {T<:Real}
    n = length(pseudo_outcome)
    p = size(history, 2)

    # Add intercept to history
    H_with_intercept = hcat(ones(T, n), history)
    p_aug = p + 1

    # Build design matrix for Q(H, a) = H'beta + a*(H'psi)
    # Equivalently: [H, a*H] * [beta; psi]
    A_times_H = treatment .* H_with_intercept
    design = hcat(H_with_intercept, A_times_H)

    # OLS: solve (X'X)^{-1} X'Y
    coefficients = design \ pseudo_outcome

    # Split coefficients
    beta = coefficients[1:p_aug]
    psi = coefficients[(p_aug+1):end]

    return beta, psi
end


"""
    _compute_blip_se(pseudo_outcome, treatment, history, blip_coef; kwargs...)

Compute standard errors for blip coefficients.

# Arguments
- `method::Symbol`: :sandwich or :bootstrap
- `n_bootstrap::Int`: Number of bootstrap replicates.

# Returns
- `Vector{Float64}`: Standard errors for blip coefficients.
"""
function _compute_blip_se(
    pseudo_outcome::AbstractVector{T},
    treatment::AbstractVector{T},
    history::AbstractMatrix{T},
    blip_coef::AbstractVector;
    method::Symbol = :sandwich,
    n_bootstrap::Int = 500,
) where {T<:Real}
    n = length(pseudo_outcome)
    p = size(history, 2)
    p_aug = p + 1

    # Add intercept
    H_with_intercept = hcat(ones(T, n), history)

    if method == :sandwich
        # Sandwich estimator for OLS
        # Full design matrix
        A_times_H = treatment .* H_with_intercept
        design = hcat(H_with_intercept, A_times_H)

        # Full coefficients and residuals
        beta_full = design \ pseudo_outcome
        residuals = pseudo_outcome .- design * beta_full

        # Sandwich: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
        XtX = design' * design
        XtX_inv = pinv(XtX)
        meat = design' * Diagonal(residuals.^2) * design
        sandwich_var = XtX_inv * meat * XtX_inv

        # Extract variance for psi (last p_aug coefficients)
        psi_var = diag(sandwich_var[(p_aug+1):end, (p_aug+1):end])
        se = sqrt.(max.(psi_var, 0.0))

    elseif method == :bootstrap
        # Nonparametric bootstrap
        bootstrap_psi = Vector{Vector{Float64}}()

        for _ in 1:n_bootstrap
            idx = rand(1:n, n)
            Y_boot = pseudo_outcome[idx]
            A_boot = treatment[idx]
            H_boot = history[idx, :]

            _, psi_boot = _fit_stage_q_function(Y_boot, A_boot, H_boot)
            push!(bootstrap_psi, psi_boot)
        end

        # Stack into matrix and compute std
        psi_matrix = hcat(bootstrap_psi...)  # p_aug x n_bootstrap
        se = vec(std(psi_matrix, dims=2))

    else
        error("Unknown method: $method")
    end

    return se
end


"""
    _bootstrap_value_se(data, n_bootstrap)

Bootstrap standard error for value function estimate.
"""
function _bootstrap_value_se(
    data::DTRData{T},
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

        # Fit Q-learning
        result = q_learning(boot_data; se_method=:sandwich)
        push!(bootstrap_values, result.value_estimate)
    end

    return std(bootstrap_values)
end
