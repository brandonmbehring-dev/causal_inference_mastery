"""
Conditional Quantile Treatment Effects via quantile regression.

Uses QuantileRegressions.jl for the underlying quantile regression.
"""

using Statistics
using LinearAlgebra
using Distributions

# types.jl is included by CausalEstimators.jl


"""
    conditional_qte(problem::QTEProblem; alpha=0.05)

Estimate conditional QTE via quantile regression.

Model: Q_τ(Y | T, X) = α + τ_q * T + β' * X

# Arguments
- `problem::QTEProblem`: Problem with outcome, treatment, covariates, quantile
- `alpha::Float64=0.05`: Significance level for CI

# Returns
- `QTESolution`: Solution with treatment coefficient as tau_q
"""
function conditional_qte(
    problem::QTEProblem{T};
    alpha::T = T(0.05)
) where T<:Real

    outcome = problem.outcome
    treatment = problem.treatment
    covariates = problem.covariates
    τ = problem.quantile

    n = length(outcome)
    n_treated = sum(treatment .== one(T))
    n_control = sum(treatment .== zero(T))

    # Build design matrix: [intercept, treatment, covariates]
    if covariates !== nothing
        X = hcat(ones(T, n), treatment, covariates)
    else
        X = hcat(ones(T, n), treatment)
    end

    # Fit quantile regression using iteratively reweighted least squares
    beta, vcov = quantile_regression(outcome, X, τ)

    # Treatment coefficient is at index 2 (after intercept)
    tau_q = beta[2]
    se = sqrt(vcov[2, 2])

    # CI using normal approximation
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = tau_q - z_crit * se
    ci_upper = tau_q + z_crit * se

    # P-value for H0: tau_q = 0
    z_stat = tau_q / se
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))

    return QTESolution(
        tau_q = tau_q,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        quantile = τ,
        method = :conditional,
        n_treated = Int(n_treated),
        n_control = Int(n_control),
        n_total = n,
        outcome_support = (minimum(outcome), maximum(outcome)),
        inference = :asymptotic
    )
end


"""
    quantile_regression(y, X, τ; max_iter=1000, tol=1e-6)

Fit quantile regression using iteratively reweighted least squares (IRLS).

Minimizes: Σ ρ_τ(y_i - x_i'β)
where ρ_τ(u) = u * (τ - I(u < 0)) is the check function.

# Returns
- `beta::Vector`: Coefficient estimates
- `vcov::Matrix`: Variance-covariance matrix (sandwich estimator)
"""
function quantile_regression(
    y::Vector{T},
    X::Matrix{T},
    τ::T;
    max_iter::Int = 1000,
    tol::T = T(1e-6)
) where T<:Real

    n, p = size(X)

    # Initialize with OLS
    beta = X \ y

    for iter in 1:max_iter
        # Compute residuals
        resid = y - X * beta

        # Compute weights for IRLS
        # ρ'(u) = τ - I(u < 0)
        # Weight: |ρ'(u)| / |u|
        weights = zeros(T, n)
        for i in 1:n
            if abs(resid[i]) < tol
                weights[i] = one(T) / tol
            else
                weights[i] = ifelse(resid[i] < zero(T), one(T) - τ, τ) / abs(resid[i])
            end
        end

        # Weighted least squares step
        W = Diagonal(weights)
        beta_new = (X' * W * X) \ (X' * W * y)

        # Check convergence
        if norm(beta_new - beta) < tol
            beta = beta_new
            break
        end

        beta = beta_new
    end

    # Compute sandwich variance-covariance matrix
    resid = y - X * beta

    # Kernel density estimate for f(0|x)
    h = T(1.06) * std(resid) * n^(-T(0.2))  # Silverman's rule
    f0 = sum(abs.(resid) .< h) / (2 * h * n)
    f0 = max(f0, T(1e-10))

    # Sandwich: (X'X)^{-1} * Ω * (X'X)^{-1}
    # where Ω = τ(1-τ) * X'X / f(0)^2
    XtX_inv = inv(X' * X)
    Omega = τ * (one(T) - τ) * (X' * X) / f0^2
    vcov = XtX_inv * Omega * XtX_inv

    return beta, vcov
end


"""
    conditional_qte(outcome, treatment, covariates; quantile=0.5, alpha=0.05)

Convenience method.
"""
function conditional_qte(
    outcome::Vector{T},
    treatment::Vector{T},
    covariates::Matrix{T};
    quantile::T = T(0.5),
    alpha::T = T(0.05)
) where T<:Real

    problem = QTEProblem(
        outcome = outcome,
        treatment = treatment,
        covariates = covariates,
        quantile = quantile
    )
    return conditional_qte(problem; alpha=alpha)
end
