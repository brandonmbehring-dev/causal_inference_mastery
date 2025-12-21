"""
Heckman two-step selection model implementation.

Implements the Heckman (1979) correction for sample selection bias,
where outcomes are only observed for a selected subsample.

# Algorithm
1. Estimate probit selection equation via MLE
2. Compute Inverse Mills Ratio for selected observations
3. Estimate outcome equation with IMR as additional regressor
4. Compute Heckman-corrected standard errors (sandwich estimator)

# References
- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error.
- Wooldridge, J. M. (2010). Econometric Analysis, Chapter 19.
"""

using LinearAlgebra
using Statistics
using Distributions
using Optim

"""
    solve(problem::HeckmanProblem, estimator::HeckmanTwoStep) -> HeckmanSolution

Estimate Heckman two-step selection model.

# Pipeline
1. Estimate selection equation (probit) on full sample: P(S=1|Z)
2. Compute Inverse Mills Ratio for selected observations
3. Estimate outcome equation (OLS) with IMR as additional regressor
4. Compute Heckman-corrected standard errors

# Returns
`HeckmanSolution` with estimate, standard errors, selection parameters, and diagnostics.

# Example
```julia
problem = HeckmanProblem(outcomes, selected, Z, X, (alpha=0.05,))
solution = solve(problem, HeckmanTwoStep())
println("Estimate: \$(solution.estimate) ± \$(solution.se)")
println("Selection test p-value: \$(solution.lambda_pvalue)")
```
"""
function solve(problem::HeckmanProblem{T}, estimator::HeckmanTwoStep) where {T}
    # Extract data
    outcomes = problem.outcomes
    selected = problem.selected
    Z = problem.selection_covariates
    X = problem.outcome_covariates
    α = problem.parameters.alpha

    n = length(outcomes)
    n_selected = sum(selected)

    # Use Z for X if not provided (with warning behavior in Python)
    if isnothing(X)
        X = Z
    end

    # =========================================================================
    # Step 1: Fit Probit Selection Equation
    # =========================================================================

    gamma, selection_probs, probit_converged = fit_probit(
        selected, Z;
        add_intercept=estimator.add_intercept,
        max_iter=estimator.max_iter,
        tol=estimator.tol
    )

    # =========================================================================
    # Step 2: Compute Inverse Mills Ratio
    # =========================================================================

    imr_full = compute_imr(selection_probs)
    imr_selected = imr_full[selected]

    # =========================================================================
    # Step 3: Fit Outcome Equation with IMR
    # =========================================================================

    # Extract selected observations
    outcomes_selected = outcomes[selected]
    X_selected = X[selected, :]

    # Build design matrix with intercept and IMR
    if estimator.add_intercept
        X_design = hcat(ones(T, n_selected), X_selected, imr_selected)
    else
        X_design = hcat(X_selected, imr_selected)
    end

    # OLS estimation
    beta, residuals, sigma_hat = fit_ols(outcomes_selected, X_design)

    # Extract IMR coefficient (always last)
    lambda_coef = beta[end]

    # Compute rho = lambda / sigma
    rho = sigma_hat > 0 ? lambda_coef / sigma_hat : zero(T)
    rho = clamp(rho, -one(T), one(T))

    # =========================================================================
    # Step 4: Compute Heckman-Corrected Standard Errors
    # =========================================================================

    # Build Z design matrix for variance computation
    if estimator.add_intercept
        Z_full = hcat(ones(T, n), Z)
    else
        Z_full = Z
    end

    vcov = compute_heckman_vcov(
        outcomes_selected, X_design, Z_full, selected,
        gamma, beta, sigma_hat, lambda_coef, imr_selected
    )

    # Standard errors from diagonal (clamp to handle numerical issues with extreme selection)
    vcov_diag = diag(vcov)
    vcov_diag = max.(vcov_diag, T(1e-16))  # Ensure non-negative
    se_beta = sqrt.(vcov_diag)

    # Lambda test
    lambda_idx = length(beta)  # Last coefficient
    lambda_se = se_beta[lambda_idx]
    lambda_t = lambda_se > 0 ? lambda_coef / lambda_se : zero(T)
    lambda_pvalue = 2 * (1 - cdf(Normal(), abs(lambda_t)))

    # =========================================================================
    # Step 5: Extract Primary Estimate
    # =========================================================================

    # First non-intercept coefficient
    estimate_idx = estimator.add_intercept ? 2 : 1
    estimate = beta[estimate_idx]
    se = se_beta[estimate_idx]

    # Confidence interval
    z_crit = quantile(Normal(), 1 - α / 2)
    ci_lower = estimate - z_crit * se
    ci_upper = estimate + z_crit * se

    # =========================================================================
    # Return Solution
    # =========================================================================

    return HeckmanSolution{T}(
        estimate,
        se,
        ci_lower,
        ci_upper,
        rho,
        sigma_hat,
        lambda_coef,
        lambda_se,
        lambda_pvalue,
        n_selected,
        n,
        selection_probs,
        imr_full,
        gamma,
        beta,
        vcov,
        probit_converged,
        α
    )
end

"""
    fit_probit(y, X; add_intercept=true, max_iter=100, tol=1e-8)

Fit probit model via maximum likelihood.

# Returns
- `gamma::Vector`: Estimated coefficients
- `probs::Vector`: Fitted selection probabilities
- `converged::Bool`: Whether optimization converged
"""
function fit_probit(
    y::Vector{Bool},
    X::Matrix{T};
    add_intercept::Bool=true,
    max_iter::Int=100,
    tol::Float64=1e-8
) where {T<:Real}

    # Add intercept if requested
    if add_intercept
        X_design = hcat(ones(T, length(y)), X)
    else
        X_design = X
    end

    n, k = size(X_design)
    y_float = convert(Vector{T}, y)

    # Negative log-likelihood for probit
    function neg_log_likelihood(gamma)
        z = X_design * gamma
        z = clamp.(z, -30, 30)  # Numerical stability
        prob = cdf.(Normal(), z)
        prob = clamp.(prob, 1e-10, 1 - 1e-10)

        ll = sum(y_float .* log.(prob) .+ (1 .- y_float) .* log.(1 .- prob))
        return -ll
    end

    # Gradient
    function gradient!(G, gamma)
        z = X_design * gamma
        z = clamp.(z, -30, 30)
        prob = cdf.(Normal(), z)
        prob = clamp.(prob, 1e-10, 1 - 1e-10)
        pdf_z = pdf.(Normal(), z)

        # λ = φ/Φ for y=1, -φ/(1-Φ) for y=0
        lam = @. ifelse(y_float == 1, pdf_z / prob, -pdf_z / (1 - prob))
        G .= -X_design' * lam
    end

    # Initial guess
    gamma_init = zeros(T, k)

    # Optimize
    result = optimize(
        neg_log_likelihood,
        gradient!,
        gamma_init,
        BFGS(),
        Optim.Options(iterations=max_iter, g_tol=tol)
    )

    gamma = Optim.minimizer(result)
    converged = Optim.converged(result)

    # Compute fitted probabilities
    z = X_design * gamma
    z = clamp.(z, -30, 30)
    probs = cdf.(Normal(), z)

    return gamma, probs, converged
end

"""
    compute_imr(selection_probs)

Compute Inverse Mills Ratio from selection probabilities.

The IMR is: λ(p) = φ(Φ⁻¹(p)) / p

where φ is the standard normal PDF and Φ is the CDF.
"""
function compute_imr(selection_probs::Vector{T}) where {T<:Real}
    # Clip to avoid numerical issues
    p = clamp.(selection_probs, 1e-6, 1 - 1e-6)

    # Inverse CDF (probit)
    z = quantile.(Normal(), p)

    # PDF at that point
    phi_z = pdf.(Normal(), z)

    # IMR = φ(z) / Φ(z) = φ(z) / p
    imr = phi_z ./ p

    return imr
end

"""
    fit_ols(y, X)

Fit OLS regression.

# Returns
- `beta::Vector`: Coefficient estimates
- `residuals::Vector`: OLS residuals
- `sigma_hat::Float64`: Estimated error standard deviation
"""
function fit_ols(y::Vector{T}, X::Matrix{T}) where {T<:Real}
    n, k = size(X)

    # OLS: β = (X'X)⁻¹X'y
    XtX = X' * X
    Xty = X' * y

    # Use pseudoinverse for numerical stability
    beta = XtX \ Xty

    # Residuals
    residuals = y - X * beta

    # Sigma estimate (degrees of freedom adjusted)
    sigma_hat = sqrt(sum(residuals .^ 2) / (n - k))

    return beta, residuals, sigma_hat
end

"""
    compute_heckman_vcov(...)

Compute Heckman-corrected variance-covariance matrix.

The standard OLS variance is invalid because:
1. IMR is estimated (introduces uncertainty)
2. Selection equation uncertainty propagates

This implements the two-step correction from Heckman (1979).
"""
function compute_heckman_vcov(
    outcomes_selected::Vector{T},
    X_selected::Matrix{T},
    Z_full::Matrix{T},
    selected::Vector{Bool},
    gamma::Vector{T},
    beta::Vector{T},
    sigma_hat::T,
    lambda_coef::T,
    imr_selected::Vector{T}
) where {T<:Real}

    n_selected = length(outcomes_selected)
    k_beta = length(beta)

    # (X'X)⁻¹
    XtX = X_selected' * X_selected
    XtX_inv = inv(XtX)

    # Compute δ = λ(λ + γ'Z) for selected observations
    Z_selected = Z_full[selected, :]
    gamma_z_selected = Z_selected * gamma

    # δ = λ(λ + γ'Z)
    delta_selected = imr_selected .* (imr_selected .+ gamma_z_selected)

    # Residuals
    residuals = outcomes_selected - X_selected * beta

    # Standard OLS variance (uncorrected)
    sigma2 = sigma_hat^2

    # Correction for heteroskedasticity from selection
    delta_diag = Diagonal(delta_selected)
    Q = X_selected' * delta_diag * X_selected

    # Compute rho squared
    rho_sq = sigma_hat > 0 ? (lambda_coef / sigma_hat)^2 : zero(T)

    # Corrected variance-covariance
    correction = rho_sq * (XtX_inv * Q * XtX_inv)
    vcov = sigma2 * XtX_inv + sigma2 * correction

    return vcov
end

"""
    selection_bias_test(solution::HeckmanSolution; alpha=0.05)

Test for selection bias using the IMR coefficient.

Tests H₀: λ = 0 (no selection bias) vs H₁: λ ≠ 0.

# Returns
Named tuple with:
- `statistic`: t-statistic
- `pvalue`: Two-sided p-value
- `reject_null`: Whether to reject H₀
- `interpretation`: Human-readable interpretation
"""
function selection_bias_test(solution::HeckmanSolution{T}; alpha::Float64=0.05) where {T}
    lambda_coef = solution.lambda_coef
    lambda_se = solution.lambda_se

    if lambda_se <= 0 || isnan(lambda_se)
        return (
            statistic = T(NaN),
            pvalue = T(NaN),
            reject_null = false,
            interpretation = "Cannot compute test: invalid standard error"
        )
    end

    t_stat = lambda_coef / lambda_se
    pvalue = 2 * (1 - cdf(Normal(), abs(t_stat)))
    reject = pvalue < alpha

    if reject
        direction = lambda_coef > 0 ? "positive" : "negative"
        interpretation = "Significant selection bias detected (p = $(round(pvalue, digits=4))). " *
            "The $direction λ indicates correlated selection and outcome errors."
    else
        interpretation = "No significant selection bias at α = $alpha (p = $(round(pvalue, digits=4)))."
    end

    return (
        statistic = t_stat,
        pvalue = pvalue,
        reject_null = reject,
        interpretation = interpretation
    )
end
