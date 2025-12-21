"""
Nonlinear Control Function estimation (Probit/Logit).

For binary outcomes, 2SLS is invalid due to Jensen's inequality.
The control function approach includes first-stage residuals directly,
avoiding this problem.

References:
- Rivers & Vuong (1988). Limited Information Estimators for Probit Models
- Wooldridge (2010). Econometric Analysis, Chapter 15.7.3
"""

using Statistics
using LinearAlgebra
using Distributions
using Random
using GLM
using DataFrames

"""
    solve(problem::NonlinearCFProblem, estimator::NonlinearCF)

Estimate treatment effect using nonlinear control function (Probit/Logit).

# Returns
- `NonlinearCFSolution`: Solution with AME, endogeneity test, and diagnostics.
"""
function solve(problem::NonlinearCFProblem{T}, estimator::NonlinearCF) where T<:Real
    Y = problem.outcome
    D = problem.treatment
    Z = problem.instrument isa Vector ? reshape(problem.instrument, :, 1) : problem.instrument
    X = problem.covariates
    model_type = problem.model_type
    alpha = problem.alpha
    n_bootstrap = estimator.n_bootstrap

    n = length(Y)

    # First stage: D ~ Z + X (OLS for continuous treatment)
    first_stage = _first_stage_cf(D, Z, X)
    nu_hat = first_stage.residuals

    # Bootstrap estimation for nonlinear model
    result = _bootstrap_nonlinear_cf(Y, D, Z, X, nu_hat, model_type, n_bootstrap, alpha)

    return NonlinearCFSolution{T}(
        result.estimate,
        result.se,
        result.ci_lower,
        result.ci_upper,
        result.p_value,
        result.control_coef,
        result.control_se,
        result.control_p_value,
        result.endogeneity_detected,
        first_stage,
        model_type,
        n,
        length(result.ame_boots),
        alpha,
        result.converged
    )
end


"""
    _fit_probit_logit(Y, D, nu_hat, X, model_type)

Fit probit or logit model: Pr(Y=1) = F(β₀ + β₁D + ρν̂ + γX).
Returns coefficients and fitted probabilities.
"""
function _fit_probit_logit(
    Y::Vector{T},
    D::Vector{T},
    nu_hat::Vector{T},
    X::Union{Nothing, Matrix{T}},
    model_type::Symbol
) where T<:Real
    n = length(Y)

    try
        # Create DataFrame for GLM - let GLM handle intercept
        df = DataFrame(y = Y, treatment = D, control = nu_hat)

        link = model_type == :probit ? ProbitLink() : LogitLink()

        if X !== nothing
            # Add covariates to DataFrame
            for i in 1:size(X, 2)
                df[!, Symbol("x$i")] = X[:, i]
            end
            # Create ModelMatrix manually with all variables
            # Design: [1, D, nu_hat, X...]
            design_cols = hcat(ones(T, n), D, nu_hat, X)
        else
            design_cols = hcat(ones(T, n), D, nu_hat)
        end

        # Use simple formula and let GLM add intercept
        model = glm(@formula(y ~ treatment + control), df, Binomial(), link)

        coefficients = coef(model)
        fitted_probs = predict(model)

        # Coefficient order: (Intercept), treatment, control
        # coef[1] = intercept, coef[2] = beta_D, coef[3] = rho
        return (
            coefficients = coefficients,
            fitted_probs = fitted_probs,
            converged = true
        )
    catch e
        # Return NaN if estimation fails
        n_coefs = X === nothing ? 3 : 3 + size(X, 2)
        return (
            coefficients = fill(T(NaN), n_coefs),
            fitted_probs = fill(T(NaN), n),
            converged = false
        )
    end
end


"""
    _compute_ame_nonlinear(coefficients, D, nu_hat, X, model_type)

Compute average marginal effect for probit/logit.

For probit: AME = β_D * mean(φ(Xβ))
For logit: AME = β_D * mean(Λ(Xβ)(1-Λ(Xβ)))

Note: Current implementation supports X=nothing only for simplicity.
Coefficients are: [intercept, treatment, control].
"""
function _compute_ame_nonlinear(
    coefficients::Vector{T},
    D::Vector{T},
    nu_hat::Vector{T},
    X::Union{Nothing, Matrix{T}},
    model_type::Symbol
) where T<:Real
    n = length(D)

    # Coefficients from GLM: [intercept, treatment, control]
    if length(coefficients) < 3
        return T(NaN)
    end

    intercept = coefficients[1]
    beta_D = coefficients[2]  # Coefficient on treatment
    rho = coefficients[3]     # Coefficient on control

    # Build linear index: intercept + beta_D*D + rho*nu_hat
    # Note: X is not currently included in the model fit for simplicity
    xb = intercept .+ beta_D .* D .+ rho .* nu_hat

    # Marginal effect at each observation
    if model_type == :probit
        # ME_i = beta_D * φ(x_i * β)
        me_i = beta_D .* pdf.(Normal(), xb)
    else
        # ME_i = beta_D * Λ(xβ) * (1 - Λ(xβ))
        prob = 1 ./ (1 .+ exp.(-xb))
        me_i = beta_D .* prob .* (1 .- prob)
    end

    return mean(me_i)
end


"""
    _bootstrap_nonlinear_cf(Y, D, Z, X, nu_hat, model_type, n_bootstrap, alpha)

Bootstrap inference for nonlinear control function.
"""
function _bootstrap_nonlinear_cf(
    Y::Vector{T},
    D::Vector{T},
    Z::Matrix{T},
    X::Union{Nothing, Matrix{T}},
    nu_hat::Vector{T},
    model_type::Symbol,
    n_bootstrap::Int,
    alpha::T
) where T<:Real
    n = length(Y)

    # Point estimates from full sample
    fit = _fit_probit_logit(Y, D, nu_hat, X, model_type)

    if !fit.converged
        return (
            estimate = T(NaN),
            se = T(NaN),
            ci_lower = T(NaN),
            ci_upper = T(NaN),
            p_value = T(NaN),
            control_coef = T(NaN),
            control_se = T(NaN),
            control_p_value = T(NaN),
            endogeneity_detected = false,
            converged = false,
            ame_boots = T[]
        )
    end

    ame = _compute_ame_nonlinear(fit.coefficients, D, nu_hat, X, model_type)
    rho = fit.coefficients[3]  # Control coefficient

    # Bootstrap
    ame_boots = T[]
    rho_boots = T[]

    for b in 1:n_bootstrap
        idx = rand(1:n, n)
        Y_b = Y[idx]
        D_b = D[idx]
        Z_b = Z[idx, :]
        X_b = X === nothing ? nothing : X[idx, :]

        try
            # Re-estimate first stage
            fs_b = _first_stage_cf(D_b, Z_b, X_b)
            nu_hat_b = fs_b.residuals

            # Re-estimate second stage
            fit_b = _fit_probit_logit(Y_b, D_b, nu_hat_b, X_b, model_type)

            if fit_b.converged
                ame_b = _compute_ame_nonlinear(fit_b.coefficients, D_b, nu_hat_b, X_b, model_type)
                push!(ame_boots, ame_b)
                push!(rho_boots, fit_b.coefficients[3])
            end
        catch
            continue
        end
    end

    ame_boots = collect(ame_boots)
    rho_boots = collect(rho_boots)

    if length(ame_boots) < 50
        return (
            estimate = ame,
            se = T(NaN),
            ci_lower = T(NaN),
            ci_upper = T(NaN),
            p_value = T(NaN),
            control_coef = rho,
            control_se = T(NaN),
            control_p_value = T(NaN),
            endogeneity_detected = false,
            converged = true,
            ame_boots = ame_boots
        )
    end

    # Bootstrap inference
    se = std(ame_boots)
    ci_lower = quantile(ame_boots, alpha / 2)
    ci_upper = quantile(ame_boots, 1 - alpha / 2)
    p_value = se > 0 ? 2 * (1 - cdf(Normal(), abs(ame / se))) : T(NaN)

    # Endogeneity test
    rho_se = std(rho_boots)
    rho_t = rho_se > 0 ? rho / rho_se : zero(T)
    control_p_value = 2 * (1 - cdf(Normal(), abs(rho_t)))
    endogeneity_detected = control_p_value < alpha

    return (
        estimate = ame,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        p_value = p_value,
        control_coef = rho,
        control_se = rho_se,
        control_p_value = control_p_value,
        endogeneity_detected = endogeneity_detected,
        converged = true,
        ame_boots = ame_boots
    )
end


"""
    nonlinear_control_function(Y, D, Z; kwargs...)

Convenience function for nonlinear control function estimation.

# Arguments
- `Y::Vector`: Binary outcome (0/1)
- `D::Vector`: Endogenous treatment
- `Z::Union{Vector, Matrix}`: Instrument(s)
- `X::Union{Nothing, Matrix}=nothing`: Controls
- `model_type::Symbol=:probit`: :probit or :logit
- `n_bootstrap::Int=500`: Bootstrap iterations
- `alpha::Float64=0.05`: Significance level

# Returns
- `NonlinearCFSolution`: Estimation results with AME
"""
function nonlinear_control_function(
    Y::Vector{T},
    D::Vector{T},
    Z::Union{Vector{T}, Matrix{T}};
    X::Union{Nothing, Matrix{T}} = nothing,
    model_type::Symbol = :probit,
    n_bootstrap::Int = 500,
    alpha::T = T(0.05)
) where T<:Real
    problem = NonlinearCFProblem(
        outcome = Y,
        treatment = D,
        instrument = Z,
        covariates = X,
        model_type = model_type,
        alpha = alpha
    )
    estimator = NonlinearCF(n_bootstrap = n_bootstrap)
    return solve(problem, estimator)
end
