"""
Linear Control Function estimation.

Implements the two-step control function approach:
1. First stage: D = π₀ + π₁Z + ν (regress treatment on instruments)
2. Second stage: Y = β₀ + β₁D + ρν̂ + u (include first-stage residuals)

The coefficient ρ tests for endogeneity (H0: ρ = 0 → no endogeneity).
In linear models, CF is numerically equivalent to 2SLS.
"""

using Statistics
using LinearAlgebra
using Distributions
using Random

"""
    solve(problem::CFProblem, estimator::ControlFunction)

Estimate treatment effect using linear control function approach.

# Returns
- `CFSolution`: Solution with treatment effect, endogeneity test, and diagnostics.
"""
function solve(problem::CFProblem{T}, estimator::ControlFunction) where T<:Real
    Y = problem.outcome
    D = problem.treatment
    Z = problem.instrument isa Vector ? reshape(problem.instrument, :, 1) : problem.instrument
    X = problem.covariates
    alpha = problem.alpha

    n = length(Y)
    n_instruments = size(Z, 2)
    n_controls = X === nothing ? 0 : size(X, 2)

    # First stage: D ~ Z + X
    first_stage = _first_stage_cf(D, Z, X)
    nu_hat = first_stage.residuals

    # Second stage with chosen inference method
    if estimator.inference == :bootstrap
        result = _bootstrap_cf(Y, D, Z, X, nu_hat, estimator.n_bootstrap, alpha)
    else
        result = _analytical_cf(Y, D, Z, X, nu_hat, first_stage, alpha)
    end

    return CFSolution{T}(
        result.estimate,
        result.se,
        result.se_naive,
        result.t_stat,
        result.p_value,
        result.ci_lower,
        result.ci_upper,
        result.control_coef,
        result.control_se,
        result.control_t_stat,
        result.control_p_value,
        result.endogeneity_detected,
        first_stage,
        result.r2,
        n,
        n_instruments,
        n_controls,
        estimator.inference,
        estimator.inference == :bootstrap ? estimator.n_bootstrap : nothing,
        alpha
    )
end


"""
    _first_stage_cf(D, Z, X)

First-stage regression: D ~ Z + X.
Returns residuals for use as control function.
"""
function _first_stage_cf(
    D::Vector{T},
    Z::Matrix{T},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(D)
    n_instruments = size(Z, 2)

    # Build design matrix: [1, Z, X]
    if X !== nothing
        design = hcat(ones(T, n), Z, X)
        n_controls = size(X, 2)
    else
        design = hcat(ones(T, n), Z)
        n_controls = 0
    end

    # OLS
    XtX = design' * design
    XtY = design' * D
    coefficients = XtX \ XtY

    # Fitted values and residuals
    fitted = design * coefficients
    residuals = D - fitted

    # Residual variance
    df = n - size(design, 2)
    sigma2 = sum(residuals.^2) / df

    # Standard errors (homoskedastic)
    vcov = sigma2 * inv(XtX)
    se = sqrt.(diag(vcov))

    # R-squared
    ss_res = sum(residuals.^2)
    ss_tot = sum((D .- mean(D)).^2)
    r2 = one(T) - ss_res / ss_tot

    # F-statistic for instruments (indices 2 to 1+n_instruments)
    if n_instruments == 1
        t_stat = coefficients[2] / se[2]
        f_statistic = t_stat^2
        f_pvalue = 2 * (1 - cdf(TDist(df), abs(t_stat)))
    else
        # Joint F-test for all instruments
        R = zeros(T, n_instruments, length(coefficients))
        for i in 1:n_instruments
            R[i, 1 + i] = one(T)
        end
        r = zeros(T, n_instruments)
        beta_sub = coefficients[2:1+n_instruments]
        vcov_sub = vcov[2:1+n_instruments, 2:1+n_instruments]
        Rbeta = R * coefficients - r
        wald = Rbeta' * inv(R * vcov * R') * Rbeta
        f_statistic = wald / n_instruments
        f_pvalue = 1 - cdf(FDist(n_instruments, df), f_statistic)
    end

    # Partial R-squared
    if X !== nothing
        # Residualize Z and D on X
        X_with_const = hcat(ones(T, n), X)
        Z_resid = Z - X_with_const * (X_with_const \ Z)
        D_resid = D - X_with_const * (X_with_const \ D)
        # Use first column of Z_resid for correlation
        partial_r2 = cor(vec(Z_resid[:, 1]), D_resid)^2
    else
        partial_r2 = r2
    end

    weak_iv_warning = f_statistic < 10

    return FirstStageCFResult{T}(
        coefficients,
        se,
        residuals,
        fitted,
        f_statistic,
        f_pvalue,
        partial_r2,
        r2,
        n,
        n_instruments,
        weak_iv_warning
    )
end


"""
    _second_stage_ols(Y, D, nu_hat, X)

Second-stage OLS: Y ~ D + nu_hat + X.
"""
function _second_stage_ols(
    Y::Vector{T},
    D::Vector{T},
    nu_hat::Vector{T},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(Y)

    # Build design: [1, D, nu_hat, X]
    if X !== nothing
        design = hcat(ones(T, n), D, nu_hat, X)
    else
        design = hcat(ones(T, n), D, nu_hat)
    end

    # OLS
    XtX = design' * design
    XtY = design' * Y
    coefficients = XtX \ XtY

    fitted = design * coefficients
    residuals = Y - fitted

    df = n - size(design, 2)
    sigma2 = sum(residuals.^2) / df

    # Robust (HC3) variance
    leverage = diag(design * inv(XtX) * design')
    u_adj = residuals ./ (one(T) .- leverage)
    meat = design' * Diagonal(u_adj.^2) * design
    vcov = inv(XtX) * meat * inv(XtX)
    se = sqrt.(diag(vcov))

    # R-squared
    ss_res = sum(residuals.^2)
    ss_tot = sum((Y .- mean(Y)).^2)
    r2 = one(T) - ss_res / ss_tot

    return (
        coefficients = coefficients,
        se = se,
        residuals = residuals,
        sigma2 = sigma2,
        r2 = r2,
        design = design,
        vcov = vcov,
        df = df
    )
end


"""
    _analytical_cf(Y, D, Z, X, nu_hat, first_stage, alpha)

Analytical (Murphy-Topel corrected) inference for control function.
"""
function _analytical_cf(
    Y::Vector{T},
    D::Vector{T},
    Z::Matrix{T},
    X::Union{Nothing, Matrix{T}},
    nu_hat::Vector{T},
    first_stage::FirstStageCFResult{T},
    alpha::T
) where T<:Real
    n = length(Y)

    # Second stage
    ss = _second_stage_ols(Y, D, nu_hat, X)

    beta = ss.coefficients[2]  # Treatment effect
    rho = ss.coefficients[3]   # Control coefficient
    se_naive = ss.se[2]
    rho_se_naive = ss.se[3]

    # Murphy-Topel correction
    # Simplified: correction factor based on rho^2
    sigma2_1 = sum(first_stage.residuals.^2) / (n - length(first_stage.coefficients))
    sigma2_2 = ss.sigma2
    correction_factor = one(T) + rho^2 * (sigma2_1 / sigma2_2)

    se_corrected = se_naive * sqrt(correction_factor)
    rho_se_corrected = rho_se_naive * sqrt(correction_factor)

    # Inference
    t_stat = beta / se_corrected
    p_value = 2 * (1 - cdf(TDist(ss.df), abs(t_stat)))
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = beta - z_crit * se_corrected
    ci_upper = beta + z_crit * se_corrected

    # Endogeneity test
    control_t_stat = rho / rho_se_corrected
    control_p_value = 2 * (1 - cdf(TDist(ss.df), abs(control_t_stat)))
    endogeneity_detected = control_p_value < alpha

    return (
        estimate = beta,
        se = se_corrected,
        se_naive = se_naive,
        t_stat = t_stat,
        p_value = p_value,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        control_coef = rho,
        control_se = rho_se_corrected,
        control_t_stat = control_t_stat,
        control_p_value = control_p_value,
        endogeneity_detected = endogeneity_detected,
        r2 = ss.r2
    )
end


"""
    _bootstrap_cf(Y, D, Z, X, nu_hat, n_bootstrap, alpha)

Bootstrap inference for control function.
Re-estimates both stages for each bootstrap sample.
"""
function _bootstrap_cf(
    Y::Vector{T},
    D::Vector{T},
    Z::Matrix{T},
    X::Union{Nothing, Matrix{T}},
    nu_hat::Vector{T},
    n_bootstrap::Int,
    alpha::T
) where T<:Real
    n = length(Y)

    # Point estimates from full sample
    ss = _second_stage_ols(Y, D, nu_hat, X)
    beta = ss.coefficients[2]
    rho = ss.coefficients[3]
    se_naive = ss.se[2]

    # Bootstrap
    beta_boots = T[]
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
            ss_b = _second_stage_ols(Y_b, D_b, nu_hat_b, X_b)
            push!(beta_boots, ss_b.coefficients[2])
            push!(rho_boots, ss_b.coefficients[3])
        catch
            # Skip failed bootstrap samples
            continue
        end
    end

    beta_boots = collect(beta_boots)
    rho_boots = collect(rho_boots)

    # Bootstrap SE and CI
    se_bootstrap = std(beta_boots)
    rho_se_bootstrap = std(rho_boots)

    ci_lower = quantile(beta_boots, alpha / 2)
    ci_upper = quantile(beta_boots, 1 - alpha / 2)

    # Inference
    t_stat = beta / se_bootstrap
    p_value = 2 * (1 - cdf(Normal(), abs(t_stat)))

    # Endogeneity test
    control_t_stat = rho / rho_se_bootstrap
    control_p_value = 2 * (1 - cdf(Normal(), abs(control_t_stat)))
    endogeneity_detected = control_p_value < alpha

    return (
        estimate = beta,
        se = se_bootstrap,
        se_naive = se_naive,
        t_stat = t_stat,
        p_value = p_value,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        control_coef = rho,
        control_se = rho_se_bootstrap,
        control_t_stat = control_t_stat,
        control_p_value = control_p_value,
        endogeneity_detected = endogeneity_detected,
        r2 = ss.r2
    )
end


"""
    control_function_ate(Y, D, Z; kwargs...)

Convenience function for control function estimation.

# Arguments
- `Y::Vector`: Outcome variable
- `D::Vector`: Endogenous treatment
- `Z::Union{Vector, Matrix}`: Instrument(s)
- `X::Union{Nothing, Matrix}=nothing`: Controls
- `inference::Symbol=:bootstrap`: :analytical or :bootstrap
- `n_bootstrap::Int=500`: Bootstrap iterations
- `alpha::Float64=0.05`: Significance level

# Returns
- `CFSolution`: Estimation results
"""
function control_function_ate(
    Y::Vector{T},
    D::Vector{T},
    Z::Union{Vector{T}, Matrix{T}};
    X::Union{Nothing, Matrix{T}} = nothing,
    inference::Symbol = :bootstrap,
    n_bootstrap::Int = 500,
    alpha::T = T(0.05)
) where T<:Real
    problem = CFProblem(
        outcome = Y,
        treatment = D,
        instrument = Z,
        covariates = X,
        alpha = alpha
    )
    estimator = ControlFunction(
        inference = inference,
        n_bootstrap = n_bootstrap
    )
    return solve(problem, estimator)
end
