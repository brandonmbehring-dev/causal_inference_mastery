"""
Mediation estimators: Baron-Kenny, NDE, NIE, CDE.

Implements causal mediation analysis following:
- Baron & Kenny (1986) linear path analysis
- Imai et al. (2010) simulation-based approach

References:
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

using Statistics
using LinearAlgebra
using Distributions
using Random


"""
    baron_kenny(outcome, treatment, mediator; covariates=nothing, alpha=0.05)

Classic Baron-Kenny mediation analysis.

Fits two linear models:
1. M = alpha_0 + alpha_1 * T + gamma'X + e_1  (mediator model)
2. Y = beta_0 + beta_1 * T + beta_2 * M + delta'X + e_2  (outcome model)

Returns path coefficients with Sobel test for indirect effect.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector`: Treatment variable T
- `mediator::Vector`: Mediator variable M
- `covariates::Union{Nothing, Matrix}`: Pre-treatment covariates X
- `alpha::Float64`: Significance level (default 0.05)

# Returns
- `BaronKennyResult`: Path coefficients, effects, and Sobel test

# Example
```julia
using Random
rng = MersenneTwister(42)
n = 500
T = Float64.(rand(rng, n) .< 0.5)
M = 0.5 .+ 0.6 .* T .+ 0.5 .* randn(rng, n)
Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ 0.5 .* randn(rng, n)
result = baron_kenny(Y, T, M)
println("Indirect: \$(result.indirect_effect)")  # ~0.48
println("Direct: \$(result.direct_effect)")      # ~0.50
```
"""
function baron_kenny(
    outcome::Vector{T},
    treatment::Vector,
    mediator::Vector;
    covariates::Union{Nothing, Matrix} = nothing,
    alpha::Float64 = 0.05
) where T<:Real
    # Input validation
    n = length(outcome)
    length(treatment) == n || error("Length mismatch: outcome ($n) != treatment ($(length(treatment)))")
    length(mediator) == n || error("Length mismatch: outcome ($n) != mediator ($(length(mediator)))")

    treatment = T.(treatment)
    mediator = T.(mediator)

    any(isnan, outcome) && error("NaN values in outcome")
    any(isnan, treatment) && error("NaN values in treatment")
    any(isnan, mediator) && error("NaN values in mediator")

    n >= 4 || error("Insufficient sample size (n=$n). Mediation requires at least 4 observations.")

    # Build design matrices
    if covariates !== nothing
        covariates = T.(covariates)
        size(covariates, 1) == n || error("Covariates row count ($(size(covariates, 1))) != n ($n)")
        X_m = hcat(ones(T, n), treatment, covariates)
        X_y = hcat(ones(T, n), treatment, mediator, covariates)
    else
        X_m = hcat(ones(T, n), treatment)
        X_y = hcat(ones(T, n), treatment, mediator)
    end

    # Step 1: Mediator model (M ~ T + X)
    beta_m = X_m \ mediator
    resid_m = mediator .- X_m * beta_m
    mse_m = sum(resid_m.^2) / (n - size(X_m, 2))
    var_beta_m = mse_m .* inv(X_m' * X_m)

    alpha_1 = beta_m[2]
    alpha_1_se = sqrt(var_beta_m[2, 2])
    alpha_1_z = alpha_1 / alpha_1_se
    alpha_1_pval = 2 * (1 - cdf(Normal(), abs(alpha_1_z)))

    # R-squared for mediator model
    ss_tot_m = sum((mediator .- mean(mediator)).^2)
    ss_res_m = sum(resid_m.^2)
    r2_m = one(T) - ss_res_m / ss_tot_m

    # Step 2: Outcome model (Y ~ T + M + X)
    beta_y = X_y \ outcome
    resid_y = outcome .- X_y * beta_y
    mse_y = sum(resid_y.^2) / (n - size(X_y, 2))
    var_beta_y = mse_y .* inv(X_y' * X_y)

    beta_1 = beta_y[2]  # Direct effect
    beta_1_se = sqrt(var_beta_y[2, 2])
    beta_1_z = beta_1 / beta_1_se
    beta_1_pval = 2 * (1 - cdf(Normal(), abs(beta_1_z)))

    beta_2 = beta_y[3]  # M -> Y effect
    beta_2_se = sqrt(var_beta_y[3, 3])
    beta_2_z = beta_2 / beta_2_se
    beta_2_pval = 2 * (1 - cdf(Normal(), abs(beta_2_z)))

    # R-squared for outcome model
    ss_tot_y = sum((outcome .- mean(outcome)).^2)
    ss_res_y = sum(resid_y.^2)
    r2_y = one(T) - ss_res_y / ss_tot_y

    # Compute effects
    indirect_effect = alpha_1 * beta_2
    direct_effect = beta_1
    total_effect = beta_1 + alpha_1 * beta_2

    # Sobel SE: sqrt(alpha_1^2 * se_beta_2^2 + beta_2^2 * se_alpha_1^2)
    sobel_se = sqrt(alpha_1^2 * beta_2_se^2 + beta_2^2 * alpha_1_se^2)
    sobel_z = sobel_se > 0 ? indirect_effect / sobel_se : zero(T)
    sobel_pval = 2 * (1 - cdf(Normal(), abs(sobel_z)))

    return BaronKennyResult{T}(
        alpha_1,
        alpha_1_se,
        alpha_1_pval,
        beta_1,
        beta_1_se,
        beta_1_pval,
        beta_2,
        beta_2_se,
        beta_2_pval,
        indirect_effect,
        sobel_se,
        direct_effect,
        total_effect,
        sobel_z,
        sobel_pval,
        r2_m,
        r2_y,
        n
    )
end


"""
    mediation_analysis(outcome, treatment, mediator; kwargs...)

Estimate causal mediation effects with bootstrap inference.

Decomposes total effect into direct and indirect components
using Baron-Kenny method with bootstrap for CIs.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector`: Treatment variable T
- `mediator::Vector`: Mediator variable M
- `covariates::Union{Nothing, Matrix}`: Pre-treatment covariates X
- `n_bootstrap::Int`: Bootstrap replications (default 1000)
- `alpha::Float64`: Significance level (default 0.05)
- `rng::Union{Nothing, AbstractRNG}`: Random number generator

# Returns
- `MediationResult`: Effects, SEs, CIs, p-values

# Notes
Identification requires Sequential Ignorability (Imai et al. 2010):
1. {Y(t,m), M(t)} ⊥ T | X  (treatment ignorability)
2. Y(t,m) ⊥ M | T, X       (mediator ignorability)
"""
function mediation_analysis(
    outcome::Vector{T},
    treatment::Vector,
    mediator::Vector;
    covariates::Union{Nothing, Matrix} = nothing,
    n_bootstrap::Int = 1000,
    alpha::Float64 = 0.05,
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real
    if rng === nothing
        rng = Random.default_rng()
    end

    n = length(outcome)
    treatment = T.(treatment)
    mediator = T.(mediator)

    # Get point estimates
    bk_result = baron_kenny(outcome, treatment, mediator; covariates=covariates, alpha=alpha)

    # Bootstrap for CIs
    boot_de = T[]
    boot_ie = T[]
    boot_te = T[]

    for _ in 1:n_bootstrap
        idx = rand(rng, 1:n, n)
        cov_boot = covariates !== nothing ? covariates[idx, :] : nothing

        try
            bk_boot = baron_kenny(
                outcome[idx], treatment[idx], mediator[idx];
                covariates=cov_boot, alpha=alpha
            )
            push!(boot_de, bk_boot.direct_effect)
            push!(boot_ie, bk_boot.indirect_effect)
            push!(boot_te, bk_boot.total_effect)
        catch
            continue
        end
    end

    # Compute bootstrap statistics
    de_se = std(boot_de, corrected=true)
    ie_se = std(boot_ie, corrected=true)
    te_se = std(boot_te, corrected=true)

    q_low = alpha / 2
    q_high = 1 - alpha / 2

    de_ci = (quantile(boot_de, q_low), quantile(boot_de, q_high))
    ie_ci = (quantile(boot_ie, q_low), quantile(boot_ie, q_high))
    te_ci = (quantile(boot_te, q_low), quantile(boot_te, q_high))

    # Proportion mediated
    te = bk_result.total_effect
    pm = abs(te) > T(1e-10) ? bk_result.indirect_effect / te : zero(T)

    boot_pm = boot_ie ./ boot_te
    boot_pm = boot_pm[isfinite.(boot_pm)]
    pm_se = length(boot_pm) > 0 ? std(boot_pm, corrected=true) : T(NaN)
    pm_ci = length(boot_pm) > 0 ? (quantile(boot_pm, q_low), quantile(boot_pm, q_high)) : (T(NaN), T(NaN))

    # P-values
    de_pval = bk_result.beta_1_pvalue
    ie_pval = bk_result.sobel_pvalue
    te_z = te_se > 0 ? te / te_se : zero(T)
    te_pval = 2 * (1 - cdf(Normal(), abs(te_z)))

    return MediationResult{T}(
        bk_result.total_effect,
        bk_result.direct_effect,
        bk_result.indirect_effect,
        pm,
        te_se,
        de_se,
        ie_se,
        pm_se,
        te_ci,
        de_ci,
        ie_ci,
        pm_ci,
        te_pval,
        de_pval,
        ie_pval,
        :baron_kenny,
        n,
        n_bootstrap
    )
end


"""
    controlled_direct_effect(outcome, treatment, mediator, mediator_value; kwargs...)

Estimate Controlled Direct Effect at fixed mediator value.

CDE(m) = E[Y(1,m) - Y(0,m)]

Unlike NDE/NIE, CDE doesn't require cross-world counterfactuals.
Simply conditions on M = m and estimates the treatment effect.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector`: Treatment variable T
- `mediator::Vector`: Mediator variable M
- `mediator_value::Real`: Value at which to fix the mediator
- `covariates::Union{Nothing, Matrix}`: Pre-treatment covariates X
- `alpha::Float64`: Significance level

# Returns
- `CDEResult`: CDE estimate, SE, CI, and p-value

# Notes
CDE is identified under weaker conditions than NDE/NIE:
- Y(t,m) ⊥ T | M=m, X
"""
function controlled_direct_effect(
    outcome::Vector{T},
    treatment::Vector,
    mediator::Vector,
    mediator_value::Real;
    covariates::Union{Nothing, Matrix} = nothing,
    alpha::Float64 = 0.05
) where T<:Real
    n = length(outcome)
    treatment = T.(treatment)
    mediator = T.(mediator)
    mediator_value = T(mediator_value)

    # Build design matrix Y ~ T + M + X
    if covariates !== nothing
        covariates = T.(covariates)
        X = hcat(ones(T, n), treatment, mediator, covariates)
    else
        X = hcat(ones(T, n), treatment, mediator)
    end

    # Fit model
    beta = X \ outcome
    resid = outcome .- X * beta
    mse = sum(resid.^2) / (n - size(X, 2))
    var_beta = mse .* inv(X' * X)

    # CDE is the treatment coefficient
    # Under linear model: CDE(m) = beta_1 (constant across m)
    cde = beta[2]
    se = sqrt(var_beta[2, 2])

    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = cde - z_crit * se
    ci_upper = cde + z_crit * se

    z_stat = se > 0 ? cde / se : zero(T)
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))

    return CDEResult{T}(
        cde,
        se,
        ci_lower,
        ci_upper,
        pvalue,
        mediator_value,
        n
    )
end


"""
    mediation_diagnostics(outcome, treatment, mediator; kwargs...)

Check mediation analysis assumptions.

# Arguments
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector`: Treatment variable T
- `mediator::Vector`: Mediator variable M
- `covariates::Union{Nothing, Matrix}`: Pre-treatment covariates X
- `alpha::Float64`: Significance level for tests

# Returns
- `MediationDiagnostics`: Diagnostic information and warnings
"""
function mediation_diagnostics(
    outcome::Vector{T},
    treatment::Vector,
    mediator::Vector;
    covariates::Union{Nothing, Matrix} = nothing,
    alpha::Float64 = 0.05
) where T<:Real
    n = length(outcome)
    treatment = T.(treatment)
    mediator = T.(mediator)

    warnings = String[]

    # Fit mediator model M ~ T + X
    if covariates !== nothing
        X_m = hcat(ones(T, n), treatment, T.(covariates))
    else
        X_m = hcat(ones(T, n), treatment)
    end

    beta_m = X_m \ mediator
    resid_m = mediator .- X_m * beta_m
    mse_m = sum(resid_m.^2) / (n - size(X_m, 2))
    var_beta_m = mse_m .* inv(X_m' * X_m)

    t_effect_on_m = beta_m[2]
    t_effect_se = sqrt(var_beta_m[2, 2])
    t_effect_z = t_effect_on_m / t_effect_se
    t_effect_pval = 2 * (1 - cdf(Normal(), abs(t_effect_z)))

    ss_tot_m = sum((mediator .- mean(mediator)).^2)
    ss_res_m = sum(resid_m.^2)
    r2_m = one(T) - ss_res_m / ss_tot_m

    # Fit full outcome model Y ~ T + M + X
    if covariates !== nothing
        X_y = hcat(ones(T, n), treatment, mediator, T.(covariates))
    else
        X_y = hcat(ones(T, n), treatment, mediator)
    end

    beta_y = X_y \ outcome
    resid_y = outcome .- X_y * beta_y
    mse_y = sum(resid_y.^2) / (n - size(X_y, 2))
    var_beta_y = mse_y .* inv(X_y' * X_y)

    m_effect_on_y = beta_y[3]
    m_effect_se = sqrt(var_beta_y[3, 3])
    m_effect_z = m_effect_on_y / m_effect_se
    m_effect_pval = 2 * (1 - cdf(Normal(), abs(m_effect_z)))

    ss_tot_y = sum((outcome .- mean(outcome)).^2)
    ss_res_y = sum(resid_y.^2)
    r2_y_full = one(T) - ss_res_y / ss_tot_y

    # Fit reduced outcome model Y ~ T + X (no mediator)
    if covariates !== nothing
        X_y_red = hcat(ones(T, n), treatment, T.(covariates))
    else
        X_y_red = hcat(ones(T, n), treatment)
    end

    beta_y_red = X_y_red \ outcome
    resid_y_red = outcome .- X_y_red * beta_y_red
    ss_res_y_red = sum(resid_y_red.^2)
    r2_y_reduced = one(T) - ss_res_y_red / ss_tot_y

    # Check for mediation path
    has_t_to_m = t_effect_pval < alpha
    has_m_to_y = m_effect_pval < alpha
    has_mediation_path = has_t_to_m && has_m_to_y

    # Generate warnings
    if !has_t_to_m
        push!(warnings, "Treatment does not significantly affect mediator (p=$(round(t_effect_pval, digits=3)))")
    end

    if !has_m_to_y
        push!(warnings, "Mediator does not significantly affect outcome (p=$(round(m_effect_pval, digits=3)))")
    end

    if r2_m < 0.01
        push!(warnings, "Treatment explains very little variance in mediator (R² = $(round(r2_m, digits=3)))")
    end

    if r2_y_full - r2_y_reduced < 0.01
        push!(warnings, "Mediator adds little explanatory power (ΔR² = $(round(r2_y_full - r2_y_reduced, digits=3)))")
    end

    return MediationDiagnostics{T}(
        t_effect_on_m,
        t_effect_pval,
        m_effect_on_y,
        m_effect_pval,
        has_mediation_path,
        r2_m,
        r2_y_full,
        r2_y_reduced,
        n,
        warnings
    )
end
