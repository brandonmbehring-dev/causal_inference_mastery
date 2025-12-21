"""
Type definitions for Mediation Analysis.

Implements Imai et al. (2010) framework for causal mediation effects.

References:
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction
- Pearl (2001). Direct and Indirect Effects
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

using Statistics


# ============================================================================
# Abstract Types
# ============================================================================

abstract type AbstractMediationProblem{T} end
abstract type AbstractMediationEstimator end
abstract type AbstractMediationResult end


# ============================================================================
# Baron-Kenny Result
# ============================================================================

"""
    BaronKennyResult{T<:Real}

Detailed Baron-Kenny mediation decomposition.

Path coefficients from the linear mediation model:
M = alpha_0 + alpha_1 * T + e_1
Y = beta_0 + beta_1 * T + beta_2 * M + e_2

# Fields
- `alpha_1::T`: Effect of treatment on mediator (T -> M)
- `alpha_1_se::T`: Standard error of alpha_1
- `alpha_1_pvalue::T`: P-value for H0: alpha_1 = 0
- `beta_1::T`: Direct effect (T -> Y controlling for M)
- `beta_1_se::T`: Standard error of beta_1
- `beta_1_pvalue::T`: P-value for H0: beta_1 = 0
- `beta_2::T`: Effect of mediator on outcome (M -> Y)
- `beta_2_se::T`: Standard error of beta_2
- `beta_2_pvalue::T`: P-value for H0: beta_2 = 0
- `indirect_effect::T`: alpha_1 * beta_2
- `indirect_se::T`: Standard error via Sobel
- `direct_effect::T`: beta_1
- `total_effect::T`: beta_1 + alpha_1 * beta_2
- `sobel_z::T`: Sobel test statistic
- `sobel_pvalue::T`: P-value from Sobel test
- `r2_mediator_model::T`: R-squared of mediator model
- `r2_outcome_model::T`: R-squared of outcome model
- `n_obs::Int`: Number of observations
"""
struct BaronKennyResult{T<:Real} <: AbstractMediationResult
    alpha_1::T
    alpha_1_se::T
    alpha_1_pvalue::T
    beta_1::T
    beta_1_se::T
    beta_1_pvalue::T
    beta_2::T
    beta_2_se::T
    beta_2_pvalue::T
    indirect_effect::T
    indirect_se::T
    direct_effect::T
    total_effect::T
    sobel_z::T
    sobel_pvalue::T
    r2_mediator_model::T
    r2_outcome_model::T
    n_obs::Int
end


# ============================================================================
# Full Mediation Result
# ============================================================================

"""
    MediationResult{T<:Real}

Result from mediation analysis.

Contains decomposition of total effect into direct and indirect effects,
with inference via bootstrap or delta method.

# Fields
- `total_effect::T`: Total effect = NDE + NIE
- `direct_effect::T`: Natural Direct Effect (NDE)
- `indirect_effect::T`: Natural Indirect Effect (NIE)
- `proportion_mediated::T`: NIE / TE
- `te_se::T`: Standard error of total effect
- `de_se::T`: Standard error of direct effect
- `ie_se::T`: Standard error of indirect effect
- `pm_se::T`: Standard error of proportion mediated
- `te_ci::Tuple{T, T}`: 95% CI for total effect
- `de_ci::Tuple{T, T}`: 95% CI for direct effect
- `ie_ci::Tuple{T, T}`: 95% CI for indirect effect
- `pm_ci::Tuple{T, T}`: 95% CI for proportion mediated
- `te_pvalue::T`: P-value for H0: TE = 0
- `de_pvalue::T`: P-value for H0: DE = 0
- `ie_pvalue::T`: P-value for H0: IE = 0
- `method::Symbol`: :baron_kenny or :simulation
- `n_obs::Int`: Number of observations
- `n_bootstrap::Int`: Number of bootstrap replications
"""
struct MediationResult{T<:Real} <: AbstractMediationResult
    total_effect::T
    direct_effect::T
    indirect_effect::T
    proportion_mediated::T
    te_se::T
    de_se::T
    ie_se::T
    pm_se::T
    te_ci::Tuple{T, T}
    de_ci::Tuple{T, T}
    ie_ci::Tuple{T, T}
    pm_ci::Tuple{T, T}
    te_pvalue::T
    de_pvalue::T
    ie_pvalue::T
    method::Symbol
    n_obs::Int
    n_bootstrap::Int
end


# ============================================================================
# Controlled Direct Effect Result
# ============================================================================

"""
    CDEResult{T<:Real}

Controlled Direct Effect result.

CDE(m) = E[Y(1,m) - Y(0,m)] at fixed mediator value m.
Simpler to identify than NDE (no cross-world counterfactuals).

# Fields
- `cde::T`: Controlled direct effect estimate
- `se::T`: Standard error
- `ci_lower::T`: Lower CI bound
- `ci_upper::T`: Upper CI bound
- `pvalue::T`: P-value for H0: CDE = 0
- `mediator_value::T`: Value at which mediator is fixed
- `n_obs::Int`: Number of observations
"""
struct CDEResult{T<:Real} <: AbstractMediationResult
    cde::T
    se::T
    ci_lower::T
    ci_upper::T
    pvalue::T
    mediator_value::T
    n_obs::Int
end


# ============================================================================
# Sensitivity Analysis Result
# ============================================================================

"""
    SensitivityResult{T<:Real}

Sensitivity analysis result for mediation.

Assesses how estimated effects change under violations of
sequential ignorability (unmeasured confounding).

# Fields
- `rho_grid::Vector{T}`: Grid of sensitivity parameter values
- `nde_at_rho::Vector{T}`: NDE estimate at each rho value
- `nie_at_rho::Vector{T}`: NIE estimate at each rho value
- `nde_ci_lower::Vector{T}`: Lower CI for NDE at each rho
- `nde_ci_upper::Vector{T}`: Upper CI for NDE at each rho
- `nie_ci_lower::Vector{T}`: Lower CI for NIE at each rho
- `nie_ci_upper::Vector{T}`: Upper CI for NIE at each rho
- `rho_at_zero_nie::T`: Rho value at which NIE = 0
- `rho_at_zero_nde::T`: Rho value at which NDE = 0
- `original_nde::T`: NDE estimate under rho = 0
- `original_nie::T`: NIE estimate under rho = 0
- `interpretation::String`: Human-readable interpretation
"""
struct SensitivityResult{T<:Real}
    rho_grid::Vector{T}
    nde_at_rho::Vector{T}
    nie_at_rho::Vector{T}
    nde_ci_lower::Vector{T}
    nde_ci_upper::Vector{T}
    nie_ci_lower::Vector{T}
    nie_ci_upper::Vector{T}
    rho_at_zero_nie::T
    rho_at_zero_nde::T
    original_nde::T
    original_nie::T
    interpretation::String
end


# ============================================================================
# Diagnostics
# ============================================================================

"""
    MediationDiagnostics{T<:Real}

Diagnostics for mediation analysis assumptions.

# Fields
- `treatment_effect_on_mediator::T`: T -> M effect
- `treatment_effect_pvalue::T`: P-value for T -> M
- `mediator_effect_on_outcome::T`: M -> Y effect (controlling for T)
- `mediator_effect_pvalue::T`: P-value for M -> Y
- `has_mediation_path::Bool`: Whether both paths are significant
- `r2_mediator::T`: Variance in M explained by T
- `r2_outcome_full::T`: Variance in Y explained by T and M
- `r2_outcome_reduced::T`: Variance in Y explained by T only
- `n_obs::Int`: Number of observations
- `warnings::Vector{String}`: Warning messages
"""
struct MediationDiagnostics{T<:Real}
    treatment_effect_on_mediator::T
    treatment_effect_pvalue::T
    mediator_effect_on_outcome::T
    mediator_effect_pvalue::T
    has_mediation_path::Bool
    r2_mediator::T
    r2_outcome_full::T
    r2_outcome_reduced::T
    n_obs::Int
    warnings::Vector{String}
end
