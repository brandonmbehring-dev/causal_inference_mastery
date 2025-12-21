"""
Type definitions for Control Function estimation.

The control function approach is an alternative to 2SLS that:
1. Explicitly estimates the endogeneity correlation (ρ)
2. Provides a built-in test for endogeneity
3. Extends naturally to nonlinear models

References:
- Wooldridge (2015). Control Function Methods in Applied Econometrics
- Murphy & Topel (1985). Estimation and Inference in Two-Step Models
- Rivers & Vuong (1988). Limited Information Estimators for Probit Models
"""

using Statistics
using LinearAlgebra
using Distributions

# ============================================================================
# Abstract Types
# ============================================================================

abstract type AbstractCFProblem{T} end
abstract type AbstractCFEstimator end
abstract type AbstractCFSolution end

# ============================================================================
# Problem Types
# ============================================================================

"""
    CFProblem{T<:Real}

Problem specification for Control Function estimation.

# Fields
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector{T}`: Endogenous treatment D
- `instrument::Union{Vector{T}, Matrix{T}}`: Instrument(s) Z
- `covariates::Union{Nothing, Matrix{T}}`: Optional exogenous controls X
- `alpha::T`: Significance level for inference
"""
struct CFProblem{T<:Real} <: AbstractCFProblem{T}
    outcome::Vector{T}
    treatment::Vector{T}
    instrument::Union{Vector{T}, Matrix{T}}
    covariates::Union{Nothing, Matrix{T}}
    alpha::T

    function CFProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        instrument::Union{Vector{T}, Matrix{T}},
        covariates::Union{Nothing, Matrix{T}} = nothing,
        alpha::Real = 0.05
    ) where T<:Real
        # Convert alpha to same type as outcome
        alpha = T(alpha)
        n = length(outcome)

        # Validate lengths
        length(treatment) == n || error("Treatment length mismatch: expected $n, got $(length(treatment))")

        if instrument isa Vector
            length(instrument) == n || error("Instrument length mismatch")
        else
            size(instrument, 1) == n || error("Instrument row count mismatch")
        end

        if covariates !== nothing
            size(covariates, 1) == n || error("Covariates row count mismatch")
        end

        # Validate no NaN/Inf
        !any(isnan, outcome) || error("NaN values in outcome")
        !any(isinf, outcome) || error("Inf values in outcome")
        !any(isnan, treatment) || error("NaN values in treatment")
        !any(isinf, treatment) || error("Inf values in treatment")

        # Validate treatment variation
        std(treatment) > 1e-10 || error("No variation in treatment")

        # Validate sample size
        n >= 10 || error("Insufficient sample size (n=$n). Need at least 10.")

        # Validate alpha
        zero(T) < alpha < one(T) || error("alpha must be in (0, 1)")

        new{T}(outcome, treatment, instrument, covariates, alpha)
    end
end


"""
    NonlinearCFProblem{T<:Real}

Problem specification for Nonlinear Control Function with binary outcome.

# Fields
- `outcome::Vector{T}`: Binary outcome Y in {0, 1}
- `treatment::Vector{T}`: Endogenous treatment D
- `instrument::Union{Vector{T}, Matrix{T}}`: Instrument(s) Z
- `covariates::Union{Nothing, Matrix{T}}`: Optional exogenous controls
- `model_type::Symbol`: :probit or :logit
- `alpha::T`: Significance level
"""
struct NonlinearCFProblem{T<:Real} <: AbstractCFProblem{T}
    outcome::Vector{T}
    treatment::Vector{T}
    instrument::Union{Vector{T}, Matrix{T}}
    covariates::Union{Nothing, Matrix{T}}
    model_type::Symbol
    alpha::T

    function NonlinearCFProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        instrument::Union{Vector{T}, Matrix{T}},
        covariates::Union{Nothing, Matrix{T}} = nothing,
        model_type::Symbol = :probit,
        alpha::Real = 0.05
    ) where T<:Real
        # Convert alpha to same type as outcome
        alpha = T(alpha)
        n = length(outcome)

        # Validate binary outcome
        all(y -> y == zero(T) || y == one(T), outcome) ||
            error("Outcome must be binary (0 or 1)")

        # Validate lengths
        length(treatment) == n || error("Treatment length mismatch")

        if instrument isa Vector
            length(instrument) == n || error("Instrument length mismatch")
        else
            size(instrument, 1) == n || error("Instrument row count mismatch")
        end

        if covariates !== nothing
            size(covariates, 1) == n || error("Covariates row count mismatch")
        end

        # Validate no NaN/Inf
        !any(isnan, treatment) || error("NaN values in treatment")
        !any(isinf, treatment) || error("Inf values in treatment")

        # Validate treatment variation
        std(treatment) > 1e-10 || error("No variation in treatment")

        # Validate sample size (need more for nonlinear)
        n >= 50 || error("Insufficient sample size (n=$n). Need at least 50 for nonlinear CF.")

        # Validate model type
        model_type in (:probit, :logit) || error("model_type must be :probit or :logit")

        new{T}(outcome, treatment, instrument, covariates, model_type, alpha)
    end
end


# ============================================================================
# Solution Types
# ============================================================================

"""
    FirstStageCFResult{T<:Real}

First-stage regression results for control function.

# Fields
- `coefficients::Vector{T}`: Regression coefficients
- `se::Vector{T}`: Standard errors
- `residuals::Vector{T}`: First-stage residuals (the control function)
- `fitted::Vector{T}`: Fitted values
- `f_statistic::T`: F-statistic for excluded instruments
- `f_pvalue::T`: P-value for F-test
- `partial_r2::T`: Partial R-squared
- `r2::T`: R-squared
- `n_obs::Int`: Number of observations
- `n_instruments::Int`: Number of instruments
- `weak_iv_warning::Bool`: True if F < 10
"""
struct FirstStageCFResult{T<:Real}
    coefficients::Vector{T}
    se::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    f_statistic::T
    f_pvalue::T
    partial_r2::T
    r2::T
    n_obs::Int
    n_instruments::Int
    weak_iv_warning::Bool
end


"""
    CFSolution{T<:Real}

Solution from Control Function estimation.

# Fields
- `estimate::T`: Treatment effect estimate (β₁)
- `se::T`: Corrected standard error
- `se_naive::T`: Naive OLS standard error (for comparison)
- `t_stat::T`: T-statistic
- `p_value::T`: Two-sided p-value
- `ci_lower::T`: Lower confidence bound
- `ci_upper::T`: Upper confidence bound
- `control_coef::T`: Control coefficient (ρ)
- `control_se::T`: SE of control coefficient
- `control_t_stat::T`: T-statistic for control coefficient
- `control_p_value::T`: P-value for endogeneity test
- `endogeneity_detected::Bool`: True if control coefficient significant
- `first_stage::FirstStageCFResult{T}`: First-stage results
- `second_stage_r2::T`: Second-stage R-squared
- `n_obs::Int`: Number of observations
- `n_instruments::Int`: Number of instruments
- `n_controls::Int`: Number of exogenous controls
- `inference::Symbol`: :analytical or :bootstrap
- `n_bootstrap::Union{Nothing, Int}`: Number of bootstrap iterations
- `alpha::T`: Significance level
"""
struct CFSolution{T<:Real} <: AbstractCFSolution
    estimate::T
    se::T
    se_naive::T
    t_stat::T
    p_value::T
    ci_lower::T
    ci_upper::T
    control_coef::T
    control_se::T
    control_t_stat::T
    control_p_value::T
    endogeneity_detected::Bool
    first_stage::FirstStageCFResult{T}
    second_stage_r2::T
    n_obs::Int
    n_instruments::Int
    n_controls::Int
    inference::Symbol
    n_bootstrap::Union{Nothing, Int}
    alpha::T
end


"""
    NonlinearCFSolution{T<:Real}

Solution from Nonlinear Control Function (Probit/Logit).

# Fields
- `estimate::T`: Average marginal effect
- `se::T`: Bootstrap standard error
- `ci_lower::T`: Lower confidence bound
- `ci_upper::T`: Upper confidence bound
- `p_value::T`: P-value
- `control_coef::T`: Control coefficient
- `control_se::T`: SE of control coefficient
- `control_p_value::T`: P-value for endogeneity test
- `endogeneity_detected::Bool`: True if control significant
- `first_stage::FirstStageCFResult{T}`: First-stage results
- `model_type::Symbol`: :probit or :logit
- `n_obs::Int`: Number of observations
- `n_bootstrap::Int`: Number of bootstrap iterations
- `alpha::T`: Significance level
- `converged::Bool`: Whether estimation converged
"""
struct NonlinearCFSolution{T<:Real} <: AbstractCFSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    control_coef::T
    control_se::T
    control_p_value::T
    endogeneity_detected::Bool
    first_stage::FirstStageCFResult{T}
    model_type::Symbol
    n_obs::Int
    n_bootstrap::Int
    alpha::T
    converged::Bool
end


# ============================================================================
# Estimator Types
# ============================================================================

"""
    ControlFunction <: AbstractCFEstimator

Linear Control Function estimator.

# Fields
- `inference::Symbol`: :analytical (Murphy-Topel) or :bootstrap
- `n_bootstrap::Int`: Number of bootstrap iterations
"""
struct ControlFunction <: AbstractCFEstimator
    inference::Symbol
    n_bootstrap::Int

    function ControlFunction(;
        inference::Symbol = :bootstrap,
        n_bootstrap::Int = 500
    )
        inference in (:analytical, :bootstrap) ||
            error("inference must be :analytical or :bootstrap")
        n_bootstrap >= 50 || error("n_bootstrap must be at least 50")
        new(inference, n_bootstrap)
    end
end


"""
    NonlinearCF <: AbstractCFEstimator

Nonlinear Control Function estimator (Probit/Logit).

# Fields
- `n_bootstrap::Int`: Number of bootstrap iterations
"""
struct NonlinearCF <: AbstractCFEstimator
    n_bootstrap::Int

    function NonlinearCF(; n_bootstrap::Int = 500)
        n_bootstrap >= 50 || error("n_bootstrap must be at least 50")
        new(n_bootstrap)
    end
end
