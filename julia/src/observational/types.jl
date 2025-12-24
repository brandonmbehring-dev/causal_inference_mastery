#=
Observational Causal Inference Types

Defines Problem-Estimator-Solution types for observational studies with confounding.
Uses inverse probability weighting (IPW) and doubly robust (DR) methods.

Architecture:
- ObservationalProblem: Data + parameters specification
- AbstractObservationalEstimator: Algorithm interface
- IPWSolution / DRSolution: Results with diagnostics

References:
- Rosenbaum & Rubin (1983). The central role of the propensity score.
- Austin & Stuart (2015). Moving towards best practice with IPTW.
- Bang & Robins (2005). Doubly robust estimation.
=#

# =============================================================================
# Abstract Type Hierarchy
# =============================================================================

"""
    AbstractObservationalProblem{T,P} <: AbstractCausalProblem{T,P}

Abstract type for observational study problems with confounding.

All observational problems must specify:
- Outcomes
- Treatment assignment
- Covariates (confounders)
- Optional: pre-computed propensity scores
"""
abstract type AbstractObservationalProblem{T<:Real,P<:NamedTuple} <: AbstractCausalProblem{T,P} end

"""
    AbstractObservationalEstimator <: AbstractCausalEstimator

Abstract type for observational causal effect estimators.

Concrete subtypes: `ObservationalIPW`, `DoublyRobust`
"""
abstract type AbstractObservationalEstimator <: AbstractCausalEstimator end

"""
    AbstractObservationalSolution <: AbstractCausalSolution

Abstract type for observational study solutions.

All solutions include:
- Point estimate
- Standard error
- Confidence interval
- Propensity score diagnostics
- Return code
"""
abstract type AbstractObservationalSolution <: AbstractCausalSolution end


# =============================================================================
# ObservationalProblem
# =============================================================================

"""
    ObservationalProblem{T,P} <: AbstractObservationalProblem{T,P}

Specification for an observational causal inference problem.

# Mathematical Framework

In observational studies, treatment assignment depends on covariates:

    P(T=1|X) = e(X)  (propensity score)

Key assumptions for IPW identification:
1. **Unconfoundedness**: Y(0), Y(1) ⟂ T | X
2. **Positivity**: 0 < e(X) < 1 for all X in support
3. **SUTVA**: No interference, no hidden treatments

# Fields
- `outcomes::Vector{T}`: Observed outcomes Y
- `treatment::Vector{Bool}`: Treatment indicators T ∈ {0,1}
- `covariates::Matrix{T}`: Covariate matrix X (n × p)
- `propensity::Union{Nothing,Vector{T}}`: Pre-computed propensity scores (optional)
- `parameters::P`: Named tuple with estimation parameters

# Parameters NamedTuple
- `alpha::Float64`: Significance level for CI (default: 0.05)
- `trim_threshold::Float64`: Trim propensities outside (ε, 1-ε) (default: 0.01)
- `stabilize::Bool`: Use stabilized weights (default: false)

# Validation
- All arrays must have same length n
- Treatment must be binary (0/1 or false/true)
- Covariates must be n × p matrix (p ≥ 1)
- If propensity provided, must have length n and values in (0,1)
- Must have observations in both treatment groups

# Example
```julia
using CausalEstimators

# Simulated observational data with confounding
n = 500
X = randn(n, 2)
logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
e_true = 1 ./ (1 .+ exp.(-logit))
T = rand(n) .< e_true
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

# Create problem
problem = ObservationalProblem(
    Y, T, X, nothing,
    (alpha=0.05, trim_threshold=0.01, stabilize=false)
)

# Solve with IPW
solution = solve(problem, ObservationalIPW())
```

# References
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity
  score in observational studies for causal effects. Biometrika, 70(1), 41-55.
"""
struct ObservationalProblem{T<:Real,P<:NamedTuple} <: AbstractObservationalProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Matrix{T}
    propensity::Union{Nothing,Vector{T}}
    parameters::P

    function ObservationalProblem(
        outcomes::AbstractVector{T},
        treatment::AbstractVector,
        covariates::AbstractMatrix{T},
        propensity::Union{Nothing,AbstractVector{T}},
        parameters::P
    ) where {T<:Real,P<:NamedTuple}
        # Validate dimensions
        n = length(outcomes)

        if length(treatment) != n
            throw(ArgumentError(
                "Treatment length ($(length(treatment))) must match outcomes length ($n)"
            ))
        end

        if size(covariates, 1) != n
            throw(ArgumentError(
                "Covariates rows ($(size(covariates, 1))) must match outcomes length ($n)"
            ))
        end

        if size(covariates, 2) < 1
            throw(ArgumentError(
                "Covariates must have at least 1 column, got $(size(covariates, 2))"
            ))
        end

        # Convert treatment to Bool
        treatment_bool = convert(Vector{Bool}, treatment)

        # Validate treatment has both groups
        n_treated = sum(treatment_bool)
        n_control = n - n_treated

        if n_treated == 0
            throw(ArgumentError("No treated units found. Treatment must be binary with both groups."))
        end

        if n_control == 0
            throw(ArgumentError("No control units found. Treatment must be binary with both groups."))
        end

        # Validate propensity if provided
        if propensity !== nothing
            if length(propensity) != n
                throw(ArgumentError(
                    "Propensity length ($(length(propensity))) must match outcomes length ($n)"
                ))
            end

            # Check propensity bounds
            if any(propensity .<= 0) || any(propensity .>= 1)
                throw(ArgumentError(
                    "Propensity scores must be in (0, 1) exclusive. " *
                    "Got min=$(minimum(propensity)), max=$(maximum(propensity))"
                ))
            end
        end

        # Validate outcomes are finite
        if any(!isfinite, outcomes)
            throw(ArgumentError("Outcomes contain non-finite values (NaN or Inf)"))
        end

        # Validate covariates are finite
        if any(!isfinite, covariates)
            throw(ArgumentError("Covariates contain non-finite values (NaN or Inf)"))
        end

        new{T,P}(
            convert(Vector{T}, outcomes),
            treatment_bool,
            convert(Matrix{T}, covariates),
            propensity === nothing ? nothing : convert(Vector{T}, propensity),
            parameters
        )
    end
end

# Convenience constructor with default parameters
function ObservationalProblem(
    outcomes::AbstractVector{T},
    treatment::AbstractVector,
    covariates::AbstractMatrix{T};
    propensity::Union{Nothing,AbstractVector{T}} = nothing,
    alpha::Float64 = 0.05,
    trim_threshold::Float64 = 0.01,
    stabilize::Bool = false
) where {T<:Real}
    params = (alpha=alpha, trim_threshold=trim_threshold, stabilize=stabilize)
    return ObservationalProblem(outcomes, treatment, covariates, propensity, params)
end


# =============================================================================
# IPWSolution
# =============================================================================

"""
    IPWSolution{T} <: AbstractObservationalSolution

Solution from Inverse Probability Weighting (IPW) estimation.

# Mathematical Formulation

IPW ATE estimator:

    τ̂_IPW = (1/n) Σᵢ [Tᵢ Yᵢ / e(Xᵢ) - (1-Tᵢ) Yᵢ / (1-e(Xᵢ))]

Stabilized IPW (sIPW):

    τ̂_sIPW = [Σᵢ Tᵢ Yᵢ / e(Xᵢ)] / [Σᵢ Tᵢ / e(Xᵢ)] -
              [Σᵢ (1-Tᵢ) Yᵢ / (1-e(Xᵢ))] / [Σᵢ (1-Tᵢ) / (1-e(Xᵢ))]

Robust variance (sandwich estimator):

    Var(τ̂) = (1/n²) Σᵢ φᵢ²

where φᵢ is the influence function.

# Fields
- `estimate::T`: IPW estimate of ATE
- `se::T`: Robust standard error
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `p_value::T`: Two-sided p-value (H₀: ATE = 0)
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `n_trimmed::Int`: Number of units trimmed for extreme propensities
- `propensity_scores::Vector{T}`: Estimated (or provided) propensity scores
- `weights::Vector{T}`: IPW weights used
- `propensity_auc::T`: AUC of propensity model (discriminatory power)
- `propensity_mean_treated::T`: Mean propensity among treated
- `propensity_mean_control::T`: Mean propensity among control
- `stabilized::Bool`: Whether stabilized weights were used
- `retcode::Symbol`: `:Success`, `:Warning`, or `:Error`
- `original_problem::ObservationalProblem`: Original problem specification

# Return Codes
- `:Success`: Estimation succeeded without issues
- `:Warning`: Estimation succeeded but with concerns (e.g., extreme weights)
- `:Error`: Estimation failed

# Example
```julia
solution = solve(problem, ObservationalIPW())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
println("Propensity AUC: \$(solution.propensity_auc)")
```
"""
struct IPWSolution{T<:Real} <: AbstractObservationalSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    n_treated::Int
    n_control::Int
    n_trimmed::Int
    propensity_scores::Vector{T}
    weights::Vector{T}
    propensity_auc::T
    propensity_mean_treated::T
    propensity_mean_control::T
    stabilized::Bool
    retcode::Symbol
    original_problem::ObservationalProblem{T}
end

# Pretty printing
function Base.show(io::IO, sol::IPWSolution)
    println(io, "IPWSolution")
    println(io, "=" ^ 50)
    println(io, "ATE Estimate:     $(round(sol.estimate, digits=4))")
    println(io, "Std. Error:       $(round(sol.se, digits=4))")
    println(io, "95% CI:           [$(round(sol.ci_lower, digits=4)), $(round(sol.ci_upper, digits=4))]")
    println(io, "p-value:          $(round(sol.p_value, digits=4))")
    println(io, "-" ^ 50)
    println(io, "n_treated:        $(sol.n_treated)")
    println(io, "n_control:        $(sol.n_control)")
    println(io, "n_trimmed:        $(sol.n_trimmed)")
    println(io, "-" ^ 50)
    println(io, "Propensity AUC:   $(round(sol.propensity_auc, digits=4))")
    println(io, "Mean P(T|X) treated: $(round(sol.propensity_mean_treated, digits=4))")
    println(io, "Mean P(T|X) control: $(round(sol.propensity_mean_control, digits=4))")
    println(io, "Stabilized:       $(sol.stabilized)")
    println(io, "Return Code:      $(sol.retcode)")
    println(io, "=" ^ 50)
end


# =============================================================================
# DRSolution (Doubly Robust)
# =============================================================================

"""
    DRSolution{T} <: AbstractObservationalSolution

Solution from Doubly Robust (AIPW) estimation.

# Mathematical Formulation

The AIPW (Augmented IPW) estimator combines IPW with outcome regression:

    τ̂_DR = (1/n) Σᵢ [
        Tᵢ/e(Xᵢ) * (Yᵢ - μ₁(Xᵢ)) + μ₁(Xᵢ)           # Treated augmentation
      - (1-Tᵢ)/(1-e(Xᵢ)) * (Yᵢ - μ₀(Xᵢ)) - μ₀(Xᵢ)   # Control augmentation
    ]

Where:
- e(X) = P(T=1|X) is the propensity score
- μ₁(X) = E[Y|T=1, X] is the outcome model for treated
- μ₀(X) = E[Y|T=0, X] is the outcome model for control

# Double Robustness Property

The estimator is consistent if EITHER:
1. The propensity model is correctly specified, OR
2. The outcome model is correctly specified

If BOTH are correct → consistent AND efficient (lowest variance)
If BOTH are wrong → biased

# Variance Estimation

Uses the influence function approach:

    IF_i = T/e(X) * (Y - μ₁(X)) + μ₁(X)
         - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X) - τ̂_DR

    Var(τ̂_DR) = (1/n) * mean(IF_i²)

# Fields
- `estimate::T`: DR estimate of ATE
- `se::T`: Robust standard error (influence function)
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `p_value::T`: Two-sided p-value (H₀: ATE = 0)
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `n_trimmed::Int`: Number of units trimmed for extreme propensities
- `propensity_scores::Vector{T}`: Estimated (or provided) propensity scores
- `mu0_predictions::Vector{T}`: E[Y|T=0, X] for all X
- `mu1_predictions::Vector{T}`: E[Y|T=1, X] for all X
- `propensity_auc::T`: AUC of propensity model
- `mu0_r2::T`: R² for control outcome model
- `mu1_r2::T`: R² for treated outcome model
- `retcode::Symbol`: `:Success`, `:Warning`, or `:Error`
- `original_problem::ObservationalProblem`: Original problem specification

# Example
```julia
solution = solve(problem, DoublyRobust())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("Propensity AUC: \$(solution.propensity_auc)")
println("Outcome R²: μ₀=\$(solution.mu0_r2), μ₁=\$(solution.mu1_r2)")
```

# References
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data
  and causal inference models. Biometrics, 61(4), 962-973.
- Kennedy, E. H. (2016). Semiparametric theory and empirical processes in
  causal inference.
"""
struct DRSolution{T<:Real} <: AbstractObservationalSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    n_treated::Int
    n_control::Int
    n_trimmed::Int
    propensity_scores::Vector{T}
    mu0_predictions::Vector{T}
    mu1_predictions::Vector{T}
    propensity_auc::T
    mu0_r2::T
    mu1_r2::T
    retcode::Symbol
    original_problem::ObservationalProblem{T}
end

# Pretty printing
function Base.show(io::IO, sol::DRSolution)
    println(io, "DRSolution (Doubly Robust)")
    println(io, "=" ^ 50)
    println(io, "ATE Estimate:     $(round(sol.estimate, digits=4))")
    println(io, "Std. Error:       $(round(sol.se, digits=4))")
    println(io, "95% CI:           [$(round(sol.ci_lower, digits=4)), $(round(sol.ci_upper, digits=4))]")
    println(io, "p-value:          $(round(sol.p_value, digits=4))")
    println(io, "-" ^ 50)
    println(io, "n_treated:        $(sol.n_treated)")
    println(io, "n_control:        $(sol.n_control)")
    println(io, "n_trimmed:        $(sol.n_trimmed)")
    println(io, "-" ^ 50)
    println(io, "Propensity AUC:   $(round(sol.propensity_auc, digits=4))")
    println(io, "Outcome μ₀ R²:    $(round(sol.mu0_r2, digits=4))")
    println(io, "Outcome μ₁ R²:    $(round(sol.mu1_r2, digits=4))")
    println(io, "Return Code:      $(sol.retcode)")
    println(io, "=" ^ 50)
end


# =============================================================================
# TMLESolution (Targeted Maximum Likelihood Estimation)
# =============================================================================

"""
    TMLESolution{T} <: AbstractObservationalSolution

Solution from Targeted Maximum Likelihood Estimation (TMLE).

# Mathematical Formulation

TMLE improves on doubly robust estimation via an iterative targeting step:

1. Initial estimates:
   - Propensity: e(X) = P(T=1|X)
   - Outcome: Q(T,X) = E[Y|T,X]

2. Targeting step (iterate until convergence):
   - Clever covariate: H = T/e - (1-T)/(1-e)
   - Fit fluctuation: Y ~ ε*H + offset(Q)
   - Update: Q* = Q + ε*H

3. ATE: mean(Q*(1,X)) - mean(Q*(0,X))

4. Variance via Efficient Influence Function:
   EIF_i = H1*(Y - Q1*)*T - H0*(Y - Q0*)*(1-T) + Q1* - Q0* - ATE

# Double Robustness + Efficiency

- Consistent if EITHER propensity OR outcome model correct
- Achieves semiparametric efficiency bound when both correct
- Better finite-sample bias than standard DR

# Fields
- `estimate::T`: TMLE estimate of ATE
- `se::T`: Standard error (from efficient influence function)
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `p_value::T`: Two-sided p-value (H₀: ATE = 0)
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `n_trimmed::Int`: Number of units trimmed for extreme propensities
- `epsilon::T`: Total fluctuation coefficient
- `n_iterations::Int`: Number of targeting iterations
- `converged::Bool`: Whether targeting converged
- `convergence_criterion::T`: Final value of mean(H * residuals)
- `propensity_scores::Vector{T}`: Estimated (or provided) propensity scores
- `Q0_initial::Vector{T}`: Initial E[Y|T=0, X] predictions
- `Q1_initial::Vector{T}`: Initial E[Y|T=1, X] predictions
- `Q0_star::Vector{T}`: Targeted E[Y|T=0, X] predictions
- `Q1_star::Vector{T}`: Targeted E[Y|T=1, X] predictions
- `eif::Vector{T}`: Efficient influence function values
- `propensity_auc::T`: AUC of propensity model
- `mu0_r2::T`: R² for control outcome model
- `mu1_r2::T`: R² for treated outcome model
- `retcode::Symbol`: `:Success`, `:Warning`, or `:Error`
- `original_problem::ObservationalProblem`: Original problem specification

# Example
```julia
solution = solve(problem, TMLE())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("Converged: \$(solution.converged) in \$(solution.n_iterations) iterations")
```

# References
- van der Laan, M. J., & Rose, S. (2011). Targeted Learning. Springer.
- Schuler, M. S., & Rose, S. (2017). TMLE for causal inference. AJE.
"""
struct TMLESolution{T<:Real} <: AbstractObservationalSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    n_treated::Int
    n_control::Int
    n_trimmed::Int
    epsilon::T
    n_iterations::Int
    converged::Bool
    convergence_criterion::T
    propensity_scores::Vector{T}
    Q0_initial::Vector{T}
    Q1_initial::Vector{T}
    Q0_star::Vector{T}
    Q1_star::Vector{T}
    eif::Vector{T}
    propensity_auc::T
    mu0_r2::T
    mu1_r2::T
    retcode::Symbol
    original_problem::ObservationalProblem{T}
end

# Pretty printing
function Base.show(io::IO, sol::TMLESolution)
    println(io, "TMLESolution (Targeted Maximum Likelihood)")
    println(io, "=" ^ 50)
    println(io, "ATE Estimate:     $(round(sol.estimate, digits=4))")
    println(io, "Std. Error:       $(round(sol.se, digits=4))")
    println(io, "95% CI:           [$(round(sol.ci_lower, digits=4)), $(round(sol.ci_upper, digits=4))]")
    println(io, "p-value:          $(round(sol.p_value, digits=4))")
    println(io, "-" ^ 50)
    println(io, "n_treated:        $(sol.n_treated)")
    println(io, "n_control:        $(sol.n_control)")
    println(io, "n_trimmed:        $(sol.n_trimmed)")
    println(io, "-" ^ 50)
    println(io, "Epsilon:          $(round(sol.epsilon, digits=6))")
    println(io, "Iterations:       $(sol.n_iterations)")
    println(io, "Converged:        $(sol.converged)")
    println(io, "Criterion:        $(round(sol.convergence_criterion, digits=8))")
    println(io, "-" ^ 50)
    println(io, "Propensity AUC:   $(round(sol.propensity_auc, digits=4))")
    println(io, "Outcome μ₀ R²:    $(round(sol.mu0_r2, digits=4))")
    println(io, "Outcome μ₁ R²:    $(round(sol.mu1_r2, digits=4))")
    println(io, "Return Code:      $(sol.retcode)")
    println(io, "=" ^ 50)
end
