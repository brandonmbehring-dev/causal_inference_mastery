"""
Selection model types and data structures.

Implements the SciML Problem-Estimator-Solution architecture for Heckman selection.

# Types
- `HeckmanProblem`: Immutable data specification for selection models
- `HeckmanSolution`: Results from Heckman two-step estimation
- `AbstractSelectionEstimator`: Abstract base for selection estimators

# References
- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error.
  Econometrica, 47(1), 153-161.
- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data.
  MIT Press, Chapter 19.
"""

using LinearAlgebra
using Statistics
using Distributions

# Abstract type hierarchy
abstract type AbstractSelectionProblem{T,P} <: AbstractCausalProblem{T,P} end
abstract type AbstractSelectionEstimator <: AbstractCausalEstimator end
abstract type AbstractSelectionSolution <: AbstractCausalSolution end

"""
    HeckmanProblem{T<:Real}

Specification of a Heckman selection problem.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y (n×1), with NaN for unselected
- `selected::Vector{Bool}`: Selection indicator S (n×1), true = outcome observed
- `selection_covariates::Matrix{T}`: Covariates for selection equation Z (n×K_z)
- `outcome_covariates::Union{Matrix{T}, Nothing}`: Covariates for outcome equation X (n×K_x)
- `parameters::NamedTuple`: Estimation parameters (alpha, add_intercept)

# Identification
For robust identification, include an exclusion restriction: at least one
variable in `selection_covariates` that is not in `outcome_covariates`.
Without this, identification relies on the nonlinearity of the IMR.

# Examples
```julia
# With exclusion restriction (Z affects selection, X affects outcome)
problem = HeckmanProblem(
    outcomes,            # NaN for unselected
    selected,            # BitVector
    [X Z],               # Selection equation covariates
    X,                   # Outcome equation covariates (excludes Z)
    (alpha=0.05, add_intercept=true)
)

# Without exclusion (fragile identification)
problem = HeckmanProblem(outcomes, selected, X, nothing, (alpha=0.05,))
```

# References
- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error.
"""
struct HeckmanProblem{T<:Real, P<:NamedTuple} <: AbstractSelectionProblem{T,P}
    outcomes::Vector{T}
    selected::Vector{Bool}
    selection_covariates::Matrix{T}
    outcome_covariates::Union{Matrix{T}, Nothing}
    parameters::P

    function HeckmanProblem(
        outcomes::Vector{T},
        selected::Union{Vector{Bool}, BitVector},
        selection_covariates::Matrix{T},
        outcome_covariates::Union{Matrix{T}, Nothing},
        parameters::P
    ) where {T<:Real, P<:NamedTuple}
        # Convert BitVector to Vector{Bool}
        selected = collect(Bool, selected)
        n = length(outcomes)

        # Validate dimensions
        if length(selected) != n
            throw(ArgumentError(
                "selected must have same length as outcomes " *
                "(got $(length(selected)), expected $n)"
            ))
        end

        if size(selection_covariates, 1) != n
            throw(ArgumentError(
                "selection_covariates must have $n rows " *
                "(got $(size(selection_covariates, 1)))"
            ))
        end

        if !isnothing(outcome_covariates) && size(outcome_covariates, 1) != n
            throw(ArgumentError(
                "outcome_covariates must have $n rows " *
                "(got $(size(outcome_covariates, 1)))"
            ))
        end

        # Validate selection
        n_selected = sum(selected)
        if n_selected == 0
            throw(ArgumentError("No observations selected (all selected == false)"))
        end

        if n_selected == n
            throw(ArgumentError("All observations selected - no selection to correct"))
        end

        # Validate outcomes for selected observations
        outcomes_selected = outcomes[selected]
        if any(isnan, outcomes_selected) || any(isinf, outcomes_selected)
            throw(ArgumentError("outcomes contains NaN or Inf for selected observations"))
        end

        # Validate covariates
        if any(isnan, selection_covariates) || any(isinf, selection_covariates)
            throw(ArgumentError("selection_covariates contains NaN or Inf values"))
        end

        if !isnothing(outcome_covariates) &&
           (any(isnan, outcome_covariates) || any(isinf, outcome_covariates))
            throw(ArgumentError("outcome_covariates contains NaN or Inf values"))
        end

        # Validate parameters
        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        if !(0 < parameters.alpha < 1)
            throw(ArgumentError("alpha must be in (0,1), got $(parameters.alpha)"))
        end

        new{T,P}(outcomes, selected, selection_covariates, outcome_covariates, parameters)
    end
end

# Convenience constructor accepting vectors
function HeckmanProblem(
    outcomes::Vector{T},
    selected::Union{Vector{Bool}, BitVector},
    selection_covariates::Vector{T},
    outcome_covariates::Union{Vector{T}, Nothing},
    parameters::P
) where {T<:Real, P<:NamedTuple}
    sel_mat = reshape(selection_covariates, :, 1)
    out_mat = isnothing(outcome_covariates) ? nothing : reshape(outcome_covariates, :, 1)
    HeckmanProblem(outcomes, collect(selected), sel_mat, out_mat, parameters)
end

"""
    HeckmanSolution{T<:Real}

Results from Heckman two-step estimation.

# Fields
- `estimate::T`: First non-intercept coefficient from outcome equation
- `se::T`: Heckman-corrected standard error
- `ci_lower::T`: Lower bound of (1-α)% CI
- `ci_upper::T`: Upper bound of (1-α)% CI
- `rho::T`: Selection correlation parameter ρ
- `sigma::T`: Outcome error standard deviation σ
- `lambda_coef::T`: IMR coefficient λ = ρσ
- `lambda_se::T`: Standard error of λ
- `lambda_pvalue::T`: P-value for H₀: λ = 0
- `n_selected::Int`: Number of selected observations
- `n_total::Int`: Total observations
- `selection_probs::Vector{T}`: Fitted selection probabilities
- `imr::Vector{T}`: Inverse Mills Ratio values
- `gamma::Vector{T}`: Selection equation coefficients
- `beta::Vector{T}`: Outcome equation coefficients
- `vcov::Matrix{T}`: Corrected variance-covariance matrix
- `probit_converged::Bool`: Whether probit optimization converged
- `alpha::T`: Significance level used

# Selection Test
The `lambda_pvalue` tests H₀: λ = 0 (no selection bias). Rejection indicates
statistically significant selection bias, suggesting OLS would be inconsistent.

# References
- Heckman (1979). Sample Selection Bias as a Specification Error.
"""
struct HeckmanSolution{T<:Real} <: AbstractSelectionSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    rho::T
    sigma::T
    lambda_coef::T
    lambda_se::T
    lambda_pvalue::T
    n_selected::Int
    n_total::Int
    selection_probs::Vector{T}
    imr::Vector{T}
    gamma::Vector{T}
    beta::Vector{T}
    vcov::Matrix{T}
    probit_converged::Bool
    alpha::T
end

"""
    HeckmanTwoStep

Heckman two-step estimator for sample selection correction.

# Algorithm
1. Estimate probit selection equation: P(S=1|Z) via MLE
2. Compute Inverse Mills Ratio for selected observations
3. Estimate outcome equation with IMR as additional regressor
4. Compute Heckman-corrected standard errors

# Options
- `add_intercept::Bool`: Whether to add intercept to both equations (default: true)
- `max_iter::Int`: Maximum iterations for probit optimization (default: 100)
- `tol::Float64`: Convergence tolerance for probit (default: 1e-8)

# Example
```julia
problem = HeckmanProblem(y, selected, Z, X, (alpha=0.05,))
solution = solve(problem, HeckmanTwoStep())
```
"""
struct HeckmanTwoStep <: AbstractSelectionEstimator
    add_intercept::Bool
    max_iter::Int
    tol::Float64

    HeckmanTwoStep(; add_intercept::Bool=true, max_iter::Int=100, tol::Float64=1e-8) =
        new(add_intercept, max_iter, tol)
end

# Display methods
function Base.show(io::IO, problem::HeckmanProblem{T}) where {T}
    n = length(problem.outcomes)
    n_sel = sum(problem.selected)
    K_z = size(problem.selection_covariates, 2)
    K_x = isnothing(problem.outcome_covariates) ? K_z : size(problem.outcome_covariates, 2)

    println(io, "HeckmanProblem{$T}")
    println(io, "  Total observations: $n")
    println(io, "  Selected: $n_sel ($(round(100*n_sel/n, digits=1))%)")
    println(io, "  Selection covariates: $K_z")
    println(io, "  Outcome covariates: $K_x")
    println(io, "  Alpha: $(problem.parameters.alpha)")
end

function Base.show(io::IO, solution::HeckmanSolution{T}) where {T}
    println(io, "HeckmanSolution{$T}")
    println(io, "  Estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Std. Error: $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  ")
    println(io, "  Selection parameters:")
    println(io, "    ρ (correlation): $(round(solution.rho, digits=4))")
    println(io, "    λ (IMR coef): $(round(solution.lambda_coef, digits=4)) ± $(round(solution.lambda_se, digits=4))")
    println(io, "    λ p-value: $(round(solution.lambda_pvalue, digits=4))")
    println(io, "  ")
    println(io, "  Sample: $(solution.n_selected)/$(solution.n_total) selected")

    if !solution.probit_converged
        println(io, "  ⚠️  Probit optimization did not converge")
    end

    if solution.lambda_pvalue < 0.05
        println(io, "  ⚠️  Significant selection bias detected (p < 0.05)")
    end
end
