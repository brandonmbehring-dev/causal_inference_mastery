#=
CATE (Conditional Average Treatment Effect) Types

Defines the Problem-Estimator-Solution types for heterogeneous treatment effect estimation.

References:
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
- Chernozhukov et al. (2018). "Double/debiased machine learning"
=#

# =============================================================================
# Abstract Type Hierarchy
# =============================================================================

"""
    AbstractCATEProblem{T,P} <: AbstractCausalProblem{T,P}

Abstract type for Conditional Average Treatment Effect (CATE) estimation problems.

Type parameters:
- T: Numeric type for outcomes (Float64, Float32, etc.)
- P: Parameter type (NamedTuple)
"""
abstract type AbstractCATEProblem{T,P} <: AbstractCausalProblem{T,P} end

"""
    AbstractCATEEstimator <: AbstractCausalEstimator

Abstract type for CATE estimators (meta-learners).
"""
abstract type AbstractCATEEstimator <: AbstractCausalEstimator end

"""
    AbstractCATESolution <: AbstractCausalSolution

Abstract type for CATE estimation results.
"""
abstract type AbstractCATESolution <: AbstractCausalSolution end

# =============================================================================
# Problem Type
# =============================================================================

"""
    CATEProblem{T<:Real, P<:NamedTuple} <: AbstractCATEProblem{T,P}

Problem specification for CATE estimation.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y of shape (n,)
- `treatment::Vector{Bool}`: Binary treatment indicator T ∈ {0,1}
- `covariates::Matrix{T}`: Covariate matrix X of shape (n, p)
- `parameters::P`: NamedTuple with estimation parameters

# Parameters (in NamedTuple)
- `alpha::Float64`: Significance level (default: 0.05)

# Constructor Validation
- Length consistency: outcomes, treatment, covariates must match
- No NaN/Inf values in outcomes or covariates
- Treatment must have variation (some treated, some control)
- At least 2 observations

# Example
```julia
using CausalEstimators
using Random

Random.seed!(42)
n = 200
X = randn(n, 3)
T = rand(n) .> 0.5
Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, SLearner())
```
"""
struct CATEProblem{T<:Real, P<:NamedTuple} <: AbstractCATEProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Matrix{T}
    parameters::P

    function CATEProblem(
        outcomes::Vector{T},
        treatment::Vector{Bool},
        covariates::Matrix{T},
        parameters::P
    ) where {T<:Real, P<:NamedTuple}
        # Validate inputs
        validate_cate_inputs(outcomes, treatment, covariates)
        new{T,P}(outcomes, treatment, covariates, parameters)
    end
end

# Convenience constructor with automatic type conversion
function CATEProblem(
    outcomes::AbstractVector,
    treatment::AbstractVector,
    covariates::AbstractMatrix,
    parameters::NamedTuple = (alpha=0.05,)
)
    T = promote_type(eltype(outcomes), eltype(covariates))
    outcomes_vec = convert(Vector{T}, outcomes)
    covariates_mat = convert(Matrix{T}, covariates)
    treatment_bool = convert(Vector{Bool}, treatment)

    CATEProblem(outcomes_vec, treatment_bool, covariates_mat, parameters)
end

"""
    validate_cate_inputs(outcomes, treatment, covariates)

Validate inputs for CATE estimation. Throws ArgumentError on invalid inputs.
"""
function validate_cate_inputs(outcomes, treatment, covariates)
    n = length(outcomes)

    # Length consistency
    if length(treatment) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Length mismatch.\n" *
            "Function: validate_cate_inputs\n" *
            "outcomes: $n, treatment: $(length(treatment))"
        ))
    end

    if size(covariates, 1) != n
        throw(ArgumentError(
            "CRITICAL ERROR: Covariate rows mismatch.\n" *
            "Function: validate_cate_inputs\n" *
            "outcomes: $n, covariate rows: $(size(covariates, 1))"
        ))
    end

    # Check for NaN/Inf
    if any(isnan.(outcomes)) || any(isinf.(outcomes))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in outcomes.\n" *
            "Function: validate_cate_inputs"
        ))
    end

    if any(isnan.(covariates)) || any(isinf.(covariates))
        throw(ArgumentError(
            "CRITICAL ERROR: NaN or Inf in covariates.\n" *
            "Function: validate_cate_inputs"
        ))
    end

    # Treatment variation
    n_treated = sum(treatment)
    n_control = n - n_treated

    if n_treated == 0 || n_control == 0
        throw(ArgumentError(
            "CRITICAL ERROR: No treatment variation.\n" *
            "Function: validate_cate_inputs\n" *
            "n_treated: $n_treated, n_control: $n_control"
        ))
    end

    # Minimum sample size
    if n < 4
        throw(ArgumentError(
            "CRITICAL ERROR: Insufficient sample size.\n" *
            "Function: validate_cate_inputs\n" *
            "n: $n (need at least 4)"
        ))
    end

    return nothing
end

# =============================================================================
# Solution Type
# =============================================================================

"""
    CATESolution{T<:Real, P<:NamedTuple} <: AbstractCATESolution

Results from CATE estimation.

# Fields
- `cate::Vector{T}`: Individual treatment effects τ̂(xᵢ) for each unit
- `ate::T`: Average treatment effect (mean of CATE)
- `se::T`: Standard error of ATE
- `ci_lower::T`: Lower bound of confidence interval
- `ci_upper::T`: Upper bound of confidence interval
- `method::Symbol`: Estimation method (:s_learner, :t_learner, etc.)
- `retcode::Symbol`: Return code (:Success, :Warning)
- `original_problem::CATEProblem{T,P}`: Original problem for reproducibility

# Example
```julia
solution = solve(problem, TLearner())
println("ATE: \$(solution.ate) ± \$(solution.se)")
println("CATE range: [\$(minimum(solution.cate)), \$(maximum(solution.cate))]")
```
"""
struct CATESolution{T<:Real, P<:NamedTuple} <: AbstractCATESolution
    cate::Vector{T}
    ate::T
    se::T
    ci_lower::T
    ci_upper::T
    method::Symbol
    retcode::Symbol
    original_problem::CATEProblem{T,P}
end

# =============================================================================
# Estimator Types
# =============================================================================

"""
    SLearner <: AbstractCATEEstimator

S-Learner (Single model) for CATE estimation.

Fits a single model μ(X, T) that includes treatment as a feature,
then estimates CATE by comparing predictions under T=1 vs T=0.

# Algorithm
1. Augment covariates: X_aug = [X, T]
2. Fit μ(X_aug) → Y
3. CATE(x) = μ̂([x, 1]) - μ̂([x, 0])

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Simple, uses all data
- Implicit regularization toward homogeneous effects

# Cons
- Treatment effect may be shrunk toward zero
- Less flexible for heterogeneity

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct SLearner <: AbstractCATEEstimator
    model::Symbol

    function SLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("SLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    TLearner <: AbstractCATEEstimator

T-Learner (Two models) for CATE estimation.

Fits separate models for treated and control groups.

# Algorithm
1. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
2. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
3. CATE(x) = μ̂₁(x) - μ̂₀(x)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Flexible, can capture different patterns in each group
- Intuitive interpretation

# Cons
- May have high variance with small groups
- Extrapolation issues if covariate distributions differ

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct TLearner <: AbstractCATEEstimator
    model::Symbol

    function TLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("TLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    XLearner <: AbstractCATEEstimator

X-Learner (Cross-learner) for CATE estimation.

Uses propensity-weighted combination of imputed treatment effects.
Particularly effective when treatment groups are imbalanced.

# Algorithm
1. Fit μ₀, μ₁ (T-learner step)
2. Compute imputed effects:
   D₁ = Y₁ - μ̂₀(X₁) for treated
   D₀ = μ̂₁(X₀) - Y₀ for control
3. Fit τ₁(X) → D₁, τ₀(X) → D₀
4. CATE(x) = ê(x)·τ₀(x) + (1-ê(x))·τ₁(x)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Handles imbalanced treatment groups well
- Uses propensity for adaptive weighting

# Cons
- More complex than S/T-learners
- Requires propensity estimation

# References
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
"""
struct XLearner <: AbstractCATEEstimator
    model::Symbol

    function XLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("XLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    RLearner <: AbstractCATEEstimator

R-Learner (Robinson transformation) for CATE estimation.

Uses residualization to achieve double robustness.

# Algorithm
1. Fit ê(X) = P(T=1|X) [propensity]
2. Fit m̂(X) = E[Y|X] [outcome]
3. Compute residuals: Ỹ = Y - m̂(X), T̃ = T - ê(X)
4. θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)

# Fields
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Doubly robust (consistent if either nuisance model correct)
- Orthogonal to nuisance estimation errors

# Cons
- Requires good nuisance model estimates
- May have higher variance than simpler methods

# References
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
- Robinson (1988). "Root-N-consistent semiparametric regression"
"""
struct RLearner <: AbstractCATEEstimator
    model::Symbol

    function RLearner(; model::Symbol = :ols)
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("RLearner model must be :ols or :ridge, got $model"))
        end
        new(model)
    end
end

"""
    DoubleMachineLearning <: AbstractCATEEstimator

Double Machine Learning with K-fold cross-fitting.

Eliminates regularization bias by using out-of-sample predictions
for nuisance parameters.

# Algorithm
1. Split data into K folds
2. For each fold k:
   - Train ê, m̂ on OTHER folds
   - Predict on fold k (out-of-sample)
3. Use cross-fitted residuals for R-learner estimation

# Fields
- `n_folds::Int`: Number of cross-fitting folds (default: 5)
- `model::Symbol`: Base learner (:ols, :ridge)

# Pros
- Eliminates regularization bias
- Valid asymptotic inference with ML nuisance models

# Cons
- Computationally more expensive (K× fitting)
- Requires sufficient sample size for cross-fitting

# References
- Chernozhukov et al. (2018). "Double/debiased machine learning"
"""
struct DoubleMachineLearning <: AbstractCATEEstimator
    n_folds::Int
    model::Symbol

    function DoubleMachineLearning(; n_folds::Int = 5, model::Symbol = :ols)
        if n_folds < 2
            throw(ArgumentError("n_folds must be ≥ 2, got $n_folds"))
        end
        if model ∉ (:ols, :ridge)
            throw(ArgumentError("DoubleMachineLearning model must be :ols or :ridge, got $model"))
        end
        new(n_folds, model)
    end
end

# =============================================================================
# Neural CATE Estimators (Session 152)
# =============================================================================

"""
    DragonNetConfig

Configuration for DragonNet neural architecture.

# Fields
- `hidden_layers::Tuple{Vararg{Int}}`: Hidden layer sizes for shared representation
- `head_layers::Tuple{Vararg{Int}}`: Hidden layer sizes for each output head
- `alpha::Float64`: L2 regularization strength
- `learning_rate::Float64`: Learning rate (for Flux backend)
- `max_iter::Int`: Maximum training iterations (for regression backend)
- `batch_size::Int`: Mini-batch size (for Flux backend)
- `random_state::Union{Int, Nothing}`: Random seed for reproducibility

# Example
```julia
config = DragonNetConfig(
    hidden_layers = (100, 50),
    head_layers = (50,),
    alpha = 0.001,
    max_iter = 200
)
```
"""
struct DragonNetConfig
    hidden_layers::Tuple{Vararg{Int}}
    head_layers::Tuple{Vararg{Int}}
    alpha::Float64
    learning_rate::Float64
    max_iter::Int
    batch_size::Int
    random_state::Union{Int, Nothing}

    function DragonNetConfig(;
        hidden_layers::Tuple{Vararg{Int}} = (200, 100),
        head_layers::Tuple{Vararg{Int}} = (100,),
        alpha::Float64 = 0.0001,
        learning_rate::Float64 = 0.001,
        max_iter::Int = 300,
        batch_size::Int = 64,
        random_state::Union{Int, Nothing} = 42
    )
        if isempty(hidden_layers)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid DragonNetConfig.\n" *
                "Function: DragonNetConfig\n" *
                "hidden_layers must have at least one layer"
            ))
        end
        if isempty(head_layers)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid DragonNetConfig.\n" *
                "Function: DragonNetConfig\n" *
                "head_layers must have at least one layer"
            ))
        end
        if alpha < 0
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid DragonNetConfig.\n" *
                "Function: DragonNetConfig\n" *
                "alpha must be non-negative, got $alpha"
            ))
        end
        if learning_rate <= 0
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid DragonNetConfig.\n" *
                "Function: DragonNetConfig\n" *
                "learning_rate must be positive, got $learning_rate"
            ))
        end
        if max_iter < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid DragonNetConfig.\n" *
                "Function: DragonNetConfig\n" *
                "max_iter must be >= 1, got $max_iter"
            ))
        end
        new(hidden_layers, head_layers, alpha, learning_rate, max_iter,
            batch_size, random_state)
    end
end


"""
    Dragonnet <: AbstractCATEEstimator

DragonNet estimator for CATE using shared representation learning.

DragonNet (Shi et al. 2019) uses a neural network architecture with:
1. Shared representation layers: X → φ(X)
2. Three output heads from shared representation:
   - Propensity head: P(T=1|φ(X)) - classification
   - Y(0) head: E[Y|T=0, φ(X)] - regression
   - Y(1) head: E[Y|T=1, φ(X)] - regression
3. CATE: τ(X) = Ŷ(1) - Ŷ(0)

The key insight is that shared representation learning improves CATE estimation
by forcing the network to learn features useful for both treatment assignment
(selection mechanism) and potential outcomes prediction.

# Fields
- `backend::Symbol`: Implementation backend (:regression or :flux)
- `config::DragonNetConfig`: Network configuration

# Backends
- `:regression` (default): Uses ridge regression with polynomial features to
  approximate neural network behavior. Always available.
- `:flux`: True neural network using Flux.jl (not yet implemented).

# Example
```julia
using CausalEstimators

# Create estimator with default settings
estimator = Dragonnet()

# Create estimator with custom architecture
estimator = Dragonnet(
    backend = :regression,
    config = DragonNetConfig(
        hidden_layers = (100, 50),
        head_layers = (50,),
        max_iter = 200
    )
)

# Solve CATE problem
problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, estimator)
```

# References
- Shi et al. (2019). "Adapting Neural Networks for the Estimation of
  Treatment Effects." NeurIPS 2019.
"""
struct Dragonnet <: AbstractCATEEstimator
    backend::Symbol
    config::DragonNetConfig

    function Dragonnet(;
        backend::Symbol = :regression,
        config::DragonNetConfig = DragonNetConfig()
    )
        if backend ∉ (:regression, :flux)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid Dragonnet configuration.\n" *
                "Function: Dragonnet\n" *
                "backend must be :regression or :flux, got :$backend"
            ))
        end
        new(backend, config)
    end
end

# Convenience constructor with direct config parameters
function Dragonnet(
    backend::Symbol;
    hidden_layers::Tuple{Vararg{Int}} = (200, 100),
    head_layers::Tuple{Vararg{Int}} = (100,),
    alpha::Float64 = 0.0001,
    learning_rate::Float64 = 0.001,
    max_iter::Int = 300,
    random_state::Union{Int, Nothing} = 42
)
    config = DragonNetConfig(;
        hidden_layers, head_layers, alpha, learning_rate, max_iter, random_state
    )
    Dragonnet(; backend, config)
end


# =============================================================================
# OML Estimators (Session 153)
# =============================================================================

"""
    IRMEstimator <: AbstractCATEEstimator

Interactive Regression Model (IRM) with K-fold cross-fitting.

IRM extends Double ML to the fully flexible model Y = g(T, X) + U,
providing doubly robust estimation of treatment effects.

# Algorithm
1. Split data into K folds
2. For each fold k:
   - Fit g0(X) = E[Y|T=0, X] on control units of OTHER folds
   - Fit g1(X) = E[Y|T=1, X] on treated units of OTHER folds
   - Fit m(X) = P(T=1|X) on ALL units of OTHER folds
   - Predict on fold k (out-of-sample)
3. Compute doubly robust score:
   ψ = (g1(X) - g0(X)) + T(Y-g1)/m - (1-T)(Y-g0)/(1-m) - θ
4. ATE = mean of plug-in + IPW corrections

# Fields
- `n_folds::Int`: Number of cross-fitting folds (default: 5)
- `model::Symbol`: Base learner (:ols, :ridge)
- `target::Symbol`: Target parameter (:ate or :atte)

# Comparison with DoubleMachineLearning (PLR)
- PLR: Y = θ*T + g(X) + U (treatment effect enters linearly)
- IRM: Y = g(T, X) + U (fully flexible in treatment)
- PLR requires outcome model correctly specified
- IRM is doubly robust: consistent if propensity OR outcome correct

# Double Robustness
The IRM score has the property that E[ψ] = 0 if either:
- The propensity model m(X) is correctly specified, OR
- The outcome models g0(X), g1(X) are correctly specified

This provides insurance against model misspecification.

# References
- Chernozhukov et al. (2018). "Double/debiased machine learning"
- Robins & Rotnitzky (1995). "Semiparametric efficiency"

# Example
```julia
using CausalEstimators

problem = CATEProblem(Y, T, X, (alpha=0.05,))

# Estimate ATE with IRM
solution = solve(problem, IRMEstimator())
println("ATE: \$(solution.ate) ± \$(solution.se)")

# Estimate ATTE (effect on treated)
solution_atte = solve(problem, IRMEstimator(target=:atte))
println("ATTE: \$(solution_atte.ate)")
```
"""
struct IRMEstimator <: AbstractCATEEstimator
    n_folds::Int
    model::Symbol
    target::Symbol

    function IRMEstimator(;
        n_folds::Int = 5,
        model::Symbol = :ridge,
        target::Symbol = :ate
    )
        if n_folds < 2
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid IRMEstimator configuration.\n" *
                "Function: IRMEstimator\n" *
                "n_folds must be >= 2, got $n_folds"
            ))
        end
        if model ∉ (:ols, :ridge)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid IRMEstimator configuration.\n" *
                "Function: IRMEstimator\n" *
                "model must be :ols or :ridge, got :$model"
            ))
        end
        if target ∉ (:ate, :atte)
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid IRMEstimator configuration.\n" *
                "Function: IRMEstimator\n" *
                "target must be :ate or :atte, got :$target"
            ))
        end
        new(n_folds, model, target)
    end
end
