"""
Instrumental Variables (IV) types and data structures.

Implements the SciML Problem-Estimator-Solution architecture for IV estimation.

# Types
- `IVProblem`: Immutable data specification for IV models
- `IVSolution`: Results from IV estimation with diagnostics
- `AbstractIVEstimator`: Abstract base for IV estimators (2SLS, LIML, GMM, etc.)

# References
- Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*.
  Princeton University Press.
- Stock, J. H., & Watson, M. W. (2015). *Introduction to Econometrics* (3rd ed.).
  Pearson.
"""

using LinearAlgebra
using Statistics

# Abstract type hierarchy
abstract type AbstractIVProblem{T,P} <: AbstractCausalProblem{T,P} end
abstract type AbstractIVEstimator <: AbstractCausalEstimator end
abstract type AbstractIVSolution <: AbstractCausalSolution end

"""
    IVProblem{T<:Real}

Specification of an instrumental variables problem.

# Fields
- `outcomes::Vector{T}`: Outcome variable Y (n×1)
- `treatment::Vector{T}`: Endogenous treatment variable D (n×1)
- `instruments::Matrix{T}`: Instrumental variables Z (n×K)
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous covariates X (n×p) or nothing
- `parameters::NamedTuple`: Estimation parameters (alpha for CI, etc.)

# Validation
- All arrays must have same number of observations
- No NaN or Inf values allowed
- At least one instrument required (K ≥ 1)
- For identification: K ≥ L (number of instruments ≥ number of endogenous variables)

# Examples
```julia
# Just-identified model (1 instrument, 1 endogenous variable)
problem = IVProblem(y, d, z, nothing, (alpha=0.05,))

# Overidentified model (2 instruments, 1 endogenous variable)
Z = hcat(z1, z2)
problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

# With exogenous covariates
problem = IVProblem(y, d, Z, X, (alpha=0.05,))
```

# Identification Requirements
For IV estimation to be identified, we need:
1. **Instrument Relevance**: Cov(Z, D) ≠ 0 (can be tested via F-statistic)
2. **Instrument Exogeneity**: Cov(Z, ε) = 0 (untestable, maintained assumption)
3. **Rank Condition**: rank(E[Z'D]) = L (number of endogenous variables)
4. **Order Condition**: K ≥ L (number of instruments ≥ number of endogenous)

# References
- Angrist & Krueger (2001). "Instrumental Variables and the Search for
  Identification." *Journal of Economic Perspectives*, 15(4), 69-85.
"""
struct IVProblem{T<:Real, P<:NamedTuple} <: AbstractIVProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{T}
    instruments::Matrix{T}
    covariates::Union{Matrix{T}, Nothing}
    parameters::P

    function IVProblem(
        outcomes::Vector{T},
        treatment::Vector{T},
        instruments::Matrix{T},
        covariates::Union{Matrix{T}, Nothing},
        parameters::P
    ) where {T<:Real, P<:NamedTuple}
        # Validate dimensions
        n = length(outcomes)

        if length(treatment) != n
            throw(ArgumentError(
                "treatment must have same length as outcomes " *
                "(got $(length(treatment)), expected $n)"
            ))
        end

        if size(instruments, 1) != n
            throw(ArgumentError(
                "instruments must have $n rows " *
                "(got $(size(instruments, 1)))"
            ))
        end

        if !isnothing(covariates) && size(covariates, 1) != n
            throw(ArgumentError(
                "covariates must have $n rows " *
                "(got $(size(covariates, 1)))"
            ))
        end

        # Validate no NaN or Inf
        if any(isnan, outcomes) || any(isinf, outcomes)
            throw(ArgumentError("outcomes contains NaN or Inf values"))
        end

        if any(isnan, treatment) || any(isinf, treatment)
            throw(ArgumentError("treatment contains NaN or Inf values"))
        end

        if any(isnan, instruments) || any(isinf, instruments)
            throw(ArgumentError("instruments contains NaN or Inf values"))
        end

        if !isnothing(covariates) && (any(isnan, covariates) || any(isinf, covariates))
            throw(ArgumentError("covariates contains NaN or Inf values"))
        end

        # Validate at least one instrument
        K = size(instruments, 2)
        if K < 1
            throw(ArgumentError("At least one instrument required (got K=$K)"))
        end

        # Check order condition: K ≥ L (assuming L=1 for now)
        # TODO: Generalize to multiple endogenous variables
        L = 1  # Number of endogenous variables
        if K < L
            throw(ArgumentError(
                "Order condition violated: need K ≥ L instruments " *
                "(got K=$K, L=$L)"
            ))
        end

        # Validate parameters
        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        if !(0 < parameters.alpha < 1)
            throw(ArgumentError("alpha must be in (0,1), got $(parameters.alpha)"))
        end

        new{T,P}(outcomes, treatment, instruments, covariates, parameters)
    end
end

"""
    IVSolution{T<:Real}

Results from instrumental variables estimation.

# Fields
- `estimate::T`: Point estimate of treatment effect β̂
- `se::T`: Standard error
- `ci_lower::T`: Lower bound of (1-α)% confidence interval
- `ci_upper::T`: Upper bound of (1-α)% confidence interval
- `p_value::T`: Two-sided p-value for H₀: β = 0
- `n::Int`: Sample size
- `n_instruments::Int`: Number of instruments (K)
- `n_covariates::Int`: Number of exogenous covariates (p)
- `first_stage_fstat::T`: F-statistic from first stage regression
- `overid_pvalue::Union{T, Nothing}`: Overidentification test p-value (if K > L)
- `weak_iv_warning::Bool`: True if first-stage F-statistic < 10
- `estimator_name::String`: Name of estimator used (2SLS, LIML, GMM, etc.)
- `alpha::T`: Significance level used for CI
- `diagnostics::NamedTuple`: Additional diagnostic information

# Diagnostics Fields
The `diagnostics` NamedTuple may include:
- `first_stage_coef`: First-stage coefficients
- `first_stage_se`: First-stage standard errors
- `reduced_form_coef`: Reduced-form coefficients
- `cragg_donald_stat`: Cragg-Donald minimum eigenvalue statistic
- `sargan_pvalue`: Sargan overidentification test p-value
- `hansen_j_pvalue`: Hansen J overidentification test p-value

# Weak IV Warning
If `weak_iv_warning` is true, the first-stage F-statistic is below the rule-of-thumb
threshold of 10 (Stock & Yogo 2005). This suggests weak instruments, and standard
inference may be invalid. Consider:
1. Using more/better instruments
2. Weak IV robust inference (Anderson-Rubin, CLR)
3. Reporting LIML instead of 2SLS

# References
- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV
  Regression." In *Identification and Inference for Econometric Models*,
  Cambridge University Press, 80-108.
"""
struct IVSolution{T<:Real} <: AbstractIVSolution
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    p_value::T
    n::Int
    n_instruments::Int
    n_covariates::Int
    first_stage_fstat::T
    overid_pvalue::Union{T, Nothing}
    weak_iv_warning::Bool
    estimator_name::String
    alpha::T
    diagnostics::NamedTuple
end

"""
Abstract base type for IV estimators.

Concrete subtypes:
- `TSLS`: Two-stage least squares
- `LIML`: Limited information maximum likelihood
- `GMM`: Generalized method of moments
- `AndersonRubin`: Weak IV robust test
- `ConditionalLR`: Moreira's conditional likelihood ratio test
"""
abstract type AbstractIVEstimator <: AbstractCausalEstimator end

# Base display methods
function Base.show(io::IO, problem::IVProblem{T}) where {T}
    n = length(problem.outcomes)
    K = size(problem.instruments, 2)
    p = isnothing(problem.covariates) ? 0 : size(problem.covariates, 2)

    println(io, "IVProblem{$T}")
    println(io, "  Observations: $n")
    println(io, "  Instruments: $K")
    println(io, "  Covariates: $p")
    println(io, "  Alpha: $(problem.parameters.alpha)")
end

function Base.show(io::IO, solution::IVSolution{T}) where {T}
    println(io, "IVSolution{$T} ($(solution.estimator_name))")
    println(io, "  Estimate: $(round(solution.estimate, digits=4))")
    println(io, "  Std. Error: $(round(solution.se, digits=4))")
    println(io, "  95% CI: [$(round(solution.ci_lower, digits=4)), $(round(solution.ci_upper, digits=4))]")
    println(io, "  p-value: $(round(solution.p_value, digits=4))")
    println(io, "  First-stage F: $(round(solution.first_stage_fstat, digits=2))")

    if solution.weak_iv_warning
        println(io, "  ⚠️  Weak instruments detected (F < 10)")
    end

    if !isnothing(solution.overid_pvalue)
        println(io, "  Overid p-value: $(round(solution.overid_pvalue, digits=4))")
    end
end
