"""
Three Stages of Instrumental Variables Regression.

Provides separate problem-solution types for each stage of 2SLS estimation,
enabling detailed inspection of first-stage strength, reduced-form effects,
and second-stage structural parameters.

# Stages

1. **First Stage**: D = π₀ + π₁Z + π₂X + ν (instrument relevance)
2. **Reduced Form**: Y = γ₀ + γ₁Z + γ₂X + u (total effect of Z on Y)
3. **Second Stage**: Y = β₀ + β₁D̂ + β₂X + ε (structural causal effect)

# Key Identity (Wald Estimator)

    γ = π × β
    (Reduced form) = (First stage) × (Second stage)

# Design Pattern

Follows SciML Problem-Estimator-Solution architecture:
- Problems specify data and parameters
- `solve(problem, estimator)` returns solutions
- Solutions contain estimates and diagnostics

# References

- Angrist & Pischke (2009). *Mostly Harmless Econometrics*, Section 4.1.2
- Wooldridge (2010). *Econometric Analysis*, Chapter 5

Session 56: Ported from Python stages.py for cross-language parity.
"""

using LinearAlgebra
using Statistics
using Distributions


# =============================================================================
# Problem Types
# =============================================================================

"""
    FirstStageProblem{T,P} <: AbstractIVProblem{T,P}

Specification for first-stage regression: D = π₀ + π₁Z + π₂X + ν.

Tests instrument relevance (do instruments predict endogenous variable?).
Key diagnostic: F-statistic for H₀: π₁ = 0.

# Fields

- `treatment::Vector{T}`: Endogenous variable D (what we're predicting)
- `instruments::Matrix{T}`: Instrumental variables Z
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous controls X
- `parameters::P`: Named tuple with `:alpha`

# Examples

```julia
# Just instruments
problem = FirstStageProblem(D, Z, nothing, (alpha=0.05,))

# With covariates
problem = FirstStageProblem(D, Z, X, (alpha=0.05,))

# Solve
solution = solve(problem, OLS())
```

# Diagnostics

- F-statistic < 10 suggests weak instruments (Stock & Yogo 2005)
- Partial R² measures variance in D explained by Z controlling for X
"""
struct FirstStageProblem{T<:Real,P<:NamedTuple} <: AbstractIVProblem{T,P}
    treatment::Vector{T}
    instruments::Matrix{T}
    covariates::Union{Matrix{T},Nothing}
    parameters::P

    function FirstStageProblem(
        treatment::Vector{T},
        instruments::Matrix{T},
        covariates::Union{Matrix{T},Nothing},
        parameters::P,
    ) where {T<:Real,P<:NamedTuple}
        n = length(treatment)

        # Validate dimensions
        if size(instruments, 1) != n
            throw(ArgumentError(
                "instruments must have $n rows (got $(size(instruments, 1)))"
            ))
        end

        if !isnothing(covariates) && size(covariates, 1) != n
            throw(ArgumentError(
                "covariates must have $n rows (got $(size(covariates, 1)))"
            ))
        end

        # Validate no NaN/Inf
        if any(isnan, treatment) || any(isinf, treatment)
            throw(ArgumentError("treatment contains NaN or Inf values"))
        end

        if any(isnan, instruments) || any(isinf, instruments)
            throw(ArgumentError("instruments contains NaN or Inf values"))
        end

        if !isnothing(covariates) && (any(isnan, covariates) || any(isinf, covariates))
            throw(ArgumentError("covariates contains NaN or Inf values"))
        end

        # Validate parameters
        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        new{T,P}(treatment, instruments, covariates, parameters)
    end
end


"""
    ReducedFormProblem{T,P} <: AbstractIVProblem{T,P}

Specification for reduced-form regression: Y = γ₀ + γ₁Z + γ₂X + u.

Shows direct effect of instruments on outcome (without explicitly modeling D).
Combines first-stage and structural effects: γ = π × β.

# Fields

- `outcomes::Vector{T}`: Outcome variable Y
- `instruments::Matrix{T}`: Instrumental variables Z
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous controls X
- `parameters::P`: Named tuple with `:alpha`

# Use Cases

- Anderson-Rubin confidence intervals (robust to weak instruments)
- Intent-to-treat (ITT) effects
- Diagnostic plots (visualize Z → Y relationship)

# Examples

```julia
problem = ReducedFormProblem(Y, Z, X, (alpha=0.05,))
solution = solve(problem, OLS())
```
"""
struct ReducedFormProblem{T<:Real,P<:NamedTuple} <: AbstractIVProblem{T,P}
    outcomes::Vector{T}
    instruments::Matrix{T}
    covariates::Union{Matrix{T},Nothing}
    parameters::P

    function ReducedFormProblem(
        outcomes::Vector{T},
        instruments::Matrix{T},
        covariates::Union{Matrix{T},Nothing},
        parameters::P,
    ) where {T<:Real,P<:NamedTuple}
        n = length(outcomes)

        # Validate dimensions
        if size(instruments, 1) != n
            throw(ArgumentError(
                "instruments must have $n rows (got $(size(instruments, 1)))"
            ))
        end

        if !isnothing(covariates) && size(covariates, 1) != n
            throw(ArgumentError(
                "covariates must have $n rows (got $(size(covariates, 1)))"
            ))
        end

        # Validate no NaN/Inf
        if any(isnan, outcomes) || any(isinf, outcomes)
            throw(ArgumentError("outcomes contains NaN or Inf values"))
        end

        if any(isnan, instruments) || any(isinf, instruments)
            throw(ArgumentError("instruments contains NaN or Inf values"))
        end

        if !isnothing(covariates) && (any(isnan, covariates) || any(isinf, covariates))
            throw(ArgumentError("covariates contains NaN or Inf values"))
        end

        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        new{T,P}(outcomes, instruments, covariates, parameters)
    end
end


"""
    SecondStageProblem{T,P}

Specification for second-stage regression: Y = β₀ + β₁D̂ + β₂X + ε.

# ⚠️ WARNING: EDUCATIONAL USE ONLY

This problem type is primarily for educational purposes.
Standard errors from second-stage OLS are **INCORRECT** (biased downward)
because they treat D̂ as fixed rather than estimated.

**For production use**: Call `solve(IVProblem(...), TSLS())` which computes
correct standard errors automatically.

# Fields

- `outcomes::Vector{T}`: Outcome variable Y
- `fitted_treatment::Vector{T}`: Predicted endogenous D̂ from first stage
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous controls X
- `parameters::P`: Named tuple with `:alpha`

# Why SEs Are Wrong

The second-stage regression treats D̂ = P_Z D as if it were observed data,
but D̂ is estimated with uncertainty from the first stage. This causes:

1. Variance underestimation (SEs too small)
2. Invalid t-statistics and p-values
3. CIs with incorrect coverage

# Correct Formula

Correct 2SLS variance: V = σ² (D'P_Z D)⁻¹

where:
- σ² estimated from residuals using original D (not D̂)
- P_Z is the projection onto instrument space

# Examples

```julia
# Educational: manual 2SLS (WRONG SEs)
first_sol = solve(FirstStageProblem(D, Z, X, params), OLS())
second_prob = SecondStageProblem(Y, first_sol.fitted_values, X, params)
second_sol = solve(second_prob, OLS())  # se_naive is WRONG!

# Production: correct 2SLS
iv_prob = IVProblem(Y, D, Z, X, params)
iv_sol = solve(iv_prob, TSLS())  # se is CORRECT
```
"""
struct SecondStageProblem{T<:Real,P<:NamedTuple}
    outcomes::Vector{T}
    fitted_treatment::Vector{T}
    covariates::Union{Matrix{T},Nothing}
    parameters::P

    function SecondStageProblem(
        outcomes::Vector{T},
        fitted_treatment::Vector{T},
        covariates::Union{Matrix{T},Nothing},
        parameters::P,
    ) where {T<:Real,P<:NamedTuple}
        n = length(outcomes)

        if length(fitted_treatment) != n
            throw(ArgumentError(
                "fitted_treatment must have length $n (got $(length(fitted_treatment)))"
            ))
        end

        if !isnothing(covariates) && size(covariates, 1) != n
            throw(ArgumentError(
                "covariates must have $n rows (got $(size(covariates, 1)))"
            ))
        end

        # Validate no NaN/Inf
        if any(isnan, outcomes) || any(isinf, outcomes)
            throw(ArgumentError("outcomes contains NaN or Inf values"))
        end

        if any(isnan, fitted_treatment) || any(isinf, fitted_treatment)
            throw(ArgumentError("fitted_treatment contains NaN or Inf values"))
        end

        if !isnothing(covariates) && (any(isnan, covariates) || any(isinf, covariates))
            throw(ArgumentError("covariates contains NaN or Inf values"))
        end

        if !haskey(parameters, :alpha)
            throw(ArgumentError("parameters must include :alpha"))
        end

        new{T,P}(outcomes, fitted_treatment, covariates, parameters)
    end
end


# =============================================================================
# Solution Types
# =============================================================================

"""
    FirstStageSolution{T<:Real}

Results from first-stage regression D ~ Z + X.

# Fields

- `coef::Vector{T}`: Coefficients on [Z, X] (excluding constant)
- `se::Vector{T}`: Standard errors
- `r2::T`: Overall R² of regression
- `partial_r2::T`: Partial R² (variance in D explained by Z | X)
- `f_statistic::T`: F-statistic for joint significance of instruments
- `f_pvalue::T`: P-value for F-test
- `fitted_values::Vector{T}`: Predicted D̂ (used in second stage)
- `residuals::Vector{T}`: First-stage residuals ν̂
- `n::Int`: Sample size
- `n_instruments::Int`: Number of instruments
- `weak_iv_warning::Bool`: True if F < 10

# Interpretation

- **F-statistic > 10**: Instruments are strong (Stock & Yogo 2005)
- **Partial R² > 0.1**: Instruments explain substantial variance
- **Coefficient significance**: Instruments predict treatment
"""
struct FirstStageSolution{T<:Real}
    coef::Vector{T}
    se::Vector{T}
    r2::T
    partial_r2::T
    f_statistic::T
    f_pvalue::T
    fitted_values::Vector{T}
    residuals::Vector{T}
    n::Int
    n_instruments::Int
    weak_iv_warning::Bool
end


"""
    ReducedFormSolution{T<:Real}

Results from reduced-form regression Y ~ Z + X.

# Fields

- `coef::Vector{T}`: Coefficients on [Z, X] (excluding constant)
- `se::Vector{T}`: Standard errors
- `r2::T`: R² of regression
- `fitted_values::Vector{T}`: Predicted Y from reduced form
- `residuals::Vector{T}`: Reduced-form residuals
- `n::Int`: Sample size
- `n_instruments::Int`: Number of instruments

# Interpretation

The reduced-form coefficient γ on Z captures the intent-to-treat (ITT) effect:
total effect of instrument on outcome, regardless of treatment compliance.

Under valid instruments: γ = π × β (first stage × structural effect)
"""
struct ReducedFormSolution{T<:Real}
    coef::Vector{T}
    se::Vector{T}
    r2::T
    fitted_values::Vector{T}
    residuals::Vector{T}
    n::Int
    n_instruments::Int
end


"""
    SecondStageSolution{T<:Real}

Results from second-stage regression Y ~ D̂ + X.

# ⚠️ WARNING: se_naive IS INCORRECT

The `se_naive` field contains standard errors that are **biased downward**.
Do NOT use for hypothesis testing or confidence intervals.

# Fields

- `coef::Vector{T}`: Coefficients on [D̂, X] (excluding constant)
- `se_naive::Vector{T}`: ⚠️ INCORRECT standard errors (for comparison only)
- `r2::T`: R² of regression
- `fitted_values::Vector{T}`: Predicted Y
- `residuals::Vector{T}`: Second-stage residuals
- `n::Int`: Sample size

# Why se_naive Is Wrong

OLS standard errors assume regressors are fixed. But D̂ is estimated,
so naive SEs understate uncertainty, leading to:
- Over-rejection of null hypothesis
- Confidence intervals that are too narrow
- Invalid inference

# Getting Correct SEs

```julia
# Use TSLS solver on full IVProblem
problem = IVProblem(Y, D, Z, X, (alpha=0.05,))
solution = solve(problem, TSLS())
correct_se = solution.se
```
"""
struct SecondStageSolution{T<:Real}
    coef::Vector{T}
    se_naive::Vector{T}  # Explicitly marked as WRONG
    r2::T
    fitted_values::Vector{T}
    residuals::Vector{T}
    n::Int
end


# =============================================================================
# OLS Estimator for Stages
# =============================================================================

"""
    OLS <: AbstractIVEstimator

Ordinary Least Squares estimator for stage regressions.

Used internally by stage solvers. For full IV estimation, use `TSLS`, `LIML`, etc.
"""
struct OLS <: AbstractIVEstimator end


# =============================================================================
# Solve Methods
# =============================================================================

"""
    solve(problem::FirstStageProblem, ::OLS) → FirstStageSolution

Fit first-stage regression D ~ Z + X via OLS.

# Returns

`FirstStageSolution` containing:
- Coefficients and standard errors
- F-statistic for instrument relevance
- Partial R² controlling for covariates
- Fitted values D̂ for second stage

# Algorithm

1. Construct design matrix [1, Z, X]
2. Fit OLS: D = [1, Z, X] β + ν
3. Compute F-statistic for H₀: β_Z = 0
4. Compute partial R² = R²(Z,X) - R²(X) if covariates present

# Examples

```julia
problem = FirstStageProblem(D, Z, X, (alpha=0.05,))
solution = solve(problem, OLS())

println("F-statistic: \$(solution.f_statistic)")
println("Partial R²: \$(solution.partial_r2)")

if solution.weak_iv_warning
    @warn "Weak instruments detected (F < 10)"
end
```
"""
function solve(problem::FirstStageProblem{T}, ::OLS) where {T<:Real}
    D = problem.treatment
    Z = problem.instruments
    X = problem.covariates
    alpha = problem.parameters.alpha

    n = length(D)
    q = size(Z, 2)  # Number of instruments

    # Construct design matrix with constant
    if isnothing(X)
        ZX = hcat(ones(T, n), Z)
        k = 1 + q  # constant + instruments
    else
        ZX = hcat(ones(T, n), Z, X)
        k = 1 + q + size(X, 2)  # constant + instruments + covariates
    end

    # OLS: β = (ZX'ZX)⁻¹ ZX'D
    ZX_ZX = ZX' * ZX
    ZX_ZX_inv = inv(ZX_ZX)
    beta = ZX_ZX_inv * (ZX' * D)

    # Fitted values and residuals
    fitted_values = ZX * beta
    residuals = D - fitted_values

    # Residual variance
    sigma2 = sum(residuals .^ 2) / (n - k)

    # Standard errors
    vcov = sigma2 * ZX_ZX_inv
    se = sqrt.(diag(vcov))

    # R²
    SS_tot = sum((D .- mean(D)) .^ 2)
    SS_res = sum(residuals .^ 2)
    r2 = 1 - SS_res / SS_tot

    # F-statistic for instruments (H₀: all instrument coefficients = 0)
    # Coefficients on Z are indices 2:(q+1) in beta
    R = zeros(T, q, k)
    for i in 1:q
        R[i, 1+i] = one(T)  # Selects Z coefficients
    end
    r = zeros(T, q)

    # Wald test: F = (Rβ - r)' [R(X'X)⁻¹R'σ²]⁻¹ (Rβ - r) / q
    R_beta = R * beta - r
    var_R_beta = sigma2 * R * ZX_ZX_inv * R'
    f_statistic = (R_beta' * inv(var_R_beta) * R_beta) / q

    # F-distribution p-value
    f_dist = FDist(q, n - k)
    f_pvalue = 1 - cdf(f_dist, f_statistic)

    # Partial R² (variance in D explained by Z controlling for X)
    if isnothing(X)
        # No controls: partial R² = R²
        partial_r2 = r2
    else
        # Fit restricted model with only [1, X]
        X_only = hcat(ones(T, n), X)
        beta_restricted = (X_only' * X_only) \ (X_only' * D)
        fitted_restricted = X_only * beta_restricted
        resid_restricted = D - fitted_restricted
        SS_res_restricted = sum(resid_restricted .^ 2)
        r2_restricted = 1 - SS_res_restricted / SS_tot

        partial_r2 = r2 - r2_restricted
    end

    # Weak IV warning
    weak_iv_warning = f_statistic < 10

    return FirstStageSolution{T}(
        beta[2:end],  # Exclude constant
        se[2:end],
        r2,
        partial_r2,
        f_statistic,
        f_pvalue,
        fitted_values,
        residuals,
        n,
        q,
        weak_iv_warning,
    )
end


"""
    solve(problem::ReducedFormProblem, ::OLS) → ReducedFormSolution

Fit reduced-form regression Y ~ Z + X via OLS.

# Returns

`ReducedFormSolution` containing:
- Coefficients and standard errors on [Z, X]
- R² and fitted values
- Residuals for diagnostics

# Examples

```julia
problem = ReducedFormProblem(Y, Z, X, (alpha=0.05,))
solution = solve(problem, OLS())

# Wald estimator identity check
first_sol = solve(FirstStageProblem(D, Z, X, params), OLS())
rf_sol = solve(ReducedFormProblem(Y, Z, X, params), OLS())

# γ ≈ π × β (with single instrument)
gamma = rf_sol.coef[1]
pi = first_sol.coef[1]
# beta = gamma / pi  (Wald estimator)
```
"""
function solve(problem::ReducedFormProblem{T}, ::OLS) where {T<:Real}
    Y = problem.outcomes
    Z = problem.instruments
    X = problem.covariates

    n = length(Y)
    q = size(Z, 2)

    # Construct design matrix with constant
    if isnothing(X)
        ZX = hcat(ones(T, n), Z)
        k = 1 + q
    else
        ZX = hcat(ones(T, n), Z, X)
        k = 1 + q + size(X, 2)
    end

    # OLS
    ZX_ZX = ZX' * ZX
    ZX_ZX_inv = inv(ZX_ZX)
    beta = ZX_ZX_inv * (ZX' * Y)

    # Fitted values and residuals
    fitted_values = ZX * beta
    residuals = Y - fitted_values

    # Residual variance and SEs
    sigma2 = sum(residuals .^ 2) / (n - k)
    vcov = sigma2 * ZX_ZX_inv
    se = sqrt.(diag(vcov))

    # R²
    SS_tot = sum((Y .- mean(Y)) .^ 2)
    SS_res = sum(residuals .^ 2)
    r2 = 1 - SS_res / SS_tot

    return ReducedFormSolution{T}(
        beta[2:end],  # Exclude constant
        se[2:end],
        r2,
        fitted_values,
        residuals,
        n,
        q,
    )
end


"""
    solve(problem::SecondStageProblem, ::OLS) → SecondStageSolution

Fit second-stage regression Y ~ D̂ + X via OLS.

# ⚠️ WARNING: Returns INCORRECT Standard Errors

This function computes naive OLS standard errors that treat D̂ as fixed.
These SEs are **biased downward** and should NOT be used for inference.

# Why This Exists

This function is provided for **educational purposes**:
1. Understand the two-stage structure of 2SLS
2. Compare naive vs correct SEs
3. Demonstrate why proper IV estimation is needed

# Getting Correct SEs

```julia
# WRONG: naive second-stage SEs
second_sol = solve(SecondStageProblem(Y, D_hat, X, params), OLS())
wrong_se = second_sol.se_naive  # TOO SMALL

# CORRECT: full 2SLS with proper variance
iv_sol = solve(IVProblem(Y, D, Z, X, params), TSLS())
correct_se = iv_sol.se  # CORRECT
```

# Examples

```julia
# Educational: manual two-stage estimation
first_sol = solve(FirstStageProblem(D, Z, X, params), OLS())
D_hat = first_sol.fitted_values

second_prob = SecondStageProblem(Y, D_hat, X, params)
second_sol = solve(second_prob, OLS())

println("Coefficient: \$(second_sol.coef[1])")
println("Naive SE (WRONG): \$(second_sol.se_naive[1])")

@warn "Do NOT use se_naive for inference!"
```
"""
function solve(problem::SecondStageProblem{T}, ::OLS) where {T<:Real}
    Y = problem.outcomes
    D_hat = problem.fitted_treatment
    X = problem.covariates

    n = length(Y)

    # Construct design matrix with constant
    if isnothing(X)
        DX = hcat(ones(T, n), D_hat)
        k = 2  # constant + fitted treatment
    else
        DX = hcat(ones(T, n), D_hat, X)
        k = 2 + size(X, 2)
    end

    # OLS
    DX_DX = DX' * DX
    DX_DX_inv = inv(DX_DX)
    beta = DX_DX_inv * (DX' * Y)

    # Fitted values and residuals
    fitted_values = DX * beta
    residuals = Y - fitted_values

    # NAIVE standard errors (INCORRECT - treats D̂ as fixed)
    sigma2 = sum(residuals .^ 2) / (n - k)
    vcov_naive = sigma2 * DX_DX_inv
    se_naive = sqrt.(diag(vcov_naive))

    # R²
    SS_tot = sum((Y .- mean(Y)) .^ 2)
    SS_res = sum(residuals .^ 2)
    r2 = 1 - SS_res / SS_tot

    # Issue warning about incorrect SEs
    @warn "SecondStageSolution.se_naive contains INCORRECT standard errors. " *
          "Do NOT use for hypothesis testing. " *
          "Use solve(IVProblem(...), TSLS()) for correct inference."

    return SecondStageSolution{T}(
        beta[2:end],  # Exclude constant
        se_naive[2:end],
        r2,
        fitted_values,
        residuals,
        n,
    )
end


# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, sol::FirstStageSolution{T}) where {T}
    println(io, "FirstStageSolution{$T}")
    println(io, "  Observations: $(sol.n)")
    println(io, "  Instruments: $(sol.n_instruments)")
    println(io, "  R²: $(round(sol.r2, digits=4))")
    println(io, "  Partial R²: $(round(sol.partial_r2, digits=4))")
    println(io, "  F-statistic: $(round(sol.f_statistic, digits=2))")
    println(io, "  F p-value: $(round(sol.f_pvalue, digits=4))")

    if sol.weak_iv_warning
        println(io, "  ⚠️  Weak instruments detected (F < 10)")
    end
end

function Base.show(io::IO, sol::ReducedFormSolution{T}) where {T}
    println(io, "ReducedFormSolution{$T}")
    println(io, "  Observations: $(sol.n)")
    println(io, "  Instruments: $(sol.n_instruments)")
    println(io, "  R²: $(round(sol.r2, digits=4))")
end

function Base.show(io::IO, sol::SecondStageSolution{T}) where {T}
    println(io, "SecondStageSolution{$T}")
    println(io, "  Observations: $(sol.n)")
    println(io, "  R²: $(round(sol.r2, digits=4))")
    println(io, "  ⚠️  WARNING: se_naive contains INCORRECT standard errors!")
end
