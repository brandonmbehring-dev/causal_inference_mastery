"""
Weak IV diagnostics for instrumental variables estimation.

Implements first-stage diagnostics to detect weak instruments:
- First-stage F-statistic (rule of thumb: F > 10)
- Cragg-Donald minimum eigenvalue statistic
- Stock-Yogo critical values for weak IV tests

# References
- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV
  Regression." In *Identification and Inference for Econometric Models*,
  Cambridge University Press, 80-108.
- Cragg, J. G., & Donald, S. G. (1993). "Testing Identifiability and Specification
  in Instrumental Variable Models." *Econometric Theory*, 9(2), 222-240.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    first_stage_fstat(treatment, instruments, covariates=nothing)

Compute first-stage F-statistic for instrument strength.

# Arguments
- `treatment::Vector{T}`: Endogenous variable D (n×1)
- `instruments::Matrix{T}`: Instrumental variables Z (n×K)
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous covariates X (n×p) or nothing

# Returns
- `fstat::T`: First-stage F-statistic
- `p_value::T`: P-value for H₀: instruments have no explanatory power

# Method

First stage regression:
```
D = Z'π + X'γ + ε
```

F-statistic tests H₀: π = 0 (instruments irrelevant):
```
F = (R² / K) / ((1 - R²) / (n - K - p - 1))
```

where:
- K = number of instruments
- p = number of exogenous covariates (0 if none)
- n = sample size

# Interpretation

**Rule of Thumb** (Stock & Yogo 2005):
- F > 10: Strong instruments (safe for 2SLS)
- F < 10: Weak instruments (2SLS biased toward OLS, use weak IV robust methods)

**Critical values** for 10% maximal IV size (worst-case bias):
- K=1: F > 16.38
- K=2: F > 19.93
- K=3: F > 22.30

# Example
```julia
fstat, p_value = first_stage_fstat(treatment, instruments, covariates)

if fstat > 10
    println("Strong instruments (F = \$(round(fstat, digits=2)))")
else
    @warn "Weak instruments detected (F = \$(round(fstat, digits=2)))"
    println("Consider: More instruments, weak IV robust inference, or LIML")
end
```

# References
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments.
- Staiger, D., & Stock, J. H. (1997). "Instrumental Variables Regression with
  Weak Instruments." *Econometrica*, 65(3), 557-586.
"""
function first_stage_fstat(
    treatment::Vector{T},
    instruments::Matrix{T},
    covariates::Union{Matrix{T},Nothing} = nothing,
) where {T<:Real}
    n = length(treatment)
    K = size(instruments, 2)
    p = isnothing(covariates) ? 0 : size(covariates, 2)

    # Build regressor matrix: [Z X] or just Z
    if isnothing(covariates)
        X_fs = instruments
    else
        X_fs = hcat(instruments, covariates)
    end

    # Add intercept
    X_fs = hcat(ones(T, n), X_fs)

    # First-stage regression: D ~ 1 + Z + X
    # Coefficients: β = (X'X)^(-1) X'D
    β = (X_fs' * X_fs) \ (X_fs' * treatment)

    # Fitted values and residuals
    D_fitted = X_fs * β
    residuals = treatment - D_fitted

    # R-squared
    TSS = sum((treatment .- mean(treatment)) .^ 2)
    RSS = sum(residuals .^ 2)
    R_squared = 1 - RSS / TSS

    # Degrees of freedom
    # Total regressors: 1 (intercept) + K (instruments) + p (covariates)
    df_model = K  # Only count instruments (testing π = 0, not intercept/covariates)
    df_resid = n - (1 + K + p)

    # F-statistic for instruments only
    # F = (R² / K) / ((1 - R²) / df_resid)
    if df_resid <= 0
        throw(
            ArgumentError(
                "Insufficient degrees of freedom for first-stage F-test\n" *
                "n=$n, K=$K, p=$p → df_resid=$df_resid ≤ 0\n" *
                "Need n > K + p + 1",
            ),
        )
    end

    fstat = (R_squared / df_model) / ((1 - R_squared) / df_resid)

    # P-value from F-distribution
    p_value = 1 - cdf(FDist(df_model, df_resid), fstat)

    return fstat, p_value
end


"""
    cragg_donald_stat(treatment, instruments, covariates=nothing)

Compute Cragg-Donald minimum eigenvalue statistic for weak IV test.

# Arguments
- `treatment::Vector{T}`: Endogenous variable D (n×1)
- `instruments::Matrix{T}`: Instrumental variables Z (n×K)
- `covariates::Union{Matrix{T}, Nothing}`: Exogenous covariates X (n×p) or nothing

# Returns
- `cd_stat::T`: Cragg-Donald minimum eigenvalue statistic

# Method

The Cragg-Donald statistic tests instrument strength via the minimum eigenvalue
of the concentration parameter matrix.

For single endogenous variable (L=1):
```
CD = λ_min(Z'M_X Z / σ̂²)
```

where:
- M_X = I - X(X'X)^(-1)X' (residual maker for exogenous variables)
- σ̂² = variance of first-stage residuals
- λ_min = minimum eigenvalue

# Interpretation

**Stock-Yogo critical values** (10% maximal IV size):
- K=1: CD > 16.38
- K=2: CD > 19.93
- K=3: CD > 22.30

For single instrument (K=1), CD ≈ F-statistic.

# Example
```julia
cd = cragg_donald_stat(treatment, instruments, covariates)

if cd > stock_yogo_critical_value(K, 0.10, "size")
    println("Instruments pass Stock-Yogo test (CD = \$(round(cd, digits=2)))")
else
    @warn "Weak instruments (CD = \$(round(cd, digits=2)))"
end
```

# References
- Cragg, J. G., & Donald, S. G. (1993). Testing identifiability and specification
  in instrumental variable models.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments.
"""
function cragg_donald_stat(
    treatment::Vector{T},
    instruments::Matrix{T},
    covariates::Union{Matrix{T},Nothing} = nothing,
) where {T<:Real}
    n = length(treatment)
    K = size(instruments, 2)

    # Residual maker for exogenous variables
    if isnothing(covariates)
        # No covariates: M_X = I - 11'/n (demean only)
        one_vec = ones(T, n)
        M_X = I - (one_vec * one_vec') / n
    else
        # With covariates: M_X = I - [1 X]([1 X]'[1 X])^(-1)[1 X]'
        X_exog = hcat(ones(T, n), covariates)
        M_X = I - X_exog * ((X_exog' * X_exog) \ X_exog')
    end

    # Residualize instruments: Z_resid = M_X * Z
    Z_resid = M_X * instruments

    # Residualize treatment: D_resid = M_X * D
    D_resid = M_X * treatment

    # First-stage residual variance
    # Regress D_resid on Z_resid
    β_fs = (Z_resid' * Z_resid) \ (Z_resid' * D_resid)
    ε_fs = D_resid - Z_resid * β_fs
    σ_sq = sum(ε_fs .^ 2) / (n - K - 1)

    # Concentration parameter matrix
    # For L=1 endogenous variable: Ω = Z'M_X Z / σ²
    Omega = (Z_resid' * Z_resid) / σ_sq

    # Cragg-Donald statistic = minimum eigenvalue
    eigenvalues = eigvals(Omega)
    cd_stat = minimum(real.(eigenvalues))  # real() handles numerical noise

    return cd_stat
end


"""
    stock_yogo_critical_value(K, α, test_type)

Stock-Yogo critical values for weak IV tests.

# Arguments
- `K::Int`: Number of instruments
- `α::Float64`: Significance level (0.05, 0.10, 0.15, 0.20, 0.25)
- `test_type::String`: Test type ("size" or "bias")

# Returns
- `critical_value::Float64`: Critical value for test

# Test Types

**Size test** (`test_type="size"`):
- Tests maximal size distortion of Wald test
- Critical values for 10%, 15%, 20%, 25% maximal size
- Use: Ensure Wald test has correct size despite weak IVs

**Bias test** (`test_type="bias"`):
- Tests maximal relative bias of IV estimator
- Critical values for 5%, 10%, 20%, 30% maximal bias
- Use: Ensure IV estimator not too biased toward OLS

# Available Values

Table shows critical values for **L=1 endogenous variable**, **size test**, **10% maximal size**:

| K | Critical Value |
|---|----------------|
| 1 | 16.38          |
| 2 | 19.93          |
| 3 | 22.30          |
| 4 | 24.58          |
| 5 | 26.87          |

# Example
```julia
K = 2
crit_val = stock_yogo_critical_value(K, 0.10, "size")
cd = cragg_donald_stat(treatment, instruments)

if cd > crit_val
    println("Pass Stock-Yogo test (CD=\$cd > \$crit_val)")
else
    @warn "Fail Stock-Yogo test (CD=\$cd < \$crit_val)"
end
```

# Notes
- Values from Stock & Yogo (2005) Table 1
- Only L=1 (single endogenous variable) currently supported
- For L>1, see Stock-Yogo tables or use alternative tests

# References
- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV
  Regression." In *Identification and Inference for Econometric Models*,
  Cambridge University Press, 80-108.
"""
function stock_yogo_critical_value(K::Int, α::Float64, test_type::String)
    # Validate inputs
    if K < 1 || K > 30
        throw(ArgumentError("K must be between 1 and 30, got K=$K"))
    end

    if test_type ∉ ["size", "bias"]
        throw(
            ArgumentError("test_type must be 'size' or 'bias', got '$test_type'"),
        )
    end

    # Stock-Yogo critical values for L=1 endogenous variable
    # Table 1: Size test (maximal size distortion)
    # Rows: K (number of instruments), Cols: maximal size (10%, 15%, 20%, 25%)
    size_table = Dict(
        0.10 => Dict(
            1 => 16.38,
            2 => 19.93,
            3 => 22.30,
            4 => 24.58,
            5 => 26.87,
            6 => 29.18,
            7 => 31.50,
            8 => 33.84,
            9 => 36.19,
            10 => 38.54,
        ),
        0.15 => Dict(
            1 => 8.96,
            2 => 11.59,
            3 => 13.91,
            4 => 16.16,
            5 => 18.37,
            6 => 20.58,
            7 => 22.76,
            8 => 24.94,
            9 => 27.13,
            10 => 29.32,
        ),
        0.20 => Dict(
            1 => 6.66,
            2 => 8.75,
            3 => 10.83,
            4 => 12.83,
            5 => 14.84,
            6 => 16.87,
            7 => 18.90,
            8 => 20.92,
            9 => 22.95,
            10 => 24.98,
        ),
        0.25 => Dict(
            1 => 5.53,
            2 => 7.25,
            3 => 9.13,
            4 => 11.03,
            5 => 12.93,
            6 => 14.85,
            7 => 16.78,
            8 => 18.70,
            9 => 20.64,
            10 => 22.57,
        ),
    )

    # Table 2: Bias test (maximal relative bias)
    # Cols: maximal bias (5%, 10%, 20%, 30%)
    bias_table = Dict(
        0.05 => Dict(
            1 => 13.91,
            2 => 16.85,
            3 => 19.45,
            4 => 22.06,
            5 => 24.58,
            6 => 27.13,
            7 => 29.68,
            8 => 32.24,
            9 => 34.81,
            10 => 37.38,
        ),
        0.10 => Dict(
            1 => 9.08,
            2 => 11.04,
            3 => 12.83,
            4 => 14.68,
            5 => 16.52,
            6 => 18.37,
            7 => 20.23,
            8 => 22.09,
            9 => 23.96,
            10 => 25.82,
        ),
        0.20 => Dict(
            1 => 6.46,
            2 => 7.77,
            3 => 9.09,
            4 => 10.44,
            5 => 11.79,
            6 => 13.15,
            7 => 14.52,
            8 => 15.89,
            9 => 17.27,
            10 => 18.66,
        ),
        0.30 => Dict(
            1 => 5.39,
            2 => 6.46,
            3 => 7.56,
            4 => 8.68,
            5 => 9.80,
            6 => 10.93,
            7 => 12.07,
            8 => 13.21,
            9 => 14.36,
            10 => 15.51,
        ),
    )

    # Select appropriate table
    table = test_type == "size" ? size_table : bias_table

    # Check if α is available
    if !haskey(table, α)
        available_alphas = sort(collect(keys(table)))
        throw(
            ArgumentError(
                "α=$α not available. Available values: $available_alphas\n" *
                "For $test_type test.",
            ),
        )
    end

    # Check if K is available in table
    if !haskey(table[α], K)
        available_K = sort(collect(keys(table[α])))
        throw(
            ArgumentError(
                "K=$K not available in Stock-Yogo tables.\n" *
                "Available K: $available_K\n" *
                "For larger K, consult Stock & Yogo (2005) original tables.",
            ),
        )
    end

    return table[α][K]
end


"""
    weak_iv_warning(fstat, cd_stat, K)

Check for weak instruments and generate warning message if needed.

# Arguments
- `fstat::Float64`: First-stage F-statistic
- `cd_stat::Float64`: Cragg-Donald statistic
- `K::Int`: Number of instruments

# Returns
- `is_weak::Bool`: True if instruments appear weak
- `warning_message::String`: Detailed warning message (empty if strong)

# Diagnostic Criteria

Instruments flagged as weak if **any** of:
1. F-statistic < 10 (rule of thumb)
2. Cragg-Donald < Stock-Yogo 10% maximal size critical value

# Example
```julia
fstat, _ = first_stage_fstat(treatment, instruments)
cd = cragg_donald_stat(treatment, instruments)
K = size(instruments, 2)

is_weak, message = weak_iv_warning(fstat, cd, K)

if is_weak
    @warn message
    println("Recommendations:")
    println("- Use more/stronger instruments")
    println("- Try weak IV robust inference (Anderson-Rubin, CLR)")
    println("- Use LIML instead of 2SLS")
end
```
"""
function weak_iv_warning(fstat::Float64, cd_stat::Float64, K::Int)
    is_weak = false
    messages = String[]

    # Check rule of thumb: F > 10
    if fstat < 10.0
        is_weak = true
        push!(
            messages,
            "First-stage F-statistic = $(round(fstat, digits=2)) < 10 (rule of thumb)",
        )
    end

    # Check Stock-Yogo critical value (if K <= 10)
    if K <= 10
        try
            crit_val = stock_yogo_critical_value(K, 0.10, "size")
            if cd_stat < crit_val
                is_weak = true
                push!(
                    messages,
                    "Cragg-Donald statistic = $(round(cd_stat, digits=2)) < " *
                    "$(round(crit_val, digits=2)) (Stock-Yogo 10% maximal size)",
                )
            end
        catch
            # K > 10 or other error - skip Stock-Yogo check
        end
    end

    # Generate warning message
    if is_weak
        warning_message =
            "⚠️  WEAK INSTRUMENTS DETECTED\n" *
            "\n" *
            "Diagnostics:\n" *
            join(["  - " * msg for msg in messages], "\n") *
            "\n\n" *
            "Implications:\n" *
            "  - 2SLS estimator biased toward OLS\n" *
            "  - Standard errors understate uncertainty\n" *
            "  - Wald tests over-reject\n" *
            "\n" *
            "Recommendations:\n" *
            "  1. Use more/stronger instruments\n" *
            "  2. Use weak IV robust inference (Anderson-Rubin, CLR)\n" *
            "  3. Use LIML instead of 2SLS (better finite-sample properties)\n" *
            "  4. Report both 2SLS and LIML estimates\n"
    else
        warning_message = ""
    end

    return is_weak, warning_message
end
