# Julia SciML Style Guide for Causal Inference

**Version**: 1.0
**Date**: 2024-11-14
**Purpose**: Reusable reference for implementing Julia causal inference packages following SciML ecosystem patterns
**Source**: Adapted from DifferentialEquations.jl, SciMLBase.jl, OrdinaryDiffEq.jl patterns

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Problem-Estimator-Solution Architecture](#problem-estimator-solution-architecture)
3. [Type System Design](#type-system-design)
4. [Testing Standards](#testing-standards)
5. [Performance Requirements](#performance-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Complete Worked Example: RCT Estimator](#complete-worked-example-rct-estimator)
8. [Integration with Brandon's Principles](#integration-with-brandons-principles)

---

## Core Philosophy

### 1.1 SciML Design Principles

The SciML ecosystem (Scientific Machine Learning) has established robust patterns for numerical computing packages. These principles apply directly to causal inference:

**Universal solve() Interface**:
```julia
solution = solve(problem, algorithm)
```
- **Problem**: Immutable specification of what to solve
- **Algorithm**: Choice of estimation method
- **Solution**: Results + metadata + convergence info

**Why this matters for causal inference**:
- Consistent API across methods (RCT, DiD, IV, RDD)
- Easy to swap estimators (compare sensitivity)
- Extensible via multiple dispatch
- User-friendly: same pattern everywhere

### 1.2 Brandon's Core Principles Integration

1. **NEVER FAIL SILENTLY** → Custom error types with diagnostic info
2. **Fail Fast** → Validation in problem constructor, not solve()
3. **Immutability by Default** → Problems and solutions are immutable structs
4. **20-50 Line Functions** → Each estimator's solve() method is focused

---

## Problem-Estimator-Solution Architecture

### 2.1 The Three-Layer Pattern

**Layer 1: Problem Types** (data specification)
```julia
abstract type AbstractCausalProblem{T,P} end

struct RCTProblem{T<:Real,P} <: AbstractCausalProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    covariates::Union{Nothing,Matrix{T}}
    strata::Union{Nothing,Vector{Int}}
    parameters::P

    # Inner constructor with validation
    function RCTProblem(outcomes::Vector{T}, treatment::Vector{Bool},
                        covariates::Union{Nothing,Matrix{T}},
                        strata::Union{Nothing,Vector{Int}},
                        parameters::P) where {T<:Real,P}
        validate_rct_inputs(outcomes, treatment, covariates, strata)
        new{T,P}(outcomes, treatment, covariates, strata, parameters)
    end
end
```

**Why parametric types `{T,P}`?**
- `T`: Numeric type (Float64, Float32, BigFloat) → type stability
- `P`: Parameter type (NamedTuple usually) → flexible configuration
- Compiler knows exact types → faster code, zero overhead

**Layer 2: Estimator Types** (algorithm specification)
```julia
abstract type AbstractCausalEstimator end

# Simple estimators are empty structs (no state)
struct SimpleATE <: AbstractCausalEstimator end

# Complex estimators hold algorithm parameters
struct PermutationTest <: AbstractCausalEstimator
    n_permutations::Union{Nothing,Int}  # Nothing = exact test
    random_seed::Union{Nothing,Int}
end
```

**Layer 3: Solution Types** (results + metadata)
```julia
struct RCTSolution{T<:Real,P}
    estimate::T
    se::T
    ci_lower::T
    ci_upper::T
    n_treated::Int
    n_control::Int
    retcode::Symbol  # :Success, :Warning, :Error
    original_problem::RCTProblem{T,P}
end
```

**Why include original_problem?**
- Full reproducibility (inputs + outputs in one object)
- Enables sensitivity analysis (remake problem with different parameters)
- Debugging (inspect inputs that led to solution)

### 2.2 The solve() Interface

```julia
function solve(problem::RCTProblem, estimator::SimpleATE)::RCTSolution
    # Extract from problem (destructuring)
    (; outcomes, treatment, parameters) = problem

    # Compute estimate
    y1 = outcomes[treatment]
    y0 = outcomes[.!treatment]
    ate = mean(y1) - mean(y0)

    # Compute uncertainty (Neyman heteroskedasticity-robust)
    se = sqrt(var(y1)/length(y1) + var(y0)/length(y0))

    # Confidence interval
    α = parameters.alpha
    z = quantile(Normal(), 1 - α/2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    # Build solution
    return RCTSolution(
        estimate=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=length(y1),
        n_control=length(y0),
        retcode=:Success,
        original_problem=problem
    )
end
```

**Key patterns**:
- Type annotation on return (`::RCTSolution`)
- Destructuring with `;` (named tuple unpack)
- Named constructor arguments for clarity
- Return code for error handling

### 2.3 Multiple Dispatch Power

```julia
# Different estimators dispatch to different methods
solve(problem::RCTProblem, estimator::SimpleATE) = ... # difference-in-means
solve(problem::RCTProblem, estimator::StratifiedATE) = ... # stratified
solve(problem::RCTProblem, estimator::RegressionATE) = ... # ANCOVA

# Different problem types dispatch differently
solve(problem::DiDProblem, estimator::TwoWayFE) = ... # panel DiD
solve(problem::IVProblem, estimator::TSLS) = ... # instrumental variables
```

---

## Type System Design

### 3.1 Abstract Type Hierarchy

**Three-level hierarchy** (not more - avoid over-engineering):

```julia
# Level 1: Universal causal inference base
abstract type AbstractCausalProblem{T,P} end
abstract type AbstractCausalEstimator end
abstract type AbstractCausalSolution end

# Level 2: Method-specific (RCT, DiD, IV, RDD, etc.)
abstract type AbstractRCTProblem{T,P} <: AbstractCausalProblem{T,P} end
abstract type AbstractRCTEstimator <: AbstractCausalEstimator end
abstract type AbstractRCTSolution <: AbstractCausalSolution end

# Level 3: Concrete types (actual implementations)
struct RCTProblem{T,P} <: AbstractRCTProblem{T,P}
    # ... fields ...
end

struct SimpleATE <: AbstractRCTEstimator end
struct StratifiedATE <: AbstractRCTEstimator end
# etc.
```

**When to add a level?**
- Need to dispatch on commonalities (e.g., all RCT estimators share validation)
- NOT for organizational purposes only
- Keep it shallow (3 levels max)

### 3.2 Type Stability

**Golden rule**: `@code_warntype` must show **no red** (unstable types).

```julia
# Type-stable ✓
function compute_ate(outcomes::Vector{Float64}, treatment::Vector{Bool})::Float64
    y1 = outcomes[treatment]
    y0 = outcomes[.!treatment]
    return mean(y1) - mean(y0)
end

# Type-unstable ✗ (return type depends on data)
function compute_ate_bad(outcomes, treatment)
    if length(outcomes) > 100
        return mean(outcomes[treatment]) - mean(outcomes[.!treatment])  # Float64
    else
        return nothing  # Nothing - TYPE INSTABILITY!
    end
end
```

**How to verify**:
```julia
@code_warntype compute_ate(rand(100), rand(Bool, 100))
# Should show: Body::Float64 (no red highlighting)
```

### 3.3 Immutability

**Default**: All problem and solution types are immutable structs.

```julia
# Immutable by default ✓
struct RCTProblem{T,P}
    outcomes::Vector{T}
    treatment::Vector{Bool}
    # ...
end

# Mutable only if profiling proves necessary ✗
# (Very rare in causal inference - data doesn't change during estimation)
```

**To modify a problem** (sensitivity analysis):
```julia
# Use remake() pattern from SciML
function remake(problem::RCTProblem; outcomes=problem.outcomes,
                                      treatment=problem.treatment,
                                      covariates=problem.covariates,
                                      strata=problem.strata,
                                      parameters=problem.parameters)
    return RCTProblem(outcomes, treatment, covariates, strata, parameters)
end

# Usage
new_problem = remake(original_problem, parameters=(alpha=0.01,))
```

---

## Testing Standards

### 4.1 Test Organization (Feature-Based, Not File-Based)

**SciML pattern**: Organize by feature, not by source file.

```julia
# test/runtests.jl
using SafeTestsets

@safetestset "Problem Construction" begin
    include("test_problems.jl")
end

@safetestset "RCT Estimators" begin
    @safetestset "SimpleATE" begin include("rct/test_simple_ate.jl") end
    @safetestset "StratifiedATE" begin include("rct/test_stratified_ate.jl") end
    @safetestset "RegressionATE" begin include("rct/test_regression_ate.jl") end
    @safetestset "PermutationTest" begin include("rct/test_permutation_test.jl") end
    @safetestset "IPWATE" begin include("rct/test_ipw_ate.jl") end
end

@safetestset "Golden Reference Validation" begin
    include("rct/test_golden_reference.jl")
end

@safetestset "Statistical Properties" begin
    include("test_properties.jl")
end
```

**Why `@safetestset`?**
- Prevents variable leakage between test sets
- Each test runs in isolated namespace
- Catches unintended dependencies
- SciML standard practice

### 4.2 ReferenceTests.jl for Golden Validation

**Pattern**: Cross-validate against Python/R implementations.

```julia
using ReferenceTests
using JSON3

@testset "Golden Reference: SimpleATE" begin
    # Load Python golden results
    golden = JSON3.read(read("golden_results/python_golden_results.json", String))
    data = golden.balanced_rct.data
    expected = golden.balanced_rct.simple_ate

    # Create problem
    problem = RCTProblem(
        Float64.(data.outcomes),
        Bool.(data.treatment),
        nothing, nothing,
        (alpha=0.05,)
    )

    # Solve
    solution = solve(problem, SimpleATE())

    # Validate to 10 decimal places
    @test solution.estimate ≈ expected.estimate rtol=1e-10
    @test solution.se ≈ expected.se rtol=1e-10
    @test solution.ci_lower ≈ expected.ci_lower rtol=1e-10
    @test solution.ci_upper ≈ expected.ci_upper rtol=1e-10
end
```

**Why rtol=1e-10?**
- Ensures near-machine precision agreement
- Catches numerical implementation differences
- Standard for cross-language validation
- If implementations differ > 10 decimal places, investigate why

### 4.3 PyCall for Development Validation

**Pattern**: Use during implementation, not in final test suite.

```julia
# test/validation/test_pycall_development.jl
using PyCall

@testset "PyCall Development: SimpleATE" begin
    # Import Python implementation
    py"""
    import sys
    sys.path.append('../../src')
    from causal_inference.rct.estimators import simple_ate
    """

    # Test data
    outcomes = [7.0, 5.0, 3.0, 1.0]
    treatment = [true, true, false, false]

    # Julia result
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
    jl_result = solve(problem, SimpleATE())

    # Python result
    py_result = py"simple_ate"(outcomes, Int.(treatment))

    # Compare
    @test jl_result.estimate ≈ py_result["estimate"] rtol=1e-10
    @test jl_result.se ≈ py_result["se"] rtol=1e-10

    println("✓ Julia and Python agree to 10 decimal places")
end
```

**When to use**:
- During implementation (immediate feedback)
- Not in `runtests.jl` (avoid Python dependency)
- Run manually: `julia --project test/validation/test_pycall_development.jl`

### 4.4 Property-Based Testing

**Pattern**: Test statistical properties, not just point estimates.

```julia
using Random

@testset "Confidence Interval Coverage" begin
    Random.seed!(12345)

    # Monte Carlo: 95% CI should contain true ATE ~95% of time
    true_ate = 5.0
    coverage_count = 0
    n_sims = 1000

    for _ in 1:n_sims
        # Simulate RCT data
        treatment = rand(Bool, 100)
        outcomes = treatment .* true_ate .+ randn(100)

        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
        solution = solve(problem, SimpleATE())

        if solution.ci_lower ≤ true_ate ≤ solution.ci_upper
            coverage_count += 1
        end
    end

    coverage_rate = coverage_count / n_sims
    @test 0.94 ≤ coverage_rate ≤ 0.96  # 95% ± 1% tolerance
end
```

**What to test**:
- Confidence interval coverage (nominal vs actual)
- Unbiasedness (estimate → true value as n → ∞)
- Variance reduction (stratified < simple)
- Type I error control (permutation tests)

### 4.5 Coverage Target

**Standard**: 90%+ coverage for production-ready code.

```julia
# Run with coverage tracking
julia --project --code-coverage=user test/runtests.jl

# Generate coverage report
using Coverage
coverage = process_folder()
covered_lines, total_lines = get_summary(coverage)
percentage = covered_lines / total_lines * 100

@test percentage ≥ 90.0  # Enforce 90%+ coverage
```

---

## Performance Requirements

### 5.1 Type Stability Verification

**Mandatory check** before declaring estimator complete.

```julia
# scripts/check_type_stability.jl
using CausalEstimators

function check_simple_ate_stability()
    outcomes = [7.0, 5.0, 3.0, 1.0]
    treatment = [true, true, false, false]
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

    println("Type stability check for SimpleATE:")
    @code_warntype solve(problem, SimpleATE())
    # Should show: Body::RCTSolution{Float64, ...} with NO RED
end

check_simple_ate_stability()
```

**What to look for**:
- Return type is concrete (not `Any` or `Union{...}`)
- No red highlighting in output
- All intermediate variables have concrete types

### 5.2 Benchmarking with BenchmarkTools.jl

**Pattern**: Statistical benchmarking, not just `@time`.

```julia
using BenchmarkTools

function benchmark_simple_ate()
    # Generate test data
    outcomes = randn(10000)
    treatment = rand(Bool, 10000)
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

    # Benchmark (runs multiple times, computes statistics)
    result = @benchmark solve($problem, SimpleATE())

    println("SimpleATE (n=10,000):")
    println("  Median time: $(median(result.times) / 1e6) ms")
    println("  Min time: $(minimum(result.times) / 1e6) ms")
    println("  Allocations: $(result.allocs)")
    println("  Memory: $(result.memory / 1024) KB")

    return result
end
```

**Benchmark multiple sizes**:
```julia
sizes = [100, 1_000, 10_000]
for n in sizes
    outcomes = randn(n)
    treatment = rand(Bool, n)
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

    result = @benchmark solve($problem, SimpleATE())
    println("n=$n: $(median(result.times) / 1e6) ms")
end
```

### 5.3 Cross-Language Performance Comparison

**Pattern**: Validate Julia speedup claim.

```julia
using PyCall
using BenchmarkTools

function compare_julia_vs_python()
    # Large dataset
    outcomes = randn(10_000)
    treatment = rand(Bool, 10_000)

    # Julia benchmark
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))
    jl_time = @belapsed solve($problem, SimpleATE())

    # Python benchmark
    py"""
    import sys
    sys.path.append('../src')
    from causal_inference.rct.estimators import simple_ate
    """
    py_time = @belapsed py"simple_ate"($outcomes, Int.($treatment))

    speedup = py_time / jl_time
    println("Julia: $(jl_time * 1000) ms")
    println("Python: $(py_time * 1000) ms")
    println("Speedup: $(round(speedup, digits=1))x")

    @test speedup ≥ 2.0  # Julia should be at least 2x faster
end
```

**Expected results**:
- Simple operations: 2-5x speedup
- Complex operations: 5-10x speedup
- If <2x: investigate type instabilities or allocations

### 5.4 Memory Allocation Tracking

**Pattern**: Minimize allocations in hot loops.

```julia
function count_allocations()
    outcomes = randn(10_000)
    treatment = rand(Bool, 10_000)
    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

    # Count allocations
    allocs = @allocated solve(problem, SimpleATE())

    println("Total allocations: $(allocs / 1024) KB")

    # For n=10,000, should be <100 KB for simple estimators
    @test allocs < 100_000  # bytes
end
```

---

## Documentation Standards

### 6.1 Docstring Template

**Every function needs**:
1. One-sentence summary
2. Extended description (1-2 paragraphs)
3. Mathematical foundation (LaTeX)
4. Arguments with types and descriptions
5. Returns with type and fields
6. Examples that actually run
7. References to papers/books

**Template**:
```julia
"""
    SimpleATE <: AbstractRCTEstimator

Simple difference-in-means estimator for average treatment effect in randomized experiments.

# Mathematical Foundation

Under randomization, the average treatment effect (ATE) is identified by:

```math
\\tau = \\mathbb{E}[Y(1) - Y(0)] = \\mathbb{E}[Y|T=1] - \\mathbb{E}[Y|T=0]
```

The sample estimator is:

```math
\\hat{\\tau} = \\bar{Y}_1 - \\bar{Y}_0
```

Standard error uses Neyman heteroskedasticity-robust variance:

```math
SE(\\hat{\\tau}) = \\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_0^2}{n_0}}
```

where ``s_t^2`` is the sample variance for treatment group ``t \\in \\{0,1\\}``.

# Usage

```julia
using CausalEstimators

# Create problem
outcomes = [10.0, 12.0, 4.0, 5.0]
treatment = [true, true, false, false]
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())

# Extract results
solution.estimate  # Point estimate
solution.se        # Standard error
solution.ci_lower  # Lower 95% CI bound
solution.ci_upper  # Upper 95% CI bound
```

# Returns

Returns `RCTSolution` with fields:
- `estimate::Float64`: Point estimate of ATE
- `se::Float64`: Standard error (Neyman variance)
- `ci_lower::Float64`: Lower confidence bound
- `ci_upper::Float64`: Upper confidence bound
- `n_treated::Int`: Number of treated units
- `n_control::Int`: Number of control units
- `retcode::Symbol`: `:Success` if estimation succeeded
- `original_problem::RCTProblem`: Original problem for reproducibility

# References

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences*. Cambridge University Press. Chapter 6.
- Neyman, J. (1923). On the Application of Probability Theory to Agricultural
  Experiments. *Statistical Science*, 5(4), 465-472.
"""
struct SimpleATE <: AbstractRCTEstimator end
```

### 6.2 README Documentation

**Must include**:
```markdown
# CausalEstimators.jl

Randomized Controlled Trial (RCT) estimators following SciML design patterns.

## Installation
```julia
] add CausalEstimators
```

## Quick Start
```julia
using CausalEstimators

# Generate RCT data
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]

# Create problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())
println("ATE: $(solution.estimate) ± $(solution.se)")
```

## Estimators

- `SimpleATE`: Difference-in-means
- `StratifiedATE`: Block randomization
- `RegressionATE`: ANCOVA adjustment
- `PermutationTest`: Fisher exact test
- `IPWATE`: Inverse probability weighting

## Performance

Julia achieves 2-10x speedup over Python implementations:
- SimpleATE (n=10,000): 0.5 ms (Julia) vs 5 ms (Python) = 10x
- RegressionATE (n=10,000): 2 ms (Julia) vs 15 ms (Python) = 7.5x

## Documentation

See [full documentation](https://...) for API reference and tutorials.
```

---

## Complete Worked Example: RCT Estimator

This section shows a complete implementation of StratifiedATE following all patterns above.

### 7.1 Problem Type (Already Defined)

```julia
# RCTProblem is already defined - works for all RCT estimators
# Just needs strata field populated
```

### 7.2 Estimator Type

```julia
"""
    StratifiedATE <: AbstractRCTEstimator

Stratified difference-in-means estimator for blocked randomization designs.

Computes weighted average of stratum-specific ATEs, where weights are
proportional to stratum sizes.

# Mathematical Foundation

For ``S`` strata, the stratified ATE is:

```math
\\hat{\\tau}_{strat} = \\sum_{s=1}^S w_s \\hat{\\tau}_s
```

where ``w_s = n_s / n`` is the weight for stratum ``s`` (proportion of sample),
and ``\\hat{\\tau}_s`` is the simple ATE within stratum ``s``.

Variance:

```math
Var(\\hat{\\tau}_{strat}) = \\sum_{s=1}^S w_s^2 Var(\\hat{\\tau}_s)
```

# Usage

```julia
# Blocked randomization data
outcomes = [100.0, 105.0, 10.0, 15.0]  # Two strata with different baselines
treatment = [true, false, true, false]
strata = [1, 1, 2, 2]

problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))
solution = solve(problem, StratifiedATE())
```

# References

- Imbens & Rubin (2015), Chapter 9: Stratified Randomized Experiments
"""
struct StratifiedATE <: AbstractRCTEstimator end
```

### 7.3 solve() Implementation

```julia
function solve(problem::RCTProblem, estimator::StratifiedATE)::RCTSolution
    # Extract from problem
    (; outcomes, treatment, strata, parameters) = problem

    # Validation: strata must be provided
    if isnothing(strata)
        throw(ArgumentError(
            "StratifiedATE requires strata. Got strata=nothing.\\n" *
            "Use SimpleATE for unstratified designs."
        ))
    end

    # Get unique strata
    unique_strata = unique(strata)
    n_strata = length(unique_strata)

    # Compute stratum-specific estimates
    stratum_estimates = Float64[]
    stratum_ses = Float64[]
    stratum_weights = Float64[]

    for s in unique_strata
        # Select units in this stratum
        in_stratum = strata .== s
        y_s = outcomes[in_stratum]
        t_s = treatment[in_stratum]

        # Check for variation in treatment within stratum
        if !any(t_s) || all(t_s)
            throw(ArgumentError(
                "Stratum $s has no treatment variation.\\n" *
                "Cannot estimate ATE without both treated and control units."
            ))
        end

        # Compute ATE within stratum
        y1_s = y_s[t_s]
        y0_s = y_s[.!t_s]
        ate_s = mean(y1_s) - mean(y0_s)

        # Variance within stratum (Neyman)
        var_s = var(y1_s)/length(y1_s) + var(y0_s)/length(y0_s)
        se_s = sqrt(var_s)

        # Weight = proportion of sample in this stratum
        weight_s = sum(in_stratum) / length(outcomes)

        push!(stratum_estimates, ate_s)
        push!(stratum_ses, se_s)
        push!(stratum_weights, weight_s)
    end

    # Weighted average ATE
    ate = sum(stratum_weights .* stratum_estimates)

    # Variance: sum of weighted variances
    var_ate = sum((stratum_weights .^ 2) .* (stratum_ses .^ 2))
    se = sqrt(var_ate)

    # Confidence interval
    α = parameters.alpha
    z = quantile(Normal(), 1 - α/2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    # Count totals
    n_treated = sum(treatment)
    n_control = sum(.!treatment)

    return RCTSolution(
        estimate=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=n_treated,
        n_control=n_control,
        retcode=:Success,
        original_problem=problem
    )
end
```

### 7.4 Tests

```julia
# test/rct/test_stratified_ate.jl
using Test, SafeTestsets
using CausalEstimators

@safetestset "StratifiedATE: Known Answer" begin
    # Two strata with different baselines
    outcomes = [100.0, 105.0, 10.0, 15.0]
    treatment = [true, false, true, false]
    strata = [1, 1, 2, 2]

    # Hand calculation:
    # Stratum 1: ATE = 100 - 105 = -5, weight = 0.5
    # Stratum 2: ATE = 10 - 15 = -5, weight = 0.5
    # Weighted ATE = 0.5*(-5) + 0.5*(-5) = -5

    problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))
    solution = solve(problem, StratifiedATE())

    @test solution.estimate ≈ -5.0 atol=1e-10
    @test solution.retcode == :Success
end

@safetestset "StratifiedATE: Missing Strata Error" begin
    outcomes = [1.0, 2.0, 3.0, 4.0]
    treatment = [true, true, false, false]

    problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

    @test_throws ArgumentError solve(problem, StratifiedATE())
end

@safetestset "StratifiedATE: No Treatment Variation" begin
    outcomes = [1.0, 2.0, 3.0, 4.0]
    treatment = [true, true, false, false]
    strata = [1, 1, 2, 2]  # Stratum 1 all treated, Stratum 2 all control

    problem = RCTProblem(outcomes, treatment, nothing, strata, (alpha=0.05,))

    @test_throws ArgumentError solve(problem, StratifiedATE())
end
```

### 7.5 Golden Reference Test

```julia
# test/rct/test_golden_reference.jl
@safetestset "Golden: StratifiedATE" begin
    golden = JSON3.read(read("golden_results/python_golden_results.json", String))
    data = golden.stratified_rct.data
    expected = golden.stratified_rct.stratified_ate

    problem = RCTProblem(
        Float64.(data.outcomes),
        Bool.(data.treatment),
        nothing,
        Int.(data.strata),
        (alpha=0.05,)
    )

    solution = solve(problem, StratifiedATE())

    @test solution.estimate ≈ expected.estimate rtol=1e-10
    @test solution.se ≈ expected.se rtol=1e-10
end
```

---

## Integration with Brandon's Principles

### 8.1 NEVER FAIL SILENTLY

**Pattern**: Explicit errors with diagnostic info.

```julia
# Bad ✗ - Silent failure
function solve(problem::RCTProblem, estimator::SimpleATE)
    y1 = outcomes[treatment]
    y0 = outcomes[.!treatment]

    if length(y1) == 0
        return RCTSolution(..., retcode=:Warning)  # SILENT!
    end
    # ...
end

# Good ✓ - Explicit error
function solve(problem::RCTProblem, estimator::SimpleATE)
    y1 = outcomes[treatment]
    y0 = outcomes[.!treatment]

    if length(y1) == 0
        throw(ArgumentError(
            "CRITICAL ERROR: No treated units in data.\\n" *
            "Function: solve(RCTProblem, SimpleATE)\\n" *
            "Cannot estimate treatment effect without treated group.\\n" *
            "Got: All units have treatment=false"
        ))
    end
    # ...
end
```

### 8.2 Fail Fast (Validation in Constructor)

```julia
# Validate in problem constructor, not in solve()
function RCTProblem(outcomes, treatment, covariates, strata, parameters)
    # Check lengths match
    if length(treatment) != length(outcomes)
        throw(ArgumentError(
            "CRITICAL ERROR: Mismatched lengths.\\n" *
            "outcomes: $(length(outcomes)), treatment: $(length(treatment))"
        ))
    end

    # Check for NaN
    if any(isnan, outcomes)
        throw(ArgumentError(
            "CRITICAL ERROR: NaN in outcomes.\\n" *
            "Indices: $(findall(isnan, outcomes))"
        ))
    end

    # Check treatment is binary
    if !all(t -> t in [0, 1, true, false], treatment)
        throw(ArgumentError(
            "CRITICAL ERROR: Treatment must be binary.\\n" *
            "Got unique values: $(unique(treatment))"
        ))
    end

    # ... more validation ...

    new{T,P}(outcomes, treatment, covariates, strata, parameters)
end
```

### 8.3 Function Length (20-50 Lines)

```julia
# If solve() exceeds 50 lines, extract helpers:

function solve(problem::RCTProblem, estimator::RegressionATE)::RCTSolution
    # Main logic only (25 lines)
    X = construct_design_matrix(problem)
    β = fit_ols(problem.outcomes, X)
    se = compute_robust_se(problem, β, X)
    ci_lower, ci_upper = confidence_interval(β[2], se, problem.parameters.alpha)

    return RCTSolution(...)
end

# Helper functions
function construct_design_matrix(problem::RCTProblem)::Matrix{Float64}
    # 15 lines
end

function fit_ols(outcomes::Vector{Float64}, X::Matrix{Float64})::Vector{Float64}
    # 10 lines
end

function compute_robust_se(problem::RCTProblem, β::Vector{Float64}, X::Matrix{Float64})::Float64
    # 20 lines
end
```

### 8.4 SciML Formatting (92-char lines)

```julia
# .JuliaFormatter.toml
style = "sciml"
margin = 92
indent = 4
always_for_in = true
whitespace_typedefs = true
whitespace_ops_in_indices = true
remove_extra_newlines = true
```

---

## Checklist for New Estimators

Before declaring an estimator complete:

- [ ] **Type System**
  - [ ] Problem type defined (or uses existing RCTProblem)
  - [ ] Estimator struct defined
  - [ ] solve() method implemented
  - [ ] Return type annotation (`::RCTSolution`)

- [ ] **Documentation**
  - [ ] Complete docstring with LaTeX math
  - [ ] Running example in docstring
  - [ ] References to papers
  - [ ] README entry

- [ ] **Testing**
  - [ ] Known-answer tests (hand-calculated)
  - [ ] Error handling tests
  - [ ] Golden reference test (rtol=1e-10)
  - [ ] PyCall development validation
  - [ ] Property-based tests
  - [ ] 90%+ coverage

- [ ] **Performance**
  - [ ] Type stability verified (`@code_warntype`)
  - [ ] Benchmarked (3 sizes: 100, 1K, 10K)
  - [ ] Compared to Python (≥2x speedup)
  - [ ] Allocations tracked

- [ ] **Code Quality**
  - [ ] Functions 20-50 lines
  - [ ] Fail-fast validation
  - [ ] Explicit error messages
  - [ ] SciML formatted (92-char margin)
  - [ ] No type instabilities
  - [ ] Immutable types

---

## Additional Resources

**SciML Documentation**:
- DifferentialEquations.jl: https://docs.sciml.ai/DiffEqDocs/stable/
- SciMLBase.jl: https://docs.sciml.ai/SciMLBase/stable/
- SciMLStyle: https://github.com/SciML/SciMLStyle

**Julia Performance**:
- Performance Tips: https://docs.julialang.org/en/v1/manual/performance-tips/
- BenchmarkTools.jl: https://juliaci.github.io/BenchmarkTools.jl/stable/

**Testing**:
- SafeTestsets.jl: https://github.com/YingboMa/SafeTestsets.jl
- ReferenceTests.jl: https://juliatesting.github.io/ReferenceTests.jl/stable/

---

**End of Style Guide v1.0**

This guide will evolve as you implement DiD, IV, RDD, and other methods. Update with lessons learned and new patterns discovered.
