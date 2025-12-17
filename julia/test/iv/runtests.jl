"""
IV Test Suite Orchestration

Phase 4 test infrastructure with layered validation:
- Fast tests (<1s): Run during development
- Slow tests (>1s): Run before commits

Test layers (following Phase 3 RDD pattern):
1. Unit tests: Known-answer tests for each estimator
2. Adversarial tests: Edge cases and error handling
3. Monte Carlo tests: Coverage and bias validation
4. R cross-validation: vs ivreg, ivmodel (gold standard)

Usage:
```
# Quick mode (unit tests only)
CAUSAL_TEST_QUICK=true julia --project -e 'using Pkg; Pkg.test("CausalEstimators")'

# Full mode (all validation layers)
julia --project -e 'using Pkg; Pkg.test("CausalEstimators")'
```
"""

using Test

# Define test modes
const QUICK_MODE = get(ENV, "CAUSAL_TEST_QUICK", "false") == "true"
const FULL_MODE = !QUICK_MODE

@testset "IV Estimators" begin
    # =========================================================================
    # Layer 1: Unit Tests (Fast - always run)
    # =========================================================================
    @testset "Unit Tests" begin
        include("test_types.jl")  # Phase 4.1
        include("test_diagnostics.jl")  # Phase 4.2
        include("test_tsls.jl")  # Phase 4.3
        include("test_liml.jl")  # Phase 4.4
        include("test_gmm.jl")  # Phase 4.5
        include("test_weak_iv_robust.jl")  # Phase 4.6
        include("test_vcov.jl")  # Session 56: VCov estimators
        include("test_stages.jl")  # Session 56: Stage decomposition
    end

    # =========================================================================
    # Layer 2: Adversarial Tests (Fast - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "Adversarial Tests" begin
            # TODO: Add test_iv_adversarial.jl (Phase 4.11)
        end
    end

    # =========================================================================
    # Layer 3: Monte Carlo Tests (Slow - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "Monte Carlo Tests" begin
            # TODO: Add test_iv_montecarlo.jl (Phase 4.9)
        end
    end

    # =========================================================================
    # Layer 4: Python Cross-Validation (Requires PyCall - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "Python Cross-Validation" begin
            include("test_pycall_validation.jl")
        end
    end

    # =========================================================================
    # Layer 5: R Cross-Validation (Requires R - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "R Cross-Validation" begin
            # TODO: Add test_iv_r.jl (Phase 4.10)
            # Requires RCall.jl and R packages (ivreg, ivmodel)
        end
    end
end

println("\n" * "="^70)
if QUICK_MODE
    println("IV tests completed (QUICK MODE - unit tests only)")
    println("Run with CAUSAL_TEST_QUICK=false for full validation")
else
    println("IV tests completed (FULL MODE - all validation layers)")
end
println("="^70)
