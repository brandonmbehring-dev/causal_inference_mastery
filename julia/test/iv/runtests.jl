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
        # TODO: Add test_diagnostics.jl (Phase 4.2)
        # TODO: Add test_tsls.jl (Phase 4.3)
        # TODO: Add test_liml.jl (Phase 4.4)
        # TODO: Add test_gmm.jl (Phase 4.5)
        # TODO: Add test_weak_iv_robust.jl (Phase 4.6)
        # TODO: Add test_sensitivity.jl (Phase 4.8)
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
    # Layer 4: R Cross-Validation (Requires R - full mode only)
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
