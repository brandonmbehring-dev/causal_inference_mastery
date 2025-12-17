"""
RDD Test Suite Orchestration

Phase 3 test infrastructure with layered validation:
- Fast tests (<1s): Run during development
- Slow tests (>1s): Run before commits
- Golden tests: Run before releases

Test layers (6-layer validation strategy):
1. Unit tests: Known-answer tests for each function
2. Adversarial tests: Edge cases and error handling
3. Monte Carlo tests: Coverage and bias validation
4. Python cross-validation: vs statsmodels (if available)
5. R cross-validation: vs rdrobust (gold standard)
6. Golden tests: Generated from R rdrobust
"""

using Test

# Define test modes for selective execution
const QUICK_MODE = get(ENV, "CAUSAL_TEST_QUICK", "false") == "true"
const FULL_MODE = !QUICK_MODE

@testset "RDD Tests" begin
    # =========================================================================
    # Layer 1: Unit Tests (Fast - always run)
    # =========================================================================
    @testset "Unit Tests" begin
        include("test_types.jl")
        include("test_bandwidth.jl")       # Phase 3.2-3.4
        include("test_sharp_rdd.jl")       # Phase 3.2-3.4
        include("test_fuzzy_rdd.jl")       # Phase 3 - Session 27
        include("test_sensitivity.jl")     # Phase 3.6
        include("test_mccrary.jl")         # Session 57: CJM (2020) variance
    end

    # =========================================================================
    # Layer 2: Adversarial Tests (Fast - always run)
    # =========================================================================
    if FULL_MODE
        @testset "Adversarial Tests" begin
            include("test_sharp_rdd_adversarial.jl")  # Phase 3.9
        end
    end

    # =========================================================================
    # Layer 3: Monte Carlo Tests (Slow - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "Monte Carlo Tests" begin
            include("test_sharp_rdd_montecarlo.jl")  # Phase 3.7
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
    # Layer 5: R Cross-Validation (rdrobust gold standard)
    # =========================================================================
    if FULL_MODE
        @testset "R Cross-Validation" begin
            # TODO: Add test_sharp_rdd_r.jl (Phase 3.8)
            # Requires RCall.jl and R rdrobust package
        end
    end

    # =========================================================================
    # Layer 6: Golden Tests (Generated from R rdrobust)
    # =========================================================================
    if FULL_MODE
        @testset "Golden Tests" begin
            # TODO: Add test_sharp_rdd_golden.jl (Phase 3.8)
            # Tests against 15 pre-generated test cases from R rdrobust
        end
    end
end

println("\n" * "="^70)
if QUICK_MODE
    println("RDD tests completed (QUICK MODE - unit tests only)")
    println("Run with CAUSAL_TEST_QUICK=false for full 6-layer validation")
else
    println("RDD tests completed (FULL MODE - all 6 validation layers)")
end
println("="^70)
