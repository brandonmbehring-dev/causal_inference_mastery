"""
Selection Model Test Suite

Session 85 test infrastructure for Heckman selection model:
- Unit tests: Known-answer tests
- Adversarial tests: Edge cases and error handling
- Monte Carlo tests: Coverage and bias validation
- Cross-language tests: Python parity (via PyCall)

Usage:
```julia
# Quick mode (unit tests only)
CAUSAL_TEST_QUICK=true julia --project -e 'include("test/selection/runtests.jl")'

# Full mode (all validation layers)
julia --project -e 'include("test/selection/runtests.jl")'
```
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# Define test modes
const QUICK_MODE = get(ENV, "CAUSAL_TEST_QUICK", "false") == "true"
const FULL_MODE = !QUICK_MODE

@testset "Selection Models" begin
    # =========================================================================
    # Layer 1: Unit Tests (Fast - always run)
    # =========================================================================
    @testset "Unit Tests" begin
        include("test_heckman.jl")
    end

    # =========================================================================
    # Layer 2: Monte Carlo Tests (Slow - full mode only)
    # =========================================================================
    if FULL_MODE
        @testset "Monte Carlo Tests" begin
            include("test_heckman_montecarlo.jl")
        end
    end
end
