#=
Observational Module Test Runner

Runs all tests for IPW, DR, and TMLE estimators for observational studies.

Test Categories:
1. Unit tests (test_ipw.jl, test_doubly_robust.jl, test_tmle.jl)
2. Monte Carlo validation (test_ipw_montecarlo.jl)
3. Adversarial edge cases (test_observational_adversarial.jl)

Session 80: Added Monte Carlo and Adversarial validation tests.
Session 100: Added TMLE tests.
=#

using Test
using CausalEstimators

@testset "Observational Tests" begin
    # Unit tests
    include("test_ipw.jl")
    include("test_doubly_robust.jl")
    include("test_tmle.jl")

    # Monte Carlo validation
    include("test_ipw_montecarlo.jl")

    # Adversarial edge cases
    include("test_observational_adversarial.jl")
end
