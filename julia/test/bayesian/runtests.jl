#=
Bayesian Module Test Runner

Runs all tests for Bayesian causal inference estimators.

Session 101: Initial test suite for conjugate ATE.
Session 102: Added Bayesian propensity tests.
Session 103: Added Bayesian DR tests.
Session 104: Added Hierarchical ATE tests (MCMC).
=#

using Test
using CausalEstimators

@testset "Bayesian Tests" begin
    include("test_conjugate_ate.jl")
    include("test_bayesian_propensity.jl")
    include("test_bayesian_dr.jl")
    include("test_hierarchical_ate.jl")
end
