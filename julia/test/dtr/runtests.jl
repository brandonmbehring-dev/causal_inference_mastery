"""
Dynamic Treatment Regimes Test Suite

Comprehensive tests for Q-learning and A-learning DTR estimators.

Coverage:
- Layer 1: Known-answer tests with hand-calculated values
- Layer 2: Adversarial tests for edge cases
- Layer 3: Monte Carlo validation for statistical properties
"""

using Test
using SafeTestsets

@safetestset "Dynamic Treatment Regimes" begin
    @safetestset "Q-Learning" begin
        include("test_q_learning.jl")
    end
    @safetestset "A-Learning" begin
        include("test_a_learning.jl")
    end
end
