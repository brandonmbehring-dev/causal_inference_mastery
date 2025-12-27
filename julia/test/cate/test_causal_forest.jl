#=
Tests for Causal Forest

Session 157: Causal Forest Julia Parity

Test Layers:
- Layer 1: Known-Answer - Constant/step/linear effect DGPs with known ground truth
- Layer 2: Adversarial - Edge cases and error handling
=#

using Test
using CausalEstimators
using Random
using Statistics

# Include shared DGP file
include("dgp_cate.jl")


# =============================================================================
# Layer 1: Known-Answer Tests - Constant Effect
# =============================================================================

@testset "Causal Forest Known-Answer" begin

    @testset "Constant effect DGP" begin
        data = dgp_constant_effect(n=1000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(n_trees=50, min_leaf_size=20))

        @test solution.retcode == :Success
        @test solution.method == :causal_forest
        @test length(solution.cate) == data.n

        # ATE should be close to true (bias < 0.5)
        @test abs(solution.ate - data.true_ate) < 0.5

        # CI should cover true ATE
        @test solution.ci_lower < data.true_ate < solution.ci_upper
    end

    @testset "Linear heterogeneity DGP" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(n_trees=50))

        @test solution.retcode == :Success

        # ATE should be reasonably close
        @test abs(solution.ate - data.true_ate) < 0.5

        # CATE should have some variation (forest captures heterogeneity)
        @test std(solution.cate) > 0.01
    end

    @testset "Complex heterogeneity (step + linear)" begin
        data = dgp_complex_heterogeneity(n=1000, base_effect=1.0, step_effect=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(n_trees=50, max_depth=8))

        @test solution.retcode == :Success

        # ATE should be in reasonable range
        @test abs(solution.ate - data.true_ate) < 1.0

        # CATE should have substantial variation for step function
        @test std(solution.cate) > 0.1
    end

    @testset "Large sample convergence" begin
        # With more data, forest should be more accurate
        data = dgp_constant_effect(n=2000, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(n_trees=100))

        @test solution.retcode == :Success

        # Larger sample should have tighter ATE estimate
        @test abs(solution.ate - data.true_ate) < 0.4
    end

end


# =============================================================================
# Layer 1: Known-Answer Tests - Step Function Heterogeneity
# =============================================================================

@testset "Causal Forest Step Heterogeneity" begin

    @testset "Step function CATE" begin
        # Forest should capture step function heterogeneity well
        Random.seed!(42)
        n = 1000
        X = randn(n, 5)
        T = rand(n) .< 0.5

        # Step CATE: τ(x) = 3 if x₁ > 0, else 1
        true_cate = ifelse.(X[:, 1] .> 0, 3.0, 1.0)
        Y = X[:, 1] .+ T .* true_cate .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))
        solution = solve(problem, CausalForestEstimator(n_trees=100, min_leaf_size=15))

        @test solution.retcode == :Success

        # ATE should be close to true average
        true_ate = mean(true_cate)
        @test abs(solution.ate - true_ate) < 0.5

        # CATE should show bimodal pattern
        cate_positive_x = solution.cate[X[:, 1] .> 0]
        cate_negative_x = solution.cate[X[:, 1] .<= 0]

        # Average CATE should be higher for positive X[:, 1]
        @test mean(cate_positive_x) > mean(cate_negative_x)
    end

end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Causal Forest Adversarial" begin

    @testset "Invalid n_trees parameter" begin
        @test_throws ArgumentError CausalForestEstimator(n_trees=0)
        @test_throws ArgumentError CausalForestEstimator(n_trees=-1)
    end

    @testset "Invalid min_leaf_size parameter" begin
        @test_throws ArgumentError CausalForestEstimator(min_leaf_size=0)
        @test_throws ArgumentError CausalForestEstimator(min_leaf_size=-1)
    end

    @testset "Invalid max_depth parameter" begin
        @test_throws ArgumentError CausalForestEstimator(max_depth=0)
    end

    @testset "Invalid subsample_ratio parameter" begin
        @test_throws ArgumentError CausalForestEstimator(subsample_ratio=0.0)
        @test_throws ArgumentError CausalForestEstimator(subsample_ratio=1.5)
        @test_throws ArgumentError CausalForestEstimator(subsample_ratio=-0.1)
    end

    @testset "Small sample size" begin
        Random.seed!(42)
        n = 100
        X = randn(n, 3)
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        # Should handle small samples (may have larger variance)
        solution = solve(problem, CausalForestEstimator(n_trees=30, min_leaf_size=5))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
        @test isfinite(solution.se)
    end

    @testset "High-dimensional covariates" begin
        data = dgp_high_dimensional(n=500, p=30, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Forest should handle high-dim with mtry regularization
        solution = solve(problem, CausalForestEstimator(n_trees=50, mtry=5))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
    end

    @testset "Imbalanced treatment" begin
        data = dgp_imbalanced_treatment(n=500, true_ate=2.0, treatment_prob=0.15, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Forest should handle imbalanced treatment
        solution = solve(problem, CausalForestEstimator(n_trees=50))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
        # May have larger bias with imbalanced treatment
        @test abs(solution.ate - data.true_ate) < 1.5
    end

    @testset "Strong confounding" begin
        data = dgp_strong_confounding(n=1000, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        # Forest with observables should still estimate effect
        solution = solve(problem, CausalForestEstimator(n_trees=50))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
        # Confounding may bias estimates
        @test abs(solution.ate - data.true_ate) < 2.0
    end

    @testset "Single feature" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 1)  # Only one covariate
        T = rand(n) .< 0.5
        Y = X[:, 1] .+ 2.0 .* T .+ randn(n) * 0.5

        problem = CATEProblem(Y, T, X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(n_trees=30))

        @test solution.retcode == :Success
        @test isfinite(solution.ate)
    end

    @testset "Subsample mode (no bootstrap)" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        solution = solve(problem, CausalForestEstimator(
            n_trees=50,
            bootstrap=false,
            subsample_ratio=0.7
        ))

        @test solution.retcode == :Success
        @test abs(solution.ate - data.true_ate) < 0.8
    end

end


# =============================================================================
# Comparison Tests
# =============================================================================

@testset "Causal Forest vs Meta-Learners" begin

    @testset "Forest comparable to T-learner on constant effect" begin
        data = dgp_constant_effect(n=500, true_ate=2.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_t = solve(problem, TLearner())
        sol_cf = solve(problem, CausalForestEstimator(n_trees=50))

        # Both should recover constant effect reasonably well
        @test abs(sol_t.ate - data.true_ate) < 0.5
        @test abs(sol_cf.ate - data.true_ate) < 0.5
    end

    @testset "Forest captures heterogeneity" begin
        data = dgp_linear_heterogeneity(n=1000, base_effect=2.0, het_coef=1.0, seed=42)
        problem = CATEProblem(data.Y, data.treatment, data.X, (alpha=0.05,))

        sol_cf = solve(problem, CausalForestEstimator(n_trees=100))

        # CATE correlation with true CATE
        cate_corr = cor(sol_cf.cate, data.true_cate)

        # Forest should capture some heterogeneity pattern
        @test cate_corr > 0.2
    end

end


# =============================================================================
# Tree Node Unit Tests
# =============================================================================

@testset "Causal Tree Node Internals" begin

    @testset "Leaf CATE estimation" begin
        # Simple case: clear treatment effect
        Y = [1.0, 1.0, 3.0, 3.0]
        T = [false, false, true, true]

        tau = CausalEstimators._estimate_leaf_tau(Y, T)
        @test tau == 2.0  # 3.0 - 1.0
    end

    @testset "Leaf with no treated" begin
        Y = [1.0, 1.0, 1.0]
        T = [false, false, false]

        tau = CausalEstimators._estimate_leaf_tau(Y, T)
        @test tau == 0.0  # Default for edge case
    end

    @testset "Heterogeneity gain computation" begin
        # Left: τ = 1, Right: τ = 3
        Y_left = [0.0, 1.0, 1.0, 2.0]  # control: 0.5, treat: 1.5
        T_left = [false, false, true, true]

        Y_right = [0.0, 1.0, 3.0, 4.0]  # control: 0.5, treat: 3.5
        T_right = [false, false, true, true]

        gain = CausalEstimators._heterogeneity_gain(Y_left, T_left, Y_right, T_right)

        # gain = (4*4)/(8^2) * (1.0 - 3.0)^2 = 0.25 * 4 = 1.0
        @test gain ≈ 1.0
    end

end
