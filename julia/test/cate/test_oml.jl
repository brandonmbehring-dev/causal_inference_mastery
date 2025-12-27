#=
Tests for Orthogonal Machine Learning (OML) / Interactive Regression Model

Session 153: OML foundation.

Tests organized by layer:
1. Known-Answer: Verify IRM recovers known treatment effects
2. Adversarial: Edge cases and challenging scenarios
=#

using Test
using Random
using Statistics
using CausalEstimators


"""
    generate_irm_dgp(n, p; true_ate, confounding_strength, noise_sd, seed)

Generate data for IRM testing.

IRM DGP:
- X ~ N(0, I_p)
- e(X) = 1 / (1 + exp(-confounding_strength * X₁))
- T ~ Bernoulli(e(X))
- g0(X) = 1 + X₁
- g1(X) = 1 + X₁ + true_ate
- Y = T * g1(X) + (1-T) * g0(X) + ε
"""
function generate_irm_dgp(;
    n::Int = 500,
    p::Int = 2,
    true_ate::Float64 = 2.0,
    confounding_strength::Float64 = 0.5,
    noise_sd::Float64 = 1.0,
    seed::Int = 42
)
    Random.seed!(seed)

    X = randn(n, p)

    # Propensity with confounding
    propensity = 1.0 ./ (1.0 .+ exp.(-confounding_strength .* X[:, 1]))
    T = [rand() < p for p in propensity]

    # Potential outcomes (IRM structure)
    g0 = 1.0 .+ X[:, 1]
    g1 = 1.0 .+ X[:, 1] .+ true_ate

    # Observed outcome
    noise = randn(n) .* noise_sd
    Y = T .* g1 .+ (1 .- T) .* g0 .+ noise

    return Y, T, X
end


@testset "OML/IRM Tests" begin

    # =========================================================================
    # Layer 1: Known-Answer Tests
    # =========================================================================

    @testset "Layer 1: Known-Answer" begin

        @testset "IRM Constant Effect ATE Recovery" begin
            Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test solution.method == :irm
            @test solution.retcode == :Success
            # ATE should be close to true value
            @test abs(solution.ate - 2.0) < 0.5
        end

        @testset "IRM Heterogeneous Effect CATE Shape" begin
            Random.seed!(42)
            n = 500
            X = randn(n, 2)
            T = rand(n) .> 0.5
            # Heterogeneous effect: τ(X) = 2 + X₁
            true_cate = 2.0 .+ X[:, 1]
            Y = 1.0 .+ X[:, 1] .+ true_cate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test length(solution.cate) == n
            @test all(isfinite.(solution.cate))
            # CATE should have variation
            @test std(solution.cate) > 0.1
        end

        @testset "IRM Returns Valid CI" begin
            Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test solution.ci_lower < solution.ate < solution.ci_upper
            ci_width = solution.ci_upper - solution.ci_lower
            @test 0.1 < ci_width < 5.0
        end

        @testset "IRM SE Positive and Finite" begin
            Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test solution.se > 0
            @test isfinite(solution.se)
        end

        @testset "IRM Different Fold Counts" begin
            Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

            for n_folds in [2, 3, 5]
                problem = CATEProblem(Y, T, X, (alpha=0.05,))
                solution = solve(problem, IRMEstimator(n_folds=n_folds))
                @test abs(solution.ate - 2.0) < 1.0
            end
        end

        @testset "ATTE Equals ATE Under RCT" begin
            Random.seed!(42)
            n = 500
            X = randn(n, 2)
            T = rand(n) .> 0.5  # Pure RCT
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution_ate = solve(problem, IRMEstimator(target=:ate))
            solution_atte = solve(problem, IRMEstimator(target=:atte))

            # Under RCT, ATE and ATTE should be similar
            @test abs(solution_ate.ate - solution_atte.ate) < 0.5
        end

        @testset "ATTE Target Parameter" begin
            Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator(target=:atte))

            @test solution.method == :irm
            # ATTE should be in reasonable range
            @test 0.5 < solution.ate < 4.0
        end

    end

    # =========================================================================
    # Layer 2: Adversarial Tests
    # =========================================================================

    @testset "Layer 2: Adversarial" begin

        @testset "IRM Confounded DGP" begin
            # Strong confounding
            Y, T, X = generate_irm_dgp(
                n=500, true_ate=2.0, confounding_strength=1.0, seed=42
            )

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            # Should still recover ATE (IRM is doubly robust)
            @test abs(solution.ate - 2.0) < 1.0
        end

        @testset "IRM High Dimensional" begin
            Random.seed!(42)
            n = 500
            p = 15
            X = randn(n, p)
            T = rand(n) .> 0.5
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator(model=:ridge))

            @test isfinite(solution.ate)
            @test abs(solution.ate - 2.0) < 1.5
        end

        @testset "IRM Extreme Propensity" begin
            Random.seed!(42)
            n = 500
            X = randn(n, 2)
            # Extreme propensity
            propensity = 1.0 ./ (1.0 .+ exp.(-2.0 .* X[:, 1]))
            T = [rand() < p for p in propensity]
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test isfinite(solution.ate)
            @test isfinite(solution.se)
        end

        @testset "IRM Unbalanced Treatment" begin
            Random.seed!(42)
            n = 500
            X = randn(n, 2)
            # 90% control, 10% treated
            T = rand(n) .> 0.9
            Y = 1.0 .+ X[:, 1] .+ 2.0 .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator())

            @test isfinite(solution.ate)
        end

        @testset "IRM Small Sample" begin
            Y, T, X = generate_irm_dgp(n=50, true_ate=2.0, seed=42)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))
            solution = solve(problem, IRMEstimator(n_folds=2))

            @test isfinite(solution.ate)
        end

        @testset "Invalid n_folds" begin
            @test_throws ArgumentError IRMEstimator(n_folds=1)
        end

        @testset "Invalid Target" begin
            @test_throws ArgumentError IRMEstimator(target=:invalid)
        end

        @testset "Invalid Model" begin
            @test_throws ArgumentError IRMEstimator(model=:invalid)
        end

    end

    # =========================================================================
    # Comparison with DML
    # =========================================================================

    @testset "IRM vs DML Comparison" begin

        @testset "IRM and DML Similar on Linear DGP" begin
            Random.seed!(42)
            n = 400
            X = randn(n, 3)
            T = rand(n) .> 0.5
            true_ate = 2.0
            # PLR-compatible DGP
            Y = 1.0 .+ X[:, 1] .+ true_ate .* T .+ randn(n)

            problem = CATEProblem(Y, T, X, (alpha=0.05,))

            irm_sol = solve(problem, IRMEstimator())
            dml_sol = solve(problem, DoubleMachineLearning())

            # Both should recover ATE
            @test abs(irm_sol.ate - true_ate) < 1.0
            @test abs(dml_sol.ate - true_ate) < 1.0

            # They should give similar results
            @test abs(irm_sol.ate - dml_sol.ate) < 0.5
        end

    end

end
