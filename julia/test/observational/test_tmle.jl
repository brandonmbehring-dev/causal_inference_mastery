#=
Unit Tests for Targeted Maximum Likelihood Estimation (TMLE)

Tests cover:
1. TMLESolution construction and display
2. TMLE helper functions (clever covariate, fluctuation, convergence)
3. TMLE estimator with known-answer DGPs
4. Double robustness property verification
5. TMLE vs DR comparison (efficiency)
6. Convergence properties
7. Edge cases and error handling
=#

using Test
using Statistics
using Random
using CausalEstimators


# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate observational data with linear outcomes (correct specification)."""
function generate_tmle_linear_data(;
    n::Int = 500,
    true_ate::Float64 = 2.0,
    confounding_strength::Float64 = 0.5,
    seed::Int = 42
)
    Random.seed!(seed)

    # Covariates
    X = randn(n, 2)

    # Propensity: depends on X
    logit = confounding_strength .* X[:, 1] .+ 0.3 .* X[:, 2]
    e_true = 1 ./ (1 .+ exp.(-logit))

    # Treatment assignment
    T = rand(n) .< e_true

    # Outcome: LINEAR in X (outcome model correctly specified)
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


"""Generate data with quadratic outcome (misspecified outcome model)."""
function generate_tmle_quadratic_outcome(;
    n::Int = 500,
    true_ate::Float64 = 2.5,
    seed::Int = 123
)
    Random.seed!(seed)

    X = randn(n, 2)

    # Linear propensity (correctly specified)
    logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # QUADRATIC outcome (misspecified if using linear model)
    Y = true_ate .* T .+ 0.5 .* X[:, 1].^2 .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


"""Generate data with misspecified propensity (non-linear selection)."""
function generate_tmle_nonlinear_propensity(;
    n::Int = 500,
    true_ate::Float64 = 3.0,
    seed::Int = 456
)
    Random.seed!(seed)

    X = randn(n, 2)

    # NON-LINEAR propensity (misspecified if using logistic)
    logit = 0.5 .* X[:, 1].^2 .+ 0.3 .* sin.(X[:, 2])
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true

    # Linear outcome (correctly specified)
    Y = true_ate .* T .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (Y = Y, T = T, X = X, e_true = e_true, true_ate = true_ate)
end


# =============================================================================
# TMLE Constructor Tests
# =============================================================================

@testset "TMLE Constructor" begin
    @testset "Default parameters" begin
        tmle = TMLE()
        @test tmle.max_iter == 100
        @test tmle.tol == 1e-6
    end

    @testset "Custom parameters" begin
        tmle = TMLE(max_iter = 50, tol = 1e-8)
        @test tmle.max_iter == 50
        @test tmle.tol == 1e-8
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError TMLE(max_iter = 0)
        @test_throws ArgumentError TMLE(max_iter = -1)
        @test_throws ArgumentError TMLE(tol = 0.0)
        @test_throws ArgumentError TMLE(tol = -1e-6)
    end
end


# =============================================================================
# TMLE Known-Answer Tests
# =============================================================================

@testset "TMLE Known-Answer" begin
    @testset "Linear DGP (both models correct)" begin
        data = generate_tmle_linear_data(n = 1000, true_ate = 2.0, seed = 10)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # Should recover true ATE well
        @test abs(solution.estimate - data.true_ate) < 0.4
        @test solution.se > 0
        @test isfinite(solution.se)
        @test solution.ci_lower < solution.ci_upper
        @test solution.retcode == :Success
    end

    @testset "Solution contains TMLE-specific fields" begin
        data = generate_tmle_linear_data(n = 500, seed = 11)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # TMLE-specific fields
        @test isfinite(solution.epsilon)
        @test solution.n_iterations >= 1
        @test solution.n_iterations <= 100
        @test typeof(solution.converged) == Bool
        @test isfinite(solution.convergence_criterion)
    end

    @testset "Targeted predictions exist" begin
        data = generate_tmle_linear_data(n = 500, seed = 12)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # Initial and targeted predictions
        @test length(solution.Q0_initial) > 0
        @test length(solution.Q1_initial) > 0
        @test length(solution.Q0_star) > 0
        @test length(solution.Q1_star) > 0
        @test all(isfinite, solution.Q0_star)
        @test all(isfinite, solution.Q1_star)
    end

    @testset "EIF computed" begin
        data = generate_tmle_linear_data(n = 500, seed = 13)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # Efficient influence function
        @test length(solution.eif) > 0
        @test all(isfinite, solution.eif)
        # EIF should be mean-zero (approximately)
        @test abs(mean(solution.eif)) < 0.1
    end

    @testset "Zero treatment effect" begin
        Random.seed!(14)
        n = 500
        X = randn(n, 2)
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 0.5 .* X[:, 1] .+ randn(n)  # No treatment effect

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, TMLE())

        # Should be close to zero
        @test abs(solution.estimate) < 0.5
        # CI should contain zero
        @test solution.ci_lower < 0 < solution.ci_upper
    end
end


# =============================================================================
# Convergence Tests
# =============================================================================

@testset "TMLE Convergence" begin
    @testset "Converges in few iterations" begin
        data = generate_tmle_linear_data(n = 500, seed = 20)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # Should converge quickly for linear DGP
        @test solution.n_iterations < 20
    end

    @testset "Always converges (simple DGP)" begin
        # Test multiple seeds
        converged_count = 0
        for seed in 100:109
            data = generate_tmle_linear_data(n = 300, seed = seed)
            problem = ObservationalProblem(data.Y, data.T, data.X)
            solution = solve(problem, TMLE())

            if solution.converged
                converged_count += 1
            end
        end

        # Should converge in vast majority
        @test converged_count >= 9
    end

    @testset "Convergence criterion near zero" begin
        data = generate_tmle_linear_data(n = 500, seed = 21)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        if solution.converged
            @test abs(solution.convergence_criterion) < 1e-4
        end
    end
end


# =============================================================================
# Double Robustness Tests
# =============================================================================

@testset "TMLE Double Robustness" begin
    @testset "Both models correct - efficient" begin
        data = generate_tmle_linear_data(n = 1000, true_ate = 2.0, seed = 30)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # Best case: should recover ATE very well
        @test abs(solution.estimate - data.true_ate) < 0.3
    end

    @testset "Propensity correct, outcome misspecified" begin
        data = generate_tmle_quadratic_outcome(n = 1000, true_ate = 2.5, seed = 31)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # TMLE should still be reasonably close (IPW component)
        @test abs(solution.estimate - data.true_ate) < 0.8
    end

    @testset "Outcome correct, propensity misspecified" begin
        data = generate_tmle_nonlinear_propensity(n = 1000, true_ate = 3.0, seed = 32)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        # TMLE should still work (outcome regression component)
        @test abs(solution.estimate - data.true_ate) < 0.8
    end
end


# =============================================================================
# TMLE vs DR Comparison
# =============================================================================

@testset "TMLE vs DR Comparison" begin
    @testset "Similar point estimates" begin
        data = generate_tmle_linear_data(n = 800, seed = 40)
        problem = ObservationalProblem(data.Y, data.T, data.X)

        tmle_sol = solve(problem, TMLE())
        dr_sol = solve(problem, DoublyRobust())

        # Point estimates should be similar
        @test abs(tmle_sol.estimate - dr_sol.estimate) < 0.3
    end

    @testset "TMLE typically similar or lower SE" begin
        # Run multiple seeds and compare SE
        tmle_ses = Float64[]
        dr_ses = Float64[]

        for seed in 50:54
            data = generate_tmle_linear_data(n = 500, seed = seed)
            problem = ObservationalProblem(data.Y, data.T, data.X)

            tmle_sol = solve(problem, TMLE())
            dr_sol = solve(problem, DoublyRobust())

            push!(tmle_ses, tmle_sol.se)
            push!(dr_ses, dr_sol.se)
        end

        # TMLE should be comparable or better
        @test mean(tmle_ses) <= mean(dr_ses) + 0.05
    end
end


# =============================================================================
# Trimming Tests
# =============================================================================

@testset "TMLE Trimming" begin
    # Generate data with strong confounding (extreme propensities)
    Random.seed!(60)
    n = 600
    X = randn(n, 2)
    logit = 1.5 .* X[:, 1] .+ 1.0 .* X[:, 2]  # Strong confounding
    e_true = 1 ./ (1 .+ exp.(-logit))
    T = rand(n) .< e_true
    Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

    @testset "No trimming" begin
        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.0)
        solution = solve(problem, TMLE())

        @test solution.n_trimmed == 0
        @test solution.n_treated + solution.n_control == n
    end

    @testset "With trimming" begin
        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.05)
        solution = solve(problem, TMLE())

        @test solution.n_trimmed >= 0
        @test solution.n_treated + solution.n_control <= n
    end
end


# =============================================================================
# Pre-computed Propensity Tests
# =============================================================================

@testset "TMLE Pre-computed Propensity" begin
    data = generate_tmle_linear_data(n = 500, seed = 70)

    @testset "Using true propensity (oracle)" begin
        problem = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution = solve(problem, TMLE())

        # Should recover ATE well with oracle propensity
        @test abs(solution.estimate - data.true_ate) < 0.4
    end

    @testset "Oracle vs estimated propensity" begin
        problem_est = ObservationalProblem(data.Y, data.T, data.X)
        solution_est = solve(problem_est, TMLE())

        problem_oracle = ObservationalProblem(
            data.Y, data.T, data.X;
            propensity = data.e_true
        )
        solution_oracle = solve(problem_oracle, TMLE())

        # Both should be reasonable
        @test abs(solution_est.estimate - data.true_ate) < 0.6
        @test abs(solution_oracle.estimate - data.true_ate) < 0.4
    end
end


# =============================================================================
# Edge Cases
# =============================================================================

@testset "TMLE Edge Cases" begin
    @testset "Small sample size" begin
        data = generate_tmle_linear_data(n = 50, seed = 80)

        problem = ObservationalProblem(data.Y, data.T, data.X)
        solution = solve(problem, TMLE())

        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
    end

    @testset "Single covariate" begin
        Random.seed!(81)
        n = 200
        X = randn(n, 1)  # Single covariate
        logit = 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, TMLE())

        @test isfinite(solution.estimate)
        @test abs(solution.estimate - 2.0) < 1.0
    end

    @testset "Many covariates" begin
        Random.seed!(82)
        n = 500
        p = 10
        X = randn(n, p)
        logit = 0.3 .* sum(X[:, 1:3], dims=2)[:]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ 0.2 .* sum(X[:, 1:3], dims=2)[:] .+ randn(n)

        problem = ObservationalProblem(Y, T, X)
        solution = solve(problem, TMLE())

        @test isfinite(solution.estimate)
        @test solution.se > 0
    end

    @testset "Imbalanced treatment" begin
        Random.seed!(83)
        n = 500
        X = randn(n, 2)
        # Strong bias toward treatment
        logit = 1.0 .+ 0.5 .* X[:, 1]
        T = rand(n) .< (1 ./ (1 .+ exp.(-logit)))
        Y = 2.0 .* T .+ X[:, 1] .+ randn(n)

        problem = ObservationalProblem(Y, T, X; trim_threshold = 0.02)
        solution = solve(problem, TMLE())

        @test isfinite(solution.estimate)
        @test solution.n_treated > solution.n_control  # Expected imbalance
    end
end


# =============================================================================
# Solution Display
# =============================================================================

@testset "TMLESolution Display" begin
    data = generate_tmle_linear_data(n = 200, seed = 90)
    problem = ObservationalProblem(data.Y, data.T, data.X)
    solution = solve(problem, TMLE())

    # Test that show method works without error
    io = IOBuffer()
    show(io, solution)
    output = String(take!(io))

    @test contains(output, "TMLESolution")
    @test contains(output, "ATE Estimate")
    @test contains(output, "Epsilon")
    @test contains(output, "Converged")
    @test contains(output, "Iterations")
end
