"""
Tests for IV Stage Decomposition (First Stage, Reduced Form, Second Stage).

Session 56: Tests for stages.jl module.

Test layers:
1. Known-answer tests with analytically computed values
2. Property tests (F-statistic bounds, R² bounds)
3. Consistency tests (Wald identity: γ = π × β)
4. Warning tests (weak IV, naive SE warning)
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics
using Random

@testset "IV Stage Decomposition" begin
    # =========================================================================
    # First Stage Tests
    # =========================================================================
    @testset "FirstStage - Known Answer" begin
        Random.seed!(42)
        n = 1000

        # Strong instrument: D = 2Z + noise
        z = randn(n)
        d = 2.0 * z + 0.5 * randn(n)

        Z = reshape(z, n, 1)
        problem = FirstStageProblem(d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, OLS())

        # Coefficient should be close to 2.0
        @test abs(solution.coef[1] - 2.0) < 0.2
        @test length(solution.coef) == 1
        @test length(solution.se) == 1

        # F-statistic should be large (strong instrument)
        @test solution.f_statistic > 100
        @test solution.f_pvalue < 0.001
        @test !solution.weak_iv_warning

        # R² should be high
        @test solution.r2 > 0.8
        @test solution.partial_r2 > 0.8

        # Fitted values should match
        @test length(solution.fitted_values) == n
        @test length(solution.residuals) == n
    end

    @testset "FirstStage - Weak Instrument Warning" begin
        Random.seed!(123)
        n = 200

        # Weak instrument: D = 0.1Z + noise
        z = randn(n)
        d = 0.1 * z + randn(n)

        Z = reshape(z, n, 1)
        problem = FirstStageProblem(d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, OLS())

        # F-statistic should be low
        @test solution.f_statistic < 10
        @test solution.weak_iv_warning == true
    end

    @testset "FirstStage - With Covariates" begin
        Random.seed!(456)
        n = 500

        # D = 1.5Z + 0.5X + noise
        z = randn(n)
        x = randn(n)
        d = 1.5 * z + 0.5 * x + 0.3 * randn(n)

        Z = reshape(z, n, 1)
        X = reshape(x, n, 1)
        problem = FirstStageProblem(d, Z, X, (alpha=0.05,))
        solution = solve(problem, OLS())

        # Should have coefficients for Z and X
        @test length(solution.coef) == 2
        @test abs(solution.coef[1] - 1.5) < 0.2  # Z coefficient
        @test abs(solution.coef[2] - 0.5) < 0.2  # X coefficient

        # Partial R² should be less than full R²
        @test solution.partial_r2 < solution.r2
        @test solution.partial_r2 > 0.5  # Still substantial
    end

    @testset "FirstStage - Multiple Instruments" begin
        Random.seed!(789)
        n = 500

        # D = Z1 + 0.8Z2 + noise
        z1 = randn(n)
        z2 = randn(n)
        d = 1.0 * z1 + 0.8 * z2 + 0.5 * randn(n)

        Z = hcat(z1, z2)
        problem = FirstStageProblem(d, Z, nothing, (alpha=0.05,))
        solution = solve(problem, OLS())

        @test length(solution.coef) == 2
        @test solution.n_instruments == 2
        @test abs(solution.coef[1] - 1.0) < 0.2
        @test abs(solution.coef[2] - 0.8) < 0.2
    end

    # =========================================================================
    # Reduced Form Tests
    # =========================================================================
    @testset "ReducedForm - Known Answer" begin
        Random.seed!(101)
        n = 1000

        # Y = 3Z + noise
        z = randn(n)
        y = 3.0 * z + randn(n)

        Z = reshape(z, n, 1)
        problem = ReducedFormProblem(y, Z, nothing, (alpha=0.05,))
        solution = solve(problem, OLS())

        # Coefficient should be close to 3.0
        @test abs(solution.coef[1] - 3.0) < 0.2
        @test length(solution.coef) == 1
        @test length(solution.se) == 1
        @test solution.n_instruments == 1

        # R² should be reasonable
        @test 0 < solution.r2 < 1

        # Fitted values and residuals
        @test length(solution.fitted_values) == n
        @test length(solution.residuals) == n
    end

    @testset "ReducedForm - With Covariates" begin
        Random.seed!(202)
        n = 500

        # Y = 2Z + 1.5X + noise
        z = randn(n)
        x = randn(n)
        y = 2.0 * z + 1.5 * x + randn(n)

        Z = reshape(z, n, 1)
        X = reshape(x, n, 1)
        problem = ReducedFormProblem(y, Z, X, (alpha=0.05,))
        solution = solve(problem, OLS())

        @test length(solution.coef) == 2
        @test abs(solution.coef[1] - 2.0) < 0.2  # Z coefficient
        @test abs(solution.coef[2] - 1.5) < 0.2  # X coefficient
    end

    # =========================================================================
    # Second Stage Tests
    # =========================================================================
    @testset "SecondStage - Educational Warning" begin
        Random.seed!(303)
        n = 500

        # Simulate Y, D̂
        d_hat = randn(n)
        y = 2.5 * d_hat + randn(n)

        problem = SecondStageProblem(y, d_hat, nothing, (alpha=0.05,))

        # Should issue warning about naive SEs
        @test_logs (:warn, r"INCORRECT standard errors") solve(problem, OLS())
    end

    @testset "SecondStage - Coefficient Accuracy" begin
        Random.seed!(404)
        n = 1000

        # Y = 3D̂ + noise
        d_hat = randn(n)
        y = 3.0 * d_hat + 0.5 * randn(n)

        problem = SecondStageProblem(y, d_hat, nothing, (alpha=0.05,))
        solution = solve(problem, OLS())

        # Coefficient should be close to 3.0
        @test abs(solution.coef[1] - 3.0) < 0.1
        @test length(solution.coef) == 1
        @test length(solution.se_naive) == 1  # Explicitly naive
    end

    @testset "SecondStage - With Covariates" begin
        Random.seed!(505)
        n = 500

        # Y = 2D̂ + 1X + noise
        d_hat = randn(n)
        x = randn(n)
        y = 2.0 * d_hat + 1.0 * x + randn(n)

        X = reshape(x, n, 1)
        problem = SecondStageProblem(y, d_hat, X, (alpha=0.05,))
        solution = solve(problem, OLS())

        @test length(solution.coef) == 2
        @test abs(solution.coef[1] - 2.0) < 0.2  # D̂ coefficient
        @test abs(solution.coef[2] - 1.0) < 0.2  # X coefficient
    end

    # =========================================================================
    # Wald Identity Test: γ = π × β
    # =========================================================================
    @testset "Wald Estimator Identity" begin
        Random.seed!(606)
        n = 2000

        # DGP: Z → D → Y
        # D = πZ + ν
        # Y = βD + ε
        # Therefore: Y = βπZ + (βν + ε)
        # Reduced form coefficient γ = βπ

        true_pi = 0.8   # First-stage coefficient
        true_beta = 2.0  # Structural coefficient
        true_gamma = true_pi * true_beta  # Reduced form = 1.6

        z = randn(n)
        d = true_pi * z + 0.5 * randn(n)
        y = true_beta * d + randn(n)

        Z = reshape(z, n, 1)

        # First stage
        fs_prob = FirstStageProblem(d, Z, nothing, (alpha=0.05,))
        fs_sol = solve(fs_prob, OLS())

        # Reduced form
        rf_prob = ReducedFormProblem(y, Z, nothing, (alpha=0.05,))
        rf_sol = solve(rf_prob, OLS())

        # Check identity: γ ≈ π × β (using estimated β from Wald)
        pi_hat = fs_sol.coef[1]
        gamma_hat = rf_sol.coef[1]
        beta_wald = gamma_hat / pi_hat

        # Should recover structural effect
        @test abs(beta_wald - true_beta) < 0.2

        # Verify identity
        @test abs(gamma_hat - pi_hat * beta_wald) < 1e-10
    end

    # =========================================================================
    # Problem Type Validation Tests
    # =========================================================================
    @testset "FirstStageProblem - Input Validation" begin
        d = [1.0, 2.0, 3.0]
        Z = reshape([0.5, 1.0, 1.5], 3, 1)

        # Valid construction
        @test FirstStageProblem(d, Z, nothing, (alpha=0.05,)) isa FirstStageProblem

        # Dimension mismatch
        @test_throws ArgumentError FirstStageProblem(
            d, reshape([1.0, 2.0], 2, 1), nothing, (alpha=0.05,)
        )

        # NaN in treatment
        @test_throws ArgumentError FirstStageProblem(
            [1.0, NaN, 3.0], Z, nothing, (alpha=0.05,)
        )

        # Missing alpha
        @test_throws ArgumentError FirstStageProblem(
            d, Z, nothing, (beta=0.1,)
        )
    end

    @testset "ReducedFormProblem - Input Validation" begin
        y = [1.0, 2.0, 3.0]
        Z = reshape([0.5, 1.0, 1.5], 3, 1)

        # Valid construction
        @test ReducedFormProblem(y, Z, nothing, (alpha=0.05,)) isa ReducedFormProblem

        # Dimension mismatch
        @test_throws ArgumentError ReducedFormProblem(
            y, reshape([1.0, 2.0], 2, 1), nothing, (alpha=0.05,)
        )
    end

    @testset "SecondStageProblem - Input Validation" begin
        y = [1.0, 2.0, 3.0]
        d_hat = [0.5, 1.0, 1.5]

        # Valid construction
        @test SecondStageProblem(y, d_hat, nothing, (alpha=0.05,)) isa SecondStageProblem

        # Dimension mismatch
        @test_throws ArgumentError SecondStageProblem(
            y, [1.0, 2.0], nothing, (alpha=0.05,)
        )
    end

    # =========================================================================
    # Display Tests
    # =========================================================================
    @testset "Solution Display" begin
        Random.seed!(707)
        n = 100

        z = randn(n)
        d = 0.8 * z + randn(n)
        y = 2.0 * d + randn(n)
        d_hat = 0.8 * z

        Z = reshape(z, n, 1)

        # First stage display
        fs_sol = solve(FirstStageProblem(d, Z, nothing, (alpha=0.05,)), OLS())
        fs_str = sprint(show, fs_sol)
        @test occursin("FirstStageSolution", fs_str)
        @test occursin("F-statistic", fs_str)

        # Reduced form display
        rf_sol = solve(ReducedFormProblem(y, Z, nothing, (alpha=0.05,)), OLS())
        rf_str = sprint(show, rf_sol)
        @test occursin("ReducedFormSolution", rf_str)

        # Second stage display (suppress warning)
        ss_sol = @test_logs (:warn,) solve(
            SecondStageProblem(y, d_hat, nothing, (alpha=0.05,)), OLS()
        )
        ss_str = sprint(show, ss_sol)
        @test occursin("SecondStageSolution", ss_str)
        @test occursin("WARNING", ss_str)
    end

    # =========================================================================
    # Naive vs Correct SE Comparison
    # =========================================================================
    @testset "Naive SE vs Correct 2SLS SE" begin
        Random.seed!(808)
        n = 1000

        # Generate IV data
        z = randn(n)
        d = 0.7 * z + 0.5 * randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        # Manual two-stage (naive SEs)
        fs_sol = solve(FirstStageProblem(d, Z, nothing, (alpha=0.05,)), OLS())
        ss_sol = @test_logs (:warn,) solve(
            SecondStageProblem(y, fs_sol.fitted_values, nothing, (alpha=0.05,)), OLS()
        )

        # Correct 2SLS
        iv_prob = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        iv_sol = solve(iv_prob, TSLS())

        # Point estimates should match
        @test abs(ss_sol.coef[1] - iv_sol.estimate) < 0.01

        # Naive SEs should be SMALLER (incorrectly so)
        # This demonstrates why naive SEs are problematic
        @test ss_sol.se_naive[1] < iv_sol.se
    end
end
