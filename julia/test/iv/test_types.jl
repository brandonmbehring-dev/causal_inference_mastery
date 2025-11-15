"""
Tests for IV types and validation.

Phase 4.1: Foundation & Types
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics

@testset "IV Types" begin
    @testset "IVProblem - Basic Construction" begin
        n = 100
        y = randn(n)
        d = randn(n)
        z = randn(n)  # Single instrument
        Z = reshape(z, n, 1)  # Convert to matrix

        # Valid construction
        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        @test problem isa IVProblem{Float64}
        @test length(problem.outcomes) == n
        @test length(problem.treatment) == n
        @test size(problem.instruments) == (n, 1)
        @test isnothing(problem.covariates)
        @test problem.parameters.alpha == 0.05
    end

    @testset "IVProblem - Multiple Instruments" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 3)  # 3 instruments

        problem = IVProblem(y, d, Z, nothing, (alpha=0.05,))

        @test size(problem.instruments) == (n, 3)
        @test problem isa IVProblem{Float64}
    end

    @testset "IVProblem - With Covariates" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 2)
        X = randn(n, 3)  # 3 covariates

        problem = IVProblem(y, d, Z, X, (alpha=0.05,))

        @test !isnothing(problem.covariates)
        @test size(problem.covariates) == (n, 3)
    end

    @testset "IVProblem - Dimension Validation" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 2)

        # Treatment wrong length
        d_wrong = randn(n - 10)
        @test_throws ArgumentError IVProblem(y, d_wrong, Z, nothing, (alpha=0.05,))

        # Instruments wrong rows
        Z_wrong = randn(n - 10, 2)
        @test_throws ArgumentError IVProblem(y, d, Z_wrong, nothing, (alpha=0.05,))

        # Covariates wrong rows
        X_wrong = randn(n - 10, 3)
        @test_throws ArgumentError IVProblem(y, d, Z, X_wrong, (alpha=0.05,))
    end

    @testset "IVProblem - NaN/Inf Validation" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 2)

        # NaN in outcomes
        y_nan = copy(y)
        y_nan[1] = NaN
        @test_throws ArgumentError IVProblem(y_nan, d, Z, nothing, (alpha=0.05,))

        # Inf in treatment
        d_inf = copy(d)
        d_inf[1] = Inf
        @test_throws ArgumentError IVProblem(y, d_inf, Z, nothing, (alpha=0.05,))

        # NaN in instruments
        Z_nan = copy(Z)
        Z_nan[1, 1] = NaN
        @test_throws ArgumentError IVProblem(y, d, Z_nan, nothing, (alpha=0.05,))

        # Inf in covariates
        X = randn(n, 2)
        X_inf = copy(X)
        X_inf[1, 1] = Inf
        @test_throws ArgumentError IVProblem(y, d, Z, X_inf, (alpha=0.05,))
    end

    @testset "IVProblem - Instrument Count Validation" begin
        n = 100
        y = randn(n)
        d = randn(n)

        # No instruments (K=0)
        Z_empty = zeros(n, 0)
        @test_throws ArgumentError IVProblem(y, d, Z_empty, nothing, (alpha=0.05,))

        # Single instrument (K=1) - should work
        Z_one = randn(n, 1)
        problem = IVProblem(y, d, Z_one, nothing, (alpha=0.05,))
        @test problem isa IVProblem

        # Many instruments (K=10) - should work
        Z_many = randn(n, 10)
        problem = IVProblem(y, d, Z_many, nothing, (alpha=0.05,))
        @test problem isa IVProblem
    end

    @testset "IVProblem - Alpha Validation" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 2)

        # Missing alpha
        @test_throws ArgumentError IVProblem(y, d, Z, nothing, (beta=0.1,))

        # Alpha out of range
        @test_throws ArgumentError IVProblem(y, d, Z, nothing, (alpha=0.0,))
        @test_throws ArgumentError IVProblem(y, d, Z, nothing, (alpha=1.0,))
        @test_throws ArgumentError IVProblem(y, d, Z, nothing, (alpha=-0.05,))
        @test_throws ArgumentError IVProblem(y, d, Z, nothing, (alpha=1.5,))

        # Valid alpha values
        for alpha in [0.01, 0.05, 0.10]
            problem = IVProblem(y, d, Z, nothing, (alpha=alpha,))
            @test problem.parameters.alpha == alpha
        end
    end

    @testset "IVProblem - Type Stability" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 2)

        # Float64 construction
        problem_f64 = IVProblem(y, d, Z, nothing, (alpha=0.05,))
        @test problem_f64 isa IVProblem{Float64}

        # Float32 construction
        y_f32 = Float32.(y)
        d_f32 = Float32.(d)
        Z_f32 = Float32.(Z)
        problem_f32 = IVProblem(y_f32, d_f32, Z_f32, nothing, (alpha=0.05f0,))
        @test problem_f32 isa IVProblem{Float32}
    end

    @testset "IVSolution - Construction" begin
        solution = IVSolution(
            5.0,        # estimate
            1.0,        # se
            3.0,        # ci_lower
            7.0,        # ci_upper
            0.001,      # p_value
            100,        # n
            2,          # n_instruments
            3,          # n_covariates
            25.0,       # first_stage_fstat
            0.5,        # overid_pvalue
            false,      # weak_iv_warning
            "2SLS",     # estimator_name
            0.05,       # alpha
            (cragg_donald=20.0,)  # diagnostics
        )

        @test solution isa IVSolution{Float64}
        @test solution.estimate == 5.0
        @test solution.se == 1.0
        @test solution.ci_lower == 3.0
        @test solution.ci_upper == 7.0
        @test solution.p_value == 0.001
        @test solution.n == 100
        @test solution.n_instruments == 2
        @test solution.n_covariates == 3
        @test solution.first_stage_fstat == 25.0
        @test solution.overid_pvalue == 0.5
        @test solution.weak_iv_warning == false
        @test solution.estimator_name == "2SLS"
        @test solution.alpha == 0.05
        @test solution.diagnostics.cragg_donald == 20.0
    end

    @testset "IVSolution - Weak IV Warning" begin
        # Strong IV (F > 10)
        solution_strong = IVSolution(
            5.0, 1.0, 3.0, 7.0, 0.001,
            100, 2, 0, 25.0, nothing, false, "2SLS", 0.05, (;)
        )
        @test solution_strong.weak_iv_warning == false

        # Weak IV (F < 10)
        solution_weak = IVSolution(
            5.0, 1.0, 3.0, 7.0, 0.001,
            100, 2, 0, 5.0, nothing, true, "2SLS", 0.05, (;)
        )
        @test solution_weak.weak_iv_warning == true
    end

    @testset "IVProblem - Display" begin
        n = 100
        y = randn(n)
        d = randn(n)
        Z = randn(n, 3)
        X = randn(n, 2)

        problem = IVProblem(y, d, Z, X, (alpha=0.05,))

        # Test that show() doesn't error
        io = IOBuffer()
        show(io, problem)
        output = String(take!(io))

        @test occursin("IVProblem", output)
        @test occursin("100", output)  # Observations
        @test occursin("3", output)    # Instruments
        @test occursin("2", output)    # Covariates
    end

    @testset "IVSolution - Display" begin
        solution = IVSolution(
            5.0, 1.0, 3.0, 7.0, 0.001,
            100, 2, 0, 25.0, 0.5, false, "2SLS", 0.05, (;)
        )

        io = IOBuffer()
        show(io, solution)
        output = String(take!(io))

        @test occursin("IVSolution", output)
        @test occursin("2SLS", output)
        @test occursin("5.0", output)  # Estimate
        @test occursin("1.0", output)  # SE
        @test occursin("25.0", output) # First-stage F
    end

    @testset "IVSolution - Display with Weak IV Warning" begin
        solution = IVSolution(
            5.0, 1.0, 3.0, 7.0, 0.001,
            100, 2, 0, 5.0, nothing, true, "2SLS", 0.05, (;)
        )

        io = IOBuffer()
        show(io, solution)
        output = String(take!(io))

        @test occursin("Weak instruments", output)
    end
end
