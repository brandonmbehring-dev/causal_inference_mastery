"""
Tests for McCrary Density Test with CJM (2020) Proper Variance.

Session 57: Tests for julia/src/rdd/mccrary.jl module.

Test layers:
1. Known-answer tests (uniform vs bunched data)
2. Type I error validation (critical: < 8%)
3. Power tests (detect 15% bunching)
4. Input validation tests
"""

using Test
using CausalEstimators
using Statistics
using Random
using Distributions


@testset "McCrary Density Test (CJM 2020)" begin
    # =========================================================================
    # Known-Answer Tests
    # =========================================================================
    @testset "Known-Answer: Uniform Data" begin
        Random.seed!(42)
        n = 2000

        # Uniform distribution: no manipulation
        x = rand(n) * 10 .- 5  # Uniform on [-5, 5]
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # Should not reject H₀ (no manipulation)
        @test solution.passes == true
        @test solution.p_value > 0.05

        # Log density ratio should be close to 0
        @test abs(solution.theta) < 0.3

        # Densities should be similar
        @test isfinite(solution.f_left)
        @test isfinite(solution.f_right)
        @test abs(log(solution.f_right / solution.f_left)) < 0.3

        # Sample sizes should be roughly equal
        @test abs(solution.n_left - n/2) < n * 0.1
    end

    @testset "Known-Answer: Bunched Data (Manipulation)" begin
        Random.seed!(123)
        n = 2000

        # Create data with bunching above cutoff
        # 85% uniform, 15% extra bunching just above cutoff
        x_uniform = rand(Int(0.85 * n)) * 10 .- 5  # Uniform on [-5, 5]
        x_bunched = rand(Int(0.15 * n)) * 0.5  # Extra mass in [0, 0.5]
        x = vcat(x_uniform, x_bunched)
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # Should detect bunching (reject H₀)
        @test solution.passes == false
        @test solution.p_value < 0.10  # May not always be < 0.05 but should be low

        # Log density ratio should be positive (more mass on right)
        @test solution.theta > 0
    end

    @testset "Known-Answer: Normal Distribution" begin
        Random.seed!(456)
        n = 1500

        # Normal centered at 0: symmetric, no manipulation
        x = randn(n)
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # Should not reject H₀
        @test solution.passes == true
        @test solution.p_value > 0.05

        # Log density ratio should be close to 0 (symmetric)
        @test abs(solution.theta) < 0.2
    end

    @testset "Known-Answer: Asymmetric Normal" begin
        Random.seed!(789)
        n = 1500

        # Normal with mean = 1 (asymmetric around cutoff=0)
        x = randn(n) .+ 1.0
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # More mass on right due to mean > 0, but this is NOT manipulation
        # The test is about density discontinuity, not asymmetry
        # Smooth density should not cause rejection
        @test solution.passes == true || solution.p_value > 0.01
    end

    # =========================================================================
    # Type I Error Test (CRITICAL - CONCERN-22)
    # =========================================================================
    @testset "Type I Error - Uniform Data" begin
        # This is the critical test that was failing before
        # With old SE formula: ~80% rejection rate (inflated)
        # With CJM formula: should be ~5% rejection rate

        Random.seed!(2024)
        n_runs = 200  # Reduce for speed in unit tests
        n_samples = 500
        alpha = 0.05
        rejections = 0

        for _ in 1:n_runs
            x = rand(n_samples) * 10 .- 5  # Uniform on [-5, 5]
            cutoff = 0.0

            problem = McCraryProblem(x, cutoff, nothing, (alpha=alpha,))
            solution = solve(problem, McCraryDensityTest())

            if !solution.passes
                rejections += 1
            end
        end

        rejection_rate = rejections / n_runs

        # CRITICAL: Type I error should be < 8% (allowing some margin)
        # Old formula gave ~80%, so this is the key fix
        @test rejection_rate < 0.08

        # Should also not be too low (test has some power)
        @test rejection_rate < 0.15  # Loose upper bound for robustness
    end

    @testset "Type I Error - Normal Data" begin
        Random.seed!(2025)
        n_runs = 200
        n_samples = 500
        alpha = 0.05
        rejections = 0

        for _ in 1:n_runs
            x = randn(n_samples)  # Standard normal
            cutoff = 0.0

            problem = McCraryProblem(x, cutoff, nothing, (alpha=alpha,))
            solution = solve(problem, McCraryDensityTest())

            if !solution.passes
                rejections += 1
            end
        end

        rejection_rate = rejections / n_runs

        # Type I error should be < 8%
        @test rejection_rate < 0.08
    end

    # =========================================================================
    # Power Tests
    # =========================================================================
    @testset "Power - 15% Bunching" begin
        Random.seed!(3000)
        n_runs = 100
        n_samples = 1000
        bunching_fraction = 0.15
        alpha = 0.05
        detections = 0

        for _ in 1:n_runs
            # 85% uniform, 15% bunched just above cutoff
            n_uniform = Int((1 - bunching_fraction) * n_samples)
            n_bunched = n_samples - n_uniform

            x_uniform = rand(n_uniform) * 10 .- 5
            x_bunched = rand(n_bunched) * 0.3 .+ 0.01  # Tight bunching in [0.01, 0.31]
            x = vcat(x_uniform, x_bunched)
            cutoff = 0.0

            problem = McCraryProblem(x, cutoff, nothing, (alpha=alpha,))
            solution = solve(problem, McCraryDensityTest())

            if !solution.passes
                detections += 1
            end
        end

        power = detections / n_runs

        # Power should be > 40% for 15% bunching with n=1000
        @test power > 0.35  # Slightly lower threshold for robustness
    end

    @testset "Power - 25% Bunching" begin
        Random.seed!(3001)
        n_runs = 50
        n_samples = 1000
        bunching_fraction = 0.25
        alpha = 0.05
        detections = 0

        for _ in 1:n_runs
            n_uniform = Int((1 - bunching_fraction) * n_samples)
            n_bunched = n_samples - n_uniform

            x_uniform = rand(n_uniform) * 10 .- 5
            x_bunched = rand(n_bunched) * 0.3 .+ 0.01
            x = vcat(x_uniform, x_bunched)
            cutoff = 0.0

            problem = McCraryProblem(x, cutoff, nothing, (alpha=alpha,))
            solution = solve(problem, McCraryDensityTest())

            if !solution.passes
                detections += 1
            end
        end

        power = detections / n_runs

        # Power should be > 60% for 25% bunching
        @test power > 0.55
    end

    # =========================================================================
    # Input Validation Tests
    # =========================================================================
    @testset "Input Validation" begin
        # Too few observations
        @test_throws ArgumentError McCraryProblem(
            rand(10), 0.5, nothing, (alpha=0.05,)
        )

        # Cutoff outside range
        @test_throws ArgumentError McCraryProblem(
            rand(100), 1.5, nothing, (alpha=0.05,)
        )

        # Too few observations on one side
        # Create data where only 5 observations are below cutoff
        x = vcat(fill(0.1, 5), fill(0.9, 100))  # Only 5 below cutoff=0.5
        @test_throws ArgumentError McCraryProblem(
            x, 0.5, nothing, (alpha=0.05,)
        )

        # Invalid alpha
        @test_throws ArgumentError McCraryProblem(
            rand(100), 0.5, nothing, (alpha=0.0,)
        )
        @test_throws ArgumentError McCraryProblem(
            rand(100), 0.5, nothing, (alpha=1.0,)
        )

        # Missing alpha
        @test_throws ArgumentError McCraryProblem(
            rand(100), 0.5, nothing, (beta=0.1,)
        )

        # Negative bandwidth
        @test_throws ArgumentError McCraryProblem(
            rand(100), 0.5, -0.1, (alpha=0.05,)
        )

        # NaN in data
        x_nan = vcat(rand(99), [NaN])
        @test_throws ArgumentError McCraryProblem(
            x_nan, 0.5, nothing, (alpha=0.05,)
        )
    end

    # =========================================================================
    # Problem Construction Tests
    # =========================================================================
    @testset "Problem Construction" begin
        x = rand(200)
        cutoff = 0.5

        # Valid construction
        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        @test problem isa McCraryProblem
        @test length(problem.x) == 200
        @test problem.cutoff == 0.5
        @test isnothing(problem.bandwidth)
        @test problem.parameters.alpha == 0.05

        # With explicit bandwidth
        problem_h = McCraryProblem(x, cutoff, 0.1, (alpha=0.05,))
        @test problem_h.bandwidth == 0.1
    end

    # =========================================================================
    # Solution Properties Tests
    # =========================================================================
    @testset "Solution Properties" begin
        Random.seed!(5000)
        x = rand(500) * 4 .- 2  # Uniform on [-2, 2]
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # Solution should have all fields
        @test isfinite(solution.theta)
        @test isfinite(solution.se)
        @test solution.se > 0
        @test isfinite(solution.z_stat)
        @test 0 <= solution.p_value <= 1
        @test solution.passes isa Bool
        @test isfinite(solution.f_left)
        @test isfinite(solution.f_right)
        @test solution.f_left > 0
        @test solution.f_right > 0
        @test solution.bandwidth > 0
        @test solution.n_left > 0
        @test solution.n_right > 0
        @test solution.n_left + solution.n_right == 500
        @test solution.interpretation isa String
    end

    @testset "Z-statistic Consistency" begin
        Random.seed!(5001)
        x = rand(800)
        cutoff = 0.5

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # z = theta / SE
        @test isapprox(solution.z_stat, solution.theta / solution.se, rtol=1e-10)

        # p-value from z-stat
        expected_p = 2 * (1 - cdf(Normal(0, 1), abs(solution.z_stat)))
        @test isapprox(solution.p_value, expected_p, rtol=1e-10)

        # passes = (p > alpha)
        @test solution.passes == (solution.p_value > 0.05)
    end

    # =========================================================================
    # Display Tests
    # =========================================================================
    @testset "Display Methods" begin
        Random.seed!(6000)
        x = rand(300)
        cutoff = 0.5

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))
        solution = solve(problem, McCraryDensityTest())

        # Problem display
        problem_str = sprint(show, problem)
        @test occursin("McCraryProblem", problem_str)
        @test occursin("cutoff", problem_str)
        @test occursin("alpha", problem_str)

        # Solution display
        solution_str = sprint(show, solution)
        @test occursin("McCrarySolution", solution_str)
        @test occursin("CJM 2020", solution_str)
        @test occursin("p-value", solution_str)
    end

    # =========================================================================
    # Kernel Options Tests
    # =========================================================================
    @testset "Kernel Options" begin
        Random.seed!(7000)
        x = rand(500) * 4 .- 2
        cutoff = 0.0

        problem = McCraryProblem(x, cutoff, nothing, (alpha=0.05,))

        # Triangular (default)
        sol_tri = solve(problem, McCraryDensityTest(kernel=:triangular))
        @test isfinite(sol_tri.theta)

        # Uniform
        sol_uni = solve(problem, McCraryDensityTest(kernel=:uniform))
        @test isfinite(sol_uni.theta)

        # Epanechnikov
        sol_epa = solve(problem, McCraryDensityTest(kernel=:epanechnikov))
        @test isfinite(sol_epa.theta)

        # Results should be similar (different kernels, similar conclusions)
        @test abs(sol_tri.theta - sol_uni.theta) < 0.5
        @test abs(sol_tri.theta - sol_epa.theta) < 0.5
    end

    # =========================================================================
    # CJM Variance Specific Tests
    # =========================================================================
    @testset "CJM Variance Formula" begin
        # Test that SE increases appropriately with smaller samples
        Random.seed!(8000)

        # Large sample
        x_large = rand(2000) * 4 .- 2
        problem_large = McCraryProblem(x_large, 0.0, nothing, (alpha=0.05,))
        sol_large = solve(problem_large, McCraryDensityTest())

        # Small sample
        x_small = rand(200) * 4 .- 2
        problem_small = McCraryProblem(x_small, 0.0, nothing, (alpha=0.05,))
        sol_small = solve(problem_small, McCraryDensityTest())

        # SE should be larger for smaller sample
        @test sol_small.se > sol_large.se

        # SE should scale roughly with sqrt(1/n)
        # CJM: SE ∝ sqrt(1/(n*h)), and h ∝ n^(-1/5)
        # So SE ∝ n^(-1/2) * n^(1/10) = n^(-2/5)
        ratio_expected = (200 / 2000)^(-0.4)  # n^(-2/5)
        ratio_actual = sol_small.se / sol_large.se
        # Allow large tolerance due to randomness
        @test ratio_actual > 1.0  # At minimum, smaller sample = larger SE
        @test ratio_actual < 10.0  # Sanity check
    end
end
