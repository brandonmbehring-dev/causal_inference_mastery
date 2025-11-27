"""
Tests for RDD sensitivity analysis functions.

Phase 3.6: Comprehensive sensitivity analysis
"""

using Test
using CausalEstimators
using Random
using Statistics
using DataFrames

@testset "Sensitivity Analysis" begin
    @testset "Bandwidth Sensitivity" begin
        # Generate RDD data
        Random.seed!(123)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0
        τ_true = 5.0
        y = 2.0 .* x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Test bandwidth sensitivity
        results = bandwidth_sensitivity(problem, estimator)

        # Should return DataFrame
        @test results isa DataFrame
        @test nrow(results) == 6  # Default: 6 bandwidths

        # Required columns
        @test :bandwidth in names(results)
        @test :estimate in names(results)
        @test :se in names(results)
        @test :ci_lower in names(results)
        @test :ci_upper in names(results)
        @test :p_value in names(results)

        # Estimates should all be reasonably close to true effect
        @test all(abs.(results.estimate .- τ_true) .< 3.0)

        # Custom bandwidths
        custom_bandwidths = [0.3, 0.5, 0.7, 1.0]
        results_custom = bandwidth_sensitivity(problem, estimator, bandwidths=custom_bandwidths)
        @test nrow(results_custom) == length(custom_bandwidths)
        @test results_custom.bandwidth == custom_bandwidths
    end

    @testset "Placebo Test" begin
        Random.seed!(456)
        n = 800
        x = randn(n) .* 2.0
        treatment = x .>= 0.0
        τ_true = 4.0
        # True effect only at cutoff=0
        y = 2.0 .* x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Run placebo test
        results = placebo_test(problem, estimator, n_placebos=10)

        # Should return DataFrame
        @test results isa DataFrame
        @test nrow(results) <= 10  # May be fewer if some fail

        # Required columns
        @test :cutoff in names(results)
        @test :estimate in names(results)
        @test :p_value in names(results)
        @test :significant in names(results)

        # Most placebos should not be significant (expect ~5% false positives)
        false_positives = sum(results.significant)
        @test false_positives <= 3  # At most 30% (conservative given randomness)

        # Custom cutoffs
        custom_cutoffs = [-2.0, -1.0, 1.0, 2.0]
        results_custom = placebo_test(problem, estimator, cutoffs=custom_cutoffs)
        @test nrow(results_custom) <= length(custom_cutoffs)
    end

    @testset "Balance Test" begin
        Random.seed!(789)
        n = 600
        x = randn(n)
        treatment = x .>= 0.0

        # Create covariates that are balanced at cutoff
        # (smooth functions of x, no discontinuity)
        covariates = hcat(
            x .+ 0.2 .* randn(n),           # Age (smooth)
            x.^2 .+ 0.3 .* randn(n),        # Income (smooth)
            sin.(x) .+ 0.1 .* randn(n)      # Education (smooth)
        )

        y = 2.0 .* x .+ 5.0 .* treatment .+ sum(covariates, dims=2)[:] .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))

        # Test balance
        results = balance_test(problem)

        # Should return DataFrame
        @test results isa DataFrame
        @test nrow(results) == 3  # 3 covariates

        # Required columns
        @test :covariate in names(results)
        @test :estimate in names(results)
        @test :p_value in names(results)
        @test :balanced in names(results)

        # All covariates should be balanced (smooth at cutoff)
        # Note: May have 1 false positive due to randomness
        @test sum(results.balanced) >= 2

        # Test error when no covariates
        problem_no_covs = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        @test_throws ArgumentError balance_test(problem_no_covs)
    end

    @testset "Donut RDD" begin
        Random.seed!(321)
        n = 1000
        x = randn(n) .* 2.0
        treatment = x .>= 0.0
        τ_true = 6.0
        y = 2.0 .* x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Baseline estimate
        baseline = solve(problem, estimator)

        # Donut with small hole
        donut_small = donut_rdd(problem, estimator, hole_radius=0.1)

        # Donut with larger hole
        donut_large = donut_rdd(problem, estimator, hole_radius=0.3)

        # Basic checks
        @test donut_small isa RDDSolution
        @test donut_large isa RDDSolution

        # Estimates should still be reasonable
        @test abs(donut_small.estimate - τ_true) < 3.0
        @test abs(donut_large.estimate - τ_true) < 3.0

        # Effective sample sizes should decrease
        @test donut_small.n_eff_left + donut_small.n_eff_right <
              baseline.n_eff_left + baseline.n_eff_right
        @test donut_large.n_eff_left + donut_large.n_eff_right <
              donut_small.n_eff_left + donut_small.n_eff_right

        # Test error for invalid hole_radius
        @test_throws ArgumentError donut_rdd(problem, estimator, hole_radius=0.0)
        @test_throws ArgumentError donut_rdd(problem, estimator, hole_radius=-0.1)
    end

    @testset "Permutation Test" begin
        Random.seed!(111)
        n = 500
        x = randn(n)
        treatment = x .>= 0.0
        τ_true = 3.0
        y = x .+ τ_true .* treatment .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Run permutation test (small n for speed)
        estimate, p_value, null_dist = permutation_test(problem, estimator, n_permutations=100)

        # Basic checks
        @test isfinite(estimate)
        @test 0.0 <= p_value <= 1.0
        @test length(null_dist) == 100

        # Estimate should match solve()
        solution = solve(problem, estimator)
        @test isapprox(estimate, solution.estimate, rtol=0.01)

        # With true effect, should reject null (p < 0.05 usually)
        # Note: May occasionally fail due to randomness
        @test p_value < 0.2  # Conservative threshold

        # Null distribution should be centered near zero
        @test abs(mean(null_dist)) < 2.0
    end

    @testset "Permutation Test - Null Effect" begin
        Random.seed!(222)
        n = 400
        x = randn(n)
        treatment = x .>= 0.0
        # No treatment effect
        y = 2.0 .* x .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Run permutation test
        estimate, p_value, null_dist = permutation_test(problem, estimator, n_permutations=100)

        # Should not reject null (p > 0.05 usually)
        # Note: May occasionally fail due to randomness
        @test p_value > 0.01  # Very conservative
    end

    @testset "Sensitivity Analysis - Integration" begin
        # Test that all sensitivity functions work together
        Random.seed!(999)
        n = 600
        x = randn(n) .* 1.5
        treatment = x .>= 0.0
        covariates = randn(n, 2)
        y = x .+ 4.0 .* treatment .+ sum(covariates, dims=2)[:] .+ randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, covariates, (alpha=0.05,))
        estimator = SharpRDD()

        # Run all sensitivity analyses
        bw_sens = bandwidth_sensitivity(problem, estimator)
        placebo = placebo_test(problem, estimator, n_placebos=5)
        balance = balance_test(problem)
        donut = donut_rdd(problem, estimator, hole_radius=0.1)
        perm_est, perm_p, perm_dist = permutation_test(problem, estimator, n_permutations=50)

        # All should complete without error
        @test bw_sens isa DataFrame
        @test placebo isa DataFrame
        @test balance isa DataFrame
        @test donut isa RDDSolution
        @test isfinite(perm_p)
    end

    @testset "Bandwidth Sensitivity - Edge Cases" begin
        Random.seed!(333)
        n = 200  # Small sample
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Should work with small samples
        results = bandwidth_sensitivity(problem, estimator)
        @test nrow(results) == 6
    end

    @testset "Placebo Test - Edge Cases" begin
        Random.seed!(444)
        n = 300
        x = randn(n)
        treatment = x .>= 0.0
        y = randn(n)

        problem = RDDProblem(y, x, treatment, 0.0, nothing, (alpha=0.05,))
        estimator = SharpRDD()

        # Test with specific cutoffs near true cutoff (should skip)
        cutoffs_near = [-0.05, -0.02, 0.02, 0.05]
        results = placebo_test(problem, estimator, cutoffs=cutoffs_near)

        # Should skip cutoffs too close to true cutoff
        @test nrow(results) < length(cutoffs_near)
    end
end
