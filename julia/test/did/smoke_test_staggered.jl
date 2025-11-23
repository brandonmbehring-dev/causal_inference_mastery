"""
Smoke test for staggered DiD implementations.

Tests basic functionality of StaggeredTWFE, CallawaySantAnna, and SunAbraham.
"""

using Test
using CausalEstimators
using Random

@testset "Staggered DiD Smoke Tests" begin

    # Create simple synthetic data
    # 3 cohorts (treated at t=2, t=3, never), 2 units per cohort, 4 periods
    Random.seed!(123)

    n_periods = 4
    n_units_per_cohort = 2

    # Unit IDs: 0-1 (cohort 2), 2-3 (cohort 3), 4-5 (never-treated)
    unit_id = repeat(0:5, inner=n_periods)

    # Time periods: 0, 1, 2, 3
    time = repeat(0:(n_periods-1), outer=6)

    # Treatment time per unit
    treatment_time = [
        2.0, 2.0,  # Units 0-1: treated at t=2
        3.0, 3.0,  # Units 2-3: treated at t=3
        Inf, Inf   # Units 4-5: never-treated
    ]
    treatment_time_full = repeat(treatment_time, inner=n_periods)

    # Treatment indicator
    treatment = zeros(Bool, length(unit_id))
    for i in 1:length(unit_id)
        uid = unit_id[i]
        t = time[i]
        tt = treatment_time[uid + 1]  # +1 for 1-indexing
        treatment[i] = isfinite(tt) && (t >= tt)
    end

    # Outcomes: Y = 10 + 5*treatment + noise
    outcomes = 10.0 .+ 5.0 * treatment .+ randn(length(unit_id))

    @testset "StaggeredDiDProblem Construction" begin
        @test_nowarn StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        @test length(problem.outcomes) == length(outcomes)
        @test length(problem.treatment) == length(treatment)
        @test length(problem.time) == length(time)
        @test length(problem.unit_id) == length(unit_id)
        @test length(problem.treatment_time) == 6  # One per unit
    end

    @testset "StaggeredTWFE Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = StaggeredTWFE(cluster_se = true)

        # Test solve
        @test_nowarn solve(problem, estimator)

        result = solve(problem, estimator)

        # Check result structure
        @test hasfield(typeof(result), :estimate)
        @test hasfield(typeof(result), :se)
        @test hasfield(typeof(result), :t_stat)
        @test hasfield(typeof(result), :p_value)
        @test hasfield(typeof(result), :ci_lower)
        @test hasfield(typeof(result), :ci_upper)
        @test hasfield(typeof(result), :n_treated)
        @test hasfield(typeof(result), :n_control)
        @test hasfield(typeof(result), :n_obs)
        @test hasfield(typeof(result), :n_periods)
        @test hasfield(typeof(result), :n_units)
        @test hasfield(typeof(result), :n_cohorts)
        @test hasfield(typeof(result), :retcode)

        # Check result types
        @test result.estimate isa Float64
        @test result.se isa Float64
        @test result.retcode == :Success

        # Check sensible values (treatment effect should be roughly 5)
        @test 2.0 < result.estimate < 8.0  # Wide range due to bias
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end

    @testset "CallawaySantAnna Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = CallawaySantAnna(
            aggregation = :simple,
            control_group = :nevertreated,
            alpha = 0.05,
            n_bootstrap = 50,  # Small for speed
            random_seed = 123
        )

        # Test solve
        @test_nowarn solve(problem, estimator)

        result = solve(problem, estimator)

        # Check result structure
        @test hasfield(typeof(result), :att)
        @test hasfield(typeof(result), :se)
        @test hasfield(typeof(result), :t_stat)
        @test hasfield(typeof(result), :p_value)
        @test hasfield(typeof(result), :ci_lower)
        @test hasfield(typeof(result), :ci_upper)
        @test hasfield(typeof(result), :att_gt)
        @test hasfield(typeof(result), :aggregated)
        @test hasfield(typeof(result), :control_group)
        @test hasfield(typeof(result), :n_bootstrap)
        @test hasfield(typeof(result), :n_cohorts)
        @test hasfield(typeof(result), :n_obs)
        @test hasfield(typeof(result), :retcode)

        # Check result types
        @test result.att isa Float64
        @test result.se isa Float64
        @test result.att_gt isa Vector
        @test result.retcode == :Success

        # Check ATT(g,t) structure
        @test length(result.att_gt) > 0
        @test all(hasfield(typeof(r), :cohort) for r in result.att_gt)
        @test all(hasfield(typeof(r), :time) for r in result.att_gt)
        @test all(hasfield(typeof(r), :event_time) for r in result.att_gt)
        @test all(hasfield(typeof(r), :att) for r in result.att_gt)

        # Check sensible values (treatment effect should be roughly 5)
        @test 2.0 < result.att < 8.0
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end

    @testset "SunAbraham Estimator" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        estimator = SunAbraham(
            alpha = 0.05,
            cluster_se = true
        )

        # Test solve
        @test_nowarn solve(problem, estimator)

        result = solve(problem, estimator)

        # Check result structure
        @test hasfield(typeof(result), :att)
        @test hasfield(typeof(result), :se)
        @test hasfield(typeof(result), :t_stat)
        @test hasfield(typeof(result), :p_value)
        @test hasfield(typeof(result), :ci_lower)
        @test hasfield(typeof(result), :ci_upper)
        @test hasfield(typeof(result), :cohort_effects)
        @test hasfield(typeof(result), :weights)
        @test hasfield(typeof(result), :n_obs)
        @test hasfield(typeof(result), :n_treated)
        @test hasfield(typeof(result), :n_control)
        @test hasfield(typeof(result), :n_cohorts)
        @test hasfield(typeof(result), :cluster_se_used)
        @test hasfield(typeof(result), :retcode)

        # Check result types
        @test result.att isa Float64
        @test result.se isa Float64
        @test result.cohort_effects isa Vector
        @test result.weights isa Vector
        @test result.retcode == :Success

        # Check cohort effects structure
        @test length(result.cohort_effects) > 0
        @test all(hasfield(typeof(r), :cohort) for r in result.cohort_effects)
        @test all(hasfield(typeof(r), :event_time) for r in result.cohort_effects)
        @test all(hasfield(typeof(r), :coef) for r in result.cohort_effects)
        @test all(hasfield(typeof(r), :se) for r in result.cohort_effects)

        # Check weights structure
        @test length(result.weights) == length(result.cohort_effects)
        @test all(hasfield(typeof(r), :weight) for r in result.weights)
        @test sum(r.weight for r in result.weights) ≈ 1.0  # Weights should sum to 1

        # Check sensible values (treatment effect should be roughly 5)
        @test 2.0 < result.att < 8.0
        @test result.se > 0
        @test 0 <= result.p_value <= 1
    end

    @testset "Different Aggregation Schemes" begin
        problem = StaggeredDiDProblem(
            outcomes,
            treatment,
            time,
            unit_id,
            treatment_time,
            (alpha = 0.05,)
        )

        # Test dynamic aggregation
        est_dynamic = CallawaySantAnna(
            aggregation = :dynamic,
            control_group = :nevertreated,
            n_bootstrap = 50,
            random_seed = 123
        )

        result_dynamic = solve(problem, est_dynamic)
        @test result_dynamic.retcode == :Success
        @test result_dynamic.aggregated isa Dict
        @test length(result_dynamic.aggregated) > 0  # Should have event time estimates

        # Test group aggregation
        est_group = CallawaySantAnna(
            aggregation = :group,
            control_group = :nevertreated,
            n_bootstrap = 50,
            random_seed = 123
        )

        result_group = solve(problem, est_group)
        @test result_group.retcode == :Success
        @test result_group.aggregated isa Dict
        @test length(result_group.aggregated) > 0  # Should have cohort estimates
    end
end

println("✓ All staggered DiD smoke tests passed!")
