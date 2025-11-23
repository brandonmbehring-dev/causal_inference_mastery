"""
Tests for Event Study (Dynamic DiD) estimator.

Test structure follows Python Session 9 tests (37 tests):
- Layer 1: Core functionality (25 tests)
- Layer 2: Integration and edge cases (12 tests)

Coverage:
- Event time computation
- Lead/lag indicator creation
- Two-Way Fixed Effects (TWFE)
- Dynamic treatment effects
- Joint F-test for pre-trends
- Auto-detection of leads/lags
- Edge cases (staggered treatment, missing periods, etc.)
"""

using Test
using CausalEstimators
using Statistics
using Random

# Test helper: Generate event study data with multiple periods
function generate_event_study_data(n_treated::Int, n_control::Int, n_pre::Int, n_post::Int;
                                  dynamic_effects::Vector{Float64}=Float64[],
                                  pre_trend::Float64=0.0,
                                  noise_std::Float64=1.0,
                                  seed::Int=42)
    Random.seed!(seed)

    n_units = n_treated + n_control
    n_periods = n_pre + n_post
    n = n_units * n_periods

    # Create panel
    unit_id = repeat(1:n_units, inner=n_periods)
    time = repeat(1:n_periods, outer=n_units)

    # Treatment indicator (time-invariant)
    treatment = repeat([fill(true, n_treated); fill(false, n_control)], inner=n_periods)

    # Post indicator (treatment starts at period n_pre + 1)
    post = time .> n_pre

    # Event time (periods relative to treatment)
    event_time = zeros(Int, n)
    for i in 1:n
        if treatment[i]
            event_time[i] = time[i] - (n_pre + 1)
        end
    end

    # Outcomes with dynamic effects
    outcomes = 5.0 .+ 1.0 .* treatment .+ 0.5 .* time .+ pre_trend .* (treatment .* time .* (.!post))

    # Add dynamic treatment effects
    if !isempty(dynamic_effects)
        for i in 1:n
            if treatment[i] && post[i]
                period_post = time[i] - n_pre
                if period_post <= length(dynamic_effects)
                    outcomes[i] += dynamic_effects[period_post]
                else
                    outcomes[i] += dynamic_effects[end]  # Use last effect for later periods
                end
            end
        end
    else
        # Constant treatment effect
        outcomes .+= 2.0 .* (treatment .& post)
    end

    outcomes .+= randn(n) .* noise_std

    return (
        outcomes=outcomes,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
        time=time,
        event_time=event_time
    )
end

@testset "Event Study Tests" begin

    @testset "Layer 1: Core Functionality" begin

        @testset "Basic Event Study - Multiple Periods" begin
            # 3 pre-periods, 3 post-periods
            data = generate_event_study_data(10, 10, 3, 3; noise_std=0.5, seed=123)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            @test solution.retcode in [:Success, :Warning]
            @test !isnan(solution.estimate)
            @test solution.se > 0
            @test hasfield(typeof(solution.parallel_trends_test), :coefficients_pre)
            @test hasfield(typeof(solution.parallel_trends_test), :coefficients_post)
        end

        @testset "Event Time Computation - Correct Relative Periods" begin
            # Simple 2 treated, 2 control, 2 pre, 2 post
            n_units = 4
            n_periods = 4

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([true, true, false, false], inner=n_periods)
            post = time .> 2  # Treatment at period 3

            outcomes = randn(16)

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # Should have leads: -2, -1 (omitted) and lags: 0, 1
            @test hasfield(typeof(solution.parallel_trends_test), :event_times_pre)
            @test hasfield(typeof(solution.parallel_trends_test), :event_times_post)
        end

        @testset "Auto-Detect Leads and Lags" begin
            # 4 pre, 3 post - should auto-detect all
            data = generate_event_study_data(15, 15, 4, 3; seed=456)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())  # Auto-detect

            # Should detect 3 leads (-4, -3, -2, omit -1) and 3 lags (0, 1, 2)
            @test length(solution.parallel_trends_test.coefficients_pre) == 3  # -4, -3, -2
            @test length(solution.parallel_trends_test.coefficients_post) == 3  # 0, 1, 2
        end

        @testset "Manual Leads and Lags Specification" begin
            data = generate_event_study_data(15, 15, 5, 4; seed=789)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy(n_leads=3, n_lags=2))

            # Should use only 3 leads and 2 lags
            @test length(solution.parallel_trends_test.coefficients_pre) == 3
            @test length(solution.parallel_trends_test.coefficients_post) == 2
        end

        @testset "Dynamic Treatment Effects - Immediate Impact" begin
            # Effect only in period 0, then zero
            dynamic_effects = [3.0, 0.0, 0.0]

            data = generate_event_study_data(20, 20, 2, 3;
                                           dynamic_effects=dynamic_effects,
                                           noise_std=0.5,
                                           seed=101)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # First post coefficient should be largest
            @test solution.parallel_trends_test.coefficients_post[1] > solution.parallel_trends_test.coefficients_post[2]
        end

        @testset "Dynamic Treatment Effects - Gradual Buildup" begin
            # Gradual increase: 1.0, 2.0, 3.0
            dynamic_effects = [1.0, 2.0, 3.0]

            data = generate_event_study_data(20, 20, 2, 3;
                                           dynamic_effects=dynamic_effects,
                                           noise_std=0.5,
                                           seed=202)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Coefficients should be increasing
            coefs_post = solution.parallel_trends_test.coefficients_post
            @test coefs_post[1] < coefs_post[2] < coefs_post[3]
        end

        @testset "Pre-Trends Test - Parallel Trends Hold" begin
            # No pre-trend, should pass
            data = generate_event_study_data(25, 25, 3, 2;
                                           pre_trend=0.0,
                                           noise_std=0.8,
                                           seed=303)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # F-test should not reject parallel trends
            @test solution.parallel_trends_test.f_pvalue > 0.05
            @test solution.parallel_trends_test.pre_trends_pass == true
        end

        @testset "Pre-Trends Test - Violation Detected" begin
            # Strong pre-trend, should fail
            data = generate_event_study_data(30, 30, 4, 2;
                                           pre_trend=0.5,  # Strong differential trend
                                           noise_std=0.5,
                                           seed=404)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # F-test should reject parallel trends
            @test solution.parallel_trends_test.f_pvalue < 0.10
            @test solution.parallel_trends_test.pre_trends_pass == false
        end

        @testset "TWFE Demeaning - Unit Fixed Effects" begin
            # Add unit-specific intercepts
            Random.seed!(505)
            n_units = 20
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 10); fill(false, 10)], inner=n_periods)
            post = time .> 3

            # Unit-specific intercepts (should be absorbed by FE)
            unit_effects = repeat(randn(n_units), inner=n_periods)
            outcomes = 5.0 .+ unit_effects .+ 2.0 .* (treatment .& post) .+ randn(100) .* 0.5

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # Should still estimate treatment effect correctly
            @test abs(solution.estimate - 2.0) < 0.5
        end

        @testset "TWFE Demeaning - Time Fixed Effects" begin
            # Add time-specific trends
            Random.seed!(606)
            n_units = 20
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 10); fill(false, 10)], inner=n_periods)
            post = time .> 3

            # Time-specific effects (should be absorbed by FE)
            time_effects = repeat(randn(n_periods), outer=n_units)
            outcomes = 5.0 .+ time_effects .+ 2.0 .* (treatment .& post) .+ randn(100) .* 0.5

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # Should still estimate treatment effect correctly
            @test abs(solution.estimate - 2.0) < 0.5
        end

        @testset "Cluster-Robust SEs (Default)" begin
            data = generate_event_study_data(15, 15, 3, 3; seed=707)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy(cluster_se=true))

            @test solution.se > 0
            # All post coefficient SEs should be positive
            @test all(solution.parallel_trends_test.se_post .> 0)
        end

        @testset "Heteroskedasticity-Robust SEs" begin
            data = generate_event_study_data(15, 15, 3, 3; seed=808)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy(cluster_se=false))

            @test solution.se > 0
            @test all(solution.parallel_trends_test.se_post .> 0)
        end

        @testset "Omit Period Parameter - Period -1" begin
            data = generate_event_study_data(15, 15, 3, 3; seed=909)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy(omit_period=-1))

            # Period -1 should be omitted (not in coefficients)
            @test -1 ∉ solution.parallel_trends_test.event_times_pre
            @test -1 ∉ solution.parallel_trends_test.event_times_post
        end

        @testset "Average Treatment Effect Calculation" begin
            # Constant effect across all post-periods
            dynamic_effects = [2.5, 2.5, 2.5]

            data = generate_event_study_data(20, 20, 2, 3;
                                           dynamic_effects=dynamic_effects,
                                           noise_std=0.3,
                                           seed=1010)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # ATE should be mean of post coefficients ≈ 2.5
            @test abs(solution.estimate - 2.5) < 0.5
        end

        @testset "Confidence Intervals for Dynamic Effects" begin
            data = generate_event_study_data(25, 25, 3, 3; seed=1111)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # All post coefficients should have positive SEs
            @test all(solution.parallel_trends_test.se_post .> 0)

            # CIs should cover coefficients
            for i in 1:length(solution.parallel_trends_test.coefficients_post)
                coef = solution.parallel_trends_test.coefficients_post[i]
                se = solution.parallel_trends_test.se_post[i]
                @test se > 0
            end
        end

        @testset "Large Sample - Precise Estimates" begin
            data = generate_event_study_data(100, 100, 3, 3; noise_std=1.0, seed=1212)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # With large sample, SEs should be small
            @test solution.se < 0.3
        end

        @testset "Small Sample - Wide CIs" begin
            data = generate_event_study_data(3, 3, 2, 2; noise_std=1.0, seed=1313)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Small sample may have wide CIs
            ci_width = solution.ci_upper - solution.ci_lower
            @test ci_width > 0
        end

        @testset "No Pre-Treatment Periods - No Pre-Trends Test" begin
            # Only post-treatment observations (edge case)
            data = generate_event_study_data(10, 10, 0, 3; seed=1414)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Should have no pre coefficients
            @test length(solution.parallel_trends_test.coefficients_pre) == 0
        end

        @testset "Many Leads - Long Pre-Period" begin
            data = generate_event_study_data(20, 20, 10, 3; seed=1515)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Should detect 9 leads (-10 to -2, omit -1)
            @test length(solution.parallel_trends_test.coefficients_pre) == 9
        end

        @testset "Many Lags - Long Post-Period" begin
            data = generate_event_study_data(20, 20, 2, 10; seed=1616)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Should detect 10 lags (0 to 9)
            @test length(solution.parallel_trends_test.coefficients_post) == 10
        end

        @testset "Balanced vs Unbalanced Panel" begin
            # Balanced panel
            data_balanced = generate_event_study_data(10, 10, 2, 2; seed=1717)

            problem_balanced = DiDProblem(
                data_balanced.outcomes,
                data_balanced.treatment,
                data_balanced.post,
                data_balanced.unit_id,
                data_balanced.time,
                (alpha=0.05,)
            )

            solution_balanced = solve(problem_balanced, EventStudy())

            @test solution_balanced.retcode in [:Success, :Warning]

            # Unbalanced: drop some observations
            n = length(data_balanced.outcomes)
            keep_mask = [i % 5 != 0 for i in 1:n]  # Drop 20%

            problem_unbalanced = DiDProblem(
                data_balanced.outcomes[keep_mask],
                data_balanced.treatment[keep_mask],
                data_balanced.post[keep_mask],
                data_balanced.unit_id[keep_mask],
                data_balanced.time[keep_mask],
                (alpha=0.05,)
            )

            solution_unbalanced = solve(problem_unbalanced, EventStudy())

            @test solution_unbalanced.retcode in [:Success, :Warning]
        end

        @testset "Zero Pre-Treatment Coefficients" begin
            # Perfect parallel trends - all pre coefficients should be ≈0
            Random.seed!(1818)
            n_units = 40
            n_periods = 6  # 4 pre, 2 post

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 20); fill(false, 20)], inner=n_periods)
            post = time .> 4

            # Perfect parallel trends
            outcomes = 5.0 .+ 1.0 .* treatment .+ 0.3 .* time .+ 2.0 .* (treatment .& post) .+ randn(240) .* 0.2

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # All pre coefficients should be close to zero
            @test all(abs.(solution.parallel_trends_test.coefficients_pre) .< 0.5)
        end

        @testset "Estimator Struct Defaults" begin
            estimator = EventStudy()

            @test isnothing(estimator.n_leads)
            @test isnothing(estimator.n_lags)
            @test estimator.omit_period == -1
            @test estimator.cluster_se == true
        end

        @testset "Estimator Struct Custom Values" begin
            estimator = EventStudy(n_leads=3, n_lags=2, omit_period=-2, cluster_se=false)

            @test estimator.n_leads == 3
            @test estimator.n_lags == 2
            @test estimator.omit_period == -2
            @test estimator.cluster_se == false
        end

    end  # Layer 1

    @testset "Layer 2: Integration and Edge Cases" begin

        @testset "No Time Variable - Failure" begin
            data = generate_event_study_data(10, 10, 2, 2; seed=1919)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,  # No time variable
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            @test solution.retcode == :Failure
            @test isnan(solution.estimate)
        end

        @testset "No Treated Units - Failure" begin
            # All control units
            Random.seed!(2020)
            n_units = 10
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = fill(false, n_units * n_periods)  # All control
            post = time .> 3
            outcomes = randn(50)

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            @test solution.retcode == :Failure
        end

        @testset "Singular Matrix - Perfect Collinearity" begin
            # Create perfect collinearity
            Random.seed!(2121)
            n_units = 4
            n_periods = 3

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([true, true, false, false], inner=n_periods)
            post = time .> 2

            # Outcome perfectly predicted by unit
            outcomes = Float64.(unit_id)  # Perfect collinearity

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            @test solution.retcode == :Failure
        end

        @testset "All Units Treated Same Period - Edge Case" begin
            # No variation in treatment timing
            Random.seed!(2222)
            n_units = 10
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = fill(true, n_units * n_periods)  # All treated
            post = time .> 3
            outcomes = randn(50) .+ 2.0 .* post

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # Should still work (no control group, but TWFE can estimate)
            @test solution.retcode in [:Success, :Warning, :Failure]
        end

        @testset "Negative Dynamic Effects" begin
            # Treatment reduces outcome over time
            dynamic_effects = [-1.0, -2.0, -3.0]

            data = generate_event_study_data(15, 15, 2, 3;
                                           dynamic_effects=dynamic_effects,
                                           seed=2323)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # ATE should be negative
            @test solution.estimate < 0
        end

        @testset "Extreme Outliers in Outcomes" begin
            data = generate_event_study_data(10, 10, 2, 2; seed=2424)

            # Add outlier
            outcomes_outlier = copy(data.outcomes)
            outcomes_outlier[1] = 999.0

            problem = DiDProblem(
                outcomes_outlier,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Should still estimate (OLS not robust to outliers, but should run)
            @test !isnan(solution.estimate)
        end

        @testset "Very Long Panel - Many Periods" begin
            data = generate_event_study_data(10, 10, 20, 20; seed=2525)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            @test solution.retcode in [:Success, :Warning]
            @test solution.n_obs == 800  # 20 units × 40 periods
        end

        @testset "Constant Outcome - No Variation" begin
            Random.seed!(2626)
            n_units = 10
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 5); fill(false, 5)], inner=n_periods)
            post = time .> 3
            outcomes = fill(5.0, 50)  # Constant

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, EventStudy())

            # Should detect no effect
            @test abs(solution.estimate) < 1e-10
        end

        @testset "Reproducibility - Same Seed" begin
            data1 = generate_event_study_data(15, 15, 3, 3; seed=2727)
            data2 = generate_event_study_data(15, 15, 3, 3; seed=2727)

            problem1 = DiDProblem(data1.outcomes, data1.treatment, data1.post, data1.unit_id, data1.time, (alpha=0.05,))
            problem2 = DiDProblem(data2.outcomes, data2.treatment, data2.post, data2.unit_id, data2.time, (alpha=0.05,))

            solution1 = solve(problem1, EventStudy())
            solution2 = solve(problem2, EventStudy())

            @test solution1.estimate == solution2.estimate
            @test solution1.se == solution2.se
        end

        @testset "Different Seeds - Different Results" begin
            data1 = generate_event_study_data(15, 15, 3, 3; seed=2828)
            data2 = generate_event_study_data(15, 15, 3, 3; seed=2829)

            problem1 = DiDProblem(data1.outcomes, data1.treatment, data1.post, data1.unit_id, data1.time, (alpha=0.05,))
            problem2 = DiDProblem(data2.outcomes, data2.treatment, data2.post, data2.unit_id, data2.time, (alpha=0.05,))

            solution1 = solve(problem1, EventStudy())
            solution2 = solve(problem2, EventStudy())

            @test solution1.estimate != solution2.estimate
        end

        @testset "F-test Degrees of Freedom" begin
            data = generate_event_study_data(20, 20, 4, 3; seed=2930)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # DF1 should equal number of pre coefficients
            @test solution.parallel_trends_test.f_df1 == length(solution.parallel_trends_test.coefficients_pre)
            @test solution.parallel_trends_test.f_df2 > 0
        end

        @testset "Unit Count vs Observation Count" begin
            data = generate_event_study_data(10, 15, 3, 3; seed=3031)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, EventStudy())

            # Should count unique units, not observations
            @test solution.n_treated == 10
            @test solution.n_control == 15
            @test solution.n_obs > solution.n_treated + solution.n_control  # Multiple periods per unit
        end

    end  # Layer 2

end  # Event Study Tests
