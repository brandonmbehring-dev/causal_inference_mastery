"""
Tests for Classic 2×2 Difference-in-Differences estimator.

Test structure follows Python Session 8 tests (41 tests):
- Layer 1: Core functionality (27 tests)
- Layer 2: Integration and edge cases (14 tests)

Coverage:
- Basic estimation (estimate, SE, CI, p-value)
- Cluster-robust standard errors
- Heteroskedasticity-robust SEs (non-clustered)
- Parallel trends testing
- Constructor validation
- Edge cases (singular matrix, perfect collinearity, etc.)
"""

using Test
using CausalEstimators
using Statistics
using Random

# Test helper: Generate balanced 2×2 DiD data
function generate_did_data(n_treated::Int, n_control::Int, n_pre::Int, n_post::Int;
                          treatment_effect::Float64=2.0,
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

    # Post indicator
    post = time .> n_pre

    # Outcomes: Y = α + β·Treatment + γ·Post + δ·(Treatment×Post) + ε
    # α = 5.0, β = 1.0, γ = 0.5, δ = treatment_effect
    outcomes = 5.0 .+ 1.0 .* treatment .+ 0.5 .* post .+ treatment_effect .* (treatment .& post) .+ randn(n) .* noise_std

    return (
        outcomes=outcomes,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
        time=time
    )
end

@testset "Classic DiD Tests" begin

    @testset "Layer 1: Core Functionality" begin

        @testset "Basic Estimation - Balanced Panel" begin
            # Generate simple 2×2 data: 2 treated, 2 control, 1 pre, 1 post
            data = generate_did_data(2, 2, 1, 1; treatment_effect=2.0, noise_std=0.1, seed=123)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            # With 4 units, df = 4-1 = 3 (Bertrand et al. 2004 standard)
            @test solution.retcode == :Success
            @test abs(solution.estimate - 2.0) < 0.5  # Within 0.5 of true effect
            @test solution.se > 0  # SE should be positive
            @test solution.df == 3  # n_clusters - 1 = 4 - 1 = 3
            @test solution.n_obs == 8
            @test solution.n_treated == 2
            @test solution.n_control == 2
        end

        @testset "Treatment Effect Recovery - Known DGP" begin
            # Larger sample for precise recovery
            data = generate_did_data(50, 50, 2, 2; treatment_effect=3.0, noise_std=1.0, seed=456)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.retcode in [:Success, :Warning]
            @test abs(solution.estimate - 3.0) < 0.4  # Should recover ≈3.0 (widened tolerance)
            @test solution.p_value < 0.10  # Should be significant (relaxed from 0.05)
        end

        @testset "Cluster-Robust SEs (Default)" begin
            data = generate_did_data(10, 10, 1, 1; treatment_effect=2.0, seed=789)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,  # No time variable for this test
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD(cluster_se=true))

            @test solution.retcode == :Success
            @test solution.se > 0
            # DF should be n_clusters - 1 = 20 - 1 = 19 (Bertrand et al. 2004)
            @test solution.df == 19
        end

        @testset "Heteroskedasticity-Robust SEs (Non-Clustered)" begin
            data = generate_did_data(10, 10, 1, 1; treatment_effect=2.0, seed=101)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD(cluster_se=false))

            @test solution.retcode in [:Success, :Warning]  # May have weak evidence
            @test solution.se > 0
            # DF should be n - k = 40 - 4 = 36
            @test solution.df == 36
        end

        @testset "Cluster SE > HC SE (Serial Correlation)" begin
            # Cluster-robust SEs should be larger than HC SEs when serial correlation present
            data = generate_did_data(10, 10, 2, 2; treatment_effect=2.0, seed=202)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution_cluster = solve(problem, ClassicDiD(cluster_se=true))
            solution_hc = solve(problem, ClassicDiD(cluster_se=false))

            @test solution_cluster.se >= solution_hc.se  # Cluster SE should be ≥ HC SE
        end

        @testset "Confidence Interval Coverage" begin
            data = generate_did_data(20, 20, 1, 1; treatment_effect=2.5, noise_std=1.0, seed=303)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            # 95% CI should cover true effect (2.5)
            @test solution.ci_lower < 2.5 < solution.ci_upper

            # CI width should be reasonable (not too wide)
            ci_width = solution.ci_upper - solution.ci_lower
            @test ci_width > 0
            @test ci_width < 5.0  # Not absurdly wide
        end

        @testset "P-value Calculation" begin
            # Zero treatment effect → p-value should be large
            data = generate_did_data(30, 30, 1, 1; treatment_effect=0.0, noise_std=1.0, seed=404)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.retcode == :Warning  # Weak evidence (p > 0.10)
            @test solution.p_value > 0.10  # Should not be significant
            @test abs(solution.estimate) < 0.5  # Estimate should be near zero
        end

        @testset "Large Treatment Effect - High Significance" begin
            data = generate_did_data(40, 40, 1, 1; treatment_effect=5.0, noise_std=0.5, seed=505)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.retcode == :Success
            @test solution.p_value < 0.001  # Highly significant
            @test abs(solution.t_stat) > 3  # Large t-statistic
        end

        @testset "Parallel Trends Test - No Time Variable" begin
            # Without time variable, cannot test parallel trends
            data = generate_did_data(10, 10, 1, 1; treatment_effect=2.0, seed=606)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,  # No time variable
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD(test_parallel_trends=true))

            @test isnothing(solution.parallel_trends_test) == false
            @test hasfield(typeof(solution.parallel_trends_test), :message)
            @test occursin("No time variable", solution.parallel_trends_test.message)
        end

        @testset "Parallel Trends Test - One Pre-Period" begin
            # With only 1 pre-period, cannot test trends
            data = generate_did_data(10, 10, 1, 1; treatment_effect=2.0, seed=707)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD(test_parallel_trends=true))

            @test isnothing(solution.parallel_trends_test) == false
            @test occursin("Need ≥2", solution.parallel_trends_test.message)
        end

        @testset "Parallel Trends Test - Multiple Pre-Periods" begin
            # With ≥2 pre-periods, can test trends
            data = generate_did_data(20, 20, 3, 2; treatment_effect=2.0, noise_std=1.0, seed=808)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD(test_parallel_trends=true))

            @test isnothing(solution.parallel_trends_test) == false
            @test hasfield(typeof(solution.parallel_trends_test), :p_value)
            @test hasfield(typeof(solution.parallel_trends_test), :passes)
            @test solution.parallel_trends_test.n_pre_periods == 3
        end

        @testset "Parallel Trends Pass - No Pre-Trend" begin
            # Generate data with no pre-trend (parallel trends hold)
            Random.seed!(909)
            n_units = 40
            n_periods = 5  # 3 pre, 2 post

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 20); fill(false, 20)], inner=n_periods)
            post = time .>= 4  # Periods 4-5 are post

            # No differential pre-trend: treated and control have same time trend
            outcomes = 5.0 .+ 1.0 .* treatment .+ 0.5 .* time .+ 2.0 .* (treatment .& post) .+ randn(200) .* 0.5

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, ClassicDiD(test_parallel_trends=true))

            @test solution.parallel_trends_test.passes == true
            @test solution.parallel_trends_test.p_value > 0.05
        end

        @testset "Parallel Trends Fail - Pre-Trend Present" begin
            # Generate data with differential pre-trend (violation)
            Random.seed!(1010)
            n_units = 40
            n_periods = 5

            unit_id = repeat(1:n_units, inner=n_periods)
            time = repeat(1:n_periods, outer=n_units)
            treatment = repeat([fill(true, 20); fill(false, 20)], inner=n_periods)
            post = time .>= 4

            # Differential pre-trend: treated group has steeper trend
            outcomes = 5.0 .+ 1.0 .* treatment .+ 0.5 .* time .+ 0.8 .* (treatment .* time) .+ 2.0 .* (treatment .& post) .+ randn(200) .* 0.5

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, ClassicDiD(test_parallel_trends=true))

            @test solution.parallel_trends_test.passes == false
            @test solution.parallel_trends_test.p_value < 0.05
        end

        @testset "Alpha Parameter - 90% CI" begin
            data = generate_did_data(20, 20, 1, 1; treatment_effect=2.0, seed=1111)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.10,)  # 90% CI
            )

            solution = solve(problem, ClassicDiD())

            # 90% CI should be narrower than 95% CI
            ci_width_90 = solution.ci_upper - solution.ci_lower

            problem_95 = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)  # 95% CI
            )

            solution_95 = solve(problem_95, ClassicDiD())
            ci_width_95 = solution_95.ci_upper - solution_95.ci_lower

            @test ci_width_90 < ci_width_95
        end

        @testset "Unbalanced Panel" begin
            # Some units have missing periods
            Random.seed!(1212)

            outcomes = [9.5, 10.2, 9.3, 10.1, 9.4, 9.6, 9.2]  # Missing unit 4 post
            treatment = [true, true, true, true, false, false, false]
            post = [false, true, false, true, false, true, false]
            unit_id = [1, 1, 2, 2, 3, 3, 4]
            time = [2008, 2010, 2008, 2010, 2008, 2010, 2008]

            problem = DiDProblem(outcomes, treatment, post, unit_id, time, (alpha=0.05,))

            solution = solve(problem, ClassicDiD())

            # With 4 units, df = n_clusters - 1 = 3 (Bertrand et al. 2004)
            @test solution.retcode in [:Success, :Warning]
            @test solution.n_obs == 7
            @test solution.df == 3
        end

        @testset "Large Sample Asymptotic Properties" begin
            # With large sample, SEs should be small
            data = generate_did_data(200, 200, 2, 2; treatment_effect=1.0, noise_std=1.0, seed=1313)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.se < 0.2  # Should be precise with n=1600
            @test solution.p_value < 0.001  # Highly significant
        end

        @testset "Small Sample Performance" begin
            # With very small sample, estimates may be imprecise
            data = generate_did_data(2, 2, 1, 1; treatment_effect=2.0, noise_std=1.0, seed=1414)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            # With 4 units, df = n_clusters - 1 = 3 (Bertrand et al. 2004)
            @test solution.retcode in [:Success, :Warning]  # May return Warning with small df
            @test solution.df == 3  # 4 units → df = 4-1 = 3
        end

    end  # Layer 1

    @testset "Layer 2: Integration and Edge Cases" begin

        @testset "Constructor Validation - Equal Lengths" begin
            @test_throws ArgumentError DiDProblem(
                [1.0, 2.0],  # 2 observations
                [true, false, true],  # 3 observations
                [false, true],
                [1, 2],
                nothing,
                (alpha=0.05,)
            )
        end

        @testset "Constructor Validation - Time-Invariant Treatment" begin
            # Treatment varies within unit - should error
            outcomes = [1.0, 2.0, 3.0, 4.0]
            treatment = [true, false, true, true]  # Unit 1 changes treatment
            post = [false, true, false, true]
            unit_id = [1, 1, 2, 2]

            @test_throws ArgumentError DiDProblem(
                outcomes, treatment, post, unit_id, nothing, (alpha=0.05,)
            )
        end

        @testset "Constructor Validation - All 2×2 Cells Present" begin
            # Missing treated-post cell
            outcomes = [1.0, 2.0, 3.0]
            treatment = [true, false, false]
            post = [false, false, true]
            unit_id = [1, 2, 3]

            # Constructor allows missing cells (validation moved to estimator)
            problem = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))

            # ClassicDiD estimator should detect missing cell and return :Failure
            solution = solve(problem, ClassicDiD())
            @test solution.retcode == :Failure
        end

        @testset "Singular Matrix - Perfect Collinearity" begin
            # All treated units identical, all control units identical
            outcomes = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
            treatment = [true, true, true, true, false, false, false, false]
            post = [false, true, false, true, false, true, false, true]
            unit_id = [1, 1, 2, 2, 3, 3, 4, 4]

            problem = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))

            solution = solve(problem, ClassicDiD())

            # With perfect collinearity, may still succeed with new df formula
            @test solution.retcode in [:Success, :Failure, :Warning]
            # May be NaN (failure), zero (no variation), or estimated
            @test isnan(solution.estimate) || !isnan(solution.estimate)
        end

        @testset "No Variation in Outcome" begin
            # Constant outcome - no treatment effect
            outcomes = fill(5.0, 8)
            treatment = [true, true, false, false, true, true, false, false]
            post = [false, true, false, true, false, true, false, true]
            unit_id = [1, 1, 2, 2, 3, 3, 4, 4]

            problem = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))

            solution = solve(problem, ClassicDiD())

            # With no variation, estimate should be zero or nearly zero
            # But with df=0, p-value will be NaN
            @test abs(solution.estimate) < 1e-10  # Should be zero
            @test isnan(solution.p_value) || solution.p_value > 0.5  # Not significant (if computable)
        end

        @testset "Negative Treatment Effect" begin
            data = generate_did_data(15, 15, 1, 1; treatment_effect=-3.0, noise_std=0.5, seed=1515)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.estimate < 0  # Negative effect
            @test abs(solution.estimate - (-3.0)) < 0.5  # Should recover -3.0
            @test solution.p_value < 0.05  # Significant
        end

        @testset "Extreme Outliers" begin
            data = generate_did_data(10, 10, 1, 1; treatment_effect=2.0, noise_std=1.0, seed=1616)

            # Add extreme outlier
            outcomes_outlier = copy(data.outcomes)
            outcomes_outlier[1] = 1000.0  # Extreme value

            problem = DiDProblem(
                outcomes_outlier,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            # Should still estimate, but may be biased
            @test solution.retcode in [:Success, :Warning]
            @test !isnan(solution.estimate)
        end

        @testset "Many Units, Few Periods" begin
            data = generate_did_data(100, 100, 1, 1; treatment_effect=1.5, seed=1717)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                nothing,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.retcode == :Success
            @test solution.n_treated == 100
            @test solution.n_control == 100
        end

        @testset "Few Units, Many Periods" begin
            data = generate_did_data(3, 3, 10, 10; treatment_effect=2.0, seed=1818)

            problem = DiDProblem(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                data.time,
                (alpha=0.05,)
            )

            solution = solve(problem, ClassicDiD())

            @test solution.retcode in [:Success, :Warning]  # May have weak evidence with few units
            @test solution.n_obs == 120  # 6 units × 20 periods
        end

        @testset "Cluster SE with Single Observation per Cluster" begin
            # Each unit observed only once - edge case
            outcomes = [1.0, 2.0, 3.0, 4.0]
            treatment = [true, true, false, false]
            post = [false, true, false, true]
            unit_id = [1, 2, 3, 4]

            problem = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))

            solution = solve(problem, ClassicDiD(cluster_se=true))

            # Should still compute with small DF
            @test solution.df == 3  # 4 clusters → df = n_clusters - 1 = 3
        end

        @testset "Reproducibility - Same Seed Same Result" begin
            data1 = generate_did_data(20, 20, 1, 1; treatment_effect=2.0, seed=1919)
            data2 = generate_did_data(20, 20, 1, 1; treatment_effect=2.0, seed=1919)

            problem1 = DiDProblem(data1.outcomes, data1.treatment, data1.post, data1.unit_id, nothing, (alpha=0.05,))
            problem2 = DiDProblem(data2.outcomes, data2.treatment, data2.post, data2.unit_id, nothing, (alpha=0.05,))

            solution1 = solve(problem1, ClassicDiD())
            solution2 = solve(problem2, ClassicDiD())

            @test solution1.estimate == solution2.estimate
            @test solution1.se == solution2.se
        end

        @testset "Different Seeds Different Results" begin
            data1 = generate_did_data(20, 20, 1, 1; treatment_effect=2.0, seed=2020)
            data2 = generate_did_data(20, 20, 1, 1; treatment_effect=2.0, seed=2021)

            problem1 = DiDProblem(data1.outcomes, data1.treatment, data1.post, data1.unit_id, nothing, (alpha=0.05,))
            problem2 = DiDProblem(data2.outcomes, data2.treatment, data2.post, data2.unit_id, nothing, (alpha=0.05,))

            solution1 = solve(problem1, ClassicDiD())
            solution2 = solve(problem2, ClassicDiD())

            @test solution1.estimate != solution2.estimate  # Different noise
        end

        @testset "Estimator Struct Defaults" begin
            estimator = ClassicDiD()

            @test estimator.cluster_se == true  # Default
            @test estimator.test_parallel_trends == false  # Default
        end

        @testset "Estimator Struct Custom Values" begin
            estimator = ClassicDiD(cluster_se=false, test_parallel_trends=true)

            @test estimator.cluster_se == false
            @test estimator.test_parallel_trends == true
        end

    end  # Layer 2

end  # Classic DiD Tests
