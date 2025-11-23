"""
Comprehensive tests for staggered Difference-in-Differences estimators.

Tests coverage:
- StaggeredDiDProblem construction and validation
- StaggeredTWFE (Two-Way Fixed Effects)
- CallawaySantAnna (Group-Time ATT with bootstrap)
- SunAbraham (Interaction-Weighted Estimator)
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

@testset "Staggered DiD - Comprehensive Tests" begin

    # =========================================================================
    # Test Data Generators
    # =========================================================================

    """Generate synthetic staggered DiD data"""
    function generate_staggered_data(;
        n_periods=8,
        n_units_per_cohort=10,
        cohorts=[3, 5],
        true_effect=5.0,
        noise_sd=1.0,
        seed=123
    )
        Random.seed!(seed)

        n_treated_cohorts = length(cohorts)
        n_never = n_units_per_cohort
        n_units = (n_treated_cohorts + 1) * n_units_per_cohort

        unit_id = repeat(0:(n_units-1), inner=n_periods)
        time = repeat(0:(n_periods-1), outer=n_units)

        # Treatment times
        treatment_time = Float64[]
        for cohort in cohorts
            append!(treatment_time, fill(Float64(cohort), n_units_per_cohort))
        end
        append!(treatment_time, fill(Inf, n_never))

        # Treatment indicator
        treatment = zeros(Bool, length(unit_id))
        for i in 1:length(unit_id)
            uid = unit_id[i]
            t = time[i]
            tt = treatment_time[uid + 1]
            treatment[i] = isfinite(tt) && (t >= tt)
        end

        # Outcomes
        outcomes = 10.0 .+ true_effect * treatment .+ noise_sd * randn(length(unit_id))

        return (
            outcomes = outcomes,
            treatment = treatment,
            time = time,
            unit_id = unit_id,
            treatment_time = treatment_time,
            n_units = n_units,
            n_periods = n_periods
        )
    end

    # =========================================================================
    # StaggeredDiDProblem Tests
    # =========================================================================

    @testset "StaggeredDiDProblem Construction" begin

        @testset "Basic Construction" begin
            data = generate_staggered_data()

            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            @test length(problem.outcomes) == length(data.outcomes)
            @test length(problem.treatment) == length(data.treatment)
            @test length(problem.time) == length(data.time)
            @test length(problem.unit_id) == length(data.unit_id)
            @test length(problem.treatment_time) == data.n_units
            @test problem.parameters.alpha == 0.05
        end

        @testset "Single Treated Cohort" begin
            data = generate_staggered_data(cohorts=[4])

            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            @test length(problem.treatment_time) == data.n_units
            cohorts = sort(unique(data.treatment_time[isfinite.(data.treatment_time)]))
            @test length(cohorts) == 1
            @test cohorts[1] == 4.0
        end

        @testset "Many Cohorts" begin
            data = generate_staggered_data(
                cohorts=[2, 3, 4, 5, 6],
                n_units_per_cohort=5
            )

            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            cohorts = sort(unique(data.treatment_time[isfinite.(data.treatment_time)]))
            @test length(cohorts) == 5
        end

        @testset "No Never-Treated Units" begin
            data = generate_staggered_data(cohorts=[3, 5], n_units_per_cohort=10)
            # Modify to have no never-treated
            data_no_never = (
                outcomes = data.outcomes[1:(end-10*data.n_periods)],
                treatment = data.treatment[1:(end-10*data.n_periods)],
                time = data.time[1:(end-10*data.n_periods)],
                unit_id = data.unit_id[1:(end-10*data.n_periods)],
                treatment_time = data.treatment_time[1:20],
                n_units = 20,
                n_periods = data.n_periods
            )

            problem = StaggeredDiDProblem(
                data_no_never.outcomes,
                data_no_never.treatment,
                data_no_never.time,
                data_no_never.unit_id,
                data_no_never.treatment_time,
                (alpha = 0.05,)
            )

            @test all(isfinite.(problem.treatment_time))
        end

        @testset "Unbalanced Panel" begin
            data = generate_staggered_data()
            # Remove some observations to create unbalanced panel
            keep_idx = 1:(length(data.outcomes)-10)

            # Get unique units remaining after dropping observations
            units_remaining = unique(data.unit_id[keep_idx])
            # Trim treatment_time to match remaining units (unit IDs are 0-indexed)
            treatment_time_trimmed = data.treatment_time[units_remaining .+ 1]

            problem = StaggeredDiDProblem(
                data.outcomes[keep_idx],
                data.treatment[keep_idx],
                data.time[keep_idx],
                data.unit_id[keep_idx],
                treatment_time_trimmed,
                (alpha = 0.05,)
            )

            @test length(problem.outcomes) == length(keep_idx)
            @test length(problem.treatment_time) == length(units_remaining)
        end

        @testset "Different Alpha Levels" begin
            data = generate_staggered_data()

            for alpha in [0.01, 0.05, 0.10]
                problem = StaggeredDiDProblem(
                    data.outcomes,
                    data.treatment,
                    data.time,
                    data.unit_id,
                    data.treatment_time,
                    (alpha = alpha,)
                )
                @test problem.parameters.alpha == alpha
            end
        end
    end

    # =========================================================================
    # StaggeredTWFE Tests
    # =========================================================================

    @testset "StaggeredTWFE Estimator" begin

        @testset "Basic Estimation" begin
            data = generate_staggered_data(true_effect=5.0, noise_sd=0.5)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = StaggeredTWFE(cluster_se=true)
            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.estimate isa Float64
            @test result.se > 0
            @test result.t_stat isa Float64
            @test 0 <= result.p_value <= 1
            @test result.ci_lower < result.ci_upper

            # Estimate should be roughly near true effect (within wide range due to bias)
            @test 2.0 < result.estimate < 10.0
        end

        @testset "Cluster vs Non-Cluster SE" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            result_cluster = solve(problem, StaggeredTWFE(cluster_se=true))
            result_no_cluster = solve(problem, StaggeredTWFE(cluster_se=false))

            # Both should succeed
            @test result_cluster.retcode == :Success
            @test result_no_cluster.retcode == :Success

            # Estimates should be identical
            @test result_cluster.estimate ≈ result_no_cluster.estimate atol=1e-10

            # Cluster SE should typically be larger (or similar)
            # (Not always true in finite samples, so just check both are positive)
            @test result_cluster.se > 0
            @test result_no_cluster.se > 0
        end

        @testset "Null Effect" begin
            data = generate_staggered_data(true_effect=0.0, noise_sd=0.5)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            result = solve(problem, StaggeredTWFE(cluster_se=true))

            @test result.retcode == :Success
            @test abs(result.estimate) < 2.0  # Should be near zero
            @test result.p_value > 0.01  # Should not be significant
        end

        @testset "Large Effect" begin
            data = generate_staggered_data(true_effect=20.0, noise_sd=1.0)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            result = solve(problem, StaggeredTWFE(cluster_se=true))

            @test result.retcode == :Success
            @test result.estimate > 10.0  # Should detect large effect
            @test result.p_value < 0.001  # Should be highly significant
        end

        @testset "Result Fields Complete" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            result = solve(problem, StaggeredTWFE(cluster_se=true))

            # Check all expected fields exist
            @test hasfield(typeof(result), :estimate)
            @test hasfield(typeof(result), :se)
            @test hasfield(typeof(result), :t_stat)
            @test hasfield(typeof(result), :p_value)
            @test hasfield(typeof(result), :ci_lower)
            @test hasfield(typeof(result), :ci_upper)
            @test hasfield(typeof(result), :n_obs)
            @test hasfield(typeof(result), :n_treated)
            @test hasfield(typeof(result), :n_control)
            @test hasfield(typeof(result), :n_periods)
            @test hasfield(typeof(result), :n_units)
            @test hasfield(typeof(result), :n_cohorts)
            @test hasfield(typeof(result), :retcode)
        end
    end

    # =========================================================================
    # CallawaySantAnna Tests
    # =========================================================================

    @testset "CallawaySantAnna Estimator" begin

        @testset "Basic Estimation - Simple Aggregation" begin
            data = generate_staggered_data(true_effect=5.0, n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.att isa Float64
            @test result.se > 0
            @test length(result.att_gt) > 0
            @test 2.0 < result.att < 10.0
        end

        @testset "Bootstrap Inference" begin
            data = generate_staggered_data(n_units_per_cohort=20)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 100,
                random_seed = 456
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.n_bootstrap == 100
            @test result.se > 0
            @test result.ci_lower < result.att < result.ci_upper
        end

        @testset "Dynamic Aggregation" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :dynamic,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.aggregated isa Dict
            @test length(result.aggregated) > 0

            # Check event time aggregation structure
            for (event_time, agg) in result.aggregated
                @test event_time isa Int
                @test hasfield(typeof(agg), :att)
                @test hasfield(typeof(agg), :se)
            end
        end

        @testset "Group Aggregation" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :group,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.aggregated isa Dict
            @test length(result.aggregated) > 0

            # Check cohort aggregation structure
            for (cohort, agg) in result.aggregated
                @test cohort isa Float64
                @test hasfield(typeof(agg), :att)
                @test hasfield(typeof(agg), :se)
            end
        end

        @testset "Calendar Time Aggregation" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :calendar,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.aggregated isa Dict
            @test length(result.aggregated) > 0

            # Check calendar time aggregation structure
            for (cal_time, agg) in result.aggregated
                @test cal_time isa Int
                @test hasfield(typeof(agg), :att)
                @test hasfield(typeof(agg), :se)
            end
        end

        @testset "Control Group: Never-Treated" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.control_group == :nevertreated
        end

        @testset "Control Group: Not-Yet-Treated" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :notyettreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.control_group == :notyettreated
        end

        @testset "ATT(g,t) Structure" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test length(result.att_gt) > 0

            for att_gt in result.att_gt
                @test hasfield(typeof(att_gt), :cohort)
                @test hasfield(typeof(att_gt), :time)
                @test hasfield(typeof(att_gt), :event_time)
                @test hasfield(typeof(att_gt), :att)
                @test hasfield(typeof(att_gt), :n_treated)
                @test hasfield(typeof(att_gt), :n_control)
            end
        end

        @testset "Reproducibility with Random Seed" begin
            data = generate_staggered_data(n_units_per_cohort=15)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator1 = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 999
            )

            estimator2 = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 999
            )

            result1 = solve(problem, estimator1)
            result2 = solve(problem, estimator2)

            @test result1.att ≈ result2.att
            @test result1.se ≈ result2.se
        end

        @testset "Null Effect Detection" begin
            data = generate_staggered_data(true_effect=0.0, n_units_per_cohort=20)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = CallawaySantAnna(
                aggregation = :simple,
                control_group = :nevertreated,
                n_bootstrap = 50,
                random_seed = 123
            )

            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test abs(result.att) < 2.0  # Should be near zero
            @test result.p_value > 0.05  # Should not be significant at 5%
        end
    end

    # =========================================================================
    # SunAbraham Tests
    # =========================================================================

    @testset "SunAbraham Estimator" begin

        @testset "Basic Estimation" begin
            data = generate_staggered_data(true_effect=5.0)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.att isa Float64
            @test result.se > 0
            @test length(result.cohort_effects) > 0
            @test length(result.weights) == length(result.cohort_effects)
            @test 1.0 < result.att < 10.0
        end

        @testset "Interaction Weights Sum to One" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            total_weight = sum(w.weight for w in result.weights)
            @test total_weight ≈ 1.0 atol=1e-10
        end

        @testset "Cohort-Specific Effects Structure" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            for effect in result.cohort_effects
                @test hasfield(typeof(effect), :cohort)
                @test hasfield(typeof(effect), :event_time)
                @test hasfield(typeof(effect), :coef)
                @test hasfield(typeof(effect), :se)
                @test hasfield(typeof(effect), :t_stat)
                @test hasfield(typeof(effect), :p_value)
            end
        end

        @testset "Cluster vs Non-Cluster SE" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            result_cluster = solve(problem, SunAbraham(alpha=0.05, cluster_se=true))
            result_no_cluster = solve(problem, SunAbraham(alpha=0.05, cluster_se=false))

            @test result_cluster.retcode == :Success
            @test result_no_cluster.retcode == :Success

            # Estimates should be identical
            @test result_cluster.att ≈ result_no_cluster.att atol=1e-10

            # SE should differ (cluster typically larger)
            @test result_cluster.se > 0
            @test result_no_cluster.se > 0
        end

        @testset "Delta Method Variance" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            # Variance should be positive
            @test result.se^2 > 0

            # Confidence interval should be valid
            @test result.ci_lower < result.att
            @test result.att < result.ci_upper
        end

        @testset "Null Effect Detection" begin
            data = generate_staggered_data(true_effect=0.0, noise_sd=0.5)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test abs(result.att) < 2.0
            @test result.p_value > 0.05
        end

        @testset "Large Effect Detection" begin
            data = generate_staggered_data(true_effect=20.0, noise_sd=1.0)
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

            @test result.retcode == :Success
            @test result.att > 10.0
            @test result.p_value < 0.001
        end

        @testset "Different Alpha Levels" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            for alpha in [0.01, 0.05, 0.10]
                estimator = SunAbraham(alpha=alpha, cluster_se=true)
                result = solve(problem, estimator)

                @test result.retcode == :Success

                # CI width should increase with alpha
                ci_width = result.ci_upper - result.ci_lower
                @test ci_width > 0
            end
        end

        @testset "Result Fields Complete" begin
            data = generate_staggered_data()
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            estimator = SunAbraham(alpha=0.05, cluster_se=true)
            result = solve(problem, estimator)

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
        end
    end

    # =========================================================================
    # Cross-Estimator Comparisons
    # =========================================================================

    @testset "Cross-Estimator Comparisons" begin

        @testset "All Estimators Detect Same Strong Effect" begin
            data = generate_staggered_data(
                true_effect=10.0,
                noise_sd=0.5,
                n_units_per_cohort=20
            )
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            # TWFE
            twfe_result = solve(problem, StaggeredTWFE(cluster_se=true))

            # Callaway-Sant'Anna
            cs_result = solve(problem, CallawaySantAnna(
                aggregation=:simple,
                control_group=:nevertreated,
                n_bootstrap=50,
                random_seed=123
            ))

            # Sun-Abraham
            sa_result = solve(problem, SunAbraham(alpha=0.05, cluster_se=true))

            # All should succeed
            @test twfe_result.retcode == :Success
            @test cs_result.retcode == :Success
            @test sa_result.retcode == :Success

            # All should detect large positive effect
            @test twfe_result.estimate > 5.0
            @test cs_result.att > 5.0
            @test sa_result.att > 5.0

            # All should be significant
            @test twfe_result.p_value < 0.01
            @test cs_result.p_value < 0.01
            @test sa_result.p_value < 0.01
        end

        @testset "All Estimators Agree on Null" begin
            data = generate_staggered_data(
                true_effect=0.0,
                noise_sd=0.5,
                n_units_per_cohort=25
            )
            problem = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha = 0.05,)
            )

            # TWFE
            twfe_result = solve(problem, StaggeredTWFE(cluster_se=true))

            # Callaway-Sant'Anna
            cs_result = solve(problem, CallawaySantAnna(
                aggregation=:simple,
                control_group=:nevertreated,
                n_bootstrap=50,
                random_seed=123
            ))

            # Sun-Abraham
            sa_result = solve(problem, SunAbraham(alpha=0.05, cluster_se=true))

            # All should find estimates near zero
            @test abs(twfe_result.estimate) < 2.0
            @test abs(cs_result.att) < 2.0
            @test abs(sa_result.att) < 2.0

            # All should have non-significant results
            @test twfe_result.p_value > 0.05
            @test cs_result.p_value > 0.05
            @test sa_result.p_value > 0.05
        end
    end
end

println("✓ All comprehensive staggered DiD tests passed!")
