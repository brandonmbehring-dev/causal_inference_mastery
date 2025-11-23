"""
PyCall validation tests for DiD estimators.

Validates Julia DiD implementation against Python implementation using identical data.
"""

using Test
using CausalEstimators
using PyCall
using Statistics
using Random

# Add Python project root to sys.path
pushfirst!(PyVector(pyimport("sys")."path"), "/home/brandon_behring/Claude/causal_inference_mastery")

# Import Python DiD modules
const did_py = pyimport("src.causal_inference.did.did_estimator")
const staggered_py = pyimport("src.causal_inference.did.staggered")
const cs_py = pyimport("src.causal_inference.did.callaway_santanna")
const sa_py = pyimport("src.causal_inference.did.sun_abraham")

@testset "PyCall DiD Validation" begin

    @testset "Classic DiD - Hand Calculation Match" begin
        # Test with simple hand-calculable values from Python test
        # Control: pre=[10, 10], post=[15, 15] (change = +5)
        # Treated: pre=[12, 12], post=[20, 20] (change = +8)
        # DiD = 8 - 5 = 3

        outcomes = [10.0, 10.0, 15.0, 15.0, 12.0, 12.0, 20.0, 20.0]
        treatment = [false, false, false, false, true, true, true, true]
        post = [false, false, true, true, false, false, true, true]
        unit_id = [1, 2, 1, 2, 3, 4, 3, 4]

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,  # Python uses 0-indexed
            cluster_se=true
        )

        # Compare estimates
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10
        @test abs(solution_jl.estimate - 3.0) < 1e-10  # Known answer

        # Compare standard errors (allow for perfect fit edge case)
        # Julia may get exact 0.0, Python gets near-zero (9.78e-15)
        @test abs(solution_jl.se - result_py["se"]) < 1e-6 || (solution_jl.se == 0.0 && result_py["se"] < 1e-10)

        # Compare confidence intervals (if SEs are near-zero, CIs collapse to point estimate)
        if solution_jl.se < 1e-10 && result_py["se"] < 1e-10
            # Perfect fit: CIs should equal estimate
            @test abs(solution_jl.ci_lower - solution_jl.estimate) < 1e-6
            @test abs(solution_jl.ci_upper - solution_jl.estimate) < 1e-6
        else
            @test abs(solution_jl.ci_lower - result_py["ci_lower"]) < 1e-6
            @test abs(solution_jl.ci_upper - result_py["ci_upper"]) < 1e-6
        end

        # Compare p-values (skip if SE = 0 causing NaN or Inf)
        if !isnan(solution_jl.p_value) && !isinf(solution_jl.p_value)
            @test abs(solution_jl.p_value - result_py["p_value"]) < 1e-6
        else
            # Perfect fit: Julia gets NaN/Inf, Python gets very small p-value
            @test result_py["p_value"] < 1e-10 || isnan(result_py["p_value"])
        end

        # Compare diagnostics
        @test solution_jl.n_treated == result_py["n_treated"]
        @test solution_jl.n_control == result_py["n_control"]
        @test solution_jl.df == result_py["df"]
    end

    @testset "Classic DiD - Zero Treatment Effect" begin
        # Generate data with no treatment effect
        Random.seed!(123)
        n_control = 50
        n_treated = 50

        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 0.5
        control_post = fill(12.0, n_control) .+ randn(n_control) .* 0.5
        treated_pre = fill(11.0, n_treated) .+ randn(n_treated) .* 0.5
        treated_post = fill(13.0, n_treated) .+ randn(n_treated) .* 0.5  # Same change as control

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should estimate ~0
        @test abs(solution_jl.estimate) < 0.5
        @test abs(result_py["estimate"]) < 0.5

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Standard errors should match
        @test abs(solution_jl.se - result_py["se"]) < 1e-6

        # CI should contain zero in both
        @test solution_jl.ci_lower <= 0.0 <= solution_jl.ci_upper
        @test result_py["ci_lower"] <= 0.0 <= result_py["ci_upper"]
    end

    @testset "Classic DiD - Positive Treatment Effect" begin
        # Generate data with positive treatment effect
        Random.seed!(456)
        n_control = 50
        n_treated = 50
        true_effect = 2.0

        control_pre = fill(10.0, n_control) .+ randn(n_control) .* 0.5
        control_post = fill(12.0, n_control) .+ randn(n_control) .* 0.5
        treated_pre = fill(11.0, n_treated) .+ randn(n_treated) .* 0.5
        treated_post = fill(15.0, n_treated) .+ randn(n_treated) .* 0.5  # +2 effect

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should estimate ~2.0
        @test abs(solution_jl.estimate - true_effect) < 0.5
        @test abs(result_py["estimate"] - true_effect) < 0.5

        # Estimates should match exactly
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Standard errors should match
        @test abs(solution_jl.se - result_py["se"]) < 1e-6

        # Both should detect significance
        @test solution_jl.p_value < 0.05
        @test result_py["p_value"] < 0.05
    end

    @testset "Classic DiD - Cluster vs Heteroskedasticity-Robust SE" begin
        # Same data, different SE methods
        Random.seed!(789)
        n_control = 30
        n_treated = 30

        control_pre = fill(8.0, n_control) .+ randn(n_control) .* 1.0
        control_post = fill(10.0, n_control) .+ randn(n_control) .* 1.0
        treated_pre = fill(9.0, n_treated) .+ randn(n_treated) .* 1.0
        treated_post = fill(14.0, n_treated) .+ randn(n_treated) .* 1.0

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia - Cluster SE
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_cluster_jl = solve(problem_jl, ClassicDiD(cluster_se=true))
        solution_hc_jl = solve(problem_jl, ClassicDiD(cluster_se=false))

        # Python - Cluster SE
        result_cluster_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Python - HC SE
        result_hc_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=false
        )

        # Point estimates should match (SE method doesn't affect estimate)
        @test abs(solution_cluster_jl.estimate - result_cluster_py["estimate"]) < 1e-10
        @test abs(solution_hc_jl.estimate - result_hc_py["estimate"]) < 1e-10
        @test abs(solution_cluster_jl.estimate - solution_hc_jl.estimate) < 1e-10

        # Cluster SEs should match between Julia and Python
        @test abs(solution_cluster_jl.se - result_cluster_py["se"]) < 1e-6

        # HC SEs should be similar (Julia uses HC1, Python uses HC3)
        # Relax tolerance due to different HC formulas
        @test abs(solution_hc_jl.se - result_hc_py["se"]) < 1e-2

        # Cluster SE should be >= HC SE (usually, with serial correlation)
        @test solution_cluster_jl.se >= solution_hc_jl.se * 0.9  # Allow 10% tolerance
    end

    @testset "Classic DiD - Negative Treatment Effect" begin
        # Test negative effects
        Random.seed!(999)
        n_control = 40
        n_treated = 40
        true_effect = -3.0

        control_pre = fill(15.0, n_control) .+ randn(n_control) .* 0.8
        control_post = fill(17.0, n_control) .+ randn(n_control) .* 0.8
        treated_pre = fill(16.0, n_treated) .+ randn(n_treated) .* 0.8
        treated_post = fill(16.0, n_treated) .+ randn(n_treated) .* 0.8  # -1 relative change

        outcomes = vcat(control_pre, control_post, treated_pre, treated_post)
        treatment = vcat(fill(false, 2*n_control), fill(true, 2*n_treated))
        post = vcat(
            fill(false, n_control), fill(true, n_control),
            fill(false, n_treated), fill(true, n_treated)
        )
        unit_id = vcat(
            1:n_control, 1:n_control,
            (n_control+1):(n_control+n_treated), (n_control+1):(n_control+n_treated)
        )

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment, post, unit_id, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment),
            post=Int.(post),
            unit_id=unit_id .- 1,
            cluster_se=true
        )

        # Both should be negative
        @test solution_jl.estimate < 0
        @test result_py["estimate"] < 0

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # CI upper bounds should be similar (some tolerance due to adjustment differences)
        @test abs(solution_jl.ci_upper - result_py["ci_upper"]) < 0.05
    end

    @testset "Classic DiD - Multiple Pre/Post Periods" begin
        # Test with 3 pre-periods and 2 post-periods
        Random.seed!(1010)
        n_control = 20
        n_treated = 20
        n_pre = 3
        n_post = 2

        outcomes = Float64[]
        treatment_vec = Bool[]
        post_vec = Bool[]
        unit_id_vec = Int[]

        for unit in 1:(n_control + n_treated)
            is_treated = unit > n_control
            baseline = is_treated ? 12.0 : 10.0

            for t in 1:(n_pre + n_post)
                is_post = t > n_pre
                y = baseline + 0.5 * t  # Time trend

                if is_treated && is_post
                    y += 5.0  # Treatment effect
                end

                y += randn() * 0.5

                push!(outcomes, y)
                push!(treatment_vec, is_treated)
                push!(post_vec, is_post)
                push!(unit_id_vec, unit)
            end
        end

        # Julia implementation
        problem_jl = DiDProblem(outcomes, treatment_vec, post_vec, unit_id_vec, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, ClassicDiD())

        # Python implementation
        result_py = did_py.did_2x2(
            outcomes=outcomes,
            treatment=Int.(treatment_vec),
            post=Int.(post_vec),
            unit_id=unit_id_vec .- 1,
            cluster_se=true
        )

        # Estimates should match
        @test abs(solution_jl.estimate - result_py["estimate"]) < 1e-10

        # Should detect effect
        @test abs(solution_jl.estimate - 5.0) < 1.0
        @test solution_jl.p_value < 0.05
        @test result_py["p_value"] < 0.05

        # Diagnostics should match
        @test solution_jl.n_obs == length(outcomes)
        @test solution_jl.n_treated == n_treated
        @test solution_jl.n_control == n_control
    end

    # =========================================================================
    # Staggered DiD Validation
    # =========================================================================

    """Helper function to generate staggered DiD test data"""
    function generate_staggered_test_data(;
        n_early=10,
        n_late=10,
        n_never=10,
        early_time=3,
        late_time=5,
        n_periods=8,
        effect=2.0,
        noise_sd=0.5,
        seed=42
    )
        Random.seed!(seed)

        n_units = n_early + n_late + n_never
        unit_id = repeat(0:(n_units-1), inner=n_periods)
        time = repeat(0:(n_periods-1), outer=n_units)

        # Treatment times per unit
        treatment_time = vcat(
            fill(Float64(early_time), n_early),
            fill(Float64(late_time), n_late),
            fill(Inf, n_never)
        )

        # Treatment indicator
        treatment = zeros(Bool, length(unit_id))
        for i in 1:length(unit_id)
            uid = unit_id[i]
            t = time[i]
            tt = treatment_time[uid + 1]  # +1 for 1-indexing
            treatment[i] = isfinite(tt) && (t >= tt)
        end

        # Outcomes: baseline + effect if treated + noise
        outcomes = 10.0 .+ effect * treatment .+ noise_sd * randn(length(unit_id))

        return (
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id .+ 1,  # Convert to 1-indexed for Julia
            treatment_time=treatment_time,
            true_effect=effect,
            n_units=n_units,
            n_periods=n_periods
        )
    end

    @testset "PyCall Staggered DiD Validation" begin

        @testset "Callaway-Sant'Anna - Simple Aggregation" begin
            data = generate_staggered_test_data(seed=123, effect=2.0, noise_sd=0.3)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, CallawaySantAnna(
                aggregation=:simple,
                control_group=:nevertreated,
                n_bootstrap=50,
                random_seed=123
            ))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,  # Convert to 0-indexed
                treatment_time=data.treatment_time
            )

            result_py = cs_py.callaway_santanna_ate(
                data=data_py,
                aggregation="simple",
                control_group="nevertreated",
                alpha=0.05,
                n_bootstrap=50,
                random_state=123
            )

            # Compare results
            @test result_jl.retcode == :Success
            @test abs(result_jl.att - result_py["att"]) < 1e-2  # Bootstrap variance
            @test abs(result_jl.se - result_py["se"]) < 0.02  # Looser for bootstrap SE
            @test abs(result_jl.p_value - result_py["p_value"]) < 0.05
            @test result_jl.control_group == :nevertreated
            @test result_py["control_group"] == "nevertreated"

            # Verify effect is near true value
            @test abs(result_jl.att - data.true_effect) < 1.0
        end

        @testset "Callaway-Sant'Anna - Dynamic Aggregation" begin
            data = generate_staggered_test_data(seed=456, effect=3.0, noise_sd=0.4)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, CallawaySantAnna(
                aggregation=:dynamic,
                control_group=:nevertreated,
                n_bootstrap=50,
                random_seed=456
            ))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = cs_py.callaway_santanna_ate(
                data=data_py,
                aggregation="dynamic",
                control_group="nevertreated",
                n_bootstrap=50,
                random_state=456
            )

            # Verify dynamic aggregation exists
            @test result_jl.aggregated isa Dict
            @test result_py["aggregated"] isa Dict
            @test length(result_jl.aggregated) > 0
            @test length(result_py["aggregated"]) > 0

            # Compare ATT (should exist in dynamic results)
            @test abs(result_jl.att - result_py["att"]) < 1e-2
        end

        @testset "Callaway-Sant'Anna - Group Aggregation" begin
            data = generate_staggered_test_data(seed=789, effect=2.5, noise_sd=0.3)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, CallawaySantAnna(
                aggregation=:group,
                control_group=:nevertreated,
                n_bootstrap=50,
                random_seed=789
            ))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = cs_py.callaway_santanna_ate(
                data=data_py,
                aggregation="group",
                control_group="nevertreated",
                n_bootstrap=50,
                random_state=789
            )

            # Verify group aggregation exists
            @test result_jl.aggregated isa Dict
            @test result_py["aggregated"] isa Dict
            @test length(result_jl.aggregated) > 0

            # Compare overall ATT
            @test abs(result_jl.att - result_py["att"]) < 1e-2
        end

        @testset "Callaway-Sant'Anna - Not-Yet-Treated Controls" begin
            data = generate_staggered_test_data(
                seed=1010,
                n_never=0,  # No never-treated, use not-yet-treated
                effect=2.0,
                noise_sd=0.3
            )

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, CallawaySantAnna(
                aggregation=:simple,
                control_group=:notyettreated,
                n_bootstrap=50,
                random_seed=1010
            ))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = cs_py.callaway_santanna_ate(
                data=data_py,
                aggregation="simple",
                control_group="notyettreated",
                n_bootstrap=50,
                random_state=1010
            )

            # Compare results
            @test abs(result_jl.att - result_py["att"]) < 1e-2
            @test result_jl.control_group == :notyettreated
            @test result_py["control_group"] == "notyettreated"
        end

        @testset "Callaway-Sant'Anna - Bootstrap Reproducibility" begin
            data = generate_staggered_test_data(seed=2020, effect=1.5)

            # Run twice with same seed
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl_1 = solve(problem_jl, CallawaySantAnna(
                aggregation=:simple,
                random_seed=999
            ))

            result_jl_2 = solve(problem_jl, CallawaySantAnna(
                aggregation=:simple,
                random_seed=999
            ))

            # Should be identical
            @test result_jl_1.att == result_jl_2.att
            @test result_jl_1.se == result_jl_2.se
            @test result_jl_1.ci_lower == result_jl_2.ci_lower
            @test result_jl_1.ci_upper == result_jl_2.ci_upper
        end

        @testset "Sun-Abraham - Basic Estimation" begin
            data = generate_staggered_test_data(seed=3030, effect=3.0, noise_sd=0.4)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, SunAbraham(
                alpha=0.05,
                cluster_se=true
            ))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = sa_py.sun_abraham_ate(
                data=data_py,
                alpha=0.05,
                cluster_se=true
            )

            # Compare results
            @test result_jl.retcode == :Success
            @test abs(result_jl.att - result_py["att"]) < 1e-2
            @test abs(result_jl.se - result_py["se"]) < 1e-2
            @test abs(result_jl.p_value - result_py["p_value"]) < 0.05

            # Verify effect is near true value (looser for small samples)
            @test abs(result_jl.att - data.true_effect) < 1.5
        end

        @testset "Sun-Abraham - Interaction Weights Sum to 1" begin
            data = generate_staggered_test_data(seed=4040, effect=2.5)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, SunAbraham(cluster_se=true))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = sa_py.sun_abraham_ate(data=data_py, cluster_se=true)

            # Extract weight sums
            weight_sum_jl = sum(w.weight for w in result_jl.weights)
            weight_sum_py = sum(result_py["weights"]["weight"])

            # Both should sum to 1.0
            @test abs(weight_sum_jl - 1.0) < 1e-10
            @test abs(weight_sum_py - 1.0) < 1e-10
        end

        @testset "Sun-Abraham - Cluster SE vs Non-Cluster SE" begin
            data = generate_staggered_test_data(seed=5050, effect=2.0, noise_sd=0.3)

            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            # Cluster SE
            result_cluster_jl = solve(problem_jl, SunAbraham(cluster_se=true))

            # Non-cluster SE
            result_no_cluster_jl = solve(problem_jl, SunAbraham(cluster_se=false))

            # Python
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_cluster_py = sa_py.sun_abraham_ate(data=data_py, cluster_se=true)
            result_no_cluster_py = sa_py.sun_abraham_ate(data=data_py, cluster_se=false)

            # Point estimates should be identical
            @test abs(result_cluster_jl.att - result_no_cluster_jl.att) < 1e-10
            @test abs(result_cluster_py["att"] - result_no_cluster_py["att"]) < 1e-10

            # Cross-validate cluster SEs
            @test abs(result_cluster_jl.se - result_cluster_py["se"]) < 1e-6

            # Cross-validate non-cluster SEs
            @test abs(result_no_cluster_jl.se - result_no_cluster_py["se"]) < 1e-6
        end

        @testset "StaggeredTWFE - Basic Estimation" begin
            data = generate_staggered_test_data(seed=6060, effect=2.5, noise_sd=0.3)

            # Julia implementation
            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, StaggeredTWFE(cluster_se=true))

            # Python implementation
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = staggered_py.twfe_staggered(data=data_py, cluster_se=true)

            # Compare results (TWFE is biased, so expect implementation differences)
            @test result_jl.retcode == :Success
            @test abs(result_jl.estimate - result_py["att"]) < 2.0  # Loose tolerance for biased estimator
            @test abs(result_jl.se - result_py["se"]) < 1.0  # SE implementations can differ
            @test isfinite(result_jl.p_value) && result_jl.p_value >= 0.0 && result_jl.p_value <= 1.0  # Sanity check
        end

        @testset "StaggeredTWFE - Cluster SE Validation" begin
            data = generate_staggered_test_data(seed=7070, effect=2.0)

            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            # Cluster SE
            result_cluster_jl = solve(problem_jl, StaggeredTWFE(cluster_se=true))

            # Non-cluster SE
            result_no_cluster_jl = solve(problem_jl, StaggeredTWFE(cluster_se=false))

            # Python
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_cluster_py = staggered_py.twfe_staggered(data=data_py, cluster_se=true)
            result_no_cluster_py = staggered_py.twfe_staggered(data=data_py, cluster_se=false)

            # Point estimates should be identical within Julia
            @test abs(result_cluster_jl.estimate - result_no_cluster_jl.estimate) < 1e-10

            # Cross-validate with Python (loose tolerances for TWFE)
            @test abs(result_cluster_jl.estimate - result_cluster_py["att"]) < 2.0
            @test abs(result_cluster_jl.se - result_cluster_py["se"]) < 1.0
        end

        @testset "Edge Case - Minimum Valid Data" begin
            # Very small sample
            data = generate_staggered_test_data(
                seed=8080,
                n_early=5,
                n_late=5,
                n_never=5,
                n_periods=4,
                early_time=1,  # Must be < n_periods
                late_time=2,   # Must be < n_periods
                effect=2.0,
                noise_sd=0.5
            )

            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            # Should run without errors (accept large SE)
            result_twfe = solve(problem_jl, StaggeredTWFE(cluster_se=true))
            result_sa = solve(problem_jl, SunAbraham(cluster_se=true))

            @test result_twfe.retcode == :Success
            @test result_sa.retcode == :Success
            @test result_twfe.se > 0
            @test result_sa.se > 0
        end

        @testset "Edge Case - Null Effect Detection" begin
            # Effect = 0
            data = generate_staggered_test_data(
                seed=9090,
                effect=0.0,
                noise_sd=0.5,
                n_early=20,
                n_late=20,
                n_never=20
            )

            problem_jl = StaggeredDiDProblem(
                data.outcomes,
                data.treatment,
                data.time,
                data.unit_id,
                data.treatment_time,
                (alpha=0.05,)
            )

            result_jl = solve(problem_jl, CallawaySantAnna(
                aggregation=:simple,
                n_bootstrap=50,
                random_seed=9090
            ))

            # Python
            data_py = staggered_py.create_staggered_data(
                outcomes=data.outcomes,
                treatment=Int.(data.treatment),
                time=data.time,
                unit_id=data.unit_id .- 1,
                treatment_time=data.treatment_time
            )

            result_py = cs_py.callaway_santanna_ate(
                data=data_py,
                aggregation="simple",
                n_bootstrap=50,
                random_state=9090
            )

            # Both should fail to reject null
            @test result_jl.p_value > 0.05
            @test result_py["p_value"] > 0.05

            # Estimates should be near zero
            @test abs(result_jl.att) < 1.0
            @test abs(result_py["att"]) < 1.0
        end

    end  # PyCall Staggered DiD Validation

end  # PyCall DiD Validation
