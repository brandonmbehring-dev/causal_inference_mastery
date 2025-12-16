"""
Tests for Fuzzy Regression Discontinuity Design (RDD) in Julia.

Validates:
- Fuzzy RDD with 2SLS estimation
- Perfect compliance (Fuzzy = Sharp)
- Partial compliance scenarios (high, moderate, low)
- First-stage diagnostics (F-statistic, compliance rate)
- Weak instrument detection
- Bandwidth selection
- Error handling

Test Structure:
- Layer 1: Known-Answer Tests (10 tests)
- Layer 2: Adversarial Tests (8 tests)
"""

using Test
using CausalEstimators
using Statistics
using Random
using LinearAlgebra
using Logging

# ===========================================================================
# Helper Functions: Data Generating Processes
# ===========================================================================

"""
Generate Fuzzy RDD data with specified compliance rate.

DGP: Y = 1.0 + 0.5 * X + tau * D + eps
     D ~ Bernoulli(p) where p depends on Z = 1{X >= cutoff}

# Arguments
- `n`: Sample size
- `cutoff`: Threshold value
- `tau`: True LATE (treatment effect for compliers)
- `p_comply_above`: P(D=1|Z=1) - treatment probability when eligible
- `p_comply_below`: P(D=1|Z=0) - treatment probability when not eligible
- `seed`: Random seed
"""
function generate_fuzzy_rdd_data(
    n::Int,
    cutoff::Float64,
    tau::Float64;
    p_comply_above::Float64 = 0.8,
    p_comply_below::Float64 = 0.2,
    noise_sd::Float64 = 0.5,
    seed::Int = 42
)
    Random.seed!(seed)

    # Running variable: X ~ Uniform(cutoff-2, cutoff+2)
    X = rand(n) .* 4 .- 2 .+ cutoff

    # Eligibility (instrument)
    Z = X .>= cutoff

    # Treatment with imperfect compliance
    D = zeros(Float64, n)
    for i in 1:n
        if Z[i]
            D[i] = rand() < p_comply_above ? 1.0 : 0.0
        else
            D[i] = rand() < p_comply_below ? 1.0 : 0.0
        end
    end

    # Outcome
    eps = randn(n) .* noise_sd
    Y = 1.0 .+ 0.5 .* X .+ tau .* D .+ eps

    # True compliance rate
    true_compliance = p_comply_above - p_comply_below

    return Y, X, D, true_compliance
end

"""Generate perfect compliance data (Sharp RDD-like)."""
function generate_perfect_compliance_data(n::Int, cutoff::Float64, tau::Float64; seed::Int = 42)
    return generate_fuzzy_rdd_data(n, cutoff, tau;
        p_comply_above = 1.0,
        p_comply_below = 0.0,
        seed = seed
    )
end

"""Generate high compliance data (≈0.8)."""
function generate_high_compliance_data(n::Int, cutoff::Float64, tau::Float64; seed::Int = 42)
    return generate_fuzzy_rdd_data(n, cutoff, tau;
        p_comply_above = 0.9,
        p_comply_below = 0.1,
        seed = seed
    )
end

"""Generate moderate compliance data (≈0.5)."""
function generate_moderate_compliance_data(n::Int, cutoff::Float64, tau::Float64; seed::Int = 42)
    return generate_fuzzy_rdd_data(n, cutoff, tau;
        p_comply_above = 0.75,
        p_comply_below = 0.25,
        seed = seed
    )
end

"""Generate low compliance data (≈0.2) - weak instrument scenario."""
function generate_low_compliance_data(n::Int, cutoff::Float64, tau::Float64; seed::Int = 42)
    return generate_fuzzy_rdd_data(n, cutoff, tau;
        p_comply_above = 0.6,
        p_comply_below = 0.4,
        seed = seed
    )
end


@testset "Fuzzy RDD Tests" begin

    # =========================================================================
    # Layer 1: Known-Answer Tests
    # =========================================================================

    @testset "Known-Answer Tests" begin

        @testset "Perfect compliance matches Sharp RDD" begin
            # Generate data with perfect compliance (D = Z)
            Y, X, D, _ = generate_perfect_compliance_data(500, 0.0, 2.0; seed=42)
            cutoff = 0.0
            tau = 2.0

            # Fuzzy RDD
            problem_fuzzy = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution_fuzzy = solve(problem_fuzzy, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Sharp RDD (with Bool treatment)
            treatment_bool = X .>= cutoff
            problem_sharp = RDDProblem(Y, X, treatment_bool, cutoff, nothing, (alpha=0.05,))
            solution_sharp = solve(problem_sharp, SharpRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Estimates should be close with perfect compliance
            @test abs(solution_fuzzy.estimate - solution_sharp.estimate) < 0.1

            # Both should recover true effect
            @test abs(solution_fuzzy.estimate - tau) < 0.4
            @test abs(solution_sharp.estimate - tau) < 0.4

            # Compliance should be ≈ 1.0
            @test solution_fuzzy.compliance_rate > 0.95
        end

        @testset "High compliance recovers LATE" begin
            Y, X, D, true_compliance = generate_high_compliance_data(500, 0.0, 2.0; seed=123)
            cutoff = 0.0
            tau = 2.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Should recover LATE within tolerance
            @test abs(solution.estimate - tau) < 0.6

            # Compliance should be ≈ 0.8
            @test 0.6 < solution.compliance_rate < 0.95

            # Strong instrument: F > 30
            @test solution.first_stage_fstat > 30

            # Should not trigger weak instrument warning
            @test !solution.weak_instrument_warning
        end

        @testset "Moderate compliance recovers LATE" begin
            Y, X, D, true_compliance = generate_moderate_compliance_data(500, 0.0, 2.0; seed=456)
            cutoff = 0.0
            tau = 2.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Should recover LATE (relaxed tolerance)
            @test abs(solution.estimate - tau) < 0.8

            # Compliance should be ≈ 0.5
            @test 0.35 < solution.compliance_rate < 0.65

            # Decent instrument: F > 15
            @test solution.first_stage_fstat > 15
        end

        @testset "Zero effect estimate near zero" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 0.0; seed=789)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # With zero true effect, estimate should be small
            @test abs(solution.estimate) < 1.0

            # Estimate and SE should be finite
            @test isfinite(solution.estimate)
            @test isfinite(solution.se)
        end

        @testset "First-stage F-statistic computation" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=101)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # F-stat should be finite and positive
            @test isfinite(solution.first_stage_fstat)
            @test solution.first_stage_fstat > 0

            # With high compliance, F should be strong
            @test solution.first_stage_fstat > 10
        end

        @testset "Compliance rate calculation" begin
            Y, X, D, true_compliance = generate_moderate_compliance_data(500, 0.0, 2.0; seed=202)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Compliance should be finite
            @test isfinite(solution.compliance_rate)

            # Compliance should be in (0, 1)
            @test 0 < solution.compliance_rate < 1

            # Should be close to true compliance (0.5)
            @test abs(solution.compliance_rate - true_compliance) < 0.2
        end

        @testset "IK bandwidth selection" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=303)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Bandwidth should be positive and reasonable
            @test solution.bandwidth > 0
            @test solution.bandwidth < 5.0  # Not too large
            @test isfinite(solution.bandwidth)
        end

        @testset "CCT bandwidth selection" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=404)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=CCTBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Bandwidth should be positive and reasonable
            @test solution.bandwidth > 0
            @test solution.bandwidth < 5.0
            @test isfinite(solution.bandwidth)

            # Estimate should be reasonable
            @test abs(solution.estimate - 2.0) < 0.6
        end

        @testset "Confidence intervals" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=505)
            cutoff = 0.0
            tau = 2.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # CI should be ordered
            @test solution.ci_lower < solution.ci_upper

            # CI should be finite
            @test isfinite(solution.ci_lower)
            @test isfinite(solution.ci_upper)

            # CI should contain true effect (with some margin)
            @test solution.ci_lower - 0.5 < tau < solution.ci_upper + 0.5
        end

        @testset "Negative treatment effect" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, -1.5; seed=606)
            cutoff = 0.0
            tau = -1.5

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Should detect negative effect
            @test solution.estimate < 0

            # Should be close to true value
            @test abs(solution.estimate - tau) < 0.6
        end

    end  # Known-Answer Tests


    # =========================================================================
    # Layer 2: Adversarial Tests
    # =========================================================================

    @testset "Adversarial Tests" begin

        @testset "Weak instrument warning" begin
            Y, X, D, _ = generate_low_compliance_data(500, 0.0, 2.0; seed=701)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))

            # Solve (may trigger warnings)
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # With low compliance, weak instrument warning should be set
            # F-stat should still be computed
            @test isfinite(solution.first_stage_fstat)

            # Low compliance should produce weak instrument (F < 10)
            @test solution.weak_instrument_warning || solution.first_stage_fstat < 15
        end

        @testset "Low compliance warning" begin
            # Very low compliance scenario
            Y, X, D, _ = generate_fuzzy_rdd_data(500, 0.0, 2.0;
                p_comply_above = 0.55,
                p_comply_below = 0.45,
                seed = 702
            )
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))

            # May trigger low compliance warning
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Compliance should be low
            @test solution.compliance_rate < 0.3
        end

        @testset "No variation in treatment - all zeros" begin
            Random.seed!(703)
            n = 500
            cutoff = 0.0

            X = rand(n) .* 4 .- 2
            D = zeros(Float64, n)  # No variation
            Y = 1.0 .+ 0.5 .* X .+ randn(n) .* 0.5

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))

            # Should error or produce NaN due to no variation
            @test_throws Union{ArgumentError, SingularException} begin
                solve(problem, FuzzyRDD(
                    bandwidth_method=IKBandwidth(),
                    kernel=TriangularKernel(),
                    run_density_test=false
                ))
            end
        end

        @testset "No variation in instrument - all observations one side" begin
            Random.seed!(704)
            n = 500
            cutoff = 0.0

            # All X < cutoff (all on left side)
            X = rand(n) .* 2 .- 3  # X in [-3, -1]
            D = rand(n) .> 0.5  # Random treatment
            Y = 1.0 .+ 0.5 .* X .+ 2.0 .* D .+ randn(n) .* 0.5

            # Should error because cutoff outside data range
            @test_throws ArgumentError begin
                RDDProblem(Y, X, Float64.(D), cutoff, nothing, (alpha=0.05,))
            end
        end

        @testset "Small sample size handling" begin
            # Very small sample
            Random.seed!(705)
            n = 50  # Small
            cutoff = 0.0

            Y, X, D, _ = generate_high_compliance_data(n, cutoff, 2.0; seed=705)

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))

            # Should still produce a result (may be imprecise)
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            @test isfinite(solution.estimate)
            @test isfinite(solution.se)
        end

        @testset "Large sample size" begin
            Y, X, D, _ = generate_high_compliance_data(2000, 0.0, 2.0; seed=706)
            cutoff = 0.0
            tau = 2.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # With more data, estimate should be more precise
            @test abs(solution.estimate - tau) < 0.3

            # SE should be smaller
            @test solution.se < 0.5

            # F-stat should be very strong
            @test solution.first_stage_fstat > 50
        end

        @testset "Uniform kernel" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=707)
            cutoff = 0.0
            tau = 2.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=UniformKernel(),
                run_density_test=false
            ))

            # Should still recover effect
            @test abs(solution.estimate - tau) < 0.6

            # Kernel should be stored correctly
            @test occursin("uniform", lowercase(string(solution.kernel)))
        end

        @testset "Non-zero cutoff" begin
            cutoff = 5.0
            tau = 2.0

            Random.seed!(708)
            n = 500
            X = rand(n) .* 6 .+ 2  # X in [2, 8], cutoff at 5
            Z = X .>= cutoff
            D = zeros(Float64, n)
            for i in 1:n
                D[i] = Z[i] ? (rand() < 0.85 ? 1.0 : 0.0) : (rand() < 0.15 ? 1.0 : 0.0)
            end
            eps = randn(n) .* 0.5
            Y = 0.5 .* X .+ tau .* D .+ eps

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Should recover effect
            @test abs(solution.estimate - tau) < 0.6
        end

    end  # Adversarial Tests


    # =========================================================================
    # Layer 3: Return Code Tests
    # =========================================================================

    @testset "Return Codes" begin

        @testset "Success return code" begin
            Y, X, D, _ = generate_high_compliance_data(500, 0.0, 2.0; seed=801)
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            @test solution.retcode == :Success
        end

        @testset "WeakInstrument return code" begin
            # Very weak compliance
            Y, X, D, _ = generate_fuzzy_rdd_data(500, 0.0, 2.0;
                p_comply_above = 0.52,
                p_comply_below = 0.48,
                seed = 802
            )
            cutoff = 0.0

            problem = RDDProblem(Y, X, D, cutoff, nothing, (alpha=0.05,))
            solution = solve(problem, FuzzyRDD(
                bandwidth_method=IKBandwidth(),
                kernel=TriangularKernel(),
                run_density_test=false
            ))

            # Should indicate weak instrument
            @test solution.retcode == :WeakInstrument || solution.first_stage_fstat < 10
        end

    end  # Return Codes

end  # Fuzzy RDD Tests
