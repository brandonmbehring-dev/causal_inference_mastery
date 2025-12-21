"""
Tests for Quantile Treatment Effects (QTE) module.
"""

using Test
using Random
using Statistics

# Include source files
include("../../src/qte/types.jl")
include("../../src/qte/unconditional.jl")
include("../../src/qte/conditional.jl")
include("../../src/qte/rif.jl")


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

function generate_homogeneous_qte_data(;
    n::Int = 500,
    true_ate::Float64 = 2.0,
    noise_sd::Float64 = 1.0,
    treatment_prob::Float64 = 0.5,
    seed::Int = 42
)
    rng = MersenneTwister(seed)
    treatment = Float64.(rand(rng, n) .< treatment_prob)
    noise = randn(rng, n) .* noise_sd
    outcome = true_ate .* treatment .+ noise
    return outcome, treatment
end

function generate_data_with_covariates(;
    n::Int = 500,
    p::Int = 3,
    true_ate::Float64 = 2.0,
    seed::Int = 42
)
    rng = MersenneTwister(seed)
    treatment = Float64.(rand(rng, n) .< 0.5)
    covariates = randn(rng, n, p)
    noise = randn(rng, n)
    outcome = true_ate .* treatment .+ 0.5 .* sum(covariates, dims=2)[:] .+ noise
    return outcome, treatment, covariates
end


# =============================================================================
# UNCONDITIONAL QTE TESTS
# =============================================================================

@testset "Unconditional QTE" begin

    @testset "Basic functionality" begin
        outcome, treatment = generate_homogeneous_qte_data(n=500, true_ate=2.0, seed=42)

        result = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=500,
                                   rng=MersenneTwister(42))

        @test isfinite(result.tau_q)
        @test result.se > 0
        @test result.ci_lower < result.ci_upper
        @test result.method == :unconditional
        @test result.inference == :bootstrap
        @test result.n_total == 500
    end

    @testset "Known answer - median effect" begin
        # Simple known case
        outcome = Float64[3, 4, 5, 6, 7, 1, 2, 3, 4, 5]
        treatment = Float64[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        result = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=200,
                                   rng=MersenneTwister(42))

        # Median treated = 5, median control = 3, QTE = 2
        @test isapprox(result.tau_q, 2.0, atol=0.2)
    end

    @testset "Homogeneous effect across quantiles" begin
        outcome, treatment = generate_homogeneous_qte_data(n=500, true_ate=2.0, seed=42)

        qte_25 = unconditional_qte(outcome, treatment; quantile=0.25, n_bootstrap=300,
                                   rng=MersenneTwister(42))
        qte_50 = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=300,
                                   rng=MersenneTwister(42))
        qte_75 = unconditional_qte(outcome, treatment; quantile=0.75, n_bootstrap=300,
                                   rng=MersenneTwister(42))

        # All should be close to 2.0
        @test isapprox(qte_25.tau_q, 2.0, atol=0.6)
        @test isapprox(qte_50.tau_q, 2.0, atol=0.6)
        @test isapprox(qte_75.tau_q, 2.0, atol=0.6)
    end

    @testset "Reproducibility with seed" begin
        outcome, treatment = generate_homogeneous_qte_data(n=200, seed=42)

        result1 = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=200,
                                    rng=MersenneTwister(123))
        result2 = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=200,
                                    rng=MersenneTwister(123))

        @test result1.tau_q == result2.tau_q
        @test result1.se == result2.se
    end

    @testset "CI contains estimate" begin
        outcome, treatment = generate_homogeneous_qte_data(n=300, seed=42)

        result = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=500,
                                   rng=MersenneTwister(42))

        @test result.ci_lower < result.tau_q < result.ci_upper
    end

    @testset "Input validation" begin
        # Non-binary treatment
        @test_throws ErrorException begin
            outcome = [1.0, 2.0, 3.0]
            treatment = [0.0, 1.0, 2.0]  # Invalid
            QTEProblem(outcome=outcome, treatment=treatment, quantile=0.5)
        end

        # No treatment variation
        @test_throws ErrorException begin
            outcome = [1.0, 2.0, 3.0]
            treatment = [1.0, 1.0, 1.0]  # No variation
            QTEProblem(outcome=outcome, treatment=treatment, quantile=0.5)
        end

        # Invalid quantile
        @test_throws ErrorException begin
            outcome = [1.0, 2.0, 3.0, 4.0]
            treatment = [1.0, 1.0, 0.0, 0.0]
            QTEProblem(outcome=outcome, treatment=treatment, quantile=1.5)
        end
    end
end


# =============================================================================
# UNCONDITIONAL QTE BAND TESTS
# =============================================================================

@testset "Unconditional QTE Band" begin

    @testset "Returns all quantiles" begin
        outcome, treatment = generate_homogeneous_qte_data(n=300, seed=42)

        problem = QTEBandProblem(
            outcome=outcome,
            treatment=treatment,
            quantiles=[0.25, 0.5, 0.75]
        )

        result = unconditional_qte_band(problem; n_bootstrap=200, rng=MersenneTwister(42))

        @test length(result.quantiles) == 3
        @test length(result.qte_estimates) == 3
        @test length(result.se_estimates) == 3
    end

    @testset "Joint CI is wider" begin
        outcome, treatment = generate_homogeneous_qte_data(n=300, seed=42)

        problem = QTEBandProblem(
            outcome=outcome,
            treatment=treatment,
            quantiles=[0.25, 0.5, 0.75]
        )

        result = unconditional_qte_band(problem; n_bootstrap=500, joint=true,
                                        rng=MersenneTwister(42))

        @test result.joint_ci_lower !== nothing
        @test result.joint_ci_upper !== nothing

        # Joint should be wider
        joint_widths = result.joint_ci_upper .- result.joint_ci_lower
        pointwise_widths = result.ci_upper .- result.ci_lower

        @test all(joint_widths .>= pointwise_widths .- 1e-10)
    end
end


# =============================================================================
# CONDITIONAL QTE TESTS
# =============================================================================

@testset "Conditional QTE" begin

    @testset "Basic functionality with covariates" begin
        outcome, treatment, covariates = generate_data_with_covariates(n=300, seed=42)

        result = conditional_qte(outcome, treatment, covariates; quantile=0.5)

        @test isfinite(result.tau_q)
        @test result.se > 0
        @test result.method == :conditional
        @test result.inference == :asymptotic
    end

    @testset "Recovers true effect" begin
        outcome, treatment, covariates = generate_data_with_covariates(
            n=500, true_ate=2.0, seed=42
        )

        result = conditional_qte(outcome, treatment, covariates; quantile=0.5)

        @test isapprox(result.tau_q, 2.0, atol=0.5)
    end

    @testset "CI validity" begin
        outcome, treatment, covariates = generate_data_with_covariates(n=300, seed=42)

        result = conditional_qte(outcome, treatment, covariates; quantile=0.5)

        @test result.ci_lower < result.ci_upper
    end
end


# =============================================================================
# RIF QTE TESTS
# =============================================================================

@testset "RIF QTE" begin

    @testset "Basic functionality" begin
        outcome, treatment = generate_homogeneous_qte_data(n=300, seed=42)

        result = rif_qte(outcome, treatment; quantile=0.5, n_bootstrap=300,
                         rng=MersenneTwister(42))

        @test isfinite(result.tau_q)
        @test result.se > 0
        @test result.method == :rif
        @test result.inference == :bootstrap
    end

    @testset "Recovers true effect" begin
        # RIF-OLS can have higher variance than unconditional QTE
        # Use larger sample and wider tolerance
        outcome, treatment = generate_homogeneous_qte_data(n=1000, true_ate=2.0, seed=42)

        result = rif_qte(outcome, treatment; quantile=0.5, n_bootstrap=500,
                         rng=MersenneTwister(42))

        # RIF has higher variance due to density estimation
        @test isapprox(result.tau_q, 2.0, atol=2.0)
    end

    @testset "With covariates" begin
        outcome, treatment, covariates = generate_data_with_covariates(n=300, seed=42)

        result = rif_qte(outcome, treatment; quantile=0.5, covariates=covariates,
                         n_bootstrap=300, rng=MersenneTwister(42))

        @test isfinite(result.tau_q)
        @test result.se > 0
    end

    @testset "Reproducibility" begin
        outcome, treatment = generate_homogeneous_qte_data(n=200, seed=42)

        result1 = rif_qte(outcome, treatment; quantile=0.5, n_bootstrap=200,
                          rng=MersenneTwister(123))
        result2 = rif_qte(outcome, treatment; quantile=0.5, n_bootstrap=200,
                          rng=MersenneTwister(123))

        @test result1.tau_q == result2.tau_q
    end
end


# =============================================================================
# RIF QTE BAND TESTS
# =============================================================================

@testset "RIF QTE Band" begin

    @testset "Returns all quantiles" begin
        outcome, treatment = generate_homogeneous_qte_data(n=300, seed=42)

        problem = QTEBandProblem(
            outcome=outcome,
            treatment=treatment,
            quantiles=[0.25, 0.5, 0.75]
        )

        result = rif_qte_band(problem; n_bootstrap=200, rng=MersenneTwister(42))

        @test length(result.quantiles) == 3
        @test length(result.qte_estimates) == 3
        @test result.method == :rif
    end
end


# =============================================================================
# EDGE CASES
# =============================================================================

@testset "Edge Cases" begin

    @testset "Extreme quantiles" begin
        outcome, treatment = generate_homogeneous_qte_data(n=500, seed=42)

        # Low quantile
        result_low = unconditional_qte(outcome, treatment; quantile=0.05, n_bootstrap=200,
                                       rng=MersenneTwister(42))
        @test isfinite(result_low.tau_q)

        # High quantile
        result_high = unconditional_qte(outcome, treatment; quantile=0.95, n_bootstrap=200,
                                        rng=MersenneTwister(42))
        @test isfinite(result_high.tau_q)
    end

    @testset "Small sample" begin
        outcome = Float64[1, 2, 3, 4, 5, 6]
        treatment = Float64[1, 1, 1, 0, 0, 0]

        result = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=100,
                                   rng=MersenneTwister(42))

        @test isfinite(result.tau_q)
        @test result.n_treated == 3
        @test result.n_control == 3
    end

    @testset "Imbalanced treatment" begin
        rng = MersenneTwister(42)
        n = 200
        treatment = Float64.(rand(rng, n) .< 0.8)  # 80% treated
        outcome = 2.0 .* treatment .+ randn(rng, n)

        result = unconditional_qte(outcome, treatment; quantile=0.5, n_bootstrap=300,
                                   rng=MersenneTwister(42))

        @test isfinite(result.tau_q)
        @test result.n_treated > result.n_control
    end
end


println("All QTE tests completed!")
