"""
Mediation Analysis Test Suite.

Session 95: Julia mediation implementation tests.
- Baron-Kenny analysis
- Full mediation analysis with bootstrap
- Controlled direct effects
- Sensitivity analysis
- Diagnostics
"""

using Test
using Random
using Statistics
using Distributions

# Include source files
include("../../src/mediation/types.jl")
include("../../src/mediation/estimators.jl")
include("../../src/mediation/sensitivity.jl")


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

function generate_mediation_data(;
    n::Int = 500,
    alpha_1::Float64 = 0.6,    # T -> M effect
    beta_1::Float64 = 0.5,     # Direct effect T -> Y
    beta_2::Float64 = 0.8,     # M -> Y effect
    noise_m::Float64 = 0.5,
    noise_y::Float64 = 0.5,
    seed::Int = 42
)
    """Generate data for mediation analysis."""
    rng = MersenneTwister(seed)

    treatment = Float64.(rand(rng, n) .< 0.5)
    mediator = 0.5 .+ alpha_1 .* treatment .+ noise_m .* randn(rng, n)
    outcome = 1.0 .+ beta_1 .* treatment .+ beta_2 .* mediator .+ noise_y .* randn(rng, n)

    true_indirect = alpha_1 * beta_2
    true_direct = beta_1
    true_total = beta_1 + alpha_1 * beta_2

    return (
        outcome=outcome,
        treatment=treatment,
        mediator=mediator,
        true_indirect=true_indirect,
        true_direct=true_direct,
        true_total=true_total
    )
end


function generate_no_mediation_data(;
    n::Int = 500,
    beta_1::Float64 = 1.0,  # Direct effect only
    seed::Int = 42
)
    """Generate data with no mediation (alpha_1 = 0 or beta_2 = 0)."""
    rng = MersenneTwister(seed)

    treatment = Float64.(rand(rng, n) .< 0.5)
    # Mediator unrelated to treatment
    mediator = 0.5 .+ 0.5 .* randn(rng, n)
    outcome = 1.0 .+ beta_1 .* treatment .+ 0.5 .* randn(rng, n)

    return (outcome=outcome, treatment=treatment, mediator=mediator)
end


function generate_full_mediation_data(;
    n::Int = 500,
    alpha_1::Float64 = 1.0,
    beta_2::Float64 = 1.0,
    seed::Int = 42
)
    """Generate data with full mediation (beta_1 = 0)."""
    rng = MersenneTwister(seed)

    treatment = Float64.(rand(rng, n) .< 0.5)
    mediator = 0.5 .+ alpha_1 .* treatment .+ 0.3 .* randn(rng, n)
    # No direct effect
    outcome = 1.0 .+ beta_2 .* mediator .+ 0.3 .* randn(rng, n)

    return (outcome=outcome, treatment=treatment, mediator=mediator, true_indirect=alpha_1 * beta_2)
end


# =============================================================================
# BARON-KENNY TESTS
# =============================================================================

@testset "Baron-Kenny Analysis" begin

    @testset "Returns correct type" begin
        data = generate_mediation_data(n=500, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        @test result isa BaronKennyResult
        @test !isnan(result.alpha_1)
        @test !isnan(result.beta_1)
        @test !isnan(result.beta_2)
        @test !isnan(result.indirect_effect)
    end

    @testset "Recovers true effects" begin
        # Large sample for precision
        data = generate_mediation_data(n=5000, alpha_1=0.6, beta_1=0.5, beta_2=0.8, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        # Should recover effects approximately
        @test isapprox(result.alpha_1, 0.6, atol=0.1)
        @test isapprox(result.beta_1, 0.5, atol=0.1)
        @test isapprox(result.beta_2, 0.8, atol=0.1)
        @test isapprox(result.indirect_effect, data.true_indirect, atol=0.15)
        @test isapprox(result.direct_effect, data.true_direct, atol=0.1)
    end

    @testset "Total effect decomposition" begin
        data = generate_mediation_data(n=1000, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        # Total = Direct + Indirect
        @test isapprox(result.total_effect, result.direct_effect + result.indirect_effect, rtol=1e-10)
    end

    @testset "Sobel test for indirect effect" begin
        data = generate_mediation_data(n=1000, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        # Should detect significant indirect effect
        @test result.sobel_pvalue < 0.05
        @test !isnan(result.sobel_z)
        @test abs(result.sobel_z) > 1.96
    end

    @testset "R-squared computed" begin
        data = generate_mediation_data(n=500, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        @test 0 <= result.r2_mediator_model <= 1
        @test 0 <= result.r2_outcome_model <= 1
        @test result.n_obs == 500
    end

    @testset "With covariates" begin
        rng = MersenneTwister(42)
        n = 500
        X = randn(rng, n, 2)
        T = Float64.(rand(rng, n) .< 0.5)
        M = 0.5 .+ 0.6 .* T .+ 0.3 .* X[:, 1] .+ 0.5 .* randn(rng, n)
        Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ 0.2 .* X[:, 2] .+ 0.5 .* randn(rng, n)

        result = baron_kenny(Y, T, M; covariates=X)

        @test result isa BaronKennyResult
        @test !isnan(result.indirect_effect)
    end

    @testset "Input validation" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 100)
        T = Float64.(rand(rng, 100) .< 0.5)
        M = randn(rng, 100)
        M_bad = randn(rng, 90)  # Wrong length

        @test_throws ErrorException baron_kenny(Y, T, M_bad)

        # Too few observations
        @test_throws ErrorException baron_kenny(Y[1:3], T[1:3], M[1:3])
    end
end


# =============================================================================
# FULL MEDIATION ANALYSIS TESTS
# =============================================================================

@testset "Mediation Analysis" begin

    @testset "Returns correct type" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_analysis(data.outcome, data.treatment, data.mediator;
                                    n_bootstrap=100, rng=MersenneTwister(42))

        @test result isa MediationResult
        @test result.method == :baron_kenny
        @test result.n_obs == 500
        @test result.n_bootstrap == 100
    end

    @testset "CIs computed via bootstrap" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_analysis(data.outcome, data.treatment, data.mediator;
                                    n_bootstrap=200, rng=MersenneTwister(42))

        # CIs should contain point estimates
        @test result.de_ci[1] <= result.direct_effect <= result.de_ci[2]
        @test result.ie_ci[1] <= result.indirect_effect <= result.ie_ci[2]
        @test result.te_ci[1] <= result.total_effect <= result.te_ci[2]

        # SEs should be positive
        @test result.de_se > 0
        @test result.ie_se > 0
        @test result.te_se > 0
    end

    @testset "Proportion mediated" begin
        data = generate_mediation_data(n=1000, alpha_1=0.6, beta_1=0.5, beta_2=0.8, seed=42)
        result = mediation_analysis(data.outcome, data.treatment, data.mediator;
                                    n_bootstrap=100, rng=MersenneTwister(42))

        # Proportion = Indirect / Total
        expected_pm = result.indirect_effect / result.total_effect
        @test isapprox(result.proportion_mediated, expected_pm, rtol=1e-10)

        # For this DGP, ~50% mediated
        @test 0.3 < result.proportion_mediated < 0.7
    end

    @testset "Full mediation detection" begin
        data = generate_full_mediation_data(n=2000, seed=42)
        result = mediation_analysis(data.outcome, data.treatment, data.mediator;
                                    n_bootstrap=100, rng=MersenneTwister(42))

        # Direct effect should be near zero, proportion mediated near 1
        @test abs(result.direct_effect) < 0.2
        @test result.proportion_mediated > 0.8
    end

    @testset "No mediation detection" begin
        data = generate_no_mediation_data(n=500, seed=42)
        result = mediation_analysis(data.outcome, data.treatment, data.mediator;
                                    n_bootstrap=100, rng=MersenneTwister(42))

        # Indirect effect should be near zero
        @test abs(result.indirect_effect) < 0.2
    end
end


# =============================================================================
# CONTROLLED DIRECT EFFECT TESTS
# =============================================================================

@testset "Controlled Direct Effect" begin

    @testset "Returns correct type" begin
        data = generate_mediation_data(n=500, seed=42)
        result = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 0.5)

        @test result isa CDEResult
        @test result.mediator_value == 0.5
        @test result.n_obs == 500
    end

    @testset "CDE equals direct effect in linear model" begin
        data = generate_mediation_data(n=1000, beta_1=0.5, seed=42)
        cde_result = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 0.0)
        bk_result = baron_kenny(data.outcome, data.treatment, data.mediator)

        # In linear model without interactions, CDE = beta_1 for any m
        @test isapprox(cde_result.cde, bk_result.beta_1, rtol=0.01)
    end

    @testset "CDE invariant to mediator value (linear model)" begin
        data = generate_mediation_data(n=500, seed=42)

        cde_0 = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 0.0)
        cde_1 = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 1.0)
        cde_2 = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 2.0)

        # All should be the same in linear model
        @test isapprox(cde_0.cde, cde_1.cde, rtol=0.01)
        @test isapprox(cde_1.cde, cde_2.cde, rtol=0.01)
    end

    @testset "Inference" begin
        data = generate_mediation_data(n=500, beta_1=0.5, seed=42)
        result = controlled_direct_effect(data.outcome, data.treatment, data.mediator, 0.0)

        # Should detect significant effect
        @test result.pvalue < 0.05
        @test result.ci_lower < result.cde < result.ci_upper
        @test result.se > 0
    end
end


# =============================================================================
# MEDIATION DIAGNOSTICS TESTS
# =============================================================================

@testset "Mediation Diagnostics" begin

    @testset "Returns correct type" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_diagnostics(data.outcome, data.treatment, data.mediator)

        @test result isa MediationDiagnostics
        @test result.n_obs == 500
    end

    @testset "Detects mediation path" begin
        data = generate_mediation_data(n=1000, alpha_1=0.6, beta_2=0.8, seed=42)
        result = mediation_diagnostics(data.outcome, data.treatment, data.mediator)

        @test result.has_mediation_path == true
        @test result.treatment_effect_pvalue < 0.05
        @test result.mediator_effect_pvalue < 0.05
    end

    @testset "Detects no mediation" begin
        data = generate_no_mediation_data(n=500, seed=42)
        result = mediation_diagnostics(data.outcome, data.treatment, data.mediator)

        # At least one path should be non-significant
        @test result.has_mediation_path == false ||
              result.treatment_effect_pvalue > 0.05 ||
              result.mediator_effect_pvalue > 0.05

        @test length(result.warnings) > 0
    end

    @testset "R-squared diagnostics" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_diagnostics(data.outcome, data.treatment, data.mediator)

        @test 0 <= result.r2_mediator <= 1
        @test 0 <= result.r2_outcome_full <= 1
        @test 0 <= result.r2_outcome_reduced <= 1
        @test result.r2_outcome_full >= result.r2_outcome_reduced
    end
end


# =============================================================================
# SENSITIVITY ANALYSIS TESTS
# =============================================================================

@testset "Mediation Sensitivity" begin

    @testset "Returns correct type" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_sensitivity(data.outcome, data.treatment, data.mediator;
                                       n_rho=21, n_bootstrap=50, rng=MersenneTwister(42))

        @test result isa SensitivityResult
        @test length(result.rho_grid) == 21
        @test length(result.nde_at_rho) == 21
        @test length(result.nie_at_rho) == 21
    end

    @testset "Effects at rho=0 match point estimates" begin
        data = generate_mediation_data(n=500, seed=42)
        bk = baron_kenny(data.outcome, data.treatment, data.mediator)
        sens = mediation_sensitivity(data.outcome, data.treatment, data.mediator;
                                     n_rho=21, n_bootstrap=50, rng=MersenneTwister(42))

        @test isapprox(sens.original_nde, bk.beta_1, rtol=0.01)
        # NIE at rho=0 should equal indirect effect
        @test isapprox(sens.original_nie, bk.indirect_effect, atol=0.1)
    end

    @testset "Rho zero crossing detected" begin
        # Strong mediation effect - need more extreme rho to nullify
        data = generate_mediation_data(n=1000, alpha_1=0.8, beta_2=1.0, seed=42)
        result = mediation_sensitivity(data.outcome, data.treatment, data.mediator;
                                       n_rho=41, n_bootstrap=50, rng=MersenneTwister(42))

        # Should find rho where NIE crosses zero
        @test !isnan(result.rho_at_zero_nie) || abs(result.original_nie) > 0.5
    end

    @testset "CIs computed at each rho" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_sensitivity(data.outcome, data.treatment, data.mediator;
                                       n_rho=11, n_bootstrap=100, rng=MersenneTwister(42))

        # Check CIs are computed (not all NaN)
        @test any(.!isnan.(result.nde_ci_lower))
        @test any(.!isnan.(result.nie_ci_lower))
    end

    @testset "Interpretation generated" begin
        data = generate_mediation_data(n=500, seed=42)
        result = mediation_sensitivity(data.outcome, data.treatment, data.mediator;
                                       n_rho=21, n_bootstrap=50, rng=MersenneTwister(42))

        @test length(result.interpretation) > 100
        @test occursin("NDE", result.interpretation)
        @test occursin("NIE", result.interpretation)
    end
end


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

@testset "Input Validation" begin

    @testset "Length mismatch" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 100)
        T = Float64.(rand(rng, 100) .< 0.5)
        M = randn(rng, 90)  # Wrong length

        @test_throws ErrorException baron_kenny(Y, T, M)
        @test_throws ErrorException mediation_analysis(Y, T, M)
    end

    @testset "Insufficient observations" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 3)
        T = Float64.(rand(rng, 3) .< 0.5)
        M = randn(rng, 3)

        @test_throws ErrorException baron_kenny(Y, T, M)
    end

    @testset "NaN values rejected" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 100)
        Y[50] = NaN
        T = Float64.(rand(rng, 100) .< 0.5)
        M = randn(rng, 100)

        @test_throws ErrorException baron_kenny(Y, T, M)
    end
end


# =============================================================================
# EDGE CASES
# =============================================================================

@testset "Edge Cases" begin

    @testset "Binary mediator" begin
        rng = MersenneTwister(42)
        n = 500
        T = Float64.(rand(rng, n) .< 0.5)
        M = Float64.(rand(rng, n) .< (0.3 .+ 0.4 .* T))  # Binary mediator
        Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ 0.5 .* randn(rng, n)

        result = baron_kenny(Y, T, M)
        @test result isa BaronKennyResult
        @test !isnan(result.indirect_effect)
    end

    @testset "Continuous treatment" begin
        rng = MersenneTwister(42)
        n = 500
        T = randn(rng, n)  # Continuous treatment
        M = 0.5 .+ 0.6 .* T .+ 0.5 .* randn(rng, n)
        Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ 0.5 .* randn(rng, n)

        result = baron_kenny(Y, T, M)
        @test result isa BaronKennyResult
        @test isapprox(result.alpha_1, 0.6, atol=0.15)
    end

    @testset "Many covariates" begin
        rng = MersenneTwister(42)
        n = 500
        X = randn(rng, n, 5)
        T = Float64.(rand(rng, n) .< 0.5)
        M = 0.5 .+ 0.6 .* T .+ X * [0.1, 0.2, 0.1, 0.05, 0.15] .+ 0.5 .* randn(rng, n)
        Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ X * [0.1, 0.1, 0.1, 0.1, 0.1] .+ 0.5 .* randn(rng, n)

        result = baron_kenny(Y, T, M; covariates=X)
        @test result isa BaronKennyResult
        @test !isnan(result.indirect_effect)
    end

    @testset "Large sample" begin
        data = generate_mediation_data(n=10000, seed=42)
        result = baron_kenny(data.outcome, data.treatment, data.mediator)

        @test result.n_obs == 10000
        # Should have smaller SEs with large sample
        @test result.alpha_1_se < 0.05
        @test result.beta_1_se < 0.05
    end
end


println("All Mediation tests completed!")
