"""
Control Function Test Suite.

Session 95: Julia Control Function implementation tests.
- Linear CF with analytical and bootstrap inference
- Nonlinear CF (Probit/Logit) with binary outcomes
- Murphy-Topel SE correction
- Endogeneity detection via control coefficient
"""

using Test
using Random
using Statistics
using Distributions

# Include source files
include("../../src/control_function/types.jl")
include("../../src/control_function/linear.jl")
include("../../src/control_function/nonlinear.jl")


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

function generate_cf_data(;
    n::Int = 500,
    true_beta::Float64 = 2.0,
    endogeneity::Float64 = 0.5,
    instrument_strength::Float64 = 0.7,
    seed::Int = 42
)
    """
    Generate data for control function testing.

    DGP:
    - Z: Instrument (exogenous)
    - U: Unobserved confounder
    - D = π₀ + π₁Z + ν, where ν ~ N(0,1) and corr(ν, U) = endogeneity
    - Y = β₀ + β₁D + U + ε
    """
    rng = MersenneTwister(seed)

    # Instrument
    Z = randn(rng, n)

    # Correlated errors
    if endogeneity > 0
        # Generate correlated ν and U
        Σ = [1.0 endogeneity; endogeneity 1.0]
        L = cholesky(Σ).L
        raw = randn(rng, n, 2)
        errors = raw * L'
        ν = errors[:, 1]
        U = errors[:, 2]
    else
        ν = randn(rng, n)
        U = randn(rng, n)
    end

    # Treatment: D = π₀ + π₁Z + ν
    D = 1.0 .+ instrument_strength .* Z .+ ν

    # Outcome: Y = β₀ + β₁D + U + ε
    ε = 0.5 .* randn(rng, n)
    Y = 0.5 .+ true_beta .* D .+ U .+ ε

    return (
        outcome = Y,
        treatment = D,
        instrument = Z,
        true_beta = true_beta,
        true_endogeneity = endogeneity > 0
    )
end


function generate_cf_data_with_controls(;
    n::Int = 500,
    p::Int = 3,
    true_beta::Float64 = 2.0,
    seed::Int = 42
)
    """Generate CF data with exogenous controls."""
    rng = MersenneTwister(seed)

    Z = randn(rng, n)
    X = randn(rng, n, p)

    # Generate correlated errors
    ν = randn(rng, n)
    U = 0.5 .* ν .+ 0.5 .* randn(rng, n)  # Correlated

    # Treatment: D = π₀ + π₁Z + γX + ν
    D = 1.0 .+ 0.7 .* Z .+ 0.3 .* sum(X, dims=2)[:] .+ ν

    # Outcome: Y = β₀ + β₁D + δX + U + ε
    ε = 0.5 .* randn(rng, n)
    Y = 0.5 .+ true_beta .* D .+ 0.5 .* sum(X, dims=2)[:] .+ U .+ ε

    return (
        outcome = Y,
        treatment = D,
        instrument = Z,
        covariates = X,
        true_beta = true_beta
    )
end


function generate_binary_outcome_data(;
    n::Int = 1000,
    true_ame::Float64 = 0.15,
    seed::Int = 42
)
    """Generate data for nonlinear CF with binary outcome."""
    rng = MersenneTwister(seed)

    Z = randn(rng, n)

    # Correlated errors for endogeneity
    ν = randn(rng, n)
    U = 0.4 .* ν .+ 0.6 .* randn(rng, n)

    # Treatment
    D = 0.5 .+ 0.6 .* Z .+ ν

    # Latent outcome (probit model)
    latent = -0.5 .+ 0.5 .* D .+ U
    prob = cdf.(Normal(), latent)

    # Binary outcome
    Y = Float64.(rand(rng, n) .< prob)

    return (
        outcome = Y,
        treatment = D,
        instrument = Z,
        true_ame = true_ame
    )
end


function generate_exogenous_data(;
    n::Int = 500,
    true_beta::Float64 = 2.0,
    seed::Int = 42
)
    """Generate data with NO endogeneity (ρ = 0)."""
    rng = MersenneTwister(seed)

    Z = randn(rng, n)
    ν = randn(rng, n)
    U = randn(rng, n)  # Independent of ν

    D = 1.0 .+ 0.7 .* Z .+ ν
    Y = 0.5 .+ true_beta .* D .+ U .+ 0.5 .* randn(rng, n)

    return (
        outcome = Y,
        treatment = D,
        instrument = Z,
        true_beta = true_beta,
        true_endogeneity = false
    )
end


# =============================================================================
# LINEAR CONTROL FUNCTION TESTS
# =============================================================================

@testset "Linear Control Function" begin

    @testset "Returns correct type" begin
        data = generate_cf_data(n=500, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        @test result isa CFSolution
        @test !isnan(result.estimate)
        @test !isnan(result.se)
        @test result.se > 0
        @test result.ci_lower < result.ci_upper
    end

    @testset "Recovers true effect with endogeneity" begin
        data = generate_cf_data(n=1000, true_beta=2.0, endogeneity=0.5, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=200
        )

        # Within 0.5 of true value
        @test abs(result.estimate - data.true_beta) < 0.5
    end

    @testset "Detects endogeneity" begin
        data = generate_cf_data(n=1000, endogeneity=0.6, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=300
        )

        # Should detect endogeneity (control coefficient significant)
        @test result.endogeneity_detected == true
        @test result.control_p_value < 0.05
    end

    @testset "No false positive for exogenous" begin
        data = generate_exogenous_data(n=500, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=200
        )

        # Should NOT detect endogeneity with high probability
        # Allow some false positives due to randomness
        @test result.control_p_value > 0.01  # Not extremely significant
    end

    @testset "First stage diagnostics" begin
        data = generate_cf_data(n=500, instrument_strength=0.7, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        fs = result.first_stage
        @test fs.f_statistic > 10  # Strong first stage
        @test 0 < fs.r2 < 1
        @test !fs.weak_iv_warning
    end

    @testset "Weak instrument warning" begin
        data = generate_cf_data(n=500, instrument_strength=0.05, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        fs = result.first_stage
        @test fs.f_statistic < 10
        @test fs.weak_iv_warning
    end

    @testset "Works with controls" begin
        data = generate_cf_data_with_controls(n=500, p=3, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            X=data.covariates, n_bootstrap=100
        )

        @test !isnan(result.estimate)
        @test result.n_controls == 3
    end

    @testset "Analytical inference" begin
        data = generate_cf_data(n=500, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            inference=:analytical
        )

        @test result.inference == :analytical
        @test !isnan(result.se)
        @test result.se_naive > 0  # Naive SE also computed
    end

    @testset "Bootstrap inference" begin
        data = generate_cf_data(n=500, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            inference=:bootstrap, n_bootstrap=200
        )

        @test result.inference == :bootstrap
        @test result.n_bootstrap == 200
    end

    @testset "CI contains true value" begin
        data = generate_cf_data(n=1000, true_beta=2.0, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=300
        )

        # True value should be in CI
        @test result.ci_lower < data.true_beta < result.ci_upper
    end

    @testset "Problem-based interface" begin
        data = generate_cf_data(n=500, seed=42)

        # Create problem (validates inputs)
        problem = CFProblem(
            outcome = data.outcome,
            treatment = data.treatment,
            instrument = data.instrument
        )

        # Verify problem fields
        @test problem isa CFProblem
        @test length(problem.outcome) == 500
        @test problem.alpha == 0.05

        # Use convenience function (which internally creates problem + estimator)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        @test result isa CFSolution
        @test !isnan(result.estimate)
    end

    @testset "Reproducibility with same seed" begin
        data = generate_cf_data(n=300, seed=42)

        # Same seed → same results
        Random.seed!(123)
        result1 = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        Random.seed!(123)
        result2 = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=100
        )

        @test result1.estimate == result2.estimate
    end
end


# =============================================================================
# NONLINEAR CONTROL FUNCTION TESTS
# =============================================================================

@testset "Nonlinear Control Function" begin

    @testset "Probit returns correct type" begin
        data = generate_binary_outcome_data(n=500, seed=42)
        result = nonlinear_control_function(
            data.outcome, data.treatment, data.instrument;
            model_type=:probit, n_bootstrap=100
        )

        @test result isa NonlinearCFSolution
        @test !isnan(result.estimate)
        @test result.model_type == :probit
    end

    @testset "Logit returns correct type" begin
        data = generate_binary_outcome_data(n=500, seed=42)
        result = nonlinear_control_function(
            data.outcome, data.treatment, data.instrument;
            model_type=:logit, n_bootstrap=100
        )

        @test result isa NonlinearCFSolution
        @test !isnan(result.estimate)
        @test result.model_type == :logit
    end

    @testset "AME has correct sign" begin
        data = generate_binary_outcome_data(n=1000, seed=42)
        result = nonlinear_control_function(
            data.outcome, data.treatment, data.instrument;
            model_type=:probit, n_bootstrap=200
        )

        # AME should be positive (treatment increases outcome probability)
        @test result.estimate > 0
    end

    @testset "CI valid" begin
        data = generate_binary_outcome_data(n=500, seed=42)
        result = nonlinear_control_function(
            data.outcome, data.treatment, data.instrument;
            model_type=:probit, n_bootstrap=200
        )

        if result.converged
            @test result.ci_lower < result.ci_upper
            @test result.ci_lower < result.estimate < result.ci_upper
        end
    end

    @testset "Problem-based interface" begin
        data = generate_binary_outcome_data(n=500, seed=42)

        # Create problem (validates inputs)
        problem = NonlinearCFProblem(
            outcome = data.outcome,
            treatment = data.treatment,
            instrument = data.instrument,
            model_type = :probit
        )

        # Verify problem fields
        @test problem isa NonlinearCFProblem
        @test problem.model_type == :probit

        # Use convenience function (which internally creates problem + estimator)
        result = nonlinear_control_function(
            data.outcome, data.treatment, data.instrument;
            model_type=:probit, n_bootstrap=100
        )

        @test result isa NonlinearCFSolution
    end
end


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

@testset "Input Validation" begin
    rng = MersenneTwister(42)

    @testset "Rejects length mismatch" begin
        Y = randn(rng, 100)
        D = randn(rng, 90)  # Wrong length
        Z = randn(rng, 100)

        @test_throws ErrorException CFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Rejects no treatment variation" begin
        Y = randn(rng, 100)
        D = ones(100)  # Constant
        Z = randn(rng, 100)

        @test_throws ErrorException CFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Rejects small sample" begin
        Y = randn(rng, 5)
        D = randn(rng, 5)
        Z = randn(rng, 5)

        @test_throws ErrorException CFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Rejects NaN in outcome" begin
        Y = randn(rng, 100)
        Y[1] = NaN
        D = randn(rng, 100)
        Z = randn(rng, 100)

        @test_throws ErrorException CFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Nonlinear rejects non-binary outcome" begin
        Y = randn(rng, 100)  # Continuous, not binary
        D = randn(rng, 100)
        Z = randn(rng, 100)

        @test_throws ErrorException NonlinearCFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Nonlinear requires larger sample" begin
        Y = Float64.(rand(rng, 30) .< 0.5)
        D = randn(rng, 30)
        Z = randn(rng, 30)

        @test_throws ErrorException NonlinearCFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "Rejects invalid alpha" begin
        Y = randn(rng, 100)
        D = randn(rng, 100)
        Z = randn(rng, 100)

        @test_throws ErrorException CFProblem(
            outcome = Y,
            treatment = D,
            instrument = Z,
            alpha = 1.5
        )
    end
end


# =============================================================================
# EDGE CASES
# =============================================================================

@testset "Edge Cases" begin

    @testset "Multiple instruments" begin
        rng = MersenneTwister(42)
        n = 500
        Z = randn(rng, n, 2)  # Two instruments
        ν = randn(rng, n)
        D = 1.0 .+ 0.5 .* Z[:, 1] .+ 0.3 .* Z[:, 2] .+ ν
        Y = 2.0 .* D .+ 0.5 .* ν .+ randn(rng, n)

        result = control_function_ate(Y, D, Z; n_bootstrap=100)

        @test !isnan(result.estimate)
        @test result.first_stage.n_instruments == 2
    end

    @testset "Large sample precision" begin
        data = generate_cf_data(n=2000, true_beta=2.0, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            n_bootstrap=200
        )

        # Should be close to true value
        @test abs(result.estimate - data.true_beta) < 0.3
        # SE should be small
        @test result.se < 0.2
    end

    @testset "Handles vector instrument" begin
        data = generate_cf_data(n=500, seed=42)
        # Instrument as Vector (not Matrix)
        Z_vec = data.instrument
        @test Z_vec isa Vector

        result = control_function_ate(
            data.outcome, data.treatment, Z_vec;
            n_bootstrap=100
        )

        @test !isnan(result.estimate)
    end

    @testset "SE correction larger than naive" begin
        data = generate_cf_data(n=500, endogeneity=0.5, seed=42)
        result = control_function_ate(
            data.outcome, data.treatment, data.instrument;
            inference=:analytical
        )

        # Murphy-Topel corrected SE should typically be >= naive
        @test result.se >= result.se_naive * 0.9  # Allow small numerical differences
    end
end


println("All Control Function tests completed!")
