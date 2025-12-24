#=
Unit Tests for Bayesian Doubly Robust ATE Estimation.

Session 103: Initial test suite.

Tests cover:
1. Basic functionality and return structure
2. Known-answer validation
3. Monte Carlo coverage validation
4. Edge cases and error handling
=#

using Test
using Random
using Statistics
using CausalEstimators


# =============================================================================
# Test Data Generator
# =============================================================================

"""Generate data for DR testing."""
function generate_dr_data(;
    n::Int = 300,
    true_ate::Float64 = 2.0,
    confounded::Bool = true,
    seed::Int = 42,
)
    Random.seed!(seed)
    X = randn(n, 2)

    if confounded
        logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
        prob = 1 ./ (1 .+ exp.(-logit))
        treatment = Float64.(rand(n) .< prob)
    else
        treatment = Float64.(rand(n) .< 0.5)
    end

    outcomes = true_ate .* treatment .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return (outcomes=outcomes, treatment=treatment, covariates=X, true_ate=true_ate)
end


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@testset "Bayesian DR Basic" begin
    data = generate_dr_data()

    @testset "Returns correct structure" begin
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            n_posterior_samples=100,
        )

        @test isa(result, BayesianDRResult)
        @test hasfield(BayesianDRResult, :estimate)
        @test hasfield(BayesianDRResult, :se)
        @test hasfield(BayesianDRResult, :ci_lower)
        @test hasfield(BayesianDRResult, :ci_upper)
        @test hasfield(BayesianDRResult, :posterior_samples)
    end

    @testset "Posterior samples shape" begin
        n_samples = 500
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            n_posterior_samples=n_samples,
        )

        @test length(result.posterior_samples) == n_samples
    end

    @testset "Estimate in credible interval" begin
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates,
        )

        @test result.ci_lower <= result.estimate <= result.ci_upper
    end

    @testset "SE positive" begin
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates,
        )

        @test result.se > 0
    end

    @testset "Sample sizes correct" begin
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates,
        )

        @test result.n == 200
        @test result.n_treated + result.n_control == 200
    end
end


# =============================================================================
# Known-Answer Tests
# =============================================================================

@testset "Bayesian DR Known-Answer" begin
    @testset "Known effect recovered" begin
        data = generate_dr_data(n=500, true_ate=2.0, seed=42)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            n_posterior_samples=500,
        )

        @test abs(result.estimate - data.true_ate) < 1.0
    end

    @testset "No treatment effect" begin
        # Use larger sample for reliable coverage of zero
        data = generate_dr_data(n=1000, true_ate=0.0, seed=456)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates,
        )

        # CI should contain zero or estimate should be close to zero
        @test result.ci_lower <= 0 <= result.ci_upper || abs(result.estimate) < 0.3
    end

    @testset "Close to frequentist" begin
        data = generate_dr_data(n=500, seed=42)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            n_posterior_samples=1000,
        )

        diff = abs(result.estimate - result.frequentist_estimate)
        @test diff < 0.5
    end
end


# =============================================================================
# Propensity Method Tests
# =============================================================================

@testset "Bayesian DR Propensity Methods" begin
    @testset "Auto method" begin
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            propensity_method="auto",
        )

        @test isfinite(result.estimate)
    end

    @testset "Logistic method" begin
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            propensity_method="logistic",
        )

        @test isfinite(result.estimate)
    end

    @testset "Stratified method" begin
        Random.seed!(42)
        n = 200
        X = Float64.(hcat(
            rand(0:1, n),
            rand(0:2, n),
        ))
        prob = 0.3 .+ 0.2 .* X[:, 1] .+ 0.1 .* X[:, 2]
        T = Float64.(rand(n) .< prob)
        Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

        result = bayesian_dr_ate(Y, T, X; propensity_method="stratified")
        @test isfinite(result.estimate)
    end
end


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@testset "Bayesian DR Edge Cases" begin
    @testset "Length mismatch raises" begin
        @test_throws ArgumentError bayesian_dr_ate(
            [1.0, 2.0, 3.0],
            [1.0, 0.0],
            randn(3, 2),
        )
    end

    @testset "Non-binary treatment raises" begin
        @test_throws ArgumentError bayesian_dr_ate(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            randn(3, 2),
        )
    end

    @testset "Invalid trim threshold raises" begin
        data = generate_dr_data(n=100)
        @test_throws ArgumentError bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            trim_threshold=0.6,
        )
    end

    @testset "Invalid credible level raises" begin
        data = generate_dr_data(n=100)
        @test_throws ArgumentError bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            credible_level=1.5,
        )
    end

    @testset "Single covariate" begin
        Random.seed!(42)
        n = 100
        X = randn(n)
        T = Float64.(rand(n) .< 0.5)
        Y = 2.0 .* T .+ 0.5 .* X .+ randn(n)

        result = bayesian_dr_ate(Y, T, X)
        @test result.n == n
    end

    @testset "Extreme propensity" begin
        Random.seed!(42)
        n = 100
        X = randn(n, 2)
        T = Float64.(rand(n) .< 0.9)  # Very imbalanced
        Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

        result = bayesian_dr_ate(Y, T, X)
        @test isfinite(result.estimate)
        @test result.se > 0
    end
end


# =============================================================================
# Monte Carlo Validation
# =============================================================================

@testset "Bayesian DR Monte Carlo" begin
    @testset "Coverage" begin
        n_sim = 100
        n = 200
        true_ate = 2.0
        covered = 0

        for i in 1:n_sim
            data = generate_dr_data(n=n, true_ate=true_ate, seed=i)
            result = bayesian_dr_ate(
                data.outcomes, data.treatment, data.covariates;
                n_posterior_samples=500,
                credible_level=0.95,
            )

            if result.ci_lower <= true_ate <= result.ci_upper
                covered += 1
            end
        end

        coverage = covered / n_sim
        @test 0.80 < coverage < 0.99
    end

    @testset "Unbiasedness" begin
        n_sim = 100
        n = 300
        true_ate = 2.0
        estimates = Float64[]

        for i in 1:n_sim
            data = generate_dr_data(n=n, true_ate=true_ate, seed=i + 1000)
            result = bayesian_dr_ate(
                data.outcomes, data.treatment, data.covariates;
                n_posterior_samples=500,
            )
            push!(estimates, result.estimate)
        end

        bias = mean(estimates) - true_ate
        @test abs(bias) < 0.15
    end
end


# =============================================================================
# Numerical Stability Tests
# =============================================================================

@testset "Bayesian DR Stability" begin
    @testset "Large outcomes" begin
        Random.seed!(42)
        n = 100
        X = randn(n, 2)
        T = Float64.(rand(n) .< 0.5)
        Y = 1000 .+ 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

        result = bayesian_dr_ate(Y, T, X)
        @test isfinite(result.estimate)
    end

    @testset "Many covariates" begin
        Random.seed!(42)
        n = 200
        p = 10
        X = randn(n, p)
        T = Float64.(rand(n) .< 0.5)
        Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

        result = bayesian_dr_ate(Y, T, X)
        @test isfinite(result.estimate)
        @test length(result.propensity_mean) == n
    end
end


# =============================================================================
# Credible Level Tests
# =============================================================================

@testset "Bayesian DR Credible Level" begin
    @testset "90% narrower than 95%" begin
        data = generate_dr_data(n=300)

        result_90 = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            credible_level=0.90,
        )

        result_95 = bayesian_dr_ate(
            data.outcomes, data.treatment, data.covariates;
            credible_level=0.95,
        )

        width_90 = result_90.ci_upper - result_90.ci_lower
        width_95 = result_95.ci_upper - result_95.ci_lower

        @test width_90 < width_95
    end
end


# =============================================================================
# Display Test
# =============================================================================

@testset "Bayesian DR Display" begin
    data = generate_dr_data(n=100, seed=70)
    result = bayesian_dr_ate(data.outcomes, data.treatment, data.covariates)

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test contains(output, "BayesianDRResult")
    @test contains(output, "ATE:")
    @test contains(output, "CI:")
end
