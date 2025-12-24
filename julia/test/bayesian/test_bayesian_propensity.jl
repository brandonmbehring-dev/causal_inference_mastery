#=
Unit Tests for Bayesian Propensity Score Estimation.

Session 102: Initial test suite.

Tests cover:
1. Stratified Beta-Binomial estimation
2. Logistic regression with Laplace approximation
3. Automatic method selection
4. Edge cases and error handling
=#

using Test
using Random
using Statistics
using CausalEstimators


# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate data with discrete covariates for stratified testing."""
function generate_discrete_data(; n::Int = 300, seed::Int = 42)
    Random.seed!(seed)
    X1 = rand(0:1, n)
    X2 = rand(0:2, n)
    X = Float64.(hcat(X1, X2))

    prob = 0.3 .+ 0.2 .* X1 .+ 0.1 .* X2
    treatment = Float64.(rand(n) .< prob)

    return (treatment = treatment, covariates = X)
end


"""Generate data with continuous covariates for logistic testing."""
function generate_continuous_data(; n::Int = 300, seed::Int = 123)
    Random.seed!(seed)
    X = randn(n, 2)

    logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    prob = 1 ./ (1 .+ exp.(-logit))
    treatment = Float64.(rand(n) .< prob)

    return (treatment = treatment, covariates = X, true_coef = [0.5, 0.3])
end


# =============================================================================
# Stratified Beta-Binomial Tests
# =============================================================================

@testset "Bayesian Propensity Stratified" begin
    data = generate_discrete_data()

    @testset "Returns correct structure" begin
        result = bayesian_propensity_stratified(data.treatment, data.covariates)

        @test isa(result, BayesianPropensityResult)
        @test result.method == "stratified_beta_binomial"
        @test result.strata !== nothing
        @test result.n_strata > 0
        @test result.stratum_info !== nothing
    end

    @testset "Posterior mean in valid range" begin
        result = bayesian_propensity_stratified(data.treatment, data.covariates)

        @test all(result.posterior_mean .>= 0)
        @test all(result.posterior_mean .<= 1)
    end

    @testset "Posterior SD positive" begin
        result = bayesian_propensity_stratified(data.treatment, data.covariates)

        @test all(result.posterior_sd .> 0)
    end

    @testset "Samples match distribution" begin
        result = bayesian_propensity_stratified(
            data.treatment, data.covariates;
            n_posterior_samples=5000
        )

        sample_mean = mean(result.posterior_samples[:, 1])
        sample_sd = std(result.posterior_samples[:, 1])

        @test isapprox(sample_mean, result.posterior_mean[1], rtol=0.1)
        @test isapprox(sample_sd, result.posterior_sd[1], rtol=0.1)
    end

    @testset "Stratum info complete" begin
        result = bayesian_propensity_stratified(data.treatment, data.covariates)

        for stratum in result.stratum_info
            @test stratum.stratum_id > 0
            @test stratum.n_obs > 0
            @test stratum.n_treated >= 0
            @test stratum.n_control >= 0
            @test stratum.posterior_alpha > 0
            @test stratum.posterior_beta > 0
        end
    end

    @testset "Uniform prior" begin
        result = bayesian_propensity_stratified(
            data.treatment, data.covariates;
            prior_alpha=1.0, prior_beta=1.0
        )

        @test result.prior_alpha == 1.0
        @test result.prior_beta == 1.0
    end
end


# =============================================================================
# Logistic Regression Tests
# =============================================================================

@testset "Bayesian Propensity Logistic" begin
    data = generate_continuous_data()

    @testset "Returns correct structure" begin
        result = bayesian_propensity_logistic(data.treatment, data.covariates)

        @test isa(result, BayesianPropensityResult)
        @test result.method == "logistic_laplace"
        @test result.coefficient_mean !== nothing
        @test result.coefficient_sd !== nothing
    end

    @testset "Posterior mean in valid range" begin
        result = bayesian_propensity_logistic(data.treatment, data.covariates)

        @test all(result.posterior_mean .>= 0)
        @test all(result.posterior_mean .<= 1)
    end

    @testset "Recovers coefficients" begin
        result = bayesian_propensity_logistic(data.treatment, data.covariates)

        # Coefficients should include intercept + 2 covariates
        @test length(result.coefficient_mean) == 3

        # Covariate coefficients should be close to true
        @test abs(result.coefficient_mean[2] - data.true_coef[1]) < 0.5
        @test abs(result.coefficient_mean[3] - data.true_coef[2]) < 0.5
    end

    @testset "Coefficient uncertainty" begin
        result = bayesian_propensity_logistic(data.treatment, data.covariates)

        @test all(result.coefficient_sd .> 0)
        @test all(isfinite.(result.coefficient_sd))
    end
end


# =============================================================================
# Automatic Method Selection Tests
# =============================================================================

@testset "Bayesian Propensity Auto" begin
    @testset "Selects stratified for discrete" begin
        data = generate_discrete_data()
        result = bayesian_propensity(data.treatment, data.covariates; method="auto")

        @test result.method == "stratified_beta_binomial"
    end

    @testset "Selects logistic for continuous" begin
        data = generate_continuous_data()
        result = bayesian_propensity(data.treatment, data.covariates; method="auto")

        @test result.method == "logistic_laplace"
    end

    @testset "Explicit method override" begin
        data = generate_discrete_data()
        result = bayesian_propensity(data.treatment, data.covariates; method="logistic")

        @test result.method == "logistic_laplace"
    end
end


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@testset "Bayesian Propensity Edge Cases" begin
    @testset "Length mismatch raises" begin
        treatment = [1.0, 0.0, 1.0]
        covariates = randn(2, 2)

        @test_throws ArgumentError bayesian_propensity_stratified(treatment, covariates)
    end

    @testset "Non-binary treatment raises" begin
        treatment = [0.0, 1.0, 2.0, 0.0]
        covariates = randn(4, 2)

        @test_throws ArgumentError bayesian_propensity_stratified(treatment, covariates)
    end

    @testset "Invalid prior raises" begin
        treatment = [1.0, 0.0, 1.0, 0.0]
        covariates = randn(4, 2)

        @test_throws ArgumentError bayesian_propensity_stratified(
            treatment, covariates; prior_alpha=-1.0
        )
    end

    @testset "Invalid method raises" begin
        treatment = [1.0, 0.0, 1.0, 0.0]
        covariates = randn(4, 2)

        @test_throws ArgumentError bayesian_propensity(
            treatment, covariates; method="invalid"
        )
    end

    @testset "Single covariate" begin
        Random.seed!(42)
        n = 100
        X = randn(n)
        T = Float64.(X .> 0)

        result = bayesian_propensity_logistic(T, X)

        @test result.n == n
        @test length(result.posterior_mean) == n
    end

    @testset "Many covariates" begin
        Random.seed!(42)
        n = 200
        p = 10
        X = randn(n, p)
        T = Float64.(X[:, 1] .> 0)

        result = bayesian_propensity_logistic(T, X)

        @test length(result.coefficient_mean) == p + 1  # +1 for intercept
    end
end


# =============================================================================
# Monte Carlo Validation
# =============================================================================

@testset "Bayesian Propensity Monte Carlo" begin
    @testset "Stratified coverage" begin
        Random.seed!(42)
        n_sim = 200
        n = 100
        covered = 0

        for _ in 1:n_sim
            X = Float64.(rand(0:1, n))
            true_prop = ifelse.(X .== 0, 0.3, 0.7)
            T = Float64.(rand(n) .< true_prop)

            result = bayesian_propensity_stratified(
                T, reshape(X, :, 1);
                n_posterior_samples=1000
            )

            samples = result.posterior_samples[:, 1]
            ci_lower = quantile(samples, 0.025)
            ci_upper = quantile(samples, 0.975)

            true_p = true_prop[1]

            if ci_lower <= true_p <= ci_upper
                covered += 1
            end
        end

        coverage = covered / n_sim
        @test 0.85 < coverage < 0.99
    end
end


# =============================================================================
# Numerical Stability Tests
# =============================================================================

@testset "Bayesian Propensity Stability" begin
    @testset "Extreme imbalance" begin
        Random.seed!(42)
        n = 100
        T = Float64.(rand(n) .< 0.95)
        X = randn(n, 2)

        result = bayesian_propensity_logistic(T, X)

        @test all(isfinite.(result.posterior_mean))
        @test mean(result.posterior_mean) > 0.8
    end

    @testset "Empty stratum handled" begin
        Random.seed!(42)
        n = 50
        X = zeros(n, 1)
        T = Float64.(rand(n) .< 0.5)

        result = bayesian_propensity_stratified(T, X)

        @test result.n_strata == 1
        @test all(isfinite.(result.posterior_mean))
    end
end


# =============================================================================
# Display Tests
# =============================================================================

@testset "Bayesian Propensity Display" begin
    data = generate_continuous_data(n=100, seed=70)
    result = bayesian_propensity_logistic(data.treatment, data.covariates)

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test contains(output, "BayesianPropensityResult")
    @test contains(output, "logistic_laplace")
    @test contains(output, "Coefficients")
end
