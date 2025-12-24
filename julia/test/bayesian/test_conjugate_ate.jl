#=
Unit Tests for Bayesian ATE with Conjugate Priors.

Session 101: Initial test suite.

Tests cover:
1. Basic functionality and output structure
2. Known-answer validation
3. Prior sensitivity analysis
4. Monte Carlo calibration
5. Edge cases and error handling
=#

using Test
using Random
using Statistics
using CausalEstimators


# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate simple RCT data for Bayesian testing."""
function generate_bayesian_data(;
    n::Int = 200,
    true_ate::Float64 = 2.0,
    seed::Int = 42
)
    Random.seed!(seed)
    treatment = Float64.(rand(0:1, n))
    outcomes = true_ate .* treatment .+ randn(n)
    return (outcomes = outcomes, treatment = treatment, true_ate = true_ate)
end


"""Generate data with covariates."""
function generate_bayesian_data_with_covariates(;
    n::Int = 300,
    true_ate::Float64 = 3.0,
    seed::Int = 123
)
    Random.seed!(seed)
    X = randn(n, 2)
    treatment = Float64.(X[:, 1] .+ randn(n) .> 0)
    outcomes = true_ate .* treatment .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)
    return (outcomes = outcomes, treatment = treatment, covariates = X, true_ate = true_ate)
end


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@testset "Bayesian ATE Basic" begin
    data = generate_bayesian_data()

    @testset "Returns correct structure" begin
        result = bayesian_ate(data.outcomes, data.treatment)

        # Check all fields exist
        @test isa(result, BayesianATEResult)
        @test hasfield(BayesianATEResult, :posterior_mean)
        @test hasfield(BayesianATEResult, :posterior_sd)
        @test hasfield(BayesianATEResult, :ci_lower)
        @test hasfield(BayesianATEResult, :ci_upper)
        @test hasfield(BayesianATEResult, :credible_level)
        @test hasfield(BayesianATEResult, :posterior_samples)
    end

    @testset "Posterior mean finite" begin
        result = bayesian_ate(data.outcomes, data.treatment)
        @test isfinite(result.posterior_mean)
    end

    @testset "Posterior SD positive" begin
        result = bayesian_ate(data.outcomes, data.treatment)
        @test result.posterior_sd > 0
    end

    @testset "Credible interval contains posterior mean" begin
        result = bayesian_ate(data.outcomes, data.treatment)
        @test result.ci_lower < result.posterior_mean < result.ci_upper
    end

    @testset "Posterior samples shape" begin
        n_samples = 3000
        result = bayesian_ate(data.outcomes, data.treatment; n_posterior_samples=n_samples)
        @test length(result.posterior_samples) == n_samples
    end

    @testset "Sample sizes consistent" begin
        result = bayesian_ate(data.outcomes, data.treatment)
        @test result.n == result.n_treated + result.n_control
        @test result.n == length(data.outcomes)
    end
end


# =============================================================================
# Known-Answer Tests
# =============================================================================

@testset "Bayesian ATE Known-Answer" begin
    @testset "Uninformative prior matches OLS" begin
        data = generate_bayesian_data(n=200, seed=10)
        result = bayesian_ate(
            data.outcomes, data.treatment;
            prior_mean=0.0, prior_sd=1000.0  # Very flat prior
        )

        # Posterior mean should be very close to OLS estimate
        @test isapprox(result.posterior_mean, result.ols_estimate, rtol=1e-3)
    end

    @testset "Recovers true ATE" begin
        data = generate_bayesian_data(n=200, true_ate=2.0, seed=11)
        result = bayesian_ate(data.outcomes, data.treatment)

        # Should be within 0.5 of true value with n=200
        @test abs(result.posterior_mean - data.true_ate) < 0.5
    end

    @testset "Credible interval covers true value" begin
        data = generate_bayesian_data(n=200, true_ate=2.0, seed=12)
        result = bayesian_ate(data.outcomes, data.treatment; credible_level=0.95)

        # 95% CI should contain true value
        @test result.ci_lower < data.true_ate < result.ci_upper
    end

    @testset "Zero effect detection" begin
        Random.seed!(13)
        n = 200
        treatment = Float64.(rand(0:1, n))
        outcomes = randn(n)  # No treatment effect

        result = bayesian_ate(outcomes, treatment; prior_mean=0.0)

        # Posterior mean should be near zero
        @test abs(result.posterior_mean) < 0.5
        # 95% CI should contain zero
        @test result.ci_lower < 0 < result.ci_upper
    end
end


# =============================================================================
# Prior Sensitivity Tests
# =============================================================================

@testset "Bayesian ATE Prior Sensitivity" begin
    data = generate_bayesian_data(seed=20)

    @testset "Prior shrinkage computed" begin
        result = bayesian_ate(data.outcomes, data.treatment)
        @test 0 <= result.prior_to_posterior_shrinkage <= 1
    end

    @testset "Wider prior less shrinkage" begin
        result_narrow = bayesian_ate(data.outcomes, data.treatment; prior_sd=1.0)
        result_wide = bayesian_ate(data.outcomes, data.treatment; prior_sd=100.0)

        # Wider prior should have less shrinkage
        @test result_wide.prior_to_posterior_shrinkage < result_narrow.prior_to_posterior_shrinkage
    end

    @testset "Strong prior dominates small sample" begin
        # Small sample
        Random.seed!(21)
        n = 20
        treatment = Float64.(rand(0:1, n))
        true_ate = 2.0
        outcomes = true_ate .* treatment .+ randn(n)

        prior_mean = 5.0
        prior_sd = 0.5  # Strong prior

        result = bayesian_ate(
            outcomes, treatment;
            prior_mean=prior_mean, prior_sd=prior_sd
        )

        # Posterior should be pulled toward prior (high shrinkage)
        @test result.prior_to_posterior_shrinkage > 0.1
    end

    @testset "Data dominates weak prior" begin
        # Large sample
        Random.seed!(22)
        n = 2000
        treatment = Float64.(rand(0:1, n))
        true_ate = 2.0
        outcomes = true_ate .* treatment .+ randn(n)

        result = bayesian_ate(
            outcomes, treatment;
            prior_mean=10.0,  # Very wrong prior mean
            prior_sd=10.0    # Weak prior
        )

        # Posterior should be near OLS despite wrong prior
        @test isapprox(result.posterior_mean, result.ols_estimate, rtol=0.05)
        # Shrinkage should be minimal
        @test result.prior_to_posterior_shrinkage < 0.1
    end
end


# =============================================================================
# Covariate Adjustment Tests
# =============================================================================

@testset "Bayesian ATE Covariates" begin
    data = generate_bayesian_data_with_covariates(seed=30)

    @testset "With covariates" begin
        result = bayesian_ate(
            data.outcomes, data.treatment;
            covariates=data.covariates
        )

        # Should recover true ATE
        @test abs(result.posterior_mean - data.true_ate) < 0.6
    end

    @testset "Covariates estimation valid" begin
        result = bayesian_ate(
            data.outcomes, data.treatment;
            covariates=data.covariates
        )

        # Estimates should be finite and reasonable
        @test isfinite(result.posterior_mean)
        @test result.posterior_sd > 0
    end
end


# =============================================================================
# Monte Carlo Calibration Tests
# =============================================================================

@testset "Bayesian ATE Monte Carlo" begin
    @testset "Credible interval calibration" begin
        Random.seed!(40)
        n_simulations = 500
        n_per_sim = 100
        true_ate = 2.0
        covered = 0

        for _ in 1:n_simulations
            treatment = Float64.(rand(0:1, n_per_sim))
            outcomes = true_ate .* treatment .+ randn(n_per_sim)

            result = bayesian_ate(
                outcomes, treatment;
                prior_mean=0.0, prior_sd=10.0, credible_level=0.95
            )

            if result.ci_lower < true_ate < result.ci_upper
                covered += 1
            end
        end

        coverage = covered / n_simulations

        # Should be approximately 95% (allowing for Monte Carlo error)
        @test 0.90 < coverage < 0.99
    end

    @testset "Posterior mean unbiased" begin
        Random.seed!(41)
        n_simulations = 500
        n_per_sim = 100
        true_ate = 2.0
        posterior_means = Float64[]

        for _ in 1:n_simulations
            treatment = Float64.(rand(0:1, n_per_sim))
            outcomes = true_ate .* treatment .+ randn(n_per_sim)

            result = bayesian_ate(outcomes, treatment; prior_mean=0.0, prior_sd=10.0)
            push!(posterior_means, result.posterior_mean)
        end

        mean_of_means = mean(posterior_means)
        bias = mean_of_means - true_ate

        # Bias should be small (< 0.1 with weak prior)
        @test abs(bias) < 0.1
    end

    @testset "Posterior samples match distribution" begin
        data = generate_bayesian_data(seed=42)
        result = bayesian_ate(data.outcomes, data.treatment; n_posterior_samples=10000)

        samples = result.posterior_samples
        sample_mean = mean(samples)
        sample_sd = std(samples)

        # Samples should match posterior parameters
        @test isapprox(sample_mean, result.posterior_mean, rtol=0.05)
        @test isapprox(sample_sd, result.posterior_sd, rtol=0.05)
    end
end


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@testset "Bayesian ATE Edge Cases" begin
    @testset "Length mismatch raises" begin
        outcomes = [1.0, 2.0, 3.0]
        treatment = [1.0, 0.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment)
    end

    @testset "Non-binary treatment raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [0.0, 1.0, 2.0, 0.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment)
    end

    @testset "Negative prior_sd raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [1.0, 1.0, 0.0, 0.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment; prior_sd=-1.0)
    end

    @testset "Invalid credible level raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [1.0, 1.0, 0.0, 0.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment; credible_level=1.5)
        @test_throws ArgumentError bayesian_ate(outcomes, treatment; credible_level=0.0)
    end

    @testset "All treated raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [1.0, 1.0, 1.0, 1.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment)
    end

    @testset "All control raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [0.0, 0.0, 0.0, 0.0]

        @test_throws ArgumentError bayesian_ate(outcomes, treatment)
    end

    @testset "Covariates length mismatch raises" begin
        outcomes = [1.0, 2.0, 3.0, 4.0]
        treatment = [1.0, 1.0, 0.0, 0.0]
        covariates = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # Wrong length

        @test_throws ArgumentError bayesian_ate(outcomes, treatment; covariates=covariates)
    end
end


# =============================================================================
# Comparison with Frequentist Tests
# =============================================================================

@testset "Bayesian vs Frequentist" begin
    data = generate_bayesian_data(seed=50)

    @testset "Weak prior matches frequentist" begin
        result = bayesian_ate(
            data.outcomes, data.treatment;
            prior_sd=1000.0,  # Very weak prior
            credible_level=0.95
        )

        # Compute frequentist CI for comparison
        ols_ci_lower = result.ols_estimate - 1.96 * result.ols_se
        ols_ci_upper = result.ols_estimate + 1.96 * result.ols_se

        # Bayesian CI should be very similar with flat prior
        @test isapprox(result.ci_lower, ols_ci_lower, rtol=0.05)
        @test isapprox(result.ci_upper, ols_ci_upper, rtol=0.05)
    end

    @testset "Effective sample size reasonable" begin
        result = bayesian_ate(data.outcomes, data.treatment)

        # ESS should be positive and <= n
        @test 0 < result.effective_sample_size <= result.n
    end
end


# =============================================================================
# Numerical Stability Tests
# =============================================================================

@testset "Bayesian ATE Numerical Stability" begin
    @testset "Extreme outcomes" begin
        Random.seed!(60)
        n = 100
        treatment = Float64.(rand(0:1, n))
        outcomes = 1e6 .* treatment .+ 1e6 .+ randn(n) .* 100

        result = bayesian_ate(outcomes, treatment)

        @test isfinite(result.posterior_mean)
        @test isfinite(result.posterior_sd)
        @test result.posterior_mean > 0
    end

    @testset "Very small variance" begin
        Random.seed!(61)
        n = 100
        treatment = Float64.(rand(0:1, n))
        outcomes = 2.0 .* treatment .+ randn(n) .* 0.01

        result = bayesian_ate(outcomes, treatment)

        @test isfinite(result.posterior_mean)
        @test result.posterior_sd > 0
    end

    @testset "Imbalanced treatment" begin
        Random.seed!(62)
        n = 200
        # 90% treatment, 10% control
        treatment = Float64.(rand(n) .< 0.9)
        outcomes = 2.0 .* treatment .+ randn(n)

        result = bayesian_ate(outcomes, treatment)

        @test isfinite(result.posterior_mean)
        @test result.n_treated > result.n_control
    end
end


# =============================================================================
# Display Tests
# =============================================================================

@testset "Bayesian ATE Display" begin
    data = generate_bayesian_data(n=100, seed=70)
    result = bayesian_ate(data.outcomes, data.treatment)

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test contains(output, "BayesianATEResult")
    @test contains(output, "Posterior Estimate")
    @test contains(output, "Prior Specification")
    @test contains(output, "Diagnostics")
    @test contains(output, "OLS Comparison")
end
