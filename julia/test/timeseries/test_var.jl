"""
Tests for Julia VAR Estimation

Session 150: Comprehensive tests for VAR following 3-layer architecture.
"""

using Test
using Random
using LinearAlgebra
using Statistics
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "VAR Estimation Tests" begin

    # ========== Layer 1: Known Answer Tests ==========

    @testset "Known Answer - VAR(1) Coefficient Recovery" begin
        # Generate VAR(1) with known coefficients
        Random.seed!(42)
        n = 500
        true_A = [0.5 0.2; 0.1 0.4]

        data = zeros(n, 2)
        for t in 2:n
            data[t, :] = true_A * data[t-1, :] + randn(2) * 0.3
        end

        result = var_estimate(data; lags=1)

        # Extract lag 1 coefficients (after intercept)
        est_A = result.coefficients[:, 2:3]

        # Should be close to true coefficients
        @test norm(est_A - true_A) < 0.2
    end

    @testset "Known Answer - VAR(2) Structure" begin
        Random.seed!(42)
        n = 300
        k = 2
        lags = 2
        data = randn(n, k)

        result = var_estimate(data; lags=lags)

        # Coefficient matrix: (k x (1 + k*p))
        # = (2 x (1 + 2*2)) = (2 x 5)
        @test size(result.coefficients) == (k, 1 + k * lags)
        @test result.lags == lags
        @test result.n_obs == n
        @test result.n_obs_effective == n - lags
    end

    @testset "Known Answer - Intercept Estimation" begin
        Random.seed!(42)
        n = 300
        true_intercept = [1.0, -0.5]
        true_A = [0.3 0.0; 0.0 0.3]

        data = zeros(n, 2)
        for t in 2:n
            data[t, :] = true_intercept + true_A * data[t-1, :] + randn(2) * 0.3
        end

        result = var_estimate(data; lags=1, include_constant=true)

        # First column is intercept
        est_intercept = result.coefficients[:, 1]
        @test abs(est_intercept[1] - true_intercept[1]) < 0.3
        @test abs(est_intercept[2] - true_intercept[2]) < 0.3
    end

    @testset "Known Answer - Residual Shape" begin
        Random.seed!(42)
        n = 200
        lags = 2
        k = 3
        data = randn(n, k)

        result = var_estimate(data; lags=lags)

        @test size(result.residuals) == (n - lags, k)
        @test result.n_obs_effective == n - lags
    end

    @testset "Known Answer - Sigma Symmetric" begin
        Random.seed!(42)
        data = randn(200, 2)
        result = var_estimate(data; lags=2)

        @test result.sigma ≈ result.sigma' rtol=1e-10
        # Should be positive semi-definite
        eigenvalues = eigvals(result.sigma)
        @test all(eigenvalues .>= -1e-10)
    end

    @testset "Known Answer - Forecast Shape" begin
        Random.seed!(42)
        data = randn(200, 2)
        result = var_estimate(data; lags=2)

        forecasts = var_forecast(result, data; steps=10)
        @test size(forecasts) == (10, 2)
    end

    @testset "Known Answer - Forecast Continuity" begin
        Random.seed!(42)
        n = 200
        A = [0.3 0.1; 0.1 0.3]
        data = zeros(n, 2)
        for t in 2:n
            data[t, :] = A * data[t-1, :] + randn(2) * 0.3
        end

        result = var_estimate(data; lags=1)
        forecasts = var_forecast(result, data; steps=5)

        # First forecast should be close to last observation
        @test all(abs.(forecasts[1, :] - data[end, :]) .< 5.0)
    end

    @testset "Known Answer - Information Criteria" begin
        Random.seed!(42)
        data = randn(200, 2)
        result = var_estimate(data; lags=2)

        @test isfinite(result.aic)
        @test isfinite(result.bic)
        @test isfinite(result.hqc)
        @test isfinite(result.log_likelihood)

        # BIC penalizes more than AIC for n > 7
        @test result.bic > result.aic
    end

    # ========== Layer 2: Adversarial Tests ==========

    @testset "Adversarial - Minimum Observations" begin
        Random.seed!(42)
        lags = 2
        k = 2
        n = lags + 2  # Minimum viable

        data = randn(n, k)
        result = var_estimate(data; lags=lags)
        @test result isa VARResult
    end

    @testset "Adversarial - Insufficient Observations" begin
        Random.seed!(42)
        data = randn(3, 2)
        @test_throws ErrorException var_estimate(data; lags=3)
    end

    @testset "Adversarial - Invalid Lags" begin
        Random.seed!(42)
        data = randn(100, 2)
        @test_throws ErrorException var_estimate(data; lags=0)
        @test_throws ErrorException var_estimate(data; lags=-1)
    end

    @testset "Adversarial - No Constant" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = var_estimate(data; lags=2, include_constant=false)

        # Without constant: (k x k*p) = (2 x 4)
        @test size(result.coefficients) == (2, 4)
    end

    @testset "Adversarial - Variable Names" begin
        Random.seed!(42)
        data = randn(100, 2)
        result = var_estimate(data; lags=1, var_names=["GDP", "Inflation"])

        @test result.var_names == ["GDP", "Inflation"]
    end

    @testset "Adversarial - Variable Name Mismatch" begin
        Random.seed!(42)
        data = randn(100, 2)
        @test_throws ErrorException var_estimate(
            data;
            lags=1,
            var_names=["A", "B", "C"]
        )
    end

    # ========== Layer 3: Monte Carlo Tests ==========

    @testset "Monte Carlo - Coefficient Consistency" begin
        # Coefficients should be consistent (converge to true as n → ∞)
        true_A = [0.4 0.1; 0.1 0.3]
        sample_sizes = [100, 200, 500]
        errors = Float64[]

        for n in sample_sizes
            Random.seed!(42)
            data = zeros(n, 2)
            for t in 2:n
                data[t, :] = true_A * data[t-1, :] + randn(2) * 0.3
            end

            result = var_estimate(data; lags=1)
            est_A = result.coefficients[:, 2:3]
            push!(errors, norm(est_A - true_A))
        end

        # Errors should generally decrease with sample size
        @test errors[3] < errors[1]
    end

    @testset "Monte Carlo - Forecast MSE Bounds" begin
        n_runs = 50
        mse_sum = 0.0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200
            true_A = [0.3 0.1; 0.1 0.3]

            data = zeros(n, 2)
            for t in 2:n
                data[t, :] = true_A * data[t-1, :] + randn(2) * 0.5
            end

            # Split into train/test
            train = data[1:180, :]
            test = data[181:200, :]

            result = var_estimate(train; lags=1)
            forecasts = var_forecast(result, train; steps=20)

            mse = mean((forecasts - test).^2)
            mse_sum += mse
        end

        avg_mse = mse_sum / n_runs
        # MSE should be reasonable
        @test avg_mse < 5.0
    end

    @testset "Monte Carlo - Information Criteria Selection" begin
        # True VAR(2), check that IC sometimes prefers lag 2
        # Note: BIC can overpenalize in small samples, so use AIC
        n_runs = 30
        correct_selection = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 500  # Larger sample for more reliable selection
            A1 = [0.4 0.0; 0.0 0.4]  # Stronger lag 1 effect
            A2 = [0.3 0.0; 0.0 0.3]  # Stronger lag 2 effect

            data = zeros(n, 2)
            for t in 3:n
                data[t, :] = A1 * data[t-1, :] + A2 * data[t-2, :] + randn(2) * 0.3
            end

            result_1 = var_estimate(data; lags=1)
            result_2 = var_estimate(data; lags=2)

            # AIC should prefer lag 2 (true model) more often than BIC
            if result_2.aic < result_1.aic
                correct_selection += 1
            end
        end

        accuracy = correct_selection / n_runs
        # AIC with larger sample should prefer true model most of the time
        @test accuracy > 0.50
    end

    @testset "Monte Carlo - Residual Properties" begin
        # Residuals should be approximately white noise
        n_runs = 50

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200
            true_A = [0.3 0.1; 0.1 0.3]

            data = zeros(n, 2)
            for t in 2:n
                data[t, :] = true_A * data[t-1, :] + randn(2) * 0.5
            end

            result = var_estimate(data; lags=1)

            # Residuals should have approximately zero mean
            resid_mean = mean(result.residuals, dims=1)
            @test all(abs.(resid_mean) .< 0.2)

            # Residuals should have reasonable variance
            resid_std = std(result.residuals, dims=1)
            @test all(resid_std .> 0.1)
            @test all(resid_std .< 2.0)
        end
    end

    @testset "Monte Carlo - Stability Check" begin
        # VAR should detect stable vs unstable systems
        n_runs = 50

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200
            # Stable VAR (eigenvalues < 1)
            A = [0.3 0.1; 0.1 0.3]

            data = zeros(n, 2)
            for t in 2:n
                data[t, :] = A * data[t-1, :] + randn(2) * 0.5
            end

            result = var_estimate(data; lags=1)

            # Check that estimation succeeded
            @test isfinite(result.aic)
            @test size(result.residuals, 1) == n - 1
        end
    end

    @testset "Monte Carlo - Trivariate VAR" begin
        Random.seed!(42)
        n = 200
        k = 3
        A = [0.3 0.1 0.0; 0.0 0.3 0.1; 0.1 0.0 0.3]

        data = zeros(n, k)
        for t in 2:n
            data[t, :] = A * data[t-1, :] + randn(k) * 0.3
        end

        result = var_estimate(data; lags=1)

        @test size(result.coefficients) == (3, 1 + 3)  # (k, 1+k*p)
        @test size(result.residuals) == (n - 1, 3)
        @test size(result.sigma) == (3, 3)
    end

end
