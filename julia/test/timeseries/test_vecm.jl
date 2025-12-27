"""
Tests for Julia VECM (Vector Error Correction Model)

Session 149: Tests for Julia VECM parity with Python.
"""

using Test
using Random
using LinearAlgebra
using Statistics
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "VECM Tests" begin

    @testset "VECM Basic Structure" begin
        Random.seed!(42)

        # Create cointegrated system
        n = 200
        trend = cumsum(randn(n))
        y1 = trend + randn(n) * 0.5
        y2 = 0.5 * trend + randn(n) * 0.5
        data = hcat(y1, y2)

        result = vecm_estimate(data; coint_rank=1, lags=2)

        @test result isa VECMResult
        @test result.coint_rank == 1
        @test result.lags == 2
        @test result.n_vars == 2
        @test result.n_obs > 0
    end

    @testset "VECM Alpha Beta Shapes" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)

        k = 2  # Number of variables
        r = 1  # Cointegration rank

        @test size(result.alpha) == (k, r)
        @test size(result.beta) == (k, r)
        @test size(result.pi) == (k, k)
    end

    @testset "VECM Gamma Shape" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=3)

        k = 2
        n_sr_lags = 2  # lags - 1

        @test size(result.gamma) == (k, k * n_sr_lags)
    end

    @testset "VECM Pi Equals Alpha Beta'" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)

        expected_pi = result.alpha * result.beta'
        @test result.pi ≈ expected_pi rtol=1e-10
    end

    @testset "VECM Residuals Shape" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)

        @test size(result.residuals, 1) == result.n_obs
        @test size(result.residuals, 2) == result.n_vars
    end

    @testset "VECM Sigma Symmetric" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)

        # Check symmetry
        @test result.sigma ≈ result.sigma' rtol=1e-10

        # Check positive semi-definite
        eigenvalues = eigvals(result.sigma)
        @test all(eigenvalues .>= -1e-10)
    end

    @testset "VECM Information Criteria" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)

        @test isfinite(result.aic)
        @test isfinite(result.bic)
        @test isfinite(result.log_likelihood)
    end

    @testset "VECM Constant Included" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result0 = vecm_estimate(data; coint_rank=1, lags=2, det_order=0)
        result1 = vecm_estimate(data; coint_rank=1, lags=2, det_order=1)
        result_m1 = vecm_estimate(data; coint_rank=1, lags=2, det_order=-1)

        @test result0.const_term !== nothing
        @test result1.const_term !== nothing
        @test result_m1.const_term === nothing
    end

    @testset "VECM Trivariate" begin
        Random.seed!(42)

        n = 200
        trend1 = cumsum(randn(n))
        trend2 = cumsum(randn(n))
        data = hcat(
            trend1 + randn(n) * 0.3,
            0.5 * trend1 + 0.5 * trend2 + randn(n) * 0.3,
            trend2 + randn(n) * 0.3
        )

        result = vecm_estimate(data; coint_rank=2, lags=2)

        @test result.n_vars == 3
        @test result.coint_rank == 2
        @test size(result.alpha) == (3, 2)
        @test size(result.beta) == (3, 2)
    end

    @testset "VECM Method OLS" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2, method=:ols)

        @test result isa VECMResult
        @test result.coint_rank == 1
        @test isfinite(result.aic)
    end

    @testset "VECM Forecast" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)
        forecasts = vecm_forecast(result, data; horizons=10)

        @test size(forecasts) == (10, 2)
    end

    @testset "VECM Forecast Continuity" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)
        forecasts = vecm_forecast(result, data; horizons=5)

        # First forecast shouldn't be too far from last observation
        diff = abs.(forecasts[1, :] .- data[end, :])
        @test all(diff .< 10)
    end

    @testset "VECM Error Correction Term" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        result = vecm_estimate(data; coint_rank=1, lags=2)
        ect = compute_error_correction_term(data, result.beta)

        @test size(ect) == (200, 1)
    end

    @testset "VECM Input Validation - Invalid coint_rank" begin
        Random.seed!(42)

        data = randn(100, 2)

        @test_throws ErrorException vecm_estimate(data; coint_rank=0, lags=2)
        @test_throws ErrorException vecm_estimate(data; coint_rank=2, lags=2)
    end

    @testset "VECM Input Validation - Invalid lags" begin
        Random.seed!(42)

        data = randn(100, 2)

        @test_throws ErrorException vecm_estimate(data; coint_rank=1, lags=0)
    end

    @testset "VECM Input Validation - Invalid det_order" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        @test_throws ErrorException vecm_estimate(data; coint_rank=1, lags=2, det_order=5)
    end

    @testset "VECM Input Validation - Insufficient Observations" begin
        Random.seed!(42)

        data = randn(20, 2)

        @test_throws ErrorException vecm_estimate(data; coint_rank=1, lags=5)
    end

    @testset "VECM Different Seeds" begin
        data1 = let
            Random.seed!(42)
            n = 200
            trend = cumsum(randn(n))
            hcat(trend + randn(n) * 0.5, 0.5 * trend + randn(n) * 0.5)
        end

        data2 = let
            Random.seed!(123)
            n = 200
            trend = cumsum(randn(n))
            hcat(trend + randn(n) * 0.5, 0.5 * trend + randn(n) * 0.5)
        end

        result1 = vecm_estimate(data1; coint_rank=1, lags=2)
        result2 = vecm_estimate(data2; coint_rank=1, lags=2)

        @test !(result1.alpha ≈ result2.alpha)
    end

    @testset "VECM Johansen Consistency" begin
        Random.seed!(42)

        n = 200
        trend = cumsum(randn(n))
        data = hcat(
            trend + randn(n) * 0.5,
            0.5 * trend + randn(n) * 0.5
        )

        johansen_result = johansen_test(data; lags=2)
        vecm_result = vecm_estimate(data; coint_rank=1, lags=2)

        # β vectors should be proportional
        johansen_beta = johansen_result.eigenvectors[:, 1]
        vecm_beta = vecm_result.beta[:, 1]

        corr = abs(cor(johansen_beta, vecm_beta))
        @test corr > 0.99
    end

end
