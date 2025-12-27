"""
Tests for Julia Granger Causality

Session 150: Comprehensive tests for Granger causality following 3-layer architecture.
"""

using Test
using Random
using LinearAlgebra
using Statistics
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "Granger Causality Tests" begin

    # ========== Layer 1: Known Answer Tests ==========

    @testset "Known Answer - Unidirectional Causality" begin
        # X causes Y with lag 1
        Random.seed!(42)
        n = 200
        x = randn(n)
        y = zeros(n)
        for t in 2:n
            y[t] = 0.7 * x[t-1] + 0.3 * y[t-1] + randn() * 0.3
        end
        data = hcat(y, x)

        # X → Y should be significant
        result = granger_causality(data; lags=2, cause_idx=2, effect_idx=1)
        @test result.granger_causes == true
        @test result.p_value < 0.01
        @test result.f_statistic > 3.0
    end

    @testset "Known Answer - No Causality" begin
        # Independent series
        Random.seed!(42)
        n = 200
        x = randn(n)
        y = randn(n)
        data = hcat(y, x)

        result = granger_causality(data; lags=2)
        @test result.granger_causes == false
        @test result.p_value > 0.05
    end

    @testset "Known Answer - Bidirectional Causality" begin
        # Feedback system: X → Y and Y → X
        Random.seed!(42)
        n = 200
        x = zeros(n)
        y = zeros(n)
        for t in 2:n
            x[t] = 0.4 * y[t-1] + 0.3 * x[t-1] + randn() * 0.3
            y[t] = 0.4 * x[t-1] + 0.3 * y[t-1] + randn() * 0.3
        end
        data = hcat(y, x)

        result_xy, result_yx = bidirectional_granger(data; lags=2)
        @test result_xy.granger_causes == true  # X → Y
        @test result_yx.granger_causes == true  # Y → X
    end

    @testset "Known Answer - Multi-lag Detection" begin
        # X causes Y with lag 3
        Random.seed!(42)
        n = 300
        x = randn(n)
        y = zeros(n)
        for t in 4:n
            y[t] = 0.5 * x[t-3] + 0.3 * y[t-1] + randn() * 0.3
        end
        data = hcat(y, x)

        # Should detect with 3 lags
        result_3 = granger_causality(data; lags=3)
        @test result_3.granger_causes == true

        # Should not detect with 1 lag
        result_1 = granger_causality(data; lags=1)
        # Might or might not detect, but R² should be lower
        @test result_3.r2_unrestricted > result_1.r2_unrestricted
    end

    @testset "Known Answer - Causality Matrix" begin
        # Build 3-var system: X₁ → X₂ → X₃
        Random.seed!(42)
        n = 200
        x1 = randn(n)
        x2 = zeros(n)
        x3 = zeros(n)
        for t in 2:n
            x2[t] = 0.6 * x1[t-1] + 0.2 * x2[t-1] + randn() * 0.3
            x3[t] = 0.6 * x2[t-1] + 0.2 * x3[t-1] + randn() * 0.3
        end
        data = hcat(x1, x2, x3)

        results, matrix = granger_causality_matrix(
            data;
            lags=2,
            var_names=["X1", "X2", "X3"]
        )

        # X1 → X2 should be detected
        @test results[("X1", "X2")].granger_causes == true
        # X2 → X3 should be detected
        @test results[("X2", "X3")].granger_causes == true
        # X3 → X1 should NOT be detected (no reverse causation)
        @test results[("X3", "X1")].granger_causes == false
    end

    @testset "Known Answer - R-squared Ordering" begin
        # Unrestricted model should have higher R²
        Random.seed!(42)
        n = 200
        x = randn(n)
        y = zeros(n)
        for t in 2:n
            y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + randn() * 0.5
        end
        data = hcat(y, x)

        result = granger_causality(data; lags=2)
        @test result.r2_unrestricted >= result.r2_restricted
        @test result.rss_unrestricted <= result.rss_restricted
    end

    @testset "Known Answer - AIC Ordering" begin
        # With true causality, unrestricted AIC should be lower
        Random.seed!(42)
        n = 200
        x = randn(n)
        y = zeros(n)
        for t in 2:n
            y[t] = 0.7 * x[t-1] + 0.2 * y[t-1] + randn() * 0.3
        end
        data = hcat(y, x)

        result = granger_causality(data; lags=2)
        @test result.granger_causes == true
        @test result.aic_unrestricted < result.aic_restricted
    end

    @testset "Known Answer - Degrees of Freedom" begin
        Random.seed!(42)
        n = 200
        lags = 3
        data = randn(n, 2)

        result = granger_causality(data; lags=lags)

        @test result.lags == lags
        @test result.df_num == lags  # Numerator df = number of restricted coefficients
        @test result.df_denom > 0
    end

    # ========== Layer 2: Adversarial Tests ==========

    @testset "Adversarial - Short Series" begin
        Random.seed!(42)
        # Minimum viable length
        data = randn(10, 2)
        result = granger_causality(data; lags=2)
        @test result isa GrangerResult
    end

    @testset "Adversarial - Too Short Raises" begin
        Random.seed!(42)
        data = randn(5, 2)
        @test_throws ErrorException granger_causality(data; lags=2)
    end

    @testset "Adversarial - Invalid Lags" begin
        Random.seed!(42)
        data = randn(100, 2)
        @test_throws ErrorException granger_causality(data; lags=0)
        @test_throws ErrorException granger_causality(data; lags=-1)
    end

    @testset "Adversarial - Invalid Index" begin
        Random.seed!(42)
        data = randn(100, 2)
        @test_throws ErrorException granger_causality(data; cause_idx=3)
        @test_throws ErrorException granger_causality(data; effect_idx=5)
    end

    @testset "Adversarial - Single Variable" begin
        Random.seed!(42)
        data = randn(100, 1)
        @test_throws ErrorException granger_causality(data)
    end

    @testset "Adversarial - Constant Series" begin
        Random.seed!(42)
        n = 100
        data = ones(n, 2)
        data[:, 1] .+= randn(n) * 0.01  # Small noise to avoid singularity
        # Should run without error
        result = granger_causality(data; lags=2)
        @test isfinite(result.f_statistic)
    end

    # ========== Layer 3: Monte Carlo Tests ==========

    @testset "Monte Carlo - Type I Error Control" begin
        n_runs = 200
        alpha = 0.05
        rejections = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            # H0: No causality (independent series)
            x = randn(150)
            y = zeros(150)
            for t in 2:150
                y[t] = 0.5 * y[t-1] + randn() * 0.5  # AR(1), no X
            end
            data = hcat(y, x)

            result = granger_causality(data; lags=2, alpha=alpha)
            if result.granger_causes
                rejections += 1
            end
        end

        type1_rate = rejections / n_runs
        # Should be around alpha (allow 2-10%)
        @test 0.02 < type1_rate < 0.12
    end

    @testset "Monte Carlo - Power Against Alternatives" begin
        n_runs = 100
        rejections = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            # H1: True causality
            x = randn(150)
            y = zeros(150)
            for t in 2:150
                y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + randn() * 0.5
            end
            data = hcat(y, x)

            result = granger_causality(data; lags=2, alpha=0.05)
            if result.granger_causes
                rejections += 1
            end
        end

        power = rejections / n_runs
        # Should have high power (>80%)
        @test power > 0.80
    end

    @testset "Monte Carlo - Lag Selection Accuracy" begin
        # True lag is 2, check that using lag 2 gives better detection
        n_runs = 50
        better_with_true_lag = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            x = randn(200)
            y = zeros(200)
            for t in 3:200
                y[t] = 0.5 * x[t-2] + 0.3 * y[t-1] + randn() * 0.5
            end
            data = hcat(y, x)

            result_1 = granger_causality(data; lags=1)
            result_2 = granger_causality(data; lags=2)

            # Model with true lag (2) should have lower p-value
            if result_2.p_value < result_1.p_value
                better_with_true_lag += 1
            end
        end

        accuracy = better_with_true_lag / n_runs
        # Should mostly identify correct lag
        @test accuracy > 0.70
    end

    @testset "Monte Carlo - Bidirectional Detection" begin
        n_runs = 50
        both_detected = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            x = zeros(150)
            y = zeros(150)
            for t in 2:150
                x[t] = 0.4 * y[t-1] + 0.3 * x[t-1] + randn() * 0.3
                y[t] = 0.4 * x[t-1] + 0.3 * y[t-1] + randn() * 0.3
            end
            data = hcat(y, x)

            result_xy, result_yx = bidirectional_granger(data; lags=2)
            if result_xy.granger_causes && result_yx.granger_causes
                both_detected += 1
            end
        end

        detection_rate = both_detected / n_runs
        # Should detect bidirectional causality in most cases
        @test detection_rate > 0.70
    end

    @testset "Monte Carlo - Chain Detection" begin
        # X₁ → X₂ → X₃ chain
        n_runs = 50
        correct_detections = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200
            x1 = randn(n)
            x2 = zeros(n)
            x3 = zeros(n)
            for t in 2:n
                x2[t] = 0.5 * x1[t-1] + 0.3 * x2[t-1] + randn() * 0.3
                x3[t] = 0.5 * x2[t-1] + 0.3 * x3[t-1] + randn() * 0.3
            end
            data = hcat(x1, x2, x3)

            results, _ = granger_causality_matrix(data; lags=2)

            # Should detect X1 → X2 and X2 → X3
            if results[("var_1", "var_2")].granger_causes &&
               results[("var_2", "var_3")].granger_causes &&
               !results[("var_3", "var_1")].granger_causes
                correct_detections += 1
            end
        end

        accuracy = correct_detections / n_runs
        @test accuracy > 0.60
    end

    @testset "Monte Carlo - F-statistic Consistency" begin
        # F-statistic should be positive and consistent
        n_runs = 100

        for seed in 1:n_runs
            Random.seed!(seed)
            data = randn(100, 2)
            result = granger_causality(data; lags=2)

            @test result.f_statistic >= 0
            @test isfinite(result.f_statistic)
            @test 0.0 <= result.p_value <= 1.0
        end
    end

end
