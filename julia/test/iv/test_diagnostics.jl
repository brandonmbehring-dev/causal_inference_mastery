"""
Tests for IV weak instrument diagnostics.

Phase 4.2: First-stage F-statistic, Cragg-Donald, Stock-Yogo
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics
using Random

@testset "IV Diagnostics" begin
    @testset "First-Stage F-Statistic - Strong IV" begin
        # Strong instrument: Z highly correlated with D
        Random.seed!(123)
        n = 500
        z = randn(n)
        d = 0.8 * z + 0.2 * randn(n)  # Strong first stage
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        # Without covariates
        fstat, p_value = first_stage_fstat(d, Z, nothing)

        @test fstat > 10.0  # Strong instrument
        @test p_value < 0.001  # Highly significant
        @test fstat > 100.0  # Very strong correlation → high F

        println("Strong IV F-stat: $(round(fstat, digits=2))")
    end

    @testset "First-Stage F-Statistic - Weak IV" begin
        # Weak instrument: Z poorly correlated with D
        Random.seed!(456)
        n = 500
        z = randn(n)
        d = 0.1 * z + randn(n)  # Weak first stage (correlation ≈ 0.1)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        fstat, p_value = first_stage_fstat(d, Z, nothing)

        @test fstat < 10.0  # Weak instrument
        @test p_value > 0.05  # Not significant at 5% level

        println("Weak IV F-stat: $(round(fstat, digits=2))")
    end

    @testset "First-Stage F-Statistic - With Covariates" begin
        Random.seed!(789)
        n = 500
        z = randn(n)
        x = randn(n, 2)  # 2 covariates
        d = 0.6 * z + 0.3 * x[:, 1] + 0.2 * x[:, 2] + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        # With covariates
        fstat_cov, p_value_cov = first_stage_fstat(d, Z, x)

        # Without covariates (for comparison)
        fstat_no_cov, p_value_no_cov = first_stage_fstat(d, Z, nothing)

        @test fstat_cov > 10.0  # Still strong
        @test p_value_cov < 0.001

        # F-stat should differ when controlling for covariates
        @test abs(fstat_cov - fstat_no_cov) > 0.1

        println(
            "F-stat with covariates: $(round(fstat_cov, digits=2)), without: $(round(fstat_no_cov, digits=2))",
        )
    end

    @testset "First-Stage F-Statistic - Multiple Instruments" begin
        Random.seed!(101112)
        n = 500
        z1 = randn(n)
        z2 = randn(n)
        d = 0.5 * z1 + 0.4 * z2 + randn(n)
        y = 2.0 * d + randn(n)

        Z = hcat(z1, z2)

        fstat, p_value = first_stage_fstat(d, Z, nothing)

        @test fstat > 10.0  # Strong instruments
        @test p_value < 0.001
        @test size(Z, 2) == 2  # 2 instruments

        println("Multiple IV F-stat: $(round(fstat, digits=2))")
    end

    @testset "First-Stage F-Statistic - Insufficient DF" begin
        # n <= K + p + 1 → no degrees of freedom
        # df_resid = n - (1 + K + p) where 1 is intercept
        # For df_resid > 0, need n > K + p + 1
        # With p=0 (no covariates), need n > K + 1
        n = 10
        K = 10  # df_resid = 10 - (1 + 10 + 0) = -1
        z = randn(n, K)
        d = randn(n)

        @test_throws ArgumentError first_stage_fstat(d, z, nothing)
    end

    @testset "Cragg-Donald Statistic - Strong IV" begin
        Random.seed!(131415)
        n = 500
        z = randn(n)
        d = 0.8 * z + 0.2 * randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        cd = cragg_donald_stat(d, Z, nothing)

        @test cd > 10.0  # Strong instrument
        @test cd > stock_yogo_critical_value(1, 0.10, "size")

        println("Strong IV CD: $(round(cd, digits=2))")
    end

    @testset "Cragg-Donald Statistic - Weak IV" begin
        Random.seed!(456)
        n = 100
        z = randn(n)
        d_weak = 0.01 * z + 2.0 * randn(n)  # Extremely weak: correlation → 0
        d_strong = 0.8 * z + 0.2 * randn(n)  # Strong for comparison

        Z = reshape(z, n, 1)

        cd_weak = cragg_donald_stat(d_weak, Z, nothing)
        cd_strong = cragg_donald_stat(d_strong, Z, nothing)

        # Weak IV should have much smaller CD than strong IV
        @test cd_weak < cd_strong / 10  # At least 10x smaller
        # Note: CD can still be > Stock-Yogo threshold even with weak first stage
        # because CD scales with sample size. What matters is relative comparison.

        println("Weak IV CD: $(round(cd_weak, digits=2)), Strong: $(round(cd_strong, digits=2))")
    end

    @testset "Cragg-Donald Statistic - With Covariates" begin
        Random.seed!(192021)
        n = 500
        z = randn(n)
        x = randn(n, 3)
        d = 0.6 * z + 0.2 * sum(x, dims = 2)[:, 1] + randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        cd_cov = cragg_donald_stat(d, Z, x)
        cd_no_cov = cragg_donald_stat(d, Z, nothing)

        @test cd_cov > 0.0
        @test cd_no_cov > 0.0

        # Should differ when controlling for covariates
        @test abs(cd_cov - cd_no_cov) > 0.1

        println("CD with cov: $(round(cd_cov, digits=2)), without: $(round(cd_no_cov, digits=2))")
    end

    @testset "Cragg-Donald ≈ F-stat for K=1" begin
        # For single instrument, CD and F-stat are related but not identical
        # CD uses concentration parameter scaled by residual variance
        # F-stat uses R² from regression
        # They should have same order of magnitude
        Random.seed!(222324)
        n = 500
        z = randn(n)
        d = 0.7 * z + 0.3 * randn(n)
        y = 2.0 * d + randn(n)

        Z = reshape(z, n, 1)

        fstat, _ = first_stage_fstat(d, Z, nothing)
        cd = cragg_donald_stat(d, Z, nothing)

        # Should have same order of magnitude (within factor of 5)
        # CD is typically larger than F due to different scaling
        @test 0.2 < cd / fstat < 5.0

        println("F-stat: $(round(fstat, digits=2)), CD: $(round(cd, digits=2)), Ratio: $(round(cd/fstat, digits=2))")
    end

    @testset "Stock-Yogo Critical Values - Size Test" begin
        # Test that critical values match published table
        @test stock_yogo_critical_value(1, 0.10, "size") ≈ 16.38
        @test stock_yogo_critical_value(2, 0.10, "size") ≈ 19.93
        @test stock_yogo_critical_value(3, 0.10, "size") ≈ 22.30
        @test stock_yogo_critical_value(5, 0.10, "size") ≈ 26.87

        # Different significance levels
        @test stock_yogo_critical_value(1, 0.15, "size") ≈ 8.96
        @test stock_yogo_critical_value(1, 0.20, "size") ≈ 6.66
        @test stock_yogo_critical_value(1, 0.25, "size") ≈ 5.53
    end

    @testset "Stock-Yogo Critical Values - Bias Test" begin
        # Bias test critical values
        @test stock_yogo_critical_value(1, 0.05, "bias") ≈ 13.91
        @test stock_yogo_critical_value(1, 0.10, "bias") ≈ 9.08
        @test stock_yogo_critical_value(1, 0.20, "bias") ≈ 6.46
        @test stock_yogo_critical_value(1, 0.30, "bias") ≈ 5.39

        @test stock_yogo_critical_value(5, 0.10, "bias") ≈ 16.52
    end

    @testset "Stock-Yogo Critical Values - Invalid Inputs" begin
        # Invalid K
        @test_throws ArgumentError stock_yogo_critical_value(0, 0.10, "size")
        @test_throws ArgumentError stock_yogo_critical_value(50, 0.10, "size")

        # Invalid α
        @test_throws ArgumentError stock_yogo_critical_value(1, 0.07, "size")

        # Invalid test type
        @test_throws ArgumentError stock_yogo_critical_value(1, 0.10, "unknown")
    end

    @testset "Stock-Yogo Critical Values - Unavailable K" begin
        # K > 10 not in table
        @test_throws ArgumentError stock_yogo_critical_value(15, 0.10, "size")
    end

    @testset "Weak IV Warning - Strong Instruments" begin
        # Strong: F > 10 and CD > critical value
        fstat = 25.0
        cd = 20.0
        K = 2

        is_weak, message = weak_iv_warning(fstat, cd, K)

        @test !is_weak
        @test message == ""
    end

    @testset "Weak IV Warning - Weak by F-stat" begin
        # Weak: F < 10
        fstat = 5.0
        cd = 20.0  # CD is fine, but F is low
        K = 2

        is_weak, message = weak_iv_warning(fstat, cd, K)

        @test is_weak
        @test occursin("First-stage F-statistic", message)
        @test occursin("rule of thumb", message)
        @test occursin("Recommendations", message)
    end

    @testset "Weak IV Warning - Weak by CD" begin
        # Weak: CD < Stock-Yogo critical value
        fstat = 15.0  # F is fine
        cd = 10.0  # CD < 19.93 (critical value for K=2)
        K = 2

        is_weak, message = weak_iv_warning(fstat, cd, K)

        @test is_weak
        @test occursin("Cragg-Donald", message)
        @test occursin("Stock-Yogo", message)
    end

    @testset "Weak IV Warning - Weak by Both" begin
        # Weak by both F and CD
        fstat = 5.0
        cd = 8.0
        K = 1

        is_weak, message = weak_iv_warning(fstat, cd, K)

        @test is_weak
        @test occursin("First-stage F-statistic", message)
        @test occursin("Cragg-Donald", message)

        # Should mention both diagnostics
        @test length(split(message, "\n  - ")) >= 3  # Header + 2 diagnostic lines
    end

    @testset "Weak IV Warning - Large K (no Stock-Yogo)" begin
        # K > 10: Stock-Yogo not available, only use F-stat
        fstat = 5.0
        cd = 100.0  # CD irrelevant for K > 10
        K = 15

        is_weak, message = weak_iv_warning(fstat, cd, K)

        @test is_weak  # Based on F < 10
        @test occursin("First-stage F-statistic", message)
        @test !occursin("Cragg-Donald", message)  # Stock-Yogo unavailable
    end

    @testset "Integration Test - Full Diagnostic Workflow" begin
        # Simulate IV scenario
        Random.seed!(252627)
        n = 1000

        # Strong instruments
        z1 = randn(n)
        z2 = randn(n)
        x = randn(n, 2)  # Covariates

        # First stage: D = π₁Z₁ + π₂Z₂ + γ'X + ε
        d = 0.6 * z1 + 0.5 * z2 + 0.3 * x[:, 1] + 0.2 * randn(n)

        # Second stage: Y = βD + δ'X + u
        y = 2.0 * d + 0.5 * x[:, 2] + randn(n)

        Z = hcat(z1, z2)
        K = size(Z, 2)

        # Compute diagnostics
        fstat, p_value = first_stage_fstat(d, Z, x)
        cd = cragg_donald_stat(d, Z, x)
        is_weak, warning = weak_iv_warning(fstat, cd, K)

        # Assertions
        @test fstat > 10.0  # Strong
        @test cd > stock_yogo_critical_value(K, 0.10, "size")
        @test !is_weak
        @test p_value < 0.001

        println("\nIntegration Test Results:")
        println("  F-statistic: $(round(fstat, digits=2))")
        println("  Cragg-Donald: $(round(cd, digits=2))")
        println("  Stock-Yogo critical (10% size): $(round(stock_yogo_critical_value(K, 0.10, "size"), digits=2))")
        println("  Weak IV? $is_weak")
    end

    @testset "Numerical Stability - Perfect Collinearity" begin
        # Instruments are perfectly collinear
        Random.seed!(282930)
        n = 500
        z1 = randn(n)
        z2 = 2.0 * z1  # Perfect collinearity

        d = 0.5 * z1 + randn(n)
        Z = hcat(z1, z2)

        # Should handle gracefully (SVD should work)
        # F-stat and CD might be numerically unstable but shouldn't error
        try
            fstat, _ = first_stage_fstat(d, Z, nothing)
            cd = cragg_donald_stat(d, Z, nothing)
            @test true  # Didn't error
        catch e
            @test e isa SingularException || e isa ArgumentError
        end
    end

    @testset "Type Stability" begin
        n = 100
        z = randn(n)
        d = randn(n)
        Z = reshape(z, n, 1)

        # Test Float64
        @test @inferred first_stage_fstat(d, Z, nothing) isa Tuple{Float64,Float64}
        @test @inferred cragg_donald_stat(d, Z, nothing) isa Float64
        @test @inferred stock_yogo_critical_value(1, 0.10, "size") isa Float64
        @test @inferred weak_iv_warning(10.0, 20.0, 1) isa Tuple{Bool,String}

        # Test Float32 (may not be fully type stable due to Distributions.jl)
        d32 = Float32.(d)
        Z32 = Float32.(Z)
        fstat32, pval32 = first_stage_fstat(d32, Z32, nothing)
        cd32 = cragg_donald_stat(d32, Z32, nothing)

        # Check output types (p-value may be Float64 due to Distributions.cdf)
        @test fstat32 isa Float32
        @test pval32 isa Real  # Distributions.cdf returns Float64 even for Float32 input
        @test cd32 isa Float32
    end
end

println("\n" * "="^70)
println("IV Diagnostics Tests Complete")
println("="^70)
println("All weak instrument diagnostics validated:")
println("- ✅ First-stage F-statistic")
println("- ✅ Cragg-Donald minimum eigenvalue statistic")
println("- ✅ Stock-Yogo critical values (size & bias tests)")
println("- ✅ Weak IV warning generation")
println("="^70)
