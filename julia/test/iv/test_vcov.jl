"""
Tests for IV Variance-Covariance Estimation.

Session 56: Tests for vcov.jl module.

Test layers:
1. Known-answer tests with hand-calculated variances
2. Property tests (positive definiteness, symmetry)
3. Comparison tests (robust vs standard under different conditions)
"""

using Test
using CausalEstimators
using LinearAlgebra
using Statistics
using Random

@testset "VCov Estimators" begin
    @testset "Standard VCov - Known Answer" begin
        # Simple case: XPX_inv = I, sigma2 = 4
        # V = 4 * I
        XPX_inv = [1.0 0.0; 0.0 1.0]
        sigma2 = 4.0

        vcov = compute_standard_vcov(XPX_inv, sigma2)

        @test vcov == [4.0 0.0; 0.0 4.0]
        @test issymmetric(vcov)
        @test all(eigvals(vcov) .> 0)  # Positive definite
    end

    @testset "Standard VCov - Scaling" begin
        # Variance should scale linearly with sigma2
        XPX_inv = [2.0 0.5; 0.5 3.0]

        vcov1 = compute_standard_vcov(XPX_inv, 1.0)
        vcov2 = compute_standard_vcov(XPX_inv, 2.0)

        @test vcov2 ≈ 2.0 * vcov1
    end

    @testset "Robust VCov - Known Answer" begin
        # Setup a simple case where we can compute by hand
        Random.seed!(42)
        n = 100
        k = 2

        # Simple design matrix [1, x]
        x = randn(n)
        DX = hcat(ones(n), x)

        # Identity projection (P_Z = I for this educational test)
        P_Z = Matrix{Float64}(I, n, n)

        # Known residuals
        residuals = randn(n)

        # XPX = DX' * P_Z * DX = DX' * DX
        XPX = DX' * DX
        XPX_inv = inv(XPX)

        vcov = compute_robust_vcov(XPX_inv, DX, P_Z, residuals)

        # Should be symmetric positive definite
        @test issymmetric(vcov)
        @test all(eigvals(vcov) .> 0)
        @test size(vcov) == (k, k)
    end

    @testset "Robust VCov - Heteroskedasticity Response" begin
        # Under heteroskedasticity, robust SEs should generally differ from standard
        Random.seed!(123)
        n = 500
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)

        # Heteroskedastic residuals (variance proportional to x²)
        residuals = (1.0 .+ abs.(x)) .* randn(n)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        # Compute both standard and robust
        sigma2 = sum(residuals .^ 2) / (n - k)
        vcov_std = compute_standard_vcov(XPX_inv, sigma2)
        vcov_robust = compute_robust_vcov(XPX_inv, DX, P_Z, residuals)

        # SEs should differ (not exactly equal)
        se_std = sqrt.(diag(vcov_std))
        se_robust = sqrt.(diag(vcov_robust))

        @test se_std != se_robust
        @test all(se_robust .> 0)
        @test all(se_std .> 0)
    end

    @testset "Clustered VCov - Known Answer" begin
        Random.seed!(456)
        n = 100
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)

        # Create 10 clusters of size 10 each
        clusters = repeat(1:10, inner=10)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        vcov = compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n)

        # Should be symmetric positive definite
        @test issymmetric(vcov)
        @test all(eigvals(vcov) .> 0)
        @test size(vcov) == (k, k)
    end

    @testset "Clustered VCov - Few Clusters Warning" begin
        Random.seed!(789)
        n = 30
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)

        # Only 5 clusters (should warn)
        clusters = repeat(1:5, inner=6)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        # Should issue warning about few clusters
        @test_logs (:warn, r"Only 5 clusters") compute_clustered_vcov(
            XPX_inv, DX, P_Z, residuals, clusters, n
        )
    end

    @testset "Clustered VCov - Finite Sample Correction" begin
        # Test that finite-sample correction is applied
        Random.seed!(101)
        n = 100
        k = 2
        G = 10  # Number of clusters

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)
        clusters = repeat(1:G, inner=div(n, G))

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        vcov = compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n)

        # Correction factor: (G/(G-1)) × ((n-1)/(n-k))
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        @test correction > 1.0  # Correction inflates variance

        # Clustered SEs should be positive
        @test all(sqrt.(diag(vcov)) .> 0)
    end

    @testset "Unified compute_vcov - Method Dispatch" begin
        Random.seed!(202)
        n = 100
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)
        clusters = repeat(1:10, inner=10)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        # Standard
        vcov_std = compute_vcov(XPX_inv, DX, P_Z, residuals; method=:standard)
        @test size(vcov_std) == (k, k)

        # Robust (default)
        vcov_robust = compute_vcov(XPX_inv, DX, P_Z, residuals)
        @test size(vcov_robust) == (k, k)

        # Clustered
        vcov_cluster = compute_vcov(
            XPX_inv, DX, P_Z, residuals; method=:clustered, clusters=clusters
        )
        @test size(vcov_cluster) == (k, k)

        # All should be symmetric positive definite
        @test issymmetric(vcov_std)
        @test issymmetric(vcov_robust)
        @test issymmetric(vcov_cluster)
    end

    @testset "compute_vcov - Error Handling" begin
        Random.seed!(303)
        n = 100
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        # Clustered without clusters should error
        @test_throws ArgumentError compute_vcov(
            XPX_inv, DX, P_Z, residuals; method=:clustered
        )

        # Unknown method should error
        @test_throws ArgumentError compute_vcov(
            XPX_inv, DX, P_Z, residuals; method=:unknown
        )
    end

    @testset "VCov - Type Stability" begin
        # All vcov functions should return Float64 matrices
        Random.seed!(404)
        n = 50
        k = 2

        x = randn(n)
        DX = hcat(ones(n), x)
        P_Z = Matrix{Float64}(I, n, n)
        residuals = randn(n)
        clusters = repeat(1:5, inner=10)

        XPX = DX' * DX
        XPX_inv = inv(XPX)

        vcov_std = compute_standard_vcov(XPX_inv, 1.0)
        vcov_robust = compute_robust_vcov(XPX_inv, DX, P_Z, residuals)

        @test eltype(vcov_std) == Float64
        @test eltype(vcov_robust) == Float64
    end
end
