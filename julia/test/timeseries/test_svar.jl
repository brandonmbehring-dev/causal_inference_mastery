#=
Tests for Structural VAR (SVAR)

Session 154: Long-run SVAR (Blanchard-Quah identification)

Test Layers:
- Layer 1: Known-Answer - Verify C(1) triangular structure
- Layer 2: Adversarial - Edge cases and error handling
=#

using Test
using Random
using Statistics
using LinearAlgebra
using CausalEstimators

# =============================================================================
# Layer 1: Known-Answer Tests for Long-Run SVAR
# =============================================================================

@testset "Long-Run SVAR Known-Answer" begin

    @testset "Basic long-run SVAR identification" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        # Check identification method
        @test svar_result.identification == CausalEstimators.TimeSeries.SVAR.LONG_RUN
        @test is_just_identified(svar_result)

        # Check B0_inv is finite and correct shape
        @test all(isfinite.(svar_result.B0_inv))
        @test size(svar_result.B0_inv) == (2, 2)
    end

    @testset "C(1) is lower triangular" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        # Compute long-run impact matrix
        C1 = long_run_impact_matrix(svar_result)

        # Upper triangle of C(1) should be zero
        upper_triangle = triu(C1, 1)
        @test isapprox(upper_triangle, zeros(2, 2), atol=1e-10)
    end

    @testset "3-variable long-run SVAR" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 3)

        var_result = var_estimate(data, lags=2, var_names=["x", "y", "z"])
        svar_result = long_run_svar(var_result)

        # C(1) should be lower triangular
        C1 = long_run_impact_matrix(svar_result)
        upper_triangle = triu(C1, 1)
        @test isapprox(upper_triangle, zeros(3, 3), atol=1e-10)

        # Check number of restrictions
        expected_restrictions = 3 * (3 - 1) ÷ 2  # = 3
        @test svar_result.n_restrictions == expected_restrictions
    end

    @testset "Structural shocks are orthogonal" begin
        Random.seed!(42)
        n = 500
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        shocks = svar_result.structural_shocks
        corr = cor(shocks[:, 1], shocks[:, 2])

        # Shocks should be approximately uncorrelated
        @test abs(corr) < 0.15
    end

    @testset "IRF computation works with long-run SVAR" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        # Compute IRF
        irf = compute_irf(svar_result, horizons=20)

        @test size(irf.irf) == (2, 2, 21)
        @test all(isfinite.(irf.irf))

        # IRF at horizon 0 should equal B0_inv
        @test isapprox(irf.irf[:, :, 1], svar_result.B0_inv, atol=1e-10)
    end

    @testset "FEVD computation works with long-run SVAR" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        # Compute FEVD
        fevd = compute_fevd(svar_result, horizons=20)

        @test size(fevd.fevd) == (2, 2, 21)

        # Rows should sum to 1
        for h in 1:21
            row_sums = sum(fevd.fevd[:, :, h], dims=2)
            @test isapprox(row_sums, ones(2, 1), atol=1e-10)
        end
    end

    @testset "Single lag long-run SVAR" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=1)
        svar_result = long_run_svar(var_result)

        @test svar_result.lags == 1

        # C(1) should still be lower triangular
        C1 = long_run_impact_matrix(svar_result)
        @test isapprox(triu(C1, 1), zeros(2, 2), atol=1e-10)
    end

end


# =============================================================================
# Layer 2: Adversarial Tests for Long-Run SVAR
# =============================================================================

@testset "Long-Run SVAR Adversarial" begin

    @testset "Invalid ordering raises error" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 3)

        var_result = var_estimate(data, lags=2, var_names=["x", "y", "z"])

        # Invalid variable name
        @test_throws ErrorException long_run_svar(var_result, ordering=["x", "y", "invalid"])

        # Wrong number of variables
        @test_throws ErrorException long_run_svar(var_result, ordering=["x", "y"])
    end

    @testset "Long-run vs Cholesky differ" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)

        svar_chol = cholesky_svar(var_result)
        svar_lr = long_run_svar(var_result)

        # Identification methods differ
        @test svar_chol.identification == CausalEstimators.TimeSeries.SVAR.CHOLESKY
        @test svar_lr.identification == CausalEstimators.TimeSeries.SVAR.LONG_RUN

        # Both are valid identifications (can still verify)
        is_valid_chol, _ = verify_identification(svar_chol.var_sigma, svar_chol.B0_inv)
        is_valid_lr, _ = verify_identification(svar_lr.var_sigma, svar_lr.B0_inv)

        @test is_valid_chol
        @test is_valid_lr
    end

    @testset "High-dimensional long-run SVAR" begin
        Random.seed!(42)
        n = 500
        k = 5

        data = randn(n, k)
        var_result = var_estimate(data, lags=2)
        svar_result = long_run_svar(var_result)

        # C(1) should be lower triangular
        C1 = long_run_impact_matrix(svar_result)
        upper_triangle = triu(C1, 1)
        @test isapprox(upper_triangle, zeros(k, k), atol=1e-10)

        # Check dimensions
        @test svar_result.n_vars == k
        @test svar_result.n_restrictions == k * (k - 1) ÷ 2  # = 10
    end

end


# =============================================================================
# Cholesky SVAR Tests (Existing functionality)
# =============================================================================

@testset "Cholesky SVAR" begin

    @testset "Basic Cholesky SVAR" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 2)

        var_result = var_estimate(data, lags=2)
        svar_result = cholesky_svar(var_result)

        # Check identification
        @test svar_result.identification == CausalEstimators.TimeSeries.SVAR.CHOLESKY
        @test is_just_identified(svar_result)

        # B0_inv should be lower triangular
        @test isapprox(triu(svar_result.B0_inv, 1), zeros(2, 2), atol=1e-10)

        # Verify identification
        is_valid, max_error = verify_identification(var_result.sigma, svar_result.B0_inv)
        @test is_valid
        @test max_error < 1e-8
    end

    @testset "Cholesky with ordering" begin
        Random.seed!(42)
        n = 300
        data = randn(n, 3)

        var_result = var_estimate(data, lags=2, var_names=["x", "y", "z"])

        # Different orderings should give different results
        svar1 = cholesky_svar(var_result, ordering=["x", "y", "z"])
        svar2 = cholesky_svar(var_result, ordering=["z", "y", "x"])

        @test !isapprox(svar1.B0_inv, svar2.B0_inv)
    end

end


# =============================================================================
# VMA and IRF Tests
# =============================================================================

@testset "VMA Coefficients" begin

    @testset "VMA at horizon 0 is identity" begin
        Random.seed!(42)
        data = randn(200, 2)
        var_result = var_estimate(data, lags=2)

        Phi = vma_coefficients(var_result, 10)

        @test size(Phi) == (2, 2, 11)
        @test isapprox(Phi[:, :, 1], Matrix{Float64}(I, 2, 2))
    end

    @testset "VMA coefficients decay for stable VAR" begin
        Random.seed!(42)
        data = randn(300, 2)
        var_result = var_estimate(data, lags=2)

        Phi = vma_coefficients(var_result, 50)

        # Coefficients should decay
        early_norm = norm(Phi[:, :, 6])  # horizon 5
        late_norm = norm(Phi[:, :, 51])  # horizon 50

        @test late_norm < early_norm
    end

end


@testset "Stability Check" begin

    @testset "Stable VAR detection" begin
        Random.seed!(42)

        # Generate stable VAR
        A1 = [0.5 0.1; 0.2 0.4]
        n = 200
        data = zeros(n, 2)
        for t in 2:n
            data[t, :] = A1 * data[t-1, :] + randn(2) * 0.5
        end

        var_result = var_estimate(data, lags=1)
        is_stable, eigenvalues = check_stability(var_result)

        @test is_stable
        @test all(abs.(eigenvalues) .< 1.0)
    end

end
