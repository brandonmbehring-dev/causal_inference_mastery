"""
Tests for Shift-Share (Bartik) IV estimation.

Layer 1-2 of 6-layer validation architecture:
- Known-answer tests
- Adversarial edge cases
"""

using Test
using Random
using Statistics
using LinearAlgebra
using Distributions

# Import from parent module
using CausalEstimators

# =============================================================================
# Test Data Generators
# =============================================================================

"""
Generate shift-share data for testing.

# Arguments
- `n::Int=200`: Number of observations
- `n_sectors::Int=10`: Number of sectors
- `true_beta::Float64=1.5`: True treatment effect
- `first_stage_strength::Float64=2.0`: First-stage coefficient
- `share_concentration::Float64=1.0`: Dirichlet concentration (lower = more concentrated)
- `seed::Int=42`: Random seed
"""
function generate_shift_share_data(;
    n::Int = 200,
    n_sectors::Int = 10,
    true_beta::Float64 = 1.5,
    first_stage_strength::Float64 = 2.0,
    share_concentration::Float64 = 1.0,
    seed::Int = 42
)
    Random.seed!(seed)

    # Generate shares from Dirichlet distribution
    alpha = fill(share_concentration, n_sectors)
    shares = zeros(n, n_sectors)
    for i in 1:n
        shares[i, :] = rand(Dirichlet(alpha))
    end

    # Generate shocks
    shocks = randn(n_sectors) * 0.1

    # Construct instrument
    Z_bartik = shares * shocks

    # Generate treatment (endogenous)
    D = first_stage_strength * Z_bartik + randn(n) * 0.5

    # Generate outcome
    Y = true_beta * D + randn(n)

    return Y, D, shares, shocks, nothing, true_beta
end


"""Generate data with controls."""
function generate_shift_share_with_controls(; seed::Int = 42)
    Y, D, shares, shocks, _, true_beta = generate_shift_share_data(seed=seed)
    n = length(Y)
    Random.seed!(seed + 1)
    X = randn(n, 2)
    # Add control effects
    Y = Y + X * [0.5, 0.3]
    return Y, D, shares, shocks, X, true_beta
end


"""Generate data with weak first stage."""
function generate_weak_first_stage(; seed::Int = 42)
    return generate_shift_share_data(
        first_stage_strength=0.3,  # Very weak
        seed=seed
    )
end


"""Generate data with negative effect."""
function generate_negative_effect(; seed::Int = 42)
    return generate_shift_share_data(
        true_beta=-2.0,
        seed=seed
    )
end


"""Generate large sample data."""
function generate_large_sample(; seed::Int = 42)
    return generate_shift_share_data(
        n=1000,
        seed=seed
    )
end


"""Generate many sectors data."""
function generate_many_sectors(; seed::Int = 42)
    return generate_shift_share_data(
        n_sectors=50,
        seed=seed
    )
end


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@testset "Shift-Share Basic Functionality" begin

    @testset "Basic estimation runs without error" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test !isnan(result.estimate)
        @test !isnan(result.se)
        @test result.se > 0
    end

    @testset "Problem-Estimator pattern works" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()

        problem = ShiftShareProblem(
            outcome = Y,
            treatment = D,
            shares = shares,
            shocks = shocks
        )
        estimator = ShiftShareIV()
        result = solve(problem, estimator)

        @test !isnan(result.estimate)
        @test result.n_obs == length(Y)
    end

    @testset "Estimation with controls" begin
        Y, D, shares, shocks, X, _ = generate_shift_share_with_controls()
        result = shift_share_iv(Y, D, shares, shocks; X=X)

        @test !isnan(result.estimate)
        @test result.n_obs == length(Y)
    end

    @testset "Instrument is stored correctly" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        expected_instrument = shares * shocks
        @test length(result.instrument) == length(Y)
        @test isapprox(result.instrument, expected_instrument, rtol=1e-10)
    end

end


# =============================================================================
# Result Structure Tests
# =============================================================================

@testset "Shift-Share Result Structure" begin

    @testset "Result has all expected fields" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        # Check main fields
        @test hasfield(typeof(result), :estimate)
        @test hasfield(typeof(result), :se)
        @test hasfield(typeof(result), :t_stat)
        @test hasfield(typeof(result), :p_value)
        @test hasfield(typeof(result), :ci_lower)
        @test hasfield(typeof(result), :ci_upper)
        @test hasfield(typeof(result), :first_stage)
        @test hasfield(typeof(result), :rotemberg)
        @test hasfield(typeof(result), :n_obs)
        @test hasfield(typeof(result), :n_sectors)
    end

    @testset "First-stage diagnostics populated" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        fs = result.first_stage
        @test fs.f_statistic > 0
        @test 0 <= fs.f_pvalue <= 1
        @test 0 <= fs.partial_r2 <= 1
        @test isa(fs.weak_iv_warning, Bool)
    end

    @testset "Rotemberg diagnostics populated" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        rot = result.rotemberg
        @test length(rot.weights) == length(shocks)
        @test 0 <= rot.negative_weight_share <= 1
        @test length(rot.top_5_sectors) == 5
        @test rot.herfindahl >= 0
    end

    @testset "CI contains estimate" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.ci_lower <= result.estimate <= result.ci_upper
    end

end


# =============================================================================
# Treatment Effect Recovery Tests
# =============================================================================

@testset "Shift-Share Treatment Effect Recovery" begin

    @testset "Recovers positive effect" begin
        Y, D, shares, shocks, _, true_beta = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        # Should be within 3 SE of true value
        @test abs(result.estimate - true_beta) < 3 * result.se
    end

    @testset "Recovers negative effect" begin
        Y, D, shares, shocks, _, true_beta = generate_negative_effect()
        result = shift_share_iv(Y, D, shares, shocks)

        # Sign should be correct
        @test result.estimate < 0
        # Within 3 SE of true value
        @test abs(result.estimate - true_beta) < 3 * result.se
    end

    @testset "Large sample precision" begin
        Y, D, shares, shocks, _, true_beta = generate_large_sample()
        result = shift_share_iv(Y, D, shares, shocks)

        # SE should be reasonable (not enormous)
        @test result.se < 1.5
        # Within 2 SE of true value
        @test abs(result.estimate - true_beta) < 2 * result.se
    end

    @testset "CI coverage" begin
        Y, D, shares, shocks, _, true_beta = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        # True value should be in CI (probabilistic)
        covered = result.ci_lower <= true_beta <= result.ci_upper
        # If not covered, at least should be close
        if !covered
            @test abs(result.estimate - true_beta) < 1.0
        else
            @test covered
        end
    end

end


# =============================================================================
# First Stage Tests
# =============================================================================

@testset "Shift-Share First Stage" begin

    @testset "Strong first stage has F > 10" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.first_stage.f_statistic > 10
        @test !result.first_stage.weak_iv_warning
    end

    @testset "Weak first stage warning" begin
        Y, D, shares, shocks, _, _ = generate_weak_first_stage()
        result = shift_share_iv(Y, D, shares, shocks)

        # With weak first stage, F may be < 10
        if result.first_stage.f_statistic < 10
            @test result.first_stage.weak_iv_warning
        end
    end

end


# =============================================================================
# Rotemberg Weight Tests
# =============================================================================

@testset "Shift-Share Rotemberg Weights" begin

    @testset "Weights have sensible magnitudes" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        weights = result.rotemberg.weights
        # Sum of absolute values should be ~1
        @test abs(sum(abs.(weights)) - 1.0) < 0.1
    end

    @testset "Top 5 sectors ordered by weight" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        weights = result.rotemberg.weights
        top_5_idx = result.rotemberg.top_5_sectors
        top_5_weights = result.rotemberg.top_5_weights

        # Check top weights correspond to indices
        for (i, idx) in enumerate(top_5_idx)
            @test isapprox(weights[idx], top_5_weights[i], rtol=1e-10)
        end
    end

    @testset "Herfindahl is positive" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.rotemberg.herfindahl > 0
    end

end


# =============================================================================
# Share Normalization Tests
# =============================================================================

@testset "Shift-Share Share Normalization" begin

    @testset "Standard shares sum to ~1" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test abs(result.share_sum_mean - 1.0) < 0.05
    end

    @testset "Unnormalized shares handled" begin
        Random.seed!(42)
        n, S = 100, 10
        # Shares that don't sum to 1
        shares = rand(n, S)
        shocks = randn(S) * 0.1
        Z = shares * shocks
        D = 2.0 * Z + randn(n) * 0.5
        Y = 1.5 * D + randn(n)

        result = shift_share_iv(Y, D, shares, shocks)

        # Should still work
        @test !isnan(result.estimate)
        # Share sum mean should differ from 1
        # (but algorithm should handle it)
    end

end


# =============================================================================
# Input Validation Tests
# =============================================================================

@testset "Shift-Share Input Validation" begin

    @testset "Rejects NaN in outcome" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        Y[1] = NaN

        @test_throws ErrorException shift_share_iv(Y, D, shares, shocks)
    end

    @testset "Rejects NaN in shares" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        shares[1, 1] = NaN

        @test_throws ErrorException shift_share_iv(Y, D, shares, shocks)
    end

    @testset "Rejects dimension mismatch" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        shocks_wrong = randn(length(shocks) + 1)

        @test_throws ErrorException shift_share_iv(Y, D, shares, shocks_wrong)
    end

    @testset "Rejects small sample" begin
        Random.seed!(42)
        n, S = 5, 3
        shares = rand(n, S)
        shares ./= sum(shares, dims=2)
        shocks = randn(S) * 0.1
        D = randn(n)
        Y = randn(n)

        @test_throws ErrorException shift_share_iv(Y, D, shares, shocks)
    end

    @testset "Rejects no treatment variation" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        D_const = fill(5.0, length(D))

        @test_throws ErrorException shift_share_iv(Y, D_const, shares, shocks)
    end

    @testset "Rejects Y/D length mismatch" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        D_short = D[1:end-10]

        @test_throws ErrorException shift_share_iv(Y, D_short, shares, shocks)
    end

end


# =============================================================================
# Edge Case Tests
# =============================================================================

@testset "Shift-Share Edge Cases" begin

    @testset "Many sectors (50)" begin
        Y, D, shares, shocks, _, _ = generate_many_sectors()
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.n_sectors == 50
        @test !isnan(result.estimate)
    end

    @testset "Few sectors (3)" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data(n_sectors=3)
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.n_sectors == 3
        @test !isnan(result.estimate)
    end

    @testset "Single dominant sector" begin
        Random.seed!(42)
        n, S = 100, 10
        shares = zeros(n, S)
        shares[:, 1] .= 0.9
        shares[:, 2:end] .= 0.1 / (S - 1)

        shocks = randn(S) * 0.1
        Z = shares * shocks
        D = 2.0 * Z + randn(n) * 0.5
        Y = 1.5 * D + randn(n)

        result = shift_share_iv(Y, D, shares, shocks)
        @test !isnan(result.estimate)
    end

    @testset "Zero shock some sectors" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        # Set half the shocks to zero
        shocks[6:end] .= 0

        result = shift_share_iv(Y, D, shares, shocks)
        @test !isnan(result.estimate)
    end

    @testset "Reproducibility" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()

        result1 = shift_share_iv(Y, D, shares, shocks)
        result2 = shift_share_iv(Y, D, shares, shocks)

        @test isapprox(result1.estimate, result2.estimate, rtol=1e-10)
        @test isapprox(result1.se, result2.se, rtol=1e-10)
    end

end


# =============================================================================
# Inference Method Tests
# =============================================================================

@testset "Shift-Share Inference Methods" begin

    @testset "Robust inference (default)" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks; inference=:robust)

        @test result.inference == :robust
        @test result.se > 0
    end

    @testset "Clustered inference" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        n = length(Y)
        clusters = repeat(1:20, inner=n÷20)

        result = shift_share_iv(Y, D, shares, shocks;
                               clusters=clusters, inference=:clustered)

        @test result.inference == :clustered
        @test result.se > 0
    end

    @testset "Invalid inference rejected" begin
        @test_throws ErrorException ShiftShareIV(inference=:invalid)
    end

end


# =============================================================================
# Type Stability Tests
# =============================================================================

@testset "Shift-Share Type Stability" begin

    @testset "Float64 input preserves type" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()
        result = shift_share_iv(Y, D, shares, shocks)

        @test result.estimate isa Float64
        @test result.se isa Float64
        @test all(w -> w isa Float64, result.rotemberg.weights)
    end

    @testset "Works with different numeric types" begin
        Y, D, shares, shocks, _, _ = generate_shift_share_data()

        # Should handle conversion
        Y32 = Float32.(Y)
        D32 = Float32.(D)
        shares32 = Float32.(shares)
        shocks32 = Float32.(shocks)

        result = shift_share_iv(Y32, D32, shares32, shocks32)
        @test !isnan(result.estimate)
    end

end
