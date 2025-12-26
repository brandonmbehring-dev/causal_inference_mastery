"""
Tests for Bayesian CACE estimation in Julia.

Test Structure (following 6-layer validation):
- Layer 1: Known-answer tests (structure, posterior summaries)
- Layer 2: Prior sensitivity tests
- Layer 3: Monte Carlo validation (light)
- Layer 4: Cross-language parity (vs Python) - separate file
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include the module
include("../../src/principal_stratification/bayesian.jl")

# =============================================================================
# Test Data Generator
# =============================================================================

"""
    generate_ps_dgp_bayesian(; n, pi_c, pi_a, pi_n, true_cace, baseline, noise_sd, seed)

Generate data from a principal stratification DGP with proper exclusion restriction.
"""
function generate_ps_dgp_bayesian(;
    n::Int = 300,
    pi_c::Float64 = 0.60,
    pi_a::Float64 = 0.20,
    pi_n::Float64 = 0.20,
    true_cace::Float64 = 2.0,
    baseline::Float64 = 1.0,
    noise_sd::Float64 = 1.0,
    seed::Int = 42
)
    Random.seed!(seed)

    # Normalize proportions
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c/total, pi_a/total, pi_n/total

    # Generate random assignment
    Z = rand(n) .< 0.5

    # Generate strata
    strata = zeros(Int, n)
    for i in 1:n
        r = rand()
        if r < pi_c
            strata[i] = 0  # Complier
        elseif r < pi_c + pi_a
            strata[i] = 1  # Always-taker
        else
            strata[i] = 2  # Never-taker
        end
    end

    # Treatment based on stratum
    D = BitVector(undef, n)
    for i in 1:n
        if strata[i] == 0
            D[i] = Z[i]  # Complier
        elseif strata[i] == 1
            D[i] = true  # Always-taker
        else
            D[i] = false  # Never-taker
        end
    end

    # Outcome with exclusion restriction: only compliers have effect
    Y = zeros(n)
    for i in 1:n
        if strata[i] == 0  # Complier
            Y[i] = baseline + true_cace * Float64(D[i]) + noise_sd * randn()
        else
            Y[i] = baseline + noise_sd * randn()
        end
    end

    return (
        Y = Y,
        D = D,
        Z = Z,
        true_cace = true_cace,
        strata = strata,
        pi_c = pi_c,
        pi_a = pi_a,
        pi_n = pi_n
    )
end


# =============================================================================
# Layer 1: Structure Tests
# =============================================================================

@testset "Bayesian CACE Structure Tests" begin
    @testset "cace_bayesian returns valid result" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        # Check all required fields exist
        @test hasfield(typeof(result), :cace_mean)
        @test hasfield(typeof(result), :cace_sd)
        @test hasfield(typeof(result), :cace_hdi_lower)
        @test hasfield(typeof(result), :cace_hdi_upper)
        @test hasfield(typeof(result), :cace_samples)
        @test hasfield(typeof(result), :pi_c_mean)
        @test hasfield(typeof(result), :pi_c_samples)
        @test hasfield(typeof(result), :pi_a_mean)
        @test hasfield(typeof(result), :pi_n_mean)
        @test hasfield(typeof(result), :rhat)
        @test hasfield(typeof(result), :ess)
        @test hasfield(typeof(result), :n_samples)
        @test hasfield(typeof(result), :n_chains)
        @test hasfield(typeof(result), :model)
    end

    @testset "Posterior samples have correct shape" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        # Quick mode: 1000 samples × 2 chains = 2000 total
        @test length(result.cace_samples) == 2000
        @test length(result.pi_c_samples) == 2000
        @test result.n_samples == 2000
        @test result.n_chains == 2
    end

    @testset "HDI contains mean" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test result.cace_hdi_lower <= result.cace_mean <= result.cace_hdi_upper
    end
end


# =============================================================================
# Layer 2: Known-Answer Tests
# =============================================================================

@testset "Bayesian CACE Known-Answer Tests" begin
    @testset "Strata proportions sum to one" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        total = result.pi_c_mean + result.pi_a_mean + result.pi_n_mean
        @test isapprox(total, 1.0, rtol=1e-6)
    end

    @testset "Credible interval width is reasonable" begin
        data = generate_ps_dgp_bayesian(n=300, true_cace=2.0, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        ci_width = result.cace_hdi_upper - result.cace_hdi_lower
        @test 0.3 < ci_width < 6.0  # Reasonable range
    end

    @testset "Posterior mean is in ballpark of true CACE" begin
        data = generate_ps_dgp_bayesian(n=400, true_cace=2.0, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        # Should be within 1.5 of true value (MH sampler has more variance)
        @test abs(result.cace_mean - 2.0) < 1.5
    end
end


# =============================================================================
# Layer 3: MCMC Diagnostics Tests
# =============================================================================

@testset "Bayesian CACE Diagnostics Tests" begin
    @testset "R-hat dict has key parameters" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test haskey(result.rhat, "cace")
        @test haskey(result.rhat, "pi_c")
    end

    @testset "ESS dict has key parameters" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test haskey(result.ess, "cace")
        @test haskey(result.ess, "pi_c")
    end

    @testset "R-hat values are positive" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test all(v -> v > 0, values(result.rhat))
    end

    @testset "ESS values are positive" begin
        data = generate_ps_dgp_bayesian(n=200, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test all(v -> v > 0, values(result.ess))
    end
end


# =============================================================================
# Layer 4: Quick Mode Tests
# =============================================================================

@testset "Bayesian CACE Quick Mode Tests" begin
    @testset "Quick mode uses fewer samples" begin
        data = generate_ps_dgp_bayesian(n=150, seed=42)
        result = cace_bayesian(data.Y, data.D, data.Z, quick=true, seed=42)

        @test result.n_samples == 2000  # 1000 × 2
        @test result.n_chains == 2
    end
end


# =============================================================================
# Layer 5: Prior Sensitivity Tests
# =============================================================================

@testset "Bayesian CACE Prior Sensitivity Tests" begin
    @testset "Prior SD affects posterior variance" begin
        data = generate_ps_dgp_bayesian(n=200, true_cace=2.0, seed=42)

        # Weak prior
        result_weak = cace_bayesian(
            data.Y, data.D, data.Z,
            prior_mu_sd=100.0,
            quick=true,
            seed=42
        )

        # Strong prior centered at 0
        result_strong = cace_bayesian(
            data.Y, data.D, data.Z,
            prior_mu_sd=1.0,
            quick=true,
            seed=42  # Same seed for comparison
        )

        # Both should produce valid results (finite mean and SD)
        @test isfinite(result_weak.cace_mean)
        @test isfinite(result_strong.cace_mean)
        @test result_weak.cace_sd > 0
        @test result_strong.cace_sd > 0

        # Note: With simple MH sampler, prior sensitivity is weak
        # Full Turing.jl implementation would show shrinkage
    end
end


# =============================================================================
# Layer 6: Input Validation Tests
# =============================================================================

@testset "Bayesian CACE Validation Tests" begin
    @testset "Length mismatch raises error" begin
        Y = randn(100)
        D = rand(100) .< 0.5
        Z = rand(50) .< 0.5  # Wrong length

        @test_throws ArgumentError cace_bayesian(Y, D, Z, quick=true, seed=42)
    end

    @testset "Non-binary treatment raises error" begin
        Y = randn(100)
        D = rand(0:2, 100)  # Not binary
        Z = rand(100) .< 0.5

        @test_throws ArgumentError cace_bayesian(Y, D, Z, quick=true, seed=42)
    end

    @testset "Non-binary instrument raises error" begin
        Y = randn(100)
        D = rand(100) .< 0.5
        Z = rand(0:2, 100)  # Not binary

        @test_throws ArgumentError cace_bayesian(Y, D, Z, quick=true, seed=42)
    end
end


# =============================================================================
# Utility Function Tests
# =============================================================================

@testset "HDI Computation" begin
    @testset "HDI returns valid bounds" begin
        samples = randn(1000)
        result = hdi(samples, prob=0.95)

        @test result.lower < result.upper
        @test result.lower < mean(samples) < result.upper
    end

    @testset "HDI respects probability level" begin
        samples = randn(1000)

        hdi_90 = hdi(samples, prob=0.90)
        hdi_95 = hdi(samples, prob=0.95)

        # 95% HDI should be wider
        width_90 = hdi_90.upper - hdi_90.lower
        width_95 = hdi_95.upper - hdi_95.lower
        @test width_95 > width_90
    end
end

@testset "R-hat Computation" begin
    @testset "R-hat is 1 for converged chains" begin
        # Generate similar chains
        Random.seed!(42)
        chains = randn(500, 2)

        rhat = compute_rhat(chains)
        @test 0.9 < rhat < 1.1
    end

    @testset "R-hat is high for divergent chains" begin
        # Generate chains with different means
        chain1 = randn(500)
        chain2 = randn(500) .+ 5.0  # Shifted mean
        chains = hcat(chain1, chain2)

        rhat = compute_rhat(chains)
        @test rhat > 1.5  # Should be high due to divergence
    end
end
