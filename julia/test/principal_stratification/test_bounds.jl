"""
Tests for principal stratification bounds in Julia.

Test Structure (following 6-layer validation):
- Layer 1: Known-answer tests (bounds contain true value)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo validation (coverage)
- Layer 4: Cross-language parity (vs Python) - separate file
"""

using Test
using Statistics
using Random

# Include the module
include("../../src/principal_stratification/bounds.jl")

# =============================================================================
# Test Data Generator
# =============================================================================

"""
Generate data from principal stratification DGP.
"""
function generate_ps_dgp(;
    n::Int = 500,
    pi_c::Float64 = 0.60,
    pi_a::Float64 = 0.20,
    pi_n::Float64 = 0.20,
    true_cace::Float64 = 2.0,
    direct_effect::Float64 = 0.0,
    baseline::Float64 = 1.0,
    noise_sd::Float64 = 1.0,
    seed::Int = 42
)
    Random.seed!(seed)

    # Normalize
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c/total, pi_a/total, pi_n/total

    # Random assignment
    Z = rand(n) .< 0.5

    # Strata
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

    # Treatment
    D = zeros(Float64, n)
    for i in 1:n
        if strata[i] == 0
            D[i] = Float64(Z[i])
        elseif strata[i] == 1
            D[i] = 1.0
        else
            D[i] = 0.0
        end
    end

    # Outcome with possible direct effect
    Y = baseline .+ true_cace .* D .+ direct_effect .* Float64.(Z) .+ noise_sd .* randn(n)

    return (
        Y = Y,
        D = D,
        Z = Float64.(Z),
        true_cace = true_cace,
        direct_effect = direct_effect,
        strata = strata,
        pi_c = pi_c,
        pi_a = pi_a,
        pi_n = pi_n
    )
end


# =============================================================================
# Layer 1: ps_bounds_monotonicity Tests
# =============================================================================

@testset "ps_bounds_monotonicity Tests" begin
    @testset "Returns valid BoundsResult" begin
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_monotonicity(data.Y, data.D, data.Z)

        @test isa(result, BoundsResult)
        @test hasfield(BoundsResult, :lower_bound)
        @test hasfield(BoundsResult, :upper_bound)
        @test hasfield(BoundsResult, :bound_width)
        @test hasfield(BoundsResult, :identified)
    end

    @testset "No direct effect point identified" begin
        data = generate_ps_dgp(n=500, direct_effect=0.0, seed=42)
        result = ps_bounds_monotonicity(data.Y, data.D, data.Z, direct_effect_bound=0.0)

        @test result.identified == true
        @test isapprox(result.bound_width, 0.0, atol=1e-10)
        @test result.lower_bound == result.upper_bound
    end

    @testset "Direct effect widens bounds" begin
        data = generate_ps_dgp(n=300, seed=42)

        result_tight = ps_bounds_monotonicity(data.Y, data.D, data.Z, direct_effect_bound=0.1)
        result_wide = ps_bounds_monotonicity(data.Y, data.D, data.Z, direct_effect_bound=1.0)

        @test result_wide.bound_width > result_tight.bound_width
    end

    @testset "Bounds contain true CACE" begin
        data = generate_ps_dgp(n=500, true_cace=2.0, direct_effect=0.0, seed=42)
        result = ps_bounds_monotonicity(data.Y, data.D, data.Z, direct_effect_bound=0.0)

        point_estimate = (result.lower_bound + result.upper_bound) / 2
        @test abs(point_estimate - 2.0) < 0.5
    end

    @testset "Weak instrument infinite bounds" begin
        Random.seed!(42)
        n = 300
        Z = Float64.(rand(n) .< 0.5)
        D = zeros(n)  # No variation in D
        Y = randn(n)

        result = ps_bounds_monotonicity(Y, D, Z)
        @test result.lower_bound == -Inf
        @test result.upper_bound == Inf
    end
end


# =============================================================================
# Layer 1: ps_bounds_no_assumption Tests
# =============================================================================

@testset "ps_bounds_no_assumption Tests" begin
    @testset "Returns valid BoundsResult" begin
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_no_assumption(data.Y, data.D, data.Z)

        @test isa(result, BoundsResult)
        @test result.lower_bound <= result.upper_bound
        @test result.identified == false
        @test result.assumptions == String[]
    end

    @testset "Bounds use outcome support" begin
        data = generate_ps_dgp(n=300, seed=42)
        Y = data.Y

        result = ps_bounds_no_assumption(data.Y, data.D, data.Z)

        Y_min, Y_max = minimum(Y), maximum(Y)
        expected_lower = Y_min - Y_max
        expected_upper = Y_max - Y_min

        @test result.lower_bound >= expected_lower - 0.1
        @test result.upper_bound <= expected_upper + 0.1
    end

    @testset "Custom support used" begin
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_no_assumption(data.Y, data.D, data.Z, outcome_support=(-5.0, 5.0))

        @test result.lower_bound >= -10.0
        @test result.upper_bound <= 10.0
    end
end


# =============================================================================
# Layer 1: ps_bounds_balke_pearl Tests
# =============================================================================

@testset "ps_bounds_balke_pearl Tests" begin
    @testset "Returns valid BoundsResult" begin
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_balke_pearl(data.Y, data.D, data.Z)

        @test isa(result, BoundsResult)
        @test result.lower_bound <= result.upper_bound
        @test "iv_constraints" in result.assumptions
    end
end


# =============================================================================
# Layer 2: Input Validation Tests
# =============================================================================

@testset "Bounds Input Validation" begin
    @testset "Length mismatch raises error" begin
        Y = randn(100)
        D = rand(100) .< 0.5
        Z = rand(50) .< 0.5  # Wrong length

        @test_throws ArgumentError ps_bounds_monotonicity(Y, Float64.(D), Float64.(Z))
    end

    @testset "Non-binary treatment raises error" begin
        Y = randn(100)
        D = rand(0:2, 100)  # Not binary
        Z = Float64.(rand(100) .< 0.5)

        @test_throws ArgumentError ps_bounds_monotonicity(Y, Float64.(D), Z)
    end

    @testset "Non-binary instrument raises error" begin
        Y = randn(100)
        D = Float64.(rand(100) .< 0.5)
        Z = rand(0:2, 100)  # Not binary

        @test_throws ArgumentError ps_bounds_monotonicity(Y, D, Float64.(Z))
    end

    @testset "Negative direct_effect_bound raises error" begin
        data = generate_ps_dgp(n=100, seed=42)
        @test_throws ArgumentError ps_bounds_monotonicity(
            data.Y, data.D, data.Z, direct_effect_bound=-0.5
        )
    end

    @testset "Invalid outcome_support raises error" begin
        data = generate_ps_dgp(n=100, seed=42)
        @test_throws ArgumentError ps_bounds_no_assumption(
            data.Y, data.D, data.Z, outcome_support=(5.0, -5.0)
        )
    end
end


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================

@testset "Bounds Monte Carlo Tests" begin
    @testset "Bounds cover true CACE" begin
        n_sims = 100
        true_cace = 2.0
        covers = 0

        for seed in 1:n_sims
            data = generate_ps_dgp(n=500, true_cace=true_cace, direct_effect=0.0, seed=seed)
            result = ps_bounds_monotonicity(data.Y, data.D, data.Z, direct_effect_bound=0.0)

            midpoint = (result.lower_bound + result.upper_bound) / 2
            if abs(midpoint - true_cace) < 1.0
                covers += 1
            end
        end

        coverage = covers / n_sims
        @test coverage > 0.80
    end
end
