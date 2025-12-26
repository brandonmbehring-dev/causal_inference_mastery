"""
Tests for SACE (Survivor Average Causal Effect) estimation in Julia.

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
include("../../src/principal_stratification/sace.jl")

# =============================================================================
# Test Data Generator
# =============================================================================

"""
Generate data with truncation by death for SACE estimation.
"""
function generate_sace_dgp(;
    n::Int = 500,
    true_sace::Float64 = 1.5,
    p_AS::Float64 = 0.50,
    p_protected::Float64 = 0.20,
    p_harmed::Float64 = 0.10,
    baseline::Float64 = 1.0,
    noise_sd::Float64 = 1.0,
    seed::Int = 42
)
    Random.seed!(seed)

    # Normalize
    p_never = max(0.0, 1.0 - p_AS - p_protected - p_harmed)
    total = p_AS + p_protected + p_harmed + p_never
    p_AS /= total
    p_protected /= total
    p_harmed /= total
    p_never /= total

    # Random treatment
    D = Float64.(rand(n) .< 0.5)

    # Strata
    strata = zeros(Int, n)
    for i in 1:n
        r = rand()
        if r < p_AS
            strata[i] = 0  # Always-survivor
        elseif r < p_AS + p_protected
            strata[i] = 1  # Protected
        elseif r < p_AS + p_protected + p_harmed
            strata[i] = 2  # Harmed
        else
            strata[i] = 3  # Never-survivor
        end
    end

    # Survival
    S = zeros(Float64, n)
    for i in 1:n
        if strata[i] == 0  # Always-survivor
            S[i] = 1.0
        elseif strata[i] == 1  # Protected
            S[i] = D[i]
        elseif strata[i] == 2  # Harmed
            S[i] = 1.0 - D[i]
        else  # Never-survivor
            S[i] = 0.0
        end
    end

    # Potential outcomes
    Y0_latent = baseline .+ noise_sd .* randn(n)
    Y1_latent = baseline .+ true_sace .+ noise_sd .* randn(n)

    # Observed outcome
    Y = zeros(n)
    for i in 1:n
        if S[i] == 1
            Y[i] = D[i] == 1 ? Y1_latent[i] : Y0_latent[i]
        else
            Y[i] = NaN
        end
    end

    return (
        Y = Y,
        D = D,
        S = S,
        true_sace = true_sace,
        strata = strata,
        p_AS = p_AS,
        p_protected = p_protected,
        p_harmed = p_harmed
    )
end


function generate_selection_monotonicity_dgp(;
    n::Int = 500,
    true_sace::Float64 = 1.5,
    p_AS::Float64 = 0.60,
    p_protected::Float64 = 0.25,
    seed::Int = 42
)
    return generate_sace_dgp(
        n=n,
        true_sace=true_sace,
        p_AS=p_AS,
        p_protected=p_protected,
        p_harmed=0.0,
        seed=seed
    )
end


# =============================================================================
# Layer 1: sace_bounds Structure Tests
# =============================================================================

@testset "SACE Bounds Structure Tests" begin
    @testset "Returns valid SACEResult" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data.Y, data.D, data.S)

        @test isa(result, SACEResult)
        @test hasfield(SACEResult, :sace)
        @test hasfield(SACEResult, :se)
        @test hasfield(SACEResult, :lower_bound)
        @test hasfield(SACEResult, :upper_bound)
        @test hasfield(SACEResult, :proportion_survivors_treat)
        @test hasfield(SACEResult, :proportion_survivors_control)
    end

    @testset "Bounds ordered" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data.Y, data.D, data.S)

        @test result.lower_bound <= result.upper_bound
    end

    @testset "Survival proportions valid" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data.Y, data.D, data.S)

        @test 0 <= result.proportion_survivors_treat <= 1
        @test 0 <= result.proportion_survivors_control <= 1
    end

    @testset "Sample size correct" begin
        n = 300
        data = generate_sace_dgp(n=n, seed=42)
        result = sace_bounds(data.Y, data.D, data.S)

        @test result.n == n
    end
end


# =============================================================================
# Layer 1: Monotonicity Options Tests
# =============================================================================

@testset "SACE Monotonicity Options Tests" begin
    @testset "No monotonicity produces valid bounds" begin
        data = generate_sace_dgp(n=500, seed=42)

        result_none = sace_bounds(data.Y, data.D, data.S, monotonicity="none")
        result_selection = sace_bounds(data.Y, data.D, data.S, monotonicity="selection")

        @test result_none.lower_bound <= result_none.upper_bound
        @test result_selection.lower_bound <= result_selection.upper_bound
        @test isfinite(result_none.lower_bound)
        @test isfinite(result_selection.lower_bound)
    end

    @testset "Selection monotonicity bounds contain true" begin
        data = generate_selection_monotonicity_dgp(n=500, seed=42)
        result = sace_bounds(data.Y, data.D, data.S, monotonicity="selection")

        @test result.lower_bound <= data.true_sace <= result.upper_bound
    end

    @testset "Both monotonicity produces valid bounds" begin
        data = generate_selection_monotonicity_dgp(n=500, seed=42)

        result_selection = sace_bounds(data.Y, data.D, data.S, monotonicity="selection")
        result_both = sace_bounds(data.Y, data.D, data.S, monotonicity="both")

        @test result_both.lower_bound <= result_both.upper_bound
    end
end


# =============================================================================
# Layer 1: sace_sensitivity Tests
# =============================================================================

@testset "SACE Sensitivity Tests" begin
    @testset "Returns expected keys" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data.Y, data.D, data.S)

        @test haskey(result, :alpha)
        @test haskey(result, :lower_bound)
        @test haskey(result, :upper_bound)
        @test haskey(result, :sace)
    end

    @testset "Output lengths match" begin
        n_points = 25
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data.Y, data.D, data.S, n_points=n_points)

        @test length(result.alpha) == n_points
        @test length(result.lower_bound) == n_points
        @test length(result.upper_bound) == n_points
        @test length(result.sace) == n_points
    end

    @testset "Alpha range correct" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data.Y, data.D, data.S, alpha_range=(0.2, 0.8), n_points=50)

        @test isapprox(result.alpha[1], 0.2)
        @test isapprox(result.alpha[end], 0.8)
    end

    @testset "SACE is midpoint" begin
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data.Y, data.D, data.S)

        midpoints = (result.lower_bound .+ result.upper_bound) ./ 2
        @test all(isapprox.(result.sace, midpoints, rtol=1e-10))
    end
end


# =============================================================================
# Layer 2: Input Validation Tests
# =============================================================================

@testset "SACE Input Validation Tests" begin
    @testset "Length mismatch raises error" begin
        Y = randn(100)
        D = Float64.(rand(100) .< 0.5)
        S = Float64.(rand(50) .< 0.8)  # Wrong length

        @test_throws ArgumentError sace_bounds(Y, D, S)
    end

    @testset "Non-binary treatment raises error" begin
        Y = randn(100)
        D = Float64.(rand(0:2, 100))  # Not binary
        S = Float64.(rand(100) .< 0.8)

        @test_throws ArgumentError sace_bounds(Y, D, S)
    end

    @testset "Non-binary survival raises error" begin
        Y = randn(100)
        D = Float64.(rand(100) .< 0.5)
        S = Float64.(rand(0:2, 100))  # Not binary

        @test_throws ArgumentError sace_bounds(Y, D, S)
    end

    @testset "No survivors raises error" begin
        Y = randn(100)
        D = Float64.(rand(100) .< 0.5)
        S = zeros(100)  # All dead

        @test_throws ArgumentError sace_bounds(Y, D, S)
    end
end


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================

@testset "SACE Monte Carlo Tests" begin
    @testset "Bounds cover true SACE under selection monotonicity" begin
        n_sims = 100
        true_sace = 1.5
        covers = 0

        for seed in 1:n_sims
            data = generate_selection_monotonicity_dgp(n=500, true_sace=true_sace, p_AS=0.65, seed=seed)
            result = sace_bounds(data.Y, data.D, data.S, monotonicity="selection")

            if result.lower_bound <= true_sace <= result.upper_bound
                covers += 1
            end
        end

        coverage = covers / n_sims
        @test coverage > 0.85
    end

    @testset "Lee bounds produce valid bounds" begin
        n_sims = 50
        true_sace = 1.5
        valid_count = 0

        for seed in 1:n_sims
            data = generate_sace_dgp(n=500, true_sace=true_sace, p_harmed=0.15, seed=seed)
            result = sace_bounds(data.Y, data.D, data.S, monotonicity="none")

            if result.lower_bound <= result.upper_bound &&
               isfinite(result.lower_bound) && isfinite(result.upper_bound)
                valid_count += 1
            end
        end

        validity = valid_count / n_sims
        @test validity == 1.0
    end
end
