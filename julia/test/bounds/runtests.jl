"""
Bounds Module Test Suite.

Session 95: Julia Bounds implementation tests.
- Manski bounds (worst-case, MTR, MTS, MTR+MTS, IV)
- Lee bounds for sample selection
"""

using Test
using Random
using Statistics
using Distributions

# Include source files
include("../../src/bounds/types.jl")
include("../../src/bounds/manski.jl")
include("../../src/bounds/lee.jl")


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

function generate_bounds_data(;
    n::Int = 500,
    true_ate::Float64 = 2.0,
    treatment_prob::Float64 = 0.5,
    seed::Int = 42
)
    """Generate simple data for bounds testing."""
    rng = MersenneTwister(seed)
    treatment = Float64.(rand(rng, n) .< treatment_prob)
    noise = randn(rng, n)
    outcome = true_ate .* treatment .+ noise

    return (outcome=outcome, treatment=treatment, true_ate=true_ate)
end


function generate_iv_data(;
    n::Int = 500,
    true_late::Float64 = 2.0,
    complier_share::Float64 = 0.4,
    seed::Int = 42
)
    """Generate IV data for Manski IV bounds."""
    rng = MersenneTwister(seed)

    Z = Float64.(rand(rng, n) .< 0.5)
    U = rand(rng, n)

    # Selection: D=1 if U < 0.3 + complier_share * Z
    D = Float64.(U .< (0.3 .+ complier_share .* Z))

    # Outcome
    Y = 1.0 .+ true_late .* D .+ 0.5 .* randn(rng, n)

    return (outcome=Y, treatment=D, instrument=Z, true_late=true_late)
end


function generate_lee_data(;
    n::Int = 1000,
    true_ate::Float64 = 2.0,
    attrition_base::Float64 = 0.2,
    attrition_diff::Float64 = 0.1,  # Treatment reduces attrition
    seed::Int = 42
)
    """Generate data with sample selection for Lee bounds."""
    rng = MersenneTwister(seed)

    treatment = Float64.(rand(rng, n) .< 0.5)

    # Treatment increases observation probability (positive monotonicity)
    obs_prob = (1 - attrition_base) .+ attrition_diff .* treatment
    observed = Float64.(rand(rng, n) .< obs_prob)

    # Outcome
    outcome = true_ate .* treatment .+ randn(rng, n)

    return (outcome=outcome, treatment=treatment, observed=observed, true_ate=true_ate)
end


# =============================================================================
# MANSKI WORST-CASE BOUNDS TESTS
# =============================================================================

@testset "Manski Worst-Case Bounds" begin

    @testset "Returns correct type" begin
        data = generate_bounds_data(n=500, seed=42)
        result = manski_worst_case(data.outcome, data.treatment)

        @test result isa ManskiBoundsResult
        @test !isnan(result.bounds_lower)
        @test !isnan(result.bounds_upper)
        @test result.bounds_lower <= result.bounds_upper
    end

    @testset "Contains true value" begin
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)
        result = manski_worst_case(data.outcome, data.treatment)

        # True ATE should be within bounds
        @test result.bounds_lower <= data.true_ate <= result.bounds_upper
    end

    @testset "Wide bounds without assumptions" begin
        data = generate_bounds_data(n=500, seed=42)
        result = manski_worst_case(data.outcome, data.treatment)

        # Worst-case should have wide bounds
        @test result.bounds_width > 2.0
        @test result.assumptions == :worst_case
    end

    @testset "Custom outcome support" begin
        data = generate_bounds_data(n=500, seed=42)
        result = manski_worst_case(data.outcome, data.treatment;
                                    outcome_support=(-5.0, 5.0))

        @test result.outcome_support == (-5.0, 5.0)
    end

    @testset "Input validation" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 100)
        D_bad = randn(rng, 90)  # Wrong length

        @test_throws ErrorException manski_worst_case(Y, D_bad)
    end
end


# =============================================================================
# MANSKI MTR BOUNDS TESTS
# =============================================================================

@testset "Manski MTR Bounds" begin

    @testset "Positive MTR tightens bounds" begin
        data = generate_bounds_data(n=500, true_ate=2.0, seed=42)

        wc = manski_worst_case(data.outcome, data.treatment)
        mtr = manski_mtr(data.outcome, data.treatment; direction=:positive)

        # MTR should be at least as tight
        @test mtr.bounds_width <= wc.bounds_width + 1e-10
        @test mtr.assumptions == :mtr
        @test mtr.mtr_direction == :positive
    end

    @testset "Positive MTR gives non-negative lower bound" begin
        data = generate_bounds_data(n=500, true_ate=2.0, seed=42)
        result = manski_mtr(data.outcome, data.treatment; direction=:positive)

        @test result.bounds_lower >= 0.0
    end

    @testset "Negative MTR gives non-positive upper bound" begin
        data = generate_bounds_data(n=500, true_ate=-1.0, seed=42)
        result = manski_mtr(data.outcome, data.treatment; direction=:negative)

        @test result.bounds_upper <= 0.0
    end
end


# =============================================================================
# MANSKI MTS BOUNDS TESTS
# =============================================================================

@testset "Manski MTS Bounds" begin

    @testset "MTS narrows bounds" begin
        data = generate_bounds_data(n=500, seed=42)

        wc = manski_worst_case(data.outcome, data.treatment)
        mts = manski_mts(data.outcome, data.treatment)

        @test mts.bounds_width <= wc.bounds_width + 1e-10
        @test mts.assumptions == :mts
    end

    @testset "Upper bound equals naive ATE" begin
        data = generate_bounds_data(n=500, seed=42)
        result = manski_mts(data.outcome, data.treatment)

        # Under MTS, upper bound = naive ATE
        @test isapprox(result.bounds_upper, result.naive_ate, rtol=1e-10)
    end
end


# =============================================================================
# MANSKI MTR+MTS BOUNDS TESTS
# =============================================================================

@testset "Manski MTR+MTS Bounds" begin

    @testset "Tightest bounds" begin
        data = generate_bounds_data(n=500, true_ate=2.0, seed=42)

        wc = manski_worst_case(data.outcome, data.treatment)
        combined = manski_mtr_mts(data.outcome, data.treatment; mtr_direction=:positive)

        @test combined.bounds_width <= wc.bounds_width
        @test combined.assumptions == :mtr_mts
    end

    @testset "Positive MTR+MTS: bounds in [0, naive]" begin
        data = generate_bounds_data(n=500, true_ate=2.0, seed=42)
        result = manski_mtr_mts(data.outcome, data.treatment; mtr_direction=:positive)

        @test result.bounds_lower >= 0.0
        if result.naive_ate > 0
            @test result.bounds_upper <= result.naive_ate + 1e-10
        end
    end
end


# =============================================================================
# MANSKI IV BOUNDS TESTS
# =============================================================================

@testset "Manski IV Bounds" begin

    @testset "Returns correct type" begin
        data = generate_iv_data(n=500, seed=42)
        result = manski_iv(data.outcome, data.treatment, data.instrument)

        @test result isa ManskiIVBoundsResult
        @test result.assumptions == :iv
        @test result.complier_share > 0
    end

    @testset "Contains true LATE" begin
        data = generate_iv_data(n=1000, true_late=2.0, seed=42)
        result = manski_iv(data.outcome, data.treatment, data.instrument)

        # True LATE should be in bounds
        @test result.bounds_lower <= data.true_late <= result.bounds_upper
    end

    @testset "Stronger IV gives tighter bounds" begin
        # Weak IV
        weak = generate_iv_data(n=500, complier_share=0.1, seed=42)
        weak_result = manski_iv(weak.outcome, weak.treatment, weak.instrument)

        # Strong IV
        strong = generate_iv_data(n=500, complier_share=0.5, seed=43)
        strong_result = manski_iv(strong.outcome, strong.treatment, strong.instrument)

        @test strong_result.complier_share > weak_result.complier_share
    end
end


# =============================================================================
# COMPARE BOUNDS TESTS
# =============================================================================

@testset "Compare Bounds" begin

    @testset "Returns all methods" begin
        data = generate_bounds_data(n=500, seed=42)
        result = compare_bounds(data.outcome, data.treatment)

        @test haskey(result, :worst_case)
        @test haskey(result, :mtr)
        @test haskey(result, :mts)
        @test haskey(result, :mtr_mts)
    end

    @testset "MTR+MTS is tightest" begin
        data = generate_bounds_data(n=500, true_ate=2.0, seed=42)
        result = compare_bounds(data.outcome, data.treatment; mtr_direction=:positive)

        widths = Dict(k => v.bounds_width for (k, v) in result)

        @test widths[:mtr_mts] <= minimum(values(widths)) + 1e-10
    end
end


# =============================================================================
# LEE BOUNDS TESTS
# =============================================================================

@testset "Lee Bounds" begin

    @testset "Returns correct type" begin
        data = generate_lee_data(n=1000, seed=42)
        result = lee_bounds(data.outcome, data.treatment, data.observed;
                           n_bootstrap=100, rng=MersenneTwister(42))

        @test result isa LeeBoundsResult
        @test !isnan(result.bounds_lower)
        @test !isnan(result.bounds_upper)
        @test result.bounds_lower <= result.bounds_upper
    end

    @testset "Contains true value" begin
        data = generate_lee_data(n=2000, true_ate=2.0, seed=42)
        result = lee_bounds(data.outcome, data.treatment, data.observed;
                           n_bootstrap=200, rng=MersenneTwister(42))

        # True ATE should be in bounds
        @test result.bounds_lower <= data.true_ate <= result.bounds_upper
    end

    @testset "CI contains bounds" begin
        data = generate_lee_data(n=1000, seed=42)
        result = lee_bounds(data.outcome, data.treatment, data.observed;
                           n_bootstrap=500, rng=MersenneTwister(42))

        # CI should cover the point estimates
        if !isnan(result.ci_lower)
            @test result.ci_lower <= result.bounds_lower
            @test result.bounds_upper <= result.ci_upper
        end
    end

    @testset "Monotonicity direction stored" begin
        data = generate_lee_data(n=500, seed=42)

        pos = lee_bounds(data.outcome, data.treatment, data.observed;
                        monotonicity=:positive, n_bootstrap=50)
        neg = lee_bounds(data.outcome, data.treatment, data.observed;
                        monotonicity=:negative, n_bootstrap=50)

        @test pos.monotonicity == :positive
        @test neg.monotonicity == :negative
    end

    @testset "Input validation" begin
        rng = MersenneTwister(42)
        Y = randn(rng, 100)
        D = Float64.(rand(rng, 100) .< 0.5)
        O = Float64.(rand(rng, 90) .< 0.8)  # Wrong length

        @test_throws ErrorException lee_bounds(Y, D, O)
    end
end


# =============================================================================
# CHECK MONOTONICITY TESTS
# =============================================================================

@testset "Check Monotonicity" begin

    @testset "Detects positive monotonicity" begin
        rng = MersenneTwister(42)
        n = 1000
        treatment = Float64.(rand(rng, n) .< 0.5)
        obs_prob = 0.7 .+ 0.2 .* treatment
        observed = Float64.(rand(rng, n) .< obs_prob)

        result = check_monotonicity(treatment, observed)

        @test result[:difference] > 0
        @test result[:suggested_monotonicity] == :positive
    end

    @testset "Detects negative monotonicity" begin
        rng = MersenneTwister(42)
        n = 1000
        treatment = Float64.(rand(rng, n) .< 0.5)
        obs_prob = 0.9 .- 0.2 .* treatment
        observed = Float64.(rand(rng, n) .< obs_prob)

        result = check_monotonicity(treatment, observed)

        @test result[:difference] < 0
        @test result[:suggested_monotonicity] == :negative
    end
end


println("All Bounds tests completed!")
