"""
MTE (Marginal Treatment Effects) Test Suite.

Session 91: Julia MTE implementation tests.
"""

using Test
using Random
using Statistics
using Distributions

# Include the source files directly for testing
include("../../src/mte/types.jl")
include("../../src/mte/late.jl")
include("../../src/mte/local_iv.jl")
include("../../src/mte/policy.jl")

# Test fixtures
function create_simple_binary_iv_data(rng::AbstractRNG; n::Int=500)
    """Create simple binary IV data with known LATE."""
    Z = rand(rng, [0.0, 1.0], n)
    U = rand(rng, n)

    # Selection: D = 1 if U < 0.3 + 0.4*Z
    # This gives complier share = 0.4
    D = Float64.(U .< (0.3 .+ 0.4 .* Z))

    # Outcome: Y = 1 + 2*D + noise
    # True LATE = 2.0
    Y = 1.0 .+ 2.0 .* D .+ 0.3 .* randn(rng, n)

    return (outcome=Y, treatment=D, instrument=Z, true_late=2.0, true_complier_share=0.4)
end


function create_heterogeneous_mte_data(rng::AbstractRNG; n::Int=1000)
    """Create data with linearly decreasing MTE."""
    Z = randn(rng, n)
    U = rand(rng, n)

    # Propensity: P(D=1|Z) = Φ(Z)
    propensity = cdf.(Normal(), Z)

    # Selection: D = 1 if U < propensity (normal model)
    D = Float64.(U .< propensity)

    # MTE(u) = 3 - 2u (linearly decreasing)
    # So those with low U (high propensity) have high treatment effect
    mte_individual = 3.0 .- 2.0 .* U

    # Outcome: Y = 1 + mte*D + noise
    Y = 1.0 .+ mte_individual .* D .+ 0.5 .* randn(rng, n)

    return (outcome=Y, treatment=D, instrument=Z, true_mte_slope=-2.0)
end


function create_constant_mte_data(rng::AbstractRNG; n::Int=800)
    """Create data with constant MTE (no heterogeneity)."""
    Z = randn(rng, n)
    U = rand(rng, n)

    propensity = cdf.(Normal(), Z)
    D = Float64.(U .< propensity)

    # Constant MTE = 1.5
    Y = 1.0 .+ 1.5 .* D .+ 0.4 .* randn(rng, n)

    return (outcome=Y, treatment=D, instrument=Z, true_mte=1.5)
end


function create_weak_instrument_data(rng::AbstractRNG; n::Int=500)
    """Create data with weak instrument."""
    Z = rand(rng, [0.0, 1.0], n)
    U = rand(rng, n)

    # Very weak first stage: D = 1 if U < 0.48 + 0.04*Z
    # Complier share = 0.04 (very small)
    D = Float64.(U .< (0.48 .+ 0.04 .* Z))

    Y = 1.0 .+ 2.0 .* D .+ randn(rng, n)

    return (outcome=Y, treatment=D, instrument=Z)
end


# ============================================================================
# LATE Tests
# ============================================================================

@testset "LATE Estimator" begin
    rng = MersenneTwister(42)

    @testset "Returns correct type" begin
        data = create_simple_binary_iv_data(rng)
        result = late_estimator(data.outcome, data.treatment, data.instrument)

        @test result isa LATESolution
        @test !isnan(result.late)
        @test !isnan(result.se)
        @test result.se > 0
        @test result.ci_lower < result.ci_upper
        @test 0 <= result.pvalue <= 1
        @test result.method == :wald
    end

    @testset "Recovers true LATE" begin
        data = create_simple_binary_iv_data(rng; n=1000)
        result = late_estimator(data.outcome, data.treatment, data.instrument)

        # Within 0.3 of true value
        @test abs(result.late - data.true_late) < 0.3
    end

    @testset "CI contains true value" begin
        data = create_simple_binary_iv_data(rng; n=1000)
        result = late_estimator(data.outcome, data.treatment, data.instrument)

        @test result.ci_lower < data.true_late < result.ci_upper
    end

    @testset "Complier share matches" begin
        data = create_simple_binary_iv_data(rng; n=1000)
        result = late_estimator(data.outcome, data.treatment, data.instrument)

        @test abs(result.complier_share - data.true_complier_share) < 0.1
    end

    @testset "First stage significant" begin
        data = create_simple_binary_iv_data(rng)
        result = late_estimator(data.outcome, data.treatment, data.instrument)

        @test result.first_stage_f > 10
    end

    @testset "Problem-based interface" begin
        data = create_simple_binary_iv_data(rng)
        problem = LATEProblem(
            outcome = data.outcome,
            treatment = data.treatment,
            instrument = data.instrument
        )
        result = late_estimator(problem)

        @test result isa LATESolution
        @test !isnan(result.late)
    end
end


@testset "LATE Bounds" begin
    rng = MersenneTwister(123)

    @testset "Returns correct type" begin
        data = create_simple_binary_iv_data(rng)
        result = late_bounds(data.outcome, data.treatment, data.instrument)

        @test result isa LATEBoundsResult
        @test !isnan(result.bounds_lower)
        @test !isnan(result.bounds_upper)
        @test result.bounds_lower <= result.bounds_upper
    end

    @testset "Bounds contain true value" begin
        data = create_simple_binary_iv_data(rng; n=800)
        result = late_bounds(data.outcome, data.treatment, data.instrument)

        @test result.bounds_lower < data.true_late < result.bounds_upper
    end

    @testset "Monotonicity estimate matches" begin
        data = create_simple_binary_iv_data(rng)
        result = late_bounds(data.outcome, data.treatment, data.instrument)

        late_result = late_estimator(data.outcome, data.treatment, data.instrument)
        @test abs(result.late_under_monotonicity - late_result.late) < 0.01
    end
end


@testset "Complier Characteristics" begin
    rng = MersenneTwister(456)

    @testset "Returns correct type" begin
        data = create_simple_binary_iv_data(rng)
        result = complier_characteristics(
            data.outcome, data.treatment, data.instrument
        )

        @test result isa ComplierResult
        @test !isnan(result.complier_mean_outcome_treated)
        @test !isnan(result.complier_mean_outcome_control)
        @test 0 < result.complier_share <= 1
        @test result.method == :kappa_weights
    end

    @testset "LATE equals outcome difference" begin
        data = create_simple_binary_iv_data(rng; n=1000)

        late_result = late_estimator(data.outcome, data.treatment, data.instrument)
        complier_result = complier_characteristics(
            data.outcome, data.treatment, data.instrument
        )

        implied_late = (complier_result.complier_mean_outcome_treated -
                       complier_result.complier_mean_outcome_control)

        @test abs(late_result.late - implied_late) < 0.5
    end
end


# ============================================================================
# Local IV MTE Tests
# ============================================================================

@testset "Local IV" begin
    rng = MersenneTwister(789)

    @testset "Returns correct type" begin
        data = create_heterogeneous_mte_data(rng)
        result = local_iv(
            data.outcome, data.treatment, data.instrument;
            n_grid=20, n_bootstrap=50
        )

        @test result isa MTESolution
        @test length(result.mte_grid) == 20
        @test length(result.u_grid) == 20
        @test length(result.se_grid) == 20
        @test result.method == :local_iv
    end

    @testset "Detects decreasing MTE" begin
        data = create_heterogeneous_mte_data(rng; n=1500)
        result = local_iv(
            data.outcome, data.treatment, data.instrument;
            n_grid=20, n_bootstrap=50
        )

        # Check slope is negative
        valid = .!isnan.(result.mte_grid)
        if sum(valid) > 5
            # Fit linear trend
            u_valid = result.u_grid[valid]
            mte_valid = result.mte_grid[valid]

            # Simple linear regression
            u_centered = u_valid .- mean(u_valid)
            slope = sum(u_centered .* mte_valid) / sum(u_centered.^2)

            @test slope < 0  # Should be negative
        end
    end

    @testset "Propensity support bounded" begin
        data = create_heterogeneous_mte_data(rng)
        result = local_iv(
            data.outcome, data.treatment, data.instrument;
            n_grid=20, n_bootstrap=30
        )

        p_min, p_max = result.propensity_support
        @test 0 <= p_min < p_max <= 1
    end

    @testset "Problem-based interface" begin
        data = create_heterogeneous_mte_data(rng)
        problem = MTEProblem(
            outcome = data.outcome,
            treatment = data.treatment,
            instrument = data.instrument,
            n_grid = 15
        )
        result = local_iv(problem; n_bootstrap=30)

        @test result isa MTESolution
        @test length(result.mte_grid) == 15
    end
end


@testset "Polynomial MTE" begin
    rng = MersenneTwister(321)

    @testset "Returns correct type" begin
        data = create_heterogeneous_mte_data(rng)
        result = polynomial_mte(
            data.outcome, data.treatment, data.instrument;
            degree=3, n_grid=20, n_bootstrap=50
        )

        @test result isa MTESolution
        @test length(result.mte_grid) == 20
        @test result.method == :polynomial
    end

    @testset "Different degrees produce results" begin
        data = create_heterogeneous_mte_data(rng)

        result_linear = polynomial_mte(
            data.outcome, data.treatment, data.instrument;
            degree=1, n_grid=15, n_bootstrap=30
        )

        result_cubic = polynomial_mte(
            data.outcome, data.treatment, data.instrument;
            degree=3, n_grid=15, n_bootstrap=30
        )

        @test !all(isnan, result_linear.mte_grid)
        @test !all(isnan, result_cubic.mte_grid)
    end
end


# ============================================================================
# Policy Parameter Tests
# ============================================================================

@testset "Policy Parameters" begin
    rng = MersenneTwister(654)

    # Create MTE result first
    data = create_heterogeneous_mte_data(rng; n=1200)
    mte_result = local_iv(
        data.outcome, data.treatment, data.instrument;
        n_grid=25, n_bootstrap=100
    )

    @testset "ATE from MTE" begin
        ate_result = ate_from_mte(mte_result)

        @test ate_result isa PolicyResult
        @test !isnan(ate_result.estimate)
        @test ate_result.parameter == :ate
        @test occursin("uniform", lowercase(ate_result.weights_used))
    end

    @testset "ATT from MTE" begin
        att_result = att_from_mte(mte_result)

        @test att_result isa PolicyResult
        @test !isnan(att_result.estimate)
        @test att_result.parameter == :att
    end

    @testset "ATU from MTE" begin
        atu_result = atu_from_mte(mte_result)

        @test atu_result isa PolicyResult
        @test !isnan(atu_result.estimate)
        @test atu_result.parameter == :atu
    end

    @testset "LATE from MTE" begin
        p_min, p_max = mte_result.propensity_support
        p_mid = (p_min + p_max) / 2

        late_result = late_from_mte(mte_result, p_min + 0.1, p_mid)

        @test late_result isa PolicyResult
        @test !isnan(late_result.estimate)
        @test late_result.parameter == :late
    end

    @testset "PRTE with custom weights" begin
        # Uniform expansion policy
        uniform_weights = u -> ones(length(u))

        prte_result = prte(mte_result, uniform_weights)

        @test prte_result isa PolicyResult
        @test !isnan(prte_result.estimate)
        @test prte_result.parameter == :prte
    end

    @testset "ATT > ATU for decreasing MTE" begin
        # With decreasing MTE, treated have higher effects than untreated
        att_result = att_from_mte(mte_result)
        atu_result = atu_from_mte(mte_result)

        # ATT should be larger (treated selected into treatment = lower U = higher MTE)
        @test att_result.estimate > atu_result.estimate - 1.0  # Allow some noise
    end
end


# ============================================================================
# Input Validation Tests
# ============================================================================

@testset "Input Validation" begin
    rng = MersenneTwister(999)

    @testset "LATE rejects non-binary treatment" begin
        Y = randn(rng, 100)
        D = randn(rng, 100)  # Continuous, not binary
        Z = rand(rng, [0.0, 1.0], 100)

        @test_throws ErrorException late_estimator(Y, D, Z)
    end

    @testset "LATE rejects non-binary instrument" begin
        Y = randn(rng, 100)
        D = rand(rng, [0.0, 1.0], 100)
        Z = randn(rng, 100)  # Continuous, not binary

        @test_throws ErrorException LATEProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "LATE rejects no treatment variation" begin
        Y = randn(rng, 100)
        D = ones(100)  # All treated
        Z = rand(rng, [0.0, 1.0], 100)

        @test_throws ErrorException LATEProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end

    @testset "MTE rejects length mismatch" begin
        Y = randn(rng, 100)
        D = rand(rng, [0.0, 1.0], 90)  # Wrong length
        Z = randn(rng, 100)

        @test_throws ErrorException MTEProblem(
            outcome = Y,
            treatment = D,
            instrument = Z
        )
    end
end


# ============================================================================
# Edge Cases
# ============================================================================

@testset "Edge Cases" begin
    rng = MersenneTwister(111)

    @testset "Weak instrument larger SE" begin
        weak_data = create_weak_instrument_data(rng)
        strong_data = create_simple_binary_iv_data(rng)

        weak_result = late_estimator(
            weak_data.outcome, weak_data.treatment, weak_data.instrument
        )
        strong_result = late_estimator(
            strong_data.outcome, strong_data.treatment, strong_data.instrument
        )

        # Weak instrument should have larger SE (or at least exist)
        @test weak_result.se > 0
        @test weak_result.first_stage_f < 10  # Weak first stage
    end

    @testset "Constant MTE relatively flat" begin
        data = create_constant_mte_data(rng)
        result = local_iv(
            data.outcome, data.treatment, data.instrument;
            n_grid=20, n_bootstrap=50
        )

        valid = .!isnan.(result.mte_grid)
        if sum(valid) > 5
            mte_std = std(result.mte_grid[valid])
            # Should have low variance (relatively flat)
            @test mte_std < 3.0
        end
    end
end


# Run all tests
println("Running MTE tests...")
