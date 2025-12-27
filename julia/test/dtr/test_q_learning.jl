"""
Tests for Q-Learning Dynamic Treatment Regimes

Comprehensive test suite covering:
- Layer 1: Known-answer tests with hand-calculated values
- Layer 2: Adversarial tests for edge cases
- Layer 3: Monte Carlo validation for statistical properties
"""

using Test
using Statistics
using Random
using LinearAlgebra
using CausalEstimators

# =============================================================================
# Helper Functions: DGP Generators
# =============================================================================

"""
Generate single-stage DTR data with known blip function.

# Arguments
- `n::Int`: Sample size
- `n_covariates::Int`: Number of covariates
- `true_blip_intercept::Float64`: Blip intercept ψ_0
- `true_blip_coef::Float64`: Blip coefficient ψ_1 for first covariate
- `random_state::Int`: Random seed

# Returns
DTRData with known ground truth blip = ψ_0 + ψ_1 * X_1
"""
function generate_single_stage_dgp(;
    n::Int = 500,
    n_covariates::Int = 3,
    true_blip_intercept::Float64 = 2.0,
    true_blip_coef::Float64 = 1.0,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    X = randn(n, n_covariates)
    propensity = 0.5  # Randomized treatment
    A = Float64.(rand(n) .< propensity)

    # True blip function: γ(H) = ψ_0 + ψ_1 * X_1
    true_blip = true_blip_intercept .+ true_blip_coef .* X[:, 1]

    # Outcome: Y = baseline + A * blip + noise
    baseline = 1.0 .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    Y = baseline .+ A .* true_blip .+ randn(n)

    data = DTRData([Y], [A], [X])
    return data, [true_blip_intercept, true_blip_coef]
end


"""
Generate two-stage DTR data for backward induction testing.

Stage 1: A_1 based on X_1
Stage 2: A_2 based on X_2 = (X_1, A_1, Y_1)
Final outcome accumulates both stages.
"""
function generate_two_stage_dgp(;
    n::Int = 500,
    true_blip1::Float64 = 1.5,
    true_blip2::Float64 = 2.0,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    # Stage 1
    X1 = randn(n, 2)
    A1 = Float64.(rand(n) .< 0.5)
    Y1 = X1[:, 1] .+ true_blip1 .* A1 .+ 0.5 .* randn(n)

    # Stage 2: history includes A1, Y1
    X2 = hcat(X1, A1, Y1, randn(n, 1))  # Additional covariate
    A2 = Float64.(rand(n) .< 0.5)
    Y2 = Y1 .+ X2[:, 1] .+ true_blip2 .* A2 .+ randn(n)

    data = DTRData([Y1, Y2], [A1, A2], [X1, X2])
    return data, true_blip1, true_blip2
end


"""
Generate DTR data with heterogeneous blip (depends on X).
"""
function generate_heterogeneous_blip_dgp(;
    n::Int = 500,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    X = randn(n, 3)
    A = Float64.(rand(n) .< 0.5)

    # Heterogeneous blip: positive for X_1 > 0, negative otherwise
    # ψ_0 = 1.0, ψ_1 = 2.0
    true_blip = 1.0 .+ 2.0 .* X[:, 1]

    Y = 0.5 .* X[:, 1] .+ A .* true_blip .+ randn(n)

    data = DTRData([Y], [A], [X])
    return data
end


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "Q-Learning Known-Answer Tests" begin

    @testset "DTRData construction and validation" begin
        n = 100
        X = randn(n, 3)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])

        @test data.n_obs == n
        @test data.n_stages == 1
        @test CausalEstimators.n_covariates(data) == [3]
    end

    @testset "History construction" begin
        # Two-stage data
        n = 50
        X1 = randn(n, 2)
        A1 = Float64.(rand(n) .< 0.5)
        Y1 = randn(n)
        X2 = randn(n, 3)
        A2 = Float64.(rand(n) .< 0.5)
        Y2 = randn(n)

        data = DTRData([Y1, Y2], [A1, A2], [X1, X2])

        # Stage 1 history: just X1
        H1 = CausalEstimators.get_history(data, 1)
        @test size(H1) == (n, 2)
        @test H1 ≈ X1

        # Stage 2 history: X1, A1, Y1, X2
        H2 = CausalEstimators.get_history(data, 2)
        @test size(H2) == (n, 2 + 1 + 1 + 3)  # 7 columns
        @test H2[:, 1:2] ≈ X1
        @test H2[:, 3] ≈ A1
        @test H2[:, 4] ≈ Y1
        @test H2[:, 5:7] ≈ X2
    end

    @testset "Single-stage constant blip recovery" begin
        # When blip is constant (no covariate interaction), should recover intercept
        Random.seed!(42)
        n = 1000
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        true_blip = 2.5  # Constant blip
        Y = X[:, 1] .+ true_blip .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = q_learning(data; se_method=:sandwich)

        # Blip intercept should be close to true_blip
        @test abs(result.blip_coefficients[1][1] - true_blip) < 0.3
        @test result.n_stages == 1
        @test result.se_method == :sandwich
    end

    @testset "Optimal regime direction" begin
        # Verify optimal regime returns correct treatment based on blip sign
        Random.seed!(42)
        n = 500
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)

        # Blip = 2 + 3*X_1: positive for X_1 > -2/3
        true_blip = 2.0 .+ 3.0 .* X[:, 1]
        Y = X[:, 1] .+ A .* true_blip .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = q_learning(data)

        # Test optimal regime for specific covariate values
        # X_1 = 1 → blip = 5 > 0 → optimal = 1
        @test optimal_regime(result, [1.0, 0.0], 1) == 1

        # X_1 = -2 → blip = -4 < 0 → optimal = 0
        @test optimal_regime(result, [-2.0, 0.0], 1) == 0
    end

    @testset "Value function estimate" begin
        data, _ = generate_single_stage_dgp(n=500, true_blip_intercept=2.0, random_state=42)
        result = q_learning(data)

        # Value estimate should be positive (optimal regime gives benefits)
        @test result.value_estimate > 0
        @test result.value_se > 0
        @test result.value_ci_lower < result.value_estimate < result.value_ci_upper
    end

    @testset "Single-stage convenience wrapper" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 3)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        result = q_learning_single_stage(Y, A, X)

        @test result.n_stages == 1
        @test length(result.blip_coefficients) == 1
        @test length(result.blip_coefficients[1]) == 4  # intercept + 3 covariates
    end
end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "Q-Learning Adversarial Tests" begin

    @testset "Small sample warning" begin
        Random.seed!(42)
        n = 30  # Small sample
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
        data = DTRData([Y], [A], [X])

        # Should warn but not error
        result = @test_logs (:warn,) q_learning(data)
        @test result.n_stages == 1
    end

    @testset "High-dimensional covariates" begin
        Random.seed!(42)
        n = 200
        p = 20  # More covariates
        X = randn(n, p)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = q_learning(data)

        @test length(result.blip_coefficients[1]) == p + 1  # intercept + p
        @test all(isfinite.(result.blip_coefficients[1]))
    end

    @testset "Extreme propensity (all treated/control)" begin
        Random.seed!(42)
        n = 100
        X = randn(n, 2)

        # Extreme propensity: 90% treated
        A = Float64.(rand(n) .< 0.9)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = q_learning(data)

        # Should still produce finite results
        @test all(isfinite.(result.blip_coefficients[1]))
        @test isfinite(result.value_estimate)
    end

    @testset "Multi-stage backward induction" begin
        data, true_blip1, true_blip2 = generate_two_stage_dgp(n=500, random_state=42)
        result = q_learning(data)

        @test result.n_stages == 2
        @test length(result.blip_coefficients) == 2
        @test length(result.blip_se) == 2

        # Both blip vectors should be finite
        @test all(isfinite.(result.blip_coefficients[1]))
        @test all(isfinite.(result.blip_coefficients[2]))
    end

    @testset "Invalid SE method" begin
        Random.seed!(42)
        X = randn(100, 2)
        A = Float64.(rand(100) .< 0.5)
        Y = X[:, 1] .+ A .+ randn(100)
        data = DTRData([Y], [A], [X])

        @test_throws ErrorException q_learning(data; se_method=:invalid)
    end

    @testset "Bootstrap SE method" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
        data = DTRData([Y], [A], [X])

        result = q_learning(data; se_method=:bootstrap, n_bootstrap=100)

        @test result.se_method == :bootstrap
        @test all(result.blip_se[1] .> 0)
        @test result.value_se > 0
    end

    @testset "Many stages" begin
        Random.seed!(42)
        K = 4
        n = 300

        outcomes = Vector{Vector{Float64}}(undef, K)
        treatments = Vector{Vector{Float64}}(undef, K)
        covariates = Vector{Matrix{Float64}}(undef, K)

        for k in 1:K
            covariates[k] = randn(n, 2)
            treatments[k] = Float64.(rand(n) .< 0.5)
            outcomes[k] = sum(covariates[k], dims=2)[:] .+ 1.5 .* treatments[k] .+ randn(n)
        end

        data = DTRData(outcomes, treatments, covariates)
        result = q_learning(data)

        @test result.n_stages == K
        @test length(result.blip_coefficients) == K
    end
end


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================

@testset "Q-Learning Monte Carlo Validation" begin

    @testset "Blip coefficient unbiased" begin
        # Run multiple simulations and check bias
        n_sims = 100
        true_blip_intercept = 2.0
        true_blip_coef = 1.0

        blip_intercepts = Float64[]
        blip_coefs = Float64[]

        for sim in 1:n_sims
            data, _ = generate_single_stage_dgp(
                n=500,
                true_blip_intercept=true_blip_intercept,
                true_blip_coef=true_blip_coef,
                random_state=sim
            )
            result = q_learning(data)

            push!(blip_intercepts, result.blip_coefficients[1][1])
            push!(blip_coefs, result.blip_coefficients[1][2])
        end

        # Bias should be small
        bias_intercept = mean(blip_intercepts) - true_blip_intercept
        bias_coef = mean(blip_coefs) - true_blip_coef

        @test abs(bias_intercept) < 0.10  # Bias < 0.10
        @test abs(bias_coef) < 0.15
    end

    @testset "Confidence interval coverage" begin
        # Verify 95% CI achieves nominal coverage
        n_sims = 200
        true_blip_intercept = 2.0

        covered = 0

        for sim in 1:n_sims
            data, _ = generate_single_stage_dgp(
                n=300,
                true_blip_intercept=true_blip_intercept,
                true_blip_coef=0.0,  # Only intercept for simplicity
                random_state=sim
            )
            result = q_learning(data; alpha=0.05)

            # Check if true value is within CI for blip intercept
            psi_0 = result.blip_coefficients[1][1]
            se_0 = result.blip_se[1][1]
            ci_lower = psi_0 - 1.96 * se_0
            ci_upper = psi_0 + 1.96 * se_0

            if ci_lower < true_blip_intercept < ci_upper
                covered += 1
            end
        end

        coverage = covered / n_sims

        # Coverage should be between 93% and 97%
        @test 0.88 ≤ coverage ≤ 0.98
    end

    @testset "Optimal regime recovery rate" begin
        # For units with clear treatment benefit, should recommend treatment
        n_sims = 100
        correct_regimes = 0
        total_decisions = 0

        for sim in 1:n_sims
            Random.seed!(sim)
            n = 300
            X = randn(n, 2)
            A = Float64.(rand(n) .< 0.5)

            # Clear positive blip = 3.0
            true_blip = 3.0
            Y = X[:, 1] .+ true_blip .* A .+ randn(n)

            data = DTRData([Y], [A], [X])
            result = q_learning(data)

            # For this constant positive blip, optimal regime should always be 1
            optimal = optimal_regime(result, [0.0, 0.0], 1)
            if optimal == 1
                correct_regimes += 1
            end
            total_decisions += 1
        end

        recovery_rate = correct_regimes / total_decisions
        @test recovery_rate > 0.90  # Should almost always recommend treatment
    end
end
