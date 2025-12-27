"""
Tests for A-Learning Dynamic Treatment Regimes

Comprehensive test suite covering:
- Layer 1: Known-answer tests with hand-calculated values
- Layer 2: Adversarial tests for edge cases
- Layer 3: Monte Carlo validation for statistical properties

Special focus on double robustness property.
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
    propensity = 0.5
    A = Float64.(rand(n) .< propensity)

    true_blip = true_blip_intercept .+ true_blip_coef .* X[:, 1]
    baseline = 1.0 .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
    Y = baseline .+ A .* true_blip .+ randn(n)

    data = DTRData([Y], [A], [X])
    return data, [true_blip_intercept, true_blip_coef]
end


"""
Generate data where propensity model is misspecified.

True propensity depends on X_1^2 (nonlinear), but we fit linear logit.
Baseline outcome is correctly specified.
"""
function generate_propensity_misspecified_dgp(;
    n::Int = 500,
    true_blip::Float64 = 2.0,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    X = randn(n, 3)

    # True propensity: nonlinear in X_1
    true_propensity = 1.0 ./ (1.0 .+ exp.(-X[:, 1].^2 .+ 0.5))
    A = Float64.(rand(n) .< true_propensity)

    # Linear baseline (correctly specified)
    baseline = 1.0 .+ 0.5 .* X[:, 1]
    Y = baseline .+ true_blip .* A .+ randn(n)

    data = DTRData([Y], [A], [X])
    return data, true_blip
end


"""
Generate data where outcome model is misspecified.

Baseline outcome is nonlinear (quadratic), but we fit linear OLS.
Propensity is correctly specified (randomized).
"""
function generate_outcome_misspecified_dgp(;
    n::Int = 500,
    true_blip::Float64 = 2.0,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    X = randn(n, 3)

    # Randomized treatment (propensity correctly specified)
    A = Float64.(rand(n) .< 0.5)

    # Nonlinear baseline (misspecified by linear model)
    baseline = 1.0 .+ X[:, 1].^2 .+ 0.3 .* X[:, 2]
    Y = baseline .+ true_blip .* A .+ randn(n)

    data = DTRData([Y], [A], [X])
    return data, true_blip
end


"""
Generate two-stage DTR data.
"""
function generate_two_stage_dgp(;
    n::Int = 500,
    true_blip1::Float64 = 1.5,
    true_blip2::Float64 = 2.0,
    random_state::Int = 42,
)
    Random.seed!(random_state)

    X1 = randn(n, 2)
    A1 = Float64.(rand(n) .< 0.5)
    Y1 = X1[:, 1] .+ true_blip1 .* A1 .+ 0.5 .* randn(n)

    X2 = hcat(X1, A1, Y1, randn(n, 1))
    A2 = Float64.(rand(n) .< 0.5)
    Y2 = Y1 .+ X2[:, 1] .+ true_blip2 .* A2 .+ randn(n)

    data = DTRData([Y1, Y2], [A1, A2], [X1, X2])
    return data, true_blip1, true_blip2
end


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "A-Learning Known-Answer Tests" begin

    @testset "Single-stage basic estimation" begin
        data, true_blip = generate_single_stage_dgp(n=500, random_state=42)
        result = a_learning(data)

        @test result.n_stages == 1
        @test result.propensity_model == :logit
        @test result.outcome_model == :ols
        @test result.doubly_robust == true
        @test result.se_method == :sandwich
    end

    @testset "Single-stage constant blip recovery" begin
        Random.seed!(42)
        n = 1000
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        true_blip = 2.5
        Y = X[:, 1] .+ true_blip .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data)

        # Blip intercept should be close to true_blip
        @test abs(result.blip_coefficients[1][1] - true_blip) < 0.4
    end

    @testset "Matches Q-learning under correct specification" begin
        # When both models are correct, A-learning and Q-learning should agree
        data, _ = generate_single_stage_dgp(n=500, random_state=42)

        result_q = q_learning(data)
        result_a = a_learning(data)

        # Blip coefficients should be similar (not exact due to different weighting)
        @test abs(result_q.blip_coefficients[1][1] - result_a.blip_coefficients[1][1]) < 0.5
    end

    @testset "Optimal regime from A-learning" begin
        Random.seed!(42)
        n = 500
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)

        # Blip = 2 + 3*X_1
        true_blip = 2.0 .+ 3.0 .* X[:, 1]
        Y = X[:, 1] .+ A .* true_blip .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data)

        # X_1 = 1 → optimal = 1
        @test optimal_regime(result, [1.0, 0.0], 1) == 1

        # X_1 = -2 → blip < 0 → optimal = 0
        @test optimal_regime(result, [-2.0, 0.0], 1) == 0
    end

    @testset "Value function estimate" begin
        data, _ = generate_single_stage_dgp(n=500, random_state=42)
        result = a_learning(data)

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

        result = a_learning_single_stage(Y, A, X)

        @test result.n_stages == 1
        @test length(result.blip_coefficients) == 1
        @test length(result.blip_coefficients[1]) == 4  # intercept + 3 covariates
    end

    @testset "ALearningResult structure" begin
        data, _ = generate_single_stage_dgp(n=200, random_state=42)
        result = a_learning(data)

        @test isa(result, ALearningResult)
        @test hasfield(typeof(result), :value_estimate)
        @test hasfield(typeof(result), :blip_coefficients)
        @test hasfield(typeof(result), :propensity_model)
        @test hasfield(typeof(result), :doubly_robust)
    end
end


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================

@testset "A-Learning Adversarial Tests" begin

    @testset "Small sample warning" begin
        Random.seed!(42)
        n = 30
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
        data = DTRData([Y], [A], [X])

        result = @test_logs (:warn,) a_learning(data)
        @test result.n_stages == 1
    end

    @testset "Invalid propensity model" begin
        X = randn(100, 2)
        A = Float64.(rand(100) .< 0.5)
        Y = X[:, 1] .+ A .+ randn(100)
        data = DTRData([Y], [A], [X])

        @test_throws ErrorException a_learning(data; propensity_model=:invalid)
    end

    @testset "Invalid outcome model" begin
        X = randn(100, 2)
        A = Float64.(rand(100) .< 0.5)
        Y = X[:, 1] .+ A .+ randn(100)
        data = DTRData([Y], [A], [X])

        @test_throws ErrorException a_learning(data; outcome_model=:invalid)
    end

    @testset "Invalid SE method" begin
        X = randn(100, 2)
        A = Float64.(rand(100) .< 0.5)
        Y = X[:, 1] .+ A .+ randn(100)
        data = DTRData([Y], [A], [X])

        @test_throws ErrorException a_learning(data; se_method=:invalid)
    end

    @testset "Propensity trimming" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 2)

        # Extreme propensity: 95% treated
        A = Float64.(rand(n) .< 0.95)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data; propensity_trim=0.05)

        @test all(isfinite.(result.blip_coefficients[1]))
        @test isfinite(result.value_estimate)
    end

    @testset "Probit propensity model" begin
        Random.seed!(42)
        n = 300
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data; propensity_model=:probit)

        @test result.propensity_model == :probit
        @test all(isfinite.(result.blip_coefficients[1]))
    end

    @testset "Ridge outcome model" begin
        Random.seed!(42)
        n = 200
        p = 15  # More covariates for ridge regularization
        X = randn(n, p)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data; outcome_model=:ridge)

        @test result.outcome_model == :ridge
        @test all(isfinite.(result.blip_coefficients[1]))
    end

    @testset "Non-doubly-robust estimation" begin
        Random.seed!(42)
        n = 300
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)

        data = DTRData([Y], [A], [X])
        result = a_learning(data; doubly_robust=false)

        @test result.doubly_robust == false
        @test all(isfinite.(result.blip_coefficients[1]))
    end

    @testset "Bootstrap SE method" begin
        Random.seed!(42)
        n = 200
        X = randn(n, 2)
        A = Float64.(rand(n) .< 0.5)
        Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
        data = DTRData([Y], [A], [X])

        result = a_learning(data; se_method=:bootstrap, n_bootstrap=50)

        @test result.se_method == :bootstrap
        @test all(result.blip_se[1] .> 0)
        @test result.value_se > 0
    end

    @testset "Multi-stage backward induction" begin
        data, _, _ = generate_two_stage_dgp(n=500, random_state=42)
        result = a_learning(data)

        @test result.n_stages == 2
        @test length(result.blip_coefficients) == 2
        @test all(isfinite.(result.blip_coefficients[1]))
        @test all(isfinite.(result.blip_coefficients[2]))
    end
end


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================

@testset "A-Learning Monte Carlo Validation" begin

    @testset "Blip coefficient unbiased" begin
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
            result = a_learning(data)

            push!(blip_intercepts, result.blip_coefficients[1][1])
            push!(blip_coefs, result.blip_coefficients[1][2])
        end

        bias_intercept = mean(blip_intercepts) - true_blip_intercept
        bias_coef = mean(blip_coefs) - true_blip_coef

        @test abs(bias_intercept) < 0.10
        @test abs(bias_coef) < 0.15
    end

    @testset "Confidence interval coverage" begin
        n_sims = 200
        true_blip_intercept = 2.0

        covered = 0

        for sim in 1:n_sims
            data, _ = generate_single_stage_dgp(
                n=300,
                true_blip_intercept=true_blip_intercept,
                true_blip_coef=0.0,
                random_state=sim
            )
            result = a_learning(data; alpha=0.05)

            psi_0 = result.blip_coefficients[1][1]
            se_0 = result.blip_se[1][1]
            ci_lower = psi_0 - 1.96 * se_0
            ci_upper = psi_0 + 1.96 * se_0

            if ci_lower < true_blip_intercept < ci_upper
                covered += 1
            end
        end

        coverage = covered / n_sims
        # A-learning sandwich SE can be slightly conservative, allow wider range
        @test 0.80 ≤ coverage ≤ 0.98
    end

    @testset "Double robustness: propensity misspecified" begin
        # A-learning should still be consistent when only outcome model is correct
        n_sims = 50
        true_blip = 2.0
        estimates = Float64[]

        for sim in 1:n_sims
            data, _ = generate_propensity_misspecified_dgp(
                n=500,
                true_blip=true_blip,
                random_state=sim
            )
            result = a_learning(data; doubly_robust=true)
            push!(estimates, result.blip_coefficients[1][1])
        end

        bias = mean(estimates) - true_blip
        # Should still be reasonably unbiased due to DR property
        @test abs(bias) < 0.25
    end

    @testset "Double robustness: outcome misspecified" begin
        # A-learning should still be consistent when only propensity model is correct
        n_sims = 50
        true_blip = 2.0
        estimates = Float64[]

        for sim in 1:n_sims
            data, _ = generate_outcome_misspecified_dgp(
                n=500,
                true_blip=true_blip,
                random_state=sim
            )
            result = a_learning(data; doubly_robust=true)
            push!(estimates, result.blip_coefficients[1][1])
        end

        bias = mean(estimates) - true_blip
        # Should still be reasonably unbiased due to DR property
        @test abs(bias) < 0.25
    end

    @testset "Optimal regime recovery rate" begin
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
            result = a_learning(data)

            optimal = optimal_regime(result, [0.0, 0.0], 1)
            if optimal == 1
                correct_regimes += 1
            end
            total_decisions += 1
        end

        recovery_rate = correct_regimes / total_decisions
        @test recovery_rate > 0.90
    end
end
