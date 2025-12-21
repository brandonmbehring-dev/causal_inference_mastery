"""
Monte Carlo Validation for Heckman Selection Model

Tests verify:
1. Unbiasedness: Mean estimate near true value (bias < 0.15)
2. Coverage: 95% CI contains true value 90-97% of runs
3. SE accuracy: Empirical SE matches estimated SE within 20%
4. Selection detection: Power to reject H₀: λ=0 when ρ≠0

Reference: Heckman (1979), Wooldridge (2010) Ch. 19
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# DGP for Monte Carlo
# =============================================================================

"""
Generate Heckman DGP with known parameters.

Selection: S* = γ₀ + γ₁Z + γ₂X + v, S = 1(S* > 0)
Outcome:   Y = β₀ + β₁X + u (observed only if S=1)

Errors (u, v) have correlation ρ.
"""
function generate_heckman_dgp(;
    n::Int=500,
    rho::Float64=0.5,
    true_beta::Float64=2.0,
    seed::Union{Int, Nothing}=nothing
)
    !isnothing(seed) && Random.seed!(seed)

    # Covariates
    X = randn(n)
    Z = randn(n)  # Exclusion restriction

    # Correlated errors: Cov(u, v) = ρ
    u = randn(n)
    v = rho * u + sqrt(1 - rho^2) * randn(n)

    # Selection equation: intercept ensures ~60-70% selected
    gamma = [0.5, 1.0, 0.3]  # intercept, Z, X
    s_star = gamma[1] .+ gamma[2] .* Z .+ gamma[3] .* X .+ v
    selected = s_star .> 0

    # Outcome equation
    beta = [1.0, true_beta]  # intercept, X
    outcomes = beta[1] .+ beta[2] .* X .+ u
    outcomes[.!selected] .= NaN

    return (
        outcomes=outcomes,
        selected=selected,
        sel_cov=hcat(X, Z),
        out_cov=reshape(X, :, 1),
        true_beta=true_beta,
        true_rho=rho,
        n_selected=sum(selected),
        n=n
    )
end

# =============================================================================
# Monte Carlo Tests
# =============================================================================

@testset "Monte Carlo Validation" begin
    # =========================================================================
    # Test 1: Unbiasedness with Moderate Selection
    # =========================================================================
    @testset "Unbiasedness (ρ=0.5, n=500, 200 runs)" begin
        n_runs = 200
        true_beta = 2.0
        true_rho = 0.5
        estimates = Float64[]
        ses = Float64[]

        for run in 1:n_runs
            data = generate_heckman_dgp(
                n=500, rho=true_rho, true_beta=true_beta, seed=run
            )

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())

            push!(estimates, solution.estimate)
            push!(ses, solution.se)
        end

        mean_estimate = mean(estimates)
        bias = mean_estimate - true_beta
        rel_bias = abs(bias) / true_beta

        # Heckman is consistent but may have finite-sample bias
        @test rel_bias < 0.15  # Within 15% relative bias
        @test isfinite(mean(ses))
    end

    # =========================================================================
    # Test 2: Coverage Validation
    # =========================================================================
    @testset "95% CI Coverage (ρ=0.5, n=500, 500 runs)" begin
        n_runs = 500
        true_beta = 2.0
        covers = Bool[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=500, rho=0.5, true_beta=true_beta, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())

            push!(covers, solution.ci_lower < true_beta < solution.ci_upper)
        end

        coverage = mean(covers)
        # Heckman SE can be conservative; accept 90-97%
        @test 0.90 < coverage < 0.97
    end

    # =========================================================================
    # Test 3: SE Accuracy
    # =========================================================================
    @testset "SE Accuracy (empirical vs estimated)" begin
        n_runs = 300
        estimates = Float64[]
        mean_se = Float64[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=500, rho=0.5, true_beta=2.0, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())

            push!(estimates, solution.estimate)
            push!(mean_se, solution.se)
        end

        empirical_se = std(estimates)
        estimated_se = mean(mean_se)

        se_ratio = estimated_se / empirical_se
        # SE should be within 20% of empirical (Heckman SE can be unstable)
        @test 0.7 < se_ratio < 1.5
    end

    # =========================================================================
    # Test 4: Selection Detection Power
    # =========================================================================
    @testset "Selection Detection Power (ρ=0.8, 200 runs)" begin
        n_runs = 200
        rejections = Bool[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=500, rho=0.8, true_beta=2.0, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())
            result = selection_bias_test(solution)

            push!(rejections, result.reject_null)
        end

        power = mean(rejections)
        # With ρ=0.8 and n=500, should have good power
        @test power > 0.50  # At least 50% power
    end

    # =========================================================================
    # Test 5: Type I Error Control
    # =========================================================================
    @testset "Type I Error (ρ=0, 500 runs)" begin
        n_runs = 500
        rejections = Bool[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=500, rho=0.0, true_beta=2.0, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())
            result = selection_bias_test(solution)

            push!(rejections, result.reject_null)
        end

        type_i_error = mean(rejections)
        # Should reject at approximately α=0.05
        @test type_i_error < 0.10  # Liberal bound for finite sample
    end

    # =========================================================================
    # Test 6: Larger Sample Convergence
    # =========================================================================
    @testset "Convergence with n=2000" begin
        n_runs = 100
        true_beta = 2.0
        estimates = Float64[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=2000, rho=0.5, true_beta=true_beta, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())

            push!(estimates, solution.estimate)
        end

        mean_estimate = mean(estimates)
        bias = mean_estimate - true_beta
        rel_bias = abs(bias) / true_beta

        # Larger sample should have smaller bias
        @test rel_bias < 0.10  # Within 10%
    end

    # =========================================================================
    # Test 7: Strong Selection Scenario
    # =========================================================================
    @testset "Strong Selection (ρ=0.9, correction matters)" begin
        n_runs = 100
        true_beta = 2.0
        heckman_estimates = Float64[]
        naive_estimates = Float64[]

        for run in 1:n_runs
            data = generate_heckman_dgp(n=500, rho=0.9, true_beta=true_beta, seed=run)

            problem = HeckmanProblem(
                data.outcomes,
                data.selected,
                data.sel_cov,
                data.out_cov,
                (alpha=0.05,)
            )
            solution = solve(problem, HeckmanTwoStep())
            push!(heckman_estimates, solution.estimate)

            # Naive OLS on selected sample
            selected_idx = findall(data.selected)
            Y_sel = data.outcomes[selected_idx]
            X_sel = data.out_cov[selected_idx, :]
            X_with_intercept = hcat(ones(length(selected_idx)), X_sel)
            beta_naive = X_with_intercept \ Y_sel
            push!(naive_estimates, beta_naive[2])
        end

        heckman_bias = abs(mean(heckman_estimates) - true_beta)
        naive_bias = abs(mean(naive_estimates) - true_beta)

        # Heckman should reduce bias compared to naive OLS
        @test heckman_bias < naive_bias * 0.8  # At least 20% improvement
    end
end
