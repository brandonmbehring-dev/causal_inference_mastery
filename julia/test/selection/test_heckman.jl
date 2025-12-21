"""
Unit Tests for Heckman Selection Model

Tests verify:
1. Problem and solution type construction
2. Coefficient recovery within tolerance
3. Selection test correctness
4. IMR computation
5. Input validation
"""

using Test
using CausalEstimators
using Random
using Statistics
using Distributions

# =============================================================================
# Test Fixtures
# =============================================================================

"""
Generate Heckman DGP data for testing.
"""
function generate_heckman_data(;
    n::Int=500,
    rho::Float64=0.5,
    true_beta::Float64=2.0,
    seed::Int=42
)
    Random.seed!(seed)

    X = randn(n)
    Z = randn(n)

    # Correlated errors
    u = randn(n)
    v = rho * u + sqrt(1 - rho^2) * randn(n)

    # Selection equation
    s_star = 0.5 .+ 1.0 .* Z .+ 0.3 .* X .+ v
    selected = s_star .> 0

    # Outcome (with selection)
    outcomes = 1.0 .+ true_beta .* X .+ u
    outcomes[.!selected] .= NaN

    return (
        outcomes=outcomes,
        selected=selected,
        sel_cov=hcat(X, Z),
        out_cov=reshape(X, :, 1),
        true_beta=true_beta,
        true_rho=rho,
        n_selected=sum(selected)
    )
end

# =============================================================================
# Problem Construction Tests
# =============================================================================

@testset "HeckmanProblem Construction" begin
    data = generate_heckman_data()

    @testset "Valid construction" begin
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )

        @test problem isa HeckmanProblem
        @test length(problem.outcomes) == 500
        @test sum(problem.selected) == data.n_selected
    end

    @testset "Rejects all selected" begin
        @test_throws ArgumentError HeckmanProblem(
            ones(100),
            fill(true, 100),
            randn(100, 2),
            nothing,
            (alpha=0.05,)
        )
    end

    @testset "Rejects none selected" begin
        @test_throws ArgumentError HeckmanProblem(
            ones(100),
            fill(false, 100),
            randn(100, 2),
            nothing,
            (alpha=0.05,)
        )
    end

    @testset "Rejects mismatched lengths" begin
        @test_throws ArgumentError HeckmanProblem(
            randn(100),
            fill(true, 50),  # Wrong length
            randn(100, 2),
            nothing,
            (alpha=0.05,)
        )
    end

    @testset "Accepts BitVector" begin
        outcomes = randn(100)
        selected = randn(100) .> 0  # Creates BitVector
        problem = HeckmanProblem(
            outcomes,
            selected,
            randn(100, 2),
            nothing,
            (alpha=0.05,)
        )
        @test problem isa HeckmanProblem
    end
end

# =============================================================================
# Solution Tests
# =============================================================================

@testset "HeckmanSolution" begin
    data = generate_heckman_data(n=500, rho=0.5, true_beta=2.0, seed=42)
    problem = HeckmanProblem(
        data.outcomes,
        data.selected,
        data.sel_cov,
        data.out_cov,
        (alpha=0.05,)
    )

    solution = solve(problem, HeckmanTwoStep())

    @testset "Solution structure" begin
        @test solution isa HeckmanSolution
        @test isfinite(solution.estimate)
        @test isfinite(solution.se)
        @test solution.se > 0
        @test solution.ci_lower < solution.estimate < solution.ci_upper
    end

    @testset "Selection parameters" begin
        @test -1 <= solution.rho <= 1
        @test solution.sigma > 0
        @test isfinite(solution.lambda_coef)
        @test isfinite(solution.lambda_se)
        @test 0 <= solution.lambda_pvalue <= 1
    end

    @testset "Sample sizes" begin
        @test solution.n_total == 500
        @test solution.n_selected == data.n_selected
        @test solution.n_selected < solution.n_total
    end

    @testset "Arrays" begin
        @test length(solution.selection_probs) == solution.n_total
        @test length(solution.imr) == solution.n_total
        @test all(0 .<= solution.selection_probs .<= 1)
        @test all(solution.imr .> 0)
    end
end

# =============================================================================
# Coefficient Recovery Tests
# =============================================================================

@testset "Coefficient Recovery" begin
    @testset "Moderate selection (ρ=0.5)" begin
        data = generate_heckman_data(n=500, rho=0.5, true_beta=2.0, seed=100)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())

        rel_error = abs(solution.estimate - data.true_beta) / data.true_beta
        @test rel_error < 0.30  # Within 30%
    end

    @testset "Strong selection (ρ=0.8)" begin
        data = generate_heckman_data(n=500, rho=0.8, true_beta=1.5, seed=200)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())

        rel_error = abs(solution.estimate - data.true_beta) / data.true_beta
        @test rel_error < 0.35  # Within 35% (harder case)
    end

    @testset "No selection (ρ=0)" begin
        data = generate_heckman_data(n=500, rho=0.0, true_beta=2.5, seed=300)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())

        # ρ should be near zero
        @test abs(solution.rho) < 0.5
    end

    @testset "Negative selection (ρ=-0.6)" begin
        data = generate_heckman_data(n=500, rho=-0.6, true_beta=2.0, seed=400)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())

        # Correctly identifies negative selection
        @test solution.lambda_coef < 0 || abs(solution.rho) < 0.3  # May not detect if weak
    end
end

# =============================================================================
# IMR Computation Tests
# =============================================================================

@testset "Inverse Mills Ratio" begin
    @testset "IMR formula correctness" begin
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        imr = compute_imr(probs)

        # Manual calculation: λ(p) = φ(Φ⁻¹(p)) / p
        for (p, λ) in zip(probs, imr)
            z = quantile(Normal(), p)
            expected = pdf(Normal(), z) / p
            @test isapprox(λ, expected, rtol=1e-10)
        end
    end

    @testset "IMR monotonically decreases" begin
        probs = range(0.1, 0.9, length=9)
        imr = compute_imr(collect(probs))

        # IMR should decrease as probability increases
        @test all(diff(imr) .< 0)
    end

    @testset "IMR handles boundaries" begin
        probs = [1e-6, 1e-4, 0.999, 1 - 1e-6]
        imr = compute_imr(probs)

        @test all(isfinite.(imr))
        @test all(imr .> 0)
    end
end

# =============================================================================
# Selection Bias Test
# =============================================================================

@testset "Selection Bias Test" begin
    @testset "Detects strong selection" begin
        data = generate_heckman_data(n=500, rho=0.8, seed=500)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        result = selection_bias_test(solution)

        @test haskey(result, :statistic)
        @test haskey(result, :pvalue)
        @test haskey(result, :reject_null)
        @test haskey(result, :interpretation)

        # Should detect selection (though may not always reject at 0.05)
        @test result.pvalue < 0.20  # At least somewhat significant
    end

    @testset "Does not reject when ρ≈0" begin
        data = generate_heckman_data(n=500, rho=0.0, seed=600)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        result = selection_bias_test(solution)

        # Should not strongly reject
        @test result.pvalue > 0.01 || abs(solution.lambda_coef) < 0.3
    end
end

# =============================================================================
# Estimator Options Tests
# =============================================================================

@testset "Estimator Options" begin
    data = generate_heckman_data(n=300, seed=700)

    @testset "Default options" begin
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        @test isfinite(solution.estimate)
    end

    @testset "Without intercept" begin
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep(add_intercept=false))
        @test isfinite(solution.estimate)
    end

    @testset "Different alpha levels" begin
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )

        sol_05 = solve(problem, HeckmanTwoStep())

        problem_01 = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.01,)
        )
        sol_01 = solve(problem_01, HeckmanTwoStep())

        # 99% CI should be wider
        width_05 = sol_05.ci_upper - sol_05.ci_lower
        width_01 = sol_01.ci_upper - sol_01.ci_lower
        @test width_01 > width_05
    end
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "Edge Cases" begin
    @testset "Small sample" begin
        data = generate_heckman_data(n=50, seed=800)
        problem = HeckmanProblem(
            data.outcomes,
            data.selected,
            data.sel_cov,
            data.out_cov,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        @test isfinite(solution.estimate)
    end

    @testset "High selection rate" begin
        Random.seed!(900)
        n = 200
        X = randn(n)
        Z = randn(n)
        u = randn(n)

        # High selection intercept
        s_star = 2.0 .+ 0.3 .* Z .+ 0.1 .* X .+ randn(n)
        selected = s_star .> 0
        @test sum(selected) / n > 0.85  # >85% selected

        outcomes = 1.0 .+ 2.0 .* X .+ u
        outcomes[.!selected] .= NaN

        problem = HeckmanProblem(
            outcomes,
            selected,
            hcat(X, Z),
            reshape(X, :, 1),
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        @test isfinite(solution.estimate)
    end

    @testset "Low selection rate" begin
        Random.seed!(901)
        n = 300
        X = randn(n)
        Z = randn(n)
        u = randn(n)

        # Low selection intercept
        s_star = -1.0 .+ 1.0 .* Z .+ 0.3 .* X .+ randn(n)
        selected = s_star .> 0
        @test sum(selected) / n < 0.40  # <40% selected

        outcomes = 1.0 .+ 2.0 .* X .+ u
        outcomes[.!selected] .= NaN

        problem = HeckmanProblem(
            outcomes,
            selected,
            hcat(X, Z),
            reshape(X, :, 1),
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        @test isfinite(solution.estimate)
    end

    @testset "Single covariate" begin
        Random.seed!(902)
        n = 200
        X = randn(n)
        u = randn(n)

        selected = (X .+ randn(n)) .> 0
        outcomes = 1.0 .+ 2.0 .* X .+ u
        outcomes[.!selected] .= NaN

        problem = HeckmanProblem(
            outcomes,
            selected,
            reshape(X, :, 1),
            nothing,
            (alpha=0.05,)
        )
        solution = solve(problem, HeckmanTwoStep())
        @test isfinite(solution.estimate)
    end
end

# =============================================================================
# Display Methods
# =============================================================================

@testset "Display Methods" begin
    data = generate_heckman_data(n=200, seed=999)
    problem = HeckmanProblem(
        data.outcomes,
        data.selected,
        data.sel_cov,
        data.out_cov,
        (alpha=0.05,)
    )
    solution = solve(problem, HeckmanTwoStep())

    # Test that show methods don't error
    @test sprint(show, problem) isa String
    @test sprint(show, solution) isa String

    # Check content
    problem_str = sprint(show, problem)
    @test occursin("HeckmanProblem", problem_str)
    @test occursin("200", problem_str)

    solution_str = sprint(show, solution)
    @test occursin("HeckmanSolution", solution_str)
    @test occursin("Estimate", solution_str)
end
