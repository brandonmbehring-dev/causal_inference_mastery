"""
Tests for CACE (Complier Average Causal Effect) estimation in Julia.

Test Structure (following 6-layer validation):
- Layer 1: Known-answer tests with hand-calculated values
- Layer 3: Monte Carlo validation
- Layer 4: Cross-language parity (vs Python)
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include the module
include("../../src/principal_stratification/cace.jl")

# =============================================================================
# Test Data Generator
# =============================================================================

"""
    generate_ps_dgp(; n, pi_c, pi_a, pi_n, true_cace, baseline, noise_sd, seed)

Generate data from a principal stratification DGP.
"""
function generate_ps_dgp(;
    n::Int = 1000,
    pi_c::Float64 = 0.70,
    pi_a::Float64 = 0.15,
    pi_n::Float64 = 0.15,
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

    # Generate strata: 0=compliers, 1=always-takers, 2=never-takers
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
            D[i] = Z[i]  # Complier: follow assignment
        elseif strata[i] == 1
            D[i] = true  # Always-taker
        else
            D[i] = false  # Never-taker
        end
    end

    # Outcome: Y = baseline + cace * D + noise
    Y = baseline .+ true_cace .* Float64.(D) .+ noise_sd .* randn(n)

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
# Layer 1: Known-Answer Tests
# =============================================================================

@testset "CACE Known-Answer Tests" begin

    @testset "CACE equals LATE (Wald)" begin
        data = generate_ps_dgp(n=5000, true_cace=2.0, seed=42)

        result_2sls = cace_2sls(data.Y, data.D, data.Z)
        result_wald = wald_estimator(data.Y, data.D, data.Z)

        # Point estimates should match
        @test isapprox(result_2sls.cace, result_wald.cace, rtol=1e-10)
    end

    @testset "First-stage equals complier proportion" begin
        pi_c = 0.60
        data = generate_ps_dgp(n=10000, pi_c=pi_c, pi_a=0.20, pi_n=0.20, seed=123)

        result = cace_2sls(data.Y, data.D, data.Z)

        # First-stage ≈ pi_c
        @test isapprox(result.first_stage_coef, pi_c, rtol=0.05)
    end

    @testset "Strata proportions sum to 1" begin
        data = generate_ps_dgp(n=1000, seed=456)
        result = cace_2sls(data.Y, data.D, data.Z)

        total = result.strata_proportions.compliers +
                result.strata_proportions.always_takers +
                result.strata_proportions.never_takers

        @test isapprox(total, 1.0, atol=1e-10)
    end

    @testset "Perfect compliance: CACE = ATE" begin
        Random.seed!(789)
        n = 2000
        true_ate = 3.0

        Z = rand(n) .< 0.5
        D = copy(Z)  # Perfect compliance
        Y = 1.0 .+ true_ate .* Float64.(D) .+ randn(n)

        result = cace_2sls(Y, D, Z)

        @test isapprox(result.cace, true_ate, rtol=0.10)
        @test isapprox(result.strata_proportions.compliers, 1.0, atol=0.01)
    end

    @testset "Reduced form = CACE × first-stage" begin
        data = generate_ps_dgp(n=5000, true_cace=2.0, seed=101)
        result = cace_2sls(data.Y, data.D, data.Z)

        γ_computed = result.cace * result.first_stage_coef

        @test isapprox(result.reduced_form, γ_computed, rtol=1e-10)
    end

end

# =============================================================================
# Layer 2: Adversarial / Edge Cases
# =============================================================================

@testset "CACE Adversarial Tests" begin

    @testset "Weak first-stage still works" begin
        # Few compliers => weak first stage
        data = generate_ps_dgp(
            n=500, pi_c=0.05, pi_a=0.475, pi_n=0.475,
            true_cace=2.0, seed=202
        )

        result = cace_2sls(data.Y, data.D, data.Z)

        @test result.first_stage_f > 0
        @test isfinite(result.cace)
    end

    @testset "No compliers raises error" begin
        Random.seed!(303)
        n = 500
        Z = rand(n) .< 0.5
        D = trues(n)  # All always-takers
        Y = 1.0 .+ 2.0 .* Float64.(D) .+ randn(n)

        @test_throws ArgumentError CACEProblem(Y, D, Z)
    end

    @testset "Extreme proportions still work" begin
        # 95% compliers
        data = generate_ps_dgp(
            n=2000, pi_c=0.95, pi_a=0.025, pi_n=0.025,
            true_cace=1.5, seed=606
        )

        result = cace_2sls(data.Y, data.D, data.Z)

        @test isfinite(result.cace)
        @test result.strata_proportions.compliers > 0.90
    end

end

# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================

@testset "CACE Monte Carlo Validation" begin

    @testset "CACE unbiased (1000 runs)" begin
        n_runs = 1000
        n_obs = 500
        true_cace = 2.0
        estimates = Float64[]

        for seed in 1:n_runs
            data = generate_ps_dgp(
                n=n_obs, true_cace=true_cace,
                pi_c=0.70, pi_a=0.15, pi_n=0.15,
                seed=seed
            )
            result = cace_2sls(data.Y, data.D, data.Z)
            push!(estimates, result.cace)
        end

        mean_estimate = mean(estimates)
        bias = mean_estimate - true_cace

        @test abs(bias) < 0.10
    end

    @testset "Coverage close to 95% (1000 runs)" begin
        n_runs = 1000
        n_obs = 500
        true_cace = 2.0
        covered = 0

        for seed in 1:n_runs
            data = generate_ps_dgp(
                n=n_obs, true_cace=true_cace,
                pi_c=0.70, pi_a=0.15, pi_n=0.15,
                seed=seed
            )
            result = cace_2sls(data.Y, data.D, data.Z)

            if result.ci_lower <= true_cace <= result.ci_upper
                covered += 1
            end
        end

        coverage = covered / n_runs

        @test 0.90 <= coverage <= 0.98
    end

    @testset "Wald matches 2SLS (100 runs)" begin
        for seed in 1:100
            data = generate_ps_dgp(n=500, true_cace=2.0, seed=seed)

            result_2sls = cace_2sls(data.Y, data.D, data.Z)
            result_wald = wald_estimator(data.Y, data.D, data.Z)

            @test isapprox(result_2sls.cace, result_wald.cace, rtol=1e-10)
        end
    end

end

# =============================================================================
# Return Type Tests
# =============================================================================

@testset "CACE Return Type" begin

    @testset "All fields present" begin
        data = generate_ps_dgp(n=500, seed=707)
        result = cace_2sls(data.Y, data.D, data.Z)

        @test haskey(result, :cace)
        @test haskey(result, :se)
        @test haskey(result, :ci_lower)
        @test haskey(result, :ci_upper)
        @test haskey(result, :z_stat)
        @test haskey(result, :pvalue)
        @test haskey(result, :strata_proportions)
        @test haskey(result, :first_stage_coef)
        @test haskey(result, :first_stage_se)
        @test haskey(result, :first_stage_f)
        @test haskey(result, :reduced_form)
        @test haskey(result, :reduced_form_se)
        @test haskey(result, :n)
        @test haskey(result, :method)
    end

    @testset "All values finite" begin
        data = generate_ps_dgp(n=1000, seed=909)
        result = cace_2sls(data.Y, data.D, data.Z)

        @test isfinite(result.cace)
        @test isfinite(result.se)
        @test isfinite(result.ci_lower)
        @test isfinite(result.ci_upper)
        @test isfinite(result.z_stat)
        @test isfinite(result.pvalue)
        @test isfinite(result.first_stage_coef)
        @test isfinite(result.first_stage_se)
        @test isfinite(result.first_stage_f)
    end

end

# =============================================================================
# Inference Tests
# =============================================================================

@testset "CACE Inference" begin

    @testset "Alpha affects CI width" begin
        data = generate_ps_dgp(n=1000, seed=1202)

        result_95 = cace_2sls(data.Y, data.D, data.Z, alpha=0.05)
        result_90 = cace_2sls(data.Y, data.D, data.Z, alpha=0.10)

        width_95 = result_95.ci_upper - result_95.ci_lower
        width_90 = result_90.ci_upper - result_90.ci_lower

        @test width_95 > width_90
    end

end

# =============================================================================
# EM Algorithm Tests
# =============================================================================

@testset "CACE EM Algorithm" begin

    @testset "EM runs and returns valid result" begin
        data = generate_ps_dgp(n=1000, true_cace=2.0, seed=2001)
        result = cace_em(data.Y, data.D, data.Z)

        @test haskey(result, :cace)
        @test haskey(result, :se)
        @test haskey(result, :ci_lower)
        @test haskey(result, :ci_upper)
        @test haskey(result, :strata_proportions)
        @test haskey(result, :method)
        @test result.method == :em
    end

    @testset "EM close to 2SLS" begin
        data = generate_ps_dgp(n=2000, true_cace=2.0, seed=2002)

        result_2sls = cace_2sls(data.Y, data.D, data.Z)
        result_em = cace_em(data.Y, data.D, data.Z)

        # Should be within 20% of each other
        @test isapprox(result_em.cace, result_2sls.cace, rtol=0.20)
    end

    @testset "EM strata sum to 1" begin
        data = generate_ps_dgp(n=1000, seed=2003)
        result = cace_em(data.Y, data.D, data.Z)

        total = result.strata_proportions.compliers +
                result.strata_proportions.always_takers +
                result.strata_proportions.never_takers

        @test isapprox(total, 1.0, atol=1e-6)
    end

    @testset "EM recovers true CACE" begin
        data = generate_ps_dgp(n=3000, true_cace=2.0, seed=2004)
        result = cace_em(data.Y, data.D, data.Z)

        # Should be within 0.5 of true value
        @test abs(result.cace - data.true_cace) < 0.5
    end

    @testset "EM CI contains true value" begin
        data = generate_ps_dgp(n=2000, true_cace=2.0, seed=2005)
        result = cace_em(data.Y, data.D, data.Z)

        @test result.ci_lower <= data.true_cace <= result.ci_upper
    end

    @testset "EM first-stage equals complier proportion" begin
        data = generate_ps_dgp(n=2000, pi_c=0.60, pi_a=0.20, pi_n=0.20, seed=2006)
        result = cace_em(data.Y, data.D, data.Z)

        # EM strata proportion should be close to truth
        @test isapprox(result.strata_proportions.compliers, 0.60, rtol=0.15)
    end

    @testset "EM method field is correct" begin
        data = generate_ps_dgp(n=500, seed=2007)
        result = cace_em(data.Y, data.D, data.Z)

        @test result.method == :em
    end

    @testset "EM with different tolerances" begin
        data = generate_ps_dgp(n=1000, seed=2008)

        result_default = cace_em(data.Y, data.D, data.Z, tol=1e-6)
        result_loose = cace_em(data.Y, data.D, data.Z, tol=1e-2)

        # Both should produce finite results
        @test isfinite(result_default.cace)
        @test isfinite(result_loose.cace)
    end

end

@testset "CACE EM Monte Carlo" begin

    @testset "EM unbiased (500 runs)" begin
        n_runs = 500
        n_obs = 500
        true_cace = 2.0
        estimates = Float64[]

        for seed in 1:n_runs
            data = generate_ps_dgp(
                n=n_obs, true_cace=true_cace,
                pi_c=0.70, pi_a=0.15, pi_n=0.15,
                seed=seed
            )
            result = cace_em(data.Y, data.D, data.Z)
            push!(estimates, result.cace)
        end

        mean_estimate = mean(estimates)
        bias = mean_estimate - true_cace

        @test abs(bias) < 0.15
    end

    @testset "EM coverage close to 95% (500 runs)" begin
        n_runs = 500
        n_obs = 500
        true_cace = 2.0
        covered = 0

        for seed in 1:n_runs
            data = generate_ps_dgp(
                n=n_obs, true_cace=true_cace,
                pi_c=0.70, pi_a=0.15, pi_n=0.15,
                seed=seed
            )
            result = cace_em(data.Y, data.D, data.Z)

            if result.ci_lower <= true_cace <= result.ci_upper
                covered += 1
            end
        end

        coverage = covered / n_runs

        # Allow wider range for EM (entropy-based SE)
        @test 0.85 <= coverage <= 0.99
    end

    @testset "EM vs 2SLS correlation (100 runs)" begin
        n_runs = 100
        estimates_em = Float64[]
        estimates_2sls = Float64[]

        for seed in 1:n_runs
            data = generate_ps_dgp(n=500, true_cace=2.0, seed=seed)

            result_em = cace_em(data.Y, data.D, data.Z)
            result_2sls = cace_2sls(data.Y, data.D, data.Z)

            push!(estimates_em, result_em.cace)
            push!(estimates_2sls, result_2sls.cace)
        end

        # Correlation should be high
        correlation = cor(estimates_em, estimates_2sls)
        @test correlation > 0.90
    end

end

println("All CACE tests passed!")
