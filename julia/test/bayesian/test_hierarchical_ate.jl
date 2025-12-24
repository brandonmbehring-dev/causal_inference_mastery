#=
Unit Tests for Hierarchical Bayesian ATE Estimation.

Session 104: Initial test suite.

Tests cover:
1. Basic functionality and return structure
2. Known-answer validation
3. Partial pooling behavior
4. MCMC diagnostics
5. Edge cases and error handling

Note: These tests require Turing.jl to be installed.
=#

using Test
using Random
using Statistics

# Skip tests if Turing is not available
const TURING_AVAILABLE = try
    using Turing
    using MCMCDiagnosticTools
    true
catch
    false
end


# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate hierarchical data with known population ATE and heterogeneity."""
function generate_hierarchical_data(;
    n_groups::Int = 5,
    n_per_group::Int = 50,
    population_ate::Float64 = 2.0,
    tau::Float64 = 0.5,
    seed::Int = 42,
)
    Random.seed!(seed)
    n = n_groups * n_per_group

    # Generate group assignments
    groups = repeat(0:n_groups-1, inner=n_per_group)

    # Generate group-specific effects
    true_group_effects = population_ate .+ tau .* randn(n_groups)

    # Generate treatment (randomized within groups)
    treatment = Float64.(rand(n) .< 0.5)

    # Generate outcomes
    group_effects = [true_group_effects[g+1] for g in groups]
    outcomes = group_effects .* treatment .+ randn(n)

    return (
        outcomes=outcomes,
        treatment=treatment,
        groups=groups,
        true_population_ate=population_ate,
        true_tau=tau,
        true_group_effects=true_group_effects,
        n_groups=n_groups,
    )
end


"""Generate data with no between-group heterogeneity (tau=0)."""
function generate_homogeneous_data(;
    n_groups::Int = 5,
    n_per_group::Int = 50,
    population_ate::Float64 = 2.0,
    seed::Int = 42,
)
    Random.seed!(seed)
    n = n_groups * n_per_group
    groups = repeat(0:n_groups-1, inner=n_per_group)
    treatment = Float64.(rand(n) .< 0.5)
    outcomes = population_ate .* treatment .+ randn(n)
    return (
        outcomes=outcomes,
        treatment=treatment,
        groups=groups,
        true_population_ate=population_ate,
        true_tau=0.0,
    )
end


"""Generate data with large between-group heterogeneity."""
function generate_heterogeneous_data(;
    n_groups::Int = 5,
    n_per_group::Int = 50,
    seed::Int = 42,
)
    Random.seed!(seed)
    n = n_groups * n_per_group
    groups = repeat(0:n_groups-1, inner=n_per_group)

    # Large spread of group effects
    true_group_effects = collect(range(-1, 5, length=n_groups))
    treatment = Float64.(rand(n) .< 0.5)
    group_effects = [true_group_effects[g+1] for g in groups]
    outcomes = group_effects .* treatment .+ randn(n)

    return (
        outcomes=outcomes,
        treatment=treatment,
        groups=groups,
        true_group_effects=true_group_effects,
        true_population_ate=mean(true_group_effects),
    )
end


# =============================================================================
# Conditional Test Block (only run if Turing available)
# =============================================================================

if TURING_AVAILABLE
    using CausalEstimators

    @testset "Hierarchical ATE Basic" begin
        @testset "Returns correct structure" begin
            data = generate_hierarchical_data(n_groups=3, n_per_group=30, seed=1)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=200,
                n_chains=2,
                n_warmup=100,
                progress=false,
            )

            @test isa(result, HierarchicalATEResult)
            @test hasfield(HierarchicalATEResult, :population_ate)
            @test hasfield(HierarchicalATEResult, :population_ate_se)
            @test hasfield(HierarchicalATEResult, :group_ates)
            @test hasfield(HierarchicalATEResult, :tau)
            @test hasfield(HierarchicalATEResult, :posterior_samples)
            @test hasfield(HierarchicalATEResult, :rhat_max)
            @test hasfield(HierarchicalATEResult, :ess_min)
        end

        @testset "Posterior samples shape" begin
            n_groups = 4
            n_samples = 300
            data = generate_hierarchical_data(n_groups=n_groups, n_per_group=25, seed=2)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=n_samples,
                n_chains=2,
                n_warmup=100,
                progress=false,
            )

            total_samples = n_samples * 2
            @test length(result.posterior_samples[:μ]) == total_samples
            @test length(result.posterior_samples[:τ]) == total_samples
            @test size(result.posterior_samples[:θ]) == (total_samples, n_groups)
        end

        @testset "Group ATEs match groups" begin
            n_groups = 5
            data = generate_hierarchical_data(n_groups=n_groups, n_per_group=20, seed=3)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=200,
                n_chains=2,
                n_warmup=100,
                progress=false,
            )

            @test length(result.group_ates) == n_groups
            @test length(result.group_ate_ses) == n_groups
            @test length(result.group_ids) == n_groups
            @test result.n_groups == n_groups
        end

        @testset "Estimate in credible interval" begin
            data = generate_hierarchical_data(seed=4)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=200,
                n_chains=2,
                n_warmup=100,
                progress=false,
            )

            @test result.population_ate_ci_lower <= result.population_ate
            @test result.population_ate <= result.population_ate_ci_upper
        end

        @testset "SE positive" begin
            data = generate_hierarchical_data(seed=5)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=200,
                n_chains=2,
                n_warmup=100,
                progress=false,
            )

            @test result.population_ate_se > 0
            @test all(result.group_ate_ses .> 0)
        end
    end


    @testset "Hierarchical ATE Known-Answer" begin
        @testset "Population ATE recovered" begin
            data = generate_hierarchical_data(
                n_groups=5,
                n_per_group=100,
                population_ate=2.0,
                tau=0.3,
                seed=42,
            )
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=500,
                n_chains=2,
                n_warmup=200,
                progress=false,
            )

            # Should be within 0.5 of true value
            @test abs(result.population_ate - data.true_population_ate) < 0.5
        end

        @testset "Homogeneous groups low tau" begin
            data = generate_homogeneous_data(n_groups=5, n_per_group=80, seed=43)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=500,
                n_chains=2,
                n_warmup=200,
                progress=false,
            )

            # Tau should be close to 0
            @test result.tau < 0.5
        end

        @testset "Heterogeneous groups high tau" begin
            data = generate_heterogeneous_data(n_groups=5, n_per_group=80, seed=44)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=500,
                n_chains=2,
                n_warmup=200,
                progress=false,
            )

            # Tau should be substantial
            @test result.tau > 0.5
        end
    end


    @testset "Hierarchical ATE MCMC Diagnostics" begin
        @testset "R-hat acceptable" begin
            data = generate_hierarchical_data(n_groups=4, n_per_group=50, seed=46)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=500,
                n_chains=4,
                n_warmup=300,
                progress=false,
            )

            # R-hat should be < 1.1 for convergence
            @test result.rhat_max < 1.1
        end

        @testset "ESS adequate" begin
            data = generate_hierarchical_data(n_groups=4, n_per_group=50, seed=47)
            result = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=500,
                n_chains=4,
                n_warmup=300,
                progress=false,
            )

            # ESS should be > 100 at minimum
            @test result.ess_min > 100
        end
    end


    @testset "Hierarchical ATE Edge Cases" begin
        @testset "Length mismatch outcomes/treatment" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0],
                [0.0, 1.0],  # Wrong length
                [0, 0, 1],
            )
        end

        @testset "Length mismatch outcomes/groups" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 0.0],
                [0, 0],  # Wrong length
            )
        end

        @testset "Non-binary treatment" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0],
                [0.0, 0.5, 1.0],  # Not binary
                [0, 0, 1],
            )
        end

        @testset "Single group raises" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 0.0],
                [0, 0, 0],  # Only one group
            )
        end

        @testset "Invalid credible level" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 0.0, 1.0],
                [0, 0, 1, 1];
                credible_level=1.5,
            )
        end

        @testset "Invalid n_samples" begin
            @test_throws ArgumentError hierarchical_bayesian_ate(
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 0.0, 1.0],
                [0, 0, 1, 1];
                n_samples=50,
            )
        end
    end


    @testset "Hierarchical ATE Credible Level" begin
        @testset "90% narrower than 95%" begin
            data = generate_hierarchical_data(n_groups=4, n_per_group=40, seed=50)

            result_90 = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=300,
                n_chains=2,
                n_warmup=100,
                credible_level=0.90,
                seed=50,
                progress=false,
            )

            result_95 = hierarchical_bayesian_ate(
                data.outcomes,
                data.treatment,
                data.groups;
                n_samples=300,
                n_chains=2,
                n_warmup=100,
                credible_level=0.95,
                seed=50,
                progress=false,
            )

            width_90 = result_90.population_ate_ci_upper - result_90.population_ate_ci_lower
            width_95 = result_95.population_ate_ci_upper - result_95.population_ate_ci_lower

            @test width_90 < width_95
        end
    end


    @testset "Hierarchical ATE Display" begin
        data = generate_hierarchical_data(n_groups=3, n_per_group=30, seed=70)
        result = hierarchical_bayesian_ate(
            data.outcomes,
            data.treatment,
            data.groups;
            n_samples=200,
            n_chains=2,
            n_warmup=100,
            progress=false,
        )

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test contains(output, "HierarchicalATEResult")
        @test contains(output, "ATE:")
        @test contains(output, "τ:")
        @test contains(output, "R-hat")
    end

else
    # Turing not available - skip all tests
    @testset "Hierarchical ATE (Turing not available)" begin
        @test_skip "Turing.jl not installed - skipping hierarchical ATE tests"
    end
end
