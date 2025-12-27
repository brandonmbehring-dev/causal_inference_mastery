"""
    Tests for Causal Discovery Module

Session 133: Validation of PC algorithm and LiNGAM.
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Include the module
include("../../src/discovery/Discovery.jl")
using .Discovery


@testset "Discovery Module" begin

    @testset "Graph Types" begin
        @testset "Graph creation and manipulation" begin
            g = Graph(5)
            @test g.n_nodes == 5
            @test n_edges(g) == 0

            add_edge!(g, 1, 2)
            @test has_edge(g, 1, 2)
            @test has_edge(g, 2, 1)
            @test n_edges(g) == 1

            remove_edge!(g, 1, 2)
            @test !has_edge(g, 1, 2)
        end

        @testset "DAG creation and properties" begin
            dag = DAG(4)
            add_edge!(dag, 1, 2)  # 1 → 2
            add_edge!(dag, 2, 3)  # 2 → 3
            add_edge!(dag, 1, 3)  # 1 → 3

            @test has_edge(dag, 1, 2)
            @test !has_edge(dag, 2, 1)
            @test 1 in parents(dag, 2)
            @test 3 in children(dag, 2)

            order = topological_order(dag)
            @test findfirst(==(1), order) < findfirst(==(2), order)
            @test findfirst(==(2), order) < findfirst(==(3), order)
        end

        @testset "CPDAG operations" begin
            cpdag = CPDAG(3)
            add_undirected_edge!(cpdag, 1, 2)
            @test has_undirected_edge(cpdag, 1, 2)
            @test has_any_edge(cpdag, 1, 2)

            add_directed_edge!(cpdag, 2, 3)
            @test has_directed_edge(cpdag, 2, 3)
            @test !has_undirected_edge(cpdag, 2, 3)
        end
    end

    @testset "DAG Generation" begin
        @testset "Random DAG is acyclic" begin
            for seed in 1:10
                dag = generate_random_dag(8; edge_prob=0.4, seed=seed)
                @test is_acyclic(dag)
            end
        end

        @testset "Data generation from DAG" begin
            dag = generate_random_dag(5; edge_prob=0.3, seed=42)
            data, B = generate_dag_data(dag, 1000; seed=42)

            @test size(data) == (1000, 5)
            @test size(B) == (5, 5)

            # B should have non-zero entries only where DAG has edges
            for i in 1:5, j in 1:5
                if dag.adjacency[i, j] == 0
                    @test B[i, j] == 0
                end
            end
        end

        @testset "Different noise types" begin
            dag = generate_random_dag(3; edge_prob=0.5, seed=42)

            for noise in [:gaussian, :laplace, :uniform, :exponential]
                data, _ = generate_dag_data(dag, 500; noise_type=noise, seed=42)
                @test size(data, 1) == 500
                @test all(isfinite, data)
            end
        end
    end

    @testset "Independence Tests" begin
        Random.seed!(42)
        n = 500

        @testset "Independent variables" begin
            X = randn(n)
            Y = randn(n)
            data = hcat(X, Y)

            result = fisher_z_test(data, 1, 2; alpha=0.05)
            @test result.pvalue > 0.05
            @test result.independent
        end

        @testset "Dependent variables" begin
            X = randn(n)
            Y = 0.8 * X + 0.3 * randn(n)
            data = hcat(X, Y)

            result = fisher_z_test(data, 1, 2; alpha=0.05)
            @test result.pvalue < 0.05
            @test !result.independent
        end

        @testset "Conditional independence" begin
            # Z → X, Z → Y (confounded)
            Z = randn(n)
            X = 0.7 * Z + 0.3 * randn(n)
            Y = 0.7 * Z + 0.3 * randn(n)
            data = hcat(X, Y, Z)

            # Unconditional: X and Y are dependent
            result_unc = fisher_z_test(data, 1, 2; alpha=0.05)
            @test !result_unc.independent

            # Conditional on Z: X ⊥ Y | Z
            result_cond = fisher_z_test(data, 1, 2, [3]; alpha=0.05)
            @test result_cond.independent
        end

        @testset "Partial correlation" begin
            Z = randn(n)
            X = 0.7 * Z + 0.3 * randn(n)
            Y = 0.7 * Z + 0.3 * randn(n)
            data = hcat(X, Y, Z)

            ρ_unc = partial_correlation(data, 1, 2)
            ρ_cond = partial_correlation(data, 1, 2, [3])

            @test abs(ρ_unc) > 0.3
            @test abs(ρ_cond) < 0.15
        end
    end

    @testset "PC Algorithm" begin
        @testset "Chain structure" begin
            # X1 → X2 → X3
            dag = DAG(3)
            add_edge!(dag, 1, 2)
            add_edge!(dag, 2, 3)

            data, _ = generate_dag_data(dag, 1000; seed=42)
            result = pc_algorithm(data; alpha=0.01)

            # Skeleton should recover both edges
            prec, rec, f1 = skeleton_f1(result.skeleton, dag)
            @test f1 >= 0.6
        end

        @testset "Collider (v-structure)" begin
            # X1 → X2 ← X3
            dag = DAG(3)
            add_edge!(dag, 1, 2)
            add_edge!(dag, 3, 2)

            data, _ = generate_dag_data(dag, 1000; seed=42)
            result = pc_algorithm(data; alpha=0.01)

            # V-structure should be detected
            @test has_directed_edge(result.cpdag, 1, 2) || has_directed_edge(result.cpdag, 3, 2)
        end

        @testset "Larger DAG" begin
            dag = generate_random_dag(6; edge_prob=0.3, seed=42)
            data, _ = generate_dag_data(dag, 1000; seed=42)

            result = pc_algorithm(data; alpha=0.01)

            # Should complete without error
            @test result.n_ci_tests > 0
            @test n_edges(result.skeleton) > 0
        end
    end

    @testset "LiNGAM" begin
        @testset "Chain with Laplace noise" begin
            dag = DAG(3)
            add_edge!(dag, 1, 2)
            add_edge!(dag, 2, 3)

            data, _ = generate_dag_data(dag, 1000; noise_type=:laplace, seed=42)
            result = direct_lingam(data; seed=42)

            # Should identify some ordering
            @test length(result.causal_order) == 3
            @test result.dag.n_nodes == 3
        end

        @testset "Fork structure" begin
            # X2 → X1, X2 → X3
            dag = DAG(3)
            add_edge!(dag, 2, 1)
            add_edge!(dag, 2, 3)

            data, _ = generate_dag_data(dag, 1000; noise_type=:laplace, seed=42)
            result = direct_lingam(data; seed=42)

            # X2 should be early in causal order
            @test findfirst(==(2), result.causal_order) == 1
        end

        @testset "Larger DAG with Laplace noise" begin
            dag = generate_random_dag(5; edge_prob=0.4, seed=42)
            data, _ = generate_dag_data(dag, 1000; noise_type=:laplace, seed=42)

            result = direct_lingam(data; seed=42)

            @test length(result.causal_order) == 5
            @test size(result.adjacency_matrix) == (5, 5)
        end
    end

    @testset "Evaluation Metrics" begin
        @testset "Skeleton F1" begin
            true_dag = DAG(4)
            add_edge!(true_dag, 1, 2)
            add_edge!(true_dag, 2, 3)
            add_edge!(true_dag, 3, 4)

            # Perfect skeleton
            skeleton = Graph(4)
            add_edge!(skeleton, 1, 2)
            add_edge!(skeleton, 2, 3)
            add_edge!(skeleton, 3, 4)

            prec, rec, f1 = skeleton_f1(skeleton, true_dag)
            @test prec ≈ 1.0
            @test rec ≈ 1.0
            @test f1 ≈ 1.0
        end

        @testset "SHD" begin
            true_dag = DAG(3)
            add_edge!(true_dag, 1, 2)
            add_edge!(true_dag, 2, 3)

            # Exact match
            cpdag = dag_to_cpdag(true_dag)
            shd = compute_shd(cpdag, true_dag)
            @test shd == 0
        end
    end

end  # main testset

println("All discovery tests passed!")
