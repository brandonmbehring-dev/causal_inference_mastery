"""
    Discovery Utilities

Session 133: DAG generation, data generation, and evaluation metrics.
"""
module DiscoveryUtils

using LinearAlgebra
using Statistics
using Random
using Distributions

include("types.jl")
using .DiscoveryTypes

export generate_random_dag, generate_dag_data
export skeleton_f1, compute_shd, dag_to_cpdag


"""
    generate_random_dag(n_nodes; edge_prob=0.3, seed=nothing)

Generate random DAG using Erdos-Renyi model with topological ordering.

# Arguments
- `n_nodes::Int`: Number of nodes
- `edge_prob::Float64`: Probability of edge between valid pairs
- `seed::Union{Int,Nothing}`: Random seed

# Returns
- `DAG`: Random directed acyclic graph
"""
function generate_random_dag(n_nodes::Int;
                             edge_prob::Float64=0.3,
                             seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)

    # Generate random order (implicit topological ordering)
    order = randperm(n_nodes)

    # Create adjacency respecting ordering
    adj = zeros(Int8, n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in i+1:n_nodes
            if rand() < edge_prob
                parent, child = order[i], order[j]
                adj[parent, child] = 1
            end
        end
    end

    DAG(n_nodes, ["X$i" for i in 0:n_nodes-1], adj)
end


"""
    generate_dag_data(dag, n_samples; noise_scale=1.0, coef_range=(0.5, 1.5),
                      noise_type=:gaussian, seed=nothing)

Generate data from linear structural causal model.

X_j = Σ_{i∈Pa(j)} B[i,j] * X_i + ε_j

# Arguments
- `dag::DAG`: Causal DAG structure
- `n_samples::Int`: Number of samples
- `noise_scale::Float64`: Standard deviation of noise
- `coef_range::Tuple{Float64,Float64}`: Range for edge coefficients
- `noise_type::Symbol`: :gaussian, :laplace, :uniform, or :exponential
- `seed::Union{Int,Nothing}`: Random seed

# Returns
- `data::Matrix{Float64}`: Data matrix (n_samples × n_nodes)
- `B::Matrix{Float64}`: Weighted adjacency matrix
"""
function generate_dag_data(dag::DAG, n_samples::Int;
                           noise_scale::Float64=1.0,
                           coef_range::Tuple{Float64,Float64}=(0.5, 1.5),
                           noise_type::Symbol=:gaussian,
                           seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)

    n_nodes = dag.n_nodes

    # Generate random coefficients
    B = zeros(n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in 1:n_nodes
            if dag.adjacency[i, j] == 1
                sign = rand([-1, 1])
                magnitude = rand() * (coef_range[2] - coef_range[1]) + coef_range[1]
                B[i, j] = sign * magnitude
            end
        end
    end

    # Get topological order
    order = topological_order(dag)

    # Generate data
    data = zeros(n_samples, n_nodes)

    for node in order
        # Generate noise
        noise = if noise_type == :gaussian
            randn(n_samples) * noise_scale
        elseif noise_type == :laplace
            rand(Laplace(0, noise_scale / sqrt(2)), n_samples)
        elseif noise_type == :uniform
            (rand(n_samples) .- 0.5) * 2 * noise_scale * sqrt(3)
        elseif noise_type == :exponential
            rand(Exponential(noise_scale), n_samples) .- noise_scale
        else
            error("Unknown noise type: $noise_type")
        end

        # Compute value
        value = copy(noise)
        for parent in parents(dag, node)
            value .+= B[parent, node] .* data[:, parent]
        end

        data[:, node] = value
    end

    data, B
end


"""
    dag_to_cpdag(dag)

Convert DAG to its CPDAG (Markov equivalence class).

Uses v-structure detection and Meek rules.
"""
function dag_to_cpdag(dag::DAG)
    n = dag.n_nodes
    cpdag = CPDAG(n; node_names=dag.node_names)

    # Find v-structures
    v_structures = Set{Tuple{Int,Int}}()
    for z in 1:n
        parents_z = collect(parents(dag, z))
        for i in 1:length(parents_z)
            for j in i+1:length(parents_z)
                x, y = parents_z[i], parents_z[j]
                if !has_edge(dag, x, y) && !has_edge(dag, y, x)
                    push!(v_structures, (x, z))
                    push!(v_structures, (y, z))
                end
            end
        end
    end

    # Initialize CPDAG
    for i in 1:n
        for j in 1:n
            if dag.adjacency[i, j] == 1
                if (i, j) in v_structures
                    add_directed_edge!(cpdag, i, j)
                else
                    add_undirected_edge!(cpdag, i, j)
                end
            end
        end
    end

    # Apply Meek rules
    changed = true
    while changed
        changed = false
        changed |= meek_r1!(cpdag)
        changed |= meek_r2!(cpdag)
        changed |= meek_r3!(cpdag)
        changed |= meek_r4!(cpdag)
    end

    cpdag
end


# Meek rules (same as in PC algorithm)
function meek_r1!(cpdag::CPDAG)
    changed = false
    n = cpdag.n_nodes
    for i in 1:n, j in 1:n
        has_undirected_edge(cpdag, i, j) || continue
        for k in 1:n
            if has_directed_edge(cpdag, k, i) && !has_any_edge(cpdag, k, j)
                add_directed_edge!(cpdag, i, j)
                changed = true
                break
            end
        end
    end
    changed
end

function meek_r2!(cpdag::CPDAG)
    changed = false
    n = cpdag.n_nodes
    for i in 1:n, j in 1:n
        has_undirected_edge(cpdag, i, j) || continue
        for k in 1:n
            if has_directed_edge(cpdag, i, k) && has_directed_edge(cpdag, k, j)
                add_directed_edge!(cpdag, i, j)
                changed = true
                break
            end
        end
    end
    changed
end

function meek_r3!(cpdag::CPDAG)
    changed = false
    n = cpdag.n_nodes
    for i in 1:n, j in 1:n
        has_undirected_edge(cpdag, i, j) || continue
        candidates = [k for k in 1:n if has_undirected_edge(cpdag, i, k) && has_directed_edge(cpdag, k, j)]
        for idx1 in 1:length(candidates)
            for idx2 in idx1+1:length(candidates)
                if !has_any_edge(cpdag, candidates[idx1], candidates[idx2])
                    add_directed_edge!(cpdag, i, j)
                    changed = true
                    break
                end
            end
            changed && break
        end
    end
    changed
end

function meek_r4!(cpdag::CPDAG)
    changed = false
    n = cpdag.n_nodes
    for i in 1:n, j in 1:n
        has_undirected_edge(cpdag, i, j) || continue
        for k in 1:n
            has_undirected_edge(cpdag, i, k) || continue
            has_any_edge(cpdag, k, j) && continue
            for l in 1:n
                if has_directed_edge(cpdag, k, l) && has_directed_edge(cpdag, l, j)
                    add_directed_edge!(cpdag, i, j)
                    changed = true
                    break
                end
            end
            changed && break
        end
    end
    changed
end


"""
    skeleton_f1(estimated, true_dag)

Compute precision, recall, F1 for skeleton recovery.
"""
function skeleton_f1(estimated::Graph, true_dag::DAG)
    n = estimated.n_nodes
    true_edges = Set{Tuple{Int,Int}}()
    est_edges = Set{Tuple{Int,Int}}()

    for i in 1:n
        for j in i+1:n
            if has_edge(true_dag, i, j) || has_edge(true_dag, j, i)
                push!(true_edges, (i, j))
            end
            if has_edge(estimated, i, j)
                push!(est_edges, (i, j))
            end
        end
    end

    true_positives = length(intersect(true_edges, est_edges))
    false_positives = length(setdiff(est_edges, true_edges))
    false_negatives = length(setdiff(true_edges, est_edges))

    precision = true_positives + false_positives > 0 ?
        true_positives / (true_positives + false_positives) : 0.0
    recall = true_positives + false_negatives > 0 ?
        true_positives / (true_positives + false_negatives) : 0.0
    f1 = precision + recall > 0 ?
        2 * precision * recall / (precision + recall) : 0.0

    precision, recall, f1
end


"""
    compute_shd(estimated, true_dag)

Compute Structural Hamming Distance.
"""
function compute_shd(estimated::CPDAG, true_dag::DAG)
    n = estimated.n_nodes
    shd = 0

    for i in 1:n
        for j in i+1:n
            true_ij = has_edge(true_dag, i, j)
            true_ji = has_edge(true_dag, j, i)
            true_edge = true_ij || true_ji

            est_ij = has_directed_edge(estimated, i, j)
            est_ji = has_directed_edge(estimated, j, i)
            est_undir = has_undirected_edge(estimated, i, j)
            est_edge = est_ij || est_ji || est_undir

            if true_edge && !est_edge
                shd += 1
            elseif !true_edge && est_edge
                shd += 1
            elseif true_edge && est_edge
                if est_ij && true_ji
                    shd += 1
                elseif est_ji && true_ij
                    shd += 1
                end
            end
        end
    end

    shd
end

end # module
