"""
    PC Algorithm

Session 133: Constraint-based causal structure learning.

# Functions
- `pc_skeleton`: Learn undirected skeleton via CI tests
- `pc_orient`: Orient skeleton to CPDAG using v-structures and Meek rules
- `pc_algorithm`: Full PC algorithm
"""
module PCAlgorithm

using Combinatorics
using LinearAlgebra

# Import from sibling modules (already included by Discovery.jl)
using ..DiscoveryTypes
using ..IndependenceTests

export pc_algorithm, pc_skeleton, pc_orient


"""
    pc_skeleton(data; alpha=0.01, max_cond_size=nothing, stable=true)

Learn undirected skeleton using conditional independence tests.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `alpha::Float64`: Significance level for CI tests
- `max_cond_size::Union{Int,Nothing}`: Maximum conditioning set size
- `stable::Bool`: Use order-independent stable variant

# Returns
- `skeleton::Graph`: Undirected skeleton
- `separating_sets::Dict`: Separating sets for removed edges
- `n_tests::Int`: Number of CI tests performed
"""
function pc_skeleton(data::Matrix{Float64};
                     alpha::Float64=0.01,
                     max_cond_size::Union{Int,Nothing}=nothing,
                     stable::Bool=true)
    n_vars = size(data, 2)
    max_k = isnothing(max_cond_size) ? n_vars - 2 : max_cond_size

    # Initialize complete graph
    adj = ones(Int8, n_vars, n_vars) - Matrix{Int8}(I, n_vars, n_vars)
    skeleton = Graph(n_vars, adj)
    separating_sets = Dict{Tuple{Int,Int}, Set{Int}}()
    n_tests = 0

    # Iterate over conditioning set sizes
    for cond_size in 0:max_k
        # Collect edges to test (for stable PC)
        edges_to_test = edges(skeleton)

        # Track edges to remove
        edges_to_remove = Tuple{Int,Int,Set{Int}}[]

        for (i, j) in edges_to_test
            has_edge(skeleton, i, j) || continue

            # Get adjacent nodes
            adj_i = setdiff(neighbors(skeleton, i), Set([j]))
            adj_j = setdiff(neighbors(skeleton, j), Set([i]))
            adjacent = length(adj_i) >= length(adj_j) ? adj_i : adj_j

            length(adjacent) < cond_size && continue

            # Test all conditioning sets of current size
            found_independent = false
            for cond_set in combinations(collect(adjacent), cond_size)
                n_tests += 1
                result = fisher_z_test(data, i, j, collect(cond_set); alpha=alpha)

                if result.independent
                    sep_key = (min(i, j), max(i, j))
                    separating_sets[sep_key] = Set(cond_set)
                    found_independent = true

                    if stable
                        push!(edges_to_remove, (i, j, Set(cond_set)))
                    else
                        remove_edge!(skeleton, i, j)
                    end
                    break
                end
            end
        end

        # For stable PC, remove edges after testing all
        if stable
            for (i, j, _) in edges_to_remove
                remove_edge!(skeleton, i, j)
            end
        end

        # Check termination
        max_degree = maximum(length(neighbors(skeleton, v)) for v in 1:n_vars)
        max_degree < cond_size + 1 && break
    end

    skeleton, separating_sets, n_tests
end


"""
    pc_orient(skeleton, separating_sets)

Orient skeleton edges to CPDAG using v-structures and Meek rules.

# Arguments
- `skeleton::Graph`: Undirected skeleton from Phase 1
- `separating_sets::Dict`: Separating sets from Phase 1

# Returns
- `CPDAG`: Completed partially directed acyclic graph
"""
function pc_orient(skeleton::Graph, separating_sets::Dict{Tuple{Int,Int}, Set{Int}})
    n = skeleton.n_nodes
    cpdag = CPDAG(n; node_names=skeleton.node_names)

    # Initialize with skeleton as undirected
    for (i, j) in edges(skeleton)
        add_undirected_edge!(cpdag, i, j)
    end

    # Step 1: Orient v-structures
    for z in 1:n
        nbrs = collect(neighbors(skeleton, z))
        for idx_x in 1:length(nbrs)
            for idx_y in idx_x+1:length(nbrs)
                x, y = nbrs[idx_x], nbrs[idx_y]

                # Check if x and y are NOT adjacent
                has_edge(skeleton, x, y) && continue

                # Check separating set for (x, y)
                sep_key = (min(x, y), max(x, y))
                if haskey(separating_sets, sep_key)
                    sep_set = separating_sets[sep_key]
                    if !(z in sep_set)
                        # V-structure: x → z ← y
                        has_undirected_edge(cpdag, x, z) && add_directed_edge!(cpdag, x, z)
                        has_undirected_edge(cpdag, y, z) && add_directed_edge!(cpdag, y, z)
                    end
                else
                    # Never tested together, treat as v-structure
                    has_undirected_edge(cpdag, x, z) && add_directed_edge!(cpdag, x, z)
                    has_undirected_edge(cpdag, y, z) && add_directed_edge!(cpdag, y, z)
                end
            end
        end
    end

    # Step 2: Apply Meek rules until convergence
    apply_meek_rules!(cpdag)

    cpdag
end


"""Apply Meek's rules R1-R4 until no changes."""
function apply_meek_rules!(cpdag::CPDAG)
    changed = true
    while changed
        changed = false
        changed |= meek_r1!(cpdag)
        changed |= meek_r2!(cpdag)
        changed |= meek_r3!(cpdag)
        changed |= meek_r4!(cpdag)
    end
end

"""Meek Rule 1: Orient i---j as i→j if k→i and k not adj j."""
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

"""Meek Rule 2: Orient i---j as i→j if i→k→j."""
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

"""Meek Rule 3: Orient i---j as i→j if i---k₁→j, i---k₂→j, k₁ not adj k₂."""
function meek_r3!(cpdag::CPDAG)
    changed = false
    n = cpdag.n_nodes
    for i in 1:n, j in 1:n
        has_undirected_edge(cpdag, i, j) || continue

        # Find candidates: k such that i---k→j
        candidates = Int[]
        for k in 1:n
            if has_undirected_edge(cpdag, i, k) && has_directed_edge(cpdag, k, j)
                push!(candidates, k)
            end
        end

        # Check pairs
        for idx1 in 1:length(candidates)
            for idx2 in idx1+1:length(candidates)
                k1, k2 = candidates[idx1], candidates[idx2]
                if !has_any_edge(cpdag, k1, k2)
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

"""Meek Rule 4: Orient i---j as i→j if i---k→l→j, k not adj j."""
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
    pc_algorithm(data; alpha=0.01, max_cond_size=nothing, stable=true)

Full PC algorithm for causal discovery.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `alpha::Float64`: Significance level for CI tests
- `max_cond_size::Union{Int,Nothing}`: Maximum conditioning set size
- `stable::Bool`: Use order-independent stable variant

# Returns
- `PCResult`: Result containing CPDAG, skeleton, and metrics

# Example
```julia
data = randn(1000, 5)
result = pc_algorithm(data; alpha=0.01)
println("CI tests: ", result.n_ci_tests)
```
"""
function pc_algorithm(data::Matrix{Float64};
                      alpha::Float64=0.01,
                      max_cond_size::Union{Int,Nothing}=nothing,
                      stable::Bool=true)
    # Phase 1: Learn skeleton
    skeleton, separating_sets, n_tests = pc_skeleton(data;
        alpha=alpha, max_cond_size=max_cond_size, stable=stable)

    # Phase 2: Orient edges
    cpdag = pc_orient(skeleton, separating_sets)

    PCResult(cpdag, skeleton, separating_sets, n_tests, alpha)
end

end # module
