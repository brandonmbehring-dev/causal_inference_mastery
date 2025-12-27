"""
    FCI Algorithm

Session 134: Causal discovery with latent confounders.

# Functions
- `fci_algorithm`: Full FCI algorithm
- `fci_orient`: Orient skeleton to PAG using FCI rules

Extends PC algorithm to handle latent confounders by outputting
PAG instead of CPDAG. Bidirected edges indicate latent confounders.
"""
module FCIAlgorithm

using Combinatorics
using LinearAlgebra

# Import from sibling modules (already included by Discovery.jl)
using ..DiscoveryTypes
using ..DiscoveryTypes: NONE, TAIL, ARROW, CIRCLE  # EdgeMark values
using ..IndependenceTests
using ..PCAlgorithm

export fci_algorithm, fci_orient


"""
    fci_algorithm(data; alpha=0.01, max_cond_size=nothing, stable=true)

FCI algorithm for causal discovery with latent confounders.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `alpha::Float64`: Significance level for CI tests
- `max_cond_size::Union{Int,Nothing}`: Maximum conditioning set size
- `stable::Bool`: Use order-independent stable variant

# Returns
- `FCIResult`: Result containing PAG, skeleton, and metrics

# Example
```julia
data = randn(1000, 5)
result = fci_algorithm(data; alpha=0.01)
println("Bidirected edges: ", n_bidirected_edges(result.pag))
```
"""
function fci_algorithm(data::Matrix{Float64};
                       alpha::Float64=0.01,
                       max_cond_size::Union{Int,Nothing}=nothing,
                       stable::Bool=true)
    # Phase 1: Learn skeleton (reuse PC skeleton)
    skeleton, separating_sets, n_tests = pc_skeleton(data;
        alpha=alpha, max_cond_size=max_cond_size, stable=stable)

    # Phase 2: Orient edges using FCI rules
    pag = fci_orient(skeleton, separating_sets)

    # Identify latent confounders (bidirected edges)
    latent_confounders = Tuple{Int,Int}[]
    for i in 1:pag.n_nodes
        for j in i+1:pag.n_nodes
            if pag.endpoints[i, j, 1] == Int8(ARROW) && pag.endpoints[i, j, 2] == Int8(ARROW)
                push!(latent_confounders, (i, j))
            end
        end
    end

    FCIResult(pag, skeleton, separating_sets, latent_confounders, n_tests, alpha)
end


"""
    fci_orient(skeleton, separating_sets)

Orient skeleton edges to PAG using FCI orientation rules.

# Arguments
- `skeleton::Graph`: Undirected skeleton from Phase 1
- `separating_sets::Dict`: Separating sets from Phase 1

# Returns
- `PAG`: Partially oriented ancestral graph
"""
function fci_orient(skeleton::Graph, separating_sets::Dict{Tuple{Int,Int}, Set{Int}})
    n = skeleton.n_nodes

    # Initialize PAG from skeleton with circle marks
    pag = PAG(skeleton)

    # Step 1: Apply R0 - V-structure detection
    fci_rule_0!(pag, skeleton, separating_sets)

    # Step 2: Apply R1-R10 until convergence
    apply_fci_rules!(pag)

    pag
end


# =============================================================================
# FCI Orientation Rules (Zhang 2008)
# =============================================================================


"""R0: Orient unshielded colliders (v-structures)."""
function fci_rule_0!(pag::PAG, skeleton::Graph, separating_sets::Dict{Tuple{Int,Int}, Set{Int}})
    n = skeleton.n_nodes

    for z in 1:n
        nbrs = collect(neighbors(skeleton, z))
        for idx_x in 1:length(nbrs)
            for idx_y in idx_x+1:length(nbrs)
                x, y = nbrs[idx_x], nbrs[idx_y]

                # Check if X and Y are not adjacent
                has_edge(skeleton, x, y) && continue

                # Check separating set for (x, y)
                sep_key = (min(x, y), max(x, y))
                if haskey(separating_sets, sep_key)
                    sep_set = separating_sets[sep_key]
                    if !(z in sep_set)
                        # V-structure: X *-> Z <-* Y
                        orient_collider!(pag, x, z, y)
                    end
                else
                    # Never tested together, treat as v-structure
                    orient_collider!(pag, x, z, y)
                end
            end
        end
    end
end


function orient_collider!(pag::PAG, x::Int, z::Int, y::Int)
    # Set arrowheads at Z for both edges
    has_edge(pag, x, z) && set_endpoint!(pag, z, x, ARROW)
    has_edge(pag, y, z) && set_endpoint!(pag, z, y, ARROW)
end


"""Apply FCI rules R1-R10 until convergence."""
function apply_fci_rules!(pag::PAG)
    changed = true
    while changed
        changed = false
        changed |= fci_rule_1!(pag)
        changed |= fci_rule_2!(pag)
        changed |= fci_rule_3!(pag)
        changed |= fci_rule_4!(pag)
        changed |= fci_rule_8!(pag)
        changed |= fci_rule_9!(pag)
        changed |= fci_rule_10!(pag)
    end
end


"""R1: Orient away from collider. If X *-> Y o-* Z and X ⊥ Z, orient Y -> Z."""
function fci_rule_1!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for y in 1:n, z in 1:n
        y == z && continue
        has_edge(pag, y, z) || continue

        # Check if Y o-* Z (circle at Y)
        get_endpoint(pag, y, z) != CIRCLE && continue

        # Look for X *-> Y where X not adjacent to Z
        for x in 1:n
            (x == y || x == z) && continue
            has_edge(pag, x, y) || continue

            # Check X *-> Y (arrow at Y)
            get_endpoint(pag, y, x) != ARROW && continue

            # Check X not adjacent to Z
            has_edge(pag, x, z) && continue

            # Orient Y -> Z: tail at Y, arrow at Z
            set_endpoint!(pag, y, z, TAIL)
            set_endpoint!(pag, z, y, ARROW)
            changed = true
        end
    end
    changed
end


"""R2: Orient to prevent cycles. If X -> Y *-> Z or X *-> Y -> Z, and X o-* Z, orient X *-> Z."""
function fci_rule_2!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for x in 1:n, z in 1:n
        x == z && continue
        has_edge(pag, x, z) || continue

        # Check if X o-* Z (circle at X)
        get_endpoint(pag, x, z) != CIRCLE && continue

        # Look for X -> Y *-> Z or X *-> Y -> Z
        for y in 1:n
            (y == x || y == z) && continue
            has_edge(pag, x, y) && has_edge(pag, y, z) || continue

            # Pattern 1: X -> Y *-> Z
            x_to_y = is_definitely_directed(pag, x, y)
            y_to_z = get_endpoint(pag, z, y) == ARROW

            # Pattern 2: X *-> Y -> Z
            x_into_y = get_endpoint(pag, y, x) == ARROW
            y_to_z_directed = is_definitely_directed(pag, y, z)

            if (x_to_y && y_to_z) || (x_into_y && y_to_z_directed)
                # Orient X *-> Z (set arrow at Z)
                set_endpoint!(pag, z, x, ARROW)
                changed = true
                break
            end
        end
    end
    changed
end


"""R3: Double-triangle rule."""
function fci_rule_3!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for y in 1:n
        # Find pairs (X, Z) forming X *-> Y <-* Z where X ⊥ Z
        pairs = Tuple{Int,Int}[]
        for x in 1:n
            x == y && continue
            has_edge(pag, x, y) || continue
            get_endpoint(pag, y, x) != ARROW && continue

            for z in x+1:n
                z == y && continue
                has_edge(pag, z, y) || continue
                get_endpoint(pag, y, z) != ARROW && continue
                has_edge(pag, x, z) && continue

                push!(pairs, (x, z))
            end
        end

        # For each pair, look for W
        for (x, z) in pairs
            for w in 1:n
                w in (x, y, z) && continue
                has_edge(pag, w, x) && has_edge(pag, w, z) && has_edge(pag, w, y) || continue

                # Check W o-* Y, X *-o W, W o-* Z
                get_endpoint(pag, w, y) == CIRCLE || continue
                get_endpoint(pag, w, x) == CIRCLE || continue
                get_endpoint(pag, w, z) == CIRCLE || continue

                # Orient W *-> Y
                set_endpoint!(pag, y, w, ARROW)
                changed = true
            end
        end
    end
    changed
end


"""R4: Discriminating path rule (simplified)."""
function fci_rule_4!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for y in 1:n, x in 1:n
        x == y && continue
        has_edge(pag, x, y) || continue

        # Check if X o-* Y (circle at X)
        get_endpoint(pag, x, y) != CIRCLE && continue

        # Look for discriminating path pattern
        for w in 1:n
            w in (x, y) && continue
            has_edge(pag, w, x) || continue

            w_to_x = is_definitely_directed(pag, w, x)
            w_bidi_x = (get_endpoint(pag, w, x) == ARROW && get_endpoint(pag, x, w) == ARROW)

            (w_to_x || w_bidi_x) || continue

            # Check W *-> Y
            has_edge(pag, w, y) || continue
            get_endpoint(pag, y, w) != ARROW && continue

            for v in 1:n
                v in (w, x, y) && continue
                has_edge(pag, v, w) || continue
                has_edge(pag, v, y) && continue

                # V -> W
                v_to_w = is_definitely_directed(pag, v, w)

                if v_to_w
                    set_endpoint!(pag, y, x, ARROW)
                    changed = true
                    break
                end
            end
            changed && break
        end
    end
    changed
end


"""R8: Orient definite non-ancestor. If X -> Y -> Z or X -o Y -> Z, and X o-> Z, orient X -> Z."""
function fci_rule_8!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for x in 1:n, z in 1:n
        x == z && continue
        has_edge(pag, x, z) || continue

        # Check X o-> Z (circle at X, arrow at Z)
        get_endpoint(pag, x, z) != CIRCLE && continue
        get_endpoint(pag, z, x) != ARROW && continue

        for y in 1:n
            y in (x, z) && continue
            has_edge(pag, x, y) && has_edge(pag, y, z) || continue

            # X -> Y or X -o Y
            x_to_y = is_definitely_directed(pag, x, y)
            x_tail_circle_y = (get_endpoint(pag, x, y) == TAIL && get_endpoint(pag, y, x) == CIRCLE)

            # Y -> Z
            y_to_z = is_definitely_directed(pag, y, z)

            if (x_to_y || x_tail_circle_y) && y_to_z
                set_endpoint!(pag, x, z, TAIL)
                changed = true
                break
            end
        end
    end
    changed
end


"""R9: Uncovered potentially directed path."""
function fci_rule_9!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for x in 1:n, y in 1:n
        x == y && continue
        has_edge(pag, x, y) || continue

        # Check X o-> Y
        get_endpoint(pag, x, y) != CIRCLE && continue
        get_endpoint(pag, y, x) != ARROW && continue

        # Look for uncovered p.d. path from X to Y
        if has_uncovered_pd_path(pag, x, y, Set([x]))
            set_endpoint!(pag, x, y, TAIL)
            changed = true
        end
    end
    changed
end


function has_uncovered_pd_path(pag::PAG, start::Int, target::Int, visited::Set{Int})
    start == target && return true

    for neighbor in adjacent(pag, start)
        neighbor in visited && continue

        # Check if edge could be oriented start -> neighbor (not arrow at start)
        get_endpoint(pag, start, neighbor) == ARROW && continue

        neighbor == target && return true

        visited_new = union(visited, Set([neighbor]))
        has_uncovered_pd_path(pag, neighbor, target, visited_new) && return true
    end

    false
end


"""R10: Three uncovered potentially directed paths."""
function fci_rule_10!(pag::PAG)
    changed = false
    n = pag.n_nodes

    for x in 1:n, y in 1:n
        x == y && continue
        has_edge(pag, x, y) || continue

        # Check X o-> Y
        get_endpoint(pag, x, y) != CIRCLE && continue
        get_endpoint(pag, y, x) != ARROW && continue

        # Count paths through different first intermediate nodes
        path_starts = Int[]
        for neighbor in adjacent(pag, x)
            neighbor == y && continue
            get_endpoint(pag, x, neighbor) == ARROW && continue

            if has_uncovered_pd_path(pag, neighbor, y, Set([x, neighbor]))
                push!(path_starts, neighbor)
            end
        end

        if length(path_starts) >= 3
            set_endpoint!(pag, x, y, TAIL)
            changed = true
        end
    end
    changed
end


end # module
