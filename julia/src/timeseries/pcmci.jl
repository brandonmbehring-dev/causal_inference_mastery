"""
PCMCI Algorithm for Time Series Causal Discovery

Session 136: Implementation of PCMCI (Runge et al., 2019).
"""
module PCMCI

using LinearAlgebra
using Statistics
using Combinatorics

using ..PCMCITypes
using ..CITests

export pcmci, pc_stable_condition_selection, mci_test_all


"""
    pcmci(data; max_lag=3, alpha=0.05, ci_test="parcorr", pc_alpha=nothing,
          min_lag=1, max_cond_size=nothing, verbosity=0)

PCMCI algorithm for time series causal discovery.

# Arguments
- `data::Matrix{Float64}`: (n_obs, n_vars) time series data
- `max_lag::Int`: Maximum time lag
- `alpha::Float64`: Significance level for MCI phase
- `ci_test::String`: CI test ("parcorr")
- `pc_alpha::Union{Float64,Nothing}`: Significance for PC phase (default: 2*alpha)
- `min_lag::Int`: Minimum lag (1 excludes contemporaneous)
- `max_cond_size::Union{Int,Nothing}`: Maximum conditioning set size
- `verbosity::Int`: Verbosity level

# Returns
- `PCMCIResult`: Discovered causal structure

# Example
```julia
using Random
Random.seed!(42)
n = 200
data = zeros(n, 3)
data[:, 1] = randn(n)
for t in 2:n
    data[t, 2] = 0.6 * data[t-1, 1] + 0.3 * data[t-1, 2] + randn() * 0.5
    data[t, 3] = 0.5 * data[t-1, 2] + 0.2 * data[t-1, 3] + randn() * 0.5
end
result = pcmci(data, max_lag=2, alpha=0.05)
println("Found \$(length(result.links)) causal links")
```
"""
function pcmci(data::Matrix{Float64};
               max_lag::Int=3,
               alpha::Float64=0.05,
               ci_test::String="parcorr",
               pc_alpha::Union{Float64,Nothing}=nothing,
               min_lag::Int=1,
               max_cond_size::Union{Int,Nothing}=nothing,
               verbosity::Int=0)

    n_obs, n_vars = size(data)

    # Validation
    if n_obs <= max_lag + 1
        error("Insufficient observations ($n_obs) for max_lag=$max_lag. " *
              "Need at least $(max_lag + 2).")
    end

    if max_lag < 1
        error("max_lag must be >= 1, got $max_lag")
    end

    if pc_alpha === nothing
        pc_alpha = min(2 * alpha, 0.2)
    end

    if verbosity > 0
        println("PCMCI: n_obs=$n_obs, n_vars=$n_vars, max_lag=$max_lag")
        println("       alpha=$alpha, pc_alpha=$pc_alpha, ci_test=$ci_test")
    end

    # Phase 1: PC-stable condition selection
    if verbosity > 0
        println("\nPhase 1: PC-stable condition selection...")
    end

    condition_result = pc_stable_condition_selection(
        data;
        max_lag=max_lag,
        alpha=pc_alpha,
        ci_test=ci_test,
        min_lag=min_lag,
        max_cond_size=max_cond_size,
        verbosity=verbosity
    )

    if verbosity > 0
        total_candidates = sum(length(p) for p in values(condition_result.parents))
        println("       Found $total_candidates candidate links")
    end

    # Phase 2: MCI testing
    if verbosity > 0
        println("\nPhase 2: MCI testing...")
    end

    mci_result = mci_test_all(
        data,
        condition_result.parents;
        max_lag=max_lag,
        alpha=alpha,
        ci_test=ci_test,
        min_lag=min_lag,
        verbosity=verbosity
    )

    if verbosity > 0
        n_significant = sum(mci_result[:graph])
        println("       Found $n_significant significant links")
    end

    # Build result
    links = build_links(
        mci_result[:p_matrix],
        mci_result[:val_matrix],
        mci_result[:graph],
        n_vars, max_lag, min_lag
    )

    # Extract final parent sets
    final_parents = Dict{Int, Vector{Tuple{Int,Int}}}()
    for j in 1:n_vars
        final_parents[j] = Tuple{Int,Int}[]
    end
    for link in links
        push!(final_parents[link.target_var], (link.source_var, link.lag))
    end

    PCMCIResult(
        links=links,
        p_matrix=mci_result[:p_matrix],
        val_matrix=mci_result[:val_matrix],
        graph=mci_result[:graph],
        parents=final_parents,
        n_vars=n_vars,
        n_obs=n_obs,
        max_lag=max_lag,
        alpha=alpha,
        ci_test=ci_test
    )
end


"""
    pc_stable_condition_selection(data; max_lag, alpha, ci_test, min_lag,
                                   max_cond_size, verbosity)

PC-stable algorithm for condition selection.
"""
function pc_stable_condition_selection(data::Matrix{Float64};
                                        max_lag::Int,
                                        alpha::Float64,
                                        ci_test::String="parcorr",
                                        min_lag::Int=1,
                                        max_cond_size::Union{Int,Nothing}=nothing,
                                        verbosity::Int=0)
    n_obs, n_vars = size(data)

    # Initialize: all possible (var, lag) pairs are candidates
    parents = Dict{Int, Vector{Tuple{Int,Int}}}()
    separating_sets = Dict{Tuple{Int,Int,Int}, Set{Tuple{Int,Int}}}()

    for target in 1:n_vars
        candidates = Tuple{Int,Int}[]
        for source in 1:n_vars
            for lag in min_lag:max_lag
                # Skip self-contemporaneous
                if source == target && lag == 0
                    continue
                end
                push!(candidates, (source, lag))
            end
        end
        parents[target] = candidates
    end

    # Iteratively test and remove
    cond_size = 0
    max_possible_cond = max_cond_size === nothing ? max_lag * n_vars : max_cond_size

    while cond_size <= max_possible_cond
        if verbosity > 1
            println("  Condition set size: $cond_size")
        end

        any_removed = false

        for target in 1:n_vars
            current_parents = copy(parents[target])

            if length(current_parents) <= cond_size
                continue
            end

            for (source, lag) in current_parents
                if !((source, lag) in parents[target])
                    continue  # Already removed
                end

                # Get conditioning candidates
                other_parents = filter(p -> p != (source, lag), parents[target])

                if length(other_parents) < cond_size
                    continue
                end

                # Test all conditioning sets
                removed = false
                for cond_set in combinations(other_parents, cond_size)
                    result = run_ci_test(
                        data, source, target, lag,
                        collect(cond_set);
                        ci_test=ci_test, alpha=alpha
                    )

                    if result.is_independent
                        # Remove link and store separator
                        filter!(p -> p != (source, lag), parents[target])
                        separating_sets[(source, target, lag)] = Set(cond_set)
                        removed = true
                        any_removed = true

                        if verbosity > 1
                            cond_str = join(["X$v(t-$l)" for (v, l) in cond_set], ", ")
                            println("    Removed: X$source(t-$lag) → X$target | {$cond_str}")
                        end
                        break
                    end
                end
            end
        end

        cond_size += 1

        if !any_removed && cond_size > 0
            break
        end
    end

    ConditionSelectionResult(parents, separating_sets, n_vars, max_lag)
end


"""
    mci_test_all(data, parents; max_lag, alpha, ci_test, min_lag, verbosity)

MCI tests for all candidate links.
"""
function mci_test_all(data::Matrix{Float64},
                      parents::Dict{Int, Vector{Tuple{Int,Int}}};
                      max_lag::Int,
                      alpha::Float64,
                      ci_test::String="parcorr",
                      min_lag::Int=1,
                      verbosity::Int=0)
    n_obs, n_vars = size(data)

    # Initialize output
    p_matrix = ones(n_vars, n_vars, max_lag + 1)
    val_matrix = zeros(n_vars, n_vars, max_lag + 1)
    graph = zeros(Int8, n_vars, n_vars, max_lag + 1)

    for target in 1:n_vars
        target_parents = get(parents, target, Tuple{Int,Int}[])

        for (source, lag) in target_parents
            # Build conditioning set: Parents(source) ∪ Parents(target) \ {(source, lag)}
            source_parents = get_lagged_parents(get(parents, source, Tuple{Int,Int}[]), lag)
            all_parents = Set(source_parents) ∪ Set(target_parents)
            delete!(all_parents, (source, lag))
            cond_set = collect(all_parents)

            # Run MCI test
            result = run_ci_test(
                data, source, target, lag, cond_set;
                ci_test=ci_test, alpha=alpha
            )

            p_matrix[source, target, lag+1] = result.p_value
            val_matrix[source, target, lag+1] = result.statistic

            if !result.is_independent
                graph[source, target, lag+1] = 1

                if verbosity > 1
                    println("  Significant: X$source(t-$lag) → X$target, ",
                            "p=$(round(result.p_value, digits=4))")
                end
            end
        end
    end

    Dict(:p_matrix => p_matrix, :val_matrix => val_matrix, :graph => graph)
end


"""Get parents with additional lag offset."""
function get_lagged_parents(parents::Vector{Tuple{Int,Int}}, additional_lag::Int)
    [(var, lag + additional_lag) for (var, lag) in parents]
end


"""Build TimeSeriesLink list from result matrices."""
function build_links(p_matrix::Array{Float64,3}, val_matrix::Array{Float64,3},
                     graph::Array{Int8,3}, n_vars::Int, max_lag::Int, min_lag::Int)
    links = TimeSeriesLink[]

    for source in 1:n_vars
        for target in 1:n_vars
            for lag in min_lag:max_lag
                if graph[source, target, lag+1] != 0
                    push!(links, TimeSeriesLink(
                        source, target, lag,
                        val_matrix[source, target, lag+1],
                        p_matrix[source, target, lag+1]
                    ))
                end
            end
        end
    end

    # Sort by p-value
    sort!(links, by=l -> l.p_value)
    return links
end

end # module
