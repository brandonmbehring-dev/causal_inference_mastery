"""
PCMCI Types for Time Series Causal Discovery

Session 136: Data structures for PCMCI algorithm (Runge et al., 2019).
"""
module PCMCITypes

using LinearAlgebra

export LinkType, TimeSeriesLink, LaggedDAG, PCMCIResult
export ConditionSelectionResult, CITestResult
export is_significant, is_lagged, add_edge!, remove_edge!, has_edge
export get_parents, get_children, n_edges, to_links
export get_lagged_dag, get_significant_links, get_parents_of


"""Link types in time-lagged causal graph."""
@enum LinkType begin
    DIRECTED   # -->
    UNDIRECTED # ---
    BIDIRECTED # <->
    NONE       # No link
end


"""
    TimeSeriesLink

A directed link in a time-lagged causal graph.

Represents a causal relationship X_{t-τ} → Y_t.
"""
struct TimeSeriesLink
    source_var::Int
    target_var::Int
    lag::Int
    strength::Float64
    p_value::Float64
    link_type::LinkType

    function TimeSeriesLink(source_var, target_var, lag, strength, p_value;
                           link_type=DIRECTED)
        lag < 0 && error("Lag must be >= 0, got $lag")
        !(0.0 <= p_value <= 1.0) && error("P-value must be in [0, 1], got $p_value")
        new(source_var, target_var, lag, strength, p_value, link_type)
    end
end

function TimeSeriesLink(;
    source_var::Int,
    target_var::Int,
    lag::Int,
    strength::Float64,
    p_value::Float64,
    link_type::LinkType=DIRECTED
)
    TimeSeriesLink(source_var, target_var, lag, strength, p_value; link_type=link_type)
end

is_significant(link::TimeSeriesLink, alpha::Float64=0.05) = link.p_value < alpha
is_lagged(link::TimeSeriesLink) = link.lag > 0

function Base.show(io::IO, link::TimeSeriesLink)
    lag_str = link.lag > 0 ? "t-$(link.lag)" : "t"
    arrow = link.link_type == DIRECTED ? "→" :
            link.link_type == UNDIRECTED ? "—" :
            link.link_type == BIDIRECTED ? "↔" : "x"
    print(io, "TimeSeriesLink(X$(link.source_var)_{$lag_str} $arrow ",
          "X$(link.target_var)_t, val=$(round(link.strength, digits=3)), ",
          "p=$(round(link.p_value, digits=4)))")
end


"""
    LaggedDAG

Time-lagged directed acyclic graph.

# Fields
- `n_vars::Int`: Number of variables
- `max_lag::Int`: Maximum lag considered
- `adjacency::Array{Int8,3}`: (n_vars, n_vars, max_lag+1) binary adjacency
- `weights::Array{Float64,3}`: Edge weights
- `var_names::Vector{String}`: Variable names
"""
mutable struct LaggedDAG
    n_vars::Int
    max_lag::Int
    adjacency::Array{Int8,3}
    weights::Array{Float64,3}
    var_names::Vector{String}

    function LaggedDAG(n_vars::Int, max_lag::Int;
                       var_names::Union{Vector{String}, Nothing}=nothing)
        adjacency = zeros(Int8, n_vars, n_vars, max_lag + 1)
        weights = zeros(Float64, n_vars, n_vars, max_lag + 1)
        names = var_names === nothing ? ["X$i" for i in 1:n_vars] : var_names
        new(n_vars, max_lag, adjacency, weights, names)
    end
end

function add_edge!(dag::LaggedDAG, source::Int, target::Int, lag::Int;
                   weight::Float64=1.0)
    (lag < 0 || lag > dag.max_lag) && error("Lag must be in [0, $(dag.max_lag)]")
    dag.adjacency[source, target, lag+1] = 1
    dag.weights[source, target, lag+1] = weight
end

function remove_edge!(dag::LaggedDAG, source::Int, target::Int, lag::Int)
    dag.adjacency[source, target, lag+1] = 0
    dag.weights[source, target, lag+1] = 0.0
end

function has_edge(dag::LaggedDAG, source::Int, target::Int, lag::Int)
    dag.adjacency[source, target, lag+1] != 0
end

function get_parents(dag::LaggedDAG, target::Int)
    parents = Tuple{Int,Int}[]
    for source in 1:dag.n_vars
        for lag in 0:dag.max_lag
            if dag.adjacency[source, target, lag+1] != 0
                push!(parents, (source, lag))
            end
        end
    end
    return parents
end

function get_children(dag::LaggedDAG, source::Int; lag::Union{Int,Nothing}=nothing)
    children = Tuple{Int,Int}[]
    lags = lag === nothing ? (0:dag.max_lag) : [lag]
    for target in 1:dag.n_vars
        for l in lags
            if dag.adjacency[source, target, l+1] != 0
                push!(children, (target, l))
            end
        end
    end
    return children
end

n_edges(dag::LaggedDAG) = sum(dag.adjacency)

function to_links(dag::LaggedDAG)
    links = TimeSeriesLink[]
    for source in 1:dag.n_vars
        for target in 1:dag.n_vars
            for lag in 0:dag.max_lag
                if dag.adjacency[source, target, lag+1] != 0
                    push!(links, TimeSeriesLink(
                        source, target, lag,
                        dag.weights[source, target, lag+1], 0.0
                    ))
                end
            end
        end
    end
    return links
end

function Base.show(io::IO, dag::LaggedDAG)
    print(io, "LaggedDAG(n_vars=$(dag.n_vars), max_lag=$(dag.max_lag), ",
          "n_edges=$(n_edges(dag)))")
end


"""
    PCMCIResult

Result from PCMCI algorithm.
"""
struct PCMCIResult
    links::Vector{TimeSeriesLink}
    p_matrix::Array{Float64,3}
    val_matrix::Array{Float64,3}
    graph::Array{Int8,3}
    parents::Dict{Int, Vector{Tuple{Int,Int}}}
    n_vars::Int
    n_obs::Int
    max_lag::Int
    alpha::Float64
    ci_test::String
end

function PCMCIResult(;
    links::Vector{TimeSeriesLink},
    p_matrix::Array{Float64,3},
    val_matrix::Array{Float64,3},
    graph::Array{Int8,3},
    parents::Dict{Int, Vector{Tuple{Int,Int}}},
    n_vars::Int,
    n_obs::Int,
    max_lag::Int,
    alpha::Float64=0.05,
    ci_test::String="parcorr"
)
    PCMCIResult(links, p_matrix, val_matrix, graph, parents,
                n_vars, n_obs, max_lag, alpha, ci_test)
end

function get_lagged_dag(result::PCMCIResult)
    dag = LaggedDAG(result.n_vars, result.max_lag)
    dag.adjacency = Int8.(result.graph)
    dag.weights = copy(result.val_matrix)
    return dag
end

function get_significant_links(result::PCMCIResult;
                               alpha::Union{Float64,Nothing}=nothing)
    threshold = alpha === nothing ? result.alpha : alpha
    filter(l -> l.p_value < threshold, result.links)
end

function get_parents_of(result::PCMCIResult, target::Int)
    get(result.parents, target, Tuple{Int,Int}[])
end

function Base.show(io::IO, r::PCMCIResult)
    print(io, "PCMCIResult(n_vars=$(r.n_vars), n_obs=$(r.n_obs), ",
          "max_lag=$(r.max_lag), n_links=$(length(r.links)), alpha=$(r.alpha))")
end


"""
    ConditionSelectionResult

Result from PC-stable condition selection phase.
"""
struct ConditionSelectionResult
    parents::Dict{Int, Vector{Tuple{Int,Int}}}
    separating_sets::Dict{Tuple{Int,Int,Int}, Set{Tuple{Int,Int}}}
    n_vars::Int
    max_lag::Int
end

function get_candidates(result::ConditionSelectionResult, target::Int)
    get(result.parents, target, Tuple{Int,Int}[])
end

function Base.show(io::IO, r::ConditionSelectionResult)
    total_parents = sum(length(p) for p in values(r.parents))
    print(io, "ConditionSelectionResult(n_vars=$(r.n_vars), ",
          "max_lag=$(r.max_lag), total_candidates=$total_parents)")
end


"""
    CITestResult

Result from conditional independence test.
"""
struct CITestResult
    statistic::Float64
    p_value::Float64
    is_independent::Bool
    dof::Int
end

CITestResult(; statistic, p_value, is_independent, dof=0) =
    CITestResult(statistic, p_value, is_independent, dof)

function Base.show(io::IO, r::CITestResult)
    ind = r.is_independent ? "⊥" : "⊥̸"
    print(io, "CITestResult($ind, stat=$(round(r.statistic, digits=4)), ",
          "p=$(round(r.p_value, digits=4)))")
end

end # module
