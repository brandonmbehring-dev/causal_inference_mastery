"""
    Discovery Types

Session 133: Type definitions for causal discovery algorithms.

# Types
- `Graph`: Undirected graph for skeleton
- `DAG`: Directed acyclic graph
- `CPDAG`: Completed partially directed acyclic graph
- `PCResult`: Result from PC algorithm
- `LiNGAMResult`: Result from LiNGAM
- `CITestResult`: Result from conditional independence test
"""
module DiscoveryTypes

export Graph, DAG, CPDAG, PCResult, LiNGAMResult, CITestResult
export add_edge!, remove_edge!, has_edge, neighbors, edges, n_edges
export parents, children, topological_order, is_acyclic
export add_directed_edge!, add_undirected_edge!
export has_directed_edge, has_undirected_edge, has_any_edge


"""
    Graph

Undirected graph for skeleton representation.

# Fields
- `n_nodes::Int`: Number of nodes
- `node_names::Vector{String}`: Node names
- `adjacency::Matrix{Int8}`: Symmetric adjacency matrix
"""
struct Graph
    n_nodes::Int
    node_names::Vector{String}
    adjacency::Matrix{Int8}
end

function Graph(n_nodes::Int; node_names::Vector{String}=String[])
    names = isempty(node_names) ? ["X$i" for i in 0:n_nodes-1] : node_names
    adj = zeros(Int8, n_nodes, n_nodes)
    Graph(n_nodes, names, adj)
end

function Graph(n_nodes::Int, adjacency::Matrix{Int8}; node_names::Vector{String}=String[])
    names = isempty(node_names) ? ["X$i" for i in 0:n_nodes-1] : node_names
    Graph(n_nodes, names, adjacency)
end

"""Complete graph with all edges."""
function complete_graph(n_nodes::Int)
    adj = ones(Int8, n_nodes, n_nodes) - I(n_nodes)
    Graph(n_nodes, adj)
end

function add_edge!(g::Graph, i::Int, j::Int)
    g.adjacency[i, j] = 1
    g.adjacency[j, i] = 1
end

function remove_edge!(g::Graph, i::Int, j::Int)
    g.adjacency[i, j] = 0
    g.adjacency[j, i] = 0
end

has_edge(g::Graph, i::Int, j::Int) = g.adjacency[i, j] == 1

function neighbors(g::Graph, i::Int)
    Set(findall(==(1), g.adjacency[i, :]))
end

function edges(g::Graph)
    edge_list = Tuple{Int,Int}[]
    for i in 1:g.n_nodes
        for j in i+1:g.n_nodes
            if g.adjacency[i, j] == 1
                push!(edge_list, (i, j))
            end
        end
    end
    edge_list
end

n_edges(g::Graph) = sum(g.adjacency) ÷ 2


"""
    DAG

Directed acyclic graph for causal structure.

# Fields
- `n_nodes::Int`: Number of nodes
- `node_names::Vector{String}`: Node names
- `adjacency::Matrix{Int8}`: Adjacency matrix (adjacency[i,j]=1 means i→j)
"""
mutable struct DAG
    n_nodes::Int
    node_names::Vector{String}
    adjacency::Matrix{Int8}
end

function DAG(n_nodes::Int; node_names::Vector{String}=String[])
    names = isempty(node_names) ? ["X$i" for i in 0:n_nodes-1] : node_names
    adj = zeros(Int8, n_nodes, n_nodes)
    DAG(n_nodes, names, adj)
end

function add_edge!(dag::DAG, i::Int, j::Int)
    dag.adjacency[i, j] = 1
end

function remove_edge!(dag::DAG, i::Int, j::Int)
    dag.adjacency[i, j] = 0
end

has_edge(dag::DAG, i::Int, j::Int) = dag.adjacency[i, j] == 1

function parents(dag::DAG, j::Int)
    Set(findall(==(1), dag.adjacency[:, j]))
end

function children(dag::DAG, i::Int)
    Set(findall(==(1), dag.adjacency[i, :]))
end

function topological_order(dag::DAG)
    n = dag.n_nodes
    in_degree = vec(sum(dag.adjacency, dims=1))
    order = Int[]
    queue = findall(==(0), in_degree)

    while !isempty(queue)
        node = popfirst!(queue)
        push!(order, node)
        for child in children(dag, node)
            in_degree[child] -= 1
            if in_degree[child] == 0
                push!(queue, child)
            end
        end
    end

    length(order) == n || error("Graph has cycle")
    order
end

function is_acyclic(dag::DAG)
    try
        topological_order(dag)
        true
    catch
        false
    end
end


"""
    CPDAG

Completed partially directed acyclic graph (Markov equivalence class).

# Fields
- `n_nodes::Int`: Number of nodes
- `node_names::Vector{String}`: Node names
- `directed::Matrix{Int8}`: Compelled (directed) edges
- `undirected::Matrix{Int8}`: Reversible (undirected) edges
"""
mutable struct CPDAG
    n_nodes::Int
    node_names::Vector{String}
    directed::Matrix{Int8}
    undirected::Matrix{Int8}
end

function CPDAG(n_nodes::Int; node_names::Vector{String}=String[])
    names = isempty(node_names) ? ["X$i" for i in 0:n_nodes-1] : node_names
    CPDAG(n_nodes, names, zeros(Int8, n_nodes, n_nodes), zeros(Int8, n_nodes, n_nodes))
end

function add_directed_edge!(cpdag::CPDAG, i::Int, j::Int)
    # Remove undirected if exists
    cpdag.undirected[i, j] = 0
    cpdag.undirected[j, i] = 0
    # Add directed
    cpdag.directed[i, j] = 1
end

function add_undirected_edge!(cpdag::CPDAG, i::Int, j::Int)
    cpdag.undirected[i, j] = 1
    cpdag.undirected[j, i] = 1
end

has_directed_edge(cpdag::CPDAG, i::Int, j::Int) = cpdag.directed[i, j] == 1
has_undirected_edge(cpdag::CPDAG, i::Int, j::Int) = cpdag.undirected[i, j] == 1
has_any_edge(cpdag::CPDAG, i::Int, j::Int) = has_directed_edge(cpdag, i, j) ||
                                               has_directed_edge(cpdag, j, i) ||
                                               has_undirected_edge(cpdag, i, j)


"""
    CITestResult

Result of conditional independence test.

# Fields
- `independent::Bool`: Whether X ⊥ Y | Z at significance level α
- `pvalue::Float64`: P-value of test
- `statistic::Float64`: Test statistic
- `alpha::Float64`: Significance level used
- `conditioning_set::Vector{Int}`: Indices of conditioning variables
"""
struct CITestResult
    independent::Bool
    pvalue::Float64
    statistic::Float64
    alpha::Float64
    conditioning_set::Vector{Int}
end


"""
    PCResult

Result from PC algorithm.

# Fields
- `cpdag::CPDAG`: Estimated CPDAG
- `skeleton::Graph`: Undirected skeleton
- `separating_sets::Dict`: Separating sets for removed edges
- `n_ci_tests::Int`: Number of CI tests performed
- `alpha::Float64`: Significance level used
"""
struct PCResult
    cpdag::CPDAG
    skeleton::Graph
    separating_sets::Dict{Tuple{Int,Int}, Set{Int}}
    n_ci_tests::Int
    alpha::Float64
end


"""
    LiNGAMResult

Result from LiNGAM algorithm.

# Fields
- `dag::DAG`: Estimated unique DAG
- `causal_order::Vector{Int}`: Causal ordering of nodes
- `adjacency_matrix::Matrix{Float64}`: Weighted adjacency matrix B
"""
struct LiNGAMResult
    dag::DAG
    causal_order::Vector{Int}
    adjacency_matrix::Matrix{Float64}
end

"""Compute causal order accuracy."""
function causal_order_accuracy(result::LiNGAMResult, true_order::Vector{Int})
    n = length(result.causal_order)
    correct = 0
    for i in 1:n
        for j in i+1:n
            # Check if relative order is preserved
            est_i = findfirst(==(result.causal_order[i]), result.causal_order)
            est_j = findfirst(==(result.causal_order[j]), result.causal_order)
            true_i = findfirst(==(result.causal_order[i]), true_order)
            true_j = findfirst(==(result.causal_order[j]), true_order)

            if (est_i < est_j) == (true_i < true_j)
                correct += 1
            end
        end
    end
    n_pairs = n * (n - 1) ÷ 2
    n_pairs > 0 ? correct / n_pairs : 1.0
end

# =============================================================================
# FCI Types (Session 134)
# =============================================================================

export EdgeMark, PAG, FCIResult
export add_edge!, remove_edge!, has_edge, adjacent
export get_endpoint, set_endpoint!, is_definitely_directed
export n_directed_edges, n_bidirected_edges, n_circle_edges


"""
    EdgeMark

Edge endpoint marks for PAG edges.

- `NONE`: No edge
- `TAIL`: Definite tail (-)
- `ARROW`: Definite arrowhead (>)
- `CIRCLE`: Unknown (o)
"""
@enum EdgeMark begin
    NONE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3
end


"""
    PAG

Partial Ancestral Graph for FCI algorithm output.

# Fields
- `n_nodes::Int`: Number of observed variables
- `node_names::Vector{String}`: Node names
- `endpoints::Array{Int8,3}`: Endpoint matrix (n_nodes, n_nodes, 2)
  - endpoints[i,j,1] = mark at i for edge i-j
  - endpoints[i,j,2] = mark at j for edge i-j

Mark encoding: 0=none, 1=tail, 2=arrow, 3=circle
"""
mutable struct PAG
    n_nodes::Int
    node_names::Vector{String}
    endpoints::Array{Int8,3}
end

function PAG(n_nodes::Int; node_names::Vector{String}=String[])
    names = isempty(node_names) ? ["X$i" for i in 0:n_nodes-1] : node_names
    endpoints = zeros(Int8, n_nodes, n_nodes, 2)
    PAG(n_nodes, names, endpoints)
end

"""Create PAG from skeleton with all circle marks."""
function PAG(skeleton::Graph)
    n = skeleton.n_nodes
    pag = PAG(n; node_names=skeleton.node_names)
    for (i, j) in edges(skeleton)
        add_edge!(pag, i, j, CIRCLE, CIRCLE)
    end
    pag
end

function add_edge!(pag::PAG, i::Int, j::Int, mark_i::EdgeMark, mark_j::EdgeMark)
    pag.endpoints[i, j, 1] = Int8(mark_i)
    pag.endpoints[i, j, 2] = Int8(mark_j)
    pag.endpoints[j, i, 1] = Int8(mark_j)
    pag.endpoints[j, i, 2] = Int8(mark_i)
end

function remove_edge!(pag::PAG, i::Int, j::Int)
    pag.endpoints[i, j, :] .= 0
    pag.endpoints[j, i, :] .= 0
end

has_edge(pag::PAG, i::Int, j::Int) = any(pag.endpoints[i, j, :] .> 0)

function get_endpoint(pag::PAG, i::Int, j::Int)::EdgeMark
    EdgeMark(pag.endpoints[i, j, 1])
end

function set_endpoint!(pag::PAG, i::Int, j::Int, mark::EdgeMark)
    pag.endpoints[i, j, 1] = Int8(mark)
    pag.endpoints[j, i, 2] = Int8(mark)
end

function is_definitely_directed(pag::PAG, i::Int, j::Int)
    pag.endpoints[i, j, 1] == Int8(TAIL) && pag.endpoints[i, j, 2] == Int8(ARROW)
end

function adjacent(pag::PAG, i::Int)
    Set(j for j in 1:pag.n_nodes if j != i && has_edge(pag, i, j))
end

function n_edges(pag::PAG)
    count = 0
    for i in 1:pag.n_nodes
        for j in i+1:pag.n_nodes
            has_edge(pag, i, j) && (count += 1)
        end
    end
    count
end

function n_directed_edges(pag::PAG)
    count = 0
    for i in 1:pag.n_nodes
        for j in 1:pag.n_nodes
            i != j && is_definitely_directed(pag, i, j) && (count += 1)
        end
    end
    count
end

function n_bidirected_edges(pag::PAG)
    count = 0
    for i in 1:pag.n_nodes
        for j in i+1:pag.n_nodes
            if pag.endpoints[i, j, 1] == Int8(ARROW) && pag.endpoints[i, j, 2] == Int8(ARROW)
                count += 1
            end
        end
    end
    count
end

function n_circle_edges(pag::PAG)
    count = 0
    for i in 1:pag.n_nodes
        for j in i+1:pag.n_nodes
            if has_edge(pag, i, j)
                if pag.endpoints[i, j, 1] == Int8(CIRCLE) || pag.endpoints[i, j, 2] == Int8(CIRCLE)
                    count += 1
                end
            end
        end
    end
    count
end


"""
    FCIResult

Result from FCI algorithm.

# Fields
- `pag::PAG`: Estimated Partial Ancestral Graph
- `skeleton::Graph`: Undirected skeleton
- `separating_sets::Dict`: Separating sets for removed edges
- `possible_latent_confounders::Vector{Tuple{Int,Int}}`: Bidirected edges
- `n_ci_tests::Int`: Number of CI tests performed
- `alpha::Float64`: Significance level used
"""
struct FCIResult
    pag::PAG
    skeleton::Graph
    separating_sets::Dict{Tuple{Int,Int}, Set{Int}}
    possible_latent_confounders::Vector{Tuple{Int,Int}}
    n_ci_tests::Int
    alpha::Float64
end


end # module
