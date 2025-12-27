#=
Causal Forest for CATE Estimation

Simplified honest forest implementation following:
- Athey & Imbens (2016). "Recursive partitioning for heterogeneous causal effects"
- Wager & Athey (2018). "Estimation and inference of heterogeneous treatment effects"

Key Features:
- Honest splitting: structure/estimation sample separation
- CATE-specific splitting criterion (heterogeneity maximization)
- Bootstrap variance estimation
- Forest aggregation for stable predictions

Session 157: Causal Forest Julia Parity.
=#

using Random
using Statistics
using LinearAlgebra


# =============================================================================
# Tree Node Structure
# =============================================================================

"""
    CausalTreeNode

Internal node structure for honest causal trees.

# Fields
- `is_leaf::Bool`: Whether this is a terminal node
- `split_var::Int`: Variable index for splitting (0 if leaf)
- `split_val::Float64`: Threshold value for splitting
- `left::Union{CausalTreeNode, Nothing}`: Left child (X[:, split_var] <= split_val)
- `right::Union{CausalTreeNode, Nothing}`: Right child (X[:, split_var] > split_val)
- `tau::Float64`: Estimated CATE at this node (populated for leaves)
- `n_treat::Int`: Number of treated units in estimation sample (for leaves)
- `n_control::Int`: Number of control units in estimation sample (for leaves)
"""
mutable struct CausalTreeNode
    is_leaf::Bool
    split_var::Int
    split_val::Float64
    left::Union{CausalTreeNode, Nothing}
    right::Union{CausalTreeNode, Nothing}
    tau::Float64
    n_treat::Int
    n_control::Int

    # Leaf constructor
    function CausalTreeNode()
        new(true, 0, 0.0, nothing, nothing, 0.0, 0, 0)
    end

    # Internal node constructor
    function CausalTreeNode(split_var::Int, split_val::Float64,
                            left::CausalTreeNode, right::CausalTreeNode)
        new(false, split_var, split_val, left, right, 0.0, 0, 0)
    end
end


# =============================================================================
# Leaf-Level CATE Estimation
# =============================================================================

"""
    _estimate_leaf_tau(Y, T) -> Float64

Estimate treatment effect at a leaf node.

Uses simple difference in means: τ̂ = Ȳ₁ - Ȳ₀

Returns 0.0 if either group is empty (for robustness).
"""
function _estimate_leaf_tau(Y::AbstractVector{<:Real}, T::AbstractVector{Bool})
    treated_mask = T
    control_mask = .!T

    n_treat = sum(treated_mask)
    n_control = sum(control_mask)

    # Return 0 if either group is empty
    if n_treat == 0 || n_control == 0
        return 0.0
    end

    y_treat = mean(Y[treated_mask])
    y_control = mean(Y[control_mask])

    return y_treat - y_control
end


"""
    _leaf_has_both_groups(T, min_per_group=1) -> Bool

Check if a leaf has enough treated and control units.
"""
function _leaf_has_both_groups(T::AbstractVector{Bool}; min_per_group::Int=1)
    n_treat = sum(T)
    n_control = length(T) - n_treat
    return n_treat >= min_per_group && n_control >= min_per_group
end


# =============================================================================
# CATE Split Criterion
# =============================================================================

"""
    _heterogeneity_gain(Y_left, T_left, Y_right, T_right) -> Float64

Compute heterogeneity gain from a split.

Criterion: Maximize squared difference in CATE between children,
weighted by sample sizes.

    gain = (n_L * n_R / (n_L + n_R)²) * (τ̂_L - τ̂_R)²

Returns -Inf if split is invalid (empty groups).
"""
function _heterogeneity_gain(
    Y_left::AbstractVector{<:Real}, T_left::AbstractVector{Bool},
    Y_right::AbstractVector{<:Real}, T_right::AbstractVector{Bool}
)
    n_left = length(Y_left)
    n_right = length(Y_right)

    # Need both children to have data
    if n_left == 0 || n_right == 0
        return -Inf
    end

    # Need both groups in each child for CATE estimation
    if !_leaf_has_both_groups(T_left) || !_leaf_has_both_groups(T_right)
        return -Inf
    end

    tau_left = _estimate_leaf_tau(Y_left, T_left)
    tau_right = _estimate_leaf_tau(Y_right, T_right)

    # Weighted squared difference in CATE
    n_total = n_left + n_right
    weight = (n_left * n_right) / (n_total^2)

    return weight * (tau_left - tau_right)^2
end


# =============================================================================
# Best Split Search
# =============================================================================

"""
    _find_best_split(Y, T, X, mtry, min_leaf_size) -> (var, val, gain)

Find the best split among candidate variables.

# Arguments
- `Y`: Outcome vector
- `T`: Treatment vector
- `X`: Covariate matrix
- `mtry`: Number of candidate variables (0 = all)
- `min_leaf_size`: Minimum observations per leaf

# Returns
- `best_var::Int`: Index of best split variable (0 if no valid split)
- `best_val::Float64`: Best split threshold
- `best_gain::Float64`: Heterogeneity gain at best split
"""
function _find_best_split(
    Y::AbstractVector{<:Real},
    T::AbstractVector{Bool},
    X::AbstractMatrix{<:Real};
    mtry::Int=0,
    min_leaf_size::Int=10
)
    n, p = size(X)
    min_n_per_side = min_leaf_size

    # Select candidate variables
    if mtry <= 0 || mtry >= p
        candidate_vars = 1:p
    else
        candidate_vars = randperm(p)[1:mtry]
    end

    best_var = 0
    best_val = 0.0
    best_gain = -Inf

    for j in candidate_vars
        x_j = @view X[:, j]

        # Get unique split candidates (midpoints)
        sorted_vals = sort(unique(x_j))
        if length(sorted_vals) < 2
            continue
        end

        # Try midpoints between consecutive values
        for i in 1:(length(sorted_vals) - 1)
            split_val = (sorted_vals[i] + sorted_vals[i + 1]) / 2

            left_mask = x_j .<= split_val
            right_mask = .!left_mask

            n_left = sum(left_mask)
            n_right = sum(right_mask)

            # Check minimum leaf size
            if n_left < min_n_per_side || n_right < min_n_per_side
                continue
            end

            # Compute gain
            gain = _heterogeneity_gain(
                Y[left_mask], T[left_mask],
                Y[right_mask], T[right_mask]
            )

            if gain > best_gain
                best_gain = gain
                best_var = j
                best_val = split_val
            end
        end
    end

    return best_var, best_val, best_gain
end


# =============================================================================
# Tree Building (Structure Phase)
# =============================================================================

"""
    _build_tree_structure(Y, T, X, depth, max_depth, mtry, min_leaf_size) -> CausalTreeNode

Recursively build tree structure using CATE splitting criterion.

This is the "structure" phase of honest splitting - determines where to split,
but leaf estimates come from a separate estimation sample.

# Arguments
- `Y, T, X`: Training data for determining splits
- `depth`: Current tree depth
- `max_depth`: Maximum allowed depth
- `mtry`: Number of candidate variables per split
- `min_leaf_size`: Minimum observations per leaf
"""
function _build_tree_structure(
    Y::AbstractVector{<:Real},
    T::AbstractVector{Bool},
    X::AbstractMatrix{<:Real};
    depth::Int=0,
    max_depth::Int=10,
    mtry::Int=0,
    min_leaf_size::Int=10
)
    n = length(Y)

    # Stopping conditions
    if depth >= max_depth || n < 2 * min_leaf_size
        return CausalTreeNode()
    end

    # Need both treated and control for splitting
    if !_leaf_has_both_groups(T, min_per_group=2)
        return CausalTreeNode()
    end

    # Find best split
    best_var, best_val, best_gain = _find_best_split(
        Y, T, X; mtry=mtry, min_leaf_size=min_leaf_size
    )

    # No valid split found
    if best_var == 0 || best_gain <= 0
        return CausalTreeNode()
    end

    # Split data
    x_split = @view X[:, best_var]
    left_mask = x_split .<= best_val
    right_mask = .!left_mask

    # Recursively build children
    left_child = _build_tree_structure(
        Y[left_mask], T[left_mask], X[left_mask, :];
        depth=depth + 1, max_depth=max_depth, mtry=mtry, min_leaf_size=min_leaf_size
    )

    right_child = _build_tree_structure(
        Y[right_mask], T[right_mask], X[right_mask, :];
        depth=depth + 1, max_depth=max_depth, mtry=mtry, min_leaf_size=min_leaf_size
    )

    return CausalTreeNode(best_var, best_val, left_child, right_child)
end


# =============================================================================
# Leaf Population (Estimation Phase)
# =============================================================================

"""
    _populate_leaves!(node, Y, T, X)

Populate leaf estimates using estimation sample (honest estimation).

Traverses tree structure and computes τ̂ at each leaf using the
estimation sample (separate from structure sample).
"""
function _populate_leaves!(
    node::CausalTreeNode,
    Y::AbstractVector{<:Real},
    T::AbstractVector{Bool},
    X::AbstractMatrix{<:Real}
)
    n = length(Y)

    if node.is_leaf || n == 0
        # Compute leaf estimate
        if n > 0
            node.tau = _estimate_leaf_tau(Y, T)
            node.n_treat = sum(T)
            node.n_control = n - node.n_treat
        end
        return
    end

    # Split estimation sample according to tree structure
    x_split = @view X[:, node.split_var]
    left_mask = x_split .<= node.split_val
    right_mask = .!left_mask

    # Recursively populate children
    _populate_leaves!(node.left, Y[left_mask], T[left_mask], X[left_mask, :])
    _populate_leaves!(node.right, Y[right_mask], T[right_mask], X[right_mask, :])
end


# =============================================================================
# Honest Tree Builder
# =============================================================================

"""
    build_honest_tree(Y, T, X; kwargs...) -> CausalTreeNode

Build an honest causal tree with separate structure and estimation samples.

# Algorithm (Athey & Imbens 2016)
1. Randomly split data into structure (S) and estimation (E) halves
2. Build tree structure on S using CATE splitting criterion
3. Populate leaf estimates using E (honest estimation)

# Arguments
- `Y::Vector`: Outcome variable
- `T::Vector{Bool}`: Treatment indicator
- `X::Matrix`: Covariates

# Keyword Arguments
- `min_leaf_size::Int=20`: Minimum observations per leaf
- `max_depth::Int=10`: Maximum tree depth
- `mtry::Int=0`: Variables per split (0 = sqrt(p))

# Returns
- `CausalTreeNode`: Root of honest causal tree
"""
function build_honest_tree(
    Y::AbstractVector{<:Real},
    T::AbstractVector{Bool},
    X::AbstractMatrix{<:Real};
    min_leaf_size::Int=20,
    max_depth::Int=10,
    mtry::Int=0
)
    n = length(Y)
    p = size(X, 2)

    # Set mtry default
    actual_mtry = mtry > 0 ? mtry : max(1, round(Int, sqrt(p)))

    # Split into structure and estimation halves
    perm = randperm(n)
    n_struct = n ÷ 2
    struct_idx = perm[1:n_struct]
    est_idx = perm[(n_struct + 1):end]

    # Structure sample
    Y_struct = Y[struct_idx]
    T_struct = T[struct_idx]
    X_struct = X[struct_idx, :]

    # Estimation sample
    Y_est = Y[est_idx]
    T_est = T[est_idx]
    X_est = X[est_idx, :]

    # Build tree structure on structure sample
    root = _build_tree_structure(
        Y_struct, T_struct, X_struct;
        depth=0, max_depth=max_depth, mtry=actual_mtry, min_leaf_size=min_leaf_size
    )

    # Populate leaves with estimation sample (honesty)
    _populate_leaves!(root, Y_est, T_est, X_est)

    return root
end


# =============================================================================
# Tree Prediction
# =============================================================================

"""
    predict_tree(node, x) -> Float64

Predict CATE for a single observation by traversing the tree.

# Arguments
- `node`: Root of causal tree
- `x`: Feature vector for single observation

# Returns
- `tau::Float64`: Predicted CATE
"""
function predict_tree(node::CausalTreeNode, x::AbstractVector{<:Real})
    # Traverse to leaf
    current = node
    while !current.is_leaf
        if x[current.split_var] <= current.split_val
            current = current.left
        else
            current = current.right
        end
    end

    return current.tau
end


"""
    predict_tree(node, X) -> Vector{Float64}

Predict CATE for multiple observations.
"""
function predict_tree(node::CausalTreeNode, X::AbstractMatrix{<:Real})
    n = size(X, 1)
    predictions = zeros(n)
    for i in 1:n
        predictions[i] = predict_tree(node, @view X[i, :])
    end
    return predictions
end


# =============================================================================
# Causal Forest Estimator
# =============================================================================

"""
    CausalForestEstimator <: AbstractCATEEstimator

Causal Forest for heterogeneous treatment effect estimation.

Implements an ensemble of honest causal trees following Wager & Athey (2018).

# Algorithm
1. For each tree b = 1, ..., B:
   a. Draw bootstrap sample (or subsample)
   b. Build honest causal tree with structure/estimation split
2. Aggregate predictions: τ̂(x) = (1/B) Σ τ̂ᵦ(x)
3. Estimate variance via bootstrap over trees

# Fields
- `n_trees::Int`: Number of trees in forest (default: 100)
- `min_leaf_size::Int`: Minimum observations per leaf (default: 20)
- `max_depth::Int`: Maximum tree depth (default: 10)
- `mtry::Int`: Variables considered per split (default: 0 = sqrt(p))
- `bootstrap::Bool`: Use bootstrap sampling (default: true)
- `subsample_ratio::Float64`: Subsample ratio if not bootstrap (default: 0.7)

# Example
```julia
estimator = CausalForestEstimator(n_trees=200, min_leaf_size=15)
solution = solve(problem, estimator)
```

# References
- Wager, S., & Athey, S. (2018). "Estimation and inference of heterogeneous
  treatment effects using random forests". JASA.
- Athey, S., & Imbens, G. (2016). "Recursive partitioning for heterogeneous
  causal effects". PNAS.
"""
struct CausalForestEstimator <: AbstractCATEEstimator
    n_trees::Int
    min_leaf_size::Int
    max_depth::Int
    mtry::Int
    bootstrap::Bool
    subsample_ratio::Float64

    function CausalForestEstimator(;
        n_trees::Int=100,
        min_leaf_size::Int=20,
        max_depth::Int=10,
        mtry::Int=0,
        bootstrap::Bool=true,
        subsample_ratio::Float64=0.7
    )
        if n_trees < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid CausalForestEstimator configuration.\n" *
                "Function: CausalForestEstimator\n" *
                "Parameter: n_trees must be >= 1, got $n_trees"
            ))
        end
        if min_leaf_size < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid CausalForestEstimator configuration.\n" *
                "Function: CausalForestEstimator\n" *
                "Parameter: min_leaf_size must be >= 1, got $min_leaf_size"
            ))
        end
        if max_depth < 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid CausalForestEstimator configuration.\n" *
                "Function: CausalForestEstimator\n" *
                "Parameter: max_depth must be >= 1, got $max_depth"
            ))
        end
        if subsample_ratio <= 0 || subsample_ratio > 1
            throw(ArgumentError(
                "CRITICAL ERROR: Invalid CausalForestEstimator configuration.\n" *
                "Function: CausalForestEstimator\n" *
                "Parameter: subsample_ratio must be in (0, 1], got $subsample_ratio"
            ))
        end
        new(n_trees, min_leaf_size, max_depth, mtry, bootstrap, subsample_ratio)
    end
end


# =============================================================================
# Bootstrap Variance Estimation
# =============================================================================

"""
    _bootstrap_variance(trees, X; n_bootstrap=50) -> Vector{Float64}

Estimate variance of CATE predictions via bootstrap over trees.

# Arguments
- `trees`: Vector of trained causal trees
- `X`: Covariate matrix for prediction
- `n_bootstrap`: Number of bootstrap samples

# Returns
- `se::Vector{Float64}`: Standard error estimate for each observation
"""
function _bootstrap_variance(
    trees::Vector{CausalTreeNode},
    X::AbstractMatrix{<:Real};
    n_bootstrap::Int=50
)
    n = size(X, 1)
    B = length(trees)

    # Collect bootstrap predictions
    boot_preds = zeros(n, n_bootstrap)

    for b in 1:n_bootstrap
        # Sample trees with replacement
        tree_idx = rand(1:B, B)

        # Aggregate predictions from sampled trees
        for i in 1:n
            x_i = @view X[i, :]
            boot_preds[i, b] = mean(predict_tree(trees[t], x_i) for t in tree_idx)
        end
    end

    # Standard deviation across bootstrap samples
    se = std(boot_preds, dims=2)[:]

    return se
end


# =============================================================================
# Solve Method
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::CausalForestEstimator) -> CATESolution

Estimate CATE using a Causal Forest.

# Algorithm
1. Build ensemble of honest causal trees
2. For each observation, predict CATE by averaging over trees
3. Estimate standard errors via bootstrap
4. Compute ATE as average of CATE predictions

# Arguments
- `problem::CATEProblem`: CATE estimation problem
- `estimator::CausalForestEstimator`: Forest configuration

# Returns
- `CATESolution`: CATE estimates with confidence intervals

# Example
```julia
problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, CausalForestEstimator(n_trees=100))

println("ATE: \$(solution.ate) ± \$(1.96 * solution.se)")
println("CATE range: [\$(minimum(solution.cate)), \$(maximum(solution.cate))]")
```
"""
function solve(
    problem::CATEProblem,
    estimator::CausalForestEstimator
)
    Y = problem.outcomes
    T = problem.treatment
    X = problem.covariates
    alpha = problem.parameters.alpha

    n, p = size(X)

    # Convert treatment to Bool if needed
    T_bool = T .> 0.5

    # Build forest
    trees = Vector{CausalTreeNode}(undef, estimator.n_trees)

    for b in 1:estimator.n_trees
        # Sample data
        if estimator.bootstrap
            # Bootstrap: sample with replacement
            sample_idx = rand(1:n, n)
        else
            # Subsample: sample without replacement
            n_sample = max(2 * estimator.min_leaf_size, round(Int, n * estimator.subsample_ratio))
            sample_idx = randperm(n)[1:n_sample]
        end

        Y_sample = Y[sample_idx]
        T_sample = T_bool[sample_idx]
        X_sample = X[sample_idx, :]

        # Build honest tree
        trees[b] = build_honest_tree(
            Y_sample, T_sample, X_sample;
            min_leaf_size=estimator.min_leaf_size,
            max_depth=estimator.max_depth,
            mtry=estimator.mtry
        )
    end

    # Predict CATE for all observations
    cate = zeros(n)
    for i in 1:n
        x_i = @view X[i, :]
        cate[i] = mean(predict_tree(t, x_i) for t in trees)
    end

    # Estimate variance via bootstrap over trees
    se_cate = _bootstrap_variance(trees, X; n_bootstrap=50)

    # Aggregate to ATE
    ate = mean(cate)

    # SE of ATE: combine individual SEs (assuming independence)
    # Conservative: sqrt(mean(se_cate^2)) / sqrt(n) + sd(cate) / sqrt(n)
    ate_se = sqrt(mean(se_cate .^ 2) / n + var(cate) / n)

    # Confidence interval
    z = quantile(Normal(), 1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATESolution(
        cate,
        ate,
        ate_se,
        ci_lower,
        ci_upper,
        :causal_forest,
        :Success,
        problem
    )
end
