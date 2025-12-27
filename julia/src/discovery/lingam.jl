"""
    LiNGAM - Linear Non-Gaussian Acyclic Model

Session 133: Functional causal discovery using ICA.

# Functions
- `direct_lingam`: DirectLiNGAM algorithm
- `ica_lingam`: ICA-based LiNGAM
"""
module LiNGAM

using LinearAlgebra
using Statistics
using Random

# Import from sibling modules (already included by Discovery.jl)
using ..DiscoveryTypes

export direct_lingam, ica_lingam


"""
    direct_lingam(data; seed=nothing)

DirectLiNGAM: Direct method without ICA iteration.

Determines causal order by iteratively finding exogenous variables.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `seed::Union{Int,Nothing}`: Random seed for tie-breaking

# Returns
- `LiNGAMResult`: Result with unique DAG and causal ordering

# Example
```julia
data = randn(1000, 5)
result = direct_lingam(data)
println("Causal order: ", result.causal_order)
```
"""
function direct_lingam(data::Matrix{Float64}; seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)

    n_samples, n_vars = size(data)

    # Center data
    X = data .- mean(data, dims=1)

    # Track remaining variables
    remaining = collect(1:n_vars)
    causal_order = Int[]
    residuals = copy(X)

    # Iteratively find causal order
    while !isempty(remaining)
        exog_scores = Float64[]

        for i in remaining
            # Regress variable i on all other remaining
            other_vars = filter(j -> j != i, remaining)

            if isempty(other_vars)
                push!(exog_scores, 0.0)
                continue
            end

            # Get residual of i regressed on others
            X_others = residuals[:, other_vars]
            y_i = residuals[:, i]

            # OLS regression
            coeffs = X_others \ y_i
            resid_i = y_i - X_others * coeffs

            # Compute independence score
            total_dep = 0.0
            for j in other_vars
                r_j = residuals[:, j]
                corr_val = cor(resid_i, r_j)
                total_dep += abs(corr_val)
            end
            push!(exog_scores, total_dep)
        end

        # Select most exogenous variable
        _, min_idx = findmin(exog_scores)
        next_var = remaining[min_idx]

        push!(causal_order, next_var)
        filter!(x -> x != next_var, remaining)

        # Remove effect from remaining variables
        if !isempty(remaining)
            x_next = residuals[:, next_var]
            x_var = var(x_next)
            for j in remaining
                y_j = residuals[:, j]
                coef = cov(y_j, x_next) / (x_var + 1e-10)
                residuals[:, j] = y_j - coef * x_next
            end
        end
    end

    # Estimate adjacency matrix
    B = estimate_adjacency(data, causal_order)

    # Build DAG
    dag = DAG(n_vars)
    for i in 1:n_vars
        for j in 1:n_vars
            if abs(B[i, j]) > 1e-10
                add_edge!(dag, i, j)
            end
        end
    end

    LiNGAMResult(dag, causal_order, B)
end


"""Estimate adjacency matrix given causal ordering."""
function estimate_adjacency(data::Matrix{Float64}, causal_order::Vector{Int})
    n_vars = length(causal_order)
    n_samples = size(data, 1)
    B = zeros(n_vars, n_vars)

    X = data .- mean(data, dims=1)

    for (idx, j) in enumerate(causal_order)
        idx == 1 && continue  # First variable has no parents

        potential_parents = causal_order[1:idx-1]
        X_parents = X[:, potential_parents]
        y_j = X[:, j]

        # OLS regression
        coeffs = X_parents \ y_j

        # Residuals for significance test
        residuals = y_j - X_parents * coeffs
        mse = var(residuals)

        # Standard errors
        XtX_inv = pinv(X_parents' * X_parents)
        se = sqrt.(diag(XtX_inv) .* mse)

        for (k_idx, parent) in enumerate(potential_parents)
            if abs(coeffs[k_idx]) > 2 * se[k_idx]  # ~95% significance
                B[parent, j] = coeffs[k_idx]
            end
        end
    end

    B
end


"""
    ica_lingam(data; seed=nothing, max_iter=1000, tol=1e-6)

ICA-based LiNGAM for causal discovery.

# Arguments
- `data::Matrix{Float64}`: Data matrix
- `seed::Union{Int,Nothing}`: Random seed
- `max_iter::Int`: Maximum ICA iterations
- `tol::Float64`: Convergence tolerance

# Returns
- `LiNGAMResult`: Result with unique DAG
"""
function ica_lingam(data::Matrix{Float64};
                    seed::Union{Int,Nothing}=nothing,
                    max_iter::Int=1000,
                    tol::Float64=1e-6)
    !isnothing(seed) && Random.seed!(seed)

    n_samples, n_vars = size(data)

    # Center and whiten
    X_centered = data .- mean(data, dims=1)
    X_whitened, whitening_matrix = whiten(X_centered)

    # Perform FastICA
    W_ica = fastica(X_whitened, n_vars; max_iter=max_iter, tol=tol)

    # Full unmixing matrix
    W = W_ica * whitening_matrix

    # Find causal order
    A = inv(W)
    causal_order, P = find_causal_order(A)

    # Extract adjacency matrix
    A_permuted = P * A * P'
    B_perm = I - inv(A_permuted)
    B_perm = tril(B_perm, -1)  # Strictly lower triangular

    # Permute back
    P_inv = P'
    B = P_inv * B_perm * P_inv'

    # Prune small edges
    threshold = 0.1
    B[abs.(B) .< threshold] .= 0

    # Build DAG
    dag = DAG(n_vars)
    for i in 1:n_vars
        for j in 1:n_vars
            if abs(B[i, j]) > 1e-10
                add_edge!(dag, i, j)
            end
        end
    end

    LiNGAMResult(dag, causal_order, B)
end


"""Whiten data to have identity covariance."""
function whiten(X::Matrix{Float64})
    C = cov(X)
    eigenvalues, eigenvectors = eigen(C)
    eigenvalues = max.(eigenvalues, 1e-10)

    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(eigenvalues))
    W = D_inv_sqrt * eigenvectors'

    X * W', W
end


"""FastICA algorithm."""
function fastica(X::Matrix{Float64}, n_components::Int;
                 max_iter::Int=1000, tol::Float64=1e-6)
    n_samples, _ = size(X)
    W = zeros(n_components, n_components)

    for p in 1:n_components
        w = randn(n_components)
        w = w / norm(w)

        for _ in 1:max_iter
            w_old = copy(w)

            # Newton iteration with G(u) = log(cosh(u))
            wx = X * w
            g_wx = tanh.(wx)
            g_prime_wx = 1 .- g_wx.^2

            w = (X' * g_wx) / n_samples - mean(g_prime_wx) * w

            # Decorrelate from previous components
            if p > 1
                w = w - W[1:p-1, :]' * (W[1:p-1, :] * w)
            end

            w = w / norm(w)

            if abs(abs(dot(w, w_old)) - 1) < tol
                break
            end
        end

        W[p, :] = w
    end

    W
end


"""Find causal ordering from mixing matrix."""
function find_causal_order(A::Matrix{Float64})
    n = size(A, 1)
    remaining = Set(1:n)
    causal_order = Int[]
    A_work = copy(A)

    for _ in 1:n
        # Find row with minimum normalized influence
        row_norms = Float64[]
        indices = Int[]

        for i in remaining
            row = copy(A_work[i, :])
            if abs(A_work[i, i]) > 1e-10
                row = row / abs(A_work[i, i])
            end
            row[i] = 0
            push!(row_norms, sum(abs.(row)))
            push!(indices, i)
        end

        _, min_idx = findmin(row_norms)
        next_var = indices[min_idx]

        push!(causal_order, next_var)
        delete!(remaining, next_var)

        # Update working matrix
        if !isempty(remaining)
            for i in remaining
                if abs(A_work[next_var, next_var]) > 1e-10
                    A_work[i, :] -= (A_work[i, next_var] / A_work[next_var, next_var]) *
                                    A_work[next_var, :]
                end
            end
        end
    end

    # Build permutation matrix
    P = zeros(n, n)
    for (new_idx, old_idx) in enumerate(causal_order)
        P[new_idx, old_idx] = 1.0
    end

    causal_order, P
end

end # module
