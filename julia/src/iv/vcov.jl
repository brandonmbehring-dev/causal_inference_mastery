"""
Variance-Covariance Matrix Computation for 2SLS Estimators.

Provides reusable variance-covariance estimators for two-stage least squares (2SLS)
instrumental variables regression. Three types of standard errors are supported:

1. Standard (homoskedastic): Assumes constant error variance
2. Robust (heteroskedasticity-robust): White/HC0 sandwich estimator
3. Clustered (cluster-robust): Accounts for within-cluster correlation

# Mathematical Framework

- Standard: V = σ² (X'P_Z X)⁻¹
- Robust:   V = (X'P_Z X)⁻¹ (X'P_Z Ω P_Z X) (X'P_Z X)⁻¹, where Ω = diag(e²)
- Clustered: V = (X'P_Z X)⁻¹ (Σ_g X_g'P_Z e_g e_g'P_Z X_g) (X'P_Z X)⁻¹

where:
- X: Design matrix [D, controls]
- P_Z: Projection matrix onto instruments Z
- e: Second-stage residuals
- g: Cluster index

# References

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator
  and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817-838.
- Cameron, A.C., and D.L. Miller (2015). "A Practitioner's Guide to Cluster-Robust
  Inference." *Journal of Human Resources*, 50(2), 317-372.
- Wooldridge, J.M. (2010). "Econometric Analysis of Cross Section and Panel Data",
  2nd ed., Chapter 8.

Session 56: Ported from Python vcov.py for cross-language parity.
"""

using LinearAlgebra


"""
    compute_standard_vcov(XPX_inv::AbstractMatrix, sigma2::Real) → Matrix

Compute standard homoskedastic variance-covariance matrix for 2SLS.

# Formula

V = σ² (X'P_Z X)⁻¹

Assumes constant error variance (homoskedasticity). This is the classical
formula but is invalid if heteroskedasticity is present.

# Arguments

- `XPX_inv::AbstractMatrix`: Inverse of (X'P_Z X) matrix, where X is the design
  matrix [D, controls, constant] and P_Z is the projection matrix onto instruments.
- `sigma2::Real`: Residual variance σ² = e'e / (n - k)

# Returns

- `Matrix{Float64}`: Variance-covariance matrix of shape (k, k)

# Notes

This estimator is consistent only under homoskedasticity.
Use robust or clustered SEs if heteroskedasticity is suspected.

# Examples

```julia
XPX_inv = inv(X' * P_Z * X)
sigma2 = sum(residuals.^2) / (n - k)
vcov = compute_standard_vcov(XPX_inv, sigma2)
se = sqrt.(diag(vcov))
```
"""
function compute_standard_vcov(XPX_inv::AbstractMatrix{T}, sigma2::Real) where {T<:Real}
    return sigma2 * XPX_inv
end


"""
    compute_robust_vcov(XPX_inv, DX, P_Z, residuals) → Matrix

Compute heteroskedasticity-robust variance-covariance matrix (White/HC0).

# Formula

V = (X'P_Z X)⁻¹ (X'P_Z Ω P_Z X) (X'P_Z X)⁻¹

where Ω = diag(e²)

This is the White (1980) sandwich estimator, also known as HC0.
It is consistent even with heteroskedasticity but may undercover
in finite samples (use HC1, HC2, or HC3 for finite-sample corrections).

# Arguments

- `XPX_inv::AbstractMatrix`: Inverse of (X'P_Z X) matrix
- `DX::AbstractMatrix`: Design matrix [D, X, constant] of shape (n, k)
- `P_Z::AbstractMatrix`: Projection matrix onto instruments of shape (n, n)
- `residuals::AbstractVector`: Second-stage residuals e = Y - X'β̂ of shape (n,)

# Returns

- `Matrix{Float64}`: Robust variance-covariance matrix of shape (k, k)

# Notes

The sandwich formula has three parts:
1. Bread: (X'P_Z X)⁻¹
2. Meat: X'P_Z Ω P_Z X, where Ω = diag(e²)
3. Bread: (X'P_Z X)⁻¹

# Examples

```julia
# After 2SLS estimation
vcov_robust = compute_robust_vcov(XPX_inv, DX, P_Z, residuals)
se_robust = sqrt.(diag(vcov_robust))
```
"""
function compute_robust_vcov(
    XPX_inv::AbstractMatrix{T},
    DX::AbstractMatrix{T},
    P_Z::AbstractMatrix{T},
    residuals::AbstractVector{T},
) where {T<:Real}
    # Ω = diag(e²)
    Omega = Diagonal(residuals .^ 2)

    # Meat: X'P_Z Ω P_Z X
    meat = DX' * P_Z * Omega * P_Z * DX

    # Sandwich: bread × meat × bread
    return XPX_inv * meat * XPX_inv
end


"""
    compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n) → Matrix

Compute cluster-robust variance-covariance matrix.

# Formula

V = (X'P_Z X)⁻¹ (Σ_g X_g'P_Z e_g e_g'P_Z X_g) (X'P_Z X)⁻¹

with finite-sample correction: (G / (G - 1)) × ((n - 1) / (n - k))

Accounts for arbitrary correlation within clusters while assuming
independence across clusters.

# Arguments

- `XPX_inv::AbstractMatrix`: Inverse of (X'P_Z X) matrix
- `DX::AbstractMatrix`: Design matrix [D, X, constant] of shape (n, k)
- `P_Z::AbstractMatrix`: Projection matrix onto instruments of shape (n, n)
- `residuals::AbstractVector`: Second-stage residuals of shape (n,)
- `clusters::AbstractVector`: Cluster identifiers (e.g., school IDs, firm IDs)
- `n::Int`: Number of observations

# Returns

- `Matrix{Float64}`: Cluster-robust variance-covariance matrix of shape (k, k)

# Warnings

Issues a warning if number of clusters < 20 (clustered SEs unreliable).

# Notes

Cluster-robust inference requires:
1. Many clusters (G ≥ 50 recommended, G ≥ 20 minimum)
2. Balanced cluster sizes (unbalanced OK, but avoid one huge cluster)
3. Independence across clusters

With few clusters (G < 20), t-tests and F-tests become unreliable.
Consider:
- Wild cluster bootstrap
- Robust SEs instead
- Aggregating to cluster level

The finite-sample correction:
- (G / (G - 1)): Corrects for estimating cluster means
- ((n - 1) / (n - k)): Corrects for estimating k parameters

# Examples

```julia
# After 2SLS estimation with clustered data
vcov_cluster = compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n)
se_cluster = sqrt.(diag(vcov_cluster))
```
"""
function compute_clustered_vcov(
    XPX_inv::AbstractMatrix{T},
    DX::AbstractMatrix{T},
    P_Z::AbstractMatrix{T},
    residuals::AbstractVector{T},
    clusters::AbstractVector,
    n::Int,
) where {T<:Real}
    unique_clusters = unique(clusters)
    G = length(unique_clusters)
    k = size(DX, 2)

    # Warn if too few clusters
    if G < 20
        @warn "Only $G clusters. Clustered standard errors may be unreliable with <20 clusters. " *
              "Consider using robust SEs instead or wild cluster bootstrap."
    end

    # Compute cluster-robust meat
    meat = zeros(T, k, k)

    for g in unique_clusters
        cluster_mask = clusters .== g
        DX_g = DX[cluster_mask, :]
        e_g = residuals[cluster_mask]

        # P_Z @ DX for cluster g observations
        PZ_DX_g = P_Z[cluster_mask, :] * DX

        # Outer product contribution: (P_Z X_g)' e_g e_g' (P_Z X_g)
        meat .+= PZ_DX_g' * (e_g * e_g') * PZ_DX_g
    end

    # Apply finite-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k))

    return correction * XPX_inv * meat * XPX_inv
end


"""
    compute_vcov(XPX_inv, DX, P_Z, residuals; method=:robust, clusters=nothing, n=nothing) → Matrix

Unified interface for computing variance-covariance matrices.

# Arguments

- `XPX_inv::AbstractMatrix`: Inverse of (X'P_Z X) matrix
- `DX::AbstractMatrix`: Design matrix of shape (n, k)
- `P_Z::AbstractMatrix`: Projection matrix onto instruments
- `residuals::AbstractVector`: Second-stage residuals

# Keyword Arguments

- `method::Symbol=:robust`: Inference method (:standard, :robust, or :clustered)
- `clusters::Union{AbstractVector, Nothing}=nothing`: Cluster identifiers (required for :clustered)
- `n::Union{Int, Nothing}=nothing`: Number of observations (inferred from residuals if not provided)

# Returns

- `Matrix{Float64}`: Variance-covariance matrix

# Examples

```julia
# Robust SEs (default)
vcov = compute_vcov(XPX_inv, DX, P_Z, residuals)

# Standard (homoskedastic) SEs
sigma2 = sum(residuals.^2) / (length(residuals) - size(DX, 2))
vcov = compute_vcov(XPX_inv, DX, P_Z, residuals; method=:standard)

# Clustered SEs
vcov = compute_vcov(XPX_inv, DX, P_Z, residuals; method=:clustered, clusters=firm_ids)
```
"""
function compute_vcov(
    XPX_inv::AbstractMatrix{T},
    DX::AbstractMatrix{T},
    P_Z::AbstractMatrix{T},
    residuals::AbstractVector{T};
    method::Symbol = :robust,
    clusters::Union{AbstractVector, Nothing} = nothing,
    n::Union{Int, Nothing} = nothing,
) where {T<:Real}
    # Infer n if not provided
    n_obs = isnothing(n) ? length(residuals) : n
    k = size(DX, 2)

    if method == :standard
        sigma2 = sum(residuals .^ 2) / (n_obs - k)
        return compute_standard_vcov(XPX_inv, sigma2)
    elseif method == :robust
        return compute_robust_vcov(XPX_inv, DX, P_Z, residuals)
    elseif method == :clustered
        if isnothing(clusters)
            throw(ArgumentError("clusters must be provided for method=:clustered"))
        end
        return compute_clustered_vcov(XPX_inv, DX, P_Z, residuals, clusters, n_obs)
    else
        throw(ArgumentError("Unknown method: $method. Use :standard, :robust, or :clustered"))
    end
end
