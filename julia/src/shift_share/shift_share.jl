"""
Shift-Share (Bartik) Instrumental Variables estimation.

The shift-share instrument exploits:
1. Cross-sectional variation in exposure to sectors (shares)
2. Time-series variation in aggregate shocks (shifts)

Instrument: Z_i = sum_s(share_{i,s} * shock_s)

Identification requires either:
- Exogeneity of shares (Goldsmith-Pinkham et al. 2020)
- Exogeneity of shocks (Borusyak et al. 2022)
"""

using Statistics
using LinearAlgebra
using Distributions

"""
    solve(problem::ShiftShareProblem, estimator::ShiftShareIV)

Estimate treatment effect using Shift-Share IV.

# Returns
- `ShiftShareSolution`: Solution with estimate, SEs, and diagnostics.
"""
function solve(problem::ShiftShareProblem{T}, estimator::ShiftShareIV) where T<:Real
    Y = problem.outcome
    D = problem.treatment
    shares = problem.shares
    shocks = problem.shocks
    X = problem.covariates
    clusters = problem.clusters
    alpha = problem.alpha

    n = length(Y)
    n_sectors = length(shocks)

    # Construct Bartik instrument: Z = shares * shocks
    Z_bartik = shares * shocks

    # Check share normalization
    share_sums = sum(shares, dims=2)
    share_sum_mean = T(mean(share_sums))

    # Compute Rotemberg weights
    rotemberg = _compute_rotemberg_weights(D, shares, shocks, X)

    # First stage: D ~ Z_bartik + X
    first_stage = _first_stage_ss(D, Z_bartik, X)

    # Second stage: 2SLS
    result = _two_stage_ls(Y, D, Z_bartik, X, clusters, estimator.inference, alpha)

    return ShiftShareSolution{T}(
        result.estimate,
        result.se,
        result.t_stat,
        result.p_value,
        result.ci_lower,
        result.ci_upper,
        first_stage,
        rotemberg,
        n,
        n_sectors,
        share_sum_mean,
        estimator.inference,
        alpha,
        Z_bartik
    )
end


"""
    _compute_rotemberg_weights(D, shares, shocks, X)

Compute Rotemberg (1983) weights for shift-share diagnostics.

The weight for sector s is proportional to:
alpha_s = shock_s * sum_i(share_{i,s} * D_i) / sum_s(...)

These weights show which sectors drive the IV estimate.
Negative weights indicate potential monotonicity violations.
"""
function _compute_rotemberg_weights(
    D::Vector{T},
    shares::Matrix{T},
    shocks::Vector{T},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(D)
    n_sectors = length(shocks)

    # Residualize D on X if controls present
    if X !== nothing
        X_with_const = hcat(ones(T, n), X)
        D_resid = D - X_with_const * (X_with_const \ D)
    else
        D_resid = D .- mean(D)
    end

    # Compute raw weights: shock_s * sum_i(share_{i,s} * D_resid_i)
    raw_weights = zeros(T, n_sectors)
    for s in 1:n_sectors
        raw_weights[s] = shocks[s] * sum(shares[:, s] .* D_resid)
    end

    # Normalize to sum of absolute values = 1
    total = sum(abs.(raw_weights))
    if total > T(1e-10)
        weights = raw_weights ./ total
    else
        weights = ones(T, n_sectors) ./ n_sectors
    end

    # Negative weight share
    negative_mask = weights .< zero(T)
    negative_weight_share = T(sum(abs.(weights[negative_mask])))

    # Top 5 sectors by absolute weight (or all if fewer)
    abs_weights = abs.(weights)
    n_top = min(5, n_sectors)
    top_idx = sortperm(abs_weights, rev=true)[1:n_top]

    # Pad to 5 if needed
    if n_top < 5
        top_5_sectors = vcat(top_idx, fill(1, 5 - n_top))
        top_5_weights = vcat(weights[top_idx], zeros(T, 5 - n_top))
    else
        top_5_sectors = top_idx
        top_5_weights = weights[top_idx]
    end

    # Herfindahl index (concentration)
    herfindahl = T(sum(weights.^2))

    return RotembergDiagnostics{T}(
        weights,
        negative_weight_share,
        top_5_sectors,
        top_5_weights,
        herfindahl
    )
end


"""
    _first_stage_ss(D, Z, X)

First-stage regression: D ~ Z + X.
Returns diagnostics including F-statistic for instrument strength.
"""
function _first_stage_ss(
    D::Vector{T},
    Z::Vector{T},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(D)

    # Build design matrix: [1, Z, X]
    if X !== nothing
        design = hcat(ones(T, n), Z, X)
        n_controls = size(X, 2)
    else
        design = hcat(ones(T, n), Z)
        n_controls = 0
    end

    # OLS
    XtX = design' * design
    XtY = design' * D
    coefficients = XtX \ XtY

    # Fitted values and residuals
    fitted = design * coefficients
    residuals = D - fitted

    # Degrees of freedom
    df = n - size(design, 2)

    # Residual variance
    sigma2 = sum(residuals.^2) / df

    # Standard errors (HC3 robust, with safeguards)
    XtX_inv = inv(XtX)
    leverage = diag(design * XtX_inv * design')
    # Cap leverage to avoid division by near-zero
    leverage_capped = min.(leverage, one(T) - T(1e-6))
    u_adj = residuals ./ (one(T) .- leverage_capped)
    meat = design' * Diagonal(u_adj.^2) * design
    vcov = XtX_inv * meat * XtX_inv
    # Ensure non-negative variance
    se = sqrt.(max.(diag(vcov), zero(T)))

    # R-squared
    ss_res = sum(residuals.^2)
    ss_tot = sum((D .- mean(D)).^2)
    r2 = one(T) - ss_res / ss_tot

    # F-statistic for instrument (index 2 is Z coefficient)
    coef = coefficients[2]
    se_coef = se[2]
    t_stat = coef / se_coef
    f_statistic = t_stat^2
    f_pvalue = 2 * (1 - cdf(TDist(df), abs(t_stat)))

    # Partial R-squared
    if X !== nothing
        # Residualize Z and D on X
        X_with_const = hcat(ones(T, n), X)
        Z_resid = Z - X_with_const * (X_with_const \ Z)
        D_resid = D - X_with_const * (X_with_const \ D)
        partial_r2 = T(cor(Z_resid, D_resid)^2)
    else
        partial_r2 = r2
    end

    weak_iv_warning = f_statistic < T(10)

    return FirstStageSSResult{T}(
        f_statistic,
        f_pvalue,
        partial_r2,
        coef,
        se_coef,
        t_stat,
        weak_iv_warning
    )
end


"""
    _two_stage_ls(Y, D, Z, X, clusters, inference, alpha)

Run 2SLS estimation with robust or clustered SEs.
"""
function _two_stage_ls(
    Y::Vector{T},
    D::Vector{T},
    Z::Vector{T},
    X::Union{Nothing, Matrix{T}},
    clusters::Union{Nothing, Vector},
    inference::Symbol,
    alpha::T
) where T<:Real
    n = length(Y)

    # Build design matrices
    if X !== nothing
        exog_first = hcat(ones(T, n), Z, X)
        n_controls = size(X, 2)
    else
        exog_first = hcat(ones(T, n), Z)
        n_controls = 0
    end

    # First stage: get fitted D
    fs_coefficients = exog_first \ D
    D_hat = exog_first * fs_coefficients

    # Second stage design: [1, D_hat, X]
    if X !== nothing
        ss_design = hcat(ones(T, n), D_hat, X)
    else
        ss_design = hcat(ones(T, n), D_hat)
    end

    # Second stage OLS
    ss_coefficients = ss_design \ Y

    # Get estimate (coefficient on D_hat, index 2)
    estimate = ss_coefficients[2]

    # Compute residuals using original D (not D_hat) for correct SE
    if X !== nothing
        final_design = hcat(ones(T, n), D, X)
    else
        final_design = hcat(ones(T, n), D)
    end
    resid = Y - final_design * ss_coefficients

    # Compute standard errors
    df = n - size(ss_design, 2)

    if inference == :clustered && clusters !== nothing
        se = _clustered_se(ss_design, resid, clusters)[2]  # Index 2 for D coefficient
    else
        # Robust (HC3) SE with safeguards
        XtX = ss_design' * ss_design
        XtX_inv = inv(XtX)
        leverage = diag(ss_design * XtX_inv * ss_design')
        # Cap leverage to avoid division by near-zero
        leverage_capped = min.(leverage, one(T) - T(1e-6))
        u_adj = resid ./ (one(T) .- leverage_capped)
        meat = ss_design' * Diagonal(u_adj.^2) * ss_design
        vcov = XtX_inv * meat * XtX_inv
        # Ensure non-negative variance
        se = sqrt(max(vcov[2, 2], zero(T)))
    end

    # Inference
    t_stat = estimate / se
    p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = estimate - z_crit * se
    ci_upper = estimate + z_crit * se

    return (
        estimate = estimate,
        se = se,
        t_stat = t_stat,
        p_value = p_value,
        ci_lower = ci_lower,
        ci_upper = ci_upper
    )
end


"""
    _clustered_se(design, residuals, clusters)

Compute cluster-robust standard errors (Liang-Zeger).
"""
function _clustered_se(
    design::Matrix{T},
    residuals::Vector{T},
    clusters::Vector
) where T<:Real
    n, k = size(design)
    unique_clusters = unique(clusters)
    G = length(unique_clusters)

    XtX = design' * design
    XtX_inv = inv(XtX)

    # Compute cluster sums
    meat = zeros(T, k, k)
    for g in unique_clusters
        idx = clusters .== g
        X_g = design[idx, :]
        u_g = residuals[idx]
        score_g = X_g' * u_g
        meat .+= score_g * score_g'
    end

    # Small sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k))

    vcov = correction * XtX_inv * meat * XtX_inv
    se = sqrt.(diag(vcov))

    return se
end


"""
    shift_share_iv(Y, D, shares, shocks; kwargs...)

Convenience function for Shift-Share IV estimation.

Constructs Bartik instrument Z = shares * shocks and runs 2SLS.

# Arguments
- `Y::Vector`: Outcome variable
- `D::Vector`: Endogenous treatment
- `shares::Matrix`: Sector shares (n × S), rows should sum to ~1
- `shocks::Vector`: Aggregate sector shocks (S)
- `X::Union{Nothing, Matrix}=nothing`: Control variables
- `clusters::Union{Nothing, Vector}=nothing`: Cluster identifiers
- `inference::Symbol=:robust`: SE type (:robust or :clustered)
- `alpha::Float64=0.05`: Significance level

# Returns
- `ShiftShareSolution`: Estimation results with Rotemberg diagnostics.

# Examples
```julia
using Random
Random.seed!(42)

n, S = 200, 15
shares = rand(n, S)
shares ./= sum(shares, dims=2)  # Normalize rows
shocks = randn(S) * 0.1
Z = shares * shocks
D = 1.5 * Z + randn(n) * 0.3
Y = 2.0 * D + randn(n)

result = shift_share_iv(Y, D, shares, shocks)
println("Estimate: \$(result.estimate)")
```
"""
function shift_share_iv(
    Y::Vector{T},
    D::Vector{T},
    shares::Matrix{T},
    shocks::Vector{T};
    X::Union{Nothing, Matrix{T}} = nothing,
    clusters::Union{Nothing, Vector} = nothing,
    inference::Symbol = :robust,
    alpha::T = T(0.05)
) where T<:Real
    problem = ShiftShareProblem(
        outcome = Y,
        treatment = D,
        shares = shares,
        shocks = shocks,
        covariates = X,
        clusters = clusters,
        alpha = alpha
    )
    estimator = ShiftShareIV(inference = inference)
    return solve(problem, estimator)
end


# Type-flexible convenience function
function shift_share_iv(
    Y::AbstractVector,
    D::AbstractVector,
    shares::AbstractMatrix,
    shocks::AbstractVector;
    X::Union{Nothing, AbstractMatrix} = nothing,
    clusters::Union{Nothing, AbstractVector} = nothing,
    inference::Symbol = :robust,
    alpha::Real = 0.05
)
    T = promote_type(eltype(Y), eltype(D), eltype(shares), eltype(shocks), typeof(alpha))
    Y_t = convert(Vector{T}, Y)
    D_t = convert(Vector{T}, D)
    shares_t = convert(Matrix{T}, shares)
    shocks_t = convert(Vector{T}, shocks)
    alpha_t = T(alpha)

    X_t = X === nothing ? nothing : convert(Matrix{T}, X)
    clusters_t = clusters

    return shift_share_iv(Y_t, D_t, shares_t, shocks_t;
                          X=X_t, clusters=clusters_t,
                          inference=inference, alpha=alpha_t)
end
