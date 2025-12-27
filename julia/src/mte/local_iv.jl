"""
Marginal Treatment Effects via Local Instrumental Variables.

Implements Heckman & Vytlacil (1999, 2005) framework for estimating
treatment effect heterogeneity indexed by unobserved resistance.

MTE(u) = dE[Y|P(Z)=p]/dp evaluated at u = p
"""

using Statistics
using Distributions
using LinearAlgebra
using Random

# Note: types.jl is included by CausalEstimators.jl before this file


"""
    local_iv(problem::MTEProblem; kwargs...)

Estimate MTE via local instrumental variables.

Computes MTE(u) at grid points across the propensity score support.

# Arguments
- `problem::MTEProblem`: Problem specification
- `bandwidth::Union{Nothing, Float64}`: Kernel bandwidth (default: auto)
- `bandwidth_rule::Symbol`: Rule for auto bandwidth (:silverman, :scott)
- `alpha::Float64`: Significance level for CI
- `n_bootstrap::Int`: Bootstrap replications for SE
- `random_state::Union{Nothing, Int}`: Random seed

# Returns
- `MTESolution`: MTE curve with SE and CI

# References
- Heckman & Vytlacil (1999). Local Instrumental Variables.
- Heckman & Vytlacil (2005). Structural Equations, Treatment Effects.
"""
function local_iv(
    problem::MTEProblem{T};
    bandwidth::Union{Nothing, T} = nothing,
    bandwidth_rule::Symbol = :silverman,
    alpha::Float64 = 0.05,
    n_bootstrap::Int = 500,
    random_state::Union{Nothing, Int} = nothing
) where T<:Real
    Y = problem.outcome
    D = problem.treatment
    Z = problem.instrument
    X = problem.covariates
    n_grid = problem.n_grid
    trim_fraction = problem.trim_fraction
    n = length(Y)

    # Estimate propensity scores P(D=1|Z)
    propensity = estimate_propensity(D, Z, X)

    # Determine support and trim
    p_min = quantile(propensity, trim_fraction)
    p_max = quantile(propensity, 1 - trim_fraction)

    # Mask for common support
    support_mask = (propensity .>= p_min) .& (propensity .<= p_max)
    n_trimmed = n - sum(support_mask)

    Y_trim = Y[support_mask]
    D_trim = D[support_mask]
    P_trim = propensity[support_mask]

    if X !== nothing
        X_trim = X[support_mask, :]
        Y_trim = residualize(Y_trim, X_trim)
    end

    # Create evaluation grid
    u_grid = collect(range(p_min, p_max, length=n_grid))

    # Select bandwidth
    bw = if bandwidth === nothing
        select_bandwidth(P_trim, bandwidth_rule)
    else
        bandwidth
    end

    # Estimate MTE at each grid point
    mte_grid = estimate_mte_grid(Y_trim, D_trim, P_trim, u_grid, bw)

    # Bootstrap for standard errors
    rng = random_state === nothing ? Random.default_rng() : Random.MersenneTwister(random_state)
    bootstrap_mte = zeros(T, n_bootstrap, n_grid)

    n_trim = length(Y_trim)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:n_trim, n_trim)
        Y_boot = Y_trim[idx]
        D_boot = D_trim[idx]
        P_boot = P_trim[idx]
        bootstrap_mte[b, :] = estimate_mte_grid(Y_boot, D_boot, P_boot, u_grid, bw)
    end

    # Standard errors and CIs
    se_grid = vec(mapslices(x -> std(filter(!isnan, x); corrected=true), bootstrap_mte; dims=1))
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = mte_grid .- z_crit .* se_grid
    ci_upper = mte_grid .+ z_crit .* se_grid

    return MTESolution(
        mte_grid = mte_grid,
        u_grid = u_grid,
        se_grid = se_grid,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        propensity_support = (p_min, p_max),
        n_obs = n,
        n_trimmed = n_trimmed,
        bandwidth = bw,
        method = :local_iv
    )
end

# Convenience method with raw arrays
function local_iv(
    outcome::Vector{T},
    treatment::Vector{T},
    instrument::Union{Vector{T}, Matrix{T}};
    covariates::Union{Nothing, Matrix{T}} = nothing,
    n_grid::Int = 50,
    trim_fraction::T = T(0.01),
    bandwidth::Union{Nothing, T} = nothing,
    bandwidth_rule::Symbol = :silverman,
    alpha::Float64 = 0.05,
    n_bootstrap::Int = 500,
    random_state::Union{Nothing, Int} = nothing
) where T<:Real
    problem = MTEProblem(
        outcome = outcome,
        treatment = treatment,
        instrument = instrument,
        covariates = covariates,
        n_grid = n_grid,
        trim_fraction = trim_fraction
    )
    return local_iv(
        problem;
        bandwidth = bandwidth,
        bandwidth_rule = bandwidth_rule,
        alpha = alpha,
        n_bootstrap = n_bootstrap,
        random_state = random_state
    )
end


"""
    polynomial_mte(outcome, treatment, instrument; degree=3, kwargs...)

Estimate MTE using polynomial approximation.

Fits E[Y|P] as polynomial in P, computes MTE = dE[Y|P]/dP.

# Arguments
- `outcome::Vector{T}`: Outcome variable
- `treatment::Vector{T}`: Binary treatment
- `instrument`: Instrument(s)
- `degree::Int`: Polynomial degree (default 3)
- `n_grid::Int`: Number of grid points
- `trim_fraction::T`: Trimming fraction
- `alpha::Float64`: Significance level
- `n_bootstrap::Int`: Bootstrap replications
- `random_state`: Random seed

# Returns
- `MTESolution`: MTE estimates via polynomial method
"""
function polynomial_mte(
    outcome::Vector{T},
    treatment::Vector{T},
    instrument::Union{Vector{T}, Matrix{T}};
    covariates::Union{Nothing, Matrix{T}} = nothing,
    degree::Int = 3,
    n_grid::Int = 50,
    trim_fraction::T = T(0.01),
    alpha::Float64 = 0.05,
    n_bootstrap::Int = 500,
    random_state::Union{Nothing, Int} = nothing
) where T<:Real
    Y = outcome
    D = treatment
    Z = instrument
    n = length(Y)

    # Validate inputs
    length(D) == n || error("Treatment length mismatch")
    Z isa Vector && length(Z) == n || Z isa Matrix && size(Z, 1) == n ||
        error("Instrument dimension mismatch")

    # Estimate propensity
    propensity = estimate_propensity(D, Z, covariates)

    # Trim
    p_min = quantile(propensity, trim_fraction)
    p_max = quantile(propensity, 1 - trim_fraction)
    support_mask = (propensity .>= p_min) .& (propensity .<= p_max)
    n_trimmed = n - sum(support_mask)

    Y_trim = Y[support_mask]
    D_trim = D[support_mask]
    P_trim = propensity[support_mask]

    if covariates !== nothing
        Y_trim = residualize(Y_trim, covariates[support_mask, :])
    end

    # Grid
    u_grid = collect(range(p_min, p_max, length=n_grid))

    # Polynomial MTE estimate
    mte_grid = polynomial_mte_estimate(Y_trim, D_trim, P_trim, u_grid, degree)

    # Bootstrap
    rng = random_state === nothing ? Random.default_rng() : Random.MersenneTwister(random_state)
    bootstrap_mte = zeros(T, n_bootstrap, n_grid)

    n_trim = length(Y_trim)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:n_trim, n_trim)
        bootstrap_mte[b, :] = polynomial_mte_estimate(
            Y_trim[idx], D_trim[idx], P_trim[idx], u_grid, degree
        )
    end

    se_grid = vec(mapslices(x -> std(filter(!isnan, x); corrected=true), bootstrap_mte; dims=1))
    z_crit = quantile(Normal(), 1 - alpha / 2)

    return MTESolution(
        mte_grid = mte_grid,
        u_grid = u_grid,
        se_grid = se_grid,
        ci_lower = mte_grid .- z_crit .* se_grid,
        ci_upper = mte_grid .+ z_crit .* se_grid,
        propensity_support = (p_min, p_max),
        n_obs = n,
        n_trimmed = n_trimmed,
        bandwidth = T(degree),  # Store degree as bandwidth
        method = :polynomial
    )
end


# ============================================================================
# Helper Functions
# ============================================================================

"""
    estimate_propensity(D, Z, X)

Estimate propensity score P(D=1|Z, X) via logistic regression.
"""
function estimate_propensity(
    D::Vector{T},
    Z::Union{Vector{T}, Matrix{T}},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(D)

    # Build design matrix
    Z_mat = Z isa Vector ? reshape(Z, n, 1) : Z

    design = if X !== nothing
        hcat(ones(T, n), Z_mat, X)
    else
        hcat(ones(T, n), Z_mat)
    end

    # Logistic regression via IRLS
    propensity = logistic_regression(D, design)

    # Clamp for numerical stability
    eps = T(1e-6)
    return clamp.(propensity, eps, one(T) - eps)
end


"""
    logistic_regression(y, X; max_iter=100, tol=1e-8)

Fit logistic regression via IRLS, return predicted probabilities.
"""
function logistic_regression(
    y::Vector{T},
    X::Matrix{T};
    max_iter::Int = 100,
    tol::Float64 = 1e-8
) where T<:Real
    n, p = size(X)
    beta = zeros(T, p)

    for _ in 1:max_iter
        # Predicted probabilities
        eta = X * beta
        eta = clamp.(eta, T(-500), T(500))
        prob = one(T) ./ (one(T) .+ exp.(-eta))
        prob = clamp.(prob, T(1e-10), one(T) - T(1e-10))

        # Weights and working response
        W = prob .* (one(T) .- prob)
        W = max.(W, T(1e-10))

        z = eta .+ (y .- prob) ./ W

        # Weighted least squares step
        W_diag = Diagonal(W)
        XtWX = X' * W_diag * X
        XtWz = X' * (W .* z)

        try
            beta_new = (XtWX + T(1e-10) * I) \ XtWz
            if maximum(abs.(beta_new .- beta)) < tol
                beta = beta_new
                break
            end
            beta = beta_new
        catch
            break
        end
    end

    # Final predictions
    eta = X * beta
    eta = clamp.(eta, T(-500), T(500))
    return one(T) ./ (one(T) .+ exp.(-eta))
end


"""
    estimate_mte_grid(Y, D, P, u_grid, bandwidth)

Estimate MTE at grid points using local linear regression.
"""
function estimate_mte_grid(
    Y::Vector{T},
    D::Vector{T},
    P::Vector{T},
    u_grid::Vector{T},
    bandwidth::T
) where T<:Real
    n_grid = length(u_grid)
    mte = zeros(T, n_grid)

    for (i, p) in enumerate(u_grid)
        # Kernel weights (Epanechnikov)
        u = (P .- p) ./ bandwidth
        weights = ifelse.(abs.(u) .<= one(T), T(0.75) .* (one(T) .- u.^2), zero(T))

        if sum(weights) < T(1e-10)
            mte[i] = T(NaN)
            continue
        end

        # Local linear regression of Y on P
        n = length(P)
        X_local = hcat(ones(T, n), P .- p)
        W = Diagonal(weights)

        try
            XtWX = X_local' * W * X_local
            XtWY = X_local' * W * Y

            # Regularize
            XtWX += T(1e-10) * I

            beta = XtWX \ XtWY
            mte[i] = beta[2]  # Slope = dE[Y|P]/dP
        catch
            mte[i] = T(NaN)
        end
    end

    # Apply smoothing
    return smooth_mte(mte)
end


"""
    polynomial_mte_estimate(Y, D, P, u_grid, degree)

Estimate MTE using polynomial approximation.
"""
function polynomial_mte_estimate(
    Y::Vector{T},
    D::Vector{T},
    P::Vector{T},
    u_grid::Vector{T},
    degree::Int
) where T<:Real
    # Create polynomial features: [1, P, P^2, ..., P^degree]
    n = length(P)
    X_poly = hcat([P.^k for k in 0:degree]...)

    # OLS fit
    beta = try
        X_poly \ Y
    catch
        return fill(T(NaN), length(u_grid))
    end

    # Compute derivative at grid points
    # d/dp [sum_k beta_k p^k] = sum_k k * beta_k * p^(k-1)
    mte = zeros(T, length(u_grid))
    for (i, p) in enumerate(u_grid)
        derivative = zero(T)
        for k in 1:degree
            derivative += k * beta[k+1] * p^(k-1)
        end
        mte[i] = derivative
    end

    return mte
end


"""
    select_bandwidth(P, rule)

Select bandwidth using specified rule.
"""
function select_bandwidth(P::Vector{T}, rule::Symbol) where T<:Real
    sigma = std(P; corrected=true)
    n = length(P)

    if sigma < T(1e-10)
        return T(0.1)
    end

    h = if rule == :silverman
        # Silverman's rule of thumb
        iqr = quantile(P, 0.75) - quantile(P, 0.25)
        sigma_iqr = min(sigma, iqr / T(1.34))
        T(1.06) * sigma_iqr * T(n)^T(-0.2)
    elseif rule == :scott
        # Scott's rule
        T(1.059) * sigma * T(n)^T(-0.2)
    else
        # Default to Silverman
        T(1.06) * sigma * T(n)^T(-0.2)
    end

    return max(h, T(0.01))
end


"""
    smooth_mte(mte; sigma=1.0)

Apply Gaussian smoothing to MTE curve.
"""
function smooth_mte(mte::Vector{T}; sigma::T = T(1.0)) where T<:Real
    valid = .!isnan.(mte)
    if sum(valid) < 3
        return mte
    end

    # Interpolate NaN values
    mte_interp = copy(mte)
    x = collect(1:length(mte))
    valid_x = x[valid]
    valid_mte = mte[valid]

    for i in x[.!valid]
        # Linear interpolation
        idx_below = findlast(valid_x .< i)
        idx_above = findfirst(valid_x .> i)

        if idx_below !== nothing && idx_above !== nothing
            x1, x2 = valid_x[idx_below], valid_x[idx_above]
            y1, y2 = valid_mte[idx_below], valid_mte[idx_above]
            mte_interp[i] = y1 + (y2 - y1) * (i - x1) / (x2 - x1)
        elseif idx_below !== nothing
            mte_interp[i] = valid_mte[idx_below]
        elseif idx_above !== nothing
            mte_interp[i] = valid_mte[idx_above]
        end
    end

    # Simple moving average smoothing (Gaussian approximation)
    n = length(mte_interp)
    mte_smooth = copy(mte_interp)
    window = max(1, round(Int, 2 * sigma))

    for i in 1:n
        start_idx = max(1, i - window)
        end_idx = min(n, i + window)
        vals = mte_interp[start_idx:end_idx]
        if !all(isnan, vals)
            mte_smooth[i] = mean(filter(!isnan, vals))
        end
    end

    # Restore NaN at original NaN positions
    mte_smooth[.!valid] .= T(NaN)

    return mte_smooth
end


# Note: residualize() is defined in late.jl (included before this file)
