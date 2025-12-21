"""
Local Average Treatment Effect (LATE) estimation.

Implements Imbens & Angrist (1994) LATE framework for binary instruments.

LATE = E[Y1 - Y0 | Complier] = Cov(Y, Z) / Cov(D, Z)
"""

using Statistics
using Distributions
using LinearAlgebra

include("types.jl")


"""
    late_estimator(problem::LATEProblem; alpha=0.05)

Estimate LATE via Wald estimator for binary instrument.

LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])

# Arguments
- `problem::LATEProblem`: Problem specification
- `alpha::Float64`: Significance level for CI (default 0.05)

# Returns
- `LATESolution`: LATE estimate with SE, CI, and diagnostics

# References
- Imbens, G.W. & Angrist, J.D. (1994). Identification and Estimation of LATE.
"""
function late_estimator(problem::LATEProblem{T}; alpha::Float64 = 0.05) where T<:Real
    Y = problem.outcome
    D = problem.treatment
    Z = problem.instrument
    X = problem.covariates
    n = length(Y)

    # Residualize outcome on covariates if provided
    Y_res = if X !== nothing
        residualize(Y, X)
    else
        Y
    end

    # Compute Wald estimator components
    z1_mask = Z .== one(T)
    z0_mask = Z .== zero(T)

    n_z1 = sum(z1_mask)
    n_z0 = sum(z0_mask)

    # Reduced form: E[Y|Z=1] - E[Y|Z=0]
    mean_y_z1 = mean(Y_res[z1_mask])
    mean_y_z0 = mean(Y_res[z0_mask])
    reduced_form = mean_y_z1 - mean_y_z0

    # First stage: E[D|Z=1] - E[D|Z=0]
    mean_d_z1 = mean(D[z1_mask])
    mean_d_z0 = mean(D[z0_mask])
    first_stage = mean_d_z1 - mean_d_z0

    # Check first stage strength
    if abs(first_stage) < T(1e-10)
        error("First stage coefficient essentially zero ($first_stage). " *
              "Instrument has no effect on treatment.")
    end

    # LATE = reduced form / first stage
    late = reduced_form / first_stage

    # Complier/always-taker/never-taker shares
    complier_share = first_stage
    always_taker_share = mean_d_z0
    never_taker_share = one(T) - mean_d_z1

    # Standard error via delta method
    var_y_z1 = n_z1 > 1 ? var(Y_res[z1_mask]; corrected=true) : zero(T)
    var_y_z0 = n_z0 > 1 ? var(Y_res[z0_mask]; corrected=true) : zero(T)
    var_d_z1 = n_z1 > 1 ? var(D[z1_mask]; corrected=true) : zero(T)
    var_d_z0 = n_z0 > 1 ? var(D[z0_mask]; corrected=true) : zero(T)

    # Variance of reduced form
    var_reduced_form = var_y_z1 / n_z1 + var_y_z0 / n_z0

    # Variance of first stage
    var_first_stage = var_d_z1 / n_z1 + var_d_z0 / n_z0

    # Covariance
    cov_yd_z1 = n_z1 > 1 ? cov(Y_res[z1_mask], D[z1_mask]) : zero(T)
    cov_yd_z0 = n_z0 > 1 ? cov(Y_res[z0_mask], D[z0_mask]) : zero(T)
    cov_rf_fs = cov_yd_z1 / n_z1 + cov_yd_z0 / n_z0

    # Delta method variance
    var_late = (one(T) / first_stage^2) * (
        var_reduced_form +
        late^2 * var_first_stage -
        T(2) * late * cov_rf_fs
    )
    se = sqrt(max(var_late, zero(T)))

    # First-stage F-statistic
    first_stage_f = compute_first_stage_f(D, Z, X)

    # Confidence interval and p-value
    z_crit = quantile(Normal(), 1 - alpha / 2)
    ci_lower = late - z_crit * se
    ci_upper = late + z_crit * se

    z_stat = se > zero(T) ? late / se : zero(T)
    pvalue = T(2) * (one(T) - cdf(Normal(), abs(z_stat)))

    return LATESolution(
        late = late,
        se = se,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        pvalue = pvalue,
        complier_share = complier_share,
        always_taker_share = always_taker_share,
        never_taker_share = never_taker_share,
        first_stage_coef = first_stage,
        first_stage_f = first_stage_f,
        n_obs = n,
        method = :wald
    )
end

# Convenience method with raw arrays
function late_estimator(
    outcome::Vector{T},
    treatment::Vector{T},
    instrument::Vector{T};
    covariates::Union{Nothing, Matrix{T}} = nothing,
    alpha::Float64 = 0.05
) where T<:Real
    problem = LATEProblem(
        outcome = outcome,
        treatment = treatment,
        instrument = instrument,
        covariates = covariates
    )
    return late_estimator(problem; alpha = alpha)
end


"""
    late_bounds(outcome, treatment, instrument; alpha=0.05)

Compute bounds on LATE when monotonicity may be violated.

# Arguments
- `outcome::Vector{T}`: Outcome variable
- `treatment::Vector{T}`: Binary treatment
- `instrument::Vector{T}`: Binary instrument
- `alpha::Float64`: Significance level (default 0.05)

# Returns
- `LATEBoundsResult`: Bounds and diagnostics
"""
function late_bounds(
    outcome::Vector{T},
    treatment::Vector{T},
    instrument::Vector{T};
    alpha::Float64 = 0.05
) where T<:Real
    Y = outcome
    D = treatment
    Z = instrument

    # Validate binary
    all(z -> z == zero(T) || z == one(T), Z) ||
        error("Instrument must be binary (0 or 1)")
    all(d -> d == zero(T) || d == one(T), D) ||
        error("Treatment must be binary (0 or 1)")

    # Outcome support
    y_min, y_max = extrema(Y)

    z1_mask = Z .== one(T)
    z0_mask = Z .== zero(T)

    # Conditional means
    mean_y_z1 = mean(Y[z1_mask])
    mean_y_z0 = mean(Y[z0_mask])
    mean_d_z1 = mean(D[z1_mask])
    mean_d_z0 = mean(D[z0_mask])

    # First stage
    first_stage = mean_d_z1 - mean_d_z0
    reduced_form = mean_y_z1 - mean_y_z0

    if abs(first_stage) < T(1e-10)
        return LATEBoundsResult(
            bounds_lower = T(-Inf),
            bounds_upper = T(Inf),
            late_under_monotonicity = T(NaN),
            first_stage = first_stage,
            reduced_form = reduced_form,
            outcome_support = (y_min, y_max),
            bounds_width = T(Inf)
        )
    end

    # Under monotonicity
    late_mono = reduced_form / first_stage

    # Manski-type bounds without monotonicity
    bounds_lower = (reduced_form - (y_max - y_min) * abs(first_stage)) / first_stage
    bounds_upper = (reduced_form + (y_max - y_min) * abs(first_stage)) / first_stage

    if bounds_lower > bounds_upper
        bounds_lower, bounds_upper = bounds_upper, bounds_lower
    end

    return LATEBoundsResult(
        bounds_lower = bounds_lower,
        bounds_upper = bounds_upper,
        late_under_monotonicity = late_mono,
        first_stage = first_stage,
        reduced_form = reduced_form,
        outcome_support = (y_min, y_max),
        bounds_width = bounds_upper - bounds_lower
    )
end


"""
    complier_characteristics(outcome, treatment, instrument; covariates=nothing)

Characterize the complier subpopulation using Abadie (2003) kappa-weighting.

# Arguments
- `outcome::Vector{T}`: Outcome variable
- `treatment::Vector{T}`: Binary treatment
- `instrument::Vector{T}`: Binary instrument
- `covariates::Matrix{T}`: Optional covariates for complier means

# Returns
- `ComplierResult`: Complier characteristics

# References
- Abadie, A. (2003). Semiparametric IV Estimation of Treatment Response Models.
"""
function complier_characteristics(
    outcome::Vector{T},
    treatment::Vector{T},
    instrument::Vector{T};
    covariates::Union{Nothing, Matrix{T}} = nothing
) where T<:Real
    Y = outcome
    D = treatment
    Z = instrument
    n = length(Y)

    # Validate
    all(z -> z == zero(T) || z == one(T), Z) ||
        error("Instrument must be binary (0 or 1)")
    all(d -> d == zero(T) || d == one(T), D) ||
        error("Treatment must be binary (0 or 1)")

    z1_mask = Z .== one(T)
    z0_mask = Z .== zero(T)

    # First stage for complier share
    mean_d_z1 = mean(D[z1_mask])
    mean_d_z0 = mean(D[z0_mask])
    complier_share = mean_d_z1 - mean_d_z0

    if complier_share <= zero(T)
        error("Non-positive complier share ($complier_share). " *
              "Monotonicity may be violated or instrument is defective.")
    end

    # Kappa weights (Abadie 2003)
    p_z1 = mean(Z)
    p_z0 = one(T) - p_z1

    kappa = ones(T, n)
    for i in 1:n
        if D[i] == one(T) && Z[i] == zero(T)
            kappa[i] = p_z0 > zero(T) ? one(T) - one(T) / p_z0 : zero(T)
        elseif D[i] == zero(T) && Z[i] == one(T)
            kappa[i] = p_z1 > zero(T) ? one(T) - one(T) / p_z1 : zero(T)
        end
    end

    # Complier mean outcomes
    # E[Y|D=1, complier]
    kappa_d1 = copy(kappa)
    kappa_d1[D .== zero(T)] .= zero(T)
    kappa_d1_sum = sum(kappa_d1)

    complier_mean_y1 = if kappa_d1_sum > zero(T)
        sum(Y .* kappa_d1) / kappa_d1_sum
    else
        T(NaN)
    end

    # E[Y|D=0, complier]
    kappa_d0 = copy(kappa)
    kappa_d0[D .== one(T)] .= zero(T)
    kappa_d0_sum = sum(kappa_d0)

    complier_mean_y0 = if kappa_d0_sum > zero(T)
        sum(Y .* kappa_d0) / kappa_d0_sum
    else
        T(NaN)
    end

    # Covariate means for compliers
    covariate_means = if covariates !== nothing
        kappa_pos = max.(kappa, zero(T))
        kappa_sum = sum(kappa_pos)
        if kappa_sum > zero(T)
            vec(sum(covariates .* kappa_pos, dims=1) ./ kappa_sum)
        else
            fill(T(NaN), size(covariates, 2))
        end
    else
        nothing
    end

    return ComplierResult(
        complier_mean_outcome_treated = complier_mean_y1,
        complier_mean_outcome_control = complier_mean_y0,
        complier_share = complier_share,
        covariate_means = covariate_means,
        method = :kappa_weights
    )
end


# ============================================================================
# Helper Functions
# ============================================================================

"""
    residualize(y, X)

Residualize y on X via OLS.
"""
function residualize(y::Vector{T}, X::Matrix{T}) where T<:Real
    n = length(y)
    X_const = hcat(ones(T, n), X)

    try
        beta = X_const \ y
        return y - X_const * beta
    catch
        return y
    end
end


"""
    compute_first_stage_f(D, Z, X)

Compute first-stage F-statistic for instrument strength.
"""
function compute_first_stage_f(
    D::Vector{T},
    Z::Vector{T},
    X::Union{Nothing, Matrix{T}}
) where T<:Real
    n = length(D)

    # Build design matrix
    design = if X !== nothing
        hcat(ones(T, n), Z, X)
    else
        hcat(ones(T, n), Z)
    end

    k_full = size(design, 2)

    # Full model: D ~ 1 + Z + X
    try
        beta_full = design \ D
        ssr_full = sum((D - design * beta_full).^2)

        # Restricted model: D ~ 1 + X (no Z)
        design_r = if X !== nothing
            hcat(ones(T, n), X)
        else
            ones(T, n, 1)
        end

        beta_r = design_r \ D
        ssr_restricted = sum((D - design_r * beta_r).^2)

        # F-statistic
        q = 1  # Number of restrictions
        df_denom = n - k_full

        if df_denom <= 0 || ssr_full <= zero(T)
            return zero(T)
        end

        f_stat = ((ssr_restricted - ssr_full) / q) / (ssr_full / df_denom)
        return max(f_stat, zero(T))
    catch
        return zero(T)
    end
end
