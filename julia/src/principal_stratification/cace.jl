"""
CACE (Complier Average Causal Effect) estimation via 2SLS.

Implements the key identification result: CACE = LATE under standard assumptions.

# Key Insight
Principal strata are defined by potential treatment values:
- Compliers: D(0)=0, D(1)=1 (respond to assignment)
- Always-takers: D(0)=1, D(1)=1 (always treated)
- Never-takers: D(0)=0, D(1)=0 (never treated)
- Defiers: D(0)=1, D(1)=0 (ruled out by monotonicity)

Under the assumptions:
1. Independence: Z ⊥ (Y(0), Y(1), D(0), D(1))
2. Exclusion: Y(d,z) = Y(d) for all d, z
3. Monotonicity: D(1) ≥ D(0) for all units (no defiers)
4. Relevance: P(D(1)=1, D(0)=0) > 0 (some compliers exist)

We have: CACE = (Reduced Form) / (First Stage) = LATE

# References
- Angrist, Imbens, Rubin (1996). Identification of Causal Effects Using IV.
- Frangakis & Rubin (2002). Principal Stratification in Causal Inference.
"""

using LinearAlgebra
using Statistics
using Distributions

# Note: types.jl is included by CausalEstimators.jl before this file

"""
    solve(problem::CACEProblem, estimator::CACETwoSLS) -> CACESolution

Estimate CACE using Two-Stage Least Squares.

# Arguments
- `problem::CACEProblem`: The problem specification
- `estimator::CACETwoSLS`: 2SLS estimator options

# Returns
- `CACESolution`: Results including CACE, SE, CI, strata proportions

# Example
```julia
problem = CACEProblem(Y, D, Z, alpha=0.05)
solution = solve(problem, CACETwoSLS())
println("CACE: \$(solution.cace) ± \$(solution.se)")
```
"""
function solve(problem::CACEProblem{T}, estimator::CACETwoSLS) where {T<:Real}
    Y = problem.outcome
    D = Float64.(problem.treatment)  # Convert Bool to Float64
    Z = Float64.(problem.instrument)
    α = problem.parameters.alpha
    inference = get(problem.parameters, :inference, :robust)

    n = length(Y)

    # Build design matrix with covariates
    if isnothing(problem.covariates)
        if estimator.add_intercept
            X_with_const = ones(T, n, 1)
        else
            X_with_const = Matrix{T}(undef, n, 0)
        end
    else
        X = problem.covariates
        if estimator.add_intercept
            X_with_const = hcat(ones(T, n), X)
        else
            X_with_const = X
        end
    end

    # ==========================================================================
    # First Stage: D = π₀ + π₁Z + π₂X + ν
    # ==========================================================================
    Z_full = hcat(X_with_const, Z)

    # OLS for first stage
    first_stage_coef = Z_full \ D
    D_hat = Z_full * first_stage_coef

    # First-stage residuals
    ν = D - D_hat

    # First-stage SE (for π₁, the coefficient on Z)
    Z_idx = size(Z_full, 2)  # Z is last column

    if inference == :robust
        first_stage_vcov = robust_vcov(Z_full, ν)
    else
        σ²_first = sum(ν.^2) / (n - size(Z_full, 2))
        first_stage_vcov = σ²_first * inv(Z_full' * Z_full)
    end

    first_stage_se = sqrt(first_stage_vcov[Z_idx, Z_idx])
    π_z = first_stage_coef[Z_idx]

    # First-stage F-statistic
    first_stage_f = (π_z / first_stage_se)^2

    # ==========================================================================
    # Reduced Form: Y = γ₀ + γ₁Z + γ₂X + η
    # ==========================================================================
    reduced_form_coef = Z_full \ Y
    Y_resid_rf = Y - Z_full * reduced_form_coef
    γ_z = reduced_form_coef[Z_idx]

    if inference == :robust
        rf_vcov = robust_vcov(Z_full, Y_resid_rf)
    else
        σ²_rf = sum(Y_resid_rf.^2) / (n - size(Z_full, 2))
        rf_vcov = σ²_rf * inv(Z_full' * Z_full)
    end

    reduced_form_se = sqrt(rf_vcov[Z_idx, Z_idx])

    # ==========================================================================
    # Second Stage: Y = β₀ + β₁D̂ + β₂X + ε (2SLS)
    # ==========================================================================
    W = hcat(X_with_const, D_hat)

    # Second-stage OLS (using D_hat)
    second_stage_coef = W \ Y

    # CACE is coefficient on D_hat (last column)
    cace = second_stage_coef[end]

    # ==========================================================================
    # Correct 2SLS Standard Errors
    # ==========================================================================
    # Use true residuals but projection matrix from Z

    W_true = hcat(X_with_const, D)
    ε = Y - W_true * second_stage_coef

    if inference == :robust
        # Robust 2SLS variance estimator
        ZtZ_inv = inv(Z_full' * Z_full)
        P_z_W = Z_full * ZtZ_inv * Z_full' * W_true

        # Meat of sandwich
        meat = zeros(T, size(W_true, 2), size(W_true, 2))
        for i in 1:n
            pzw_i = vec(Z_full[i:i, :] * ZtZ_inv * Z_full' * W_true)
            meat .+= (ε[i]^2) * (pzw_i * pzw_i')
        end

        # Bread
        bread = inv(P_z_W' * W_true)
        vcov_2sls = bread * meat * bread
    else
        # Homoskedastic 2SLS variance
        σ² = sum(ε.^2) / (n - size(W_true, 2))
        ZtZ_inv = inv(Z_full' * Z_full)
        WtPzW = W_true' * Z_full * ZtZ_inv * Z_full' * W_true
        vcov_2sls = σ² * inv(WtPzW)
    end

    cace_se = sqrt(vcov_2sls[end, end])

    # ==========================================================================
    # Inference
    # ==========================================================================
    z_stat = cace / cace_se
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))
    z_crit = quantile(Normal(), 1 - α/2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # ==========================================================================
    # Strata Proportions
    # ==========================================================================
    strata_props = compute_strata_proportions(D, Z, first_stage_se)

    # ==========================================================================
    # Sample sizes
    # ==========================================================================
    n_treated_assigned = sum(problem.instrument)
    n_control_assigned = n - n_treated_assigned

    return CACESolution{T}(
        cace,
        cace_se,
        ci_lower,
        ci_upper,
        z_stat,
        pvalue,
        strata_props,
        π_z,
        first_stage_se,
        first_stage_f,
        γ_z,
        reduced_form_se,
        n,
        n_treated_assigned,
        n_control_assigned,
        :twosls
    )
end

"""
    solve(problem::CACEProblem, ::WaldEstimator) -> CACESolution

Estimate CACE using the simple Wald/ratio estimator.

CACE = [E(Y|Z=1) - E(Y|Z=0)] / [E(D|Z=1) - E(D|Z=0)]

Uses delta method for standard error. No covariates supported.
"""
function solve(problem::CACEProblem{T}, ::WaldEstimator) where {T<:Real}
    Y = problem.outcome
    D = problem.treatment
    Z = problem.instrument
    α = problem.parameters.alpha

    # Split by instrument
    Y_z1 = Y[Z]
    Y_z0 = Y[.!Z]
    D_z1 = D[Z]
    D_z0 = D[.!Z]

    n1 = length(Y_z1)
    n0 = length(Y_z0)
    n = n1 + n0

    # Reduced form: E[Y|Z=1] - E[Y|Z=0]
    γ = mean(Y_z1) - mean(Y_z0)
    var_γ = var(Y_z1) / n1 + var(Y_z0) / n0
    γ_se = sqrt(var_γ)

    # First stage: E[D|Z=1] - E[D|Z=0]
    π = mean(D_z1) - mean(D_z0)
    var_π = var(Float64.(D_z1)) / n1 + var(Float64.(D_z0)) / n0
    π_se = sqrt(var_π)

    # Wald/CACE estimate
    cace = γ / π

    # Delta method for SE
    var_cace = (var_γ + cace^2 * var_π) / π^2
    cace_se = sqrt(var_cace)

    # Inference
    z_stat = cace / cace_se
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))
    z_crit = quantile(Normal(), 1 - α/2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # First-stage F
    first_stage_f = (π / π_se)^2

    # Strata proportions
    strata_props = compute_strata_proportions(Float64.(D), Float64.(Z), π_se)

    return CACESolution{T}(
        cace,
        cace_se,
        ci_lower,
        ci_upper,
        z_stat,
        pvalue,
        strata_props,
        π,
        π_se,
        first_stage_f,
        γ,
        γ_se,
        n,
        n1,
        n0,
        :wald
    )
end

"""
    cace_2sls(Y, D, Z; covariates=nothing, alpha=0.05, inference=:robust)

Convenience function to estimate CACE via 2SLS.

# Arguments
- `Y`: Outcome vector
- `D`: Treatment received (binary)
- `Z`: Instrument/assignment (binary)
- `covariates`: Optional covariate matrix
- `alpha`: Significance level (default: 0.05)
- `inference`: `:robust` or `:standard` (default: :robust)

# Returns
NamedTuple with CACE estimate and all diagnostics.

# Example
```julia
result = cace_2sls(Y, D, Z)
println("CACE: \$(result.cace) (SE: \$(result.se))")
println("Complier proportion: \$(result.strata_proportions.compliers)")
```
"""
function cace_2sls(
    outcome::Vector{T},
    treatment::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
    instrument::Union{Vector{Bool}, BitVector, Vector{<:Integer}};
    covariates::Union{Matrix{<:Real}, Nothing}=nothing,
    alpha::Float64=0.05,
    inference::Symbol=:robust
) where {T<:Real}
    # Create problem and solve
    problem = CACEProblem(outcome, treatment, instrument, covariates,
                          (alpha=alpha, inference=inference))
    solution = solve(problem, CACETwoSLS())

    # Return as NamedTuple for easier access
    return (
        cace = solution.cace,
        se = solution.se,
        ci_lower = solution.ci_lower,
        ci_upper = solution.ci_upper,
        z_stat = solution.z_stat,
        pvalue = solution.pvalue,
        strata_proportions = NamedTuple(solution.strata_proportions),
        first_stage_coef = solution.first_stage_coef,
        first_stage_se = solution.first_stage_se,
        first_stage_f = solution.first_stage_f,
        reduced_form = solution.reduced_form,
        reduced_form_se = solution.reduced_form_se,
        n = solution.n,
        n_treated_assigned = solution.n_treated_assigned,
        n_control_assigned = solution.n_control_assigned,
        method = solution.method
    )
end

"""
    wald_estimator(Y, D, Z; alpha=0.05) -> NamedTuple

Simple Wald estimator for CACE (no covariates).
"""
function wald_estimator(
    outcome::Vector{T},
    treatment::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
    instrument::Union{Vector{Bool}, BitVector, Vector{<:Integer}};
    alpha::Float64=0.05
) where {T<:Real}
    problem = CACEProblem(outcome, treatment, instrument, nothing,
                          (alpha=alpha, inference=:robust))
    solution = solve(problem, WaldEstimator())

    return (
        cace = solution.cace,
        se = solution.se,
        ci_lower = solution.ci_lower,
        ci_upper = solution.ci_upper,
        z_stat = solution.z_stat,
        pvalue = solution.pvalue,
        strata_proportions = NamedTuple(solution.strata_proportions),
        first_stage_coef = solution.first_stage_coef,
        first_stage_se = solution.first_stage_se,
        first_stage_f = solution.first_stage_f,
        reduced_form = solution.reduced_form,
        reduced_form_se = solution.reduced_form_se,
        n = solution.n,
        n_treated_assigned = solution.n_treated_assigned,
        n_control_assigned = solution.n_control_assigned,
        method = solution.method
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_strata_proportions(D, Z, first_stage_se) -> StrataProportions

Compute principal strata proportions from observed data.

Under monotonicity (no defiers):
- π_c (compliers) = P(D=1|Z=1) - P(D=1|Z=0)
- π_a (always-takers) = P(D=1|Z=0)
- π_n (never-takers) = 1 - P(D=1|Z=1)
"""
function compute_strata_proportions(
    D::Vector{Float64},
    Z::Vector{Float64},
    first_stage_se::Float64
)
    # Conditional treatment probabilities
    p_d1_z1 = mean(D[Z .== 1.0])
    p_d1_z0 = mean(D[Z .== 0.0])

    # Strata proportions
    π_c = p_d1_z1 - p_d1_z0  # Compliers
    π_a = p_d1_z0            # Always-takers
    π_n = 1 - p_d1_z1        # Never-takers

    # Ensure non-negative (numerical issues)
    π_c = max(0.0, min(1.0, π_c))
    π_a = max(0.0, min(1.0, π_a))
    π_n = max(0.0, min(1.0, π_n))

    # Normalize to sum to 1
    total = π_c + π_a + π_n
    if !isapprox(total, 1.0, atol=1e-6)
        π_c /= total
        π_a /= total
        π_n /= total
    end

    return StrataProportions(π_c, π_a, π_n, first_stage_se)
end

"""
    robust_vcov(X, residuals) -> Matrix

Compute heteroskedasticity-robust (HC0) variance-covariance matrix.

V = (X'X)⁻¹ · (Σᵢ εᵢ² · xᵢxᵢ') · (X'X)⁻¹
"""
function robust_vcov(X::Matrix{T}, residuals::Vector{T}) where {T<:Real}
    n, k = size(X)
    XtX_inv = inv(X' * X)

    # Meat of sandwich
    meat = zeros(T, k, k)
    for i in 1:n
        x_i = X[i, :]
        meat .+= (residuals[i]^2) * (x_i * x_i')
    end

    return XtX_inv * meat * XtX_inv
end

# =============================================================================
# EM Estimator
# =============================================================================

"""
    EMEstimator <: AbstractPSEstimator

EM algorithm for CACE estimation treating strata as latent variables.

# Fields
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)
- `add_intercept::Bool`: Add intercept to covariates (default: true)
"""
struct EMEstimator <: AbstractPSEstimator
    max_iter::Int
    tol::Float64
    add_intercept::Bool
end

EMEstimator(; max_iter::Int=100, tol::Float64=1e-6, add_intercept::Bool=true) =
    EMEstimator(max_iter, tol, add_intercept)

"""
    solve(problem::CACEProblem, estimator::EMEstimator) -> CACESolution

Estimate CACE using EM algorithm with latent strata membership.

# Algorithm
E-Step: Compute posterior strata probabilities P(S|Y,D,Z;θ)
M-Step: Update parameters using weighted maximum likelihood

# Key Insight: Strata Identification from (D,Z)
| Observed (D,Z) | Possible Strata           | Identification |
|----------------|---------------------------|----------------|
| D=1, Z=0       | Always-taker (definite)   | Identified     |
| D=0, Z=1       | Never-taker (definite)    | Identified     |
| D=1, Z=1       | Complier OR Always-taker  | Ambiguous      |
| D=0, Z=0       | Complier OR Never-taker   | Ambiguous      |

The EM algorithm marginalizes over ambiguous cases using outcome distribution.
"""
function solve(problem::CACEProblem{T}, estimator::EMEstimator) where {T<:Real}
    Y = problem.outcome
    D = Float64.(problem.treatment)
    Z = Float64.(problem.instrument)
    α = problem.parameters.alpha

    n = length(Y)
    max_iter = estimator.max_iter
    tol = estimator.tol

    # =========================================================================
    # Initialize from 2SLS (warm start)
    # =========================================================================
    init_result = solve(problem, CACETwoSLS())

    μ_c1 = init_result.cace  # E[Y(1)|complier]
    μ_c0 = 0.0               # E[Y(0)|complier] = baseline
    μ_a = mean(Y[(D .== 1.0) .& (Z .== 0.0)])  # Always-takers
    μ_n = mean(Y[(D .== 0.0) .& (Z .== 1.0)])  # Never-takers
    σ² = var(Y)

    π_c = init_result.strata_proportions.compliers
    π_a = init_result.strata_proportions.always_takers
    π_n = init_result.strata_proportions.never_takers

    # =========================================================================
    # EM Iteration
    # =========================================================================
    converged = false
    n_iter = 0
    prev_ll = -Inf

    for iter in 1:max_iter
        n_iter = iter

        # E-Step: Compute posterior strata probabilities
        post_c, post_a, post_n = e_step_ps(Y, D, Z, π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ²)

        # M-Step: Update parameters
        π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ² = m_step_ps(
            Y, D, Z, post_c, post_a, post_n
        )

        # Compute log-likelihood for convergence check
        ll = compute_em_log_likelihood(Y, D, Z, π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ²)

        # Check convergence
        if iter > 1 && abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol
            converged = true
            break
        end
        prev_ll = ll
    end

    # Warn if not converged
    if !converged
        @warn "EM did not converge in $max_iter iterations"
    end

    # =========================================================================
    # Compute CACE and Standard Error
    # =========================================================================
    cace = μ_c1 - μ_c0

    # SE via Louis Information approximation
    cace_se = compute_em_variance(Y, D, Z, π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ², cace)

    # =========================================================================
    # Inference
    # =========================================================================
    z_stat = cace / cace_se
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))
    z_crit = quantile(Normal(), 1 - α/2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # =========================================================================
    # First stage and reduced form (for compatibility)
    # =========================================================================
    first_stage_coef = π_c  # Complier proportion
    first_stage_se = sqrt(π_c * (1 - π_c) / n)
    first_stage_f = (first_stage_coef / first_stage_se)^2

    # Reduced form
    Y_z1 = mean(Y[Z .== 1.0])
    Y_z0 = mean(Y[Z .== 0.0])
    reduced_form = Y_z1 - Y_z0
    reduced_form_se = sqrt(var(Y[Z .== 1.0]) / sum(Z .== 1.0) +
                           var(Y[Z .== 0.0]) / sum(Z .== 0.0))

    # =========================================================================
    # Package result
    # =========================================================================
    strata_props = StrataProportions(π_c, π_a, π_n, first_stage_se)

    n_treated_assigned = sum(problem.instrument)
    n_control_assigned = n - n_treated_assigned

    return CACESolution{T}(
        cace,
        cace_se,
        ci_lower,
        ci_upper,
        z_stat,
        pvalue,
        strata_props,
        first_stage_coef,
        first_stage_se,
        first_stage_f,
        reduced_form,
        reduced_form_se,
        n,
        n_treated_assigned,
        n_control_assigned,
        :em
    )
end

"""
    e_step_ps(Y, D, Z, π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ²)

E-Step: Compute posterior strata probabilities using Bayes' rule.

P(S=s|Y,D,Z) ∝ P(S=s) · P(Y|S=s,D) · P(D|S=s,Z)

Uses log-sum-exp trick for numerical stability.
"""
function e_step_ps(
    Y::Vector{T}, D::Vector{Float64}, Z::Vector{Float64},
    π_c::Float64, π_a::Float64, π_n::Float64,
    μ_c0::Float64, μ_c1::Float64, μ_a::Float64, μ_n::Float64, σ²::Float64
) where {T<:Real}
    n = length(Y)
    σ = sqrt(σ²)

    post_c = zeros(n)
    post_a = zeros(n)
    post_n = zeros(n)

    for i in 1:n
        d_i = D[i]
        z_i = Z[i]
        y_i = Y[i]

        # Compute log-likelihoods for each stratum

        if d_i == 1.0 && z_i == 0.0
            # Definitely always-taker
            post_c[i] = 0.0
            post_a[i] = 1.0
            post_n[i] = 0.0

        elseif d_i == 0.0 && z_i == 1.0
            # Definitely never-taker
            post_c[i] = 0.0
            post_a[i] = 0.0
            post_n[i] = 1.0

        elseif d_i == 1.0 && z_i == 1.0
            # Could be complier (treated) or always-taker
            log_c = log(max(π_c, 1e-10)) + logpdf(Normal(μ_c1, σ), y_i)
            log_a = log(max(π_a, 1e-10)) + logpdf(Normal(μ_a, σ), y_i)

            # Log-sum-exp for normalization
            max_log = max(log_c, log_a)
            sum_exp = exp(log_c - max_log) + exp(log_a - max_log)

            post_c[i] = exp(log_c - max_log) / sum_exp
            post_a[i] = exp(log_a - max_log) / sum_exp
            post_n[i] = 0.0

        else  # d_i == 0.0 && z_i == 0.0
            # Could be complier (untreated) or never-taker
            log_c = log(max(π_c, 1e-10)) + logpdf(Normal(μ_c0, σ), y_i)
            log_n = log(max(π_n, 1e-10)) + logpdf(Normal(μ_n, σ), y_i)

            # Log-sum-exp for normalization
            max_log = max(log_c, log_n)
            sum_exp = exp(log_c - max_log) + exp(log_n - max_log)

            post_c[i] = exp(log_c - max_log) / sum_exp
            post_a[i] = 0.0
            post_n[i] = exp(log_n - max_log) / sum_exp
        end
    end

    return post_c, post_a, post_n
end

"""
    m_step_ps(Y, D, Z, post_c, post_a, post_n)

M-Step: Update parameters using weighted maximum likelihood.
"""
function m_step_ps(
    Y::Vector{T}, D::Vector{Float64}, Z::Vector{Float64},
    post_c::Vector{Float64}, post_a::Vector{Float64}, post_n::Vector{Float64}
) where {T<:Real}
    n = length(Y)

    # Update strata proportions
    π_c = sum(post_c) / n
    π_a = sum(post_a) / n
    π_n = sum(post_n) / n

    # Normalize
    total = π_c + π_a + π_n
    π_c /= total
    π_a /= total
    π_n /= total

    # Clip to valid range
    π_c = clamp(π_c, 1e-10, 1.0 - 2e-10)
    π_a = clamp(π_a, 1e-10, 1.0 - 2e-10)
    π_n = clamp(π_n, 1e-10, 1.0 - 2e-10)

    # Update outcome means
    # Compliers when treated (D=1, Z=1)
    mask_c1 = (D .== 1.0) .& (Z .== 1.0)
    weights_c1 = post_c[mask_c1]
    if sum(weights_c1) > 1e-10
        μ_c1 = sum(weights_c1 .* Y[mask_c1]) / sum(weights_c1)
    else
        μ_c1 = mean(Y)
    end

    # Compliers when untreated (D=0, Z=0)
    mask_c0 = (D .== 0.0) .& (Z .== 0.0)
    weights_c0 = post_c[mask_c0]
    if sum(weights_c0) > 1e-10
        μ_c0 = sum(weights_c0 .* Y[mask_c0]) / sum(weights_c0)
    else
        μ_c0 = mean(Y)
    end

    # Always-takers (D=1, any Z)
    mask_a = D .== 1.0
    weights_a = post_a[mask_a]
    if sum(weights_a) > 1e-10
        μ_a = sum(weights_a .* Y[mask_a]) / sum(weights_a)
    else
        μ_a = mean(Y[D .== 1.0])
    end

    # Never-takers (D=0, any Z)
    mask_n = D .== 0.0
    weights_n = post_n[mask_n]
    if sum(weights_n) > 1e-10
        μ_n = sum(weights_n .* Y[mask_n]) / sum(weights_n)
    else
        μ_n = mean(Y[D .== 0.0])
    end

    # Update variance (weighted pooled)
    total_weight = 0.0
    weighted_ss = 0.0

    # Compliers treated
    for i in findall(mask_c1)
        weighted_ss += post_c[i] * (Y[i] - μ_c1)^2
        total_weight += post_c[i]
    end

    # Compliers untreated
    for i in findall(mask_c0)
        weighted_ss += post_c[i] * (Y[i] - μ_c0)^2
        total_weight += post_c[i]
    end

    # Always-takers
    for i in findall(mask_a)
        weighted_ss += post_a[i] * (Y[i] - μ_a)^2
        total_weight += post_a[i]
    end

    # Never-takers
    for i in findall(mask_n)
        weighted_ss += post_n[i] * (Y[i] - μ_n)^2
        total_weight += post_n[i]
    end

    σ² = weighted_ss / max(total_weight, 1e-10)
    σ² = max(σ², 1e-6)  # Floor variance

    return π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ²
end

"""
    compute_em_log_likelihood(...)

Compute complete-data log-likelihood for convergence monitoring.
"""
function compute_em_log_likelihood(
    Y::Vector{T}, D::Vector{Float64}, Z::Vector{Float64},
    π_c::Float64, π_a::Float64, π_n::Float64,
    μ_c0::Float64, μ_c1::Float64, μ_a::Float64, μ_n::Float64, σ²::Float64
) where {T<:Real}
    σ = sqrt(σ²)
    ll = 0.0

    for i in eachindex(Y)
        d_i = D[i]
        z_i = Z[i]
        y_i = Y[i]

        if d_i == 1.0 && z_i == 0.0
            # Always-taker
            ll += log(max(π_a, 1e-10)) + logpdf(Normal(μ_a, σ), y_i)
        elseif d_i == 0.0 && z_i == 1.0
            # Never-taker
            ll += log(max(π_n, 1e-10)) + logpdf(Normal(μ_n, σ), y_i)
        elseif d_i == 1.0 && z_i == 1.0
            # Complier (treated) or always-taker
            p_c = π_c * pdf(Normal(μ_c1, σ), y_i)
            p_a = π_a * pdf(Normal(μ_a, σ), y_i)
            ll += log(max(p_c + p_a, 1e-300))
        else
            # Complier (untreated) or never-taker
            p_c = π_c * pdf(Normal(μ_c0, σ), y_i)
            p_n = π_n * pdf(Normal(μ_n, σ), y_i)
            ll += log(max(p_c + p_n, 1e-300))
        end
    end

    return ll
end

"""
    compute_em_variance(...)

Compute CACE variance using Louis Information Formula approximation.

Uses weighted variance from both treatment groups with entropy-based inflation.
"""
function compute_em_variance(
    Y::Vector{T}, D::Vector{Float64}, Z::Vector{Float64},
    π_c::Float64, π_a::Float64, π_n::Float64,
    μ_c0::Float64, μ_c1::Float64, μ_a::Float64, μ_n::Float64,
    σ²::Float64, cace::Float64
) where {T<:Real}
    n = length(Y)

    # Recompute posteriors at final parameters
    post_c, post_a, post_n = e_step_ps(Y, D, Z, π_c, π_a, π_n, μ_c0, μ_c1, μ_a, μ_n, σ²)

    # Effective sample sizes for compliers in each treatment group
    mask_d1 = D .== 1.0
    mask_d0 = D .== 0.0

    n_c_d1 = sum(post_c[mask_d1])
    n_c_d0 = sum(post_c[mask_d0])

    # Variance of μ_1 (compliers with D=1)
    if n_c_d1 > 1
        var_mu_1 = sum(post_c[mask_d1] .* (Y[mask_d1] .- μ_c1).^2) / n_c_d1
        var_mu_1 /= max(n_c_d1, 1.0)
    else
        var_mu_1 = σ² / n
    end

    # Variance of μ_0 (compliers with D=0)
    if n_c_d0 > 1
        var_mu_0 = sum(post_c[mask_d0] .* (Y[mask_d0] .- μ_c0).^2) / n_c_d0
        var_mu_0 /= max(n_c_d0, 1.0)
    else
        var_mu_0 = σ² / n
    end

    # Base variance of CACE = Var(μ_1 - μ_0) ≈ Var(μ_1) + Var(μ_0)
    base_var = var_mu_1 + var_mu_0

    # Entropy-based inflation for missing data uncertainty
    entropy = 0.0
    for i in eachindex(Y)
        for p in [post_c[i], post_a[i], post_n[i]]
            if p > 1e-10 && p < 1.0 - 1e-10
                entropy -= p * log(p)
            end
        end
    end

    # Normalize entropy per observation
    avg_entropy = entropy / n

    # Inflation factor: higher entropy → more uncertainty
    # Use 2x base inflation to be more conservative
    inflation = 1.0 + 2.0 * avg_entropy

    # Total variance
    total_var = base_var * inflation

    return sqrt(max(total_var, 1e-10))
end

"""
    cace_em(Y, D, Z; max_iter=100, tol=1e-6, alpha=0.05) -> NamedTuple

Convenience function to estimate CACE via EM algorithm.

# Arguments
- `Y`: Outcome vector
- `D`: Treatment received (binary)
- `Z`: Instrument/assignment (binary)
- `max_iter`: Maximum EM iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `alpha`: Significance level (default: 0.05)

# Returns
NamedTuple with CACE estimate, SE, CI, and strata proportions.

# Example
```julia
result = cace_em(Y, D, Z)
println("CACE (EM): \$(result.cace) ± \$(result.se)")
println("Method: \$(result.method)")
```
"""
function cace_em(
    outcome::Vector{T},
    treatment::Union{Vector{Bool}, BitVector, Vector{<:Integer}},
    instrument::Union{Vector{Bool}, BitVector, Vector{<:Integer}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    alpha::Float64=0.05
) where {T<:Real}
    problem = CACEProblem(outcome, treatment, instrument, nothing,
                          (alpha=alpha, inference=:robust))
    solution = solve(problem, EMEstimator(max_iter=max_iter, tol=tol))

    return (
        cace = solution.cace,
        se = solution.se,
        ci_lower = solution.ci_lower,
        ci_upper = solution.ci_upper,
        z_stat = solution.z_stat,
        pvalue = solution.pvalue,
        strata_proportions = NamedTuple(solution.strata_proportions),
        first_stage_coef = solution.first_stage_coef,
        first_stage_se = solution.first_stage_se,
        first_stage_f = solution.first_stage_f,
        reduced_form = solution.reduced_form,
        reduced_form_se = solution.reduced_form_se,
        n = solution.n,
        n_treated_assigned = solution.n_treated_assigned,
        n_control_assigned = solution.n_control_assigned,
        method = solution.method
    )
end

# Export
export CACEProblem, CACESolution, StrataProportions
export CACETwoSLS, WaldEstimator, EMEstimator
export solve, cace_2sls, wald_estimator, cace_em
