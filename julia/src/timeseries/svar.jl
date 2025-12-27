"""
Structural VAR Estimation

Session 137: SVAR identification, IRF, and FEVD.
"""
module SVAR

using LinearAlgebra
using Statistics

using ..TimeSeriesTypes
using ..VAR: var_estimate
using ..SVARTypes: SVARResult, IRFResult, FEVDResult, IdentificationMethod, CHOLESKY, LONG_RUN

export cholesky_svar, long_run_svar, companion_form, vma_coefficients, structural_vma_coefficients
export check_stability, long_run_impact_matrix, verify_identification
export compute_irf, compute_fevd


"""
    cholesky_svar(var_result; ordering=nothing)

Structural VAR using Cholesky (recursive) identification.

The Cholesky decomposition of Σ_u yields a lower triangular B₀⁻¹:
    Σ_u = P P'  =>  B₀⁻¹ = P

Interpretation: Variables are ordered causally.
- First variable is exogenous (not affected contemporaneously by others)
- Last variable affected by all others contemporaneously

# Arguments
- `var_result::VARResult`: Estimated reduced-form VAR model
- `ordering::Vector{String}`: Variable ordering for Cholesky decomposition

# Returns
- `SVARResult`: Structural VAR estimation results

# Example
```julia
using Random
Random.seed!(42)
data = randn(200, 3)
var_result = var_estimate(data, lags=2)
svar_result = cholesky_svar(var_result)
```
"""
function cholesky_svar(var_result::VARResult; ordering::Union{Vector{String},Nothing}=nothing)
    n_vars = length(var_result.var_names)
    sigma = var_result.sigma

    # Handle ordering
    if ordering !== nothing
        if length(ordering) != n_vars
            error("ordering has $(length(ordering)) elements, expected $n_vars")
        end
        for name in ordering
            if !(name in var_result.var_names)
                error("Variable '$name' not in VAR model")
            end
        end

        # Permute covariance matrix
        perm = [findfirst(==(name), var_result.var_names) for name in ordering]
        P = zeros(n_vars, n_vars)
        for (i, j) in enumerate(perm)
            P[i, j] = 1.0
        end
        sigma_ordered = P * sigma * P'
    else
        ordering = var_result.var_names
        sigma_ordered = sigma
        P = Matrix{Float64}(I, n_vars, n_vars)
    end

    # Cholesky decomposition: Σ = L L'
    L = cholesky(Symmetric(sigma_ordered)).L

    # B₀⁻¹ = L (in reordered space)
    B0_inv_ordered = Matrix(L)

    # Transform back to original ordering
    if ordering != var_result.var_names
        P_inv = P'
        B0_inv = P_inv * B0_inv_ordered * P_inv'
    else
        B0_inv = B0_inv_ordered
    end

    # Compute B₀ = (B₀⁻¹)⁻¹
    B0 = inv(B0_inv)

    # Compute structural shocks: ε_t = B₀ u_t
    residuals = var_result.residuals
    structural_shocks = Matrix((B0 * residuals')')

    # Identification info
    n_restrictions = n_vars * (n_vars - 1) ÷ 2

    SVARResult(
        var_coefficients=var_result.coefficients,
        var_residuals=residuals,
        var_sigma=sigma,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=CHOLESKY,
        n_restrictions=n_restrictions,
        n_vars=n_vars,
        n_obs=var_result.n_obs_effective,
        lags=var_result.lags,
        var_names=var_result.var_names,
        ordering=ordering,
        log_likelihood=var_result.log_likelihood
    )
end


"""
    long_run_svar(var_result; ordering=nothing)

Structural VAR using Blanchard-Quah long-run identification.

Imposes restrictions on the long-run cumulative impact matrix C(1):
    C(1) = (I - A₁ - ... - Aₚ)⁻¹ B₀⁻¹ = Ξ · B₀⁻¹

The Cholesky decomposition of the long-run covariance yields a lower
triangular C(1), meaning shock j has no permanent effect on variable i
for j > i.

# Arguments
- `var_result::VARResult`: Estimated reduced-form VAR model. Must be stable.
- `ordering::Vector{String}`: Variable ordering for long-run restrictions

# Returns
- `SVARResult`: Structural VAR estimation results with long-run identification

# Throws
- Error if VAR is not stable (has unit root or explosive roots)
- Error if ordering contains invalid variable names

# Example
```julia
using Random
Random.seed!(42)
data = randn(300, 2)
var_result = var_estimate(data, lags=2)
svar_result = long_run_svar(var_result)
# Verify C(1) is lower triangular
C1 = long_run_impact_matrix(svar_result)
@assert isapprox(triu(C1, 1), zeros(2, 2), atol=1e-10)
```

# References
Blanchard, O. J., & Quah, D. (1989). The dynamic effects of aggregate
demand and supply disturbances. American Economic Review, 79(4), 655-673.
"""
function long_run_svar(var_result::VARResult; ordering::Union{Vector{String},Nothing}=nothing)
    n_vars = length(var_result.var_names)
    sigma = var_result.sigma

    # Step 0: Check stability (required for long-run identification)
    is_stable, eigenvalues = check_stability(var_result)
    if !is_stable
        max_modulus = maximum(abs.(eigenvalues))
        error("VAR is not stable (max eigenvalue modulus = $(round(max_modulus, digits=4)) >= 1). " *
              "Long-run identification requires a stable VAR. " *
              "Consider differencing the data or using fewer lags.")
    end

    # Handle ordering (same pattern as cholesky_svar)
    if ordering !== nothing
        if length(ordering) != n_vars
            error("ordering has $(length(ordering)) elements, expected $n_vars")
        end
        for name in ordering
            if !(name in var_result.var_names)
                error("Variable '$name' not in VAR model")
            end
        end

        # Permute to ordering space
        perm = [findfirst(==(name), var_result.var_names) for name in ordering]
        P_perm = zeros(n_vars, n_vars)
        for (i, j) in enumerate(perm)
            P_perm[i, j] = 1.0
        end
        sigma_ordered = P_perm * sigma * P_perm'
    else
        ordering = var_result.var_names
        sigma_ordered = sigma
        P_perm = Matrix{Float64}(I, n_vars, n_vars)
    end

    # Step 1: Compute sum of lag coefficient matrices (in ordering space)
    A_sum = zeros(n_vars, n_vars)
    for lag in 1:var_result.lags
        A_lag = get_lag_matrix(var_result, lag)
        if ordering != var_result.var_names
            A_lag = P_perm * A_lag * P_perm'
        end
        A_sum += A_lag
    end

    # Step 2: Compute Ξ = (I - A_sum)⁻¹ (long-run multiplier)
    I_minus_A = Matrix{Float64}(I, n_vars, n_vars) - A_sum
    Xi = try
        inv(I_minus_A)
    catch e
        error("Cannot compute long-run multiplier: (I - A_sum) is singular. " *
              "This typically indicates a unit root or near-unit-root process.")
    end

    # Step 3: Long-run reduced-form covariance: Ω = Ξ Σᵤ Ξ'
    Omega = Xi * sigma_ordered * Xi'

    # Step 4: Cholesky decomposition: Ω = P P' (P is lower triangular)
    P_chol = try
        cholesky(Symmetric(Omega)).L
    catch e
        error("Long-run covariance matrix Ω is not positive definite. " *
              "This may indicate numerical issues or near-singularity. " *
              "Original error: $e")
    end

    # Step 5: Back out B₀⁻¹ = Ξ⁻¹ P = (I - A_sum) P
    B0_inv_ordered = I_minus_A * P_chol

    # Transform back to original ordering
    if ordering != var_result.var_names
        P_perm_inv = P_perm'
        B0_inv = P_perm_inv * B0_inv_ordered * P_perm_inv'
    else
        B0_inv = Matrix(B0_inv_ordered)
    end

    # Compute B₀ = (B₀⁻¹)⁻¹
    B0 = inv(B0_inv)

    # Compute structural shocks: ε_t = B₀ u_t
    residuals = var_result.residuals
    structural_shocks = Matrix((B0 * residuals')')

    # Identification info
    n_restrictions = n_vars * (n_vars - 1) ÷ 2

    SVARResult(
        var_coefficients=var_result.coefficients,
        var_residuals=residuals,
        var_sigma=sigma,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=LONG_RUN,
        n_restrictions=n_restrictions,
        n_vars=n_vars,
        n_obs=var_result.n_obs_effective,
        lags=var_result.lags,
        var_names=var_result.var_names,
        ordering=ordering,
        log_likelihood=var_result.log_likelihood
    )
end


"""
    companion_form(var_result)

Build VAR companion matrix for IRF computation.

For VAR(p): Y_t = A₁ Y_{t-1} + ... + Aₚ Y_{t-p} + u_t

The companion form is:
[Y_t    ]   [A₁ A₂ ... A_{p-1} Aₚ] [Y_{t-1}  ]
[Y_{t-1}] = [I  0  ... 0      0 ] [Y_{t-2}  ]
[  ...  ]   [0  I  ... 0      0 ] [  ...    ]
[Y_{t-p+1}] [0  0  ... I      0 ] [Y_{t-p}  ]

# Arguments
- `var_result::VARResult`: Estimated VAR model

# Returns
- `Matrix{Float64}`: (n_vars * lags, n_vars * lags) companion matrix
"""
function companion_form(var_result::VARResult)
    n_vars = length(var_result.var_names)
    lags = var_result.lags
    m = n_vars * lags

    F = zeros(m, m)

    # Fill in lag coefficient matrices in first block row
    for lag in 1:lags
        A_lag = get_lag_matrix(var_result, lag)
        col_start = (lag - 1) * n_vars + 1
        col_end = lag * n_vars
        F[1:n_vars, col_start:col_end] = A_lag
    end

    # Fill in identity blocks on sub-diagonal
    if lags > 1
        F[n_vars+1:end, 1:(lags-1)*n_vars] = Matrix{Float64}(I, (lags-1)*n_vars, (lags-1)*n_vars)
    end

    return F
end


"""
    vma_coefficients(var_result, horizons)

Compute VMA (Vector Moving Average) coefficients.

Y_t = Σ_{h=0}^{∞} Φ_h u_{t-h}

where Φ_0 = I and Φ_h are the impulse response coefficients.
"""
function vma_coefficients(var_result::VARResult, horizons::Int)
    n_vars = length(var_result.var_names)
    lags = var_result.lags

    Phi = zeros(n_vars, n_vars, horizons + 1)
    Phi[:, :, 1] = Matrix{Float64}(I, n_vars, n_vars)  # Φ_0 = I

    if lags == 0
        return Phi
    end

    # Use companion form
    F = companion_form(var_result)
    m = size(F, 1)

    # Selector: J = [I_k, 0, ..., 0]
    J = zeros(n_vars, m)
    J[1:n_vars, 1:n_vars] = Matrix{Float64}(I, n_vars, n_vars)

    # Φ_h = J F^h J'
    F_power = Matrix{Float64}(I, m, m)
    for h in 1:horizons
        F_power = F_power * F
        Phi[:, :, h+1] = J * F_power * J'
    end

    return Phi
end


"""
    structural_vma_coefficients(svar_result, horizons)

Compute structural VMA coefficients (orthogonalized IRF).

Ψ_h = Φ_h B₀⁻¹
"""
function structural_vma_coefficients(svar_result::SVARResult, horizons::Int)
    # Reconstruct VARResult-like object for vma_coefficients
    # Using positional constructor: coefficients, residuals, aic, bic, hqc, lags, n_obs, n_obs_effective, var_names, sigma, log_likelihood
    var_result = VARResult(
        svar_result.var_coefficients,
        svar_result.var_residuals,
        0.0,  # aic
        0.0,  # bic
        0.0,  # hqc
        svar_result.lags,
        svar_result.n_obs + svar_result.lags,
        svar_result.n_obs,
        svar_result.var_names,
        svar_result.var_sigma,
        svar_result.log_likelihood
    )

    Phi = vma_coefficients(var_result, horizons)
    B0_inv = svar_result.B0_inv

    n_vars = svar_result.n_vars
    Psi = zeros(n_vars, n_vars, horizons + 1)

    for h in 0:horizons
        Psi[:, :, h+1] = Phi[:, :, h+1] * B0_inv
    end

    return Psi
end


"""
    check_stability(var_result)

Check VAR stability (stationarity).

VAR is stable if all eigenvalues of companion matrix are inside unit circle.
"""
function check_stability(var_result::VARResult)
    F = companion_form(var_result)
    eigenvalues = eigvals(F)
    moduli = abs.(eigenvalues)
    is_stable = all(moduli .< 1.0)
    return is_stable, eigenvalues
end


"""
    long_run_impact_matrix(svar_result)

Compute long-run impact matrix.

Ξ = (I - A₁ - ... - Aₚ)⁻¹ B₀⁻¹
"""
function long_run_impact_matrix(svar_result::SVARResult)
    n_vars = svar_result.n_vars
    lags = svar_result.lags

    # Reconstruct VARResult to get lag matrices
    # Using positional constructor: coefficients, residuals, aic, bic, hqc, lags, n_obs, n_obs_effective, var_names, sigma, log_likelihood
    var_result = VARResult(
        svar_result.var_coefficients,
        svar_result.var_residuals,
        0.0,  # aic
        0.0,  # bic
        0.0,  # hqc
        svar_result.lags,
        svar_result.n_obs + svar_result.lags,
        svar_result.n_obs,
        svar_result.var_names,
        svar_result.var_sigma,
        svar_result.log_likelihood
    )

    # Sum of lag coefficient matrices
    A_sum = zeros(n_vars, n_vars)
    for lag in 1:lags
        A_sum += get_lag_matrix(var_result, lag)
    end

    # (I - A₁ - ... - Aₚ)⁻¹
    long_run_mult = inv(Matrix{Float64}(I, n_vars, n_vars) - A_sum)

    return long_run_mult * svar_result.B0_inv
end


"""
    verify_identification(sigma, B0_inv; tol=1e-8)

Verify SVAR identification by checking Σ_u = B₀⁻¹ (B₀⁻¹)'.
"""
function verify_identification(sigma::Matrix{Float64}, B0_inv::Matrix{Float64}; tol::Float64=1e-8)
    reconstructed = B0_inv * B0_inv'
    error_matrix = abs.(reconstructed - sigma)
    max_error = maximum(error_matrix)
    is_valid = max_error < tol
    return is_valid, max_error
end


"""
    compute_irf(svar_result; horizons=20, cumulative=false)

Compute impulse response functions for SVAR.

# Arguments
- `svar_result::SVARResult`: Structural VAR result
- `horizons::Int`: Maximum horizon (0 to horizons inclusive)
- `cumulative::Bool`: If true, return cumulative IRF

# Returns
- `IRFResult`: Impulse response function results
"""
function compute_irf(svar_result::SVARResult; horizons::Int=20, cumulative::Bool=false)
    if horizons < 0
        error("horizons must be >= 0, got $horizons")
    end

    irf = structural_vma_coefficients(svar_result, horizons)

    if cumulative
        irf_cumulative = similar(irf)
        for h in 1:(horizons+1)
            irf_cumulative[:, :, h] = sum(irf[:, :, 1:h], dims=3)[:, :, 1]
        end
        irf = irf_cumulative
    end

    IRFResult(
        irf=irf,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=true,
        var_names=svar_result.var_names
    )
end


"""
    compute_fevd(svar_result; horizons=20)

Compute Forecast Error Variance Decomposition.

FEVD measures what proportion of variable i's h-step ahead forecast
error variance is due to structural shock j.
"""
function compute_fevd(svar_result::SVARResult; horizons::Int=20)
    if horizons < 0
        error("horizons must be >= 0, got $horizons")
    end

    Psi = structural_vma_coefficients(svar_result, horizons)
    n_vars = svar_result.n_vars

    fevd = zeros(n_vars, n_vars, horizons + 1)

    for h in 0:horizons
        # Sum of squared IRFs up to horizon h
        cumsum_squared = zeros(n_vars, n_vars)
        for k in 0:h
            cumsum_squared += Psi[:, :, k+1].^2
        end

        # Total MSE for each variable
        total_mse = sum(cumsum_squared, dims=2)[:, 1]

        # FEVD = cumulative contribution / total MSE
        for i in 1:n_vars
            if total_mse[i] > 1e-12
                fevd[i, :, h+1] = cumsum_squared[i, :] / total_mse[i]
            else
                fevd[i, :, h+1] .= 1.0 / n_vars
            end
        end
    end

    FEVDResult(
        fevd=fevd,
        horizons=horizons,
        var_names=svar_result.var_names
    )
end

end # module
