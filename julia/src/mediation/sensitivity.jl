"""
Sensitivity analysis for mediation.

Assesses robustness of mediation effects to violations of
sequential ignorability (unmeasured confounding).

Based on Imai, Keele, Yamamoto (2010) sensitivity approach.

References:
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

using Statistics
using LinearAlgebra
using Distributions
using Random


"""
    mediation_sensitivity(outcome, treatment, mediator; kwargs...)

Sensitivity analysis for unmeasured confounding.

Assesses how NDE/NIE estimates change under varying degrees of
unmeasured confounding between mediator and outcome.

The sensitivity parameter rho represents the correlation between
the error terms in the mediator and outcome models:
- rho = 0: No unmeasured confounding (sequential ignorability holds)
- rho > 0: Positive confounding
- rho < 0: Negative confounding

# Arguments
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector`: Treatment variable T
- `mediator::Vector`: Mediator variable M
- `covariates::Union{Nothing, Matrix}`: Pre-treatment covariates X
- `rho_range::Tuple{T,T}`: Range of sensitivity parameter
- `n_rho::Int`: Number of rho values in grid
- `n_bootstrap::Int`: Bootstrap replications for CIs
- `alpha::Float64`: Significance level
- `rng::Union{Nothing, AbstractRNG}`: Random number generator

# Returns
- `SensitivityResult`: Grid of effects at each rho value with CIs

# Notes
The interpretation depends on the magnitude of rho_at_zero_*:
- Large |rho| needed to nullify effect: Robust finding
- Small |rho| needed to nullify effect: Sensitive to confounding

Example interpretation:
- rho_at_zero_nie = 0.3: Only moderate confounding needed to explain
  away the indirect effect → proceed with caution
- rho_at_zero_nie = 0.8: Very strong confounding needed → robust

# Example
```julia
using Random
rng = MersenneTwister(42)
n = 500
T = Float64.(rand(rng, n) .< 0.5)
M = 0.5 .+ 0.6 .* T .+ 0.5 .* randn(rng, n)
Y = 1.0 .+ 0.5 .* T .+ 0.8 .* M .+ 0.5 .* randn(rng, n)
result = mediation_sensitivity(Y, T, M; n_bootstrap=100)
println("rho to nullify NIE: \$(result.rho_at_zero_nie)")
```
"""
function mediation_sensitivity(
    outcome::Vector{T},
    treatment::Vector,
    mediator::Vector;
    covariates::Union{Nothing, Matrix} = nothing,
    rho_range::Tuple{Real, Real} = (-0.9, 0.9),
    n_rho::Int = 41,
    n_bootstrap::Int = 100,
    alpha::Float64 = 0.05,
    rng::Union{Nothing, AbstractRNG} = nothing
) where T<:Real
    if rng === nothing
        rng = Random.default_rng()
    end

    n = length(outcome)
    treatment = T.(treatment)
    mediator = T.(mediator)

    # Build design matrices
    if covariates !== nothing
        covariates = T.(covariates)
        X_m = hcat(ones(T, n), treatment, covariates)
        X_y = hcat(ones(T, n), treatment, mediator, covariates)
    else
        X_m = hcat(ones(T, n), treatment)
        X_y = hcat(ones(T, n), treatment, mediator)
    end

    # Fit baseline models
    beta_m = X_m \ mediator
    resid_m = mediator .- X_m * beta_m
    m_resid_var = sum(resid_m.^2) / (n - size(X_m, 2))

    beta_y = X_y \ outcome
    resid_y = outcome .- X_y * beta_y
    y_resid_var = sum(resid_y.^2) / (n - size(X_y, 2))

    # Create rho grid
    rho_grid = collect(range(T(rho_range[1]), T(rho_range[2]), length=n_rho))

    # Initialize output arrays
    nde_at_rho = zeros(T, n_rho)
    nie_at_rho = zeros(T, n_rho)
    nde_ci_lower = zeros(T, n_rho)
    nde_ci_upper = zeros(T, n_rho)
    nie_ci_lower = zeros(T, n_rho)
    nie_ci_upper = zeros(T, n_rho)

    # Key coefficients
    alpha_1 = beta_m[2]  # T -> M
    beta_1 = beta_y[2]   # Direct effect (T -> Y)
    beta_2 = beta_y[3]   # M -> Y

    sigma_m = sqrt(m_resid_var)
    sigma_y = sqrt(y_resid_var)

    for (i, rho) in enumerate(rho_grid)
        # Compute adjusted effects under this rho
        # Following linear sensitivity model from Imai et al. (2010)
        # Adjusted beta_2 = beta_2 - rho * sigma_y / sigma_m
        adjustment = rho * sigma_y / sigma_m
        beta_2_adj = beta_2 - adjustment

        nde = beta_1  # Direct effect unchanged in linear model
        nie = alpha_1 * beta_2_adj

        nde_at_rho[i] = nde
        nie_at_rho[i] = nie

        # Bootstrap for CIs at this rho
        boot_nde = T[]
        boot_nie = T[]

        for _ in 1:n_bootstrap
            idx = rand(rng, 1:n, n)

            if covariates !== nothing
                X_m_b = hcat(ones(T, n), treatment[idx], covariates[idx, :])
                X_y_b = hcat(ones(T, n), treatment[idx], mediator[idx], covariates[idx, :])
            else
                X_m_b = hcat(ones(T, n), treatment[idx])
                X_y_b = hcat(ones(T, n), treatment[idx], mediator[idx])
            end

            try
                beta_m_b = X_m_b \ mediator[idx]
                resid_m_b = mediator[idx] .- X_m_b * beta_m_b

                beta_y_b = X_y_b \ outcome[idx]
                resid_y_b = outcome[idx] .- X_y_b * beta_y_b

                alpha_1_b = beta_m_b[2]
                beta_1_b = beta_y_b[2]
                beta_2_b = beta_y_b[3]

                sigma_m_b = sqrt(sum(resid_m_b.^2) / (n - size(X_m_b, 2)))
                sigma_y_b = sqrt(sum(resid_y_b.^2) / (n - size(X_y_b, 2)))

                adj_b = rho * sigma_y_b / sigma_m_b
                beta_2_adj_b = beta_2_b - adj_b

                push!(boot_nde, beta_1_b)
                push!(boot_nie, alpha_1_b * beta_2_adj_b)
            catch
                continue
            end
        end

        if length(boot_nde) > 10
            q_low = alpha / 2
            q_high = 1 - alpha / 2
            nde_ci_lower[i] = quantile(boot_nde, q_low)
            nde_ci_upper[i] = quantile(boot_nde, q_high)
            nie_ci_lower[i] = quantile(boot_nie, q_low)
            nie_ci_upper[i] = quantile(boot_nie, q_high)
        else
            nde_ci_lower[i] = T(NaN)
            nde_ci_upper[i] = T(NaN)
            nie_ci_lower[i] = T(NaN)
            nie_ci_upper[i] = T(NaN)
        end
    end

    # Find rho at which effects cross zero
    rho_at_zero_nie = _find_zero_crossing(rho_grid, nie_at_rho)
    rho_at_zero_nde = _find_zero_crossing(rho_grid, nde_at_rho)

    # Original estimates (rho = 0)
    zero_idx = argmin(abs.(rho_grid))
    original_nde = nde_at_rho[zero_idx]
    original_nie = nie_at_rho[zero_idx]

    # Generate interpretation
    interpretation = _generate_interpretation(
        original_nde, original_nie, rho_at_zero_nde, rho_at_zero_nie
    )

    return SensitivityResult{T}(
        rho_grid,
        nde_at_rho,
        nie_at_rho,
        nde_ci_lower,
        nde_ci_upper,
        nie_ci_lower,
        nie_ci_upper,
        rho_at_zero_nie,
        rho_at_zero_nde,
        original_nde,
        original_nie,
        interpretation
    )
end


"""
    _find_zero_crossing(rho_grid, effect_grid)

Find rho value where effect crosses zero via linear interpolation.
"""
function _find_zero_crossing(rho_grid::Vector{T}, effect_grid::Vector{T}) where T<:Real
    # Check for sign changes
    signs = sign.(effect_grid)
    sign_changes = findall(diff(signs) .!= 0)

    if isempty(sign_changes)
        return T(NaN)
    end

    # Use first crossing
    idx = sign_changes[1]
    rho1, rho2 = rho_grid[idx], rho_grid[idx + 1]
    eff1, eff2 = effect_grid[idx], effect_grid[idx + 1]

    # Linear interpolation to find zero crossing
    if abs(eff2 - eff1) < T(1e-10)
        return (rho1 + rho2) / 2
    end

    rho_zero = rho1 - eff1 * (rho2 - rho1) / (eff2 - eff1)

    return rho_zero
end


"""
    _generate_interpretation(original_nde, original_nie, rho_at_zero_nde, rho_at_zero_nie)

Generate human-readable interpretation of sensitivity results.
"""
function _generate_interpretation(
    original_nde::T,
    original_nie::T,
    rho_at_zero_nde::T,
    rho_at_zero_nie::T
) where T<:Real
    lines = String[]

    push!(lines, "Mediation Sensitivity Analysis Results")
    push!(lines, "=" ^ 40)
    push!(lines, "Original NDE (at rho=0): $(round(original_nde, digits=4))")
    push!(lines, "Original NIE (at rho=0): $(round(original_nie, digits=4))")
    push!(lines, "")

    # Interpret NIE sensitivity
    if isnan(rho_at_zero_nie)
        push!(lines,
            "NIE: Effect does not cross zero in the examined rho range. " *
            "The indirect effect appears robust to moderate confounding."
        )
    else
        abs_rho = abs(rho_at_zero_nie)
        if abs_rho < 0.2
            robustness = "NOT ROBUST"
            advice = "Very weak confounding could explain away the indirect effect."
        elseif abs_rho < 0.4
            robustness = "MODERATELY SENSITIVE"
            advice = "Moderate confounding could nullify the indirect effect."
        elseif abs_rho < 0.6
            robustness = "MODERATELY ROBUST"
            advice = "Substantial confounding would be needed to nullify the effect."
        else
            robustness = "ROBUST"
            advice = "Only strong confounding could explain away the indirect effect."
        end

        push!(lines,
            "NIE: Effect becomes zero at rho = $(round(rho_at_zero_nie, digits=3)) ($robustness)"
        )
        push!(lines, "     $advice")
    end

    push!(lines, "")

    # Interpret NDE sensitivity
    if isnan(rho_at_zero_nde)
        push!(lines,
            "NDE: Direct effect does not cross zero in the examined range. " *
            "The direct effect appears robust."
        )
    else
        abs_rho = abs(rho_at_zero_nde)
        if abs_rho < 0.3
            push!(lines,
                "NDE: Effect becomes zero at rho = $(round(rho_at_zero_nde, digits=3)) (SENSITIVE)"
            )
        else
            push!(lines,
                "NDE: Effect becomes zero at rho = $(round(rho_at_zero_nde, digits=3)) (ROBUST)"
            )
        end
    end

    return join(lines, "\n")
end
