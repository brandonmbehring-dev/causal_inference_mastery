"""
    Dynamic Treatment Regime (DTR) Types

Type definitions for multi-stage treatment data and estimation results
for Q-learning and A-learning methods.

References
----------
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating
    optimal dynamic treatment regimes. Statistical Science.
Robins, J. M. (2004). Optimal structural nested models for optimal sequential
    decisions. In Proceedings of the Second Seattle Symposium on Biostatistics.
"""

# Abstract type hierarchy for DTR
abstract type AbstractDTRProblem{T} end
abstract type AbstractDTREstimator end
abstract type AbstractDTRSolution end


"""
    DTRData{T<:Real}

Multi-stage treatment data for Dynamic Treatment Regimes.

Stores sequential decision data where each stage has outcomes, treatments,
and covariates/history. Supports K >= 1 decision stages.

# Fields
- `outcomes::Vector{Vector{T}}`: Outcomes at each stage [Y_1, ..., Y_K].
- `treatments::Vector{Vector{T}}`: Binary treatments at each stage [A_1, ..., A_K].
- `covariates::Vector{Matrix{T}}`: Covariates at each stage [X_1, ..., X_K].
- `n_stages::Int`: Number of decision stages K.
- `n_obs::Int`: Number of observations n.

# Properties (computed)
- `n_covariates`: Number of covariates at each stage [p_1, ..., p_K].

# Examples
```julia
using Random
Random.seed!(42)

# Single-stage DTR (like standard CATE)
n = 500
X = randn(n, 3)
A = Float64.(rand(n) .< 0.5)
Y = X[:, 1] .+ 2.0 .* A .+ randn(n)
data = DTRData([Y], [A], [X])
println("Stages: \$(data.n_stages), Obs: \$(data.n_obs)")

# Two-stage DTR
X1 = randn(n, 3)
A1 = Float64.(rand(n) .< 0.5)
Y1 = X1[:, 1] .+ A1 .+ randn(n)
X2 = hcat(X1, A1, Y1)  # history includes prior A, Y
A2 = Float64.(rand(n) .< 0.5)
Y2 = Y1 .+ X2[:, 1] .+ 2.0 .* A2 .+ randn(n)
data = DTRData([Y1, Y2], [A1, A2], [X1, X2])
println("Stages: \$(data.n_stages)")
```
"""
struct DTRData{T<:Real} <: AbstractDTRProblem{T}
    outcomes::Vector{Vector{T}}
    treatments::Vector{Vector{T}}
    covariates::Vector{Matrix{T}}
    n_stages::Int
    n_obs::Int

    function DTRData(
        outcomes::Vector{<:AbstractVector{<:Real}},
        treatments::Vector{<:AbstractVector{<:Real}},
        covariates::Vector{<:AbstractMatrix{<:Real}},
    )
        # Check non-empty
        isempty(outcomes) && error(
            "CRITICAL ERROR: Empty data.\n" *
            "Function: DTRData\n" *
            "outcomes list is empty. Must have at least one stage."
        )

        n_stages = length(outcomes)

        # Check consistent number of stages
        length(treatments) == n_stages || error(
            "CRITICAL ERROR: Stage count mismatch.\n" *
            "Function: DTRData\n" *
            "outcomes has $n_stages stages, treatments has $(length(treatments))"
        )
        length(covariates) == n_stages || error(
            "CRITICAL ERROR: Stage count mismatch.\n" *
            "Function: DTRData\n" *
            "outcomes has $n_stages stages, covariates has $(length(covariates))"
        )

        # Determine common type
        T = promote_type(
            eltype(outcomes[1]),
            eltype(treatments[1]),
            eltype(covariates[1])
        )

        # Convert and validate each stage
        converted_outcomes = Vector{Vector{T}}(undef, n_stages)
        converted_treatments = Vector{Vector{T}}(undef, n_stages)
        converted_covariates = Vector{Matrix{T}}(undef, n_stages)

        n_obs = length(outcomes[1])

        for k in 1:n_stages
            # Convert to common type
            converted_outcomes[k] = convert(Vector{T}, outcomes[k])
            converted_treatments[k] = convert(Vector{T}, treatments[k])

            # Handle 1D covariates
            if ndims(covariates[k]) == 1
                converted_covariates[k] = reshape(convert(Vector{T}, covariates[k]), :, 1)
            else
                converted_covariates[k] = convert(Matrix{T}, covariates[k])
            end

            # Check consistent n_obs across stages
            n_k = length(converted_outcomes[k])
            n_k == n_obs || error(
                "CRITICAL ERROR: Observation count mismatch.\n" *
                "Function: DTRData\n" *
                "Stage 1 has $n_obs obs, stage $k has $n_k obs"
            )

            # Check treatment and covariate lengths
            length(converted_treatments[k]) == n_obs || error(
                "CRITICAL ERROR: Length mismatch at stage $k.\n" *
                "Function: DTRData\n" *
                "outcomes: $n_obs, treatments: $(length(converted_treatments[k]))"
            )
            size(converted_covariates[k], 1) == n_obs || error(
                "CRITICAL ERROR: Length mismatch at stage $k.\n" *
                "Function: DTRData\n" *
                "outcomes: $n_obs, covariate rows: $(size(converted_covariates[k], 1))"
            )

            # Check binary treatment
            unique_treatments = unique(converted_treatments[k])
            all(a -> a in (0.0, 1.0), unique_treatments) || error(
                "CRITICAL ERROR: Non-binary treatment at stage $k.\n" *
                "Function: DTRData\n" *
                "Treatment must be binary (0 or 1), found: $unique_treatments"
            )

            # Check for NaN/Inf
            any(isnan, converted_outcomes[k]) && error(
                "CRITICAL ERROR: NaN in outcomes at stage $k.\n" *
                "Function: DTRData"
            )
            any(isinf, converted_outcomes[k]) && error(
                "CRITICAL ERROR: Inf in outcomes at stage $k.\n" *
                "Function: DTRData"
            )
        end

        new{T}(converted_outcomes, converted_treatments, converted_covariates, n_stages, n_obs)
    end
end


"""
    n_covariates(data::DTRData)

Number of covariates at each stage [p_1, ..., p_K].
"""
n_covariates(data::DTRData) = [size(X, 2) for X in data.covariates]


"""
    get_history(data::DTRData, stage::Int)

Get full history available at a given stage.

# Arguments
- `data::DTRData`: DTR data.
- `stage::Int`: Stage index (1-indexed, so stage=1 is first stage).

# Returns
- `Matrix{T}`: History H_k = (X_1, A_1, Y_1, ..., X_{k-1}, A_{k-1}, Y_{k-1}, X_k)

# Examples
```julia
data = DTRData([Y1, Y2], [A1, A2], [X1, X2])
H1 = get_history(data, 1)  # Just X1
H2 = get_history(data, 2)  # X1, A1, Y1, X2
```
"""
function get_history(data::DTRData{T}, stage::Int) where T
    (stage < 1 || stage > data.n_stages) && error(
        "CRITICAL ERROR: Invalid stage.\n" *
        "Function: get_history\n" *
        "stage must be in [1, $(data.n_stages)], got $stage"
    )

    k = stage  # Already 1-indexed

    # Build history forward
    history_parts = Vector{Matrix{T}}()

    for j in 1:(k-1)
        push!(history_parts, data.covariates[j])
        push!(history_parts, reshape(data.treatments[j], :, 1))
        push!(history_parts, reshape(data.outcomes[j], :, 1))
    end
    push!(history_parts, data.covariates[k])

    return length(history_parts) > 1 ? hcat(history_parts...) : data.covariates[k]
end


"""
    QLearningResult <: AbstractDTRSolution

Result from Q-learning estimation.

# Fields
- `value_estimate::Float64`: Estimated expected outcome under optimal regime E[Y^{d*}].
- `value_se::Float64`: Standard error of value estimate.
- `value_ci_lower::Float64`: Lower bound of confidence interval.
- `value_ci_upper::Float64`: Upper bound of confidence interval.
- `blip_coefficients::Vector{Vector{Float64}}`: Blip coefficients [ψ_1, ..., ψ_K].
- `blip_se::Vector{Vector{Float64}}`: Standard errors for blip coefficients.
- `n_stages::Int`: Number of decision stages.
- `se_method::Symbol`: SE method (:sandwich or :bootstrap).

# Notes
The optimal regime is d*_k(H) = I(H'ψ_k > 0), where H includes intercept.

# Examples
```julia
result = q_learning(data)
println("Optimal value: \$(result.value_estimate) ± \$(result.value_se)")
println("Stage 1 blip intercept: \$(result.blip_coefficients[1][1])")
```
"""
struct QLearningResult <: AbstractDTRSolution
    value_estimate::Float64
    value_se::Float64
    value_ci_lower::Float64
    value_ci_upper::Float64
    blip_coefficients::Vector{Vector{Float64}}
    blip_se::Vector{Vector{Float64}}
    n_stages::Int
    se_method::Symbol
end


"""
    optimal_regime(result::QLearningResult, history::AbstractVector, stage::Int=1)

Compute optimal treatment for given history at a stage.

# Arguments
- `result::QLearningResult`: Q-learning result.
- `history::AbstractVector`: History/covariates (without intercept).
- `stage::Int`: Decision stage (1-indexed).

# Returns
- `Int`: Optimal treatment {0, 1}.
"""
function optimal_regime(result::QLearningResult, history::AbstractVector, stage::Int=1)
    (stage < 1 || stage > result.n_stages) && error(
        "CRITICAL ERROR: Invalid stage.\n" *
        "Function: optimal_regime\n" *
        "stage must be in [1, $(result.n_stages)], got $stage"
    )

    # Add intercept
    h_aug = vcat(1.0, history)
    blip = dot(h_aug, result.blip_coefficients[stage])
    return blip > 0 ? 1 : 0
end


"""
    summary(result::QLearningResult)

Generate summary string of Q-learning results.
"""
function Base.summary(result::QLearningResult)
    lines = [
        "Q-Learning Results",
        "=" ^ 50,
        "Number of stages: $(result.n_stages)",
        "SE method: $(result.se_method)",
        "",
        "Value Function:",
        "  Optimal value: $(round(result.value_estimate, digits=4)) (SE: $(round(result.value_se, digits=4)))",
        "  95% CI: [$(round(result.value_ci_lower, digits=4)), $(round(result.value_ci_upper, digits=4))]",
        "",
        "Blip Coefficients by Stage:",
    ]

    for k in 1:result.n_stages
        push!(lines, "  Stage $k:")
        for (j, (coef, se)) in enumerate(zip(result.blip_coefficients[k], result.blip_se[k]))
            push!(lines, "    psi[$(j-1)]: $(round(coef, digits=4)) (SE: $(round(se, digits=4)))")
        end
    end

    return join(lines, "\n")
end


"""
    ALearningResult <: AbstractDTRSolution

Result from A-learning (doubly robust) estimation.

A-learning is consistent if EITHER the propensity score model OR the
baseline outcome model is correctly specified.

# Fields
- `value_estimate::Float64`: Estimated expected outcome under optimal regime.
- `value_se::Float64`: Standard error of value estimate.
- `value_ci_lower::Float64`: Lower bound of confidence interval.
- `value_ci_upper::Float64`: Upper bound of confidence interval.
- `blip_coefficients::Vector{Vector{Float64}}`: Blip coefficients [ψ_1, ..., ψ_K].
- `blip_se::Vector{Vector{Float64}}`: Standard errors for blip coefficients.
- `n_stages::Int`: Number of decision stages.
- `propensity_model::Symbol`: Propensity model (:logit or :probit).
- `outcome_model::Symbol`: Baseline outcome model (:ols or :ridge).
- `doubly_robust::Bool`: Whether DR estimation was used.
- `se_method::Symbol`: SE method (:sandwich or :bootstrap).

# Notes
A-learning is doubly robust: it is consistent if EITHER:
1. The propensity model P(A=1|H) is correctly specified, OR
2. The baseline outcome model E[Y|H, A=0] is correctly specified

# Examples
```julia
result = a_learning(data)
println("Optimal value: \$(result.value_estimate)")
println("Doubly robust: \$(result.doubly_robust)")
```
"""
struct ALearningResult <: AbstractDTRSolution
    value_estimate::Float64
    value_se::Float64
    value_ci_lower::Float64
    value_ci_upper::Float64
    blip_coefficients::Vector{Vector{Float64}}
    blip_se::Vector{Vector{Float64}}
    n_stages::Int
    propensity_model::Symbol
    outcome_model::Symbol
    doubly_robust::Bool
    se_method::Symbol
end


"""
    optimal_regime(result::ALearningResult, history::AbstractVector, stage::Int=1)

Compute optimal treatment for given history at a stage.
"""
function optimal_regime(result::ALearningResult, history::AbstractVector, stage::Int=1)
    (stage < 1 || stage > result.n_stages) && error(
        "CRITICAL ERROR: Invalid stage.\n" *
        "Function: optimal_regime\n" *
        "stage must be in [1, $(result.n_stages)], got $stage"
    )

    # Add intercept
    h_aug = vcat(1.0, history)
    blip = dot(h_aug, result.blip_coefficients[stage])
    return blip > 0 ? 1 : 0
end


"""
    summary(result::ALearningResult)

Generate summary string of A-learning results.
"""
function Base.summary(result::ALearningResult)
    lines = [
        "A-Learning Results",
        "=" ^ 50,
        "Number of stages: $(result.n_stages)",
        "Propensity model: $(result.propensity_model)",
        "Outcome model: $(result.outcome_model)",
        "Doubly robust: $(result.doubly_robust)",
        "SE method: $(result.se_method)",
        "",
        "Value Function:",
        "  Optimal value: $(round(result.value_estimate, digits=4)) (SE: $(round(result.value_se, digits=4)))",
        "  95% CI: [$(round(result.value_ci_lower, digits=4)), $(round(result.value_ci_upper, digits=4))]",
        "",
        "Blip Coefficients by Stage:",
    ]

    for k in 1:result.n_stages
        push!(lines, "  Stage $k:")
        for (j, (coef, se)) in enumerate(zip(result.blip_coefficients[k], result.blip_se[k]))
            push!(lines, "    psi[$(j-1)]: $(round(coef, digits=4)) (SE: $(round(se, digits=4)))")
        end
    end

    return join(lines, "\n")
end
