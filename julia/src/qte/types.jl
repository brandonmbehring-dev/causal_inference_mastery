"""
Type definitions for Quantile Treatment Effects (QTE) estimation.

Follows SciML Problem-Estimator-Solution architecture.
"""

"""
    QTEProblem{T<:Real}

Problem specification for QTE estimation.

# Fields
- `outcome::Vector{T}`: Outcome variable Y
- `treatment::Vector{T}`: Binary treatment indicator T ∈ {0, 1}
- `covariates::Union{Nothing, Matrix{T}}`: Optional covariates X
- `quantile::T`: Target quantile τ ∈ (0, 1)
"""
struct QTEProblem{T<:Real}
    outcome::Vector{T}
    treatment::Vector{T}
    covariates::Union{Nothing, Matrix{T}}
    quantile::T

    function QTEProblem(;
        outcome::Vector{T},
        treatment::Vector{T},
        covariates::Union{Nothing, Matrix{T}} = nothing,
        quantile::T = T(0.5)
    ) where T<:Real
        n = length(outcome)

        # Validate lengths
        length(treatment) == n || error("Treatment length mismatch")
        if covariates !== nothing
            size(covariates, 1) == n || error("Covariates row count mismatch")
        end

        # Validate treatment is binary
        all(t -> t == zero(T) || t == one(T), treatment) ||
            error("Treatment must be binary (0 or 1)")

        # Validate treatment variation
        length(unique(treatment)) >= 2 ||
            error("No treatment variation")

        # Validate quantile
        zero(T) < quantile < one(T) ||
            error("Quantile must be in (0, 1), got $quantile")

        # Validate no NaN/Inf
        !any(isnan, outcome) || error("NaN in outcome")
        !any(isinf, outcome) || error("Inf in outcome")

        new{T}(outcome, treatment, covariates, quantile)
    end
end


"""
    QTESolution{T<:Real}

Result from QTE estimation.
"""
struct QTESolution{T<:Real}
    tau_q::T
    se::T
    ci_lower::T
    ci_upper::T
    quantile::T
    method::Symbol
    n_treated::Int
    n_control::Int
    n_total::Int
    outcome_support::Tuple{T, T}
    inference::Symbol
end

# Keyword constructor
function QTESolution(;
    tau_q::T,
    se::T,
    ci_lower::T,
    ci_upper::T,
    quantile::T,
    method::Symbol,
    n_treated::Int,
    n_control::Int,
    n_total::Int,
    outcome_support::Tuple{T, T},
    inference::Symbol
) where T<:Real
    QTESolution{T}(tau_q, se, ci_lower, ci_upper, quantile, method,
                   n_treated, n_control, n_total, outcome_support, inference)
end


"""
    QTEBandProblem{T<:Real}

Problem specification for QTE band estimation across multiple quantiles.
"""
struct QTEBandProblem{T<:Real}
    outcome::Vector{T}
    treatment::Vector{T}
    covariates::Union{Nothing, Matrix{T}}
    quantiles::Vector{T}
end

function QTEBandProblem(;
    outcome::Vector{T},
    treatment::Vector{T},
    covariates::Union{Nothing, Matrix{T}} = nothing,
    quantiles::Vector{T} = T[0.1, 0.25, 0.5, 0.75, 0.9]
) where T<:Real
    QTEBandProblem{T}(outcome, treatment, covariates, quantiles)
end


"""
    QTEBandSolution{T<:Real}

Result from QTE band estimation.
"""
struct QTEBandSolution{T<:Real}
    quantiles::Vector{T}
    qte_estimates::Vector{T}
    se_estimates::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    joint_ci_lower::Union{Nothing, Vector{T}}
    joint_ci_upper::Union{Nothing, Vector{T}}
    method::Symbol
    n_bootstrap::Int
    n_treated::Int
    n_control::Int
    n_total::Int
    alpha::T
end

function QTEBandSolution(;
    quantiles::Vector{T},
    qte_estimates::Vector{T},
    se_estimates::Vector{T},
    ci_lower::Vector{T},
    ci_upper::Vector{T},
    joint_ci_lower::Union{Nothing, Vector{T}},
    joint_ci_upper::Union{Nothing, Vector{T}},
    method::Symbol,
    n_bootstrap::Int,
    n_treated::Int,
    n_control::Int,
    n_total::Int,
    alpha::T
) where T<:Real
    QTEBandSolution{T}(quantiles, qte_estimates, se_estimates, ci_lower, ci_upper,
                       joint_ci_lower, joint_ci_upper, method, n_bootstrap,
                       n_treated, n_control, n_total, alpha)
end
