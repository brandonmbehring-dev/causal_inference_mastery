#=
Orthogonal Machine Learning (OML) - Interactive Regression Model (IRM)

IRM extends DML to the fully flexible model Y = g(T, X) + U,
providing doubly robust estimation of treatment effects.

Key difference from DML:
- DML (PLR): Y = θT + g(X) + U (linear in T, fits joint E[Y|X])
- IRM: Y = g(T,X) + U (fully flexible, fits separate g0, g1)

Double robustness: IRM is consistent if propensity OR outcome model correct.

Session 153: OML foundation.

References:
- Chernozhukov et al. (2018). "Double/debiased machine learning"
- Robins & Rotnitzky (1995). "Semiparametric efficiency"
=#

using LinearAlgebra
using Statistics
using Random
using Distributions


# =============================================================================
# Cross-Fitting for IRM
# =============================================================================

"""
    _cross_fit_irm_nuisance(outcomes, treatment, covariates, n_folds, model)

Cross-fit nuisance models for Interactive Regression Model.

Unlike DML which fits joint E[Y|X], IRM fits SEPARATE outcome models
on control (g0) and treated (g1) subsets.

Returns (g0_hat, g1_hat, m_hat) - all cross-fitted predictions.
"""
function _cross_fit_irm_nuisance(
    outcomes::Vector{T},
    treatment::Vector{Bool},
    covariates::Matrix{T},
    n_folds::Int,
    model::Symbol
) where {T<:Real}
    n = length(outcomes)

    # Storage for cross-fitted predictions
    g0_hat = zeros(T, n)
    g1_hat = zeros(T, n)
    m_hat = zeros(T, n)

    # Create fold indices
    perm = randperm(n)
    fold_size = div(n, n_folds)

    fold_indices = zeros(Int, n)
    for k in 1:n_folds
        start_idx = (k - 1) * fold_size + 1
        end_idx = k == n_folds ? n : k * fold_size
        fold_indices[perm[start_idx:end_idx]] .= k
    end

    X_int = add_intercept(covariates)

    for k in 1:n_folds
        # Training/test masks
        train_idx = fold_indices .!= k
        test_idx = fold_indices .== k

        # Training data
        Y_train = outcomes[train_idx]
        T_train = treatment[train_idx]
        X_train = X_int[train_idx, :]
        X_cov_train = covariates[train_idx, :]

        # Test data
        X_test = X_int[test_idx, :]
        X_cov_test = covariates[test_idx, :]

        # Masks for control/treated in training
        control_mask = .!T_train
        treated_mask = T_train

        n_control = sum(control_mask)
        n_treated = sum(treated_mask)

        # --- Fit g0: E[Y|T=0, X] on CONTROL units only ---
        if n_control >= 2
            β_g0, _, _ = fit_model(X_train[control_mask, :], Y_train[control_mask], model)
            g0_hat[test_idx] = predict_ols(X_test, β_g0)
        else
            # Fallback: use control mean
            g0_hat[test_idx] .= n_control > 0 ? mean(Y_train[control_mask]) : T(0)
        end

        # --- Fit g1: E[Y|T=1, X] on TREATED units only ---
        if n_treated >= 2
            β_g1, _, _ = fit_model(X_train[treated_mask, :], Y_train[treated_mask], model)
            g1_hat[test_idx] = predict_ols(X_test, β_g1)
        else
            # Fallback: use treated mean
            g1_hat[test_idx] .= n_treated > 0 ? mean(Y_train[treated_mask]) : T(0)
        end

        # --- Fit propensity m(X) on ALL training data ---
        m_hat[test_idx] = _predict_propensity(X_cov_train, T_train, X_cov_test)
    end

    return g0_hat, g1_hat, m_hat
end


# =============================================================================
# Score Functions
# =============================================================================

"""
    _irm_score(Y, T, g0, g1, m, theta)

Compute IRM doubly robust influence function scores.

The doubly robust score for ATE under IRM:
ψ = (g1(X) - g0(X)) + T(Y-g1(X))/m(X) - (1-T)(Y-g0(X))/(1-m(X)) - θ

This score is Neyman-orthogonal and doubly robust.
"""
function _irm_score(
    Y::Vector{T},
    treatment::Vector{Bool},
    g0::Vector{T},
    g1::Vector{T},
    m::Vector{T},
    theta::T
) where {T<:Real}
    T_float = T.(treatment)

    # Outcome regression component
    or_component = g1 .- g0

    # IPW correction for treated
    ipw_treated = T_float .* (Y .- g1) ./ m

    # IPW correction for control
    ipw_control = (one(T) .- T_float) .* (Y .- g0) ./ (one(T) .- m)

    # Full doubly robust score
    psi = or_component .+ ipw_treated .- ipw_control .- theta

    return psi
end


"""
    _atte_score(Y, treatment, g0, m, theta)

Compute ATTE (Average Treatment Effect on Treated) influence function scores.

The score for ATTE:
ψ = T(Y - g0(X) - θ) / P(T=1) - m(X)(1-T)(Y - g0(X)) / (P(T=1)(1-m(X)))
"""
function _atte_score(
    Y::Vector{T},
    treatment::Vector{Bool},
    g0::Vector{T},
    m::Vector{T},
    theta::T
) where {T<:Real}
    T_float = T.(treatment)
    p_treated = mean(T_float)

    if p_treated < 1e-10
        return zeros(T, length(Y))
    end

    # Component 1: Direct effect on treated
    direct = T_float .* (Y .- g0 .- theta) ./ p_treated

    # Component 2: IPW adjustment for control units
    ipw_adjust = m .* (one(T) .- T_float) .* (Y .- g0) ./ (p_treated .* (one(T) .- m))

    psi = direct .- ipw_adjust

    return psi
end


"""
    _irm_influence_se(psi)

Compute standard error from influence function scores.

SE = sqrt(Var(ψ) / n)
"""
function _irm_influence_se(psi::Vector{T}) where {T<:Real}
    n = length(psi)
    if n < 2
        return T(NaN)
    end
    return T(sqrt(var(psi, corrected=true) / n))
end


# =============================================================================
# Main Solve Function
# =============================================================================

"""
    solve(problem::CATEProblem, estimator::IRMEstimator) -> CATESolution

Estimate CATE using Interactive Regression Model with cross-fitting.

# Algorithm
1. Split data into K folds
2. For each fold k:
   - Fit g0(X) = E[Y|T=0, X] on control units of OTHER folds
   - Fit g1(X) = E[Y|T=1, X] on treated units of OTHER folds
   - Fit m(X) = P(T=1|X) on ALL units of OTHER folds
   - Predict on fold k (out-of-sample)
3. Compute doubly robust score
4. ATE = plug-in + IPW corrections
5. SE via influence function

# Comparison with DoubleMachineLearning (PLR)
- PLR: Y = θT + g(X) + U (linear in T)
- IRM: Y = g(T, X) + U (fully flexible)
- PLR requires outcome model correctly specified
- IRM is doubly robust: consistent if propensity OR outcome correct

# Example
```julia
using CausalEstimators
using Random

Random.seed!(42)
n = 500
X = randn(n, 3)
propensity = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
T = rand(n) .< propensity
# IRM DGP: g0 = 1 + X, g1 = 1 + X + 2 (ATE = 2)
Y = (1 .- T) .* (1 .+ X[:, 1]) .+ T .* (3 .+ X[:, 1]) .+ randn(n)

problem = CATEProblem(Y, T, X, (alpha=0.05,))
solution = solve(problem, IRMEstimator())

println("IRM ATE: \$(solution.ate) ± \$(solution.se)")
```
"""
function solve(
    problem::CATEProblem{T,P},
    estimator::IRMEstimator
)::CATESolution{T,P} where {T<:Real, P<:NamedTuple}
    # Extract from problem
    outcomes = problem.outcomes
    treatment = problem.treatment
    covariates = problem.covariates
    parameters = problem.parameters
    alpha = get(parameters, :alpha, 0.05)

    n = length(outcomes)
    n_folds = estimator.n_folds
    target = estimator.target

    # =========================================================================
    # Step 1: Cross-fit nuisance models for IRM
    # =========================================================================
    g0_hat, g1_hat, m_hat = _cross_fit_irm_nuisance(
        outcomes, treatment, covariates, n_folds, estimator.model
    )

    # Clip propensity more aggressively for IRM (has 1/m terms)
    m_hat = clamp.(m_hat, T(0.025), T(0.975))

    # =========================================================================
    # Step 2: Compute CATE estimates
    # =========================================================================
    # For IRM, CATE = g1(X) - g0(X) (plug-in estimator)
    cate = g1_hat .- g0_hat

    # =========================================================================
    # Step 3: Estimate target parameter (ATE or ATTE)
    # =========================================================================
    T_float = T.(treatment)

    if target == :ate
        # ATE: E[Y(1) - Y(0)]
        # Doubly robust estimator
        or_component = mean(g1_hat .- g0_hat)
        ipw_treated = mean(T_float .* (outcomes .- g1_hat) ./ m_hat)
        ipw_control = mean((one(T) .- T_float) .* (outcomes .- g0_hat) ./ (one(T) .- m_hat))

        theta = T(or_component + ipw_treated - ipw_control)

        # Compute influence function scores for SE
        psi = _irm_score(outcomes, treatment, g0_hat, g1_hat, m_hat, theta)

    else  # target == :atte
        # ATTE: E[Y(1) - Y(0) | T=1]
        p_treated = mean(T_float)

        if p_treated < 1e-10
            throw(ErrorException(
                "CRITICAL ERROR: No treated units.\n" *
                "Function: solve (IRMEstimator)\n" *
                "Cannot estimate ATTE without treated units."
            ))
        end

        # ATTE = E[T(Y - g0)] / P(T=1) with IPW adjustment
        direct = mean(T_float .* (outcomes .- g0_hat)) / p_treated
        ipw_adjust = mean(m_hat .* (one(T) .- T_float) .* (outcomes .- g0_hat) ./
                         (one(T) .- m_hat)) / p_treated

        theta = T(direct - ipw_adjust)

        # Compute ATTE influence function scores
        psi = _atte_score(outcomes, treatment, g0_hat, m_hat, theta)
    end

    # =========================================================================
    # Step 4: Standard error via influence function
    # =========================================================================
    se = _irm_influence_se(psi)

    # Handle degenerate SE
    if !isfinite(se) || se <= 0
        se = T(std(cate, corrected=true) / sqrt(n))
    end

    # Confidence interval
    z_crit = T(quantile(Normal(), 1 - alpha / 2))
    ci_lower = theta - z_crit * se
    ci_upper = theta + z_crit * se

    return CATESolution{T,P}(
        cate,
        theta,
        se,
        ci_lower,
        ci_upper,
        :irm,
        :Success,
        problem
    )
end
