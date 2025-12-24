#=
Targeted Maximum Likelihood Estimation (TMLE) for Causal Inference

Implements TMLE for ATE estimation, an improvement over standard doubly robust
(DR/AIPW) estimation that achieves the semiparametric efficiency bound through
an iterative targeting procedure.

Key Properties of TMLE:
1. Double robustness: Consistent if either propensity OR outcome model correct
2. Efficiency: Achieves the semiparametric efficiency bound when both correct
3. Better finite-sample: Targeting step improves bias compared to standard DR
4. Valid inference: Influence function-based standard errors are valid

Algorithm:
1. Estimate initial nuisance functions:
   - Propensity: g(X) = P(T=1|X)
   - Outcome: Q(T,X) = E[Y|T,X]

2. Targeting step (iterate until convergence):
   - Compute clever covariate: H = T/g - (1-T)/(1-g)
   - Fit fluctuation: Y ~ ε*H + offset(Q)
   - Update: Q* = Q + ε*H

3. Estimate ATE: mean(Q*(1,X)) - mean(Q*(0,X))

4. Inference via efficient influence function

References:
- van der Laan, M. J., & Rose, S. (2011). Targeted Learning. Springer.
- Schuler, M. S., & Rose, S. (2017). TMLE for causal inference. AJE 185(1).
=#

"""
    TMLE <: AbstractObservationalEstimator

Targeted Maximum Likelihood Estimator for average treatment effect.

TMLE improves on standard doubly robust estimation by iteratively updating
the outcome model to satisfy the efficient score equation, achieving the
semiparametric efficiency bound.

# Parameters
- `max_iter::Int`: Maximum targeting iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Example
```julia
problem = ObservationalProblem(Y, T, X)
solution = solve(problem, TMLE())
```
"""
struct TMLE <: AbstractObservationalEstimator
    max_iter::Int
    tol::Float64

    function TMLE(; max_iter::Int = 100, tol::Float64 = 1e-6)
        if max_iter < 1
            throw(ArgumentError("max_iter must be at least 1, got $max_iter"))
        end
        if tol <= 0
            throw(ArgumentError("tol must be positive, got $tol"))
        end
        new(max_iter, tol)
    end
end


# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_clever_covariate(treatment, propensity) -> Vector{T}

Compute the clever covariate H for TMLE targeting.

H = T/g(X) - (1-T)/(1-g(X))

Where T is treatment and g(X) is propensity.
"""
function compute_clever_covariate(
    treatment::AbstractVector{Bool},
    propensity::AbstractVector{T}
) where {T<:Real}
    # Clip propensity to prevent division by zero
    epsilon = T(1e-6)
    g = clamp.(propensity, epsilon, 1 - epsilon)

    # H = T/g - (1-T)/(1-g)
    T_numeric = convert(Vector{T}, treatment)
    H = T_numeric ./ g .- (1 .- T_numeric) ./ (1 .- g)

    return H
end


"""
    fit_fluctuation(outcomes, clever_covariate, initial_predictions) -> (epsilon, updated_predictions)

Fit the fluctuation parameter epsilon.

Fits the targeting submodel: Y = Q(X) + ε*H + noise
Using regression: ε = sum(H * residuals) / sum(H²)
"""
function fit_fluctuation(
    outcomes::AbstractVector{T},
    clever_covariate::AbstractVector{T},
    initial_predictions::AbstractVector{T}
) where {T<:Real}
    # Residuals from initial predictions
    residuals = outcomes .- initial_predictions

    # Fit epsilon via simple regression: residuals ~ H
    numerator = sum(clever_covariate .* residuals)
    denominator = sum(clever_covariate .^ 2)

    epsilon = if abs(denominator) < 1e-10
        T(0)  # Avoid division by zero
    else
        numerator / denominator
    end

    # Update predictions
    updated_predictions = initial_predictions .+ epsilon .* clever_covariate

    return epsilon, updated_predictions
end


"""
    check_tmle_convergence(outcomes, predictions, clever_covariate, tol) -> (converged, criterion)

Check TMLE targeting convergence.

Convergence is achieved when: |mean(H * (Y - Q*))| < tol
"""
function check_tmle_convergence(
    outcomes::AbstractVector{T},
    predictions::AbstractVector{T},
    clever_covariate::AbstractVector{T},
    tol::Float64
) where {T<:Real}
    residuals = outcomes .- predictions
    criterion = mean(clever_covariate .* residuals)

    converged = abs(criterion) < tol

    return converged, T(criterion)
end


"""
    compute_efficient_influence_function(outcomes, treatment, propensity, Q1_star, Q0_star, ate) -> Vector{T}

Compute the efficient influence function (EIF) for TMLE.

EIF_i = H1*(Y - Q1*)*T - H0*(Y - Q0*)*(1-T) + Q1* - Q0* - ATE

Where H1 = 1/g and H0 = 1/(1-g).
"""
function compute_efficient_influence_function(
    outcomes::AbstractVector{T},
    treatment::AbstractVector{Bool},
    propensity::AbstractVector{T},
    Q1_star::AbstractVector{T},
    Q0_star::AbstractVector{T},
    ate::T
) where {T<:Real}
    epsilon = T(1e-6)
    g = clamp.(propensity, epsilon, 1 - epsilon)

    # Clever covariate components
    H1 = 1 ./ g
    H0 = 1 ./ (1 .- g)

    T_numeric = convert(Vector{T}, treatment)

    # EIF components
    treated_component = T_numeric .* H1 .* (outcomes .- Q1_star)
    control_component = (1 .- T_numeric) .* H0 .* (outcomes .- Q0_star)
    outcome_component = Q1_star .- Q0_star

    eif = treated_component .- control_component .+ outcome_component .- ate

    return eif
end


# =============================================================================
# Main solve() Implementation
# =============================================================================

"""
    solve(problem::ObservationalProblem, estimator::TMLE) -> TMLESolution

Estimate ATE using Targeted Maximum Likelihood Estimation.

# Algorithm

1. Estimate propensity scores e(X) via logistic regression
2. Fit outcome models μ₀(X) and μ₁(X) via linear regression
3. (Optional) Trim extreme propensities
4. Targeting step (iterate until convergence):
   - Compute clever covariate: H = T/e - (1-T)/(1-e)
   - Fit fluctuation: Y ~ ε*H + offset(Q)
   - Update: Q* = Q + ε*H
5. Compute ATE = mean(Q1*) - mean(Q0*)
6. Compute variance via efficient influence function

# Arguments
- `problem::ObservationalProblem`: Problem specification with outcomes, treatment, covariates
- `estimator::TMLE`: TMLE estimator with max_iter and tol parameters

# Returns
- `TMLESolution`: Solution with ATE estimate, SE, CI, and diagnostics

# Example
```julia
# Generate observational data with confounding
n = 500
X = randn(n, 2)
logit = 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2]
e_true = 1 ./ (1 .+ exp.(-logit))
T = rand(n) .< e_true
Y = 2.0 .* T .+ 0.5 .* X[:, 1] .+ randn(n)

problem = ObservationalProblem(Y, T, X)
solution = solve(problem, TMLE())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("Converged: \$(solution.converged) in \$(solution.n_iterations) iterations")
```

# Notes
- TMLE vs Doubly Robust (DR/AIPW):
  - Both are doubly robust (consistent if either model correct)
  - TMLE achieves efficiency bound; DR may not
  - TMLE typically has lower finite-sample bias
  - TMLE requires iterative targeting; DR is one-shot
"""
function solve(
    problem::ObservationalProblem{T},
    estimator::TMLE
)::TMLESolution{T} where {T<:Real}

    # Extract data
    outcomes = problem.outcomes
    treatment = problem.treatment
    covariates = problem.covariates
    alpha = problem.parameters.alpha
    trim_threshold = problem.parameters.trim_threshold

    n = length(outcomes)
    max_iter = estimator.max_iter
    tol = estimator.tol

    # =========================================================================
    # Step 1: Estimate Propensity Scores
    # =========================================================================

    if problem.propensity !== nothing
        propensity = copy(problem.propensity)
    else
        prop_result = estimate_propensity_scores(treatment, covariates)
        propensity = prop_result.propensity
    end

    # =========================================================================
    # Step 2: Trim Extreme Propensities (Optional)
    # =========================================================================

    n_trimmed = 0
    if trim_threshold > 0
        trim_result = trim_propensities(
            propensity, treatment, outcomes, covariates;
            trim_at = (trim_threshold, 1 - trim_threshold)
        )
        propensity = trim_result.propensity
        treatment = trim_result.treatment
        outcomes = trim_result.outcomes
        covariates = trim_result.covariates
        n_trimmed = trim_result.n_trimmed
        n = length(outcomes)
    end

    # Clip propensity for numerical stability
    epsilon_clip = T(1e-6)
    propensity_clipped = clamp.(propensity, epsilon_clip, 1 - epsilon_clip)

    # =========================================================================
    # Step 3: Fit Outcome Models
    # =========================================================================

    outcome_result = fit_outcome_models(outcomes, treatment, covariates)
    Q0_initial = outcome_result.mu0_predictions
    Q1_initial = outcome_result.mu1_predictions
    mu0_r2 = outcome_result.mu0_r2
    mu1_r2 = outcome_result.mu1_r2

    # =========================================================================
    # Step 4: Targeting Step (Iterative Update)
    # =========================================================================

    # Create observed predictions based on actual treatment
    T_numeric = convert(Vector{T}, treatment)
    Q_observed = @. T_numeric * Q1_initial + (1 - T_numeric) * Q0_initial

    # Compute clever covariate
    H = compute_clever_covariate(treatment, propensity_clipped)

    # Iterative targeting
    Q_star = copy(Q_observed)
    epsilon_total = T(0)
    converged = false
    convergence_criterion = T(Inf)
    n_iterations = 0

    for iteration in 1:max_iter
        # Check convergence
        converged, convergence_criterion = check_tmle_convergence(
            outcomes, Q_star, H, tol
        )

        if converged
            n_iterations = iteration
            break
        end

        # Fit fluctuation
        epsilon, Q_star = fit_fluctuation(outcomes, H, Q_star)
        epsilon_total += epsilon
        n_iterations = iteration
    end

    # =========================================================================
    # Step 5: Compute Targeted Predictions for Both Treatment Levels
    # =========================================================================

    # Apply total fluctuation to initial predictions
    H1 = 1 ./ propensity_clipped
    H0 = -1 ./ (1 .- propensity_clipped)

    Q1_star = Q1_initial .+ epsilon_total .* H1
    Q0_star = Q0_initial .+ epsilon_total .* H0

    # =========================================================================
    # Step 6: Compute ATE and Inference
    # =========================================================================

    ate = mean(Q1_star) - mean(Q0_star)

    # Efficient influence function for variance
    eif = compute_efficient_influence_function(
        outcomes, treatment, propensity_clipped, Q1_star, Q0_star, ate
    )

    variance = mean(eif .^ 2) / n
    se = sqrt(variance)

    # Confidence interval
    if n < 50
        df = n - 2
        critical = quantile(TDist(df), 1 - alpha / 2)
        p_value = 2 * ccdf(TDist(df), abs(ate / se))
    else
        critical = quantile(Normal(), 1 - alpha / 2)
        p_value = 2 * ccdf(Normal(), abs(ate / se))
    end

    ci_lower = ate - critical * se
    ci_upper = ate + critical * se

    # =========================================================================
    # Step 7: Compute Diagnostics
    # =========================================================================

    propensity_auc = compute_propensity_auc(propensity_clipped, treatment)

    n_treated = sum(treatment)
    n_control = n - n_treated

    # =========================================================================
    # Step 8: Construct Solution
    # =========================================================================

    return TMLESolution{T}(
        T(ate),
        T(se),
        T(ci_lower),
        T(ci_upper),
        T(p_value),
        n_treated,
        n_control,
        n_trimmed,
        T(epsilon_total),
        n_iterations,
        converged,
        T(convergence_criterion),
        propensity_clipped,
        Q0_initial,
        Q1_initial,
        Q0_star,
        Q1_star,
        eif,
        propensity_auc,
        mu0_r2,
        mu1_r2,
        :Success,
        problem
    )
end
