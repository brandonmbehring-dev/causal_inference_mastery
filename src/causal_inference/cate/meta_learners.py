"""Meta-learners for CATE estimation.

Implements S-Learner and T-Learner approaches for estimating heterogeneous
treatment effects (Conditional Average Treatment Effects, CATE).

Meta-learners are generic frameworks that use any supervised learning algorithm
as a base learner to estimate treatment effect heterogeneity.

Algorithm Overview
------------------
**S-Learner** (Single model):
1. Fit μ(X, T) on combined data: augment X with T as additional feature
2. Predict: μ̂(xᵢ, 1) and μ̂(xᵢ, 0) for each unit
3. CATE: τ̂(xᵢ) = μ̂(xᵢ, 1) - μ̂(xᵢ, 0)

**T-Learner** (Two models):
1. Fit μ₀(X) on control group: X[T=0] → Y[T=0]
2. Fit μ₁(X) on treated group: X[T=1] → Y[T=1]
3. CATE: τ̂(xᵢ) = μ̂₁(xᵢ) - μ̂₀(xᵢ)

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
"""

import numpy as np
from typing import Literal
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .base import CATEResult, validate_cate_inputs


def _get_model(model_type: str, **kwargs):
    """Get sklearn model instance by name.

    Parameters
    ----------
    model_type : str
        One of "linear", "ridge", "random_forest".
    **kwargs
        Additional arguments passed to model constructor.

    Returns
    -------
    sklearn estimator
        Instantiated model.

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    if model_type == "linear":
        return LinearRegression(**kwargs)
    elif model_type == "ridge":
        return Ridge(**kwargs)
    elif model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"CRITICAL ERROR: Unknown model type.\n"
            f"Function: _get_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'linear', 'ridge', 'random_forest'"
        )


def s_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using S-Learner (Single model approach).

    The S-Learner fits a single model μ(X, T) that includes treatment as a feature,
    then estimates CATE by comparing predictions under T=1 vs T=0.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for outcome modeling:
        - "linear": OLS regression
        - "ridge": L2-regularized regression
        - "random_forest": Random forest regressor
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "s_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)  # True ATE = 2
    >>> result = s_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.00

    Notes
    -----
    **Known Limitation**: S-learner is biased toward 0 when the treatment effect
    is small relative to the main effects of X. This is because the model may
    "regularize away" the treatment effect if it's not a strong predictor.

    The S-learner works best when:
    - Treatment effects are large relative to outcome variance
    - The base learner can capture treatment-covariate interactions

    See Also
    --------
    t_learner : Two-model approach that avoids regularization bias.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Build augmented feature matrix [X | T]
    X_augmented = np.column_stack([covariates, treatment])

    # Fit single model on combined data
    learner = _get_model(model)
    learner.fit(X_augmented, outcomes)

    # Create counterfactual feature matrices
    X_treated = np.column_stack([covariates, np.ones(n)])
    X_control = np.column_stack([covariates, np.zeros(n)])

    # Predict potential outcomes
    mu_1 = learner.predict(X_treated)
    mu_0 = learner.predict(X_control)

    # CATE estimates
    cate = mu_1 - mu_0

    # Compute ATE
    ate = np.mean(cate)

    # SE estimation: Use residual-based approach
    # For ATE, we use the standard error of the treatment coefficient
    # Residuals from the model
    y_pred = learner.predict(X_augmented)
    residuals = outcomes - y_pred

    # SE of ATE using influence function approach
    # For S-learner: SE ≈ sqrt(Var(residuals) * (1/n1 + 1/n0))
    n1 = np.sum(treatment == 1)
    n0 = np.sum(treatment == 0)
    residual_var = np.var(residuals, ddof=1)
    ate_se = np.sqrt(residual_var * (1/n1 + 1/n0))

    # Confidence interval using normal approximation
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="s_learner",
    )


def t_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using T-Learner (Two-model approach).

    The T-Learner fits separate models for treated and control groups,
    then estimates CATE as the difference in predictions.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for outcome modeling:
        - "linear": OLS regression
        - "ridge": L2-regularized regression
        - "random_forest": Random forest regressor
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "t_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)  # True ATE = 2
    >>> result = t_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.01

    Notes
    -----
    **Known Limitation**: T-learner can overfit when treatment and control groups
    have different covariate distributions (lack of overlap). Each model only
    sees part of the covariate space, which can lead to poor extrapolation.

    The T-learner works best when:
    - There is good covariate overlap between treatment groups
    - Sample sizes are large enough to fit two separate models
    - Treatment effect heterogeneity is substantial

    **Advantages over S-learner**:
    - No regularization bias toward zero
    - Can capture different functional forms for μ₁(x) and μ₀(x)

    See Also
    --------
    s_learner : Single-model approach.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Split data by treatment status
    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_treated = covariates[treated_mask]
    Y_treated = outcomes[treated_mask]

    X_control = covariates[control_mask]
    Y_control = outcomes[control_mask]

    # Fit separate models
    model_1 = _get_model(model)
    model_0 = _get_model(model)

    model_1.fit(X_treated, Y_treated)
    model_0.fit(X_control, Y_control)

    # Predict potential outcomes for all units
    mu_1 = model_1.predict(covariates)
    mu_0 = model_0.predict(covariates)

    # CATE estimates
    cate = mu_1 - mu_0

    # Compute ATE
    ate = np.mean(cate)

    # SE estimation using residuals from each model
    # Residuals for treated and control models
    residuals_1 = Y_treated - model_1.predict(X_treated)
    residuals_0 = Y_control - model_0.predict(X_control)

    n1 = len(Y_treated)
    n0 = len(Y_control)

    # Pooled variance approach for T-learner
    var_1 = np.var(residuals_1, ddof=1)
    var_0 = np.var(residuals_0, ddof=1)
    ate_se = np.sqrt(var_1/n1 + var_0/n0)

    # Confidence interval using normal approximation
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="t_learner",
    )


def x_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "random_forest",
    propensity_model: Literal["logistic", "random_forest"] = "logistic",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using X-Learner (Cross-learner approach).

    The X-Learner extends T-learner by using imputed treatment effects and
    propensity-weighted combination. It performs better when treatment groups
    are imbalanced, leveraging the larger group to improve smaller group estimates.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="random_forest"
        Base learner for outcome and CATE modeling.
    propensity_model : {"logistic", "random_forest"}, default="logistic"
        Model for propensity score estimation.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "x_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.3, n)  # Imbalanced: 30% treated
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = x_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.01

    Notes
    -----
    **Algorithm** (Künzel et al. 2019):

    1. **Stage 1**: Fit T-learner models μ̂₀(X) and μ̂₁(X)
    2. **Stage 2**: Compute imputed treatment effects:
       - For treated: D₁ = Y - μ̂₀(X) (observed - counterfactual)
       - For control: D₀ = μ̂₁(X) - Y (counterfactual - observed)
    3. **Stage 3**: Fit CATE models τ̂₁(X), τ̂₀(X) on imputed effects
    4. **Stage 4**: Combine using propensity weighting:
       τ̂(x) = g(x)·τ̂₀(x) + (1-g(x))·τ̂₁(x)

    **Key Advantage**: X-learner leverages the larger group to improve
    estimates for the smaller group, making it ideal for imbalanced data.

    References
    ----------
    Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment
    effects using machine learning." PNAS 116(10): 4156-4165.

    See Also
    --------
    t_learner : Simpler two-model approach.
    r_learner : Doubly robust Robinson transformation approach.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Split data by treatment status
    treated_mask = treatment == 1
    control_mask = treatment == 0

    X_treated = covariates[treated_mask]
    Y_treated = outcomes[treated_mask]

    X_control = covariates[control_mask]
    Y_control = outcomes[control_mask]

    # =========================================================================
    # Stage 1: Fit T-learner models
    # =========================================================================
    model_1 = _get_model(model)  # μ₁(X): E[Y|X, T=1]
    model_0 = _get_model(model)  # μ₀(X): E[Y|X, T=0]

    model_1.fit(X_treated, Y_treated)
    model_0.fit(X_control, Y_control)

    # =========================================================================
    # Stage 2: Compute imputed treatment effects
    # =========================================================================
    # For treated units: D₁ = Y(1) - μ̂₀(X) (actual outcome - counterfactual)
    D_1 = Y_treated - model_0.predict(X_treated)

    # For control units: D₀ = μ̂₁(X) - Y(0) (counterfactual - actual outcome)
    D_0 = model_1.predict(X_control) - Y_control

    # =========================================================================
    # Stage 3: Fit CATE models on imputed effects
    # =========================================================================
    tau_model_1 = _get_model(model)  # τ̂₁(X) trained on treated
    tau_model_0 = _get_model(model)  # τ̂₀(X) trained on control

    tau_model_1.fit(X_treated, D_1)
    tau_model_0.fit(X_control, D_0)

    # Predict CATE from both models for all units
    tau_1 = tau_model_1.predict(covariates)  # CATE estimates from treated model
    tau_0 = tau_model_0.predict(covariates)  # CATE estimates from control model

    # =========================================================================
    # Stage 4: Propensity-weighted combination
    # =========================================================================
    # Estimate propensity scores g(x) = P(T=1|X)
    if propensity_model == "logistic":
        prop_model = LogisticRegression(max_iter=1000, random_state=42)
    else:  # random_forest
        prop_model = RandomForestClassifier(n_estimators=100, random_state=42)

    prop_model.fit(covariates, treatment)
    propensity = prop_model.predict_proba(covariates)[:, 1]

    # Clip propensity to avoid extreme weights
    propensity = np.clip(propensity, 0.01, 0.99)

    # Combine: τ̂(x) = g(x)·τ̂₀(x) + (1-g(x))·τ̂₁(x)
    # Note: g(x) weights the control model, (1-g(x)) weights the treated model
    # This is because the control model is trained on imputed effects from controls
    cate = propensity * tau_0 + (1 - propensity) * tau_1

    # =========================================================================
    # Compute ATE and SE
    # =========================================================================
    ate = np.mean(cate)

    # SE estimation: Use variance of CATE estimates
    # For X-learner, we use a heuristic SE based on imputed effect variance
    n1 = len(Y_treated)
    n0 = len(Y_control)

    # Variance from each stage
    var_D1 = np.var(D_1, ddof=1)
    var_D0 = np.var(D_0, ddof=1)

    # Approximate SE using weighted variance
    ate_se = np.sqrt(var_D1/n1 + var_D0/n0)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="x_learner",
    )


def r_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using R-Learner (Robinson transformation).

    The R-Learner uses orthogonalization to obtain doubly robust CATE estimates.
    It residualizes both outcomes and treatment, then regresses outcome residuals
    on the product of CATE and treatment residuals.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for CATE modeling. Linear is common for R-learner.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "r_learner"

    Raises
    ------
    ValueError
        If inputs are invalid or model type is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> # Confounded treatment: propensity depends on X
    >>> propensity = 1 / (1 + np.exp(-X[:, 0]))
    >>> T = np.random.binomial(1, propensity, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = r_learner(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.05

    Notes
    -----
    **Algorithm** (Robinson Transformation, Nie & Wager 2021):

    1. **Stage 1**: Estimate nuisance functions
       - ê(X) = P(T=1|X) via logistic regression (propensity)
       - m̂(X) = E[Y|X] via outcome regression (marginal)

    2. **Stage 2**: Compute residuals
       - Ỹ = Y - m̂(X) (outcome residual)
       - T̃ = T - ê(X) (treatment residual)

    3. **Stage 3**: Estimate CATE
       - Solve: τ̂ = argmin_τ Σᵢ (Ỹᵢ - τ(Xᵢ)·T̃ᵢ)²
       - For linear τ(X) = X'β: weighted least squares

    **Key Property**: R-learner is "orthogonal" (Neyman-orthogonal) - first-order
    errors in propensity or outcome models don't bias CATE estimates.

    **Doubly Robust**: Consistent if either propensity OR outcome model is correct.

    References
    ----------
    - Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment
      effects." Biometrika 108(2): 299-319.
    - Robinson (1988). "Root-N-consistent semiparametric regression."
      Econometrica 56(4): 931-954.

    See Also
    --------
    x_learner : Cross-learner for imbalanced groups.
    t_learner : Simple two-model approach.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # =========================================================================
    # Stage 1: Estimate nuisance functions
    # =========================================================================

    # Propensity model: ê(X) = P(T=1|X)
    prop_model = LogisticRegression(max_iter=1000, random_state=42)
    prop_model.fit(covariates, treatment)
    e_hat = prop_model.predict_proba(covariates)[:, 1]

    # Clip propensity to avoid division issues
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # Outcome model: m̂(X) = E[Y|X] (marginal over T)
    outcome_model = _get_model("ridge")  # Use ridge for regularization
    outcome_model.fit(covariates, outcomes)
    m_hat = outcome_model.predict(covariates)

    # =========================================================================
    # Stage 2: Compute residuals
    # =========================================================================

    # Outcome residual: Ỹ = Y - m̂(X)
    Y_tilde = outcomes - m_hat

    # Treatment residual: T̃ = T - ê(X)
    T_tilde = treatment - e_hat

    # =========================================================================
    # Stage 3: Estimate CATE via weighted regression
    # =========================================================================

    # The R-learner objective is:
    #   min_τ Σᵢ (Ỹᵢ - τ(Xᵢ)·T̃ᵢ)²
    #
    # For linear τ(X) = X'β + β₀:
    #   Ỹᵢ = (X'β + β₀)·T̃ᵢ + error
    #   Ỹᵢ = (Xᵢ·T̃ᵢ)'β + β₀·T̃ᵢ + error
    #
    # This is weighted regression with features [X·T̃, T̃] and target Ỹ

    # Create transformed features: each covariate multiplied by T̃
    X_transformed = covariates * T_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, T_tilde])

    # Fit model: Ỹ ~ X·T̃ + T̃
    cate_model = _get_model(model)
    cate_model.fit(X_with_intercept, Y_tilde)

    # Predict CATE for each unit
    # τ̂(X) = X'β + β₀ (the intercept term gives base effect)
    # To get CATE, we predict with T̃=1 (counterfactual "full treatment")
    X_pred = np.column_stack([covariates, np.ones(n)])
    cate = cate_model.predict(X_pred)

    # =========================================================================
    # Compute ATE and SE
    # =========================================================================
    ate = np.mean(cate)

    # SE estimation using influence function approach
    # For R-learner, the influence function involves residuals
    # Approximate SE using residual variance
    Y_pred = cate * T_tilde
    residuals = Y_tilde - Y_pred

    # SE of ATE (approximation)
    # Use weighted variance accounting for T̃
    weights = T_tilde ** 2
    weights = weights / np.sum(weights)  # Normalize

    # Standard approach: variance of pseudo-outcomes divided by effective n
    pseudo_outcomes = Y_tilde / (T_tilde + 1e-8)  # Avoid division by zero
    # Only use observations with reasonable T̃
    valid_mask = np.abs(T_tilde) > 0.1
    if np.sum(valid_mask) > 10:
        ate_se = np.std(pseudo_outcomes[valid_mask], ddof=1) / np.sqrt(np.sum(valid_mask))
    else:
        # Fallback to simple SE
        ate_se = np.std(cate, ddof=1) / np.sqrt(n)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="r_learner",
    )
