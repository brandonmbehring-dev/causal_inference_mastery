"""Double Machine Learning for CATE estimation.

Implements Double/Debiased Machine Learning (DML) with K-fold cross-fitting
to eliminate regularization bias in treatment effect estimation.

Key Features
------------
- K-fold cross-fitting eliminates in-sample bias
- Influence function based standard errors
- Valid asymptotic inference with ML nuisance models

Algorithm Overview
------------------
The partially linear model (Chernozhukov et al. 2018):

Y = θ(X)·T + g(X) + ε    (outcome equation)
T = m(X) + η             (treatment equation)

Cross-fitting procedure:
1. Split data into K folds
2. For each fold k:
   - Train nuisance models (propensity ê, outcome m̂) on OTHER folds
   - Predict ê(Xₖ), m̂(Xₖ) for fold k (out-of-sample)
3. Compute residuals: Ỹ = Y - m̂(X), T̃ = T - ê(X)
4. Estimate θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)

References
----------
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
- Robinson (1988). "Root-N-consistent semiparametric regression."
  Econometrica 56(4): 931-954.
"""

import numpy as np
from typing import Literal
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .base import CATEResult, validate_cate_inputs


def _get_outcome_model(model_type: str, **kwargs):
    """Get sklearn regression model by name.

    Parameters
    ----------
    model_type : str
        One of "linear", "ridge", "random_forest".
    **kwargs
        Additional arguments passed to model constructor.

    Returns
    -------
    sklearn estimator
        Instantiated regression model.
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
            f"Function: _get_outcome_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'linear', 'ridge', 'random_forest'"
        )


def _get_propensity_model(model_type: str, **kwargs):
    """Get sklearn classification model for propensity estimation.

    Parameters
    ----------
    model_type : str
        One of "logistic", "random_forest".
    **kwargs
        Additional arguments passed to model constructor.

    Returns
    -------
    sklearn estimator
        Instantiated classification model.
    """
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"CRITICAL ERROR: Unknown propensity model type.\n"
            f"Function: _get_propensity_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'logistic', 'random_forest'"
        )


def _cross_fit_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    n_folds: int,
    nuisance_model: str,
    propensity_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cross-fitted predictions for nuisance models.

    For each fold, trains models on out-of-fold data and predicts
    on in-fold data. This eliminates in-sample bias.

    Parameters
    ----------
    X : np.ndarray
        Covariates of shape (n, p).
    y : np.ndarray
        Outcomes of shape (n,).
    T : np.ndarray
        Treatment indicators of shape (n,).
    n_folds : int
        Number of folds for cross-fitting.
    nuisance_model : str
        Model type for outcome regression.
    propensity_model : str
        Model type for propensity estimation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (m_hat, e_hat) where:
        - m_hat: Cross-fitted outcome predictions E[Y|X]
        - e_hat: Cross-fitted propensity predictions P(T=1|X)
    """
    n = len(y)
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # Get train/test splits
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        T_train = T[train_idx]

        # Train outcome model on training fold: E[Y|X]
        outcome_model = _get_outcome_model(nuisance_model)
        outcome_model.fit(X_train, y_train)
        m_hat[test_idx] = outcome_model.predict(X_test)

        # Train propensity model on training fold: P(T=1|X)
        prop_model = _get_propensity_model(propensity_model)
        prop_model.fit(X_train, T_train)
        e_hat[test_idx] = prop_model.predict_proba(X_test)[:, 1]

    return m_hat, e_hat


def _influence_function_se(
    Y_tilde: np.ndarray,
    T_tilde: np.ndarray,
    theta: float,
) -> float:
    """Compute standard error using influence function.

    For the partially linear model, the influence function is:
    ψᵢ = (Ỹᵢ - θ·T̃ᵢ)·T̃ᵢ / E[T̃²]

    The variance of θ̂ is Var(θ̂) = E[ψ²] / n

    Parameters
    ----------
    Y_tilde : np.ndarray
        Outcome residuals (Y - m̂(X)).
    T_tilde : np.ndarray
        Treatment residuals (T - ê(X)).
    theta : float
        Point estimate of treatment effect.

    Returns
    -------
    float
        Standard error of θ̂.
    """
    n = len(Y_tilde)

    # Denominator: E[T̃²]
    T_tilde_sq_mean = np.mean(T_tilde ** 2)

    if T_tilde_sq_mean < 1e-10:
        # Fallback for degenerate case
        return np.std(Y_tilde, ddof=1) / np.sqrt(n)

    # Influence function: ψᵢ = (Ỹᵢ - θ·T̃ᵢ)·T̃ᵢ / E[T̃²]
    psi = (Y_tilde - theta * T_tilde) * T_tilde / T_tilde_sq_mean

    # Variance of θ̂ = Var(ψ) / n
    var_theta = np.var(psi, ddof=1) / n

    return np.sqrt(var_theta)


def double_ml(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    nuisance_model: Literal["linear", "ridge", "random_forest"] = "ridge",
    propensity_model: Literal["logistic", "random_forest"] = "logistic",
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate treatment effects using Double Machine Learning.

    Implements the partially linear model with K-fold cross-fitting to
    eliminate regularization bias from ML nuisance models.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    n_folds : int, default=5
        Number of folds for cross-fitting. Must be >= 2.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for CATE modeling (final stage).
    nuisance_model : {"linear", "ridge", "random_forest"}, default="ridge"
        Model for outcome regression E[Y|X]. Ridge is recommended for
        regularization without overfitting.
    propensity_model : {"logistic", "random_forest"}, default="logistic"
        Model for propensity estimation P(T=1|X).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect
        - ate_se: Standard error of ATE (influence function based)
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "double_ml"

    Raises
    ------
    ValueError
        If inputs are invalid, n_folds < 2, or model types unknown.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> # Confounded treatment: propensity depends on X
    >>> propensity = 1 / (1 + np.exp(-X[:, 0]))
    >>> T = np.random.binomial(1, propensity, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)  # True ATE = 2
    >>> result = double_ml(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.05

    Notes
    -----
    **Why Cross-Fitting Matters** (CONCERN-29):

    Without cross-fitting (like R-learner), using the same data for:
    1. Training nuisance models (propensity, outcome)
    2. Estimating treatment effects

    introduces "regularization bias" - the nuisance model errors are
    correlated with the effect estimation, biasing θ̂.

    Cross-fitting breaks this correlation by ensuring out-of-sample
    predictions for each observation.

    **Comparison with R-Learner**:
    - R-learner: Same idea (Robinson transformation) but in-sample
    - Double ML: K× computation but eliminates regularization bias
    - Double ML achieves √n-consistency even with slow nuisance rates

    References
    ----------
    - Chernozhukov et al. (2018). "Double/debiased machine learning."

    See Also
    --------
    r_learner : In-sample Robinson transformation (faster, some bias).
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Validate n_folds
    if n_folds < 2:
        raise ValueError(
            f"CRITICAL ERROR: n_folds must be >= 2.\n"
            f"Function: double_ml\n"
            f"Got: n_folds = {n_folds}\n"
            f"Cross-fitting requires at least 2 folds."
        )

    if n_folds > n // 10:
        # Warn if folds are too small
        import warnings
        warnings.warn(
            f"n_folds={n_folds} results in small fold sizes ({n // n_folds}). "
            f"Consider using fewer folds for n={n} observations.",
            UserWarning,
        )

    # =========================================================================
    # Step 1-2: Cross-fit nuisance models
    # =========================================================================
    m_hat, e_hat = _cross_fit_nuisance(
        X=covariates,
        y=outcomes,
        T=treatment,
        n_folds=n_folds,
        nuisance_model=nuisance_model,
        propensity_model=propensity_model,
    )

    # Clip propensity to avoid extreme weights
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = outcomes - m_hat  # Outcome residual: Ỹ = Y - m̂(X)
    T_tilde = treatment - e_hat  # Treatment residual: T̃ = T - ê(X)

    # =========================================================================
    # Step 4: Estimate ATE
    # =========================================================================
    # For the partially linear model: θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
    T_tilde_sq_sum = np.sum(T_tilde ** 2)

    if T_tilde_sq_sum < 1e-10:
        raise ValueError(
            f"CRITICAL ERROR: Treatment residuals too small.\n"
            f"Function: double_ml\n"
            f"Sum of T̃² = {T_tilde_sq_sum:.2e}\n"
            f"This indicates propensity is almost constant (no treatment variation).\n"
            f"Check that treatment assignment has variation conditional on X."
        )

    ate = np.sum(Y_tilde * T_tilde) / T_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE(X) - heterogeneous effects
    # =========================================================================
    # For CATE, we use the cross-fitted residuals with a final model
    # τ̂(X) = E[Ỹ/T̃ | X] via weighted regression

    # Create transformed features
    X_transformed = covariates * T_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, T_tilde])

    # Fit CATE model
    cate_model = _get_outcome_model(model)
    cate_model.fit(X_with_intercept, Y_tilde)

    # Predict CATE for each unit
    X_pred = np.column_stack([covariates, np.ones(n)])
    cate = cate_model.predict(X_pred)

    # =========================================================================
    # Step 6: Standard error via influence function
    # =========================================================================
    ate_se = _influence_function_se(Y_tilde, T_tilde, ate)

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
        method="double_ml",
    )
