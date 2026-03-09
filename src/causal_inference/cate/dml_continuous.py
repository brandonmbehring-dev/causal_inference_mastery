"""Double Machine Learning for Continuous Treatment.

Extends DML to handle continuous (non-binary) treatments. The key difference
from binary DML is that we model E[D|X] using regression instead of P(T=1|X)
using classification.

Key Features
------------
- Continuous treatment D in R (not just {0, 1})
- K-fold cross-fitting for nuisance models
- Heterogeneous effects tau(X) supported
- Influence function based standard errors

Algorithm Overview
------------------
The partially linear model for continuous treatment:

Y = theta(X) * D + g(X) + epsilon    (outcome equation)
D = m(X) + eta                        (treatment equation)

Cross-fitting procedure:
1. Split data into K folds
2. For each fold k:
   - Train outcome model m_hat(X) = E[Y|X] on OTHER folds
   - Train treatment model d_hat(X) = E[D|X] on OTHER folds
   - Predict for fold k (out-of-sample)
3. Compute residuals: Y_tilde = Y - m_hat(X), D_tilde = D - d_hat(X)
4. Estimate theta = Cov(Y_tilde, D_tilde) / Var(D_tilde)

References
----------
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
- Colangelo & Lee (2020). "Double Debiased Machine Learning Nonparametric
  Inference with Continuous Treatments."
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


@dataclass
class DMLContinuousResult:
    """Result from DML with continuous treatment.

    Attributes
    ----------
    cate : np.ndarray
        Individual treatment effects tau(x_i) for each unit. Shape (n,).
    ate : float
        Average treatment effect (average marginal effect dE[Y]/dD).
    ate_se : float
        Standard error of the ATE estimate.
    ci_lower : float
        Lower bound of (1-alpha)% confidence interval for ATE.
    ci_upper : float
        Upper bound of (1-alpha)% confidence interval for ATE.
    method : str
        "dml_continuous"
    fold_estimates : np.ndarray
        Per-fold ATE estimates for stability analysis.
    fold_ses : np.ndarray
        Per-fold standard errors.
    outcome_r2 : float
        R-squared of outcome model (diagnostic).
    treatment_r2 : float
        R-squared of treatment model (diagnostic).
    n : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    """

    cate: np.ndarray
    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    method: str
    fold_estimates: np.ndarray
    fold_ses: np.ndarray
    outcome_r2: float
    treatment_r2: float
    n: int
    n_folds: int


def _get_regression_model(model_type: str, **kwargs):
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
            f"Function: _get_regression_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'linear', 'ridge', 'random_forest'"
        )


def _validate_continuous_inputs(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for continuous treatment DML.

    Unlike binary DML, we do NOT require treatment to be {0, 1}.
    We only check for sufficient variation.

    Parameters
    ----------
    outcomes : array-like
        Outcome variable Y of shape (n,).
    treatment : array-like
        Continuous treatment D of shape (n,).
    covariates : array-like
        Covariate matrix X of shape (n, p) or (n,) for single covariate.

    Returns
    -------
    tuple
        Validated (outcomes, treatment, covariates) as numpy arrays.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or insufficient variation.
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    n = len(outcomes)

    # Validate lengths
    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"Function: _validate_continuous_inputs\n"
            f"outcomes has {n} observations, treatment has {len(treatment)}.\n"
            f"All inputs must have the same number of observations."
        )

    # Handle 1D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    if len(covariates) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"Function: _validate_continuous_inputs\n"
            f"outcomes has {n} observations, covariates has {len(covariates)}.\n"
            f"All inputs must have the same number of observations."
        )

    if covariates.ndim != 2:
        raise ValueError(
            f"CRITICAL ERROR: Invalid covariate shape.\n"
            f"Function: _validate_continuous_inputs\n"
            f"covariates must be 2D array (n, p), got shape {covariates.shape}.\n"
            f"Use covariates.reshape(-1, 1) for single covariate."
        )

    # Check for NaN/Inf
    if np.any(np.isnan(outcomes)) or np.any(np.isinf(outcomes)):
        raise ValueError(
            f"CRITICAL ERROR: Invalid outcome values.\n"
            f"Function: _validate_continuous_inputs\n"
            f"outcomes contains NaN or Inf values."
        )

    if np.any(np.isnan(treatment)) or np.any(np.isinf(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: Invalid treatment values.\n"
            f"Function: _validate_continuous_inputs\n"
            f"treatment contains NaN or Inf values."
        )

    # Check for sufficient treatment variation
    treatment_std = np.std(treatment)
    if treatment_std < 1e-10:
        raise ValueError(
            f"CRITICAL ERROR: No treatment variation.\n"
            f"Function: _validate_continuous_inputs\n"
            f"Treatment std = {treatment_std:.2e}.\n"
            f"Continuous DML requires variation in treatment."
        )

    return outcomes, treatment, covariates


def _cross_fit_continuous_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    n_folds: int,
    outcome_model: str,
    treatment_model: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate cross-fitted predictions for continuous treatment nuisance models.

    Unlike binary DML which uses classification for propensity, we use
    regression for both outcome E[Y|X] and treatment E[D|X].

    Parameters
    ----------
    X : np.ndarray
        Covariates of shape (n, p).
    y : np.ndarray
        Outcomes of shape (n,).
    D : np.ndarray
        Continuous treatment of shape (n,).
    n_folds : int
        Number of folds for cross-fitting.
    outcome_model : str
        Model type for outcome regression.
    treatment_model : str
        Model type for treatment regression.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list]
        (m_hat, d_hat, fold_info) where:
        - m_hat: Cross-fitted outcome predictions E[Y|X]
        - d_hat: Cross-fitted treatment predictions E[D|X]
        - fold_info: List of (train_idx, test_idx) for each fold
    """
    n = len(y)
    m_hat = np.zeros(n)
    d_hat = np.zeros(n)
    fold_info = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        fold_info.append((train_idx, test_idx))

        # Get train/test splits
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        D_train = D[train_idx]

        # Train outcome model on training fold: E[Y|X]
        outcome_mod = _get_regression_model(outcome_model)
        outcome_mod.fit(X_train, y_train)
        m_hat[test_idx] = outcome_mod.predict(X_test)

        # Train treatment model on training fold: E[D|X]
        # KEY DIFFERENCE: regression instead of classification
        treatment_mod = _get_regression_model(treatment_model)
        treatment_mod.fit(X_train, D_train)
        d_hat[test_idx] = treatment_mod.predict(X_test)

    return m_hat, d_hat, fold_info


def _influence_function_se_continuous(
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    theta: float,
) -> float:
    """Compute standard error using influence function for continuous treatment.

    For the partially linear model with continuous D, the influence function is:
    psi_i = (Y_tilde_i - theta * D_tilde_i) * D_tilde_i / E[D_tilde^2]

    The variance of theta_hat is Var(theta_hat) = E[psi^2] / n

    Parameters
    ----------
    Y_tilde : np.ndarray
        Outcome residuals (Y - m_hat(X)).
    D_tilde : np.ndarray
        Treatment residuals (D - d_hat(X)).
    theta : float
        Point estimate of treatment effect.

    Returns
    -------
    float
        Standard error of theta_hat.
    """
    n = len(Y_tilde)

    # Denominator: E[D_tilde^2]
    D_tilde_sq_mean = np.mean(D_tilde**2)

    if D_tilde_sq_mean < 1e-10:
        # Fallback for degenerate case
        return np.std(Y_tilde, ddof=1) / np.sqrt(n)

    # Influence function: psi_i = (Y_tilde_i - theta * D_tilde_i) * D_tilde_i / E[D_tilde^2]
    psi = (Y_tilde - theta * D_tilde) * D_tilde / D_tilde_sq_mean

    # Variance of theta_hat = Var(psi) / n
    var_theta = np.var(psi, ddof=1) / n

    return np.sqrt(var_theta)


def _compute_fold_estimates(
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    fold_info: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-fold ATE estimates for stability analysis.

    Parameters
    ----------
    Y_tilde : np.ndarray
        Outcome residuals.
    D_tilde : np.ndarray
        Treatment residuals.
    fold_info : list
        List of (train_idx, test_idx) tuples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (fold_estimates, fold_ses) arrays.
    """
    n_folds = len(fold_info)
    fold_estimates = np.zeros(n_folds)
    fold_ses = np.zeros(n_folds)

    for i, (_, test_idx) in enumerate(fold_info):
        Y_fold = Y_tilde[test_idx]
        D_fold = D_tilde[test_idx]

        # Fold-specific estimate
        D_sq_sum = np.sum(D_fold**2)
        if D_sq_sum > 1e-10:
            fold_estimates[i] = np.sum(Y_fold * D_fold) / D_sq_sum
            fold_ses[i] = _influence_function_se_continuous(Y_fold, D_fold, fold_estimates[i])
        else:
            fold_estimates[i] = np.nan
            fold_ses[i] = np.nan

    return fold_estimates, fold_ses


def dml_continuous(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    outcome_model: Literal["linear", "ridge", "random_forest"] = "ridge",
    treatment_model: Literal["linear", "ridge", "random_forest"] = "ridge",
    alpha: float = 0.05,
) -> DMLContinuousResult:
    """Estimate treatment effects using Double ML with continuous treatment.

    Implements the partially linear model with K-fold cross-fitting for
    continuous (non-binary) treatment effects. This is also known as the
    "dose-response" setting.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Continuous treatment D of shape (n,). Unlike binary DML, this can
        take any real values.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    n_folds : int, default=5
        Number of folds for cross-fitting. Must be >= 2.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Base learner for CATE modeling (final stage).
    outcome_model : {"linear", "ridge", "random_forest"}, default="ridge"
        Model for outcome regression E[Y|X]. Ridge is recommended for
        regularization without overfitting.
    treatment_model : {"linear", "ridge", "random_forest"}, default="ridge"
        Model for treatment regression E[D|X]. Note: This is REGRESSION
        (not classification like binary DML).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    DMLContinuousResult
        Dataclass with fields:
        - cate: Individual treatment effects tau(x_i) of shape (n,)
        - ate: Average treatment effect (marginal effect dE[Y]/dD)
        - ate_se: Standard error of ATE (influence function based)
        - ci_lower: Lower bound of (1-alpha)% CI
        - ci_upper: Upper bound of (1-alpha)% CI
        - method: "dml_continuous"
        - fold_estimates: Per-fold ATE estimates
        - fold_ses: Per-fold standard errors
        - outcome_r2: R-squared of outcome model
        - treatment_r2: R-squared of treatment model
        - n: Number of observations
        - n_folds: Number of folds

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
    >>> # Continuous treatment: D depends on X
    >>> D = X[:, 0] + np.random.randn(n)  # Continuous, not binary
    >>> Y = 1 + X[:, 0] + 2 * D + np.random.randn(n)  # True effect = 2
    >>> result = dml_continuous(Y, D, X)
    >>> print(f"ATE: {result.ate:.2f}")
    ATE: 2.01

    Notes
    -----
    **Key Difference from Binary DML**:

    - Binary DML: Uses P(T=1|X) via classification (propensity score)
    - Continuous DML: Uses E[D|X] via regression (no propensity)

    The treatment effect theta represents the marginal effect:
    theta = dE[Y|D,X] / dD

    Under the partial linear model, this is constant in D but can
    vary with X (heterogeneous effects).

    **Interpretation**:
    - For binary treatment: ATE = E[Y(1) - Y(0)]
    - For continuous treatment: ATE = dE[Y]/dD (marginal effect at mean)

    References
    ----------
    - Chernozhukov et al. (2018). "Double/debiased machine learning."
    - Colangelo & Lee (2020). "Double Debiased ML with Continuous Treatments."

    See Also
    --------
    double_ml : DML for binary treatment.
    dml_panel : DML for panel data.
    """
    # Validate inputs (no binary check)
    outcomes, treatment, covariates = _validate_continuous_inputs(outcomes, treatment, covariates)

    n = len(outcomes)

    # Validate n_folds
    if n_folds < 2:
        raise ValueError(
            f"CRITICAL ERROR: n_folds must be >= 2.\n"
            f"Function: dml_continuous\n"
            f"Got: n_folds = {n_folds}\n"
            f"Cross-fitting requires at least 2 folds."
        )

    if n_folds > n // 10:
        import warnings

        warnings.warn(
            f"n_folds={n_folds} results in small fold sizes ({n // n_folds}). "
            f"Consider using fewer folds for n={n} observations.",
            UserWarning,
        )

    # =========================================================================
    # Step 1-2: Cross-fit nuisance models
    # =========================================================================
    m_hat, d_hat, fold_info = _cross_fit_continuous_nuisance(
        X=covariates,
        y=outcomes,
        D=treatment,
        n_folds=n_folds,
        outcome_model=outcome_model,
        treatment_model=treatment_model,
    )

    # Compute R-squared for diagnostics
    ss_total_y = np.sum((outcomes - np.mean(outcomes)) ** 2)
    ss_resid_y = np.sum((outcomes - m_hat) ** 2)
    outcome_r2 = 1 - ss_resid_y / ss_total_y if ss_total_y > 0 else 0.0

    ss_total_d = np.sum((treatment - np.mean(treatment)) ** 2)
    ss_resid_d = np.sum((treatment - d_hat) ** 2)
    treatment_r2 = 1 - ss_resid_d / ss_total_d if ss_total_d > 0 else 0.0

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = outcomes - m_hat  # Outcome residual: Y_tilde = Y - m_hat(X)
    D_tilde = treatment - d_hat  # Treatment residual: D_tilde = D - d_hat(X)

    # =========================================================================
    # Step 4: Estimate ATE
    # =========================================================================
    # For continuous treatment: theta_hat = Sum(Y_tilde * D_tilde) / Sum(D_tilde^2)
    D_tilde_sq_sum = np.sum(D_tilde**2)

    if D_tilde_sq_sum < 1e-10:
        raise ValueError(
            f"CRITICAL ERROR: Treatment residuals too small.\n"
            f"Function: dml_continuous\n"
            f"Sum of D_tilde^2 = {D_tilde_sq_sum:.2e}\n"
            f"This indicates treatment is almost perfectly predicted by X.\n"
            f"Check that treatment has variation conditional on X."
        )

    ate = np.sum(Y_tilde * D_tilde) / D_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE(X) - heterogeneous effects
    # =========================================================================
    # For CATE, we use weighted regression:
    # tau(X) = E[Y_tilde / D_tilde | X] via weighted regression with weights D_tilde^2

    # Create transformed features
    X_transformed = covariates * D_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, D_tilde])

    # Fit CATE model
    cate_model = _get_regression_model(model)
    cate_model.fit(X_with_intercept, Y_tilde)

    # Predict CATE for each unit
    X_pred = np.column_stack([covariates, np.ones(n)])
    cate = cate_model.predict(X_pred)

    # =========================================================================
    # Step 6: Standard error via influence function
    # =========================================================================
    ate_se = _influence_function_se_continuous(Y_tilde, D_tilde, ate)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    # =========================================================================
    # Step 7: Per-fold estimates for stability analysis
    # =========================================================================
    fold_estimates, fold_ses = _compute_fold_estimates(Y_tilde, D_tilde, fold_info)

    return DMLContinuousResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="dml_continuous",
        fold_estimates=fold_estimates,
        fold_ses=fold_ses,
        outcome_r2=float(outcome_r2),
        treatment_r2=float(treatment_r2),
        n=n,
        n_folds=n_folds,
    )
