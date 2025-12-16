"""
Augmented Synthetic Control Method (ASCM)

Implements the augmented synthetic control method from:
Ben-Michael, E., Feller, A., & Rothstein, J. (2021).
"The Augmented Synthetic Control Method"

ASCM augments traditional SCM with an outcome model to:
1. Reduce bias when pre-treatment fit is imperfect
2. Improve efficiency by using covariate information
3. Handle cases with few pre-treatment periods

The estimator combines:
- SCM weights (data-driven matching)
- Outcome model (ridge regression for extrapolation)
"""

from typing import Optional, TypedDict

import numpy as np
from scipy import stats

from .types import validate_panel_data
from .weights import compute_scm_weights, compute_pre_treatment_fit


class ASCMResult(TypedDict):
    """
    Result container for Augmented Synthetic Control Method.

    Extends SCMResult with augmentation-specific fields.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    weights: np.ndarray
    ridge_coef: np.ndarray
    pre_rmse: float
    pre_r_squared: float
    n_treated: int
    n_control: int
    n_pre_periods: int
    n_post_periods: int
    synthetic_control: np.ndarray
    augmented_control: np.ndarray  # Augmented version
    treated_series: np.ndarray
    gap: np.ndarray
    lambda_ridge: float


def augmented_synthetic_control(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    lambda_ridge: Optional[float] = None,
    inference: str = "jackknife",
) -> ASCMResult:
    """
    Estimate treatment effect using Augmented Synthetic Control Method.

    Combines synthetic control weights with ridge regression to reduce bias
    when pre-treatment fit is imperfect.

    The ASCM estimator is:
        τ̂_ASCM = τ̂_SCM + (m̂₁ - ∑ᵢ wᵢm̂ᵢ)

    where m̂ is the ridge regression prediction of post-treatment outcomes
    from pre-treatment outcomes.

    Parameters
    ----------
    outcomes : np.ndarray
        Panel data with shape (n_units, n_periods)
    treatment : np.ndarray
        Binary treatment indicator with shape (n_units,)
    treatment_period : int
        Period when treatment starts (0-indexed)
    covariates : np.ndarray, optional
        Pre-treatment covariates with shape (n_units, n_covariates)
    alpha : float
        Significance level for confidence intervals
    lambda_ridge : float, optional
        Ridge penalty parameter. If None, selected by cross-validation.
    inference : str
        "jackknife" (leave-one-out), "bootstrap", or "none"

    Returns
    -------
    ASCMResult
        Dictionary with estimation results

    References
    ----------
    Ben-Michael, E., Feller, A., & Rothstein, J. (2021).
    "The Augmented Synthetic Control Method". Journal of the American
    Statistical Association.
    """
    # Validate inputs
    validate_panel_data(outcomes, treatment, treatment_period, covariates)

    n_units, n_periods = outcomes.shape
    n_pre = treatment_period
    n_post = n_periods - treatment_period

    # Identify treated and control
    treated_mask = treatment == 1
    control_mask = treatment == 0
    n_treated = np.sum(treated_mask)
    n_control = np.sum(control_mask)

    treated_outcomes = outcomes[treated_mask, :]
    control_outcomes = outcomes[control_mask, :]

    # Split into pre/post
    treated_pre = treated_outcomes[:, :n_pre]
    control_pre = control_outcomes[:, :n_pre]
    treated_post = treated_outcomes[:, n_pre:]
    control_post = control_outcomes[:, n_pre:]

    # Step 1: Compute SCM weights
    weights, _ = compute_scm_weights(treated_pre, control_pre)

    # Step 2: Fit ridge regression (outcome model)
    # m(X) = E[Y_post | Y_pre] using ridge regression
    if lambda_ridge is None:
        lambda_ridge = _select_lambda_cv(control_pre, control_post)

    # Ridge regression: predict post from pre
    ridge_coef = _fit_ridge_outcome_model(control_pre, control_post, lambda_ridge)

    # Step 3: Compute predictions
    treated_avg = treated_outcomes.mean(axis=0) if n_treated > 1 else treated_outcomes.flatten()
    treated_pre_avg = treated_avg[:n_pre]
    treated_post_avg = treated_avg[n_pre:]

    # SCM counterfactual
    scm_synthetic = control_outcomes.T @ weights  # (n_periods,)

    # Ridge predictions
    m_treated = _predict_ridge(treated_pre_avg, ridge_coef)  # (n_post,)
    m_control = np.array([_predict_ridge(control_pre[i, :], ridge_coef) for i in range(n_control)])
    m_synthetic = m_control.T @ weights  # (n_post,)

    # Augmentation term: bias correction
    augmentation = m_treated - m_synthetic

    # Augmented synthetic control (post-treatment only)
    augmented_post = scm_synthetic[n_pre:] + augmentation

    # Full augmented series
    augmented_control = np.concatenate([scm_synthetic[:n_pre], augmented_post])

    # Gap and estimate
    gap = treated_avg - augmented_control
    estimate = np.mean(gap[n_pre:])

    # Pre-treatment fit (using SCM, not augmented)
    pre_rmse, pre_r_squared = compute_pre_treatment_fit(
        treated_pre.mean(axis=0), control_pre, weights
    )

    # Inference
    if inference == "jackknife":
        se = _jackknife_se(
            control_pre, control_post, treated_pre.mean(axis=0),
            treated_post.mean(axis=0), weights, lambda_ridge
        )
    elif inference == "bootstrap":
        se = _bootstrap_se(
            control_pre, control_post, treated_pre.mean(axis=0),
            treated_post.mean(axis=0), lambda_ridge, n_bootstrap=200
        )
    elif inference == "none":
        se = np.nan
    else:
        raise ValueError(f"Unknown inference: {inference}")

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = estimate - z * se if not np.isnan(se) else np.nan
    ci_upper = estimate + z * se if not np.isnan(se) else np.nan

    # P-value (two-sided, normal approximation)
    if not np.isnan(se) and se > 0:
        z_stat = estimate / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = np.nan

    return ASCMResult(
        estimate=float(estimate),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        weights=weights,
        ridge_coef=ridge_coef,
        pre_rmse=float(pre_rmse),
        pre_r_squared=float(pre_r_squared),
        n_treated=int(n_treated),
        n_control=int(n_control),
        n_pre_periods=int(n_pre),
        n_post_periods=int(n_post),
        synthetic_control=scm_synthetic,
        augmented_control=augmented_control,
        treated_series=treated_avg,
        gap=gap,
        lambda_ridge=float(lambda_ridge),
    )


def _fit_ridge_outcome_model(
    X_pre: np.ndarray,
    Y_post: np.ndarray,
    lambda_ridge: float,
) -> np.ndarray:
    """
    Fit ridge regression to predict post-treatment from pre-treatment.

    Parameters
    ----------
    X_pre : np.ndarray
        Pre-treatment outcomes (n_control, n_pre)
    Y_post : np.ndarray
        Post-treatment outcomes (n_control, n_post)
    lambda_ridge : float
        Ridge penalty

    Returns
    -------
    coef : np.ndarray
        Ridge coefficients (n_pre, n_post)
    """
    n_control, n_pre = X_pre.shape
    n_post = Y_post.shape[1]

    # Add intercept
    X = np.hstack([np.ones((n_control, 1)), X_pre])  # (n_control, n_pre + 1)

    # Ridge solution: β = (X'X + λI)⁻¹ X'Y
    XtX = X.T @ X
    I = np.eye(n_pre + 1)
    I[0, 0] = 0  # Don't penalize intercept
    reg = XtX + lambda_ridge * I

    coef = np.linalg.solve(reg, X.T @ Y_post)  # (n_pre + 1, n_post)

    return coef


def _predict_ridge(
    x_pre: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    """
    Predict post-treatment outcomes from pre-treatment using ridge model.

    Parameters
    ----------
    x_pre : np.ndarray
        Pre-treatment outcomes (n_pre,)
    coef : np.ndarray
        Ridge coefficients (n_pre + 1, n_post)

    Returns
    -------
    y_post : np.ndarray
        Predicted post-treatment (n_post,)
    """
    x = np.concatenate([[1.0], x_pre])  # Add intercept
    return x @ coef


def _select_lambda_cv(
    X_pre: np.ndarray,
    Y_post: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    n_folds: int = 5,
) -> float:
    """
    Select ridge penalty by cross-validation.

    Parameters
    ----------
    X_pre : np.ndarray
        Pre-treatment outcomes (n_control, n_pre)
    Y_post : np.ndarray
        Post-treatment outcomes (n_control, n_post)
    lambdas : np.ndarray, optional
        Grid of lambda values to try
    n_folds : int
        Number of CV folds

    Returns
    -------
    best_lambda : float
        Lambda with lowest CV error
    """
    n_control = X_pre.shape[0]

    if lambdas is None:
        lambdas = np.logspace(-2, 4, 20)

    if n_control < n_folds:
        n_folds = n_control  # Leave-one-out

    # Random fold assignment
    fold_ids = np.arange(n_control) % n_folds
    np.random.shuffle(fold_ids)

    cv_errors = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        fold_errors = []

        for fold in range(n_folds):
            train_mask = fold_ids != fold
            test_mask = fold_ids == fold

            X_train = X_pre[train_mask, :]
            Y_train = Y_post[train_mask, :]
            X_test = X_pre[test_mask, :]
            Y_test = Y_post[test_mask, :]

            try:
                coef = _fit_ridge_outcome_model(X_train, Y_train, lam)

                # Predict on test set
                for j in range(X_test.shape[0]):
                    pred = _predict_ridge(X_test[j, :], coef)
                    error = np.mean((Y_test[j, :] - pred) ** 2)
                    fold_errors.append(error)
            except Exception:
                fold_errors.append(np.inf)

        cv_errors[i] = np.mean(fold_errors)

    best_idx = np.argmin(cv_errors)
    return float(lambdas[best_idx])


def _jackknife_se(
    control_pre: np.ndarray,
    control_post: np.ndarray,
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    weights: np.ndarray,
    lambda_ridge: float,
) -> float:
    """
    Compute SE using jackknife (leave-one-out).

    Parameters
    ----------
    control_pre : np.ndarray
        Control pre-treatment (n_control, n_pre)
    control_post : np.ndarray
        Control post-treatment (n_control, n_post)
    treated_pre : np.ndarray
        Treated pre-treatment (n_pre,)
    treated_post : np.ndarray
        Treated post-treatment (n_post,)
    weights : np.ndarray
        SCM weights (n_control,)
    lambda_ridge : float
        Ridge penalty

    Returns
    -------
    se : float
        Jackknife standard error
    """
    n_control = control_pre.shape[0]
    n_post = control_post.shape[1]

    jackknife_estimates = []

    for i in range(n_control):
        # Leave out unit i
        mask = np.ones(n_control, dtype=bool)
        mask[i] = False

        loo_control_pre = control_pre[mask, :]
        loo_control_post = control_post[mask, :]
        loo_weights = weights[mask]
        loo_weights = loo_weights / loo_weights.sum()  # Renormalize

        try:
            # Refit ridge
            coef = _fit_ridge_outcome_model(loo_control_pre, loo_control_post, lambda_ridge)

            # Predictions
            m_treated = _predict_ridge(treated_pre, coef)
            m_control = np.array([
                _predict_ridge(loo_control_pre[j, :], coef)
                for j in range(n_control - 1)
            ])
            m_synthetic = m_control.T @ loo_weights

            # SCM part
            scm_synthetic_post = loo_control_post.T @ loo_weights

            # Augmented estimate
            augmented_post = scm_synthetic_post + (m_treated - m_synthetic)
            estimate = np.mean(treated_post - augmented_post)
            jackknife_estimates.append(estimate)

        except Exception:
            continue

    if len(jackknife_estimates) < 2:
        return np.nan

    jackknife_estimates = np.array(jackknife_estimates)

    # Jackknife SE formula
    n = len(jackknife_estimates)
    mean_est = np.mean(jackknife_estimates)
    se = np.sqrt((n - 1) / n * np.sum((jackknife_estimates - mean_est) ** 2))

    return float(se)


def _bootstrap_se(
    control_pre: np.ndarray,
    control_post: np.ndarray,
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    lambda_ridge: float,
    n_bootstrap: int = 200,
) -> float:
    """
    Compute SE using bootstrap resampling of control units.

    Parameters
    ----------
    control_pre : np.ndarray
        Control pre-treatment (n_control, n_pre)
    control_post : np.ndarray
        Control post-treatment (n_control, n_post)
    treated_pre : np.ndarray
        Treated pre-treatment (n_pre,)
    treated_post : np.ndarray
        Treated post-treatment (n_post,)
    lambda_ridge : float
        Ridge penalty
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    se : float
        Bootstrap standard error
    """
    n_control = control_pre.shape[0]

    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample control units
        idx = np.random.choice(n_control, size=n_control, replace=True)
        boot_control_pre = control_pre[idx, :]
        boot_control_post = control_post[idx, :]

        try:
            # Recompute weights
            weights, _ = compute_scm_weights(treated_pre.reshape(1, -1), boot_control_pre)

            # Refit ridge
            coef = _fit_ridge_outcome_model(boot_control_pre, boot_control_post, lambda_ridge)

            # Predictions
            m_treated = _predict_ridge(treated_pre, coef)
            m_control = np.array([
                _predict_ridge(boot_control_pre[j, :], coef)
                for j in range(n_control)
            ])
            m_synthetic = m_control.T @ weights

            # SCM part
            scm_synthetic_post = boot_control_post.T @ weights

            # Augmented estimate
            augmented_post = scm_synthetic_post + (m_treated - m_synthetic)
            estimate = np.mean(treated_post - augmented_post)
            bootstrap_estimates.append(estimate)

        except Exception:
            continue

    if len(bootstrap_estimates) < 2:
        return np.nan

    return float(np.std(bootstrap_estimates, ddof=1))
