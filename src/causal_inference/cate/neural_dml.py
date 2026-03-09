"""Neural Double Machine Learning for CATE estimation.

This module implements Double ML with neural network nuisance models,
using K-fold cross-fitting to avoid regularization bias.

Methods
-------
- neural_double_ml: Cross-fitted DML with neural networks

References
----------
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects."
  Biometrika 108(2): 299-319.
"""

from typing import Tuple

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import CATEResult, validate_cate_inputs


def _get_mlp_regressor(
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPRegressor:
    """Create configured MLPRegressor with early stopping."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        n_iter_no_change=10,
    )


def _get_mlp_classifier(
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPClassifier:
    """Create configured MLPClassifier with early stopping."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        n_iter_no_change=10,
    )


def _cross_fit_neural_nuisance(
    covariates: np.ndarray,
    outcomes: np.ndarray,
    treatment: np.ndarray,
    n_folds: int,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-fit neural network nuisance models.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix X, shape (n, p).
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment T, shape (n,).
    n_folds : int
        Number of cross-fitting folds.
    hidden_layers : tuple of int
        Hidden layer sizes for neural networks.
    max_iter : int
        Maximum training iterations.

    Returns
    -------
    m_hat : np.ndarray
        Out-of-fold outcome predictions, shape (n,).
    e_hat : np.ndarray
        Out-of-fold propensity predictions, shape (n,).
    """
    n = len(outcomes)
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(covariates)):
        X_train, X_test = covariates[train_idx], covariates[test_idx]
        Y_train = outcomes[train_idx]
        T_train = treatment[train_idx]

        # Fit outcome model
        outcome_model = _get_mlp_regressor(hidden_layers, max_iter, random_state=42 + fold_idx)
        outcome_model.fit(X_train, Y_train)
        m_hat[test_idx] = outcome_model.predict(X_test)

        # Fit propensity model
        prop_model = _get_mlp_classifier(hidden_layers, max_iter, random_state=100 + fold_idx)
        prop_model.fit(X_train, T_train.astype(int))
        e_hat[test_idx] = prop_model.predict_proba(X_test)[:, 1]

    return m_hat, e_hat


def _influence_function_se(
    Y_tilde: np.ndarray,
    T_tilde: np.ndarray,
    ate: float,
) -> float:
    """Compute standard error using influence function.

    Parameters
    ----------
    Y_tilde : np.ndarray
        Residualized outcomes.
    T_tilde : np.ndarray
        Residualized treatment.
    ate : float
        Estimated ATE.

    Returns
    -------
    float
        Standard error of ATE estimate.
    """
    n = len(Y_tilde)

    # Influence function: psi_i = (Y_tilde_i - ate * T_tilde_i) * T_tilde_i / E[T_tilde^2]
    T_tilde_sq_mean = np.mean(T_tilde**2)

    if T_tilde_sq_mean < 1e-10:
        # Fallback if T_tilde has no variance
        return float(np.std(Y_tilde, ddof=1) / np.sqrt(n))

    psi = (Y_tilde - ate * T_tilde) * T_tilde / T_tilde_sq_mean
    se = float(np.std(psi, ddof=1) / np.sqrt(n))

    return se


def neural_double_ml(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    hidden_layers: Tuple[int, ...] = (100, 50),
    max_iter: int = 200,
    alpha: float = 0.05,
) -> CATEResult:
    """Neural Double Machine Learning for CATE estimation.

    Cross-fitted DML using neural networks for nuisance estimation.
    This eliminates regularization bias through sample splitting.

    Algorithm
    ---------
    1. K-fold cross-fit: Train propensity e(X) and outcome m(X) on each fold
    2. Out-of-fold predictions for residualization
    3. Compute residuals: Y_tilde = Y - m(X), T_tilde = T - e(X)
    4. Estimate ATE: theta = sum(Y_tilde * T_tilde) / sum(T_tilde^2)
    5. Estimate CATE via transformed features

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    n_folds : int, default=5
        Number of cross-fitting folds.
    hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for neural networks.
    max_iter : int, default=200
        Maximum training iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = neural_double_ml(Y, T, X, n_folds=3)
    >>> abs(result["ate"] - 2.0) < 0.5
    True

    Notes
    -----
    The cross-fitting procedure ensures that predictions used for residualization
    are made on held-out data, eliminating the regularization bias that would
    occur if the same data were used for both fitting and predicting.
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Check minimum samples per fold
    if n < n_folds * 10:
        raise ValueError(
            f"Insufficient samples for {n_folds}-fold cross-fitting. "
            f"Got {n} samples, need at least {n_folds * 10}."
        )

    # Cross-fit nuisance models
    m_hat, e_hat = _cross_fit_neural_nuisance(
        covariates, outcomes, treatment, n_folds, hidden_layers, max_iter
    )

    # Clip propensity scores
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # Compute residuals
    Y_tilde = outcomes - m_hat
    T_tilde = treatment - e_hat

    # Estimate ATE from partially linear model
    # theta = sum(Y_tilde * T_tilde) / sum(T_tilde^2)
    T_tilde_sq_sum = np.sum(T_tilde**2)
    if T_tilde_sq_sum < 1e-10:
        raise ValueError(
            "Propensity model nearly perfectly predicts treatment. "
            "Cannot identify treatment effect."
        )

    ate = float(np.sum(Y_tilde * T_tilde) / T_tilde_sq_sum)

    # Estimate CATE with transformed features
    X_transformed = covariates * T_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, T_tilde])

    cate_model = _get_mlp_regressor(hidden_layers, max_iter, random_state=200)
    cate_model.fit(X_with_intercept, Y_tilde)

    # Predict CATE
    X_pred = np.column_stack([covariates, np.ones(n)])
    cate = cate_model.predict(X_pred)

    # Compute SE using influence function
    ate_se = _influence_function_se(Y_tilde, T_tilde, ate)

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="neural_double_ml",
    )
