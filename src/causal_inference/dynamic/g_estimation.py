"""Sequential G-Estimation for Dynamic Treatment Effects.

Implements the sequential g-estimation algorithm from Lewis & Syrgkanis (2021)
for estimating dynamic treatment effects with machine learning nuisance estimation.

The key insight is "peeling off" treatment effects sequentially from the most
distant lag to the contemporaneous effect, using Neyman-orthogonal moment
conditions at each step.

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.

Robins, J. M. (1986). A new approach to causal inference in mortality studies
with a sustained exposure period. Mathematical Modelling, 7(9-12), 1393-1512.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler


def get_nuisance_model(
    model_type: Literal["ridge", "random_forest", "gradient_boosting"],
    task: Literal["regression", "classification"] = "regression",
) -> object:
    """Get a nuisance model for outcome or propensity estimation.

    Parameters
    ----------
    model_type : {"ridge", "random_forest", "gradient_boosting"}
        Type of model to use.
    task : {"regression", "classification"}
        Whether this is for outcome (regression) or propensity (classification).

    Returns
    -------
    sklearn estimator
        Fitted model with fit() and predict() methods.
    """
    if model_type == "ridge":
        if task == "regression":
            return Ridge(alpha=1.0)
        else:
            return RidgeClassifier(alpha=1.0)

    elif model_type == "random_forest":
        if task == "regression":
            return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    elif model_type == "gradient_boosting":
        if task == "regression":
            return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def sequential_g_estimation(
    Y: np.ndarray,
    T_lagged: np.ndarray,
    X: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    nuisance_model_type: Literal["ridge", "random_forest", "gradient_boosting"] = "ridge",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Core sequential g-estimation algorithm.

    Implements the peeling procedure from Lewis & Syrgkanis (2021).

    For t = max_lag, max_lag-1, ..., 0:
        1. Adjust outcome: Ỹ_t = Y - Σ_{k>t} θ̂_k T_{t-k}
        2. Estimate nuisances: q(X) = E[Ỹ_t | X], p(X) = E[T_t | X]
        3. Solve moment: E[(Ỹ_t - θ_t T_t - q(X)) (T_t - p(X))] = 0
        4. Estimate θ_t = Cov(Ỹ_t - q(X), T_t - p(X)) / Var(T_t - p(X))

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable, shape (n,).
    T_lagged : np.ndarray
        Lagged treatment matrix, shape (n, max_lag + 1, n_treatments).
        T_lagged[t, h, :] = treatment at time t-h.
    X : np.ndarray
        Covariates/states, shape (n, p).
    train_mask : np.ndarray
        Boolean mask for training observations, shape (n,).
    test_mask : np.ndarray
        Boolean mask for test observations, shape (n,).
    nuisance_model_type : {"ridge", "random_forest", "gradient_boosting"}
        Model for nuisance estimation.

    Returns
    -------
    theta : np.ndarray
        Treatment effect at each lag, shape (max_lag + 1, n_treatments).
    influence : np.ndarray
        Influence scores for test obs, shape (n_test, max_lag + 1).
    nuisance_preds : dict
        Cross-fitted nuisance predictions.
    nuisance_r2 : dict
        R-squared values for nuisance models.

    Notes
    -----
    This function processes one train/test fold. The main dynamic_dml
    function calls this for each cross-fitting fold.
    """
    n, max_lag_plus_1, n_treatments = T_lagged.shape
    max_lag = max_lag_plus_1 - 1

    # Storage
    theta = np.zeros((max_lag_plus_1, n_treatments))
    influence_scores = np.zeros((np.sum(test_mask), max_lag_plus_1))
    nuisance_r2 = {"outcome_r2": [], "propensity_r2": []}

    # Get train/test data
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]
    X_train = X[train_mask]
    X_test = X[test_mask]
    T_train = T_lagged[train_mask]
    T_test = T_lagged[test_mask]

    # Adjusted outcome (will be modified as we peel)
    Y_adj_train = Y_train.copy()
    Y_adj_test = Y_test.copy()

    # Sequential peeling: from most distant lag to lag 0
    for h in range(max_lag, -1, -1):
        # Current treatment at lag h (for simplicity, use first treatment if multiple)
        T_h_train = T_train[:, h, 0] if n_treatments == 1 else T_train[:, h, :]
        T_h_test = T_test[:, h, 0] if n_treatments == 1 else T_test[:, h, :]

        # Ensure 1D for single treatment
        if T_h_train.ndim > 1:
            T_h_train = T_h_train[:, 0]
            T_h_test = T_h_test[:, 0]

        # Step 1: Estimate outcome nuisance q(X) = E[Y_adj | X]
        outcome_model = get_nuisance_model(nuisance_model_type, "regression")
        outcome_model.fit(X_train, Y_adj_train)
        q_train = outcome_model.predict(X_train)
        q_test = outcome_model.predict(X_test)

        # Outcome R²
        ss_res = np.sum((Y_adj_train - q_train) ** 2)
        ss_tot = np.sum((Y_adj_train - Y_adj_train.mean()) ** 2)
        r2_outcome = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        nuisance_r2["outcome_r2"].append(r2_outcome)

        # Step 2: Estimate propensity nuisance p(X) = E[T | X]
        propensity_model = get_nuisance_model(nuisance_model_type, "regression")
        propensity_model.fit(X_train, T_h_train)
        p_train = propensity_model.predict(X_train)
        p_test = propensity_model.predict(X_test)

        # Propensity R²
        ss_res_p = np.sum((T_h_train - p_train) ** 2)
        ss_tot_p = np.sum((T_h_train - T_h_train.mean()) ** 2)
        r2_propensity = 1 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0
        nuisance_r2["propensity_r2"].append(r2_propensity)

        # Step 3: Compute residuals
        Y_tilde_train = Y_adj_train - q_train
        T_tilde_train = T_h_train - p_train
        Y_tilde_test = Y_adj_test - q_test
        T_tilde_test = T_h_test - p_test

        # Step 4: Estimate theta_h via moment condition
        # theta = Cov(Y_tilde, T_tilde) / Var(T_tilde)
        # Using training data for estimation
        cov = np.mean(Y_tilde_train * T_tilde_train)
        var_t = np.mean(T_tilde_train**2)

        if var_t > 1e-10:
            theta_h = cov / var_t
        else:
            theta_h = 0.0

        theta[h, 0] = theta_h

        # Step 5: Compute influence scores for test observations
        # ψ = (Y_tilde - θ T_tilde) * T_tilde / Var(T_tilde)
        if var_t > 1e-10:
            psi = (Y_tilde_test - theta_h * T_tilde_test) * T_tilde_test / var_t
        else:
            psi = np.zeros(len(Y_tilde_test))

        influence_scores[:, h] = psi

        # Step 6: Adjust outcomes for next iteration (peel off this lag's effect)
        Y_adj_train = Y_adj_train - theta_h * T_h_train
        Y_adj_test = Y_adj_test - theta_h * T_h_test

    # Store nuisance predictions
    nuisance_preds = {
        "q_test": q_test,
        "p_test": p_test,
    }

    return theta, influence_scores, nuisance_preds, nuisance_r2


def aggregate_fold_estimates(
    fold_thetas: list[np.ndarray],
    fold_influences: list[np.ndarray],
    fold_n_test: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate treatment effect estimates across cross-fitting folds.

    Uses sample-size weighted averaging for point estimates and
    concatenates influence functions for variance estimation.

    Parameters
    ----------
    fold_thetas : list of np.ndarray
        Treatment effects from each fold, each shape (max_lag + 1, n_treatments).
    fold_influences : list of np.ndarray
        Influence scores from each fold, each shape (n_test_fold, max_lag + 1).
    fold_n_test : list of int
        Number of test observations in each fold.

    Returns
    -------
    theta : np.ndarray
        Aggregated treatment effects, shape (max_lag + 1, n_treatments).
    influence : np.ndarray
        Concatenated influence scores, shape (n_total, max_lag + 1).
    """
    n_total = sum(fold_n_test)
    max_lag_plus_1 = fold_thetas[0].shape[0]
    n_treatments = fold_thetas[0].shape[1]

    # Weighted average of theta estimates
    theta = np.zeros((max_lag_plus_1, n_treatments))
    for fold_theta, n_test in zip(fold_thetas, fold_n_test):
        theta += fold_theta * n_test
    theta /= n_total

    # Concatenate influence scores
    influence = np.vstack(fold_influences)

    return theta, influence


def compute_cumulative_effect(
    theta: np.ndarray,
    discount_factor: float = 0.99,
) -> tuple[float, np.ndarray]:
    """Compute discounted cumulative treatment effect.

    The cumulative effect is:
        Θ = Σ_{h=0}^{max_lag} δ^h θ_h

    where δ is the discount factor.

    Parameters
    ----------
    theta : np.ndarray
        Treatment effects at each lag, shape (max_lag + 1,) or (max_lag + 1, n_treatments).
    discount_factor : float
        Discount factor δ ∈ (0, 1].

    Returns
    -------
    cumulative : float
        Discounted sum of effects.
    weights : np.ndarray
        Discount weights δ^h for each lag.
    """
    if theta.ndim > 1:
        theta = theta[:, 0]  # Use first treatment for now

    max_lag_plus_1 = len(theta)
    weights = np.array([discount_factor**h for h in range(max_lag_plus_1)])
    cumulative = np.sum(weights * theta)

    return cumulative, weights


def compute_cumulative_influence(
    influence: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute influence function for cumulative effect.

    Parameters
    ----------
    influence : np.ndarray
        Influence scores, shape (n, max_lag + 1).
    weights : np.ndarray
        Discount weights for each lag.

    Returns
    -------
    np.ndarray
        Influence for cumulative effect, shape (n,).
    """
    return influence @ weights
