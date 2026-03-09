"""A-learning for Dynamic Treatment Regimes.

Implements A-learning (Advantage Learning) with double robustness for estimating
optimal dynamic treatment regimes. A-learning is consistent if either the
propensity score model or the baseline outcome model is correctly specified.

Algorithm
---------
A-learning estimates the optimal treatment regime by solving the estimating
equation:

    E[(A - π(H)) * (Y - γ(H,A;ψ) - m(H)) * ∂γ/∂ψ] = 0

where:
- π(H) = P(A=1|H) is the propensity score
- γ(H,A;ψ) = A * H'ψ is the blip function (treatment contrast)
- m(H) = E[Y|H, A=0] is the baseline outcome model

For multi-stage settings, backward induction is used similar to Q-learning.

Double Robustness
-----------------
A-learning is consistent if EITHER:
1. π(H) is correctly specified, OR
2. m(H) is correctly specified

This makes it more robust to model misspecification than Q-learning.

References
----------
Robins, J. M. (2004). Optimal structural nested models for optimal sequential
    decisions. In Proceedings of the Second Seattle Symposium on Biostatistics.
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B, 65(2), 331-355.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating optimal
    dynamic treatment regimes. Statistical Science, 29(4), 640-661.
"""

import numpy as np
from scipy import stats
from scipy.special import expit, logit
from typing import Callable, Optional
import warnings

from .types import DTRData, ALearningResult


def a_learning(
    data: DTRData,
    propensity_model: str = "logit",
    outcome_model: str = "ols",
    doubly_robust: bool = True,
    se_method: str = "sandwich",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    propensity_trim: float = 0.01,
) -> ALearningResult:
    """A-learning for optimal dynamic treatment regime estimation.

    Estimates the optimal treatment regime using doubly robust A-learning
    with backward induction for multi-stage settings.

    Parameters
    ----------
    data : DTRData
        Multi-stage treatment data with outcomes, treatments, and covariates.
    propensity_model : str, default="logit"
        Model for P(A=1|H): "logit" or "probit".
    outcome_model : str, default="ols"
        Model for E[Y|H, A=0]: "ols" or "ridge".
    doubly_robust : bool, default=True
        Use doubly robust estimator. If False, uses simple weighted regression.
    se_method : str, default="sandwich"
        Standard error method: "sandwich" or "bootstrap".
    n_bootstrap : int, default=500
        Number of bootstrap replicates if se_method="bootstrap".
    alpha : float, default=0.05
        Significance level for confidence intervals.
    propensity_trim : float, default=0.01
        Trim propensity scores to [trim, 1-trim] to avoid extreme weights.

    Returns
    -------
    ALearningResult
        Contains optimal value, blip coefficients, and regime.

    Notes
    -----
    A-learning is doubly robust: consistent if either propensity or outcome
    model is correct. This is particularly useful when one is uncertain about
    model specification.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.dtr import DTRData, a_learning
    >>> n = 500
    >>> X = np.random.randn(n, 3)
    >>> A = np.random.binomial(1, 0.5, n)
    >>> Y = X[:, 0] + 2.0 * A + np.random.randn(n)
    >>> data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
    >>> result = a_learning(data)
    >>> print(f"Optimal value: {result.value_estimate:.3f}")
    """
    # Validate inputs
    if propensity_model not in ["logit", "probit"]:
        raise ValueError(
            f"CRITICAL ERROR: Unknown propensity_model.\n"
            f"Function: a_learning\n"
            f"propensity_model must be 'logit' or 'probit', got '{propensity_model}'"
        )
    if outcome_model not in ["ols", "ridge"]:
        raise ValueError(
            f"CRITICAL ERROR: Unknown outcome_model.\n"
            f"Function: a_learning\n"
            f"outcome_model must be 'ols' or 'ridge', got '{outcome_model}'"
        )
    if se_method not in ["sandwich", "bootstrap"]:
        raise ValueError(
            f"CRITICAL ERROR: Unknown se_method.\n"
            f"Function: a_learning\n"
            f"se_method must be 'sandwich' or 'bootstrap', got '{se_method}'"
        )
    if data.n_obs < 50:
        warnings.warn(
            f"Small sample size (n={data.n_obs}). A-learning estimates may be unstable.",
            UserWarning,
        )

    K = data.n_stages
    n = data.n_obs

    # Storage for results
    blip_coefficients = [None] * K
    blip_se = [None] * K

    # Backward induction: k = K, K-1, ..., 1 (0-indexed: K-1, K-2, ..., 0)
    future_value = np.zeros(n)

    for k in range(K - 1, -1, -1):
        # Get data for this stage
        Y_k = data.outcomes[k]
        A_k = data.treatments[k]
        H_k = data.get_history(k + 1)

        # Pseudo-outcome for this stage
        pseudo_outcome = Y_k + future_value

        # Fit propensity score
        propensity, _ = _fit_propensity(A_k, H_k, propensity_model)

        # Trim propensity scores
        propensity = np.clip(propensity, propensity_trim, 1 - propensity_trim)

        # Fit baseline outcome model (on controls)
        if doubly_robust:
            baseline_pred, _ = _fit_baseline_outcome(pseudo_outcome, A_k, H_k, outcome_model)
        else:
            baseline_pred = np.zeros(n)

        # Solve A-learning estimating equation
        psi_k = _solve_a_learning_equation(pseudo_outcome, A_k, H_k, propensity, baseline_pred)

        # Compute standard errors
        psi_se = _compute_a_learning_se(
            pseudo_outcome=pseudo_outcome,
            treatment=A_k,
            history=H_k,
            propensity=propensity,
            baseline_pred=baseline_pred,
            blip_coef=psi_k,
            method=se_method,
            n_bootstrap=n_bootstrap,
        )

        # Store results
        blip_coefficients[k] = psi_k
        blip_se[k] = psi_se

        # Compute value function for next iteration
        # Add intercept for blip computation
        H_k_aug = np.column_stack([np.ones(n), H_k])
        blip_values = H_k_aug @ psi_k
        optimal_A_k = (blip_values > 0).astype(float)
        # Value = baseline + max(0, blip)
        future_value = baseline_pred + np.maximum(0, blip_values)

    # Build optimal regime function
    def optimal_regime(history: np.ndarray, stage: int = 1) -> int:
        """Return optimal treatment for given history at stage."""
        k = stage - 1
        if k < 0 or k >= K:
            raise ValueError(f"stage must be in [1, {K}], got {stage}")
        history = np.atleast_1d(history)
        history_aug = np.concatenate([[1.0], history])
        blip = history_aug @ blip_coefficients[k]
        return int(blip > 0)

    # Estimate value under optimal regime
    H_1 = data.get_history(1)
    H_1_aug = np.column_stack([np.ones(n), H_1])

    # Refit baseline for stage 1 for value estimation
    if doubly_robust:
        baseline_1, _ = _fit_baseline_outcome(
            data.outcomes[0], data.treatments[0], H_1, outcome_model
        )
    else:
        baseline_1 = np.zeros(n)

    value_estimate = np.mean(baseline_1 + np.maximum(0, H_1_aug @ blip_coefficients[0]))

    # Compute value SE
    if se_method == "bootstrap":
        value_se = _bootstrap_value_se(
            data, propensity_model, outcome_model, doubly_robust, n_bootstrap
        )
    else:
        V_1 = baseline_1 + np.maximum(0, H_1_aug @ blip_coefficients[0])
        value_se = np.std(V_1) / np.sqrt(n)

    # Confidence intervals
    z_crit = stats.norm.ppf(1 - alpha / 2)
    value_ci_lower = value_estimate - z_crit * value_se
    value_ci_upper = value_estimate + z_crit * value_se

    return ALearningResult(
        value_estimate=value_estimate,
        value_se=value_se,
        value_ci_lower=value_ci_lower,
        value_ci_upper=value_ci_upper,
        blip_coefficients=blip_coefficients,
        blip_se=blip_se,
        optimal_regime=optimal_regime,
        n_stages=K,
        propensity_model=propensity_model,
        outcome_model=outcome_model,
        doubly_robust=doubly_robust,
        se_method=se_method,
    )


def a_learning_single_stage(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    propensity_model: str = "logit",
    outcome_model: str = "ols",
    doubly_robust: bool = True,
    se_method: str = "sandwich",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> ALearningResult:
    """Convenience wrapper for single-stage A-learning.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment A of shape (n,).
    covariates : np.ndarray
        Covariates X of shape (n, p).
    propensity_model : str, default="logit"
        Propensity score model.
    outcome_model : str, default="ols"
        Baseline outcome model.
    doubly_robust : bool, default=True
        Use doubly robust estimation.
    se_method : str, default="sandwich"
        Standard error method.
    n_bootstrap : int, default=500
        Bootstrap replicates.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    ALearningResult
        A-learning results for single-stage problem.
    """
    data = DTRData(
        outcomes=[outcome],
        treatments=[treatment],
        covariates=[covariates],
    )
    return a_learning(
        data=data,
        propensity_model=propensity_model,
        outcome_model=outcome_model,
        doubly_robust=doubly_robust,
        se_method=se_method,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def _fit_propensity(
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: str = "logit",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit propensity score model P(A=1|H).

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment of shape (n,).
    covariates : np.ndarray
        Covariates of shape (n, p).
    model : str
        "logit" or "probit".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (propensity_scores, coefficients)
    """
    n, p = covariates.shape

    # Add intercept
    X = np.column_stack([np.ones(n), covariates])

    if model == "logit":
        # Iteratively reweighted least squares for logistic regression
        # Simple Newton-Raphson implementation
        beta = np.zeros(X.shape[1])

        for _ in range(25):  # Max iterations
            eta = X @ beta
            mu = expit(eta)
            # Avoid numerical issues
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            W = np.diag(mu * (1 - mu))
            z = eta + (treatment - mu) / (mu * (1 - mu))

            # Weighted least squares update
            try:
                beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]

            if np.max(np.abs(beta_new - beta)) < 1e-8:
                break
            beta = beta_new

        propensity = expit(X @ beta)

    elif model == "probit":
        # Simplified probit via logit approximation (scale by 1.7)
        # For proper probit, would need scipy.optimize
        beta = np.zeros(X.shape[1])

        for _ in range(25):
            eta = X @ beta * 1.7  # Probit-logit scaling
            mu = expit(eta)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            W = np.diag(mu * (1 - mu))
            z = eta + (treatment - mu) / (mu * (1 - mu))

            try:
                beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]

            if np.max(np.abs(beta_new - beta)) < 1e-8:
                break
            beta = beta_new

        propensity = expit(X @ beta * 1.7)

    else:
        raise ValueError(f"Unknown model: {model}")

    return propensity, beta


def _fit_baseline_outcome(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: str = "ols",
    ridge_lambda: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit baseline outcome model E[Y|H, A=0] on control observations.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome of shape (n,).
    treatment : np.ndarray
        Treatment of shape (n,).
    covariates : np.ndarray
        Covariates of shape (n, p).
    model : str
        "ols" or "ridge".
    ridge_lambda : float
        Ridge penalty (only if model="ridge").

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (predictions_for_all, coefficients)
    """
    n, p = covariates.shape

    # Add intercept
    X = np.column_stack([np.ones(n), covariates])

    # Fit on control observations (A=0)
    control_mask = treatment == 0
    X_control = X[control_mask]
    Y_control = outcome[control_mask]

    if len(Y_control) < 5:
        # Not enough controls, return zeros
        return np.zeros(n), np.zeros(X.shape[1])

    if model == "ols":
        try:
            beta = np.linalg.lstsq(X_control, Y_control, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(X.shape[1])

    elif model == "ridge":
        # Ridge regression
        XtX = X_control.T @ X_control
        XtY = X_control.T @ Y_control
        penalty = ridge_lambda * np.eye(XtX.shape[0])
        penalty[0, 0] = 0  # Don't penalize intercept
        try:
            beta = np.linalg.solve(XtX + penalty, XtY)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtX + penalty, XtY, rcond=None)[0]

    else:
        raise ValueError(f"Unknown model: {model}")

    # Predict for all observations
    predictions = X @ beta

    return predictions, beta


def _solve_a_learning_equation(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    propensity: np.ndarray,
    baseline_pred: np.ndarray,
) -> np.ndarray:
    """Solve A-learning estimating equation for blip coefficients.

    The A-learning estimating equation is:
        E[(A - π(H)) * (Y - m(H) - A*H'ψ) * H] = 0

    This is solved via weighted least squares:
        min_ψ Σᵢ wᵢ * (Yᵢ - m̂ᵢ - Aᵢ*Hᵢ'ψ)²

    where wᵢ = (Aᵢ - πᵢ)².

    Parameters
    ----------
    outcome : np.ndarray
        Outcome of shape (n,).
    treatment : np.ndarray
        Treatment of shape (n,).
    covariates : np.ndarray
        Covariates of shape (n, p).
    propensity : np.ndarray
        Propensity scores of shape (n,).
    baseline_pred : np.ndarray
        Baseline predictions m̂(H) of shape (n,).

    Returns
    -------
    np.ndarray
        Blip coefficients ψ of shape (p+1,) with intercept.
    """
    n, p = covariates.shape

    # Add intercept
    H = np.column_stack([np.ones(n), covariates])

    # Residual after removing baseline
    residual = outcome - baseline_pred

    # Weights: (A - π)²
    weights = (treatment - propensity) ** 2

    # For treated observations: residual = A*H'ψ + noise
    # Design matrix for blip: A * H
    blip_design = treatment.reshape(-1, 1) * H

    # Weighted least squares: min_ψ Σᵢ wᵢ * (residualᵢ - Aᵢ*Hᵢ'ψ)²
    W = np.diag(weights)
    XtWX = blip_design.T @ W @ blip_design
    XtWY = blip_design.T @ W @ residual

    try:
        psi = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        psi = np.linalg.lstsq(XtWX, XtWY, rcond=None)[0]

    return psi


def _compute_a_learning_se(
    pseudo_outcome: np.ndarray,
    treatment: np.ndarray,
    history: np.ndarray,
    propensity: np.ndarray,
    baseline_pred: np.ndarray,
    blip_coef: np.ndarray,
    method: str = "sandwich",
    n_bootstrap: int = 500,
) -> np.ndarray:
    """Compute standard errors for A-learning blip coefficients.

    Parameters
    ----------
    pseudo_outcome : np.ndarray
        Pseudo-outcome of shape (n,).
    treatment : np.ndarray
        Treatment of shape (n,).
    history : np.ndarray
        History/covariates of shape (n, p).
    propensity : np.ndarray
        Propensity scores of shape (n,).
    baseline_pred : np.ndarray
        Baseline predictions of shape (n,).
    blip_coef : np.ndarray
        Blip coefficients of shape (p+1,).
    method : str
        "sandwich" or "bootstrap".
    n_bootstrap : int
        Number of bootstrap replicates.

    Returns
    -------
    np.ndarray
        Standard errors of shape (p+1,).
    """
    n, p = history.shape

    # Add intercept
    H = np.column_stack([np.ones(n), history])

    if method == "sandwich":
        # Influence function approach
        # Residual
        residual = pseudo_outcome - baseline_pred - treatment * (H @ blip_coef)

        # Weights
        weights = (treatment - propensity) ** 2

        # Design
        blip_design = treatment.reshape(-1, 1) * H

        # Meat: Σᵢ wᵢ² * residualᵢ² * Hᵢ Hᵢ'
        W = np.diag(weights)
        XtWX = blip_design.T @ W @ blip_design

        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)

        # Meat matrix
        meat = np.zeros((H.shape[1], H.shape[1]))
        for i in range(n):
            Hi = blip_design[i : i + 1].T
            meat += weights[i] ** 2 * residual[i] ** 2 * (Hi @ Hi.T)

        # Sandwich variance
        var_mat = XtWX_inv @ meat @ XtWX_inv
        se = np.sqrt(np.diag(var_mat))

    elif method == "bootstrap":
        bootstrap_psi = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)

            psi_boot = _solve_a_learning_equation(
                outcome=pseudo_outcome[idx],
                treatment=treatment[idx],
                covariates=history[idx],
                propensity=propensity[idx],
                baseline_pred=baseline_pred[idx],
            )
            bootstrap_psi.append(psi_boot)

        bootstrap_psi = np.array(bootstrap_psi)
        se = np.std(bootstrap_psi, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return se


def _bootstrap_value_se(
    data: DTRData,
    propensity_model: str,
    outcome_model: str,
    doubly_robust: bool,
    n_bootstrap: int,
) -> float:
    """Bootstrap standard error for value function estimate.

    Parameters
    ----------
    data : DTRData
        Original data.
    propensity_model : str
        Propensity model.
    outcome_model : str
        Outcome model.
    doubly_robust : bool
        Use DR estimation.
    n_bootstrap : int
        Number of bootstrap replicates.

    Returns
    -------
    float
        Bootstrap standard error.
    """
    n = data.n_obs
    K = data.n_stages
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)

        boot_data = DTRData(
            outcomes=[data.outcomes[k][idx] for k in range(K)],
            treatments=[data.treatments[k][idx] for k in range(K)],
            covariates=[data.covariates[k][idx] for k in range(K)],
        )

        result = a_learning(
            boot_data,
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            doubly_robust=doubly_robust,
            se_method="sandwich",
        )
        bootstrap_values.append(result.value_estimate)

    return np.std(bootstrap_values)
