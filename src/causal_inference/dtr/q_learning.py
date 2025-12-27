"""Q-learning for Dynamic Treatment Regimes.

Implements Q-learning with backward induction for estimating optimal
dynamic treatment regimes in multi-stage settings.

Algorithm
---------
Q-learning estimates the optimal treatment regime by fitting Q-functions
via backward induction from the final stage to the first:

    For k = K down to 1:
        1. Pseudo-outcome: Y_tilde_k = Y_k + V_hat_{k+1}(H_{k+1})
        2. Fit Q_k(H_k, a) = mu_k(H_k) + a * gamma_k(H_k)
        3. Blip: gamma_k(H_k) = Q_k(H_k, 1) - Q_k(H_k, 0) = H_k' @ psi_k
        4. Optimal: d*_k(H_k) = I(gamma_k(H_k) > 0)
        5. Value: V_k(H_k) = max_a Q_k(H_k, a)

References
----------
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B, 65(2), 331-355.
Murphy, S. A. (2005). An experimental design for the development of adaptive
    treatment strategies. Statistics in Medicine, 24(10), 1455-1481.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating optimal
    dynamic treatment regimes. Statistical Science, 29(4), 640-661.
"""

import numpy as np
from scipy import stats
from typing import Callable, Optional
import warnings

from .types import DTRData, QLearningResult


def q_learning(
    data: DTRData,
    model: str = "ols",
    se_method: str = "sandwich",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> QLearningResult:
    """Q-learning for optimal dynamic treatment regime estimation.

    Estimates the optimal treatment regime using backward induction,
    fitting Q-functions from the final stage to the first.

    Parameters
    ----------
    data : DTRData
        Multi-stage treatment data with outcomes, treatments, and covariates.
    model : str, default="ols"
        Regression model for Q-function. Currently supports "ols".
    se_method : str, default="sandwich"
        Standard error method: "sandwich" (fast, exact for linear) or
        "bootstrap" (robust, slower).
    n_bootstrap : int, default=500
        Number of bootstrap replicates if se_method="bootstrap".
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    QLearningResult
        Contains optimal value, blip coefficients, Q-functions, and regime.

    Notes
    -----
    The Q-function at each stage is parameterized as:
        Q_k(H_k, a) = H_k' @ beta_k + a * (H_k' @ psi_k)

    where beta_k captures the baseline outcome model and psi_k captures
    the treatment effect (blip function).

    The optimal treatment at stage k is:
        d*_k(H_k) = I(H_k' @ psi_k > 0)

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.dtr import DTRData, q_learning
    >>> # Generate simple single-stage data
    >>> n = 500
    >>> X = np.random.randn(n, 3)
    >>> A = np.random.binomial(1, 0.5, n)
    >>> Y = X[:, 0] + 2.0 * A + np.random.randn(n)
    >>> data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
    >>> result = q_learning(data)
    >>> print(f"Optimal value: {result.value_estimate:.3f}")
    """
    # Validate inputs
    if model not in ["ols"]:
        raise ValueError(
            f"CRITICAL ERROR: Unknown model.\n"
            f"Function: q_learning\n"
            f"model must be 'ols', got '{model}'"
        )
    if se_method not in ["sandwich", "bootstrap"]:
        raise ValueError(
            f"CRITICAL ERROR: Unknown se_method.\n"
            f"Function: q_learning\n"
            f"se_method must be 'sandwich' or 'bootstrap', got '{se_method}'"
        )
    if data.n_obs < 50:
        warnings.warn(
            f"Small sample size (n={data.n_obs}). "
            "Q-learning estimates may be unstable.",
            UserWarning,
        )

    K = data.n_stages
    n = data.n_obs

    # Storage for results
    blip_coefficients = [None] * K
    blip_se = [None] * K
    baseline_coefficients = [None] * K
    stage_q_functions = [None] * K

    # Backward induction: k = K, K-1, ..., 1 (0-indexed: K-1, K-2, ..., 0)
    # Initialize value function for stage after K (i.e., V_{K+1} = 0)
    future_value = np.zeros(n)

    for k in range(K - 1, -1, -1):
        # Get data for this stage
        Y_k = data.outcomes[k]
        A_k = data.treatments[k]
        H_k = data.get_history(k + 1)  # 1-indexed for get_history

        # Pseudo-outcome: Y_tilde_k = Y_k + V_{k+1}
        pseudo_outcome = Y_k + future_value

        # Fit Q-function
        beta_k, psi_k, q_func = _fit_stage_q_function(
            pseudo_outcome=pseudo_outcome,
            treatment=A_k,
            history=H_k,
            model=model,
        )

        # Compute standard errors
        psi_se = _compute_blip_se(
            pseudo_outcome=pseudo_outcome,
            treatment=A_k,
            history=H_k,
            blip_coef=psi_k,
            method=se_method,
            n_bootstrap=n_bootstrap,
        )

        # Store results
        blip_coefficients[k] = psi_k
        blip_se[k] = psi_se
        baseline_coefficients[k] = beta_k
        stage_q_functions[k] = q_func

        # Compute value function for this stage (for next iteration)
        # V_k(H_k) = max_a Q_k(H_k, a) = Q_k(H_k, d*_k(H_k))
        # Since Q(H,a) = H'beta + a*(H'psi), max is at a=1 if H'psi > 0, else a=0
        # Note: beta_k and psi_k include intercept, so add intercept to H_k
        H_k_aug = np.column_stack([np.ones(n), H_k])
        blip_values = H_k_aug @ psi_k
        optimal_A_k = (blip_values > 0).astype(float)
        future_value = H_k_aug @ beta_k + optimal_A_k * blip_values

    # Build optimal regime function
    def optimal_regime(history: np.ndarray, stage: int = 1) -> int:
        """Return optimal treatment for given history at stage."""
        k = stage - 1  # Convert to 0-indexed
        if k < 0 or k >= K:
            raise ValueError(f"stage must be in [1, {K}], got {stage}")
        history = np.atleast_1d(history)
        # Add intercept to match blip coefficients
        history_aug = np.concatenate([[1.0], history])
        blip = history_aug @ blip_coefficients[k]
        return int(blip > 0)

    # Estimate value under optimal regime
    # This is the mean of V_1(H_1) under the first-stage optimal decisions
    H_1 = data.get_history(1)
    H_1_aug = np.column_stack([np.ones(n), H_1])
    value_estimate = np.mean(H_1_aug @ baseline_coefficients[0] +
                            np.maximum(0, H_1_aug @ blip_coefficients[0]))

    # Compute value SE via bootstrap or influence function
    if se_method == "bootstrap":
        value_se = _bootstrap_value_se(data, model, n_bootstrap)
    else:
        # Simplified SE: variance of V_1(H_1)
        V_1 = H_1_aug @ baseline_coefficients[0] + np.maximum(0, H_1_aug @ blip_coefficients[0])
        value_se = np.std(V_1) / np.sqrt(n)

    # Confidence intervals
    z_crit = stats.norm.ppf(1 - alpha / 2)
    value_ci_lower = value_estimate - z_crit * value_se
    value_ci_upper = value_estimate + z_crit * value_se

    return QLearningResult(
        value_estimate=value_estimate,
        value_se=value_se,
        value_ci_lower=value_ci_lower,
        value_ci_upper=value_ci_upper,
        blip_coefficients=blip_coefficients,
        blip_se=blip_se,
        stage_q_functions=stage_q_functions,
        optimal_regime=optimal_regime,
        n_stages=K,
        se_method=se_method,
    )


def q_learning_single_stage(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    model: str = "ols",
    se_method: str = "sandwich",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> QLearningResult:
    """Convenience wrapper for single-stage Q-learning.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment A of shape (n,).
    covariates : np.ndarray
        Covariates X of shape (n, p).
    model : str, default="ols"
        Regression model for Q-function.
    se_method : str, default="sandwich"
        Standard error method.
    n_bootstrap : int, default=500
        Bootstrap replicates if se_method="bootstrap".
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    QLearningResult
        Q-learning results for single-stage problem.

    Examples
    --------
    >>> import numpy as np
    >>> n = 500
    >>> X = np.random.randn(n, 3)
    >>> A = np.random.binomial(1, 0.5, n)
    >>> true_blip = 2.0
    >>> Y = X[:, 0] + true_blip * A + np.random.randn(n)
    >>> result = q_learning_single_stage(Y, A, X)
    >>> print(f"Estimated blip intercept: {result.blip_coefficients[0][0]:.3f}")
    """
    data = DTRData(
        outcomes=[outcome],
        treatments=[treatment],
        covariates=[covariates],
    )
    return q_learning(
        data=data,
        model=model,
        se_method=se_method,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def _fit_stage_q_function(
    pseudo_outcome: np.ndarray,
    treatment: np.ndarray,
    history: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, Callable]:
    """Fit Q-function at a single stage.

    Models Q(H, a) = H' @ beta + a * (H' @ psi) via regression.

    Parameters
    ----------
    pseudo_outcome : np.ndarray
        Pseudo-outcome Y_tilde = Y + V_{k+1} of shape (n,).
    treatment : np.ndarray
        Treatment A of shape (n,).
    history : np.ndarray
        History/covariates H of shape (n, p).
    model : str
        Regression model type.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Callable]
        (beta, psi, q_function) where:
        - beta: baseline coefficients of shape (p,)
        - psi: blip coefficients of shape (p,)
        - q_function: callable Q(H, a) -> float
    """
    n, p = history.shape

    # Add intercept to history
    H_with_intercept = np.column_stack([np.ones(n), history])
    p_aug = p + 1

    # Build design matrix for Q(H, a) = H'beta + a*(H'psi)
    # Equivalently: [H, a*H] @ [beta; psi]
    A_times_H = treatment.reshape(-1, 1) * H_with_intercept
    design = np.column_stack([H_with_intercept, A_times_H])

    if model == "ols":
        # OLS: solve (X'X)^{-1} X'Y
        coefficients, residuals, rank, s = np.linalg.lstsq(
            design, pseudo_outcome, rcond=None
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Split coefficients
    beta = coefficients[:p_aug]
    psi = coefficients[p_aug:]

    # Build Q-function
    def q_function(H: np.ndarray, a: float) -> np.ndarray:
        """Evaluate Q(H, a) for given history and treatment."""
        H = np.atleast_2d(H)
        H_aug = np.column_stack([np.ones(len(H)), H])
        return H_aug @ beta + a * (H_aug @ psi)

    return beta, psi, q_function


def _compute_blip_se(
    pseudo_outcome: np.ndarray,
    treatment: np.ndarray,
    history: np.ndarray,
    blip_coef: np.ndarray,
    method: str = "sandwich",
    n_bootstrap: int = 500,
) -> np.ndarray:
    """Compute standard errors for blip coefficients.

    Parameters
    ----------
    pseudo_outcome : np.ndarray
        Pseudo-outcome of shape (n,).
    treatment : np.ndarray
        Treatment of shape (n,).
    history : np.ndarray
        History/covariates of shape (n, p).
    blip_coef : np.ndarray
        Blip coefficients of shape (p+1,) with intercept.
    method : str
        "sandwich" or "bootstrap".
    n_bootstrap : int
        Number of bootstrap replicates.

    Returns
    -------
    np.ndarray
        Standard errors for blip coefficients of shape (p+1,).
    """
    n, p = history.shape
    p_aug = p + 1

    # Add intercept
    H_with_intercept = np.column_stack([np.ones(n), history])

    if method == "sandwich":
        # Sandwich estimator for OLS on the blip part
        # The model is Y = H'beta + A*(H'psi) + eps
        # For the blip part, we use the treated observations weighted regression
        # or equivalently the interaction terms

        # Full design matrix
        A_times_H = treatment.reshape(-1, 1) * H_with_intercept
        design = np.column_stack([H_with_intercept, A_times_H])

        # Residuals
        beta_full, _, _, _ = np.linalg.lstsq(design, pseudo_outcome, rcond=None)
        residuals = pseudo_outcome - design @ beta_full

        # Sandwich: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
        XtX = design.T @ design
        XtX_inv = np.linalg.pinv(XtX)
        meat = design.T @ np.diag(residuals**2) @ design
        sandwich_var = XtX_inv @ meat @ XtX_inv

        # Extract variance for psi (last p_aug coefficients)
        psi_var = np.diag(sandwich_var[p_aug:, p_aug:])
        se = np.sqrt(psi_var)

    elif method == "bootstrap":
        # Nonparametric bootstrap
        bootstrap_psi = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            Y_boot = pseudo_outcome[idx]
            A_boot = treatment[idx]
            H_boot = history[idx]

            _, psi_boot, _ = _fit_stage_q_function(
                pseudo_outcome=Y_boot,
                treatment=A_boot,
                history=H_boot,
                model="ols",
            )
            bootstrap_psi.append(psi_boot)

        bootstrap_psi = np.array(bootstrap_psi)
        se = np.std(bootstrap_psi, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return se


def _bootstrap_value_se(
    data: DTRData,
    model: str,
    n_bootstrap: int,
) -> float:
    """Bootstrap standard error for value function estimate.

    Parameters
    ----------
    data : DTRData
        Original data.
    model : str
        Regression model.
    n_bootstrap : int
        Number of bootstrap replicates.

    Returns
    -------
    float
        Bootstrap standard error for value estimate.
    """
    n = data.n_obs
    K = data.n_stages
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Resample observations
        idx = np.random.choice(n, size=n, replace=True)

        # Create bootstrap data
        boot_data = DTRData(
            outcomes=[data.outcomes[k][idx] for k in range(K)],
            treatments=[data.treatments[k][idx] for k in range(K)],
            covariates=[data.covariates[k][idx] for k in range(K)],
        )

        # Fit Q-learning
        result = q_learning(boot_data, model=model, se_method="sandwich")
        bootstrap_values.append(result.value_estimate)

    return np.std(bootstrap_values)
