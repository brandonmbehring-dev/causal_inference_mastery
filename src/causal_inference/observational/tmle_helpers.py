"""Helper functions for Targeted Maximum Likelihood Estimation (TMLE).

This module provides the core computational components for TMLE:
- Clever covariate computation
- Fluctuation parameter fitting
- Convergence checking
- Outcome model updating

These functions support the targeting step that distinguishes TMLE from
standard doubly robust estimation.

References
----------
- van der Laan, M. J., & Rose, S. (2011). Targeted Learning: Causal Inference
  for Observational and Experimental Data. Springer.
- Schuler, M. S., & Rose, S. (2017). Targeted Maximum Likelihood Estimation
  for Causal Inference in Observational Studies. American Journal of
  Epidemiology, 185(1), 65-73. doi:10.1093/aje/kww165
"""

import numpy as np
from typing import Tuple, Dict, Any


def compute_clever_covariate(
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    """
    Compute the clever covariate H for TMLE targeting.

    The clever covariate is the efficient influence function component
    that guides the fluctuation step toward the efficient estimator.

    For ATE estimation:
        H = T/g(X) - (1-T)/(1-g(X))

    Where:
        - T is the treatment indicator
        - g(X) = P(T=1|X) is the propensity score

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (1=treated, 0=control), shape (n,).
    propensity : np.ndarray
        Propensity scores P(T=1|X), shape (n,).

    Returns
    -------
    np.ndarray
        Clever covariate H, shape (n,).

    Notes
    -----
    The clever covariate is bounded by clipping propensity scores to
    [1e-6, 1-1e-6] to prevent numerical overflow.
    """
    # Clip propensity to prevent division by zero
    g = np.clip(propensity, 1e-6, 1 - 1e-6)

    # H = T/g - (1-T)/(1-g)
    H = treatment / g - (1 - treatment) / (1 - g)

    return H


def compute_clever_covariate_components(
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute separate clever covariate components for treated and control.

    Used for updating Q(1,X) and Q(0,X) separately.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    propensity : np.ndarray
        Propensity scores, shape (n,).

    Returns
    -------
    H1 : np.ndarray
        Clever covariate for Q(1,X): 1/g(X), shape (n,).
    H0 : np.ndarray
        Clever covariate for Q(0,X): 1/(1-g(X)), shape (n,).
    """
    g = np.clip(propensity, 1e-6, 1 - 1e-6)

    H1 = 1.0 / g
    H0 = 1.0 / (1 - g)

    return H1, H0


def fit_fluctuation(
    outcomes: np.ndarray,
    clever_covariate: np.ndarray,
    initial_predictions: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Fit the fluctuation parameter epsilon.

    Fits the targeting submodel:
        Y = Q(X) + epsilon * H + noise

    Using weighted least squares with H as the clever covariate
    and Q(X) as the offset (initial predictions).

    For continuous outcomes, we use a linear fluctuation:
        Q*(X) = Q(X) + epsilon * H

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes Y, shape (n,).
    clever_covariate : np.ndarray
        Clever covariate H, shape (n,).
    initial_predictions : np.ndarray
        Initial outcome predictions Q(X), shape (n,).

    Returns
    -------
    epsilon : float
        Fitted fluctuation parameter.
    updated_predictions : np.ndarray
        Updated predictions Q*(X) = Q(X) + epsilon * H.

    Notes
    -----
    The fluctuation is fit as a simple linear regression of
    (Y - Q) on H, i.e., epsilon = cov(Y - Q, H) / var(H).
    """
    # Residuals from initial predictions
    residuals = outcomes - initial_predictions

    # Fit epsilon via simple regression: residuals ~ H
    # epsilon = sum(H * residuals) / sum(H^2)
    numerator = np.sum(clever_covariate * residuals)
    denominator = np.sum(clever_covariate ** 2)

    if np.abs(denominator) < 1e-10:
        # Avoid division by zero - no fluctuation needed
        epsilon = 0.0
    else:
        epsilon = numerator / denominator

    # Update predictions
    updated_predictions = initial_predictions + epsilon * clever_covariate

    return epsilon, updated_predictions


def fit_fluctuation_logistic(
    outcomes: np.ndarray,
    clever_covariate: np.ndarray,
    initial_predictions: np.ndarray,
    max_iter: int = 100,
) -> Tuple[float, np.ndarray]:
    """
    Fit fluctuation parameter using logistic submodel for bounded outcomes.

    For outcomes bounded in [0, 1], uses the logistic fluctuation:
        logit(Q*(X)) = logit(Q(X)) + epsilon * H

    This ensures Q*(X) remains in [0, 1].

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes Y in [0, 1], shape (n,).
    clever_covariate : np.ndarray
        Clever covariate H, shape (n,).
    initial_predictions : np.ndarray
        Initial outcome predictions Q(X) in (0, 1), shape (n,).
    max_iter : int, default=100
        Maximum Newton-Raphson iterations.

    Returns
    -------
    epsilon : float
        Fitted fluctuation parameter on logit scale.
    updated_predictions : np.ndarray
        Updated predictions Q*(X) = expit(logit(Q(X)) + epsilon * H).

    Notes
    -----
    Uses Newton-Raphson to solve the score equation:
        sum(H * (Y - Q*)) = 0
    """
    # Clip predictions to avoid log(0) or log(1)
    Q = np.clip(initial_predictions, 1e-6, 1 - 1e-6)
    logit_Q = np.log(Q / (1 - Q))  # logit(Q)

    epsilon = 0.0

    for _ in range(max_iter):
        # Current predictions on probability scale
        linear = logit_Q + epsilon * clever_covariate
        Q_star = 1 / (1 + np.exp(-linear))  # expit

        # Score: sum(H * (Y - Q*))
        score = np.sum(clever_covariate * (outcomes - Q_star))

        # Information (Hessian): sum(H^2 * Q* * (1 - Q*))
        info = np.sum(clever_covariate ** 2 * Q_star * (1 - Q_star))

        if np.abs(info) < 1e-10:
            break

        # Newton-Raphson update
        delta = score / info
        epsilon = epsilon + delta

        if np.abs(delta) < 1e-8:
            break

    # Final updated predictions
    linear = logit_Q + epsilon * clever_covariate
    updated_predictions = 1 / (1 + np.exp(-linear))

    return epsilon, updated_predictions


def check_convergence(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    clever_covariate: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Check TMLE targeting convergence.

    Convergence is achieved when the efficient score equation is satisfied:
        mean(H * (Y - Q*)) ≈ 0

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes Y, shape (n,).
    predictions : np.ndarray
        Current outcome predictions Q*(X), shape (n,).
    clever_covariate : np.ndarray
        Clever covariate H, shape (n,).
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    converged : bool
        True if |mean(H * residuals)| < tol.
    criterion : float
        Value of mean(H * residuals).
    """
    residuals = outcomes - predictions
    criterion = np.mean(clever_covariate * residuals)

    converged = np.abs(criterion) < tol

    return converged, criterion


def compute_tmle_ate(
    Q1_star: np.ndarray,
    Q0_star: np.ndarray,
) -> float:
    """
    Compute the TMLE estimate of ATE from targeted predictions.

    ATE = E[Q*(1,X)] - E[Q*(0,X)]
        = mean(Q1*) - mean(Q0*)

    Parameters
    ----------
    Q1_star : np.ndarray
        Targeted predictions for treatment group Q*(1,X), shape (n,).
    Q0_star : np.ndarray
        Targeted predictions for control group Q*(0,X), shape (n,).

    Returns
    -------
    float
        TMLE estimate of ATE.
    """
    return np.mean(Q1_star) - np.mean(Q0_star)


def compute_efficient_influence_function(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    Q1_star: np.ndarray,
    Q0_star: np.ndarray,
    ate: float,
) -> np.ndarray:
    """
    Compute the efficient influence function (EIF) for TMLE.

    The EIF is the core component for valid statistical inference:

        EIF_i = H1_i * (Y_i - Q1*(X_i)) * T_i
              - H0_i * (Y_i - Q0*(X_i)) * (1-T_i)
              + Q1*(X_i) - Q0*(X_i)
              - ATE

    Where H1 = 1/g and H0 = 1/(1-g).

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    propensity : np.ndarray
        Propensity scores g(X), shape (n,).
    Q1_star : np.ndarray
        Targeted predictions Q*(1,X), shape (n,).
    Q0_star : np.ndarray
        Targeted predictions Q*(0,X), shape (n,).
    ate : float
        TMLE ATE estimate.

    Returns
    -------
    np.ndarray
        Efficient influence function values, shape (n,).
    """
    g = np.clip(propensity, 1e-6, 1 - 1e-6)

    # Clever covariate components
    H1 = 1.0 / g
    H0 = 1.0 / (1 - g)

    # EIF components
    treated_component = treatment * H1 * (outcomes - Q1_star)
    control_component = (1 - treatment) * H0 * (outcomes - Q0_star)
    outcome_component = Q1_star - Q0_star

    eif = treated_component - control_component + outcome_component - ate

    return eif


def compute_tmle_variance(eif: np.ndarray) -> float:
    """
    Compute TMLE variance from the efficient influence function.

    Var(ATE) = Var(EIF) / n = E[EIF^2] / n

    Parameters
    ----------
    eif : np.ndarray
        Efficient influence function values, shape (n,).

    Returns
    -------
    float
        Estimated variance of ATE.
    """
    n = len(eif)
    variance = np.mean(eif ** 2) / n

    return variance
