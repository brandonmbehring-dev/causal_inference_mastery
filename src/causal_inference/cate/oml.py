"""Orthogonal Machine Learning for CATE estimation.

Implements the Interactive Regression Model (IRM) with doubly robust scores
for treatment effect estimation. Extends DML to the general OML framework.

Key Features
------------
- Interactive Regression Model: Y = g(T, X) + U (fully flexible)
- Doubly robust scores: consistent if propensity OR outcome model correct
- Multiple targets: ATE (Average Treatment Effect), ATTE (Average on Treated)
- K-fold cross-fitting for valid inference

Algorithm Overview
------------------
The Interactive Regression Model (Chernozhukov et al. 2018):

Y = g(T, X) + U = T*g1(X) + (1-T)*g0(X) + U

Cross-fitting procedure:
1. Split data into K folds
2. For each fold k:
   - Train g0(X), g1(X), m(X) on OTHER folds (separately on T=0, T=1)
   - Predict on fold k (out-of-sample)
3. Compute doubly robust score:
   ψ = (g1 - g0) + T(Y-g1)/m - (1-T)(Y-g0)/(1-m) - θ
4. Estimate θ via solving E[ψ] = 0

Comparison with DML (Partially Linear Model)
-------------------------------------------
- PLR: Y = θ*T + g(X) + U (treatment enters linearly)
- IRM: Y = g(T, X) + U (fully flexible in T)
- PLR requires outcome model correctly specified
- IRM is doubly robust: consistent if propensity OR outcome correct

References
----------
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
- Robins & Rotnitzky (1995). "Semiparametric efficiency in multivariate
  regression models with missing data." JASA 90(429): 122-129.
"""

import numpy as np
from typing import Literal
from scipy import stats
from sklearn.model_selection import KFold

from .base import OMLResult, validate_cate_inputs
from .dml import _get_outcome_model, _get_propensity_model


def _cross_fit_irm_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    n_folds: int,
    nuisance_model: str,
    propensity_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit nuisance models for Interactive Regression Model.

    Unlike PLR which fits joint E[Y|X], IRM fits SEPARATE outcome models
    on control (g0) and treated (g1) subsets.

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
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (g0_hat, g1_hat, m_hat) where:
        - g0_hat: Cross-fitted E[Y|T=0, X] predictions
        - g1_hat: Cross-fitted E[Y|T=1, X] predictions
        - m_hat: Cross-fitted P(T=1|X) propensity predictions
    """
    n = len(y)
    g0_hat = np.zeros(n)
    g1_hat = np.zeros(n)
    m_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # Get train/test splits
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        T_train = T[train_idx]

        # Masks for treatment groups in training data
        control_mask = T_train == 0
        treated_mask = T_train == 1

        # =================================================================
        # Fit g0: E[Y|T=0, X] on CONTROL units only
        # =================================================================
        if np.sum(control_mask) >= 2:
            g0_model = _get_outcome_model(nuisance_model)
            g0_model.fit(X_train[control_mask], y_train[control_mask])
            g0_hat[test_idx] = g0_model.predict(X_test)
        else:
            # Fallback: use control mean
            g0_hat[test_idx] = np.mean(y_train[control_mask]) if np.any(control_mask) else 0.0

        # =================================================================
        # Fit g1: E[Y|T=1, X] on TREATED units only
        # =================================================================
        if np.sum(treated_mask) >= 2:
            g1_model = _get_outcome_model(nuisance_model)
            g1_model.fit(X_train[treated_mask], y_train[treated_mask])
            g1_hat[test_idx] = g1_model.predict(X_test)
        else:
            # Fallback: use treated mean
            g1_hat[test_idx] = np.mean(y_train[treated_mask]) if np.any(treated_mask) else 0.0

        # =================================================================
        # Fit propensity: P(T=1|X) on all training data
        # =================================================================
        prop_model = _get_propensity_model(propensity_model)
        prop_model.fit(X_train, T_train)
        m_hat[test_idx] = prop_model.predict_proba(X_test)[:, 1]

    return g0_hat, g1_hat, m_hat


def _irm_score(
    Y: np.ndarray,
    T: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Compute IRM doubly robust influence function scores.

    The doubly robust score for ATE under IRM:
    ψ = (g1(X) - g0(X)) + T(Y-g1(X))/m(X) - (1-T)(Y-g0(X))/(1-m(X)) - θ

    This score is Neyman-orthogonal and doubly robust: it is consistent
    if EITHER the propensity model OR the outcome models are correct.

    Parameters
    ----------
    Y : np.ndarray
        Outcomes of shape (n,).
    T : np.ndarray
        Treatment indicators of shape (n,).
    g0 : np.ndarray
        Predicted E[Y|T=0, X] of shape (n,).
    g1 : np.ndarray
        Predicted E[Y|T=1, X] of shape (n,).
    m : np.ndarray
        Predicted P(T=1|X) of shape (n,).
    theta : float
        Current estimate of treatment effect.

    Returns
    -------
    np.ndarray
        Influence function scores ψᵢ of shape (n,).
    """
    # Outcome regression component
    or_component = g1 - g0

    # IPW correction for treated
    ipw_treated = T * (Y - g1) / m

    # IPW correction for control
    ipw_control = (1 - T) * (Y - g0) / (1 - m)

    # Full doubly robust score
    psi = or_component + ipw_treated - ipw_control - theta

    return psi


def _atte_score(
    Y: np.ndarray,
    T: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Compute ATTE (Average Treatment Effect on Treated) influence function scores.

    The score for ATTE:
    ψ = T(Y - g0(X) - θ) / P(T=1) - m(X)(1-T)(Y - g0(X)) / (P(T=1)(1-m(X)))

    Parameters
    ----------
    Y : np.ndarray
        Outcomes of shape (n,).
    T : np.ndarray
        Treatment indicators of shape (n,).
    g0 : np.ndarray
        Predicted E[Y|T=0, X] of shape (n,).
    m : np.ndarray
        Predicted P(T=1|X) of shape (n,).
    theta : float
        Current estimate of ATTE.

    Returns
    -------
    np.ndarray
        Influence function scores ψᵢ of shape (n,).
    """
    # Marginal treatment probability
    p_treated = np.mean(T)

    if p_treated < 1e-10:
        # Degenerate case: no treated units
        return np.zeros_like(Y)

    # Component 1: Direct effect on treated
    direct = T * (Y - g0 - theta) / p_treated

    # Component 2: IPW adjustment for control units
    ipw_adjust = m * (1 - T) * (Y - g0) / (p_treated * (1 - m))

    psi = direct - ipw_adjust

    return psi


def _irm_influence_se(psi: np.ndarray) -> float:
    """Compute standard error from influence function scores.

    SE = sqrt(Var(ψ) / n)

    Parameters
    ----------
    psi : np.ndarray
        Influence function scores of shape (n,).

    Returns
    -------
    float
        Standard error estimate.
    """
    n = len(psi)
    if n < 2:
        return np.nan
    return np.sqrt(np.var(psi, ddof=1) / n)


def irm_dml(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    target: Literal["ate", "atte"] = "ate",
    nuisance_model: Literal["linear", "ridge", "random_forest"] = "ridge",
    propensity_model: Literal["logistic", "random_forest"] = "logistic",
    alpha: float = 0.05,
) -> OMLResult:
    """Estimate treatment effects using Interactive Regression Model with DML.

    Implements the IRM model with K-fold cross-fitting and doubly robust scores.
    This is the general OML framework applied to treatment effect estimation.

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
    target : {"ate", "atte"}, default="ate"
        Target parameter:
        - "ate": Average Treatment Effect E[Y(1) - Y(0)]
        - "atte": Average Treatment Effect on Treated E[Y(1) - Y(0) | T=1]
    nuisance_model : {"linear", "ridge", "random_forest"}, default="ridge"
        Model for outcome regression E[Y|T,X]. Ridge is recommended.
    propensity_model : {"logistic", "random_forest"}, default="logistic"
        Model for propensity estimation P(T=1|X).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    OMLResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (or ATTE if target="atte")
        - ate_se: Standard error (influence function based)
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "irm_dml"
        - target: "ate" or "atte"
        - score_type: "irm"

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
    >>> # Confounded treatment
    >>> propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    >>> T = np.random.binomial(1, propensity, n)
    >>> # IRM DGP: g0 = 1 + X, g1 = 1 + X + 2 (ATE = 2)
    >>> Y = (1 - T) * (1 + X[:, 0]) + T * (1 + X[:, 0] + 2) + np.random.randn(n)
    >>> result = irm_dml(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f} (target: {result['target']})")
    ATE: 2.05 (target: ate)

    Notes
    -----
    **Why IRM over PLR (standard DML)?**

    - PLR assumes Y = θ*T + g(X) + U (linear in T)
    - IRM allows Y = g(T, X) + U (fully flexible)
    - IRM is doubly robust: consistent if propensity OR outcome correct
    - PLR requires outcome model correctly specified

    **Double Robustness**:

    The IRM score ψ = (g1-g0) + T(Y-g1)/m - (1-T)(Y-g0)/(1-m) - θ has:
    - E[ψ|g correct] = 0 (regardless of propensity)
    - E[ψ|m correct] = 0 (regardless of outcome)

    This provides insurance against model misspecification.

    **ATTE vs ATE**:

    - ATE: E[Y(1) - Y(0)] - effect on random person
    - ATTE: E[Y(1) - Y(0) | T=1] - effect on those who got treatment

    Under selection on observables, ATTE ≠ ATE when treatment effects
    are heterogeneous and correlated with treatment propensity.

    References
    ----------
    - Chernozhukov et al. (2018). "Double/debiased machine learning."
    - Robins & Rotnitzky (1995). "Semiparametric efficiency."

    See Also
    --------
    double_ml : PLR model (simpler, requires outcome model correct).
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
            f"Function: irm_dml\n"
            f"Got: n_folds = {n_folds}\n"
            f"Cross-fitting requires at least 2 folds."
        )

    # Validate target
    if target not in ("ate", "atte"):
        raise ValueError(
            f"CRITICAL ERROR: Invalid target parameter.\n"
            f"Function: irm_dml\n"
            f"Got: target = '{target}'\n"
            f"Valid options: 'ate', 'atte'"
        )

    if n_folds > n // 10:
        import warnings
        warnings.warn(
            f"n_folds={n_folds} results in small fold sizes ({n // n_folds}). "
            f"Consider using fewer folds for n={n} observations.",
            UserWarning,
        )

    # =========================================================================
    # Step 1-2: Cross-fit nuisance models for IRM
    # =========================================================================
    g0_hat, g1_hat, m_hat = _cross_fit_irm_nuisance(
        X=covariates,
        y=outcomes,
        T=treatment,
        n_folds=n_folds,
        nuisance_model=nuisance_model,
        propensity_model=propensity_model,
    )

    # Clip propensity more aggressively for IRM (has 1/m terms)
    m_hat = np.clip(m_hat, 0.025, 0.975)

    # =========================================================================
    # Step 3: Compute CATE estimates
    # =========================================================================
    # For IRM, CATE = g1(X) - g0(X) (plug-in estimator)
    cate = g1_hat - g0_hat

    # =========================================================================
    # Step 4: Estimate target parameter (ATE or ATTE)
    # =========================================================================
    if target == "ate":
        # ATE: E[Y(1) - Y(0)]
        # Initial estimate from plug-in
        theta_init = np.mean(cate)

        # Refine using doubly robust estimator
        # θ = E[g1 - g0] + E[T(Y-g1)/m] - E[(1-T)(Y-g0)/(1-m)]
        or_component = np.mean(g1_hat - g0_hat)
        ipw_treated = np.mean(treatment * (outcomes - g1_hat) / m_hat)
        ipw_control = np.mean((1 - treatment) * (outcomes - g0_hat) / (1 - m_hat))

        theta = or_component + ipw_treated - ipw_control

        # Compute influence function scores for SE
        psi = _irm_score(outcomes, treatment, g0_hat, g1_hat, m_hat, theta)

    else:  # target == "atte"
        # ATTE: E[Y(1) - Y(0) | T=1]
        # Initial estimate from plug-in on treated
        treated_mask = treatment == 1
        theta_init = np.mean(cate[treated_mask]) if np.any(treated_mask) else 0.0

        # Refine using doubly robust ATTE estimator
        p_treated = np.mean(treatment)

        if p_treated < 1e-10:
            raise ValueError(
                f"CRITICAL ERROR: No treated units.\n"
                f"Function: irm_dml\n"
                f"Cannot estimate ATTE without treated units."
            )

        # ATTE = E[T(Y - g0)] / P(T=1)
        direct = np.mean(treatment * (outcomes - g0_hat)) / p_treated

        # Add IPW adjustment
        ipw_adjust = np.mean(m_hat * (1 - treatment) * (outcomes - g0_hat) / (1 - m_hat)) / p_treated

        theta = direct - ipw_adjust

        # Compute ATTE influence function scores
        psi = _atte_score(outcomes, treatment, g0_hat, m_hat, theta)

    # =========================================================================
    # Step 5: Standard error via influence function
    # =========================================================================
    ate_se = _irm_influence_se(psi)

    # Handle degenerate SE
    if not np.isfinite(ate_se) or ate_se <= 0:
        ate_se = np.std(cate, ddof=1) / np.sqrt(n)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = theta - z_crit * ate_se
    ci_upper = theta + z_crit * ate_se

    return OMLResult(
        cate=cate,
        ate=float(theta),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="irm_dml",
        target=target,
        score_type="irm",
    )
