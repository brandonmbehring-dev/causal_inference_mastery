"""Double Machine Learning with Correlated Random Effects for Binary Treatment.

Implements DML-CRE using the Mundlak (1978) approach to handle panel data
with unobserved unit heterogeneity.

Key Features
------------
- Stratified cross-fitting by unit (preserves panel structure)
- Mundlak projection: includes time-means X̄ᵢ as covariates
- Clustered standard errors at unit level
- Supports balanced and unbalanced panels

Algorithm Overview
------------------
1. Compute unit means: X̄ᵢ = mean(Xᵢₜ) over t
2. Augment covariates: [Xᵢₜ, X̄ᵢ]
3. Cross-fit by unit (not by observation):
   - Each fold contains complete unit histories
   - Train nuisance models on other folds
   - Predict for held-out units
4. Outcome model: m(Xᵢₜ, X̄ᵢ) ≈ E[Yᵢₜ | Xᵢₜ, X̄ᵢ]
5. Propensity model: e(Xᵢₜ, X̄ᵢ) ≈ P(Dᵢₜ=1 | Xᵢₜ, X̄ᵢ)
6. Compute residuals: Ỹᵢₜ = Yᵢₜ - m̂, D̃ᵢₜ = Dᵢₜ - ê
7. ATE: θ̂ = Σᵢₜ(Ỹᵢₜ·D̃ᵢₜ) / Σᵢₜ(D̃ᵢₜ²)
8. SE via clustered influence function

References
----------
- Mundlak, Y. (1978). "On the pooling of time series and cross section data."
- Chernozhukov et al. (2018). "Double/debiased machine learning."
"""

import numpy as np
from typing import Literal, Optional
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .types import PanelData, DMLCREResult


def _get_propensity_model(model_type: str, **kwargs):
    """Get sklearn propensity model by name."""
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, **kwargs)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"CRITICAL ERROR: Unknown propensity model.\n"
            f"Function: _get_propensity_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'logistic', 'random_forest'"
        )


def _get_regression_model(model_type: str, **kwargs):
    """Get sklearn regression model by name."""
    if model_type == "linear":
        return LinearRegression(**kwargs)
    elif model_type == "ridge":
        return Ridge(**kwargs)
    elif model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"CRITICAL ERROR: Unknown outcome model.\n"
            f"Function: _get_regression_model\n"
            f"Got: model_type = '{model_type}'\n"
            f"Valid options: 'linear', 'ridge', 'random_forest'"
        )


def _stratified_kfold_by_unit(
    unit_id: np.ndarray,
    n_folds: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified K-fold splits by unit.

    Each fold contains complete unit histories - no unit is split
    across train and test.

    Parameters
    ----------
    unit_id : np.ndarray
        Unit identifiers for each observation.
    n_folds : int
        Number of folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of (train_idx, test_idx) tuples
        Indices for each fold.
    """
    np.random.seed(random_state)

    unique_units = np.unique(unit_id)
    n_units = len(unique_units)

    if n_folds > n_units:
        raise ValueError(
            f"CRITICAL ERROR: n_folds > n_units.\n"
            f"Function: _stratified_kfold_by_unit\n"
            f"n_folds: {n_folds}, n_units: {n_units}\n"
            f"Cannot have more folds than units."
        )

    # Shuffle units
    shuffled_units = np.random.permutation(unique_units)

    # Assign units to folds
    fold_assignment = np.zeros(n_units, dtype=int)
    for i, unit in enumerate(shuffled_units):
        fold_assignment[i] = i % n_folds

    # Create fold splits
    folds = []
    for k in range(n_folds):
        # Get units in this fold
        test_units = shuffled_units[fold_assignment == k]
        train_units = shuffled_units[fold_assignment != k]

        # Get observation indices
        test_idx = np.where(np.isin(unit_id, test_units))[0]
        train_idx = np.where(np.isin(unit_id, train_units))[0]

        folds.append((train_idx, test_idx))

    return folds


def _cross_fit_panel_nuisance(
    panel: PanelData,
    X_augmented: np.ndarray,
    n_folds: int,
    outcome_model: str,
    propensity_model: str,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Cross-fit nuisance models with stratified folds by unit.

    Parameters
    ----------
    panel : PanelData
        Panel data structure.
    X_augmented : np.ndarray
        Covariates augmented with unit means [Xᵢₜ, X̄ᵢ].
    n_folds : int
        Number of folds.
    outcome_model : str
        Outcome model type.
    propensity_model : str
        Propensity model type.

    Returns
    -------
    tuple
        (m_hat, e_hat, fold_info) cross-fitted predictions.
    """
    n = panel.n_obs
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)

    fold_info = _stratified_kfold_by_unit(panel.unit_id, n_folds)

    for train_idx, test_idx in fold_info:
        X_train = X_augmented[train_idx]
        X_test = X_augmented[test_idx]
        Y_train = panel.outcomes[train_idx]
        D_train = panel.treatment[train_idx]

        # Fit outcome model: E[Y|X, X̄]
        outcome_mod = _get_regression_model(outcome_model)
        outcome_mod.fit(X_train, Y_train)
        m_hat[test_idx] = outcome_mod.predict(X_test)

        # Fit propensity model: P(D=1|X, X̄)
        prop_mod = _get_propensity_model(propensity_model)
        prop_mod.fit(X_train, D_train)
        e_hat[test_idx] = prop_mod.predict_proba(X_test)[:, 1]

    # Clip propensity to avoid extreme weights
    e_hat = np.clip(e_hat, 0.01, 0.99)

    return m_hat, e_hat, fold_info


def _clustered_influence_se(
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    theta: float,
    unit_id: np.ndarray,
) -> float:
    """Compute clustered standard error via influence function.

    For panel data, we cluster the influence function at the unit level
    to account for within-unit correlation.

    Parameters
    ----------
    Y_tilde : np.ndarray
        Outcome residuals.
    D_tilde : np.ndarray
        Treatment residuals.
    theta : float
        Point estimate.
    unit_id : np.ndarray
        Unit identifiers.

    Returns
    -------
    float
        Clustered standard error.
    """
    n = len(Y_tilde)
    unique_units = np.unique(unit_id)
    n_units = len(unique_units)

    # Denominator: E[D_tilde²]
    D_tilde_sq_mean = np.mean(D_tilde**2)

    if D_tilde_sq_mean < 1e-10:
        return np.std(Y_tilde) / np.sqrt(n)

    # Compute influence function for each observation
    psi = (Y_tilde - theta * D_tilde) * D_tilde / D_tilde_sq_mean

    # Sum influence functions within each cluster (unit)
    cluster_psi = np.zeros(n_units)
    for i, unit in enumerate(unique_units):
        unit_idx = np.where(unit_id == unit)[0]
        cluster_psi[i] = np.sum(psi[unit_idx])

    # Clustered variance: Var(θ̂) = (1/n²) * Σᵢ (Σₜ ψᵢₜ)²
    var_theta = np.sum(cluster_psi**2) / (n**2)

    return np.sqrt(var_theta)


def _compute_unit_effects(
    panel: PanelData,
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Estimate unit fixed effects from residuals.

    α̂ᵢ = (1/Tᵢ) Σₜ (Ỹᵢₜ - θ·D̃ᵢₜ)

    Parameters
    ----------
    panel : PanelData
        Panel data.
    Y_tilde : np.ndarray
        Outcome residuals.
    D_tilde : np.ndarray
        Treatment residuals.
    theta : float
        Treatment effect estimate.

    Returns
    -------
    np.ndarray
        Unit effects of shape (n_units,).
    """
    unique_units = panel.get_unique_units()
    n_units = len(unique_units)
    unit_effects = np.zeros(n_units)

    for i, unit in enumerate(unique_units):
        unit_idx = panel.get_unit_indices(unit)
        # Residual after removing treatment effect
        residual = Y_tilde[unit_idx] - theta * D_tilde[unit_idx]
        unit_effects[i] = np.mean(residual)

    return unit_effects


def _compute_fold_estimates(
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    unit_id: np.ndarray,
    fold_info: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-fold ATE estimates for stability analysis."""
    n_folds = len(fold_info)
    fold_estimates = np.zeros(n_folds)
    fold_ses = np.zeros(n_folds)

    for i, (_, test_idx) in enumerate(fold_info):
        Y_fold = Y_tilde[test_idx]
        D_fold = D_tilde[test_idx]
        unit_fold = unit_id[test_idx]

        D_sq_sum = np.sum(D_fold**2)
        if D_sq_sum > 1e-10:
            fold_estimates[i] = np.sum(Y_fold * D_fold) / D_sq_sum
            fold_ses[i] = _clustered_influence_se(Y_fold, D_fold, fold_estimates[i], unit_fold)
        else:
            fold_estimates[i] = np.nan
            fold_ses[i] = np.nan

    return fold_estimates, fold_ses


def dml_cre(
    panel: PanelData,
    n_folds: int = 5,
    outcome_model: Literal["linear", "ridge", "random_forest"] = "ridge",
    propensity_model: Literal["logistic", "random_forest"] = "logistic",
    alpha: float = 0.05,
) -> DMLCREResult:
    """Estimate treatment effects using Panel DML-CRE with binary treatment.

    Implements Double Machine Learning with Correlated Random Effects
    for panel data following Mundlak (1978).

    Parameters
    ----------
    panel : PanelData
        Panel data with outcomes, treatment, covariates, unit_id, time.
    n_folds : int, default=5
        Number of folds for cross-fitting. Must be <= n_units.
    outcome_model : {"linear", "ridge", "random_forest"}, default="ridge"
        Model for outcome regression E[Y|X, X̄].
    propensity_model : {"logistic", "random_forest"}, default="logistic"
        Model for propensity P(D=1|X, X̄).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    DMLCREResult
        Results including ATE, SE, CATE, unit effects, and diagnostics.

    Raises
    ------
    ValueError
        If treatment is not binary or inputs are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.panel import PanelData, dml_cre
    >>> np.random.seed(42)
    >>> n_units, n_periods = 50, 10
    >>> n_obs = n_units * n_periods
    >>> unit_id = np.repeat(np.arange(n_units), n_periods)
    >>> time = np.tile(np.arange(n_periods), n_units)
    >>> X = np.random.randn(n_obs, 3)
    >>> # Unit effects correlated with X
    >>> alpha_i = 0.5 * np.repeat(np.mean(X.reshape(n_units, n_periods, -1), axis=1)[:, 0], n_periods)
    >>> D = (np.random.rand(n_obs) < 0.5 + 0.2 * X[:, 0]).astype(float)
    >>> Y = alpha_i + X[:, 0] + 2.0 * D + np.random.randn(n_obs)
    >>> panel = PanelData(Y, D, X, unit_id, time)
    >>> result = dml_cre(panel)
    >>> print(f"ATE: {result.ate:.2f} ± {result.ate_se:.2f}")

    Notes
    -----
    **Mundlak (1978) Approach**:

    The key insight is that unobserved unit effects αᵢ may be correlated
    with covariates. Mundlak's solution:

    1. Assume E[αᵢ | Xᵢ] = γ·X̄ᵢ (linear projection)
    2. Include X̄ᵢ = mean(Xᵢₜ over t) as additional covariates
    3. This controls for time-invariant confounding through the projection

    **Stratified Cross-Fitting**:

    Unlike standard DML which splits by observation, Panel DML-CRE splits
    by unit to preserve the panel structure. All observations for unit i
    are in the same fold.

    References
    ----------
    - Mundlak (1978). "On the pooling of time series and cross section data."
    - Chernozhukov et al. (2018). "Double/debiased machine learning."
    """
    # Validate binary treatment
    unique_treatments = np.unique(panel.treatment)
    if not np.array_equal(np.sort(unique_treatments), np.array([0.0, 1.0])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0, 1).\n"
            f"Function: dml_cre\n"
            f"Found unique values: {unique_treatments}\n"
            f"Use dml_cre_continuous for continuous treatment."
        )

    # Validate n_folds
    if n_folds < 2:
        raise ValueError(
            f"CRITICAL ERROR: n_folds must be >= 2.\nFunction: dml_cre\nGot: n_folds = {n_folds}"
        )

    if n_folds > panel.n_units:
        raise ValueError(
            f"CRITICAL ERROR: n_folds > n_units.\n"
            f"Function: dml_cre\n"
            f"n_folds: {n_folds}, n_units: {panel.n_units}\n"
            f"Cannot have more folds than units."
        )

    n = panel.n_obs

    # =========================================================================
    # Step 1: Compute unit means (Mundlak projection)
    # =========================================================================
    unit_means = panel.compute_unit_means()

    # Augment covariates with unit means: [Xᵢₜ, X̄ᵢ]
    X_augmented = np.column_stack([panel.covariates, unit_means])

    # =========================================================================
    # Step 2: Cross-fit nuisance models
    # =========================================================================
    m_hat, e_hat, fold_info = _cross_fit_panel_nuisance(
        panel, X_augmented, n_folds, outcome_model, propensity_model
    )

    # Compute R-squared for diagnostics
    ss_total_y = np.sum((panel.outcomes - np.mean(panel.outcomes)) ** 2)
    ss_resid_y = np.sum((panel.outcomes - m_hat) ** 2)
    outcome_r2 = 1 - ss_resid_y / ss_total_y if ss_total_y > 0 else 0.0

    # Pseudo R² for propensity (McFadden)
    null_ll = np.sum(panel.treatment * np.log(0.5) + (1 - panel.treatment) * np.log(0.5))
    model_ll = np.sum(
        panel.treatment * np.log(e_hat + 1e-10) + (1 - panel.treatment) * np.log(1 - e_hat + 1e-10)
    )
    treatment_r2 = 1 - model_ll / null_ll if null_ll != 0 else 0.0

    # =========================================================================
    # Step 3: Compute residuals
    # =========================================================================
    Y_tilde = panel.outcomes - m_hat
    D_tilde = panel.treatment - e_hat

    # =========================================================================
    # Step 4: Estimate ATE
    # =========================================================================
    D_tilde_sq_sum = np.sum(D_tilde**2)

    if D_tilde_sq_sum < 1e-10:
        raise ValueError(
            f"CRITICAL ERROR: Treatment residuals too small.\n"
            f"Function: dml_cre\n"
            f"Sum of D_tilde² = {D_tilde_sq_sum:.2e}\n"
            f"Propensity model may be too good (no residual variation)."
        )

    ate = np.sum(Y_tilde * D_tilde) / D_tilde_sq_sum

    # =========================================================================
    # Step 5: Estimate CATE (heterogeneous effects)
    # =========================================================================
    # For CATE, use weighted regression on X_augmented
    X_transformed = X_augmented * D_tilde[:, np.newaxis]
    X_with_intercept = np.column_stack([X_transformed, D_tilde])

    cate_model = _get_regression_model("linear")
    cate_model.fit(X_with_intercept, Y_tilde)

    X_pred = np.column_stack([X_augmented, np.ones(n)])
    cate = cate_model.predict(X_pred)

    # =========================================================================
    # Step 6: Compute SE via clustered influence function
    # =========================================================================
    ate_se = _clustered_influence_se(Y_tilde, D_tilde, ate, panel.unit_id)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_crit * ate_se
    ci_upper = ate + z_crit * ate_se

    # =========================================================================
    # Step 7: Compute unit effects
    # =========================================================================
    unit_effects = _compute_unit_effects(panel, Y_tilde, D_tilde, ate)

    # =========================================================================
    # Step 8: Per-fold estimates
    # =========================================================================
    fold_estimates, fold_ses = _compute_fold_estimates(Y_tilde, D_tilde, panel.unit_id, fold_info)

    return DMLCREResult(
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        cate=cate,
        method="dml_cre",
        n_units=panel.n_units,
        n_obs=n,
        n_folds=n_folds,
        outcome_r2=float(outcome_r2),
        treatment_r2=float(treatment_r2),
        unit_effects=unit_effects,
        fold_estimates=fold_estimates,
        fold_ses=fold_ses,
    )
