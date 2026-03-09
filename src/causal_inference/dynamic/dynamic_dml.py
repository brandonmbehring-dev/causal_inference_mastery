"""Dynamic Double Machine Learning for Time Series/Panel Treatment Effects.

Implements the Dynamic DML estimator from Lewis & Syrgkanis (2021) for estimating
treatment effects over time with flexible machine learning nuisance estimation.

The estimator:
1. Uses cross-fitting to avoid overfitting bias
2. Employs sequential g-estimation to peel off lag effects
3. Provides HAC-robust inference for autocorrelated data

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.

Chernozhukov, V., et al. (2018). Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import stats

from .cross_fitting import (
    BlockedTimeSeriesSplit,
    PanelStratifiedSplit,
    ProgressiveBlockSplit,
    RollingOriginSplit,
    get_cross_validator,
)
from .g_estimation import (
    aggregate_fold_estimates,
    compute_cumulative_effect,
    compute_cumulative_influence,
    sequential_g_estimation,
)
from .hac_inference import (
    confidence_interval,
    influence_function_se,
    newey_west_variance,
    optimal_bandwidth,
)
from .types import DynamicDMLResult, TimeSeriesPanelData, validate_dynamic_inputs


def dynamic_dml(
    outcomes: np.ndarray,
    treatments: np.ndarray,
    states: np.ndarray,
    max_lag: int = 5,
    n_folds: int = 5,
    cross_fitting: Literal["blocked", "rolling", "panel", "progressive"] = "blocked",
    nuisance_model: Literal["ridge", "random_forest", "gradient_boosting"] = "ridge",
    hac_kernel: Literal["bartlett", "qs"] = "bartlett",
    hac_bandwidth: Optional[int] = None,
    alpha: float = 0.05,
    discount_factor: float = 0.99,
    unit_id: Optional[np.ndarray] = None,
) -> DynamicDMLResult:
    """Estimate dynamic treatment effects using Double/Debiased Machine Learning.

    Implements the Dynamic DML estimator following Lewis & Syrgkanis (2021).
    Estimates the effect of treatment at each lag h = 0, 1, ..., max_lag.

    The estimator uses sequential g-estimation, "peeling off" effects from
    the most distant lag to the contemporaneous effect, with cross-fitting
    to avoid regularization bias.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (T,) for single series or (n_obs,) for panel.
    treatments : np.ndarray
        Treatment variable(s), shape (T,) or (T, n_treatments).
        Currently only binary treatments are fully supported.
    states : np.ndarray
        Covariate/state variables X, shape (T, p).
        Should include lagged outcomes, lagged treatments, and other controls.
    max_lag : int, default=5
        Maximum treatment lag to estimate effects for.
        Effects θ_h will be estimated for h = 0, 1, ..., max_lag.
    n_folds : int, default=5
        Number of cross-fitting folds.
    cross_fitting : {"blocked", "rolling", "panel", "progressive"}, default="blocked"
        Cross-fitting strategy:
        - "blocked": Divide into K contiguous blocks (allows future training)
        - "rolling": Expanding window, strictly forward-looking
        - "panel": Split by unit (requires unit_id)
        - "progressive": Progressive block estimation
    nuisance_model : {"ridge", "random_forest", "gradient_boosting"}, default="ridge"
        Model for nuisance estimation (outcome and propensity).
    hac_kernel : {"bartlett", "qs"}, default="bartlett"
        HAC kernel for variance estimation:
        - "bartlett": Newey-West (triangular weights)
        - "qs": Quadratic spectral
    hac_bandwidth : int, optional
        HAC bandwidth. Default: Newey-West optimal floor(4 * (n/100)^{2/9}).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    discount_factor : float, default=0.99
        Discount factor δ for cumulative effect: Θ = Σ_h δ^h θ_h.
    unit_id : np.ndarray, optional
        Unit identifiers for panel data. Required if cross_fitting="panel".

    Returns
    -------
    DynamicDMLResult
        Result object containing:
        - theta: Treatment effects at each lag
        - theta_se: HAC-robust standard errors
        - ci_lower, ci_upper: Confidence bounds
        - cumulative_effect: Discounted sum of effects
        - influence_function: For downstream inference
        - nuisance_r2: Goodness of fit for nuisance models

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible.

    Notes
    -----
    **Key Assumptions**:

    1. Sequential Unconfoundedness: E[Y_{t+1}(a) | X_t, A_t, ..., A_0] = E[Y_{t+1}(a) | X_t]
    2. Overlap: 0 < P(A_t = 1 | X_t) < 1
    3. Correct nuisance specification (or sufficiently flexible ML models)

    **Interpretation**:

    θ_h represents the effect of treatment at time t on outcome at time t+h.
    - θ_0: Contemporaneous effect
    - θ_1: Effect with 1-period lag
    - θ_h: Effect with h-period lag

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> # Simulate dynamic treatment effect
    >>> T = 500
    >>> X = np.random.randn(T, 3)  # States
    >>> D = (0.5 + 0.3 * X[:, 0] + np.random.randn(T) > 0).astype(float)  # Treatment
    >>> # Outcome with contemporaneous and lagged effects
    >>> Y = 2.0 * D + 1.0 * np.roll(D, 1) + 0.5 * np.roll(D, 2) + X @ [1, 0.5, 0.2] + np.random.randn(T)
    >>>
    >>> result = dynamic_dml(Y, D, X, max_lag=3, n_folds=5)
    >>> print(result.summary())  # doctest: +SKIP

    See Also
    --------
    sequential_g_estimation : Core estimation algorithm
    BlockedTimeSeriesSplit : Cross-fitting for single time series
    PanelStratifiedSplit : Cross-fitting for panel data
    """
    # Validate inputs
    outcomes, treatments, states = validate_dynamic_inputs(outcomes, treatments, states, max_lag)

    # Validate cross-fitting choice
    if cross_fitting == "panel" and unit_id is None:
        raise ValueError(
            "CRITICAL ERROR: unit_id required for panel cross-fitting.\n"
            "Either provide unit_id or use a different cross_fitting strategy."
        )

    # Create data structure
    data = TimeSeriesPanelData.from_arrays(
        outcomes=outcomes,
        treatments=treatments.ravel() if treatments.shape[1] == 1 else treatments,
        states=states,
        unit_id=unit_id,
    )

    # Get lagged data
    Y, T_lagged, X, valid_mask = data.get_lagged_data(max_lag)
    n_valid = len(Y)
    n_treatments = T_lagged.shape[2]

    # Set up cross-validator
    if cross_fitting == "panel" and unit_id is not None:
        cv = get_cross_validator("panel", n_folds=n_folds)
        # Need to extract unit_id for valid observations
        valid_unit_id = unit_id[valid_mask]
        splits = list(cv.split(Y, unit_id=valid_unit_id))
    else:
        cv = get_cross_validator(cross_fitting, n_samples=n_valid, n_folds=n_folds)
        splits = list(cv.split(Y))

    # Storage for fold results
    fold_thetas = []
    fold_influences = []
    fold_n_test = []
    all_nuisance_r2 = {"outcome_r2": [], "propensity_r2": []}

    # Cross-fitting loop
    for train_idx, test_idx in splits:
        # Create masks
        train_mask = np.zeros(n_valid, dtype=bool)
        test_mask = np.zeros(n_valid, dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # Run sequential g-estimation on this fold
        theta_fold, influence_fold, _, nuisance_r2_fold = sequential_g_estimation(
            Y=Y,
            T_lagged=T_lagged,
            X=X,
            train_mask=train_mask,
            test_mask=test_mask,
            nuisance_model_type=nuisance_model,
        )

        fold_thetas.append(theta_fold)
        fold_influences.append(influence_fold)
        fold_n_test.append(len(test_idx))

        # Accumulate nuisance R² values
        all_nuisance_r2["outcome_r2"].extend(nuisance_r2_fold["outcome_r2"])
        all_nuisance_r2["propensity_r2"].extend(nuisance_r2_fold["propensity_r2"])

    # Aggregate across folds
    theta, influence = aggregate_fold_estimates(fold_thetas, fold_influences, fold_n_test)

    # Compute HAC standard errors
    if hac_bandwidth is None:
        hac_bandwidth = optimal_bandwidth(n_valid)

    # Standard errors for each lag
    theta_se = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        se_h = influence_function_se(
            influence[:, h],
            bandwidth=hac_bandwidth,
            kernel=hac_kernel,
        )
        theta_se[h] = se_h

    # Confidence intervals
    theta_1d = theta[:, 0] if theta.ndim > 1 else theta
    ci_lower, ci_upper = confidence_interval(theta_1d, theta_se, alpha=alpha, method="normal")

    # Cumulative effect
    cumulative, weights = compute_cumulative_effect(theta, discount_factor)
    cumulative_influence = compute_cumulative_influence(influence, weights)
    cumulative_se = float(
        influence_function_se(cumulative_influence, bandwidth=hac_bandwidth, kernel=hac_kernel)
    )

    # Build result
    result = DynamicDMLResult(
        theta=theta_1d,
        theta_se=theta_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cumulative_effect=cumulative,
        cumulative_effect_se=cumulative_se,
        influence_function=influence,
        nuisance_r2=all_nuisance_r2,
        method="dynamic_dml",
        max_lag=max_lag,
        n_folds=n_folds,
        hac_bandwidth=hac_bandwidth,
        hac_kernel=hac_kernel,
        n_obs=n_valid,
        alpha=alpha,
        discount_factor=discount_factor,
    )

    return result


def dynamic_dml_panel(
    outcomes: np.ndarray,
    treatments: np.ndarray,
    states: np.ndarray,
    unit_id: np.ndarray,
    max_lag: int = 5,
    n_folds: int = 5,
    nuisance_model: Literal["ridge", "random_forest", "gradient_boosting"] = "ridge",
    hac_kernel: Literal["bartlett", "qs"] = "bartlett",
    hac_bandwidth: Optional[int] = None,
    alpha: float = 0.05,
    discount_factor: float = 0.99,
) -> DynamicDMLResult:
    """Convenience function for panel data dynamic DML.

    Uses panel cross-fitting (split by unit) which is the recommended
    approach for panel data following Lewis & Syrgkanis (2021).

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable, shape (n_obs,).
    treatments : np.ndarray
        Treatment variable(s), shape (n_obs,) or (n_obs, n_treatments).
    states : np.ndarray
        Covariates, shape (n_obs, p).
    unit_id : np.ndarray
        Unit identifiers, shape (n_obs,).
    max_lag : int, default=5
        Maximum treatment lag.
    n_folds : int, default=5
        Number of cross-fitting folds.
    nuisance_model : str, default="ridge"
        Nuisance model type.
    hac_kernel : str, default="bartlett"
        HAC kernel.
    hac_bandwidth : int, optional
        HAC bandwidth.
    alpha : float, default=0.05
        Significance level.
    discount_factor : float, default=0.99
        Discount factor for cumulative effect.

    Returns
    -------
    DynamicDMLResult
        Estimation results.

    See Also
    --------
    dynamic_dml : Main estimation function
    """
    return dynamic_dml(
        outcomes=outcomes,
        treatments=treatments,
        states=states,
        max_lag=max_lag,
        n_folds=n_folds,
        cross_fitting="panel",
        nuisance_model=nuisance_model,
        hac_kernel=hac_kernel,
        hac_bandwidth=hac_bandwidth,
        alpha=alpha,
        discount_factor=discount_factor,
        unit_id=unit_id,
    )


def simulate_dynamic_dgp(
    n_obs: int = 500,
    n_lags: int = 3,
    true_effects: Optional[np.ndarray] = None,
    n_covariates: int = 3,
    treatment_prob: float = 0.5,
    confounding_strength: float = 0.3,
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate data from a dynamic treatment effect DGP.

    Generates panel/time series data with known treatment effects at each lag.
    Useful for validation and Monte Carlo studies.

    Parameters
    ----------
    n_obs : int, default=500
        Number of observations (time periods for single series).
    n_lags : int, default=3
        Number of lagged effects (effects at lags 0, 1, ..., n_lags-1).
    true_effects : np.ndarray, optional
        True treatment effects at each lag, shape (n_lags,).
        Default: [2.0, 1.0, 0.5] (decaying effects).
    n_covariates : int, default=3
        Number of covariate state variables.
    treatment_prob : float, default=0.5
        Base treatment probability.
    confounding_strength : float, default=0.3
        Strength of confounding (covariate → treatment and outcome).
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Y : np.ndarray
        Outcomes, shape (n_obs,).
    D : np.ndarray
        Treatments, shape (n_obs,).
    X : np.ndarray
        Covariates, shape (n_obs, n_covariates).
    true_effects : np.ndarray
        True treatment effects used in DGP.

    Examples
    --------
    >>> Y, D, X, effects = simulate_dynamic_dgp(n_obs=1000, n_lags=3, seed=42)
    >>> print(f"True effects: {effects}")
    True effects: [2.  1.  0.5]
    """
    if seed is not None:
        np.random.seed(seed)

    # Default true effects (decaying)
    if true_effects is None:
        true_effects = np.array([2.0, 1.0, 0.5])[:n_lags]
    else:
        true_effects = np.asarray(true_effects)
        n_lags = len(true_effects)

    # Generate covariates (autocorrelated)
    X = np.zeros((n_obs, n_covariates))
    X[0, :] = np.random.randn(n_covariates)
    for t in range(1, n_obs):
        X[t, :] = 0.5 * X[t - 1, :] + np.sqrt(0.75) * np.random.randn(n_covariates)

    # Generate treatment (confounded by covariates)
    # P(D_t = 1 | X_t) = Φ(confounding_strength * X_t[0])
    propensity = treatment_prob + confounding_strength * X[:, 0]
    propensity = np.clip(propensity, 0.1, 0.9)
    D = (np.random.rand(n_obs) < propensity).astype(float)

    # Generate outcomes with dynamic treatment effects
    # Y_t = Σ_h θ_h D_{t-h} + β'X_t + ε_t
    covariate_effects = np.array([1.0, 0.5, 0.2])[:n_covariates]
    if len(covariate_effects) < n_covariates:
        covariate_effects = np.pad(covariate_effects, (0, n_covariates - len(covariate_effects)))

    Y = np.zeros(n_obs)
    for t in range(n_obs):
        # Add covariate effect
        Y[t] = X[t, :] @ covariate_effects

        # Add lagged treatment effects
        for h in range(n_lags):
            if t >= h:
                Y[t] += true_effects[h] * D[t - h]

        # Add noise
        Y[t] += noise_scale * np.random.randn()

    return Y, D, X, true_effects
