"""Panel Quantile Treatment Effects via RIF Regression.

Implements RIF-OLS approach (Firpo, Fortin, Lemieux 2009) adapted for panel data
with Mundlak projection and clustered standard errors.

Key Features
------------
- RIF (Recentered Influence Function) transformation for unconditional QTE
- Mundlak projection: includes time-means X̄ᵢ as covariates
- Clustered standard errors at unit level
- Supports balanced and unbalanced panels

Algorithm Overview
------------------
1. Compute unit means: X̄ᵢ = mean(Xᵢₜ) over t (Mundlak projection)
2. Estimate pooled quantile: q̂_τ = Quantile(Y, τ)
3. Estimate density at quantile: f̂_Y(q̂_τ) via kernel density
4. Compute RIF: RIF_it = q̂_τ + (τ - I(Y_it ≤ q̂_τ)) / f̂_Y(q̂_τ)
5. OLS regression: RIF ~ [1, X, X̄, D]
6. QTE(τ) = coefficient on D
7. Clustered SE by unit

References
----------
- Firpo, S., Fortin, N., & Lemieux, T. (2009). "Unconditional Quantile Regressions."
- Mundlak, Y. (1978). "On the pooling of time series and cross section data."
"""

import numpy as np
from typing import List, Optional, Sequence, Union
import warnings

from .types import PanelData, PanelQTEResult, PanelQTEBandResult


def _silverman_bandwidth(y: np.ndarray) -> float:
    """Compute Silverman's rule-of-thumb bandwidth for kernel density.

    h = 1.06 × σ̂_Y × n^(-0.2)

    Parameters
    ----------
    y : np.ndarray
        Data vector.

    Returns
    -------
    float
        Bandwidth h.
    """
    n = len(y)
    sigma = np.std(y, ddof=1)

    # Use IQR-based estimate if data is heavy-tailed
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    sigma_robust = min(sigma, iqr / 1.34)  # 1.34 = 2 * norm.ppf(0.75)

    # Silverman's rule
    h = 1.06 * sigma_robust * (n ** (-0.2))

    return max(h, 1e-10)  # Avoid zero bandwidth


def _kernel_density_at_quantile(
    y: np.ndarray,
    q_tau: float,
    bandwidth: float,
) -> float:
    """Estimate kernel density at a specific point.

    Uses Gaussian kernel: K(u) = (1/√2π) exp(-u²/2)

    Parameters
    ----------
    y : np.ndarray
        Data vector.
    q_tau : float
        Point at which to evaluate density.
    bandwidth : float
        Kernel bandwidth h.

    Returns
    -------
    float
        Density estimate f̂(q_tau).
    """
    n = len(y)
    h = bandwidth

    # Normalized distances
    u = (y - q_tau) / h

    # Gaussian kernel
    kernel_vals = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    # Density estimate
    f_hat = np.mean(kernel_vals) / h

    return max(f_hat, 1e-10)  # Avoid division by zero


def _compute_rif(
    y: np.ndarray,
    tau: float,
    q_tau: float,
    f_q: float,
) -> np.ndarray:
    """Compute Recentered Influence Function for a quantile.

    RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)

    Parameters
    ----------
    y : np.ndarray
        Outcome values.
    tau : float
        Target quantile.
    q_tau : float
        Sample quantile at τ.
    f_q : float
        Density at quantile.

    Returns
    -------
    np.ndarray
        RIF values for each observation.
    """
    indicator = (y <= q_tau).astype(float)
    rif = q_tau + (tau - indicator) / f_q

    return rif


def _panel_clustered_se(
    residuals: np.ndarray,
    D: np.ndarray,
    unit_id: np.ndarray,
) -> float:
    """Compute clustered standard error for OLS coefficient.

    For panel data, we cluster at the unit level to account for
    within-unit correlation.

    Variance formula:
        Var(β_D) = (X'X)^{-1} (Σᵢ (Σₜ xᵢₜ·eᵢₜ)²) (X'X)^{-1}

    For the simplified case of just the D coefficient:
        Var(β_D) ≈ (1/n²) × Σᵢ (Σₜ ψᵢₜ)²

    where ψᵢₜ = residual × D

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals (RIF - X·β).
    D : np.ndarray
        Treatment variable.
    unit_id : np.ndarray
        Unit identifiers.

    Returns
    -------
    float
        Clustered standard error.
    """
    n = len(residuals)
    unique_units = np.unique(unit_id)
    n_units = len(unique_units)

    # Influence function contribution for D coefficient
    # For HC3-style, weight by leverage, but for clustered we just use raw
    psi = residuals * D

    # Aggregate within clusters
    cluster_sums = np.zeros(n_units)
    for i, unit in enumerate(unique_units):
        mask = unit_id == unit
        cluster_sums[i] = np.sum(psi[mask])

    # Clustered variance
    # With finite-sample correction: (G / (G-1)) * (n-1) / (n-k)
    # Simplified: just use (1/n²) * sum of squared cluster sums
    var_clustered = np.sum(cluster_sums**2) / (n**2)

    # Small-sample correction (optional)
    # Adjust for number of clusters
    correction = n_units / (n_units - 1) if n_units > 1 else 1.0
    var_clustered *= correction

    return np.sqrt(var_clustered)


def _validate_panel_qte_inputs(
    panel: PanelData,
    quantile: float,
) -> None:
    """Validate inputs for panel QTE estimation.

    Parameters
    ----------
    panel : PanelData
        Panel data structure.
    quantile : float
        Target quantile.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if quantile <= 0 or quantile >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid quantile.\n"
            f"Function: _validate_panel_qte_inputs\n"
            f"Expected: quantile in (0, 1), Got: {quantile}"
        )

    # Warn for extreme quantiles
    if quantile < 0.05 or quantile > 0.95:
        n_obs = panel.n_obs
        n_tail = int(n_obs * min(quantile, 1 - quantile))
        if n_tail < 20:
            warnings.warn(
                f"Extreme quantile τ={quantile:.2f} with only ~{n_tail} "
                f"observations in the tail. Estimates may be unstable.",
                UserWarning,
            )

    # Warn for small clusters
    obs_per_unit = []
    for unit in panel.get_unique_units():
        obs_per_unit.append(len(panel.get_unit_indices(unit)))
    median_t = np.median(obs_per_unit)
    if median_t < 5:
        warnings.warn(
            f"Small clusters detected (median Tᵢ = {median_t:.0f}). "
            f"Clustered standard errors may be inflated.",
            UserWarning,
        )


def panel_rif_qte(
    panel: PanelData,
    quantile: float = 0.5,
    alpha: float = 0.05,
    include_covariates: bool = True,
) -> PanelQTEResult:
    """Estimate panel quantile treatment effect via RIF regression.

    Uses the Recentered Influence Function (Firpo et al. 2009) with
    Mundlak projection to control for unobserved unit effects.

    Parameters
    ----------
    panel : PanelData
        Panel data structure with outcomes, treatment, covariates, unit_id, time.
    quantile : float, default=0.5
        The quantile τ ∈ (0, 1) at which to estimate the effect.
    alpha : float, default=0.05
        Significance level for confidence interval.
    include_covariates : bool, default=True
        If True, include covariates and their unit means (Mundlak projection).
        If False, only regress RIF on treatment.

    Returns
    -------
    PanelQTEResult
        Result containing QTE estimate, SE, CI, and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.panel import PanelData, panel_rif_qte
    >>> # Create panel data
    >>> n_units, n_periods = 50, 10
    >>> n_obs = n_units * n_periods
    >>> unit_id = np.repeat(np.arange(n_units), n_periods)
    >>> time = np.tile(np.arange(n_periods), n_units)
    >>> X = np.random.randn(n_obs, 2)
    >>> D = np.random.binomial(1, 0.5, n_obs)
    >>> Y = 1.0 + X[:, 0] + 2.0 * D + np.random.randn(n_obs)
    >>> panel = PanelData(Y, D, X, unit_id, time)
    >>> result = panel_rif_qte(panel, quantile=0.5)
    >>> print(f"Median QTE: {result.qte:.3f} ± {result.qte_se:.3f}")

    Notes
    -----
    The RIF-OLS approach transforms outcomes using:

        RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)

    Then regresses RIF on [1, X, X̄, D] where X̄ are unit means (Mundlak).

    The coefficient on D estimates the unconditional QTE - the effect
    of treatment on the τ-th quantile of the marginal distribution.

    Standard errors are clustered at the unit level.
    """
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    _validate_panel_qte_inputs(panel, quantile)

    Y = panel.outcomes
    D = panel.treatment
    X = panel.covariates
    unit_id = panel.unit_id

    n_obs = panel.n_obs
    n_units = panel.n_units

    # ========================================================================
    # MUNDLAK PROJECTION
    # ========================================================================

    if include_covariates:
        X_bar = panel.compute_unit_means()  # n_obs × p
        X_augmented = np.hstack([X, X_bar])  # n_obs × 2p
    else:
        X_augmented = None

    # ========================================================================
    # COMPUTE RIF
    # ========================================================================

    # Pooled quantile
    q_tau = np.quantile(Y, quantile)

    # Kernel density at quantile
    h = _silverman_bandwidth(Y)
    f_q = _kernel_density_at_quantile(Y, q_tau, h)

    # RIF transformation
    rif = _compute_rif(Y, quantile, q_tau, f_q)

    # Warn if density is very sparse
    if f_q < 0.01:
        warnings.warn(
            f"Sparse density at quantile (f̂(q_{quantile:.2f}) = {f_q:.4f}). "
            f"RIF estimates may be unstable.",
            UserWarning,
        )

    # ========================================================================
    # OLS REGRESSION: RIF ~ [1, X, X̄, D]
    # ========================================================================

    # Build design matrix
    if include_covariates:
        Z = np.column_stack([np.ones(n_obs), X_augmented, D])
    else:
        Z = np.column_stack([np.ones(n_obs), D])

    # OLS via normal equations (stable for moderate dimensions)
    try:
        ZtZ = Z.T @ Z
        ZtRIF = Z.T @ rif
        beta = np.linalg.solve(ZtZ, ZtRIF)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        beta = np.linalg.lstsq(Z, rif, rcond=None)[0]

    # QTE is the coefficient on D (last column)
    qte = beta[-1]

    # ========================================================================
    # CLUSTERED STANDARD ERROR
    # ========================================================================

    # OLS residuals
    residuals = rif - Z @ beta

    # Clustered SE at unit level
    se = _panel_clustered_se(residuals, D, unit_id)

    # Confidence interval
    z_crit = 1.96  # For alpha=0.05 (could generalize with scipy.stats.norm.ppf)
    from scipy import stats

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = qte - z_crit * se
    ci_upper = qte + z_crit * se

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return PanelQTEResult(
        qte=float(qte),
        qte_se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        quantile=quantile,
        n_obs=n_obs,
        n_units=n_units,
        outcome_quantile=float(q_tau),
        density_at_quantile=float(f_q),
        bandwidth=float(h),
        method="panel_rif_qte",
    )


def panel_rif_qte_band(
    panel: PanelData,
    quantiles: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    include_covariates: bool = True,
) -> PanelQTEBandResult:
    """Estimate panel QTE across multiple quantiles.

    Estimates the quantile treatment effect at each specified quantile
    using RIF regression with Mundlak projection.

    Parameters
    ----------
    panel : PanelData
        Panel data structure.
    quantiles : sequence of float, optional
        Quantiles to estimate. Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    alpha : float, default=0.05
        Significance level for confidence intervals.
    include_covariates : bool, default=True
        Include covariates and their unit means.

    Returns
    -------
    PanelQTEBandResult
        Arrays of estimates across quantiles.

    Examples
    --------
    >>> result = panel_rif_qte_band(panel, quantiles=[0.1, 0.5, 0.9])
    >>> for q, qte, se in zip(result.quantiles, result.qtes, result.qte_ses):
    ...     print(f"τ={q:.1f}: QTE={qte:.3f} ± {se:.3f}")

    Notes
    -----
    A common quantile band is [0.1, 0.25, 0.5, 0.75, 0.9], which
    characterizes treatment effects across the outcome distribution.

    Heterogeneous effects across quantiles indicate the treatment
    affects different parts of the distribution differently.
    """
    # Default quantiles
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    quantiles_arr = np.asarray(quantiles)
    n_quantiles = len(quantiles_arr)

    # Validate
    if np.any(quantiles_arr <= 0) or np.any(quantiles_arr >= 1):
        raise ValueError(
            f"CRITICAL ERROR: Quantiles must be in (0, 1).\n"
            f"Function: panel_rif_qte_band\n"
            f"Got: {quantiles}"
        )

    # Estimate at each quantile
    qtes = np.zeros(n_quantiles)
    qte_ses = np.zeros(n_quantiles)
    ci_lowers = np.zeros(n_quantiles)
    ci_uppers = np.zeros(n_quantiles)

    for i, q in enumerate(quantiles_arr):
        result = panel_rif_qte(
            panel, quantile=q, alpha=alpha, include_covariates=include_covariates
        )
        qtes[i] = result.qte
        qte_ses[i] = result.qte_se
        ci_lowers[i] = result.ci_lower
        ci_uppers[i] = result.ci_upper

    return PanelQTEBandResult(
        quantiles=quantiles_arr,
        qtes=qtes,
        qte_ses=qte_ses,
        ci_lowers=ci_lowers,
        ci_uppers=ci_uppers,
        n_obs=panel.n_obs,
        n_units=panel.n_units,
        method="panel_rif_qte_band",
    )


def panel_unconditional_qte(
    panel: PanelData,
    quantile: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    cluster_bootstrap: bool = True,
    random_state: Optional[int] = None,
) -> PanelQTEResult:
    """Estimate simple unconditional QTE for panel data.

    Computes the difference in quantiles between treated and control groups:
        QTE(τ) = Q_τ(Y | D=1) - Q_τ(Y | D=0)

    This is a baseline estimator that doesn't control for covariates
    but uses cluster bootstrap for valid inference.

    Parameters
    ----------
    panel : PanelData
        Panel data structure.
    quantile : float, default=0.5
        The quantile τ ∈ (0, 1).
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    alpha : float, default=0.05
        Significance level.
    cluster_bootstrap : bool, default=True
        If True, resample clusters (units) rather than observations.
        This accounts for within-unit correlation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PanelQTEResult
        Result containing QTE estimate, SE, CI.

    Notes
    -----
    This estimator does NOT control for confounding via covariates
    or unit effects. It is provided as a baseline comparison.

    For panel data with confounding, use panel_rif_qte which includes
    Mundlak projection and covariate adjustment.
    """
    Y = panel.outcomes
    D = panel.treatment
    unit_id = panel.unit_id

    n_obs = panel.n_obs
    n_units = panel.n_units

    # Validate
    _validate_panel_qte_inputs(panel, quantile)

    # ========================================================================
    # POINT ESTIMATE
    # ========================================================================

    y_treated = Y[D == 1]
    y_control = Y[D == 0]

    if len(y_treated) < 2 or len(y_control) < 2:
        raise ValueError(
            f"CRITICAL ERROR: Insufficient observations.\n"
            f"Function: panel_unconditional_qte\n"
            f"Treated: {len(y_treated)}, Control: {len(y_control)}"
        )

    q_treated = np.quantile(y_treated, quantile)
    q_control = np.quantile(y_control, quantile)
    qte = q_treated - q_control

    # Use q_tau as the pooled quantile for consistency
    q_tau = np.quantile(Y, quantile)
    h = _silverman_bandwidth(Y)
    f_q = _kernel_density_at_quantile(Y, q_tau, h)

    # ========================================================================
    # CLUSTER BOOTSTRAP FOR SE
    # ========================================================================

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    unique_units = np.unique(unit_id)

    for b in range(n_bootstrap):
        if cluster_bootstrap:
            # Resample clusters
            boot_units = rng.choice(unique_units, len(unique_units), replace=True)

            # Gather observations from resampled clusters
            boot_idx = []
            for unit in boot_units:
                unit_idx = np.where(unit_id == unit)[0]
                boot_idx.extend(unit_idx)
            boot_idx = np.array(boot_idx)
        else:
            # Simple observation-level bootstrap (ignores clustering)
            boot_idx = rng.choice(n_obs, n_obs, replace=True)

        Y_boot = Y[boot_idx]
        D_boot = D[boot_idx]

        y_t_boot = Y_boot[D_boot == 1]
        y_c_boot = Y_boot[D_boot == 0]

        if len(y_t_boot) > 0 and len(y_c_boot) > 0:
            q_t = np.quantile(y_t_boot, quantile)
            q_c = np.quantile(y_c_boot, quantile)
            bootstrap_estimates[b] = q_t - q_c
        else:
            bootstrap_estimates[b] = np.nan

    # Remove NaN bootstrap samples
    valid_bootstrap = bootstrap_estimates[~np.isnan(bootstrap_estimates)]

    if len(valid_bootstrap) < 10:
        raise ValueError(
            f"CRITICAL ERROR: Too few valid bootstrap samples.\n"
            f"Function: panel_unconditional_qte\n"
            f"Valid: {len(valid_bootstrap)}"
        )

    se = np.std(valid_bootstrap, ddof=1)
    ci_lower = np.percentile(valid_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(valid_bootstrap, 100 * (1 - alpha / 2))

    return PanelQTEResult(
        qte=float(qte),
        qte_se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        quantile=quantile,
        n_obs=n_obs,
        n_units=n_units,
        outcome_quantile=float(q_tau),
        density_at_quantile=float(f_q),
        bandwidth=float(h),
        method="panel_unconditional_qte",
    )
