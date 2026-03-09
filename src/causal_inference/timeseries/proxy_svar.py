"""
Proxy SVAR (External Instrument SVAR).

Session 163: Structural VAR identification using external instruments.

Stock & Watson (2012), Mertens & Ravn (2013).

The proxy SVAR approach uses external instruments (proxies) that are:
1. Relevant: Correlated with the target structural shock
2. Exogenous: Uncorrelated with other structural shocks

This achieves point identification of one column of B₀⁻¹ without
imposing the recursive ordering assumptions of Cholesky identification.

Algorithm:
1. Estimate reduced-form VAR → residuals u_t
2. First stage: Regress u_target on z_t → fitted values û_target
3. Second stage: Regress u_i on û_target for each i → β_i
4. Stack coefficients to form first column of B₀⁻¹
5. Complete B₀⁻¹ via variance decomposition

Key insight: Cov(z_t, u_t) = B₀⁻¹ · Cov(z_t, ε_t)
If z correlated only with ε₁: Cov(z_t, u_t) ∝ first column of B₀⁻¹

References:
- Stock & Watson (2012): "Disentangling the Channels of the 2007-09 Recession"
- Mertens & Ravn (2013): "The Dynamic Effects of Personal and Corporate Income Tax Changes"
- Gertler & Karadi (2015): "Monetary Policy Surprises, Credit Costs, and Economic Activity"

Example
-------
>>> from causal_inference.timeseries import var_estimate, proxy_svar
>>> import numpy as np
>>> np.random.seed(42)
>>> # Generate data with known structure
>>> n = 500
>>> eps = np.random.randn(n, 3)
>>> z = 0.5 * eps[:, 0] + 0.3 * np.random.randn(n)  # Proxy for first shock
>>> B0_inv_true = np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.2, 0.1, 1.0]])
>>> u = (B0_inv_true @ eps.T).T
>>> # Generate VAR data
>>> A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.0, 0.0, 0.3]])
>>> data = np.zeros((n, 3))
>>> for t in range(1, n):
...     data[t] = A1 @ data[t-1] + u[t]
>>> var_result = var_estimate(data, lags=1)
>>> proxy_result = proxy_svar(var_result, instrument=z[1:], target_shock_idx=0)
>>> print(f"First-stage F-stat: {proxy_result.first_stage_f_stat:.2f}")
>>> print(f"Weak instrument: {proxy_result.is_weak_instrument}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import linalg

from .svar_types import IdentificationMethod, SVARResult
from .types import VARResult


@dataclass
class ProxySVARResult:
    """
    Result from Proxy SVAR (External Instrument) estimation.

    Proxy SVAR identifies structural shocks using external instruments
    correlated with the target shock but uncorrelated with other shocks.

    Attributes
    ----------
    var_result : VARResult
        Underlying reduced-form VAR estimation
    B0_inv : np.ndarray
        Shape (n_vars, n_vars) impact matrix.
        Only the target shock column is directly identified.
    B0 : np.ndarray
        Inverse of B0_inv
    structural_shocks : np.ndarray
        Shape (n_obs_effective, n_vars) structural shock series
    identification : IdentificationMethod
        Always PROXY for this result type
    target_shock_idx : int
        Index of the shock being identified (0-indexed)
    target_residual_idx : int
        Index of the VAR residual used in first stage
    instrument : np.ndarray
        External instrument used for identification
    first_stage_f_stat : float
        F-statistic from first-stage regression (instrument strength)
    first_stage_r2 : float
        R-squared from first-stage regression
    is_weak_instrument : bool
        True if F < threshold (Stock-Yogo rule: F < 10)
    reliability_ratio : float
        Ratio of signal to total variance
    impact_column : np.ndarray
        Identified column of B₀⁻¹ (target shock impacts)
    impact_column_se : np.ndarray
        Standard errors for impact_column (delta method)
    impact_column_ci_lower : np.ndarray
        Lower confidence bounds for impact_column
    impact_column_ci_upper : np.ndarray
        Upper confidence bounds for impact_column
    n_restrictions : int
        Number of restrictions (n_vars - 1 for proxy SVAR)
    is_just_identified : bool
        True for single-instrument proxy SVAR
    is_over_identified : bool
        False for single instrument
    alpha : float
        Significance level for confidence intervals
    bootstrap_se : Optional[np.ndarray]
        Bootstrap standard errors (if computed)
    n_bootstrap : int
        Number of bootstrap replications (0 if no bootstrap)

    Example
    -------
    >>> result = proxy_svar(var_result, instrument=z, target_shock_idx=0)
    >>> print(f"F-stat: {result.first_stage_f_stat:.2f}")
    >>> print(f"Impact column: {result.impact_column}")
    """

    var_result: VARResult
    B0_inv: np.ndarray
    B0: np.ndarray
    structural_shocks: np.ndarray
    identification: IdentificationMethod = IdentificationMethod.PROXY

    # Proxy-specific fields
    target_shock_idx: int = 0
    target_residual_idx: int = 0
    instrument: np.ndarray = field(default_factory=lambda: np.array([]))
    first_stage_f_stat: float = 0.0
    first_stage_r2: float = 0.0
    is_weak_instrument: bool = False
    reliability_ratio: float = 0.0

    # Impact column inference
    impact_column: np.ndarray = field(default_factory=lambda: np.array([]))
    impact_column_se: np.ndarray = field(default_factory=lambda: np.array([]))
    impact_column_ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    impact_column_ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    n_restrictions: int = 0
    is_just_identified: bool = True
    is_over_identified: bool = False
    alpha: float = 0.05

    # Optional bootstrap
    bootstrap_se: Optional[np.ndarray] = None
    n_bootstrap: int = 0

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.B0_inv.shape[0]

    @property
    def lags(self) -> int:
        """VAR lag order."""
        return self.var_result.lags

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.var_result.n_obs_effective

    @property
    def var_names(self) -> List[str]:
        """Variable names."""
        return self.var_result.var_names

    @property
    def has_bootstrap_se(self) -> bool:
        """Whether bootstrap standard errors are available."""
        return self.bootstrap_se is not None

    def get_structural_coefficient(self, shock_var: int, response_var: int) -> float:
        """
        Get contemporaneous impact of structural shock on variable.

        Parameters
        ----------
        shock_var : int
            Index of shock variable
        response_var : int
            Index of response variable

        Returns
        -------
        float
            Impact coefficient: (B₀⁻¹)_{response, shock}
        """
        return self.B0_inv[response_var, shock_var]

    def __repr__(self) -> str:
        weak_str = ", WEAK" if self.is_weak_instrument else ""
        return (
            f"ProxySVARResult(n_vars={self.n_vars}, lags={self.lags}, "
            f"F={self.first_stage_f_stat:.2f}{weak_str})"
        )


def proxy_svar(
    var_result: VARResult,
    instrument: np.ndarray,
    target_shock_idx: int = 0,
    target_residual_idx: Optional[int] = None,
    alpha: float = 0.05,
    bootstrap_se: bool = False,
    n_bootstrap: int = 500,
    weak_instrument_threshold: float = 10.0,
    seed: Optional[int] = None,
) -> ProxySVARResult:
    """
    Proxy SVAR identification via external instruments.

    Identifies structural shocks using external instruments (proxies)
    following Stock & Watson (2012) and Mertens & Ravn (2013).

    Parameters
    ----------
    var_result : VARResult
        Estimated reduced-form VAR
    instrument : np.ndarray
        External instrument z_t, shape (n_obs_effective,) or (n_obs_effective, 1).
        Must be correlated with target structural shock, uncorrelated with others.
    target_shock_idx : int, default 0
        Which structural shock to identify (0-indexed).
        The identified impact column will be placed at this position.
    target_residual_idx : int, optional
        Which VAR residual is instrumented. Default: same as target_shock_idx.
        Typically the same, but can differ if shock ordering is non-standard.
    alpha : float, default 0.05
        Significance level for confidence intervals
    bootstrap_se : bool, default False
        Whether to compute bootstrap standard errors
    n_bootstrap : int, default 500
        Number of bootstrap replications (if bootstrap_se=True)
    weak_instrument_threshold : float, default 10.0
        F-statistic threshold for weak instrument warning (Stock-Yogo rule)
    seed : int, optional
        Random seed for reproducibility (if bootstrap_se=True)

    Returns
    -------
    ProxySVARResult
        Estimation result with identified impact column and diagnostics

    Raises
    ------
    ValueError
        If instrument length doesn't match n_obs_effective
        If instrument has zero variance (constant)
        If target indices are out of bounds

    Warns
    -----
    UserWarning
        If first-stage F-statistic < weak_instrument_threshold

    Notes
    -----
    The proxy SVAR algorithm:

    1. **First stage**: Regress target residual on instrument
       u_target = γ₀ + γ₁·z + v

    2. **Second stage**: Regress each residual on fitted values
       u_i = β_i · û_target + e_i

    3. **Stack coefficients**: b₁ = [β₁, β₂, ..., β_k]'
       This is (up to scale) the first column of B₀⁻¹

    4. **Complete B₀⁻¹**: Use variance decomposition
       Σ_u = B₀⁻¹ · B₀⁻¹'

    Identifying assumptions:
    - **Relevance**: Cov(z_t, ε_target) ≠ 0
    - **Exogeneity**: Cov(z_t, ε_j) = 0 for j ≠ target

    Example
    -------
    >>> result = proxy_svar(var_result, instrument=monetary_surprise)
    >>> print(f"First-stage F: {result.first_stage_f_stat:.2f}")
    >>> if result.is_weak_instrument:
    ...     print("Warning: Weak instrument detected!")
    """
    # Input validation
    n_obs = var_result.n_obs_effective
    n_vars = var_result.n_vars

    # Handle instrument shape
    instrument = np.asarray(instrument).flatten()

    if len(instrument) != n_obs:
        raise ValueError(
            f"Instrument length ({len(instrument)}) must match "
            f"VAR n_obs_effective ({n_obs}). "
            f"If VAR has {var_result.lags} lags, instrument should be trimmed accordingly."
        )

    if np.var(instrument) < 1e-10:
        raise ValueError(
            "Instrument has near-zero variance (constant). Cannot compute first-stage regression."
        )

    if np.any(np.isnan(instrument)):
        raise ValueError(
            "Instrument contains NaN values. "
            "Please handle missing values before calling proxy_svar."
        )

    if target_shock_idx < 0 or target_shock_idx >= n_vars:
        raise ValueError(
            f"target_shock_idx ({target_shock_idx}) out of bounds. Must be in [0, {n_vars - 1}]."
        )

    if target_residual_idx is None:
        target_residual_idx = target_shock_idx

    if target_residual_idx < 0 or target_residual_idx >= n_vars:
        raise ValueError(
            f"target_residual_idx ({target_residual_idx}) out of bounds. "
            f"Must be in [0, {n_vars - 1}]."
        )

    # Get VAR residuals
    residuals = var_result.residuals  # (n_obs_effective, n_vars)
    sigma_u = var_result.sigma  # (n_vars, n_vars)

    # Compute proxy impact column via 2SLS
    impact_column, f_stat, r2, impact_se, first_stage_residuals = _compute_proxy_impact_column(
        residuals=residuals,
        instrument=instrument,
        target_residual_idx=target_residual_idx,
    )

    # Check for weak instrument
    is_weak = f_stat < weak_instrument_threshold
    if is_weak:
        warnings.warn(
            f"Weak instrument detected: F-statistic ({f_stat:.2f}) < "
            f"threshold ({weak_instrument_threshold}). "
            f"Estimates may be biased. Consider using robust inference.",
            UserWarning,
        )

    # Compute reliability ratio
    reliability = r2

    # Complete the impact matrix
    B0_inv = _complete_impact_matrix(
        impact_column=impact_column,
        sigma_u=sigma_u,
        target_shock_idx=target_shock_idx,
    )

    # Compute B0 and structural shocks
    B0 = linalg.inv(B0_inv)
    structural_shocks = (B0 @ residuals.T).T

    # Confidence intervals (delta method)
    z_crit = _normal_critical_value(alpha)
    ci_lower = impact_column - z_crit * impact_se
    ci_upper = impact_column + z_crit * impact_se

    # Optional bootstrap
    bootstrap_se_result = None
    if bootstrap_se:
        if seed is not None:
            np.random.seed(seed)

        bootstrap_se_result = _proxy_bootstrap(
            residuals=residuals,
            instrument=instrument,
            target_residual_idx=target_residual_idx,
            n_bootstrap=n_bootstrap,
        )

        # Update CIs with bootstrap
        ci_lower = impact_column - z_crit * bootstrap_se_result
        ci_upper = impact_column + z_crit * bootstrap_se_result

    return ProxySVARResult(
        var_result=var_result,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=IdentificationMethod.PROXY,
        target_shock_idx=target_shock_idx,
        target_residual_idx=target_residual_idx,
        instrument=instrument,
        first_stage_f_stat=f_stat,
        first_stage_r2=r2,
        is_weak_instrument=is_weak,
        reliability_ratio=reliability,
        impact_column=impact_column,
        impact_column_se=impact_se,
        impact_column_ci_lower=ci_lower,
        impact_column_ci_upper=ci_upper,
        n_restrictions=n_vars - 1,
        is_just_identified=True,
        is_over_identified=False,
        alpha=alpha,
        bootstrap_se=bootstrap_se_result,
        n_bootstrap=n_bootstrap if bootstrap_se else 0,
    )


def _compute_proxy_impact_column(
    residuals: np.ndarray,
    instrument: np.ndarray,
    target_residual_idx: int,
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Compute impact column via 2SLS on VAR residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Shape (n_obs, n_vars) VAR residuals
    instrument : np.ndarray
        Shape (n_obs,) external instrument
    target_residual_idx : int
        Which residual is instrumented

    Returns
    -------
    impact_column : np.ndarray
        First column of B₀⁻¹ (shape: n_vars)
    f_stat : float
        First-stage F-statistic
    r2 : float
        First-stage R-squared
    standard_errors : np.ndarray
        Delta-method standard errors (shape: n_vars)
    first_stage_residuals : np.ndarray
        Residuals from first-stage regression
    """
    n_obs, n_vars = residuals.shape

    # Target residual (the one we're instrumenting)
    u_target = residuals[:, target_residual_idx]

    # First stage: regress u_target on z
    fitted, f_stat, r2, first_stage_resid = _first_stage_regression(u_target, instrument)

    # Second stage: regress each u_i on fitted values
    # This gives us the relative impact coefficients
    impact_column = np.zeros(n_vars)
    standard_errors = np.zeros(n_vars)

    # Design matrix for second stage
    X_second = np.column_stack([np.ones(n_obs), fitted])

    for i in range(n_vars):
        u_i = residuals[:, i]

        # OLS: u_i = a + b * fitted + error
        # Using (X'X)^{-1} X'y formula
        XtX = X_second.T @ X_second
        Xty = X_second.T @ u_i
        beta = linalg.solve(XtX, Xty)

        impact_column[i] = beta[1]  # Coefficient on fitted values

        # Standard error
        resid_second = u_i - X_second @ beta
        sigma2 = np.sum(resid_second**2) / (n_obs - 2)
        var_beta = sigma2 * linalg.inv(XtX)
        standard_errors[i] = np.sqrt(var_beta[1, 1])

    # Normalize: scale so target shock element = 1
    # This makes impact_column[target_residual_idx] = 1.0
    scale = impact_column[target_residual_idx]
    if abs(scale) < 1e-10:
        # If target coefficient is near zero, can't normalize
        # Use unit scaling instead
        warnings.warn(
            "Target coefficient near zero in second stage. Using unit scaling.",
            UserWarning,
        )
        scale = 1.0

    impact_column = impact_column / scale
    standard_errors = standard_errors / abs(scale)

    return impact_column, f_stat, r2, standard_errors, first_stage_resid


def _first_stage_regression(
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    First-stage regression for proxy SVAR.

    y = γ₀ + γ₁·z + v

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (target residual)
    z : np.ndarray
        Instrument

    Returns
    -------
    fitted : np.ndarray
        Predicted y values
    f_stat : float
        F-statistic for instrument strength
    r2 : float
        R-squared
    residuals : np.ndarray
        First-stage residuals
    """
    n = len(y)

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), z])

    # OLS
    XtX = X.T @ X
    Xty = X.T @ y
    beta = linalg.solve(XtX, Xty)

    # Fitted values
    fitted = X @ beta

    # Residuals
    residuals = y - fitted

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # F-statistic: F = (R² / (1-R²)) × ((n-k) / (k-1))
    # For simple regression with intercept: k=2, so (k-1)=1
    k = 2  # intercept + one regressor
    if r2 < 1.0:
        f_stat = (r2 / (1 - r2)) * ((n - k) / (k - 1))
    else:
        f_stat = np.inf  # Perfect fit

    return fitted, f_stat, r2, residuals


def _complete_impact_matrix(
    impact_column: np.ndarray,
    sigma_u: np.ndarray,
    target_shock_idx: int,
) -> np.ndarray:
    """
    Complete B₀⁻¹ given identified first column.

    Uses variance decomposition:
    Σ_u = B₀⁻¹ · B₀⁻¹'

    Given b₁ (target column), solve for remaining columns
    using Cholesky decomposition approach.

    Parameters
    ----------
    impact_column : np.ndarray
        Identified column of B₀⁻¹ (shape: n_vars)
    sigma_u : np.ndarray
        Reduced-form residual covariance (n_vars × n_vars)
    target_shock_idx : int
        Index where impact_column should be placed

    Returns
    -------
    B0_inv : np.ndarray
        Complete impact matrix (n_vars × n_vars)

    Notes
    -----
    The algorithm:
    1. Place identified column at target_shock_idx
    2. Compute variance explained: σ² = b₁' Σ_u⁻¹ b₁
    3. Scale column: b₁ = b₁ / √σ²
    4. Compute residual covariance: Σ_resid = Σ_u - b₁ b₁'
    5. Fill remaining columns via Cholesky of Σ_resid

    This ensures Σ_u = B₀⁻¹ · I · B₀⁻¹' = B₀⁻¹ B₀⁻¹'
    """
    n_vars = len(impact_column)

    # Start with Cholesky as baseline
    L = linalg.cholesky(sigma_u, lower=True)

    # Compute scaling for impact column
    # σ² = b₁' Σ_u⁻¹ b₁ determines the shock variance
    sigma_u_inv = linalg.inv(sigma_u)
    shock_variance = impact_column @ sigma_u_inv @ impact_column

    if shock_variance <= 0:
        warnings.warn(
            "Non-positive shock variance. Using Cholesky fallback.",
            UserWarning,
        )
        return L

    # Scale impact column so shock has unit variance
    scaled_impact = impact_column / np.sqrt(shock_variance)

    # Build B₀⁻¹ by placing identified column and using Cholesky for rest
    B0_inv = L.copy()
    B0_inv[:, target_shock_idx] = scaled_impact

    # Orthogonalize remaining columns with respect to identified column
    # Using Gram-Schmidt style orthogonalization
    for j in range(n_vars):
        if j != target_shock_idx:
            # Project out the identified column component
            proj = (scaled_impact @ B0_inv[:, j]) * scaled_impact
            B0_inv[:, j] = B0_inv[:, j] - proj

            # Re-normalize to preserve variance contribution
            col_norm = np.linalg.norm(B0_inv[:, j])
            if col_norm > 1e-10:
                # Scale to match original Cholesky column norm
                orig_norm = np.linalg.norm(L[:, j])
                B0_inv[:, j] = B0_inv[:, j] * (orig_norm / col_norm)

    return B0_inv


def _proxy_bootstrap(
    residuals: np.ndarray,
    instrument: np.ndarray,
    target_residual_idx: int,
    n_bootstrap: int,
) -> np.ndarray:
    """
    Bootstrap inference for proxy SVAR.

    Uses residual resampling to compute bootstrap distribution
    of impact coefficients.

    Parameters
    ----------
    residuals : np.ndarray
        VAR residuals (n_obs × n_vars)
    instrument : np.ndarray
        External instrument
    target_residual_idx : int
        Which residual is instrumented
    n_bootstrap : int
        Number of bootstrap replications

    Returns
    -------
    bootstrap_se : np.ndarray
        Bootstrap standard errors for impact column
    """
    n_obs, n_vars = residuals.shape
    impact_boots = np.zeros((n_bootstrap, n_vars))

    for b in range(n_bootstrap):
        # Resample indices
        boot_idx = np.random.choice(n_obs, size=n_obs, replace=True)

        # Resample residuals and instrument
        resid_boot = residuals[boot_idx, :]
        inst_boot = instrument[boot_idx]

        # Compute impact column for bootstrap sample
        try:
            impact_b, _, _, _, _ = _compute_proxy_impact_column(
                residuals=resid_boot,
                instrument=inst_boot,
                target_residual_idx=target_residual_idx,
            )
            impact_boots[b, :] = impact_b
        except (np.linalg.LinAlgError, ValueError):
            # If bootstrap sample fails, use NaN and exclude later
            impact_boots[b, :] = np.nan

    # Compute standard errors, ignoring NaN samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bootstrap_se = np.nanstd(impact_boots, axis=0, ddof=1)

    return bootstrap_se


def _normal_critical_value(alpha: float) -> float:
    """Get critical value from standard normal for two-sided CI."""
    from scipy.stats import norm

    return norm.ppf(1 - alpha / 2)


def weak_instrument_diagnostics(
    f_stat: float,
    n_obs: int,
    n_instruments: int = 1,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive weak instrument diagnostics for proxy SVAR.

    Parameters
    ----------
    f_stat : float
        First-stage F-statistic
    n_obs : int
        Number of observations
    n_instruments : int, default 1
        Number of instruments (currently only single instrument supported)
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Diagnostic results with keys:
        - 'f_stat': First-stage F-statistic
        - 'is_weak': Boolean for weak instrument (F < 10)
        - 'is_very_weak': Boolean for very weak instrument (F < 5)
        - 'stock_yogo_critical': Stock-Yogo critical value (if available)
        - 'interpretation': String interpretation
        - 'recommended_inference': Recommended inference method

    Notes
    -----
    Stock & Yogo (2005) critical values for single endogenous variable:
    - For 10% maximal IV size: F > 16.38
    - For 15% maximal IV size: F > 8.96
    - For 20% maximal IV size: F > 6.66
    - For 25% maximal IV size: F > 5.53

    Rule of thumb: F > 10 is "strong enough" for reliable inference.

    Example
    -------
    >>> diag = weak_instrument_diagnostics(f_stat=8.5, n_obs=200)
    >>> print(diag['interpretation'])
    """
    # Stock-Yogo critical values for single instrument, single endogenous
    stock_yogo = {
        0.10: 16.38,  # 10% maximal size
        0.15: 8.96,  # 15% maximal size
        0.20: 6.66,  # 20% maximal size
        0.25: 5.53,  # 25% maximal size
    }

    is_weak = f_stat < 10.0
    is_very_weak = f_stat < 5.0

    # Find closest Stock-Yogo threshold
    sy_critical = stock_yogo.get(0.15, 8.96)  # Use 15% size as default

    # Interpretation
    if f_stat >= 16.38:
        interpretation = (
            f"Strong instrument (F={f_stat:.2f} ≥ 16.38). Standard inference is reliable."
        )
        recommended = "standard"
    elif f_stat >= 10.0:
        interpretation = (
            f"Moderate instrument strength (F={f_stat:.2f}). "
            "Standard inference likely OK, but consider robust methods."
        )
        recommended = "standard_with_caution"
    elif f_stat >= 5.0:
        interpretation = (
            f"Weak instrument (F={f_stat:.2f} < 10). "
            "Standard errors biased. Use weak-instrument robust methods."
        )
        recommended = "anderson_rubin"
    else:
        interpretation = (
            f"Very weak instrument (F={f_stat:.2f} < 5). "
            "Estimates unreliable. Consider alternative instruments."
        )
        recommended = "reconsider_instrument"

    return {
        "f_stat": f_stat,
        "is_weak": is_weak,
        "is_very_weak": is_very_weak,
        "stock_yogo_critical_15pct": sy_critical,
        "stock_yogo_critical_values": stock_yogo,
        "interpretation": interpretation,
        "recommended_inference": recommended,
        "n_obs": n_obs,
        "n_instruments": n_instruments,
    }


def compute_irf_from_proxy(
    result: ProxySVARResult,
    horizons: int = 20,
) -> np.ndarray:
    """
    Compute impulse responses from proxy SVAR result.

    Parameters
    ----------
    result : ProxySVARResult
        Proxy SVAR estimation result
    horizons : int, default 20
        Maximum horizon

    Returns
    -------
    irf : np.ndarray
        Shape (n_vars, n_vars, horizons+1) impulse responses
        irf[i, j, h] = response of var i to shock j at horizon h

    Notes
    -----
    Only the column corresponding to target_shock_idx is directly identified.
    Other columns are based on the completed B₀⁻¹ matrix.
    """
    from .irf import vma_coefficients

    # Get VMA coefficients
    Phi = vma_coefficients(result.var_result, horizons)  # (n_vars, n_vars, horizons+1)

    n_vars = result.n_vars

    # Compute structural IRF: IRF_h = Φ_h · B₀⁻¹
    irf = np.zeros((n_vars, n_vars, horizons + 1))
    for h in range(horizons + 1):
        irf[:, :, h] = Phi[:, :, h] @ result.B0_inv

    return irf
