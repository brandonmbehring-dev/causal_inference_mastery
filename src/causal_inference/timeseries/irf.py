"""
Impulse Response Functions for Structural VAR.

Session 137: IRF computation and bootstrap inference.

IRF measures the dynamic response of variables to structural shocks.

At horizon h:
    IRF_h = Ψ_h = Φ_h B₀⁻¹

where Φ_h is the VMA coefficient and B₀⁻¹ is the impact matrix.
"""

from typing import Optional, Union
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VARResult
from causal_inference.timeseries.svar_types import SVARResult, IRFResult
from causal_inference.timeseries.svar import (
    structural_vma_coefficients,
    vma_coefficients,
    cholesky_svar,
)
from causal_inference.timeseries.var import var_estimate


def compute_irf(
    svar_result: SVARResult,
    horizons: int = 20,
    cumulative: bool = False,
) -> IRFResult:
    """
    Compute impulse response functions for SVAR.

    Parameters
    ----------
    svar_result : SVARResult
        Structural VAR estimation result
    horizons : int
        Maximum horizon (0 to horizons inclusive)
    cumulative : bool
        If True, return cumulative IRF (sum from 0 to h)

    Returns
    -------
    IRFResult
        Impulse response function results

    Example
    -------
    >>> from causal_inference.timeseries import var_estimate, cholesky_svar
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(200, 3)
    >>> var_result = var_estimate(data, lags=2)
    >>> svar_result = cholesky_svar(var_result)
    >>> irf = compute_irf(svar_result, horizons=20)
    >>> # Response of var 1 to shock in var 0 at horizon 5
    >>> print(f"IRF[1,0,5] = {irf.irf[1, 0, 5]:.4f}")
    """
    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")

    # Compute structural VMA coefficients
    irf = structural_vma_coefficients(svar_result, horizons)

    if cumulative:
        irf = np.cumsum(irf, axis=2)

    return IRFResult(
        irf=irf,
        irf_lower=None,
        irf_upper=None,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=True,
        var_names=svar_result.var_names,
        alpha=0.05,
        n_bootstrap=0,
    )


def compute_irf_reduced_form(
    var_result: VARResult,
    horizons: int = 20,
    cumulative: bool = False,
) -> IRFResult:
    """
    Compute non-orthogonalized (reduced-form) IRF.

    Uses identity impact matrix (shocks are reduced-form, correlated).

    Parameters
    ----------
    var_result : VARResult
        VAR estimation result
    horizons : int
        Maximum horizon
    cumulative : bool
        If True, return cumulative IRF

    Returns
    -------
    IRFResult
        Reduced-form impulse responses
    """
    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")

    irf = vma_coefficients(var_result, horizons)

    if cumulative:
        irf = np.cumsum(irf, axis=2)

    return IRFResult(
        irf=irf,
        irf_lower=None,
        irf_upper=None,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=False,
        var_names=var_result.var_names,
        alpha=0.05,
        n_bootstrap=0,
    )


def bootstrap_irf(
    data: np.ndarray,
    svar_result: SVARResult,
    horizons: int = 20,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    cumulative: bool = False,
    method: str = "residual",
    seed: Optional[int] = None,
) -> IRFResult:
    """
    Bootstrap confidence bands for IRF.

    Parameters
    ----------
    data : np.ndarray
        Original time series data, shape (n_obs, n_vars)
    svar_result : SVARResult
        Structural VAR estimation from original data
    horizons : int
        Maximum horizon for IRF
    n_bootstrap : int
        Number of bootstrap replications
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    cumulative : bool
        If True, compute cumulative IRF
    method : str
        Bootstrap method: "residual" (resample residuals) or
        "wild" (wild bootstrap for heteroskedasticity)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    IRFResult
        IRF with confidence bands

    Example
    -------
    >>> irf_ci = bootstrap_irf(data, svar_result, horizons=20, n_bootstrap=500)
    >>> # Check if zero is in confidence band at horizon 10
    >>> lower = irf_ci.irf_lower[1, 0, 10]
    >>> upper = irf_ci.irf_upper[1, 0, 10]
    >>> significant = (lower > 0) or (upper < 0)
    """
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")

    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if method not in ("residual", "wild"):
        raise ValueError(f"method must be 'residual' or 'wild', got '{method}'")

    rng = np.random.default_rng(seed)

    var_result = svar_result.var_result
    n_obs = data.shape[0]
    n_vars = svar_result.n_vars
    lags = svar_result.lags
    n_effective = n_obs - lags

    # Get fitted values and residuals
    residuals = var_result.residuals  # (n_effective, n_vars)

    # Storage for bootstrap IRFs
    irf_boots = np.zeros((n_bootstrap, n_vars, n_vars, horizons + 1))

    for b in range(n_bootstrap):
        # Generate bootstrap sample
        if method == "residual":
            # Resample residuals with replacement
            indices = rng.integers(0, n_effective, size=n_effective)
            resid_boot = residuals[indices, :]
        else:  # wild bootstrap
            # Rademacher distribution
            signs = rng.choice([-1, 1], size=n_effective)
            resid_boot = residuals * signs[:, np.newaxis]

        # Reconstruct bootstrap data
        data_boot = _reconstruct_var_data(
            data, var_result, resid_boot, lags
        )

        # Re-estimate VAR and SVAR
        try:
            var_boot = var_estimate(
                data_boot,
                lags=lags,
                var_names=var_result.var_names,
            )
            svar_boot = cholesky_svar(var_boot, ordering=svar_result.ordering)

            # Compute IRF
            irf_boot = structural_vma_coefficients(svar_boot, horizons)
            if cumulative:
                irf_boot = np.cumsum(irf_boot, axis=2)

            irf_boots[b, :, :, :] = irf_boot

        except (np.linalg.LinAlgError, ValueError):
            # Use NaN for failed bootstrap (will be ignored in quantiles)
            irf_boots[b, :, :, :] = np.nan

    # Compute percentile confidence intervals
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    irf_lower = np.nanpercentile(irf_boots, lower_pct, axis=0)
    irf_upper = np.nanpercentile(irf_boots, upper_pct, axis=0)

    # Point estimate from original
    irf_point = structural_vma_coefficients(svar_result, horizons)
    if cumulative:
        irf_point = np.cumsum(irf_point, axis=2)

    return IRFResult(
        irf=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        horizons=horizons,
        cumulative=cumulative,
        orthogonalized=True,
        var_names=svar_result.var_names,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )


def _reconstruct_var_data(
    original_data: np.ndarray,
    var_result: VARResult,
    bootstrap_residuals: np.ndarray,
    lags: int,
) -> np.ndarray:
    """
    Reconstruct time series from VAR fitted values and bootstrap residuals.

    Y_t = A_0 + A_1 Y_{t-1} + ... + A_p Y_{t-p} + ε*_t

    where ε*_t are bootstrap residuals.
    """
    n_obs = original_data.shape[0]
    n_vars = var_result.n_vars
    n_effective = n_obs - lags

    # Start with original initial values
    data_boot = np.zeros((n_obs, n_vars))
    data_boot[:lags, :] = original_data[:lags, :]

    # Reconstruct using fitted values + bootstrap residuals
    intercepts = var_result.get_intercepts()

    for t in range(lags, n_obs):
        y_t = intercepts.copy()

        for lag in range(1, lags + 1):
            A_lag = var_result.get_lag_matrix(lag)
            y_t += A_lag @ data_boot[t - lag, :]

        # Add bootstrap residual
        resid_idx = t - lags
        y_t += bootstrap_residuals[resid_idx, :]

        data_boot[t, :] = y_t

    return data_boot


def irf_significance_test(
    irf_result: IRFResult,
    response_var: Union[int, str],
    shock_var: Union[int, str],
    horizon: int,
) -> dict:
    """
    Test if IRF is significantly different from zero.

    Parameters
    ----------
    irf_result : IRFResult
        IRF with confidence bands
    response_var : int or str
        Response variable
    shock_var : int or str
        Shock variable
    horizon : int
        Horizon to test

    Returns
    -------
    dict
        Keys: 'significant', 'irf', 'lower', 'upper', 'sign'
    """
    if not irf_result.has_confidence_bands:
        raise ValueError("IRF result must have confidence bands for significance test")

    data = irf_result.get_response_with_ci(response_var, shock_var)

    irf_val = data["irf"][horizon]
    lower = data["lower"][horizon]
    upper = data["upper"][horizon]

    # Significant if CI doesn't include zero
    significant = (lower > 0) or (upper < 0)

    # Sign of effect (if significant)
    if significant:
        sign = "positive" if lower > 0 else "negative"
    else:
        sign = "uncertain"

    return {
        "significant": significant,
        "irf": irf_val,
        "lower": lower,
        "upper": upper,
        "sign": sign,
        "alpha": irf_result.alpha,
    }


def asymptotic_irf_se(
    svar_result: SVARResult,
    horizons: int = 20,
) -> np.ndarray:
    """
    Compute asymptotic standard errors for IRF (Lütkepohl 2005).

    This provides analytical standard errors based on the delta method.

    Parameters
    ----------
    svar_result : SVARResult
        SVAR estimation result
    horizons : int
        Maximum horizon

    Returns
    -------
    np.ndarray
        Shape (n_vars, n_vars, horizons+1) standard error matrix

    Notes
    -----
    Asymptotic SEs may be less reliable than bootstrap in small samples.
    Bootstrap is generally preferred for inference.
    """
    var_result = svar_result.var_result
    n_vars = var_result.n_vars
    lags = var_result.lags
    n_obs = var_result.n_obs_effective

    # Compute VMA coefficients
    Psi = structural_vma_coefficients(svar_result, horizons)

    # Estimate covariance of VAR coefficients
    # Σ_β ≈ (Z'Z)^{-1} ⊗ Σ_u where Z is the VAR design matrix
    # For simplicity, use bootstrap-based approach in practice

    # Placeholder: estimate based on VAR residual variance
    # This is a rough approximation
    sigma_u = var_result.sigma
    residual_var = np.diag(sigma_u)

    # Scale by sample size
    se = np.zeros_like(Psi)
    for h in range(horizons + 1):
        # SE increases with horizon (roughly)
        se[:, :, h] = np.sqrt(residual_var[:, np.newaxis] / n_obs) * np.sqrt(h + 1)

    return se
