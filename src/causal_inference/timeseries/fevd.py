"""
Forecast Error Variance Decomposition for Structural VAR.

Session 137: FEVD measures the proportion of forecast error variance
attributable to each structural shock.

At horizon h, the FEVD for variable i due to shock j:
    FEVD_{i,j,h} = Σ_{k=0}^{h} (Ψ_{i,j,k})² / MSE_{i,h}

where MSE_{i,h} = Σ_{j} Σ_{k=0}^{h} (Ψ_{i,j,k})²

FEVD rows sum to 1: Σ_j FEVD_{i,j,h} = 1 for all i, h.
"""

from typing import Optional, Union
import numpy as np

from causal_inference.timeseries.svar_types import (
    SVARResult,
    FEVDResult,
    HistoricalDecompositionResult,
)
from causal_inference.timeseries.svar import structural_vma_coefficients


def compute_fevd(
    svar_result: SVARResult,
    horizons: int = 20,
) -> FEVDResult:
    """
    Compute Forecast Error Variance Decomposition.

    FEVD measures what proportion of variable i's h-step ahead forecast
    error variance is due to structural shock j.

    Parameters
    ----------
    svar_result : SVARResult
        Structural VAR estimation result
    horizons : int
        Maximum horizon (0 to horizons inclusive)

    Returns
    -------
    FEVDResult
        Variance decomposition results

    Example
    -------
    >>> fevd = compute_fevd(svar_result, horizons=20)
    >>> # Proportion of var 1's 10-step FEV due to shock 0
    >>> print(f"FEVD[1,0,10] = {fevd.fevd[1, 0, 10]:.2%}")
    >>> # Check rows sum to 1
    >>> assert np.allclose(fevd.fevd[:, :, 10].sum(axis=1), 1.0)

    Notes
    -----
    At horizon 0, FEVD equals the squared contemporaneous impact matrix
    (normalized). At long horizons, it converges to the long-run variance
    contribution of each shock.
    """
    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")

    # Get structural VMA coefficients: Ψ[i,j,h] = response of i to shock j at h
    Psi = structural_vma_coefficients(svar_result, horizons)

    n_vars = svar_result.n_vars

    # Initialize FEVD array
    fevd = np.zeros((n_vars, n_vars, horizons + 1))

    # Compute cumulative squared IRFs
    for h in range(horizons + 1):
        # Sum of squared IRFs up to horizon h
        # MSE contribution from shock j to variable i
        cumsum_squared = np.zeros((n_vars, n_vars))
        for k in range(h + 1):
            cumsum_squared += Psi[:, :, k] ** 2

        # Total MSE for each variable (sum across shocks)
        total_mse = cumsum_squared.sum(axis=1)  # (n_vars,)

        # FEVD = cumulative contribution / total MSE
        # Handle zero MSE (shouldn't happen with valid SVAR)
        for i in range(n_vars):
            if total_mse[i] > 1e-12:
                fevd[i, :, h] = cumsum_squared[i, :] / total_mse[i]
            else:
                # Equal contribution if MSE is zero
                fevd[i, :, h] = 1.0 / n_vars

    return FEVDResult(
        fevd=fevd,
        horizons=horizons,
        var_names=svar_result.var_names,
    )


def historical_decomposition(
    svar_result: SVARResult,
    data: Optional[np.ndarray] = None,
) -> HistoricalDecompositionResult:
    """
    Compute historical decomposition.

    Decomposes each variable's realized time path into contributions
    from each structural shock.

    Y_t = baseline_t + Σ_j contribution_{j,t}

    where contribution_{j,t} = Σ_{s=0}^{t-1} Ψ_s ε_{j,t-s}

    Parameters
    ----------
    svar_result : SVARResult
        Structural VAR estimation result
    data : np.ndarray, optional
        Original data. If None, uses data from VAR estimation.

    Returns
    -------
    HistoricalDecompositionResult
        Historical decomposition results

    Example
    -------
    >>> hd = historical_decomposition(svar_result)
    >>> # Contribution of shock 0 to variable 1 over time
    >>> contrib = hd.get_shock_contribution(1, 0)
    """
    var_result = svar_result.var_result
    n_vars = svar_result.n_vars
    lags = svar_result.lags
    n_obs = var_result.n_obs_effective

    # Structural shocks
    shocks = svar_result.structural_shocks  # (n_obs, n_vars)

    # Get VMA coefficients for full sample horizon
    Psi = structural_vma_coefficients(svar_result, n_obs - 1)

    # Initialize contributions array
    contributions = np.zeros((n_vars, n_vars, n_obs))

    # Compute contributions: shock j's contribution to variable i at time t
    for t in range(n_obs):
        for s in range(min(t + 1, n_obs)):
            # Contribution at time t from shock at time t-s
            if t - s >= 0:
                contributions[:, :, t] += np.outer(
                    Psi[:, :, s].sum(axis=1), shocks[t - s, :]
                )
                # More precise: contributions[i, j, t] += Ψ[i,j,s] * ε[j,t-s]
                for i in range(n_vars):
                    for j in range(n_vars):
                        contributions[i, j, t] = 0  # Reset for precise calc

    # Recompute more precisely
    contributions = np.zeros((n_vars, n_vars, n_obs))
    for t in range(n_obs):
        for j in range(n_vars):  # Shock source
            for s in range(min(t + 1, Psi.shape[2])):
                if t - s >= 0:
                    contributions[:, j, t] += Psi[:, j, s] * shocks[t - s, j]

    # Baseline: deterministic component (fitted without shocks)
    # This would be zero for mean-zero shocks, or intercept effects
    baseline = np.zeros((n_vars, n_obs))

    # Actual values (from VAR residuals + fitted)
    if data is not None:
        actual = data[lags:, :].T  # (n_vars, n_obs)
    else:
        # Reconstruct from residuals
        actual = var_result.residuals.T  # Simplified; ideally use fitted values

    return HistoricalDecompositionResult(
        contributions=contributions,
        baseline=baseline,
        actual=actual,
        var_names=svar_result.var_names,
    )


def fevd_convergence(
    svar_result: SVARResult,
    max_horizon: int = 100,
    tol: float = 1e-4,
) -> dict:
    """
    Analyze FEVD convergence to long-run values.

    Parameters
    ----------
    svar_result : SVARResult
        SVAR result
    max_horizon : int
        Maximum horizon to check
    tol : float
        Convergence tolerance

    Returns
    -------
    dict
        Keys: 'converged', 'horizon_converged', 'long_run_fevd'
    """
    n_vars = svar_result.n_vars

    # Compute FEVD at increasing horizons
    fevd_prev = None
    converged = False
    horizon_converged = max_horizon

    for h in range(1, max_horizon + 1, 5):  # Check every 5 periods
        fevd_result = compute_fevd(svar_result, h)
        fevd_current = fevd_result.fevd[:, :, h]

        if fevd_prev is not None:
            max_change = np.max(np.abs(fevd_current - fevd_prev))
            if max_change < tol:
                converged = True
                horizon_converged = h
                break

        fevd_prev = fevd_current.copy()

    # Final long-run FEVD
    final_fevd = compute_fevd(svar_result, horizon_converged)

    return {
        "converged": converged,
        "horizon_converged": horizon_converged,
        "long_run_fevd": final_fevd.fevd[:, :, -1],
        "tolerance": tol,
    }


def variance_contribution_table(
    fevd_result: FEVDResult,
    horizon: int,
) -> dict:
    """
    Create a summary table of variance contributions at specific horizon.

    Parameters
    ----------
    fevd_result : FEVDResult
        FEVD result
    horizon : int
        Horizon to summarize

    Returns
    -------
    dict
        Nested dict: result[response_var][shock_var] = contribution
    """
    if horizon > fevd_result.horizons:
        raise ValueError(
            f"horizon {horizon} exceeds max {fevd_result.horizons}"
        )

    table = {}
    for i, resp_name in enumerate(fevd_result.var_names):
        table[resp_name] = {}
        for j, shock_name in enumerate(fevd_result.var_names):
            table[resp_name][shock_name] = fevd_result.fevd[i, j, horizon]

    return table
