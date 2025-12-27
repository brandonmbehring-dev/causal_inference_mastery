"""
Lag Order Selection for VAR Models.

Session 135: Information criteria for optimal lag selection.

Implements:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- HQC (Hannan-Quinn Criterion)
"""

from typing import Dict, List, Optional
import numpy as np

from causal_inference.timeseries.types import LagSelectionResult, VARResult
from causal_inference.timeseries.var import var_estimate


def select_lag_order(
    data: np.ndarray,
    max_lags: int = 10,
    criterion: str = "aic",
    var_names: Optional[List[str]] = None,
) -> LagSelectionResult:
    """
    Select optimal VAR lag order using information criteria.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    max_lags : int
        Maximum lag order to test
    criterion : str
        Primary criterion for selection: "aic", "bic", or "hqc"
    var_names : List[str], optional
        Variable names

    Returns
    -------
    LagSelectionResult
        Optimal lag and all criterion values

    Example
    -------
    >>> np.random.seed(42)
    >>> # Generate VAR(2) data
    >>> n, k = 200, 2
    >>> data = np.random.randn(n, k)
    >>> result = select_lag_order(data, max_lags=5)
    >>> print(f"Optimal lag: {result.optimal_lag}")
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if criterion not in ["aic", "bic", "hqc"]:
        raise ValueError(f"criterion must be 'aic', 'bic', or 'hqc', got '{criterion}'")

    # Ensure we have enough observations
    min_obs_needed = max_lags + 2
    if n_obs < min_obs_needed:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for max_lags={max_lags}. "
            f"Need at least {min_obs_needed}."
        )

    aic_values: Dict[int, float] = {}
    bic_values: Dict[int, float] = {}
    hqc_values: Dict[int, float] = {}
    all_lags: List[int] = []

    for lag in range(1, max_lags + 1):
        try:
            result = var_estimate(data, lags=lag, var_names=var_names)
            aic_values[lag] = result.aic
            bic_values[lag] = result.bic
            hqc_values[lag] = result.hqc
            all_lags.append(lag)
        except (ValueError, np.linalg.LinAlgError):
            # Skip lags that fail estimation
            continue

    if not all_lags:
        raise ValueError("Could not estimate VAR for any lag order")

    # Select optimal based on criterion
    if criterion == "aic":
        all_values = aic_values
    elif criterion == "bic":
        all_values = bic_values
    else:
        all_values = hqc_values

    optimal_lag = min(all_values, key=all_values.get)

    return LagSelectionResult(
        optimal_lag=optimal_lag,
        criterion=criterion,
        all_values=all_values,
        all_lags=all_lags,
        aic_values=aic_values,
        bic_values=bic_values,
        hqc_values=hqc_values,
    )


def compute_aic(
    log_likelihood: float,
    n_params: int,
    n_obs: Optional[int] = None,
) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = -2 * log_likelihood + 2 * n_params

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters
    n_obs : int, optional
        Number of observations (not used in standard AIC)

    Returns
    -------
    float
        AIC value (lower is better)

    Example
    -------
    >>> compute_aic(log_likelihood=-100, n_params=5)
    210.0
    """
    return -2 * log_likelihood + 2 * n_params


def compute_bic(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = -2 * log_likelihood + n_params * log(n_obs)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters
    n_obs : int
        Number of observations

    Returns
    -------
    float
        BIC value (lower is better)

    Example
    -------
    >>> compute_bic(log_likelihood=-100, n_params=5, n_obs=100)
    223.02585092994046
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be > 0, got {n_obs}")

    return -2 * log_likelihood + n_params * np.log(n_obs)


def compute_hqc(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
) -> float:
    """
    Compute Hannan-Quinn Criterion.

    HQC = -2 * log_likelihood + 2 * n_params * log(log(n_obs))

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters
    n_obs : int
        Number of observations

    Returns
    -------
    float
        HQC value (lower is better)

    Example
    -------
    >>> compute_hqc(log_likelihood=-100, n_params=5, n_obs=100)
    215.30256282829406
    """
    if n_obs <= 1:
        raise ValueError(f"n_obs must be > 1, got {n_obs}")

    return -2 * log_likelihood + 2 * n_params * np.log(np.log(n_obs))


def compare_lag_orders(
    data: np.ndarray,
    lags_to_compare: List[int],
    var_names: Optional[List[str]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compare multiple lag orders using all information criteria.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) time series data
    lags_to_compare : List[int]
        List of lag orders to compare
    var_names : List[str], optional
        Variable names

    Returns
    -------
    Dict[int, Dict[str, float]]
        Mapping from lag to dict of criterion values

    Example
    -------
    >>> np.random.seed(42)
    >>> data = np.random.randn(200, 2)
    >>> results = compare_lag_orders(data, [1, 2, 3, 4])
    >>> for lag, criteria in results.items():
    ...     print(f"Lag {lag}: AIC={criteria['aic']:.2f}")
    """
    data = np.asarray(data, dtype=np.float64)
    results = {}

    for lag in lags_to_compare:
        try:
            var_result = var_estimate(data, lags=lag, var_names=var_names)
            results[lag] = {
                "aic": var_result.aic,
                "bic": var_result.bic,
                "hqc": var_result.hqc,
                "log_likelihood": var_result.log_likelihood,
            }
        except (ValueError, np.linalg.LinAlgError):
            continue

    return results
