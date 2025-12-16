"""
Synthetic Control Methods - Type Definitions and Validation

Provides:
- SCMResult TypedDict for return values
- Input validation functions with fail-fast behavior
"""

from typing import Optional, TypedDict

import numpy as np


class SCMResult(TypedDict):
    """
    Result container for Synthetic Control Method estimation.

    Attributes
    ----------
    estimate : float
        Average treatment effect on treated (ATT) estimate
    se : float
        Standard error (from placebo or bootstrap)
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    p_value : float
        P-value from placebo distribution
    weights : np.ndarray
        Synthetic control weights for donor units
    pre_rmse : float
        Root mean squared error of pre-treatment fit
    pre_r_squared : float
        R-squared of pre-treatment fit
    n_treated : int
        Number of treated units
    n_control : int
        Number of control (donor) units
    n_pre_periods : int
        Number of pre-treatment periods
    n_post_periods : int
        Number of post-treatment periods
    synthetic_control : np.ndarray
        Synthetic control series (all periods)
    treated_series : np.ndarray
        Observed treated series (all periods)
    gap : np.ndarray
        Period-by-period treatment effects (treated - synthetic)
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    weights: np.ndarray
    pre_rmse: float
    pre_r_squared: float
    n_treated: int
    n_control: int
    n_pre_periods: int
    n_post_periods: int
    synthetic_control: np.ndarray
    treated_series: np.ndarray
    gap: np.ndarray


def validate_panel_data(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,
    covariates: Optional[np.ndarray] = None,
) -> None:
    """
    Validate inputs for synthetic control estimation.

    Parameters
    ----------
    outcomes : np.ndarray
        Panel data (n_units, n_periods)
    treatment : np.ndarray
        Treatment indicator (n_units,) - binary
    treatment_period : int
        Period when treatment starts (0-indexed)
    covariates : np.ndarray, optional
        Pre-treatment covariates (n_units, n_covariates)

    Raises
    ------
    ValueError
        If inputs fail validation
    TypeError
        If inputs have wrong types
    """
    # Type checks
    if not isinstance(outcomes, np.ndarray):
        raise TypeError(f"outcomes must be np.ndarray, got {type(outcomes).__name__}")
    if not isinstance(treatment, np.ndarray):
        raise TypeError(f"treatment must be np.ndarray, got {type(treatment).__name__}")

    # Shape validation
    if outcomes.ndim != 2:
        raise ValueError(
            f"outcomes must be 2D (n_units, n_periods), got {outcomes.ndim}D"
        )

    n_units, n_periods = outcomes.shape

    if treatment.ndim != 1:
        raise ValueError(f"treatment must be 1D (n_units,), got {treatment.ndim}D")

    if len(treatment) != n_units:
        raise ValueError(
            f"treatment length ({len(treatment)}) != number of units ({n_units})"
        )

    # Treatment indicator validation
    unique_vals = np.unique(treatment)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"treatment must be binary (0 or 1), got values: {unique_vals}"
        )

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated units found (treatment must have at least one 1)")

    if n_control == 0:
        raise ValueError("No control units found (need donor pool for synthetic control)")

    if n_control < 2:
        raise ValueError(
            f"Need at least 2 control units for synthetic control, got {n_control}"
        )

    # Treatment period validation
    if not isinstance(treatment_period, (int, np.integer)):
        raise TypeError(
            f"treatment_period must be int, got {type(treatment_period).__name__}"
        )

    if treatment_period < 1:
        raise ValueError(
            f"treatment_period must be >= 1 (need pre-treatment periods), got {treatment_period}"
        )

    if treatment_period >= n_periods:
        raise ValueError(
            f"treatment_period ({treatment_period}) >= n_periods ({n_periods}), "
            "no post-treatment data"
        )

    n_pre_periods = treatment_period
    n_post_periods = n_periods - treatment_period

    # Warn about few pre-treatment periods (but don't fail)
    if n_pre_periods < 5:
        import warnings

        warnings.warn(
            f"Only {n_pre_periods} pre-treatment periods. SCM works best with >= 5. "
            "Consider using Augmented SCM for better bias properties.",
            UserWarning,
            stacklevel=3,
        )

    # Covariates validation
    if covariates is not None:
        if not isinstance(covariates, np.ndarray):
            raise TypeError(
                f"covariates must be np.ndarray, got {type(covariates).__name__}"
            )

        if covariates.ndim != 2:
            raise ValueError(
                f"covariates must be 2D (n_units, n_covariates), got {covariates.ndim}D"
            )

        if covariates.shape[0] != n_units:
            raise ValueError(
                f"covariates rows ({covariates.shape[0]}) != n_units ({n_units})"
            )

    # Check for missing values
    if np.any(np.isnan(outcomes)):
        raise ValueError("outcomes contains NaN values - SCM requires complete panel data")

    if covariates is not None and np.any(np.isnan(covariates)):
        raise ValueError("covariates contains NaN values")


def validate_weights(weights: np.ndarray, n_control: int, tol: float = 1e-6) -> None:
    """
    Validate synthetic control weights.

    Parameters
    ----------
    weights : np.ndarray
        Weights for control units
    n_control : int
        Expected number of control units
    tol : float
        Tolerance for constraint violations

    Raises
    ------
    ValueError
        If weights violate constraints
    """
    if len(weights) != n_control:
        raise ValueError(
            f"weights length ({len(weights)}) != n_control ({n_control})"
        )

    if np.any(weights < -tol):
        min_w = np.min(weights)
        raise ValueError(
            f"weights must be non-negative, min weight = {min_w:.6f}"
        )

    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > tol:
        raise ValueError(
            f"weights must sum to 1, sum = {weight_sum:.6f}"
        )
