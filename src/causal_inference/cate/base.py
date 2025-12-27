"""Base types and utilities for CATE estimation.

Defines CATEResult TypedDict and common validation functions used across
all meta-learner implementations.
"""

import numpy as np
from typing import TypedDict


class CATEResult(TypedDict):
    """Return type for CATE estimators.

    Attributes
    ----------
    cate : np.ndarray
        Individual treatment effects τ(xᵢ) for each unit. Shape (n,).
    ate : float
        Average treatment effect: mean of CATE estimates.
    ate_se : float
        Standard error of the ATE estimate.
    ci_lower : float
        Lower bound of (1-alpha)% confidence interval for ATE.
    ci_upper : float
        Upper bound of (1-alpha)% confidence interval for ATE.
    method : str
        Name of the estimation method (e.g., "s_learner", "t_learner").
    """

    cate: np.ndarray
    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    method: str


class OMLResult(TypedDict):
    """Return type for Orthogonal Machine Learning estimators.

    Extends CATEResult with OML-specific fields for target parameter
    and score function type.

    Attributes
    ----------
    cate : np.ndarray
        Individual treatment effects τ(xᵢ) for each unit. Shape (n,).
    ate : float
        Average treatment effect (or ATTE if target="atte").
    ate_se : float
        Standard error of the ATE/ATTE estimate.
    ci_lower : float
        Lower bound of (1-alpha)% confidence interval.
    ci_upper : float
        Upper bound of (1-alpha)% confidence interval.
    method : str
        Name of the estimation method (e.g., "irm_dml").
    target : str
        Target parameter: "ate" (Average Treatment Effect) or
        "atte" (Average Treatment Effect on Treated).
    score_type : str
        Score function type: "irm" (Interactive Regression Model, doubly robust)
        or "plr" (Partially Linear Regression).
    """

    cate: np.ndarray
    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    method: str
    target: str
    score_type: str


def validate_cate_inputs(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert inputs for CATE estimation.

    Parameters
    ----------
    outcomes : array-like
        Outcome variable Y of shape (n,).
    treatment : array-like
        Binary treatment indicator of shape (n,).
    covariates : array-like
        Covariate matrix X of shape (n, p) or (n,) for single covariate.

    Returns
    -------
    tuple
        Validated (outcomes, treatment, covariates) as numpy arrays.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or values.

    Examples
    --------
    >>> y, t, X = validate_cate_inputs([1, 2, 3], [0, 1, 0], [[1], [2], [3]])
    >>> y.shape, t.shape, X.shape
    ((3,), (3,), (3, 1))
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    n = len(outcomes)

    # Validate lengths
    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"Function: validate_cate_inputs\n"
            f"outcomes has {n} observations, treatment has {len(treatment)}.\n"
            f"All inputs must have the same number of observations."
        )

    # Handle 1D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    if len(covariates) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"Function: validate_cate_inputs\n"
            f"outcomes has {n} observations, covariates has {len(covariates)}.\n"
            f"All inputs must have the same number of observations."
        )

    if covariates.ndim != 2:
        raise ValueError(
            f"CRITICAL ERROR: Invalid covariate shape.\n"
            f"Function: validate_cate_inputs\n"
            f"covariates must be 2D array (n, p), got shape {covariates.shape}.\n"
            f"Use covariates.reshape(-1, 1) for single covariate."
        )

    # Validate treatment is binary
    unique_t = np.unique(treatment)
    if not np.all(np.isin(unique_t, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary.\n"
            f"Function: validate_cate_inputs\n"
            f"Got unique values: {unique_t}.\n"
            f"Treatment must contain only 0 and 1."
        )

    # Check both treatment groups exist
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError(
            f"CRITICAL ERROR: No treated units.\n"
            f"Function: validate_cate_inputs\n"
            f"All {n} units are controls (T=0).\n"
            f"Need at least one treated unit to estimate treatment effects."
        )

    if n_control == 0:
        raise ValueError(
            f"CRITICAL ERROR: No control units.\n"
            f"Function: validate_cate_inputs\n"
            f"All {n} units are treated (T=1).\n"
            f"Need at least one control unit to estimate treatment effects."
        )

    return outcomes, treatment, covariates
