"""Type definitions for Heckman selection model.

This module provides TypedDict definitions for structured return types,
ensuring type safety and IDE support.
"""

from typing import TypedDict, Dict, Any, Optional
import numpy as np


class SelectionDiagnostics(TypedDict):
    """Diagnostics for selection equation."""

    pseudo_r_squared: float
    log_likelihood: float
    n_iterations: int
    converged: bool


class HeckmanResult(TypedDict):
    """Return type for heckman_two_step() estimator.

    Attributes
    ----------
    estimate : float
        Coefficient of interest (e.g., treatment effect or beta_X).
    se : float
        Standard error with Heckman correction for two-stage estimation.
    ci_lower : float
        Lower bound of (1-alpha)% confidence interval.
    ci_upper : float
        Upper bound of (1-alpha)% confidence interval.
    rho : float
        Correlation between selection and outcome errors (selection parameter).
        rho = 0 implies no selection bias.
    sigma : float
        Standard deviation of outcome equation errors.
    lambda_coef : float
        Coefficient on Inverse Mills Ratio (= rho * sigma).
    lambda_se : float
        Standard error of lambda coefficient.
    lambda_pvalue : float
        P-value for test H0: lambda = 0 (no selection).
    n_selected : int
        Number of observations with observed outcome (selected sample).
    n_total : int
        Total number of observations.
    selection_probs : np.ndarray
        Predicted selection probabilities from probit model.
    imr : np.ndarray
        Inverse Mills Ratio values for selected sample.
    gamma : np.ndarray
        Coefficients from selection (probit) equation.
    beta : np.ndarray
        Coefficients from outcome equation (including lambda).
    vcov : np.ndarray
        Variance-covariance matrix with Heckman correction.
    selection_diagnostics : SelectionDiagnostics
        Diagnostics from selection equation fit.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    rho: float
    sigma: float
    lambda_coef: float
    lambda_se: float
    lambda_pvalue: float
    n_selected: int
    n_total: int
    selection_probs: np.ndarray
    imr: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    vcov: np.ndarray
    selection_diagnostics: SelectionDiagnostics


class SelectionTestResult(TypedDict):
    """Return type for selection test."""

    statistic: float
    pvalue: float
    reject_null: bool
    interpretation: str
