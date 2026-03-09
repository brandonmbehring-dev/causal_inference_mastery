"""
Type definitions for Marginal Treatment Effects (MTE) estimation.

Implements Heckman & Vytlacil (2005) framework for treatment effect heterogeneity.
"""

from typing import Tuple, Optional, Literal
from typing_extensions import TypedDict
import numpy as np


class MTEResult(TypedDict):
    """
    Result from Marginal Treatment Effect estimation.

    MTE(u) = E[Y₁ - Y₀ | U = u] where U is unobserved resistance to treatment.

    Attributes
    ----------
    mte_grid : np.ndarray
        MTE(u) evaluated at grid points in propensity support
    u_grid : np.ndarray
        Grid points u ∈ [p_min, p_max] where p is propensity
    se_grid : np.ndarray
        Standard errors at each grid point
    ci_lower : np.ndarray
        Lower CI bound at each grid point
    ci_upper : np.ndarray
        Upper CI bound at each grid point
    propensity_support : Tuple[float, float]
        (min, max) of estimated propensity scores
    n_obs : int
        Total sample size
    n_trimmed : int
        Units trimmed for common support
    bandwidth : float
        Bandwidth used for local estimation
    method : str
        Estimation method: "local_iv" | "polynomial"
    """

    mte_grid: np.ndarray
    u_grid: np.ndarray
    se_grid: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    propensity_support: Tuple[float, float]
    n_obs: int
    n_trimmed: int
    bandwidth: float
    method: str


class LATEResult(TypedDict):
    """
    Result from Local Average Treatment Effect estimation.

    LATE = E[Y₁ - Y₀ | Complier] for binary instrument.

    Attributes
    ----------
    late : float
        Local average treatment effect estimate
    se : float
        Standard error (delta method or bootstrap)
    ci_lower : float
        Lower 95% CI bound
    ci_upper : float
        Upper 95% CI bound
    pvalue : float
        Two-sided p-value for H₀: LATE = 0
    complier_share : float
        Proportion of population that are compliers (P(D₁ > D₀))
    always_taker_share : float
        Proportion always treated (P(D₁ = D₀ = 1))
    never_taker_share : float
        Proportion never treated (P(D₁ = D₀ = 0))
    first_stage_coef : float
        First-stage coefficient (E[D|Z=1] - E[D|Z=0])
    first_stage_f : float
        First-stage F-statistic
    n_obs : int
        Total sample size
    method : str
        Estimation method: "wald" | "2sls"
    """

    late: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    complier_share: float
    always_taker_share: float
    never_taker_share: float
    first_stage_coef: float
    first_stage_f: float
    n_obs: int
    method: str


class PolicyResult(TypedDict):
    """
    Result from policy-relevant treatment effect estimation.

    Integrates MTE to obtain population parameters (ATE, ATT, ATU, PRTE).

    Attributes
    ----------
    estimate : float
        Policy parameter estimate
    se : float
        Standard error (bootstrap or delta method)
    ci_lower : float
        Lower 95% CI bound
    ci_upper : float
        Upper 95% CI bound
    parameter : str
        Type: "ate" | "att" | "atu" | "prte"
    weights_used : str
        Description of weighting scheme
    n_obs : int
        Sample size
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    parameter: str
    weights_used: str
    n_obs: int


class ComplierResult(TypedDict):
    """
    Result from complier characteristic analysis.

    Describes the subpopulation whose treatment is affected by the instrument.

    Attributes
    ----------
    complier_mean_outcome_treated : float
        E[Y₁ | Complier]
    complier_mean_outcome_control : float
        E[Y₀ | Complier]
    complier_share : float
        Fraction of compliers in population
    covariate_means : Optional[np.ndarray]
        Mean covariate values for compliers (if covariates provided)
    covariate_names : Optional[list]
        Names of covariates
    method : str
        Estimation method: "kappa_weights" | "bounds"
    """

    complier_mean_outcome_treated: float
    complier_mean_outcome_control: float
    complier_share: float
    covariate_means: Optional[np.ndarray]
    covariate_names: Optional[list]
    method: str


class CommonSupportResult(TypedDict):
    """
    Result from common support diagnostics.

    Checks propensity score overlap between treatment groups.

    Attributes
    ----------
    has_support : bool
        Whether common support exists
    support_region : Tuple[float, float]
        (min, max) propensity in common support
    n_outside_support : int
        Units outside common support
    fraction_outside : float
        Proportion outside support
    recommendation : str
        Trimming/reweighting recommendation
    """

    has_support: bool
    support_region: Tuple[float, float]
    n_outside_support: int
    fraction_outside: float
    recommendation: str
