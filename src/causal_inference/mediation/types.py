"""
Type definitions for Mediation Analysis.

Implements Imai et al. (2010) framework for causal mediation effects.

References
----------
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction
- Pearl (2001). Direct and Indirect Effects
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict


class MediationResult(TypedDict):
    """
    Result from mediation analysis.

    Contains decomposition of total effect into direct and indirect effects,
    with inference via bootstrap or delta method.

    Attributes
    ----------
    total_effect : float
        Total effect = NDE + NIE
    direct_effect : float
        Natural Direct Effect (NDE) or beta_1 in Baron-Kenny
    indirect_effect : float
        Natural Indirect Effect (NIE) or alpha_1 * beta_2
    proportion_mediated : float
        Proportion of total effect through mediator (NIE / TE)
    te_se : float
        Standard error of total effect
    de_se : float
        Standard error of direct effect
    ie_se : float
        Standard error of indirect effect
    pm_se : float
        Standard error of proportion mediated
    te_ci : Tuple[float, float]
        95% CI for total effect
    de_ci : Tuple[float, float]
        95% CI for direct effect
    ie_ci : Tuple[float, float]
        95% CI for indirect effect
    pm_ci : Tuple[float, float]
        95% CI for proportion mediated
    te_pvalue : float
        Two-sided p-value for H0: TE = 0
    de_pvalue : float
        Two-sided p-value for H0: DE = 0
    ie_pvalue : float
        Two-sided p-value for H0: IE = 0
    method : str
        Estimation method ("baron_kenny", "simulation", "cde")
    n_obs : int
        Number of observations
    n_bootstrap : int
        Number of bootstrap replications
    treatment_control : Tuple[float, float]
        (control_value, treatment_value)
    mediator_model : str
        Model for mediator ("linear" or "logistic")
    outcome_model : str
        Model for outcome ("linear" or "logistic")
    """

    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float

    te_se: float
    de_se: float
    ie_se: float
    pm_se: float

    te_ci: Tuple[float, float]
    de_ci: Tuple[float, float]
    ie_ci: Tuple[float, float]
    pm_ci: Tuple[float, float]

    te_pvalue: float
    de_pvalue: float
    ie_pvalue: float

    method: Literal["baron_kenny", "simulation", "cde"]
    n_obs: int
    n_bootstrap: int
    treatment_control: Tuple[float, float]
    mediator_model: str
    outcome_model: str


class BaronKennyResult(TypedDict):
    """
    Detailed Baron-Kenny mediation decomposition.

    Returns path coefficients from the linear mediation model:
    M = alpha_0 + alpha_1 * T + e_1
    Y = beta_0 + beta_1 * T + beta_2 * M + e_2

    Attributes
    ----------
    alpha_1 : float
        Effect of treatment on mediator (T -> M)
    alpha_1_se : float
        Standard error of alpha_1
    alpha_1_pvalue : float
        P-value for H0: alpha_1 = 0
    beta_1 : float
        Direct effect (T -> Y controlling for M)
    beta_1_se : float
        Standard error of beta_1
    beta_1_pvalue : float
        P-value for H0: beta_1 = 0
    beta_2 : float
        Effect of mediator on outcome (M -> Y)
    beta_2_se : float
        Standard error of beta_2
    beta_2_pvalue : float
        P-value for H0: beta_2 = 0
    indirect_effect : float
        alpha_1 * beta_2
    indirect_se : float
        Standard error via Sobel or bootstrap
    direct_effect : float
        beta_1
    total_effect : float
        beta_1 + alpha_1 * beta_2
    sobel_z : float
        Sobel test statistic for indirect effect
    sobel_pvalue : float
        P-value from Sobel test
    r2_mediator_model : float
        R-squared of mediator model
    r2_outcome_model : float
        R-squared of outcome model
    n_obs : int
        Number of observations
    """

    alpha_1: float
    alpha_1_se: float
    alpha_1_pvalue: float

    beta_1: float
    beta_1_se: float
    beta_1_pvalue: float

    beta_2: float
    beta_2_se: float
    beta_2_pvalue: float

    indirect_effect: float
    indirect_se: float
    direct_effect: float
    total_effect: float

    sobel_z: float
    sobel_pvalue: float

    r2_mediator_model: float
    r2_outcome_model: float
    n_obs: int


class SensitivityResult(TypedDict):
    """
    Sensitivity analysis result for mediation.

    Assesses how estimated effects change under violations of
    sequential ignorability (unmeasured confounding).

    Based on Imai et al. (2010) sensitivity parameter rho.

    Attributes
    ----------
    rho_grid : NDArray[np.floating]
        Grid of sensitivity parameter values
    nde_at_rho : NDArray[np.floating]
        NDE estimate at each rho value
    nie_at_rho : NDArray[np.floating]
        NIE estimate at each rho value
    nde_ci_lower : NDArray[np.floating]
        Lower CI for NDE at each rho
    nde_ci_upper : NDArray[np.floating]
        Upper CI for NDE at each rho
    nie_ci_lower : NDArray[np.floating]
        Lower CI for NIE at each rho
    nie_ci_upper : NDArray[np.floating]
        Upper CI for NIE at each rho
    rho_at_zero_nie : float
        Rho value at which NIE = 0 (if exists, else NaN)
    rho_at_zero_nde : float
        Rho value at which NDE = 0 (if exists, else NaN)
    original_nde : float
        NDE estimate under rho = 0 (no confounding)
    original_nie : float
        NIE estimate under rho = 0
    interpretation : str
        Human-readable interpretation of results
    """

    rho_grid: NDArray[np.floating]
    nde_at_rho: NDArray[np.floating]
    nie_at_rho: NDArray[np.floating]
    nde_ci_lower: NDArray[np.floating]
    nde_ci_upper: NDArray[np.floating]
    nie_ci_lower: NDArray[np.floating]
    nie_ci_upper: NDArray[np.floating]
    rho_at_zero_nie: float
    rho_at_zero_nde: float
    original_nde: float
    original_nie: float
    interpretation: str


class CDEResult(TypedDict):
    """
    Controlled Direct Effect result.

    CDE(m) = E[Y(1,m) - Y(0,m)] at fixed mediator value m.
    Simpler to identify than NDE (no cross-world counterfactuals).

    Attributes
    ----------
    cde : float
        Controlled direct effect estimate
    se : float
        Standard error
    ci_lower : float
        Lower 95% CI bound
    ci_upper : float
        Upper 95% CI bound
    pvalue : float
        P-value for H0: CDE = 0
    mediator_value : float
        Value at which mediator is fixed
    n_obs : int
        Number of observations
    method : str
        Estimation method
    """

    cde: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    mediator_value: float
    n_obs: int
    method: str


class MediationDiagnostics(TypedDict):
    """
    Diagnostics for mediation analysis assumptions.

    Attributes
    ----------
    treatment_effect_on_mediator : float
        Estimated T -> M effect
    treatment_effect_pvalue : float
        P-value for T -> M effect
    mediator_effect_on_outcome : float
        Estimated M -> Y effect (controlling for T)
    mediator_effect_pvalue : float
        P-value for M -> Y effect
    has_mediation_path : bool
        Whether both T->M and M->Y paths are significant
    r2_mediator : float
        Variance in M explained by T
    r2_outcome_full : float
        Variance in Y explained by T and M
    r2_outcome_reduced : float
        Variance in Y explained by T only
    n_obs : int
        Number of observations
    warnings : list
        List of warning messages
    """

    treatment_effect_on_mediator: float
    treatment_effect_pvalue: float
    mediator_effect_on_outcome: float
    mediator_effect_pvalue: float
    has_mediation_path: bool
    r2_mediator: float
    r2_outcome_full: float
    r2_outcome_reduced: float
    n_obs: int
    warnings: list
