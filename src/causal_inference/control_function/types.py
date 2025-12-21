"""
Type definitions for Control Function estimation.

Defines TypedDict result types for control function analysis outputs,
including linear and nonlinear (probit/logit) models.

References
----------
- Wooldridge (2015). "Control Function Methods in Applied Econometrics"
- Murphy & Topel (1985). "Estimation and Inference in Two-Step Models"
"""

from typing import Literal, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray


class FirstStageResult(TypedDict):
    """Result from first-stage regression.

    The first stage regresses the endogenous variable D on instruments Z
    and controls X: D = pi_0 + pi_1*Z + pi_2*X + nu

    Attributes
    ----------
    coefficients : NDArray[np.float64]
        First-stage coefficients on intercept, instruments, and controls.
    se : NDArray[np.float64]
        Standard errors of first-stage coefficients.
    residuals : NDArray[np.float64]
        First-stage residuals nu_hat = D - D_hat (the control function).
    fitted_values : NDArray[np.float64]
        Predicted treatment D_hat.
    f_statistic : float
        F-statistic for excluded instrument relevance.
    f_pvalue : float
        P-value for F-test.
    partial_r2 : float
        Partial R-squared (variance explained by Z | X).
    r2 : float
        Overall R-squared.
    n_obs : int
        Number of observations.
    n_instruments : int
        Number of excluded instruments.
    weak_iv_warning : bool
        True if F < 10 (Stock-Yogo weak IV threshold).
    """

    coefficients: NDArray[np.float64]
    se: NDArray[np.float64]
    residuals: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    f_statistic: float
    f_pvalue: float
    partial_r2: float
    r2: float
    n_obs: int
    n_instruments: int
    weak_iv_warning: bool


class ControlFunctionResult(TypedDict):
    """Result from Control Function estimation.

    The control function approach estimates:
    - First stage: D = pi*Z + pi_X*X + nu
    - Second stage: Y = beta*D + rho*nu_hat + gamma*X + u

    The coefficient rho captures endogeneity. If rho = 0, OLS is consistent.

    Attributes
    ----------
    estimate : float
        Control function estimate of causal effect (beta_1).
    se : float
        Standard error (bootstrap or Murphy-Topel corrected).
    se_naive : float
        Naive OLS SE from second stage (INCORRECT, for comparison only).
    t_stat : float
        T-statistic for treatment effect.
    p_value : float
        Two-sided p-value for H0: beta = 0.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    control_coef : float
        Coefficient on first-stage residual (rho).
    control_se : float
        Standard error of control coefficient.
    control_t_stat : float
        T-statistic for endogeneity test (H0: rho = 0).
    control_p_value : float
        P-value for endogeneity test.
    endogeneity_detected : bool
        True if control coefficient significantly != 0 at alpha level.
    first_stage : FirstStageResult
        Full first-stage regression results.
    second_stage_r2 : float
        R-squared of control function second stage.
    n_obs : int
        Number of observations.
    n_instruments : int
        Number of excluded instruments.
    n_controls : int
        Number of exogenous controls (excluding intercept).
    inference : Literal['analytical', 'bootstrap']
        Type of standard errors computed.
    n_bootstrap : Optional[int]
        Number of bootstrap iterations (if bootstrap).
    alpha : float
        Significance level used for tests and CIs.
    message : str
        Descriptive message about estimation.
    """

    estimate: float
    se: float
    se_naive: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    control_coef: float
    control_se: float
    control_t_stat: float
    control_p_value: float
    endogeneity_detected: bool
    first_stage: FirstStageResult
    second_stage_r2: float
    n_obs: int
    n_instruments: int
    n_controls: int
    inference: Literal["analytical", "bootstrap"]
    n_bootstrap: Optional[int]
    alpha: float
    message: str


class NonlinearCFResult(TypedDict):
    """Result from nonlinear Control Function estimation (probit/logit).

    For binary outcomes, extends control function to probit/logit models
    where 2SLS is not valid. Uses generalized residuals from first stage.

    Model:
        First stage:  D = pi*Z + nu (continuous or probit)
        Second stage: Y* = beta*D + rho*nu_hat + u (latent)
                      Y = 1{Y* > 0} (binary outcome)

    Attributes
    ----------
    estimate : float
        Average marginal effect of treatment.
    se : float
        Bootstrap standard error.
    ci_lower : float
        Lower CI bound.
    ci_upper : float
        Upper CI bound.
    p_value : float
        P-value for treatment effect.
    control_coef : float
        Coefficient on control function in latent equation.
    control_se : float
        Standard error of control coefficient.
    control_p_value : float
        P-value for endogeneity test.
    endogeneity_detected : bool
        True if control coefficient significant.
    first_stage : FirstStageResult
        First-stage results.
    model_type : Literal['probit', 'logit']
        Type of nonlinear model for second stage.
    n_obs : int
        Number of observations.
    n_bootstrap : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    convergence : bool
        Whether estimation converged.
    message : str
        Status message.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    control_coef: float
    control_se: float
    control_p_value: float
    endogeneity_detected: bool
    first_stage: FirstStageResult
    model_type: Literal["probit", "logit"]
    n_obs: int
    n_bootstrap: int
    alpha: float
    convergence: bool
    message: str
