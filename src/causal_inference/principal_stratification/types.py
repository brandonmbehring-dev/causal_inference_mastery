"""Type definitions for Principal Stratification methods.

This module provides TypedDict definitions for structured return types,
ensuring type safety and IDE support.

Principal strata are defined by joint potential treatment values:
- Compliers: D(0)=0, D(1)=1 (take treatment iff assigned)
- Always-takers: D(0)=1, D(1)=1 (always take treatment)
- Never-takers: D(0)=0, D(1)=0 (never take treatment)
- Defiers: D(0)=1, D(1)=0 (do opposite of assignment) - ruled out by monotonicity

References
----------
- Frangakis, C. E., & Rubin, D. B. (2002). Principal Stratification in Causal Inference.
  Biometrics, 58(1), 21-29.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of Causal Effects
  Using Instrumental Variables. Journal of the American Statistical Association, 91(434), 444-455.
"""

from typing import TypedDict, Dict, Any, Optional, Literal
import numpy as np


class StrataProportions(TypedDict):
    """Estimated proportions of principal strata.

    Under monotonicity (no defiers), we can identify proportions from:
    - pi_c (compliers) = P(D=1|Z=1) - P(D=1|Z=0)
    - pi_a (always-takers) = P(D=1|Z=0)
    - pi_n (never-takers) = P(D=0|Z=1) = 1 - P(D=1|Z=1)

    Attributes
    ----------
    compliers : float
        Proportion of compliers (pi_c). Equals first-stage coefficient.
    always_takers : float
        Proportion of always-takers (pi_a).
    never_takers : float
        Proportion of never-takers (pi_n).
    compliers_se : float
        Standard error of complier proportion estimate.
    """

    compliers: float
    always_takers: float
    never_takers: float
    compliers_se: float


class CACEResult(TypedDict):
    """Return type for CACE (Complier Average Causal Effect) estimator.

    CACE is the average treatment effect for compliers only:
    CACE = E[Y(1) - Y(0) | D(0)=0, D(1)=1]

    Under standard IV assumptions (independence, exclusion, monotonicity, relevance),
    CACE = LATE = (Reduced Form) / (First Stage) = Cov(Y,Z)/Cov(D,Z)

    Attributes
    ----------
    cace : float
        Complier Average Causal Effect estimate.
    se : float
        Standard error of CACE (from 2SLS or delta method).
    ci_lower : float
        Lower bound of (1-alpha)% confidence interval.
    ci_upper : float
        Upper bound of (1-alpha)% confidence interval.
    z_stat : float
        Z-statistic for test H0: CACE = 0.
    pvalue : float
        P-value for two-sided test.
    strata_proportions : StrataProportions
        Estimated proportions of each stratum.
    first_stage_coef : float
        First-stage coefficient (E[D|Z=1] - E[D|Z=0]).
    first_stage_se : float
        Standard error of first-stage coefficient.
    first_stage_f : float
        First-stage F-statistic for weak instrument diagnostic.
    reduced_form : float
        Reduced-form coefficient (E[Y|Z=1] - E[Y|Z=0]).
    reduced_form_se : float
        Standard error of reduced-form coefficient.
    n : int
        Sample size.
    n_treated_assigned : int
        Number assigned to treatment (Z=1).
    n_control_assigned : int
        Number assigned to control (Z=0).
    method : str
        Estimation method used ('2sls', 'em', 'bayesian').
    """

    cace: float
    se: float
    ci_lower: float
    ci_upper: float
    z_stat: float
    pvalue: float
    strata_proportions: StrataProportions
    first_stage_coef: float
    first_stage_se: float
    first_stage_f: float
    reduced_form: float
    reduced_form_se: float
    n: int
    n_treated_assigned: int
    n_control_assigned: int
    method: str


class SACEResult(TypedDict):
    """Return type for SACE (Survivor Average Causal Effect) estimator.

    SACE is the treatment effect for units who would survive under both treatment conditions.
    Used when there is truncation by death / attrition.

    Attributes
    ----------
    sace : float
        Survivor Average Causal Effect estimate.
    se : float
        Standard error (may be bounds-based).
    lower_bound : float
        Lower bound on SACE.
    upper_bound : float
        Upper bound on SACE.
    proportion_survivors_treat : float
        P(S(1)=1) - survival probability under treatment.
    proportion_survivors_control : float
        P(S(0)=1) - survival probability under control.
    n : int
        Sample size.
    method : str
        Estimation method ('bounds', 'sensitivity').
    """

    sace: float
    se: float
    lower_bound: float
    upper_bound: float
    proportion_survivors_treat: float
    proportion_survivors_control: float
    n: int
    method: str


class BoundsResult(TypedDict):
    """Return type for principal stratification bounds.

    When identification fails (e.g., no instrument, no monotonicity), we can
    compute worst-case bounds on treatment effects.

    Attributes
    ----------
    lower_bound : float
        Lower bound on CACE.
    upper_bound : float
        Upper bound on CACE.
    bound_width : float
        Width of bounds (upper - lower).
    identified : bool
        Whether effect is point-identified (bounds collapse to point).
    assumptions : list[str]
        List of assumptions used for identification.
    method : str
        Method used for bounds computation.
    """

    lower_bound: float
    upper_bound: float
    bound_width: float
    identified: bool
    assumptions: list
    method: str


class BayesianPSResult(TypedDict):
    """Return type for Bayesian principal stratification.

    Full posterior inference on strata membership and causal effects.

    Attributes
    ----------
    cace_mean : float
        Posterior mean of CACE.
    cace_sd : float
        Posterior standard deviation of CACE.
    cace_hdi_lower : float
        Lower bound of 95% HDI (highest density interval).
    cace_hdi_upper : float
        Upper bound of 95% HDI.
    cace_samples : np.ndarray
        Posterior samples of CACE.
    pi_c_mean : float
        Posterior mean of complier proportion.
    pi_c_samples : np.ndarray
        Posterior samples of complier proportion.
    pi_a_mean : float
        Posterior mean of always-taker proportion.
    pi_n_mean : float
        Posterior mean of never-taker proportion.
    stratum_probs : np.ndarray
        Posterior probability of stratum membership for each unit.
        Shape: (n_samples, n_units, 3) for complier/always/never.
    rhat : Dict[str, float]
        Gelman-Rubin convergence diagnostic for each parameter.
    ess : Dict[str, float]
        Effective sample size for each parameter.
    n_samples : int
        Number of posterior samples.
    n_chains : int
        Number of MCMC chains.
    model : str
        Description of Bayesian model used.
    """

    cace_mean: float
    cace_sd: float
    cace_hdi_lower: float
    cace_hdi_upper: float
    cace_samples: np.ndarray
    pi_c_mean: float
    pi_c_samples: np.ndarray
    pi_a_mean: float
    pi_n_mean: float
    stratum_probs: np.ndarray
    rhat: Dict[str, float]
    ess: Dict[str, float]
    n_samples: int
    n_chains: int
    model: str


class MonotonicityTestResult(TypedDict):
    """Return type for monotonicity assumption test.

    Monotonicity (D(1) >= D(0) for all units) is untestable directly,
    but we can test necessary conditions.

    Attributes
    ----------
    passed : bool
        Whether necessary conditions for monotonicity are satisfied.
    test_stat : float
        Test statistic value.
    pvalue : float
        P-value for the test.
    defier_proportion_bound : float
        Upper bound on proportion of defiers (should be ~0).
    interpretation : str
        Human-readable interpretation of results.
    warning : str
        Warning about untestability of the full assumption.
    """

    passed: bool
    test_stat: float
    pvalue: float
    defier_proportion_bound: float
    interpretation: str
    warning: str
