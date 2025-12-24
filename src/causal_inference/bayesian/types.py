"""
Type definitions for Bayesian causal inference.

Session 101: Initial types for Bayesian ATE with conjugate priors.
Session 102: Added types for Bayesian propensity scores.
Session 103: Added BayesianDRResult.
Session 104: Added HierarchicalATEResult.
"""

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray


class BayesianATEResult(TypedDict):
    """
    Result from Bayesian Average Treatment Effect estimation.

    Attributes
    ----------
    posterior_mean : float
        Posterior mean of the treatment effect, E[tau | data].
    posterior_sd : float
        Posterior standard deviation, SD[tau | data].
    ci_lower : float
        Lower bound of credible interval (alpha/2 quantile).
    ci_upper : float
        Upper bound of credible interval (1 - alpha/2 quantile).
    credible_level : float
        Credible interval level (e.g., 0.95 for 95% CI).
    prior_mean : float
        Prior mean for the treatment effect.
    prior_sd : float
        Prior standard deviation for the treatment effect.
    posterior_samples : NDArray[np.float64]
        Samples from the posterior distribution, shape (n_samples,).
    n : int
        Total sample size.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    prior_to_posterior_shrinkage : float
        Measure of how much the prior influenced the posterior.
        Range [0, 1]: 0 = data dominates, 1 = prior dominates.
    effective_sample_size : float
        Effective sample size for the treatment effect estimate.
    ols_estimate : float
        Maximum likelihood (OLS) estimate for comparison.
    ols_se : float
        Standard error of OLS estimate.
    sigma2_mle : float
        MLE estimate of residual variance.

    Notes
    -----
    **Credible vs Confidence Intervals**:

    The credible interval has a direct probability interpretation:
    "There is a 95% probability that the true treatment effect
    lies within this interval, given the data and prior."

    This differs from frequentist confidence intervals, which have
    a coverage interpretation over repeated sampling.

    **Prior Shrinkage**:

    The shrinkage metric quantifies the relative influence of the prior:

        shrinkage = prior_precision / (prior_precision + likelihood_precision)

    - shrinkage near 0: Data dominates (many observations, weak prior)
    - shrinkage near 1: Prior dominates (few observations, strong prior)
    """

    # Point estimates
    posterior_mean: float
    posterior_sd: float

    # Credible intervals
    ci_lower: float
    ci_upper: float
    credible_level: float

    # Prior specification
    prior_mean: float
    prior_sd: float

    # Posterior samples
    posterior_samples: NDArray[np.float64]

    # Sample sizes
    n: int
    n_treated: int
    n_control: int

    # Diagnostics
    prior_to_posterior_shrinkage: float
    effective_sample_size: float

    # Comparison with frequentist
    ols_estimate: float
    ols_se: float
    sigma2_mle: float


class StratumInfo(TypedDict):
    """
    Information about a propensity stratum in Beta-Binomial estimation.

    Attributes
    ----------
    stratum_id : int
        Unique identifier for this stratum.
    n_obs : int
        Number of observations in this stratum.
    n_treated : int
        Number of treated units in this stratum.
    n_control : int
        Number of control units in this stratum.
    posterior_alpha : float
        Posterior Beta distribution alpha parameter.
    posterior_beta : float
        Posterior Beta distribution beta parameter.
    posterior_mean : float
        Posterior mean of propensity in this stratum.
    posterior_sd : float
        Posterior SD of propensity in this stratum.
    """

    stratum_id: int
    n_obs: int
    n_treated: int
    n_control: int
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_sd: float


class BayesianPropensityResult(TypedDict, total=False):
    """
    Result from Bayesian propensity score estimation.

    Attributes
    ----------
    posterior_samples : NDArray[np.float64]
        Posterior samples of propensity scores, shape (n_samples, n).
    posterior_mean : NDArray[np.float64]
        Posterior mean of propensity for each observation, shape (n,).
    posterior_sd : NDArray[np.float64]
        Posterior SD of propensity for each observation, shape (n,).
    strata : NDArray[np.int64] or None
        Stratum assignment for each observation (stratified method only).
    n_strata : int
        Number of strata (stratified method only).
    stratum_info : List[StratumInfo] or None
        Detailed information for each stratum (stratified method only).
    prior_alpha : float
        Beta prior alpha parameter (stratified method).
    prior_beta : float
        Beta prior beta parameter (stratified method).
    method : str
        Estimation method used ("stratified_beta_binomial" or "logistic_laplace").
    n : int
        Total sample size.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    mean_uncertainty : float
        Mean posterior SD across all observations.
    propensity_range : float
        Range of posterior mean propensities.

    Logistic Method Only
    --------------------
    coefficient_mean : NDArray[np.float64]
        Posterior mean of logistic regression coefficients.
    coefficient_sd : NDArray[np.float64]
        Posterior SD of logistic regression coefficients.
    coefficient_samples : NDArray[np.float64]
        Posterior samples of coefficients, shape (n_samples, n_coef).
    prior_sd : float
        Prior SD for coefficients.

    Notes
    -----
    **Stratified Beta-Binomial**:
    Best for discrete/categorical covariates. Uses exact conjugate inference.

    **Logistic Laplace**:
    Best for continuous covariates. Uses normal approximation to posterior.
    """

    # Core results
    posterior_samples: NDArray[np.float64]
    posterior_mean: NDArray[np.float64]
    posterior_sd: NDArray[np.float64]

    # Stratified method fields
    strata: Optional[NDArray[np.int64]]
    n_strata: int
    stratum_info: Optional[List[StratumInfo]]
    prior_alpha: float
    prior_beta: float

    # Common fields
    method: str
    n: int
    n_treated: int
    n_control: int
    mean_uncertainty: float
    propensity_range: float

    # Logistic method fields
    coefficient_mean: NDArray[np.float64]
    coefficient_sd: NDArray[np.float64]
    coefficient_samples: NDArray[np.float64]
    prior_sd: float


class BayesianDRResult(TypedDict):
    """
    Result from Bayesian Doubly Robust ATE estimation.

    Combines Bayesian propensity score estimation with frequentist outcome
    models, propagating propensity uncertainty through the AIPW formula.

    Attributes
    ----------
    estimate : float
        Posterior mean of the ATE, E[tau | data].
    se : float
        Posterior standard deviation, SD[tau | data].
    ci_lower : float
        Lower bound of credible interval.
    ci_upper : float
        Upper bound of credible interval.
    credible_level : float
        Credible interval level (e.g., 0.95).
    posterior_samples : NDArray[np.float64]
        Samples from the posterior distribution of ATE, shape (n_samples,).
    n : int
        Total sample size.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    propensity_mean : NDArray[np.float64]
        Posterior mean propensity for each observation, shape (n,).
    propensity_mean_uncertainty : float
        Mean posterior SD of propensity across observations.
    outcome_r2 : float
        R-squared from outcome models (average of treated and control).
    frequentist_estimate : float
        Frequentist DR estimate for comparison.
    frequentist_se : float
        Standard error of frequentist DR estimate.

    Notes
    -----
    **Uncertainty Propagation**:

    The Bayesian DR estimator propagates propensity uncertainty through
    the AIPW formula. For each propensity posterior sample e_k:

        ATE_k = (1/n) * sum[
            T/e_k * (Y - mu1) + mu1 - (1-T)/(1-e_k) * (Y - mu0) - mu0
        ]

    The posterior distribution {ATE_1, ..., ATE_S} captures uncertainty
    from propensity estimation.

    **Double Robustness**:

    Like frequentist DR, the Bayesian version is consistent when EITHER:
    - Propensity model is correctly specified, OR
    - Outcome models are correctly specified

    **Credible vs Confidence Intervals**:

    The credible interval has direct probability interpretation:
    "There is a 95% probability that the true ATE lies within this
    interval, given the data and prior."
    """

    # Point estimates
    estimate: float
    se: float

    # Credible intervals
    ci_lower: float
    ci_upper: float
    credible_level: float

    # Posterior samples
    posterior_samples: NDArray[np.float64]

    # Sample sizes
    n: int
    n_treated: int
    n_control: int

    # Propensity diagnostics
    propensity_mean: NDArray[np.float64]
    propensity_mean_uncertainty: float

    # Outcome model diagnostics
    outcome_r2: float

    # Frequentist comparison
    frequentist_estimate: float
    frequentist_se: float


class HierarchicalATEResult(TypedDict):
    """
    Result from Hierarchical Bayesian ATE estimation with MCMC.

    Provides partial pooling across groups/sites, shrinking group-specific
    estimates toward the population mean. Uses MCMC for full posterior inference.

    Attributes
    ----------
    population_ate : float
        Posterior mean of population-level ATE.
    population_ate_se : float
        Posterior SD of population-level ATE.
    population_ate_ci_lower : float
        Lower bound of population ATE credible interval.
    population_ate_ci_upper : float
        Upper bound of population ATE credible interval.
    group_ates : NDArray[np.float64]
        Posterior mean of group-specific ATEs, shape (n_groups,).
    group_ate_ses : NDArray[np.float64]
        Posterior SD of group-specific ATEs, shape (n_groups,).
    group_ids : List[Any]
        Unique group identifiers in order corresponding to group_ates.
    tau : float
        Posterior mean of between-group SD (heterogeneity).
    tau_ci_lower : float
        Lower bound of tau credible interval.
    tau_ci_upper : float
        Upper bound of tau credible interval.
    posterior_samples : Dict[str, NDArray[np.float64]]
        Full posterior samples for all parameters:
        - "mu": population ATE samples, shape (n_samples,)
        - "tau": heterogeneity samples, shape (n_samples,)
        - "theta": group-specific ATEs, shape (n_samples, n_groups)
    n_groups : int
        Number of groups in the analysis.
    n_obs : int
        Total number of observations.
    credible_level : float
        Credible interval level (e.g., 0.95).
    rhat_max : float
        Worst R-hat across all parameters (convergence diagnostic).
    ess_min : float
        Minimum effective sample size across parameters.
    divergences : int
        Number of divergent transitions (NUTS sampler health).

    Notes
    -----
    **Partial Pooling**:

    Hierarchical models provide optimal bias-variance tradeoff:
    - Small groups: Shrink more toward population mean
    - Large groups: Retain more group-specific signal
    - Population mean: Weighted average of group effects

    **MCMC Diagnostics**:

    - R-hat < 1.05: Chain convergence
    - ESS > 400: Adequate effective samples
    - Divergences = 0: NUTS sampler healthy

    **Model Structure**:

        μ ~ Normal(0, σ_μ)           # Population ATE
        τ ~ HalfNormal(σ_τ)          # Between-group SD
        θⱼ ~ Normal(μ, τ)            # Group-specific ATE
        Yᵢⱼ | Tᵢⱼ ~ Normal(α + θⱼ × Tᵢⱼ, σ)
    """

    # Population-level estimates
    population_ate: float
    population_ate_se: float
    population_ate_ci_lower: float
    population_ate_ci_upper: float

    # Group-specific estimates
    group_ates: NDArray[np.float64]
    group_ate_ses: NDArray[np.float64]
    group_ids: List[Any]

    # Heterogeneity
    tau: float
    tau_ci_lower: float
    tau_ci_upper: float

    # Full posterior samples
    posterior_samples: Dict[str, NDArray[np.float64]]

    # Metadata
    n_groups: int
    n_obs: int
    credible_level: float

    # MCMC diagnostics
    rhat_max: float
    ess_min: float
    divergences: int
