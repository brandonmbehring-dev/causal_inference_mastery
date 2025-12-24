"""Bayesian Doubly Robust (DR) estimation for causal inference.

This module implements Bayesian DR ATE estimation that combines Bayesian propensity
score estimation with frequentist outcome regression. Propensity uncertainty is
propagated through the AIPW formula to obtain a posterior distribution for the ATE.

Key property: Uncertainty quantification via propensity posterior
- Each propensity posterior sample generates one ATE estimate
- Posterior distribution captures estimation uncertainty
- Double robustness property preserved

Session 103: Initial implementation.

References
----------
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. JASA, 89(427), 846-866.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and
  causal inference models. Biometrics, 61(4), 962-973.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Optional, Union

from .types import BayesianDRResult
from .bayesian_propensity import bayesian_propensity

# Import directly from module files to avoid circular import via __init__.py
from causal_inference.observational.outcome_regression import fit_outcome_models
from causal_inference.observational.doubly_robust import dr_ate


def _bayesian_dr_core(
    outcomes: NDArray[np.float64],
    treatment: NDArray[np.float64],
    e_samples: NDArray[np.float64],
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
    trim_threshold: float,
) -> NDArray[np.float64]:
    """
    Propagate propensity uncertainty through the DR formula.

    For each propensity posterior sample, computes one ATE estimate using
    the AIPW formula.

    Parameters
    ----------
    outcomes : NDArray
        Observed outcomes, shape (n,).
    treatment : NDArray
        Binary treatment indicator, shape (n,).
    e_samples : NDArray
        Propensity posterior samples, shape (n_samples, n).
    mu0 : NDArray
        Outcome model predictions for control, shape (n,).
    mu1 : NDArray
        Outcome model predictions for treated, shape (n,).
    trim_threshold : float
        Clipping threshold for propensity scores.

    Returns
    -------
    NDArray
        Posterior samples of ATE, shape (n_samples,).
    """
    n_samples = e_samples.shape[0]
    ate_samples = np.zeros(n_samples)

    for k in range(n_samples):
        # Clip propensity sample
        e_k = np.clip(e_samples[k, :], trim_threshold, 1 - trim_threshold)

        # AIPW formula
        treated = treatment / e_k * (outcomes - mu1) + mu1
        control = (1 - treatment) / (1 - e_k) * (outcomes - mu0) + mu0

        ate_samples[k] = np.mean(treated - control)

    return ate_samples


def bayesian_dr_ate(
    outcomes: Union[NDArray[np.float64], list],
    treatment: Union[NDArray[np.float64], list],
    covariates: Union[NDArray[np.float64], list],
    *,
    propensity_method: str = "auto",
    propensity_prior_alpha: float = 1.0,
    propensity_prior_beta: float = 1.0,
    propensity_prior_sd: float = 10.0,
    n_posterior_samples: int = 1000,
    credible_level: float = 0.95,
    trim_threshold: float = 0.01,
) -> BayesianDRResult:
    """
    Bayesian Doubly Robust ATE estimation.

    Combines Bayesian propensity score estimation with frequentist outcome
    models, propagating propensity uncertainty through the AIPW formula.

    Parameters
    ----------
    outcomes : NDArray or list
        Observed outcomes for all units, shape (n,).
    treatment : NDArray or list
        Binary treatment indicator (1=treated, 0=control), shape (n,).
    covariates : NDArray or list
        Covariate matrix, shape (n, p). Can be 1D for single covariate.
    propensity_method : str, default="auto"
        Method for Bayesian propensity estimation:
        - "auto": Automatically select based on covariate type
        - "stratified": Use Beta-Binomial (best for discrete covariates)
        - "logistic": Use Laplace approximation (best for continuous)
    propensity_prior_alpha : float, default=1.0
        Beta prior alpha parameter (stratified method only).
    propensity_prior_beta : float, default=1.0
        Beta prior beta parameter (stratified method only).
    propensity_prior_sd : float, default=10.0
        Prior SD for logistic regression coefficients (logistic method only).
    n_posterior_samples : int, default=1000
        Number of posterior samples for propensity.
    credible_level : float, default=0.95
        Credible interval level (e.g., 0.95 for 95% CI).
    trim_threshold : float, default=0.01
        Propensity clipping threshold (applied to each sample).

    Returns
    -------
    BayesianDRResult
        Dictionary with keys:
        - 'estimate': Posterior mean of ATE
        - 'se': Posterior standard deviation
        - 'ci_lower', 'ci_upper': Credible interval bounds
        - 'credible_level': The credible level used
        - 'posterior_samples': Full posterior samples, shape (n_samples,)
        - 'n', 'n_treated', 'n_control': Sample sizes
        - 'propensity_mean': Posterior mean propensity, shape (n,)
        - 'propensity_mean_uncertainty': Mean SD across observations
        - 'outcome_r2': R-squared from outcome models
        - 'frequentist_estimate', 'frequentist_se': For comparison

    Raises
    ------
    ValueError
        If inputs invalid, mismatched lengths, or insufficient data.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.normal(0, 1, (n, 2))
    >>> e_X = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    >>> T = np.random.binomial(1, e_X).astype(float)
    >>> Y = 2.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, n)
    >>> result = bayesian_dr_ate(Y, T, X)
    >>> print(f"ATE: {result['estimate']:.3f} +/- {result['se']:.3f}")

    Notes
    -----
    **Uncertainty Propagation**:

    The algorithm:
    1. Estimate Bayesian propensity → posterior samples e_samples[k, i]
    2. Fit frequentist outcome models → deterministic μ₀(X), μ₁(X)
    3. For each propensity sample k:
           ATE_k = (1/n) Σ [T/e_k * (Y - μ₁) + μ₁ - (1-T)/(1-e_k) * (Y - μ₀) - μ₀]
    4. Posterior: ATE ~ {ATE_1, ..., ATE_S}

    **Double Robustness**:

    Like frequentist DR, consistent when EITHER:
    - Propensity model correctly specified, OR
    - Outcome models correctly specified

    **Credible vs Confidence Intervals**:

    Credible intervals have direct probability interpretation:
    "There is a 95% probability that the true ATE lies within this interval."
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    outcomes = np.asarray(outcomes, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    if len(treatment) != n or covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Input arrays have different lengths.\n"
            f"Function: bayesian_dr_ate\n"
            f"Expected: All arrays length {n}\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"covariates.shape={covariates.shape}"
        )

    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)) or np.any(np.isnan(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: bayesian_dr_ate\n"
            f"NaN indicates data quality issues that must be addressed."
        )

    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)) or np.any(np.isinf(covariates)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: bayesian_dr_ate"
        )

    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: bayesian_dr_ate\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    if trim_threshold <= 0 or trim_threshold >= 0.5:
        raise ValueError(
            f"CRITICAL ERROR: trim_threshold must be in (0, 0.5).\n"
            f"Function: bayesian_dr_ate\n"
            f"Got: trim_threshold = {trim_threshold}"
        )

    if credible_level <= 0 or credible_level >= 1:
        raise ValueError(
            f"CRITICAL ERROR: credible_level must be in (0, 1).\n"
            f"Function: bayesian_dr_ate\n"
            f"Got: credible_level = {credible_level}"
        )

    # ============================================================================
    # Step 1: Bayesian Propensity Score Estimation
    # ============================================================================

    # Build propensity kwargs based on method
    prop_kwargs = {
        "method": propensity_method,
        "n_posterior_samples": n_posterior_samples,
    }

    # Stratified method uses prior_alpha/beta, logistic uses prior_sd
    # The bayesian_propensity function auto-selects, so we pass method-specific params
    if propensity_method == "stratified":
        prop_kwargs["prior_alpha"] = propensity_prior_alpha
        prop_kwargs["prior_beta"] = propensity_prior_beta
    elif propensity_method == "logistic":
        prop_kwargs["prior_sd"] = propensity_prior_sd
    # For "auto", pass both and let the function handle selection
    elif propensity_method == "auto":
        # Pass to the top-level function which will select appropriate method
        prop_kwargs["prior_alpha"] = propensity_prior_alpha
        prop_kwargs["prior_beta"] = propensity_prior_beta
        prop_kwargs["prior_sd"] = propensity_prior_sd

    prop_result = bayesian_propensity(treatment, covariates, **prop_kwargs)

    e_samples = prop_result["posterior_samples"]  # (n_samples, n)
    propensity_mean = prop_result["posterior_mean"]  # (n,)
    propensity_mean_uncertainty = prop_result["mean_uncertainty"]

    # ============================================================================
    # Step 2: Fit Outcome Models (Frequentist)
    # ============================================================================

    outcome_result = fit_outcome_models(outcomes, treatment, covariates)
    mu0 = outcome_result["mu0_predictions"]
    mu1 = outcome_result["mu1_predictions"]

    # Average R² across treated and control models
    outcome_r2 = (
        outcome_result["diagnostics"].get("r2_treated", 0.0)
        + outcome_result["diagnostics"].get("r2_control", 0.0)
    ) / 2

    # ============================================================================
    # Step 3: Propagate Propensity Uncertainty Through DR Formula
    # ============================================================================

    ate_samples = _bayesian_dr_core(
        outcomes=outcomes,
        treatment=treatment,
        e_samples=e_samples,
        mu0=mu0,
        mu1=mu1,
        trim_threshold=trim_threshold,
    )

    # ============================================================================
    # Step 4: Add Outcome Model Uncertainty
    # ============================================================================

    # The propensity-only posterior underestimates total uncertainty.
    # We add outcome model variance using the influence function approach.
    # This creates a hybrid Bayesian-frequentist uncertainty quantification.

    # Compute influence function residuals (outcome model contribution)
    propensity_mean_clipped = np.clip(propensity_mean, trim_threshold, 1 - trim_threshold)

    treated_contrib = (
        treatment / propensity_mean_clipped * (outcomes - mu1) + mu1
    )
    control_contrib = (
        (1 - treatment) / (1 - propensity_mean_clipped) * (outcomes - mu0) + mu0
    )

    dr_point_estimate = np.mean(treated_contrib - control_contrib)
    influence_function = treated_contrib - control_contrib - dr_point_estimate

    # Outcome model variance contribution (from influence function)
    outcome_variance = np.mean(influence_function**2) / n

    # Propensity uncertainty contribution (from posterior samples)
    propensity_variance = np.var(ate_samples)

    # Combine variances (avoid double-counting by taking max)
    # The outcome variance already includes some propensity contribution,
    # so we use a conservative combination
    total_variance = max(outcome_variance, propensity_variance + outcome_variance * 0.5)

    # Inflate posterior samples to reflect total uncertainty
    propensity_sd = np.std(ate_samples)
    target_sd = np.sqrt(total_variance)
    if propensity_sd > 0:
        inflation_factor = target_sd / propensity_sd
        ate_samples = np.mean(ate_samples) + (ate_samples - np.mean(ate_samples)) * inflation_factor

    # ============================================================================
    # Step 5: Summarize Posterior Distribution
    # ============================================================================

    posterior_mean = float(np.mean(ate_samples))
    posterior_sd = float(np.std(ate_samples))

    alpha = 1 - credible_level
    ci_lower = float(np.quantile(ate_samples, alpha / 2))
    ci_upper = float(np.quantile(ate_samples, 1 - alpha / 2))

    # ============================================================================
    # Step 5: Frequentist DR for Comparison
    # ============================================================================

    freq_result = dr_ate(outcomes, treatment, covariates)
    frequentist_estimate = freq_result["estimate"]
    frequentist_se = freq_result["se"]

    # ============================================================================
    # Step 6: Sample Sizes
    # ============================================================================

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    return BayesianDRResult(
        estimate=posterior_mean,
        se=posterior_sd,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        credible_level=credible_level,
        posterior_samples=ate_samples,
        n=n,
        n_treated=n_treated,
        n_control=n_control,
        propensity_mean=propensity_mean,
        propensity_mean_uncertainty=propensity_mean_uncertainty,
        outcome_r2=outcome_r2,
        frequentist_estimate=frequentist_estimate,
        frequentist_se=frequentist_se,
    )
