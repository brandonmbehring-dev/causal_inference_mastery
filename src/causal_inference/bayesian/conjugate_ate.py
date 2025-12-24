"""
Bayesian ATE Estimation with Conjugate Priors.

Implements closed-form posterior computation using the normal-normal
conjugate model. No MCMC required.

Session 101: Initial implementation.

Mathematical Foundation
-----------------------
The conjugate normal-normal model:

    Prior:      tau ~ N(mu_0, sigma_0^2)
    Likelihood: Y | T, X ~ N(alpha + tau*T + X*beta, sigma^2)
    Posterior:  tau | Y, T, X ~ N(mu_n, sigma_n^2)  [closed form]

Where:
    mu_n = sigma_n^2 * (mu_0/sigma_0^2 + tau_ols/var_tau_ols)
    sigma_n^2 = 1 / (1/sigma_0^2 + 1/var_tau_ols)

The posterior is a precision-weighted average of prior and likelihood.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from causal_inference.bayesian.types import BayesianATEResult


def _compute_conjugate_posterior(
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    X: Optional[NDArray[np.float64]],
    prior_mean: float,
    prior_var: float,
) -> Tuple[float, float, float, float, float]:
    """
    Compute posterior mean and variance using conjugate formulas.

    Parameters
    ----------
    Y : ndarray
        Outcomes, shape (n,).
    T : ndarray
        Treatment indicators (0/1), shape (n,).
    X : ndarray or None
        Covariates, shape (n, p) or None.
    prior_mean : float
        Prior mean for treatment effect.
    prior_var : float
        Prior variance for treatment effect.

    Returns
    -------
    post_mean : float
        Posterior mean of treatment effect.
    post_var : float
        Posterior variance of treatment effect.
    tau_ols : float
        OLS estimate of treatment effect.
    var_tau_ols : float
        Variance of OLS estimate.
    sigma2_mle : float
        MLE estimate of residual variance.
    """
    n = len(Y)

    # Build design matrix: [intercept, treatment, covariates]
    if X is not None:
        design = np.column_stack([np.ones(n), T, X])
    else:
        design = np.column_stack([np.ones(n), T])

    # OLS estimates (MLE)
    # Use lstsq for numerical stability
    beta_ols, residuals, rank, s = np.linalg.lstsq(design, Y, rcond=None)

    # Residual variance (MLE with bias correction)
    fitted = design @ beta_ols
    residuals_vec = Y - fitted
    df = n - design.shape[1]
    sigma2_mle = np.sum(residuals_vec**2) / df

    # Treatment coefficient is second element (after intercept)
    tau_ols = beta_ols[1]

    # Variance of OLS estimate for tau
    # Var(tau_ols) = sigma^2 * (X'X)^{-1}[1,1]
    XtX = design.T @ design
    XtX_inv = np.linalg.inv(XtX)
    var_tau_ols = sigma2_mle * XtX_inv[1, 1]

    # Conjugate posterior update
    # Posterior precision = prior precision + likelihood precision
    prior_prec = 1.0 / prior_var
    lik_prec = 1.0 / var_tau_ols
    post_prec = prior_prec + lik_prec
    post_var = 1.0 / post_prec

    # Posterior mean = weighted average of prior and likelihood
    post_mean = post_var * (prior_mean * prior_prec + tau_ols * lik_prec)

    return post_mean, post_var, tau_ols, var_tau_ols, sigma2_mle


def bayesian_ate(
    outcomes: NDArray[np.float64],
    treatment: NDArray[np.float64],
    covariates: Optional[NDArray[np.float64]] = None,
    prior_mean: float = 0.0,
    prior_sd: float = 10.0,
    credible_level: float = 0.95,
    n_posterior_samples: int = 5000,
) -> BayesianATEResult:
    """
    Bayesian estimation of Average Treatment Effect.

    Uses normal-normal conjugate prior for closed-form posterior.
    No MCMC required.

    Parameters
    ----------
    outcomes : ndarray
        Observed outcomes Y, shape (n,).
    treatment : ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : ndarray, optional
        Covariate matrix X, shape (n, p). If None, simple difference.
    prior_mean : float, default=0.0
        Prior mean for treatment effect.
        Default: 0 (centered on no effect).
    prior_sd : float, default=10.0
        Prior standard deviation.
        Default: 10 (weakly informative).
    credible_level : float, default=0.95
        Credible interval level.
    n_posterior_samples : int, default=5000
        Number of posterior samples to draw.

    Returns
    -------
    BayesianATEResult
        Posterior mean, SD, credible interval, samples, and diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid (length mismatch, non-binary treatment, etc.).

    Notes
    -----
    The estimator uses the conjugate normal-normal model:

        Prior:      tau ~ N(prior_mean, prior_sd^2)
        Likelihood: Y | T, X ~ N(alpha + tau*T + X*beta, sigma^2)
        Posterior:  tau | Y, T, X ~ N(mu_post, sigma_post^2)

    The posterior is computed in closed form (no MCMC needed).

    **Credible Intervals**:

    The credible interval has a direct probability interpretation:
    P(tau in [ci_lower, ci_upper] | data) = credible_level.

    This differs from frequentist confidence intervals.

    **Prior Sensitivity**:

    - Weak prior (large prior_sd): Data dominates, posterior near OLS
    - Strong prior (small prior_sd): Prior influences posterior more
    - Default prior_sd=10 is weakly informative for standardized outcomes

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> outcomes = 2.0 * treatment + np.random.normal(0, 1, n)
    >>> result = bayesian_ate(outcomes, treatment)
    >>> print(f"Posterior mean: {result['posterior_mean']:.3f}")
    >>> print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    """
    # Input validation
    outcomes = np.asarray(outcomes, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    if len(outcomes) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcomes ({len(outcomes)}) != treatment ({len(treatment)})"
        )

    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError("Treatment must be binary (0 or 1)")

    if prior_sd <= 0:
        raise ValueError(f"prior_sd must be positive, got {prior_sd}")

    if not 0 < credible_level < 1:
        raise ValueError(f"credible_level must be in (0, 1), got {credible_level}")

    if n_posterior_samples < 1:
        raise ValueError(f"n_posterior_samples must be >= 1, got {n_posterior_samples}")

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if len(covariates) != len(outcomes):
            raise ValueError(
                f"Length mismatch: covariates ({len(covariates)}) != outcomes ({len(outcomes)})"
            )

    n = len(outcomes)
    n_treated = int(np.sum(treatment))
    n_control = n - n_treated

    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Both treatment groups must be non-empty. "
            f"Got n_treated={n_treated}, n_control={n_control}"
        )

    # Compute conjugate posterior
    prior_var = prior_sd**2
    post_mean, post_var, tau_ols, var_tau_ols, sigma2_mle = _compute_conjugate_posterior(
        outcomes, treatment, covariates, prior_mean, prior_var
    )
    post_sd = np.sqrt(post_var)

    # Credible interval
    alpha = 1 - credible_level
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ci_lower = post_mean - z_alpha * post_sd
    ci_upper = post_mean + z_alpha * post_sd

    # Draw posterior samples
    posterior_samples = np.random.normal(post_mean, post_sd, size=n_posterior_samples)

    # Compute diagnostics
    # Prior shrinkage: proportion of posterior precision from prior
    prior_prec = 1.0 / prior_var
    lik_prec = 1.0 / var_tau_ols
    total_prec = prior_prec + lik_prec
    prior_to_posterior_shrinkage = prior_prec / total_prec

    # Effective sample size for treatment effect
    # ESS = n * (1 - shrinkage), approximating information from data
    effective_sample_size = n * (1 - prior_to_posterior_shrinkage)

    return BayesianATEResult(
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        credible_level=credible_level,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        posterior_samples=posterior_samples,
        n=n,
        n_treated=n_treated,
        n_control=n_control,
        prior_to_posterior_shrinkage=prior_to_posterior_shrinkage,
        effective_sample_size=effective_sample_size,
        ols_estimate=tau_ols,
        ols_se=np.sqrt(var_tau_ols),
        sigma2_mle=sigma2_mle,
    )
