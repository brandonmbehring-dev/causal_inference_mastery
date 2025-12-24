"""
Bayesian Propensity Score Estimation.

Implements two approaches:
1. Stratified Beta-Binomial (conjugate, for discrete covariates)
2. Bayesian Logistic Regression (Laplace approximation, for continuous covariates)

Session 102: Initial implementation.

Mathematical Foundation
-----------------------
**Stratified Beta-Binomial**:

    Prior:      e_s ~ Beta(α₀, β₀) for each stratum s
    Likelihood: T_i | s ~ Bernoulli(e_s)
    Posterior:  e_s | T ~ Beta(α₀ + n_treated_s, β₀ + n_control_s)

**Bayesian Logistic Regression** (Laplace approximation):

    Prior:      β ~ N(0, σ²_prior I)
    Likelihood: T | X ~ Bernoulli(sigmoid(Xβ))
    Posterior:  β | T, X ≈ N(β_MLE, H⁻¹)

where H is the Hessian of the negative log-posterior at β_MLE.
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function

from causal_inference.bayesian.types import (
    BayesianPropensityResult,
    StratumInfo,
)


def _create_strata(
    covariates: NDArray[np.float64],
    n_bins: int = 5,
) -> Tuple[NDArray[np.int64], int]:
    """
    Create strata from covariates by discretizing continuous variables.

    Parameters
    ----------
    covariates : ndarray
        Covariate matrix, shape (n, p).
    n_bins : int
        Number of bins for each continuous covariate.

    Returns
    -------
    strata : ndarray
        Stratum assignment for each observation, shape (n,).
    n_strata : int
        Total number of unique strata.
    """
    n, p = covariates.shape

    # Discretize each covariate
    discretized = np.zeros((n, p), dtype=np.int64)
    for j in range(p):
        col = covariates[:, j]
        n_unique = len(np.unique(col))

        if n_unique <= n_bins:
            # Already discrete, use as-is
            _, discretized[:, j] = np.unique(col, return_inverse=True)
        else:
            # Discretize using quantile bins
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(col, percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            discretized[:, j] = np.digitize(col, bins[1:-1])

    # Create unique stratum for each combination
    # Use tuple hashing approach
    strata = np.zeros(n, dtype=np.int64)
    stratum_map = {}
    for i in range(n):
        key = tuple(discretized[i, :])
        if key not in stratum_map:
            stratum_map[key] = len(stratum_map)
        strata[i] = stratum_map[key]

    return strata, len(stratum_map)


def bayesian_propensity_stratified(
    treatment: NDArray[np.float64],
    covariates: NDArray[np.float64],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_bins: int = 5,
    n_posterior_samples: int = 1000,
) -> BayesianPropensityResult:
    """
    Bayesian propensity score estimation using stratified Beta-Binomial.

    For each stratum defined by covariate combinations, estimates propensity
    using Beta-Binomial conjugate model.

    Parameters
    ----------
    treatment : ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : ndarray
        Covariate matrix, shape (n, p).
    prior_alpha : float, default=1.0
        Beta prior alpha parameter (prior successes + 1).
    prior_beta : float, default=1.0
        Beta prior beta parameter (prior failures + 1).
        Default (1,1) is uniform prior.
    n_bins : int, default=5
        Number of bins for discretizing continuous covariates.
    n_posterior_samples : int, default=1000
        Number of posterior samples to draw.

    Returns
    -------
    BayesianPropensityResult
        Posterior propensity samples, means, and stratum information.

    Notes
    -----
    The Beta(1,1) prior is uniform on [0,1], representing no prior information.
    Beta(0.5, 0.5) is Jeffreys prior, which is weakly informative.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> T = (X[:, 0] > 0).astype(float)
    >>> result = bayesian_propensity_stratified(T, X)
    >>> print(f"Mean propensity: {result['posterior_mean'].mean():.3f}")
    """
    # Input validation
    treatment = np.asarray(treatment, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(treatment)

    if len(covariates) != n:
        raise ValueError(
            f"Length mismatch: treatment ({n}) != covariates ({len(covariates)})"
        )

    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError("Treatment must be binary (0 or 1)")

    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError(
            f"prior_alpha and prior_beta must be positive, "
            f"got alpha={prior_alpha}, beta={prior_beta}"
        )

    # Create strata
    strata, n_strata = _create_strata(covariates, n_bins)

    # Compute posterior for each stratum
    stratum_info = []
    posterior_samples = np.zeros((n_posterior_samples, n))

    for s in range(n_strata):
        mask = strata == s
        n_in_stratum = np.sum(mask)
        n_treated = np.sum(treatment[mask])
        n_control = n_in_stratum - n_treated

        # Posterior parameters
        post_alpha = prior_alpha + n_treated
        post_beta = prior_beta + n_control

        # Posterior mean
        post_mean = post_alpha / (post_alpha + post_beta)

        # Draw samples from posterior
        samples = np.random.beta(post_alpha, post_beta, size=n_posterior_samples)

        # Assign to all observations in this stratum
        posterior_samples[:, mask] = samples[:, np.newaxis]

        stratum_info.append(StratumInfo(
            stratum_id=s,
            n_obs=int(n_in_stratum),
            n_treated=int(n_treated),
            n_control=int(n_control),
            posterior_alpha=post_alpha,
            posterior_beta=post_beta,
            posterior_mean=post_mean,
            posterior_sd=np.sqrt(
                (post_alpha * post_beta) /
                ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
            ),
        ))

    # Compute summary statistics
    posterior_mean = np.mean(posterior_samples, axis=0)
    posterior_sd = np.std(posterior_samples, axis=0)

    # Uncertainty metrics
    mean_uncertainty = np.mean(posterior_sd)
    propensity_range = np.ptp(posterior_mean)

    return BayesianPropensityResult(
        posterior_samples=posterior_samples,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        strata=strata,
        n_strata=n_strata,
        stratum_info=stratum_info,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        method="stratified_beta_binomial",
        n=n,
        n_treated=int(np.sum(treatment)),
        n_control=int(n - np.sum(treatment)),
        mean_uncertainty=mean_uncertainty,
        propensity_range=propensity_range,
    )


def _neg_log_posterior_logistic(
    beta: NDArray[np.float64],
    X: NDArray[np.float64],
    T: NDArray[np.float64],
    prior_var: float,
) -> float:
    """Negative log-posterior for logistic regression."""
    linear = X @ beta
    # Clip for numerical stability
    linear = np.clip(linear, -500, 500)
    prob = expit(linear)

    # Log-likelihood
    ll = np.sum(T * np.log(prob + 1e-10) + (1 - T) * np.log(1 - prob + 1e-10))

    # Log-prior (normal)
    log_prior = -0.5 * np.sum(beta**2) / prior_var

    return -(ll + log_prior)


def _hessian_logistic(
    beta: NDArray[np.float64],
    X: NDArray[np.float64],
    prior_var: float,
) -> NDArray[np.float64]:
    """Hessian of negative log-posterior for logistic regression."""
    linear = X @ beta
    linear = np.clip(linear, -500, 500)
    prob = expit(linear)

    # Weight matrix
    W = prob * (1 - prob)

    # Hessian = X' diag(W) X + I/prior_var
    hessian = X.T @ (W[:, np.newaxis] * X) + np.eye(X.shape[1]) / prior_var

    return hessian


def bayesian_propensity_logistic(
    treatment: NDArray[np.float64],
    covariates: NDArray[np.float64],
    prior_sd: float = 10.0,
    n_posterior_samples: int = 1000,
    include_intercept: bool = True,
) -> BayesianPropensityResult:
    """
    Bayesian propensity score estimation using logistic regression with Laplace approximation.

    Uses normal approximation to posterior of logistic regression coefficients.

    Parameters
    ----------
    treatment : ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : ndarray
        Covariate matrix, shape (n, p).
    prior_sd : float, default=10.0
        Prior standard deviation for regression coefficients.
        Larger values = weaker prior.
    n_posterior_samples : int, default=1000
        Number of posterior samples to draw.
    include_intercept : bool, default=True
        Whether to include an intercept term.

    Returns
    -------
    BayesianPropensityResult
        Posterior propensity samples, means, and coefficient information.

    Notes
    -----
    The Laplace approximation uses a normal distribution centered at the
    maximum a posteriori (MAP) estimate with covariance given by the
    inverse Hessian of the negative log-posterior.

    This approximation is accurate when:
    - Sample size is reasonably large
    - Posterior is approximately normal (unimodal, symmetric)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.randn(n, 2)
    >>> logit = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    >>> T = (np.random.rand(n) < 1/(1 + np.exp(-logit))).astype(float)
    >>> result = bayesian_propensity_logistic(T, X)
    >>> print(f"Coefficient posterior means: {result['coefficient_mean']}")
    """
    # Input validation
    treatment = np.asarray(treatment, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape

    if len(treatment) != n:
        raise ValueError(
            f"Length mismatch: treatment ({len(treatment)}) != covariates ({n})"
        )

    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError("Treatment must be binary (0 or 1)")

    if prior_sd <= 0:
        raise ValueError(f"prior_sd must be positive, got {prior_sd}")

    # Add intercept if requested
    if include_intercept:
        X = np.column_stack([np.ones(n), covariates])
    else:
        X = covariates

    n_coef = X.shape[1]
    prior_var = prior_sd**2

    # Find MAP estimate
    beta_init = np.zeros(n_coef)
    result = minimize(
        _neg_log_posterior_logistic,
        beta_init,
        args=(X, treatment, prior_var),
        method="BFGS",
    )
    beta_map = result.x

    # Compute Hessian at MAP for Laplace approximation
    hessian = _hessian_logistic(beta_map, X, prior_var)
    try:
        cov_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        cov_matrix = np.linalg.pinv(hessian)

    # Draw coefficient samples from approximate posterior
    try:
        beta_samples = np.random.multivariate_normal(
            beta_map, cov_matrix, size=n_posterior_samples
        )
    except np.linalg.LinAlgError:
        # Fallback to independent normals if covariance is problematic
        stds = np.sqrt(np.diag(cov_matrix))
        beta_samples = np.random.normal(
            beta_map, stds, size=(n_posterior_samples, n_coef)
        )

    # Compute propensity samples
    linear_samples = beta_samples @ X.T  # (n_samples, n)
    propensity_samples = expit(linear_samples)

    # Summary statistics
    posterior_mean = np.mean(propensity_samples, axis=0)
    posterior_sd = np.std(propensity_samples, axis=0)

    # Coefficient summaries
    coefficient_mean = np.mean(beta_samples, axis=0)
    coefficient_sd = np.std(beta_samples, axis=0)

    # Uncertainty metrics
    mean_uncertainty = np.mean(posterior_sd)
    propensity_range = np.ptp(posterior_mean)

    return BayesianPropensityResult(
        posterior_samples=propensity_samples,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        strata=None,
        n_strata=0,
        stratum_info=None,
        prior_alpha=0.0,
        prior_beta=0.0,
        method="logistic_laplace",
        n=n,
        n_treated=int(np.sum(treatment)),
        n_control=int(n - np.sum(treatment)),
        mean_uncertainty=mean_uncertainty,
        propensity_range=propensity_range,
        coefficient_mean=coefficient_mean,
        coefficient_sd=coefficient_sd,
        coefficient_samples=beta_samples,
        prior_sd=prior_sd,
    )


def bayesian_propensity(
    treatment: NDArray[np.float64],
    covariates: NDArray[np.float64],
    method: str = "auto",
    **kwargs,
) -> BayesianPropensityResult:
    """
    Bayesian propensity score estimation.

    Automatically selects between stratified Beta-Binomial and logistic
    regression based on covariate characteristics.

    Parameters
    ----------
    treatment : ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : ndarray
        Covariate matrix, shape (n, p).
    method : str, default="auto"
        Estimation method:
        - "auto": Choose based on covariate characteristics
        - "stratified": Beta-Binomial stratified estimation
        - "logistic": Logistic regression with Laplace approximation
    **kwargs
        Additional arguments passed to the specific method.

    Returns
    -------
    BayesianPropensityResult
        Posterior propensity samples and summary statistics.
    """
    covariates = np.asarray(covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    if method == "auto":
        # Heuristic: use stratified if few unique values per covariate
        n_unique = [len(np.unique(covariates[:, j])) for j in range(covariates.shape[1])]
        if all(nu <= 10 for nu in n_unique):
            method = "stratified"
        else:
            method = "logistic"

    # Filter kwargs to only include parameters accepted by the selected method
    stratified_params = {"prior_alpha", "prior_beta", "n_bins", "n_posterior_samples"}
    logistic_params = {"prior_sd", "n_posterior_samples", "include_intercept"}

    if method == "stratified":
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in stratified_params}
        return bayesian_propensity_stratified(treatment, covariates, **filtered_kwargs)
    elif method == "logistic":
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in logistic_params}
        return bayesian_propensity_logistic(treatment, covariates, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'stratified', or 'logistic'.")
