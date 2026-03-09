"""
Hierarchical Bayesian ATE Estimation with MCMC.

Implements partial pooling across groups/sites for multi-site studies.
Uses PyMC for MCMC sampling via the NUTS algorithm.

Session 104: Initial implementation.

References
----------
- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and
  Multilevel/Hierarchical Models. Cambridge University Press.
- Stan Development Team (2023). Stan User's Guide: Hierarchical Models.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .types import HierarchicalATEResult


def _check_pymc_installed() -> None:
    """Check if PyMC is installed and raise informative error if not."""
    try:
        import pymc  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyMC is required for hierarchical Bayesian models. "
            "Install with: pip install 'causal-inference-mastery[mcmc]' "
            "or: pip install pymc arviz"
        ) from e


def hierarchical_bayesian_ate(
    outcomes: NDArray[np.float64],
    treatment: NDArray[np.float64],
    groups: NDArray[Any],
    *,
    # Prior hyperparameters
    mu_prior_sd: float = 10.0,
    tau_prior_sd: float = 5.0,
    sigma_prior_sd: float = 10.0,
    # MCMC settings
    n_samples: int = 2000,
    n_chains: int = 4,
    n_tune: int = 1000,
    target_accept: float = 0.9,
    # Output
    credible_level: float = 0.95,
    random_seed: int | None = None,
    progressbar: bool = True,
) -> HierarchicalATEResult:
    """
    Hierarchical Bayesian ATE with partial pooling across groups.

    Uses a hierarchical model with non-centered parameterization and
    MCMC sampling via PyMC's NUTS algorithm.

    Parameters
    ----------
    outcomes : NDArray[np.float64]
        Observed outcomes, shape (n,).
    treatment : NDArray[np.float64]
        Binary treatment indicator (0/1), shape (n,).
    groups : NDArray
        Group identifiers for each observation, shape (n,).
        Can be integers, strings, or any hashable type.
    mu_prior_sd : float, default 10.0
        Prior SD for population-level ATE.
    tau_prior_sd : float, default 5.0
        Prior SD for between-group heterogeneity (tau).
    sigma_prior_sd : float, default 10.0
        Prior SD for observation-level noise.
    n_samples : int, default 2000
        Number of MCMC samples per chain (after tuning).
    n_chains : int, default 4
        Number of parallel MCMC chains.
    n_tune : int, default 1000
        Number of tuning (warmup) samples per chain.
    target_accept : float, default 0.9
        Target acceptance rate for NUTS sampler.
    credible_level : float, default 0.95
        Credible interval level.
    random_seed : int or None, default None
        Random seed for reproducibility.
    progressbar : bool, default True
        Whether to show MCMC progress bar.

    Returns
    -------
    HierarchicalATEResult
        Contains population ATE, group-specific ATEs, heterogeneity (tau),
        full posterior samples, and MCMC diagnostics.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    ValueError
        If inputs are invalid (length mismatch, non-binary treatment, etc.).

    Notes
    -----
    **Model Structure** (non-centered parameterization):

        μ ~ Normal(0, mu_prior_sd)           # Population ATE
        τ ~ HalfNormal(tau_prior_sd)         # Between-group SD
        θ_raw_j ~ Normal(0, 1)               # Standardized group effects
        θⱼ = μ + τ × θ_raw_j                 # Group-specific ATE
        α ~ Normal(0, 10)                    # Intercept
        σ ~ HalfNormal(sigma_prior_sd)       # Observation noise
        Yᵢⱼ ~ Normal(α + θⱼ × Tᵢⱼ, σ)

    **Partial Pooling**:

    - Small groups shrink toward population mean
    - Large groups retain more group-specific signal
    - τ (heterogeneity) estimated from data

    **MCMC Diagnostics**:

    - R-hat < 1.05: Good chain convergence
    - ESS > 400: Adequate effective samples
    - Divergences = 0: NUTS sampler healthy

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 300
    >>> groups = np.repeat(np.arange(5), 60)  # 5 groups, 60 obs each
    >>> treatment = np.random.binomial(1, 0.5, n).astype(float)
    >>> true_effects = [1.0, 1.5, 2.0, 2.5, 3.0]  # Group effects
    >>> outcomes = np.array([true_effects[g] for g in groups]) * treatment + np.random.randn(n)
    >>> result = hierarchical_bayesian_ate(outcomes, treatment, groups, n_samples=500)
    >>> print(f"Population ATE: {result['population_ate']:.3f}")
    """
    # Check PyMC availability
    _check_pymc_installed()

    import arviz as az
    import pymc as pm

    # ==========================================================================
    # Input Validation
    # ==========================================================================

    Y = np.asarray(outcomes, dtype=np.float64)
    T = np.asarray(treatment, dtype=np.float64)
    G = np.asarray(groups)

    n = len(Y)

    if len(T) != n:
        raise ValueError(f"Length mismatch: outcomes ({n}) != treatment ({len(T)})")

    if len(G) != n:
        raise ValueError(f"Length mismatch: outcomes ({n}) != groups ({len(G)})")

    if not np.all(np.isin(T, [0, 1])):
        raise ValueError("Treatment must be binary (0 or 1)")

    if not (0 < credible_level < 1):
        raise ValueError("credible_level must be in (0, 1)")

    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")

    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")

    # Encode groups as integers
    unique_groups = np.unique(G)
    n_groups = len(unique_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    group_idx = np.array([group_to_idx[g] for g in G])

    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups for hierarchical model, got {n_groups}. "
            "For single-group analysis, use bayesian_ate() instead."
        )

    # ==========================================================================
    # Build PyMC Model
    # ==========================================================================

    with pm.Model() as hierarchical_model:
        # --- Population-level priors ---
        mu = pm.Normal("mu", mu=0, sigma=mu_prior_sd)
        tau = pm.HalfNormal("tau", sigma=tau_prior_sd)
        sigma = pm.HalfNormal("sigma", sigma=sigma_prior_sd)
        alpha = pm.Normal("alpha", mu=0, sigma=10)

        # --- Group-level effects (non-centered parameterization) ---
        # This avoids funnel geometry issues in hierarchical models
        theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=n_groups)
        theta = pm.Deterministic("theta", mu + tau * theta_raw)

        # --- Likelihood ---
        mu_obs = alpha + theta[group_idx] * T
        pm.Normal("Y_obs", mu=mu_obs, sigma=sigma, observed=Y)

    # ==========================================================================
    # Run MCMC Sampling
    # ==========================================================================

    with hierarchical_model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    # ==========================================================================
    # Extract Results
    # ==========================================================================

    # Get posterior samples
    mu_samples = trace.posterior["mu"].values.flatten()
    tau_samples = trace.posterior["tau"].values.flatten()
    theta_samples = trace.posterior["theta"].values.reshape(-1, n_groups)

    # Compute credible intervals
    alpha_ci = (1 - credible_level) / 2

    # Population ATE
    population_ate = float(np.mean(mu_samples))
    population_ate_se = float(np.std(mu_samples))
    population_ate_ci_lower = float(np.percentile(mu_samples, alpha_ci * 100))
    population_ate_ci_upper = float(np.percentile(mu_samples, (1 - alpha_ci) * 100))

    # Heterogeneity (tau)
    tau_mean = float(np.mean(tau_samples))
    tau_ci_lower = float(np.percentile(tau_samples, alpha_ci * 100))
    tau_ci_upper = float(np.percentile(tau_samples, (1 - alpha_ci) * 100))

    # Group-specific ATEs
    group_ates = np.mean(theta_samples, axis=0)
    group_ate_ses = np.std(theta_samples, axis=0)

    # ==========================================================================
    # MCMC Diagnostics
    # ==========================================================================

    # R-hat (should be < 1.05)
    rhat = az.rhat(trace)
    rhat_values = []
    for var in ["mu", "tau", "sigma", "alpha"]:
        if var in rhat:
            val = rhat[var].values
            if np.isscalar(val):
                rhat_values.append(float(val))
            else:
                rhat_values.extend(val.flatten().tolist())
    if "theta" in rhat:
        rhat_values.extend(rhat["theta"].values.flatten().tolist())
    rhat_max = float(np.max(rhat_values)) if rhat_values else np.nan

    # Effective sample size (should be > 400)
    ess = az.ess(trace)
    ess_values = []
    for var in ["mu", "tau", "sigma", "alpha"]:
        if var in ess:
            val = ess[var].values
            if np.isscalar(val):
                ess_values.append(float(val))
            else:
                ess_values.extend(val.flatten().tolist())
    if "theta" in ess:
        ess_values.extend(ess["theta"].values.flatten().tolist())
    ess_min = float(np.min(ess_values)) if ess_values else np.nan

    # Divergences (should be 0)
    divergences = int(trace.sample_stats["diverging"].sum())

    # ==========================================================================
    # Build Result
    # ==========================================================================

    result: HierarchicalATEResult = {
        # Population-level estimates
        "population_ate": population_ate,
        "population_ate_se": population_ate_se,
        "population_ate_ci_lower": population_ate_ci_lower,
        "population_ate_ci_upper": population_ate_ci_upper,
        # Group-specific estimates
        "group_ates": group_ates,
        "group_ate_ses": group_ate_ses,
        "group_ids": list(unique_groups),
        # Heterogeneity
        "tau": tau_mean,
        "tau_ci_lower": tau_ci_lower,
        "tau_ci_upper": tau_ci_upper,
        # Full posterior samples
        "posterior_samples": {
            "mu": mu_samples,
            "tau": tau_samples,
            "theta": theta_samples,
        },
        # Metadata
        "n_groups": n_groups,
        "n_obs": n,
        "credible_level": credible_level,
        # MCMC diagnostics
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }

    return result
