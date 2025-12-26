"""Bayesian Principal Stratification for CACE estimation.

This module provides Bayesian inference for the Complier Average Causal Effect (CACE)
using PyMC for MCMC sampling. The approach models principal strata as latent variables
and uses a marginalized likelihood for stable inference.

Key Features
------------
- Full posterior inference on CACE and strata proportions
- Marginalized likelihood (no direct sampling of discrete strata)
- MCMC diagnostics with automatic warnings
- Quick mode for fast exploratory analysis

Generative Model
----------------
Strata proportions: (π_c, π_a, π_n) ~ Dirichlet(α_c, α_a, α_n)
Outcome means:
  μ_c0 ~ Normal(0, σ_prior)  # Complier untreated
  μ_c1 ~ Normal(0, σ_prior)  # Complier treated
  μ_a  ~ Normal(0, σ_prior)  # Always-taker
  μ_n  ~ Normal(0, σ_prior)  # Never-taker
Noise: σ ~ HalfNormal(1)
Likelihood: Marginalized over strata based on (D_i, Z_i)

References
----------
- Imbens, G. W., & Rubin, D. B. (1997). Bayesian Inference for Causal Effects in
  Randomized Experiments with Noncompliance. Annals of Statistics, 25(1), 305-327.
- Hirano, K., Imbens, G. W., Rubin, D. B., & Zhou, X. H. (2000). Assessing the Effect
  of an Influenza Vaccine in an Encouragement Design. Biostatistics, 1(1), 69-88.
"""

import warnings
from typing import Tuple, Optional, Dict, Any

import numpy as np
from numpy.typing import ArrayLike

from .types import BayesianPSResult


def _check_pymc_installed() -> None:
    """Check if PyMC is installed, raise helpful error if not.

    Raises
    ------
    ImportError
        If PyMC is not installed, with installation instructions.
    """
    try:
        import pymc  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyMC required for Bayesian CACE. Install with:\n"
            "  pip install 'causal-inference-mastery[bayesian]'\n"
            "or:\n"
            "  pip install pymc>=5.10 arviz>=0.18"
        ) from e


def _emit_diagnostic_warnings(
    rhat: Dict[str, float],
    ess: Dict[str, float],
    divergences: int,
    rhat_threshold: float = 1.05,
    ess_threshold: float = 400,
) -> None:
    """Emit warnings if MCMC diagnostics indicate problems.

    Parameters
    ----------
    rhat : Dict[str, float]
        Gelman-Rubin R-hat statistics for each parameter.
    ess : Dict[str, float]
        Effective sample sizes for each parameter.
    divergences : int
        Number of divergent transitions.
    rhat_threshold : float
        Maximum acceptable R-hat (default 1.05).
    ess_threshold : float
        Minimum acceptable ESS (default 400).
    """
    max_rhat = max(rhat.values()) if rhat else 0.0
    min_ess = min(ess.values()) if ess else float("inf")

    if max_rhat > rhat_threshold:
        warnings.warn(
            f"R-hat {max_rhat:.3f} > {rhat_threshold}: chains may not have converged. "
            "Consider increasing n_samples or n_chains.",
            RuntimeWarning,
            stacklevel=3,
        )

    if min_ess < ess_threshold:
        warnings.warn(
            f"ESS {min_ess:.0f} < {ess_threshold}: low effective sample size. "
            "Consider increasing n_samples.",
            RuntimeWarning,
            stacklevel=3,
        )

    if divergences > 0:
        warnings.warn(
            f"MCMC had {divergences} divergent transitions. "
            "Results may be biased. Consider increasing target_accept.",
            RuntimeWarning,
            stacklevel=3,
        )


def _compute_log_likelihood_marginalized(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    log_pi_c: Any,
    log_pi_a: Any,
    log_pi_n: Any,
    mu_c0: Any,
    mu_c1: Any,
    mu_a: Any,
    mu_n: Any,
    sigma: Any,
) -> Any:
    """Compute marginalized log-likelihood over latent strata.

    For each observation, we compute P(Y|D,Z,θ) = Σ_s P(s|θ) P(Y|s,D,θ)

    The key insight is that (D,Z) patterns constrain possible strata:
    - D=1, Z=0: Must be always-taker (compliers don't take when Z=0)
    - D=0, Z=1: Must be never-taker (compliers take when Z=1)
    - D=1, Z=1: Could be complier or always-taker
    - D=0, Z=0: Could be complier or never-taker

    Parameters
    ----------
    Y, D, Z : np.ndarray
        Outcome, treatment, and instrument arrays.
    log_pi_c, log_pi_a, log_pi_n : PyMC tensor
        Log probabilities of each stratum.
    mu_c0, mu_c1, mu_a, mu_n : PyMC tensor
        Mean outcomes for each stratum-treatment combination.
    sigma : PyMC tensor
        Outcome standard deviation.

    Returns
    -------
    total_ll : PyMC tensor
        Total log-likelihood (summed over all observations).
    """
    import pymc as pm
    import pytensor.tensor as pt

    n = len(Y)
    ll = pt.zeros(n)

    # Masks for each (D, Z) pattern
    mask_d1_z0 = (D == 1) & (Z == 0)  # Always-takers only
    mask_d0_z1 = (D == 0) & (Z == 1)  # Never-takers only
    mask_d1_z1 = (D == 1) & (Z == 1)  # Compliers (treated) or always-takers
    mask_d0_z0 = (D == 0) & (Z == 0)  # Compliers (untreated) or never-takers

    # Log-likelihood for identified strata
    # D=1, Z=0: Must be always-taker
    ll = pt.set_subtensor(
        ll[mask_d1_z0],
        log_pi_a + pm.logp(pm.Normal.dist(mu=mu_a, sigma=sigma), Y[mask_d1_z0]),
    )

    # D=0, Z=1: Must be never-taker
    ll = pt.set_subtensor(
        ll[mask_d0_z1],
        log_pi_n + pm.logp(pm.Normal.dist(mu=mu_n, sigma=sigma), Y[mask_d0_z1]),
    )

    # D=1, Z=1: Mixture of compliers (treated) and always-takers
    ll_c_d1z1 = log_pi_c + pm.logp(pm.Normal.dist(mu=mu_c1, sigma=sigma), Y[mask_d1_z1])
    ll_a_d1z1 = log_pi_a + pm.logp(pm.Normal.dist(mu=mu_a, sigma=sigma), Y[mask_d1_z1])
    ll = pt.set_subtensor(ll[mask_d1_z1], pt.logaddexp(ll_c_d1z1, ll_a_d1z1))

    # D=0, Z=0: Mixture of compliers (untreated) and never-takers
    ll_c_d0z0 = log_pi_c + pm.logp(pm.Normal.dist(mu=mu_c0, sigma=sigma), Y[mask_d0_z0])
    ll_n_d0z0 = log_pi_n + pm.logp(pm.Normal.dist(mu=mu_n, sigma=sigma), Y[mask_d0_z0])
    ll = pt.set_subtensor(ll[mask_d0_z0], pt.logaddexp(ll_c_d0z0, ll_n_d0z0))

    return pt.sum(ll)


def _build_ps_model(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    prior_alpha: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    prior_mu_sd: float = 10.0,
) -> Any:
    """Build PyMC model for principal stratification.

    Uses marginalized likelihood for stable inference (no discrete latent strata sampling).

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable.
    D : np.ndarray
        Treatment received (binary).
    Z : np.ndarray
        Instrument/assignment (binary).
    prior_alpha : Tuple[float, float, float]
        Dirichlet prior parameters for (compliers, always-takers, never-takers).
        Default (1,1,1) is uniform.
    prior_mu_sd : float
        Prior SD for outcome means.

    Returns
    -------
    model : pm.Model
        PyMC model ready for sampling.
    """
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        # Strata proportions: Dirichlet prior
        # pi = [pi_c, pi_a, pi_n]
        pi = pm.Dirichlet("pi", a=np.array(prior_alpha))
        log_pi = pt.log(pi)

        # Outcome means for each stratum-treatment combination
        mu_c0 = pm.Normal("mu_c0", mu=0, sigma=prior_mu_sd)  # Complier untreated
        mu_c1 = pm.Normal("mu_c1", mu=0, sigma=prior_mu_sd)  # Complier treated
        mu_a = pm.Normal("mu_a", mu=0, sigma=prior_mu_sd)  # Always-taker (always D=1)
        mu_n = pm.Normal("mu_n", mu=0, sigma=prior_mu_sd)  # Never-taker (always D=0)

        # Outcome noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # CACE as derived quantity
        cace = pm.Deterministic("cace", mu_c1 - mu_c0)

        # Marginalized likelihood
        ll = _compute_log_likelihood_marginalized(
            Y,
            D,
            Z,
            log_pi[0],  # log(pi_c)
            log_pi[1],  # log(pi_a)
            log_pi[2],  # log(pi_n)
            mu_c0,
            mu_c1,
            mu_a,
            mu_n,
            sigma,
        )
        pm.Potential("likelihood", ll)

    return model


def cace_bayesian(
    outcome: ArrayLike,
    treatment: ArrayLike,
    instrument: ArrayLike,
    prior_alpha: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    prior_mu_sd: float = 10.0,
    n_samples: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    random_seed: Optional[int] = None,
    credible_level: float = 0.95,
    quick: bool = False,
) -> BayesianPSResult:
    """Estimate CACE using Bayesian inference with PyMC.

    Provides full posterior inference on the Complier Average Causal Effect (CACE)
    and principal strata proportions using MCMC sampling.

    Parameters
    ----------
    outcome : ArrayLike
        Outcome variable (continuous).
    treatment : ArrayLike
        Treatment received (binary 0/1).
    instrument : ArrayLike
        Instrument/randomization (binary 0/1).
    prior_alpha : Tuple[float, float, float]
        Dirichlet prior parameters for strata proportions (compliers, always-takers,
        never-takers). Default (1,1,1) is uniform.
    prior_mu_sd : float
        Prior standard deviation for outcome means. Default 10.0 (weakly informative).
    n_samples : int
        Number of posterior samples per chain. Default 2000.
    n_chains : int
        Number of MCMC chains. Default 4.
    target_accept : float
        Target acceptance rate for NUTS sampler. Default 0.9.
    random_seed : Optional[int]
        Random seed for reproducibility.
    credible_level : float
        Credible interval level (default 0.95 for 95% HDI).
    quick : bool
        If True, use fast settings (1000 samples, 2 chains, target_accept=0.8).
        Good for exploration, not for final inference.

    Returns
    -------
    BayesianPSResult
        Dictionary containing posterior summaries and samples.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    ValueError
        If inputs have invalid values.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> Z = np.random.binomial(1, 0.5, n)
    >>> # Compliance behavior
    >>> is_complier = np.random.binomial(1, 0.5, n)
    >>> is_always_taker = np.random.binomial(1, 0.2, n) * (1 - is_complier)
    >>> D = np.where(is_complier, Z, is_always_taker)
    >>> # Outcome
    >>> Y = 2.0 * D * is_complier + np.random.normal(0, 1, n)  # CACE = 2
    >>> result = cace_bayesian(Y, D, Z, quick=True, random_seed=42)
    >>> print(f"CACE: {result['cace_mean']:.2f} [{result['cace_hdi_lower']:.2f}, {result['cace_hdi_upper']:.2f}]")
    """
    _check_pymc_installed()
    import pymc as pm
    import arviz as az

    # Convert inputs
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    Z = np.asarray(instrument, dtype=float)

    # Input validation
    if len(Y) != len(D) or len(Y) != len(Z):
        raise ValueError(
            f"Length mismatch: outcome ({len(Y)}), treatment ({len(D)}), "
            f"instrument ({len(Z)}) must have same length"
        )

    if not np.all(np.isin(D, [0, 1])):
        raise ValueError("Treatment must be binary (0 or 1)")

    if not np.all(np.isin(Z, [0, 1])):
        raise ValueError("Instrument must be binary (0 or 1)")

    # Quick mode overrides
    if quick:
        n_samples = 1000
        n_chains = 2
        target_accept = 0.8

    # Build and sample model
    model = _build_ps_model(Y, D, Z, prior_alpha, prior_mu_sd)

    with model:
        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract posterior samples
    posterior = trace.posterior

    cace_samples = posterior["cace"].values.flatten()
    pi_samples = posterior["pi"].values.reshape(-1, 3)
    pi_c_samples = pi_samples[:, 0]
    pi_a_samples = pi_samples[:, 1]
    pi_n_samples = pi_samples[:, 2]

    # Compute summaries
    cace_mean = float(np.mean(cace_samples))
    cace_sd = float(np.std(cace_samples))

    # HDI (Highest Density Interval)
    alpha = 1 - credible_level
    cace_hdi = az.hdi(cace_samples, hdi_prob=credible_level)
    cace_hdi_lower = float(cace_hdi[0])
    cace_hdi_upper = float(cace_hdi[1])

    # Strata proportions
    pi_c_mean = float(np.mean(pi_c_samples))
    pi_a_mean = float(np.mean(pi_a_samples))
    pi_n_mean = float(np.mean(pi_n_samples))

    # MCMC diagnostics
    rhat_dict: Dict[str, float] = {}
    ess_dict: Dict[str, float] = {}

    summary = az.summary(trace, var_names=["cace", "pi", "mu_c0", "mu_c1", "mu_a", "mu_n", "sigma"])

    for var in ["cace", "pi[0]", "pi[1]", "pi[2]", "mu_c0", "mu_c1", "mu_a", "mu_n", "sigma"]:
        if var in summary.index:
            rhat_dict[var] = float(summary.loc[var, "r_hat"])
            ess_dict[var] = float(summary.loc[var, "ess_bulk"])

    # Count divergences
    divergences = int(trace.sample_stats["diverging"].sum().values)

    # Emit warnings if diagnostics are bad
    _emit_diagnostic_warnings(rhat_dict, ess_dict, divergences)

    # Construct stratum_probs (not easily available with marginalized likelihood)
    # We'll return empty array - full stratum membership would require Gibbs sampling
    stratum_probs = np.array([])

    return BayesianPSResult(
        cace_mean=cace_mean,
        cace_sd=cace_sd,
        cace_hdi_lower=cace_hdi_lower,
        cace_hdi_upper=cace_hdi_upper,
        cace_samples=cace_samples,
        pi_c_mean=pi_c_mean,
        pi_c_samples=pi_c_samples,
        pi_a_mean=pi_a_mean,
        pi_n_mean=pi_n_mean,
        stratum_probs=stratum_probs,
        rhat=rhat_dict,
        ess=ess_dict,
        n_samples=n_samples * n_chains,
        n_chains=n_chains,
        model="marginalized_likelihood_dirichlet_normal",
    )
