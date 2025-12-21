"""
Data Generating Processes for Quantile Treatment Effects Monte Carlo validation.

Provides DGPs with known true QTEs for validating estimator bias and coverage.
"""

from typing import Tuple, Optional
import numpy as np


def generate_homogeneous_qte_dgp(
    n: int = 500,
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    treatment_prob: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate data with homogeneous treatment effect across all quantiles.

    DGP: Y = true_ate * T + noise
    where noise ~ N(0, noise_sd^2)

    With this DGP:
    - QTE(tau) = true_ate for all tau in (0, 1)
    - Treatment shifts the entire distribution by true_ate

    Parameters
    ----------
    n : int
        Sample size
    true_ate : float
        True average treatment effect (constant across quantiles)
    noise_sd : float
        Standard deviation of noise
    treatment_prob : float
        Probability of treatment
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator
    true_qte : float
        True QTE (same at all quantiles for this DGP)
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)
    noise = rng.normal(0, noise_sd, n)
    outcome = true_ate * treatment + noise

    return outcome, treatment, true_ate


def generate_heterogeneous_qte_dgp(
    n: int = 500,
    base_effect: float = 2.0,
    heterogeneity: float = 1.0,
    treatment_prob: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate data with heterogeneous treatment effects across quantiles.

    DGP: Y = base_effect * T + heterogeneity * T * U + noise
    where U ~ Uniform(0, 1) is latent type, noise ~ N(0, 1)

    This creates QTE heterogeneity:
    - Lower quantiles: smaller effect
    - Higher quantiles: larger effect

    Parameters
    ----------
    n : int
        Sample size
    base_effect : float
        Base treatment effect
    heterogeneity : float
        Degree of heterogeneity across quantiles
    treatment_prob : float
        Probability of treatment
    seed : int, optional
        Random seed

    Returns
    -------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator
    true_qtes : dict
        Dictionary mapping quantiles to true QTEs
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)
    latent_type = rng.uniform(0, 1, n)
    noise = rng.normal(0, 1, n)

    # Outcome with heterogeneous effect
    outcome = base_effect * treatment + heterogeneity * treatment * latent_type + noise

    # True QTEs at standard quantiles
    # At quantile tau, the marginal individual has type ~tau
    true_qtes = {}
    for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
        true_qtes[tau] = base_effect + heterogeneity * tau

    return outcome, treatment, true_qtes


def generate_qte_with_covariates_dgp(
    n: int = 500,
    p: int = 3,
    true_ate: float = 2.0,
    covariate_effect: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate data with covariates for conditional QTE estimation.

    DGP: Y = true_ate * T + covariate_effect * sum(X) + noise

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of covariates
    true_ate : float
        True treatment effect
    covariate_effect : float
        Effect of each covariate
    seed : int, optional
        Random seed

    Returns
    -------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator
    covariates : np.ndarray
        Covariate matrix (n x p)
    true_qte : float
        True QTE (constant in this DGP)
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, 0.5, n).astype(float)
    covariates = rng.normal(0, 1, (n, p))
    noise = rng.normal(0, 1, n)

    covariate_sum = covariates.sum(axis=1)
    outcome = true_ate * treatment + covariate_effect * covariate_sum + noise

    return outcome, treatment, covariates, true_ate


def generate_location_scale_shift_dgp(
    n: int = 500,
    location_shift: float = 2.0,
    scale_ratio: float = 1.5,
    treatment_prob: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate data with location-scale treatment effect.

    Control: Y0 ~ N(0, 1)
    Treated: Y1 ~ N(location_shift, scale_ratio^2)

    This creates asymmetric QTEs:
    - Lower quantiles: location_shift - (scale_ratio - 1) * |z_tau|
    - Higher quantiles: location_shift + (scale_ratio - 1) * z_tau

    Parameters
    ----------
    n : int
        Sample size
    location_shift : float
        Mean shift from treatment
    scale_ratio : float
        Ratio of treated to control variance
    treatment_prob : float
        Probability of treatment
    seed : int, optional
        Random seed

    Returns
    -------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator
    true_qtes : dict
        Dictionary mapping quantiles to true QTEs
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)
    n_treated = int(treatment.sum())
    n_control = n - n_treated

    # Generate outcomes from different distributions
    y_control = rng.normal(0, 1, n_control)
    y_treated = rng.normal(location_shift, scale_ratio, n_treated)

    # Combine
    outcome = np.empty(n)
    outcome[treatment == 0] = y_control
    outcome[treatment == 1] = y_treated

    # True QTEs: Q_tau(Y1) - Q_tau(Y0)
    # Q_tau(Y0) = Phi^{-1}(tau)
    # Q_tau(Y1) = location_shift + scale_ratio * Phi^{-1}(tau)
    true_qtes = {}
    for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
        z_tau = norm.ppf(tau)
        q0 = z_tau  # N(0,1) quantile
        q1 = location_shift + scale_ratio * z_tau
        true_qtes[tau] = q1 - q0

    return outcome, treatment, true_qtes


def generate_extreme_quantile_dgp(
    n: int = 1000,
    true_ate: float = 2.0,
    tail_weight: float = 3.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate data with heavy tails for extreme quantile testing.

    Uses t-distribution with specified degrees of freedom.

    Parameters
    ----------
    n : int
        Sample size (should be large for extreme quantiles)
    true_ate : float
        True average treatment effect
    tail_weight : float
        Degrees of freedom for t-distribution (lower = heavier tails)
    seed : int, optional
        Random seed

    Returns
    -------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator
    true_qte : float
        True QTE (constant for this DGP)
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, 0.5, n).astype(float)
    # t-distribution noise for heavy tails
    noise = rng.standard_t(tail_weight, n)
    outcome = true_ate * treatment + noise

    return outcome, treatment, true_ate
