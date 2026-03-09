"""
Test fixtures for Quantile Treatment Effects (QTE) module.

Provides data generating processes (DGPs) for testing QTE estimators
with known true values for validation.
"""

from typing import Tuple

import numpy as np
import pytest


def generate_homogeneous_qte_dgp(
    n: int = 500,
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    treatment_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate RCT data with homogeneous treatment effect.

    For homogeneous effects, QTE is the same at all quantiles.

    DGP:
    - T ~ Bernoulli(treatment_prob)
    - Y = tau * T + epsilon, where epsilon ~ N(0, noise_sd^2)

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True treatment effect (constant across quantiles).
    noise_sd : float
        Standard deviation of noise.
    treatment_prob : float
        Probability of treatment.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (outcome, treatment) arrays.
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)
    noise = rng.normal(0, noise_sd, n)
    outcome = true_ate * treatment + noise

    return outcome, treatment


def generate_heterogeneous_qte_dgp(
    n: int = 500,
    base_effect: float = 1.0,
    heterogeneity: float = 1.0,
    noise_sd: float = 1.0,
    treatment_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate RCT data with heterogeneous treatment effect across quantiles.

    Treatment effect varies with individual-level heterogeneity:
    tau_i = base_effect + heterogeneity * u_i

    This creates different QTEs at different quantiles.

    DGP:
    - T ~ Bernoulli(treatment_prob)
    - u ~ N(0, 1) (individual heterogeneity)
    - Y = (base_effect + heterogeneity * u) * T + u

    At quantile tau, the QTE is approximately:
    QTE(tau) = base_effect + heterogeneity * Phi^{-1}(tau)

    Parameters
    ----------
    n : int
        Sample size.
    base_effect : float
        Baseline treatment effect (effect at median).
    heterogeneity : float
        Degree of heterogeneity (effect variation).
    noise_sd : float
        Noise standard deviation.
    treatment_prob : float
        Probability of treatment.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (outcome, treatment) arrays.
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)

    # Individual heterogeneity
    u = rng.normal(0, 1, n)

    # Heterogeneous treatment effect
    tau_i = base_effect + heterogeneity * u

    # Outcome: baseline + heterogeneous effect * treatment
    outcome = u + tau_i * treatment

    return outcome, treatment


def generate_qte_dgp_with_covariates(
    n: int = 500,
    p: int = 3,
    true_ate: float = 2.0,
    covariate_effects: float = 0.5,
    noise_sd: float = 1.0,
    treatment_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate RCT data with covariates.

    DGP:
    - X ~ N(0, I_p)
    - T ~ Bernoulli(treatment_prob)
    - Y = tau * T + beta' * X + epsilon

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    true_ate : float
        True treatment effect.
    covariate_effects : float
        Coefficient for each covariate.
    noise_sd : float
        Noise standard deviation.
    treatment_prob : float
        Probability of treatment.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (outcome, treatment, covariates) arrays.
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, treatment_prob, n).astype(float)
    covariates = rng.normal(0, 1, (n, p))
    noise = rng.normal(0, noise_sd, n)

    # Outcome with covariate effects
    outcome = true_ate * treatment + covariate_effects * covariates.sum(axis=1) + noise

    return outcome, treatment, covariates


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_rct_data():
    """Simple RCT with homogeneous effect of 2.0."""
    return generate_homogeneous_qte_dgp(n=500, true_ate=2.0, seed=42)


@pytest.fixture
def large_rct_data():
    """Larger RCT for more precise estimates."""
    return generate_homogeneous_qte_dgp(n=2000, true_ate=2.0, seed=42)


@pytest.fixture
def heterogeneous_data():
    """RCT with heterogeneous treatment effects."""
    return generate_heterogeneous_qte_dgp(n=500, base_effect=1.0, heterogeneity=1.0, seed=42)


@pytest.fixture
def data_with_covariates():
    """RCT data with 3 covariates."""
    return generate_qte_dgp_with_covariates(n=500, p=3, true_ate=2.0, seed=42)


@pytest.fixture
def small_sample_data():
    """Small sample RCT (n=50)."""
    return generate_homogeneous_qte_dgp(n=50, true_ate=2.0, seed=42)


@pytest.fixture
def known_quantile_data():
    """
    Data with known exact quantiles for validation.

    Treated: outcomes uniformly spaced [3, 4, 5, 6, 7]
    Control: outcomes uniformly spaced [1, 2, 3, 4, 5]

    Median treated = 5, Median control = 3
    Expected QTE(0.5) = 2.0
    """
    treatment = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    outcome = np.array([3, 4, 5, 6, 7, 1, 2, 3, 4, 5], dtype=float)
    return outcome, treatment


@pytest.fixture
def zero_effect_data():
    """Data with true zero treatment effect."""
    return generate_homogeneous_qte_dgp(n=500, true_ate=0.0, seed=42)


@pytest.fixture
def negative_effect_data():
    """Data with negative treatment effect."""
    return generate_homogeneous_qte_dgp(n=500, true_ate=-1.5, seed=42)


@pytest.fixture
def imbalanced_treatment_data():
    """Data with imbalanced treatment (80% treated)."""
    return generate_homogeneous_qte_dgp(n=500, true_ate=2.0, treatment_prob=0.8, seed=42)


@pytest.fixture
def high_noise_data():
    """Data with high noise (lower signal-to-noise ratio)."""
    return generate_homogeneous_qte_dgp(n=500, true_ate=1.0, noise_sd=3.0, seed=42)


# =============================================================================
# ADVERSARIAL FIXTURES
# =============================================================================


@pytest.fixture
def empty_arrays():
    """Empty arrays for error testing."""
    return {"outcome": np.array([]), "treatment": np.array([])}


@pytest.fixture
def nan_in_outcome():
    """Outcome array with NaN values."""
    outcome = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    treatment = np.array([1, 1, 0, 0, 0], dtype=float)
    return outcome, treatment


@pytest.fixture
def inf_in_outcome():
    """Outcome array with infinite values."""
    outcome = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
    treatment = np.array([1, 1, 0, 0, 0], dtype=float)
    return outcome, treatment


@pytest.fixture
def non_binary_treatment():
    """Non-binary treatment values."""
    outcome = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = np.array([0, 1, 2, 0, 1], dtype=float)  # Invalid: contains 2
    return outcome, treatment


@pytest.fixture
def no_variation_treatment():
    """Treatment with no variation (all treated)."""
    outcome = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = np.array([1, 1, 1, 1, 1], dtype=float)
    return outcome, treatment


@pytest.fixture
def length_mismatch():
    """Arrays with different lengths."""
    outcome = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = np.array([1, 0, 1], dtype=float)  # Wrong length
    return outcome, treatment


@pytest.fixture
def minimal_sample():
    """Minimal sample (2 treated, 2 control)."""
    outcome = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    treatment = np.array([1, 1, 0, 0], dtype=float)
    return outcome, treatment
