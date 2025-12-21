"""
Pytest fixtures for Manski bounds tests.

Provides DGP generators for various treatment effect scenarios.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_rct_data():
    """Simple RCT data with known ATE = 2.0."""
    np.random.seed(42)
    n = 1000
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + np.random.randn(n)
    return outcome, treatment, 2.0  # outcome, treatment, true_ate


@pytest.fixture
def positive_selection_data():
    """
    Data with positive selection bias.

    Higher potential outcomes select into treatment.
    Naive ATE overestimates true effect.
    """
    np.random.seed(123)
    n = 1000

    # Latent ability
    ability = np.random.randn(n)

    # Treatment depends on ability (positive selection)
    p_treat = 1 / (1 + np.exp(-ability))
    treatment = np.random.binomial(1, p_treat)

    # Outcomes depend on ability and treatment
    # Y₀ = ability + noise
    # Y₁ = ability + 1.0 + noise (true ATE = 1.0)
    noise = 0.5 * np.random.randn(n)
    outcome = ability + 1.0 * treatment + noise

    return outcome, treatment, 1.0  # true ATE


@pytest.fixture
def bounded_outcome_data():
    """Data with known bounded outcome support [0, 10]."""
    np.random.seed(456)
    n = 500
    treatment = np.random.binomial(1, 0.5, n)

    # Outcomes bounded in [0, 10]
    base = 5.0 + np.random.randn(n)
    effect = 1.5 * treatment
    outcome = np.clip(base + effect, 0, 10)

    return outcome, treatment, (0.0, 10.0)  # outcome, treatment, support


@pytest.fixture
def mtr_positive_data():
    """
    Data consistent with positive MTR (Y₁ ≥ Y₀).

    Treatment never hurts - like a beneficial training program.
    """
    np.random.seed(789)
    n = 800

    treatment = np.random.binomial(1, 0.4, n)

    # Y₀ ~ N(5, 1)
    y0 = 5.0 + np.random.randn(n)

    # Y₁ = Y₀ + positive effect (always ≥ 0)
    individual_effect = np.abs(np.random.randn(n))  # Always positive
    y1 = y0 + individual_effect

    outcome = np.where(treatment == 1, y1, y0)
    true_ate = individual_effect.mean()

    return outcome, treatment, true_ate


@pytest.fixture
def mtr_negative_data():
    """
    Data consistent with negative MTR (Y₁ ≤ Y₀).

    Treatment never helps - like a harmful exposure.
    """
    np.random.seed(101)
    n = 800

    treatment = np.random.binomial(1, 0.5, n)

    # Y₀ ~ N(5, 1)
    y0 = 5.0 + np.random.randn(n)

    # Y₁ = Y₀ - positive effect (always ≤ Y₀)
    individual_effect = np.abs(np.random.randn(n))
    y1 = y0 - individual_effect

    outcome = np.where(treatment == 1, y1, y0)
    true_ate = -individual_effect.mean()

    return outcome, treatment, true_ate


@pytest.fixture
def iv_data():
    """
    Data with a valid instrument.

    Instrument affects treatment but not outcome directly.
    """
    np.random.seed(202)
    n = 1000

    # Instrument (e.g., lottery)
    instrument = np.random.binomial(1, 0.5, n)

    # Unobserved confounder
    u = np.random.randn(n)

    # Treatment depends on instrument and confounder
    p_treat = 1 / (1 + np.exp(-(0.5 + 1.0 * instrument + 0.5 * u)))
    treatment = np.random.binomial(1, p_treat)

    # Outcome depends on treatment and confounder (not instrument)
    true_ate = 2.0
    outcome = true_ate * treatment + u + np.random.randn(n)

    return outcome, treatment, instrument, true_ate


@pytest.fixture
def null_effect_data():
    """Data with true ATE = 0 (null effect)."""
    np.random.seed(303)
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    outcome = np.random.randn(n)  # No treatment effect
    return outcome, treatment, 0.0


@pytest.fixture
def large_effect_data():
    """Data with large treatment effect (ATE = 5.0)."""
    np.random.seed(404)
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 5.0 * treatment + np.random.randn(n)
    return outcome, treatment, 5.0


@pytest.fixture
def heterogeneous_effect_data():
    """Data with heterogeneous treatment effects."""
    np.random.seed(505)
    n = 1000

    # Covariate
    x = np.random.randn(n)

    treatment = np.random.binomial(1, 0.5, n)

    # Heterogeneous effect: ATE = 2 + x
    individual_effect = 2.0 + x
    outcome = individual_effect * treatment + np.random.randn(n)

    true_ate = individual_effect.mean()
    return outcome, treatment, true_ate


def generate_manski_dgp(
    n: int = 1000,
    true_ate: float = 2.0,
    selection_strength: float = 0.0,
    random_seed: int = 42,
):
    """
    Generate DGP for Manski bounds testing.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    selection_strength : float
        Strength of selection into treatment (0 = random, 1 = strong positive).
    random_seed : int
        Random seed.

    Returns
    -------
    tuple
        (outcome, treatment, true_ate)
    """
    np.random.seed(random_seed)

    # Latent type
    u = np.random.randn(n)

    # Selection into treatment
    if selection_strength > 0:
        p_treat = 1 / (1 + np.exp(-selection_strength * u))
    else:
        p_treat = np.full(n, 0.5)

    treatment = np.random.binomial(1, p_treat)

    # Outcomes
    y0 = u + np.random.randn(n)
    y1 = y0 + true_ate
    outcome = np.where(treatment == 1, y1, y0)

    return outcome, treatment, true_ate
