"""Shared pytest fixtures for causal inference tests.

This file provides common fixtures used across all test modules.
Following patterns from annuity_forecasting and double_ml_time_series.
"""

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Simple Synthetic Data Fixtures
# ============================================================================


@pytest.fixture
def simple_rct_data() -> dict:
    """
    Simple RCT data with known treatment effect for quick tests.

    Returns
    -------
    dict
        - 'outcomes': np.ndarray of shape (4,)
        - 'treatment': np.ndarray of shape (4,)
        - 'true_effect': float (known value = 4.0)

    Notes
    -----
    Hand-calculated expected values:
    - Treated mean: (7.0 + 5.0) / 2 = 6.0
    - Control mean: (3.0 + 1.0) / 2 = 2.0
    - ATE: 6.0 - 2.0 = 4.0
    """
    return {
        "outcomes": np.array([7.0, 5.0, 3.0, 1.0]),
        "treatment": np.array([1, 1, 0, 0]),
        "true_effect": 4.0,
    }


@pytest.fixture
def balanced_rct_data() -> dict:
    """
    Balanced RCT data (equal n) with known variance for SE testing.

    Returns
    -------
    dict
        - 'outcomes': np.ndarray of shape (6,)
        - 'treatment': np.ndarray of shape (6,)
        - 'true_effect': float (known value = 3.0)
        - 'expected_se': float (hand-calculated)

    Notes
    -----
    Hand-calculated:
    - Treated: [3, 5, 7], mean=5.0, var=4.0, n=3
    - Control: [1, 2, 3], mean=2.0, var=1.0, n=3
    - ATE: 5.0 - 2.0 = 3.0
    - SE: sqrt(4.0/3 + 1.0/3) = sqrt(5/3) ≈ 1.291
    """
    return {
        "outcomes": np.array([3.0, 5.0, 7.0, 1.0, 2.0, 3.0]),
        "treatment": np.array([1, 1, 1, 0, 0, 0]),
        "true_effect": 3.0,
        "expected_se": np.sqrt(5 / 3),
    }


# ============================================================================
# Synthetic Data Generation Fixtures
# ============================================================================


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_size_small() -> int:
    """Small sample size for unit tests (fast execution)."""
    return 50


@pytest.fixture
def sample_size_medium() -> int:
    """Medium sample size for integration tests."""
    return 200


@pytest.fixture
def sample_size_large() -> int:
    """Large sample size for Monte Carlo validation."""
    return 1000


# ============================================================================
# Error Testing Fixtures
# ============================================================================


@pytest.fixture
def empty_arrays() -> dict:
    """Empty arrays for error testing."""
    return {
        "outcomes": np.array([]),
        "treatment": np.array([]),
    }


@pytest.fixture
def mismatched_arrays() -> dict:
    """Arrays with different lengths for error testing."""
    return {
        "outcomes": np.array([1.0, 2.0, 3.0]),
        "treatment": np.array([1, 0]),
    }


@pytest.fixture
def arrays_with_nan() -> dict:
    """Arrays containing NaN values for error testing."""
    return {
        "outcomes": np.array([1.0, 2.0, np.nan, 4.0]),
        "treatment": np.array([1, 1, 0, 0]),
    }


@pytest.fixture
def all_treated() -> dict:
    """Data where all units are treated (no variation)."""
    return {
        "outcomes": np.array([1.0, 2.0, 3.0, 4.0]),
        "treatment": np.array([1, 1, 1, 1]),
    }


@pytest.fixture
def all_control() -> dict:
    """Data where all units are control (no variation)."""
    return {
        "outcomes": np.array([1.0, 2.0, 3.0, 4.0]),
        "treatment": np.array([0, 0, 0, 0]),
    }


# ============================================================================
# Standard Effect Sizes for Power Analysis
# ============================================================================


@pytest.fixture
def small_effect_size() -> float:
    """Cohen's d small effect size."""
    return 0.2


@pytest.fixture
def medium_effect_size() -> float:
    """Cohen's d medium effect size."""
    return 0.5


@pytest.fixture
def large_effect_size() -> float:
    """Cohen's d large effect size."""
    return 0.8


# ============================================================================
# Statistical Test Parameters
# ============================================================================


@pytest.fixture
def alpha_standard() -> float:
    """Standard significance level."""
    return 0.05


@pytest.fixture
def power_standard() -> float:
    """Standard statistical power."""
    return 0.80


# ============================================================================
# Tolerance Parameters for Numerical Comparisons
# ============================================================================


@pytest.fixture
def rtol_strict() -> float:
    """Strict relative tolerance for cross-language validation."""
    return 1e-10


@pytest.fixture
def rtol_standard() -> float:
    """Standard relative tolerance for most tests."""
    return 1e-6


@pytest.fixture
def rtol_loose() -> float:
    """Loose tolerance for Monte Carlo / stochastic tests."""
    return 1e-2
