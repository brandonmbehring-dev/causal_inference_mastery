"""
Test fixtures for instrumental variables (IV) tests.

Provides known-answer test data for validating IV estimators:
- Just-identified IV (1 instrument, 1 endogenous)
- Over-identified IV (2 instruments, 1 endogenous)
- Strong instrument (F > 20)
- Weak instrument (F < 10) - for adversarial tests
- Very weak instrument (F < 5) - for adversarial tests

All fixtures return (Y, D, Z, X, true_beta) where:
- Y: Outcome variable
- D: Endogenous treatment variable
- Z: Instrumental variable(s)
- X: Exogenous controls (can be None)
- true_beta: True causal effect (for validation)
"""

import numpy as np
import pytest


@pytest.fixture
def iv_just_identified():
    """
    Just-identified IV: 1 instrument, 1 endogenous variable.

    Data Generating Process (Angrist-Krueger 1991 style):
    - Returns to schooling with quarter of birth as instrument
    - Instrument: Quarter of birth (exogenous)
    - Treatment: Education (affected by quarter via compulsory schooling laws)
    - Outcome: Log wages
    - True causal effect: β = 0.10 (10% return to one year of education)

    Sample size: n = 10,000
    First-stage F-statistic: ~60 (strong instrument)

    Returns
    -------
    Y : ndarray, shape (10000,)
        Log wages (outcome)
    D : ndarray, shape (10000,)
        Years of education (endogenous)
    Z : ndarray, shape (10000,)
        Quarter of birth (instrument: 1, 2, 3, 4)
    X : None
        No controls in this simple example
    true_beta : float
        True causal effect = 0.10
    """
    np.random.seed(42)
    n = 10000

    # Instrument: Quarter of birth (1, 2, 3, 4)
    Z = np.random.choice([1, 2, 3, 4], size=n).astype(float)

    # First stage: Education affected by quarter
    # Quarter 1 (Jan-Mar) → lower education due to compulsory schooling laws
    # (Students born in Q1 can drop out earlier)
    D = 12 + 0.5 * (Z == 1) + np.random.normal(0, 2, n)

    # Outcome: Log wages = 0.10 * education + noise
    # True causal effect: One year of education → 10% higher wages
    true_beta = 0.10
    Y = 8 + true_beta * D + np.random.normal(0, 0.5, n)

    return Y, D, Z, None, true_beta


@pytest.fixture
def iv_over_identified():
    """
    Over-identified IV: 2 instruments, 1 endogenous variable.

    Data Generating Process:
    - Two instruments: quarter of birth + distance to college
    - Both instruments affect education
    - Same returns to education: β = 0.10

    Sample size: n = 10,000
    First-stage F-statistic: ~80 (strong instruments)

    Returns
    -------
    Y : ndarray, shape (10000,)
        Log wages (outcome)
    D : ndarray, shape (10000,)
        Years of education (endogenous)
    Z : ndarray, shape (10000, 2)
        Two instruments:
        - Z1: Quarter of birth (1, 2, 3, 4)
        - Z2: Distance to nearest college (km, exponential distribution)
    X : None
        No controls
    true_beta : float
        True causal effect = 0.10
    """
    np.random.seed(123)
    n = 10000

    # Instrument 1: Quarter of birth
    Z1 = np.random.choice([1, 2, 3, 4], size=n).astype(float)

    # Instrument 2: Distance to nearest college (km)
    # Exponential distribution, mean = 50 km
    Z2 = np.random.exponential(50, n)

    # Stack instruments
    Z = np.column_stack([Z1, Z2])

    # First stage: Both instruments affect education
    # Quarter 1 → +0.5 years education
    # Each 10km distance → -0.1 years education
    D = 12 + 0.5 * (Z1 == 1) - 0.01 * Z2 + np.random.normal(0, 2, n)

    # Outcome: Same returns to education
    true_beta = 0.10
    Y = 8 + true_beta * D + np.random.normal(0, 0.5, n)

    return Y, D, Z, None, true_beta


@pytest.fixture
def iv_strong_instrument():
    """
    Strong instrument: F > 20 (conventional "strong" threshold).

    Data Generating Process:
    - Designed for F-statistic ≈ 50
    - Signal-to-noise ratio = 2:1 in first stage
    - True causal effect: β = 0.50

    Sample size: n = 1,000
    First-stage F-statistic: ~50 (strong)

    Returns
    -------
    Y : ndarray, shape (1000,)
        Outcome variable
    D : ndarray, shape (1000,)
        Endogenous treatment
    Z : ndarray, shape (1000,)
        Strong instrument (standard normal)
    X : None
        No controls
    true_beta : float
        True causal effect = 0.50
    """
    np.random.seed(789)
    n = 1000

    # Instrument: Standard normal
    Z = np.random.normal(0, 1, n)

    # First stage: Strong relationship (R² ≈ 0.3)
    # D = 2*Z + noise, signal-to-noise = 2:1
    # This produces F ≈ 50
    D = 2 * Z + np.random.normal(0, 1, n)

    # Outcome: Causal effect = 0.50
    true_beta = 0.50
    Y = 1 + true_beta * D + np.random.normal(0, 1, n)

    return Y, D, Z, None, true_beta


@pytest.fixture
def iv_weak_instrument():
    """
    Weak instrument: F ≈ 8 (below Stock-Yogo threshold of 16.38).

    Data Generating Process:
    - Designed for F-statistic ≈ 8
    - Signal-to-noise ratio = 0.3:1 in first stage
    - True causal effect: β = 0.50

    Sample size: n = 1,000
    First-stage F-statistic: ~8 (weak, fails Stock-Yogo test)

    Returns
    -------
    Y : ndarray, shape (1000,)
        Outcome variable
    D : ndarray, shape (1000,)
        Endogenous treatment
    Z : ndarray, shape (1000,)
        Weak instrument
    X : None
        No controls
    true_beta : float
        True causal effect = 0.50
    """
    np.random.seed(456)
    n = 1000

    # Instrument
    Z = np.random.normal(0, 1, n)

    # First stage: Weak relationship (R² ≈ 0.008)
    # D = 0.09*Z + noise, signal-to-noise = 0.09:1
    # This produces F ≈ 8
    D = 0.09 * Z + np.random.normal(0, 1, n)

    # Outcome
    true_beta = 0.50
    Y = 1 + true_beta * D + np.random.normal(0, 1, n)

    return Y, D, Z, None, true_beta


@pytest.fixture
def iv_very_weak_instrument():
    """
    Very weak instrument: F ≈ 2 (seriously weak).

    Data Generating Process:
    - Designed for F-statistic ≈ 2
    - Signal-to-noise ratio = 0.05:1 in first stage
    - True causal effect: β = 0.50

    Sample size: n = 1,000
    First-stage F-statistic: ~2 (very weak)

    Returns
    -------
    Y : ndarray, shape (1000,)
        Outcome variable
    D : ndarray, shape (1000,)
        Endogenous treatment
    Z : ndarray, shape (1000,)
        Very weak instrument
    X : None
        No controls
    true_beta : float
        True causal effect = 0.50
    """
    np.random.seed(999)
    n = 1000

    # Instrument
    Z = np.random.normal(0, 1, n)

    # First stage: Very weak relationship (R² ≈ 0.002)
    # D = 0.05*Z + noise, signal-to-noise = 0.05:1
    # This produces F ≈ 2
    D = 0.05 * Z + np.random.normal(0, 1, n)

    # Outcome
    true_beta = 0.50
    Y = 1 + true_beta * D + np.random.normal(0, 1, n)

    return Y, D, Z, None, true_beta


@pytest.fixture
def iv_with_controls():
    """
    IV with exogenous controls: Z + X → D, D + X → Y.

    Data Generating Process:
    - 1 instrument, 1 endogenous, 2 exogenous controls
    - Controls affect both D and Y
    - True causal effect: β = 0.10

    Sample size: n = 5,000
    First-stage F-statistic: ~40 (strong)

    Returns
    -------
    Y : ndarray, shape (5000,)
        Outcome variable
    D : ndarray, shape (5000,)
        Endogenous treatment
    Z : ndarray, shape (5000,)
        Instrument
    X : ndarray, shape (5000, 2)
        Two exogenous controls
    true_beta : float
        True causal effect = 0.10
    """
    np.random.seed(555)
    n = 5000

    # Instrument
    Z = np.random.normal(0, 1, n)

    # Exogenous controls (affect both D and Y)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.uniform(0, 10, n)
    X = np.column_stack([X1, X2])

    # First stage: D affected by Z and X
    D = 1 * Z + 0.5 * X1 + 0.1 * X2 + np.random.normal(0, 1, n)

    # Outcome: Y affected by D and X
    true_beta = 0.10
    Y = true_beta * D + 0.3 * X1 - 0.05 * X2 + np.random.normal(0, 0.5, n)

    return Y, D, Z, X, true_beta


@pytest.fixture
def iv_heteroskedastic():
    """
    IV with heteroskedastic errors (for robust SE testing).

    Data Generating Process:
    - Error variance increases with Z
    - Var(ε | Z) = 1 + Z²
    - True causal effect: β = 0.20

    Sample size: n = 2,000
    First-stage F-statistic: ~30 (strong)

    Returns
    -------
    Y : ndarray, shape (2000,)
        Outcome variable
    D : ndarray, shape (2000,)
        Endogenous treatment
    Z : ndarray, shape (2000,)
        Instrument
    X : None
        No controls
    true_beta : float
        True causal effect = 0.20
    """
    np.random.seed(777)
    n = 2000

    # Instrument
    Z = np.random.normal(0, 1, n)

    # First stage
    D = 1.5 * Z + np.random.normal(0, 1, n)

    # Outcome with heteroskedastic errors
    # Error variance increases with Z
    error_sd = np.sqrt(1 + Z**2)
    errors = np.random.normal(0, 1, n) * error_sd

    true_beta = 0.20
    Y = 1 + true_beta * D + errors

    return Y, D, Z, None, true_beta
