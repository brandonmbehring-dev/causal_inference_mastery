"""
Test fixtures for Regression Discontinuity Design (RDD) tests.

Provides known-answer test data for validating Sharp RDD estimators:
- Linear DGP (simple discontinuity)
- Quadratic DGP (tests local linear with curvature)
- Zero effect (no jump at cutoff)
- Large effect (obvious discontinuity)
- Sparse data near cutoff
- Heteroskedastic errors (robust SE testing)
- All observations on one side (should raise error)
- Boundary cutoff (adversarial test)

All fixtures return (Y, X, cutoff, true_tau) where:
- Y: Outcome variable
- X: Running variable (assignment variable)
- cutoff: Threshold value where treatment changes
- true_tau: True treatment effect at cutoff
"""

import numpy as np
import pytest


@pytest.fixture
def sharp_rdd_linear_dgp():
    """
    Linear DGP with sharp discontinuity at cutoff=0.

    Data Generating Process:
    - Y = X + 2*(X >= 0) + ε
    - Running variable: X ~ U(-5, 5)
    - Treatment effect: τ = 2.0 (sharp jump at X=0)
    - Error: ε ~ N(0, 1)

    Sample size: n = 1000
    Expected estimate: τ̂ ≈ 2.0 (±0.20 tolerance)

    Returns
    -------
    Y : ndarray, shape (1000,)
        Outcome variable
    X : ndarray, shape (1000,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 2.0
    """
    np.random.seed(42)
    n = 1000

    # Running variable: uniform on [-5, 5]
    X = np.random.uniform(-5, 5, n)

    # Treatment indicator
    D = (X >= 0).astype(float)

    # Outcome: linear with jump of 2.0 at cutoff
    Y = X + 2.0 * D + np.random.normal(0, 1, n)

    return Y, X, 0.0, 2.0


@pytest.fixture
def sharp_rdd_quadratic_dgp():
    """
    Quadratic DGP with sharp discontinuity (tests local linear with curvature).

    Data Generating Process:
    - Y = X² + 3*(X >= 0) + ε
    - Running variable: X ~ U(-4, 4)
    - Treatment effect: τ = 3.0 (jump at X=0)
    - Error: ε ~ N(0, 0.8)

    Sample size: n = 800
    Expected estimate: τ̂ ≈ 3.0 (±0.25 tolerance, higher due to curvature)

    Notes
    -----
    Local linear regression should handle curvature well by focusing
    on observations near the cutoff.

    Returns
    -------
    Y : ndarray, shape (800,)
        Outcome variable
    X : ndarray, shape (800,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 3.0
    """
    np.random.seed(123)
    n = 800

    # Running variable
    X = np.random.uniform(-4, 4, n)

    # Treatment indicator
    D = (X >= 0).astype(float)

    # Outcome: quadratic with jump
    Y = X**2 + 3.0 * D + np.random.normal(0, 0.8, n)

    return Y, X, 0.0, 3.0


@pytest.fixture
def sharp_rdd_zero_effect_dgp():
    """
    No treatment effect (tests that RDD doesn't find phantom effects).

    Data Generating Process:
    - Y = X + ε (no discontinuity)
    - Running variable: X ~ U(-3, 3)
    - Treatment effect: τ = 0.0 (no jump)
    - Error: ε ~ N(0, 1.2)

    Sample size: n = 500
    Expected estimate: τ̂ ≈ 0.0, p-value > 0.05 (not significant)

    Returns
    -------
    Y : ndarray, shape (500,)
        Outcome variable
    X : ndarray, shape (500,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 0.0
    """
    np.random.seed(456)
    n = 500

    # Running variable
    X = np.random.uniform(-3, 3, n)

    # Outcome: no discontinuity, just linear trend
    Y = X + np.random.normal(0, 1.2, n)

    return Y, X, 0.0, 0.0


@pytest.fixture
def sharp_rdd_large_effect_dgp():
    """
    Large treatment effect (obvious discontinuity).

    Data Generating Process:
    - Y = 2*X + 10*(X >= 0) + ε
    - Running variable: X ~ U(-6, 6)
    - Treatment effect: τ = 10.0 (large jump)
    - Error: ε ~ N(0, 1.5)

    Sample size: n = 1200
    Expected estimate: τ̂ ≈ 10.0, p-value < 0.001 (highly significant)

    Returns
    -------
    Y : ndarray, shape (1200,)
        Outcome variable
    X : ndarray, shape (1200,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 10.0
    """
    np.random.seed(789)
    n = 1200

    # Running variable
    X = np.random.uniform(-6, 6, n)

    # Treatment indicator
    D = (X >= 0).astype(float)

    # Outcome: steeper slope (2*X) with large jump
    Y = 2 * X + 10.0 * D + np.random.normal(0, 1.5, n)

    return Y, X, 0.0, 10.0


@pytest.fixture
def sharp_rdd_sparse_data_dgp():
    """
    Sparse data near cutoff (tests small effective sample size warnings).

    Data Generating Process:
    - Y = 0.5*X + 2.5*(X >= 0) + ε
    - Running variable: X ~ U(-10, -1) ∪ U(1, 10) (gap around cutoff)
    - Treatment effect: τ = 2.5
    - Error: ε ~ N(0, 1)

    Sample size: n = 300
    Expected: RuntimeWarning about small effective sample size

    Notes
    -----
    With a gap around the cutoff, most observations will have low
    kernel weights, resulting in small effective sample sizes.

    Returns
    -------
    Y : ndarray, shape (300,)
        Outcome variable
    X : ndarray, shape (300,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 2.5
    """
    np.random.seed(999)
    n = 300

    # Running variable with gap around cutoff
    # Half on left: U(-10, -1), half on right: U(1, 10)
    X_left = np.random.uniform(-10, -1, n // 2)
    X_right = np.random.uniform(1, 10, n // 2)
    X = np.concatenate([X_left, X_right])

    # Treatment indicator
    D = (X >= 0).astype(float)

    # Outcome
    Y = 0.5 * X + 2.5 * D + np.random.normal(0, 1, n)

    return Y, X, 0.0, 2.5


@pytest.fixture
def sharp_rdd_heteroskedastic_dgp():
    """
    Heteroskedastic errors (tests robust SE vs standard SE).

    Data Generating Process:
    - Y = X + 1.5*(X >= 0) + ε
    - Running variable: X ~ U(-5, 5)
    - Treatment effect: τ = 1.5
    - Error variance: Var(ε | X) = 1 + X²

    Sample size: n = 1000
    Expected: Robust SEs >= standard SEs

    Notes
    -----
    Error variance increases with |X|, violating homoskedasticity.
    Robust SEs should be larger than standard SEs.

    Returns
    -------
    Y : ndarray, shape (1000,)
        Outcome variable
    X : ndarray, shape (1000,)
        Running variable
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 1.5
    """
    np.random.seed(555)
    n = 1000

    # Running variable
    X = np.random.uniform(-5, 5, n)

    # Treatment indicator
    D = (X >= 0).astype(float)

    # Heteroskedastic errors: variance increases with X²
    error_sd = np.sqrt(1 + X**2)
    errors = np.random.normal(0, 1, n) * error_sd

    # Outcome
    Y = X + 1.5 * D + errors

    return Y, X, 0.0, 1.5


@pytest.fixture
def sharp_rdd_all_left_dgp():
    """
    All observations on left side of cutoff (should raise error).

    Data Generating Process:
    - Y = X + 2*(X >= 0) + ε
    - Running variable: X ~ U(-5, -0.1) (all negative)
    - Cutoff: 0.0
    - Treatment effect: τ = 2.0 (but unidentified)

    Sample size: n = 500
    Expected: ValueError - cannot estimate right side

    Notes
    -----
    This is an adversarial test. SharpRDD should detect that there
    are no observations with X >= cutoff and raise an error.

    Returns
    -------
    Y : ndarray, shape (500,)
        Outcome variable
    X : ndarray, shape (500,)
        Running variable (all < 0)
    cutoff : float
        Cutoff value = 0.0
    true_tau : float
        True treatment effect = 2.0 (unidentified)
    """
    np.random.seed(111)
    n = 500

    # Running variable: all on left side
    X = np.random.uniform(-5, -0.1, n)

    # Outcome (but no treatment since all X < 0)
    Y = X + np.random.normal(0, 1, n)

    return Y, X, 0.0, 2.0


@pytest.fixture
def sharp_rdd_boundary_cutoff_dgp():
    """
    Cutoff at data boundary (adversarial test).

    Data Generating Process:
    - Y = X + 2*(X >= 5) + ε
    - Running variable: X ~ U(0, 10)
    - Cutoff: 5.0 (middle of range, but could be min or max in practice)
    - Treatment effect: τ = 2.0

    Sample size: n = 400
    Expected: Valid estimate, but may have warnings about bandwidth

    Notes
    -----
    When cutoff is near the boundary, bandwidth selection may be
    problematic. This tests whether the estimator handles this gracefully.

    Returns
    -------
    Y : ndarray, shape (400,)
        Outcome variable
    X : ndarray, shape (400,)
        Running variable
    cutoff : float
        Cutoff value = 5.0
    true_tau : float
        True treatment effect = 2.0
    """
    np.random.seed(222)
    n = 400

    # Running variable
    X = np.random.uniform(0, 10, n)

    # Treatment indicator
    D = (X >= 5.0).astype(float)

    # Outcome
    Y = X + 2.0 * D + np.random.normal(0, 1, n)

    return Y, X, 5.0, 2.0


@pytest.fixture
def rdd_bunching_dgp():
    """
    DGP with bunching at cutoff (manipulation).

    Running variable has excess mass right at cutoff (X=0).
    McCrary test should detect this (p < 0.05).
    """
    np.random.seed(333)
    n = 800

    # Base distribution: uniform
    X_base = np.random.uniform(-5, 5, int(n * 0.9))

    # Add bunching at cutoff (10% of observations)
    X_bunched = np.random.normal(0, 0.05, int(n * 0.1))  # Tight cluster at 0

    X = np.concatenate([X_base, X_bunched])
    D = (X >= 0).astype(float)
    Y = X + 2.0 * D + np.random.normal(0, 1, len(X))

    return Y, X, 0.0, 2.0


@pytest.fixture
def rdd_with_covariates_dgp():
    """
    Valid RDD with balanced pre-treatment covariates.

    Covariates (age, gender) have no discontinuity at cutoff.
    Balance tests should pass (p > 0.05).
    """
    np.random.seed(444)
    n = 1000

    X = np.random.uniform(-5, 5, n)
    D = (X >= 0).astype(float)

    # Pre-treatment covariates (no relationship to cutoff)
    age = 30 + 10 * np.random.normal(0, 1, n)
    gender = np.random.binomial(1, 0.5, n).astype(float)
    W = np.column_stack([age, gender])

    # Outcome depends on covariates AND treatment
    Y = 0.5 * age + 2.0 * gender + 3.0 * D + np.random.normal(0, 1, n)

    return Y, X, 0.0, 3.0, W


@pytest.fixture
def rdd_sorted_on_covariate_dgp():
    """
    Invalid RDD with sorting on covariates.

    Units with high income sort to right of cutoff.
    Balance test should detect this (p < 0.05).
    """
    np.random.seed(555)
    n = 1000

    X = np.random.uniform(-5, 5, n)

    # Income is higher on right side (sorting)
    income_baseline = 50 + 10 * np.random.normal(0, 1, n)
    income_boost = 15 * (X >= 0)  # Discontinuity in covariate!
    income = income_baseline + income_boost

    W = income.reshape(-1, 1)

    # Outcome
    D = (X >= 0).astype(float)
    Y = 0.3 * income + 2.0 * D + np.random.normal(0, 1, n)

    return Y, X, 0.0, 2.0, W


@pytest.fixture
def rdd_nonlinear_dgp():
    """
    Nonlinear DGP (cubic) for polynomial sensitivity testing.

    Y = X³ + τ*(X >= 0) + ε

    Local linear may have bias, but local cubic should work well.
    """
    np.random.seed(666)
    n = 1200

    X = np.random.uniform(-3, 3, n)
    D = (X >= 0).astype(float)

    # Cubic relationship
    Y = X**3 + 4.0 * D + np.random.normal(0, 1.5, n)

    return Y, X, 0.0, 4.0
