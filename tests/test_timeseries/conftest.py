"""
Test Fixtures for Time Series Causal Inference.

Session 135: Data generating processes for Granger causality and VAR tests.
"""

import numpy as np
import pytest
from typing import Tuple


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_stationary_series(seed):
    """Generate stationary AR(1) series."""
    np.random.seed(seed)
    n = 200
    y = np.zeros(n)
    phi = 0.5  # AR coefficient
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.randn()
    return y


@pytest.fixture
def sample_nonstationary_series(seed):
    """Generate non-stationary random walk series."""
    np.random.seed(seed)
    n = 200
    return np.cumsum(np.random.randn(n))


@pytest.fixture
def sample_granger_causal_pair(seed):
    """
    Generate bivariate series where X Granger-causes Y.

    X -> Y with lag 1, coefficient 0.5
    """
    np.random.seed(seed)
    n = 200
    x = np.random.randn(n)
    y = np.zeros(n)

    for t in range(1, n):
        y[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + np.random.randn() * 0.5

    return np.column_stack([y, x])  # [effect, cause]


@pytest.fixture
def sample_no_granger_causality(seed):
    """
    Generate bivariate series with NO Granger causality.

    X and Y are independent AR(1) processes.
    """
    np.random.seed(seed)
    n = 200

    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + np.random.randn()
        y[t] = 0.5 * y[t - 1] + np.random.randn()

    return np.column_stack([y, x])


@pytest.fixture
def sample_bidirectional_causality(seed):
    """
    Generate bivariate series with bidirectional Granger causality.

    X -> Y and Y -> X (feedback system).
    """
    np.random.seed(seed)
    n = 200

    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + 0.4 * y[t - 1] + np.random.randn() * 0.5
        y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + np.random.randn() * 0.5

    return np.column_stack([y, x])


@pytest.fixture
def sample_var1_data(seed):
    """
    Generate data from VAR(1) process with known coefficients.

    Y_t = A_0 + A_1 Y_{t-1} + epsilon_t

    where A_1 = [[0.5, 0.1], [0.2, 0.4]]
    """
    np.random.seed(seed)
    n = 200
    k = 2

    A0 = np.array([0.0, 0.0])
    A1 = np.array([[0.5, 0.1], [0.2, 0.4]])

    data = np.zeros((n, k))

    for t in range(1, n):
        data[t, :] = A0 + A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

    return data, A1


@pytest.fixture
def sample_var2_data(seed):
    """
    Generate data from VAR(2) process with known coefficients.
    """
    np.random.seed(seed)
    n = 200
    k = 2

    A0 = np.array([0.0, 0.0])
    A1 = np.array([[0.4, 0.1], [0.1, 0.3]])
    A2 = np.array([[0.2, 0.05], [0.05, 0.2]])

    data = np.zeros((n, k))

    for t in range(2, n):
        data[t, :] = A0 + A1 @ data[t - 1, :] + A2 @ data[t - 2, :] + np.random.randn(k) * 0.5

    return data, (A1, A2)


@pytest.fixture
def sample_chain_causality(seed):
    """
    Generate three-variable chain: X1 -> X2 -> X3

    Only X1 -> X2 and X2 -> X3 should be detected,
    not X1 -> X3 (indirect).
    """
    np.random.seed(seed)
    n = 300

    x1 = np.random.randn(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)

    for t in range(1, n):
        x2[t] = 0.6 * x1[t - 1] + 0.2 * x2[t - 1] + np.random.randn() * 0.5
        x3[t] = 0.6 * x2[t - 1] + 0.2 * x3[t - 1] + np.random.randn() * 0.5

    return np.column_stack([x1, x2, x3])


@pytest.fixture
def sample_confounder_data(seed):
    """
    Generate data with confounded causality.

    Z -> X and Z -> Y, but X does NOT cause Y.
    Spurious Granger causality may be detected.
    """
    np.random.seed(seed)
    n = 200

    z = np.random.randn(n)  # Confounder
    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.5 * z[t - 1] + 0.2 * x[t - 1] + np.random.randn() * 0.5
        y[t] = 0.5 * z[t - 1] + 0.2 * y[t - 1] + np.random.randn() * 0.5

    return np.column_stack([y, x]), z


def generate_granger_causal_pair(
    n: int = 200,
    lag: int = 1,
    effect_size: float = 0.5,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Generate bivariate series where X Granger-causes Y.

    Parameters
    ----------
    n : int
        Number of observations
    lag : int
        Lag at which causality occurs
    effect_size : float
        Coefficient of X on Y
    noise_scale : float
        Scale of error terms
    seed : int
        Random seed

    Returns
    -------
    data : np.ndarray
        Shape (n, 2) with [Y, X]
    true_effect : float
        True causal effect size
    """
    np.random.seed(seed)
    x = np.random.randn(n)
    y = np.zeros(n)

    for t in range(lag, n):
        y[t] = effect_size * x[t - lag] + 0.3 * y[t - 1] + np.random.randn() * noise_scale

    return np.column_stack([y, x]), effect_size


def generate_var_data(
    n: int = 200,
    k: int = 2,
    lags: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate VAR(p) data with random stable coefficients.

    Parameters
    ----------
    n : int
        Number of observations
    k : int
        Number of variables
    lags : int
        Lag order
    seed : int
        Random seed

    Returns
    -------
    data : np.ndarray
        Shape (n, k)
    coefficients : np.ndarray
        True coefficient matrices
    """
    np.random.seed(seed)

    # Generate stable VAR coefficients
    all_A = []
    for _ in range(lags):
        A = np.random.randn(k, k) * 0.3
        # Ensure stability by scaling eigenvalues
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 0.8 / lags:
            A = A * (0.8 / lags) / max_eig
        all_A.append(A)

    data = np.zeros((n, k))

    for t in range(lags, n):
        for lag_idx, A in enumerate(all_A):
            data[t, :] += A @ data[t - lag_idx - 1, :]
        data[t, :] += np.random.randn(k) * 0.5

    return data, np.array(all_A)
