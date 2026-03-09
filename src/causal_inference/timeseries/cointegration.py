"""
Cointegration Tests for Time Series.

Session 145: Johansen cointegration test for multivariate systems.

Cointegration occurs when a linear combination of non-stationary I(1) series
is stationary. This implies a long-run equilibrium relationship.

The Johansen procedure tests for the number of cointegrating relationships
(cointegration rank r) using both trace and maximum eigenvalue statistics.

References
----------
- Johansen (1988). "Statistical analysis of cointegration vectors."
  Journal of Economic Dynamics and Control 12: 231-254.
- Johansen (1991). "Estimation and hypothesis testing of cointegration
  vectors in Gaussian vector autoregressive models." Econometrica 59: 1551-1580.
- MacKinnon-Haug-Michelis (1999). "Numerical distribution functions for
  unit root and cointegration tests." J. Appl. Econometrics 14: 563-577.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import JohansenResult


# Critical values for Johansen tests
# MacKinnon-Haug-Michelis (1999) tables
# Format: JOHANSEN_CV[det_order][n_vars][rank] = {"90%": ..., "95%": ..., "99%": ...}

# Trace statistic critical values
JOHANSEN_TRACE_CV = {
    # det_order = 0 (restricted constant, no trend - most common)
    0: {
        2: {
            0: {"90%": 13.43, "95%": 15.49, "99%": 19.94},
            1: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        3: {
            0: {"90%": 27.07, "95%": 29.80, "99%": 35.46},
            1: {"90%": 13.43, "95%": 15.49, "99%": 19.94},
            2: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        4: {
            0: {"90%": 44.49, "95%": 47.86, "99%": 54.68},
            1: {"90%": 27.07, "95%": 29.80, "99%": 35.46},
            2: {"90%": 13.43, "95%": 15.49, "99%": 19.94},
            3: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        5: {
            0: {"90%": 65.82, "95%": 69.82, "99%": 77.82},
            1: {"90%": 44.49, "95%": 47.86, "99%": 54.68},
            2: {"90%": 27.07, "95%": 29.80, "99%": 35.46},
            3: {"90%": 13.43, "95%": 15.49, "99%": 19.94},
            4: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        6: {
            0: {"90%": 91.11, "95%": 95.75, "99%": 104.96},
            1: {"90%": 65.82, "95%": 69.82, "99%": 77.82},
            2: {"90%": 44.49, "95%": 47.86, "99%": 54.68},
            3: {"90%": 27.07, "95%": 29.80, "99%": 35.46},
            4: {"90%": 13.43, "95%": 15.49, "99%": 19.94},
            5: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
    },
    # det_order = -1 (no constant, no trend)
    -1: {
        2: {
            0: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
            1: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        3: {
            0: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            1: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
            2: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        4: {
            0: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            1: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            2: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
            3: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        5: {
            0: {"90%": 50.52, "95%": 54.07, "99%": 61.27},
            1: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            2: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            3: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
            4: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        6: {
            0: {"90%": 72.77, "95%": 76.97, "99%": 85.34},
            1: {"90%": 50.52, "95%": 54.07, "99%": 61.27},
            2: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            3: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            4: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
            5: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
    },
    # det_order = 1 (unrestricted constant)
    1: {
        2: {
            0: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            1: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
        },
        3: {
            0: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            1: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            2: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
        },
        4: {
            0: {"90%": 50.52, "95%": 54.07, "99%": 61.27},
            1: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            2: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            3: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
        },
        5: {
            0: {"90%": 72.77, "95%": 76.97, "99%": 85.34},
            1: {"90%": 50.52, "95%": 54.07, "99%": 61.27},
            2: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            3: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            4: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
        },
        6: {
            0: {"90%": 98.42, "95%": 103.18, "99%": 112.74},
            1: {"90%": 72.77, "95%": 76.97, "99%": 85.34},
            2: {"90%": 50.52, "95%": 54.07, "99%": 61.27},
            3: {"90%": 32.09, "95%": 35.19, "99%": 41.20},
            4: {"90%": 17.98, "95%": 20.26, "99%": 25.08},
            5: {"90%": 7.52, "95%": 9.16, "99%": 12.97},
        },
    },
}

# Maximum eigenvalue critical values
JOHANSEN_MAX_EIGEN_CV = {
    # det_order = 0 (restricted constant)
    0: {
        2: {
            0: {"90%": 12.30, "95%": 14.26, "99%": 18.52},
            1: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        3: {
            0: {"90%": 18.89, "95%": 21.13, "99%": 25.86},
            1: {"90%": 12.30, "95%": 14.26, "99%": 18.52},
            2: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        4: {
            0: {"90%": 25.12, "95%": 27.58, "99%": 32.72},
            1: {"90%": 18.89, "95%": 21.13, "99%": 25.86},
            2: {"90%": 12.30, "95%": 14.26, "99%": 18.52},
            3: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        5: {
            0: {"90%": 31.24, "95%": 33.88, "99%": 39.37},
            1: {"90%": 25.12, "95%": 27.58, "99%": 32.72},
            2: {"90%": 18.89, "95%": 21.13, "99%": 25.86},
            3: {"90%": 12.30, "95%": 14.26, "99%": 18.52},
            4: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        6: {
            0: {"90%": 37.28, "95%": 40.07, "99%": 45.87},
            1: {"90%": 31.24, "95%": 33.88, "99%": 39.37},
            2: {"90%": 25.12, "95%": 27.58, "99%": 32.72},
            3: {"90%": 18.89, "95%": 21.13, "99%": 25.86},
            4: {"90%": 12.30, "95%": 14.26, "99%": 18.52},
            5: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
    },
    # det_order = -1 (no constant)
    -1: {
        2: {
            0: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
            1: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        3: {
            0: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            1: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
            2: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        4: {
            0: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            1: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            2: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
            3: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        5: {
            0: {"90%": 24.16, "95%": 26.53, "99%": 31.73},
            1: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            2: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            3: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
            4: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
        6: {
            0: {"90%": 29.57, "95%": 32.12, "99%": 37.61},
            1: {"90%": 24.16, "95%": 26.53, "99%": 31.73},
            2: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            3: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            4: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
            5: {"90%": 2.71, "95%": 3.84, "99%": 6.63},
        },
    },
    # det_order = 1 (unrestricted constant)
    1: {
        2: {
            0: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            1: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
        },
        3: {
            0: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            1: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            2: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
        },
        4: {
            0: {"90%": 24.16, "95%": 26.53, "99%": 31.73},
            1: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            2: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            3: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
        },
        5: {
            0: {"90%": 29.57, "95%": 32.12, "99%": 37.61},
            1: {"90%": 24.16, "95%": 26.53, "99%": 31.73},
            2: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            3: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            4: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
        },
        6: {
            0: {"90%": 34.87, "95%": 37.61, "99%": 43.27},
            1: {"90%": 29.57, "95%": 32.12, "99%": 37.61},
            2: {"90%": 24.16, "95%": 26.53, "99%": 31.73},
            3: {"90%": 18.63, "95%": 20.78, "99%": 25.42},
            4: {"90%": 12.78, "95%": 14.59, "99%": 18.78},
            5: {"90%": 6.69, "95%": 8.18, "99%": 11.65},
        },
    },
}


def johansen_test(
    data: np.ndarray,
    lags: int = 1,
    det_order: int = 0,
    alpha: float = 0.05,
) -> JohansenResult:
    """
    Johansen cointegration test.

    Tests for cointegration rank in a VAR system. Determines the number r
    of cointegrating relationships among n variables.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_obs, n_vars) multivariate time series
    lags : int
        Number of VAR lags (p). The VECM uses p-1 lagged differences.
    det_order : int
        Deterministic terms specification:
        - -1: No constant, no trend
        - 0: Restricted constant (most common)
        - 1: Unrestricted constant
    alpha : float
        Significance level for rank determination

    Returns
    -------
    JohansenResult
        Test results including rank, test statistics, eigenvalues,
        and cointegrating vectors

    Example
    -------
    >>> np.random.seed(42)
    >>> # Create cointegrated system: y1 and y2 share common stochastic trend
    >>> n = 200
    >>> trend = np.cumsum(np.random.randn(n))  # Common trend
    >>> y1 = trend + np.random.randn(n) * 0.5
    >>> y2 = 0.5 * trend + np.random.randn(n) * 0.5
    >>> data = np.column_stack([y1, y2])
    >>> result = johansen_test(data, lags=2)
    >>> print(f"Cointegration rank: {result.rank}")

    Notes
    -----
    The Johansen procedure works by reformulating the VAR in VECM form:
        ΔY_t = Π Y_{t-1} + Γ_1 ΔY_{t-1} + ... + Γ_{p-1} ΔY_{t-p+1} + ε_t

    where Π = αβ' contains the long-run information:
    - α: Adjustment coefficients (speed of adjustment to equilibrium)
    - β: Cointegrating vectors (long-run relationships)

    The rank of Π equals the number of cointegrating relationships.

    References
    ----------
    Johansen (1988, 1991). Statistical analysis of cointegration vectors.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if n_vars < 2:
        raise ValueError(f"Need at least 2 variables for cointegration, got {n_vars}")

    if n_vars > 6:
        raise ValueError(f"Critical values only available for up to 6 variables, got {n_vars}")

    if n_obs < 3 * n_vars + lags:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for {n_vars} variables and {lags} lags"
        )

    if det_order not in [-1, 0, 1]:
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}")

    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    # Step 1: Compute first differences and levels
    dy = np.diff(data, axis=0)  # ΔY_t (T-1, n_vars)
    y_lag = data[:-1, :]  # Y_{t-1} (T-1, n_vars)

    # Effective sample size
    T = dy.shape[0] - lags + 1

    # Trim to account for lags
    dy_trimmed = dy[lags - 1 :, :]  # (T, n_vars)
    y_lag_trimmed = y_lag[lags - 1 :, :]  # (T, n_vars)

    # Step 2: Build regressors for lagged differences
    X_parts = []

    # Add lagged differences ΔY_{t-1}, ..., ΔY_{t-p+1}
    for i in range(1, lags):
        lag_diff = dy[lags - 1 - i : -i, :]
        X_parts.append(lag_diff)

    # Add deterministic terms
    if det_order >= 0:
        # Constant
        X_parts.append(np.ones((T, 1)))

    if len(X_parts) > 0:
        X = np.hstack(X_parts)
    else:
        X = None

    # Step 3: Reduced rank regression via canonical correlations
    # Regress ΔY_t on X to get residuals R0
    # Regress Y_{t-1} on X to get residuals R1
    # Then solve eigenvalue problem with S_ij = R_i' R_j / T

    if X is not None and X.shape[1] > 0:
        # Residuals from regressing ΔY on X
        beta0 = np.linalg.lstsq(X, dy_trimmed, rcond=None)[0]
        R0 = dy_trimmed - X @ beta0

        # Residuals from regressing Y_{t-1} on X
        beta1 = np.linalg.lstsq(X, y_lag_trimmed, rcond=None)[0]
        R1 = y_lag_trimmed - X @ beta1
    else:
        # No regressors - use centered data
        R0 = dy_trimmed - np.mean(dy_trimmed, axis=0)
        R1 = y_lag_trimmed - np.mean(y_lag_trimmed, axis=0)

    # Step 4: Form moment matrices
    S00 = R0.T @ R0 / T
    S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T
    S10 = R1.T @ R0 / T

    # Step 5: Solve generalized eigenvalue problem
    # |λ S_11 - S_10 S_00^{-1} S_01| = 0

    try:
        S00_inv = np.linalg.inv(S00)
    except np.linalg.LinAlgError:
        S00_inv = np.linalg.pinv(S00)

    try:
        S11_inv = np.linalg.inv(S11)
    except np.linalg.LinAlgError:
        S11_inv = np.linalg.pinv(S11)

    # Matrix for eigenvalue problem
    M = S11_inv @ S10 @ S00_inv @ S01

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Take real parts (should be real for symmetric problem)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Ensure eigenvalues are in [0, 1] (numerical stability)
    eigenvalues = np.clip(eigenvalues, 0, 1 - 1e-10)

    # Step 6: Compute trace and max eigenvalue statistics
    trace_stats = np.zeros(n_vars)
    max_eigen_stats = np.zeros(n_vars)

    for r in range(n_vars):
        # Trace statistic: -T * sum(ln(1 - λ_i)) for i = r+1 to n
        trace_stats[r] = -T * np.sum(np.log(1 - eigenvalues[r:]))

        # Max eigenvalue statistic: -T * ln(1 - λ_{r+1})
        if r < n_vars:
            max_eigen_stats[r] = -T * np.log(1 - eigenvalues[r])

    # Step 7: Get critical values and determine rank
    trace_crit = np.zeros(n_vars)
    max_eigen_crit = np.zeros(n_vars)
    trace_pvalues = np.zeros(n_vars)
    max_eigen_pvalues = np.zeros(n_vars)

    # Get critical value level string
    if alpha <= 0.01:
        cv_level = "99%"
    elif alpha <= 0.05:
        cv_level = "95%"
    else:
        cv_level = "90%"

    for r in range(n_vars):
        trace_cv_dict = JOHANSEN_TRACE_CV.get(det_order, {}).get(n_vars, {}).get(r, {})
        max_cv_dict = JOHANSEN_MAX_EIGEN_CV.get(det_order, {}).get(n_vars, {}).get(r, {})

        trace_crit[r] = trace_cv_dict.get(cv_level, np.nan)
        max_eigen_crit[r] = max_cv_dict.get(cv_level, np.nan)

        # Approximate p-values
        trace_pvalues[r] = _johansen_pvalue(trace_stats[r], trace_cv_dict)
        max_eigen_pvalues[r] = _johansen_pvalue(max_eigen_stats[r], max_cv_dict)

    # Determine rank by trace test (sequential testing)
    # Start with r=0, reject if stat > CV, continue until fail to reject
    rank = 0
    for r in range(n_vars):
        if trace_stats[r] > trace_crit[r]:
            rank = r + 1
        else:
            break

    # Step 8: Extract cointegrating vectors and adjustment coefficients
    # β = eigenvectors (columns are cointegrating vectors)
    # α = S_01 * β * (β' S_11 β)^{-1}

    beta = eigenvectors  # Cointegrating vectors

    # Compute adjustment coefficients
    # For numerical stability, compute for all vectors
    try:
        beta_S11_beta_inv = np.linalg.inv(beta.T @ S11 @ beta)
        alpha = S01 @ beta @ beta_S11_beta_inv
    except np.linalg.LinAlgError:
        alpha = np.zeros_like(beta)

    return JohansenResult(
        rank=rank,
        trace_stats=trace_stats,
        trace_crit=trace_crit,
        trace_pvalues=trace_pvalues,
        max_eigen_stats=max_eigen_stats,
        max_eigen_crit=max_eigen_crit,
        max_eigen_pvalues=max_eigen_pvalues,
        eigenvalues=eigenvalues,
        eigenvectors=beta,
        adjustment=alpha,
        lags=lags,
        n_obs=T,
        n_vars=n_vars,
        det_order=det_order,
        alpha=alpha,
    )


def _johansen_pvalue(stat: float, cv_dict: dict) -> float:
    """
    Compute approximate p-value for Johansen statistic.

    Uses linear interpolation between critical values.
    """
    if not cv_dict:
        return np.nan

    cv_90 = cv_dict.get("90%", np.nan)
    cv_95 = cv_dict.get("95%", np.nan)
    cv_99 = cv_dict.get("99%", np.nan)

    if np.isnan(cv_90):
        return np.nan

    if stat <= cv_90:
        return 0.15  # Above 10%
    elif stat <= cv_95:
        return 0.10 - 0.05 * (stat - cv_90) / (cv_95 - cv_90)
    elif stat <= cv_99:
        return 0.05 - 0.04 * (stat - cv_95) / (cv_99 - cv_95)
    else:
        return 0.005


def engle_granger_test(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Simpler alternative to Johansen for bivariate case.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (1D)
    x : np.ndarray
        Independent variable(s) (1D or 2D)
    alpha : float
        Significance level

    Returns
    -------
    dict
        Contains cointegrating regression coefficients, residuals,
        ADF test on residuals, and cointegration decision

    Example
    -------
    >>> np.random.seed(42)
    >>> n = 200
    >>> trend = np.cumsum(np.random.randn(n))
    >>> y = trend + np.random.randn(n) * 0.3
    >>> x = 2 * trend + np.random.randn(n) * 0.3
    >>> result = engle_granger_test(y, x)
    >>> print(f"Cointegrated: {result['is_cointegrated']}")

    References
    ----------
    Engle & Granger (1987). "Co-integration and error correction:
    Representation, estimation, and testing." Econometrica 55: 251-276.
    """
    from causal_inference.timeseries.stationarity import adf_test

    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n = len(y)

    if x.shape[0] != n:
        raise ValueError(f"Length mismatch: y ({n}) vs x ({x.shape[0]})")

    # Step 1: Cointegrating regression
    # y_t = β_0 + β_1 x_t + u_t
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Step 2: Test residuals for stationarity
    # Use ADF test on residuals
    # Critical values are different for cointegration residuals (more negative)
    # but we use standard ADF as approximation
    adf_result = adf_test(residuals, regression="n", alpha=alpha)

    # Cointegration critical values (MacKinnon 1991)
    # These are more negative than standard ADF
    # For 2 variables: 1%: -3.90, 5%: -3.34, 10%: -3.04
    coint_cv = {"1%": -3.90, "5%": -3.34, "10%": -3.04}

    # Determine cointegration based on cointegration-specific CVs
    if alpha <= 0.01:
        cv = coint_cv["1%"]
    elif alpha <= 0.05:
        cv = coint_cv["5%"]
    else:
        cv = coint_cv["10%"]

    is_cointegrated = adf_result.statistic < cv

    return {
        "beta": beta,
        "residuals": residuals,
        "adf_result": adf_result,
        "coint_critical_values": coint_cv,
        "is_cointegrated": is_cointegrated,
    }
