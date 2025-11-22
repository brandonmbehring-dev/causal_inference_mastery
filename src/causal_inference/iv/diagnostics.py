"""
Weak Instrument Diagnostics for Instrumental Variables.

This module provides diagnostic tools for detecting and handling weak instruments
in IV regression. Key diagnostics include Stock-Yogo critical values, Cragg-Donald
statistic, and Anderson-Rubin confidence intervals.

References
----------
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression.
  In D. W. K. Andrews & J. H. Stock (Eds.), Identification and Inference for Econometric
  Models: Essays in Honor of Thomas Rothenberg (pp. 80-108). Cambridge University Press.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single equation
  in a complete system of stochastic equations. Annals of Mathematical Statistics, 20(1), 46-63.
- Angrist & Pischke (2009). Mostly Harmless Econometrics, Section 4.6.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


# Stock-Yogo (2005) Critical Values for Weak Instrument Test
# Table shows critical values for first-stage F-statistic
# Null hypothesis: Maximum bias of 2SLS estimator exceeds r% of OLS bias
# Rows: Number of instruments (q)
# Columns: Number of endogenous regressors (p)
# Values: Critical values at 10% maximal bias

STOCK_YOGO_CRITICAL_VALUES = {
    "10pct_maximal_bias": {
        # q instruments, p endogenous → critical value
        (1, 1): 16.38,  # 1 instrument, 1 endogenous
        (2, 1): 19.93,
        (3, 1): 22.30,
        (4, 1): 24.58,
        (5, 1): 26.87,
        (10, 1): 38.54,
        (1, 2): 7.03,  # 1 instrument, 2 endogenous (underidentified)
        (2, 2): 11.04,
        (3, 2): 13.43,
        (4, 2): 15.09,
        (5, 2): 16.76,
    },
    "15pct_maximal_bias": {
        (1, 1): 8.96,
        (2, 1): 11.59,
        (3, 1): 12.83,
        (4, 1): 13.96,
        (5, 1): 15.09,
        (10, 1): 22.30,
        (2, 2): 7.56,
        (3, 2): 9.48,
        (4, 2): 10.83,
        (5, 2): 12.20,
    },
    "20pct_maximal_bias": {
        (1, 1): 6.66,
        (2, 1): 8.75,
        (3, 1): 9.93,
        (4, 1): 11.04,
        (5, 1): 12.20,
        (10, 1): 18.37,
        (2, 2): 6.28,
        (3, 2): 8.00,
        (4, 2): 9.31,
        (5, 2): 10.63,
    },
}


def classify_instrument_strength(
    f_statistic: float,
    n_instruments: int = 1,
    n_endogenous: int = 1,
    bias_threshold: Literal["10pct", "15pct", "20pct"] = "10pct",
) -> Tuple[str, float, str]:
    """
    Classify instrument strength using Stock-Yogo critical values.

    Parameters
    ----------
    f_statistic : float
        First-stage F-statistic for joint significance of instruments.
    n_instruments : int, default=1
        Number of instrumental variables (q).
    n_endogenous : int, default=1
        Number of endogenous regressors (p).
    bias_threshold : {"10pct", "15pct", "20pct"}, default="10pct"
        Maximal bias threshold (10%, 15%, or 20% of OLS bias).

    Returns
    -------
    classification : str
        One of: "strong", "weak", "very_weak"
    critical_value : float
        Stock-Yogo critical value for given (q, p) and bias threshold.
        Returns np.nan if no critical value available.
    interpretation : str
        Human-readable interpretation of the result.

    Notes
    -----
    Classification rules:
    - Strong: F > critical value (instruments pass Stock-Yogo test)
    - Weak: 10 < F <= critical value (instruments fail Stock-Yogo test)
    - Very weak: F <= 10 (instruments are severely weak)

    If no critical value is available (e.g., q=1, p>2), uses rule of thumb:
    - Strong: F > 20 (conventional threshold from Angrist & Pischke)
    - Weak: 10 < F <= 20
    - Very weak: F <= 10

    Examples
    --------
    >>> classify_instrument_strength(f_statistic=25.0, n_instruments=1, n_endogenous=1)
    ('strong', 16.38, 'Instruments pass Stock-Yogo test (F=25.00 > 16.38)')

    >>> classify_instrument_strength(f_statistic=12.0, n_instruments=1, n_endogenous=1)
    ('weak', 16.38, 'Instruments fail Stock-Yogo test (F=12.00 <= 16.38)')

    >>> classify_instrument_strength(f_statistic=8.0, n_instruments=1, n_endogenous=1)
    ('very_weak', 16.38, 'Instruments are severely weak (F=8.00 <= 10)')
    """
    # Get critical value from Stock-Yogo table
    key = f"{bias_threshold}_maximal_bias"
    critical_value = STOCK_YOGO_CRITICAL_VALUES[key].get((n_instruments, n_endogenous), np.nan)

    # Classify using critical value if available
    if not np.isnan(critical_value):
        if f_statistic > critical_value:
            classification = "strong"
            interpretation = (
                f"Instruments pass Stock-Yogo test (F={f_statistic:.2f} > {critical_value:.2f})"
            )
        elif f_statistic > 10:
            classification = "weak"
            interpretation = (
                f"Instruments fail Stock-Yogo test (F={f_statistic:.2f} <= {critical_value:.2f})"
            )
        else:
            classification = "very_weak"
            interpretation = f"Instruments are severely weak (F={f_statistic:.2f} <= 10)"
    else:
        # Fallback to rule of thumb (Angrist & Pischke 2009)
        if f_statistic > 20:
            classification = "strong"
            interpretation = f"Instruments likely strong (F={f_statistic:.2f} > 20, rule of thumb)"
        elif f_statistic > 10:
            classification = "weak"
            interpretation = (
                f"Instruments may be weak (10 < F={f_statistic:.2f} <= 20, rule of thumb)"
            )
        else:
            classification = "very_weak"
            interpretation = f"Instruments are severely weak (F={f_statistic:.2f} <= 10)"

    return classification, critical_value, interpretation


def cragg_donald_statistic(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    X: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Cragg-Donald statistic for weak instrument test.

    The Cragg-Donald (CD) statistic is a multivariate generalization of the
    first-stage F-statistic. It tests whether instruments are weak when there
    are multiple endogenous regressors (p > 1).

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable (not used in calculation, but kept for API consistency).
    D : array-like, shape (n, p)
        Endogenous regressors (p endogenous variables).
    Z : array-like, shape (n, q)
        Instrumental variables (q instruments).
    X : array-like, shape (n, k), optional
        Exogenous controls.

    Returns
    -------
    cd_stat : float
        Cragg-Donald statistic. Compare to Stock-Yogo critical values.

    Notes
    -----
    The Cragg-Donald statistic is defined as:

        CD = (n - k - q) / q × min eigenvalue of (Π̂' Z'Z Π̂ / σ̂²)

    where:
    - Π̂ is the matrix of first-stage coefficients (q × p)
    - σ̂² is the residual variance from first stage
    - n is sample size, k is number of controls, q is number of instruments

    For just-identified case (q = p = 1), CD reduces to F-statistic.

    References
    ----------
    Cragg, J. G., & Donald, S. G. (1993). Testing identifiability and
    specification in instrumental variable models. Econometric Theory, 9(2), 222-240.

    Examples
    --------
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, (n, 2))  # 2 instruments
    >>> D = Z @ [0.5, 0.3] + np.random.normal(0, 1, (n, 2))  # 2 endogenous
    >>> Y = D @ [0.2, 0.1] + np.random.normal(0, 1, n)
    >>> cd = cragg_donald_statistic(Y, D, Z)
    >>> cd > 10  # Check if instruments are strong
    """
    # Convert to arrays
    D = np.asarray(D)
    Z = np.asarray(Z)
    X = np.asarray(X) if X is not None else None

    # Ensure 2D
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = D.shape[0]
    p = D.shape[1]  # Number of endogenous
    q = Z.shape[1]  # Number of instruments

    # Construct first-stage design matrix: [Z, X] or just Z
    if X is not None:
        ZX = np.column_stack([Z, X])
        k = X.shape[1]
    else:
        ZX = Z
        k = 0

    # Add constant
    import statsmodels.api as sm

    ZX = sm.add_constant(ZX, has_constant="add")

    # First-stage regression: D ~ Z + X
    # Fit for each endogenous variable
    Pi_hat = []  # Coefficients on instruments
    sigma_sq_sum = 0.0

    for j in range(p):
        model = sm.OLS(D[:, j], ZX)
        result = model.fit()

        # Extract coefficients on instruments (exclude constant and controls)
        pi_j = result.params[1 : q + 1]
        Pi_hat.append(pi_j)

        # Residual variance
        sigma_sq_sum += result.scale

    Pi_hat = np.column_stack(Pi_hat)  # Shape: (q, p)
    sigma_sq = sigma_sq_sum / p  # Average residual variance

    # Cragg-Donald statistic
    # CD = (n - k - q) / q × min eigenvalue of (Π̂' (Z'Z/n) Π̂ / σ̂²)

    # Compute Z'Z/n (sample covariance)
    ZtZ_over_n = (Z.T @ Z) / n

    # Compute Π̂' (Z'Z/n) Π̂
    M = Pi_hat.T @ ZtZ_over_n @ Pi_hat / sigma_sq

    # Minimum eigenvalue
    eigenvalues = np.linalg.eigvalsh(M)
    min_eigenvalue = eigenvalues.min()

    # Cragg-Donald statistic
    cd_stat = ((n - k - q) / q) * min_eigenvalue

    return cd_stat


def anderson_rubin_test(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    X: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_grid: int = 100,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Anderson-Rubin test and confidence interval for IV regression.

    The Anderson-Rubin (AR) test is robust to weak instruments. It inverts
    the AR statistic to construct a confidence interval for the treatment effect.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable.
    D : array-like, shape (n,) or (n, 1)
        Endogenous treatment variable (currently supports p=1 only).
    Z : array-like, shape (n,) or (n, q)
        Instrumental variables.
    X : array-like, shape (n, k), optional
        Exogenous controls.
    alpha : float, default=0.05
        Significance level (default: 95% CI).
    n_grid : int, default=100
        Number of grid points for CI inversion.

    Returns
    -------
    ar_statistic : float
        Anderson-Rubin test statistic (chi-squared distributed).
    p_value : float
        P-value for test of H₀: β = 0.
    ci : tuple of (float, float)
        (1 - alpha) confidence interval for treatment effect.

    Notes
    -----
    The AR statistic tests the null hypothesis H₀: β = β₀ by checking
    whether the reduced-form residuals (Y - β₀ D) are uncorrelated with
    instruments Z.

    AR statistic:
        AR(β₀) = (Y - β₀ D)' P_Z (Y - β₀ D) / σ̂²

    where P_Z = Z(Z'Z)⁻¹Z' is the projection matrix onto instruments.

    Under H₀ and weak instruments, AR(β₀) ~ χ²(q) where q = number of instruments.

    The confidence interval inverts the AR test: CI = {β : AR(β) <= χ²_α(q)}.

    **Important**: AR test is robust to weak instruments but has lower power
    than 2SLS when instruments are strong.

    References
    ----------
    Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
    equation in a complete system of stochastic equations. Annals of Mathematical
    Statistics, 20(1), 46-63.

    Examples
    --------
    >>> n = 500
    >>> Z = np.random.normal(0, 1, n)
    >>> D = 0.1 * Z + np.random.normal(0, 1, n)  # Weak instrument
    >>> Y = 0.5 * D + np.random.normal(0, 1, n)
    >>> ar_stat, p_val, ci = anderson_rubin_test(Y, D, Z)
    >>> print(f"AR CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    AR CI: [0.12, 0.88]  # Wide CI due to weak instrument
    """
    # Convert to arrays
    Y = np.asarray(Y).flatten()
    D = np.asarray(D).flatten()
    Z = np.asarray(Z)
    X = np.asarray(X) if X is not None else None

    # Ensure Z is 2D
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = len(Y)
    q = Z.shape[1]

    # Construct design matrix: [Z, X] or just Z
    if X is not None:
        ZX = np.column_stack([Z, X])
        k = X.shape[1]  # Number of controls
    else:
        ZX = Z
        k = 0

    # Add constant for reduced form regression
    import statsmodels.api as sm

    ZX_with_const = sm.add_constant(ZX, has_constant="add")

    # Projection matrix: P_Z = Z (Z'Z)⁻¹ Z' (instruments only, no X or constant)
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    P_Z = Z @ ZtZ_inv @ Z.T

    # Compute residual variance from unrestricted reduced form Y ~ Z + X
    # This is fixed across all β values
    model_unrestricted = sm.OLS(Y, ZX_with_const)
    result_unrestricted = model_unrestricted.fit()

    # Residual sum of squares from unrestricted model
    ssr_unrestricted = result_unrestricted.ssr
    df_unrestricted = n - (q + k + 1)  # n - (instruments + controls + constant)
    sigma_sq = ssr_unrestricted / df_unrestricted

    # Helper function: Compute AR statistic for given β
    def ar_statistic_beta(beta: float) -> float:
        """
        Compute AR(β) for given β.

        Formula: AR(β) = (1/q) × [ũ' P_Z ũ] / σ̂² ~ χ²(q)
        where:
        - ũ = Y - βD (residuals under H₀: β = β₀)
        - σ̂² = residual variance from unrestricted reduced form Y ~ Z + X

        This is the standard formulation from Angrist & Pischke (2009, Section 4.6.3)
        and Andrews, Moreira & Stock (2006).
        """
        # Reduced-form residuals under H₀: β = beta
        u_tilde = Y - beta * D

        # AR statistic
        ar = (1.0 / q) * (u_tilde.T @ P_Z @ u_tilde) / sigma_sq
        return ar

    # Test H₀: β = 0
    ar_stat_zero = ar_statistic_beta(0.0)
    p_value = 1 - stats.chi2.cdf(ar_stat_zero, df=q)

    # Invert AR test to find confidence interval
    # CI = {β : AR(β) <= χ²_α(q)}
    chi2_critical = stats.chi2.ppf(1 - alpha, df=q)

    # Grid search to find CI bounds
    # Start with rough bounds from 2SLS estimate
    from .two_stage_least_squares import TwoStageLeastSquares

    iv = TwoStageLeastSquares(inference="standard")
    iv.fit(Y, D, Z, X)
    beta_2sls = iv.coef_[0]
    se_2sls = iv.se_[0]

    # Search grid: β_2sls ± 20 SE (wider for over-identified/weak IV cases)
    beta_min = beta_2sls - 20 * se_2sls
    beta_max = beta_2sls + 20 * se_2sls
    beta_grid = np.linspace(beta_min, beta_max, n_grid)

    ar_stats = np.array([ar_statistic_beta(b) for b in beta_grid])

    # Find where AR(β) crosses critical value
    in_ci = ar_stats <= chi2_critical

    if not in_ci.any():
        # CI is empty (very rare, instruments may be invalid)
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        # Find bounds
        ci_indices = np.where(in_ci)[0]
        ci_lower = beta_grid[ci_indices.min()]
        ci_upper = beta_grid[ci_indices.max()]

    return ar_stat_zero, p_value, (ci_lower, ci_upper)


def weak_instrument_summary(
    f_statistic: float,
    n_instruments: int,
    n_endogenous: int,
    cragg_donald: Optional[float] = None,
    ar_ci: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Generate comprehensive weak instrument diagnostic summary.

    Parameters
    ----------
    f_statistic : float
        First-stage F-statistic.
    n_instruments : int
        Number of instruments (q).
    n_endogenous : int
        Number of endogenous regressors (p).
    cragg_donald : float, optional
        Cragg-Donald statistic (for p > 1).
    ar_ci : tuple of (float, float), optional
        Anderson-Rubin confidence interval.

    Returns
    -------
    summary : pd.DataFrame
        Table with diagnostic results and interpretation.

    Examples
    --------
    >>> summary = weak_instrument_summary(
    ...     f_statistic=25.0,
    ...     n_instruments=1,
    ...     n_endogenous=1,
    ...     ar_ci=(0.08, 0.12)
    ... )
    >>> print(summary)
    """
    # Stock-Yogo classification
    classification, critical_value, interpretation = classify_instrument_strength(
        f_statistic, n_instruments, n_endogenous
    )

    # Build summary rows
    rows = []

    # First-stage F-statistic
    rows.append(
        {
            "Diagnostic": "First-Stage F-Statistic",
            "Value": f"{f_statistic:.2f}",
            "Interpretation": interpretation,
        }
    )

    # Stock-Yogo critical value
    if not np.isnan(critical_value):
        rows.append(
            {
                "Diagnostic": "Stock-Yogo Critical Value (10% bias)",
                "Value": f"{critical_value:.2f}",
                "Interpretation": f"Reject weak IV if F > {critical_value:.2f}",
            }
        )

    # Cragg-Donald statistic
    if cragg_donald is not None:
        rows.append(
            {
                "Diagnostic": "Cragg-Donald Statistic",
                "Value": f"{cragg_donald:.2f}",
                "Interpretation": "Multivariate weak IV test (compare to Stock-Yogo)",
            }
        )

    # Anderson-Rubin CI
    if ar_ci is not None:
        ci_lower, ci_upper = ar_ci
        if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
            rows.append(
                {
                    "Diagnostic": "Anderson-Rubin 95% CI",
                    "Value": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                    "Interpretation": "Robust to weak instruments",
                }
            )

    # Overall recommendation
    if classification == "strong":
        recommendation = "✓ Instruments appear strong. 2SLS estimates are reliable."
    elif classification == "weak":
        recommendation = "⚠ Instruments may be weak. Consider LIML or AR CI."
    else:
        recommendation = "✗ Instruments are very weak. Use AR CI or find better instruments."

    rows.append({"Diagnostic": "Recommendation", "Value": "", "Interpretation": recommendation})

    return pd.DataFrame(rows)
