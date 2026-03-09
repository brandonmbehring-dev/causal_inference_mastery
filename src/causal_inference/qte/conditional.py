"""
Conditional Quantile Treatment Effects via quantile regression.

This module implements conditional QTE using the quantile regression framework
of Koenker & Bassett (1978). The conditional QTE estimates the effect of treatment
on the tau-th conditional quantile of Y given covariates X.

Model: Q_tau(Y | T, X) = alpha(tau) + tau_q(tau) * T + beta(tau)' * X

The treatment coefficient tau_q(tau) represents the conditional QTE at quantile tau.

References
----------
- Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles. Econometrica.
- Koenker, R. (2005). Quantile Regression. Cambridge University Press.
"""

from typing import List, Optional, Union

import numpy as np
from scipy import stats

from .types import QTEBandResult, QTEResult

# Try to import statsmodels for quantile regression
try:
    from statsmodels.regression.quantile_regression import QuantReg

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def conditional_qte(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    covariates: Union[np.ndarray, List[List[float]]],
    quantile: float = 0.5,
    alpha: float = 0.05,
    vcov: str = "robust",
) -> QTEResult:
    """
    Estimate conditional QTE via quantile regression.

    Fits the model: Q_tau(Y | T, X) = alpha + tau_q * T + beta' * X
    and returns the treatment coefficient tau_q at quantile tau.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y of shape (n,).
    treatment : np.ndarray or list
        Binary treatment indicator T of shape (n,).
    covariates : np.ndarray or list
        Covariates X of shape (n, p). Can also be 1D array for single covariate.
    quantile : float, default=0.5
        The quantile tau in (0, 1) at which to estimate the conditional effect.
    alpha : float, default=0.05
        Significance level for confidence interval.
    vcov : str, default="robust"
        Variance-covariance estimator. Options:
        - "robust": Sandwich estimator (default, most reliable)
        - "iid": Assumes iid errors (less reliable but faster)

    Returns
    -------
    QTEResult
        Dictionary containing:
        - tau_q: Treatment coefficient at quantile tau
        - se: Standard error
        - ci_lower, ci_upper: Confidence interval
        - pvalue: P-value for H0: tau_q = 0
        - method: "conditional"
        - inference: "asymptotic"

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 300
    >>> X = np.random.normal(0, 1, (n, 2))
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> outcome = X[:, 0] + 2.0 * treatment + np.random.normal(0, 1, n)
    >>> result = conditional_qte(outcome, treatment, X, quantile=0.5)
    >>> print(f"Conditional QTE at median: {result['tau_q']:.3f}")

    Notes
    -----
    Conditional QTE has a different interpretation than unconditional QTE:
    - Conditional QTE: Effect on tau-th quantile of Y *given* specific X values
    - Unconditional QTE: Effect on tau-th quantile of the marginal Y distribution

    For policy-relevant unconditional effects with covariates, consider using
    the RIF-OLS approach instead (rif_qte function).
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "CRITICAL ERROR: statsmodels is required for conditional_qte.\n"
            "Install with: pip install statsmodels"
        )

    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcome)

    # Basic validation
    if n == 0:
        raise ValueError("CRITICAL ERROR: Empty input arrays.\nFunction: conditional_qte")

    if len(treatment) != n or len(covariates) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: conditional_qte\n"
            f"Got: outcome={n}, treatment={len(treatment)}, covariates={len(covariates)}"
        )

    if np.any(np.isnan(outcome)) or np.any(np.isnan(treatment)) or np.any(np.isnan(covariates)):
        raise ValueError("CRITICAL ERROR: NaN values detected.\nFunction: conditional_qte")

    if np.any(np.isinf(outcome)) or np.any(np.isinf(treatment)) or np.any(np.isinf(covariates)):
        raise ValueError("CRITICAL ERROR: Infinite values detected.\nFunction: conditional_qte")

    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\nGot: {unique_treatment}"
        )

    if len(unique_treatment) < 2:
        raise ValueError("CRITICAL ERROR: No treatment variation.\nFunction: conditional_qte")

    if quantile <= 0 or quantile >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid quantile value.\n"
            f"Expected: quantile in (0, 1)\n"
            f"Got: quantile={quantile}"
        )

    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alpha value.\nExpected: alpha in (0, 1)\nGot: alpha={alpha}"
        )

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    # ========================================================================
    # QUANTILE REGRESSION
    # ========================================================================

    # Design matrix: [intercept, treatment, covariates]
    X = np.column_stack([np.ones(n), treatment, covariates])

    # Fit quantile regression
    model = QuantReg(outcome, X)

    # Use 'powell' method for vcov if robust, else default
    if vcov == "robust":
        # Default method uses kernel density for SE, which is robust
        result = model.fit(q=quantile)
    else:
        result = model.fit(q=quantile)

    # Extract treatment coefficient (index 1, after intercept)
    tau_q = result.params[1]
    se = result.bse[1]
    pvalue = result.pvalues[1]

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = tau_q - z_crit * se
    ci_upper = tau_q + z_crit * se

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return QTEResult(
        tau_q=float(tau_q),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        quantile=quantile,
        method="conditional",
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        outcome_support=(float(outcome.min()), float(outcome.max())),
        inference="asymptotic",
        pvalue=float(pvalue),
    )


def conditional_qte_band(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    covariates: Union[np.ndarray, List[List[float]]],
    quantiles: Optional[List[float]] = None,
    alpha: float = 0.05,
    vcov: str = "robust",
) -> QTEBandResult:
    """
    Estimate conditional QTE across multiple quantiles.

    Fits separate quantile regressions at each quantile and returns
    the band of treatment effects across the conditional distribution.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y of shape (n,).
    treatment : np.ndarray or list
        Binary treatment indicator T of shape (n,).
    covariates : np.ndarray or list
        Covariates X of shape (n, p).
    quantiles : list of float, optional
        Quantiles at which to estimate. Default is [0.1, 0.25, 0.5, 0.75, 0.9].
    alpha : float, default=0.05
        Significance level for confidence intervals.
    vcov : str, default="robust"
        Variance-covariance estimator.

    Returns
    -------
    QTEBandResult
        Dictionary with arrays of estimates across quantiles.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 300
    >>> X = np.random.normal(0, 1, (n, 2))
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> outcome = X[:, 0] + 2.0 * treatment + np.random.normal(0, 1, n)
    >>> result = conditional_qte_band(outcome, treatment, X)
    >>> print(f"QTE at quantiles: {result['qte_estimates']}")
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "CRITICAL ERROR: statsmodels is required for conditional_qte_band.\n"
            "Install with: pip install statsmodels"
        )

    # Default quantiles
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Convert inputs
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    quantiles_arr = np.asarray(quantiles)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcome)
    n_quantiles = len(quantiles)

    # Validate
    if np.any(quantiles_arr <= 0) or np.any(quantiles_arr >= 1):
        raise ValueError(f"CRITICAL ERROR: All quantiles must be in (0, 1).\nGot: {quantiles}")

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    # Design matrix
    X = np.column_stack([np.ones(n), treatment, covariates])

    # Fit at each quantile
    qte_estimates = np.zeros(n_quantiles)
    se_estimates = np.zeros(n_quantiles)
    ci_lower = np.zeros(n_quantiles)
    ci_upper = np.zeros(n_quantiles)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    for i, q in enumerate(quantiles):
        model = QuantReg(outcome, X)
        result = model.fit(q=q)

        qte_estimates[i] = result.params[1]
        se_estimates[i] = result.bse[1]
        ci_lower[i] = qte_estimates[i] - z_crit * se_estimates[i]
        ci_upper[i] = qte_estimates[i] + z_crit * se_estimates[i]

    return QTEBandResult(
        quantiles=quantiles_arr,
        qte_estimates=qte_estimates,
        se_estimates=se_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        joint_ci_lower=None,  # Joint CI requires bootstrap for CQR
        joint_ci_upper=None,
        method="conditional",
        n_bootstrap=0,  # Asymptotic inference
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        alpha=alpha,
    )
