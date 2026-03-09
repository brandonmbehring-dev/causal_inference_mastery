"""
Recentered Influence Function (RIF) regression for unconditional QTE.

This module implements the RIF-OLS approach of Firpo, Fortin, & Lemieux (2009)
for estimating unconditional quantile treatment effects with covariates.

The key advantage over conditional quantile regression: RIF-OLS recovers the
marginal effect on the unconditional quantile, not the conditional quantile.

RIF(Y; q_tau) = q_tau + (tau - I(Y <= q_tau)) / f_Y(q_tau)

where:
- q_tau = sample quantile at tau
- f_Y(q_tau) = kernel density estimate at the quantile
- I(.) = indicator function

References
----------
- Firpo, S. (2007). Efficient Semiparametric Estimation of Quantile Treatment Effects.
- Firpo, S., Fortin, N., & Lemieux, T. (2009). Unconditional Quantile Regressions.
"""

from typing import List, Literal, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

from .types import QTEBandResult, QTEResult


def rif_qte(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    covariates: Optional[Union[np.ndarray, List[List[float]]]] = None,
    quantile: float = 0.5,
    bandwidth: Literal["silverman", "scott", "auto"] = "silverman",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> QTEResult:
    """
    Estimate unconditional QTE via RIF-OLS (Firpo et al. 2009).

    Computes the Recentered Influence Function for the quantile, then
    regresses it on treatment (and optionally covariates). The treatment
    coefficient is the marginal effect on the unconditional quantile.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y of shape (n,).
    treatment : np.ndarray or list
        Binary treatment indicator T of shape (n,).
    covariates : np.ndarray or list, optional
        Covariates X of shape (n, p). If provided, controls for X while
        still recovering unconditional (marginal) effect.
    quantile : float, default=0.5
        The quantile tau in (0, 1) at which to estimate the effect.
    bandwidth : str, default="silverman"
        Bandwidth selection method for kernel density estimation.
        Options: "silverman", "scott", "auto".
    n_bootstrap : int, default=1000
        Number of bootstrap replications for standard error estimation.
    alpha : float, default=0.05
        Significance level for confidence interval.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    QTEResult
        Dictionary containing:
        - tau_q: RIF treatment coefficient (unconditional QTE)
        - se: Bootstrap standard error
        - ci_lower, ci_upper: Confidence interval
        - method: "rif"
        - inference: "bootstrap"

    Raises
    ------
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
    >>> result = rif_qte(outcome, treatment, X, quantile=0.5)
    >>> print(f"Unconditional QTE (RIF): {result['tau_q']:.3f}")

    Notes
    -----
    The RIF-OLS approach has the following interpretation:

    1. Without covariates: Equivalent to unconditional_qte (difference in quantiles)

    2. With covariates: Recovers the UNCONDITIONAL effect by:
       - Computing RIF for each observation
       - Regressing RIF on treatment and covariates
       - Treatment coefficient = marginal effect on unconditional quantile

    This differs from conditional quantile regression, which estimates effects
    on CONDITIONAL quantiles (different interpretation).

    When to use RIF-OLS vs conditional QTE:
    - RIF-OLS: "What is the average effect on the 10th percentile of wages?"
    - CQR: "What is the effect for someone at the 10th percentile given their X?"
    """
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n = len(outcome)

    # Basic validation
    if n == 0:
        raise ValueError("CRITICAL ERROR: Empty input arrays.\nFunction: rif_qte")

    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: rif_qte\n"
            f"Got: outcome={n}, treatment={len(treatment)}"
        )

    if np.any(np.isnan(outcome)) or np.any(np.isnan(treatment)):
        raise ValueError("CRITICAL ERROR: NaN values detected.\nFunction: rif_qte")

    if np.any(np.isinf(outcome)) or np.any(np.isinf(treatment)):
        raise ValueError("CRITICAL ERROR: Infinite values detected.\nFunction: rif_qte")

    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(f"CRITICAL ERROR: Treatment must be binary.\nGot: {unique_treatment}")

    if len(unique_treatment) < 2:
        raise ValueError("CRITICAL ERROR: No treatment variation.\nFunction: rif_qte")

    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"CRITICAL ERROR: Invalid quantile.\nExpected: in (0, 1), Got: {quantile}")

    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"CRITICAL ERROR: Invalid alpha.\nExpected: in (0, 1), Got: {alpha}")

    # Handle covariates
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if len(covariates) != n:
            raise ValueError(
                f"CRITICAL ERROR: Covariates length mismatch.\n"
                f"Got: outcome={n}, covariates={len(covariates)}"
            )
        if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
            raise ValueError("CRITICAL ERROR: NaN/Inf in covariates.\nFunction: rif_qte")

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    # ========================================================================
    # COMPUTE RIF
    # ========================================================================

    rif = _compute_rif(outcome, quantile, bandwidth)

    # ========================================================================
    # OLS REGRESSION OF RIF ON TREATMENT (+ COVARIATES)
    # ========================================================================

    # Design matrix
    if covariates is not None:
        X = np.column_stack([np.ones(n), treatment, covariates])
    else:
        X = np.column_stack([np.ones(n), treatment])

    # Point estimate via OLS
    beta = np.linalg.lstsq(X, rif, rcond=None)[0]
    tau_q = beta[1]  # Treatment coefficient

    # ========================================================================
    # BOOTSTRAP INFERENCE
    # ========================================================================

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)

        y_boot = outcome[idx]
        t_boot = treatment[idx]
        rif_boot = _compute_rif(y_boot, quantile, bandwidth)

        if covariates is not None:
            X_boot = np.column_stack([np.ones(n), t_boot, covariates[idx]])
        else:
            X_boot = np.column_stack([np.ones(n), t_boot])

        beta_boot = np.linalg.lstsq(X_boot, rif_boot, rcond=None)[0]
        bootstrap_estimates[b] = beta_boot[1]

    se = np.std(bootstrap_estimates, ddof=1)
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return QTEResult(
        tau_q=float(tau_q),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        quantile=quantile,
        method="rif",
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        outcome_support=(float(outcome.min()), float(outcome.max())),
        inference="bootstrap",
        pvalue=None,  # Not computed
    )


def rif_qte_band(
    outcome: Union[np.ndarray, List[float]],
    treatment: Union[np.ndarray, List[int]],
    covariates: Optional[Union[np.ndarray, List[List[float]]]] = None,
    quantiles: Optional[List[float]] = None,
    bandwidth: Literal["silverman", "scott", "auto"] = "silverman",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    joint: bool = False,
    random_state: Optional[int] = None,
) -> QTEBandResult:
    """
    Estimate RIF-based unconditional QTE across multiple quantiles.

    Parameters
    ----------
    outcome : np.ndarray or list
        Outcome variable Y.
    treatment : np.ndarray or list
        Binary treatment indicator.
    covariates : np.ndarray or list, optional
        Covariates X.
    quantiles : list of float, optional
        Quantiles to estimate. Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    bandwidth : str, default="silverman"
        Bandwidth method for KDE.
    n_bootstrap : int, default=1000
        Bootstrap replications.
    alpha : float, default=0.05
        Significance level.
    joint : bool, default=False
        Compute joint confidence band.
    random_state : int, optional
        Random seed.

    Returns
    -------
    QTEBandResult
        Arrays of estimates across quantiles.
    """
    # Default quantiles
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    quantiles_arr = np.asarray(quantiles)

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    n = len(outcome)
    n_quantiles = len(quantiles)

    # Validate
    if np.any(quantiles_arr <= 0) or np.any(quantiles_arr >= 1):
        raise ValueError(f"CRITICAL ERROR: Quantiles must be in (0, 1). Got: {quantiles}")

    n_treated = int(np.sum(treatment == 1))
    n_control = int(np.sum(treatment == 0))

    rng = np.random.default_rng(random_state)

    # ========================================================================
    # POINT ESTIMATES AT EACH QUANTILE
    # ========================================================================

    qte_estimates = np.zeros(n_quantiles)

    for i, q in enumerate(quantiles):
        rif = _compute_rif(outcome, q, bandwidth)
        if covariates is not None:
            X = np.column_stack([np.ones(n), treatment, covariates])
        else:
            X = np.column_stack([np.ones(n), treatment])
        beta = np.linalg.lstsq(X, rif, rcond=None)[0]
        qte_estimates[i] = beta[1]

    # ========================================================================
    # BOOTSTRAP INFERENCE
    # ========================================================================

    bootstrap_matrix = np.zeros((n_bootstrap, n_quantiles))

    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_boot = outcome[idx]
        t_boot = treatment[idx]

        for i, q in enumerate(quantiles):
            rif_boot = _compute_rif(y_boot, q, bandwidth)
            if covariates is not None:
                X_boot = np.column_stack([np.ones(n), t_boot, covariates[idx]])
            else:
                X_boot = np.column_stack([np.ones(n), t_boot])
            beta_boot = np.linalg.lstsq(X_boot, rif_boot, rcond=None)[0]
            bootstrap_matrix[b, i] = beta_boot[1]

    se_estimates = np.std(bootstrap_matrix, axis=0, ddof=1)
    ci_lower = np.percentile(bootstrap_matrix, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(bootstrap_matrix, 100 * (1 - alpha / 2), axis=0)

    # Joint CI
    joint_ci_lower: Optional[np.ndarray] = None
    joint_ci_upper: Optional[np.ndarray] = None

    if joint:
        t_stats = np.abs(bootstrap_matrix - qte_estimates) / np.maximum(se_estimates, 1e-10)
        sup_t_stats = np.max(t_stats, axis=1)
        critical_value = np.percentile(sup_t_stats, 100 * (1 - alpha))
        joint_ci_lower = qte_estimates - critical_value * se_estimates
        joint_ci_upper = qte_estimates + critical_value * se_estimates

    return QTEBandResult(
        quantiles=quantiles_arr,
        qte_estimates=qte_estimates,
        se_estimates=se_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        joint_ci_lower=joint_ci_lower,
        joint_ci_upper=joint_ci_upper,
        method="rif",
        n_bootstrap=n_bootstrap,
        n_treated=n_treated,
        n_control=n_control,
        n_total=n,
        alpha=alpha,
    )


def _compute_rif(
    outcome: np.ndarray,
    quantile: float,
    bandwidth: str = "silverman",
) -> np.ndarray:
    """
    Compute Recentered Influence Function for a quantile.

    RIF(Y; q_tau) = q_tau + (tau - I(Y <= q_tau)) / f_Y(q_tau)

    Parameters
    ----------
    outcome : np.ndarray
        Outcome values.
    quantile : float
        Target quantile tau.
    bandwidth : str
        Bandwidth method for kernel density.

    Returns
    -------
    np.ndarray
        RIF values for each observation.
    """
    # Sample quantile
    q_tau = np.quantile(outcome, quantile)

    # Kernel density estimate at the quantile
    try:
        if bandwidth == "silverman":
            kde = gaussian_kde(outcome, bw_method="silverman")
        elif bandwidth == "scott":
            kde = gaussian_kde(outcome, bw_method="scott")
        else:  # auto
            kde = gaussian_kde(outcome)

        f_q = kde(q_tau)[0]
    except np.linalg.LinAlgError:
        # Fallback: use histogram-based estimate if KDE fails
        h = 1.06 * np.std(outcome) * len(outcome) ** (-1 / 5)  # Silverman
        f_q = np.mean(np.abs(outcome - q_tau) < h) / (2 * h)

    # Avoid division by zero
    f_q = max(f_q, 1e-10)

    # RIF = q_tau + (tau - I(Y <= q_tau)) / f_Y(q_tau)
    indicator = (outcome <= q_tau).astype(float)
    rif = q_tau + (quantile - indicator) / f_q

    return rif
