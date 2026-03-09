"""Inverse Probability Weighting for observational studies with confounding.

This module extends the RCT IPW estimator to handle observational data by adding
propensity score estimation from covariates, weight trimming, and stabilization.

References
----------
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity
  score in observational studies for causal effects. Biometrika, 70(1), 41-55.
  doi:10.1093/biomet/70.1.41

- Austin, P. C., & Stuart, E. A. (2015). Moving towards best practice when using
  inverse probability of treatment weighting (IPTW) using the propensity score.
  Statistics in Medicine, 34(28), 3661-3679. doi:10.1002/sim.6607

- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient estimation of average
  treatment effects using the estimated propensity score. Econometrica, 71(4), 1161-1189.
  doi:10.1111/1468-0262.00442
"""

import warnings
import numpy as np
from typing import Dict, Any, Tuple, Union, Optional, TypedDict
from src.causal_inference.rct.estimators_ipw import ipw_ate
from src.causal_inference.observational.propensity import (
    estimate_propensity,
    trim_propensity,
    stabilize_weights,
)


class PropensitySummary(TypedDict):
    """Summary statistics for propensity scores."""

    min: float
    max: float
    mean: float
    median: float
    std: float


class IPWResult(TypedDict):
    """Return type for ipw_ate_observational() estimator."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    n_trimmed: int
    n_propensity_clipped: int
    propensity_diagnostics: Dict[str, Any]
    propensity_summary: PropensitySummary


def ipw_ate_observational(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    propensity: Optional[Union[np.ndarray, list]] = None,
    trim_at: Optional[Tuple[float, float]] = None,
    stabilize: bool = False,
    alpha: float = 0.05,
) -> IPWResult:
    """
    IPW ATE estimator for observational data with confounding.

    Workflow:
    1. Estimate propensity scores P(T=1|X) from covariates (or use provided)
    2. Optionally trim extreme propensity scores
    3. Optionally compute stabilized weights
    4. Call RCT IPW estimator with propensities
    5. Return results + propensity diagnostics

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Binary treatment indicator (1=treated, 0=control).
    covariates : np.ndarray or list
        Covariate matrix (n_samples, n_features). Can be 1D for single covariate.
    propensity : np.ndarray or list, optional
        Pre-computed propensity scores P(T=1|X). If None, estimated from covariates.
    trim_at : tuple of (float, float), optional
        Trim at (lower_percentile, upper_percentile), e.g., (0.01, 0.99).
        Units with extreme propensities are removed. Default: no trimming.
    stabilize : bool, default=False
        If True, use stabilized weights SW = P(T) / P(T|X).
        Reduces variance while maintaining unbiasedness.
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'estimate': IPW estimate of ATE
        - 'se': Standard error (robust)
        - 'ci_lower': Lower bound of (1-alpha)% CI
        - 'ci_upper': Upper bound of (1-alpha)% CI
        - 'n_treated': Number of treated units (after trimming)
        - 'n_control': Number of control units (after trimming)
        - 'n_trimmed': Number of units removed by trimming (0 if no trimming)
        - 'propensity_diagnostics': dict with AUC, pseudo-R², convergence (if estimated)
        - 'propensity_summary': dict with min/max/mean propensity scores

    Raises
    ------
    ValueError
        If inputs invalid, propensity estimation fails, or trimming removes all units.

    Examples
    --------
    >>> # Basic observational IPW
    >>> X = np.random.normal(0, 1, (100, 2))
    >>> logit = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    >>> T = (np.random.uniform(0, 1, 100) < 1 / (1 + np.exp(-logit))).astype(float)
    >>> Y = 3.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, 100)
    >>> result = ipw_ate_observational(Y, T, X)
    >>> result['estimate']  # Should be near 3.0
    >>> result['propensity_diagnostics']['auc']  # > 0.5 indicates confounding

    >>> # With trimming and stabilization
    >>> result = ipw_ate_observational(
    ...     Y, T, X,
    ...     trim_at=(0.01, 0.99),
    ...     stabilize=True
    ... )
    >>> result['n_trimmed']  # Units removed by trimming

    >>> # With pre-computed propensities (advanced)
    >>> prop_result = estimate_propensity(T, X)
    >>> result = ipw_ate_observational(Y, T, X, propensity=prop_result['propensity'])

    Notes
    -----
    - Propensity estimation uses logistic regression: logit(P(T=1|X)) = β₀ + β'X
    - AUC > 0.7 indicates strong confounding (good discrimination)
    - AUC ≈ 0.5 indicates weak confounding (T nearly independent of X)
    - Trimming reduces variance but introduces bias (bias-variance tradeoff)
    - Stabilized weights have mean ≈ 1 and lower variance than standard IPW
    - IPW assumes: positivity (0 < P(T=1|X) < 1), no unmeasured confounding
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Handle 1D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Check lengths match
    if len(treatment) != n or covariates.shape[0] != n:
        raise ValueError(
            f"CRITICAL ERROR: Input arrays have different lengths.\\n"
            f"Function: ipw_ate_observational\\n"
            f"Expected: All arrays length {n}\\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}, "
            f"covariates.shape={covariates.shape}"
        )

    # If propensity provided, validate
    if propensity is not None:
        propensity = np.asarray(propensity, dtype=float)
        if len(propensity) != n:
            raise ValueError(
                f"CRITICAL ERROR: Propensity length mismatch.\\n"
                f"Function: ipw_ate_observational\\n"
                f"Expected: len(propensity) == {n}\\n"
                f"Got: len(propensity) == {len(propensity)}"
            )

    # ============================================================================
    # Step 1: Estimate Propensity (if not provided)
    # ============================================================================

    propensity_diagnostics = {}

    if propensity is None:
        # Estimate propensity from covariates
        prop_result = estimate_propensity(treatment, covariates)
        propensity = prop_result["propensity"]
        propensity_diagnostics = prop_result["diagnostics"]
    else:
        # Pre-computed propensity provided
        propensity_diagnostics = {"provided": True}

    # ============================================================================
    # Step 2: Trim Extreme Propensities (if requested)
    # ============================================================================

    n_trimmed = 0

    if trim_at is not None:
        # Trim extreme propensities
        trim_result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=trim_at)

        # Update arrays to trimmed versions
        propensity = trim_result["propensity"]
        treatment = trim_result["treatment"]
        outcomes = trim_result["outcomes"]
        covariates = trim_result["covariates"]
        n_trimmed = trim_result["n_trimmed"]

        # Check that trimming didn't remove all units
        if trim_result["n_kept"] == 0:
            raise ValueError(
                f"CRITICAL ERROR: Trimming removed all units.\\n"
                f"Function: ipw_ate_observational\\n"
                f"trim_at={trim_at} removed all {n} units.\\n"
                f"Options: Relax trim_at thresholds or check propensity distribution."
            )

    # ============================================================================
    # Step 3: Stabilize Weights (if requested)
    # ============================================================================

    # Compute stabilized weights if requested
    stabilized_weights = None
    if stabilize:
        # Compute stabilized weights: SW = P(T) / P(T|X) for treated,
        #                              SW = (1-P(T)) / (1-P(T|X)) for control
        # Stabilized weights have mean ≈ 1.0 and reduce variance compared to
        # standard IPW weights, especially when propensity scores are extreme.
        stabilized_weights = stabilize_weights(propensity, treatment)

    # ============================================================================
    # Step 4: Clip Extreme Propensities
    # ============================================================================

    # RCT ipw_ate requires propensities in (0,1) exclusive.
    # With perfect separation or extreme confounding, propensities can hit 0 or 1.
    # Clip to [epsilon, 1-epsilon] to avoid division by zero in weight calculation.
    epsilon = 1e-6
    propensity_clipped = np.clip(propensity, epsilon, 1 - epsilon)

    n_propensity_clipped = int(np.sum((propensity < epsilon) | (propensity > 1 - epsilon)))
    if n_propensity_clipped > 0:
        # Warn user about clipping - silent failures are unacceptable
        warnings.warn(
            f"Propensity values clipped for numerical stability. "
            f"{n_propensity_clipped} values outside ({epsilon}, {1 - epsilon}) were clipped. "
            f"This may indicate positivity violations.",
            UserWarning,
        )

    # ============================================================================
    # Step 5: Call RCT IPW Estimator
    # ============================================================================

    # Pass stabilized weights if computed, otherwise ipw_ate computes standard weights
    ipw_result = ipw_ate(
        outcomes, treatment, propensity_clipped, alpha=alpha, weights=stabilized_weights
    )

    # ============================================================================
    # Step 5: Add Observational Diagnostics
    # ============================================================================

    # Propensity summary statistics
    propensity_summary = {
        "min": float(np.min(propensity)),
        "max": float(np.max(propensity)),
        "mean": float(np.mean(propensity)),
        "median": float(np.median(propensity)),
        "std": float(np.std(propensity)),
    }

    # Combine results
    result = {
        **ipw_result,  # ATE estimate, SE, CI, n_treated, n_control
        "n_trimmed": n_trimmed,
        "n_propensity_clipped": n_propensity_clipped,
        "stabilized": stabilize,
        "propensity_diagnostics": propensity_diagnostics,
        "propensity_summary": propensity_summary,
    }

    return result
