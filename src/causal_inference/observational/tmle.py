"""Targeted Maximum Likelihood Estimation (TMLE) for causal inference.

This module implements TMLE for ATE estimation, an improvement over standard
doubly robust (DR/AIPW) estimation that achieves the semiparametric efficiency
bound through an iterative targeting procedure.

Key Properties of TMLE
----------------------
1. **Double robustness**: Consistent if either propensity OR outcome model correct
2. **Efficiency**: Achieves the semiparametric efficiency bound when both correct
3. **Better finite-sample**: Targeting step improves bias compared to standard DR
4. **Valid inference**: Influence function-based standard errors are valid

Algorithm
---------
1. Estimate initial nuisance functions:
   - Propensity: g(X) = P(T=1|X)
   - Outcome: Q(T,X) = E[Y|T,X]

2. Targeting step (iterate until convergence):
   - Compute clever covariate: H = T/g - (1-T)/(1-g)
   - Fit fluctuation: Y ~ epsilon*H + offset(Q)
   - Update: Q* = Q + epsilon*H

3. Estimate ATE: mean(Q*(1,X)) - mean(Q*(0,X))

4. Inference via efficient influence function

References
----------
- van der Laan, M. J., & Rose, S. (2011). Targeted Learning: Causal Inference
  for Observational and Experimental Data. Springer.
- Schuler, M. S., & Rose, S. (2017). Targeted Maximum Likelihood Estimation
  for Causal Inference in Observational Studies. American Journal of
  Epidemiology, 185(1), 65-73. doi:10.1093/aje/kww165
- Gruber, S., & van der Laan, M. J. (2012). tmle: An R Package for Targeted
  Maximum Likelihood Estimation. Journal of Statistical Software, 51(13), 1-35.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, TypedDict

from .propensity import estimate_propensity, trim_propensity
from .outcome_regression import fit_outcome_models
from .tmle_helpers import (
    compute_clever_covariate,
    fit_fluctuation,
    check_convergence,
    compute_tmle_ate,
    compute_efficient_influence_function,
    compute_tmle_variance,
)


class TMLEResult(TypedDict):
    """Return type for tmle_ate() estimator."""

    # Core estimates
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float

    # Sample information
    n: int
    n_treated: int
    n_control: int
    n_trimmed: int

    # TMLE-specific
    epsilon: float
    n_iterations: int
    converged: bool
    convergence_criterion: float

    # Diagnostics
    propensity_diagnostics: Dict[str, Any]
    outcome_diagnostics: Dict[str, Any]

    # Intermediate values (for debugging/analysis)
    propensity: np.ndarray
    Q0_initial: np.ndarray
    Q1_initial: np.ndarray
    Q0_star: np.ndarray
    Q1_star: np.ndarray
    eif: np.ndarray


def tmle_ate(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    covariates: Union[np.ndarray, list],
    propensity: Optional[Union[np.ndarray, list]] = None,
    outcome_models: Optional[Dict[str, Any]] = None,
    trim_at: Optional[tuple] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> TMLEResult:
    """
    Targeted Maximum Likelihood Estimation for Average Treatment Effect.

    TMLE improves on standard doubly robust estimation by iteratively
    updating the outcome model to satisfy the efficient score equation,
    achieving the semiparametric efficiency bound.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units, shape (n,).
    treatment : np.ndarray or list
        Binary treatment indicator (1=treated, 0=control), shape (n,).
    covariates : np.ndarray or list
        Covariate matrix, shape (n, p). Can be 1D for single covariate.
    propensity : np.ndarray or list, optional
        Pre-computed propensity scores P(T=1|X), shape (n,).
        If None, estimated via logistic regression.
    outcome_models : dict, optional
        Pre-computed outcome model predictions.
        Expected keys: 'mu0_predictions', 'mu1_predictions'.
        If None, fitted via linear regression.
    trim_at : tuple of float, optional
        Propensity trimming percentiles (lower, upper). E.g., (0.01, 0.99).
        If None, no trimming applied.
    max_iter : int, default=100
        Maximum targeting iterations.
    tol : float, default=1e-6
        Convergence tolerance for targeting step.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    TMLEResult
        Dictionary containing:
        - estimate : TMLE estimate of ATE
        - se : Standard error (from efficient influence function)
        - ci_lower, ci_upper : Confidence interval bounds
        - n, n_treated, n_control, n_trimmed : Sample sizes
        - epsilon : Final fluctuation coefficient
        - n_iterations : Number of targeting iterations
        - converged : Whether targeting converged
        - convergence_criterion : Final score equation value
        - propensity_diagnostics : Propensity model fit information
        - outcome_diagnostics : Outcome model fit information
        - propensity : Propensity scores used
        - Q0_initial, Q1_initial : Initial outcome predictions
        - Q0_star, Q1_star : Targeted outcome predictions
        - eif : Efficient influence function values

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, non-binary treatment,
        or other validation failures.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.observational.tmle import tmle_ate
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)
    >>> T = (X[:, 0] + np.random.randn(n) > 0).astype(int)
    >>> Y = 2 * T + X[:, 0] + np.random.randn(n)
    >>> result = tmle_ate(Y, T, X)
    >>> print(f"ATE: {result['estimate']:.3f} (SE: {result['se']:.3f})")
    ATE: 2.012 (SE: 0.095)

    Notes
    -----
    TMLE vs Doubly Robust (DR/AIPW):
    - Both are doubly robust (consistent if either model correct)
    - TMLE achieves efficiency bound; DR may not
    - TMLE typically has lower finite-sample bias
    - TMLE requires iterative targeting; DR is one-shot

    The targeting step solves the efficient score equation:
        E[H * (Y - Q*)] = 0
    where H is the clever covariate and Q* is the updated outcome model.
    """
    # =========================================================================
    # Input validation
    # =========================================================================
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n = len(outcomes)

    # Validate array lengths
    if len(treatment) != n or len(covariates) != n:
        raise ValueError(
            f"CRITICAL ERROR: Array length mismatch. "
            f"outcomes={n}, treatment={len(treatment)}, covariates={len(covariates)}. "
            f"All arrays must have the same number of observations."
        )

    # Validate binary treatment
    unique_t = np.unique(treatment)
    if not np.array_equal(np.sort(unique_t), np.array([0.0, 1.0])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1). "
            f"Found values: {unique_t}. "
            f"Check for missing values or incorrect encoding."
        )

    # Validate finite values
    if not np.all(np.isfinite(outcomes)):
        n_nan = np.sum(~np.isfinite(outcomes))
        raise ValueError(
            f"CRITICAL ERROR: outcomes contains {n_nan} non-finite values. "
            f"Remove or impute NaN/Inf before estimation."
        )

    if not np.all(np.isfinite(covariates)):
        n_nan = np.sum(~np.isfinite(covariates))
        raise ValueError(
            f"CRITICAL ERROR: covariates contains {n_nan} non-finite values. "
            f"Remove or impute NaN/Inf before estimation."
        )

    # =========================================================================
    # Step 1: Estimate or validate propensity scores
    # =========================================================================
    n_trimmed = 0
    propensity_diagnostics = {}

    if propensity is None:
        prop_result = estimate_propensity(treatment, covariates)
        propensity = prop_result["propensity"]
        propensity_diagnostics = prop_result["diagnostics"]
    else:
        propensity = np.asarray(propensity, dtype=float)
        if len(propensity) != n:
            raise ValueError(
                f"CRITICAL ERROR: propensity length ({len(propensity)}) != n ({n}). "
                f"Propensity scores must be provided for all observations."
            )
        propensity_diagnostics = {"source": "user_provided"}

    # Apply trimming if specified
    if trim_at is not None:
        trim_result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at)
        outcomes = trim_result["outcomes"]
        treatment = trim_result["treatment"]
        covariates = trim_result["covariates"]
        propensity = trim_result["propensity"]
        n_trimmed = trim_result["n_trimmed"]
        n = len(outcomes)
        propensity_diagnostics["n_trimmed"] = n_trimmed

    # Clip propensity for numerical stability
    propensity = np.clip(propensity, 1e-6, 1 - 1e-6)

    # =========================================================================
    # Step 2: Estimate or validate outcome models
    # =========================================================================
    if outcome_models is None:
        outcome_results = fit_outcome_models(outcomes, treatment, covariates)
        Q0_initial = outcome_results["mu0_predictions"]
        Q1_initial = outcome_results["mu1_predictions"]
        outcome_diagnostics = outcome_results.get("diagnostics", {})
    else:
        Q0_initial = np.asarray(outcome_models["mu0_predictions"], dtype=float)
        Q1_initial = np.asarray(outcome_models["mu1_predictions"], dtype=float)
        outcome_diagnostics = {"source": "user_provided"}

    # =========================================================================
    # Step 3: Targeting step (iterative update)
    # =========================================================================
    # Create observed predictions based on actual treatment
    Q_observed = np.where(treatment == 1, Q1_initial, Q0_initial)

    # Compute clever covariate
    H = compute_clever_covariate(treatment, propensity)

    # Iterative targeting
    Q_star = Q_observed.copy()
    epsilon_total = 0.0
    converged = False
    convergence_criterion = np.inf

    for iteration in range(max_iter):
        # Check convergence
        converged, convergence_criterion = check_convergence(outcomes, Q_star, H, tol)

        if converged:
            break

        # Fit fluctuation
        epsilon, Q_star = fit_fluctuation(outcomes, H, Q_star)
        epsilon_total += epsilon

    n_iterations = iteration + 1

    # =========================================================================
    # Step 4: Compute targeted predictions for both treatment levels
    # =========================================================================
    # Apply total fluctuation to initial predictions
    H1 = 1.0 / propensity  # Clever covariate for treated
    H0 = -1.0 / (1 - propensity)  # Clever covariate for control

    Q1_star = Q1_initial + epsilon_total * H1
    Q0_star = Q0_initial + epsilon_total * H0

    # =========================================================================
    # Step 5: Compute ATE and inference
    # =========================================================================
    ate = compute_tmle_ate(Q1_star, Q0_star)

    # Efficient influence function for variance
    eif = compute_efficient_influence_function(
        outcomes, treatment, propensity, Q1_star, Q0_star, ate
    )

    variance = compute_tmle_variance(eif)
    se = np.sqrt(variance)

    # Confidence interval
    if n < 50:
        # Use t-distribution for small samples
        df = n - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        # Use normal approximation for large samples
        t_crit = stats.norm.ppf(1 - alpha / 2)

    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    # =========================================================================
    # Step 6: Assemble results
    # =========================================================================
    n_treated = int(np.sum(treatment))
    n_control = n - n_treated

    return TMLEResult(
        estimate=float(ate),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n=int(n),
        n_treated=int(n_treated),
        n_control=int(n_control),
        n_trimmed=int(n_trimmed),
        epsilon=float(epsilon_total),
        n_iterations=int(n_iterations),
        converged=bool(converged),
        convergence_criterion=float(convergence_criterion),
        propensity_diagnostics=propensity_diagnostics,
        outcome_diagnostics=outcome_diagnostics,
        propensity=propensity,
        Q0_initial=Q0_initial,
        Q1_initial=Q1_initial,
        Q0_star=Q0_star,
        Q1_star=Q1_star,
        eif=eif,
    )
