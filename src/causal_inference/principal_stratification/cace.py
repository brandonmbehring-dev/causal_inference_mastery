"""
CACE (Complier Average Causal Effect) Estimation via Principal Stratification.

This module implements CACE estimation using the IV/2SLS approach, exploiting
the key identification result: CACE = LATE under standard assumptions.

Key Insight
-----------
Principal strata are defined by potential treatment values:
- Compliers: D(0)=0, D(1)=1 (respond to assignment)
- Always-takers: D(0)=1, D(1)=1 (always treated regardless of assignment)
- Never-takers: D(0)=0, D(1)=0 (never treated regardless of assignment)
- Defiers: D(0)=1, D(1)=0 (do opposite) - ruled out by monotonicity

Under the assumptions:
1. Independence: Z ⊥ (Y(0), Y(1), D(0), D(1))
2. Exclusion: Y(d,z) = Y(d) for all d, z
3. Monotonicity: D(1) >= D(0) for all units (no defiers)
4. Relevance: P(D(1)=1, D(0)=0) > 0 (some compliers exist)

We have: CACE = E[Y|Z=1] - E[Y|Z=0]  /  E[D|Z=1] - E[D|Z=0]
              = Reduced Form / First Stage
              = LATE (Local Average Treatment Effect)

References
----------
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of Causal Effects
  Using Instrumental Variables. Journal of the American Statistical Association, 91(434), 444-455.
- Frangakis, C. E., & Rubin, D. B. (2002). Principal Stratification in Causal Inference.
  Biometrics, 58(1), 21-29.
- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction. Cambridge University Press. Chapter 23-24.
"""

from typing import Optional, Literal, Dict, Any, Tuple
import warnings
import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike

from .types import CACEResult, StrataProportions


def cace_2sls(
    outcome: ArrayLike,
    treatment: ArrayLike,
    instrument: ArrayLike,
    covariates: Optional[ArrayLike] = None,
    alpha: float = 0.05,
    inference: Literal["standard", "robust"] = "robust",
) -> CACEResult:
    """
    Estimate CACE using 2SLS (Two-Stage Least Squares).

    Uses the equivalence CACE = LATE under standard IV assumptions.
    This is the simplest and most widely-used approach for principal stratification
    with binary instrument and treatment.

    Parameters
    ----------
    outcome : array-like, shape (n,)
        Outcome variable Y.
    treatment : array-like, shape (n,)
        Actual treatment received D (binary: 0 or 1).
    instrument : array-like, shape (n,)
        Instrument / random assignment Z (binary: 0 or 1).
    covariates : array-like, shape (n, p), optional
        Additional covariates X to include in both stages.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    inference : {'standard', 'robust'}, default='robust'
        Type of standard errors:
        - 'standard': Homoskedastic (classical) SEs
        - 'robust': Heteroskedasticity-robust (HC0) SEs

    Returns
    -------
    CACEResult
        Dictionary containing CACE estimate, standard error, confidence interval,
        strata proportions, first-stage diagnostics, and sample sizes.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong dimensions, non-binary treatment/instrument,
        no compliers detected, etc.)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> # Random assignment
    >>> Z = np.random.binomial(1, 0.5, n)
    >>> # 70% compliers, 15% always-takers, 15% never-takers
    >>> strata = np.random.choice([0, 1, 2], n, p=[0.70, 0.15, 0.15])
    >>> # Treatment based on stratum
    >>> D = np.where(strata == 0, Z,  # compliers: follow Z
    ...              np.where(strata == 1, 1,  # always-takers: D=1
    ...                      0))  # never-takers: D=0
    >>> # Outcome with true CACE = 2.0
    >>> Y = 1.0 + 2.0 * D + np.random.normal(0, 1, n)
    >>> # Estimate
    >>> result = cace_2sls(Y, D, Z)
    >>> print(f"CACE: {result['cace']:.3f} (SE: {result['se']:.3f})")

    Notes
    -----
    **CRITICAL**: Under LATE/CACE assumptions:

    - First-stage coefficient = complier proportion (π_c)
    - CACE is only identified for compliers, not the full population
    - Always-takers and never-takers have treatment effect = 0 (exclusion restriction)
    - The exclusion restriction is untestable

    If first-stage F < 10, standard errors may be biased (weak instruments).
    Consider using weak-IV robust methods for inference.
    """
    # Convert to numpy arrays
    Y = np.asarray(outcome, dtype=np.float64)
    D = np.asarray(treatment, dtype=np.float64)
    Z = np.asarray(instrument, dtype=np.float64)

    # Validate inputs
    _validate_ps_inputs(Y, D, Z, covariates, alpha)

    n = len(Y)

    # Handle covariates
    if covariates is not None:
        X = np.asarray(covariates, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Add intercept
        X_with_const = np.column_stack([np.ones(n), X])
    else:
        X_with_const = np.ones((n, 1))

    # ==========================================================================
    # First Stage: D = π₀ + π₁Z + π₂X + ν
    # ==========================================================================
    Z_full = np.column_stack([X_with_const, Z])

    # OLS for first stage
    first_stage_coef = np.linalg.lstsq(Z_full, D, rcond=None)[0]
    D_hat = Z_full @ first_stage_coef

    # First-stage residuals
    nu = D - D_hat

    # First-stage SE (for π₁, the coefficient on Z)
    # Z is the last column
    Z_idx = Z_full.shape[1] - 1

    if inference == "robust":
        first_stage_vcov = _robust_vcov(Z_full, nu)
    else:
        sigma2_first = np.sum(nu**2) / (n - Z_full.shape[1])
        first_stage_vcov = sigma2_first * np.linalg.inv(Z_full.T @ Z_full)

    first_stage_se = np.sqrt(first_stage_vcov[Z_idx, Z_idx])
    pi_z = first_stage_coef[Z_idx]

    # First-stage F-statistic (for Z only)
    first_stage_f = (pi_z / first_stage_se) ** 2

    # ==========================================================================
    # Reduced Form: Y = γ₀ + γ₁Z + γ₂X + η
    # ==========================================================================
    reduced_form_coef = np.linalg.lstsq(Z_full, Y, rcond=None)[0]
    Y_resid_rf = Y - Z_full @ reduced_form_coef
    gamma_z = reduced_form_coef[Z_idx]

    if inference == "robust":
        rf_vcov = _robust_vcov(Z_full, Y_resid_rf)
    else:
        sigma2_rf = np.sum(Y_resid_rf**2) / (n - Z_full.shape[1])
        rf_vcov = sigma2_rf * np.linalg.inv(Z_full.T @ Z_full)

    reduced_form_se = np.sqrt(rf_vcov[Z_idx, Z_idx])

    # ==========================================================================
    # Second Stage: Y = β₀ + β₁D̂ + β₂X + ε (2SLS)
    # ==========================================================================
    # Create second-stage design matrix with D_hat
    W = np.column_stack([X_with_const, D_hat])

    # Second-stage OLS (using D_hat, not D)
    second_stage_coef = np.linalg.lstsq(W, Y, rcond=None)[0]

    # CACE is coefficient on D_hat (last column)
    cace = second_stage_coef[-1]

    # ==========================================================================
    # Correct 2SLS Standard Errors
    # ==========================================================================
    # Key: Use true residuals (Y - W @ beta) but projection matrix from Z
    # Var(beta) = sigma^2 * (W'P_z W)^{-1} where P_z = Z(Z'Z)^{-1}Z'

    # True second-stage residuals using D (not D_hat)
    W_true = np.column_stack([X_with_const, D])
    epsilon = Y - W_true @ second_stage_coef

    if inference == "robust":
        # Robust 2SLS variance estimator
        # V = (W'P_z W)^{-1} * (sum_i eps_i^2 * z_i z_i') * (W'P_z W)^{-1}
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        P_z_W = Z_full @ ZtZ_inv @ Z_full.T @ W_true

        # Meat of sandwich
        meat = np.zeros((W_true.shape[1], W_true.shape[1]))
        for i in range(n):
            pzw_i = (Z_full[i : i + 1] @ ZtZ_inv @ Z_full.T @ W_true).flatten()
            meat += (epsilon[i] ** 2) * np.outer(pzw_i, pzw_i)

        # Bread
        bread = np.linalg.inv(P_z_W.T @ W_true)

        vcov_2sls = bread @ meat @ bread
    else:
        # Homoskedastic 2SLS variance
        sigma2 = np.sum(epsilon**2) / (n - W_true.shape[1])
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        WtPzW = W_true.T @ Z_full @ ZtZ_inv @ Z_full.T @ W_true
        vcov_2sls = sigma2 * np.linalg.inv(WtPzW)

    cace_se = np.sqrt(vcov_2sls[-1, -1])

    # ==========================================================================
    # Inference
    # ==========================================================================
    z_stat = cace / cace_se
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # ==========================================================================
    # Strata Proportions
    # ==========================================================================
    strata_props = _compute_strata_proportions(D, Z, first_stage_se)

    # ==========================================================================
    # Sample sizes
    # ==========================================================================
    n_treated_assigned = int(np.sum(Z == 1))
    n_control_assigned = int(np.sum(Z == 0))

    return CACEResult(
        cace=float(cace),
        se=float(cace_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_stat=float(z_stat),
        pvalue=float(pvalue),
        strata_proportions=strata_props,
        first_stage_coef=float(pi_z),
        first_stage_se=float(first_stage_se),
        first_stage_f=float(first_stage_f),
        reduced_form=float(gamma_z),
        reduced_form_se=float(reduced_form_se),
        n=n,
        n_treated_assigned=n_treated_assigned,
        n_control_assigned=n_control_assigned,
        method="2sls",
    )


def _validate_ps_inputs(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    covariates: Optional[ArrayLike],
    alpha: float,
) -> None:
    """Validate inputs for principal stratification estimators."""
    # Check dimensions
    if Y.ndim != 1:
        raise ValueError(f"outcome must be 1-dimensional. Got shape {Y.shape}")
    if D.ndim != 1:
        raise ValueError(f"treatment must be 1-dimensional. Got shape {D.shape}")
    if Z.ndim != 1:
        raise ValueError(f"instrument must be 1-dimensional. Got shape {Z.shape}")

    n = len(Y)
    if len(D) != n:
        raise ValueError(f"treatment length ({len(D)}) must match outcome length ({n})")
    if len(Z) != n:
        raise ValueError(f"instrument length ({len(Z)}) must match outcome length ({n})")

    # Check binary treatment
    unique_D = np.unique(D[~np.isnan(D)])
    if not np.all(np.isin(unique_D, [0, 1])):
        raise ValueError(f"treatment must be binary (0 or 1). Got unique values: {unique_D}")

    # Check binary instrument
    unique_Z = np.unique(Z[~np.isnan(Z)])
    if not np.all(np.isin(unique_Z, [0, 1])):
        raise ValueError(f"instrument must be binary (0 or 1). Got unique values: {unique_Z}")

    # Check for variation in instrument
    if len(unique_Z) < 2:
        raise ValueError("instrument must have variation (both 0 and 1 present)")

    # Check alpha
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1. Got: {alpha}")

    # Check covariates if provided
    if covariates is not None:
        X = np.asarray(covariates)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError(f"covariates rows ({X.shape[0]}) must match outcome length ({n})")

    # Check for compliers (first-stage coefficient > 0)
    D_z1 = D[Z == 1]
    D_z0 = D[Z == 0]
    first_stage = np.mean(D_z1) - np.mean(D_z0)

    if first_stage <= 0:
        raise ValueError(
            f"No compliers detected: first-stage coefficient = {first_stage:.4f} <= 0. "
            "This violates the relevance assumption. Check that Z causes D."
        )


def _compute_strata_proportions(
    D: np.ndarray, Z: np.ndarray, first_stage_se: float
) -> StrataProportions:
    """
    Compute principal strata proportions from observed data.

    Under monotonicity (no defiers):
    - π_c (compliers) = P(D=1|Z=1) - P(D=1|Z=0) = first-stage coefficient
    - π_a (always-takers) = P(D=1|Z=0)
    - π_n (never-takers) = P(D=0|Z=1) = 1 - P(D=1|Z=1)

    These sum to 1.
    """
    # Conditional treatment probabilities
    p_d1_z1 = np.mean(D[Z == 1])  # P(D=1 | Z=1)
    p_d1_z0 = np.mean(D[Z == 0])  # P(D=1 | Z=0)

    # Strata proportions
    pi_c = p_d1_z1 - p_d1_z0  # Compliers
    pi_a = p_d1_z0  # Always-takers
    pi_n = 1 - p_d1_z1  # Never-takers

    # Verify they sum to 1 (sanity check)
    total = pi_c + pi_a + pi_n
    if not np.isclose(total, 1.0, atol=1e-10):
        # Numerical adjustment if needed
        pi_c = pi_c / total
        pi_a = pi_a / total
        pi_n = pi_n / total

    return StrataProportions(
        compliers=float(pi_c),
        always_takers=float(pi_a),
        never_takers=float(pi_n),
        compliers_se=float(first_stage_se),
    )


def _robust_vcov(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    Compute heteroskedasticity-robust (HC0) variance-covariance matrix.

    V = (X'X)^{-1} * (sum_i eps_i^2 * x_i x_i') * (X'X)^{-1}
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    # Meat of sandwich
    meat = np.zeros((k, k))
    for i in range(n):
        x_i = X[i : i + 1].T
        meat += (residuals[i] ** 2) * (x_i @ x_i.T)

    return XtX_inv @ meat @ XtX_inv


def wald_estimator(
    outcome: ArrayLike,
    treatment: ArrayLike,
    instrument: ArrayLike,
    alpha: float = 0.05,
) -> CACEResult:
    """
    Simple Wald estimator for CACE (ratio of reduced form to first stage).

    This is the most basic form of IV estimation:
    CACE = [E(Y|Z=1) - E(Y|Z=0)] / [E(D|Z=1) - E(D|Z=0)]

    Equivalent to 2SLS without covariates, but uses the simple ratio formula.

    Parameters
    ----------
    outcome : array-like, shape (n,)
        Outcome variable Y.
    treatment : array-like, shape (n,)
        Actual treatment received D (binary).
    instrument : array-like, shape (n,)
        Instrument / random assignment Z (binary).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CACEResult
        Dictionary with CACE estimate and inference.

    Notes
    -----
    Standard error computed using the delta method:
    Var(CACE) ≈ [Var(γ) + CACE² * Var(π)] / π²

    where γ = reduced form coefficient, π = first stage coefficient.
    """
    Y = np.asarray(outcome, dtype=np.float64)
    D = np.asarray(treatment, dtype=np.float64)
    Z = np.asarray(instrument, dtype=np.float64)

    _validate_ps_inputs(Y, D, Z, None, alpha)

    # Split by instrument
    Y_z1, Y_z0 = Y[Z == 1], Y[Z == 0]
    D_z1, D_z0 = D[Z == 1], D[Z == 0]

    n1, n0 = len(Y_z1), len(Y_z0)
    n = n1 + n0

    # Reduced form: E[Y|Z=1] - E[Y|Z=0]
    gamma = np.mean(Y_z1) - np.mean(Y_z0)
    var_gamma = np.var(Y_z1, ddof=1) / n1 + np.var(Y_z0, ddof=1) / n0
    gamma_se = np.sqrt(var_gamma)

    # First stage: E[D|Z=1] - E[D|Z=0]
    pi = np.mean(D_z1) - np.mean(D_z0)
    var_pi = np.var(D_z1, ddof=1) / n1 + np.var(D_z0, ddof=1) / n0
    pi_se = np.sqrt(var_pi)

    # Wald/CACE estimate
    cace = gamma / pi

    # Delta method for SE
    # d(gamma/pi)/dgamma = 1/pi, d(gamma/pi)/dpi = -gamma/pi^2
    var_cace = (var_gamma + cace**2 * var_pi) / (pi**2)
    cace_se = np.sqrt(var_cace)

    # Inference
    z_stat = cace / cace_se
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # First-stage F
    first_stage_f = (pi / pi_se) ** 2

    # Strata proportions
    strata_props = _compute_strata_proportions(D, Z, pi_se)

    return CACEResult(
        cace=float(cace),
        se=float(cace_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_stat=float(z_stat),
        pvalue=float(pvalue),
        strata_proportions=strata_props,
        first_stage_coef=float(pi),
        first_stage_se=float(pi_se),
        first_stage_f=float(first_stage_f),
        reduced_form=float(gamma),
        reduced_form_se=float(gamma_se),
        n=n,
        n_treated_assigned=n1,
        n_control_assigned=n0,
        method="wald",
    )


# =============================================================================
# EM Algorithm for CACE Estimation
# =============================================================================


def cace_em(
    outcome: ArrayLike,
    treatment: ArrayLike,
    instrument: ArrayLike,
    covariates: Optional[ArrayLike] = None,
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    inference: Literal["standard", "robust"] = "robust",
) -> CACEResult:
    """
    Estimate CACE using the Expectation-Maximization (EM) algorithm.

    The EM algorithm treats strata membership (complier, always-taker, never-taker)
    as latent variables and iteratively estimates CACE by maximum likelihood.

    Parameters
    ----------
    outcome : array-like, shape (n,)
        Outcome variable Y.
    treatment : array-like, shape (n,)
        Actual treatment received D (binary: 0 or 1).
    instrument : array-like, shape (n,)
        Instrument / random assignment Z (binary: 0 or 1).
    covariates : array-like, shape (n, p), optional
        Additional covariates (not currently used in EM, reserved for future).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance (relative change in log-likelihood).
    inference : {'standard', 'robust'}, default='robust'
        Type of standard errors (uses Louis Information Formula).

    Returns
    -------
    CACEResult
        Dictionary containing CACE estimate, standard error, confidence interval,
        strata proportions, convergence diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    **Algorithm**:

    E-Step: Compute posterior strata probabilities
        P(S_i = s | Y_i, D_i, Z_i; θ) for s ∈ {complier, always-taker, never-taker}

    M-Step: Update parameters
        - π_c, π_a, π_n: strata proportions
        - μ_0, μ_1: baseline means (untreated, treated)
        - CACE = μ_1 - μ_0 for compliers
        - σ²: residual variance

    **Identification**:
    - D=1, Z=0 → definitely always-taker
    - D=0, Z=1 → definitely never-taker
    - D=1, Z=1 → complier OR always-taker (ambiguous)
    - D=0, Z=0 → complier OR never-taker (ambiguous)

    For ambiguous cases, we use the outcome distribution to compute posteriors.

    References
    ----------
    - Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from
      Incomplete Data via the EM Algorithm. JRSS-B.
    - Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data. Wiley.
    """
    # Convert to numpy arrays
    Y = np.asarray(outcome, dtype=np.float64)
    D = np.asarray(treatment, dtype=np.float64)
    Z = np.asarray(instrument, dtype=np.float64)

    # Validate inputs
    _validate_ps_inputs(Y, D, Z, covariates, alpha)

    n = len(Y)

    # Note: Covariates not currently used in EM (simpler model)
    # This matches standard principal stratification literature
    if covariates is not None:
        warnings.warn(
            "covariates are not used in EM algorithm. Use cace_2sls() for covariate adjustment.",
            UserWarning,
        )

    # ==========================================================================
    # Initialize from 2SLS (warm start)
    # ==========================================================================
    init_result = cace_2sls(Y, D, Z, None, alpha, inference)

    # Initial parameter estimates
    pi_c = init_result["strata_proportions"]["compliers"]
    pi_a = init_result["strata_proportions"]["always_takers"]
    pi_n = init_result["strata_proportions"]["never_takers"]

    # Initial outcome model parameters
    # For compliers: Y = μ_0 + CACE * D + error
    # For always-takers: Y = μ_a + error (D always 1)
    # For never-takers: Y = μ_n + error (D always 0)
    cace = init_result["cace"]
    mu_0 = np.mean(Y[D == 0])  # Baseline for untreated
    mu_1 = mu_0 + cace  # Mean for treated compliers
    mu_a = np.mean(Y[(D == 1) & (Z == 0)])  # Always-taker mean (identified)
    mu_n = np.mean(Y[(D == 0) & (Z == 1)])  # Never-taker mean (identified)
    sigma2 = np.var(Y, ddof=1)  # Initial variance

    # Track log-likelihood history
    ll_history = []
    converged = False
    final_iteration = 0

    # ==========================================================================
    # EM Algorithm
    # ==========================================================================
    for iteration in range(max_iter):
        # E-Step: Compute posterior strata probabilities
        strata_probs, ll = _e_step_ps(Y, D, Z, pi_c, pi_a, pi_n, mu_0, mu_1, mu_a, mu_n, sigma2)
        ll_history.append(ll)

        # Check convergence
        if iteration > 0:
            rel_change = (ll - ll_history[-2]) / (np.abs(ll_history[-2]) + 1e-10)
            if abs(rel_change) < tol:
                converged = True
                final_iteration = iteration + 1
                break

        # M-Step: Update parameters
        pi_c, pi_a, pi_n, mu_0, mu_1, mu_a, mu_n, sigma2 = _m_step_ps(Y, D, Z, strata_probs)

        # Compute CACE from updated parameters
        cace = mu_1 - mu_0

        final_iteration = iteration + 1

    # Warn if didn't converge
    if not converged:
        warnings.warn(
            f"EM algorithm did not converge after {max_iter} iterations. "
            f"Final relative LL change: {abs(ll_history[-1] - ll_history[-2]) / (np.abs(ll_history[-2]) + 1e-10):.2e}",
            UserWarning,
        )

    # ==========================================================================
    # Compute Standard Errors (Louis Information Formula)
    # ==========================================================================
    cace_se, strata_ses = _compute_em_variance(
        Y, D, Z, strata_probs, pi_c, pi_a, pi_n, mu_0, mu_1, mu_a, mu_n, sigma2
    )

    # ==========================================================================
    # Inference
    # ==========================================================================
    z_stat = cace / cace_se
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = cace - z_crit * cace_se
    ci_upper = cace + z_crit * cace_se

    # First-stage and reduced-form (from final strata probs)
    # First-stage ≈ π_c
    first_stage_coef = pi_c
    first_stage_se = strata_ses["compliers_se"]
    first_stage_f = (first_stage_coef / first_stage_se) ** 2 if first_stage_se > 0 else np.inf

    # Reduced form = CACE * π_c
    reduced_form = cace * pi_c
    reduced_form_se = np.sqrt((pi_c**2) * (cace_se**2) + (cace**2) * (first_stage_se**2))

    # Strata proportions
    strata_props = StrataProportions(
        compliers=float(pi_c),
        always_takers=float(pi_a),
        never_takers=float(pi_n),
        compliers_se=float(first_stage_se),
    )

    # Sample sizes
    n_treated_assigned = int(np.sum(Z == 1))
    n_control_assigned = int(np.sum(Z == 0))

    return CACEResult(
        cace=float(cace),
        se=float(cace_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_stat=float(z_stat),
        pvalue=float(pvalue),
        strata_proportions=strata_props,
        first_stage_coef=float(first_stage_coef),
        first_stage_se=float(first_stage_se),
        first_stage_f=float(first_stage_f),
        reduced_form=float(reduced_form),
        reduced_form_se=float(reduced_form_se),
        n=n,
        n_treated_assigned=n_treated_assigned,
        n_control_assigned=n_control_assigned,
        method="em",
    )


def _e_step_ps(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    pi_c: float,
    pi_a: float,
    pi_n: float,
    mu_0: float,
    mu_1: float,
    mu_a: float,
    mu_n: float,
    sigma2: float,
) -> Tuple[np.ndarray, float]:
    """
    E-Step: Compute posterior strata probabilities.

    For each unit, compute P(S_i = s | Y_i, D_i, Z_i; θ) for s ∈ {c, a, n}.

    Returns
    -------
    strata_probs : np.ndarray, shape (n, 3)
        Posterior probabilities. Columns: [P(complier), P(always-taker), P(never-taker)]
    log_likelihood : float
        Observed-data log-likelihood.
    """
    n = len(Y)
    sigma = np.sqrt(sigma2)

    # Initialize posterior probabilities
    strata_probs = np.zeros((n, 3))  # [complier, always-taker, never-taker]

    log_likelihood = 0.0

    for i in range(n):
        y_i = Y[i]
        d_i = int(D[i])
        z_i = int(Z[i])

        # Compute likelihood for each stratum given (D, Z) configuration
        # Key insight: Some (D, Z) combinations definitively identify stratum

        if d_i == 1 and z_i == 0:
            # D=1, Z=0 → Definitely always-taker
            strata_probs[i] = [0.0, 1.0, 0.0]
            # Likelihood: f(Y | always-taker, D=1)
            ll_i = stats.norm.logpdf(y_i, mu_a, sigma)

        elif d_i == 0 and z_i == 1:
            # D=0, Z=1 → Definitely never-taker
            strata_probs[i] = [0.0, 0.0, 1.0]
            # Likelihood: f(Y | never-taker, D=0)
            ll_i = stats.norm.logpdf(y_i, mu_n, sigma)

        elif d_i == 1 and z_i == 1:
            # D=1, Z=1 → Complier OR Always-taker (ambiguous)
            # Need Bayes' rule

            # Prior probabilities given Z=1
            # P(complier | Z=1) ∝ π_c
            # P(always-taker | Z=1) ∝ π_a
            # Never-taker impossible since D=1

            # Likelihoods
            # Complier with Z=1 → D=1 → Y ~ N(μ_1, σ²)
            ll_c = stats.norm.logpdf(y_i, mu_1, sigma)
            # Always-taker → D=1 → Y ~ N(μ_a, σ²)
            ll_a = stats.norm.logpdf(y_i, mu_a, sigma)

            # Log posteriors (unnormalized)
            log_post_c = np.log(pi_c + 1e-10) + ll_c
            log_post_a = np.log(pi_a + 1e-10) + ll_a

            # Normalize using log-sum-exp for numerical stability
            max_log_post = max(log_post_c, log_post_a)
            denom = max_log_post + np.log(
                np.exp(log_post_c - max_log_post) + np.exp(log_post_a - max_log_post)
            )

            p_c = np.exp(log_post_c - denom)
            p_a = np.exp(log_post_a - denom)

            # Clip for numerical stability
            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
            p_a = 1 - p_c

            strata_probs[i] = [p_c, p_a, 0.0]

            # Observed likelihood (marginal over ambiguous strata)
            ll_i = denom

        else:
            # D=0, Z=0 → Complier OR Never-taker (ambiguous)
            # Need Bayes' rule

            # Likelihoods
            # Complier with Z=0 → D=0 → Y ~ N(μ_0, σ²)
            ll_c = stats.norm.logpdf(y_i, mu_0, sigma)
            # Never-taker → D=0 → Y ~ N(μ_n, σ²)
            ll_n = stats.norm.logpdf(y_i, mu_n, sigma)

            # Log posteriors (unnormalized)
            log_post_c = np.log(pi_c + 1e-10) + ll_c
            log_post_n = np.log(pi_n + 1e-10) + ll_n

            # Normalize
            max_log_post = max(log_post_c, log_post_n)
            denom = max_log_post + np.log(
                np.exp(log_post_c - max_log_post) + np.exp(log_post_n - max_log_post)
            )

            p_c = np.exp(log_post_c - denom)
            p_n = np.exp(log_post_n - denom)

            # Clip
            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
            p_n = 1 - p_c

            strata_probs[i] = [p_c, 0.0, p_n]

            # Observed likelihood
            ll_i = denom

        log_likelihood += ll_i

    return strata_probs, log_likelihood


def _m_step_ps(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    strata_probs: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    M-Step: Update parameters given posterior strata probabilities.

    Returns
    -------
    pi_c, pi_a, pi_n : float
        Updated strata proportions.
    mu_0, mu_1 : float
        Updated baseline and treatment means for compliers.
    mu_a, mu_n : float
        Updated means for always-takers and never-takers.
    sigma2 : float
        Updated residual variance.
    """
    n = len(Y)

    # Extract posterior probabilities
    w_c = strata_probs[:, 0]  # P(complier | data)
    w_a = strata_probs[:, 1]  # P(always-taker | data)
    w_n = strata_probs[:, 2]  # P(never-taker | data)

    # Update strata proportions (weighted counts)
    pi_c = np.mean(w_c)
    pi_a = np.mean(w_a)
    pi_n = np.mean(w_n)

    # Normalize to ensure sum = 1
    total = pi_c + pi_a + pi_n
    pi_c /= total
    pi_a /= total
    pi_n /= total

    # Update outcome means
    # Compliers with D=0 (i.e., Z=0) contribute to μ_0
    # Compliers with D=1 (i.e., Z=1) contribute to μ_1

    # μ_0: Mean for compliers with D=0
    # These are observations where Z=0 and they're compliers
    mask_c_d0 = D == 0
    weights_c_d0 = w_c[mask_c_d0]
    if np.sum(weights_c_d0) > 1e-10:
        mu_0 = np.average(Y[mask_c_d0], weights=weights_c_d0)
    else:
        mu_0 = np.mean(Y[D == 0])

    # μ_1: Mean for compliers with D=1
    mask_c_d1 = D == 1
    weights_c_d1 = w_c[mask_c_d1]
    if np.sum(weights_c_d1) > 1e-10:
        mu_1 = np.average(Y[mask_c_d1], weights=weights_c_d1)
    else:
        mu_1 = np.mean(Y[D == 1])

    # μ_a: Mean for always-takers (all have D=1)
    mask_a = D == 1
    weights_a = w_a[mask_a]
    if np.sum(weights_a) > 1e-10:
        mu_a = np.average(Y[mask_a], weights=weights_a)
    else:
        mu_a = np.mean(Y[D == 1])

    # μ_n: Mean for never-takers (all have D=0)
    mask_n = D == 0
    weights_n = w_n[mask_n]
    if np.sum(weights_n) > 1e-10:
        mu_n = np.average(Y[mask_n], weights=weights_n)
    else:
        mu_n = np.mean(Y[D == 0])

    # Update variance (weighted residual sum of squares)
    # σ² = (1/n) * Σᵢ Σₛ wᵢₛ * (Yᵢ - μₛ(Dᵢ))²

    resid_sq = 0.0
    for i in range(n):
        d_i = int(D[i])

        # Complier contribution
        if d_i == 0:
            resid_sq += w_c[i] * (Y[i] - mu_0) ** 2
        else:
            resid_sq += w_c[i] * (Y[i] - mu_1) ** 2

        # Always-taker contribution (always D=1)
        resid_sq += w_a[i] * (Y[i] - mu_a) ** 2

        # Never-taker contribution (always D=0)
        resid_sq += w_n[i] * (Y[i] - mu_n) ** 2

    sigma2 = resid_sq / n
    sigma2 = max(sigma2, 1e-10)  # Ensure positive

    return pi_c, pi_a, pi_n, mu_0, mu_1, mu_a, mu_n, sigma2


def _compute_em_variance(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    strata_probs: np.ndarray,
    pi_c: float,
    pi_a: float,
    pi_n: float,
    mu_0: float,
    mu_1: float,
    mu_a: float,
    mu_n: float,
    sigma2: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute variance using Louis Information Formula.

    For EM, the observed Fisher information is:
    I_obs = I_complete - I_missing

    where I_missing accounts for the uncertainty in latent strata membership.

    For simplicity, we use a numerical approximation based on the
    posterior strata probabilities.

    Returns
    -------
    cace_se : float
        Standard error of CACE estimate.
    strata_ses : dict
        Standard errors for strata proportions.
    """
    n = len(Y)
    cace = mu_1 - mu_0

    # Extract weights
    w_c = strata_probs[:, 0]
    w_a = strata_probs[:, 1]
    w_n = strata_probs[:, 2]

    # Effective sample sizes
    n_eff_c = np.sum(w_c)
    n_eff_c_d0 = np.sum(w_c[D == 0])
    n_eff_c_d1 = np.sum(w_c[D == 1])

    # Variance of CACE = Var(μ_1 - μ_0)
    # Using delta method with weighted variance estimates

    # Variance of μ_0 (compliers with D=0)
    if n_eff_c_d0 > 1:
        mask_d0 = D == 0
        var_mu_0 = np.sum(w_c[mask_d0] * (Y[mask_d0] - mu_0) ** 2) / (
            n_eff_c_d0 * (n_eff_c_d0 - 1) / n_eff_c_d0
        )
        var_mu_0 /= n_eff_c_d0
    else:
        var_mu_0 = sigma2 / n

    # Variance of μ_1 (compliers with D=1)
    if n_eff_c_d1 > 1:
        mask_d1 = D == 1
        var_mu_1 = np.sum(w_c[mask_d1] * (Y[mask_d1] - mu_1) ** 2) / (
            n_eff_c_d1 * (n_eff_c_d1 - 1) / n_eff_c_d1
        )
        var_mu_1 /= n_eff_c_d1
    else:
        var_mu_1 = sigma2 / n

    # CACE variance (assuming independence between μ_0 and μ_1 estimates)
    var_cace = var_mu_0 + var_mu_1

    # Add adjustment for missing data uncertainty (Louis formula approximation)
    # The additional variance comes from posterior strata uncertainty
    # This inflates variance when strata are ambiguous

    # Entropy-based adjustment: Higher entropy in posteriors → more uncertainty
    entropy = 0.0
    for i in range(n):
        for s in range(3):
            if strata_probs[i, s] > 1e-10:
                entropy -= strata_probs[i, s] * np.log(strata_probs[i, s])
    avg_entropy = entropy / n

    # Scale factor: more uncertainty when posteriors are spread out
    # Max entropy for 3 classes is log(3) ≈ 1.1
    uncertainty_factor = 1.0 + avg_entropy / np.log(3)

    var_cace *= uncertainty_factor

    cace_se = np.sqrt(max(var_cace, 1e-10))

    # Standard error for complier proportion
    # Var(π_c) ≈ (1/n) * Σᵢ Var(wᵢc) + ...
    # Simplified: use variance of posterior probabilities
    var_pi_c = np.var(w_c, ddof=1) / n
    compliers_se = np.sqrt(max(var_pi_c, 1e-10))

    strata_ses = {
        "compliers_se": compliers_se,
        "always_takers_se": np.sqrt(np.var(w_a, ddof=1) / n),
        "never_takers_se": np.sqrt(np.var(w_n, ddof=1) / n),
    }

    return cace_se, strata_ses
