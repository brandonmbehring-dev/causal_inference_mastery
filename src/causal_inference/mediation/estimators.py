"""
Mediation estimators: Baron-Kenny, NDE, NIE, CDE.

Implements causal mediation analysis following:
- Baron & Kenny (1986) linear path analysis
- Imai et al. (2010) simulation-based approach

References
----------
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

from .types import BaronKennyResult, CDEResult, MediationResult


def baron_kenny(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]] = None,
    robust_se: bool = True,
    alpha: float = 0.05,
) -> BaronKennyResult:
    """
    Classic Baron-Kenny mediation analysis.

    Fits two linear models:
    1. M = alpha_0 + alpha_1 * T + gamma'X + e_1  (mediator model)
    2. Y = beta_0 + beta_1 * T + beta_2 * M + delta'X + e_2  (outcome model)

    Returns path coefficients with Sobel test for indirect effect.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y, shape (n,)
    treatment : ndarray
        Treatment variable T (binary or continuous), shape (n,)
    mediator : ndarray
        Mediator variable M, shape (n,)
    covariates : ndarray, optional
        Pre-treatment covariates X, shape (n, p)
    robust_se : bool
        Use HC3 robust standard errors (default True)
    alpha : float
        Significance level for tests (default 0.05)

    Returns
    -------
    BaronKennyResult
        Path coefficients, effects, and Sobel test

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or invalid values

    Notes
    -----
    Baron-Kenny assumes linear models. For nonlinear relationships
    (binary mediator/outcome), use simulation-based methods instead.

    The Sobel test can have low power in small samples; consider
    bootstrap CIs for indirect effect inference.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> T = np.random.binomial(1, 0.5, n).astype(float)
    >>> M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    >>> Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5
    >>> result = baron_kenny(Y, T, M)
    >>> print(f"Indirect: {result['indirect_effect']:.3f}")  # ~0.48
    >>> print(f"Direct: {result['direct_effect']:.3f}")      # ~0.50
    """
    # Validate inputs
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    mediator = np.asarray(mediator, dtype=np.float64)

    n = len(outcome)
    if len(treatment) != n or len(mediator) != n:
        raise ValueError(
            f"Length mismatch: outcome ({n}), treatment ({len(treatment)}), "
            f"mediator ({len(mediator)})"
        )

    if np.any(np.isnan(outcome)) or np.any(np.isnan(treatment)) or np.any(np.isnan(mediator)):
        raise ValueError("NaN values in input arrays")

    if np.any(np.isinf(outcome)) or np.any(np.isinf(treatment)) or np.any(np.isinf(mediator)):
        raise ValueError("Infinite values in input arrays")

    if n < 4:
        raise ValueError(
            f"Insufficient sample size (n={n}). Mediation requires at least 4 observations."
        )

    # Build design matrices
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if len(covariates) != n:
            raise ValueError(f"Covariates length ({len(covariates)}) != n ({n})")
        if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
            raise ValueError("NaN or infinite values in covariates")
        X_m = np.column_stack([np.ones(n), treatment, covariates])
        X_y = np.column_stack([np.ones(n), treatment, mediator, covariates])
    else:
        X_m = np.column_stack([np.ones(n), treatment])
        X_y = np.column_stack([np.ones(n), treatment, mediator])

    cov_type = "HC3" if robust_se else "nonrobust"

    # Step 1: Mediator model (M ~ T + X)
    model_m = sm.OLS(mediator, X_m)
    fit_m = model_m.fit(cov_type=cov_type)

    alpha_1 = fit_m.params[1]
    alpha_1_se = fit_m.bse[1]
    alpha_1_pval = fit_m.pvalues[1]
    r2_m = fit_m.rsquared

    # Step 2: Outcome model (Y ~ T + M + X)
    model_y = sm.OLS(outcome, X_y)
    fit_y = model_y.fit(cov_type=cov_type)

    beta_1 = fit_y.params[1]  # Direct effect
    beta_1_se = fit_y.bse[1]
    beta_1_pval = fit_y.pvalues[1]

    beta_2 = fit_y.params[2]  # M -> Y effect
    beta_2_se = fit_y.bse[2]
    beta_2_pval = fit_y.pvalues[2]

    r2_y = fit_y.rsquared

    # Compute effects
    indirect_effect = alpha_1 * beta_2
    direct_effect = beta_1
    total_effect = beta_1 + alpha_1 * beta_2

    # Sobel standard error: sqrt(alpha_1^2 * se_beta_2^2 + beta_2^2 * se_alpha_1^2)
    sobel_se = np.sqrt(alpha_1**2 * beta_2_se**2 + beta_2**2 * alpha_1_se**2)
    sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0.0
    sobel_pval = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    return BaronKennyResult(
        alpha_1=alpha_1,
        alpha_1_se=alpha_1_se,
        alpha_1_pvalue=alpha_1_pval,
        beta_1=beta_1,
        beta_1_se=beta_1_se,
        beta_1_pvalue=beta_1_pval,
        beta_2=beta_2,
        beta_2_se=beta_2_se,
        beta_2_pvalue=beta_2_pval,
        indirect_effect=indirect_effect,
        indirect_se=sobel_se,
        direct_effect=direct_effect,
        total_effect=total_effect,
        sobel_z=sobel_z,
        sobel_pvalue=sobel_pval,
        r2_mediator_model=r2_m,
        r2_outcome_model=r2_y,
        n_obs=n,
    )


def mediation_analysis(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]] = None,
    method: Literal["baron_kenny", "simulation"] = "simulation",
    n_simulations: int = 1000,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    treat_value: float = 1.0,
    control_value: float = 0.0,
    mediator_model: Literal["linear", "logistic"] = "linear",
    outcome_model: Literal["linear", "logistic"] = "linear",
    random_state: Optional[int] = None,
) -> MediationResult:
    """
    Estimate causal mediation effects.

    Decomposes total effect into direct and indirect components
    using either Baron-Kenny (linear) or simulation-based (general) methods.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y
    treatment : ndarray
        Treatment variable T (binary or continuous)
    mediator : ndarray
        Mediator variable M
    covariates : ndarray, optional
        Pre-treatment covariates X
    method : str
        "baron_kenny" for linear OLS, "simulation" for general
    n_simulations : int
        Monte Carlo simulations for counterfactuals (simulation method)
    n_bootstrap : int
        Bootstrap replications for inference
    alpha : float
        Significance level for CIs
    treat_value : float
        Treatment value for comparison (default 1.0)
    control_value : float
        Control value for comparison (default 0.0)
    mediator_model : str
        "linear" or "logistic" for mediator model
    outcome_model : str
        "linear" or "logistic" for outcome model
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    MediationResult
        Effects, SEs, CIs, p-values, and metadata

    Notes
    -----
    Identification requires Sequential Ignorability (Imai et al. 2010):
    1. {Y(t,m), M(t)} ⊥ T | X  (treatment ignorability)
    2. Y(t,m) ⊥ M | T, X       (mediator ignorability)

    The mediator ignorability assumption is often violated in practice.
    Use mediation_sensitivity() to assess robustness.

    References
    ----------
    Imai, Keele, Yamamoto (2010). A General Approach to Causal
    Mediation Analysis. Psychological Methods 15(4):309-334.
    """
    # Validate inputs
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    mediator = np.asarray(mediator, dtype=np.float64)

    n = len(outcome)
    if len(treatment) != n or len(mediator) != n:
        raise ValueError("Input arrays must have same length")

    if random_state is not None:
        np.random.seed(random_state)

    if method == "baron_kenny":
        # Use Baron-Kenny with bootstrap for CIs
        bk_result = baron_kenny(outcome, treatment, mediator, covariates)

        # Bootstrap for proportion mediated and better CIs
        boot_de = []
        boot_ie = []
        boot_te = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            try:
                cov_boot = covariates[idx] if covariates is not None else None
                bk_boot = baron_kenny(
                    outcome[idx], treatment[idx], mediator[idx], cov_boot, robust_se=False
                )
                boot_de.append(bk_boot["direct_effect"])
                boot_ie.append(bk_boot["indirect_effect"])
                boot_te.append(bk_boot["total_effect"])
            except Exception:
                continue

        boot_de = np.array(boot_de)
        boot_ie = np.array(boot_ie)
        boot_te = np.array(boot_te)

        # Compute bootstrap statistics
        de_se = np.std(boot_de, ddof=1)
        ie_se = np.std(boot_ie, ddof=1)
        te_se = np.std(boot_te, ddof=1)

        q_low = alpha / 2 * 100
        q_high = (1 - alpha / 2) * 100

        de_ci = (np.percentile(boot_de, q_low), np.percentile(boot_de, q_high))
        ie_ci = (np.percentile(boot_ie, q_low), np.percentile(boot_ie, q_high))
        te_ci = (np.percentile(boot_te, q_low), np.percentile(boot_te, q_high))

        # Proportion mediated
        te = bk_result["total_effect"]
        pm = bk_result["indirect_effect"] / te if abs(te) > 1e-10 else 0.0
        boot_pm = boot_ie / boot_te
        boot_pm = boot_pm[np.isfinite(boot_pm)]
        pm_se = np.std(boot_pm, ddof=1) if len(boot_pm) > 0 else np.nan
        pm_ci = (np.percentile(boot_pm, q_low), np.percentile(boot_pm, q_high)) if len(boot_pm) > 0 else (np.nan, np.nan)

        # P-values (two-sided)
        de_pval = bk_result["beta_1_pvalue"]
        ie_pval = bk_result["sobel_pvalue"]
        te_z = te / te_se if te_se > 0 else 0.0
        te_pval = 2 * (1 - stats.norm.cdf(abs(te_z)))

        return MediationResult(
            total_effect=bk_result["total_effect"],
            direct_effect=bk_result["direct_effect"],
            indirect_effect=bk_result["indirect_effect"],
            proportion_mediated=pm,
            te_se=te_se,
            de_se=de_se,
            ie_se=ie_se,
            pm_se=pm_se,
            te_ci=te_ci,
            de_ci=de_ci,
            ie_ci=ie_ci,
            pm_ci=pm_ci,
            te_pvalue=te_pval,
            de_pvalue=de_pval,
            ie_pvalue=ie_pval,
            method="baron_kenny",
            n_obs=n,
            n_bootstrap=n_bootstrap,
            treatment_control=(control_value, treat_value),
            mediator_model="linear",
            outcome_model="linear",
        )

    else:  # simulation method
        return _simulation_mediation(
            outcome=outcome,
            treatment=treatment,
            mediator=mediator,
            covariates=covariates,
            n_simulations=n_simulations,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            treat_value=treat_value,
            control_value=control_value,
            mediator_model=mediator_model,
            outcome_model=outcome_model,
        )


def _simulation_mediation(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]],
    n_simulations: int,
    n_bootstrap: int,
    alpha: float,
    treat_value: float,
    control_value: float,
    mediator_model: str,
    outcome_model: str,
) -> MediationResult:
    """Simulation-based mediation analysis (Imai et al. 2010)."""
    n = len(outcome)

    # Fit mediator and outcome models
    m_model, y_model, m_sigma, y_sigma = _fit_models(
        outcome, treatment, mediator, covariates, mediator_model, outcome_model
    )

    # Single run of simulation for point estimates
    nde, nie = _simulate_effects_once(
        treatment, covariates, m_model, y_model, m_sigma, y_sigma,
        n_simulations, treat_value, control_value, mediator_model, outcome_model
    )

    te = nde + nie
    pm = nie / te if abs(te) > 1e-10 else 0.0

    # Bootstrap for inference
    boot_nde = []
    boot_nie = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        cov_boot = covariates[idx] if covariates is not None else None

        try:
            m_model_b, y_model_b, m_sigma_b, y_sigma_b = _fit_models(
                outcome[idx], treatment[idx], mediator[idx], cov_boot,
                mediator_model, outcome_model
            )
            nde_b, nie_b = _simulate_effects_once(
                treatment[idx], cov_boot, m_model_b, y_model_b, m_sigma_b, y_sigma_b,
                n_simulations // 2,  # Fewer sims in bootstrap for speed
                treat_value, control_value, mediator_model, outcome_model
            )
            boot_nde.append(nde_b)
            boot_nie.append(nie_b)
        except Exception:
            continue

    boot_nde = np.array(boot_nde)
    boot_nie = np.array(boot_nie)
    boot_te = boot_nde + boot_nie

    # Statistics
    de_se = np.std(boot_nde, ddof=1)
    ie_se = np.std(boot_nie, ddof=1)
    te_se = np.std(boot_te, ddof=1)

    q_low = alpha / 2 * 100
    q_high = (1 - alpha / 2) * 100

    de_ci = (np.percentile(boot_nde, q_low), np.percentile(boot_nde, q_high))
    ie_ci = (np.percentile(boot_nie, q_low), np.percentile(boot_nie, q_high))
    te_ci = (np.percentile(boot_te, q_low), np.percentile(boot_te, q_high))

    boot_pm = boot_nie / boot_te
    boot_pm = boot_pm[np.isfinite(boot_pm)]
    pm_se = np.std(boot_pm, ddof=1) if len(boot_pm) > 0 else np.nan
    pm_ci = (np.percentile(boot_pm, q_low), np.percentile(boot_pm, q_high)) if len(boot_pm) > 0 else (np.nan, np.nan)

    # P-values (proportion of bootstrap samples crossing zero)
    de_pval = 2 * min(np.mean(boot_nde <= 0), np.mean(boot_nde >= 0))
    ie_pval = 2 * min(np.mean(boot_nie <= 0), np.mean(boot_nie >= 0))
    te_pval = 2 * min(np.mean(boot_te <= 0), np.mean(boot_te >= 0))

    return MediationResult(
        total_effect=te,
        direct_effect=nde,
        indirect_effect=nie,
        proportion_mediated=pm,
        te_se=te_se,
        de_se=de_se,
        ie_se=ie_se,
        pm_se=pm_se,
        te_ci=te_ci,
        de_ci=de_ci,
        ie_ci=ie_ci,
        pm_ci=pm_ci,
        te_pvalue=te_pval,
        de_pvalue=de_pval,
        ie_pvalue=ie_pval,
        method="simulation",
        n_obs=n,
        n_bootstrap=n_bootstrap,
        treatment_control=(control_value, treat_value),
        mediator_model=mediator_model,
        outcome_model=outcome_model,
    )


def _fit_models(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]],
    mediator_model: str,
    outcome_model: str,
) -> Tuple:
    """Fit mediator and outcome models, return models and residual SDs."""
    n = len(outcome)

    # Design matrices
    if covariates is not None:
        X_m = np.column_stack([np.ones(n), treatment, covariates])
        X_y = np.column_stack([np.ones(n), treatment, mediator, covariates])
    else:
        X_m = np.column_stack([np.ones(n), treatment])
        X_y = np.column_stack([np.ones(n), treatment, mediator])

    # Mediator model
    if mediator_model == "linear":
        m_fit = sm.OLS(mediator, X_m).fit()
        m_sigma = np.sqrt(m_fit.mse_resid)
    else:  # logistic
        m_fit = sm.Logit(mediator, X_m).fit(disp=0)
        m_sigma = 1.0  # Not used for logistic

    # Outcome model
    if outcome_model == "linear":
        y_fit = sm.OLS(outcome, X_y).fit()
        y_sigma = np.sqrt(y_fit.mse_resid)
    else:  # logistic
        y_fit = sm.Logit(outcome, X_y).fit(disp=0)
        y_sigma = 1.0

    return m_fit, y_fit, m_sigma, y_sigma


def _simulate_effects_once(
    treatment: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]],
    m_model,
    y_model,
    m_sigma: float,
    y_sigma: float,
    n_simulations: int,
    treat_value: float,
    control_value: float,
    mediator_model: str,
    outcome_model: str,
) -> Tuple[float, float]:
    """Simulate counterfactuals and compute NDE, NIE."""
    n = len(treatment)

    # Base covariates for prediction
    if covariates is not None:
        base_cov = covariates
    else:
        base_cov = None

    nde_samples = []
    nie_samples = []

    for _ in range(n_simulations):
        # Generate counterfactual mediators M(0) and M(1)
        if covariates is not None:
            X_m0 = np.column_stack([np.ones(n), np.full(n, control_value), base_cov])
            X_m1 = np.column_stack([np.ones(n), np.full(n, treat_value), base_cov])
        else:
            X_m0 = np.column_stack([np.ones(n), np.full(n, control_value)])
            X_m1 = np.column_stack([np.ones(n), np.full(n, treat_value)])

        if mediator_model == "linear":
            M_0 = m_model.predict(X_m0) + np.random.randn(n) * m_sigma
            M_1 = m_model.predict(X_m1) + np.random.randn(n) * m_sigma
        else:  # logistic
            prob_0 = 1 / (1 + np.exp(-m_model.predict(X_m0)))
            prob_1 = 1 / (1 + np.exp(-m_model.predict(X_m1)))
            M_0 = (np.random.rand(n) < prob_0).astype(float)
            M_1 = (np.random.rand(n) < prob_1).astype(float)

        # Generate counterfactual outcomes
        # Y(1, M(0)), Y(0, M(0)), Y(1, M(1))
        if covariates is not None:
            X_y10 = np.column_stack([np.ones(n), np.full(n, treat_value), M_0, base_cov])
            X_y00 = np.column_stack([np.ones(n), np.full(n, control_value), M_0, base_cov])
            X_y11 = np.column_stack([np.ones(n), np.full(n, treat_value), M_1, base_cov])
        else:
            X_y10 = np.column_stack([np.ones(n), np.full(n, treat_value), M_0])
            X_y00 = np.column_stack([np.ones(n), np.full(n, control_value), M_0])
            X_y11 = np.column_stack([np.ones(n), np.full(n, treat_value), M_1])

        if outcome_model == "linear":
            Y_10 = y_model.predict(X_y10)
            Y_00 = y_model.predict(X_y00)
            Y_11 = y_model.predict(X_y11)
        else:  # logistic
            Y_10 = 1 / (1 + np.exp(-y_model.predict(X_y10)))
            Y_00 = 1 / (1 + np.exp(-y_model.predict(X_y00)))
            Y_11 = 1 / (1 + np.exp(-y_model.predict(X_y11)))

        # NDE = E[Y(1, M(0)) - Y(0, M(0))]
        # NIE = E[Y(1, M(1)) - Y(1, M(0))]
        nde_samples.append(np.mean(Y_10 - Y_00))
        nie_samples.append(np.mean(Y_11 - Y_10))

    nde = np.mean(nde_samples)
    nie = np.mean(nie_samples)

    return nde, nie


def natural_direct_effect(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]] = None,
    n_simulations: int = 1000,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    treat_value: float = 1.0,
    control_value: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Estimate Natural Direct Effect via simulation.

    NDE = E[Y(1, M(0)) - Y(0, M(0))]

    The direct effect of treatment holding the mediator at its
    natural value under control.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y
    treatment : ndarray
        Treatment variable T
    mediator : ndarray
        Mediator variable M
    covariates : ndarray, optional
        Pre-treatment covariates X
    n_simulations : int
        Monte Carlo simulations for counterfactuals
    n_bootstrap : int
        Bootstrap replications for SE
    alpha : float
        Significance level
    treat_value : float
        Treatment value
    control_value : float
        Control value
    random_state : int, optional
        Random seed

    Returns
    -------
    Tuple[float, float, Tuple[float, float]]
        (estimate, se, (ci_lower, ci_upper))
    """
    if random_state is not None:
        np.random.seed(random_state)

    result = mediation_analysis(
        outcome=outcome,
        treatment=treatment,
        mediator=mediator,
        covariates=covariates,
        method="simulation",
        n_simulations=n_simulations,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        treat_value=treat_value,
        control_value=control_value,
    )

    return (result["direct_effect"], result["de_se"], result["de_ci"])


def natural_indirect_effect(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]] = None,
    n_simulations: int = 1000,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    treat_value: float = 1.0,
    control_value: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Estimate Natural Indirect Effect via simulation.

    NIE = E[Y(1, M(1)) - Y(1, M(0))]

    The effect of treatment operating through the mediator,
    holding treatment at the treated level.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y
    treatment : ndarray
        Treatment variable T
    mediator : ndarray
        Mediator variable M
    covariates : ndarray, optional
        Pre-treatment covariates X
    n_simulations : int
        Monte Carlo simulations for counterfactuals
    n_bootstrap : int
        Bootstrap replications for SE
    alpha : float
        Significance level
    treat_value : float
        Treatment value
    control_value : float
        Control value
    random_state : int, optional
        Random seed

    Returns
    -------
    Tuple[float, float, Tuple[float, float]]
        (estimate, se, (ci_lower, ci_upper))
    """
    if random_state is not None:
        np.random.seed(random_state)

    result = mediation_analysis(
        outcome=outcome,
        treatment=treatment,
        mediator=mediator,
        covariates=covariates,
        method="simulation",
        n_simulations=n_simulations,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        treat_value=treat_value,
        control_value=control_value,
    )

    return (result["indirect_effect"], result["ie_se"], result["ie_ci"])


def controlled_direct_effect(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    mediator_value: float,
    covariates: Optional[NDArray[np.floating]] = None,
    alpha: float = 0.05,
) -> CDEResult:
    """
    Estimate Controlled Direct Effect at fixed mediator value.

    CDE(m) = E[Y(1,m) - Y(0,m)]

    Unlike NDE/NIE, CDE doesn't require cross-world counterfactuals.
    Simply conditions on M = m and estimates the treatment effect.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y
    treatment : ndarray
        Treatment variable T
    mediator : ndarray
        Mediator variable M
    mediator_value : float
        Value at which to fix the mediator
    covariates : ndarray, optional
        Pre-treatment covariates X
    alpha : float
        Significance level

    Returns
    -------
    CDEResult
        CDE estimate, SE, CI, and p-value

    Notes
    -----
    CDE is identified under weaker conditions than NDE/NIE:
    - Y(t,m) ⊥ T | M=m, X

    This is the standard ignorability assumption, not requiring
    the stronger sequential ignorability for natural effects.

    The CDE can vary with the choice of m, so interpretation
    requires specifying the mediator level of interest.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    mediator = np.asarray(mediator, dtype=np.float64)

    n = len(outcome)

    # Fit outcome model Y ~ T + M + X at the specified mediator value
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        X = np.column_stack([np.ones(n), treatment, mediator, covariates])
    else:
        X = np.column_stack([np.ones(n), treatment, mediator])

    model = sm.OLS(outcome, X)
    fit = model.fit(cov_type="HC3")

    # CDE is the treatment coefficient at the specified M level
    # Under linear model: CDE(m) = beta_1 (constant across m)
    # For interaction models, would need: beta_1 + beta_TM * m
    cde = fit.params[1]
    se = fit.bse[1]

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = cde - z_crit * se
    ci_upper = cde + z_crit * se

    pvalue = fit.pvalues[1]

    return CDEResult(
        cde=cde,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        mediator_value=mediator_value,
        n_obs=n,
        method="ols",
    )
