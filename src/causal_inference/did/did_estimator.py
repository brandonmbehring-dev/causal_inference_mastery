"""
Difference-in-Differences (DiD) estimators with cluster-robust standard errors.

This module implements classic 2×2 DiD estimation with:
- Cluster-robust standard errors (Bertrand, Duflo, Mullainathan 2004)
- Parallel trends testing
- Comprehensive diagnostics
"""

from typing import Dict, Optional, Any, Literal
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se
from src.causal_inference.utils.validation import validate_did_inputs


def did_2x2(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    post: np.ndarray,
    unit_id: np.ndarray,
    alpha: float = 0.05,
    cluster_se: bool = True,
    se_method: Optional[Literal["cluster", "wild_bootstrap", "naive"]] = None,
    n_bootstrap: int = 999,
) -> Dict[str, Any]:
    """
    2×2 Difference-in-Differences estimator with cluster-robust standard errors.
    
    Estimates the causal effect of a binary treatment using the difference-in-differences
    design: (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
    
    Mathematically:
        DiD = E[Y_{i,post} - Y_{i,pre} | D_i=1] - E[Y_{i,post} - Y_{i,pre} | D_i=0]
    
    Uses OLS regression: Y_it = β₀ + β₁·Treat_i + β₂·Post_t + β₃·(Treat_i × Post_t) + ε_it
    where β₃ is the DiD estimator.
    
    **Identification Assumption**: Parallel trends
        E[Y_{1t} - Y_{0t} | D=1] = E[Y_{1t} - Y_{0t} | D=0] for all t < T
        (Treated and control would have parallel trends in absence of treatment)
    
    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable (length n*T for n units, T time periods).
        Must be continuous.
    treatment : np.ndarray
        Treatment indicator (1=treated unit, 0=control unit).
        Must be binary and constant within units (unit-level treatment).
    post : np.ndarray
        Post-treatment indicator (1=post period, 0=pre period).
        Must be binary (time-level indicator).
    unit_id : np.ndarray
        Unit identifier for clustering standard errors.
        Used to account for serial correlation within units.
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05).
    cluster_se : bool, optional
        DEPRECATED: Use se_method instead.
        Use cluster-robust standard errors (default: True).
        Set to False only for comparison purposes (biased SEs).
    se_method : {'cluster', 'wild_bootstrap', 'naive'}, optional
        Standard error method:
        - 'cluster': Cluster-robust SEs (default, recommended for n_clusters >= 30)
        - 'wild_bootstrap': Wild cluster bootstrap (recommended for n_clusters < 30)
        - 'naive': Heteroskedasticity-robust but not cluster-robust (biased)
        If None, defaults based on cluster_se parameter for backward compatibility.
    n_bootstrap : int, default=999
        Number of bootstrap replications when se_method='wild_bootstrap'.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - estimate : float - DiD estimate (β₃ coefficient)
        - se : float - Standard error (cluster-robust if cluster_se=True)
        - t_stat : float - t-statistic
        - p_value : float - Two-sided p-value
        - ci_lower : float - Lower confidence interval bound
        - ci_upper : float - Upper confidence interval bound
        - n_treated : int - Number of treated units
        - n_control : int - Number of control units
        - n_pre : int - Number of pre-treatment periods
        - n_post : int - Number of post-treatment periods
        - n_obs : int - Total observations
        - n_clusters : int - Number of clusters (units)
        - cluster_se_used : bool - Whether cluster SEs were used (deprecated)
        - se_method : str - SE method used ('cluster', 'wild_bootstrap', 'naive')
        - df : int - Degrees of freedom for t-distribution
    
    Raises
    ------
    ValueError
        If inputs have mismatched lengths, invalid values, or violate DiD assumptions.
    
    References
    ----------
    - Bertrand, Duflo, Mullainathan (2004). "How much should we trust DD estimates?"
      Quarterly Journal of Economics 119(1): 249-275.
    - Angrist & Pischke (2009). Mostly Harmless Econometrics, Chapter 5.
    
    Examples
    --------
    >>> # Simple 2×2 DiD with 100 units, 2 periods
    >>> n_units = 100
    >>> outcomes = np.array([...])  # 200 observations
    >>> treatment = np.repeat([0]*50 + [1]*50, 2)  # 50 control, 50 treated
    >>> post = np.tile([0, 1], 100)  # Pre, post for each unit
    >>> unit_id = np.repeat(np.arange(100), 2)  # Unit IDs
    >>> result = did_2x2(outcomes, treatment, post, unit_id)
    >>> print(f"DiD estimate: {result['estimate']:.3f} (SE: {result['se']:.3f})")
    """
    # Input validation (using shared validation utilities)
    validate_did_inputs(outcomes, treatment, post, unit_id)

    # Check alpha
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1. Got: {alpha}")

    # DiD-specific validations
    # Check treatment is unit-level (constant within units)
    df = pd.DataFrame({"unit_id": unit_id, "treatment": treatment})
    treatment_varies = df.groupby("unit_id")["treatment"].nunique()
    if (treatment_varies > 1).any():
        raise ValueError(
            "treatment must be constant within units (unit-level treatment). "
            "Found units with time-varying treatment. Use staggered DiD for time-varying treatment."
        )

    # Count treated/control units
    units_treated = df.groupby("unit_id")["treatment"].first()
    n_treated = (units_treated == 1).sum()
    n_control = (units_treated == 0).sum()

    # Count periods
    n_pre = (post == 0).sum() // len(np.unique(unit_id))
    n_post = (post == 1).sum() // len(np.unique(unit_id))

    # Create interaction term
    treat_post = treatment * post

    # Determine SE method (backward compatibility with cluster_se parameter)
    if se_method is None:
        se_method_used = "cluster" if cluster_se else "naive"
    else:
        se_method_used = se_method

    # Use statsmodels for OLS
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels is required for cluster-robust SEs. "
            "Install with: pip install statsmodels"
        )

    # Create design matrix: intercept, treatment, post, treatment*post
    X = np.column_stack([
        np.ones(len(outcomes)),  # Intercept
        treatment,               # Treatment indicator
        post,                    # Post indicator
        treat_post               # Interaction (DiD estimate)
    ])

    # Get number of clusters for all methods
    n_clusters = len(np.unique(unit_id))

    # Fit OLS model
    model = sm.OLS(outcomes, X)

    if se_method_used == "wild_bootstrap":
        # Wild cluster bootstrap for few clusters
        # First get OLS estimates and residuals
        results = model.fit()
        did_estimate = results.params[3]
        residuals = results.resid

        # Compute wild bootstrap SE
        wb_result = wild_cluster_bootstrap_se(
            X=X,
            y=outcomes,
            residuals=residuals,
            cluster_id=unit_id,
            coef_idx=3,  # DiD coefficient
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )

        did_se = wb_result["se"]
        ci_lower = wb_result["ci_lower"]
        ci_upper = wb_result["ci_upper"]
        did_pvalue = wb_result["p_value"]
        df_resid = n_clusters - 1  # For reporting consistency
        did_tstat = did_estimate / did_se if did_se > 0 else np.nan

    elif se_method_used == "cluster":
        # Cluster-robust SEs
        # Degrees of freedom: n_clusters - 1 (following Bertrand et al. 2004)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': unit_id})
        df_resid = n_clusters - 1

        if n_clusters < 30:
            warnings.warn(
                f"Small number of clusters (n={n_clusters}). "
                "Cluster-robust SEs may be biased with <30 clusters. "
                "Consider using se_method='wild_bootstrap' for valid inference.",
                RuntimeWarning
            )

        # Extract DiD estimate
        did_estimate = results.params[3]
        did_se = results.bse[3]
        did_tstat = results.tvalues[3]
        did_pvalue = results.pvalues[3]

        # Confidence interval using t-distribution
        t_crit = stats.t.ppf(1 - alpha/2, df=df_resid)
        ci_lower = did_estimate - t_crit * did_se
        ci_upper = did_estimate + t_crit * did_se

    else:  # naive
        # Naive (heteroskedasticity-robust) SEs
        results = model.fit(cov_type='HC3')
        df_resid = results.df_resid

        # Extract DiD estimate
        did_estimate = results.params[3]
        did_se = results.bse[3]
        did_tstat = results.tvalues[3]
        did_pvalue = results.pvalues[3]

        # Confidence interval using t-distribution
        t_crit = stats.t.ppf(1 - alpha/2, df=df_resid)
        ci_lower = did_estimate - t_crit * did_se
        ci_upper = did_estimate + t_crit * did_se

    return {
        "estimate": float(did_estimate),
        "se": float(did_se),
        "t_stat": float(did_tstat),
        "p_value": float(did_pvalue),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "n_pre": int(n_pre),
        "n_post": int(n_post),
        "n_obs": int(len(outcomes)),
        "n_clusters": int(n_clusters),
        "cluster_se_used": bool(se_method_used == "cluster"),  # Backward compat
        "se_method": se_method_used,
        "df": int(df_resid),
    }


def check_parallel_trends(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test parallel trends assumption using pre-treatment periods.
    
    Tests whether treated and control groups had parallel trends before treatment
    by regressing outcomes on treatment×time interaction in pre-treatment periods:
        Y_it = β₀ + β₁·Treat_i + β₂·Time_t + β₃·(Treat_i × Time_t) + ε_it
    
    Under parallel trends: β₃ = 0 (no differential trend between groups).
    
    **Note**: This tests **pre-treatment trends**, not the parallel trends assumption itself
    (which is untestable). Rejecting H₀: β₃=0 suggests parallel trends may be violated.
    
    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable (length n*T).
    treatment : np.ndarray
        Treatment indicator (unit-level, constant within units).
    time : np.ndarray
        Time period indicator (0, 1, 2, ..., T-1).
        Must be numeric and increasing.
    unit_id : np.ndarray
        Unit identifier for clustering.
    treatment_time : int
        Time period when treatment begins (periods < treatment_time are pre-treatment).
    alpha : float, optional
        Significance level for test (default: 0.05).
    
    Returns
    -------
    dict
        Dictionary with keys:
        - pre_trend_diff : float - Differential trend coefficient (β₃)
        - se : float - Cluster-robust standard error
        - t_stat : float - t-statistic
        - p_value : float - Two-sided p-value for H₀: β₃=0
        - parallel_trends_plausible : bool - True if p > alpha (fail to reject H₀)
        - n_pre_periods : int - Number of pre-treatment periods used
        - n_obs : int - Number of observations in pre-treatment sample
        - warning : str or None - Warning message if test may be underpowered
    
    Raises
    ------
    ValueError
        If insufficient pre-treatment periods (<2) or invalid inputs.
    
    References
    ----------
    - Roth (2022). "Pretest with caution: Event-study estimates after testing for parallel trends"
      American Economic Review: Insights 4(3): 305-322.
    
    Examples
    --------
    >>> # Test parallel trends with 3 pre-periods, treatment starts at t=3
    >>> result = check_parallel_trends(outcomes, treatment, time, unit_id, treatment_time=3)
    >>> if result['parallel_trends_plausible']:
    ...     print("Parallel trends plausible (p > 0.05)")
    ... else:
    ...     print(f"Warning: Differential pre-trends detected (p = {result['p_value']:.3f})")
    """
    # Input validation
    if not (len(outcomes) == len(treatment) == len(time) == len(unit_id)):
        raise ValueError("All inputs must have same length")

    if len(outcomes) == 0:
        raise ValueError("Inputs cannot be empty")

    # Check alpha
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1. Got: {alpha}")

    # Check for NaN/inf
    if np.any(~np.isfinite(outcomes)):
        raise ValueError("outcomes contains NaN or inf values")
    if np.any(~np.isfinite(treatment)):
        raise ValueError("treatment contains NaN or inf values")
    if np.any(~np.isfinite(time)):
        raise ValueError("time contains NaN or inf values")

    # Filter to pre-treatment periods
    pre_mask = time < treatment_time
    
    if pre_mask.sum() == 0:
        raise ValueError(
            f"No pre-treatment periods found (all time >= {treatment_time}). "
            "Cannot test parallel trends without pre-treatment data."
        )
    
    outcomes_pre = outcomes[pre_mask]
    treatment_pre = treatment[pre_mask]
    time_pre = time[pre_mask]
    unit_id_pre = unit_id[pre_mask]
    
    # Count pre-treatment periods
    n_pre_periods = len(np.unique(time_pre))
    
    if n_pre_periods < 2:
        raise ValueError(
            f"Need at least 2 pre-treatment periods to test trends. Got {n_pre_periods}. "
            "Parallel trends test requires variation in time to estimate trends."
        )
    
    # Create treatment×time interaction
    treat_time = treatment_pre * time_pre
    
    # Use statsmodels for OLS with cluster-robust SEs
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
    # Design matrix: intercept, treatment, time, treatment*time
    X = np.column_stack([
        np.ones(len(outcomes_pre)),
        treatment_pre,
        time_pre,
        treat_time  # Differential trend coefficient
    ])
    
    # Fit OLS with cluster-robust SEs
    model = sm.OLS(outcomes_pre, X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': unit_id_pre})
    
    # Extract differential trend coefficient (treatment×time interaction)
    pre_trend_diff = results.params[3]
    pre_trend_se = results.bse[3]
    pre_trend_tstat = results.tvalues[3]
    pre_trend_pvalue = results.pvalues[3]
    
    # Test H₀: β₃ = 0 (parallel trends)
    parallel_trends_plausible = pre_trend_pvalue > alpha
    
    # Generate warning if test may be underpowered
    warning_msg = None
    if n_pre_periods <= 2:
        warning_msg = (
            f"Only {n_pre_periods} pre-treatment periods available. "
            "Test may be underpowered to detect violations of parallel trends. "
            "Consider: (1) More pre-periods if available, (2) Event study design."
        )
    
    n_clusters = len(np.unique(unit_id_pre))
    if n_clusters < 20:
        warning_msg = (
            f"Small number of clusters (n={n_clusters}) in pre-treatment sample. "
            "Cluster-robust SEs may be biased. Consider bootstrap or aggregation."
        )
    
    return {
        "pre_trend_diff": float(pre_trend_diff),
        "se": float(pre_trend_se),
        "t_stat": float(pre_trend_tstat),
        "p_value": float(pre_trend_pvalue),
        "parallel_trends_plausible": bool(parallel_trends_plausible),
        "n_pre_periods": int(n_pre_periods),
        "n_obs": int(len(outcomes_pre)),
        "warning": warning_msg,
    }
