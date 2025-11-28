"""
Event study design for Difference-in-Differences with dynamic treatment effects.

This module implements event study estimation with:
- Leads/lags coefficients (treatment effects by period relative to treatment)
- Two-way fixed effects (TWFE) specification
- Joint F-test for pre-trends
- Event study plots with confidence intervals
"""

from typing import Dict, Optional, Any, Tuple, TypedDict
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import statsmodels.api as sm


class EventStudyCoefficient(TypedDict):
    """Coefficient information for a single event time period."""

    estimate: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float


class EventStudyResult(TypedDict):
    """Return type for event_study() estimator."""

    leads: Dict[int, EventStudyCoefficient]
    lags: Dict[int, EventStudyCoefficient]
    joint_pretrends_pvalue: float
    parallel_trends_plausible: bool
    omitted_period: int
    n_leads: int
    n_lags: int
    n_obs: int
    n_treated: int
    n_control: int
    n_clusters: int
    df: int
    cluster_se_used: bool


def event_study(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: int,
    n_leads: Optional[int] = None,
    n_lags: Optional[int] = None,
    alpha: float = 0.05,
    cluster_se: bool = True,
    omit_period: int = -1,
) -> EventStudyResult:
    """
    Event study design for DiD with dynamic treatment effects.

    Estimates separate treatment effects for each time period relative to treatment
    using two-way fixed effects (TWFE) specification:

        Y_it = α_i + λ_t + Σ_{k≠omit} β_k·D_i·1{t - T_i = k} + ε_it

    where:
        - α_i: Unit fixed effects (differences out time-invariant characteristics)
        - λ_t: Time fixed effects (differences out common time trends)
        - β_k: Treatment effect k periods relative to treatment
        - D_i: Treatment indicator (1 if unit i ever treated)
        - T_i: Treatment time for unit i
        - k: Time relative to treatment (k<0 = leads/pre, k≥0 = lags/post)
        - omit: Reference period (typically k=-1 to avoid collinearity)

    **Leads (k < 0)**: Pre-treatment periods, should be zero under parallel trends
    **Lags (k ≥ 0)**: Post-treatment periods, dynamic treatment effects

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable (length n*T for n units, T time periods).
    treatment : np.ndarray
        Treatment indicator (1=ever treated, 0=never treated).
        Must be constant within units (unit-level treatment).
    time : np.ndarray
        Time period indicator (0, 1, 2, ..., T-1).
    unit_id : np.ndarray
        Unit identifier for clustering.
    treatment_time : int
        Time period when treatment begins (periods < treatment_time are pre-treatment).
    n_leads : int, optional
        Number of lead periods to include (default: auto-detect from data).
    n_lags : int, optional
        Number of lag periods to include (default: auto-detect from data).
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05).
    cluster_se : bool, optional
        Use cluster-robust standard errors (default: True).
    omit_period : int, optional
        Period to omit as reference (default: -1, period immediately before treatment).

    Returns
    -------
    dict
        Dictionary with keys:
        - leads : dict - Lead coefficients {period: {estimate, se, t_stat, p_value, ci_lower, ci_upper}}
        - lags : dict - Lag coefficients {period: {estimate, se, t_stat, p_value, ci_lower, ci_upper}}
        - joint_pretrends_pvalue : float - Joint F-test p-value for all leads = 0
        - parallel_trends_plausible : bool - True if joint test p > alpha
        - omitted_period : int - Which period was omitted
        - n_leads : int - Number of lead periods
        - n_lags : int - Number of lag periods
        - n_obs : int - Number of observations
        - n_clusters : int - Number of clusters
        - df : int - Degrees of freedom for t-distribution
        - cluster_se_used : bool - Whether cluster SEs were used

    Raises
    ------
    ValueError
        If inputs are invalid or violate event study assumptions.

    References
    ----------
    - Angrist & Pischke (2009). Mostly Harmless Econometrics, Chapter 5.
    - Roth (2022). "Pretest with caution: Event-study estimates after testing for
      parallel trends." American Economic Review: Insights 4(3): 305-322.
    - Sun & Abraham (2021). "Estimating Dynamic Treatment Effects in Event Studies
      with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.

    Examples
    --------
    >>> # Event study with 3 pre-periods, 3 post-periods
    >>> result = event_study(
    ...     outcomes=outcomes,
    ...     treatment=treatment,
    ...     time=time,
    ...     unit_id=unit_id,
    ...     treatment_time=5,
    ...     n_leads=3,
    ...     n_lags=3,
    ... )
    >>> # Check pre-trends
    >>> if result['parallel_trends_plausible']:
    ...     print("Parallel trends plausible (joint F-test p > 0.05)")
    >>> # View dynamic effects
    >>> for period, coef in result['lags'].items():
    ...     print(f"Period {period}: {coef['estimate']:.3f} (SE: {coef['se']:.3f})")
    """
    # Input validation
    if not (len(outcomes) == len(treatment) == len(time) == len(unit_id)):
        raise ValueError(
            f"All inputs must have same length. Got: outcomes={len(outcomes)}, "
            f"treatment={len(treatment)}, time={len(time)}, unit_id={len(unit_id)}"
        )

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

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.array_equal(unique_treatment, [0, 1]):
        raise ValueError(
            f"treatment must be binary (0, 1). Got unique values: {unique_treatment}"
        )

    # Check treatment is unit-level (constant within units)
    df = pd.DataFrame({"unit_id": unit_id, "treatment": treatment, "time": time})
    treatment_varies = df.groupby("unit_id")["treatment"].nunique()
    if (treatment_varies > 1).any():
        raise ValueError(
            "treatment must be constant within units (unit-level treatment). "
            "Found units with time-varying treatment."
        )

    # Check we have both treated and control units
    units_treated = df.groupby("unit_id")["treatment"].first()
    n_treated = (units_treated == 1).sum()
    n_control = (units_treated == 0).sum()

    if n_treated == 0:
        raise ValueError("No treated units found (all treatment=0)")
    if n_control == 0:
        raise ValueError("No control units found (all treatment=1)")

    # Check treatment_time is valid
    time_min, time_max = time.min(), time.max()
    if not (time_min <= treatment_time <= time_max):
        raise ValueError(
            f"treatment_time must be within time range [{time_min}, {time_max}]. "
            f"Got: {treatment_time}"
        )

    # Auto-detect n_leads and n_lags if not provided
    # n_leads: number of pre-treatment periods available
    # n_lags: number of post-treatment periods available (including treatment period as lag 0)
    if n_leads is None:
        n_leads = int(treatment_time - time_min)
    if n_lags is None:
        n_lags = int(time_max - treatment_time) + 1  # +1 to include treatment period

    # Validate n_leads and n_lags
    if n_leads < 0:
        raise ValueError(f"n_leads must be non-negative. Got: {n_leads}")
    if n_lags < 0:
        raise ValueError(f"n_lags must be non-negative. Got: {n_lags}")

    # Maximum possible leads/lags based on available periods
    max_possible_leads = int(treatment_time - time_min)
    max_possible_lags = int(time_max - treatment_time) + 1  # +1 to include treatment period

    if n_leads > max_possible_leads:
        raise ValueError(
            f"n_leads ({n_leads}) exceeds maximum possible ({max_possible_leads}) "
            f"given treatment_time={treatment_time} and time_min={time_min}"
        )
    if n_lags > max_possible_lags:
        raise ValueError(
            f"n_lags ({n_lags}) exceeds maximum possible ({max_possible_lags}) "
            f"given treatment_time={treatment_time} and time_max={time_max}"
        )

    # Check we have at least one pre or post period
    if n_leads == 0 and n_lags == 0:
        raise ValueError("Must have at least one lead or lag period")

    # Validate omit_period
    valid_periods = list(range(-n_leads, 0)) + list(range(0, n_lags + 1))
    if omit_period not in valid_periods:
        raise ValueError(
            f"omit_period ({omit_period}) must be in valid periods {valid_periods}"
        )

    # Create relative time variable (time - treatment_time for each observation)
    df["relative_time"] = df["time"] - treatment_time

    # Create event time dummies (one for each lead/lag, excluding omitted period)
    # Leads: k ∈ {-n_leads, ..., -2, -1} (n_leads coefficients)
    # Lags: k ∈ {0, 1, 2, ..., n_lags-1} (n_lags coefficients)
    event_time_dummies = {}

    for k in range(-n_leads, n_lags):
        if k == omit_period:
            continue  # Skip omitted period (reference category)

        # Create dummy: D_i * 1{relative_time == k}
        dummy_name = f"event_time_{k}"
        df[dummy_name] = (df["treatment"] == 1) & (df["relative_time"] == k)
        df[dummy_name] = df[dummy_name].astype(float)
        event_time_dummies[k] = dummy_name

    # Create unit and time fixed effects
    unit_dummies = pd.get_dummies(df["unit_id"], prefix="unit", drop_first=True).astype(float)
    time_dummies = pd.get_dummies(df["time"], prefix="time", drop_first=True).astype(float)

    # Combine: intercept + event time dummies + unit FE + time FE
    X_list = [pd.Series(np.ones(len(df)), name="intercept")]
    for k in sorted(event_time_dummies.keys()):
        X_list.append(df[event_time_dummies[k]].astype(float))
    X_list.extend([unit_dummies, time_dummies])

    X = pd.concat(X_list, axis=1)
    y = outcomes

    # Fit OLS model with cluster-robust SEs
    model = sm.OLS(y, X)

    if cluster_se:
        # Cluster-robust SEs
        n_clusters = len(np.unique(unit_id))
        results = model.fit(cov_type="cluster", cov_kwds={"groups": unit_id})
        df_resid = n_clusters - 1

        if n_clusters < 30:
            warnings.warn(
                f"Small number of clusters (n={n_clusters}). "
                "Cluster-robust SEs may be biased with <30 clusters. "
                "Consider: (1) Bootstrap, (2) Wild cluster bootstrap, "
                "(3) Aggregating to cluster level.",
                RuntimeWarning,
            )
    else:
        # Naive (heteroskedasticity-robust) SEs
        results = model.fit(cov_type="HC3")
        df_resid = results.df_resid
        n_clusters = len(np.unique(unit_id))

    # Extract coefficients for leads and lags
    leads = {}
    lags = {}

    param_idx = 1  # Start after intercept

    for k in sorted(event_time_dummies.keys()):
        estimate = results.params.iloc[param_idx]
        se = results.bse.iloc[param_idx]
        t_stat = results.tvalues.iloc[param_idx]
        p_value = results.pvalues.iloc[param_idx]

        # Confidence interval using t-distribution
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_resid)
        ci_lower = estimate - t_crit * se
        ci_upper = estimate + t_crit * se

        coef_dict = {
            "estimate": float(estimate),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

        if k < 0:
            leads[k] = coef_dict
        else:
            lags[k] = coef_dict

        param_idx += 1

    # Joint F-test for all leads = 0 (pre-trends test)
    if len(leads) > 0:
        # Build hypothesis matrix: one row per lead coefficient
        lead_indices = list(range(1, 1 + len(leads)))  # Indices in params vector
        hypothesis = np.zeros((len(leads), len(results.params)))
        for i, idx in enumerate(lead_indices):
            hypothesis[i, idx] = 1

        # Wald test: H0: all lead coefficients = 0
        wald_test = results.wald_test(hypothesis)
        joint_pretrends_pvalue = float(wald_test.pvalue)
        parallel_trends_plausible = joint_pretrends_pvalue > alpha
    else:
        # No leads to test
        joint_pretrends_pvalue = np.nan
        parallel_trends_plausible = None

    return {
        "leads": leads,
        "lags": lags,
        "joint_pretrends_pvalue": joint_pretrends_pvalue,
        "parallel_trends_plausible": parallel_trends_plausible,
        "omitted_period": int(omit_period),
        "n_leads": int(n_leads),
        "n_lags": int(n_lags),
        "n_obs": int(len(outcomes)),
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "n_clusters": int(n_clusters),
        "df": int(df_resid),
        "cluster_se_used": bool(cluster_se),
    }


def plot_event_study(
    result: Dict[str, Any],
    title: str = "Event Study",
    xlabel: str = "Periods Relative to Treatment",
    ylabel: str = "Treatment Effect",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_omitted: bool = True,
) -> None:
    """
    Create event study plot with leads/lags coefficients and confidence intervals.

    Plots treatment effect estimates over time relative to treatment, with:
    - Point estimates for each lead/lag
    - 95% confidence intervals (shaded region)
    - Vertical line at treatment time (period 0)
    - Horizontal line at zero (null hypothesis)
    - Omitted period marked (if show_omitted=True)

    Parameters
    ----------
    result : dict
        Result dictionary from event_study() function.
    title : str, optional
        Plot title (default: "Event Study").
    xlabel : str, optional
        X-axis label (default: "Periods Relative to Treatment").
    ylabel : str, optional
        Y-axis label (default: "Treatment Effect").
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    save_path : str, optional
        Path to save plot (default: None, display only).
    show_omitted : bool, optional
        Mark omitted period on plot (default: True).

    Returns
    -------
    None
        Displays plot and optionally saves to file.

    Examples
    --------
    >>> result = event_study(outcomes, treatment, time, unit_id, treatment_time=5)
    >>> plot_event_study(result, title="My Event Study", save_path="event_study.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Extract data from result
    leads = result["leads"]
    lags = result["lags"]
    omitted_period = result["omitted_period"]

    # Combine leads and lags into single sorted dict
    all_periods = {}
    all_periods.update(leads)
    all_periods.update(lags)

    if len(all_periods) == 0:
        raise ValueError("No coefficients to plot (empty leads and lags)")

    # Sort by period
    periods = sorted(all_periods.keys())
    estimates = [all_periods[k]["estimate"] for k in periods]
    ci_lowers = [all_periods[k]["ci_lower"] for k in periods]
    ci_uppers = [all_periods[k]["ci_upper"] for k in periods]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot point estimates
    ax.plot(periods, estimates, "o-", color="steelblue", linewidth=2, markersize=6, label="Estimate")

    # Plot confidence intervals (shaded region)
    ax.fill_between(periods, ci_lowers, ci_uppers, alpha=0.3, color="steelblue", label="95% CI")

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Add vertical line at treatment time (period 0)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Treatment")

    # Mark omitted period if requested
    if show_omitted and omitted_period in periods:
        omit_idx = periods.index(omitted_period)
        ax.plot(
            omitted_period,
            estimates[omit_idx],
            "x",
            color="red",
            markersize=10,
            markeredgewidth=2,
            label=f"Omitted (k={omitted_period})",
        )

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="best", fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot
    plt.show()
