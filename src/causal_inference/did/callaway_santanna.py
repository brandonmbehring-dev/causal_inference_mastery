"""
Callaway-Sant'Anna (2021) Difference-in-Differences Estimator.

This module implements the Callaway-Sant'Anna estimator for staggered DiD designs
with heterogeneous treatment effects. Unlike TWFE, this estimator is unbiased even
when treatment effects vary across cohorts or over time.

Key References:
    - Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with
      Multiple Time Periods." Journal of Econometrics 225(2): 200-230.
      https://doi.org/10.1016/j.jeconom.2020.12.001
"""

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .staggered import StaggeredData


def _aggregate_by_method(
    att_gt_df: pd.DataFrame,
    aggregation: str,
) -> tuple[float, Dict[str, float]]:
    """
    Aggregate ATT(g,t) using specified method.

    Parameters
    ----------
    att_gt_df : pd.DataFrame
        DataFrame with ATT(g,t) estimates
    aggregation : str
        Aggregation method: 'simple', 'dynamic', or 'group'

    Returns
    -------
    tuple
        (overall_att, aggregated_dict)
        - overall_att: Scalar ATT estimate
        - aggregated_dict: For simple, {"att": float}
                          For dynamic, {event_time: att_k}
                          For group, {cohort: att_g}
    """
    if aggregation == "simple":
        att, _ = _aggregate_simple(att_gt_df)
        aggregated = {"att": float(att)}
    elif aggregation == "dynamic":
        att_dynamic, att, _ = _aggregate_dynamic(att_gt_df)
        aggregated = {int(k): float(v) for k, v in att_dynamic.items()}
    elif aggregation == "group":
        att_group, att, _ = _aggregate_group(att_gt_df)
        aggregated = {int(k): float(v) for k, v in att_group.items()}
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return att, aggregated


def callaway_santanna_ate(
    data: StaggeredData,
    aggregation: Literal["simple", "dynamic", "group"] = "simple",
    control_group: Literal["nevertreated", "notyettreated"] = "nevertreated",
    alpha: float = 0.05,
    n_bootstrap: int = 250,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Callaway-Sant'Anna (2021) group-time ATT estimator with aggregation.

    Two-step procedure:
    1. Estimate ATT(g,t) for each cohort g and time t >= g
       ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]
       where C is the control group (never-treated or not-yet-treated)

    2. Aggregate ATT(g,t) to summary estimand:
       - Simple: Average over all (g,t) with weights = group size
       - Dynamic: Average by event time (k = t-g)
       - Group: Average by cohort g

    Unlike TWFE, this estimator:
    - Uses only valid control groups (never or not-yet treated)
    - Avoids forbidden comparisons (no "already treated" as controls)
    - Produces non-negative weights
    - Unbiased with heterogeneous treatment effects

    Parameters:
        data: StaggeredData instance
        aggregation: Type of aggregation
            - "simple": Average ATT across all cohort-time cells
            - "dynamic": ATT by event time (periods since treatment)
            - "group": ATT by cohort (treatment timing group)
        control_group: Which units to use as controls
            - "nevertreated": Only never-treated units (preferred if available)
            - "notyettreated": Not-yet-treated units (includes future-treated)
        alpha: Significance level for confidence intervals (default 0.05)
        n_bootstrap: Number of bootstrap samples for standard errors (default 250)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
            - att: Overall ATT (if aggregation="simple")
            - se: Bootstrap standard error
            - t_stat: t-statistic for H0: ATT=0
            - p_value: p-value
            - ci_lower, ci_upper: (1-alpha)*100% confidence interval
            - att_gt: DataFrame with ATT(g,t) for each cohort × time
                     Columns: cohort, time, att, weight, n_treated, n_control
            - aggregated: Dict with aggregation-specific results
                - For "simple": same as top-level att
                - For "dynamic": Dict mapping event_time → att
                - For "group": Dict mapping cohort → att
            - control_group: Which control group was used
            - n_bootstrap: Number of bootstrap samples
            - n_cohorts: Number of treatment cohorts
            - n_obs: Total observations

    Raises:
        ValueError: If no never-treated units and control_group="nevertreated",
                   or if data structure is invalid

    Example:
        >>> data = create_staggered_data(outcomes, treatment, time, unit_id)
        >>> result = callaway_santanna_ate(data, aggregation="simple")
        >>> result["att"]  # Overall ATT, unbiased estimate
        >>> result["se"]   # Bootstrap standard error
        >>> result["att_gt"]  # DataFrame with ATT(g,t) for all cohorts × times

        >>> # Dynamic aggregation (ATT by event time)
        >>> result_dynamic = callaway_santanna_ate(data, aggregation="dynamic")
        >>> result_dynamic["aggregated"]  # Dict: {0: att_0, 1: att_1, ...}

    References:
        Callaway & Sant'Anna (2021) provide theoretical justification and show
        this estimator is √n-consistent and asymptotically normal under
        standard regularity conditions.
    """
    # Validate inputs
    if control_group == "nevertreated" and not np.any(data.never_treated_mask):
        raise ValueError(
            'control_group="nevertreated" requires never-treated units, but none found. '
            'Use control_group="notyettreated" or add never-treated units to data.'
        )

    if aggregation not in ["simple", "dynamic", "group"]:
        raise ValueError(
            f'aggregation must be "simple", "dynamic", or "group". Got: {aggregation}'
        )

    if n_bootstrap < 50:
        raise ValueError(
            f"n_bootstrap must be >= 50 for reliable inference. Got: {n_bootstrap}"
        )

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_state)

    # Step 1: Compute ATT(g,t) for all cohort-time cells
    att_gt_df = _compute_att_gt(data, control_group)

    # Step 2: Aggregate ATT(g,t)
    att, aggregated = _aggregate_by_method(att_gt_df, aggregation)

    # Step 3: Bootstrap standard errors
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Resample units with replacement
        boot_data = _bootstrap_resample(data, rng)

        # Compute ATT(g,t) on bootstrap sample
        try:
            boot_att_gt = _compute_att_gt(boot_data, control_group)

            # Aggregate bootstrap ATT(g,t) the same way
            boot_att, _ = _aggregate_by_method(boot_att_gt, aggregation)

            bootstrap_estimates.append(boot_att)
        except Exception:
            # Skip bootstrap samples that fail (e.g., no control units after resample)
            continue

    # Compute bootstrap SE
    if len(bootstrap_estimates) < n_bootstrap * 0.8:
        raise ValueError(
            f"Bootstrap failed: Only {len(bootstrap_estimates)}/{n_bootstrap} samples succeeded. "
            f"Data may be too small or imbalanced for bootstrap inference."
        )

    se = float(np.std(bootstrap_estimates, ddof=1))

    # Inference
    t_stat = att / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(bootstrap_estimates) - 1))

    # Confidence interval (percentile method)
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

    return {
        "att": float(att),
        "se": se,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "att_gt": att_gt_df,
        "aggregated": aggregated,
        "control_group": control_group,
        "n_bootstrap": n_bootstrap,
        "n_cohorts": data.n_cohorts,
        "n_obs": data.n_obs,
    }


def _compute_att_gt(
    data: StaggeredData, control_group: str
) -> pd.DataFrame:
    """
    Compute ATT(g,t) for each cohort g and time t.

    ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]

    where:
    - g: Treatment cohort (time when cohort is first treated)
    - t: Current time period (t >= g for post-treatment)
    - Y_t: Outcome at time t
    - Y_{g-1}: Outcome at time g-1 (period before treatment)
    - C: Control group (never-treated or not-yet-treated at time t)

    Parameters:
        data: StaggeredData instance
        control_group: "nevertreated" or "notyettreated"

    Returns:
        DataFrame with columns:
        - cohort: Treatment cohort g
        - time: Time period t
        - att: ATT(g,t) estimate
        - weight: Number of treated units in cohort g
        - n_treated: Number of treated observations
        - n_control: Number of control observations used
    """
    cohorts = data.cohorts
    periods = np.unique(data.time)

    att_gt_list = []

    # Get unique units (sorted) to align with treatment_time array
    unique_units = np.sort(np.unique(data.unit_id))

    for g in cohorts:
        # Find units in cohort g
        # treatment_time has one entry per unique unit
        cohort_mask = data.treatment_time == g

        # Get unit IDs in cohort g
        cohort_units = unique_units[cohort_mask]

        for t in periods:
            # Only compute ATT for post-treatment periods (t >= g)
            if t < g:
                continue

            # Get pre-treatment period (g-1)
            pre_period = g - 1

            # Skip if pre-period not in data
            if pre_period not in periods:
                continue

            # Compute ATT(g,t) using double difference
            # Treated group: units in cohort g
            treated_t = _get_outcome_for_units(data, cohort_units, t)
            treated_pre = _get_outcome_for_units(data, cohort_units, pre_period)
            treated_diff = treated_t - treated_pre

            # Control group: depends on control_group parameter
            if control_group == "nevertreated":
                # Never-treated units
                control_units = unique_units[data.never_treated_mask]
            else:  # "notyettreated"
                # Not-yet-treated at time t (includes never-treated and future cohorts)
                not_yet_mask = data.treatment_time > t
                control_units = unique_units[not_yet_mask]

            # Skip if no control units available
            if len(control_units) == 0:
                continue

            control_t = _get_outcome_for_units(data, control_units, t)
            control_pre = _get_outcome_for_units(data, control_units, pre_period)
            control_diff = control_t - control_pre

            # ATT(g,t) = double difference
            att_gt = np.mean(treated_diff) - np.mean(control_diff)

            att_gt_list.append(
                {
                    "cohort": int(g),
                    "time": int(t),
                    "event_time": int(t - g),
                    "att": float(att_gt),
                    "weight": len(cohort_units),  # Number of units in cohort
                    "n_treated": len(treated_diff),
                    "n_control": len(control_diff),
                }
            )

    return pd.DataFrame(att_gt_list)


def _get_outcome_for_units(
    data: StaggeredData, units: np.ndarray, time_period: int
) -> np.ndarray:
    """
    Get outcomes for specific units at a specific time period.

    Parameters:
        data: StaggeredData instance
        units: Array of unit IDs
        time_period: Time period

    Returns:
        Array of outcomes for those units at that time
    """
    mask = (np.isin(data.unit_id, units)) & (data.time == time_period)
    return data.outcomes[mask]


def _aggregate_simple(att_gt_df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """
    Simple aggregation: weighted average over all ATT(g,t).

    Weights are group sizes (number of units in each cohort).

    Returns:
        - att: Weighted average ATT
        - weights: Array of weights used
    """
    if len(att_gt_df) == 0:
        raise ValueError("No ATT(g,t) estimates to aggregate")

    weights = att_gt_df["weight"].values
    atts = att_gt_df["att"].values

    # Weighted average
    att = np.average(atts, weights=weights)

    return att, weights


def _aggregate_dynamic(att_gt_df: pd.DataFrame) -> tuple[Dict[int, float], float, np.ndarray]:
    """
    Dynamic aggregation: Average ATT by event time (k = t - g).

    Returns:
        - att_dynamic: Dict mapping event_time → ATT
        - att_overall: Overall ATT (average over all event times)
        - weights: Array of weights used
    """
    if len(att_gt_df) == 0:
        raise ValueError("No ATT(g,t) estimates to aggregate")

    # Group by event time
    event_times = att_gt_df["event_time"].unique()
    att_dynamic = {}

    all_atts = []
    all_weights = []

    for k in sorted(event_times):
        k_df = att_gt_df[att_gt_df["event_time"] == k]
        weights = k_df["weight"].values
        atts = k_df["att"].values

        # Weighted average for this event time
        att_k = np.average(atts, weights=weights)
        att_dynamic[int(k)] = att_k

        all_atts.append(att_k)
        all_weights.append(np.sum(weights))

    # Overall ATT: average over event times weighted by total group size at each event time
    att_overall = np.average(all_atts, weights=all_weights)

    return att_dynamic, att_overall, np.array(all_weights)


def _aggregate_group(att_gt_df: pd.DataFrame) -> tuple[Dict[int, float], float, np.ndarray]:
    """
    Group aggregation: Average ATT by cohort g.

    Returns:
        - att_group: Dict mapping cohort → ATT
        - att_overall: Overall ATT (average over all cohorts)
        - weights: Array of weights used
    """
    if len(att_gt_df) == 0:
        raise ValueError("No ATT(g,t) estimates to aggregate")

    # Group by cohort
    cohorts = att_gt_df["cohort"].unique()
    att_group = {}

    all_atts = []
    all_weights = []

    for g in sorted(cohorts):
        g_df = att_gt_df[att_gt_df["cohort"] == g]
        weights = g_df["weight"].values
        atts = g_df["att"].values

        # Weighted average for this cohort (over time periods)
        att_g = np.average(atts, weights=weights)
        att_group[int(g)] = att_g

        all_atts.append(att_g)
        all_weights.append(g_df["weight"].iloc[0])  # Cohort size (same across times)

    # Overall ATT: average over cohorts weighted by cohort size
    att_overall = np.average(all_atts, weights=all_weights)

    return att_group, att_overall, np.array(all_weights)


def _bootstrap_resample(data: StaggeredData, rng: np.random.RandomState) -> StaggeredData:
    """
    Resample units with replacement for bootstrap.

    Parameters:
        data: Original StaggeredData
        rng: Random number generator

    Returns:
        Resampled StaggeredData (same number of units, resampled with replacement)
    """
    # Get unique units
    unique_units = np.unique(data.unit_id)
    n_units = len(unique_units)

    # Resample units with replacement
    resampled_units = rng.choice(unique_units, size=n_units, replace=True)

    # Build resampled data
    resampled_outcomes = []
    resampled_treatment = []
    resampled_time = []
    resampled_unit_id = []
    resampled_treatment_time = []

    new_unit_id = 0
    for unit in resampled_units:
        # Get all observations for this unit
        unit_mask = data.unit_id == unit
        resampled_outcomes.append(data.outcomes[unit_mask])
        resampled_treatment.append(data.treatment[unit_mask])
        resampled_time.append(data.time[unit_mask])

        # Assign new unit ID (to handle duplicates from resampling)
        n_obs_for_unit = np.sum(unit_mask)
        resampled_unit_id.append(np.full(n_obs_for_unit, new_unit_id))

        # Get treatment time for this unit
        unit_idx = np.where(unique_units == unit)[0][0]
        resampled_treatment_time.append(data.treatment_time[unit_idx])

        new_unit_id += 1

    # Concatenate all resampled data
    return StaggeredData(
        outcomes=np.concatenate(resampled_outcomes),
        treatment=np.concatenate(resampled_treatment),
        time=np.concatenate(resampled_time),
        unit_id=np.concatenate(resampled_unit_id),
        treatment_time=np.array(resampled_treatment_time),
    )
