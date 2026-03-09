"""
Comparison and bias demonstration for DiD estimators.

This module provides tools to compare TWFE, Callaway-Sant'Anna, and Sun-Abraham
estimators on the same data, and to demonstrate TWFE bias with heterogeneous effects.

Key References:
    - Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing."
      Journal of Econometrics 225(2): 254-277.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .staggered import StaggeredData, create_staggered_data, twfe_staggered
from .callaway_santanna import callaway_santanna_ate
from .sun_abraham import sun_abraham_ate


def compare_did_methods(
    data: StaggeredData,
    true_effect: Optional[float] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 250,
) -> pd.DataFrame:
    """
    Compare TWFE, Callaway-Sant'Anna, and Sun-Abraham on same data.

    Runs all three estimators and returns a comparison table showing:
    - Estimated ATT
    - Standard error
    - 95% confidence interval
    - Bias (if true_effect provided)

    Parameters:
        data: StaggeredData instance
        true_effect: Optional true ATT for bias calculation
        alpha: Significance level for CIs (default 0.05)
        n_bootstrap: Bootstrap samples for CS (default 250)

    Returns:
        DataFrame with columns:
            - method: "TWFE", "Callaway-Sant'Anna", "Sun-Abraham"
            - att: Estimated ATT
            - se: Standard error
            - ci_lower, ci_upper: (1-alpha)*100% confidence interval
            - bias: att - true_effect (if true_effect provided, else NaN)
            - abs_bias: |bias| (if true_effect provided, else NaN)

    Example:
        >>> data = create_staggered_data(...)
        >>> comparison = compare_did_methods(data, true_effect=2.5)
        >>> print(comparison)
               method       att    se  ci_lower  ci_upper   bias  abs_bias
        0        TWFE  1.234...  0.15   0.94...   1.53...  -1.27   1.27
        1  Callaway...  2.487...  0.22   2.05...   2.92...  -0.01   0.01
        2  Sun-Abr...  2.501...  0.19   2.13...   2.87...   0.00   0.00

    Notes:
        - TWFE may show large bias with heterogeneous treatment effects
        - CS and SA should be approximately unbiased
        - All three should agree when effects are homogeneous
    """
    results = []

    # 1. TWFE (potentially biased)
    try:
        twfe_result = twfe_staggered(data, alpha=alpha)
        results.append(
            {
                "method": "TWFE",
                "att": twfe_result["att"],
                "se": twfe_result["se"],
                "ci_lower": twfe_result["ci_lower"],
                "ci_upper": twfe_result["ci_upper"],
            }
        )
    except Exception as e:
        results.append(
            {
                "method": "TWFE",
                "att": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "error": str(e),
            }
        )

    # 2. Callaway-Sant'Anna (unbiased)
    try:
        cs_result = callaway_santanna_ate(
            data, aggregation="simple", alpha=alpha, n_bootstrap=n_bootstrap
        )
        results.append(
            {
                "method": "Callaway-Sant'Anna",
                "att": cs_result["att"],
                "se": cs_result["se"],
                "ci_lower": cs_result["ci_lower"],
                "ci_upper": cs_result["ci_upper"],
            }
        )
    except Exception as e:
        results.append(
            {
                "method": "Callaway-Sant'Anna",
                "att": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "error": str(e),
            }
        )

    # 3. Sun-Abraham (unbiased)
    try:
        sa_result = sun_abraham_ate(data, alpha=alpha)
        results.append(
            {
                "method": "Sun-Abraham",
                "att": sa_result["att"],
                "se": sa_result["se"],
                "ci_lower": sa_result["ci_lower"],
                "ci_upper": sa_result["ci_upper"],
            }
        )
    except Exception as e:
        results.append(
            {
                "method": "Sun-Abraham",
                "att": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "error": str(e),
            }
        )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Add bias columns if true_effect provided
    if true_effect is not None:
        comparison_df["bias"] = comparison_df["att"] - true_effect
        comparison_df["abs_bias"] = np.abs(comparison_df["bias"])

    return comparison_df


def demonstrate_twfe_bias(
    n_units: int = 300,
    n_periods: int = 10,
    cohorts: List[int] = [5, 7],
    true_effects: Dict[int, float] = {5: 2.0, 7: 4.0},
    n_sims: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Demonstrate TWFE bias with staggered adoption via Monte Carlo simulation.

    Data Generating Process (DGP):
    - n_units units, n_periods time periods
    - Cohorts treated at different times with different effects
    - Example: Cohort 5 treated at t=5 with effect=2.0, Cohort 7 treated at t=7 with effect=4.0
    - Heterogeneous treatment effects across cohorts

    Runs n_sims simulations and compares:
    - TWFE (biased with heterogeneous effects)
    - Callaway-Sant'Anna (unbiased)
    - Sun-Abraham (unbiased)

    Parameters:
        n_units: Number of units (default 300)
        n_periods: Number of time periods (default 10)
        cohorts: List of treatment times (default [5, 7])
        true_effects: Dict mapping cohort → true effect (default {5: 2.0, 7: 4.0})
        n_sims: Number of simulations (default 1000)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            - method: "TWFE", "Callaway-Sant'Anna", "Sun-Abraham"
            - mean_estimate: Mean estimate over n_sims
            - true_effect: True overall ATT (weighted average of cohort effects)
            - bias: mean_estimate - true_effect
            - rmse: Root mean squared error
            - coverage: Proportion of 95% CIs containing true effect
            - mean_se: Mean standard error

    Example:
        >>> results = demonstrate_twfe_bias(n_sims=1000)
        >>> print(results)
                       method  mean_estimate  true_effect   bias   rmse  coverage  mean_se
        0               TWFE         1.234...         3.0  -1.77   1.85     0.45     0.15
        1  Callaway-Sant'Anna         2.987...         3.0  -0.01   0.25     0.95     0.24
        2        Sun-Abraham         3.012...         3.0   0.01   0.23     0.95     0.22

    Notes:
        - TWFE shows large negative bias (uses "already treated" as controls)
        - Coverage far below 95% for TWFE (invalid inference)
        - CS and SA approximately unbiased with correct coverage
    """
    rng = np.random.RandomState(random_state)

    # Validate inputs
    if not all(c in true_effects for c in cohorts):
        raise ValueError(
            f"true_effects must contain entries for all cohorts. "
            f"Cohorts: {cohorts}, true_effects keys: {list(true_effects.keys())}"
        )

    if len(cohorts) < 2:
        raise ValueError(
            f"Need at least 2 cohorts for meaningful heterogeneity. Got {len(cohorts)}"
        )

    # Compute true overall ATT (weighted average by cohort size)
    units_per_cohort = n_units // (len(cohorts) + 1)  # +1 for never-treated
    total_treated = units_per_cohort * len(cohorts)
    true_att = sum(true_effects[g] * units_per_cohort for g in cohorts) / total_treated

    # Storage for simulation results
    twfe_estimates = []
    cs_estimates = []
    sa_estimates = []
    twfe_ses = []
    cs_ses = []
    sa_ses = []
    twfe_cis = []
    cs_cis = []
    sa_cis = []

    for sim in range(n_sims):
        # Generate data
        data = _generate_staggered_data(
            n_units=n_units,
            n_periods=n_periods,
            cohorts=cohorts,
            true_effects=true_effects,
            rng=rng,
        )

        # Run estimators
        try:
            # TWFE
            twfe_result = twfe_staggered(data)
            twfe_estimates.append(twfe_result["att"])
            twfe_ses.append(twfe_result["se"])
            twfe_cis.append((twfe_result["ci_lower"], twfe_result["ci_upper"]))

            # Callaway-Sant'Anna (use fewer bootstrap for speed)
            cs_result = callaway_santanna_ate(data, n_bootstrap=100)
            cs_estimates.append(cs_result["att"])
            cs_ses.append(cs_result["se"])
            cs_cis.append((cs_result["ci_lower"], cs_result["ci_upper"]))

            # Sun-Abraham
            sa_result = sun_abraham_ate(data)
            sa_estimates.append(sa_result["att"])
            sa_ses.append(sa_result["se"])
            sa_cis.append((sa_result["ci_lower"], sa_result["ci_upper"]))

        except Exception:
            # Skip failed simulations (should be rare)
            continue

    # Compute summary statistics
    def compute_stats(estimates, ses, cis, true_value):
        estimates = np.array(estimates)
        ses = np.array(ses)

        mean_est = np.mean(estimates)
        bias = mean_est - true_value
        rmse = np.sqrt(np.mean((estimates - true_value) ** 2))
        coverage = np.mean([(ci[0] <= true_value <= ci[1]) for ci in cis])
        mean_se = np.mean(ses)

        return {
            "mean_estimate": mean_est,
            "true_effect": true_value,
            "bias": bias,
            "rmse": rmse,
            "coverage": coverage,
            "mean_se": mean_se,
        }

    results = [
        {"method": "TWFE", **compute_stats(twfe_estimates, twfe_ses, twfe_cis, true_att)},
        {
            "method": "Callaway-Sant'Anna",
            **compute_stats(cs_estimates, cs_ses, cs_cis, true_att),
        },
        {
            "method": "Sun-Abraham",
            **compute_stats(sa_estimates, sa_ses, sa_cis, true_att),
        },
    ]

    return pd.DataFrame(results)


def _generate_staggered_data(
    n_units: int,
    n_periods: int,
    cohorts: List[int],
    true_effects: Dict[int, float],
    rng: np.random.RandomState,
) -> StaggeredData:
    """
    Generate synthetic staggered DiD data with heterogeneous effects.

    DGP:
        Y_it = α_i + λ_t + Σ_g τ_g·D_it^g + ε_it

    where:
        - α_i ~ N(0, 1): Unit fixed effect
        - λ_t = 0.5·t: Linear time trend
        - D_it^g = 1{G_i = g and t >= g}: Treatment indicator for cohort g
        - τ_g: Treatment effect for cohort g (heterogeneous across cohorts)
        - ε_it ~ N(0, 1): Idiosyncratic error

    Parameters:
        n_units: Number of units
        n_periods: Number of time periods
        cohorts: List of treatment times
        true_effects: Dict mapping cohort → effect
        rng: Random number generator

    Returns:
        StaggeredData instance
    """
    # Assign units to cohorts (equal-sized cohorts + never-treated)
    units_per_cohort = n_units // (len(cohorts) + 1)

    treatment_time = []
    for g in cohorts:
        treatment_time.extend([g] * units_per_cohort)
    # Never-treated units
    treatment_time.extend([np.inf] * (n_units - len(treatment_time)))
    treatment_time = np.array(treatment_time)

    # Generate unit fixed effects
    unit_fe = rng.randn(n_units)

    # Build panel data
    outcomes = []
    treatment = []
    time = []
    unit_id = []

    for i in range(n_units):
        for t in range(n_periods):
            # Base outcome: unit FE + time trend
            y = unit_fe[i] + 0.5 * t

            # Add treatment effect if treated
            g_i = treatment_time[i]
            if np.isfinite(g_i) and t >= g_i:
                y += true_effects[g_i]
                d_it = 1
            else:
                d_it = 0

            # Add noise
            y += rng.randn()

            outcomes.append(y)
            treatment.append(d_it)
            time.append(t)
            unit_id.append(i)

    return StaggeredData(
        outcomes=np.array(outcomes),
        treatment=np.array(treatment),
        time=np.array(time),
        unit_id=np.array(unit_id),
        treatment_time=treatment_time,
    )
