"""
Sun-Abraham (2021) Difference-in-Differences Estimator.

This module implements the Sun-Abraham interaction-weighted estimator for staggered DiD
designs with heterogeneous treatment effects. Unlike TWFE, this estimator uses cohort ×
event time interactions and proper weighting to avoid bias.

Key References:
    - Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event
      Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.
      https://doi.org/10.1016/j.jeconom.2020.12.001
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from .staggered import StaggeredData


def sun_abraham_ate(
    data: StaggeredData,
    alpha: float = 0.05,
    cluster_se: bool = True,
) -> Dict[str, Any]:
    """
    Sun-Abraham (2021) interaction-weighted estimator.

    Regression model:
        Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_it^{g,l} + ε_it

    where:
        - α_i: Unit fixed effects
        - λ_t: Time fixed effects
        - D_it^{g,l} = 1{G_i = g}·1{t - G_i = l} (cohort g × event time l interaction)
        - β_{g,l}: Treatment effect for cohort g at event time l (relative to treatment)

    Then aggregate:
        ATT = Σ_{g,l} w_{g,l}·β_{g,l}

    where w_{g,l} are the share of treated observations in cohort g at event time l:
        w_{g,l} = N_{g,l} / Σ_{g',l'} N_{g',l'}

    Unlike TWFE:
    - Saturated model with cohort × event time interactions (flexible)
    - Never-treated and not-yet-treated as clean control group (no forbidden comparisons)
    - Weights based on sample composition (non-negative)
    - Unbiased with heterogeneous treatment effects

    Parameters:
        data: StaggeredData instance
        alpha: Significance level for confidence intervals (default 0.05)
        cluster_se: If True, use cluster-robust SEs at unit level (default True)

    Returns:
        Dictionary with:
            - att: Weighted average treatment effect
            - se: Standard error (cluster-robust if cluster_se=True)
            - t_stat: t-statistic for H0: ATT=0
            - p_value: p-value
            - ci_lower, ci_upper: (1-alpha)*100% confidence interval
            - cohort_effects: DataFrame with β_{g,l} for each cohort × event time
                             Columns: cohort, event_time, coef, se, t_stat, p_value
            - weights: DataFrame with w_{g,l} weights used
                      Columns: cohort, event_time, weight, n_obs
            - n_obs: Total observations
            - n_treated: Total treated observations
            - n_control: Total control observations
            - n_cohorts: Number of treatment cohorts
            - cluster_se_used: Whether cluster SEs were used

    Raises:
        ValueError: If no never-treated units (need clean control group),
                   or if data structure is invalid

    Example:
        >>> data = create_staggered_data(outcomes, treatment, time, unit_id)
        >>> result = sun_abraham_ate(data)
        >>> result["att"]  # Overall ATT, unbiased estimate
        >>> result["se"]   # Cluster-robust standard error
        >>> result["cohort_effects"]  # DataFrame with β_{g,l} for all cohorts × event times
        >>> result["weights"]  # DataFrame with w_{g,l} weights

    References:
        Sun & Abraham (2021) show this estimator is unbiased with heterogeneous
        treatment effects and provides a clean event study visualization without
        TWFE bias.
    """
    # Validate: Need never-treated units as clean control group
    if not np.any(data.never_treated_mask):
        raise ValueError(
            "Sun-Abraham estimator requires never-treated units as control group. "
            "No never-treated units found in data. Consider using Callaway-Sant'Anna "
            'with control_group="notyettreated" if no never-treated units available.'
        )

    # Validate: Need at least 2 cohorts for meaningful comparison
    if data.n_cohorts < 2:
        raise ValueError(
            f"Sun-Abraham estimator requires at least 2 cohorts. Found {data.n_cohorts}. "
            f"For single treatment time, use event_study() instead."
        )

    # Step 1: Create cohort × event time interactions
    df, interaction_cols = _create_interactions(data)

    # Step 2: Run regression with unit + time FE + interactions
    results = _fit_sun_abraham_regression(df, interaction_cols, cluster_se)

    # Step 3: Extract cohort effects β_{g,l}
    cohort_effects_df = _extract_cohort_effects(results, interaction_cols, alpha)

    # Step 4: Compute weights w_{g,l} based on sample composition
    weights_df = _compute_weights(df, interaction_cols)

    # Step 5: Aggregate ATT = Σ w_{g,l}·β_{g,l}
    att, se_att = _aggregate_sun_abraham(cohort_effects_df, weights_df, results, cluster_se)

    # Inference
    t_stat = att / se_att if se_att > 0 else np.inf
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=results.df_resid))

    # Confidence interval using delta method
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=results.df_resid)
    ci_lower = att - t_crit * se_att
    ci_upper = att + t_crit * se_att

    # Diagnostics
    n_obs = len(data.outcomes)
    n_treated = int(np.sum(data.treatment))
    n_control = n_obs - n_treated

    return {
        "att": float(att),
        "se": float(se_att),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cohort_effects": cohort_effects_df,
        "weights": weights_df,
        "n_obs": n_obs,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_cohorts": data.n_cohorts,
        "cluster_se_used": cluster_se,
    }


def _create_interactions(data: StaggeredData) -> tuple[pd.DataFrame, list[str]]:
    """
    Create cohort × event time interaction dummies.

    For each cohort g and event time l (where l >= 0, post-treatment):
        D_it^{g,l} = 1{G_i = g}·1{t - G_i = l}

    Parameters:
        data: StaggeredData instance

    Returns:
        - df: DataFrame with outcome, unit_id, time, and interaction dummies
        - interaction_cols: List of interaction column names
    """
    # Create DataFrame
    df = pd.DataFrame(
        {
            "outcome": data.outcomes,
            "unit_id": data.unit_id,
            "time": data.time,
            "treatment_time": data.treatment_time[data.unit_id.astype(int)],
        }
    )

    # Compute event time for each observation
    df["event_time"] = df["time"] - df["treatment_time"]

    # Create interaction dummies for each cohort × event time
    # Only for post-treatment periods (event_time >= 0) and treated units
    interaction_cols = []

    cohorts = data.cohorts

    for g in cohorts:
        cohort_mask = df["treatment_time"] == g

        # Get event times for this cohort (only post-treatment)
        event_times = df.loc[cohort_mask & (df["event_time"] >= 0), "event_time"].unique()

        for l in sorted(event_times):
            col_name = f"cohort_{int(g)}_event_{int(l)}"
            df[col_name] = ((df["treatment_time"] == g) & (df["event_time"] == l)).astype(float)
            interaction_cols.append(col_name)

    return df, interaction_cols


def _fit_sun_abraham_regression(
    df: pd.DataFrame,
    interaction_cols: list[str],
    cluster_se: bool,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit Sun-Abraham regression with unit + time FE + interactions.

    Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_it^{g,l} + ε_it

    Parameters:
        df: DataFrame with outcomes and interaction dummies
        interaction_cols: List of interaction column names
        cluster_se: Whether to use cluster-robust SEs

    Returns:
        statsmodels RegressionResultsWrapper
    """
    # Create unit and time fixed effects
    unit_dummies = pd.get_dummies(df["unit_id"], prefix="unit", drop_first=True).astype(float)
    time_dummies = pd.get_dummies(df["time"], prefix="time", drop_first=True).astype(float)

    # Construct design matrix: [interactions, unit FE, time FE]
    X = pd.concat([df[interaction_cols], unit_dummies, time_dummies], axis=1)
    y = df["outcome"]

    # Fit OLS regression
    if cluster_se:
        model = sm.OLS(y, X)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df["unit_id"].values})
    else:
        model = sm.OLS(y, X)
        results = model.fit()

    return results


def _extract_cohort_effects(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    interaction_cols: list[str],
    alpha: float,
) -> pd.DataFrame:
    """
    Extract cohort × event time coefficients β_{g,l} from regression results.

    Parameters:
        results: Regression results
        interaction_cols: List of interaction column names
        alpha: Significance level for confidence intervals

    Returns:
        DataFrame with columns: cohort, event_time, coef, se, t_stat, p_value, ci_lower, ci_upper
    """
    cohort_effects = []

    for col in interaction_cols:
        # Parse cohort and event time from column name
        # Format: "cohort_{g}_event_{l}"
        parts = col.split("_")
        cohort = int(parts[1])
        event_time = int(parts[3])

        # Extract coefficient and inference
        coef = results.params[col]
        se = results.bse[col]
        t_stat = results.tvalues[col]
        p_value = results.pvalues[col]

        # Confidence interval
        ci = results.conf_int(alpha=alpha).loc[col]

        cohort_effects.append(
            {
                "cohort": cohort,
                "event_time": event_time,
                "coef": float(coef),
                "se": float(se),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
            }
        )

    return pd.DataFrame(cohort_effects)


def _compute_weights(df: pd.DataFrame, interaction_cols: list[str]) -> pd.DataFrame:
    """
    Compute sample share weights w_{g,l} for aggregation.

    w_{g,l} = N_{g,l} / Σ_{g',l'} N_{g',l'}

    where N_{g,l} is the number of treated observations with cohort g at event time l.

    Parameters:
        df: DataFrame with interaction dummies
        interaction_cols: List of interaction column names

    Returns:
        DataFrame with columns: cohort, event_time, weight, n_obs
    """
    weights = []
    total_treated = 0

    # First pass: Count treated observations for each cohort × event time
    for col in interaction_cols:
        n_obs = df[col].sum()
        total_treated += n_obs

        # Parse cohort and event time
        parts = col.split("_")
        cohort = int(parts[1])
        event_time = int(parts[3])

        weights.append(
            {
                "cohort": cohort,
                "event_time": event_time,
                "n_obs": int(n_obs),
                "weight": 0.0,  # Will compute in second pass
            }
        )

    # Second pass: Compute weights as proportions
    for w in weights:
        w["weight"] = w["n_obs"] / total_treated if total_treated > 0 else 0.0

    return pd.DataFrame(weights)


def _aggregate_sun_abraham(
    cohort_effects_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    results: sm.regression.linear_model.RegressionResultsWrapper,
    cluster_se: bool,
) -> tuple[float, float]:
    """
    Aggregate cohort effects using sample share weights.

    ATT = Σ_{g,l} w_{g,l}·β_{g,l}

    Standard error via delta method:
        Var(ATT) = w' Σ w
    where Σ is the covariance matrix of β_{g,l}

    Parameters:
        cohort_effects_df: DataFrame with cohort effects
        weights_df: DataFrame with weights
        results: Regression results (for covariance matrix)
        cluster_se: Whether cluster SEs were used

    Returns:
        - att: Aggregated ATT
        - se: Standard error
    """
    # Merge cohort effects and weights
    merged = cohort_effects_df.merge(weights_df, on=["cohort", "event_time"], how="inner")

    # Compute weighted average
    att = np.sum(merged["coef"] * merged["weight"])

    # Standard error via delta method
    # Extract covariance matrix for interaction coefficients
    interaction_cols = [
        f"cohort_{int(row['cohort'])}_event_{int(row['event_time'])}"
        for _, row in merged.iterrows()
    ]

    # Get covariance matrix for these coefficients
    cov_matrix = results.cov_params().loc[interaction_cols, interaction_cols].values

    # Weights vector
    w = merged["weight"].values

    # Var(ATT) = w' Σ w
    var_att = w @ cov_matrix @ w

    se = np.sqrt(var_att)

    return att, se
