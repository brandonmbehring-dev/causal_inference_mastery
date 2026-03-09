"""
Staggered Difference-in-Differences Implementation.

This module provides infrastructure for staggered DiD designs where treatment timing
varies across units (cohorts treated at different times).

WARNING: The TWFE estimator in this module is BIASED with heterogeneous treatment effects.
Use Callaway-Sant'Anna or Sun-Abraham estimators instead (see callaway_santanna.py and sun_abraham.py).

Key References:
    - Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing."
      Journal of Econometrics 225(2): 254-277.
    - Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with
      Multiple Time Periods." Journal of Econometrics 225(2): 200-230.
    - Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event
      Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class StaggeredData:
    """
    Container for staggered DiD data.

    In staggered designs, treatment timing varies across units:
    - Some units treated at t=3, others at t=5, others never treated
    - Creates multiple "cohorts" based on treatment timing

    Attributes:
        outcomes: Outcome variable (n_obs,)
        treatment: Binary treatment indicator (0/1) for each observation (n_obs,)
        time: Time period for each observation (n_obs,)
        unit_id: Unit identifier for each observation (n_obs,)
        treatment_time: Treatment time for each unit (n_units,)
                       np.inf for never-treated units

    Example:
        >>> # 3 units: treated at t=3, t=5, never
        >>> outcomes = np.array([...])  # 3 units × 10 periods = 30 obs
        >>> treatment = np.array([...])  # 0/1 indicators
        >>> time = np.array([0,1,2,...,9, 0,1,2,...,9, 0,1,2,...,9])
        >>> unit_id = np.array([1,1,1,...,1, 2,2,2,...,2, 3,3,3,...,3])
        >>> treatment_time = np.array([3, 5, np.inf])
        >>> data = StaggeredData(outcomes, treatment, time, unit_id, treatment_time)
        >>> data.cohorts  # array([3, 5]) - two treatment cohorts
        >>> data.never_treated_mask  # array([False, False, True])
    """

    outcomes: np.ndarray
    treatment: np.ndarray
    time: np.ndarray
    unit_id: np.ndarray
    treatment_time: np.ndarray  # Treatment time per unit (np.inf for never-treated)

    def __post_init__(self) -> None:
        """Validate staggered data structure."""
        # Check array lengths
        n_obs = len(self.outcomes)
        if not (
            len(self.treatment) == n_obs and len(self.time) == n_obs and len(self.unit_id) == n_obs
        ):
            raise ValueError(
                f"outcomes, treatment, time, unit_id must have same length. "
                f"Got outcomes={len(self.outcomes)}, treatment={len(self.treatment)}, "
                f"time={len(self.time)}, unit_id={len(self.unit_id)}"
            )

        # Check treatment_time length matches number of unique units
        unique_units = np.unique(self.unit_id)
        if len(self.treatment_time) != len(unique_units):
            raise ValueError(
                f"treatment_time must have one entry per unit. "
                f"Got {len(self.treatment_time)} entries but {len(unique_units)} unique units"
            )

        # Check treatment_time values are reasonable
        time_min = self.time.min()
        time_max = self.time.max()
        finite_treatment_times = self.treatment_time[np.isfinite(self.treatment_time)]
        if len(finite_treatment_times) > 0:
            if finite_treatment_times.min() < time_min:
                raise ValueError(
                    f"treatment_time contains values ({finite_treatment_times.min()}) "
                    f"before first time period ({time_min})"
                )
            # Allow treatment at last period but warn
            if finite_treatment_times.max() > time_max:
                raise ValueError(
                    f"treatment_time contains values ({finite_treatment_times.max()}) "
                    f"after last time period ({time_max})"
                )

        # Check for variation in treatment timing (required for staggered design)
        if len(self.cohorts) < 2 and not np.any(self.never_treated_mask):
            raise ValueError(
                "Staggered design requires variation in treatment timing. "
                "Found only one cohort and no never-treated units. "
                "For single treatment time, use did_2x2() or event_study() instead."
            )

    @property
    def cohorts(self) -> np.ndarray:
        """
        Unique treatment times (cohorts/groups).

        Returns:
            Array of treatment times for treated cohorts (excludes np.inf)

        Example:
            >>> data.treatment_time = np.array([3, 5, 3, np.inf])
            >>> data.cohorts  # array([3, 5])
        """
        return np.unique(self.treatment_time[np.isfinite(self.treatment_time)])

    @property
    def never_treated_mask(self) -> np.ndarray:
        """
        Boolean mask for never-treated units.

        Returns:
            Boolean array (n_units,) where True indicates never-treated

        Example:
            >>> data.treatment_time = np.array([3, 5, np.inf])
            >>> data.never_treated_mask  # array([False, False, True])
        """
        return np.isinf(self.treatment_time)

    @property
    def n_cohorts(self) -> int:
        """Number of treatment cohorts (excludes never-treated)."""
        return len(self.cohorts)

    @property
    def n_units(self) -> int:
        """Total number of units."""
        return len(np.unique(self.unit_id))

    @property
    def n_periods(self) -> int:
        """Number of time periods."""
        return len(np.unique(self.time))

    @property
    def n_obs(self) -> int:
        """Total number of observations (unit × time)."""
        return len(self.outcomes)


def create_staggered_data(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: Optional[np.ndarray] = None,
) -> StaggeredData:
    """
    Create StaggeredData from arrays.

    If treatment_time is not provided, it will be inferred from the treatment array
    (first time period where treatment=1 for each unit).

    Parameters:
        outcomes: Outcome variable (n_obs,)
        treatment: Binary treatment indicator (n_obs,)
        time: Time period (n_obs,)
        unit_id: Unit identifier (n_obs,)
        treatment_time: Optional treatment time per unit (n_units,)
                       If not provided, inferred from treatment array
                       Use np.inf for never-treated units

    Returns:
        StaggeredData instance

    Raises:
        ValueError: If treatment_time cannot be inferred or data is invalid

    Example:
        >>> outcomes = np.array([...])
        >>> treatment = np.array([0,0,1,1, 0,1,1,1, 0,0,0,0])  # 3 units, 4 periods each
        >>> time = np.array([0,1,2,3, 0,1,2,3, 0,1,2,3])
        >>> unit_id = np.array([1,1,1,1, 2,2,2,2, 3,3,3,3])
        >>> data = create_staggered_data(outcomes, treatment, time, unit_id)
        >>> data.treatment_time  # array([2, 1, inf]) - inferred from treatment array
    """
    # Validate input arrays
    n_obs = len(outcomes)
    if not (len(treatment) == n_obs and len(time) == n_obs and len(unit_id) == n_obs):
        raise ValueError(
            f"All input arrays must have same length. "
            f"Got outcomes={len(outcomes)}, treatment={len(treatment)}, "
            f"time={len(time)}, unit_id={len(unit_id)}"
        )

    # Check for NaN in outcomes
    if np.any(np.isnan(outcomes)):
        raise ValueError("outcomes contains NaN values. Remove or impute missing data.")

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.array_equal(unique_treatment, [0, 1]) and not np.array_equal(unique_treatment, [0]):
        raise ValueError(f"treatment must be binary (0, 1). Got unique values: {unique_treatment}")

    # Infer treatment_time if not provided
    if treatment_time is None:
        treatment_time = _infer_treatment_time(treatment, time, unit_id)

    # Validate treatment_time has correct length
    unique_units = np.unique(unit_id)
    if len(treatment_time) != len(unique_units):
        raise ValueError(
            f"treatment_time must have one entry per unit. "
            f"Got {len(treatment_time)} entries but {len(unique_units)} unique units. "
            f"Ensure units are numbered 0, 1, ..., n-1 or provide explicit mapping."
        )

    return StaggeredData(
        outcomes=outcomes,
        treatment=treatment,
        time=time,
        unit_id=unit_id,
        treatment_time=treatment_time,
    )


def _infer_treatment_time(
    treatment: np.ndarray, time: np.ndarray, unit_id: np.ndarray
) -> np.ndarray:
    """
    Infer treatment time for each unit from treatment array.

    Treatment time is the first period where treatment=1. If a unit is never treated,
    treatment time is np.inf.

    Parameters:
        treatment: Binary treatment indicator (n_obs,)
        time: Time period (n_obs,)
        unit_id: Unit identifier (n_obs,)

    Returns:
        treatment_time: Array of treatment times (n_units,)

    Raises:
        ValueError: If treatment timing is not time-invariant for any unit
    """
    unique_units = np.unique(unit_id)
    treatment_time = np.full(len(unique_units), np.inf)

    for i, unit in enumerate(unique_units):
        unit_mask = unit_id == unit
        unit_treatment = treatment[unit_mask]
        unit_time = time[unit_mask]

        # Check if unit ever treated
        if np.any(unit_treatment == 1):
            # Find first period where treatment=1
            first_treatment_idx = np.where(unit_treatment == 1)[0][0]
            treatment_time[i] = unit_time[first_treatment_idx]

            # Validate time-invariant treatment (once treated, always treated)
            if not np.all(unit_treatment[first_treatment_idx:] == 1):
                raise ValueError(
                    f"Unit {unit} has time-varying treatment. "
                    f"Treatment must be time-invariant (once treated, always treated)."
                )

    return treatment_time


def identify_cohorts(treatment_time: np.ndarray) -> Dict[float, np.ndarray]:
    """
    Map cohort (treatment time) to unit indices.

    Parameters:
        treatment_time: Treatment time for each unit (n_units,)

    Returns:
        Dictionary mapping cohort (treatment time) to array of unit indices
        Never-treated units (treatment_time=np.inf) are stored under key np.inf

    Example:
        >>> treatment_time = np.array([3, 5, 3, np.inf, 5])
        >>> cohorts = identify_cohorts(treatment_time)
        >>> cohorts[3]  # array([0, 2]) - units 0 and 2 treated at t=3
        >>> cohorts[5]  # array([1, 4]) - units 1 and 4 treated at t=5
        >>> cohorts[np.inf]  # array([3]) - unit 3 never treated
    """
    cohort_map = {}
    unique_times = np.unique(treatment_time)

    for t in unique_times:
        cohort_map[t] = np.where(treatment_time == t)[0]

    return cohort_map


def twfe_staggered(
    data: StaggeredData,
    cluster_se: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-Way Fixed Effects (TWFE) estimator with staggered adoption.

    WARNING: This estimator is BIASED with heterogeneous treatment effects across
    cohorts or over time. It is included only for comparison purposes to demonstrate
    the bias problem. Use callaway_santanna() or sun_abraham() for unbiased estimation.

    Regression:
        Y_it = α_i + λ_t + β·D_it + ε_it

    where:
        - α_i: Unit fixed effects
        - λ_t: Time fixed effects
        - β: Single treatment effect coefficient (assumes homogeneity)
        - D_it: Treatment indicator (1 if unit i treated at time t)

    The problem with TWFE:
        - With heterogeneous effects, β is a weighted average of cohort effects
        - Some weights can be NEGATIVE (forbidden comparisons)
        - Uses "already treated" units as controls (invalid)
        - Results in biased estimates, even asymptotically

    Parameters:
        data: StaggeredData instance
        cluster_se: If True, use cluster-robust SEs at unit level
        alpha: Significance level for confidence intervals (default 0.05)

    Returns:
        Dictionary with:
            - att: Estimated average treatment effect (BIASED with heterogeneity)
            - se: Standard error
            - t_stat: t-statistic
            - p_value: p-value for H0: β=0
            - ci_lower, ci_upper: (1-alpha)*100% confidence interval
            - n_obs: Number of observations
            - n_treated: Number of treated observations
            - n_control: Number of control observations
            - n_units: Number of units
            - n_cohorts: Number of treatment cohorts
            - cluster_se_used: Whether cluster SEs were used
            - warning: String warning about bias with heterogeneous effects

    Raises:
        ValueError: If no treated units, no control units, or other data issues

    Example:
        >>> # WARNING: This will be biased if cohorts have different effects!
        >>> result = twfe_staggered(data)
        >>> print(result["warning"])  # Will warn about bias
        >>> result["att"]  # Biased estimate (do not use for inference!)

    References:
        Goodman-Bacon (2021) decomposes TWFE with staggered timing and shows
        bias with heterogeneous treatment effects.
    """
    # Input validation
    if not np.any(data.treatment == 1):
        raise ValueError("No treated units found. treatment array contains only zeros.")

    if not np.any(data.treatment == 0):
        raise ValueError("No control observations found. treatment array contains only ones.")

    # Create DataFrame for regression
    df = pd.DataFrame(
        {
            "outcome": data.outcomes,
            "treatment": data.treatment,
            "unit_id": data.unit_id,
            "time": data.time,
        }
    )

    # Create unit and time fixed effects
    unit_dummies = pd.get_dummies(df["unit_id"], prefix="unit", drop_first=True).astype(float)
    time_dummies = pd.get_dummies(df["time"], prefix="time", drop_first=True).astype(float)

    # Construct design matrix: [treatment, unit FE, time FE]
    X = pd.concat([df[["treatment"]], unit_dummies, time_dummies], axis=1)
    y = df["outcome"]

    # Fit OLS regression
    if cluster_se:
        # Cluster-robust standard errors at unit level
        model = sm.OLS(y, X)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df["unit_id"].values})
    else:
        # Regular OLS standard errors
        model = sm.OLS(y, X)
        results = model.fit()

    # Extract treatment effect coefficient (first column)
    att = results.params.iloc[0]
    se = results.bse.iloc[0]
    t_stat = results.tvalues.iloc[0]
    p_value = results.pvalues.iloc[0]

    # Confidence interval
    ci = results.conf_int(alpha=alpha).iloc[0]
    ci_lower = ci[0]
    ci_upper = ci[1]

    # Diagnostics
    n_obs = len(data.outcomes)
    n_treated = int(np.sum(data.treatment))
    n_control = n_obs - n_treated
    n_units = data.n_units
    n_cohorts = data.n_cohorts

    # WARNING about bias
    warning = (
        "WARNING: TWFE estimator is BIASED with heterogeneous treatment effects. "
        f"With {n_cohorts} cohorts, effects may vary across cohorts or over time. "
        "TWFE uses 'already treated' units as implicit controls (forbidden comparison). "
        "This can produce NEGATIVE estimates even when all true effects are positive. "
        "Use callaway_santanna() or sun_abraham() for unbiased estimation."
    )

    return {
        "att": float(att),
        "se": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_obs": n_obs,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_units": n_units,
        "n_cohorts": n_cohorts,
        "cluster_se_used": cluster_se,
        "warning": warning,
    }
