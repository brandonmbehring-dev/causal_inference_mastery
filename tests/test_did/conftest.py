"""Test fixtures for DiD tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_did_data():
    """
    Simple 2×2 DiD dataset with known treatment effect.

    Setup:
    - 50 treated units, 50 control units
    - 1 pre-period (t=0), 1 post-period (t=1)
    - True DiD effect = 2.0
    - Baseline outcomes: control=10, treated=12 (2-unit difference)
    - Common time trend: +3 units in post-period
    - Treatment effect: +2 units for treated in post only

    Expected DiD: (14 - 12) - (13 - 10) = 2 - 3 = -1... wait, let me recalculate.

    Actually:
    - Control: pre=10, post=13 (change = +3)
    - Treated: pre=12, post=17 (change = +5)
    - DiD = (17-12) - (13-10) = 5 - 3 = 2.0 ✓
    """
    np.random.seed(42)
    n_treated = 50
    n_control = 50

    # Control group
    control_pre = np.full(n_control, 10.0) + np.random.normal(0, 0.5, n_control)
    control_post = np.full(n_control, 13.0) + np.random.normal(0, 0.5, n_control)

    # Treated group
    treated_pre = np.full(n_treated, 12.0) + np.random.normal(0, 0.5, n_treated)
    treated_post = np.full(n_treated, 17.0) + np.random.normal(0, 0.5, n_treated)

    # Combine
    outcomes = np.concatenate([control_pre, control_post, treated_pre, treated_post])
    treatment = np.concatenate([
        np.zeros(n_control * 2),
        np.ones(n_treated * 2)
    ])
    post = np.concatenate([
        np.zeros(n_control), np.ones(n_control),
        np.zeros(n_treated), np.ones(n_treated)
    ])
    unit_id = np.concatenate([
        np.arange(n_control), np.arange(n_control),
        np.arange(n_control, n_control + n_treated),
        np.arange(n_control, n_control + n_treated)
    ])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "post": post,
        "unit_id": unit_id,
        "true_did": 2.0,
        "n_treated": n_treated,
        "n_control": n_control,
    }


@pytest.fixture
def balanced_panel_data():
    """
    Balanced panel with multiple pre/post periods.

    Setup:
    - 40 treated, 40 control
    - 3 pre-periods (t=-2, -1, 0), 3 post-periods (t=1, 2, 3)
    - Treatment at t=1
    - True effect = 5.0 (constant in all post-periods)
    - Parallel pre-trends
    """
    np.random.seed(123)
    n_treated = 40
    n_control = 40
    n_units = n_treated + n_control
    n_pre = 3
    n_post = 3
    n_periods = n_pre + n_post

    outcomes = []
    treatment_vec = []
    time_vec = []
    post_vec = []
    unit_id_vec = []

    # Time trend common to both groups
    time_trend = np.array([-2, -1, 0, 1, 2, 3]) * 0.5  # Slight upward trend

    for unit in range(n_units):
        is_treated = unit >= n_control
        baseline = 20.0 if is_treated else 18.0  # Treated have higher baseline

        for t_idx, t in enumerate(range(-2, 4)):
            is_post = t >= 1

            # Outcome = baseline + time trend + treatment effect (if treated & post)
            y = baseline + time_trend[t_idx]
            if is_treated and is_post:
                y += 5.0  # True DiD effect

            y += np.random.normal(0, 1.0)  # Noise

            outcomes.append(y)
            treatment_vec.append(1 if is_treated else 0)
            time_vec.append(t)
            post_vec.append(1 if is_post else 0)
            unit_id_vec.append(unit)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment_vec),
        "post": np.array(post_vec),
        "time": np.array(time_vec),
        "unit_id": np.array(unit_id_vec),
        "true_did": 5.0,
        "n_treated": n_treated,
        "n_control": n_control,
        "treatment_time": 1,
    }


@pytest.fixture
def heterogeneous_baselines_data():
    """
    DiD with heterogeneous baseline levels across units.

    Setup:
    - 30 treated, 30 control
    - Baseline levels vary widely (control: 5-15, treated: 8-18)
    - Common time effect: +4
    - Treatment effect: +3
    """
    np.random.seed(456)
    n_treated = 30
    n_control = 30

    # Control group with heterogeneous baselines
    control_baselines = np.random.uniform(5, 15, n_control)
    control_pre = control_baselines + np.random.normal(0, 0.5, n_control)
    control_post = control_baselines + 4.0 + np.random.normal(0, 0.5, n_control)

    # Treated group with heterogeneous baselines
    treated_baselines = np.random.uniform(8, 18, n_treated)
    treated_pre = treated_baselines + np.random.normal(0, 0.5, n_treated)
    treated_post = treated_baselines + 4.0 + 3.0 + np.random.normal(0, 0.5, n_treated)

    # Combine
    outcomes = np.concatenate([control_pre, control_post, treated_pre, treated_post])
    treatment = np.concatenate([
        np.zeros(n_control * 2),
        np.ones(n_treated * 2)
    ])
    post = np.concatenate([
        np.zeros(n_control), np.ones(n_control),
        np.zeros(n_treated), np.ones(n_treated)
    ])
    unit_id = np.concatenate([
        np.arange(n_control), np.arange(n_control),
        np.arange(n_control, n_control + n_treated),
        np.arange(n_control, n_control + n_treated)
    ])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "post": post,
        "unit_id": unit_id,
        "true_did": 3.0,
        "n_treated": n_treated,
        "n_control": n_control,
    }


@pytest.fixture
def zero_effect_data():
    """
    DiD with zero treatment effect (pure common trends).

    Setup:
    - 50 treated, 50 control
    - True effect = 0.0
    - Common time trend: +2
    """
    np.random.seed(789)
    n_treated = 50
    n_control = 50

    # Control group
    control_pre = np.full(n_control, 15.0) + np.random.normal(0, 1.0, n_control)
    control_post = np.full(n_control, 17.0) + np.random.normal(0, 1.0, n_control)

    # Treated group (same time trend, no treatment effect)
    treated_pre = np.full(n_treated, 18.0) + np.random.normal(0, 1.0, n_treated)
    treated_post = np.full(n_treated, 20.0) + np.random.normal(0, 1.0, n_treated)

    # Combine
    outcomes = np.concatenate([control_pre, control_post, treated_pre, treated_post])
    treatment = np.concatenate([
        np.zeros(n_control * 2),
        np.ones(n_treated * 2)
    ])
    post = np.concatenate([
        np.zeros(n_control), np.ones(n_control),
        np.zeros(n_treated), np.ones(n_treated)
    ])
    unit_id = np.concatenate([
        np.arange(n_control), np.arange(n_control),
        np.arange(n_control, n_control + n_treated),
        np.arange(n_control, n_control + n_treated)
    ])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "post": post,
        "unit_id": unit_id,
        "true_did": 0.0,
        "n_treated": n_treated,
        "n_control": n_control,
    }


@pytest.fixture
def negative_effect_data():
    """
    DiD with negative treatment effect.

    Setup:
    - 60 treated, 60 control
    - True effect = -4.0 (treatment harms outcomes)
    - Common trend: +5
    """
    np.random.seed(101112)
    n_treated = 60
    n_control = 60

    # Control group
    control_pre = np.full(n_control, 25.0) + np.random.normal(0, 1.5, n_control)
    control_post = np.full(n_control, 30.0) + np.random.normal(0, 1.5, n_control)

    # Treated group (same trend but treatment reduces outcome by 4)
    treated_pre = np.full(n_treated, 24.0) + np.random.normal(0, 1.5, n_treated)
    treated_post = np.full(n_treated, 25.0) + np.random.normal(0, 1.5, n_treated)  # 24 + 5 - 4 = 25

    # Combine
    outcomes = np.concatenate([control_pre, control_post, treated_pre, treated_post])
    treatment = np.concatenate([
        np.zeros(n_control * 2),
        np.ones(n_treated * 2)
    ])
    post = np.concatenate([
        np.zeros(n_control), np.ones(n_control),
        np.zeros(n_treated), np.ones(n_treated)
    ])
    unit_id = np.concatenate([
        np.arange(n_control), np.arange(n_control),
        np.arange(n_control, n_control + n_treated),
        np.arange(n_control, n_control + n_treated)
    ])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "post": post,
        "unit_id": unit_id,
        "true_did": -4.0,
        "n_treated": n_treated,
        "n_control": n_control,
    }


# ================================
# Event Study Fixtures
# ================================


@pytest.fixture
def event_study_constant_effect_data():
    """
    Event study with constant treatment effect across all post-periods.

    Setup:
    - 30 treated, 30 control
    - 3 pre-periods (t=0, 1, 2), 3 post-periods (t=3, 4, 5)
    - Treatment at t=3
    - True effect = 2.5 (constant in all post-periods)
    - Parallel pre-trends (leads should be ~0)
    """
    np.random.seed(111)
    n_treated = 30
    n_control = 30
    n_units = n_treated + n_control
    n_periods = 6
    treatment_time = 3

    outcomes = []
    treatment_vec = []
    time_vec = []
    unit_id_vec = []

    # Time trend common to both groups
    time_trend = np.arange(n_periods) * 0.3

    for unit in range(n_units):
        is_treated = unit >= n_control
        baseline = 15.0 if is_treated else 12.0

        for t in range(n_periods):
            is_post = t >= treatment_time

            # Outcome = baseline + time trend + treatment effect (if treated & post)
            y = baseline + time_trend[t]
            if is_treated and is_post:
                y += 2.5  # Constant treatment effect

            y += np.random.normal(0, 0.5)  # Noise

            outcomes.append(y)
            treatment_vec.append(1 if is_treated else 0)
            time_vec.append(t)
            unit_id_vec.append(unit)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment_vec),
        "time": np.array(time_vec),
        "unit_id": np.array(unit_id_vec),
        "treatment_time": treatment_time,
        "true_effect": 2.5,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_periods": n_periods,
    }


@pytest.fixture
def event_study_dynamic_effect_data():
    """
    Event study with dynamic treatment effects (increasing over time).

    Setup:
    - 40 treated, 40 control
    - 3 pre-periods (t=0, 1, 2), 4 post-periods (t=3, 4, 5, 6)
    - Treatment at t=3
    - True effects: [1.0, 2.0, 3.0, 4.0] in periods 3, 4, 5, 6
    - Parallel pre-trends (leads should be ~0)
    """
    np.random.seed(222)
    n_treated = 40
    n_control = 40
    n_units = n_treated + n_control
    n_periods = 7
    treatment_time = 3

    # Dynamic treatment effects by period relative to treatment
    true_effects = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

    outcomes = []
    treatment_vec = []
    time_vec = []
    unit_id_vec = []

    # Time trend common to both groups
    time_trend = np.arange(n_periods) * 0.4

    for unit in range(n_units):
        is_treated = unit >= n_control
        baseline = 18.0 if is_treated else 16.0

        for t in range(n_periods):
            is_post = t >= treatment_time

            # Outcome = baseline + time trend + treatment effect (if treated & post)
            y = baseline + time_trend[t]
            if is_treated and is_post:
                relative_time = t - treatment_time
                y += true_effects[relative_time]  # Dynamic effect

            y += np.random.normal(0, 0.6)  # Noise

            outcomes.append(y)
            treatment_vec.append(1 if is_treated else 0)
            time_vec.append(t)
            unit_id_vec.append(unit)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment_vec),
        "time": np.array(time_vec),
        "unit_id": np.array(unit_id_vec),
        "treatment_time": treatment_time,
        "true_effects": true_effects,  # Dictionary: {lag: effect}
        "n_treated": n_treated,
        "n_control": n_control,
        "n_periods": n_periods,
    }


@pytest.fixture
def event_study_anticipation_data():
    """
    Event study with anticipation effects (violation of parallel trends).

    Setup:
    - 35 treated, 35 control
    - 4 pre-periods (t=0, 1, 2, 3), 3 post-periods (t=4, 5, 6)
    - Treatment at t=4
    - Anticipation effect: +1.5 in t=3 (one period before treatment)
    - True post-treatment effects: [3.0, 3.5, 4.0]
    - Leads should show significant effect at t=-1
    """
    np.random.seed(333)
    n_treated = 35
    n_control = 35
    n_units = n_treated + n_control
    n_periods = 7
    treatment_time = 4

    outcomes = []
    treatment_vec = []
    time_vec = []
    unit_id_vec = []

    # Time trend common to both groups
    time_trend = np.arange(n_periods) * 0.25

    for unit in range(n_units):
        is_treated = unit >= n_control
        baseline = 20.0 if is_treated else 18.0

        for t in range(n_periods):
            # Outcome = baseline + time trend + effects
            y = baseline + time_trend[t]

            if is_treated:
                # Anticipation effect one period before treatment
                if t == treatment_time - 1:
                    y += 1.5  # Anticipation

                # Post-treatment effects
                if t >= treatment_time:
                    relative_time = t - treatment_time
                    effects = {0: 3.0, 1: 3.5, 2: 4.0}
                    y += effects[relative_time]

            y += np.random.normal(0, 0.7)  # Noise

            outcomes.append(y)
            treatment_vec.append(1 if is_treated else 0)
            time_vec.append(t)
            unit_id_vec.append(unit)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment_vec),
        "time": np.array(time_vec),
        "unit_id": np.array(unit_id_vec),
        "treatment_time": treatment_time,
        "anticipation_effect": 1.5,  # Expected at t=-1
        "true_post_effects": {0: 3.0, 1: 3.5, 2: 4.0},
        "n_treated": n_treated,
        "n_control": n_control,
        "n_periods": n_periods,
    }


@pytest.fixture
def event_study_many_periods_data():
    """
    Event study with many pre/post periods.

    Setup:
    - 50 treated, 50 control
    - 10 pre-periods (t=0-9), 10 post-periods (t=10-19)
    - Treatment at t=10
    - True effect = 3.0 (constant in all post-periods)
    - Parallel pre-trends (all leads should be ~0)
    """
    np.random.seed(444)
    n_treated = 50
    n_control = 50
    n_units = n_treated + n_control
    n_periods = 20
    treatment_time = 10

    outcomes = []
    treatment_vec = []
    time_vec = []
    unit_id_vec = []

    # Time trend common to both groups
    time_trend = np.arange(n_periods) * 0.2

    for unit in range(n_units):
        is_treated = unit >= n_control
        baseline = 25.0 if is_treated else 23.0

        for t in range(n_periods):
            is_post = t >= treatment_time

            # Outcome = baseline + time trend + treatment effect (if treated & post)
            y = baseline + time_trend[t]
            if is_treated and is_post:
                y += 3.0  # Constant treatment effect

            y += np.random.normal(0, 0.8)  # Noise

            outcomes.append(y)
            treatment_vec.append(1 if is_treated else 0)
            time_vec.append(t)
            unit_id_vec.append(unit)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment_vec),
        "time": np.array(time_vec),
        "unit_id": np.array(unit_id_vec),
        "treatment_time": treatment_time,
        "true_effect": 3.0,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_periods": n_periods,
        "n_leads": 10,
        "n_lags": 10,
    }

# ============================================================================
# Staggered DiD Fixtures (for modern methods)
# ============================================================================


@pytest.fixture
def staggered_homogeneous_data():
    """
    Staggered DiD with HOMOGENEOUS treatment effects (TWFE unbiased).

    Setup:
    - 150 units: 50 each in cohorts 5, 7, never-treated
    - 10 time periods (0-9)
    - Cohort 5 treated at t=5 with effect = 2.5
    - Cohort 7 treated at t=7 with effect = 2.5 (SAME effect)
    - Overall true ATT = 2.5 (homogeneous)

    Expected:
    - TWFE unbiased (effects are homogeneous)
    - Callaway-Sant'Anna ≈ 2.5
    - Sun-Abraham ≈ 2.5
    """
    np.random.seed(100)
    n_periods = 10
    cohorts = [5, 7]
    effect = 2.5  # Same for both cohorts

    units_per_cohort = 50
    n_units = units_per_cohort * 3  # 2 cohorts + never-treated

    # Generate unit fixed effects
    unit_fe = np.random.normal(0, 1, n_units)

    # Assign units to cohorts
    treatment_time = np.concatenate([
        np.full(units_per_cohort, 5),    # Cohort 5
        np.full(units_per_cohort, 7),    # Cohort 7
        np.full(units_per_cohort, np.inf)  # Never-treated
    ])

    # Build panel
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
                y += effect  # Same effect for all cohorts
                d_it = 1
            else:
                d_it = 0

            # Add noise
            y += np.random.normal(0, 1)

            outcomes.append(y)
            treatment.append(d_it)
            time.append(t)
            unit_id.append(i)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment),
        "time": np.array(time),
        "unit_id": np.array(unit_id),
        "treatment_time": treatment_time,
        "true_effect": effect,
        "cohorts": cohorts,
        "n_units": n_units,
        "n_periods": n_periods,
    }


@pytest.fixture
def staggered_heterogeneous_data():
    """
    Staggered DiD with HETEROGENEOUS treatment effects (TWFE biased).

    Setup:
    - 150 units: 50 each in cohorts 5, 7, never-treated
    - 10 time periods (0-9)
    - Cohort 5 treated at t=5 with effect = 1.0
    - Cohort 7 treated at t=7 with effect = 5.0 (DIFFERENT effect - stronger heterogeneity)
    - Overall true ATT = 3.0 (weighted average: (1.0 + 5.0) / 2)

    Expected:
    - TWFE BIASED (negative bias due to forbidden comparisons)
    - Callaway-Sant'Anna ≈ 3.0 (unbiased)
    - Sun-Abraham ≈ 3.0 (unbiased)
    """
    np.random.seed(101)
    n_periods = 10
    cohorts = [5, 7]
    effects = {5: 1.0, 7: 5.0}  # Heterogeneous! (stronger: 4.0 difference instead of 2.0)

    units_per_cohort = 50
    n_units = units_per_cohort * 3

    # Generate unit fixed effects
    unit_fe = np.random.normal(0, 1, n_units)

    # Assign units to cohorts
    treatment_time = np.concatenate([
        np.full(units_per_cohort, 5),
        np.full(units_per_cohort, 7),
        np.full(units_per_cohort, np.inf)
    ])

    # Build panel
    outcomes = []
    treatment = []
    time = []
    unit_id = []

    for i in range(n_units):
        for t in range(n_periods):
            y = unit_fe[i] + 0.5 * t

            g_i = treatment_time[i]
            if np.isfinite(g_i) and t >= g_i:
                y += effects[g_i]  # Different effect per cohort
                d_it = 1
            else:
                d_it = 0

            y += np.random.normal(0, 1)

            outcomes.append(y)
            treatment.append(d_it)
            time.append(t)
            unit_id.append(i)

    # True ATT = weighted average of cohort effects
    true_att = np.mean([effects[g] for g in cohorts])

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment),
        "time": np.array(time),
        "unit_id": np.array(unit_id),
        "treatment_time": treatment_time,
        "true_effect": true_att,
        "cohort_effects": effects,
        "cohorts": cohorts,
        "n_units": n_units,
        "n_periods": n_periods,
    }


@pytest.fixture
def staggered_dynamic_data():
    """
    Staggered DiD with dynamic treatment effects over time.

    Setup:
    - 100 units: 50 treated at t=5, 50 never-treated
    - 10 time periods (0-9)
    - Effects increase over time: [1.0, 1.5, 2.0, 2.5, 3.0] at event times [0, 1, 2, 3, 4]

    Expected:
    - Callaway-Sant'Anna dynamic aggregation matches true event time effects
    - Sun-Abraham cohort × event time coefficients match true effects
    """
    np.random.seed(102)
    n_periods = 10
    treatment_time_val = 5
    n_treated = 50
    n_control = 50
    n_units = n_treated + n_control

    # Dynamic effects: increase over time since treatment
    event_time_effects = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0}

    unit_fe = np.random.normal(0, 1, n_units)

    treatment_time = np.concatenate([
        np.full(n_treated, treatment_time_val),
        np.full(n_control, np.inf)
    ])

    outcomes = []
    treatment = []
    time = []
    unit_id = []

    for i in range(n_units):
        for t in range(n_periods):
            y = unit_fe[i] + 0.5 * t

            g_i = treatment_time[i]
            if np.isfinite(g_i) and t >= g_i:
                event_time = int(t - g_i)
                if event_time in event_time_effects:
                    y += event_time_effects[event_time]
                else:
                    y += event_time_effects[max(event_time_effects.keys())]  # Constant after
                d_it = 1
            else:
                d_it = 0

            y += np.random.normal(0, 1)

            outcomes.append(y)
            treatment.append(d_it)
            time.append(t)
            unit_id.append(i)

    true_att = np.mean(list(event_time_effects.values()))

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment),
        "time": np.array(time),
        "unit_id": np.array(unit_id),
        "treatment_time": treatment_time,
        "true_effect": true_att,
        "event_time_effects": event_time_effects,
        "n_units": n_units,
        "n_periods": n_periods,
    }
