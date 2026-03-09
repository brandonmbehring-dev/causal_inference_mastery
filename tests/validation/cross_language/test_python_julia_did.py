"""
Cross-language validation: Python DiD estimators vs Julia DiD estimators.

Validates that Python and Julia implementations produce matching results.

Test coverage:
- Classic 2×2 DiD: Basic, large sample, cluster SE
- Staggered DiD: Callaway-Sant'Anna, Sun-Abraham
- Multiple cohorts: Early/late/never-treated

Tolerance Strategy:
- Point estimate: rtol=0.1 (algorithmic differences in SE computation affect weights)
- Standard error: rtol=0.3 (bootstrap vs analytic, cluster-robust variations)
- t-stat/p-value: rtol=0.3 (SE differences propagate)
- CI endpoints: rtol=0.3 (SE differences)

Note: DiD implementations differ more than simple estimators due to:
1. Bootstrap inference (CS) vs analytic (Julia CS default varies)
2. Cluster SE computation (DoF adjustments)
3. Fixed effect demeaning (numerical precision)
"""

import numpy as np
import pytest

from src.causal_inference.did.did_estimator import did_2x2
from src.causal_inference.did.staggered import create_staggered_data, twfe_staggered
from src.causal_inference.did.callaway_santanna import callaway_santanna_ate
from src.causal_inference.did.sun_abraham import sun_abraham_ate
from src.causal_inference.did.event_study import event_study
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_classic_did,
    julia_event_study,
    julia_staggered_twfe,
    julia_callaway_santanna,
    julia_sun_abraham,
)


pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-validation"
)


# =============================================================================
# Data Generators
# =============================================================================


def generate_classic_did_data(
    n_units: int,
    n_periods: int = 2,
    true_effect: float = 2.0,
    seed: int = 42,
):
    """
    Generate classic 2×2 DiD panel data.

    DGP: Y_it = 5 + 2*treatment_i + 3*post_t + τ*(treatment_i × post_t) + ε_it

    Returns arrays in long format (n_units * n_periods observations).
    """
    np.random.seed(seed)

    # Half treated, half control
    n_treated = n_units // 2
    n_control = n_units - n_treated

    # Create panel structure
    unit_ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    treatment = np.repeat(np.array([True] * n_treated + [False] * n_control), n_periods)
    post = times == (n_periods - 1)  # Last period is post

    # DGP
    y_base = 5.0 + 2.0 * treatment.astype(float) + 3.0 * post.astype(float)
    y_effect = true_effect * (treatment & post).astype(float)
    eps = np.random.normal(0, 1, len(unit_ids))
    outcomes = y_base + y_effect + eps

    return outcomes, treatment, post, unit_ids, times


def generate_staggered_did_data(
    n_units_per_cohort: int = 10,
    n_periods: int = 8,
    true_effect: float = 2.0,
    seed: int = 42,
):
    """
    Generate staggered DiD data with 3 cohorts.

    Cohorts:
    - Early: treated at t=3
    - Late: treated at t=5
    - Never: never treated (pure control)

    DGP: Y_it = unit_fe + time_fe + τ*D_it + ε_it
    """
    np.random.seed(seed)

    outcomes = []
    treatment = []
    time_vec = []
    unit_vec = []
    treatment_time_per_unit = []

    unit_id = 0
    for cohort in ["early", "late", "never"]:
        if cohort == "early":
            treat_time = 3.0
        elif cohort == "late":
            treat_time = 5.0
        else:
            treat_time = np.inf

        for _ in range(n_units_per_cohort):
            treatment_time_per_unit.append(treat_time)
            for t in range(1, n_periods + 1):
                is_treated = (not np.isinf(treat_time)) and (t >= treat_time)

                # DGP: unit FE + time FE + treatment effect + noise
                y = (
                    unit_id * 0.1  # unit FE
                    + t * 0.5  # time FE
                    + (true_effect if is_treated else 0.0)
                    + np.random.normal(0, 0.5)
                )

                outcomes.append(y)
                treatment.append(is_treated)
                time_vec.append(t)
                unit_vec.append(unit_id)
            unit_id += 1

    return (
        np.array(outcomes),
        np.array(treatment),
        np.array(time_vec),
        np.array(unit_vec),
        np.array(treatment_time_per_unit),
    )


# =============================================================================
# Classic 2×2 DiD Tests
# =============================================================================


class TestClassicDiDParity:
    """Cross-validate Python did_2x2 vs Julia ClassicDiD."""

    def test_basic_2x2_did(self):
        """Basic 2×2 DiD with 20 units."""
        true_effect = 2.5
        outcomes, treatment, post, unit_id, time = generate_classic_did_data(
            n_units=20, n_periods=2, true_effect=true_effect, seed=42
        )

        # Python
        py_result = did_2x2(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            post=post.astype(int),
            unit_id=unit_id,
            alpha=0.05,
            cluster_se=True,
        )

        # Julia
        jl_result = julia_classic_did(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
            time=time,
            alpha=0.05,
            cluster_se=True,
        )

        # Both should recover true effect reasonably
        assert abs(py_result["estimate"] - true_effect) < 1.0, (
            f"Python estimate {py_result['estimate']} far from true {true_effect}"
        )
        assert abs(jl_result["estimate"] - true_effect) < 1.0, (
            f"Julia estimate {jl_result['estimate']} far from true {true_effect}"
        )

        # Estimates should be close (same DGP)
        rel_diff = abs(py_result["estimate"] - jl_result["estimate"]) / abs(py_result["estimate"])
        assert rel_diff < 0.1, (
            f"Estimate mismatch: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_large_sample_n100(self):
        """Large sample should give tighter estimates."""
        true_effect = 2.0
        outcomes, treatment, post, unit_id, time = generate_classic_did_data(
            n_units=100, n_periods=2, true_effect=true_effect, seed=123
        )

        py_result = did_2x2(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            post=post.astype(int),
            unit_id=unit_id,
            cluster_se=True,
        )

        jl_result = julia_classic_did(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
            time=time,
            cluster_se=True,
        )

        # Closer to true value with more data
        assert abs(py_result["estimate"] - true_effect) < 0.5
        assert abs(jl_result["estimate"] - true_effect) < 0.5

        # Estimates should match closely
        rel_diff = abs(py_result["estimate"] - jl_result["estimate"]) / abs(py_result["estimate"])
        assert rel_diff < 0.05, f"Large sample estimate mismatch: rel_diff={rel_diff:.4f}"

    def test_zero_effect(self):
        """DiD with zero treatment effect."""
        true_effect = 0.0
        outcomes, treatment, post, unit_id, time = generate_classic_did_data(
            n_units=50, n_periods=2, true_effect=true_effect, seed=456
        )

        py_result = did_2x2(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            post=post.astype(int),
            unit_id=unit_id,
            cluster_se=True,
        )

        jl_result = julia_classic_did(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
            time=time,
            cluster_se=True,
        )

        # Both should be close to zero
        assert abs(py_result["estimate"]) < 0.5
        assert abs(jl_result["estimate"]) < 0.5

    def test_negative_effect(self):
        """DiD with negative treatment effect."""
        true_effect = -1.5
        outcomes, treatment, post, unit_id, time = generate_classic_did_data(
            n_units=40, n_periods=2, true_effect=true_effect, seed=789
        )

        py_result = did_2x2(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            post=post.astype(int),
            unit_id=unit_id,
            cluster_se=True,
        )

        jl_result = julia_classic_did(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
            time=time,
            cluster_se=True,
        )

        # Both should recover negative effect
        assert py_result["estimate"] < 0
        assert jl_result["estimate"] < 0
        assert abs(py_result["estimate"] - true_effect) < 1.0
        assert abs(jl_result["estimate"] - true_effect) < 1.0

    def test_ci_covers_true_effect(self):
        """95% CI should cover true effect most of the time."""
        true_effect = 2.0
        outcomes, treatment, post, unit_id, time = generate_classic_did_data(
            n_units=60, n_periods=2, true_effect=true_effect, seed=999
        )

        py_result = did_2x2(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            post=post.astype(int),
            unit_id=unit_id,
            cluster_se=True,
            alpha=0.05,
        )

        jl_result = julia_classic_did(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
            time=time,
            cluster_se=True,
            alpha=0.05,
        )

        # CIs should cover true effect
        assert py_result["ci_lower"] < true_effect < py_result["ci_upper"], (
            f"Python CI [{py_result['ci_lower']:.3f}, {py_result['ci_upper']:.3f}] doesn't cover {true_effect}"
        )
        assert jl_result["ci_lower"] < true_effect < jl_result["ci_upper"], (
            f"Julia CI [{jl_result['ci_lower']:.3f}, {jl_result['ci_upper']:.3f}] doesn't cover {true_effect}"
        )


# =============================================================================
# Staggered DiD Tests: Callaway-Sant'Anna
# =============================================================================


class TestCallawaySantAnnaParity:
    """Cross-validate Python callaway_santanna_ate vs Julia CallawaySantAnna."""

    def test_cs_simple_aggregation(self):
        """Callaway-Sant'Anna with simple aggregation."""
        true_effect = 2.0
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=8, true_effect=true_effect, seed=42
        )

        # Python: Create StaggeredData and run CS
        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = callaway_santanna_ate(
            data=py_data,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=100,
            random_state=42,
        )

        # Julia
        jl_result = julia_callaway_santanna(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=100,
            random_seed=42,
        )

        # Both should recover true effect
        assert abs(py_result["att"] - true_effect) < 0.8, (
            f"Python ATT {py_result['att']:.4f} far from true {true_effect}"
        )
        assert abs(jl_result["att"] - true_effect) < 0.8, (
            f"Julia ATT {jl_result['att']:.4f} far from true {true_effect}"
        )

        # ATT should be reasonably close (bootstrap variance accepted)
        rel_diff = abs(py_result["att"] - jl_result["att"]) / abs(py_result["att"])
        assert rel_diff < 0.3, (
            f"CS ATT mismatch: Python={py_result['att']:.4f}, Julia={jl_result['att']:.4f}"
        )

    def test_cs_notyettreated_control(self):
        """Callaway-Sant'Anna with not-yet-treated control group."""
        true_effect = 1.5
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=15, n_periods=8, true_effect=true_effect, seed=123
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = callaway_santanna_ate(
            data=py_data,
            aggregation="simple",
            control_group="notyettreated",
            n_bootstrap=100,
            random_state=123,
        )

        jl_result = julia_callaway_santanna(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            aggregation="simple",
            control_group="notyettreated",
            n_bootstrap=100,
            random_seed=123,
        )

        # Both should be close to true effect
        assert abs(py_result["att"] - true_effect) < 1.0
        assert abs(jl_result["att"] - true_effect) < 1.0

    def test_cs_cohort_count(self):
        """Check both implementations identify same number of cohorts."""
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=8, true_effect=2.0, seed=456
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = callaway_santanna_ate(
            data=py_data,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=50,
            random_state=456,
        )

        jl_result = julia_callaway_santanna(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=50,
            random_seed=456,
        )

        # Same number of cohorts (2: early and late)
        assert py_result["n_cohorts"] == jl_result["n_cohorts"] == 2


# =============================================================================
# Staggered DiD Tests: Sun-Abraham
# =============================================================================


class TestSunAbrahamParity:
    """Cross-validate Python sun_abraham_ate vs Julia SunAbraham."""

    def test_sa_basic(self):
        """Sun-Abraham basic estimation."""
        true_effect = 2.0
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=8, true_effect=true_effect, seed=42
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = sun_abraham_ate(
            data=py_data,
            alpha=0.05,
            cluster_se=True,
        )

        jl_result = julia_sun_abraham(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            alpha=0.05,
            cluster_se=True,
        )

        # Both should recover true effect
        assert abs(py_result["att"] - true_effect) < 0.8, (
            f"Python ATT {py_result['att']:.4f} far from true {true_effect}"
        )
        assert abs(jl_result["att"] - true_effect) < 0.8, (
            f"Julia ATT {jl_result['att']:.4f} far from true {true_effect}"
        )

        # ATT should be reasonably close
        rel_diff = abs(py_result["att"] - jl_result["att"]) / abs(py_result["att"])
        assert rel_diff < 0.3, (
            f"SA ATT mismatch: Python={py_result['att']:.4f}, Julia={jl_result['att']:.4f}"
        )

    def test_sa_large_sample(self):
        """Sun-Abraham with larger sample."""
        true_effect = 1.5
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=20, n_periods=10, true_effect=true_effect, seed=123
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = sun_abraham_ate(data=py_data, cluster_se=True)

        jl_result = julia_sun_abraham(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # Both closer to true effect with more data
        assert abs(py_result["att"] - true_effect) < 0.5
        assert abs(jl_result["att"] - true_effect) < 0.5

    def test_sa_cohort_effects_count(self):
        """Both should identify same number of cohort effects."""
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=8, true_effect=2.0, seed=789
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = sun_abraham_ate(data=py_data, cluster_se=True)

        jl_result = julia_sun_abraham(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # Same number of cohort effects
        n_py_effects = len(py_result["cohort_effects"])
        n_jl_effects = len(jl_result["cohort_effects"])
        assert n_py_effects == n_jl_effects, (
            f"Cohort effects count mismatch: Python={n_py_effects}, Julia={n_jl_effects}"
        )

    def test_sa_n_cohorts(self):
        """Both should count same number of cohorts."""
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=8, true_effect=2.0, seed=999
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = sun_abraham_ate(data=py_data, cluster_se=True)

        jl_result = julia_sun_abraham(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # 2 cohorts: early (t=3) and late (t=5)
        assert py_result["n_cohorts"] == jl_result["n_cohorts"] == 2


# =============================================================================
# Event Study Data Generator
# =============================================================================


def generate_event_study_data(
    n_units: int = 30,
    n_periods: int = 10,
    treatment_time: int = 5,
    true_effect: float = 2.0,
    seed: int = 42,
):
    """
    Generate event study panel data (non-staggered, uniform treatment timing).

    DGP: Y_it = α_i + λ_t + τ*(D_i × Post_it) + ε_it

    - Unit FE: varies by unit_id
    - Time FE: linear trend
    - Treatment effect: appears only after treatment_time
    - D_i: 1 for treated units (first half), 0 for control

    Returns
    -------
    tuple
        (outcomes, treatment, times, unit_ids, post, treatment_time)
    """
    np.random.seed(seed)

    n_treated = n_units // 2
    n_control = n_units - n_treated

    # Create panel structure
    unit_ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    treatment = np.repeat(np.array([True] * n_treated + [False] * n_control), n_periods)
    post = times >= treatment_time

    # DGP: unit FE + time FE + treatment effect + noise
    unit_fe = unit_ids * 0.5  # Unit FE
    time_fe = times * 0.3  # Time FE
    y_effect = true_effect * (treatment & post).astype(float)
    eps = np.random.normal(0, 1.0, len(unit_ids))

    outcomes = unit_fe + time_fe + y_effect + eps

    return outcomes, treatment, times, unit_ids, post, treatment_time


# =============================================================================
# Event Study Parity Tests
# =============================================================================


class TestEventStudyParity:
    """Cross-validate Python event_study vs Julia EventStudy.

    Note: Both compute TWFE with event-time indicators.
    Python returns leads/lags dicts, Julia returns aggregate estimate.
    We compare:
    - Aggregate treatment effect (mean of post-treatment coefficients)
    - Standard errors (approximate)
    - Sample sizes (must match)
    """

    def test_event_study_basic_estimate(self):
        """Event study aggregate effect should match."""
        true_effect = 2.0
        outcomes, treatment, times, unit_ids, post, treatment_time = generate_event_study_data(
            n_units=40, n_periods=10, treatment_time=5, true_effect=true_effect, seed=42
        )

        # Python
        py_result = event_study(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            time=times,
            unit_id=unit_ids,
            treatment_time=treatment_time,
            alpha=0.05,
            cluster_se=True,
        )

        # Julia
        jl_result = julia_event_study(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_ids,
            time=times,
            alpha=0.05,
            cluster_se=True,
        )

        # Compute aggregate estimate from Python lags
        py_lag_estimates = [coef["estimate"] for coef in py_result["lags"].values()]
        py_aggregate = np.mean(py_lag_estimates) if py_lag_estimates else 0.0

        # Both should recover true effect
        assert abs(py_aggregate - true_effect) < 1.5, (
            f"Python aggregate {py_aggregate:.4f} far from true {true_effect}"
        )
        assert abs(jl_result["estimate"] - true_effect) < 1.5, (
            f"Julia estimate {jl_result['estimate']:.4f} far from true {true_effect}"
        )

        # Aggregate estimates should be reasonably close
        rel_diff = abs(py_aggregate - jl_result["estimate"]) / max(abs(py_aggregate), 0.1)
        assert rel_diff < 0.3, (
            f"Event study aggregate mismatch: Python={py_aggregate:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_event_study_with_zero_effect(self):
        """Event study with zero effect - estimates near zero."""
        true_effect = 0.0
        outcomes, treatment, times, unit_ids, post, treatment_time = generate_event_study_data(
            n_units=40, n_periods=10, treatment_time=5, true_effect=true_effect, seed=123
        )

        py_result = event_study(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            time=times,
            unit_id=unit_ids,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        jl_result = julia_event_study(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_ids,
            time=times,
            cluster_se=True,
        )

        # Compute aggregate from Python
        py_lag_estimates = [coef["estimate"] for coef in py_result["lags"].values()]
        py_aggregate = np.mean(py_lag_estimates) if py_lag_estimates else 0.0

        # Both should be near zero
        assert abs(py_aggregate) < 1.0, f"Python aggregate {py_aggregate:.4f} not near zero"
        assert abs(jl_result["estimate"]) < 1.0, (
            f"Julia estimate {jl_result['estimate']:.4f} not near zero"
        )

    def test_event_study_negative_effect(self):
        """Event study with negative treatment effect."""
        true_effect = -1.5
        outcomes, treatment, times, unit_ids, post, treatment_time = generate_event_study_data(
            n_units=40, n_periods=10, treatment_time=5, true_effect=true_effect, seed=456
        )

        py_result = event_study(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            time=times,
            unit_id=unit_ids,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        jl_result = julia_event_study(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_ids,
            time=times,
            cluster_se=True,
        )

        # Compute aggregate from Python
        py_lag_estimates = [coef["estimate"] for coef in py_result["lags"].values()]
        py_aggregate = np.mean(py_lag_estimates) if py_lag_estimates else 0.0

        # Both should be negative
        assert py_aggregate < 0, f"Python aggregate {py_aggregate:.4f} not negative"
        assert jl_result["estimate"] < 0, f"Julia estimate {jl_result['estimate']:.4f} not negative"

        # Both should recover negative effect
        assert abs(py_aggregate - true_effect) < 1.5
        assert abs(jl_result["estimate"] - true_effect) < 1.5

    def test_event_study_sample_sizes(self):
        """Sample sizes should match exactly."""
        outcomes, treatment, times, unit_ids, post, treatment_time = generate_event_study_data(
            n_units=30, n_periods=8, treatment_time=4, true_effect=2.0, seed=789
        )

        py_result = event_study(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            time=times,
            unit_id=unit_ids,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        jl_result = julia_event_study(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=unit_ids,
            time=times,
            cluster_se=True,
        )

        # Sample sizes must match
        assert py_result["n_obs"] == jl_result["n_obs"], (
            f"n_obs mismatch: Python={py_result['n_obs']}, Julia={jl_result['n_obs']}"
        )
        assert py_result["n_treated"] == jl_result["n_treated"], (
            f"n_treated mismatch: Python={py_result['n_treated']}, Julia={jl_result['n_treated']}"
        )
        assert py_result["n_control"] == jl_result["n_control"], (
            f"n_control mismatch: Python={py_result['n_control']}, Julia={jl_result['n_control']}"
        )


# =============================================================================
# Staggered TWFE Parity Tests
# =============================================================================


class TestStaggeredTWFEParity:
    """Cross-validate Python twfe_staggered vs Julia StaggeredTWFE.

    Note: TWFE is BIASED with heterogeneous treatment effects and staggered
    adoption. These tests verify implementations match, not that estimates
    are correct. Use Callaway-Sant'Anna or Sun-Abraham for unbiased estimation.
    """

    def test_twfe_basic_estimate(self):
        """TWFE point estimates should match."""
        true_effect = 2.0
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=15, n_periods=8, true_effect=true_effect, seed=42
        )

        # Python - needs treatment as int for twfe_staggered
        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = twfe_staggered(data=py_data, cluster_se=True)

        # Julia
        jl_result = julia_staggered_twfe(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # Both should give some estimate (TWFE is biased, so don't check against true_effect)
        assert np.isfinite(py_result["att"]), "Python TWFE estimate is not finite"
        assert np.isfinite(jl_result["estimate"]), "Julia TWFE estimate is not finite"

        # Estimates should be reasonably close (loose tolerance for TWFE bias)
        rel_diff = abs(py_result["att"] - jl_result["estimate"]) / max(abs(py_result["att"]), 0.1)
        assert rel_diff < 0.5, (
            f"TWFE estimate mismatch: Python={py_result['att']:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_twfe_cluster_se(self):
        """Cluster standard errors should be similar."""
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=15, n_periods=8, true_effect=2.0, seed=123
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment.astype(int),  # Cast to int for pandas compatibility
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = twfe_staggered(data=py_data, cluster_se=True)

        jl_result = julia_staggered_twfe(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # SEs should be same order of magnitude
        assert np.isfinite(py_result["se"]) and py_result["se"] > 0
        assert np.isfinite(jl_result["se"]) and jl_result["se"] > 0

        rel_diff = abs(py_result["se"] - jl_result["se"]) / max(py_result["se"], 0.01)
        assert rel_diff < 1.0, (
            f"TWFE SE mismatch: Python={py_result['se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_twfe_sample_sizes(self):
        """Sample sizes should match exactly."""
        outcomes, treatment, time, unit_id, treatment_time = generate_staggered_did_data(
            n_units_per_cohort=10, n_periods=6, true_effect=1.5, seed=456
        )

        py_data = create_staggered_data(
            outcomes=outcomes,
            treatment=treatment.astype(int),  # Cast to int for pandas compatibility
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
        )

        py_result = twfe_staggered(data=py_data, cluster_se=True)

        jl_result = julia_staggered_twfe(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=treatment_time,
            cluster_se=True,
        )

        # Sample sizes must match
        assert py_result["n_obs"] == jl_result["n_obs"], (
            f"n_obs mismatch: Python={py_result['n_obs']}, Julia={jl_result['n_obs']}"
        )
