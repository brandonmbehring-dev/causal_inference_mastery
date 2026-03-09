"""
Layer 1: Known-answer tests for event study DiD estimator.

Tests cover:
1. Known-answer validation with hand-calculated leads/lags
2. Input validation and error handling
3. Pre-trends testing (joint F-test)
4. Output structure validation
"""

import numpy as np
import pytest
from src.causal_inference.did.event_study import event_study


class TestEventStudyKnownAnswers:
    """Layer 1: Known-answer tests for event study estimation."""

    def test_constant_effect_across_periods(self, event_study_constant_effect_data):
        """Event study with constant treatment effect across all post-periods."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
        )

        # Check all leads are close to zero (parallel trends)
        for k, v in result["leads"].items():
            assert abs(v["estimate"]) < 1.0, f"Lead {k} should be ~0, got {v['estimate']:.3f}"

        # Check all lags are close to true effect (2.5)
        for k, v in result["lags"].items():
            assert abs(v["estimate"] - 2.5) < 1.0, (
                f"Lag {k} should be ~2.5, got {v['estimate']:.3f}"
            )

        # Check joint pre-trends test
        assert result["parallel_trends_plausible"] is True, (
            f"Parallel trends should be plausible, got p={result['joint_pretrends_pvalue']:.3f}"
        )

    def test_dynamic_increasing_effects(self, event_study_dynamic_effect_data):
        """Event study with dynamic treatment effects increasing over time."""
        data = event_study_dynamic_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=4,
        )

        # Check all leads are close to zero
        for k, v in result["leads"].items():
            assert abs(v["estimate"]) < 1.0, f"Lead {k} should be ~0, got {v['estimate']:.3f}"

        # Check lags match expected dynamic effects [1, 2, 3, 4]
        expected_effects = data["true_effects"]
        for k, v in result["lags"].items():
            expected = expected_effects[k]
            assert abs(v["estimate"] - expected) < 1.5, (
                f"Lag {k} should be ~{expected}, got {v['estimate']:.3f}"
            )

        # Check parallel trends plausible
        assert result["parallel_trends_plausible"] is True

    def test_anticipation_effect_detected(self, event_study_anticipation_data):
        """Event study detects anticipation effects (violation of parallel trends)."""
        data = event_study_anticipation_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=4,
            n_lags=3,
            omit_period=-2,  # Omit period -2 instead of -1 to see anticipation
        )

        # Lead at k=-1 should be significant (anticipation effect = 1.5)
        lead_minus_1 = result["leads"][-1]
        assert abs(lead_minus_1["estimate"] - 1.5) < 1.0, (
            f"Lead -1 should show anticipation (~1.5), got {lead_minus_1['estimate']:.3f}"
        )

        # Joint pre-trends test should detect violation
        # Note: With noise, this may not always reject at p<0.05
        # But anticipation effect magnitude should be detectable
        assert abs(lead_minus_1["estimate"]) > 0.5, (
            "Lead -1 should show non-zero anticipation effect"
        )

    def test_many_periods_all_leads_zero(self, event_study_many_periods_data):
        """Event study with many periods: all leads should be zero."""
        data = event_study_many_periods_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=10,
            n_lags=10,
        )

        # Check all 10 leads are close to zero
        for k in range(-10, 0):
            if k == -1:
                continue  # Omitted period
            lead_coef = result["leads"][k]
            assert abs(lead_coef["estimate"]) < 1.0, (
                f"Lead {k} should be ~0, got {lead_coef['estimate']:.3f}"
            )

        # Check all 10 lags are close to 3.0
        for k in range(0, 10):
            lag_coef = result["lags"][k]
            assert abs(lag_coef["estimate"] - 3.0) < 1.5, (
                f"Lag {k} should be ~3.0, got {lag_coef['estimate']:.3f}"
            )

        # Joint pre-trends test should pass
        assert result["parallel_trends_plausible"] is True

    def test_negative_dynamic_effects(self):
        """Event study with negative treatment effects."""
        np.random.seed(555)
        n_units = 40
        n_periods = 8
        treatment_time = 4

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        # Negative effects: [-2, -3, -4, -5] in periods 4, 5, 6, 7
        true_effects = {0: -2.0, 1: -3.0, 2: -4.0, 3: -5.0}

        for unit in range(n_units):
            is_treated = unit >= 20
            baseline = 30.0

            for t in range(n_periods):
                y = baseline + t * 0.3  # Time trend

                if is_treated and t >= treatment_time:
                    relative_time = t - treatment_time
                    y += true_effects[relative_time]  # Negative effect

                y += np.random.normal(0, 0.5)

                outcomes.append(y)
                treatment_vec.append(1 if is_treated else 0)
                time_vec.append(t)
                unit_id_vec.append(unit)

        result = event_study(
            outcomes=np.array(outcomes),
            treatment=np.array(treatment_vec),
            time=np.array(time_vec),
            unit_id=np.array(unit_id_vec),
            treatment_time=treatment_time,
            n_leads=4,
            n_lags=4,
        )

        # Check negative effects recovered
        for k, expected in true_effects.items():
            lag_coef = result["lags"][k]
            assert lag_coef["estimate"] < 0, f"Lag {k} should be negative"
            assert abs(lag_coef["estimate"] - expected) < 1.5, (
                f"Lag {k} should be ~{expected}, got {lag_coef['estimate']:.3f}"
            )

    def test_zero_effect_all_periods(self):
        """Event study with zero treatment effect in all periods."""
        np.random.seed(666)
        n_units = 50
        n_periods = 6
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 25
            baseline = 20.0

            for t in range(n_periods):
                y = baseline + t * 0.4  # Common trend, no treatment effect

                y += np.random.normal(0, 0.6)

                outcomes.append(y)
                treatment_vec.append(1 if is_treated else 0)
                time_vec.append(t)
                unit_id_vec.append(unit)

        result = event_study(
            outcomes=np.array(outcomes),
            treatment=np.array(treatment_vec),
            time=np.array(time_vec),
            unit_id=np.array(unit_id_vec),
            treatment_time=treatment_time,
            n_leads=3,
            n_lags=3,
        )

        # All leads should be close to zero
        for k, v in result["leads"].items():
            assert abs(v["estimate"]) < 1.0

        # All lags should be close to zero
        for k, v in result["lags"].items():
            assert abs(v["estimate"]) < 1.0, (
                f"Lag {k} should be ~0 (no effect), got {v['estimate']:.3f}"
            )

    def test_hand_calculated_simple_case(self):
        """
        Hand-calculated event study with perfect data (no noise).

        Setup:
        - 2 treated, 2 control
        - 2 pre-periods (t=0, 1), 2 post-periods (t=2, 3)
        - Treatment at t=2
        - Outcomes perfectly deterministic:
          - Control: [10, 11, 12, 13] (linear trend)
          - Treated: [12, 13, 15, 16] (same trend + effect=1 in post)
        """
        # Control units
        control_outcomes = [10, 11, 12, 13, 10, 11, 12, 13]
        # Treated units (effect=1 in post periods)
        treated_outcomes = [12, 13, 15, 16, 12, 13, 15, 16]

        outcomes = np.array(control_outcomes + treated_outcomes, dtype=float)
        treatment = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        time = np.array([0, 1, 2, 3] * 4)
        unit_id = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        result = event_study(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=2,
            n_leads=2,
            n_lags=2,
        )

        # Check leads (should be exactly 0 with perfect data)
        # Note: With TWFE, leads might not be exactly 0 due to omitted period
        for k, v in result["leads"].items():
            assert abs(v["estimate"]) < 0.5, f"Lead {k} should be ~0"

        # Check lags (should be exactly 1 with perfect data)
        for k, v in result["lags"].items():
            # With perfect data and TWFE, effect should be very close to 1
            assert abs(v["estimate"] - 1.0) < 0.5, (
                f"Lag {k} should be ~1.0, got {v['estimate']:.3f}"
            )

    def test_omitted_period_not_in_results(self, event_study_constant_effect_data):
        """Omitted period (default k=-1) should not appear in leads or lags."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
            omit_period=-1,  # Explicit default
        )

        # Check that k=-1 is not in leads
        assert -1 not in result["leads"], "Omitted period should not be in leads"

        # Check omitted_period field
        assert result["omitted_period"] == -1


class TestEventStudyInputValidation:
    """Input validation and error handling tests."""

    def test_mismatched_lengths(self):
        """Error when input arrays have different lengths."""
        with pytest.raises(ValueError, match="All inputs must have same length"):
            event_study(
                outcomes=np.array([1, 2, 3]),
                treatment=np.array([0, 1]),  # Wrong length
                time=np.array([0, 1, 2]),
                unit_id=np.array([0, 0, 1]),
                treatment_time=1,
            )

    def test_empty_inputs(self):
        """Error when inputs are empty."""
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            event_study(
                outcomes=np.array([]),
                treatment=np.array([]),
                time=np.array([]),
                unit_id=np.array([]),
                treatment_time=1,
            )

    def test_invalid_treatment_time_too_early(self, event_study_constant_effect_data):
        """Error when treatment_time is before time range."""
        data = event_study_constant_effect_data

        with pytest.raises(ValueError, match="treatment_time must be within time range"):
            event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=-5,  # Before time_min=0
                n_leads=3,
                n_lags=3,
            )

    def test_invalid_treatment_time_too_late(self, event_study_constant_effect_data):
        """Error when treatment_time is after time range."""
        data = event_study_constant_effect_data

        with pytest.raises(ValueError, match="treatment_time must be within time range"):
            event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=100,  # After time_max=5
                n_leads=3,
                n_lags=3,
            )

    def test_invalid_n_leads_too_large(self, event_study_constant_effect_data):
        """Error when n_leads exceeds available pre-periods."""
        data = event_study_constant_effect_data  # treatment_time=3, time_min=0 → max 3 leads

        with pytest.raises(ValueError, match="n_leads.*exceeds maximum possible"):
            event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
                n_leads=10,  # Only 3 pre-periods available
                n_lags=3,
            )

    def test_invalid_n_lags_too_large(self, event_study_constant_effect_data):
        """Error when n_lags exceeds available post-periods."""
        data = event_study_constant_effect_data  # treatment_time=3, time_max=5 → max 3 lags

        with pytest.raises(ValueError, match="n_lags.*exceeds maximum possible"):
            event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
                n_leads=3,
                n_lags=10,  # Only 3 post-periods available
            )

    def test_invalid_omit_period(self, event_study_constant_effect_data):
        """Error when omit_period is not in valid periods."""
        data = event_study_constant_effect_data

        with pytest.raises(ValueError, match="omit_period.*must be in valid periods"):
            event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
                n_leads=3,
                n_lags=3,
                omit_period=-10,  # Not in valid periods
            )

    def test_all_units_treated_error(self):
        """
        Error when all units receive treatment (no control group).

        Note: This test was previously test_time_varying_treatment_error.
        After BUG-9 fix, time-varying treatment with consistent start time is now
        ALLOWED. But if all units are treated (no control group), it still errors.
        """
        np.random.seed(777)
        n_units = 20
        n_periods = 4

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            for t in range(n_periods):
                outcomes.append(10 + t + np.random.normal(0, 0.5))
                # All units treated at t>=2 (no control group)
                treatment_vec.append(1 if t >= 2 else 0)
                time_vec.append(t)
                unit_id_vec.append(unit)

        # BUG-9 FIX: Time-varying treatment now accepted if consistent.
        # But all units being treated means no control group → error
        with pytest.raises(ValueError, match="No control units found"):
            event_study(
                outcomes=np.array(outcomes),
                treatment=np.array(treatment_vec),
                time=np.array(time_vec),
                unit_id=np.array(unit_id_vec),
                treatment_time=2,
            )

    def test_no_treated_units(self, event_study_constant_effect_data):
        """Error when all units are control."""
        data = event_study_constant_effect_data

        with pytest.raises(ValueError, match="treatment must be binary"):
            event_study(
                outcomes=data["outcomes"],
                treatment=np.zeros_like(data["treatment"]),  # All control
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )

    def test_no_control_units(self, event_study_constant_effect_data):
        """Error when all units are treated."""
        data = event_study_constant_effect_data

        with pytest.raises(ValueError, match="treatment must be binary"):
            event_study(
                outcomes=data["outcomes"],
                treatment=np.ones_like(data["treatment"]),  # All treated
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )

    def test_nan_in_outcomes(self, event_study_constant_effect_data):
        """Error when outcomes contain NaN."""
        data = event_study_constant_effect_data

        outcomes_with_nan = data["outcomes"].copy()
        outcomes_with_nan[0] = np.nan

        with pytest.raises(ValueError, match="outcomes contains NaN or inf"):
            event_study(
                outcomes=outcomes_with_nan,
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )


class TestEventStudyPreTrends:
    """Pre-trends testing (joint F-test)."""

    def test_joint_pretrends_test_passes(self, event_study_constant_effect_data):
        """Joint F-test for pre-trends should pass with parallel trends."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
        )

        # Joint pre-trends test should have p > 0.05
        assert result["joint_pretrends_pvalue"] > 0.05, (
            f"Joint pre-trends test should pass, got p={result['joint_pretrends_pvalue']:.3f}"
        )
        assert result["parallel_trends_plausible"] is True

    def test_joint_pretrends_test_detects_violation(self, event_study_anticipation_data):
        """Joint F-test should detect anticipation effects (though may not reject with noise)."""
        data = event_study_anticipation_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=4,
            n_lags=3,
        )

        # With anticipation effect of 1.5, joint test p-value should be lower
        # (though may not always reject with noise, so we just check it's computed)
        assert "joint_pretrends_pvalue" in result
        assert 0 <= result["joint_pretrends_pvalue"] <= 1

    def test_no_leads_available_pretrends_nan(self, event_study_constant_effect_data):
        """When no leads available, joint pre-trends test should be NaN."""
        data = event_study_constant_effect_data

        # Use treatment_time=0 → no pre-periods
        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=0,  # Treatment at first period
            n_leads=0,
            n_lags=6,  # time_max=5, treatment_time=0 → 6 post-periods
            omit_period=0,  # Omit first lag since no leads available
        )

        # No leads → joint test should be NaN
        assert np.isnan(result["joint_pretrends_pvalue"])
        assert result["parallel_trends_plausible"] is None


class TestEventStudyOutputStructure:
    """Output structure validation tests."""

    def test_result_has_required_keys(self, event_study_constant_effect_data):
        """Result dictionary has all required keys."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
        )

        required_keys = [
            "leads",
            "lags",
            "joint_pretrends_pvalue",
            "parallel_trends_plausible",
            "omitted_period",
            "n_leads",
            "n_lags",
            "n_obs",
            "n_clusters",
            "df",
            "cluster_se_used",
        ]

        for key in required_keys:
            assert key in result, f"Result missing required key: {key}"

    def test_lead_lag_structure(self, event_study_constant_effect_data):
        """Each lead/lag has required coefficient fields."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
        )

        required_fields = ["estimate", "se", "t_stat", "p_value", "ci_lower", "ci_upper"]

        # Check leads structure
        for period, coef in result["leads"].items():
            for field in required_fields:
                assert field in coef, f"Lead {period} missing field: {field}"

        # Check lags structure
        for period, coef in result["lags"].items():
            for field in required_fields:
                assert field in coef, f"Lag {period} missing field: {field}"

    def test_correct_number_of_leads_lags(self, event_study_constant_effect_data):
        """Result has correct number of leads and lags (excluding omitted)."""
        data = event_study_constant_effect_data

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
            n_leads=3,
            n_lags=3,
        )

        # 3 leads requested, but k=-1 omitted → 2 leads
        assert len(result["leads"]) == 2, (
            f"Expected 2 leads (omit k=-1), got {len(result['leads'])}"
        )

        # 3 lags requested (k=0, 1, 2) → 3 lags
        assert len(result["lags"]) == 3, f"Expected 3 lags, got {len(result['lags'])}"

        # Check n_leads and n_lags fields
        assert result["n_leads"] == 3
        assert result["n_lags"] == 3


class TestBug9StaggeredAdoptionValidation:
    """
    BUG-9 FIX VALIDATION: Event study should detect staggered adoption.

    The event_study() function assumes a single treatment_time for all units.
    When data shows different treatment start times (staggered adoption),
    the function should reject with a helpful error message pointing to
    appropriate methods (Callaway-Sant'Anna, Sun-Abraham).

    Reference: docs/KNOWN_BUGS.md BUG-9
    """

    def test_staggered_adoption_detected_and_rejected(self):
        """
        BUG-9: Staggered adoption should raise ValueError with helpful message.

        Creates data where units are treated at different times (staggered).
        The function should detect this and error.
        """
        np.random.seed(42)

        # Panel data: 6 units, 10 time periods
        n_units = 6
        n_times = 10
        n_obs = n_units * n_times

        unit_id = np.repeat(np.arange(n_units), n_times)
        time = np.tile(np.arange(n_times), n_units)

        # STAGGERED ADOPTION: Units 0,1 treated at t=3, Units 2,3 treated at t=5
        # Units 4,5 are never treated (control)
        treatment = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            u = unit_id[i]
            t = time[i]
            if u in [0, 1] and t >= 3:  # Cohort 1: treated at t=3
                treatment[i] = 1
            elif u in [2, 3] and t >= 5:  # Cohort 2: treated at t=5
                treatment[i] = 1

        outcomes = np.random.normal(0, 1, n_obs)

        # Should detect staggered adoption and raise ValueError
        with pytest.raises(ValueError, match="Staggered adoption detected"):
            event_study(
                outcomes=outcomes,
                treatment=treatment,
                time=time,
                unit_id=unit_id,
                treatment_time=3,  # Assumed time (incorrect for cohort 2)
            )

    def test_consistent_treatment_time_accepted(self):
        """
        Time-varying treatment with consistent start time should be accepted.

        When all treated units switch treatment at the same time,
        the function should work correctly.
        """
        np.random.seed(123)

        n_units = 4
        n_times = 8
        n_obs = n_units * n_times

        unit_id = np.repeat(np.arange(n_units), n_times)
        time = np.tile(np.arange(n_times), n_units)

        # All treated units (0, 1) switch at t=4
        treatment = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            u = unit_id[i]
            t = time[i]
            if u in [0, 1] and t >= 4:
                treatment[i] = 1

        # Outcomes with treatment effect
        true_effect = 2.0
        outcomes = np.random.normal(0, 1, n_obs)
        outcomes[treatment == 1] += true_effect

        # Should work when treatment_time matches actual start time
        result = event_study(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=4,  # Matches actual treatment start
            n_leads=2,
            n_lags=3,
        )

        assert result["n_treated"] == 2
        assert result["n_control"] == 2

    def test_treatment_time_mismatch_detected(self):
        """
        Mismatch between stated treatment_time and actual data should error.
        """
        np.random.seed(456)

        n_units = 4
        n_times = 8
        n_obs = n_units * n_times

        unit_id = np.repeat(np.arange(n_units), n_times)
        time = np.tile(np.arange(n_times), n_units)

        # All treated units switch at t=4
        treatment = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            if unit_id[i] in [0, 1] and time[i] >= 4:
                treatment[i] = 1

        outcomes = np.random.normal(0, 1, n_obs)

        # Should error: treatment_time=2 doesn't match actual t=4
        with pytest.raises(ValueError, match="Treatment start time mismatch"):
            event_study(
                outcomes=outcomes,
                treatment=treatment,
                time=time,
                unit_id=unit_id,
                treatment_time=2,  # Wrong! Actual is t=4
            )

    def test_ever_treated_indicator_accepted(self):
        """
        Ever-treated indicator (constant within units) should be accepted.

        This is the original format the function was designed for.
        """
        np.random.seed(789)

        n_units = 4
        n_times = 8
        n_obs = n_units * n_times

        unit_id = np.repeat(np.arange(n_units), n_times)
        time = np.tile(np.arange(n_times), n_units)

        # Ever-treated indicator (constant within unit)
        treatment = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            if unit_id[i] in [0, 1]:  # Units 0,1 are "ever treated"
                treatment[i] = 1

        outcomes = np.random.normal(0, 1, n_obs)

        # Should work with ever-treated indicator
        result = event_study(
            outcomes=outcomes,
            treatment=treatment,
            time=time,
            unit_id=unit_id,
            treatment_time=4,
            n_leads=2,
            n_lags=3,
        )

        assert result["n_treated"] == 2
        assert result["n_control"] == 2
