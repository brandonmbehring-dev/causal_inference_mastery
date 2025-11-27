"""
Layer 2: Adversarial tests for event study DiD estimator.

Stress-tests edge cases:
- Minimum/maximum periods
- Extreme imbalance
- High/zero variance
- Perfect separation
- Unbalanced panels
- Negative outcomes
"""

import numpy as np
import pytest
from src.causal_inference.did.event_study import event_study


class TestEventStudyMinimumPeriods:
    """Test event study with minimum number of periods."""

    def test_minimum_viable_periods(self):
        """Event study with 1 pre-period, 1 post-period (minimum)."""
        np.random.seed(111)
        n_units = 30
        n_periods = 2
        treatment_time = 1

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 15
            baseline = 20.0

            for t in range(n_periods):
                y = baseline + t * 0.5
                if is_treated and t >= treatment_time:
                    y += 2.0  # Treatment effect

                y += np.random.normal(0, 1.0)

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
            n_leads=1,
            n_lags=1,
            omit_period=0,  # Omit lag 0 since only 1 pre-period
        )

        # Should run without error
        assert "leads" in result
        assert "lags" in result
        # Only 1 pre-period, so no leads (k=-1 doesn't exist since omit_period=0)
        # Wait, if n_leads=1 and omit_period=0, leads should include k=-1
        assert len(result["leads"]) == 1  # k=-1
        assert len(result["lags"]) == 0  # k=0 omitted

    def test_two_pre_two_post(self):
        """Event study with 2 pre-periods, 2 post-periods."""
        np.random.seed(222)
        n_units = 40
        n_periods = 4
        treatment_time = 2

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 20
            baseline = 15.0

            for t in range(n_periods):
                y = baseline + t * 0.3
                if is_treated and t >= treatment_time:
                    y += 1.5

                y += np.random.normal(0, 0.8)

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
            n_leads=2,
            n_lags=2,
        )

        # Should run and estimate effect
        assert len(result["leads"]) == 1  # k=-2 (k=-1 omitted)
        assert len(result["lags"]) == 2  # k=0, 1


class TestEventStudyMaximumPeriods:
    """Test event study with many periods."""

    def test_many_leads_and_lags(self):
        """Event study with 15 pre-periods, 15 post-periods."""
        np.random.seed(333)
        n_units = 60
        n_periods = 30
        treatment_time = 15

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 30
            baseline = 25.0

            for t in range(n_periods):
                y = baseline + t * 0.15
                if is_treated and t >= treatment_time:
                    y += 2.5

                y += np.random.normal(0, 1.2)

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
            n_leads=15,
            n_lags=15,
        )

        # Should run with many periods
        assert len(result["leads"]) == 14  # 15 leads minus omitted k=-1
        assert len(result["lags"]) == 15  # All 15 lags


class TestEventStudyExtremeImbalance:
    """Test event study with extreme treatment group imbalance."""

    def test_90_percent_treated(self):
        """Event study with 90% treated, 10% control."""
        np.random.seed(444)
        n_units = 50
        n_periods = 6
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        n_control = 5
        n_treated = 45

        for unit in range(n_units):
            is_treated = unit >= n_control
            baseline = 18.0

            for t in range(n_periods):
                y = baseline + t * 0.4
                if is_treated and t >= treatment_time:
                    y += 1.8

                y += np.random.normal(0, 1.0)

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

        # Should run despite imbalance (wider CIs expected)
        assert result["n_treated"] == n_treated
        assert result["n_control"] == n_control
        assert result["n_obs"] == n_units * n_periods

    def test_10_percent_treated(self):
        """Event study with 10% treated, 90% control."""
        np.random.seed(555)
        n_units = 50
        n_periods = 6
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        n_control = 45
        n_treated = 5

        for unit in range(n_units):
            is_treated = unit >= n_control
            baseline = 20.0

            for t in range(n_periods):
                y = baseline + t * 0.35
                if is_treated and t >= treatment_time:
                    y += 2.2

                y += np.random.normal(0, 0.9)

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

        # Should run despite imbalance
        assert result["n_treated"] == n_treated
        assert result["n_control"] == n_control


class TestEventStudyHighVariance:
    """Test event study with high outcome variance."""

    def test_very_high_variance(self):
        """Event study with σ=50 (high noise)."""
        np.random.seed(666)
        n_units = 80
        n_periods = 8
        treatment_time = 4

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 40
            baseline = 100.0

            for t in range(n_periods):
                y = baseline + t * 2.0
                if is_treated and t >= treatment_time:
                    y += 10.0  # Large effect relative to noise

                y += np.random.normal(0, 50.0)  # High variance

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

        # Should run (wide CIs expected)
        assert "estimate" in result["lags"][0]
        # CIs should be wider with high variance
        ci_width = result["lags"][0]["ci_upper"] - result["lags"][0]["ci_lower"]
        assert ci_width > 5.0, "CI should be wide with high variance"

    def test_zero_variance(self):
        """Event study with zero variance (deterministic outcomes)."""
        n_units = 40
        n_periods = 6
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 20
            baseline = 50.0

            for t in range(n_periods):
                y = baseline + t * 1.0
                if is_treated and t >= treatment_time:
                    y += 3.0  # Deterministic effect

                # No noise added (zero variance)

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

        # Should run with deterministic data (perfect fit)
        # Effect should be exactly 3.0
        for k in result["lags"].keys():
            assert abs(result["lags"][k]["estimate"] - 3.0) < 0.01, (
                f"Lag {k} should be exactly 3.0 with zero variance"
            )


class TestEventStudyPerfectSeparation:
    """Test event study with perfect separation (extreme baseline differences)."""

    def test_hundred_fold_baseline_difference(self):
        """Event study with 100x baseline difference between groups."""
        np.random.seed(777)
        n_units = 50
        n_periods = 7
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 25
            baseline = 1000.0 if is_treated else 10.0  # 100x difference

            for t in range(n_periods):
                y = baseline + t * 0.5
                if is_treated and t >= treatment_time:
                    y += 5.0

                y += np.random.normal(0, 2.0)

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
            n_lags=4,
        )

        # TWFE should difference out baseline differences
        # Effect should still be ~5.0
        for k in result["lags"].keys():
            assert abs(result["lags"][k]["estimate"] - 5.0) < 3.0, (
                f"Lag {k} should recover treatment effect despite baseline separation"
            )


class TestEventStudyUnbalancedPanel:
    """Test event study with unbalanced panel (missing observations)."""

    def test_randomly_missing_observations(self):
        """Event study with ~20% missing observations."""
        np.random.seed(888)
        n_units = 60
        n_periods = 8
        treatment_time = 4

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 30
            baseline = 22.0

            for t in range(n_periods):
                # Randomly drop ~20% of observations
                if np.random.rand() < 0.2:
                    continue  # Skip this observation

                y = baseline + t * 0.4
                if is_treated and t >= treatment_time:
                    y += 2.0

                y += np.random.normal(0, 1.0)

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

        # Should handle unbalanced panel
        assert result["n_obs"] < n_units * n_periods, "Panel should be unbalanced"
        assert result["n_obs"] > 0.6 * n_units * n_periods, "Should keep ~80% of observations"


class TestEventStudyNegativeOutcomes:
    """Test event study with negative outcomes."""

    def test_all_negative_outcomes(self):
        """Event study with all negative outcomes."""
        np.random.seed(999)
        n_units = 50
        n_periods = 6
        treatment_time = 3

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 25
            baseline = -50.0

            for t in range(n_periods):
                y = baseline + t * 0.5
                if is_treated and t >= treatment_time:
                    y -= 3.0  # Negative treatment effect on negative outcomes

                y += np.random.normal(0, 2.0)

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

        # Should handle negative outcomes
        # Effect should be ~-3.0
        for k in result["lags"].keys():
            assert result["lags"][k]["estimate"] < 0, f"Lag {k} should be negative"
            assert abs(result["lags"][k]["estimate"] - (-3.0)) < 2.0

    def test_mixed_sign_outcomes(self):
        """Event study with outcomes crossing zero."""
        np.random.seed(1111)
        n_units = 60
        n_periods = 8
        treatment_time = 4

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= 30
            baseline = -10.0

            for t in range(n_periods):
                y = baseline + t * 3.0  # Crosses zero around t=3-4
                if is_treated and t >= treatment_time:
                    y += 5.0

                y += np.random.normal(0, 1.5)

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

        # Should handle mixed signs
        # Effect should be ~5.0
        for k in result["lags"].keys():
            assert abs(result["lags"][k]["estimate"] - 5.0) < 2.5


class TestEventStudyIdenticalEffects:
    """Test event study with all lag coefficients identical (constant effect)."""

    def test_constant_effect_all_periods(self):
        """All post-treatment lags should have same coefficient."""
        np.random.seed(1212)
        n_units = 50
        n_periods = 10
        treatment_time = 5

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        constant_effect = 4.0

        for unit in range(n_units):
            is_treated = unit >= 25
            baseline = 30.0

            for t in range(n_periods):
                y = baseline + t * 0.6
                if is_treated and t >= treatment_time:
                    y += constant_effect  # Same effect in all post-periods

                y += np.random.normal(0, 1.2)

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
            n_leads=5,
            n_lags=5,
        )

        # All lags should have similar coefficients (~4.0)
        lag_estimates = [result["lags"][k]["estimate"] for k in result["lags"].keys()]
        for est in lag_estimates:
            assert abs(est - constant_effect) < 2.0, (
                f"All lags should be ~{constant_effect}, got {est}"
            )
