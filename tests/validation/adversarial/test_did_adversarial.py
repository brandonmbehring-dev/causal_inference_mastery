"""Layer 2: Adversarial tests for DiD estimator edge cases."""

import numpy as np
import pytest

from src.causal_inference.did.did_estimator import did_2x2, check_parallel_trends


class TestDiDMinimumSamples:
    """Test DiD with very small sample sizes."""

    def test_minimum_viable_sample(self):
        """DiD with n=5 per group (minimum for cluster SE)."""
        np.random.seed(100)
        n_treated = 5
        n_control = 5

        # Generate data with known effect
        control_pre = np.array([10.0, 11.0, 9.0, 10.5, 10.2])
        control_post = np.array([13.0, 14.0, 12.0, 13.5, 13.2])
        treated_pre = np.array([12.0, 13.0, 11.0, 12.5, 12.2])
        treated_post = np.array([18.0, 19.0, 17.0, 18.5, 18.2])

        outcomes = np.concatenate([control_pre, control_post, treated_pre, treated_post])
        treatment = np.concatenate([np.zeros(10), np.ones(10)])
        post = np.concatenate([np.zeros(5), np.ones(5), np.zeros(5), np.ones(5)])
        unit_id = np.concatenate([
            np.arange(5), np.arange(5),
            np.arange(5, 10), np.arange(5, 10)
        ])

        # Should run but with warning about small clusters
        with pytest.warns(RuntimeWarning, match="Small number of clusters"):
            result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        assert "estimate" in result
        assert result["n_clusters"] == 10

    def test_extreme_small_sample(self):
        """DiD with n=2 per group (extreme minimum)."""
        # 2 control, 2 treated
        outcomes = np.array([10.0, 15.0, 11.0, 16.0,  # Control
                             12.0, 20.0, 13.0, 21.0])  # Treated
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        with pytest.warns(RuntimeWarning, match="Small number of clusters"):
            result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD = (20.5 - 12.5) - (15.5 - 10.5) = 8 - 5 = 3
        assert abs(result["estimate"] - 3.0) < 0.5
        assert result["n_clusters"] == 4
        assert result["df"] == 3  # n_clusters - 1


class TestDiDExtremeImbalance:
    """Test DiD with severe treatment/control imbalance."""

    def test_ninety_ten_split(self):
        """90% treated, 10% control."""
        np.random.seed(200)
        n_treated = 90
        n_control = 10

        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 1.0, n_control)
        control_post = np.full(n_control, 13.0) + np.random.normal(0, 1.0, n_control)
        treated_pre = np.full(n_treated, 12.0) + np.random.normal(0, 1.0, n_treated)
        treated_post = np.full(n_treated, 17.0) + np.random.normal(0, 1.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # Should still work despite imbalance
        assert result["n_treated"] == 90
        assert result["n_control"] == 10
        assert abs(result["estimate"] - 2.0) < 1.0  # True effect ≈ 2

    def test_ten_ninety_split(self):
        """10% treated, 90% control (opposite imbalance)."""
        np.random.seed(201)
        n_treated = 10
        n_control = 90

        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 1.0, n_control)
        control_post = np.full(n_control, 13.0) + np.random.normal(0, 1.0, n_control)
        treated_pre = np.full(n_treated, 12.0) + np.random.normal(0, 1.0, n_treated)
        treated_post = np.full(n_treated, 17.0) + np.random.normal(0, 1.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        assert result["n_treated"] == 10
        assert result["n_control"] == 90
        assert abs(result["estimate"] - 2.0) < 1.0


class TestDiDHighVariance:
    """Test DiD with extreme outcome variance."""

    def test_high_variance_outcomes(self):
        """Outcomes with very high variance (σ=50)."""
        np.random.seed(300)
        n_treated = 50
        n_control = 50

        # High variance noise
        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 50.0, n_control)
        control_post = np.full(n_control, 13.0) + np.random.normal(0, 50.0, n_control)
        treated_pre = np.full(n_treated, 12.0) + np.random.normal(0, 50.0, n_treated)
        treated_post = np.full(n_treated, 17.0) + np.random.normal(0, 50.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # Should run but with wide CIs
        assert "estimate" in result
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 10.0, "CI should be wide with high variance"

    def test_zero_variance_outcomes(self):
        """Outcomes with zero variance (constant within groups)."""
        # Perfect deterministic outcomes
        outcomes = np.array([10, 10, 15, 15, 12, 12, 20, 20])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        unit_id = np.array([0, 1, 0, 1, 2, 3, 2, 3])

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD = (20 - 12) - (15 - 10) = 8 - 5 = 3
        assert abs(result["estimate"] - 3.0) < 1e-10
        # SE should be very small (but not zero due to cluster adjustment)
        assert result["se"] < 1.0


class TestDiDManyPeriods:
    """Test DiD with many time periods."""

    def test_twenty_periods(self):
        """DiD with 10 pre-periods and 10 post-periods."""
        np.random.seed(400)
        n_treated = 30
        n_control = 30
        n_units = n_treated + n_control
        n_periods = 20  # t = 0 to 19, treatment at t=10

        outcomes = []
        treatment_vec = []
        post_vec = []
        unit_id_vec = []

        for unit in range(n_units):
            is_treated = unit >= n_control
            baseline = 12.0 if is_treated else 10.0

            for t in range(n_periods):
                is_post = t >= 10

                # Common time trend
                y = baseline + 0.2 * t

                # Treatment effect in post-period
                if is_treated and is_post:
                    y += 3.0

                y += np.random.normal(0, 0.5)

                outcomes.append(y)
                treatment_vec.append(1 if is_treated else 0)
                post_vec.append(1 if is_post else 0)
                unit_id_vec.append(unit)

        result = did_2x2(
            outcomes=np.array(outcomes),
            treatment=np.array(treatment_vec),
            post=np.array(post_vec),
            unit_id=np.array(unit_id_vec),
            cluster_se=True,
        )

        # Should recover treatment effect (3.0)
        assert abs(result["estimate"] - 3.0) < 0.5
        assert result["n_obs"] == 60 * 20  # 60 units × 20 periods


class TestDiDUnbalancedPanel:
    """Test DiD with unbalanced panels (missing observations)."""

    def test_subset_of_units(self):
        """DiD with only a subset of available units."""
        np.random.seed(500)
        n_treated = 40
        n_control = 40

        # Start with full panel
        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 0.5, n_control)
        control_post = np.full(n_control, 13.0) + np.random.normal(0, 0.5, n_control)
        treated_pre = np.full(n_treated, 12.0) + np.random.normal(0, 0.5, n_treated)
        treated_post = np.full(n_treated, 17.0) + np.random.normal(0, 0.5, n_treated)

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

        # Use only first 30 control and 30 treated units (drop 10 of each)
        # Keep units 0-29 (control) and 40-69 (treated)
        keep_units = list(range(30)) + list(range(40, 70))
        keep_mask = np.isin(unit_id, keep_units)

        result = did_2x2(
            outcomes=outcomes[keep_mask],
            treatment=treatment[keep_mask],
            post=post[keep_mask],
            unit_id=unit_id[keep_mask],
            cluster_se=True,
        )

        # Should work with 60 units instead of 80
        assert "estimate" in result
        assert result["n_clusters"] == 60
        assert result["n_treated"] == 30
        assert result["n_control"] == 30


class TestDiDPerfectSeparation:
    """Test DiD with perfect separation or extreme baseline differences."""

    def test_extreme_baseline_difference(self):
        """Treated and control groups have very different baselines."""
        np.random.seed(600)
        n_treated = 40
        n_control = 40

        # Control group: outcomes around 10
        # Treated group: outcomes around 1000 (100x difference)
        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 1.0, n_control)
        control_post = np.full(n_control, 13.0) + np.random.normal(0, 1.0, n_control)
        treated_pre = np.full(n_treated, 1000.0) + np.random.normal(0, 10.0, n_treated)
        treated_post = np.full(n_treated, 1005.0) + np.random.normal(0, 10.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD should still work (differences out baseline differences)
        # Expected: (1005 - 1000) - (13 - 10) = 5 - 3 = 2
        # With noise, tolerance needs to be wider
        assert abs(result["estimate"] - 2.0) < 2.0, f"Expected ~2.0, got {result['estimate']}"


class TestDiDNegativeOutcomes:
    """Test DiD with negative or extreme outcome values."""

    def test_negative_outcomes(self):
        """All outcomes are negative."""
        np.random.seed(700)
        n_treated = 30
        n_control = 30

        control_pre = np.full(n_control, -100.0) + np.random.normal(0, 5.0, n_control)
        control_post = np.full(n_control, -97.0) + np.random.normal(0, 5.0, n_control)
        treated_pre = np.full(n_treated, -98.0) + np.random.normal(0, 5.0, n_treated)
        treated_post = np.full(n_treated, -90.0) + np.random.normal(0, 5.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD = (-90 - -98) - (-97 - -100) = 8 - 3 = 5
        # With noise (σ=5), tolerance needs to be wider
        assert abs(result["estimate"] - 5.0) < 3.0, f"Expected ~5.0, got {result['estimate']}"

    def test_mixed_sign_outcomes(self):
        """Outcomes cross zero."""
        outcomes = np.array([-5, 5, -3, 7, -2, 8, 0, 12])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD = (10 - -1) - (6 - -4) = 11 - 10 = 1
        assert "estimate" in result


class TestDiDParallelTrendsAdversarial:
    """Adversarial tests for parallel trends checking."""

    def test_extreme_differential_pre_trends(self):
        """Very strong differential pre-trends."""
        np.random.seed(800)
        n_treated = 40
        n_control = 40
        n_units = n_treated + n_control

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        # Treated group: steep upward trend pre-treatment
        # Control group: flat
        for unit in range(n_units):
            is_treated = unit >= n_control
            baseline = 10.0

            for t in range(-5, 5):  # 5 pre, 5 post
                is_post = t >= 0

                if is_treated and not is_post:
                    y = baseline + 5 * t  # Strong upward trend
                elif not is_post:
                    y = baseline  # Flat
                else:
                    y = baseline + 10  # Post-period level

                y += np.random.normal(0, 1.0)

                outcomes.append(y)
                treatment_vec.append(1 if is_treated else 0)
                time_vec.append(t)
                unit_id_vec.append(unit)

        result = check_parallel_trends(
            outcomes=np.array(outcomes),
            treatment=np.array(treatment_vec),
            time=np.array(time_vec),
            unit_id=np.array(unit_id_vec),
            treatment_time=0,
        )

        # Should strongly reject parallel trends
        assert result["p_value"] < 0.001, "Should strongly reject with extreme differential trends"
        assert result["parallel_trends_plausible"] is False

    def test_minimal_pre_periods(self):
        """Parallel trends test with exactly 2 pre-periods (minimum)."""
        np.random.seed(801)
        n_treated = 30
        n_control = 30

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        for unit in range(60):
            is_treated = unit >= n_control
            baseline = 12.0 if is_treated else 10.0

            for t in [-1, 0, 1, 2]:  # 2 pre, 2 post
                is_post = t >= 1
                y = baseline + 0.5 * t

                if is_treated and is_post:
                    y += 3.0

                y += np.random.normal(0, 0.5)

                outcomes.append(y)
                treatment_vec.append(1 if is_treated else 0)
                time_vec.append(t)
                unit_id_vec.append(unit)

        result = check_parallel_trends(
            outcomes=np.array(outcomes),
            treatment=np.array(treatment_vec),
            time=np.array(time_vec),
            unit_id=np.array(unit_id_vec),
            treatment_time=1,
        )

        # Should work but may have warning about limited power
        assert result["n_pre_periods"] == 2
        assert result["warning"] is not None  # Should warn about limited periods


class TestDiDCollinearity:
    """Test DiD with collinear covariates (not used in estimator but good to test)."""

    def test_identical_pre_post_means(self):
        """Pre and post have same mean (no time effect)."""
        np.random.seed(900)
        n_treated = 40
        n_control = 40

        # Same mean in pre and post (no time trend)
        control_pre = np.full(n_control, 10.0) + np.random.normal(0, 2.0, n_control)
        control_post = np.full(n_control, 10.0) + np.random.normal(0, 2.0, n_control)
        treated_pre = np.full(n_treated, 10.0) + np.random.normal(0, 2.0, n_treated)
        treated_post = np.full(n_treated, 15.0) + np.random.normal(0, 2.0, n_treated)

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

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # DiD = (15 - 10) - (10 - 10) = 5 - 0 = 5
        # With noise (σ=2), tolerance needs to be wider
        assert abs(result["estimate"] - 5.0) < 2.0, f"Expected ~5.0, got {result['estimate']}"
