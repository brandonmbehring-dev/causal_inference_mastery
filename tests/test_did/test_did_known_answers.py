"""Layer 1: Known-answer tests for Difference-in-Differences estimator."""

import numpy as np
import pytest

from src.causal_inference.did.did_estimator import did_2x2, check_parallel_trends


class TestDiD2x2KnownAnswers:
    """Layer 1: Known-answer tests for 2×2 DiD."""

    def test_zero_treatment_effect(self, zero_effect_data):
        """DiD estimate should be zero with no treatment effect."""
        result = did_2x2(
            outcomes=zero_effect_data["outcomes"],
            treatment=zero_effect_data["treatment"],
            post=zero_effect_data["post"],
            unit_id=zero_effect_data["unit_id"],
            cluster_se=True,
        )

        # Should be close to zero
        assert abs(result["estimate"]) < 0.5, f"Expected ~0, got {result['estimate']}"
        assert result["ci_lower"] <= 0.0 <= result["ci_upper"], "CI should contain 0"

        # Check diagnostics
        assert result["n_treated"] == zero_effect_data["n_treated"]
        assert result["n_control"] == zero_effect_data["n_control"]
        assert result["n_clusters"] == zero_effect_data["n_treated"] + zero_effect_data["n_control"]

    def test_known_positive_effect(self, simple_did_data):
        """DiD estimate should match known positive effect."""
        result = did_2x2(
            outcomes=simple_did_data["outcomes"],
            treatment=simple_did_data["treatment"],
            post=simple_did_data["post"],
            unit_id=simple_did_data["unit_id"],
            cluster_se=True,
        )

        # Should be close to true effect (2.0)
        true_did = simple_did_data["true_did"]
        assert abs(result["estimate"] - true_did) < 0.5, f"Expected ~{true_did}, got {result['estimate']}"
        assert result["ci_lower"] <= true_did <= result["ci_upper"], "CI should contain true value"

        # p-value should be significant
        assert result["p_value"] < 0.05, "Should detect non-zero effect"

    def test_common_time_trend_removal(self, simple_did_data):
        """DiD should remove common time trends."""
        # Calculate manual DiD
        outcomes = simple_did_data["outcomes"]
        treatment = simple_did_data["treatment"]
        post = simple_did_data["post"]

        # Control group: change from pre to post
        control_pre = outcomes[(treatment == 0) & (post == 0)]
        control_post = outcomes[(treatment == 0) & (post == 1)]
        control_change = control_post.mean() - control_pre.mean()

        # Treated group: change from pre to post
        treated_pre = outcomes[(treatment == 1) & (post == 0)]
        treated_post = outcomes[(treatment == 1) & (post == 1)]
        treated_change = treated_post.mean() - treated_pre.mean()

        # DiD = difference in changes
        manual_did = treated_change - control_change

        # Estimate via regression
        result = did_2x2(
            outcomes=outcomes,
            treatment=treatment,
            post=post,
            unit_id=simple_did_data["unit_id"],
            cluster_se=True,
        )

        # Should match manual calculation
        assert abs(result["estimate"] - manual_did) < 0.01, "Regression DiD should match manual DiD"

    def test_simple_2x2_hand_calculation(self):
        """Test with simple hand-calculable values."""
        # Setup: 2 control, 2 treated, 1 pre, 1 post
        # Control: pre=[10, 10], post=[15, 15] (change = +5)
        # Treated: pre=[12, 12], post=[20, 20] (change = +8)
        # DiD = 8 - 5 = 3

        outcomes = np.array([10.0, 10.0, 15.0, 15.0,  # Control
                             12.0, 12.0, 20.0, 20.0])  # Treated
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        unit_id = np.array([0, 1, 0, 1, 2, 3, 2, 3])

        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        # Should exactly equal 3.0
        assert abs(result["estimate"] - 3.0) < 1e-10, f"Expected 3.0, got {result['estimate']}"
        assert result["n_clusters"] == 4

    def test_smaller_sample_size(self, simple_did_data):
        """DiD should work with smaller samples."""
        # Use only first 20 control and 20 treated units (40 total)
        # Data structure: control_pre (50), control_post (50), treated_pre (50), treated_post (50)
        # Indices: [0:50] control_pre, [50:100] control_post, [100:150] treated_pre, [150:200] treated_post

        keep_control = 20
        keep_treated = 20

        # Keep first 20 from each group
        mask = np.concatenate([
            np.ones(keep_control, dtype=bool), np.zeros(50-keep_control, dtype=bool),  # control_pre
            np.ones(keep_control, dtype=bool), np.zeros(50-keep_control, dtype=bool),  # control_post
            np.ones(keep_treated, dtype=bool), np.zeros(50-keep_treated, dtype=bool),  # treated_pre
            np.ones(keep_treated, dtype=bool), np.zeros(50-keep_treated, dtype=bool),  # treated_post
        ])

        outcomes = simple_did_data["outcomes"][mask]
        treatment = simple_did_data["treatment"][mask]
        post = simple_did_data["post"][mask]
        unit_id = simple_did_data["unit_id"][mask]

        # Should still run with 20 treated and 20 control
        result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)
        assert "estimate" in result
        assert result["n_treated"] == keep_treated
        assert result["n_control"] == keep_control

    def test_cluster_se_larger_than_naive(self, simple_did_data):
        """Cluster-robust SE should be larger than naive SE with serial correlation."""
        # Test with cluster SEs
        result_cluster = did_2x2(
            outcomes=simple_did_data["outcomes"],
            treatment=simple_did_data["treatment"],
            post=simple_did_data["post"],
            unit_id=simple_did_data["unit_id"],
            cluster_se=True,
        )

        # Test without cluster SEs
        result_naive = did_2x2(
            outcomes=simple_did_data["outcomes"],
            treatment=simple_did_data["treatment"],
            post=simple_did_data["post"],
            unit_id=simple_did_data["unit_id"],
            cluster_se=False,
        )

        # Cluster SE should typically be larger (or equal)
        assert result_cluster["se"] >= result_naive["se"] * 0.9, \
            "Cluster SE should be at least 90% of naive SE"

        # Both should have same point estimate
        assert abs(result_cluster["estimate"] - result_naive["estimate"]) < 1e-10

    def test_multiple_pre_post_periods(self, balanced_panel_data):
        """DiD should work with multiple pre/post periods."""
        result = did_2x2(
            outcomes=balanced_panel_data["outcomes"],
            treatment=balanced_panel_data["treatment"],
            post=balanced_panel_data["post"],
            unit_id=balanced_panel_data["unit_id"],
            cluster_se=True,
        )

        # Should recover true effect (5.0)
        true_did = balanced_panel_data["true_did"]
        assert abs(result["estimate"] - true_did) < 1.0, f"Expected ~{true_did}, got {result['estimate']}"

        # With larger sample, should be more precise
        assert result["p_value"] < 0.05, "Should detect effect with multiple periods"

    def test_heterogeneous_baseline_levels(self, heterogeneous_baselines_data):
        """DiD should handle heterogeneous baseline levels across units."""
        result = did_2x2(
            outcomes=heterogeneous_baselines_data["outcomes"],
            treatment=heterogeneous_baselines_data["treatment"],
            post=heterogeneous_baselines_data["post"],
            unit_id=heterogeneous_baselines_data["unit_id"],
            cluster_se=True,
        )

        # Should recover true effect (3.0) despite heterogeneous baselines
        true_did = heterogeneous_baselines_data["true_did"]
        assert abs(result["estimate"] - true_did) < 1.0, \
            f"Expected ~{true_did}, got {result['estimate']} (heterogeneous baselines)"

    def test_treatment_effect_only_in_post(self, simple_did_data):
        """Treatment effect should only appear in post-period."""
        outcomes = simple_did_data["outcomes"]
        treatment = simple_did_data["treatment"]
        post = simple_did_data["post"]

        # Pre-period: treated and control should have constant difference
        pre_mask = post == 0
        treated_pre_mean = outcomes[(treatment == 1) & pre_mask].mean()
        control_pre_mean = outcomes[(treatment == 0) & pre_mask].mean()
        pre_diff = treated_pre_mean - control_pre_mean

        # Post-period: difference should be larger
        post_mask = post == 1
        treated_post_mean = outcomes[(treatment == 1) & post_mask].mean()
        control_post_mean = outcomes[(treatment == 0) & post_mask].mean()
        post_diff = treated_post_mean - control_post_mean

        # DiD = change in difference
        did_manual = post_diff - pre_diff

        # Should be close to true effect
        assert abs(did_manual - simple_did_data["true_did"]) < 0.5

    def test_negative_treatment_effect(self, negative_effect_data):
        """DiD should correctly identify negative treatment effects."""
        result = did_2x2(
            outcomes=negative_effect_data["outcomes"],
            treatment=negative_effect_data["treatment"],
            post=negative_effect_data["post"],
            unit_id=negative_effect_data["unit_id"],
            cluster_se=True,
        )

        # Should be negative and close to true effect (-4.0)
        true_did = negative_effect_data["true_did"]
        assert result["estimate"] < 0, "Effect should be negative"
        assert abs(result["estimate"] - true_did) < 1.0, \
            f"Expected ~{true_did}, got {result['estimate']}"

        # CI should not contain zero
        assert result["ci_upper"] < 0, "CI should be entirely negative"

    def test_single_pre_single_post_minimum(self):
        """DiD should work with minimum 1 pre + 1 post period."""
        # 20 control, 20 treated, 1 pre, 1 post
        np.random.seed(999)
        n_control = 20
        n_treated = 20

        control_pre = np.full(n_control, 8.0) + np.random.normal(0, 0.5, n_control)
        control_post = np.full(n_control, 10.0) + np.random.normal(0, 0.5, n_control)
        treated_pre = np.full(n_treated, 9.0) + np.random.normal(0, 0.5, n_treated)
        treated_post = np.full(n_treated, 14.0) + np.random.normal(0, 0.5, n_treated)

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

        # DiD = (14 - 9) - (10 - 8) = 5 - 2 = 3
        assert abs(result["estimate"] - 3.0) < 0.5, "Should work with minimum periods"
        assert result["n_clusters"] == 40


class TestDiDInputValidation:
    """Test input validation and error handling."""

    def test_mismatched_lengths(self):
        """Should raise error if input arrays have different lengths."""
        outcomes = np.array([1, 2, 3, 4])
        treatment = np.array([0, 1])  # Wrong length
        post = np.array([0, 0, 1, 1])
        unit_id = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="All inputs must have same length"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_non_binary_treatment(self):
        """Should raise error if treatment is not binary."""
        outcomes = np.array([1, 2, 3, 4])
        treatment = np.array([0, 1, 2, 1])  # Contains 2
        post = np.array([0, 0, 1, 1])
        unit_id = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="must be binary"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_non_binary_post(self):
        """Should raise error if post is not binary."""
        outcomes = np.array([1, 2, 3, 4])
        treatment = np.array([0, 1, 0, 1])
        post = np.array([0, 0, 1, 2])  # Contains 2
        unit_id = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="must be binary"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_nan_in_outcomes(self):
        """Should raise error if outcomes contain NaN."""
        outcomes = np.array([1.0, np.nan, 3.0, 4.0])
        treatment = np.array([0, 1, 0, 1])
        post = np.array([0, 0, 1, 1])
        unit_id = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="non-finite"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_time_varying_treatment(self):
        """Should raise error if treatment varies within unit."""
        # Unit 0 has treatment=0 in period 0, treatment=1 in period 1
        outcomes = np.array([10, 15, 12, 20])
        treatment = np.array([0, 1, 1, 1])  # Unit 0 changes treatment
        post = np.array([0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="treatment must be constant within units"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_all_treated_units(self):
        """Should raise error if no control units."""
        outcomes = np.array([10, 15, 12, 20])
        treatment = np.array([1, 1, 1, 1])  # All treated
        post = np.array([0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="No control units"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_all_control_units(self):
        """Should raise error if no treated units."""
        outcomes = np.array([10, 15, 12, 20])
        treatment = np.array([0, 0, 0, 0])  # All control
        post = np.array([0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="No treated units"):
            did_2x2(outcomes, treatment, post, unit_id)

    def test_invalid_alpha(self):
        """Should raise error if alpha is not in (0, 1)."""
        outcomes = np.array([10, 15, 12, 20])
        treatment = np.array([0, 0, 1, 1])
        post = np.array([0, 1, 0, 1])
        unit_id = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            did_2x2(outcomes, treatment, post, unit_id, alpha=1.5)


class TestParallelTrendsTesting:
    """Test parallel trends testing functionality."""

    def test_parallel_trends_with_balanced_panel(self, balanced_panel_data):
        """Should detect parallel trends in pre-period."""
        result = check_parallel_trends(
            outcomes=balanced_panel_data["outcomes"],
            treatment=balanced_panel_data["treatment"],
            time=balanced_panel_data["time"],
            unit_id=balanced_panel_data["unit_id"],
            treatment_time=balanced_panel_data["treatment_time"],
        )

        # p-value should be high (fail to reject parallel trends)
        assert result["p_value"] > 0.05, "Should not reject parallel trends"
        assert result["parallel_trends_plausible"] is True

        # Pre-trend difference should be small
        assert abs(result["pre_trend_diff"]) < 1.0

    def test_parallel_trends_violation_detected(self):
        """Should detect differential pre-trends."""
        np.random.seed(555)
        n_treated = 30
        n_control = 30
        n_units = n_treated + n_control

        outcomes = []
        treatment_vec = []
        time_vec = []
        unit_id_vec = []

        # Treated group has upward pre-trend, control group flat
        for unit in range(n_units):
            is_treated = unit >= n_control
            baseline = 10.0

            for t in range(-2, 3):  # 3 pre-periods, 2 post-periods
                is_post = t >= 1

                # Control: flat pre-trend
                # Treated: +2 per period pre-trend
                if is_treated and t < 1:
                    y = baseline + 2 * (t + 2)  # Upward pre-trend
                else:
                    y = baseline

                if is_post:
                    y += 5.0  # Post-period jump for both

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

        # Should detect differential pre-trend
        assert result["p_value"] < 0.05, "Should reject parallel trends"
        assert result["parallel_trends_plausible"] is False
        # Note: warning_msg only set for n_pre_periods <= 2 or n_clusters < 20
        # This test has 3 pre-periods and 60 units, so no warning expected

    def test_insufficient_pre_periods(self):
        """Should raise error if <2 pre-periods."""
        # Only 1 pre-period
        outcomes = np.array([10, 15, 12, 20])
        treatment = np.array([0, 0, 1, 1])
        time = np.array([0, 1, 0, 1])  # t=0 is pre, t=1 is post
        unit_id = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="at least 2 pre-treatment periods"):
            check_parallel_trends(outcomes, treatment, time, unit_id, treatment_time=1)

    def test_no_time_variation_in_pre_period(self):
        """Should raise error if all pre-observations at same time."""
        # All pre-period observations at t=0
        outcomes = np.array([10, 15, 12, 20, 11, 16])
        treatment = np.array([0, 0, 1, 1, 0, 1])
        time = np.array([0, 0, 0, 0, 1, 1])  # All pre at t=0
        unit_id = np.array([0, 1, 2, 3, 0, 2])

        with pytest.raises(ValueError, match="at least 2 pre-treatment periods"):
            check_parallel_trends(outcomes, treatment, time, unit_id, treatment_time=1)

    def test_parallel_trends_nan_handling(self):
        """Should raise error if NaN in inputs."""
        outcomes = np.array([10, np.nan, 12, 20, 11, 16])
        treatment = np.array([0, 0, 1, 1, 0, 1])
        time = np.array([-1, 0, -1, 0, 1, 1])
        unit_id = np.array([0, 0, 1, 1, 0, 1])

        with pytest.raises(ValueError, match="contains NaN or inf"):
            check_parallel_trends(outcomes, treatment, time, unit_id, treatment_time=1)


class TestDiDDiagnostics:
    """Test diagnostic information returned by DiD estimator."""

    def test_diagnostics_structure(self, simple_did_data):
        """Should return all expected diagnostic fields."""
        result = did_2x2(
            outcomes=simple_did_data["outcomes"],
            treatment=simple_did_data["treatment"],
            post=simple_did_data["post"],
            unit_id=simple_did_data["unit_id"],
        )

        # Check required fields
        required_fields = [
            "estimate", "se", "t_stat", "p_value",
            "ci_lower", "ci_upper",
            "n_treated", "n_control", "n_clusters",
            "df", "cluster_se_used"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_degrees_of_freedom_calculation(self, simple_did_data):
        """Should use n_clusters - 1 for degrees of freedom."""
        result = did_2x2(
            outcomes=simple_did_data["outcomes"],
            treatment=simple_did_data["treatment"],
            post=simple_did_data["post"],
            unit_id=simple_did_data["unit_id"],
            cluster_se=True,
        )

        expected_df = result["n_clusters"] - 1
        assert result["df"] == expected_df, f"Expected df={expected_df}, got {result['df']}"

    def test_small_cluster_warning(self):
        """Should warn when n_clusters < 30."""
        # Create dataset with only 20 clusters
        np.random.seed(777)
        n_treated = 10
        n_control = 10

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

        # Should issue warning (RuntimeWarning, not UserWarning)
        with pytest.warns(RuntimeWarning, match="Small number of clusters"):
            result = did_2x2(outcomes, treatment, post, unit_id, cluster_se=True)

        assert result["n_clusters"] == 20
