"""
Tests for SCM Inference Methods

Tests placebo tests, bootstrap inference, and p-value computation.
"""

import numpy as np
import pytest

from src.causal_inference.scm import (
    synthetic_control,
    placebo_test_in_space,
    placebo_test_in_time,
    bootstrap_se,
    compute_confidence_interval,
    compute_p_value,
)


class TestPlaceboTestInSpace:
    """Tests for in-space placebo tests."""

    def test_placebo_returns_effects(self, balanced_panel):
        """Placebo test should return array of placebo effects."""
        control_mask = balanced_panel["treatment"] == 0
        control_outcomes = balanced_panel["outcomes"][control_mask, :]

        result = placebo_test_in_space(
            control_outcomes=control_outcomes,
            treatment_period=balanced_panel["treatment_period"],
            observed_effect=2.5,
            n_placebo=None,
        )

        assert len(result["placebo_effects"]) > 0
        assert result["se"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_placebo_with_limited_placebos(self, balanced_panel):
        """Should respect n_placebo limit."""
        control_mask = balanced_panel["treatment"] == 0
        control_outcomes = balanced_panel["outcomes"][control_mask, :]

        result = placebo_test_in_space(
            control_outcomes=control_outcomes,
            treatment_period=balanced_panel["treatment_period"],
            observed_effect=2.5,
            n_placebo=5,
        )

        # Should have at most 5 placebo effects
        assert len(result["placebo_effects"]) <= 5

    def test_large_effect_small_pvalue(self, balanced_panel):
        """Very large observed effect should yield small p-value."""
        control_mask = balanced_panel["treatment"] == 0
        control_outcomes = balanced_panel["outcomes"][control_mask, :]

        result = placebo_test_in_space(
            control_outcomes=control_outcomes,
            treatment_period=balanced_panel["treatment_period"],
            observed_effect=100.0,  # Very large
            n_placebo=None,
        )

        # P-value should be small for extreme effect
        assert result["p_value"] < 0.3


class TestPlaceboTestInTime:
    """Tests for in-time placebo tests."""

    def test_in_time_placebo(self, balanced_panel):
        """In-time placebo should compute pseudo effect."""
        # First get SCM result
        result = synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        placebo_result = placebo_test_in_time(
            treated_series=result["treated_series"],
            synthetic_series=result["synthetic_control"],
            treatment_period=balanced_panel["treatment_period"],
            pseudo_treatment_period=4,  # Pseudo treatment at period 4
        )

        assert "pseudo_effect" in placebo_result
        assert "actual_effect" in placebo_result
        assert "ratio" in placebo_result

    def test_invalid_pseudo_period(self, balanced_panel):
        """Pseudo period >= treatment period should raise error."""
        result = synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        with pytest.raises(ValueError, match="must be <"):
            placebo_test_in_time(
                treated_series=result["treated_series"],
                synthetic_series=result["synthetic_control"],
                treatment_period=balanced_panel["treatment_period"],
                pseudo_treatment_period=balanced_panel["treatment_period"],
            )


class TestBootstrapSE:
    """Tests for bootstrap standard error."""

    def test_bootstrap_returns_se(self, balanced_panel):
        """Bootstrap should return positive SE."""
        treated_mask = balanced_panel["treatment"] == 1
        control_mask = balanced_panel["treatment"] == 0

        treated = balanced_panel["outcomes"][treated_mask, :]
        control = balanced_panel["outcomes"][control_mask, :]

        se, effects = bootstrap_se(
            treated_outcomes=treated,
            control_outcomes=control,
            treatment_period=balanced_panel["treatment_period"],
            n_bootstrap=50,
            seed=42,
        )

        assert se > 0
        assert len(effects) > 0

    def test_bootstrap_with_block_length(self, balanced_panel):
        """Should accept custom block length."""
        treated_mask = balanced_panel["treatment"] == 1
        control_mask = balanced_panel["treatment"] == 0

        treated = balanced_panel["outcomes"][treated_mask, :]
        control = balanced_panel["outcomes"][control_mask, :]

        se, effects = bootstrap_se(
            treated_outcomes=treated,
            control_outcomes=control,
            treatment_period=balanced_panel["treatment_period"],
            n_bootstrap=30,
            block_length=3,
            seed=42,
        )

        assert se > 0


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_normal_ci(self):
        """Normal CI should be symmetric around estimate."""
        estimate = 2.0
        se = 0.5

        ci_lower, ci_upper = compute_confidence_interval(estimate, se, alpha=0.05, method="normal")

        # 95% CI should be approximately [1.02, 2.98]
        assert ci_lower < estimate < ci_upper
        assert np.isclose(estimate - ci_lower, ci_upper - estimate, atol=0.01)

    def test_percentile_ci(self):
        """Percentile CI from bootstrap distribution."""
        np.random.seed(42)
        bootstrap_effects = np.random.normal(2.0, 0.5, size=1000)

        ci_lower, ci_upper = compute_confidence_interval(
            estimate=2.0,
            se=0.5,
            alpha=0.05,
            method="percentile",
            bootstrap_effects=bootstrap_effects,
        )

        assert ci_lower < ci_upper
        # Should be roughly [1.0, 3.0] for 95% CI
        assert 0.8 < ci_lower < 1.2
        assert 2.8 < ci_upper < 3.2

    def test_invalid_method(self):
        """Invalid method should raise error."""
        with pytest.raises(ValueError, match="Unknown CI method"):
            compute_confidence_interval(2.0, 0.5, method="invalid")


class TestPValue:
    """Tests for p-value computation."""

    def test_two_sided_pvalue(self):
        """Two-sided p-value for observed effect."""
        placebo_effects = np.array([-1, -0.5, 0, 0.5, 1, 1.5])
        observed = 2.0

        p_value = compute_p_value(observed, placebo_effects, alternative="two-sided")

        # Only 0 placebos have |effect| >= 2.0, so p = 1/7 ≈ 0.14
        assert 0 < p_value < 0.3

    def test_greater_pvalue(self):
        """One-sided p-value (greater)."""
        placebo_effects = np.array([0, 0.5, 1, 1.5, 2, 2.5])
        observed = 2.0

        p_value = compute_p_value(observed, placebo_effects, alternative="greater")

        # 2 placebos >= 2.0 (2, 2.5), so p = 3/7 ≈ 0.43
        assert 0.3 < p_value < 0.6

    def test_less_pvalue(self):
        """One-sided p-value (less)."""
        placebo_effects = np.array([0, 0.5, 1, 1.5, 2, 2.5])
        observed = 0.5

        p_value = compute_p_value(observed, placebo_effects, alternative="less")

        # 2 placebos <= 0.5 (0, 0.5), so p = 3/7 ≈ 0.43
        assert 0.3 < p_value < 0.6

    def test_empty_placebo(self):
        """Empty placebo distribution returns NaN."""
        p_value = compute_p_value(1.0, np.array([]), alternative="two-sided")
        assert np.isnan(p_value)

    def test_invalid_alternative(self):
        """Invalid alternative should raise error."""
        with pytest.raises(ValueError, match="Unknown alternative"):
            compute_p_value(1.0, np.array([0, 1]), alternative="invalid")
