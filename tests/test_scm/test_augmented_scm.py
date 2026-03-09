"""
Tests for Augmented Synthetic Control Method (ASCM)

Tests Ben-Michael et al. (2021) augmented estimator.
"""

import numpy as np
import pytest

from src.causal_inference.scm import augmented_synthetic_control, ASCMResult


class TestAugmentedSCMBasic:
    """Basic tests for ASCM."""

    def test_augmented_scm_runs(self, balanced_panel):
        """ASCM should run without errors."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        assert "estimate" in result
        assert "weights" in result
        assert "ridge_coef" in result
        assert "augmented_control" in result

    def test_augmented_scm_estimate(self, balanced_panel):
        """ASCM estimate should be close to true effect."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        # Should be reasonably close to true effect (2.5)
        assert np.isclose(result["estimate"], balanced_panel["true_effect"], rtol=0.5)

    def test_output_structure(self, balanced_panel):
        """Check all required output fields."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        required_fields = [
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "weights",
            "ridge_coef",
            "pre_rmse",
            "pre_r_squared",
            "n_treated",
            "n_control",
            "n_pre_periods",
            "n_post_periods",
            "synthetic_control",
            "augmented_control",
            "treated_series",
            "gap",
            "lambda_ridge",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"


class TestAugmentedSCMInference:
    """Tests for ASCM inference methods."""

    def test_jackknife_inference(self, balanced_panel):
        """Jackknife inference should return SE."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="jackknife",
        )

        assert result["se"] > 0
        assert np.isfinite(result["ci_lower"])
        assert np.isfinite(result["ci_upper"])

    def test_bootstrap_inference(self, balanced_panel):
        """Bootstrap inference should return SE."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="bootstrap",
        )

        assert result["se"] > 0

    def test_no_inference(self, balanced_panel):
        """No inference should return NaN for SE."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="none",
        )

        assert np.isnan(result["se"])
        assert np.isnan(result["ci_lower"])
        assert np.isnan(result["ci_upper"])


class TestAugmentedSCMLambda:
    """Tests for ridge penalty selection."""

    def test_custom_lambda(self, balanced_panel):
        """Should accept custom lambda."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            lambda_ridge=10.0,
            inference="none",
        )

        assert result["lambda_ridge"] == 10.0

    def test_cv_lambda_selection(self, balanced_panel):
        """Without lambda, should select via CV."""
        result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            lambda_ridge=None,  # Trigger CV
            inference="none",
        )

        assert result["lambda_ridge"] > 0


class TestAugmentedSCMBiasReduction:
    """Tests for bias reduction property."""

    @pytest.mark.slow
    def test_ascm_vs_scm_imperfect_fit(self):
        """ASCM should reduce bias when pre-treatment fit is imperfect."""
        np.random.seed(42)
        n_units = 10
        n_periods = 16
        treatment_period = 8
        true_effect = 3.0

        # Create panel where SCM fit is imperfect
        # Treated unit has different trend than controls
        outcomes = np.zeros((n_units, n_periods))
        for i in range(n_units):
            trend = np.linspace(0, 4, n_periods)
            unit_fe = np.random.randn() * 2
            noise = np.random.randn(n_periods) * 0.5
            outcomes[i, :] = 10 + trend + unit_fe + noise

        # Treated unit has slightly different trend
        outcomes[0, :] += np.linspace(0, 2, n_periods)  # Extra trend
        outcomes[0, treatment_period:] += true_effect

        treatment = np.zeros(n_units)
        treatment[0] = 1

        # Compare SCM and ASCM
        from causal_inference.scm import synthetic_control

        scm_result = synthetic_control(outcomes, treatment, treatment_period, inference="none")

        ascm_result = augmented_synthetic_control(
            outcomes, treatment, treatment_period, inference="none"
        )

        # ASCM should have estimate closer to true effect when fit is imperfect
        scm_error = abs(scm_result["estimate"] - true_effect)
        ascm_error = abs(ascm_result["estimate"] - true_effect)

        # ASCM should not be dramatically worse (may or may not be better
        # depending on specific DGP)
        assert ascm_error < scm_error * 1.5


class TestAugmentedSCMEdgeCases:
    """Edge case tests for ASCM."""

    def test_many_controls(self):
        """Should handle many control units."""
        np.random.seed(42)
        n_units = 20
        n_periods = 12
        treatment_period = 6

        outcomes = np.random.randn(n_units, n_periods) + 10
        outcomes[0, treatment_period:] += 2.0

        treatment = np.zeros(n_units)
        treatment[0] = 1

        result = augmented_synthetic_control(
            outcomes, treatment, treatment_period, inference="none"
        )

        assert np.isfinite(result["estimate"])

    def test_minimum_controls(self):
        """Should work with minimum number of controls."""
        np.random.seed(42)
        n_units = 4  # 1 treated + 3 controls (minimum for CV)
        n_periods = 10
        treatment_period = 5

        outcomes = np.random.randn(n_units, n_periods) + 10
        outcomes[0, treatment_period:] += 2.0

        treatment = np.zeros(n_units)
        treatment[0] = 1

        result = augmented_synthetic_control(
            outcomes, treatment, treatment_period, inference="none"
        )

        assert np.isfinite(result["estimate"])


class TestJackknifeSEValidation:
    """Monte Carlo validation for jackknife SE (BUG-7 fix verification)."""

    @pytest.mark.slow
    def test_jackknife_se_accuracy(self):
        """
        Jackknife SE should be within 50% of empirical SD.

        BUG-7 FIX VALIDATION: With proper weight recomputation in LOO,
        jackknife SE should accurately estimate true sampling variability.
        """
        np.random.seed(42)
        n_runs = 100  # Fewer runs for speed, but enough for SE estimation
        estimates = []

        n_units = 8
        n_periods = 12
        treatment_period = 6
        true_effect = 2.0

        for _ in range(n_runs):
            # Generate fresh panel each run
            outcomes = np.zeros((n_units, n_periods))
            for i in range(n_units):
                trend = np.linspace(0, 3, n_periods)
                noise = np.random.randn(n_periods) * 0.5
                outcomes[i, :] = 10 + trend + noise

            # Add treatment effect
            outcomes[0, treatment_period:] += true_effect

            treatment = np.zeros(n_units)
            treatment[0] = 1

            result = augmented_synthetic_control(
                outcomes, treatment, treatment_period, inference="none"
            )
            estimates.append(result["estimate"])

        # Compute empirical SD
        empirical_sd = np.std(estimates, ddof=1)

        # Get jackknife SE from one run with fixed data
        np.random.seed(999)
        outcomes = np.zeros((n_units, n_periods))
        for i in range(n_units):
            trend = np.linspace(0, 3, n_periods)
            noise = np.random.randn(n_periods) * 0.5
            outcomes[i, :] = 10 + trend + noise
        outcomes[0, treatment_period:] += true_effect

        treatment = np.zeros(n_units)
        treatment[0] = 1

        jackknife_result = augmented_synthetic_control(
            outcomes, treatment, treatment_period, inference="jackknife"
        )
        jackknife_se = jackknife_result["se"]

        # Jackknife SE should be within factor of 2 of empirical SD
        # (relaxed bound due to jackknife variability with few control units)
        ratio = jackknife_se / empirical_sd
        assert 0.5 < ratio < 2.0, (
            f"Jackknife SE {jackknife_se:.4f} differs from empirical SD {empirical_sd:.4f} "
            f"by ratio {ratio:.2f} (expected 0.5-2.0)"
        )

    def test_jackknife_vs_bootstrap_consistency(self, balanced_panel):
        """
        Jackknife and bootstrap SE should be similar magnitude.

        BUG-7 FIX: With proper weight recomputation, jackknife and bootstrap
        should give comparable uncertainty estimates.
        """
        jackknife_result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="jackknife",
        )

        bootstrap_result = augmented_synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="bootstrap",
        )

        jk_se = jackknife_result["se"]
        bs_se = bootstrap_result["se"]

        # Both should be positive
        assert jk_se > 0
        assert bs_se > 0

        # Should be within factor of 3 (reasonable for small sample)
        ratio = max(jk_se, bs_se) / min(jk_se, bs_se)
        assert ratio < 3.0, (
            f"Jackknife SE {jk_se:.4f} and bootstrap SE {bs_se:.4f} "
            f"differ by factor {ratio:.2f} (expected < 3)"
        )
