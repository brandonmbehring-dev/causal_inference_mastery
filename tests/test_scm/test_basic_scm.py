"""
Tests for Synthetic Control Method - Basic Estimator

Layer 1: Known-answer tests with hand-calculated expected values
Layer 2: Adversarial tests for edge cases and validation
"""

import numpy as np
import pytest

from src.causal_inference.scm import (
    synthetic_control,
    compute_scm_weights,
    compute_pre_treatment_fit,
    validate_panel_data,
    SCMResult,
)


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestSyntheticControlKnownAnswer:
    """Known-answer tests with hand-calculated expected values."""

    def test_perfect_match_dominant_control(self, simple_panel):
        """
        With control_0 identical to treated pre-treatment,
        first weight should dominate and effect should match true_effect.
        """
        result = synthetic_control(
            outcomes=simple_panel["outcomes"],
            treatment=simple_panel["treatment"],
            treatment_period=simple_panel["treatment_period"],
            inference="none",
        )

        # First control should get most weight (≈1.0)
        assert len(result["weights"]) == 2
        assert result["weights"][0] > 0.9

        # Effect should be close to true effect
        assert np.isclose(result["estimate"], simple_panel["true_effect"], rtol=0.1)

        # Pre-treatment fit should be near-perfect
        assert result["pre_rmse"] < 0.1
        assert result["pre_r_squared"] > 0.99

    def test_weighted_combination(self, multi_control_panel):
        """
        When treated = 0.5*control_0 + 0.5*control_1,
        weights should be approximately [0.5, 0.5, 0, 0, 0].
        """
        result = synthetic_control(
            outcomes=multi_control_panel["outcomes"],
            treatment=multi_control_panel["treatment"],
            treatment_period=multi_control_panel["treatment_period"],
            inference="none",
        )

        weights = result["weights"]

        # First two weights should dominate
        assert weights[0] + weights[1] > 0.7

        # Effect should be close to true effect
        assert np.isclose(result["estimate"], multi_control_panel["true_effect"], rtol=0.2)

    def test_output_structure(self, simple_panel):
        """Verify all required fields are present in result."""
        result = synthetic_control(
            outcomes=simple_panel["outcomes"],
            treatment=simple_panel["treatment"],
            treatment_period=simple_panel["treatment_period"],
            inference="none",
        )

        # Check required fields
        required_fields = [
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "weights",
            "pre_rmse",
            "pre_r_squared",
            "n_treated",
            "n_control",
            "n_pre_periods",
            "n_post_periods",
            "synthetic_control",
            "treated_series",
            "gap",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_gap_dimensions(self, simple_panel):
        """Gap should have length n_periods."""
        result = synthetic_control(
            outcomes=simple_panel["outcomes"],
            treatment=simple_panel["treatment"],
            treatment_period=simple_panel["treatment_period"],
            inference="none",
        )

        n_periods = simple_panel["outcomes"].shape[1]
        assert len(result["gap"]) == n_periods
        assert len(result["synthetic_control"]) == n_periods
        assert len(result["treated_series"]) == n_periods

    def test_pre_treatment_gap_near_zero(self, simple_panel):
        """Pre-treatment gap should be near zero for good fit."""
        result = synthetic_control(
            outcomes=simple_panel["outcomes"],
            treatment=simple_panel["treatment"],
            treatment_period=simple_panel["treatment_period"],
            inference="none",
        )

        pre_gap = result["gap"][: simple_panel["treatment_period"]]
        assert np.abs(pre_gap).mean() < 0.5

    def test_simplex_constraint(self, multi_control_panel):
        """Weights must be non-negative and sum to 1."""
        result = synthetic_control(
            outcomes=multi_control_panel["outcomes"],
            treatment=multi_control_panel["treatment"],
            treatment_period=multi_control_panel["treatment_period"],
            inference="none",
        )

        weights = result["weights"]

        # Non-negative
        assert np.all(weights >= -1e-6)

        # Sum to 1
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)

    def test_n_counts(self, multi_control_panel):
        """Verify unit and period counts."""
        result = synthetic_control(
            outcomes=multi_control_panel["outcomes"],
            treatment=multi_control_panel["treatment"],
            treatment_period=multi_control_panel["treatment_period"],
            inference="none",
        )

        assert result["n_treated"] == 1
        assert result["n_control"] == 5
        assert result["n_pre_periods"] == 10
        assert result["n_post_periods"] == 10


class TestWeightComputation:
    """Tests for weight optimization."""

    def test_compute_weights_perfect_match(self):
        """Single control identical to treated should get weight 1.0."""
        np.random.seed(42)
        n_pre = 10

        treated = np.random.randn(n_pre)
        control = treated.copy().reshape(1, -1)  # (1, n_pre)

        weights, result = compute_scm_weights(treated.reshape(1, -1), control)

        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0, atol=1e-6)

    def test_compute_weights_uniform(self):
        """When all controls identical, weights should be uniform-ish."""
        np.random.seed(42)
        n_pre = 10
        n_control = 4

        treated = np.random.randn(n_pre)
        control = np.tile(treated, (n_control, 1))  # All identical

        weights, result = compute_scm_weights(treated.reshape(1, -1), control)

        # All weights should be equal (within tolerance)
        assert np.allclose(weights, 0.25, atol=0.1)

    def test_pre_treatment_fit_perfect(self):
        """Perfect fit should have RMSE=0, R²=1."""
        np.random.seed(42)
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = treated.copy().reshape(1, -1)
        weights = np.array([1.0])

        rmse, r2 = compute_pre_treatment_fit(treated, control, weights)

        assert np.isclose(rmse, 0.0, atol=1e-10)
        assert np.isclose(r2, 1.0, atol=1e-10)


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestInputValidation:
    """Adversarial tests for input validation."""

    def test_wrong_outcomes_dim(self):
        """outcomes must be 2D."""
        with pytest.raises(ValueError, match="2D"):
            validate_panel_data(
                outcomes=np.array([1, 2, 3]),  # 1D
                treatment=np.array([1, 0, 0]),
                treatment_period=2,
            )

    def test_treatment_length_mismatch(self):
        """treatment length must match n_units."""
        with pytest.raises(ValueError, match="length"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0]),  # Wrong length
                treatment_period=5,
            )

    def test_non_binary_treatment(self):
        """treatment must be binary."""
        with pytest.raises(ValueError, match="binary"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 2, 0]),  # Has 2
                treatment_period=5,
            )

    def test_no_treated_units(self):
        """Must have at least one treated unit."""
        with pytest.raises(ValueError, match="No treated"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([0, 0, 0, 0, 0]),  # All control
                treatment_period=5,
            )

    def test_no_control_units(self):
        """Must have control units."""
        with pytest.raises(ValueError, match="No control"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 1, 1, 1, 1]),  # All treated
                treatment_period=5,
            )

    def test_only_one_control(self):
        """Need at least 2 control units."""
        with pytest.raises(ValueError, match="at least 2 control"):
            validate_panel_data(
                outcomes=np.random.randn(2, 10),
                treatment=np.array([1, 0]),  # Only 1 control
                treatment_period=5,
            )

    def test_treatment_period_zero(self):
        """treatment_period must be >= 1."""
        with pytest.raises(ValueError, match=">= 1"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=0,  # No pre-treatment periods
            )

    def test_treatment_period_too_late(self):
        """treatment_period must be < n_periods."""
        with pytest.raises(ValueError, match="no post-treatment"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=10,  # No post-treatment
            )

    def test_nan_in_outcomes(self):
        """outcomes cannot contain NaN."""
        outcomes = np.random.randn(5, 10)
        outcomes[2, 5] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            validate_panel_data(
                outcomes=outcomes,
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=5,
            )

    def test_wrong_covariates_dim(self):
        """covariates must be 2D."""
        with pytest.raises(ValueError, match="2D"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=5,
                covariates=np.array([1, 2, 3, 4, 5]),  # 1D
            )

    def test_covariates_row_mismatch(self):
        """covariates rows must match n_units."""
        with pytest.raises(ValueError, match="rows"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=5,
                covariates=np.random.randn(3, 2),  # Wrong rows
            )

    def test_few_pre_periods_warning(self):
        """Should warn when n_pre_periods < 5."""
        with pytest.warns(UserWarning, match="pre-treatment periods"):
            validate_panel_data(
                outcomes=np.random.randn(5, 6),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=3,  # Only 3 pre-periods
            )


class TestEdgeCases:
    """Edge cases for SCM estimation."""

    def test_large_treatment_effect(self):
        """SCM should detect large positive effect."""
        np.random.seed(42)
        n_periods = 12
        treatment_period = 6

        # Create panel
        outcomes = np.random.randn(4, n_periods) + 10
        outcomes[0, treatment_period:] += 10.0  # Large effect

        treatment = np.array([1, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, treatment_period, inference="none")

        assert result["estimate"] > 5.0

    def test_negative_treatment_effect(self):
        """SCM should detect negative effect."""
        np.random.seed(42)
        n_periods = 12
        treatment_period = 6

        outcomes = np.random.randn(4, n_periods) + 10
        outcomes[0, treatment_period:] -= 5.0  # Negative effect

        treatment = np.array([1, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, treatment_period, inference="none")

        assert result["estimate"] < -2.0

    def test_multiple_treated_units(self):
        """SCM should handle multiple treated units (averaging)."""
        np.random.seed(42)
        n_periods = 10
        treatment_period = 5

        outcomes = np.random.randn(5, n_periods) + 10
        # Two treated units with same effect
        outcomes[0, treatment_period:] += 2.0
        outcomes[1, treatment_period:] += 2.0

        treatment = np.array([1, 1, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, treatment_period, inference="none")

        assert result["n_treated"] == 2
        assert result["n_control"] == 3
        assert np.isclose(result["estimate"], 2.0, rtol=0.5)

    def test_with_covariates(self, panel_with_covariates):
        """SCM should accept and use covariates."""
        result = synthetic_control(
            outcomes=panel_with_covariates["outcomes"],
            treatment=panel_with_covariates["treatment"],
            treatment_period=panel_with_covariates["treatment_period"],
            covariates=panel_with_covariates["covariates"],
            inference="none",
        )

        # Should still estimate close to true effect
        assert np.isclose(
            result["estimate"],
            panel_with_covariates["true_effect"],
            rtol=0.5,
        )


class TestInference:
    """Tests for inference methods."""

    def test_placebo_inference(self, balanced_panel):
        """Placebo inference should return SE and p-value."""
        result = synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="placebo",
            n_placebo=50,
        )

        # SE should be positive
        assert result["se"] > 0

        # P-value should be in [0, 1]
        assert 0 <= result["p_value"] <= 1

        # CI should be finite
        assert np.isfinite(result["ci_lower"])
        assert np.isfinite(result["ci_upper"])

    def test_bootstrap_inference(self, balanced_panel):
        """Bootstrap inference should return SE and p-value."""
        result = synthetic_control(
            outcomes=balanced_panel["outcomes"],
            treatment=balanced_panel["treatment"],
            treatment_period=balanced_panel["treatment_period"],
            inference="bootstrap",
            n_placebo=50,
        )

        assert result["se"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_no_inference(self, simple_panel):
        """inference='none' should return NaN for SE and p-value."""
        result = synthetic_control(
            outcomes=simple_panel["outcomes"],
            treatment=simple_panel["treatment"],
            treatment_period=simple_panel["treatment_period"],
            inference="none",
        )

        assert np.isnan(result["se"])
        assert np.isnan(result["p_value"])
        assert np.isnan(result["ci_lower"])
        assert np.isnan(result["ci_upper"])

    def test_invalid_inference_method(self, multi_control_panel):
        """Invalid inference method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown inference"):
            synthetic_control(
                outcomes=multi_control_panel["outcomes"],
                treatment=multi_control_panel["treatment"],
                treatment_period=multi_control_panel["treatment_period"],
                inference="invalid",
            )

    def test_null_effect_p_value(self, no_effect_panel):
        """With no true effect, p-value should be high."""
        result = synthetic_control(
            outcomes=no_effect_panel["outcomes"],
            treatment=no_effect_panel["treatment"],
            treatment_period=no_effect_panel["treatment_period"],
            inference="placebo",
            n_placebo=50,
        )

        # P-value should not be very small with null effect
        # (This is probabilistic, but should usually be > 0.05)
        # We just check it's not extremely small
        assert result["p_value"] > 0.01


# =============================================================================
# Layer 3: Monte Carlo (Simplified)
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo validation tests."""

    @pytest.mark.slow
    def test_unbiasedness(self):
        """SCM should be approximately unbiased over many runs."""
        np.random.seed(42)
        n_runs = 50
        true_effect = 2.0
        estimates = []

        for run in range(n_runs):
            # Generate panel
            n_units = 6
            n_periods = 16
            treatment_period = 8

            outcomes = np.random.randn(n_units, n_periods) + 10
            outcomes[0, treatment_period:] += true_effect

            treatment = np.zeros(n_units)
            treatment[0] = 1

            result = synthetic_control(outcomes, treatment, treatment_period, inference="none")
            estimates.append(result["estimate"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_effect

        # Bias should be small (< 0.5 for SCM with noise)
        assert abs(bias) < 0.5, f"Bias too large: {bias:.3f}"

    @pytest.mark.slow
    def test_ci_coverage(self):
        """CI should have approximately correct coverage."""
        np.random.seed(42)
        n_runs = 30
        true_effect = 2.0
        covered = 0

        for run in range(n_runs):
            n_units = 8
            n_periods = 14
            treatment_period = 7

            outcomes = np.random.randn(n_units, n_periods) + 10
            outcomes[0, treatment_period:] += true_effect

            treatment = np.zeros(n_units)
            treatment[0] = 1

            result = synthetic_control(
                outcomes,
                treatment,
                treatment_period,
                inference="placebo",
                n_placebo=30,
                alpha=0.10,
            )

            if result["ci_lower"] < true_effect < result["ci_upper"]:
                covered += 1

        coverage = covered / n_runs

        # Coverage should be roughly 90% (alpha=0.10)
        # Allow wide range due to Monte Carlo variance
        assert 0.6 < coverage < 1.0, f"Coverage: {coverage:.2%}"
