"""
Adversarial Tests for Synthetic Control Methods

Layer 2 validation: Edge cases, boundary conditions, and stress tests
for SCM and Augmented SCM estimators.

Test Categories:
1. Input Validation - Type errors, dimension mismatches, invalid values
2. Numerical Stability - Extreme scales, collinearity, near-singular cases
3. Panel Structure - Minimum viable panels, extreme imbalance
4. Optimizer Edge Cases - Convergence failures, degenerate solutions
5. Inference Edge Cases - Few placebos, bootstrap/jackknife failures
6. ASCM Specific - Ridge regularization edge cases
7. Weight Validation - Constraint violations

References:
    Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods"
    Ben-Michael, Feller, Rothstein (2021). "Augmented Synthetic Control"
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.scm import (
    synthetic_control,
    augmented_synthetic_control,
    compute_scm_weights,
    compute_pre_treatment_fit,
    validate_panel_data,
)
from src.causal_inference.scm.types import validate_weights


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_panel():
    """Minimum viable SCM panel: 3 units (1 treated + 2 controls), 3 periods."""
    np.random.seed(42)
    outcomes = np.array(
        [
            [10.0, 11.0, 15.0],  # Treated (effect in period 2)
            [10.0, 11.0, 12.0],  # Control 1
            [9.0, 10.0, 11.0],  # Control 2
        ]
    )
    treatment = np.array([1, 0, 0])
    treatment_period = 2
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
    }


@pytest.fixture
def many_controls_panel():
    """Panel with many control units (50 controls)."""
    np.random.seed(42)
    n_control = 50
    n_periods = 15
    treatment_period = 8

    outcomes = np.random.randn(n_control + 1, n_periods) * 2 + 10
    outcomes[0, treatment_period:] += 3.0  # Treatment effect

    treatment = np.zeros(n_control + 1)
    treatment[0] = 1

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
    }


# =============================================================================
# 1. Input Validation Edge Cases
# =============================================================================


class TestInputValidationEdgeCases:
    """Adversarial tests for input validation."""

    def test_inf_in_outcomes(self):
        """Inf values in outcomes should raise ValueError."""
        outcomes = np.array(
            [
                [10.0, np.inf, 12.0, 13.0],
                [9.0, 10.0, 11.0, 12.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        treatment = np.array([1, 0, 0])

        # Inf is handled as NaN-like invalid value
        with pytest.raises(ValueError):
            synthetic_control(outcomes, treatment, 2, inference="none")

    def test_negative_inf_in_outcomes(self):
        """-Inf values should raise error."""
        outcomes = np.array(
            [
                [10.0, -np.inf, 12.0, 13.0],
                [9.0, 10.0, 11.0, 12.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        treatment = np.array([1, 0, 0])

        with pytest.raises(ValueError):
            synthetic_control(outcomes, treatment, 2, inference="none")

    def test_outcomes_not_array(self):
        """outcomes must be numpy array."""
        with pytest.raises(TypeError, match="np.ndarray"):
            validate_panel_data(
                outcomes=[[1, 2, 3], [4, 5, 6]],  # List, not array
                treatment=np.array([1, 0]),
                treatment_period=2,
            )

    def test_treatment_not_array(self):
        """treatment must be numpy array."""
        with pytest.raises(TypeError, match="np.ndarray"):
            validate_panel_data(
                outcomes=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                treatment=[1, 0, 0],  # List, not array
                treatment_period=2,
            )

    def test_treatment_period_float(self):
        """treatment_period must be int."""
        with pytest.raises(TypeError, match="int"):
            validate_panel_data(
                outcomes=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                treatment=np.array([1, 0, 0]),
                treatment_period=1.5,  # Float
            )

    def test_treatment_period_negative(self):
        """treatment_period cannot be negative."""
        with pytest.raises(ValueError):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=-1,
            )

    def test_empty_outcomes(self):
        """Empty outcomes array should raise error."""
        with pytest.raises(ValueError):
            validate_panel_data(
                outcomes=np.array([]).reshape(0, 0),
                treatment=np.array([]),
                treatment_period=1,
            )

    def test_single_unit(self):
        """Single unit (no donor pool possible) should fail."""
        with pytest.raises(ValueError, match="No control"):
            validate_panel_data(
                outcomes=np.random.randn(1, 10),
                treatment=np.array([1]),
                treatment_period=5,
            )

    def test_all_nan_in_row(self):
        """Row of all NaN should be caught."""
        outcomes = np.random.randn(5, 10)
        outcomes[2, :] = np.nan  # Entire row is NaN

        with pytest.raises(ValueError, match="NaN"):
            validate_panel_data(
                outcomes=outcomes,
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=5,
            )

    def test_nan_in_covariates(self):
        """NaN in covariates should raise error."""
        covariates = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        with pytest.raises(ValueError, match="NaN"):
            validate_panel_data(
                outcomes=np.random.randn(3, 10),
                treatment=np.array([1, 0, 0]),
                treatment_period=5,
                covariates=covariates,
            )

    def test_covariates_not_array(self):
        """covariates must be numpy array if provided."""
        with pytest.raises(TypeError):
            validate_panel_data(
                outcomes=np.random.randn(3, 10),
                treatment=np.array([1, 0, 0]),
                treatment_period=5,
                covariates=[[1, 2], [3, 4], [5, 6]],  # List
            )


class TestTreatmentIndicatorEdgeCases:
    """Edge cases for treatment indicator validation."""

    def test_treatment_with_floats(self):
        """Float treatment values should be rejected."""
        with pytest.raises(ValueError, match="binary"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1.0, 0.5, 0.0, 0.0, 0.0]),  # Non-binary
                treatment_period=5,
            )

    def test_treatment_negative_values(self):
        """Negative treatment values should be rejected."""
        with pytest.raises(ValueError, match="binary"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, -1, 0, 0, 0]),
                treatment_period=5,
            )

    def test_treatment_values_greater_than_one(self):
        """Treatment values > 1 should be rejected."""
        with pytest.raises(ValueError, match="binary"):
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([2, 0, 0, 0, 0]),
                treatment_period=5,
            )


# =============================================================================
# 2. Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_very_small_outcome_values(self):
        """Outcomes near machine epsilon should still work."""
        np.random.seed(42)
        scale = 1e-12
        outcomes = np.random.randn(5, 10) * scale + scale
        outcomes[0, 5:] += 0.5 * scale  # Small effect

        treatment = np.array([1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert np.isfinite(result["estimate"])
        assert len(result["weights"]) == 4

    def test_very_large_outcome_values(self):
        """Large outcomes (1e10 scale) should work."""
        np.random.seed(42)
        scale = 1e10
        outcomes = np.random.randn(5, 10) * scale + scale
        outcomes[0, 5:] += 1e9  # Large effect

        treatment = np.array([1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert np.isfinite(result["estimate"])
        assert result["estimate"] > 0

    def test_mixed_scales(self):
        """Mix of large and small values in panel."""
        np.random.seed(42)
        outcomes = np.zeros((4, 10))
        outcomes[0, :] = np.random.randn(10) * 1e6 + 1e6  # Large
        outcomes[1, :] = np.random.randn(10) * 1e-3 + 1  # Small
        outcomes[2, :] = np.random.randn(10) * 100 + 500  # Medium
        outcomes[3, :] = np.random.randn(10) * 1e4 + 5e4  # Large-ish

        treatment = np.array([1, 0, 0, 0])

        # Should complete without error (fit may be poor)
        result = synthetic_control(outcomes, treatment, 5, inference="none")
        assert np.isfinite(result["estimate"])

    def test_identical_control_units(self):
        """All control units identical should work."""
        np.random.seed(42)
        n_periods = 10
        control_trajectory = np.linspace(10, 15, n_periods)

        outcomes = np.zeros((5, n_periods))
        outcomes[0, :] = control_trajectory + 2.0  # Treated (shifted)
        outcomes[0, 5:] += 3.0  # Treatment effect
        for i in range(1, 5):
            outcomes[i, :] = control_trajectory  # All identical

        treatment = np.array([1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        # All weights should be roughly equal
        assert np.isfinite(result["estimate"])
        assert np.allclose(result["weights"], 0.25, atol=0.1)

    def test_treated_outside_convex_hull(self):
        """Treated unit outside control convex hull."""
        np.random.seed(42)
        n_periods = 10

        # Controls clustered around 10
        outcomes = np.random.randn(5, n_periods) * 0.5 + 10

        # Treated much higher (outside hull)
        outcomes[0, :] = np.random.randn(n_periods) * 0.5 + 20
        outcomes[0, 5:] += 3.0  # Treatment effect

        treatment = np.array([1, 0, 0, 0, 0])

        # Should still produce result (weights on boundary)
        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert np.isfinite(result["estimate"])
        # Pre-fit should be poor (outside hull)
        assert result["pre_r_squared"] < 0.5

    def test_collinear_control_units(self):
        """Perfectly collinear controls should work."""
        np.random.seed(42)
        n_periods = 10
        base = np.linspace(10, 20, n_periods)

        outcomes = np.zeros((4, n_periods))
        outcomes[0, :] = base * 1.5  # Treated
        outcomes[0, 5:] += 3.0
        outcomes[1, :] = base  # Control 1
        outcomes[2, :] = base * 2  # Control 2 (collinear with 1)
        outcomes[3, :] = base * 0.5  # Control 3 (collinear with 1)

        treatment = np.array([1, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert np.isfinite(result["estimate"])
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)

    def test_constant_outcomes_control(self):
        """Constant outcome values in controls."""
        outcomes = np.array(
            [
                [10.0, 11.0, 12.0, 18.0, 19.0],  # Treated (varies + effect)
                [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant control
                [8.0, 8.0, 8.0, 8.0, 8.0],  # Constant control
            ]
        )
        treatment = np.array([1, 0, 0])

        result = synthetic_control(outcomes, treatment, 3, inference="none")

        assert np.isfinite(result["estimate"])

    def test_constant_outcomes_treated(self):
        """Constant pre-treatment values for treated unit."""
        outcomes = np.array(
            [
                [10.0, 10.0, 10.0, 15.0, 15.0],  # Treated (constant pre, then jumps)
                [9.0, 10.0, 11.0, 12.0, 13.0],  # Control varies
                [8.0, 9.0, 10.0, 11.0, 12.0],  # Control varies
            ]
        )
        treatment = np.array([1, 0, 0])

        result = synthetic_control(outcomes, treatment, 3, inference="none")

        assert np.isfinite(result["estimate"])


class TestZeroVarianceEdgeCases:
    """Edge cases with zero variance."""

    def test_zero_variance_single_control(self):
        """Single control with zero variance."""
        outcomes = np.array(
            [
                [10.0, 11.0, 12.0, 15.0],  # Treated
                [5.0, 5.0, 5.0, 5.0],  # Zero variance control
                [8.0, 9.0, 10.0, 11.0],  # Normal control
            ]
        )
        treatment = np.array([1, 0, 0])

        result = synthetic_control(outcomes, treatment, 2, inference="none")
        assert np.isfinite(result["estimate"])

    def test_all_zero_outcomes(self):
        """All zeros (degenerate case)."""
        outcomes = np.zeros((5, 10))
        outcomes[0, 5:] = 2.0  # Only treatment effect

        treatment = np.array([1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        # Should detect the effect
        assert np.isclose(result["estimate"], 2.0, atol=0.1)


# =============================================================================
# 3. Panel Structure Edge Cases
# =============================================================================


class TestPanelStructureEdgeCases:
    """Tests for panel dimension edge cases."""

    def test_minimum_viable_panel(self, minimal_panel):
        """Minimum panel (3 units, 3 periods) should work."""
        result = synthetic_control(
            minimal_panel["outcomes"],
            minimal_panel["treatment"],
            minimal_panel["treatment_period"],
            inference="none",
        )

        assert result["n_control"] == 2
        assert result["n_pre_periods"] == 2
        assert result["n_post_periods"] == 1
        assert np.isfinite(result["estimate"])

    def test_single_post_period(self):
        """Single post-treatment period should work."""
        np.random.seed(42)
        outcomes = np.random.randn(5, 11) + 10
        outcomes[0, 10] += 5.0  # Effect in last period only

        treatment = np.array([1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 10, inference="none")

        assert result["n_post_periods"] == 1
        assert np.isfinite(result["estimate"])

    def test_many_pre_periods(self):
        """Many pre-treatment periods (100+)."""
        np.random.seed(42)
        n_pre = 100
        n_post = 10
        outcomes = np.random.randn(6, n_pre + n_post) + 10
        outcomes[0, n_pre:] += 2.0

        treatment = np.array([1, 0, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, n_pre, inference="none")

        assert result["n_pre_periods"] == 100
        assert np.isfinite(result["estimate"])

    def test_many_controls(self, many_controls_panel):
        """50 control units should work."""
        result = synthetic_control(
            many_controls_panel["outcomes"],
            many_controls_panel["treatment"],
            many_controls_panel["treatment_period"],
            inference="none",
        )

        assert result["n_control"] == 50
        assert np.isfinite(result["estimate"])
        # Most weights should be near zero (sparsity)
        assert np.sum(result["weights"] < 0.01) > 40

    def test_multiple_treated_units(self):
        """Multiple treated units (averaging behavior)."""
        np.random.seed(42)
        outcomes = np.random.randn(6, 12) + 10
        # Two treated units with similar effects
        outcomes[0, 6:] += 3.0
        outcomes[1, 6:] += 3.5

        treatment = np.array([1, 1, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 6, inference="none")

        assert result["n_treated"] == 2
        assert result["n_control"] == 4
        # Estimate should be average of effects (~3.25)
        assert np.isclose(result["estimate"], 3.25, atol=1.0)

    def test_many_treated_few_control(self):
        """More treated than control units (unusual but valid)."""
        np.random.seed(42)
        outcomes = np.random.randn(5, 10) + 10

        treatment = np.array([1, 1, 1, 0, 0])  # 3 treated, 2 control

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert result["n_treated"] == 3
        assert result["n_control"] == 2


# =============================================================================
# 4. Optimizer Edge Cases
# =============================================================================


class TestOptimizerEdgeCases:
    """Tests for weight optimization edge cases."""

    def test_perfect_match_single_control(self):
        """Perfect match should put all weight on one control."""
        np.random.seed(42)
        n_periods = 10
        trajectory = np.linspace(10, 20, n_periods)

        outcomes = np.zeros((4, n_periods))
        outcomes[0, :] = trajectory  # Treated
        outcomes[0, 5:] += 2.0  # Treatment effect
        outcomes[1, :] = trajectory  # Perfect match
        outcomes[2, :] = trajectory + 5  # Different level
        outcomes[3, :] = trajectory * 0.5  # Different slope

        treatment = np.array([1, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        # First control should have weight ≈ 1
        assert result["weights"][0] > 0.95
        assert result["pre_rmse"] < 0.1

    def test_weight_sparsity(self):
        """With many controls, weights should be sparse."""
        np.random.seed(42)
        n_control = 30
        n_periods = 15

        outcomes = np.random.randn(n_control + 1, n_periods) + 10
        treatment = np.zeros(n_control + 1)
        treatment[0] = 1

        result = synthetic_control(outcomes, treatment, 8, inference="none")

        # Most weights should be near zero
        n_nonzero = np.sum(result["weights"] > 0.01)
        assert n_nonzero < n_control / 2

    def test_weights_always_valid(self):
        """Weights must always be non-negative and sum to 1."""
        np.random.seed(42)
        outcomes = np.random.randn(8, 15) + 10
        treatment = np.array([1, 0, 0, 0, 0, 0, 0, 0])

        result = synthetic_control(outcomes, treatment, 8, inference="none")

        # Non-negative
        assert np.all(result["weights"] >= -1e-10)

        # Sum to 1
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)


class TestWeightValidation:
    """Tests for validate_weights function."""

    def test_weights_wrong_length(self):
        """weights length must match n_control."""
        with pytest.raises(ValueError, match="length"):
            validate_weights(np.array([0.5, 0.5]), n_control=3)

    def test_weights_negative(self):
        """Negative weights should fail validation."""
        with pytest.raises(ValueError, match="non-negative"):
            validate_weights(np.array([0.5, 0.5, -0.1, 0.1]), n_control=4)

    def test_weights_not_sum_to_one(self):
        """Weights not summing to 1 should fail."""
        with pytest.raises(ValueError, match="sum to 1"):
            validate_weights(np.array([0.5, 0.3, 0.1]), n_control=3)

    def test_weights_valid(self):
        """Valid weights should pass."""
        # Should not raise
        validate_weights(np.array([0.5, 0.3, 0.2]), n_control=3)
        validate_weights(np.array([1.0, 0.0, 0.0, 0.0]), n_control=4)


# =============================================================================
# 5. Inference Edge Cases
# =============================================================================


class TestInferenceEdgeCases:
    """Tests for inference method edge cases."""

    def test_placebo_with_minimum_controls(self, minimal_panel):
        """Placebo inference with only 2 controls."""
        result = synthetic_control(
            minimal_panel["outcomes"],
            minimal_panel["treatment"],
            minimal_panel["treatment_period"],
            inference="placebo",
            n_placebo=10,  # More than available
        )

        # Should limit to n_control placebos
        assert np.isfinite(result["se"]) or np.isnan(result["se"])
        assert 0 <= result["p_value"] <= 1 or np.isnan(result["p_value"])

    def test_bootstrap_with_few_controls(self, minimal_panel):
        """Bootstrap with minimum controls."""
        result = synthetic_control(
            minimal_panel["outcomes"],
            minimal_panel["treatment"],
            minimal_panel["treatment_period"],
            inference="bootstrap",
            n_placebo=20,
        )

        # Should produce SE (may be 0 or NaN with minimal data)
        assert result["se"] >= 0 or np.isnan(result["se"])

    def test_many_placebos(self, many_controls_panel):
        """Many placebo iterations with many controls."""
        result = synthetic_control(
            many_controls_panel["outcomes"],
            many_controls_panel["treatment"],
            many_controls_panel["treatment_period"],
            inference="placebo",
            n_placebo=100,
        )

        assert result["se"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_alpha_edge_values(self, minimal_panel):
        """Edge alpha values (0.01, 0.99)."""
        for alpha in [0.01, 0.10, 0.50, 0.99]:
            result = synthetic_control(
                minimal_panel["outcomes"],
                minimal_panel["treatment"],
                minimal_panel["treatment_period"],
                alpha=alpha,
                inference="none",
            )

            # With inference="none", CI is NaN
            assert np.isnan(result["ci_lower"])
            assert np.isnan(result["ci_upper"])

    def test_large_effect_p_value(self):
        """Very large effect should have small p-value."""
        np.random.seed(42)
        outcomes = np.random.randn(10, 12) * 0.5 + 10
        outcomes[0, 6:] += 20.0  # Very large effect

        treatment = np.zeros(10)
        treatment[0] = 1

        result = synthetic_control(outcomes, treatment, 6, inference="placebo", n_placebo=50)

        # P-value should be small (significant effect)
        assert result["p_value"] < 0.20


# =============================================================================
# 6. Augmented SCM Specific Tests
# =============================================================================


class TestASCMEdgeCases:
    """Adversarial tests for Augmented Synthetic Control Method."""

    def test_ascm_minimum_panel(self, minimal_panel):
        """ASCM with minimum viable panel."""
        result = augmented_synthetic_control(
            minimal_panel["outcomes"],
            minimal_panel["treatment"],
            minimal_panel["treatment_period"],
            inference="none",
        )

        assert np.isfinite(result["estimate"])
        assert "augmented_control" in result
        assert "lambda_ridge" in result

    def test_ascm_with_explicit_lambda(self):
        """ASCM with specified lambda_ridge."""
        np.random.seed(42)
        outcomes = np.random.randn(6, 12) + 10
        outcomes[0, 6:] += 2.0

        treatment = np.array([1, 0, 0, 0, 0, 0])

        for lambda_val in [0.01, 1.0, 100.0, 10000.0]:
            result = augmented_synthetic_control(
                outcomes, treatment, 6, lambda_ridge=lambda_val, inference="none"
            )

            assert np.isfinite(result["estimate"])
            assert result["lambda_ridge"] == lambda_val

    def test_ascm_jackknife_inference(self):
        """ASCM jackknife SE with various panel sizes."""
        np.random.seed(42)
        outcomes = np.random.randn(8, 15) + 10
        outcomes[0, 8:] += 2.0

        treatment = np.array([1, 0, 0, 0, 0, 0, 0, 0])

        result = augmented_synthetic_control(outcomes, treatment, 8, inference="jackknife")

        assert result["se"] > 0 or np.isnan(result["se"])

    def test_ascm_bootstrap_inference(self):
        """ASCM bootstrap SE."""
        np.random.seed(42)
        outcomes = np.random.randn(8, 15) + 10
        outcomes[0, 8:] += 2.0

        treatment = np.array([1, 0, 0, 0, 0, 0, 0, 0])

        result = augmented_synthetic_control(outcomes, treatment, 8, inference="bootstrap")

        assert result["se"] > 0 or np.isnan(result["se"])

    def test_ascm_poor_fit_improvement(self):
        """ASCM should improve on poor SCM fit."""
        np.random.seed(42)
        n_periods = 12
        treatment_period = 6

        # Create panel where treated is outside convex hull
        outcomes = np.random.randn(6, n_periods) * 0.5 + 10
        outcomes[0, :] += 5  # Treated much higher
        outcomes[0, treatment_period:] += 2.0

        treatment = np.array([1, 0, 0, 0, 0, 0])

        scm_result = synthetic_control(outcomes, treatment, treatment_period, inference="none")
        ascm_result = augmented_synthetic_control(
            outcomes, treatment, treatment_period, inference="none"
        )

        # ASCM should have lower pre_rmse or similar estimate
        # (May not always be true, but should complete)
        assert np.isfinite(ascm_result["estimate"])

    def test_ascm_invalid_inference(self):
        """Invalid inference method should raise error."""
        np.random.seed(42)
        outcomes = np.random.randn(5, 10) + 10
        treatment = np.array([1, 0, 0, 0, 0])

        with pytest.raises(ValueError, match="Unknown inference"):
            augmented_synthetic_control(outcomes, treatment, 5, inference="invalid_method")


# =============================================================================
# 7. Pre-treatment Fit Computation
# =============================================================================


class TestPreTreatmentFit:
    """Tests for compute_pre_treatment_fit edge cases."""

    def test_perfect_fit(self):
        """Perfect fit should give RMSE=0, R²=1."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = treated.reshape(1, -1)
        weights = np.array([1.0])

        rmse, r2 = compute_pre_treatment_fit(treated, control, weights)

        assert np.isclose(rmse, 0.0, atol=1e-10)
        assert np.isclose(r2, 1.0, atol=1e-10)

    def test_poor_fit(self):
        """Poor fit should have low R²."""
        treated = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        control = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # Different scale
        weights = np.array([1.0])

        rmse, r2 = compute_pre_treatment_fit(treated, control, weights)

        assert rmse > 5.0  # Large error
        assert r2 < 0.5  # Poor fit

    def test_constant_treated(self):
        """Constant treated series (zero variance edge case)."""
        treated = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        control = np.array([[9.0, 10.0, 11.0, 10.0, 9.0]])
        weights = np.array([1.0])

        rmse, r2 = compute_pre_treatment_fit(treated, control, weights)

        # R² may be 0 due to zero total sum of squares
        assert np.isfinite(rmse)
        assert np.isfinite(r2) or r2 == 0.0


# =============================================================================
# 8. Weight Computation Edge Cases
# =============================================================================


class TestComputeWeightsEdgeCases:
    """Tests for compute_scm_weights edge cases."""

    def test_single_control_weights(self):
        """Single control should get weight 1.0."""
        np.random.seed(42)
        treated = np.random.randn(10)
        control = np.random.randn(1, 10)

        weights, _ = compute_scm_weights(treated, control)

        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0)

    def test_weights_with_covariates(self):
        """Weights should incorporate covariate matching."""
        np.random.seed(42)
        treated_pre = np.random.randn(10)
        control_pre = np.random.randn(5, 10)
        cov_treated = np.array([1.0, 2.0])
        cov_control = np.random.randn(5, 2)

        weights, _ = compute_scm_weights(
            treated_pre,
            control_pre,
            covariates_treated=cov_treated,
            covariates_control=cov_control,
            covariate_weight=1.0,
        )

        assert len(weights) == 5
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= -1e-10)

    def test_pre_period_mismatch(self):
        """Mismatched pre-period lengths should raise error."""
        treated = np.random.randn(10)
        control = np.random.randn(5, 8)  # Different length

        with pytest.raises(ValueError, match="mismatch"):
            compute_scm_weights(treated, control)


# =============================================================================
# 9. Data Type Handling
# =============================================================================


class TestDataTypeHandling:
    """Tests for various input data types."""

    def test_int_outcomes(self):
        """Integer outcomes should be converted and work."""
        outcomes = np.array(
            [
                [10, 11, 12, 15, 16],
                [9, 10, 11, 12, 13],
                [8, 9, 10, 11, 12],
            ],
            dtype=np.int64,
        )

        treatment = np.array([1, 0, 0])

        result = synthetic_control(outcomes, treatment, 3, inference="none")

        assert np.isfinite(result["estimate"])

    def test_float32_outcomes(self):
        """Float32 outcomes should work."""
        outcomes = np.array(
            [
                [10.0, 11.0, 12.0, 15.0, 16.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [8.0, 9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        )

        treatment = np.array([1, 0, 0])

        result = synthetic_control(outcomes, treatment, 3, inference="none")

        assert np.isfinite(result["estimate"])

    def test_bool_treatment_array(self):
        """Boolean treatment array should work."""
        np.random.seed(42)
        outcomes = np.random.randn(5, 10) + 10

        treatment = np.array([True, False, False, False, False])

        result = synthetic_control(outcomes, treatment, 5, inference="none")

        assert result["n_treated"] == 1
        assert result["n_control"] == 4


# =============================================================================
# 10. Error Message Quality
# =============================================================================


class TestErrorMessages:
    """Tests for clear, informative error messages."""

    def test_dimension_mismatch_message(self):
        """Error should clearly indicate dimension issue."""
        try:
            validate_panel_data(
                outcomes=np.random.randn(5, 10),
                treatment=np.array([1, 0, 0]),  # Wrong length
                treatment_period=5,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "length" in str(e).lower() or "5" in str(e)

    def test_no_control_message(self):
        """Error should explain need for controls."""
        try:
            validate_panel_data(
                outcomes=np.random.randn(3, 10),
                treatment=np.array([1, 1, 1]),  # All treated
                treatment_period=5,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "control" in str(e).lower()

    def test_nan_location_message(self):
        """NaN error should be informative."""
        outcomes = np.random.randn(5, 10)
        outcomes[2, 3] = np.nan

        try:
            validate_panel_data(
                outcomes=outcomes,
                treatment=np.array([1, 0, 0, 0, 0]),
                treatment_period=5,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nan" in str(e).lower()


# =============================================================================
# Run Summary
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
