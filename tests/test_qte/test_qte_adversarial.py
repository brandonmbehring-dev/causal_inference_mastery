"""
Adversarial tests for QTE estimation.

Tests edge cases, boundary conditions, and numerical stability.
"""

import numpy as np
import pytest

from src.causal_inference.qte import (
    conditional_qte,
    rif_qte,
    unconditional_qte,
    unconditional_qte_band,
)


class TestExtremeQuantiles:
    """Tests for extreme quantile values."""

    def test_very_low_quantile(self):
        """Test quantile near 0 (tau = 0.01)."""
        np.random.seed(42)
        n = 500
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        result = unconditional_qte(
            outcome, treatment, quantile=0.01, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])
        # SE should be larger at extreme quantiles
        assert result["se"] > 0

    def test_very_high_quantile(self):
        """Test quantile near 1 (tau = 0.99)."""
        np.random.seed(42)
        n = 500
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        result = unconditional_qte(
            outcome, treatment, quantile=0.99, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])


class TestSmallSamples:
    """Tests with very small sample sizes."""

    def test_minimal_sample_unconditional(self):
        """Test with minimal sample (3 per group)."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        treatment = np.array([1, 1, 1, 0, 0, 0], dtype=float)

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=200, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        # Median treated = 2, median control = 5, so QTE should be negative
        # Actually treated [1,2,3] median=2, control [4,5,6] median=5
        # QTE = 2 - 5 = -3
        assert result["tau_q"] < 0

    def test_single_observation_per_group_fails(self):
        """Single observation per group should raise error."""
        outcome = np.array([1.0, 2.0])
        treatment = np.array([1, 0], dtype=float)

        with pytest.raises(ValueError, match="Insufficient"):
            unconditional_qte(outcome, treatment, quantile=0.5)


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_large_values(self):
        """Test with very large outcome values."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(1e8, 1e6, n) + 2e6 * treatment

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=300, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])

    def test_small_values(self):
        """Test with very small outcome values."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(1e-8, 1e-9, n) + 2e-9 * treatment

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=300, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])

    def test_identical_treated_outcomes(self):
        """Test when all treated outcomes are identical."""
        n = 100
        treatment = np.array([1] * 50 + [0] * 50, dtype=float)
        outcome = np.concatenate(
            [
                np.ones(50) * 5.0,  # All treated have same outcome
                np.random.normal(3, 1, 50),  # Control varies
            ]
        )

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=300, random_state=42
        )

        # Median of treated = 5, median of control ~ 3
        assert np.isfinite(result["tau_q"])
        assert result["tau_q"] > 0

    def test_high_variance_ratio(self):
        """Test when one group has much higher variance."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)

        # Treated has high variance, control has low variance
        outcome = np.where(
            treatment == 1,
            np.random.normal(5, 10, n),  # High variance
            np.random.normal(3, 0.1, n),  # Low variance
        )

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])


class TestTiedValues:
    """Tests with tied (duplicate) outcome values."""

    def test_many_ties(self):
        """Test with many tied values."""
        # Many repeated values
        outcome = np.array([1, 1, 1, 2, 2, 3, 4, 4, 4, 5] * 10, dtype=float)
        treatment = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0] * 10, dtype=float)

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=300, random_state=42
        )

        assert np.isfinite(result["tau_q"])

    def test_all_same_value(self):
        """Test when all values in one group are the same."""
        n = 50
        treatment = np.array([1] * 25 + [0] * 25, dtype=float)
        outcome = np.concatenate(
            [
                np.ones(25) * 10,  # All treated = 10
                np.random.normal(5, 1, 25),  # Control varies
            ]
        )

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=300, random_state=42
        )

        # QTE should be around 10 - 5 = 5
        assert result["tau_q"] > 0


class TestRIFEdgeCases:
    """Edge cases specific to RIF-OLS."""

    def test_rif_with_sparse_covariates(self):
        """RIF with sparse covariate matrix."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        # Sparse covariates (mostly zeros)
        covariates = np.zeros((n, 5))
        covariates[np.random.choice(n, 20), :] = np.random.normal(0, 1, (20, 5))

        result = rif_qte(
            outcome, treatment, covariates, quantile=0.5, n_bootstrap=200, random_state=42
        )

        assert np.isfinite(result["tau_q"])

    def test_rif_kde_fallback(self):
        """Test RIF when KDE might fail (e.g., near-constant data)."""
        n = 100
        treatment = np.array([1] * 50 + [0] * 50, dtype=float)
        # Nearly constant with small variation
        outcome = np.ones(n) + np.random.normal(0, 0.001, n)
        outcome[treatment == 1] += 0.5

        result = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=200, random_state=42)

        assert np.isfinite(result["tau_q"])


class TestBandEdgeCases:
    """Edge cases for band estimation."""

    def test_band_single_quantile(self):
        """Band with single quantile should work."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        result = unconditional_qte_band(
            outcome, treatment, quantiles=[0.5], n_bootstrap=200, random_state=42
        )

        assert len(result["quantiles"]) == 1
        assert len(result["qte_estimates"]) == 1

    def test_band_many_quantiles(self):
        """Band with many quantiles should work."""
        np.random.seed(42)
        n = 500
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        quantiles = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]

        result = unconditional_qte_band(
            outcome, treatment, quantiles=quantiles, n_bootstrap=200, random_state=42
        )

        assert len(result["quantiles"]) == len(quantiles)
        assert np.all(np.isfinite(result["qte_estimates"]))


class TestConditionalQTEEdgeCases:
    """Edge cases for conditional QTE."""

    def test_many_covariates(self):
        """Test with many covariates (p > n/10)."""
        np.random.seed(42)
        n = 200
        p = 30  # Many covariates
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        covariates = np.random.normal(0, 1, (n, p))
        outcome = np.random.normal(0, 1, n) + 2.0 * treatment

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert np.isfinite(result["tau_q"])

    def test_collinear_covariates(self):
        """Test with collinear covariates."""
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n).astype(float)

        # X2 is nearly collinear with X1
        x1 = np.random.normal(0, 1, n)
        x2 = x1 + np.random.normal(0, 0.01, n)  # Nearly identical
        x3 = np.random.normal(0, 1, n)
        covariates = np.column_stack([x1, x2, x3])

        outcome = x1 + 2.0 * treatment + np.random.normal(0, 1, n)

        # May have numerical issues but should still work
        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert np.isfinite(result["tau_q"])


class TestMixedInputTypes:
    """Test with different input types."""

    def test_integer_outcome(self):
        """Test with integer outcomes."""
        outcome = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # int
        treatment = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])

    def test_integer_treatment(self):
        """Test with integer treatment."""
        np.random.seed(42)
        n = 100
        outcome = np.random.normal(0, 1, n)
        treatment = np.random.binomial(1, 0.5, n)  # int, not float

        outcome = outcome + 2.0 * treatment

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])
