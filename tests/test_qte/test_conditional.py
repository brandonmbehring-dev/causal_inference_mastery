"""
Tests for conditional QTE via quantile regression.

Tests cover:
- Known-answer validation
- Property tests
- Comparison with unconditional QTE
"""

import numpy as np
import pytest

from src.causal_inference.qte import conditional_qte, conditional_qte_band


class TestConditionalQTEKnownAnswers:
    """Test conditional QTE with known expected values."""

    def test_median_effect_with_covariates(self, data_with_covariates):
        """Test median treatment effect with covariates."""
        outcome, treatment, covariates = data_with_covariates

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        # True effect is 2.0, should be close
        assert np.isclose(result["tau_q"], 2.0, atol=0.5)
        assert result["quantile"] == 0.5
        assert result["method"] == "conditional"
        assert result["inference"] == "asymptotic"

    def test_pvalue_computed(self, data_with_covariates):
        """P-value should be computed for conditional QTE."""
        outcome, treatment, covariates = data_with_covariates

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert result["pvalue"] is not None
        assert 0 <= result["pvalue"] <= 1


class TestConditionalQTEProperties:
    """Property-based tests for conditional QTE."""

    def test_se_positive(self, data_with_covariates):
        """Standard error should be positive."""
        outcome, treatment, covariates = data_with_covariates

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert result["se"] > 0

    def test_ci_valid(self, data_with_covariates):
        """CI should be valid (lower < upper)."""
        outcome, treatment, covariates = data_with_covariates

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert result["ci_lower"] < result["ci_upper"]

    def test_1d_covariates(self, simple_rct_data):
        """Should handle 1D covariate array."""
        outcome, treatment = simple_rct_data
        covariates = np.random.normal(0, 1, len(outcome))

        result = conditional_qte(outcome, treatment, covariates, quantile=0.5)

        assert np.isfinite(result["tau_q"])


class TestConditionalQTEBand:
    """Tests for conditional_qte_band function."""

    def test_band_returns_all_quantiles(self, data_with_covariates):
        """Band should return estimates at all requested quantiles."""
        outcome, treatment, covariates = data_with_covariates

        quantiles = [0.25, 0.5, 0.75]
        result = conditional_qte_band(outcome, treatment, covariates, quantiles=quantiles)

        assert len(result["quantiles"]) == 3
        assert len(result["qte_estimates"]) == 3
        assert result["n_bootstrap"] == 0  # Asymptotic inference

    def test_band_homogeneous_effect(self, data_with_covariates):
        """With homogeneous DGP, QTE should be similar across quantiles."""
        outcome, treatment, covariates = data_with_covariates

        result = conditional_qte_band(outcome, treatment, covariates, quantiles=[0.25, 0.5, 0.75])

        # Effects should be similar (within 1.0)
        assert np.std(result["qte_estimates"]) < 1.0


class TestConditionalQTEInputValidation:
    """Input validation tests."""

    def test_empty_arrays(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="CRITICAL ERROR.*Empty"):
            conditional_qte(np.array([]), np.array([]), np.array([]).reshape(-1, 1), quantile=0.5)

    def test_invalid_quantile(self, data_with_covariates):
        """Invalid quantile should raise ValueError."""
        outcome, treatment, covariates = data_with_covariates

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            conditional_qte(outcome, treatment, covariates, quantile=1.5)

    def test_non_binary_treatment(self, data_with_covariates):
        """Non-binary treatment should raise ValueError."""
        outcome, treatment, covariates = data_with_covariates
        treatment[0] = 2  # Make non-binary

        with pytest.raises(ValueError, match="CRITICAL ERROR.*binary"):
            conditional_qte(outcome, treatment, covariates, quantile=0.5)

    def test_length_mismatch_covariates(self, data_with_covariates):
        """Mismatched covariate length should raise ValueError."""
        outcome, treatment, covariates = data_with_covariates

        with pytest.raises(ValueError, match="CRITICAL ERROR.*lengths"):
            conditional_qte(
                outcome,
                treatment,
                covariates[:-10],
                quantile=0.5,  # Wrong length
            )
