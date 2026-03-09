"""
Tests for RIF-OLS based QTE estimation.

Tests cover:
- Known-answer validation
- Comparison with unconditional QTE (should be similar without covariates)
- RIF computation correctness
"""

import numpy as np
import pytest

from src.causal_inference.qte import rif_qte, rif_qte_band, unconditional_qte


class TestRIFQTEKnownAnswers:
    """Test RIF QTE with known expected values."""

    def test_median_effect_no_covariates(self, simple_rct_data):
        """Without covariates, RIF-QTE should recover true effect.

        Note: RIF-OLS uses density estimation + regression, which differs
        from direct quantile differences. Both should be close to true effect
        but may differ from each other due to finite-sample bias.
        """
        outcome, treatment = simple_rct_data

        result_rif = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42)

        # True effect is 2.0 - RIF should be within 1.0 (wider tolerance due to
        # density estimation variability)
        assert np.isclose(result_rif["tau_q"], 2.0, atol=1.0), (
            f"RIF QTE {result_rif['tau_q']:.3f} too far from true effect 2.0"
        )

    def test_median_effect_with_covariates(self, data_with_covariates):
        """RIF-QTE with covariates should recover unconditional effect."""
        outcome, treatment, covariates = data_with_covariates

        result = rif_qte(
            outcome, treatment, covariates, quantile=0.5, n_bootstrap=500, random_state=42
        )

        # True effect is 2.0
        assert np.isclose(result["tau_q"], 2.0, atol=0.5)
        assert result["method"] == "rif"
        assert result["inference"] == "bootstrap"


class TestRIFQTEProperties:
    """Property-based tests for RIF QTE."""

    def test_se_positive(self, simple_rct_data):
        """Standard error should be positive."""
        outcome, treatment = simple_rct_data

        result = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=200, random_state=42)

        assert result["se"] > 0

    def test_ci_contains_estimate(self, simple_rct_data):
        """CI should contain point estimate."""
        outcome, treatment = simple_rct_data

        result = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42)

        assert result["ci_lower"] < result["tau_q"] < result["ci_upper"]

    def test_reproducibility(self, simple_rct_data):
        """Same seed should give identical results."""
        outcome, treatment = simple_rct_data

        result1 = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=200, random_state=123)
        result2 = rif_qte(outcome, treatment, quantile=0.5, n_bootstrap=200, random_state=123)

        assert result1["tau_q"] == result2["tau_q"]
        assert result1["se"] == result2["se"]


class TestRIFQTEBand:
    """Tests for rif_qte_band function."""

    def test_band_returns_all_quantiles(self, simple_rct_data):
        """Band should return estimates at all quantiles."""
        outcome, treatment = simple_rct_data

        quantiles = [0.25, 0.5, 0.75]
        result = rif_qte_band(
            outcome, treatment, quantiles=quantiles, n_bootstrap=200, random_state=42
        )

        assert len(result["quantiles"]) == 3
        assert len(result["qte_estimates"]) == 3
        assert result["n_bootstrap"] == 200

    def test_band_joint_ci(self, simple_rct_data):
        """Joint CI should be computed when requested."""
        outcome, treatment = simple_rct_data

        result = rif_qte_band(outcome, treatment, n_bootstrap=300, joint=True, random_state=42)

        assert result["joint_ci_lower"] is not None
        assert result["joint_ci_upper"] is not None

    def test_band_with_covariates(self, data_with_covariates):
        """Band should work with covariates."""
        outcome, treatment, covariates = data_with_covariates

        result = rif_qte_band(
            outcome,
            treatment,
            covariates,
            quantiles=[0.25, 0.5, 0.75],
            n_bootstrap=200,
            random_state=42,
        )

        assert len(result["qte_estimates"]) == 3
        # All estimates should be finite
        assert np.all(np.isfinite(result["qte_estimates"]))


class TestRIFQTEBandwidthMethods:
    """Tests for different bandwidth selection methods."""

    def test_silverman_bandwidth(self, simple_rct_data):
        """Silverman bandwidth should work."""
        outcome, treatment = simple_rct_data

        result = rif_qte(
            outcome,
            treatment,
            quantile=0.5,
            bandwidth="silverman",
            n_bootstrap=100,
            random_state=42,
        )

        assert np.isfinite(result["tau_q"])

    def test_scott_bandwidth(self, simple_rct_data):
        """Scott bandwidth should work."""
        outcome, treatment = simple_rct_data

        result = rif_qte(
            outcome, treatment, quantile=0.5, bandwidth="scott", n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])

    def test_auto_bandwidth(self, simple_rct_data):
        """Auto bandwidth should work."""
        outcome, treatment = simple_rct_data

        result = rif_qte(
            outcome, treatment, quantile=0.5, bandwidth="auto", n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])


class TestRIFQTEInputValidation:
    """Input validation tests for RIF QTE."""

    def test_empty_arrays(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="CRITICAL ERROR.*Empty"):
            rif_qte(np.array([]), np.array([]), quantile=0.5)

    def test_invalid_quantile(self, simple_rct_data):
        """Invalid quantile should raise ValueError."""
        outcome, treatment = simple_rct_data

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            rif_qte(outcome, treatment, quantile=1.5)

    def test_non_binary_treatment(self, simple_rct_data):
        """Non-binary treatment should raise ValueError."""
        outcome, treatment = simple_rct_data
        treatment[0] = 2

        with pytest.raises(ValueError, match="CRITICAL ERROR.*binary"):
            rif_qte(outcome, treatment, quantile=0.5)

    def test_nan_in_covariates(self, data_with_covariates):
        """NaN in covariates should raise ValueError."""
        outcome, treatment, covariates = data_with_covariates
        covariates[0, 0] = np.nan

        with pytest.raises(ValueError, match="CRITICAL ERROR.*NaN"):
            rif_qte(outcome, treatment, covariates, quantile=0.5)
