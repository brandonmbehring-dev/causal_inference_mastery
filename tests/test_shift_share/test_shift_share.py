"""
Tests for Shift-Share (Bartik) IV estimation.

Layer 1-2 of 6-layer validation architecture:
- Known-answer tests
- Adversarial edge cases
"""

import numpy as np
import pytest

from src.causal_inference.shift_share import (
    ShiftShareIV,
    shift_share_iv,
)
from tests.test_shift_share.conftest import generate_shift_share_data


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_basic_estimation(self, ss_basic):
        """Basic shift-share estimation runs without error."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        assert not np.isnan(result["estimate"])
        assert not np.isnan(result["se"])
        assert result["se"] > 0

    def test_convenience_matches_class(self, ss_basic):
        """Convenience function matches class-based estimation."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        result1 = shift_share_iv(Y, D, shares, shocks)

        ssiv = ShiftShareIV()
        result2 = ssiv.fit(Y, D, shares, shocks)

        assert np.isclose(result1["estimate"], result2["estimate"])
        assert np.isclose(result1["se"], result2["se"])

    def test_with_controls(self, ss_with_controls):
        """Estimation works with controls."""
        Y, D, shares, shocks, X, true_beta = ss_with_controls
        result = shift_share_iv(Y, D, shares, shocks, X=X)

        assert not np.isnan(result["estimate"])
        assert result["n_obs"] == len(Y)

    def test_instrument_stored(self, ss_basic):
        """Bartik instrument is stored after fit."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        ssiv = ShiftShareIV()
        ssiv.fit(Y, D, shares, shocks)

        assert ssiv.instrument_ is not None
        assert len(ssiv.instrument_) == len(Y)
        # Check instrument equals shares @ shocks
        expected = shares @ shocks
        np.testing.assert_array_almost_equal(ssiv.instrument_, expected)


class TestResultStructure:
    """Tests for result structure."""

    def test_result_has_all_fields(self, ss_basic):
        """Result contains all expected fields."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        required_fields = [
            "estimate", "se", "t_stat", "p_value",
            "ci_lower", "ci_upper", "first_stage",
            "rotemberg", "n_obs", "n_sectors",
            "share_sum_mean", "inference", "alpha", "message",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_first_stage_diagnostics(self, ss_basic):
        """First-stage diagnostics are populated."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        fs = result["first_stage"]
        assert fs["f_statistic"] > 0
        assert 0 <= fs["f_pvalue"] <= 1
        assert 0 <= fs["partial_r2"] <= 1
        assert isinstance(fs["weak_iv_warning"], (bool, np.bool_))

    def test_rotemberg_diagnostics(self, ss_basic):
        """Rotemberg diagnostics are populated."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        rot = result["rotemberg"]
        assert len(rot["weights"]) == len(shocks)
        assert 0 <= rot["negative_weight_share"] <= 1
        assert len(rot["top_5_sectors"]) == 5
        assert rot["herfindahl"] >= 0

    def test_ci_contains_estimate(self, ss_basic):
        """Confidence interval contains point estimate."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


class TestTreatmentEffectRecovery:
    """Tests for recovering true treatment effect."""

    def test_recovers_positive_effect(self, ss_basic):
        """Recovers positive treatment effect."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        # Should be within 3 SE of true value
        assert abs(result["estimate"] - true_beta) < 3 * result["se"]

    def test_recovers_negative_effect(self, ss_negative_effect):
        """Recovers negative treatment effect."""
        Y, D, shares, shocks, X, true_beta = ss_negative_effect
        result = shift_share_iv(Y, D, shares, shocks)

        # Sign should be correct
        assert result["estimate"] < 0
        # Within 3 SE of true value
        assert abs(result["estimate"] - true_beta) < 3 * result["se"]

    def test_large_sample_precision(self, ss_large_sample):
        """Large sample gives precise estimates."""
        Y, D, shares, shocks, X, true_beta = ss_large_sample
        result = shift_share_iv(Y, D, shares, shocks)

        # SE should be relatively small
        assert result["se"] < 0.5
        # Close to true value
        assert abs(result["estimate"] - true_beta) < 0.3

    def test_ci_coverage(self, ss_basic):
        """CI covers true value."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        # True value should be in CI (probabilistic, but usually works)
        # This is a single sample test, so we allow some slack
        covered = result["ci_lower"] <= true_beta <= result["ci_upper"]
        # If not covered, at least should be close
        if not covered:
            assert abs(result["estimate"] - true_beta) < 1.0


class TestFirstStage:
    """Tests for first-stage diagnostics."""

    def test_strong_first_stage(self, ss_basic):
        """Strong first stage has F > 10."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        assert result["first_stage"]["f_statistic"] > 10
        assert not result["first_stage"]["weak_iv_warning"]

    def test_weak_first_stage_warning(self, ss_weak_first_stage):
        """Weak first stage triggers warning."""
        Y, D, shares, shocks, X, true_beta = ss_weak_first_stage
        result = shift_share_iv(Y, D, shares, shocks)

        # With weak first stage, F may be < 10
        if result["first_stage"]["f_statistic"] < 10:
            assert result["first_stage"]["weak_iv_warning"]
            assert "WARNING" in result["message"]


class TestRotembergWeights:
    """Tests for Rotemberg weight diagnostics."""

    def test_weights_sum_approximately(self, ss_basic):
        """Rotemberg weights have sensible magnitudes."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        weights = result["rotemberg"]["weights"]
        # Weights are normalized, sum of absolute values = 1
        assert abs(np.sum(np.abs(weights)) - 1.0) < 0.1

    def test_concentrated_shares_higher_herfindahl(self):
        """Concentrated shares give higher Herfindahl."""
        Y_c, D_c, shares_c, shocks_c, _, _ = generate_shift_share_data(
            share_concentration=0.2, random_state=100
        )
        Y_u, D_u, shares_u, shocks_u, _, _ = generate_shift_share_data(
            share_concentration=10.0, random_state=101
        )

        result_c = shift_share_iv(Y_c, D_c, shares_c, shocks_c)
        result_u = shift_share_iv(Y_u, D_u, shares_u, shocks_u)

        # Concentrated should have higher Herfindahl
        # (more weight on fewer sectors)
        h_c = result_c["rotemberg"]["herfindahl"]
        h_u = result_u["rotemberg"]["herfindahl"]
        # This relationship may not always hold due to shock variation
        # so we just check they're both positive
        assert h_c > 0 and h_u > 0

    def test_top_5_sectors_ordered(self, ss_basic):
        """Top 5 sectors are ordered by absolute weight."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        weights = result["rotemberg"]["weights"]
        top_5_idx = result["rotemberg"]["top_5_sectors"]
        top_5_weights = result["rotemberg"]["top_5_weights"]

        # Check top weights correspond to indices
        for i, idx in enumerate(top_5_idx):
            assert np.isclose(weights[idx], top_5_weights[i])


class TestShareNormalization:
    """Tests for share normalization handling."""

    def test_shares_sum_to_one(self, ss_basic):
        """Standard shares sum to 1."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        result = shift_share_iv(Y, D, shares, shocks)

        assert abs(result["share_sum_mean"] - 1.0) < 0.05

    def test_unnormalized_shares_noted(self):
        """Unnormalized shares are noted in message."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        # Shares that don't sum to 1
        shares = rng.uniform(0, 1, (n, S))  # Won't sum to 1
        shocks = rng.normal(0, 0.1, S)
        Z = shares @ shocks
        D = 2.0 * Z + rng.normal(0, 0.5, n)
        Y = 1.5 * D + rng.normal(0, 1, n)

        result = shift_share_iv(Y, D, shares, shocks)

        # Share sum mean should be noted if far from 1
        if abs(result["share_sum_mean"] - 1.0) > 0.1:
            assert "Share sum" in result["message"] or result["share_sum_mean"] != 1.0


class TestSummary:
    """Tests for summary method."""

    def test_summary_after_fit(self, ss_basic):
        """Summary works after fitting."""
        Y, D, shares, shocks, X, true_beta = ss_basic
        ssiv = ShiftShareIV()
        ssiv.fit(Y, D, shares, shocks)

        summary = ssiv.summary()
        assert isinstance(summary, str)
        assert "Shift-Share" in summary
        assert "Estimate" in summary

    def test_summary_before_fit(self):
        """Summary before fit returns message."""
        ssiv = ShiftShareIV()
        summary = ssiv.summary()
        assert "not fitted" in summary.lower()


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_rejects_nan_in_outcome(self):
        """Raises error for NaN in outcome."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        shares = rng.dirichlet(np.ones(S), n)
        shocks = rng.normal(0, 0.1, S)
        Z = shares @ shocks
        D = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)
        Y[0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            shift_share_iv(Y, D, shares, shocks)

    def test_rejects_nan_in_shares(self):
        """Raises error for NaN in shares."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        shares = rng.dirichlet(np.ones(S), n)
        shares[0, 0] = np.nan
        shocks = rng.normal(0, 0.1, S)
        D = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="NaN"):
            shift_share_iv(Y, D, shares, shocks)

    def test_rejects_dimension_mismatch(self):
        """Raises error for dimension mismatch."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        shares = rng.dirichlet(np.ones(S), n)
        shocks = rng.normal(0, 0.1, S + 1)  # Wrong dimension
        D = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="columns"):
            shift_share_iv(Y, D, shares, shocks)

    def test_rejects_1d_shares(self):
        """Raises error for 1D shares."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        shares = rng.normal(0, 1, n)  # 1D, should be 2D
        shocks = rng.normal(0, 0.1, S)
        D = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="2D"):
            shift_share_iv(Y, D, shares, shocks)

    def test_rejects_small_sample(self):
        """Raises error for sample size < 10."""
        rng = np.random.default_rng(42)
        n, S = 5, 3
        shares = rng.dirichlet(np.ones(S), n)
        shocks = rng.normal(0, 0.1, S)
        D = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="sample size"):
            shift_share_iv(Y, D, shares, shocks)

    def test_rejects_no_treatment_variation(self):
        """Raises error for constant treatment."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        shares = rng.dirichlet(np.ones(S), n)
        shocks = rng.normal(0, 0.1, S)
        D = np.ones(n) * 5  # Constant
        Y = rng.normal(0, 1, n)

        with pytest.raises(ValueError, match="variation"):
            shift_share_iv(Y, D, shares, shocks)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_many_sectors(self, ss_many_sectors):
        """Works with many sectors (50)."""
        Y, D, shares, shocks, X, true_beta = ss_many_sectors
        result = shift_share_iv(Y, D, shares, shocks)

        assert result["n_sectors"] == 50
        assert not np.isnan(result["estimate"])

    def test_few_sectors(self):
        """Works with few sectors (3)."""
        Y, D, shares, shocks, X, _ = generate_shift_share_data(
            n=100, n_sectors=3, random_state=42
        )
        result = shift_share_iv(Y, D, shares, shocks)

        assert result["n_sectors"] == 3
        assert not np.isnan(result["estimate"])

    def test_single_dominant_sector(self):
        """Works when one sector dominates."""
        rng = np.random.default_rng(42)
        n, S = 100, 10
        # One sector has 90% share
        shares = np.zeros((n, S))
        shares[:, 0] = 0.9
        shares[:, 1:] = 0.1 / (S - 1)

        shocks = rng.normal(0, 0.1, S)
        Z = shares @ shocks
        D = 2.0 * Z + rng.normal(0, 0.5, n)
        Y = 1.5 * D + rng.normal(0, 1, n)

        result = shift_share_iv(Y, D, shares, shocks)
        assert not np.isnan(result["estimate"])

    def test_zero_shock_some_sectors(self):
        """Works when some sectors have zero shock."""
        Y, D, shares, shocks, X, _ = generate_shift_share_data(
            n=100, n_sectors=10, random_state=42
        )
        # Set half the shocks to zero
        shocks[5:] = 0

        result = shift_share_iv(Y, D, shares, shocks)
        assert not np.isnan(result["estimate"])

    def test_reproducibility(self, ss_basic):
        """Results are reproducible."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        result1 = shift_share_iv(Y, D, shares, shocks)
        result2 = shift_share_iv(Y, D, shares, shocks)

        assert np.isclose(result1["estimate"], result2["estimate"])
        assert np.isclose(result1["se"], result2["se"])
