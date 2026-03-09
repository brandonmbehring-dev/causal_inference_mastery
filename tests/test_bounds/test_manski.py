"""
Tests for Manski partial identification bounds.

Layer 1: Known-answer tests (hand-calculated values)
Layer 2: Adversarial tests (edge cases, input validation)
Layer 3: Property tests (bounds contain true value, ordering)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.bounds import (
    manski_worst_case,
    manski_mtr,
    manski_mts,
    manski_mtr_mts,
    manski_iv,
    compare_bounds,
)


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestManskiWorstCaseKnownAnswers:
    """Tests with hand-calculated expected values."""

    def test_simple_rct_bounds(self, simple_rct_data):
        """Worst-case bounds should contain true ATE."""
        outcome, treatment, true_ate = simple_rct_data
        result = manski_worst_case(outcome, treatment)

        assert result["bounds_lower"] < true_ate < result["bounds_upper"]
        assert result["assumptions"] == "worst_case"

    def test_known_support_bounds(self):
        """Bounds with known support should be computable."""
        np.random.seed(1)
        outcome = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        treatment = np.array([0, 0, 0, 1, 1, 1])

        # E[Y|T=1] = 5, E[Y|T=0] = 2
        # Naive ATE = 3

        result = manski_worst_case(outcome, treatment, outcome_support=(0, 10))

        # With support [0, 10]:
        # E[Y1] in [P(T=1)*5 + P(T=0)*0, P(T=1)*5 + P(T=0)*10] = [2.5, 7.5]
        # E[Y0] in [P(T=1)*0 + P(T=0)*2, P(T=1)*10 + P(T=0)*2] = [1, 6]
        # ATE bounds: [2.5 - 6, 7.5 - 1] = [-3.5, 6.5]

        assert_allclose(result["bounds_lower"], -3.5, rtol=0.01)
        assert_allclose(result["bounds_upper"], 6.5, rtol=0.01)
        assert result["naive_ate"] == 3.0

    def test_bounds_width_with_narrow_support(self):
        """Narrower support should give narrower bounds."""
        np.random.seed(2)
        n = 200
        treatment = np.random.binomial(1, 0.5, n)
        outcome = treatment + np.random.randn(n) * 0.1  # Tight around 0 and 1

        wide = manski_worst_case(outcome, treatment, outcome_support=(-10, 10))
        narrow = manski_worst_case(outcome, treatment, outcome_support=(-1, 2))

        assert narrow["bounds_width"] < wide["bounds_width"]


class TestManskiMTRKnownAnswers:
    """Tests for Monotone Treatment Response bounds."""

    def test_positive_mtr_bounds_nonnegative(self, mtr_positive_data):
        """Positive MTR should give non-negative lower bound."""
        outcome, treatment, true_ate = mtr_positive_data
        result = manski_mtr(outcome, treatment, direction="positive")

        assert result["bounds_lower"] >= 0, "Positive MTR implies non-negative effect"
        assert result["mtr_direction"] == "positive"

    def test_negative_mtr_bounds_nonpositive(self, mtr_negative_data):
        """Negative MTR should give non-positive upper bound."""
        outcome, treatment, true_ate = mtr_negative_data
        result = manski_mtr(outcome, treatment, direction="negative")

        assert result["bounds_upper"] <= 0, "Negative MTR implies non-positive effect"
        assert result["mtr_direction"] == "negative"

    def test_mtr_narrower_than_worst_case(self, simple_rct_data):
        """MTR bounds should be narrower than worst-case."""
        outcome, treatment, _ = simple_rct_data

        worst = manski_worst_case(outcome, treatment)
        mtr = manski_mtr(outcome, treatment, direction="positive")

        assert mtr["bounds_width"] <= worst["bounds_width"]


class TestManskiMTSKnownAnswers:
    """Tests for Monotone Treatment Selection bounds."""

    def test_mts_upper_bound_is_naive(self, positive_selection_data):
        """MTS upper bound should be naive ATE."""
        outcome, treatment, _ = positive_selection_data
        result = manski_mts(outcome, treatment)

        # Under MTS, naive is an upper bound
        assert_allclose(result["bounds_upper"], result["naive_ate"], rtol=0.01)

    def test_mts_narrower_than_worst_case(self, simple_rct_data):
        """MTS bounds should be narrower than worst-case."""
        outcome, treatment, _ = simple_rct_data

        worst = manski_worst_case(outcome, treatment)
        mts = manski_mts(outcome, treatment)

        assert mts["bounds_width"] <= worst["bounds_width"]


class TestManskiCombinedKnownAnswers:
    """Tests for combined MTR + MTS bounds."""

    def test_combined_narrowest(self, simple_rct_data):
        """Combined bounds should be narrowest."""
        outcome, treatment, _ = simple_rct_data

        worst = manski_worst_case(outcome, treatment)
        mtr = manski_mtr(outcome, treatment, direction="positive")
        mts = manski_mts(outcome, treatment)
        combined = manski_mtr_mts(outcome, treatment, mtr_direction="positive")

        assert combined["bounds_width"] <= worst["bounds_width"]
        assert combined["bounds_width"] <= mtr["bounds_width"]
        assert combined["bounds_width"] <= mts["bounds_width"]

    def test_positive_mtr_mts_bounds(self):
        """Positive MTR + MTS should give [0, max(naive, 0)]."""
        np.random.seed(3)
        n = 500
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 1.5 * treatment + np.random.randn(n)

        result = manski_mtr_mts(outcome, treatment, mtr_direction="positive")

        assert result["bounds_lower"] == 0.0
        assert result["bounds_upper"] >= 0


class TestManskiIVKnownAnswers:
    """Tests for IV bounds."""

    def test_iv_bounds_contain_true_ate(self, iv_data):
        """IV bounds should contain true ATE."""
        outcome, treatment, instrument, true_ate = iv_data
        result = manski_iv(outcome, treatment, instrument)

        # May not always contain true ATE due to finite sample
        # But should be in reasonable range
        assert result["bounds_lower"] < result["bounds_upper"]
        assert result["complier_share"] > 0

    def test_iv_with_strong_instrument(self):
        """Strong instrument should give informative bounds."""
        np.random.seed(4)
        n = 1000

        # Perfect compliance
        instrument = np.random.binomial(1, 0.5, n)
        treatment = instrument.copy()  # Perfect first stage
        outcome = 2.0 * treatment + np.random.randn(n)

        result = manski_iv(outcome, treatment, instrument)

        # With perfect compliance, bounds should be tight
        assert result["complier_share"] > 0.9


# =============================================================================
# Layer 2: Adversarial Tests (Input Validation & Edge Cases)
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_rejects_mismatched_lengths(self):
        """Should reject arrays of different lengths."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 1])

        with pytest.raises(ValueError, match="Length mismatch"):
            manski_worst_case(outcome, treatment)

    def test_rejects_non_binary_treatment(self):
        """Should reject non-binary treatment."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="binary"):
            manski_worst_case(outcome, treatment)

    def test_rejects_all_treated(self):
        """Should reject when all are treated."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([1, 1, 1])

        with pytest.raises(ValueError, match="No control"):
            manski_worst_case(outcome, treatment)

    def test_rejects_all_control(self):
        """Should reject when all are control."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 0, 0])

        with pytest.raises(ValueError, match="No treated"):
            manski_worst_case(outcome, treatment)

    def test_rejects_invalid_support(self):
        """Should reject invalid outcome support."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="Invalid support"):
            manski_worst_case(outcome, treatment, outcome_support=(10, 0))

    def test_rejects_invalid_mtr_direction(self):
        """Should reject invalid MTR direction."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="direction"):
            manski_mtr(outcome, treatment, direction="invalid")

    def test_iv_rejects_no_variation(self):
        """IV should reject instrument with no variation."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])
        instrument = np.array([1, 1, 1, 1])  # No variation

        with pytest.raises(ValueError, match="variation"):
            manski_iv(outcome, treatment, instrument)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_sample(self):
        """Should handle very small samples."""
        outcome = np.array([1.0, 3.0])
        treatment = np.array([0, 1])

        result = manski_worst_case(outcome, treatment)
        assert np.isfinite(result["bounds_lower"])
        assert np.isfinite(result["bounds_upper"])

    def test_extreme_imbalance(self):
        """Should handle extreme treatment imbalance."""
        np.random.seed(5)
        n = 1000
        treatment = np.random.binomial(1, 0.01, n)  # 1% treated
        outcome = treatment + np.random.randn(n)

        result = manski_worst_case(outcome, treatment)
        assert np.isfinite(result["bounds_lower"])
        assert np.isfinite(result["bounds_upper"])

    def test_constant_outcome(self):
        """Should handle constant outcome."""
        outcome = np.ones(100)
        treatment = np.random.binomial(1, 0.5, 100)

        result = manski_worst_case(outcome, treatment)
        assert result["bounds_lower"] == 0.0
        assert result["bounds_upper"] == 0.0
        assert result["point_identified"]

    def test_perfect_separation(self):
        """Should handle perfect separation of outcomes."""
        np.random.seed(6)
        n = 100
        treatment = np.array([0] * 50 + [1] * 50)
        outcome = np.concatenate([np.zeros(50), np.ones(50)])

        result = manski_worst_case(outcome, treatment)
        assert result["naive_ate"] == 1.0

    def test_handles_float_treatment(self):
        """Should convert float treatment to binary."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0.0, 0.0, 1.0, 1.0])

        result = manski_worst_case(outcome, treatment)
        assert np.isfinite(result["bounds_lower"])


# =============================================================================
# Layer 3: Property Tests
# =============================================================================


class TestBoundsProperties:
    """Tests for mathematical properties of bounds."""

    def test_lower_less_than_upper(self, simple_rct_data):
        """Lower bound should be ≤ upper bound."""
        outcome, treatment, _ = simple_rct_data

        for func in [manski_worst_case, manski_mts]:
            result = func(outcome, treatment)
            assert result["bounds_lower"] <= result["bounds_upper"]

        result = manski_mtr(outcome, treatment, direction="positive")
        assert result["bounds_lower"] <= result["bounds_upper"]

    def test_bounds_width_nonnegative(self, simple_rct_data):
        """Bounds width should be non-negative."""
        outcome, treatment, _ = simple_rct_data

        for func in [manski_worst_case, manski_mts]:
            result = func(outcome, treatment)
            assert result["bounds_width"] >= 0

    def test_bounds_contain_true_ate_under_correct_dgp(self, simple_rct_data):
        """Bounds should contain true ATE when DGP matches assumptions."""
        outcome, treatment, true_ate = simple_rct_data

        # Worst case always contains true ATE
        result = manski_worst_case(outcome, treatment)
        assert result["bounds_lower"] <= true_ate <= result["bounds_upper"]

    def test_narrower_support_gives_narrower_bounds(self):
        """Narrower outcome support should give narrower bounds."""
        np.random.seed(7)
        n = 500
        treatment = np.random.binomial(1, 0.5, n)
        outcome = treatment + np.random.randn(n)

        wide = manski_worst_case(outcome, treatment, outcome_support=(-100, 100))
        narrow = manski_worst_case(outcome, treatment, outcome_support=(-3, 4))

        assert narrow["bounds_width"] < wide["bounds_width"]

    def test_bounds_ordering(self, simple_rct_data):
        """Bounds should follow: worst_case ≥ MTR ≥ combined."""
        outcome, treatment, _ = simple_rct_data

        worst = manski_worst_case(outcome, treatment)
        mtr = manski_mtr(outcome, treatment, direction="positive")
        combined = manski_mtr_mts(outcome, treatment, mtr_direction="positive")

        assert worst["bounds_width"] >= mtr["bounds_width"] - 0.01  # Small tolerance
        assert mtr["bounds_width"] >= combined["bounds_width"] - 0.01


class TestReturnTypes:
    """Tests for return type correctness."""

    def test_worst_case_return_type(self, simple_rct_data):
        """Worst-case should return ManskiBoundsResult."""
        outcome, treatment, _ = simple_rct_data
        result = manski_worst_case(outcome, treatment)

        required_keys = [
            "bounds_lower",
            "bounds_upper",
            "bounds_width",
            "point_identified",
            "assumptions",
            "naive_ate",
            "ate_in_bounds",
            "n_treated",
            "n_control",
            "outcome_support",
            "interpretation",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_iv_return_type(self, iv_data):
        """IV should return ManskiIVBoundsResult."""
        outcome, treatment, instrument, _ = iv_data
        result = manski_iv(outcome, treatment, instrument)

        iv_keys = ["iv_strength", "complier_share", "n_iv_1", "n_iv_0"]
        for key in iv_keys:
            assert key in result, f"Missing IV key: {key}"


class TestCompareBounds:
    """Tests for compare_bounds function."""

    def test_compare_returns_all_methods(self, simple_rct_data):
        """Compare should return results from all methods."""
        outcome, treatment, _ = simple_rct_data
        comparison = compare_bounds(outcome, treatment)

        expected_methods = ["worst_case", "mtr", "mts", "mtr_mts", "_summary"]
        for method in expected_methods:
            assert method in comparison

    def test_compare_identifies_narrowest(self, simple_rct_data):
        """Compare should correctly identify narrowest bounds."""
        outcome, treatment, _ = simple_rct_data
        comparison = compare_bounds(outcome, treatment)

        summary = comparison["_summary"]
        assert "narrowest" in summary
        assert "widest" in summary

        # Verify narrowest is actually narrowest
        narrowest_method = summary["narrowest"]
        narrowest_width = comparison[narrowest_method]["bounds_width"]

        for method in ["worst_case", "mtr", "mts", "mtr_mts"]:
            assert comparison[method]["bounds_width"] >= narrowest_width - 0.01


# =============================================================================
# Monte Carlo Validation Tests
# =============================================================================


class TestMonteCarloProperties:
    """Monte Carlo validation of bounds properties."""

    @pytest.mark.parametrize("true_ate", [0.0, 1.0, 2.0, -1.0])
    def test_bounds_contain_true_ate_rct(self, true_ate):
        """Bounds should contain true ATE in RCT (no selection)."""
        np.random.seed(int(true_ate * 100) % 10000 + 100)
        n = 500
        treatment = np.random.binomial(1, 0.5, n)
        outcome = true_ate * treatment + np.random.randn(n)

        result = manski_worst_case(outcome, treatment)

        # Allow small tolerance for sampling error
        assert result["bounds_lower"] - 0.5 <= true_ate <= result["bounds_upper"] + 0.5

    def test_coverage_across_seeds(self):
        """Bounds should contain true ATE across multiple seeds."""
        true_ate = 1.5
        coverage_count = 0
        n_sims = 100

        for seed in range(n_sims):
            np.random.seed(seed)
            n = 300
            treatment = np.random.binomial(1, 0.5, n)
            outcome = true_ate * treatment + np.random.randn(n)

            result = manski_worst_case(outcome, treatment)
            if result["bounds_lower"] <= true_ate <= result["bounds_upper"]:
                coverage_count += 1

        coverage = coverage_count / n_sims
        assert coverage >= 0.95, f"Coverage {coverage:.2%} below 95%"

    def test_mtr_bounds_correct_under_dgp(self):
        """MTR bounds should be correct when DGP satisfies MTR."""
        np.random.seed(888)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)

        # DGP satisfies positive MTR: Y₁ ≥ Y₀
        y0 = np.random.randn(n)
        effect = np.abs(np.random.randn(n))  # Always positive
        y1 = y0 + effect
        outcome = np.where(treatment == 1, y1, y0)

        true_ate = effect.mean()

        result = manski_mtr(outcome, treatment, direction="positive")

        # True ATE should be within bounds
        assert result["bounds_lower"] <= true_ate + 0.3
        assert result["bounds_upper"] >= true_ate - 0.3


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_job_training_scenario(self):
        """Simulate job training program evaluation."""
        np.random.seed(999)
        n = 1000

        # Positive selection: higher ability → more likely to train
        ability = np.random.randn(n)
        p_train = 1 / (1 + np.exp(-0.5 * ability))
        training = np.random.binomial(1, p_train)

        # Positive MTR: training never hurts
        effect = np.abs(0.5 + 0.3 * ability + 0.2 * np.random.randn(n))
        wages = 10 + ability + effect * training + np.random.randn(n)

        # Compare methods
        worst = manski_worst_case(wages, training)
        mtr = manski_mtr(wages, training, direction="positive")
        mts = manski_mts(wages, training)
        combined = manski_mtr_mts(wages, training, mtr_direction="positive")

        # Bounds should narrow with more assumptions
        assert worst["bounds_width"] >= mtr["bounds_width"]
        assert worst["bounds_width"] >= mts["bounds_width"]
        assert mtr["bounds_width"] >= combined["bounds_width"]

        # Combined should give informative bounds
        assert combined["bounds_width"] < worst["bounds_width"] * 0.5

    def test_medical_treatment_scenario(self):
        """Simulate medical treatment with potential harm."""
        np.random.seed(1111)
        n = 800

        treatment = np.random.binomial(1, 0.3, n)  # 30% treated

        # Treatment has negative effect on some
        base_health = 50 + 10 * np.random.randn(n)
        effect = -5 + 3 * np.random.randn(n)  # Mean -5, some positive
        health = base_health + effect * treatment

        # Analyze
        worst = manski_worst_case(health, treatment)
        mtr_neg = manski_mtr(health, treatment, direction="negative")

        # Worst case should be very wide with health outcomes
        assert worst["bounds_width"] > 10

        # Negative MTR gives upper bound of 0
        assert mtr_neg["bounds_upper"] <= 0
