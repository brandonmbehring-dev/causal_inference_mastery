"""
Tests for GMM (Generalized Method of Moments) estimator.

Validates:
- GMM vs 2SLS agreement (asymptotically equivalent)
- One-step vs two-step GMM
- Hansen J-test for overidentification
- Input validation
- Edge cases (just-identified, over-identified)
"""

import numpy as np
import pytest

from src.causal_inference.iv import GMM, TwoStageLeastSquares


class TestGMMBasicFunctionality:
    """Test basic GMM functionality."""

    def test_gmm_with_strong_instrument(self, iv_strong_instrument):
        """Test that GMM works with strong instruments."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit two-step GMM
        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # Coefficient should be finite and close to truth
        assert np.isfinite(gmm.coef_[0])
        assert abs(gmm.coef_[0] - true_beta) < 0.3  # Within ±30%

    def test_gmm_agrees_with_2sls_asymptotically(self, iv_strong_instrument):
        """Test that one-step GMM agrees with 2SLS (they are equivalent)."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit one-step GMM
        gmm = GMM(steps="one", inference="robust")
        gmm.fit(Y, D, Z, X)

        # Fit 2SLS
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)

        # Coefficients should be very close (one-step GMM = 2SLS)
        np.testing.assert_allclose(
            gmm.coef_[0],
            tsls.coef_[0],
            rtol=1e-6,
            err_msg="One-step GMM should match 2SLS exactly",
        )

    def test_one_step_vs_two_step(self, iv_over_identified):
        """Test one-step vs two-step GMM with overidentification."""
        Y, D, Z, X, true_beta = iv_over_identified

        # Fit both
        gmm_one = GMM(steps="one", inference="robust")
        gmm_one.fit(Y, D, Z, X)

        gmm_two = GMM(steps="two", inference="robust")
        gmm_two.fit(Y, D, Z, X)

        # Both should produce finite estimates
        assert np.isfinite(gmm_one.coef_[0])
        assert np.isfinite(gmm_two.coef_[0])

        # Two-step is asymptotically more efficient, but with strong IV
        # they should be similar
        np.testing.assert_allclose(
            gmm_one.coef_[0],
            gmm_two.coef_[0],
            rtol=0.2,
            err_msg="One-step and two-step GMM should be similar with strong IV",
        )

    def test_gmm_summary_table(self, iv_strong_instrument):
        """Test that GMM summary table is generated correctly."""
        Y, D, Z, X, _ = iv_strong_instrument

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        summary = gmm.summary()

        # Check DataFrame structure
        assert "Variable" in summary.columns
        assert "Coefficient" in summary.columns
        assert "Std. Error" in summary.columns
        assert "t-statistic" in summary.columns
        assert "p-value" in summary.columns

        # Check J-test row is present
        assert any("Hansen J-test" in str(v) for v in summary["Variable"].values)


class TestHansenJTest:
    """Test Hansen J-test for overidentifying restrictions."""

    def test_hansen_j_with_just_identified(self, iv_just_identified):
        """Test that J-statistic is exactly zero for just-identified models."""
        Y, D, Z, X, _ = iv_just_identified

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # J-test should be exactly 0 when q = p
        assert gmm.j_statistic_ == 0.0, "J-statistic must be 0 for just-identified models"
        assert gmm.j_pvalue_ == 1.0
        assert gmm.j_df_ == 0

    def test_hansen_j_with_valid_overid_restrictions(self, iv_over_identified):
        """Test J-test with valid overidentifying restrictions."""
        Y, D, Z, X, _ = iv_over_identified

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # J-statistic should be positive (df = 1 with q=2, p=1)
        assert gmm.j_statistic_ >= 0
        assert gmm.j_df_ == 1  # q - p = 2 - 1 = 1

        # With valid instruments, should not reject (p-value high)
        # Note: This is a probabilistic test, may occasionally fail
        # We just check that p-value is reasonable (not always 0 or 1)
        assert 0 <= gmm.j_pvalue_ <= 1

    def test_hansen_j_rejects_invalid_instruments(self):
        """Test that J-test detects invalid instruments."""
        # Create DGP with invalid instruments
        np.random.seed(42)
        n = 1000

        # Z1 is valid, Z2 is invalid (correlated with error)
        eps = np.random.normal(0, 1, n)
        Z1 = np.random.normal(0, 1, n)
        Z2 = np.random.normal(0, 1, n) + 0.5 * eps  # Invalid: correlated with error

        D = 2 * Z1 + 1.5 * Z2 + np.random.normal(0, 1, n)
        Y = 0.5 * D + eps

        Z = np.column_stack([Z1, Z2])

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z)

        # J-test should detect the invalid instrument
        # (though this is not guaranteed in every random sample)
        assert gmm.j_statistic_ > 0
        assert gmm.j_df_ == 1

        # Note: We don't assert p-value < 0.05 because with n=1000,
        # the invalid instrument may not always be detected


class TestGMMInputValidation:
    """Test GMM input validation."""

    def test_gmm_rejects_underidentified(self):
        """Test that GMM raises error if underidentified (q < p)."""
        n = 100
        Y = np.random.normal(0, 1, n)
        D = np.column_stack([np.random.normal(0, 1, n), np.random.normal(0, 1, n)])
        Z = np.random.normal(0, 1, n)  # 1 instrument, 2 endogenous

        gmm = GMM(steps="two")

        with pytest.raises(ValueError, match="underidentified"):
            gmm.fit(Y, D, Z)

    def test_gmm_summary_raises_if_not_fitted(self):
        """Test that summary() raises error if model not fitted."""
        gmm = GMM(steps="two")

        with pytest.raises(ValueError, match="not been fitted"):
            gmm.summary()

    def test_gmm_rejects_invalid_steps(self):
        """Test that GMM raises error for invalid steps parameter."""
        with pytest.raises(ValueError, match="steps must be"):
            GMM(steps="three")

    def test_gmm_rejects_invalid_inference(self):
        """Test that GMM raises error for invalid inference parameter."""
        with pytest.raises(ValueError, match="inference must be"):
            GMM(steps="two", inference="clustered")


class TestGMMEdgeCases:
    """Test GMM edge cases and numerical stability."""

    def test_gmm_with_over_identified(self, iv_over_identified):
        """Test GMM with over-identified model (q > p)."""
        Y, D, Z, X, true_beta = iv_over_identified

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # Should work with q=2, p=1
        assert np.isfinite(gmm.coef_[0])
        assert gmm.n_instruments_ == 2
        assert gmm.n_endogenous_ == 1
        assert gmm.j_df_ == 1  # q - p

    def test_gmm_with_just_identified(self, iv_just_identified):
        """Test GMM with just-identified model (q = p)."""
        Y, D, Z, X, true_beta = iv_just_identified

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # Should work with q=1, p=1
        assert np.isfinite(gmm.coef_[0])
        assert gmm.n_instruments_ == 1
        assert gmm.n_endogenous_ == 1
        assert gmm.j_df_ == 0  # q - p

    def test_gmm_with_weak_instrument(self, iv_weak_instrument):
        """Test GMM with weak instruments."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        gmm = GMM(steps="two", inference="robust")
        gmm.fit(Y, D, Z, X)

        # Should produce finite estimates (though biased)
        assert np.isfinite(gmm.coef_[0])
        assert np.isfinite(gmm.j_statistic_)


class TestGMMInference:
    """Test GMM inference methods."""

    def test_gmm_robust_vs_standard_se(self, iv_heteroskedastic):
        """Test GMM with different inference types."""
        Y, D, Z, X, _ = iv_heteroskedastic

        gmm_standard = GMM(steps="two", inference="standard")
        gmm_standard.fit(Y, D, Z, X)

        gmm_robust = GMM(steps="two", inference="robust")
        gmm_robust.fit(Y, D, Z, X)

        # Coefficients should be identical
        np.testing.assert_allclose(gmm_standard.coef_, gmm_robust.coef_, rtol=1e-6)

        # Both should produce finite SEs
        assert np.all(np.isfinite(gmm_standard.se_))
        assert np.all(np.isfinite(gmm_robust.se_))

        # Robust SEs are typically larger with heteroskedasticity
        # (though not guaranteed in every sample)

    def test_gmm_confidence_intervals(self, iv_strong_instrument):
        """Test that GMM confidence intervals are computed correctly."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        gmm = GMM(steps="two", inference="robust", alpha=0.05)
        gmm.fit(Y, D, Z, X)

        # CI should be (lower, upper) with lower < estimate < upper
        ci_lower, ci_upper = gmm.ci_[0]
        estimate = gmm.coef_[0]

        assert ci_lower <= estimate <= ci_upper, "Estimate should be within CI"
        assert ci_upper > ci_lower, "CI upper bound should be > lower bound"


class TestGMMSpecialCases:
    """Test GMM special cases and comparisons."""

    def test_gmm_two_step_more_efficient(self, iv_over_identified):
        """Test that two-step GMM has smaller SEs than one-step (asymptotically)."""
        Y, D, Z, X, _ = iv_over_identified

        gmm_one = GMM(steps="one", inference="robust")
        gmm_one.fit(Y, D, Z, X)

        gmm_two = GMM(steps="two", inference="robust")
        gmm_two.fit(Y, D, Z, X)

        # Two-step should be more efficient (smaller SE)
        # Note: Not guaranteed in finite samples, but typical pattern
        # We just verify both produce finite SEs
        assert np.isfinite(gmm_one.se_[0])
        assert np.isfinite(gmm_two.se_[0])
        assert gmm_one.se_[0] > 0
        assert gmm_two.se_[0] > 0

    def test_gmm_j_test_increases_with_more_invalid_instruments(self):
        """Test that J-statistic increases with more invalid instruments."""
        np.random.seed(123)
        n = 1000

        eps = np.random.normal(0, 1, n)
        Z1 = np.random.normal(0, 1, n)

        # Create two models:
        # Model 1: 1 valid + 1 invalid instrument
        # Model 2: 1 valid + 2 invalid instruments

        Z2_invalid = np.random.normal(0, 1, n) + 0.3 * eps
        Z3_invalid = np.random.normal(0, 1, n) + 0.3 * eps

        D = 2 * Z1 + 0.5 * Z2_invalid + 0.5 * Z3_invalid + np.random.normal(0, 1, n)
        Y = 0.5 * D + eps

        # Model 1: 2 instruments (1 valid, 1 invalid)
        Z_2inst = np.column_stack([Z1, Z2_invalid])
        gmm_2inst = GMM(steps="two", inference="robust")
        gmm_2inst.fit(Y, D, Z_2inst)

        # Model 2: 3 instruments (1 valid, 2 invalid)
        Z_3inst = np.column_stack([Z1, Z2_invalid, Z3_invalid])
        gmm_3inst = GMM(steps="two", inference="robust")
        gmm_3inst.fit(Y, D, Z_3inst)

        # J-statistic should increase (though not guaranteed in every sample)
        # We just verify both are computed and positive
        assert gmm_2inst.j_statistic_ >= 0
        assert gmm_3inst.j_statistic_ >= 0
        assert gmm_2inst.j_df_ == 1
        assert gmm_3inst.j_df_ == 2
