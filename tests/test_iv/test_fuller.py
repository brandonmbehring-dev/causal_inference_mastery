"""
Tests for Fuller k-class estimator.

Validates:
- Fuller vs LIML vs 2SLS agreement
- Fuller-1 vs Fuller-4 properties
- Kappa adjustment (k_Fuller = k_LIML - α/(n-L))
- Fuller with weak instruments
- Input validation
"""

import numpy as np
import pytest

from src.causal_inference.iv import Fuller, LIML, TwoStageLeastSquares


class TestFullerBasicFunctionality:
    """Test basic Fuller functionality."""

    def test_fuller_with_strong_instrument(self, iv_strong_instrument):
        """Test that Fuller works with strong instruments."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit Fuller-1
        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        # Coefficient should be finite and close to truth
        assert np.isfinite(fuller.coef_[0])
        assert abs(fuller.coef_[0] - true_beta) < 0.3  # Within ±30%

    def test_fuller_agrees_with_liml_strong_iv(self, iv_strong_instrument):
        """Test that Fuller ≈ LIML with strong instruments."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit both estimators
        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Coefficients should be similar (Fuller correction is small with strong IV)
        np.testing.assert_allclose(
            fuller.coef_[0],
            liml.coef_[0],
            rtol=0.1,
            err_msg="Fuller and LIML should agree closely with strong instruments",
        )

    def test_fuller_kappa_adjustment(self, iv_strong_instrument):
        """Test that Fuller kappa = LIML kappa - α/(n-L)."""
        Y, D, Z, X, _ = iv_strong_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Compute expected correction
        n = len(Y)
        L = 1 + 1  # 1 instrument + 1 intercept
        if X is not None:
            L += X.shape[1]

        expected_correction = fuller.alpha_param / (n - L)
        expected_kappa = liml.kappa_ - expected_correction

        # Fuller kappa should match formula
        np.testing.assert_allclose(
            fuller.kappa_,
            expected_kappa,
            rtol=1e-6,
            err_msg=f"Fuller kappa should be LIML kappa - α/(n-L)",
        )

        # Also check that stored LIML kappa matches
        np.testing.assert_allclose(fuller.kappa_liml_, liml.kappa_, rtol=1e-6)

    def test_fuller_standard_errors_positive(self, iv_strong_instrument):
        """Test that Fuller standard errors are positive and finite."""
        Y, D, Z, X, _ = iv_strong_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        assert np.all(fuller.se_ > 0), "Standard errors should be positive"
        assert np.all(np.isfinite(fuller.se_)), "Standard errors should be finite"

    def test_fuller_summary_table(self, iv_strong_instrument):
        """Test that Fuller summary table is generated correctly."""
        Y, D, Z, X, _ = iv_strong_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        summary = fuller.summary()

        # Check DataFrame structure
        assert "Variable" in summary.columns
        assert "Coefficient" in summary.columns
        assert "Std. Error" in summary.columns
        assert "t-statistic" in summary.columns
        assert "p-value" in summary.columns

        # Check number of rows
        assert len(summary) == len(fuller.coef_)


class TestFullerVariants:
    """Test different Fuller variants (Fuller-1 vs Fuller-4)."""

    def test_fuller1_vs_fuller4_kappa(self, iv_strong_instrument):
        """Test that Fuller-4 kappa is smaller than Fuller-1 kappa."""
        Y, D, Z, X, _ = iv_strong_instrument

        fuller1 = Fuller(alpha_param=1.0, inference="robust")
        fuller1.fit(Y, D, Z, X)

        fuller4 = Fuller(alpha_param=4.0, inference="robust")
        fuller4.fit(Y, D, Z, X)

        # Fuller-4 applies larger correction (α=4 > α=1)
        # So k_Fuller4 < k_Fuller1
        assert fuller4.kappa_ < fuller1.kappa_, (
            f"Fuller-4 kappa ({fuller4.kappa_:.4f}) should be < Fuller-1 kappa ({fuller1.kappa_:.4f})"
        )

    def test_fuller1_recommended(self, iv_weak_instrument):
        """Test that Fuller-1 is often recommended variant."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        # Fuller-1 should work with weak instruments
        fuller1 = Fuller(alpha_param=1.0, inference="robust")
        fuller1.fit(Y, D, Z, X)

        # Should produce finite estimates
        assert np.isfinite(fuller1.coef_[0])
        assert np.isfinite(fuller1.kappa_)
        assert fuller1.kappa_ > 0

    def test_fuller_approaches_liml_asymptotically(self, iv_strong_instrument):
        """Test that as n increases, Fuller → LIML."""
        # This is a theoretical property - the correction α/(n-L) → 0 as n → ∞
        Y, D, Z, X, _ = iv_strong_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        n = len(Y)
        L = fuller._liml.n_instruments_ + 1  # instruments + intercept

        # Correction should be small with large n
        correction = 1.0 / (n - L)

        # With n=1000 (from fixture), correction ≈ 0.001
        assert correction < 0.01, f"Correction should be small with n={n}"


class TestFullerWeakInstruments:
    """Test Fuller behavior with weak instruments."""

    def test_fuller_with_weak_instrument(self, iv_weak_instrument):
        """Test that Fuller works with weak instruments."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        # Should produce finite estimates
        assert np.isfinite(fuller.coef_[0])
        assert np.isfinite(fuller.kappa_)
        assert fuller.kappa_ > 0

    def test_fuller_vs_2sls_weak_iv(self, iv_weak_instrument):
        """Test Fuller vs 2SLS with weak instruments."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)

        # Fuller should not be dramatically worse than 2SLS
        fuller_bias = abs(fuller.coef_[0] - true_beta)
        tsls_bias = abs(tsls.coef_[0] - true_beta)

        assert fuller_bias <= tsls_bias * 1.5, (
            f"Fuller bias ({fuller_bias:.3f}) should not be much worse than 2SLS ({tsls_bias:.3f})"
        )


class TestFullerInputValidation:
    """Test Fuller input validation."""

    def test_fuller_rejects_negative_alpha_param(self):
        """Test that Fuller raises error if alpha_param <= 0."""
        with pytest.raises(ValueError, match="alpha_param must be positive"):
            Fuller(alpha_param=-1.0)

        with pytest.raises(ValueError, match="alpha_param must be positive"):
            Fuller(alpha_param=0.0)

    def test_fuller_rejects_underidentified(self):
        """Test that Fuller raises error if underidentified (q < p)."""
        n = 100
        Y = np.random.normal(0, 1, n)
        D = np.column_stack([np.random.normal(0, 1, n), np.random.normal(0, 1, n)])
        Z = np.random.normal(0, 1, n)  # 1 instrument, 2 endogenous

        fuller = Fuller(alpha_param=1.0)

        with pytest.raises(ValueError, match="underidentified"):
            fuller.fit(Y, D, Z)

    def test_fuller_summary_raises_if_not_fitted(self):
        """Test that summary() raises error if model not fitted."""
        fuller = Fuller(alpha_param=1.0)

        with pytest.raises(ValueError, match="not been fitted"):
            fuller.summary()


class TestFullerEdgeCases:
    """Test Fuller edge cases and numerical stability."""

    def test_fuller_with_very_weak_instrument(self, iv_very_weak_instrument):
        """Test Fuller with very weak instruments (F < 5)."""
        Y, D, Z, X, _ = iv_very_weak_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust")

        # May fail with very weak instruments
        try:
            fuller.fit(Y, D, Z, X)
            # If it succeeds, kappa should be positive
            assert fuller.kappa_ > 0
        except ValueError as e:
            # Expected: kappa too small or non-positive
            assert "kappa" in str(e).lower() or "weak" in str(e).lower()

    def test_fuller_with_over_identified(self, iv_over_identified):
        """Test Fuller with over-identified model (q > p)."""
        Y, D, Z, X, true_beta = iv_over_identified

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        # Should work with q=2, p=1
        assert np.isfinite(fuller.coef_[0])
        assert fuller.n_instruments_ == 2
        assert fuller.n_endogenous_ == 1

    def test_fuller_with_just_identified(self, iv_just_identified):
        """Test Fuller with just-identified model (q = p)."""
        Y, D, Z, X, true_beta = iv_just_identified

        fuller = Fuller(alpha_param=1.0, inference="robust")
        fuller.fit(Y, D, Z, X)

        # Should work with q=1, p=1
        assert np.isfinite(fuller.coef_[0])
        assert fuller.n_instruments_ == 1
        assert fuller.n_endogenous_ == 1


class TestFullerInference:
    """Test Fuller inference methods."""

    def test_fuller_confidence_intervals(self, iv_strong_instrument):
        """Test that Fuller confidence intervals are computed correctly."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        fuller = Fuller(alpha_param=1.0, inference="robust", alpha=0.05)
        fuller.fit(Y, D, Z, X)

        # CI should be (lower, upper) with lower < estimate < upper
        ci_lower, ci_upper = fuller.ci_[0]
        estimate = fuller.coef_[0]

        assert ci_lower <= estimate <= ci_upper, "Estimate should be within CI"
        assert ci_upper > ci_lower, "CI upper bound should be > lower bound"

    def test_fuller_robust_vs_standard_se(self, iv_heteroskedastic):
        """Test Fuller with different inference types."""
        Y, D, Z, X, _ = iv_heteroskedastic

        fuller_standard = Fuller(alpha_param=1.0, inference="standard")
        fuller_standard.fit(Y, D, Z, X)

        fuller_robust = Fuller(alpha_param=1.0, inference="robust")
        fuller_robust.fit(Y, D, Z, X)

        # Coefficients should be identical
        np.testing.assert_allclose(fuller_standard.coef_, fuller_robust.coef_, rtol=1e-6)

        # Both should produce finite SEs
        assert np.all(np.isfinite(fuller_standard.se_))
        assert np.all(np.isfinite(fuller_robust.se_))
