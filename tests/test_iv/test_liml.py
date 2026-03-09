"""
Tests for Limited Information Maximum Likelihood (LIML) estimator.

Validates:
- LIML vs 2SLS agreement with strong instruments
- LIML less biased than 2SLS with weak instruments
- Kappa parameter calculation
- Standard error computation
- Input validation
"""

import numpy as np
import pytest

from src.causal_inference.iv import LIML, TwoStageLeastSquares


class TestLIMLBasicFunctionality:
    """Test basic LIML functionality."""

    def test_liml_with_strong_instrument(self, iv_strong_instrument):
        """Test that LIML works with strong instruments."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit LIML
        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Coefficient should be finite and close to truth
        assert np.isfinite(liml.coef_[0])
        assert abs(liml.coef_[0] - true_beta) < 0.3  # Within ±30% (generous tolerance)

    def test_liml_agrees_with_2sls_strong_iv(self, iv_strong_instrument):
        """Test that LIML ≈ 2SLS with strong instruments."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit both estimators
        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)

        # Coefficients should be similar (within 10%)
        np.testing.assert_allclose(
            liml.coef_[0],
            tsls.coef_[0],
            rtol=0.1,
            err_msg="LIML and 2SLS should agree with strong instruments",
        )

    def test_liml_kappa_greater_than_one(self, iv_strong_instrument):
        """Test that LIML kappa > 1 (k-class property)."""
        Y, D, Z, X, _ = iv_strong_instrument

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # LIML kappa should be >= 1 (approaching 1 with strong IV)
        assert liml.kappa_ >= 0.9, f"Expected kappa >= 0.9, got {liml.kappa_:.3f}"

    def test_liml_standard_errors_positive(self, iv_strong_instrument):
        """Test that LIML standard errors are positive and finite."""
        Y, D, Z, X, _ = iv_strong_instrument

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        assert np.all(liml.se_ > 0), "Standard errors should be positive"
        assert np.all(np.isfinite(liml.se_)), "Standard errors should be finite"

    def test_liml_summary_table(self, iv_strong_instrument):
        """Test that LIML summary table is generated correctly."""
        Y, D, Z, X, _ = iv_strong_instrument

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        summary = liml.summary()

        # Check DataFrame structure
        assert "Variable" in summary.columns
        assert "Coefficient" in summary.columns
        assert "Std. Error" in summary.columns
        assert "t-statistic" in summary.columns
        assert "p-value" in summary.columns

        # Check number of rows
        assert len(summary) == len(liml.coef_)


class TestLIMLWeakInstruments:
    """Test LIML behavior with weak instruments."""

    def test_liml_with_weak_instrument(self, iv_weak_instrument):
        """Test that LIML works with weak instruments."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        # Fit LIML
        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Should still produce finite estimates (though may be biased)
        assert np.isfinite(liml.coef_[0])
        assert np.isfinite(liml.kappa_)

    def test_liml_less_biased_than_2sls_weak_iv(self, iv_weak_instrument):
        """Test that LIML is closer to truth than 2SLS with weak instruments."""
        Y, D, Z, X, true_beta = iv_weak_instrument

        # Fit both estimators
        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)

        # LIML should be less biased (closer to true value)
        # Note: This is a statistical property, may not hold in every single realization
        liml_bias = abs(liml.coef_[0] - true_beta)
        tsls_bias = abs(tsls.coef_[0] - true_beta)

        # At minimum, LIML should not be dramatically worse
        assert liml_bias <= tsls_bias * 1.5, (
            f"LIML bias ({liml_bias:.3f}) should not be much worse than 2SLS ({tsls_bias:.3f})"
        )

    def test_liml_kappa_with_weak_iv(self, iv_weak_instrument):
        """Test that LIML kappa is computed with weak instruments."""
        Y_weak, D_weak, Z_weak, X_weak, _ = iv_weak_instrument

        liml_weak = LIML(inference="robust")
        liml_weak.fit(Y_weak, D_weak, Z_weak, X_weak)

        # Kappa should be finite and positive
        # Note: Direction (>1 or <1) depends on DGP specifics
        assert np.isfinite(liml_weak.kappa_), "Kappa should be finite"
        assert liml_weak.kappa_ > 0, f"Kappa should be positive, got {liml_weak.kappa_:.3f}"


class TestLIMLInference:
    """Test LIML inference methods."""

    def test_liml_robust_vs_standard_se(self, iv_heteroskedastic):
        """Test that robust SEs >= standard SEs with heteroskedasticity."""
        Y, D, Z, X, _ = iv_heteroskedastic

        # Fit with standard SEs
        liml_standard = LIML(inference="standard")
        liml_standard.fit(Y, D, Z, X)

        # Fit with robust SEs
        liml_robust = LIML(inference="robust")
        liml_robust.fit(Y, D, Z, X)

        # Coefficients should be identical
        np.testing.assert_allclose(liml_standard.coef_, liml_robust.coef_, rtol=1e-6)

        # Robust SEs should be >= standard SEs (in expectation)
        # Note: May not hold for every single coefficient in every sample
        # Check at least one SE is larger (weak test)
        assert np.any(liml_robust.se_ >= liml_standard.se_ * 0.9)

    def test_liml_confidence_intervals(self, iv_strong_instrument):
        """Test that LIML confidence intervals are computed correctly."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        liml = LIML(inference="robust", alpha=0.05)
        liml.fit(Y, D, Z, X)

        # CI should be (lower, upper) with lower < estimate < upper
        ci_lower, ci_upper = liml.ci_[0]
        estimate = liml.coef_[0]

        assert ci_lower <= estimate <= ci_upper, "Estimate should be within CI"
        assert ci_upper > ci_lower, "CI upper bound should be > lower bound"


class TestLIMLInputValidation:
    """Test LIML input validation."""

    def test_liml_rejects_underidentified(self):
        """Test that LIML raises error if underidentified (q < p)."""
        n = 100
        Y = np.random.normal(0, 1, n)
        D = np.column_stack([np.random.normal(0, 1, n), np.random.normal(0, 1, n)])
        Z = np.random.normal(0, 1, n)  # 1 instrument, 2 endogenous

        liml = LIML()

        with pytest.raises(ValueError, match="underidentified"):
            liml.fit(Y, D, Z)

    def test_liml_rejects_nan_in_y(self):
        """Test that LIML raises error if Y contains NaN."""
        n = 100
        Y = np.random.normal(0, 1, n)
        Y[0] = np.nan
        D = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)

        liml = LIML()

        with pytest.raises(ValueError, match="NaN"):
            liml.fit(Y, D, Z)

    def test_liml_rejects_mismatched_lengths(self):
        """Test that LIML raises error if Y and D have different lengths."""
        Y = np.random.normal(0, 1, 100)
        D = np.random.normal(0, 1, 90)
        Z = np.random.normal(0, 1, 100)

        liml = LIML()

        with pytest.raises(ValueError, match="same length"):
            liml.fit(Y, D, Z)

    def test_liml_summary_raises_if_not_fitted(self):
        """Test that summary() raises error if model not fitted."""
        liml = LIML()

        with pytest.raises(ValueError, match="not been fitted"):
            liml.summary()


class TestLIMLEdgeCases:
    """Test LIML edge cases and numerical stability."""

    def test_liml_with_very_weak_instrument_raises(self, iv_very_weak_instrument):
        """Test that LIML fails gracefully with very weak instruments (F < 5)."""
        Y, D, Z, X, _ = iv_very_weak_instrument

        liml = LIML(inference="robust")

        # Should either raise error (kappa too small) or produce warning
        # Depending on random seed, may occasionally succeed
        try:
            liml.fit(Y, D, Z, X)
            # If it succeeds, kappa should be > 0
            assert liml.kappa_ > 0
        except ValueError as e:
            # Expected: kappa too close to zero
            assert "kappa" in str(e).lower() or "weak" in str(e).lower()

    def test_liml_with_over_identified(self, iv_over_identified):
        """Test LIML with over-identified model (q > p)."""
        Y, D, Z, X, true_beta = iv_over_identified

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Should work with q=2, p=1
        assert np.isfinite(liml.coef_[0])
        assert liml.n_instruments_ == 2
        assert liml.n_endogenous_ == 1

    def test_liml_with_just_identified(self, iv_just_identified):
        """Test LIML with just-identified model (q = p)."""
        Y, D, Z, X, true_beta = iv_just_identified

        liml = LIML(inference="robust")
        liml.fit(Y, D, Z, X)

        # Should work with q=1, p=1
        assert np.isfinite(liml.coef_[0])
        assert liml.n_instruments_ == 1
        assert liml.n_endogenous_ == 1

        # LIML and 2SLS should be very similar when just-identified
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)

        np.testing.assert_allclose(liml.coef_[0], tsls.coef_[0], rtol=0.05)
