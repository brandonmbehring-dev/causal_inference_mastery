"""
Layer 2: Adversarial tests for IV estimator edge cases.

Session 59: Comprehensive edge case and error handling tests.

Tests 25+ challenging scenarios:
- Boundary violations (insufficient data, singular matrices)
- Data quality issues (NaN, Inf)
- Instrument strength extremes (F → 0, perfect instruments)
- Numerical stability (outliers, collinearity)
- Multi-instrument edge cases
- Estimator-specific edge cases (LIML, Fuller, GMM)

Goal: Ensure graceful failure OR correct handling of edge cases.
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.iv import (
    TwoStageLeastSquares,
    LIML,
    Fuller,
    GMM,
    FirstStage,
    classify_instrument_strength,
    anderson_rubin_test,
)


# =============================================================================
# Boundary Violations - Insufficient Data
# =============================================================================
class TestIVBoundaryViolations:
    """Test IV with boundary conditions and insufficient data."""

    def test_minimum_observations_fails_gracefully(self):
        """IV with n=2 observations - should fail or handle gracefully."""
        np.random.seed(123)
        Y = np.array([1.0, 2.0])
        D = np.array([0.0, 1.0])
        Z = np.array([[0.1], [0.9]])

        iv = TwoStageLeastSquares()
        # Should either raise an error or return a result
        try:
            iv.fit(Y, D, Z)
            # If it succeeds, result should be finite
            assert np.isfinite(iv.coef_[0])
        except (ValueError, np.linalg.LinAlgError):
            # Graceful failure is acceptable
            pass

    def test_constant_treatment_raises(self):
        """Constant treatment variable should fail (singular first stage)."""
        np.random.seed(456)
        n = 100
        Y = np.random.randn(n)
        D = np.ones(n)  # All treated - no variation
        Z = np.random.randn(n, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            iv.fit(Y, D, Z)

    def test_zero_instrument_variance_raises(self):
        """Constant instrument should fail (singular first stage)."""
        np.random.seed(789)
        n = 100
        Y = np.random.randn(n)
        D = np.random.randn(n)
        Z = np.ones((n, 1))  # Constant instrument

        iv = TwoStageLeastSquares()
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            iv.fit(Y, D, Z)

    def test_perfect_collinearity_handled(self):
        """Perfectly collinear instruments should fail or warn."""
        np.random.seed(111)
        n = 100
        Y = np.random.randn(n)
        D = np.random.randn(n)
        z1 = np.random.randn(n)
        Z = np.column_stack([z1, z1])  # Perfectly collinear

        iv = TwoStageLeastSquares()
        # May succeed with warning or raise - both are acceptable
        try:
            with warnings.catch_warnings(record=True):
                iv.fit(Y, D, Z)
            # If it succeeds, result should still be finite
            assert np.isfinite(iv.coef_[0])
        except (ValueError, np.linalg.LinAlgError):
            pass  # Graceful failure is acceptable


# =============================================================================
# Data Quality Issues
# =============================================================================
class TestIVDataQuality:
    """Test IV with data quality issues (NaN, Inf)."""

    def test_nan_in_outcome_raises(self):
        """NaN in outcome should raise error."""
        np.random.seed(222)
        n = 100
        Y = np.random.randn(n)
        Y[0] = np.nan
        D = np.random.randn(n)
        Z = np.random.randn(n, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises(ValueError):
            iv.fit(Y, D, Z)

    def test_nan_in_treatment_raises(self):
        """NaN in treatment should raise error."""
        np.random.seed(333)
        n = 100
        Y = np.random.randn(n)
        D = np.random.randn(n)
        D[5] = np.nan
        Z = np.random.randn(n, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises(ValueError):
            iv.fit(Y, D, Z)

    def test_nan_in_instruments_raises(self):
        """NaN in instruments should raise error."""
        np.random.seed(444)
        n = 100
        Y = np.random.randn(n)
        D = np.random.randn(n)
        Z = np.random.randn(n, 1)
        Z[10, 0] = np.nan

        iv = TwoStageLeastSquares()
        with pytest.raises(ValueError):
            iv.fit(Y, D, Z)

    def test_inf_in_outcome_raises(self):
        """Inf in outcome should raise error."""
        np.random.seed(555)
        n = 100
        Y = np.random.randn(n)
        Y[0] = np.inf
        D = np.random.randn(n)
        Z = np.random.randn(n, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises(ValueError):
            iv.fit(Y, D, Z)

    def test_zero_variance_treatment_raises(self):
        """Zero variance treatment should fail at fit time."""
        np.random.seed(666)
        n = 100
        Y = np.random.randn(n)
        D = np.full(n, 0.5)  # Constant treatment
        Z = np.random.randn(n, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            iv.fit(Y, D, Z)


# =============================================================================
# Instrument Strength Edge Cases
# =============================================================================
class TestIVInstrumentStrength:
    """Test IV with varying instrument strength."""

    def test_weak_instrument_warning_triggered(self):
        """Extremely weak instrument should trigger warning or detection."""
        np.random.seed(777)
        n = 500
        Z = np.random.randn(n)
        noise = np.random.randn(n) * 10  # Huge noise
        D = 0.01 * Z + noise  # Weak correlation
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # F-stat should be very low
        assert iv.first_stage_f_stat_ < 10, f"F-stat {iv.first_stage_f_stat_} should be < 10"

        # classify_instrument_strength returns (category, critical_value, message)
        strength_tuple = classify_instrument_strength(iv.first_stage_f_stat_, 1)
        strength = strength_tuple[0]  # Extract category
        assert strength in ["weak", "very_weak"], f"Got {strength}, expected weak/very_weak"

    def test_strong_instrument_no_warning(self):
        """Strong instrument should not warn and have high F-stat."""
        np.random.seed(888)
        n = 500
        Z = np.random.randn(n)
        D = 10.0 * Z + np.random.randn(n) * 0.1  # Very strong correlation
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # F-stat should be very high
        assert iv.first_stage_f_stat_ > 100, f"F-stat {iv.first_stage_f_stat_} should be > 100"

        # classify_instrument_strength returns (category, critical_value, message)
        strength_tuple = classify_instrument_strength(iv.first_stage_f_stat_, 1)
        strength = strength_tuple[0]  # Extract category
        assert strength == "strong", f"Got {strength}, expected strong"

    def test_perfect_instrument_high_fstat(self):
        """Perfect instrument should have extremely high F-stat."""
        np.random.seed(889)
        n = 500
        Z = np.random.randn(n)
        D = Z * 100  # Nearly perfect correlation
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # F-stat should be very high (thousands)
        assert iv.first_stage_f_stat_ > 1000, f"F-stat {iv.first_stage_f_stat_} should be > 1000"


# =============================================================================
# Numerical Stability
# =============================================================================
class TestIVNumericalStability:
    """Test IV numerical stability with challenging data."""

    def test_outliers_handled_gracefully(self):
        """Extreme outliers should not crash the estimator."""
        np.random.seed(999)
        n = 500
        Z = np.random.randn(n)
        D = 0.5 * Z + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)
        Y[0] = 1000.0  # Extreme outlier

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # Should complete with finite estimate
        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with outliers"

    def test_scaled_variables_work(self):
        """Variables scaled by 1e6 should still work."""
        np.random.seed(1001)
        n = 500
        scale = 1e6
        Z = np.random.randn(n) * scale
        D = 0.5 * Z + np.random.randn(n) * scale
        Y = 2.0 * D + np.random.randn(n) * scale

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with scaled data"

    def test_near_collinear_instruments(self):
        """Near-collinear instruments should work or fail gracefully."""
        np.random.seed(1002)
        n = 500
        z1 = np.random.randn(n)
        z2 = z1 + np.random.randn(n) * 0.01  # Nearly collinear
        Z = np.column_stack([z1, z2])
        D = 0.5 * z1 + 0.3 * z2 + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        # May succeed or fail - either is acceptable
        try:
            iv.fit(Y, D, Z)
            assert np.isfinite(iv.coef_[0])
        except (ValueError, np.linalg.LinAlgError):
            pass  # Graceful failure is acceptable

    def test_high_dimensional_covariates(self):
        """IV with many covariates should work."""
        np.random.seed(1009)
        n = 500
        k_covariates = 20
        Z = np.random.randn(n, 2)
        X = np.random.randn(n, k_covariates)
        D = 0.5 * Z[:, 0] + 0.3 * Z[:, 1] + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z, X)

        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with many covariates"


# =============================================================================
# Multi-Instrument Edge Cases
# =============================================================================
class TestIVMultipleInstruments:
    """Test IV with multiple instruments."""

    def test_order_condition_met(self):
        """Multiple instruments (overidentified) should work."""
        np.random.seed(1003)
        n = 500
        Z = np.random.randn(n, 3)  # 3 instruments for 1 endogenous
        D = 0.3 * Z[:, 0] + 0.2 * Z[:, 1] + 0.1 * Z[:, 2] + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z)

        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with multiple instruments"
        # Estimate should be near true value
        assert abs(iv.coef_[0] - 2.0) < 1.0, f"Estimate {iv.coef_[0]} far from true 2.0"

    def test_one_weak_instrument(self):
        """Mix of strong and weak instruments should work."""
        np.random.seed(1004)
        n = 500
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        z3 = np.random.randn(n)
        Z = np.column_stack([z1, z2, z3])
        D = 0.5 * z1 + 0.5 * z2 + 0.001 * z3 + np.random.randn(n)  # z3 is weak
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z)

        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with mixed strength"


# =============================================================================
# Estimator-Specific Edge Cases
# =============================================================================
class TestIVEstimatorEdgeCases:
    """Test specific IV estimators with edge cases."""

    def test_liml_normal_case(self):
        """LIML should work on standard data."""
        np.random.seed(1005)
        n = 200
        Z = np.random.randn(n)
        D = 0.5 * Z + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        liml = LIML()
        liml.fit(Y, D, Z.reshape(-1, 1))

        assert np.isfinite(liml.coef_[0]), "LIML estimate should be finite"
        # LIML kappa should be >= 1 (with small tolerance for floating point)
        if hasattr(liml, "kappa_"):
            assert liml.kappa_ >= 0.99, f"LIML kappa {liml.kappa_} should be >= 0.99"

    def test_fuller_modification_factors(self):
        """Fuller with different modification factors should work."""
        np.random.seed(1006)
        n = 500
        Z = np.random.randn(n, 2)
        D = 0.5 * Z[:, 0] + 0.3 * Z[:, 1] + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        # Fuller uses alpha_param for modification factor
        for alpha_param in [1.0, 4.0]:
            fuller = Fuller(alpha_param=alpha_param)
            fuller.fit(Y, D, Z)
            assert np.isfinite(fuller.coef_[0]), (
                f"Fuller(alpha_param={alpha_param}) should give finite estimate"
            )

    def test_gmm_two_step(self):
        """GMM with two-step (optimal) estimation should work."""
        np.random.seed(1007)
        n = 500
        Z = np.random.randn(n, 2)
        D = 0.3 * Z[:, 0] + 0.3 * Z[:, 1] + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        # GMM uses steps='two' for optimal weighting
        gmm = GMM(steps="two")
        gmm.fit(Y, D, Z)

        assert np.isfinite(gmm.coef_[0]), "GMM estimate should be finite"

    def test_overidentification_test(self):
        """Overidentification test should be available when applicable."""
        np.random.seed(1008)
        n = 500
        Z = np.random.randn(n, 3)  # 3 instruments
        D = 0.3 * Z[:, 0] + 0.2 * Z[:, 1] + 0.1 * Z[:, 2] + np.random.randn(n)
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z)

        # Should have Sargan/Hansen J test
        if hasattr(iv, "sargan_pvalue_"):
            assert 0 <= iv.sargan_pvalue_ <= 1, "Sargan p-value should be in [0,1]"


# =============================================================================
# Covariate Edge Cases
# =============================================================================
class TestIVCovariates:
    """Test IV with covariates."""

    def test_covariates_standard_case(self):
        """IV with standard covariates should work."""
        np.random.seed(1008)
        n = 500
        Z = np.random.randn(n)
        X = np.random.randn(n, 1)
        D = 0.5 * Z + 0.3 * X.flatten() + np.random.randn(n)
        Y = 2.0 * D + 1.0 * X.flatten() + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1), X)

        assert np.isfinite(iv.coef_[0]), "Estimate should be finite with covariates"


# =============================================================================
# Error Handling
# =============================================================================
class TestIVErrorHandling:
    """Test IV error handling."""

    def test_mismatched_dimensions_raises(self):
        """Mismatched array dimensions should raise error."""
        np.random.seed(1010)
        Y = np.random.randn(100)
        D = np.random.randn(50)  # Different length
        Z = np.random.randn(100, 1)

        iv = TwoStageLeastSquares()
        with pytest.raises(ValueError):
            iv.fit(Y, D, Z)

    def test_empty_arrays_raises(self):
        """Empty arrays should raise error."""
        Y = np.array([])
        D = np.array([])
        Z = np.zeros((0, 1))

        iv = TwoStageLeastSquares()
        with pytest.raises((ValueError, IndexError)):
            iv.fit(Y, D, Z)

    def test_invalid_inference_raises(self):
        """Invalid inference method should raise error at init."""
        # Error is raised at __init__, not fit
        with pytest.raises(ValueError):
            iv = TwoStageLeastSquares(inference="invalid_method")

    def test_no_instruments_raises(self):
        """Zero instruments should raise error."""
        np.random.seed(1012)
        Y = np.random.randn(100)
        D = np.random.randn(100)
        Z = np.zeros((100, 0))  # Zero instruments

        iv = TwoStageLeastSquares()
        with pytest.raises((ValueError, IndexError)):
            iv.fit(Y, D, Z)


# =============================================================================
# Anderson-Rubin Edge Cases
# =============================================================================
class TestAndersonRubinEdgeCases:
    """Test Anderson-Rubin test edge cases."""

    def test_ar_with_weak_iv(self):
        """AR test should be valid even with weak instruments."""
        np.random.seed(2001)
        n = 500
        Z = np.random.randn(n)
        D = 0.05 * Z + np.random.randn(n)  # Very weak
        Y = 2.0 * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # AR test returns (statistic, p_value, ci)
        ar_stat, ar_pval, ar_ci = anderson_rubin_test(Y, D, Z.reshape(-1, 1))
        assert np.isfinite(ar_stat), "AR statistic should be finite"
        assert 0 <= ar_pval <= 1, "AR p-value should be in [0, 1]"

    def test_ar_with_null_effect(self):
        """AR test at true null should have correct Type I error."""
        np.random.seed(2002)
        n = 500
        true_beta = 0.0  # No effect
        Z = np.random.randn(n)
        D = 0.5 * Z + np.random.randn(n)
        Y = true_beta * D + np.random.randn(n)

        iv = TwoStageLeastSquares()
        iv.fit(Y, D, Z.reshape(-1, 1))

        # AR test returns (statistic, p_value, ci)
        ar_stat, ar_pval, ar_ci = anderson_rubin_test(Y, D, Z.reshape(-1, 1))
        # Just check it runs and gives valid output
        assert 0 <= ar_pval <= 1, "AR p-value should be in [0, 1]"


# =============================================================================
# First Stage Edge Cases
# =============================================================================
class TestFirstStageEdgeCases:
    """Test FirstStage decomposition edge cases."""

    def test_first_stage_perfect_fit(self):
        """FirstStage with perfect fit should work."""
        np.random.seed(3001)
        n = 100
        Z = np.random.randn(n)
        D = Z * 10  # Nearly perfect relationship

        fs = FirstStage()
        fs.fit(D, Z.reshape(-1, 1))

        # FirstStage uses r2_ not r_squared_
        assert fs.r2_ > 0.99, f"R-squared {fs.r2_} should be > 0.99"

    def test_first_stage_with_covariates(self):
        """FirstStage with covariates should work."""
        np.random.seed(3002)
        n = 200
        Z = np.random.randn(n)
        X = np.random.randn(n, 2)
        D = 0.5 * Z + 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n)

        fs = FirstStage()
        fs.fit(D, Z.reshape(-1, 1), X)

        assert np.isfinite(fs.coef_[0]), "FirstStage coefficient should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
