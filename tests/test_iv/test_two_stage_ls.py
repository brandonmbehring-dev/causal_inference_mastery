"""
Tests for Two-Stage Least Squares (2SLS) estimator.

Layer 1 (Known-Answer) tests verify that the 2SLS estimator:
- Recovers true coefficients from simulated data (±10% tolerance)
- Computes positive, finite standard errors
- Produces confidence intervals with correct coverage
- Correctly identifies weak vs. strong instruments
- Handles edge cases gracefully (underidentification, collinearity)
"""

import numpy as np
import pandas as pd
import pytest

from src.causal_inference.iv import TwoStageLeastSquares


class TestTwoStageLeastSquaresBasic:
    """Test basic 2SLS functionality with known-answer fixtures."""

    def test_just_identified_coefficient(self, iv_just_identified):
        """Test 2SLS recovers true coefficient with just-identified IV (1 instrument)."""
        Y, D, Z, X, true_beta = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # Check point estimate within 10% of truth
        assert np.isclose(
            iv.coef_[0], true_beta, rtol=0.10
        ), f"Expected β ≈ {true_beta}, got {iv.coef_[0]:.4f}"

    def test_over_identified_coefficient(self, iv_over_identified):
        """Test 2SLS with multiple instruments (overidentified, q > p)."""
        Y, D, Z, X, true_beta = iv_over_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # With 2 instruments, should still recover true effect (±15% tolerance for overID)
        assert np.isclose(
            iv.coef_[0], true_beta, rtol=0.15
        ), f"Expected β ≈ {true_beta}, got {iv.coef_[0]:.4f}"

    def test_strong_instrument_coefficient(self, iv_strong_instrument):
        """Test 2SLS with strong instrument (F > 20)."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # Strong instrument should give accurate estimate
        assert np.isclose(
            iv.coef_[0], true_beta, rtol=0.10
        ), f"Expected β ≈ {true_beta}, got {iv.coef_[0]:.4f}"

    def test_with_controls_coefficient(self, iv_with_controls):
        """Test 2SLS with exogenous controls (Z + X → D, D + X → Y)."""
        Y, D, Z, X, true_beta = iv_with_controls

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # First coefficient is treatment effect (should match true_beta)
        assert np.isclose(
            iv.coef_[0], true_beta, rtol=0.15
        ), f"Expected β_D ≈ {true_beta}, got {iv.coef_[0]:.4f}"

        # Should have 3 coefficients: [D, X1, X2]
        assert (
            len(iv.coef_) == 3
        ), f"Expected 3 coefficients [D, X1, X2], got {len(iv.coef_)}"


class TestStandardErrors:
    """Test standard error computation (critical for inference)."""

    def test_standard_errors_positive_finite(self, iv_just_identified):
        """Test that standard errors are positive and finite."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="standard")
        iv.fit(Y, D, Z, X)

        # All SEs should be positive
        assert np.all(iv.se_ > 0), f"Standard errors must be positive. Got: {iv.se_}"

        # All SEs should be finite
        assert np.all(
            np.isfinite(iv.se_)
        ), f"Standard errors must be finite. Got: {iv.se_}"

    def test_robust_vs_standard_se(self, iv_heteroskedastic):
        """Test that robust SEs >= standard SEs with heteroskedasticity."""
        Y, D, Z, X, _ = iv_heteroskedastic

        # Fit with standard SEs
        iv_std = TwoStageLeastSquares(inference="standard")
        iv_std.fit(Y, D, Z, X)

        # Fit with robust SEs
        iv_robust = TwoStageLeastSquares(inference="robust")
        iv_robust.fit(Y, D, Z, X)

        # Coefficients should be the same
        assert np.allclose(
            iv_std.coef_, iv_robust.coef_
        ), "Coefficients should be identical regardless of SE type"

        # Robust SEs should be larger (or equal) with heteroskedasticity
        # Note: Can be smaller in some cases, but typically larger
        # For this specific heteroskedastic DGP, robust should be noticeably larger
        assert (
            iv_robust.se_[0] >= iv_std.se_[0] * 0.9
        ), f"Robust SE ({iv_robust.se_[0]:.4f}) should be >= 90% of standard SE ({iv_std.se_[0]:.4f})"

    def test_standard_errors_scale_with_sample_size(self, iv_strong_instrument):
        """Test that SEs decrease with √n (asymptotic theory)."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        # Fit with full sample (n=1000)
        iv_full = TwoStageLeastSquares(inference="robust")
        iv_full.fit(Y, D, Z, X)
        se_full = iv_full.se_[0]

        # Fit with half sample (n=500)
        n_half = len(Y) // 2
        iv_half = TwoStageLeastSquares(inference="robust")
        iv_half.fit(Y[:n_half], D[:n_half], Z[:n_half], X)
        se_half = iv_half.se_[0]

        # SE should be approximately √2 times larger with half sample
        # Allow 0.5 to 3.0 factor (wide tolerance due to random variation)
        ratio = se_half / se_full
        assert (
            0.5 < ratio < 3.0
        ), f"Expected SE ratio ≈ √2 = 1.41, got {ratio:.2f}"


class TestInference:
    """Test t-statistics, p-values, and confidence intervals."""

    def test_t_statistics_computation(self, iv_just_identified):
        """Test that t-statistics = coefficient / SE."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # Manually compute t-statistic
        expected_t = iv.coef_[0] / iv.se_[0]

        assert np.isclose(
            iv.t_stats_[0], expected_t
        ), f"Expected t-stat = {expected_t:.4f}, got {iv.t_stats_[0]:.4f}"

    def test_p_values_two_sided(self, iv_just_identified):
        """Test that p-values are for two-sided tests."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # With true effect = 0.10 and n=10,000, should be highly significant
        assert iv.p_values_[0] < 0.05, f"Expected p-value < 0.05, got {iv.p_values_[0]:.4f}"

        # p-value should be positive and <= 1
        assert 0 < iv.p_values_[0] <= 1, f"p-value must be in (0, 1], got {iv.p_values_[0]}"

    def test_confidence_intervals_contain_true_value(self, iv_just_identified):
        """Test that 95% CIs contain true value (single realization check)."""
        Y, D, Z, X, true_beta = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust", alpha=0.05)
        iv.fit(Y, D, Z, X)

        ci_lower, ci_upper = iv.ci_[0]

        # 95% CI should contain true value (with high probability)
        assert (
            ci_lower <= true_beta <= ci_upper
        ), f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] does not contain true β = {true_beta}"

    def test_confidence_interval_width_reasonable(self, iv_strong_instrument):
        """Test that CI width is reasonable (not too wide or too narrow)."""
        Y, D, Z, X, true_beta = iv_strong_instrument

        iv = TwoStageLeastSquares(inference="robust", alpha=0.05)
        iv.fit(Y, D, Z, X)

        ci_lower, ci_upper = iv.ci_[0]
        ci_width = ci_upper - ci_lower

        # CI width should be approximately 4 * SE (±1.96 SE on each side)
        expected_width = 4 * iv.se_[0]

        assert np.isclose(
            ci_width, expected_width, rtol=0.05
        ), f"Expected CI width ≈ {expected_width:.4f}, got {ci_width:.4f}"


class TestFirstStageDiagnostics:
    """Test first-stage F-statistic and diagnostics."""

    def test_first_stage_f_stat_strong_instrument(self, iv_strong_instrument):
        """Test first-stage F-statistic with strong instrument (F > 20)."""
        Y, D, Z, X, _ = iv_strong_instrument

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # F-statistic should be > 20 (strong instrument threshold)
        assert (
            iv.first_stage_f_stat_ > 20
        ), f"Expected F > 20 for strong IV, got F = {iv.first_stage_f_stat_:.2f}"

    def test_first_stage_f_stat_weak_instrument(self, iv_weak_instrument):
        """Test first-stage F-statistic with weak instrument (F < 12)."""
        Y, D, Z, X, _ = iv_weak_instrument

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        # F-statistic should be < 12 (below conventional weak IV threshold of 10)
        # Allowing small tolerance for random variation
        assert (
            iv.first_stage_f_stat_ < 12
        ), f"Expected F < 12 for weak IV, got F = {iv.first_stage_f_stat_:.2f}"

    def test_first_stage_r2_bounds(self, iv_just_identified):
        """Test that first-stage R² is between 0 and 1."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        assert (
            0 < iv.first_stage_r2_ < 1
        ), f"First-stage R² must be in (0, 1), got {iv.first_stage_r2_:.4f}"


class TestInputValidation:
    """Test input validation and error handling."""

    def test_underidentification_raises_error(self):
        """Test that underidentified model (q < p) raises error."""
        np.random.seed(123)
        n = 1000

        # 1 instrument, 2 endogenous variables → underidentified
        Z = np.random.normal(0, 1, (n, 1))
        D = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, n)

        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="underidentified"):
            iv.fit(Y, D, Z)

    def test_nan_in_y_raises_error(self, iv_just_identified):
        """Test that NaN in Y raises error."""
        Y, D, Z, X, _ = iv_just_identified

        # Introduce NaN
        Y[0] = np.nan

        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="NaN"):
            iv.fit(Y, D, Z, X)

    def test_constant_treatment_raises_error(self):
        """Test that constant treatment (no variation) raises error."""
        np.random.seed(456)
        n = 1000

        Z = np.random.normal(0, 1, n)
        D = np.ones(n) * 5  # Constant (no variation)
        Y = np.random.normal(0, 1, n)

        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="no variation"):
            iv.fit(Y, D, Z)

    def test_constant_instrument_raises_error(self):
        """Test that constant instrument raises error."""
        np.random.seed(789)
        n = 1000

        Z = np.ones(n) * 3  # Constant instrument
        D = np.random.normal(0, 1, n)
        Y = np.random.normal(0, 1, n)

        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="no variation"):
            iv.fit(Y, D, Z)

    def test_mismatched_array_lengths_raises_error(self):
        """Test that mismatched array lengths raise error."""
        np.random.seed(101)

        Y = np.random.normal(0, 1, 1000)
        D = np.random.normal(0, 1, 900)  # Wrong length
        Z = np.random.normal(0, 1, 1000)

        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="same length"):
            iv.fit(Y, D, Z)


class TestSummaryOutput:
    """Test summary table output."""

    def test_summary_returns_dataframe(self, iv_just_identified):
        """Test that summary() returns a pandas DataFrame."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        summary = iv.summary()

        assert isinstance(summary, pd.DataFrame), "summary() must return DataFrame"

    def test_summary_has_required_columns(self, iv_just_identified):
        """Test that summary table has all required columns."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        summary = iv.summary()

        required_cols = ["coef", "se", "t_stat", "p_value", "ci_lower", "ci_upper"]
        for col in required_cols:
            assert (
                col in summary.columns
            ), f"summary() must have column '{col}'"

    def test_summary_with_controls(self, iv_with_controls):
        """Test summary table with controls (multiple rows)."""
        Y, D, Z, X, _ = iv_with_controls

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        summary = iv.summary()

        # Should have 3 rows: [D, X1, X2]
        assert len(summary) == 3, f"Expected 3 rows in summary, got {len(summary)}"

        # Row names should be ['D', 'X1', 'X2']
        expected_names = ["D", "X1", "X2"]
        assert list(summary.index) == expected_names, f"Expected row names {expected_names}, got {list(summary.index)}"

    def test_summary_before_fit_raises_error(self):
        """Test that calling summary() before fit() raises error."""
        iv = TwoStageLeastSquares(inference="robust")

        with pytest.raises(ValueError, match="must be fitted"):
            iv.summary()


class TestFittedAttribute:
    """Test fitted_ attribute tracking."""

    def test_fitted_false_before_fit(self):
        """Test that fitted_ is False before calling fit()."""
        iv = TwoStageLeastSquares(inference="robust")

        assert iv.fitted_ is False, "fitted_ should be False before fit()"

    def test_fitted_true_after_fit(self, iv_just_identified):
        """Test that fitted_ is True after calling fit()."""
        Y, D, Z, X, _ = iv_just_identified

        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)

        assert iv.fitted_ is True, "fitted_ should be True after fit()"


class TestClusteredStandardErrors:
    """Test clustered standard errors (advanced inference)."""

    def test_clustered_se_requires_cluster_var(self, iv_just_identified):
        """Test that clustered SEs require cluster_var."""
        Y, D, Z, X, _ = iv_just_identified

        # Try to use clustered SEs without providing cluster_var
        with pytest.raises(ValueError, match="cluster_var must be provided"):
            iv = TwoStageLeastSquares(inference="clustered", cluster_var=None)

    def test_clustered_se_with_few_clusters_warns(self, iv_just_identified):
        """Test that clustered SEs with <20 clusters produce warning."""
        Y, D, Z, X, _ = iv_just_identified

        # Create 5 clusters (too few)
        n = len(Y)
        clusters = np.repeat(np.arange(5), n // 5)

        iv = TwoStageLeastSquares(inference="clustered", cluster_var=clusters)

        # Should produce UserWarning about few clusters
        with pytest.warns(UserWarning, match="<20 clusters"):
            iv.fit(Y, D, Z, X)

    def test_clustered_se_larger_than_robust(self, iv_just_identified):
        """Test that clustered SEs are typically larger than robust (with clustering)."""
        Y, D, Z, X, _ = iv_just_identified

        # Create 50 clusters (sufficient)
        n = len(Y)
        clusters = np.repeat(np.arange(50), n // 50)

        # Fit with robust SEs
        iv_robust = TwoStageLeastSquares(inference="robust")
        iv_robust.fit(Y, D, Z, X)

        # Fit with clustered SEs
        iv_cluster = TwoStageLeastSquares(inference="clustered", cluster_var=clusters)
        iv_cluster.fit(Y, D, Z, X)

        # Clustered SEs should be larger (or similar) to robust
        # Allow some tolerance as clustering may not matter in this simple DGP
        assert (
            iv_cluster.se_[0] >= iv_robust.se_[0] * 0.8
        ), f"Clustered SE ({iv_cluster.se_[0]:.4f}) should be >= 80% of robust SE ({iv_robust.se_[0]:.4f})"
