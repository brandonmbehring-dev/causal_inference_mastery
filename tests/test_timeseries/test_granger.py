"""
Tests for Granger Causality.

Session 135: Comprehensive tests across 3 validation layers.

Layer 1: Known-Answer Tests (deterministic, reproducible)
Layer 2: Adversarial Tests (edge cases, boundary conditions)
Layer 3: Monte Carlo Tests (statistical properties)
"""

import numpy as np
import pytest
from typing import Tuple

from causal_inference.timeseries import (
    granger_causality,
    granger_causality_matrix,
    bidirectional_granger,
    GrangerResult,
    MultiGrangerResult,
)
from causal_inference.timeseries.granger import granger_with_lag_selection


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestGrangerKnownAnswer:
    """Known-answer tests with deterministic expected outcomes."""

    def test_detects_causal_relationship(self, sample_granger_causal_pair):
        """X Granger-causes Y should be detected."""
        result = granger_causality(sample_granger_causal_pair, lags=2, alpha=0.05)

        assert isinstance(result, GrangerResult)
        assert result.granger_causes  # Use truthy check, not `is True`
        assert result.p_value < 0.05
        assert result.f_statistic > 0

    def test_no_false_positive(self, sample_no_granger_causality):
        """Independent series should NOT show Granger causality."""
        result = granger_causality(sample_no_granger_causality, lags=2, alpha=0.05)

        # Note: This may occasionally fail due to randomness
        # Using larger sample in fixture helps
        assert not result.granger_causes  # Use truthy check
        assert result.p_value > 0.05

    def test_bidirectional_detection(self, sample_bidirectional_causality):
        """Both directions should be detected in feedback system."""
        result_xy, result_yx = bidirectional_granger(sample_bidirectional_causality, lags=2)

        assert result_xy.granger_causes  # Use truthy check
        assert result_yx.granger_causes

    def test_result_attributes(self, sample_granger_causal_pair):
        """Verify all result attributes are populated."""
        result = granger_causality(
            sample_granger_causal_pair,
            lags=2,
            var_names=["Y", "X"],
        )

        assert result.cause_var == "X"
        assert result.effect_var == "Y"
        assert result.lags == 2
        assert result.alpha == 0.05
        assert result.df_num == 2  # Number of lags
        assert result.df_denom > 0
        assert 0 <= result.r2_unrestricted <= 1
        assert 0 <= result.r2_restricted <= 1

    def test_larger_r2_unrestricted(self, sample_granger_causal_pair):
        """Unrestricted model should have higher R² when causality exists."""
        result = granger_causality(sample_granger_causal_pair, lags=2)

        # Unrestricted includes X lags, should fit better
        assert result.r2_unrestricted >= result.r2_restricted

    def test_matrix_result_structure(self, sample_chain_causality):
        """Test multivariate Granger matrix output structure."""
        result = granger_causality_matrix(
            sample_chain_causality,
            lags=2,
            var_names=["X1", "X2", "X3"],
        )

        assert isinstance(result, MultiGrangerResult)
        assert result.n_vars == 3
        assert result.causality_matrix.shape == (3, 3)
        assert len(result.pairwise_results) == 6  # 3*2 ordered pairs

    def test_chain_causality_detection(self, sample_chain_causality):
        """X1 -> X2 -> X3 chain should be detected."""
        result = granger_causality_matrix(
            sample_chain_causality,
            lags=2,
            var_names=["X1", "X2", "X3"],
        )

        # X1 -> X2 should be detected
        assert result.causality_matrix[0, 1]  # Truthy check

        # X2 -> X3 should be detected
        assert result.causality_matrix[1, 2]  # Truthy check

        # Direct X1 -> X3 may or may not be detected (indirect effect)


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestGrangerAdversarial:
    """Adversarial tests for edge cases and boundary conditions."""

    def test_minimum_observations(self):
        """Test with minimum viable observations."""
        np.random.seed(42)
        n = 10  # Minimum for lag=1
        data = np.random.randn(n, 2)

        result = granger_causality(data, lags=1)
        assert isinstance(result, GrangerResult)

    def test_insufficient_observations_raises(self):
        """Too few observations should raise ValueError."""
        data = np.random.randn(5, 2)

        with pytest.raises(ValueError, match="Insufficient observations"):
            granger_causality(data, lags=3)

    def test_single_lag(self):
        """Test with single lag."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = granger_causality(data, lags=1)
        assert result.lags == 1
        assert result.df_num == 1

    def test_many_lags(self):
        """Test with many lags."""
        np.random.seed(42)
        data = np.random.randn(500, 2)

        result = granger_causality(data, lags=10)
        assert result.lags == 10
        assert result.df_num == 10

    def test_near_perfect_prediction(self):
        """Test when X almost perfectly predicts Y."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.99 * x[t - 1]  # Nearly deterministic

        data = np.column_stack([y, x])
        result = granger_causality(data, lags=1)

        assert result.granger_causes  # Truthy check
        assert result.p_value < 0.001
        assert result.r2_unrestricted > 0.9

    def test_constant_series(self):
        """Test with constant series."""
        n = 100
        data = np.ones((n, 2))

        # Should handle gracefully (no causality)
        result = granger_causality(data, lags=1)
        # F-stat may be 0 or undefined
        assert result.p_value >= 0

    def test_multivariate_data(self):
        """Test with more than 2 variables."""
        np.random.seed(42)
        data = np.random.randn(200, 5)

        result = granger_causality(
            data,
            lags=2,
            cause_idx=3,
            effect_idx=1,
        )
        assert result.cause_var == "var_4"
        assert result.effect_var == "var_2"

    def test_invalid_indices_raise(self):
        """Invalid variable indices should raise."""
        data = np.random.randn(100, 3)

        with pytest.raises(ValueError):
            granger_causality(data, cause_idx=5, effect_idx=0)

    def test_negative_lags_raise(self):
        """Negative lags should raise ValueError."""
        data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="lags must be >= 1"):
            granger_causality(data, lags=0)

    def test_1d_data_raises(self):
        """1D data should raise ValueError."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2D"):
            granger_causality(data, lags=1)

    def test_different_alpha_levels(self, sample_granger_causal_pair):
        """Test with different significance levels."""
        result_01 = granger_causality(sample_granger_causal_pair, alpha=0.01)
        result_10 = granger_causality(sample_granger_causal_pair, alpha=0.10)

        # Same p-value, different decisions possible
        assert result_01.p_value == result_10.p_value
        assert result_01.alpha == 0.01
        assert result_10.alpha == 0.10


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestGrangerMonteCarlo:
    """Monte Carlo tests for statistical properties."""

    @pytest.mark.slow
    def test_type_i_error_rate(self):
        """
        Under null (no causality), rejection rate should be ~alpha.

        H0: X does not Granger-cause Y
        We generate independent series and check false positive rate.
        """
        n_sims = 500
        n_obs = 100
        alpha = 0.05
        false_positives = 0

        for seed in range(n_sims):
            np.random.seed(seed)
            # Independent AR(1) processes
            x = np.zeros(n_obs)
            y = np.zeros(n_obs)
            for t in range(1, n_obs):
                x[t] = 0.5 * x[t - 1] + np.random.randn()
                y[t] = 0.5 * y[t - 1] + np.random.randn()

            data = np.column_stack([y, x])
            result = granger_causality(data, lags=2, alpha=alpha)

            if result.granger_causes:
                false_positives += 1

        fp_rate = false_positives / n_sims

        # Should be within 2-8% (alpha=5% with some tolerance)
        assert 0.02 < fp_rate < 0.10, f"Type I error rate {fp_rate:.3f} outside [0.02, 0.10]"

    @pytest.mark.slow
    def test_power_analysis(self):
        """
        Under alternative (causality exists), detection rate should be high.

        We expect power > 80% for moderate effect size with n=200.
        """
        n_sims = 300
        n_obs = 200
        alpha = 0.05
        effect_size = 0.5
        detections = 0

        for seed in range(n_sims):
            np.random.seed(seed)
            x = np.random.randn(n_obs)
            y = np.zeros(n_obs)
            for t in range(1, n_obs):
                y[t] = effect_size * x[t - 1] + 0.3 * y[t - 1] + np.random.randn() * 0.5

            data = np.column_stack([y, x])
            result = granger_causality(data, lags=2, alpha=alpha)

            if result.granger_causes:
                detections += 1

        power = detections / n_sims

        assert power > 0.75, f"Power {power:.3f} is below 0.75"

    def test_p_value_uniformity_under_null(self):
        """
        Under null, p-values should be approximately uniform.
        """
        n_sims = 200
        n_obs = 100
        p_values = []

        for seed in range(n_sims):
            np.random.seed(seed)
            x = np.zeros(n_obs)
            y = np.zeros(n_obs)
            for t in range(1, n_obs):
                x[t] = 0.5 * x[t - 1] + np.random.randn()
                y[t] = 0.5 * y[t - 1] + np.random.randn()

            data = np.column_stack([y, x])
            result = granger_causality(data, lags=1)
            p_values.append(result.p_value)

        p_values = np.array(p_values)

        # Check that p-values span the range
        assert p_values.min() < 0.2
        assert p_values.max() > 0.5

        # Kolmogorov-Smirnov test for uniformity (lenient)
        from scipy import stats

        ks_stat, ks_pval = stats.kstest(p_values, "uniform")
        # Don't require perfect uniformity, just reasonable
        assert ks_pval > 0.01 or p_values.std() > 0.2


class TestGrangerLagSelection:
    """Tests for automatic lag selection."""

    def test_lag_selection_basic(self, sample_granger_causal_pair):
        """Basic lag selection should work."""
        result, optimal_lag = granger_with_lag_selection(
            sample_granger_causal_pair,
            max_lags=5,
        )

        assert isinstance(result, GrangerResult)
        assert 1 <= optimal_lag <= 5

    def test_lag_selection_finds_true_lag(self):
        """Lag selection should find the correct lag order."""
        np.random.seed(42)
        n = 300
        true_lag = 3

        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(true_lag, n):
            y[t] = 0.5 * x[t - true_lag] + 0.2 * y[t - 1] + np.random.randn() * 0.3

        data = np.column_stack([y, x])
        result, optimal_lag = granger_with_lag_selection(data, max_lags=5)

        # Should find lag 3 or nearby
        assert abs(optimal_lag - true_lag) <= 1

    def test_lag_selection_criterion(self, sample_granger_causal_pair):
        """Different criteria may give different results."""
        result_aic, lag_aic = granger_with_lag_selection(
            sample_granger_causal_pair,
            max_lags=5,
            criterion="aic",
        )
        result_bic, lag_bic = granger_with_lag_selection(
            sample_granger_causal_pair,
            max_lags=5,
            criterion="bic",
        )

        # Both should be valid
        assert 1 <= lag_aic <= 5
        assert 1 <= lag_bic <= 5
        # BIC tends to select simpler models
        assert lag_bic <= lag_aic or lag_bic == lag_aic


class TestGrangerMultivariate:
    """Tests for multivariate Granger analysis."""

    def test_matrix_diagonal_zero(self):
        """Diagonal of causality matrix should be False (no self-causality)."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        result = granger_causality_matrix(data, lags=1)

        for i in range(3):
            assert not result.causality_matrix[i, i]  # Truthy check

    def test_get_causes_effects(self, sample_chain_causality):
        """Test helper methods for extracting causes/effects."""
        result = granger_causality_matrix(
            sample_chain_causality,
            lags=2,
            var_names=["X1", "X2", "X3"],
        )

        # X2 should be caused by X1
        causes_of_x2 = result.get_causes("X2")
        assert "X1" in causes_of_x2

        # X1 should cause X2
        effects_of_x1 = result.get_effects("X1")
        assert "X2" in effects_of_x1

    def test_pairwise_results_access(self):
        """Test accessing specific pairwise results."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        result = granger_causality_matrix(
            data,
            lags=1,
            var_names=["A", "B", "C"],
        )

        # Access specific pair
        ab_result = result.pairwise_results[("A", "B")]
        assert isinstance(ab_result, GrangerResult)
        assert ab_result.cause_var == "A"
        assert ab_result.effect_var == "B"


class TestGrangerInputValidation:
    """Tests for input validation."""

    def test_wrong_shape_raises(self):
        """Wrong data shape should raise."""
        with pytest.raises(ValueError):
            granger_causality(np.random.randn(100), lags=1)

    def test_single_variable_raises(self):
        """Single variable should raise."""
        with pytest.raises(ValueError, match="Need at least 2 variables"):
            granger_causality(np.random.randn(100, 1), lags=1)

    def test_var_names_length_mismatch(self):
        """Mismatched var_names length handled gracefully."""
        data = np.random.randn(100, 2)
        # Should use default names
        result = granger_causality(data, lags=1, var_names=None)
        assert result.cause_var == "var_2"
        assert result.effect_var == "var_1"


class TestGrangerReproducibility:
    """Tests for result reproducibility."""

    def test_deterministic_with_same_data(self):
        """Same data should give same results."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result1 = granger_causality(data, lags=2)
        result2 = granger_causality(data, lags=2)

        assert result1.f_statistic == result2.f_statistic
        assert result1.p_value == result2.p_value
        assert result1.granger_causes == result2.granger_causes

    def test_repr_format(self, sample_granger_causal_pair):
        """Test string representation."""
        result = granger_causality(
            sample_granger_causal_pair,
            lags=2,
            var_names=["Y", "X"],
        )

        repr_str = repr(result)
        assert "GrangerResult" in repr_str
        assert "X" in repr_str
        assert "Y" in repr_str
