"""Tests for outcome regression models.

Tests cover:
1. Linear outcome models (correct specification)
2. Misspecified outcome models
3. Single and multiple covariates
4. Error handling

Following test-first principles with known-answer tests.
"""

import numpy as np
import pytest
from src.causal_inference.observational.outcome_regression import fit_outcome_models


class TestOutcomeModels:
    """Test outcome regression with correct specification."""

    def test_linear_outcome_correct_specification(self):
        """
        Test with true linear outcome model.

        DGP:
            X ~ N(0, 1)
            Y(0) = 2 + 0.5*X + ε  # Control outcome
            Y(1) = 5 + 0.5*X + ε  # Treated outcome (ATE = 3)
        """
        np.random.seed(42)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)
        np.random.shuffle(T)

        # Generate outcomes with true ATE = 3.0
        Y = np.where(T == 1, 5 + 0.5 * X, 2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = fit_outcome_models(Y, T, X)

        # Should recover ATE ≈ 3.0 from difference in predictions
        ate_estimate = np.mean(result["mu1_predictions"] - result["mu0_predictions"])
        assert np.abs(ate_estimate - 3.0) < 0.3

        # R² should be reasonable (correct specification + noise)
        assert result["diagnostics"]["mu0_r2"] > 0.4
        assert result["diagnostics"]["mu1_r2"] > 0.4

        # Predictions should have correct shape
        assert result["mu0_predictions"].shape == (n,)
        assert result["mu1_predictions"].shape == (n,)

    def test_multiple_covariates(self):
        """Test with multiple covariates."""
        np.random.seed(123)
        n = 300

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        T = np.array([1] * 150 + [0] * 150)
        np.random.shuffle(T)

        # Outcome depends on both covariates
        Y = np.where(T == 1, 4 + 0.5 * X1 + 0.3 * X2, 1 + 0.5 * X1 + 0.3 * X2)
        Y += np.random.normal(0, 0.5, n)

        result = fit_outcome_models(Y, T, X)

        # ATE ≈ 3.0
        ate_estimate = np.mean(result["mu1_predictions"] - result["mu0_predictions"])
        assert np.abs(ate_estimate - 3.0) < 0.3

        # Both models should fit well
        assert result["diagnostics"]["mu0_r2"] > 0.5
        assert result["diagnostics"]["mu1_r2"] > 0.5

    def test_single_covariate_1d_array(self):
        """Test with 1D covariate array."""
        np.random.seed(456)
        n = 200

        X = np.random.normal(0, 1, n)  # 1D array
        T = np.array([1] * 100 + [0] * 100)
        Y = np.where(T == 1, 3 + X, X) + np.random.normal(0, 0.5, n)

        result = fit_outcome_models(Y, T, X)

        # Should handle 1D covariates
        ate_estimate = np.mean(result["mu1_predictions"] - result["mu0_predictions"])
        assert np.abs(ate_estimate - 3.0) < 0.3

    def test_diagnostics_included(self):
        """Test that diagnostics are properly computed."""
        np.random.seed(789)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)
        Y = np.where(T == 1, 3 + X, X) + np.random.normal(0, 0.5, n)

        result = fit_outcome_models(Y, T, X)

        diag = result["diagnostics"]

        # Check all diagnostics present
        assert "mu0_r2" in diag
        assert "mu1_r2" in diag
        assert "mu0_rmse" in diag
        assert "mu1_rmse" in diag
        assert "n_control" in diag
        assert "n_treated" in diag

        # Check values reasonable
        assert 0 <= diag["mu0_r2"] <= 1
        assert 0 <= diag["mu1_r2"] <= 1
        assert diag["mu0_rmse"] > 0
        assert diag["mu1_rmse"] > 0
        assert diag["n_control"] == 100
        assert diag["n_treated"] == 100


class TestMisspecifiedModels:
    """Test with misspecified outcome models."""

    def test_quadratic_truth_linear_fit(self):
        """
        Test with quadratic truth but linear model fit.

        DGP:
            Y(0) = 2 + 0.5*X² + ε  # Quadratic
            Y(1) = 5 + 0.5*X² + ε
        Fitted:
            μ(X) = β₀ + β₁*X      # Linear (misspecified!)
        """
        np.random.seed(101)
        n = 300

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 150 + [0] * 150)
        np.random.shuffle(T)

        # True outcome is quadratic in X
        Y = np.where(T == 1, 5 + 0.5 * X**2, 2 + 0.5 * X**2)
        Y += np.random.normal(0, 0.5, n)

        # Fit linear model (misspecified)
        result = fit_outcome_models(Y, T, X)

        # Should still run and produce estimates
        assert result["mu0_predictions"].shape == (n,)
        assert result["mu1_predictions"].shape == (n,)

        # R² will be lower (misspecification)
        assert result["diagnostics"]["mu0_r2"] >= 0  # Non-negative
        assert result["diagnostics"]["mu1_r2"] >= 0

        # ATE estimate may be biased but finite
        ate_estimate = np.mean(result["mu1_predictions"] - result["mu0_predictions"])
        assert np.isfinite(ate_estimate)

    def test_interaction_effects_not_modeled(self):
        """Test with treatment × covariate interaction (misspecified)."""
        np.random.seed(202)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)

        # Treatment effect varies with X (heterogeneous)
        Y = np.where(T == 1, 3 * (1 + 0.5 * X), 0) + X + np.random.normal(0, 0.5, n)

        # Separate models can capture some heterogeneity
        result = fit_outcome_models(Y, T, X)

        # Should run successfully
        assert result["mu0_predictions"].shape == (n,)
        assert result["mu1_predictions"].shape == (n,)


class TestErrorHandling:
    """Test error handling for fit_outcome_models."""

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50)  # Wrong length
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "different lengths" in error_msg

    def test_nan_in_outcomes_fails_fast(self):
        """Test that NaN in outcomes raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)
        Y[10] = np.nan

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "NaN" in error_msg

    def test_infinite_values_fails_fast(self):
        """Test that infinite values raise ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)
        Y[10] = np.inf

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Infinite" in error_msg

    def test_non_binary_treatment_fails_fast(self):
        """Test that non-binary treatment raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([0, 1, 2] * 33 + [0])  # Has value 2
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "binary" in error_msg

    def test_insufficient_control_units_fails_fast(self):
        """Test that too few control units raises ValueError."""
        X = np.random.normal(0, 1, 10)
        T = np.array([1] * 9 + [0] * 1)  # Only 1 control
        Y = np.random.normal(0, 1, 10)

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Insufficient control" in error_msg

    def test_insufficient_treated_units_fails_fast(self):
        """Test that too few treated units raises ValueError."""
        X = np.random.normal(0, 1, 10)
        T = np.array([1] * 1 + [0] * 9)  # Only 1 treated
        Y = np.random.normal(0, 1, 10)

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Insufficient treated" in error_msg

    def test_invalid_method_fails_fast(self):
        """Test that invalid method raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            fit_outcome_models(Y, T, X, method="random_forest")

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Unsupported modeling method" in error_msg
