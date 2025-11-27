"""Tests for doubly robust (DR) estimation.

Tests cover:
1. Both models correct (ideal case, lowest variance)
2. Propensity correct, outcome misspecified (IPW protects)
3. Outcome correct, propensity misspecified (regression protects)
4. Both models misspecified (biased but runs)
5. Trimming integration
6. Pre-computed inputs
7. Error handling

Following test-first principles with known-answer tests.
"""

import numpy as np
import pytest
from src.causal_inference.observational.doubly_robust import dr_ate


class TestDoublyRobustBothCorrect:
    """Test DR estimator when both propensity and outcome models are correct."""

    def test_both_models_correct_linear(self):
        """
        Test with both models correctly specified (ideal case).

        DGP:
            X ~ N(0, 1)
            e(X) = logistic(0.8*X)  # Propensity (correctly modeled)
            μ(X) = 2 + 0.5*X        # Outcome (correctly modeled)
            Y = τ*T + μ(X) + ε      # True ATE = 3.0

        Both models correct → DR should recover ATE with lowest variance.
        """
        np.random.seed(42)
        n = 300

        X = np.random.normal(0, 1, n)

        # Generate treatment via logistic propensity
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Generate outcomes with true ATE = 3.0
        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should recover ATE ≈ 3.0
        assert np.abs(result["estimate"] - 3.0) < 0.3

        # Should have reasonable SE
        assert 0.05 < result["se"] < 0.30

        # CI should cover true ATE
        assert result["ci_lower"] < 3.0 < result["ci_upper"]

        # Diagnostics present
        assert "propensity_diagnostics" in result
        assert "outcome_diagnostics" in result

        # Sample sizes correct
        assert result["n"] == n
        assert result["n_treated"] + result["n_control"] == n

    def test_both_models_correct_multiple_covariates(self):
        """Test with multiple covariates, both models correct."""
        np.random.seed(123)
        n = 400

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        # Propensity depends on both covariates
        logit_prop = 0.5 * X1 - 0.3 * X2
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome depends on both covariates
        Y = 2.5 * T + (1 + 0.4 * X1 + 0.3 * X2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should recover ATE ≈ 2.5
        assert np.abs(result["estimate"] - 2.5) < 0.3

        # Propensity diagnostics should show confounding
        assert result["propensity_diagnostics"]["auc"] > 0.6


class TestDoublyRobustPropensityCorrect:
    """Test DR when propensity correct but outcome model misspecified."""

    def test_propensity_correct_outcome_wrong_quadratic(self):
        """
        Test when propensity correct but outcome model misspecified.

        DGP:
            e(X) = logistic(0.8*X)      # Correct (logistic in X)
            μ_true(X) = 2 + 0.5*X²      # True outcome (quadratic)
            μ_fit(X) = β₀ + β₁*X        # Fitted outcome (linear, WRONG!)
            Y = τ*T + μ_true(X) + ε

        DR should still be consistent because propensity is correct (IPW protects).
        """
        np.random.seed(456)
        n = 400

        X = np.random.normal(0, 1, n)

        # Propensity: linear in X (correctly modeled by logistic regression)
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome: QUADRATIC in X (misspecified by linear regression)
        Y = 3.0 * T + (2 + 0.5 * X**2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # DR should still recover ATE despite outcome misspecification
        # (IPW protects via correct propensity)
        assert np.abs(result["estimate"] - 3.0) < 0.4  # Slightly more bias tolerated

        # Should run without error
        assert result["n"] == n

    def test_propensity_correct_outcome_interaction(self):
        """Test with propensity correct but outcome has treatment × covariate interaction."""
        np.random.seed(789)
        n = 350

        X = np.random.normal(0, 1, n)

        # Propensity: linear in X
        logit_prop = 0.6 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome: treatment effect varies with X (heterogeneous)
        # This means separate linear models for T=0 and T=1 will be misspecified
        Y = (2.0 + 0.5 * X) * T + (1 + 0.3 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run and produce finite estimate
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])


class TestDoublyRobustOutcomeCorrect:
    """Test DR when outcome correct but propensity model misspecified."""

    def test_outcome_correct_propensity_wrong_quadratic(self):
        """
        Test when outcome correct but propensity misspecified.

        DGP:
            e_true(X) = logistic(0.8*X²)    # True propensity (quadratic)
            e_fit(X) = logistic(β₀ + β₁*X)  # Fitted propensity (linear, WRONG!)
            μ(X) = 2 + 0.5*X                # Outcome (linear, correct)
            Y = τ*T + μ(X) + ε

        DR should still be consistent because outcome model is correct (regression protects).
        """
        np.random.seed(101)
        n = 400

        X = np.random.normal(0, 1, n)

        # Propensity: QUADRATIC in X (misspecified by linear logistic regression)
        logit_prop = 0.8 * X**2 - 1.0  # Shift to avoid extreme probabilities
        e_X = 1 / (1 + np.exp(-logit_prop))
        e_X = np.clip(e_X, 0.1, 0.9)  # Clip to avoid perfect separation
        T = np.random.binomial(1, e_X)

        # Outcome: LINEAR in X (correctly modeled by linear regression)
        Y = 2.5 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # DR should still recover ATE despite propensity misspecification
        # (Outcome model protects via correct regression)
        assert np.abs(result["estimate"] - 2.5) < 0.5  # More bias tolerated

        # Should run without error
        assert result["n"] == n

    def test_outcome_correct_propensity_interaction(self):
        """Test with outcome correct but propensity depends on interaction."""
        np.random.seed(202)
        n = 300

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        # Propensity: interaction term (misspecified by additive model)
        logit_prop = 0.5 * X1 * X2  # Interaction
        e_X = 1 / (1 + np.exp(-logit_prop))
        e_X = np.clip(e_X, 0.2, 0.8)
        T = np.random.binomial(1, e_X)

        # Outcome: additive (correctly modeled)
        Y = 3.0 * T + (1 + 0.4 * X1 + 0.3 * X2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run and produce finite estimate
        assert np.isfinite(result["estimate"])
        assert result["n"] == n


class TestDoublyRobustBothWrong:
    """Test DR when both models are misspecified (no protection)."""

    def test_both_models_wrong_runs_without_error(self):
        """
        Test that DR runs even when both models are wrong (may be biased).

        DGP:
            e_true(X) = logistic(0.8*X²)    # Quadratic
            e_fit(X) = logistic(β₀ + β₁*X)  # Linear (WRONG!)
            μ_true(X) = 2 + 0.5*X²          # Quadratic
            μ_fit(X) = β₀ + β₁*X            # Linear (WRONG!)

        DR not guaranteed consistent, but should run without crashing.
        """
        np.random.seed(303)
        n = 300

        X = np.random.normal(0, 1, n)

        # Propensity: quadratic (misspecified as linear)
        logit_prop = 0.8 * X**2 - 1.0
        e_X = 1 / (1 + np.exp(-logit_prop))
        e_X = np.clip(e_X, 0.15, 0.85)
        T = np.random.binomial(1, e_X)

        # Outcome: quadratic (misspecified as linear)
        Y = 3.0 * T + (2 + 0.5 * X**2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run without error
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])
        assert result["n"] == n

        # CI should be valid
        assert result["ci_lower"] < result["ci_upper"]


class TestDoublyRobustTrimming:
    """Test DR with propensity trimming."""

    def test_trimming_reduces_sample_size(self):
        """Test that trimming removes units with extreme propensities."""
        np.random.seed(404)
        n = 300

        X = np.random.normal(0, 1, n)

        # Create strong confounding (extreme propensities)
        logit_prop = 2.0 * X  # Strong effect
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        # Without trimming
        result_no_trim = dr_ate(Y, T, X)
        assert result_no_trim["n_trimmed"] == 0

        # With trimming
        result_trimmed = dr_ate(Y, T, X, trim_at=(0.05, 0.95))

        # Should trim some units
        assert result_trimmed["n_trimmed"] > 0
        assert result_trimmed["n"] < n
        assert result_trimmed["n"] + result_trimmed["n_trimmed"] == n

    def test_trimming_with_both_models_correct(self):
        """Test that trimming works when both models correct."""
        np.random.seed(505)
        n = 400

        X = np.random.normal(0, 1, n)

        logit_prop = 1.5 * X  # Moderate confounding
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 2.5 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X, trim_at=(0.1, 0.9))

        # Should recover ATE reasonably well
        assert np.abs(result["estimate"] - 2.5) < 0.4

        # Should have trimmed
        assert result["n_trimmed"] > 0


class TestDoublyRobustPrecomputedInputs:
    """Test DR with pre-computed propensity scores and outcome models."""

    def test_precomputed_propensity(self):
        """Test with pre-computed propensity scores."""
        np.random.seed(606)
        n = 200

        X = np.random.normal(0, 1, n)

        # True propensity
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        # Provide exact propensity scores
        result = dr_ate(Y, T, X, propensity=e_X)

        # Should use provided propensity
        assert result["propensity_diagnostics"]["provided"] is True

        # Should recover ATE
        assert np.abs(result["estimate"] - 3.0) < 0.4

    def test_precomputed_outcome_models(self):
        """Test with pre-computed outcome model predictions."""
        np.random.seed(707)
        n = 200

        X = np.random.normal(0, 1, n)

        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        # True outcome models
        mu0_true = 2 + 0.5 * X
        mu1_true = 5 + 0.5 * X  # Shifted by ATE = 3.0

        outcome_models = {"mu0_predictions": mu0_true, "mu1_predictions": mu1_true}

        result = dr_ate(Y, T, X, outcome_models=outcome_models)

        # Should use provided models
        assert result["outcome_diagnostics"]["provided"] is True

        # Should recover ATE very well (true models provided)
        assert np.abs(result["estimate"] - 3.0) < 0.3

    def test_precomputed_both(self):
        """Test with both pre-computed propensity and outcome models."""
        np.random.seed(808)
        n = 200

        X = np.random.normal(0, 1, n)

        # True propensity
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        # True models
        mu0_true = 2 + 0.5 * X
        mu1_true = 5 + 0.5 * X

        outcome_models = {"mu0_predictions": mu0_true, "mu1_predictions": mu1_true}

        result = dr_ate(Y, T, X, propensity=e_X, outcome_models=outcome_models)

        # Both should be marked as provided
        assert result["propensity_diagnostics"]["provided"] is True
        assert result["outcome_diagnostics"]["provided"] is True

        # Should recover ATE very accurately (oracle case)
        assert np.abs(result["estimate"] - 3.0) < 0.2


class TestDoublyRobustErrorHandling:
    """Test error handling for dr_ate."""

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50)  # Wrong length
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            dr_ate(Y, T, X)

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
            dr_ate(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "NaN" in error_msg

    def test_non_binary_treatment_fails_fast(self):
        """Test that non-binary treatment raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([0, 1, 2] * 33 + [0])  # Has value 2
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            dr_ate(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "binary" in error_msg

    def test_propensity_length_mismatch_fails_fast(self):
        """Test that propensity length mismatch raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)
        prop = np.random.uniform(0.2, 0.8, 50)  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            dr_ate(Y, T, X, propensity=prop)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Propensity length mismatch" in error_msg

    def test_outcome_models_missing_keys_fails_fast(self):
        """Test that outcome models without required keys raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)

        # Missing keys
        outcome_models = {"mu0": np.zeros(100)}  # Should be "mu0_predictions"

        with pytest.raises(ValueError) as exc_info:
            dr_ate(Y, T, X, outcome_models=outcome_models)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "mu0_predictions" in error_msg

    def test_outcome_models_length_mismatch_fails_fast(self):
        """Test that outcome model predictions length mismatch raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)

        # Wrong length
        outcome_models = {
            "mu0_predictions": np.zeros(50),  # Should be 100
            "mu1_predictions": np.ones(100),
        }

        with pytest.raises(ValueError) as exc_info:
            dr_ate(Y, T, X, outcome_models=outcome_models)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "length mismatch" in error_msg
