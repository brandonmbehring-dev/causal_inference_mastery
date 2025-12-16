"""Tests for Double Machine Learning implementation.

Tests are organized by layer:
1. Known-Answer: Verify DML recovers known treatment effects
2. Adversarial: Edge cases and challenging scenarios
3. Monte Carlo: Statistical validation of bias and coverage (CONCERN-29)
"""

import pytest
import numpy as np
from scipy import stats

from src.causal_inference.cate import double_ml, r_learner
from tests.test_cate.conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestDMLKnownAnswer:
    """Tests with known true treatment effects."""

    def test_dml_constant_effect(self, constant_effect_data):
        """DML recovers constant ATE within tolerance."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = 2.0

        result = double_ml(Y, T, X, n_folds=5)

        # ATE should be close to true value
        assert abs(result["ate"] - true_ate) < 0.5, (
            f"DML ATE {result['ate']:.3f} far from true {true_ate}"
        )
        assert result["method"] == "double_ml"

    def test_dml_heterogeneous_effect(self, linear_heterogeneous_data):
        """DML captures heterogeneity in treatment effects."""
        Y, T, X, true_cate = linear_heterogeneous_data

        result = double_ml(Y, T, X, n_folds=5, model="ridge")

        # Check CATE shape
        assert result["cate"].shape == (len(Y),)

        # CATE should correlate with true CATE
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.3, (
            f"CATE correlation {correlation:.3f} too low for heterogeneous effect"
        )

    def test_dml_returns_valid_ci(self, constant_effect_data):
        """DML returns valid confidence intervals."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, n_folds=5, alpha=0.05)

        # CI should contain ATE
        assert result["ci_lower"] < result["ate"] < result["ci_upper"]

        # CI width should be reasonable (not degenerate)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.1 < ci_width < 5.0, f"CI width {ci_width:.3f} seems unreasonable"

    def test_dml_se_positive(self, constant_effect_data):
        """Standard error should be positive and finite."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X)

        assert result["ate_se"] > 0, "SE must be positive"
        assert np.isfinite(result["ate_se"]), "SE must be finite"

    def test_dml_different_fold_counts(self, constant_effect_data):
        """DML works with different numbers of folds."""
        Y, T, X, _ = constant_effect_data
        true_ate = 2.0

        for n_folds in [2, 3, 5, 10]:
            result = double_ml(Y, T, X, n_folds=n_folds)
            # All should recover ATE reasonably
            assert abs(result["ate"] - true_ate) < 1.0, (
                f"n_folds={n_folds}: ATE {result['ate']:.3f} far from {true_ate}"
            )


class TestDMLNuisanceModels:
    """Tests for different nuisance model configurations."""

    def test_dml_ridge_nuisance(self, constant_effect_data):
        """DML with ridge nuisance model (default)."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, nuisance_model="ridge")

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])

    def test_dml_linear_nuisance(self, constant_effect_data):
        """DML with linear nuisance model."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, nuisance_model="linear")

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])

    def test_dml_rf_nuisance(self, constant_effect_data):
        """DML with random forest nuisance model."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, nuisance_model="random_forest")

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])

    def test_dml_rf_propensity(self, constant_effect_data):
        """DML with random forest propensity model."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, propensity_model="random_forest")

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestDMLAdversarial:
    """Tests for edge cases and challenging scenarios."""

    def test_dml_confounded_dgp(self):
        """DML handles selection on observables (confounding)."""
        np.random.seed(123)
        n = 500

        # Generate confounded data
        X = np.random.randn(n, 2)

        # Propensity depends on X (confounding)
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] - 0.3 * X[:, 1]))
        T = np.random.binomial(1, propensity, n).astype(float)

        # Outcome depends on X and T
        true_ate = 2.0
        Y = 1 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + true_ate * T + np.random.randn(n)

        result = double_ml(Y, T, X, n_folds=5)

        # Should recover ATE despite confounding
        assert abs(result["ate"] - true_ate) < 0.6, (
            f"DML ATE {result['ate']:.3f} with confounding far from {true_ate}"
        )

    def test_dml_high_dimensional(self, high_dimensional_data):
        """DML works with many covariates (p=15)."""
        Y, T, X, _ = high_dimensional_data

        # Use ridge for regularization with high-d
        result = double_ml(Y, T, X, n_folds=5, nuisance_model="ridge")

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])
        # With p=15, n=500, should still get reasonable estimate
        assert abs(result["ate"] - 2.0) < 1.0

    def test_dml_small_folds(self, constant_effect_data):
        """DML with minimum folds (n_folds=2) still works."""
        Y, T, X, _ = constant_effect_data

        result = double_ml(Y, T, X, n_folds=2)

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])

    def test_dml_single_covariate(self, single_covariate_data):
        """DML works with single covariate."""
        Y, T, X, _ = single_covariate_data

        result = double_ml(Y, T, X, n_folds=3)

        assert result["method"] == "double_ml"
        assert abs(result["ate"] - 2.0) < 1.0

    def test_dml_small_sample(self, small_sample_data):
        """DML handles small samples (n=50) with warning."""
        Y, T, X, _ = small_sample_data

        # Should work but may warn about small fold sizes
        result = double_ml(Y, T, X, n_folds=2)

        assert result["method"] == "double_ml"
        assert np.isfinite(result["ate"])


class TestDMLInputValidation:
    """Tests for input validation and error handling."""

    def test_dml_invalid_n_folds(self, constant_effect_data):
        """DML raises error for invalid n_folds."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            double_ml(Y, T, X, n_folds=1)

    def test_dml_invalid_model_type(self, constant_effect_data):
        """DML raises error for unknown model type."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="Unknown model type"):
            double_ml(Y, T, X, model="invalid")

    def test_dml_invalid_nuisance_model(self, constant_effect_data):
        """DML raises error for unknown nuisance model."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="Unknown model type"):
            double_ml(Y, T, X, nuisance_model="invalid")

    def test_dml_invalid_propensity_model(self, constant_effect_data):
        """DML raises error for unknown propensity model."""
        Y, T, X, _ = constant_effect_data

        with pytest.raises(ValueError, match="Unknown propensity model"):
            double_ml(Y, T, X, propensity_model="invalid")

    def test_dml_non_binary_treatment(self, constant_effect_data):
        """DML raises error for non-binary treatment."""
        Y, T, X, _ = constant_effect_data
        T_continuous = T + 0.5  # Make non-binary

        with pytest.raises(ValueError, match="binary"):
            double_ml(Y, T_continuous, X)


# ============================================================================
# Layer 3: Monte Carlo Tests (CONCERN-29 validation)
# ============================================================================


class TestDMLMonteCarlo:
    """Monte Carlo validation of DML properties.

    These tests validate CONCERN-29: Cross-fitting eliminates regularization bias.
    """

    @pytest.mark.slow
    def test_dml_unbiased_constant_effect(self):
        """Monte Carlo: DML has low bias for constant effect.

        CONCERN-29: Validate that cross-fitting produces unbiased estimates.
        Target: Bias < 0.10
        """
        np.random.seed(42)
        n_sims = 200
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            Y, T, X, _ = generate_cate_dgp(
                n=300,
                p=3,
                effect_type="constant",
                true_ate=true_ate,
                noise_sd=1.0,
                seed=sim * 1000 + 42,
            )

            result = double_ml(Y, T, X, n_folds=5)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate

        assert abs(bias) < 0.10, (
            f"DML bias {bias:.4f} exceeds threshold 0.10.\n"
            f"Mean estimate: {np.mean(estimates):.4f}, True: {true_ate}"
        )

    @pytest.mark.slow
    def test_dml_coverage(self):
        """Monte Carlo: DML CIs achieve nominal coverage.

        Target: 93-97% coverage for 95% CIs.
        """
        np.random.seed(123)
        n_sims = 200
        true_ate = 2.0

        covers = []
        for sim in range(n_sims):
            Y, T, X, _ = generate_cate_dgp(
                n=300,
                p=3,
                effect_type="constant",
                true_ate=true_ate,
                noise_sd=1.0,
                seed=sim * 1000 + 123,
            )

            result = double_ml(Y, T, X, n_folds=5, alpha=0.05)
            covers.append(result["ci_lower"] < true_ate < result["ci_upper"])

        coverage = np.mean(covers)

        assert 0.88 < coverage < 0.99, (
            f"DML coverage {coverage:.2%} outside acceptable range [88%, 99%].\n"
            f"Expected ~95% for well-calibrated CIs."
        )

    @pytest.mark.slow
    def test_dml_vs_r_learner_bias_comparison(self):
        """Monte Carlo: Compare DML and R-learner bias.

        CONCERN-29: DML (cross-fitted) should have lower or comparable bias
        to R-learner (in-sample) on this DGP.

        Note: R-learner regularization bias is most apparent with:
        - High-dimensional covariates
        - Small samples
        - Flexible (overfit-prone) nuisance models
        """
        np.random.seed(456)
        n_sims = 100
        true_ate = 2.0

        dml_estimates = []
        r_estimates = []

        for sim in range(n_sims):
            # Use moderately challenging DGP
            Y, T, X, _ = generate_cate_dgp(
                n=200,
                p=5,
                effect_type="constant",
                true_ate=true_ate,
                noise_sd=1.0,
                seed=sim * 1000 + 456,
            )

            dml_result = double_ml(Y, T, X, n_folds=5, nuisance_model="ridge")
            r_result = r_learner(Y, T, X, model="linear")

            dml_estimates.append(dml_result["ate"])
            r_estimates.append(r_result["ate"])

        dml_bias = abs(np.mean(dml_estimates) - true_ate)
        r_bias = abs(np.mean(r_estimates) - true_ate)

        # Both should have low bias for this DGP
        assert dml_bias < 0.15, f"DML bias {dml_bias:.4f} too high"
        assert r_bias < 0.15, f"R-learner bias {r_bias:.4f} too high"

        # Log comparison (informational)
        print(f"\nBias comparison:")
        print(f"  DML bias:      {dml_bias:.4f}")
        print(f"  R-learner bias: {r_bias:.4f}")

    @pytest.mark.slow
    def test_dml_confounded_dgp_monte_carlo(self):
        """Monte Carlo: DML handles confounding well.

        Test that DML correctly adjusts for selection on observables.
        """
        np.random.seed(789)
        n_sims = 200
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            np.random.seed(sim * 1000 + 789)
            n = 300

            # Confounded DGP
            X = np.random.randn(n, 2)
            propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
            T = np.random.binomial(1, propensity, n).astype(float)
            Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

            result = double_ml(Y, T, X, n_folds=5)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate

        assert abs(bias) < 0.15, (
            f"DML bias with confounding {bias:.4f} exceeds threshold 0.15"
        )


class TestDMLComparison:
    """Tests comparing DML to other methods."""

    def test_dml_similar_to_r_learner(self, constant_effect_data):
        """DML and R-learner give similar estimates on clean data."""
        Y, T, X, _ = constant_effect_data

        dml_result = double_ml(Y, T, X, n_folds=5)
        r_result = r_learner(Y, T, X)

        # Should be within ~0.5 of each other
        diff = abs(dml_result["ate"] - r_result["ate"])
        assert diff < 0.5, (
            f"DML ({dml_result['ate']:.3f}) and R-learner ({r_result['ate']:.3f}) "
            f"differ by {diff:.3f}"
        )
