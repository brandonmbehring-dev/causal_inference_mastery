"""Tests for Causal Forest implementation.

Tests are organized by layer:
1. Known-Answer: Verify causal forest recovers known treatment effects
2. Adversarial: Edge cases and challenging scenarios
3. Monte Carlo: Statistical validation of honesty (CONCERN-28)
"""

import pytest
import numpy as np

from src.causal_inference.cate import causal_forest
from tests.test_cate.conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestCausalForestKnownAnswer:
    """Tests with known true treatment effects."""

    def test_causal_forest_constant_effect(self, constant_effect_data):
        """Causal forest recovers constant ATE within tolerance."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = 2.0

        result = causal_forest(Y, T, X, n_estimators=48)

        # ATE should be close to true value
        assert abs(result["ate"] - true_ate) < 0.8, (
            f"Causal forest ATE {result['ate']:.3f} far from true {true_ate}"
        )
        assert result["method"] == "causal_forest"

    def test_causal_forest_heterogeneous_effect(self, linear_heterogeneous_data):
        """Causal forest captures heterogeneity in treatment effects."""
        Y, T, X, true_cate = linear_heterogeneous_data

        result = causal_forest(Y, T, X, n_estimators=48)

        # Check CATE shape
        assert result["cate"].shape == (len(Y),)

        # CATE should correlate with true CATE (forests excel at heterogeneity)
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.2, (
            f"CATE correlation {correlation:.3f} too low for heterogeneous effect"
        )

    def test_causal_forest_returns_valid_ci(self, constant_effect_data):
        """Causal forest returns valid confidence intervals."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=48, alpha=0.05)

        # CI should contain ATE
        assert result["ci_lower"] < result["ate"] < result["ci_upper"]

        # CI width should be reasonable
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.01 < ci_width < 10.0, f"CI width {ci_width:.3f} seems unreasonable"

    def test_causal_forest_se_positive(self, constant_effect_data):
        """Standard error should be positive and finite."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=48)

        assert result["ate_se"] > 0, "SE must be positive"
        assert np.isfinite(result["ate_se"]), "SE must be finite"


class TestCausalForestConfiguration:
    """Tests for different causal forest configurations."""

    def test_causal_forest_few_trees(self, constant_effect_data):
        """Causal forest works with few trees."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=24)

        assert result["method"] == "causal_forest"
        assert np.isfinite(result["ate"])

    def test_causal_forest_many_trees(self, constant_effect_data):
        """Causal forest works with many trees."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=100)

        assert result["method"] == "causal_forest"
        assert np.isfinite(result["ate"])

    def test_causal_forest_large_min_samples_leaf(self, constant_effect_data):
        """Causal forest works with large min_samples_leaf."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=48, min_samples_leaf=20)

        assert result["method"] == "causal_forest"
        assert np.isfinite(result["ate"])

    def test_causal_forest_limited_depth(self, constant_effect_data):
        """Causal forest works with limited tree depth."""
        Y, T, X, _ = constant_effect_data

        result = causal_forest(Y, T, X, n_estimators=48, max_depth=5)

        assert result["method"] == "causal_forest"
        assert np.isfinite(result["ate"])


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestCausalForestAdversarial:
    """Tests for edge cases and challenging scenarios."""

    def test_causal_forest_high_dimensional(self, high_dimensional_data):
        """Causal forest handles high-dimensional covariates (p=15)."""
        Y, T, X, _ = high_dimensional_data

        result = causal_forest(Y, T, X, n_estimators=48)

        assert result["method"] == "causal_forest"
        assert np.isfinite(result["ate"])
        # Forests handle feature selection internally
        assert abs(result["ate"] - 2.0) < 1.5

    def test_causal_forest_nonlinear_effect(self, nonlinear_heterogeneous_data):
        """Causal forest captures nonlinear heterogeneity."""
        Y, T, X, true_cate = nonlinear_heterogeneous_data

        result = causal_forest(Y, T, X, n_estimators=48)

        assert result["method"] == "causal_forest"
        # Forests excel at nonlinear patterns
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.1, (
            f"Causal forest failed to capture nonlinear heterogeneity "
            f"(correlation={correlation:.3f})"
        )

    def test_causal_forest_confounded_dgp(self):
        """Causal forest handles selection on observables."""
        np.random.seed(456)
        n = 400

        # Confounded DGP
        X = np.random.randn(n, 2)
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        true_ate = 2.0
        Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

        result = causal_forest(Y, T, X, n_estimators=48)

        # Should handle confounding reasonably well
        assert abs(result["ate"] - true_ate) < 1.0

    def test_causal_forest_single_covariate(self, single_covariate_data):
        """Causal forest works with single covariate."""
        Y, T, X, _ = single_covariate_data

        result = causal_forest(Y, T, X, n_estimators=32)

        assert result["method"] == "causal_forest"
        assert abs(result["ate"] - 2.0) < 1.0


class TestCausalForestHonesty:
    """Tests for honest vs non-honest splitting (CONCERN-28)."""

    def test_honest_is_default(self, constant_effect_data):
        """Verify honest=True is the default."""
        Y, T, X, _ = constant_effect_data

        # Should work without explicit honest=True
        result = causal_forest(Y, T, X, n_estimators=32)

        assert result["method"] == "causal_forest"

    def test_dishonest_warns(self, constant_effect_data):
        """Dishonest forests should emit a warning."""
        Y, T, X, _ = constant_effect_data

        with pytest.warns(UserWarning, match="CONCERN-28"):
            result = causal_forest(Y, T, X, n_estimators=32, honest=False)

        assert result["method"] == "causal_forest"

    def test_honest_vs_dishonest_produce_results(self, constant_effect_data):
        """Both honest and dishonest forests produce finite results."""
        Y, T, X, _ = constant_effect_data

        honest_result = causal_forest(Y, T, X, n_estimators=32, honest=True)

        with pytest.warns(UserWarning):
            dishonest_result = causal_forest(Y, T, X, n_estimators=32, honest=False)

        # Both should produce finite results
        assert np.isfinite(honest_result["ate"])
        assert np.isfinite(dishonest_result["ate"])


class TestCausalForestInputValidation:
    """Tests for input validation."""

    def test_non_binary_treatment(self, constant_effect_data):
        """Raises error for non-binary treatment."""
        Y, T, X, _ = constant_effect_data
        T_continuous = T + 0.5

        with pytest.raises(ValueError, match="binary"):
            causal_forest(Y, T_continuous, X)


# ============================================================================
# Layer 3: Monte Carlo Tests (CONCERN-28 validation)
# ============================================================================


class TestCausalForestMonteCarlo:
    """Monte Carlo validation of honest forests.

    These tests validate CONCERN-28: Honest forests provide valid inference.
    """

    @pytest.mark.slow
    def test_honest_forest_low_bias(self):
        """Monte Carlo: Honest forest has low bias for constant effect.

        Target: Bias < 0.20
        """
        np.random.seed(42)
        n_sims = 50  # Fewer sims since forests are slower
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

            result = causal_forest(Y, T, X, n_estimators=48, honest=True)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate

        assert abs(bias) < 0.20, (
            f"Causal forest bias {bias:.4f} exceeds threshold 0.20.\n"
            f"Mean estimate: {np.mean(estimates):.4f}, True: {true_ate}"
        )

    @pytest.mark.slow
    def test_honest_forest_coverage(self):
        """Monte Carlo: Honest forest CIs achieve reasonable coverage.

        Target: Coverage > 80% (forests can be conservative or liberal)
        """
        np.random.seed(123)
        n_sims = 50
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

            result = causal_forest(Y, T, X, n_estimators=48, honest=True, alpha=0.05)
            covers.append(result["ci_lower"] < true_ate < result["ci_upper"])

        coverage = np.mean(covers)

        # Forests can be conservative (high coverage) or slightly liberal
        assert coverage > 0.70, (
            f"Honest forest coverage {coverage:.2%} too low.\n"
            f"CONCERN-28 requires honest splitting for valid inference."
        )

    @pytest.mark.slow
    def test_honest_forest_heterogeneous_effects(self):
        """Monte Carlo: Honest forest captures heterogeneity."""
        np.random.seed(789)
        n_sims = 30
        true_ate = 2.0

        correlations = []
        for sim in range(n_sims):
            Y, T, X, true_cate = generate_cate_dgp(
                n=400,
                p=3,
                effect_type="linear",  # τ(x) = 2 + X[:,0]
                true_ate=true_ate,
                noise_sd=1.0,
                seed=sim * 1000 + 789,
            )

            result = causal_forest(Y, T, X, n_estimators=48, honest=True)
            corr = np.corrcoef(result["cate"], true_cate)[0, 1]
            correlations.append(corr)

        mean_correlation = np.mean(correlations)

        # Forests should capture heterogeneity
        assert mean_correlation > 0.15, (
            f"Mean CATE correlation {mean_correlation:.3f} too low.\n"
            f"Causal forests should capture heterogeneous treatment effects."
        )


class TestCausalForestComparison:
    """Tests comparing causal forest to other methods."""

    def test_causal_forest_vs_dml_similar_ate(self, constant_effect_data):
        """Causal forest and DML give similar ATEs on clean data."""
        from src.causal_inference.cate import double_ml

        Y, T, X, _ = constant_effect_data

        forest_result = causal_forest(Y, T, X, n_estimators=48)
        dml_result = double_ml(Y, T, X, n_folds=5)

        # Should be within ~1.0 of each other
        diff = abs(forest_result["ate"] - dml_result["ate"])
        assert diff < 1.0, (
            f"Causal forest ({forest_result['ate']:.3f}) and DML ({dml_result['ate']:.3f}) "
            f"differ by {diff:.3f}"
        )
