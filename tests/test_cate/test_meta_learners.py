"""Tests for meta-learners (S, T, X, R-Learner).

Test Layers:
- Layer 1 (Known-Answer): Tests with deterministic/expected results
- Layer 2 (Adversarial): Edge cases and error handling
- Layer 3 (Monte Carlo): Statistical validation (marked slow)
"""

import pytest
import numpy as np
from scipy import stats

from src.causal_inference.cate import s_learner, t_learner, x_learner, r_learner, CATEResult
from .conftest import generate_cate_dgp


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestSLearnerKnownAnswer:
    """Known-answer tests for S-Learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """S-learner recovers ATE within tolerance for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = s_learner(Y, T, X)

        # ATE should be close to true value (within 0.3 for n=500)
        assert abs(result["ate"] - true_ate) < 0.3, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )

    def test_constant_effect_cate_shape(self, constant_effect_data):
        """S-learner returns CATE array of correct shape."""
        Y, T, X, _ = constant_effect_data

        result = s_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)
        assert result["method"] == "s_learner"

    def test_linear_heterogeneous_effect_rf(self, linear_heterogeneous_data):
        """S-learner with RF captures heterogeneous effects.

        Note: S-learner with linear regression cannot capture heterogeneity
        because it lacks X*T interaction terms. Use random_forest for
        heterogeneous effect estimation.
        """
        Y, T, X, true_cate = linear_heterogeneous_data

        # Use random forest which can capture nonlinear patterns
        result = s_learner(Y, T, X, model="random_forest")

        # Correlation between estimated and true CATE should be positive
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.2, (
            f"CATE correlation {correlation:.3f} too low (expected > 0.2)"
        )

    def test_confidence_interval_valid(self, constant_effect_data):
        """CI should be valid (lower < upper, reasonable width)."""
        Y, T, X, _ = constant_effect_data

        result = s_learner(Y, T, X, alpha=0.05)

        # CI should be valid
        assert result["ci_lower"] < result["ci_upper"]
        # CI width should be reasonable (not too narrow or too wide)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.1 < ci_width < 1.0, f"CI width {ci_width:.3f} outside reasonable range"

    def test_se_positive(self, constant_effect_data):
        """Standard error should be positive."""
        Y, T, X, _ = constant_effect_data

        result = s_learner(Y, T, X)

        assert result["ate_se"] > 0


class TestTLearnerKnownAnswer:
    """Known-answer tests for T-Learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """T-learner recovers ATE within tolerance for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = t_learner(Y, T, X)

        # T-learner should recover ATE within tolerance
        assert abs(result["ate"] - true_ate) < 0.3, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )

    def test_constant_effect_cate_shape(self, constant_effect_data):
        """T-learner returns CATE array of correct shape."""
        Y, T, X, _ = constant_effect_data

        result = t_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)
        assert result["method"] == "t_learner"

    def test_linear_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """T-learner CATE correlates with true CATE for linear heterogeneity."""
        Y, T, X, true_cate = linear_heterogeneous_data

        result = t_learner(Y, T, X)

        # T-learner should capture heterogeneity better than S-learner
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.3, (
            f"CATE correlation {correlation:.3f} too low (expected > 0.3)"
        )

    def test_confidence_interval_valid(self, constant_effect_data):
        """CI should be valid (lower < upper, reasonable width)."""
        Y, T, X, _ = constant_effect_data

        result = t_learner(Y, T, X, alpha=0.05)

        # CI should be valid
        assert result["ci_lower"] < result["ci_upper"]
        # CI width should be reasonable
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.1 < ci_width < 1.0, f"CI width {ci_width:.3f} outside reasonable range"

    def test_se_positive(self, constant_effect_data):
        """Standard error should be positive."""
        Y, T, X, _ = constant_effect_data

        result = t_learner(Y, T, X)

        assert result["ate_se"] > 0


class TestMetaLearnerComparison:
    """Comparative tests between S-learner and T-learner."""

    def test_both_recover_ate_constant_effect(self, constant_effect_data):
        """Both learners recover ATE for constant treatment effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        # Both should be close to true ATE
        assert abs(s_result["ate"] - true_ate) < 0.4
        assert abs(t_result["ate"] - true_ate) < 0.4

    def test_ate_similar_for_constant_effect(self, constant_effect_data):
        """S-learner and T-learner produce similar ATE for constant effect."""
        Y, T, X, _ = constant_effect_data

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        # ATEs should be similar (within 0.5)
        assert abs(s_result["ate"] - t_result["ate"]) < 0.5


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_empty_treatment_group_raises(self):
        """Raise ValueError when all units are control."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)  # All control
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="No treated units"):
            s_learner(Y, T, X)

        with pytest.raises(ValueError, match="No treated units"):
            t_learner(Y, T, X)

    def test_empty_control_group_raises(self):
        """Raise ValueError when all units are treated."""
        n = 100
        Y = np.random.randn(n)
        T = np.ones(n)  # All treated
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="No control units"):
            s_learner(Y, T, X)

        with pytest.raises(ValueError, match="No control units"):
            t_learner(Y, T, X)

    def test_mismatched_lengths_raises(self):
        """Raise ValueError when input lengths don't match."""
        Y = np.random.randn(100)
        T = np.random.binomial(1, 0.5, 50)  # Wrong length
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="Length mismatch"):
            s_learner(Y, T, X)

    def test_non_binary_treatment_raises(self):
        """Raise ValueError for non-binary treatment."""
        n = 100
        Y = np.random.randn(n)
        T = np.random.randn(n)  # Continuous, not binary
        X = np.random.randn(n, 2)

        with pytest.raises(ValueError, match="binary"):
            s_learner(Y, T, X)

    def test_invalid_model_type_raises(self):
        """Raise ValueError for unknown model type."""
        Y, T, X, _ = generate_cate_dgp(n=100, seed=42)

        with pytest.raises(ValueError, match="Unknown model type"):
            s_learner(Y, T, X, model="unknown_model")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_covariate(self, single_covariate_data):
        """Works with single covariate (p=1)."""
        Y, T, X, true_cate = single_covariate_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        # Should still recover ATE
        assert abs(s_result["ate"] - true_ate) < 0.5
        assert abs(t_result["ate"] - true_ate) < 0.5

    def test_high_dimensional_covariates(self, high_dimensional_data):
        """Works with many covariates (p=15)."""
        Y, T, X, true_cate = high_dimensional_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        # Should still recover ATE (may be less precise)
        assert abs(s_result["ate"] - true_ate) < 0.6
        assert abs(t_result["ate"] - true_ate) < 0.6

    def test_small_sample(self, small_sample_data):
        """Works with small sample (n=50)."""
        Y, T, X, true_cate = small_sample_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        # Wider tolerance for small sample
        assert abs(s_result["ate"] - true_ate) < 1.0
        assert abs(t_result["ate"] - true_ate) < 1.0

    def test_1d_covariate_array(self):
        """Handles 1D covariate array (auto-reshapes)."""
        n = 200
        np.random.seed(42)
        X = np.random.randn(n)  # 1D array
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 1 + 0.5 * X + 2 * T + np.random.randn(n)

        # Should work without error (auto-reshape)
        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)

        assert s_result["cate"].shape == (n,)
        assert t_result["cate"].shape == (n,)


class TestModelVariants:
    """Tests for different base learner models."""

    def test_ridge_model(self, constant_effect_data):
        """Works with ridge regression."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X, model="ridge")
        t_result = t_learner(Y, T, X, model="ridge")

        assert abs(s_result["ate"] - true_ate) < 0.4
        assert abs(t_result["ate"] - true_ate) < 0.4

    def test_random_forest_model(self, constant_effect_data):
        """Works with random forest."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X, model="random_forest")
        t_result = t_learner(Y, T, X, model="random_forest")

        # Random forest may be less precise for linear DGP
        assert abs(s_result["ate"] - true_ate) < 0.6
        assert abs(t_result["ate"] - true_ate) < 0.6


class TestResultStructure:
    """Tests for CATEResult structure."""

    def test_result_keys(self, constant_effect_data):
        """Result contains all expected keys."""
        Y, T, X, _ = constant_effect_data

        result = s_learner(Y, T, X)

        expected_keys = {"cate", "ate", "ate_se", "ci_lower", "ci_upper", "method"}
        assert set(result.keys()) == expected_keys

    def test_result_types(self, constant_effect_data):
        """Result values have correct types."""
        Y, T, X, _ = constant_effect_data

        result = s_learner(Y, T, X)

        assert isinstance(result["cate"], np.ndarray)
        assert isinstance(result["ate"], float)
        assert isinstance(result["ate_se"], float)
        assert isinstance(result["ci_lower"], float)
        assert isinstance(result["ci_upper"], float)
        assert isinstance(result["method"], str)


# ============================================================================
# Layer 3: Monte Carlo Tests (marked slow)
# ============================================================================


class TestMonteCarlo:
    """Monte Carlo validation tests."""

    @pytest.mark.slow
    def test_s_learner_ate_unbiased(self):
        """S-learner ATE is approximately unbiased over simulations."""
        n_simulations = 500
        true_ate = 2.0
        ates = []

        for seed in range(n_simulations):
            Y, T, X, _ = generate_cate_dgp(
                n=300, effect_type="constant", true_ate=true_ate, seed=seed
            )
            result = s_learner(Y, T, X)
            ates.append(result["ate"])

        # Bias should be small
        mean_ate = np.mean(ates)
        bias = mean_ate - true_ate

        assert abs(bias) < 0.1, f"Bias {bias:.4f} exceeds threshold 0.1"

    @pytest.mark.slow
    def test_t_learner_ate_unbiased(self):
        """T-learner ATE is approximately unbiased over simulations."""
        n_simulations = 500
        true_ate = 2.0
        ates = []

        for seed in range(n_simulations):
            Y, T, X, _ = generate_cate_dgp(
                n=300, effect_type="constant", true_ate=true_ate, seed=seed
            )
            result = t_learner(Y, T, X)
            ates.append(result["ate"])

        mean_ate = np.mean(ates)
        bias = mean_ate - true_ate

        assert abs(bias) < 0.1, f"Bias {bias:.4f} exceeds threshold 0.1"

    @pytest.mark.slow
    def test_s_learner_coverage(self):
        """S-learner 95% CI has correct coverage."""
        n_simulations = 500
        true_ate = 2.0
        covered = 0

        for seed in range(n_simulations):
            Y, T, X, _ = generate_cate_dgp(
                n=300, effect_type="constant", true_ate=true_ate, seed=seed
            )
            result = s_learner(Y, T, X, alpha=0.05)

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                covered += 1

        coverage = covered / n_simulations

        # 95% CI should have ~95% coverage (allow 90-99% range)
        assert 0.88 < coverage < 0.99, f"Coverage {coverage:.2%} outside expected range"

    @pytest.mark.slow
    def test_t_learner_cate_recovery(self):
        """T-learner recovers heterogeneous CATE."""
        n_simulations = 200
        cate_correlations = []

        for seed in range(n_simulations):
            Y, T, X, true_cate = generate_cate_dgp(
                n=500, effect_type="linear", true_ate=2.0, seed=seed
            )
            result = t_learner(Y, T, X)

            corr = np.corrcoef(result["cate"], true_cate)[0, 1]
            cate_correlations.append(corr)

        mean_corr = np.mean(cate_correlations)

        # Should have positive correlation with true CATE
        assert mean_corr > 0.2, f"Mean CATE correlation {mean_corr:.3f} too low"


# ============================================================================
# X-Learner Tests
# ============================================================================


class TestXLearnerKnownAnswer:
    """Known-answer tests for X-Learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """X-learner recovers ATE within tolerance for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = x_learner(Y, T, X)

        # ATE should be close to true value
        assert abs(result["ate"] - true_ate) < 0.5, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )

    def test_cate_shape_and_method(self, constant_effect_data):
        """X-learner returns correct shape and method."""
        Y, T, X, _ = constant_effect_data

        result = x_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)
        assert result["method"] == "x_learner"

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """X-learner captures heterogeneous effects."""
        Y, T, X, true_cate = linear_heterogeneous_data

        result = x_learner(Y, T, X, model="random_forest")

        # X-learner should capture heterogeneity
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.2, (
            f"CATE correlation {correlation:.3f} too low (expected > 0.2)"
        )

    def test_imbalanced_treatment_groups(self):
        """X-learner handles imbalanced treatment well."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        # Imbalanced: only 20% treated
        T = np.random.binomial(1, 0.2, n).astype(float)
        true_ate = 2.0
        Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

        result = x_learner(Y, T, X)

        # Should still recover ATE reasonably
        assert abs(result["ate"] - true_ate) < 0.6, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )


class TestXLearnerAdversarial:
    """Adversarial tests for X-Learner."""

    def test_propensity_edge_cases(self):
        """X-learner handles varying propensity gracefully."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 2)
        # Propensity depends on X
        propensity = 1 / (1 + np.exp(-X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)

        # Should not crash
        result = x_learner(Y, T, X)
        assert np.isfinite(result["ate"])
        assert np.all(np.isfinite(result["cate"]))

    def test_with_linear_model(self, constant_effect_data):
        """X-learner works with linear base model."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = x_learner(Y, T, X, model="linear")

        # Should recover ATE
        assert abs(result["ate"] - true_ate) < 0.5


# ============================================================================
# R-Learner Tests
# ============================================================================


class TestRLearnerKnownAnswer:
    """Known-answer tests for R-Learner."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """R-learner recovers ATE within tolerance for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        result = r_learner(Y, T, X)

        # R-learner may have more variance but should be close
        assert abs(result["ate"] - true_ate) < 0.6, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )

    def test_cate_shape_and_method(self, constant_effect_data):
        """R-learner returns correct shape and method."""
        Y, T, X, _ = constant_effect_data

        result = r_learner(Y, T, X)

        assert result["cate"].shape == (len(Y),)
        assert result["method"] == "r_learner"

    def test_confounded_dgp(self):
        """R-learner handles confounding (selection on observables)."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        # Confounded treatment: propensity depends on X
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        true_ate = 2.0
        # Outcome also depends on X (confounder)
        Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

        result = r_learner(Y, T, X)

        # R-learner should handle confounding via Robinson transformation
        assert abs(result["ate"] - true_ate) < 0.8, (
            f"ATE {result['ate']:.3f} too far from true {true_ate:.3f}"
        )


class TestRLearnerAdversarial:
    """Adversarial tests for R-Learner."""

    def test_balanced_propensity(self):
        """R-learner works when propensity near 0.5 everywhere."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 2)
        T = np.random.binomial(1, 0.5, n).astype(float)  # Balanced
        Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)

        result = r_learner(Y, T, X)
        assert np.isfinite(result["ate"])
        assert np.all(np.isfinite(result["cate"]))

    def test_high_dimensional(self, high_dimensional_data):
        """R-learner works with high-dimensional covariates."""
        Y, T, X, true_cate = high_dimensional_data
        true_ate = np.mean(true_cate)

        result = r_learner(Y, T, X)

        # May be less precise but should be finite
        assert np.isfinite(result["ate"])
        assert abs(result["ate"] - true_ate) < 1.0


# ============================================================================
# Comparison Tests
# ============================================================================


class TestMetaLearnerComparison:
    """Comparative tests between all meta-learners."""

    def test_all_learners_recover_constant_ate(self, constant_effect_data):
        """All four learners recover ATE for constant effect."""
        Y, T, X, true_cate = constant_effect_data
        true_ate = np.mean(true_cate)

        s_result = s_learner(Y, T, X)
        t_result = t_learner(Y, T, X)
        x_result = x_learner(Y, T, X)
        r_result = r_learner(Y, T, X)

        # All should be reasonably close to true ATE
        for name, result in [("S", s_result), ("T", t_result),
                              ("X", x_result), ("R", r_result)]:
            assert abs(result["ate"] - true_ate) < 0.6, (
                f"{name}-learner ATE {result['ate']:.3f} too far from {true_ate:.3f}"
            )

    def test_all_learners_return_valid_ci(self, constant_effect_data):
        """All learners return valid confidence intervals."""
        Y, T, X, _ = constant_effect_data

        for learner, name in [(s_learner, "S"), (t_learner, "T"),
                               (x_learner, "X"), (r_learner, "R")]:
            result = learner(Y, T, X)
            assert result["ci_lower"] < result["ci_upper"], (
                f"{name}-learner CI invalid: [{result['ci_lower']}, {result['ci_upper']}]"
            )
            assert result["ate_se"] > 0, f"{name}-learner SE not positive"


# ============================================================================
# X/R-Learner Monte Carlo Tests
# ============================================================================


class TestXRLearnerMonteCarlo:
    """Monte Carlo tests for X and R learners."""

    @pytest.mark.slow
    def test_x_learner_imbalanced_performance(self):
        """X-learner performs well on imbalanced data."""
        n_simulations = 200
        true_ate = 2.0
        x_ates = []
        t_ates = []

        for seed in range(n_simulations):
            np.random.seed(seed)
            n = 400
            X = np.random.randn(n, 2)
            # Imbalanced: 25% treated
            T = np.random.binomial(1, 0.25, n).astype(float)
            Y = 1 + 0.5 * X[:, 0] + true_ate * T + np.random.randn(n)

            x_result = x_learner(Y, T, X)
            t_result = t_learner(Y, T, X)

            x_ates.append(x_result["ate"])
            t_ates.append(t_result["ate"])

        # X-learner should have lower MSE on imbalanced data
        x_mse = np.mean((np.array(x_ates) - true_ate) ** 2)
        t_mse = np.mean((np.array(t_ates) - true_ate) ** 2)

        # X-learner should not be significantly worse
        assert x_mse < t_mse * 2, (
            f"X-learner MSE {x_mse:.4f} much worse than T-learner {t_mse:.4f}"
        )

    @pytest.mark.slow
    def test_r_learner_ate_unbiased(self):
        """R-learner ATE is approximately unbiased."""
        n_simulations = 300
        true_ate = 2.0
        ates = []

        for seed in range(n_simulations):
            Y, T, X, _ = generate_cate_dgp(
                n=300, effect_type="constant", true_ate=true_ate, seed=seed
            )
            result = r_learner(Y, T, X)
            ates.append(result["ate"])

        mean_ate = np.mean(ates)
        bias = mean_ate - true_ate

        # R-learner should have low bias
        assert abs(bias) < 0.2, f"R-learner bias {bias:.4f} exceeds threshold 0.2"
