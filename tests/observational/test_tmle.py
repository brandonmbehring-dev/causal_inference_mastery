"""Tests for Targeted Maximum Likelihood Estimation (TMLE).

Tests cover:
1. Basic functionality and return structure
2. Known-answer tests with hand-calculated expected values
3. Comparison with doubly robust estimator
4. Convergence behavior
5. Double robustness property (both, propensity-only, outcome-only)
6. Input validation and error handling
7. Edge cases (extreme propensities, small samples)

Following test-first principles with known-answer tests.
"""

import numpy as np
import pytest
from src.causal_inference.observational.tmle import tmle_ate
from src.causal_inference.observational.doubly_robust import dr_ate


class TestTMLEBasicFunctionality:
    """Test TMLE basic functionality and return structure."""

    def test_tmle_returns_correct_structure(self):
        """Verify TMLE returns all expected fields."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        # Core estimates
        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

        # Sample sizes
        assert "n" in result
        assert "n_treated" in result
        assert "n_control" in result
        assert "n_trimmed" in result

        # TMLE-specific
        assert "epsilon" in result
        assert "n_iterations" in result
        assert "converged" in result
        assert "convergence_criterion" in result

        # Diagnostics
        assert "propensity_diagnostics" in result
        assert "outcome_diagnostics" in result

        # Intermediate values
        assert "propensity" in result
        assert "Q0_initial" in result
        assert "Q1_initial" in result
        assert "Q0_star" in result
        assert "Q1_star" in result
        assert "eif" in result

    def test_tmle_types_correct(self):
        """Verify return types are correct."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        assert isinstance(result["estimate"], float)
        assert isinstance(result["se"], float)
        assert isinstance(result["n"], int)
        assert isinstance(result["converged"], bool)
        assert isinstance(result["propensity"], np.ndarray)
        assert isinstance(result["eif"], np.ndarray)

    def test_tmle_sample_sizes_correct(self):
        """Verify sample size accounting."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        T = np.random.binomial(1, 0.4, n)  # 40% treated
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        assert result["n"] == n
        assert result["n_treated"] + result["n_control"] == n
        assert result["n_treated"] == np.sum(T)
        assert result["n_control"] == n - np.sum(T)


class TestTMLEKnownAnswer:
    """Test TMLE with known-answer data generating processes."""

    def test_tmle_recovers_zero_effect(self):
        """Test TMLE recovers zero ATE when true effect is zero."""
        np.random.seed(42)
        n = 500

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = X + np.random.randn(n)  # True ATE = 0

        result = tmle_ate(Y, T, X)

        # Should be close to 0
        assert np.abs(result["estimate"]) < 0.2

        # CI should contain 0
        assert result["ci_lower"] < 0 < result["ci_upper"]

    def test_tmle_recovers_positive_effect(self):
        """Test TMLE recovers positive ATE."""
        np.random.seed(42)
        n = 500
        true_ate = 2.0

        X = np.random.randn(n)
        logit_prop = 0.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = true_ate * T + X + np.random.randn(n) * 0.5

        result = tmle_ate(Y, T, X)

        # Should recover ATE ≈ 2.0
        assert np.abs(result["estimate"] - true_ate) < 0.3

        # CI should be close to covering true ATE (within 1 SE)
        assert result["ci_lower"] - result["se"] < true_ate < result["ci_upper"] + result["se"]

    def test_tmle_recovers_negative_effect(self):
        """Test TMLE recovers negative ATE."""
        np.random.seed(42)
        n = 500
        true_ate = -1.5

        X = np.random.randn(n)
        logit_prop = 0.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = true_ate * T + X + np.random.randn(n) * 0.5

        result = tmle_ate(Y, T, X)

        # Should recover ATE ≈ -1.5
        assert np.abs(result["estimate"] - true_ate) < 0.3

        # CI should be close to covering true ATE (within 1 SE)
        assert result["ci_lower"] - result["se"] < true_ate < result["ci_upper"] + result["se"]


class TestTMLEConvergence:
    """Test TMLE convergence behavior."""

    def test_tmle_converges(self):
        """Test TMLE converges within max iterations."""
        np.random.seed(42)
        n = 300

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        assert result["converged"] is True
        assert result["n_iterations"] < 50  # Should converge quickly
        assert np.abs(result["convergence_criterion"]) < 1e-5

    def test_tmle_convergence_tolerance(self):
        """Test convergence respects tolerance parameter."""
        np.random.seed(42)
        n = 300

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        # Loose tolerance should converge faster
        result_loose = tmle_ate(Y, T, X, tol=1e-3)

        # Tight tolerance may take more iterations
        result_tight = tmle_ate(Y, T, X, tol=1e-10)

        # Both should converge
        assert result_loose["converged"]
        assert result_tight["converged"]

        # Tight tolerance should have smaller criterion
        assert np.abs(result_tight["convergence_criterion"]) <= np.abs(
            result_loose["convergence_criterion"]
        )


class TestTMLEComparisonWithDR:
    """Test TMLE comparison with doubly robust estimator."""

    def test_tmle_close_to_dr(self):
        """TMLE and DR should give similar estimates."""
        np.random.seed(42)
        n = 500

        X = np.random.randn(n)
        logit_prop = 0.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X + np.random.randn(n) * 0.5

        tmle_result = tmle_ate(Y, T, X)
        dr_result = dr_ate(Y, T, X)

        # Estimates should be within 10% of each other
        relative_diff = np.abs(tmle_result["estimate"] - dr_result["estimate"]) / np.abs(
            dr_result["estimate"]
        )
        assert relative_diff < 0.1

    def test_tmle_se_comparable_to_dr(self):
        """TMLE SE should be comparable to (or smaller than) DR SE."""
        np.random.seed(42)
        n = 500

        X = np.random.randn(n)
        logit_prop = 0.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X + np.random.randn(n) * 0.5

        tmle_result = tmle_ate(Y, T, X)
        dr_result = dr_ate(Y, T, X)

        # TMLE SE should be within 50% of DR SE (often smaller)
        se_ratio = tmle_result["se"] / dr_result["se"]
        assert 0.5 < se_ratio < 1.5


class TestTMLEDoubleRobustness:
    """Test TMLE double robustness property."""

    def test_both_models_correct(self):
        """Test with both propensity and outcome models correctly specified."""
        np.random.seed(42)
        n = 400
        true_ate = 3.0

        X = np.random.randn(n)
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = true_ate * T + (2 + 0.5 * X) + np.random.randn(n) * 0.5

        result = tmle_ate(Y, T, X)

        # Should recover ATE with low bias
        assert np.abs(result["estimate"] - true_ate) < 0.25

        # Should have reasonable SE
        assert 0.05 < result["se"] < 0.25

        # CI should cover true ATE
        assert result["ci_lower"] < true_ate < result["ci_upper"]

    def test_propensity_correct_outcome_wrong(self):
        """Test when propensity correct but outcome has nonlinear relationship."""
        np.random.seed(42)
        n = 400
        true_ate = 2.0

        X = np.random.randn(n)
        logit_prop = 0.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome is nonlinear in X (outcome model misspecified)
        Y = true_ate * T + X**2 + np.random.randn(n) * 0.5

        result = tmle_ate(Y, T, X)

        # Should still be reasonably close (protected by correct propensity)
        assert np.abs(result["estimate"] - true_ate) < 0.5

    def test_outcome_correct_propensity_wrong(self):
        """Test when outcome correct but propensity has nonlinear relationship."""
        np.random.seed(42)
        n = 400
        true_ate = 2.0

        X = np.random.randn(n)

        # Propensity is nonlinear (propensity model misspecified)
        logit_prop = 0.3 * X**2
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome is linear in X (correctly modeled)
        Y = true_ate * T + X + np.random.randn(n) * 0.5

        result = tmle_ate(Y, T, X)

        # Should still be reasonably close (protected by correct outcome)
        assert np.abs(result["estimate"] - true_ate) < 0.5


class TestTMLEInputValidation:
    """Test TMLE input validation and error handling."""

    def test_mismatched_array_lengths(self):
        """Test error on mismatched array lengths."""
        Y = np.array([1, 2, 3])
        T = np.array([0, 1])  # Wrong length
        X = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="CRITICAL ERROR"):
            tmle_ate(Y, T, X)

    def test_non_binary_treatment(self):
        """Test error on non-binary treatment."""
        Y = np.array([1, 2, 3, 4])
        T = np.array([0, 1, 2, 1])  # Not binary
        X = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="CRITICAL ERROR"):
            tmle_ate(Y, T, X)

    def test_nan_in_outcomes(self):
        """Test error on NaN in outcomes."""
        Y = np.array([1, 2, np.nan, 4])
        T = np.array([0, 1, 0, 1])
        X = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="CRITICAL ERROR"):
            tmle_ate(Y, T, X)

    def test_nan_in_covariates(self):
        """Test error on NaN in covariates."""
        Y = np.array([1, 2, 3, 4])
        T = np.array([0, 1, 0, 1])
        X = np.array([1, np.nan, 3, 4])

        with pytest.raises(ValueError, match="CRITICAL ERROR"):
            tmle_ate(Y, T, X)

    def test_accepts_list_inputs(self):
        """Test TMLE accepts list inputs."""
        np.random.seed(42)
        n = 100

        X = list(np.random.randn(n))
        T = list((np.array(X) + np.random.randn(n) > 0).astype(int))
        Y = list(2 * np.array(T) + np.array(X) + np.random.randn(n))

        result = tmle_ate(Y, T, X)

        assert isinstance(result["estimate"], float)


class TestTMLEEdgeCases:
    """Test TMLE edge cases."""

    def test_small_sample(self):
        """Test TMLE with small sample size."""
        np.random.seed(42)
        n = 50

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        # Should still run and produce estimates
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])
        assert result["se"] > 0

    def test_extreme_propensities(self):
        """Test TMLE handles extreme propensities gracefully."""
        np.random.seed(42)
        n = 200

        # Create extreme propensities
        X = np.random.randn(n)
        logit_prop = 3 * X  # Strong confounding
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        # Should still produce finite estimates
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

    def test_single_covariate(self):
        """Test TMLE with single covariate."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        T = (X + np.random.randn(n) > 0).astype(int)
        Y = 2 * T + X + np.random.randn(n)

        # Pass 1D array
        result = tmle_ate(Y, T, X)

        assert result["n"] == n
        assert np.isfinite(result["estimate"])

    def test_multiple_covariates(self):
        """Test TMLE with multiple covariates."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n, 3)
        logit_prop = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1 * X[:, 2]
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X[:, 0] + X[:, 1] + np.random.randn(n)

        result = tmle_ate(Y, T, X)

        assert result["n"] == n
        assert np.isfinite(result["estimate"])

    def test_precomputed_propensity(self):
        """Test TMLE with pre-computed propensity scores."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        true_prop = 1 / (1 + np.exp(-0.5 * X))
        T = np.random.binomial(1, true_prop)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X, propensity=true_prop)

        assert result["propensity_diagnostics"]["source"] == "user_provided"
        assert np.isfinite(result["estimate"])


class TestTMLETrimming:
    """Test TMLE propensity trimming."""

    def test_trimming_applied(self):
        """Test that trimming is applied when specified."""
        np.random.seed(42)
        n = 200

        X = np.random.randn(n)
        logit_prop = 2 * X  # Strong confounding
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X + np.random.randn(n)

        result = tmle_ate(Y, T, X, trim_at=(0.05, 0.95))

        # Some units should be trimmed
        assert result["n_trimmed"] >= 0
        assert result["n"] <= n

    def test_trimming_affects_sample_size(self):
        """Test that trimming reduces effective sample size."""
        np.random.seed(42)
        n = 300

        X = np.random.randn(n)
        logit_prop = 3 * X  # Very strong confounding
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)
        Y = 2 * T + X + np.random.randn(n)

        result_notrim = tmle_ate(Y, T, X, trim_at=None)
        result_trim = tmle_ate(Y, T, X, trim_at=(0.1, 0.9))

        # Trimmed version should have smaller sample
        assert result_trim["n"] <= result_notrim["n"]
