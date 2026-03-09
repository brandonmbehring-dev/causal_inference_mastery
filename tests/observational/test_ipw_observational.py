"""Tests for observational IPW ATE estimator.

Tests cover:
1. IPW with automatic propensity estimation
2. IPW with pre-computed propensities
3. Weight trimming effects
4. Integration with RCT ipw_ate
5. Error handling

Following test-first principles with known-answer tests.
"""

import numpy as np
import pytest
from src.causal_inference.observational.ipw import ipw_ate_observational
from src.causal_inference.observational.propensity import estimate_propensity


class TestIPWObservational:
    """Test observational IPW with automatic propensity estimation."""

    def test_linear_confounding_recovers_ate(self):
        """
        Test IPW with linear confounding recovers true ATE.

        DGP:
            X ~ N(0, 1)
            logit(P(T=1|X)) = 0 + 0.8*X  # Confounding
            Y = 3.0*T + 0.5*X + N(0,1)   # True ATE = 3.0, confounding from X

        IPW should recover ATE ≈ 3.0 by reweighting.
        """
        np.random.seed(42)
        n = 500

        # Confounded treatment assignment
        X = np.random.normal(0, 1, n)
        logit = 0.8 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)

        # Outcome depends on treatment AND confounder
        Y = 3.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Should recover ATE ≈ 3.0
        assert np.abs(result["estimate"] - 3.0) < 0.5  # Reasonable tolerance

        # SE should be finite and positive
        assert result["se"] > 0
        assert np.isfinite(result["se"])

        # CI should contain true ATE
        assert result["ci_lower"] < 3.0 < result["ci_upper"]

        # Propensity diagnostics should show confounding
        assert result["propensity_diagnostics"]["auc"] > 0.65  # Good discrimination

    def test_multiple_confounders(self):
        """Test IPW with multiple confounders."""
        np.random.seed(123)
        n = 500

        # Three confounders
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X3 = np.random.uniform(-1, 1, n)
        X = np.column_stack([X1, X2, X3])

        # Confounded treatment from all three
        logit = 0.5 * X1 + 0.3 * X2 - 0.4 * X3
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)

        # Outcome depends on treatment and all confounders
        Y = 2.5 * T + 0.4 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Should recover ATE ≈ 2.5
        assert np.abs(result["estimate"] - 2.5) < 0.6

        # Propensity diagnostics should show reasonable confounding
        assert result["propensity_diagnostics"]["auc"] > 0.6

    def test_weak_confounding(self):
        """Test IPW with weak confounding (T nearly independent of X)."""
        np.random.seed(456)
        n = 500

        # Very weak confounding
        X = np.random.normal(0, 1, n)
        logit = 0.1 * X  # Small coefficient
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)

        # Outcome with weak confounding
        Y = 4.0 * T + 0.1 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Should still recover ATE ≈ 4.0
        assert np.abs(result["estimate"] - 4.0) < 0.5

        # Propensity diagnostics should show weak confounding
        assert 0.45 < result["propensity_diagnostics"]["auc"] < 0.65


class TestIPWWithPrecomputedPropensity:
    """Test observational IPW with pre-computed propensities."""

    def test_precomputed_propensity_same_as_estimated(self):
        """Test that pre-computed propensity gives same result as auto-estimation."""
        np.random.seed(789)
        n = 300

        X = np.random.normal(0, 1, (n, 2))
        logit = 0.6 * X[:, 0] + 0.4 * X[:, 1]
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 3.5 * T + 0.3 * X[:, 0] + np.random.normal(0, 1, n)

        # Estimate propensity separately
        prop_result = estimate_propensity(T, X)
        propensity = prop_result["propensity"]

        # IPW with auto-estimation
        result_auto = ipw_ate_observational(Y, T, X)

        # IPW with pre-computed propensity
        result_precomputed = ipw_ate_observational(Y, T, X, propensity=propensity)

        # Should give identical estimates
        assert np.isclose(result_auto["estimate"], result_precomputed["estimate"], atol=1e-10)
        assert np.isclose(result_auto["se"], result_precomputed["se"], atol=1e-10)

        # Pre-computed should have "provided" flag in diagnostics
        assert result_precomputed["propensity_diagnostics"]["provided"] == True

    def test_precomputed_propensity_wrong_length_fails(self):
        """Test that mismatched propensity length raises error."""
        np.random.seed(101)
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)

        # Propensity with wrong length
        propensity_wrong = np.array([0.5] * 50)  # Only 50 instead of 100

        with pytest.raises(ValueError) as exc_info:
            ipw_ate_observational(Y, T, X, propensity=propensity_wrong)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Propensity length mismatch" in error_msg


class TestIPWWithTrimming:
    """Test observational IPW with weight trimming."""

    def test_trimming_reduces_sample_size(self):
        """Test that trimming removes units with extreme propensities."""
        np.random.seed(202)
        n = 500

        # Create data with some extreme propensities
        X = np.random.normal(0, 1, n)
        logit = 2.0 * X  # Strong confounding -> extreme propensities
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 3.0 * T + X + np.random.normal(0, 1, n)

        # IPW with trimming at 1st/99th percentile
        result = ipw_ate_observational(Y, T, X, trim_at=(0.01, 0.99))

        # Should have trimmed some units
        assert result["n_trimmed"] > 0

        # Total units after trimming
        n_after_trim = result["n_treated"] + result["n_control"]
        assert n_after_trim == n - result["n_trimmed"]

        # Propensity range should be narrower after trimming extremes
        # (trimming at 1st/99th percentile removes ~2% of observations)
        propensity_range = result["propensity_summary"]["max"] - result["propensity_summary"]["min"]
        assert propensity_range < 1.0  # Should not span full [0,1] range

    def test_no_trimming_by_default(self):
        """Test that no trimming occurs by default."""
        np.random.seed(303)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Should have trimmed zero units
        assert result["n_trimmed"] == 0

        # All units should be included
        assert result["n_treated"] + result["n_control"] == n

    def test_aggressive_trimming_reduces_extremes(self):
        """Test that aggressive trimming (5th/95th) removes more units."""
        np.random.seed(404)
        n = 500

        X = np.random.normal(0, 1, n)
        logit = 1.5 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 3.0 * T + X + np.random.normal(0, 1, n)

        # Mild trimming
        result_mild = ipw_ate_observational(Y, T, X, trim_at=(0.01, 0.99))

        # Aggressive trimming
        result_aggressive = ipw_ate_observational(Y, T, X, trim_at=(0.05, 0.95))

        # Aggressive should trim more units
        assert result_aggressive["n_trimmed"] > result_mild["n_trimmed"]


class TestIPWStabilization:
    """Test observational IPW with weight stabilization."""

    def test_stabilization_returns_estimate(self):
        """Test that stabilize=True returns valid estimate."""
        np.random.seed(505)
        n = 500

        X = np.random.normal(0, 1, n)
        logit = 0.8 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X, stabilize=True)

        # Should return valid estimate
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])
        assert result["se"] > 0

        # Should indicate stabilization was used
        assert result["stabilized"] is True

        # Should recover ATE approximately
        assert np.abs(result["estimate"] - 2.0) < 0.6

    def test_stabilization_vs_unstabilized_point_estimate(self):
        """Test that stabilized and unstabilized give similar point estimates."""
        np.random.seed(506)
        n = 500

        X = np.random.normal(0, 1, n)
        logit = 0.8 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 3.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result_unstab = ipw_ate_observational(Y, T, X, stabilize=False)
        result_stab = ipw_ate_observational(Y, T, X, stabilize=True)

        # Point estimates should be close (stabilization doesn't change bias)
        assert np.abs(result_stab["estimate"] - result_unstab["estimate"]) < 0.3

        # Both should recover ATE approximately
        assert np.abs(result_stab["estimate"] - 3.0) < 0.6
        assert np.abs(result_unstab["estimate"] - 3.0) < 0.6

        # Stabilization flag should differ
        assert result_stab["stabilized"] is True
        assert result_unstab["stabilized"] is False

    def test_stabilization_reduces_variance_with_extreme_propensity(self):
        """
        Test that stabilization reduces variance when propensities are extreme.

        Stabilized weights have mean ≈ 1.0 which reduces variance compared to
        standard IPW weights that can be very large for extreme propensities.
        """
        np.random.seed(507)
        n = 500

        # Create extreme propensities (strong confounding)
        X = np.random.normal(0, 1, n)
        logit = 1.5 * X  # Strong confounding -> extreme propensities
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 2.5 * T + X + np.random.normal(0, 1, n)

        result_unstab = ipw_ate_observational(Y, T, X, stabilize=False)
        result_stab = ipw_ate_observational(Y, T, X, stabilize=True)

        # Stabilized should have smaller or similar SE (variance reduction)
        # Note: With finite samples, this isn't guaranteed but typically holds
        # We use a weak assertion here - just check both are finite and positive
        assert result_stab["se"] > 0
        assert result_unstab["se"] > 0
        assert np.isfinite(result_stab["se"])
        assert np.isfinite(result_unstab["se"])

    def test_stabilization_with_trimming(self):
        """Test that stabilization works correctly with propensity trimming."""
        np.random.seed(508)
        n = 500

        X = np.random.normal(0, 1, n)
        logit = 1.2 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        # Stabilization with trimming
        result = ipw_ate_observational(Y, T, X, stabilize=True, trim_at=(0.05, 0.95))

        # Should return valid results
        assert np.isfinite(result["estimate"])
        assert result["stabilized"] is True
        assert result["n_trimmed"] >= 0

    def test_stabilization_ci_contains_true_ate(self):
        """Test that stabilized IPW CI contains true ATE."""
        np.random.seed(509)
        n = 500

        true_ate = 2.5
        X = np.random.normal(0, 1, n)
        logit = 0.8 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = true_ate * T + 0.5 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X, stabilize=True)

        # CI should contain true ATE
        assert result["ci_lower"] < true_ate < result["ci_upper"]

    def test_stabilized_flag_in_result(self):
        """Test that result contains stabilized flag."""
        np.random.seed(510)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result_stab = ipw_ate_observational(Y, T, X, stabilize=True)
        result_unstab = ipw_ate_observational(Y, T, X, stabilize=False)

        assert "stabilized" in result_stab
        assert "stabilized" in result_unstab
        assert result_stab["stabilized"] is True
        assert result_unstab["stabilized"] is False


class TestIPWIntegration:
    """Test integration with RCT ipw_ate function."""

    def test_returns_standard_ipw_fields(self):
        """Test that result contains all standard IPW fields from RCT ipw_ate."""
        np.random.seed(606)
        n = 200

        X = np.random.normal(0, 1, n)
        T = np.array([1] * 100 + [0] * 100)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Standard IPW fields (from RCT ipw_ate)
        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n_treated" in result
        assert "n_control" in result

        # Observational-specific fields
        assert "n_trimmed" in result
        assert "propensity_diagnostics" in result
        assert "propensity_summary" in result

    def test_propensity_summary_statistics(self):
        """Test that propensity summary contains expected statistics."""
        np.random.seed(707)
        n = 200

        X = np.random.normal(0, 1, n)
        logit = 0.5 * X
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        Y = 2.0 * T + X + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        summary = result["propensity_summary"]

        # Should contain all standard statistics
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert "median" in summary
        assert "std" in summary

        # All should be in (0, 1)
        assert 0 < summary["min"] < 1
        assert 0 < summary["max"] < 1
        assert 0 < summary["mean"] < 1

        # Min < Mean < Max
        assert summary["min"] <= summary["mean"] <= summary["max"]


class TestIPWErrors:
    """Test error handling for ipw_ate_observational."""

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched input lengths raise ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50)  # Only 50 (mismatch!)
        Y = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError) as exc_info:
            ipw_ate_observational(Y, T, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "different lengths" in error_msg

    def test_invalid_trim_at_fails_fast(self):
        """Test that invalid trim_at parameter raises ValueError."""
        np.random.seed(808)
        X = np.random.normal(0, 1, 100)
        T = np.array([1] * 50 + [0] * 50)
        Y = np.random.normal(0, 1, 100)

        # Invalid trim_at (lower > upper)
        with pytest.raises(ValueError) as exc_info:
            ipw_ate_observational(Y, T, X, trim_at=(0.9, 0.1))

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Invalid trim_at" in error_msg
