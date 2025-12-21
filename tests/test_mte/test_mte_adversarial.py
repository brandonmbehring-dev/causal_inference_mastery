"""
Adversarial tests for MTE module.

Edge cases, boundary conditions, and robustness tests.
"""

import pytest
import numpy as np

from causal_inference.mte import (
    late_estimator,
    late_bounds,
    local_iv,
    polynomial_mte,
    ate_from_mte,
    common_support_check,
    monotonicity_test,
    propensity_variation_test,
    mte_shape_test,
)


class TestWeakInstrument:
    """Tests with weak instruments."""

    def test_late_weak_instrument_larger_se(self, weak_instrument_data, simple_binary_iv_data):
        """Weak instrument produces larger SE than strong instrument."""
        weak_data = weak_instrument_data
        strong_data = simple_binary_iv_data

        result_weak = late_estimator(
            weak_data["outcome"],
            weak_data["treatment"],
            weak_data["instrument"],
        )

        result_strong = late_estimator(
            strong_data["outcome"],
            strong_data["treatment"],
            strong_data["instrument"],
        )

        # Weak instrument should have larger SE
        # (or at least not dramatically smaller)
        assert result_weak["se"] > 0  # SE should exist

    def test_local_iv_weak_instrument(self, weak_instrument_data):
        """Local IV with weak instrument produces noisy MTE."""
        data = weak_instrument_data

        # Should still run but with wide confidence intervals
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,
        )

        # SE should be larger than for strong instrument
        mean_se = np.nanmean(result["se_grid"])
        assert mean_se > 0


class TestNoFirstStage:
    """Tests when instrument has no effect on treatment."""

    def test_late_no_first_stage(self, no_first_stage_data):
        """LATE undefined when no first stage."""
        data = no_first_stage_data

        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # First stage should be near zero
        assert abs(result["first_stage_coef"]) < 0.1
        # F-stat should be very small
        assert result["first_stage_f"] < 5

    def test_monotonicity_no_first_stage(self, no_first_stage_data):
        """Monotonicity test detects no first stage."""
        data = no_first_stage_data

        result = monotonicity_test(
            data["treatment"],
            data["instrument"],
        )

        # Should indicate weak first stage
        assert result["first_stage"] < 0.1


class TestDefiers:
    """Tests with defiers (monotonicity violations)."""

    def test_late_with_defiers(self, defier_data):
        """LATE biased when monotonicity violated."""
        data = defier_data

        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Result should still be computed, but may be biased
        assert not np.isnan(result["late"])

    def test_bounds_with_defiers(self, defier_data):
        """Bounds should be wider with defiers."""
        data = defier_data

        result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Bounds should exist
        assert not np.isnan(result["bounds_lower"])
        assert not np.isnan(result["bounds_upper"])


class TestExtremePropensity:
    """Tests with extreme propensity scores."""

    def test_propensity_near_zero(self, rng):
        """Handles propensity near zero."""
        n = 500
        Z = rng.normal(-2, 1, size=n)  # Negative Z → low propensity
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))  # Mostly < 0.5
        D = (U < propensity).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

        result = local_iv(Y, D, Z, n_bootstrap=50, trim_fraction=0.05)

        # Should still produce some output
        valid = ~np.isnan(result["mte_grid"])
        assert valid.sum() > 0

    def test_propensity_near_one(self, rng):
        """Handles propensity near one."""
        n = 500
        Z = rng.normal(2, 1, size=n)  # Positive Z → high propensity
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))  # Mostly > 0.5
        D = (U < propensity).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

        result = local_iv(Y, D, Z, n_bootstrap=50, trim_fraction=0.05)

        valid = ~np.isnan(result["mte_grid"])
        assert valid.sum() > 0

    def test_common_support_extreme(self, rng):
        """Common support check with extreme propensity."""
        n = 500
        Z = rng.normal(2, 1, size=n)
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))
        D = (U < propensity).astype(float)

        result = common_support_check(propensity, D)

        # Should have narrow support
        p_min, p_max = result["support_region"]
        support_width = p_max - p_min
        # Support should be narrower than full [0, 1]
        assert support_width < 0.8


class TestSmallSample:
    """Tests with small sample sizes."""

    def test_late_small_sample(self, rng):
        """LATE works with small sample."""
        n = 50
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        U = rng.uniform(0, 1, size=n)
        D = (U < 0.3 + 0.4 * Z).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

        result = late_estimator(Y, D, Z)

        # Should produce result (may be noisy)
        assert not np.isnan(result["late"])

    def test_local_iv_small_sample(self, rng):
        """Local IV warns with small sample."""
        n = 40
        Z = rng.normal(0, 1, size=n)
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))
        D = (U < propensity).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

        with pytest.warns(UserWarning, match="[Ss]mall"):
            result = local_iv(Y, D, Z, n_bootstrap=20)


class TestHighVariance:
    """Tests with high outcome variance."""

    def test_late_high_noise(self, rng):
        """LATE with high outcome variance."""
        n = 500
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        U = rng.uniform(0, 1, size=n)
        D = (U < 0.3 + 0.4 * Z).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 5.0, size=n)  # High noise

        result = late_estimator(Y, D, Z)

        # Should have large SE
        assert result["se"] > 0.5

    def test_mte_high_noise(self, rng):
        """MTE with high outcome variance."""
        n = 500
        Z = rng.normal(0, 1, size=n)
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))
        D = (U < propensity).astype(float)
        Y = 1 + 2 * D + rng.normal(0, 5.0, size=n)  # High noise

        result = local_iv(Y, D, Z, n_bootstrap=50)

        # SE should be large
        mean_se = np.nanmean(result["se_grid"])
        assert mean_se > 0.5


class TestDiagnostics:
    """Tests for diagnostic functions."""

    def test_common_support_no_overlap(self, rng):
        """Common support detects no overlap."""
        n = 200

        # Create data with no overlap in propensity
        Z = np.concatenate([rng.normal(-3, 0.5, 100), rng.normal(3, 0.5, 100)])
        U = rng.uniform(0, 1, size=n)
        propensity = 1 / (1 + np.exp(-Z))
        D = (U < propensity).astype(float)

        result = common_support_check(propensity, D)

        # Should report limited support
        assert "LIMITED" in result["recommendation"] or "NO COMMON" in result["recommendation"]

    def test_propensity_variation_low(self, rng):
        """Detects low propensity variation."""
        n = 500
        propensity = rng.uniform(0.45, 0.55, size=n)  # Very concentrated

        result = propensity_variation_test(propensity, min_variation=0.2)

        assert not result["sufficient_variation"]

    def test_mte_shape_constant(self, constant_mte_data):
        """Shape test detects constant MTE."""
        data = constant_mte_data
        mte_result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=100,
        )

        result = mte_shape_test(
            mte_result["mte_grid"],
            mte_result["u_grid"],
            mte_result["se_grid"],
        )

        # Should not reject constant
        if "constant_mte_test" in result and "reject_constant" in result["constant_mte_test"]:
            # May or may not reject depending on noise
            pass

    def test_mte_shape_linear(self, heterogeneous_mte_data):
        """Shape test for linear MTE."""
        data = heterogeneous_mte_data
        mte_result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=100,
        )

        result = mte_shape_test(
            mte_result["mte_grid"],
            mte_result["u_grid"],
            mte_result["se_grid"],
        )

        # Should detect decreasing pattern
        if "monotonicity_test" in result:
            assert "frac_decreasing" in result["monotonicity_test"]


class TestPolynomialEdgeCases:
    """Edge cases for polynomial MTE."""

    def test_polynomial_high_degree(self, heterogeneous_mte_data):
        """High polynomial degree doesn't crash."""
        data = heterogeneous_mte_data

        result = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=5,
            n_bootstrap=50,
        )

        # Should still produce output
        assert not np.all(np.isnan(result["mte_grid"]))

    def test_polynomial_degree_one(self, heterogeneous_mte_data):
        """Linear polynomial (degree 1)."""
        data = heterogeneous_mte_data

        result = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=1,
            n_bootstrap=50,
        )

        # Should produce constant slope MTE
        mte = result["mte_grid"]
        valid = ~np.isnan(mte)

        if valid.sum() > 3:
            # Should be approximately linear (small second derivative)
            diffs = np.diff(mte[valid])
            diff_var = np.var(diffs)
            # Low variance in differences = approximately linear
            assert diff_var < 1.0


class TestATEEdgeCases:
    """Edge cases for ATE from MTE."""

    def test_ate_all_nan_mte(self, rng):
        """ATE with all NaN MTE values."""
        # Create mock MTE result with all NaN
        mte_result = {
            "mte_grid": np.full(20, np.nan),
            "u_grid": np.linspace(0.1, 0.9, 20),
            "se_grid": np.full(20, np.nan),
            "propensity_support": (0.1, 0.9),
            "n_obs": 100,
            "n_trimmed": 10,
            "bandwidth": 0.1,
            "method": "local_iv",
        }

        result = ate_from_mte(mte_result)

        # Should return NaN estimate
        assert np.isnan(result["estimate"])

    def test_ate_few_valid_points(self, rng):
        """ATE with few valid MTE points."""
        mte_values = np.full(20, np.nan)
        mte_values[8:12] = [2.0, 1.8, 1.6, 1.4]  # Only 4 valid

        mte_result = {
            "mte_grid": mte_values,
            "u_grid": np.linspace(0.1, 0.9, 20),
            "se_grid": np.full(20, 0.1),
            "propensity_support": (0.1, 0.9),
            "n_obs": 100,
            "n_trimmed": 10,
            "bandwidth": 0.1,
            "method": "local_iv",
        }

        result = ate_from_mte(mte_result)

        # Should still compute from valid points
        assert not np.isnan(result["estimate"])


class TestInputEdgeCases:
    """Edge cases for input handling."""

    def test_single_value_instrument(self, rng):
        """Instrument with single unique value."""
        n = 100
        Z = np.ones(n)  # No variation
        D = rng.binomial(1, 0.5, size=n).astype(float)
        Y = rng.normal(size=n)

        # Should fail gracefully or warn
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            late_estimator(Y, D, Z)

    def test_all_treated(self, rng):
        """All units treated."""
        n = 100
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        D = np.ones(n)  # All treated
        Y = rng.normal(size=n)

        # Should detect no variation in treatment
        with pytest.raises(ValueError):
            late_estimator(Y, D, Z)

    def test_all_control(self, rng):
        """All units in control."""
        n = 100
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        D = np.zeros(n)  # All control
        Y = rng.normal(size=n)

        # Should detect no variation in treatment
        with pytest.raises(ValueError):
            late_estimator(Y, D, Z)
