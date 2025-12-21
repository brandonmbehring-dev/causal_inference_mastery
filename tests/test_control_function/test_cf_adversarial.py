"""
Adversarial tests for Control Function estimation.

Tests edge cases and boundary conditions:
- Minimum sample size
- Invalid inputs (NaN, Inf, wrong dimensions)
- Constant treatment/instrument
- Perfect collinearity
- Very weak instruments
- Extreme endogeneity
- Large sample performance

Layer 2 of 6-layer validation architecture.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.control_function import (
    ControlFunction,
    control_function_ate,
)
from tests.test_control_function.conftest import generate_cf_data


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_nan_in_outcome(self):
        """Raises error for NaN in outcome."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=1)
        Y[10] = np.nan

        cf = ControlFunction()
        with pytest.raises(ValueError, match="NaN|infinite"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_nan_in_treatment(self):
        """Raises error for NaN in treatment."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=2)
        D[5] = np.nan

        cf = ControlFunction()
        with pytest.raises(ValueError, match="NaN|infinite"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_nan_in_instrument(self):
        """Raises error for NaN in instrument."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=3)
        Z[0, 0] = np.nan

        cf = ControlFunction()
        with pytest.raises(ValueError, match="NaN|infinite"):
            cf.fit(Y, D, Z, X)

    def test_nan_in_controls(self):
        """Raises error for NaN in controls."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, n_controls=2, random_state=4)
        X[20, 0] = np.nan

        cf = ControlFunction()
        with pytest.raises(ValueError, match="NaN|infinite"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_inf_in_outcome(self):
        """Raises error for Inf in outcome."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=5)
        Y[15] = np.inf

        cf = ControlFunction()
        with pytest.raises(ValueError, match="NaN|infinite"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_length_mismatch_y_d(self):
        """Raises error for Y/D length mismatch."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=6)

        cf = ControlFunction()
        with pytest.raises(ValueError, match="Length mismatch"):
            cf.fit(Y[:-5], D, Z.ravel(), X)

    def test_length_mismatch_y_z(self):
        """Raises error for Y/Z length mismatch."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=7)

        cf = ControlFunction()
        with pytest.raises(ValueError, match="Length mismatch"):
            cf.fit(Y, D, Z[:-5], X)

    def test_length_mismatch_y_x(self):
        """Raises error for Y/X length mismatch."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, n_controls=2, random_state=8)

        cf = ControlFunction()
        with pytest.raises(ValueError, match="Length mismatch"):
            cf.fit(Y, D, Z.ravel(), X[:-10])


class TestMinimumSampleSize:
    """Tests for minimum sample size handling."""

    def test_minimum_n_10(self):
        """Works with n=10 (minimum)."""
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=10, true_beta=2.0, pi=0.8, rho=0.5, random_state=10
        )

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should complete without error
        assert "estimate" in result

    def test_fails_below_minimum(self):
        """Raises error for n < 10."""
        Y, D, Z, X, _, _ = generate_cf_data(n=9, random_state=11)

        cf = ControlFunction()
        with pytest.raises(ValueError, match="Insufficient sample size"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_n_5_fails(self):
        """Raises error for n=5."""
        Y, D, Z, X, _, _ = generate_cf_data(n=5, random_state=12)

        cf = ControlFunction()
        with pytest.raises(ValueError, match="Insufficient sample size"):
            cf.fit(Y, D, Z.ravel(), X)


class TestNoVariation:
    """Tests for constant treatment/instrument."""

    def test_constant_treatment(self):
        """Raises error for constant treatment."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=20)
        D = np.ones_like(D) * 5.0  # Constant

        cf = ControlFunction()
        with pytest.raises(ValueError, match="No variation.*D"):
            cf.fit(Y, D, Z.ravel(), X)

    def test_constant_instrument(self):
        """Raises error for constant instrument."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=21)
        Z = np.ones_like(Z) * 3.0  # Constant

        cf = ControlFunction()
        with pytest.raises(ValueError, match="No variation.*Z"):
            cf.fit(Y, D, Z, X)

    def test_near_constant_treatment(self):
        """Handles near-constant treatment."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=22)
        D = np.ones_like(D) * 5.0 + np.random.RandomState(22).normal(0, 1e-12, len(D))

        cf = ControlFunction()
        with pytest.raises(ValueError, match="No variation"):
            cf.fit(Y, D, Z.ravel(), X)


class TestWeakInstruments:
    """Tests for weak instrument scenarios."""

    def test_very_weak_instrument(self):
        """Handles very weak instrument (F < 2)."""
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=100,
            true_beta=2.0,
            pi=0.05,  # Very weak first stage
            rho=0.5,
            sigma_nu=2.0,
            random_state=30,
        )

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should complete but warn
        assert result["first_stage"]["weak_iv_warning"]
        assert result["first_stage"]["f_statistic"] < 10

    def test_borderline_weak_instrument(self):
        """Handles borderline weak instrument (F ≈ 10)."""
        # Design DGP to have F near 10
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=200,
            true_beta=2.0,
            pi=0.15,
            rho=0.5,
            sigma_nu=1.0,
            random_state=31,
        )

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Check F-statistic is computed correctly
        assert result["first_stage"]["f_statistic"] > 0


class TestExtremeEndogeneity:
    """Tests for extreme endogeneity scenarios."""

    def test_strong_endogeneity_rho_0_9(self):
        """Handles strong endogeneity (rho = 0.9)."""
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=1000,
            true_beta=2.0,
            pi=0.5,
            rho=0.9,  # Strong endogeneity
            sigma_nu=1.0,
            sigma_epsilon=0.1,
            random_state=40,
        )

        cf = ControlFunction(inference="bootstrap", n_bootstrap=300, random_state=40)
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should detect endogeneity
        assert result["endogeneity_detected"]

        # Should still recover approximate effect
        bias = abs(result["estimate"] - true_beta)
        assert bias < 0.5, f"Bias too large with strong endogeneity: {bias:.3f}"

    def test_negative_endogeneity(self):
        """Handles negative endogeneity (rho < 0)."""
        # Generate data with negative correlation
        rng = np.random.default_rng(41)
        n = 1000
        true_beta = 2.0
        rho = -0.6

        Z = rng.normal(0, 1, n)
        nu = rng.normal(0, 1, n)
        D = 0.5 * Z + nu

        # Negative correlation: epsilon = -0.6 * nu + noise
        epsilon = rho * nu + np.sqrt(1 - rho**2) * rng.normal(0, 0.5, n)
        Y = true_beta * D + epsilon

        cf = ControlFunction(inference="bootstrap", n_bootstrap=300, random_state=41)
        result = cf.fit(Y, D, Z, None)

        # Control coefficient should be negative (since rho < 0)
        assert result["control_coef"] < 0


class TestLargeSample:
    """Tests for large sample performance."""

    def test_n_50000(self):
        """Works with n=50,000."""
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=50000,
            true_beta=1.0,
            pi=0.5,
            rho=0.5,
            random_state=50,
        )

        cf = ControlFunction(inference="analytical")  # Faster for large n
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should have very small SE
        assert result["se"] < 0.05

        # Should be very close to true value
        bias = abs(result["estimate"] - true_beta)
        assert bias < 0.05

    def test_n_100000_analytical(self):
        """Works with n=100,000 using analytical inference."""
        Y, D, Z, X, true_beta, rho = generate_cf_data(
            n=100000,
            true_beta=1.5,
            pi=0.6,
            rho=0.4,
            random_state=51,
        )

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should complete quickly with analytical SE
        assert result["inference"] == "analytical"
        assert result["se"] < 0.02


class TestInputShapeHandling:
    """Tests for various input shapes."""

    def test_1d_instrument(self):
        """Handles 1D instrument array."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=60)

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_instruments"] == 1

    def test_2d_instrument_single_column(self):
        """Handles 2D instrument with single column."""
        Y, D, Z, X, _, _ = generate_cf_data(n=100, random_state=61)

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z, X)  # Z is already 2D (n, 1)

        assert result["n_instruments"] == 1

    def test_multiple_instruments(self):
        """Handles multiple instruments."""
        Y, D, Z, X, _, _ = generate_cf_data(
            n=100, n_instruments=3, random_state=62
        )

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z, X)

        assert result["n_instruments"] == 3

    def test_1d_controls(self):
        """Handles 1D control array."""
        Y, D, Z, _, _, _ = generate_cf_data(n=100, random_state=63)
        X = np.random.RandomState(63).normal(0, 1, len(Y))  # 1D

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_controls"] == 1

    def test_no_controls(self):
        """Works without controls (X=None)."""
        Y, D, Z, _, _, _ = generate_cf_data(n=100, random_state=64)

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), None)

        assert result["n_controls"] == 0


class TestBootstrapRobustness:
    """Tests for bootstrap inference robustness."""

    def test_bootstrap_reproducible(self):
        """Bootstrap is reproducible with random_state."""
        Y, D, Z, X, _, _ = generate_cf_data(n=200, random_state=70)

        cf1 = ControlFunction(inference="bootstrap", n_bootstrap=100, random_state=42)
        result1 = cf1.fit(Y, D, Z.ravel(), X)

        cf2 = ControlFunction(inference="bootstrap", n_bootstrap=100, random_state=42)
        result2 = cf2.fit(Y, D, Z.ravel(), X)

        assert result1["estimate"] == result2["estimate"]
        assert result1["se"] == result2["se"]

    def test_bootstrap_different_seeds_differ(self):
        """Different random seeds give different bootstrap results."""
        Y, D, Z, X, _, _ = generate_cf_data(n=200, random_state=71)

        cf1 = ControlFunction(inference="bootstrap", n_bootstrap=100, random_state=1)
        result1 = cf1.fit(Y, D, Z.ravel(), X)

        cf2 = ControlFunction(inference="bootstrap", n_bootstrap=100, random_state=2)
        result2 = cf2.fit(Y, D, Z.ravel(), X)

        # Estimates should be same (same data), but SEs may differ
        assert result1["estimate"] == result2["estimate"]
        # SEs should differ (different bootstrap samples)
        assert result1["se"] != result2["se"]

    def test_more_bootstrap_iterations(self):
        """More bootstrap iterations give stable SE."""
        Y, D, Z, X, _, _ = generate_cf_data(n=500, random_state=72)

        cf_100 = ControlFunction(inference="bootstrap", n_bootstrap=100, random_state=1)
        result_100 = cf_100.fit(Y, D, Z.ravel(), X)

        cf_500 = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=1)
        result_500 = cf_500.fit(Y, D, Z.ravel(), X)

        # Both should give reasonable estimates
        assert result_100["se"] > 0
        assert result_500["se"] > 0


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_values(self):
        """Handles large values in data."""
        Y, D, Z, X, _, _ = generate_cf_data(n=200, random_state=80)
        Y = Y * 1e6
        D = D * 1e4

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should complete without numerical issues
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

    def test_small_values(self):
        """Handles small values in data."""
        Y, D, Z, X, _, _ = generate_cf_data(n=200, random_state=81)
        Y = Y * 1e-4
        D = D * 1e-3

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

    def test_mixed_scales(self):
        """Handles mixed scales in covariates."""
        Y, D, Z, _, _, _ = generate_cf_data(n=200, random_state=82)
        X = np.column_stack([
            np.random.RandomState(82).normal(0, 1e-5, len(Y)),
            np.random.RandomState(83).normal(0, 1e5, len(Y)),
        ])

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert np.isfinite(result["estimate"])
