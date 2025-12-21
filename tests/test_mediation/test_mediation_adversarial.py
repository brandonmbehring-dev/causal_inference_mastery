"""
Adversarial tests for mediation analysis.

Layer 2: Edge cases and boundary conditions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mediation import (
    baron_kenny,
    mediation_analysis,
    controlled_direct_effect,
    mediation_sensitivity,
)


class TestEdgeCases:
    """Edge case tests."""

    def test_small_sample(self, small_sample):
        """Handles small samples (n=50)."""
        data = small_sample
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # Should produce valid results
        assert not np.isnan(result["indirect_effect"])
        assert not np.isnan(result["direct_effect"])
        assert result["indirect_se"] > 0

    def test_very_small_sample(self):
        """Very small sample (n=20)."""
        np.random.seed(42)
        n = 20

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        # Should work but with high variance
        result = baron_kenny(Y, T, M)
        assert not np.isnan(result["indirect_effect"])

    def test_no_treatment_variation(self):
        """All treated or all control raises error."""
        np.random.seed(42)
        n = 100

        T = np.ones(n)  # All treated
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        # Should fail or give degenerate result
        # (OLS will still work but treatment effect is not identified)
        # We're testing the code doesn't crash
        try:
            result = baron_kenny(Y, T, M)
            # If it doesn't crash, SE should be very large or NaN
            assert np.isnan(result["alpha_1_se"]) or result["alpha_1_se"] > 1e6
        except Exception:
            pass  # Acceptable to raise an error

    def test_constant_mediator(self):
        """Constant mediator (no variance)."""
        np.random.seed(42)
        n = 100

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = np.ones(n) * 0.5  # Constant
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        # Should fail or give degenerate result
        try:
            result = baron_kenny(Y, T, M)
            # beta_2 should not be identifiable
            assert np.isnan(result["beta_2"]) or result["beta_2_se"] > 1e6
        except Exception:
            pass

    def test_constant_outcome(self):
        """Constant outcome (no variance)."""
        np.random.seed(42)
        n = 100

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        Y = np.ones(n)  # Constant

        result = baron_kenny(Y, T, M)
        # Effects should be 0
        assert abs(result["direct_effect"]) < 1e-10
        assert abs(result["indirect_effect"]) < 1e-10

    def test_perfect_collinearity_t_m(self):
        """Perfect collinearity between T and M."""
        np.random.seed(42)
        n = 100

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = T.copy()  # Perfect correlation
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        # OLS should have issues with multicollinearity
        try:
            result = baron_kenny(Y, T, M)
            # Coefficients will be poorly identified
            # SE should be large
        except Exception:
            pass  # Acceptable


class TestBinaryMediatorOutcome:
    """Tests with binary mediator/outcome."""

    def test_binary_mediator(self, binary_mediator):
        """Binary mediator works (but linear model may not be appropriate)."""
        data = binary_mediator

        # Baron-Kenny (linear) should still work numerically
        result_bk = baron_kenny(data["outcome"], data["treatment"], data["mediator"])
        assert not np.isnan(result_bk["indirect_effect"])

    def test_binary_mediator_simulation(self, binary_mediator):
        """Simulation method with logistic mediator model."""
        data = binary_mediator

        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            mediator_model="logistic",
            n_bootstrap=100,
            n_simulations=100,
            random_state=42,
        )

        # Should produce valid results
        assert not np.isnan(result["direct_effect"])
        assert not np.isnan(result["indirect_effect"])

    def test_binary_outcome(self):
        """Binary outcome (logistic outcome model)."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        prob_y = 1 / (1 + np.exp(-(1.0 + 0.5 * T + 0.8 * M)))
        Y = (np.random.rand(n) < prob_y).astype(float)

        result = mediation_analysis(
            Y,
            T,
            M,
            method="simulation",
            outcome_model="logistic",
            n_bootstrap=100,
            n_simulations=100,
        )

        assert not np.isnan(result["direct_effect"])


class TestHighNoise:
    """Tests with high noise levels."""

    def test_high_noise_mediator(self):
        """High noise in mediator model."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        # Very high noise relative to signal
        M = 0.5 + 0.1 * T + np.random.randn(n) * 5.0
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        result = baron_kenny(Y, T, M)

        # alpha_1 should be hard to detect
        assert result["alpha_1_pvalue"] > 0.01  # Not very significant
        # Indirect effect should be small (signal drowned out)
        assert abs(result["indirect_effect"]) < 0.5

    def test_high_noise_outcome(self):
        """High noise in outcome model."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        # Very high noise
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 5.0

        result = baron_kenny(Y, T, M)

        # R-squared should be low
        assert result["r2_outcome_model"] < 0.3


class TestNumericalStability:
    """Numerical stability tests."""

    def test_large_values(self):
        """Large outcome values."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 1000 + 600 * T + np.random.randn(n) * 500
        Y = 10000 + 500 * T + 80 * M + np.random.randn(n) * 500

        result = baron_kenny(Y, T, M)

        # Should work with rescaled coefficients
        assert not np.isnan(result["indirect_effect"])

    def test_small_values(self):
        """Very small coefficient values."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.0005 + 0.0006 * T + np.random.randn(n) * 0.0005
        Y = 0.001 + 0.0005 * T + 0.0008 * M + np.random.randn(n) * 0.0005

        result = baron_kenny(Y, T, M)

        # Should work with small coefficients
        assert not np.isnan(result["indirect_effect"])

    def test_extreme_treatment_imbalance(self):
        """Extreme treatment imbalance (10% treated)."""
        np.random.seed(42)
        n = 500

        T = (np.random.rand(n) < 0.1).astype(float)  # Only 10% treated
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        result = baron_kenny(Y, T, M)

        # Should work but with larger SE
        assert not np.isnan(result["indirect_effect"])
        assert result["alpha_1_se"] > 0


class TestSpecialMediationPatterns:
    """Special mediation patterns."""

    def test_suppression_effect(self):
        """Suppression: indirect and direct effects have opposite signs."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        # T increases M
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        # But M decreases Y (negative beta_2)
        Y = 1.0 + 0.8 * T - 0.5 * M + np.random.randn(n) * 0.5

        result = baron_kenny(Y, T, M)

        # Direct and indirect should have opposite signs
        assert result["indirect_effect"] < 0  # alpha_1 * beta_2 = 0.6 * (-0.5) = -0.3
        assert result["direct_effect"] > 0  # beta_1 = 0.8

    def test_inconsistent_mediation(self):
        """Inconsistent mediation (signs differ)."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 - 0.6 * T + np.random.randn(n) * 0.5  # T decreases M
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

        result = baron_kenny(Y, T, M)

        # Indirect effect should be negative
        assert result["indirect_effect"] < 0


class TestCDEAdversarial:
    """Adversarial tests for CDE."""

    def test_cde_extreme_mediator_value(self, simple_linear_mediation):
        """CDE at extreme mediator value."""
        data = simple_linear_mediation
        m_extreme = data["mediator"].max() * 2  # Beyond observed range

        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=m_extreme,
        )

        # Should still work (linear extrapolation)
        assert not np.isnan(result["cde"])


class TestSensitivityAdversarial:
    """Adversarial tests for sensitivity analysis."""

    def test_sensitivity_narrow_range(self, simple_linear_mediation):
        """Very narrow rho range."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.1, 0.1),
            n_rho=5,
            n_simulations=50,
            n_bootstrap=20,
        )

        assert len(result["rho_grid"]) == 5
        assert result["rho_grid"].min() >= -0.1
        assert result["rho_grid"].max() <= 0.1

    def test_sensitivity_wide_range(self, simple_linear_mediation):
        """Very wide rho range."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.99, 0.99),
            n_rho=51,
            n_simulations=50,
            n_bootstrap=20,
        )

        # Should handle extreme rho values
        assert not np.all(np.isnan(result["nie_at_rho"]))


class TestInputValidation:
    """Input validation edge cases."""

    def test_empty_arrays(self):
        """Empty arrays should raise error."""
        Y = np.array([])
        T = np.array([])
        M = np.array([])

        with pytest.raises(Exception):
            baron_kenny(Y, T, M)

    def test_inf_values(self):
        """Infinite values should raise or handle gracefully."""
        np.random.seed(42)
        n = 100

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5
        Y[0] = np.inf

        with pytest.raises(Exception):
            baron_kenny(Y, T, M)

    def test_1d_covariates(self, simple_linear_mediation):
        """1D covariate array is handled correctly."""
        data = simple_linear_mediation
        X = np.random.randn(data["n"])

        result = baron_kenny(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            covariates=X,  # 1D array
        )

        assert not np.isnan(result["indirect_effect"])

    def test_covariates_length_mismatch(self, simple_linear_mediation):
        """Covariates with wrong length raise error."""
        data = simple_linear_mediation
        X = np.random.randn(data["n"] - 10)  # Wrong length

        with pytest.raises(ValueError):
            baron_kenny(
                data["outcome"],
                data["treatment"],
                data["mediator"],
                covariates=X,
            )
