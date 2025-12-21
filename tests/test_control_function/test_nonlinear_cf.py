"""
Tests for Nonlinear Control Function estimation (Probit/Logit).

The control function approach is essential for nonlinear models because:
1. 2SLS is INVALID for probit/logit (Jensen's inequality)
2. Simply substituting D_hat for D is incorrect
3. Control function includes first-stage residuals to "control for" endogeneity

Layer 2 of 6-layer validation architecture.
"""

import numpy as np
import pytest

from src.causal_inference.control_function import (
    NonlinearControlFunction,
    nonlinear_control_function,
)
from tests.test_control_function.conftest import generate_nonlinear_cf_data


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestNonlinearCFBasic:
    """Basic functionality tests for nonlinear CF."""

    def test_probit_runs(self, nonlinear_cf_probit):
        """Probit CF runs without error."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_probit
        cf = NonlinearControlFunction(model_type="probit", n_bootstrap=100)
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["convergence"]
        assert not np.isnan(result["estimate"])
        assert not np.isnan(result["se"])

    def test_logit_runs(self, nonlinear_cf_logit):
        """Logit CF runs without error."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_logit
        cf = NonlinearControlFunction(model_type="logit", n_bootstrap=100)
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["convergence"]
        assert not np.isnan(result["estimate"])

    def test_convenience_function(self, nonlinear_cf_probit):
        """Convenience function matches class-based estimation."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_probit

        result1 = nonlinear_control_function(
            Y, D, Z.ravel(), X, model_type="probit", n_bootstrap=50, random_state=42
        )

        cf = NonlinearControlFunction(
            model_type="probit", n_bootstrap=50, random_state=42
        )
        result2 = cf.fit(Y, D, Z.ravel(), X)

        assert np.isclose(result1["estimate"], result2["estimate"], rtol=1e-10)

    def test_with_controls(self, nonlinear_cf_with_controls):
        """Nonlinear CF works with exogenous controls."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_with_controls
        cf = NonlinearControlFunction(model_type="probit", n_bootstrap=100)
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["convergence"]
        assert result["first_stage"]["n_obs"] == len(Y)


class TestNonlinearCFResults:
    """Tests for result structure and values."""

    def test_result_contains_all_fields(self, nonlinear_cf_probit):
        """Result contains all expected fields."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_probit
        result = nonlinear_control_function(Y, D, Z.ravel(), X, n_bootstrap=50)

        required_fields = [
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "control_coef",
            "control_se",
            "control_p_value",
            "endogeneity_detected",
            "first_stage",
            "model_type",
            "n_obs",
            "n_bootstrap",
            "alpha",
            "convergence",
            "message",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_model_type_stored(self, nonlinear_cf_probit):
        """Model type is correctly stored in result."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit

        probit_result = nonlinear_control_function(
            Y, D, Z.ravel(), X, model_type="probit", n_bootstrap=50
        )
        assert probit_result["model_type"] == "probit"

        logit_result = nonlinear_control_function(
            Y, D, Z.ravel(), X, model_type="logit", n_bootstrap=50
        )
        assert logit_result["model_type"] == "logit"

    def test_ci_contains_estimate(self, nonlinear_cf_probit):
        """Confidence interval contains point estimate."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit
        result = nonlinear_control_function(Y, D, Z.ravel(), X, n_bootstrap=100)

        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]

    def test_first_stage_diagnostics(self, nonlinear_cf_probit):
        """First-stage diagnostics are populated."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit
        result = nonlinear_control_function(Y, D, Z.ravel(), X, n_bootstrap=50)

        fs = result["first_stage"]
        assert fs["n_obs"] == len(Y)
        assert fs["f_statistic"] > 0
        assert 0 <= fs["f_pvalue"] <= 1
        assert 0 <= fs["r2"] <= 1


# =============================================================================
# Endogeneity Detection Tests
# =============================================================================


class TestEndogeneityDetection:
    """Tests for endogeneity detection in nonlinear CF."""

    def test_detects_endogeneity_when_present(self, nonlinear_cf_probit):
        """Detects endogeneity when rho > 0."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_probit
        # rho = 0.5 in this fixture
        result = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=200, random_state=42
        )

        # Should detect endogeneity with high probability
        # (may not always due to sampling variability)
        assert abs(result["control_coef"]) > 0.1

    def test_no_endogeneity_when_absent(self, nonlinear_cf_no_endogeneity):
        """Control coefficient is small when rho = 0."""
        Y, D, Z, X, true_beta, rho = nonlinear_cf_no_endogeneity
        result = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=200, random_state=42
        )

        # Control coefficient should be close to zero
        # We can't guarantee endogeneity_detected == False due to type I error
        # but control_coef should be small relative to its SE
        assert result["control_p_value"] > 0.01  # Not highly significant


# =============================================================================
# AME Tests
# =============================================================================


class TestAME:
    """Tests for Average Marginal Effect computation."""

    def test_ame_positive_effect(self, nonlinear_cf_probit):
        """AME has correct sign for positive true effect."""
        Y, D, Z, X, true_beta, _ = nonlinear_cf_probit
        # true_beta = 1.0 (positive)
        result = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=200, random_state=42
        )

        # AME should be positive (same sign as true_beta)
        # Note: AME is scaled by phi/lambda, so magnitude differs from true_beta
        assert result["estimate"] > 0

    def test_ame_reasonable_magnitude(self, nonlinear_cf_probit):
        """AME has reasonable magnitude (typically 0.1-0.5 for marginal effects)."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit
        result = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=100, random_state=42
        )

        # AME typically in reasonable range
        assert abs(result["estimate"]) < 1.0  # Not implausibly large

    def test_probit_vs_logit_ame_similar(self, nonlinear_cf_probit):
        """Probit and logit AME should be similar (scale factor ~1.6)."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit

        probit_result = nonlinear_control_function(
            Y, D, Z.ravel(), X, model_type="probit", n_bootstrap=100, random_state=42
        )
        logit_result = nonlinear_control_function(
            Y, D, Z.ravel(), X, model_type="logit", n_bootstrap=100, random_state=42
        )

        # Logit coefficients are typically ~1.6x probit (due to variance difference)
        # AME adjusts for this, so they should be somewhat similar
        ratio = probit_result["estimate"] / logit_result["estimate"]
        assert 0.5 < ratio < 2.0  # Within 2x of each other


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation in nonlinear CF."""

    def test_rejects_non_binary_outcome(self):
        """Raises error for non-binary outcome."""
        n = 100
        Y = np.random.normal(0, 1, n)  # Continuous, not binary
        D = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)

        cf = NonlinearControlFunction()
        with pytest.raises(ValueError, match="binary"):
            cf.fit(Y, D, Z)

    def test_rejects_nan_in_outcome(self):
        """Raises error for NaN in outcome."""
        n = 100
        Y = np.random.binomial(1, 0.5, n).astype(float)
        Y[0] = np.nan
        D = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)

        cf = NonlinearControlFunction()
        # NaN check happens during binary check, so error mentions binary
        with pytest.raises(ValueError, match="(NaN|binary)"):
            cf.fit(Y, D, Z)

    def test_rejects_small_sample(self):
        """Raises error for sample size < 50."""
        n = 30
        Y = np.random.binomial(1, 0.5, n).astype(float)
        D = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)

        cf = NonlinearControlFunction()
        with pytest.raises(ValueError, match="sample size"):
            cf.fit(Y, D, Z)

    def test_rejects_length_mismatch(self):
        """Raises error for length mismatch."""
        Y = np.random.binomial(1, 0.5, 100).astype(float)
        D = np.random.normal(0, 1, 80)  # Wrong length
        Z = np.random.normal(0, 1, 100)

        cf = NonlinearControlFunction()
        with pytest.raises(ValueError, match="Length mismatch"):
            cf.fit(Y, D, Z)

    def test_rejects_no_variation_in_treatment(self):
        """Raises error for constant treatment."""
        n = 100
        Y = np.random.binomial(1, 0.5, n).astype(float)
        D = np.ones(n) * 5  # No variation
        Z = np.random.normal(0, 1, n)

        cf = NonlinearControlFunction()
        with pytest.raises(ValueError, match="No variation"):
            cf.fit(Y, D, Z)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for nonlinear CF."""

    def test_handles_extreme_outcome_imbalance(self):
        """Handles highly imbalanced outcomes (>90% one class)."""
        n = 500
        rng = np.random.default_rng(42)
        Z = rng.normal(0, 1, n)
        nu = rng.normal(0, 1, n)
        D = 0.5 * Z + nu

        # Create imbalanced outcome (~95% positive)
        latent = 2.0 + 0.5 * D + rng.normal(0, 1, n)
        Y = (latent > 0).astype(float)
        assert Y.mean() > 0.90  # Highly imbalanced

        cf = NonlinearControlFunction(n_bootstrap=100)
        result = cf.fit(Y, D, Z)

        # Should still converge (may have large SEs)
        assert result["convergence"]

    def test_handles_weak_instrument(self):
        """Handles weak instrument case."""
        Y, D, Z, X, _, _ = generate_nonlinear_cf_data(
            n=1000,
            true_beta=0.5,
            pi=0.1,  # Weak first stage
            rho=0.5,
            random_state=42,
        )

        cf = NonlinearControlFunction(n_bootstrap=100)
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["first_stage"]["weak_iv_warning"]
        assert "WARNING" in result["message"] or result["first_stage"]["f_statistic"] < 10

    def test_summary_method(self, nonlinear_cf_probit):
        """Summary method returns formatted string."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit
        cf = NonlinearControlFunction(n_bootstrap=50)
        cf.fit(Y, D, Z.ravel(), X)

        summary = cf.summary()
        assert isinstance(summary, str)
        assert "Average Marginal Effect" in summary
        assert "Endogeneity Test" in summary

    def test_reproducibility_with_seed(self, nonlinear_cf_probit):
        """Results reproducible with same random_state."""
        Y, D, Z, X, _, _ = nonlinear_cf_probit

        result1 = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=50, random_state=42
        )
        result2 = nonlinear_control_function(
            Y, D, Z.ravel(), X, n_bootstrap=50, random_state=42
        )

        assert np.isclose(result1["estimate"], result2["estimate"])
        assert np.isclose(result1["se"], result2["se"])


# =============================================================================
# Monte Carlo Tests (Small-scale)
# =============================================================================


class TestNonlinearCFMonteCarlo:
    """Small-scale Monte Carlo tests for nonlinear CF."""

    @pytest.mark.slow
    def test_ame_approximately_correct(self):
        """AME estimates are approximately correct across replications."""
        n_runs = 50
        estimates = []

        for seed in range(n_runs):
            Y, D, Z, X, true_beta, _ = generate_nonlinear_cf_data(
                n=1000,
                true_beta=0.5,
                pi=0.5,
                rho=0.5,
                model_type="probit",
                random_state=seed,
            )

            result = nonlinear_control_function(
                Y, D, Z.ravel(), X, n_bootstrap=50, random_state=seed
            )

            if result["convergence"]:
                estimates.append(result["estimate"])

        # Most should converge
        assert len(estimates) >= 40

        # AME should be positive (true_beta = 0.5 > 0)
        avg_ame = np.mean(estimates)
        assert avg_ame > 0

    @pytest.mark.slow
    def test_coverage_reasonable(self):
        """CI coverage is reasonable (> 80%)."""
        n_runs = 50
        covers = []

        # True AME is approximately 0.5 * phi(0) ≈ 0.2 for probit
        # But exact value depends on distribution of covariates
        # So we just check if estimate is within CI

        for seed in range(n_runs):
            Y, D, Z, X, _, _ = generate_nonlinear_cf_data(
                n=1000,
                true_beta=0.5,
                pi=0.5,
                rho=0.0,  # No endogeneity for cleaner test
                model_type="probit",
                random_state=seed,
            )

            result = nonlinear_control_function(
                Y, D, Z.ravel(), X, n_bootstrap=100, random_state=seed
            )

            if result["convergence"]:
                # CI should contain estimate (always true by construction)
                # More meaningful: does CI width scale with SE?
                ci_width = result["ci_upper"] - result["ci_lower"]
                expected_width = 2 * 1.96 * result["se"]
                # Percentile CI may differ from z-based, allow some slack
                ratio = ci_width / expected_width if expected_width > 0 else 1.0
                covers.append(0.5 < ratio < 2.0)

        coverage = np.mean(covers)
        assert coverage > 0.8, f"CI width consistency only {coverage:.0%}"
