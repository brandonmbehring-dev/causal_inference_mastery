"""
Cross-Language Parity Tests for IV Stage Decomposition (Python ↔ Julia).

Session 56: Tests FirstStage, ReducedForm, SecondStage cross-language parity.

Test Methodology:
1. Generate identical DGP in Python
2. Call Python stages.py functions
3. Call Julia stages.jl via juliacall
4. Compare coefficients, F-statistics, fitted values

Tolerance:
- Coefficients: rtol=0.01 (numerical precision)
- F-statistics: rtol=0.05 (more sensitive to implementation)
- R²: rtol=0.01
"""

import numpy as np
import pytest

from src.causal_inference.iv.stages import FirstStage, ReducedForm, SecondStage

try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_first_stage,
        julia_reduced_form,
        julia_second_stage,
    )

    JULIA_AVAILABLE = is_julia_available()
except ImportError:
    JULIA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not JULIA_AVAILABLE, reason="Julia not available for cross-language validation"
)


class TestFirstStageParity:
    """Cross-language parity tests for FirstStage regression."""

    def test_first_stage_coefficients_simple(self):
        """FirstStage coefficients match Python ↔ Julia (no covariates)."""
        np.random.seed(42)
        n = 1000

        # Strong first stage: D = 0.8Z + noise
        z = np.random.normal(0, 1, n)
        d = 0.8 * z + 0.5 * np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python
        py_model = FirstStage()
        py_model.fit(d, Z)

        # Julia
        jl_result = julia_first_stage(d, Z)

        # Compare coefficients
        assert np.allclose(py_model.coef_, jl_result["coef"], rtol=0.01), (
            f"Coefficient mismatch: Python={py_model.coef_}, Julia={jl_result['coef']}"
        )

        # Compare R²
        assert np.isclose(py_model.r2_, jl_result["r2"], rtol=0.01), (
            f"R² mismatch: Python={py_model.r2_}, Julia={jl_result['r2']}"
        )

    def test_first_stage_f_statistic(self):
        """FirstStage F-statistic matches Python ↔ Julia."""
        np.random.seed(123)
        n = 500

        z = np.random.normal(0, 1, n)
        d = 0.7 * z + 0.3 * np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python
        py_model = FirstStage()
        py_model.fit(d, Z)

        # Julia
        jl_result = julia_first_stage(d, Z)

        # Compare F-statistic (more tolerance due to implementation differences)
        assert np.isclose(py_model.f_statistic_, jl_result["f_statistic"], rtol=0.05), (
            f"F-stat mismatch: Python={py_model.f_statistic_}, Julia={jl_result['f_statistic']}"
        )

    def test_first_stage_partial_r2(self):
        """FirstStage partial R² matches Python ↔ Julia (with covariates)."""
        np.random.seed(456)
        n = 500

        # D = 0.6Z + 0.4X + noise
        z = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n)
        d = 0.6 * z + 0.4 * x + 0.3 * np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)
        X = x.reshape(-1, 1)

        # Python
        py_model = FirstStage()
        py_model.fit(d, Z, X)

        # Julia
        jl_result = julia_first_stage(d, Z, X)

        # Compare partial R²
        assert np.isclose(py_model.partial_r2_, jl_result["partial_r2"], rtol=0.05), (
            f"Partial R² mismatch: Python={py_model.partial_r2_}, Julia={jl_result['partial_r2']}"
        )

    def test_first_stage_fitted_values(self):
        """FirstStage fitted values match Python ↔ Julia."""
        np.random.seed(789)
        n = 200

        z = np.random.normal(0, 1, n)
        d = 0.5 * z + np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python
        py_model = FirstStage()
        py_model.fit(d, Z)

        # Julia
        jl_result = julia_first_stage(d, Z)

        # Compare fitted values (correlation should be ~1.0)
        corr = np.corrcoef(py_model.fitted_values_, jl_result["fitted_values"])[0, 1]
        assert corr > 0.999, f"Fitted values correlation too low: {corr}"


class TestReducedFormParity:
    """Cross-language parity tests for ReducedForm regression."""

    def test_reduced_form_coefficients(self):
        """ReducedForm coefficients match Python ↔ Julia."""
        np.random.seed(101)
        n = 1000

        # Y = 2Z + noise (direct reduced form)
        z = np.random.normal(0, 1, n)
        y = 2.0 * z + np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python
        py_model = ReducedForm()
        py_model.fit(y, Z)

        # Julia
        jl_result = julia_reduced_form(y, Z)

        # Compare coefficients
        assert np.allclose(py_model.coef_, jl_result["coef"], rtol=0.01), (
            f"Coefficient mismatch: Python={py_model.coef_}, Julia={jl_result['coef']}"
        )

    def test_reduced_form_with_covariates(self):
        """ReducedForm with covariates matches Python ↔ Julia."""
        np.random.seed(202)
        n = 500

        # Y = 1.5Z + 0.8X + noise
        z = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n)
        y = 1.5 * z + 0.8 * x + np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)
        X = x.reshape(-1, 1)

        # Python
        py_model = ReducedForm()
        py_model.fit(y, Z, X)

        # Julia
        jl_result = julia_reduced_form(y, Z, X)

        # Compare both coefficients
        assert np.allclose(py_model.coef_, jl_result["coef"], rtol=0.01), (
            f"Coefficient mismatch: Python={py_model.coef_}, Julia={jl_result['coef']}"
        )

    def test_reduced_form_r2(self):
        """ReducedForm R² matches Python ↔ Julia."""
        np.random.seed(303)
        n = 500

        z = np.random.normal(0, 1, n)
        y = 3.0 * z + 0.5 * np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python
        py_model = ReducedForm()
        py_model.fit(y, Z)

        # Julia
        jl_result = julia_reduced_form(y, Z)

        # Compare R²
        assert np.isclose(py_model.r2_, jl_result["r2"], rtol=0.01), (
            f"R² mismatch: Python={py_model.r2_}, Julia={jl_result['r2']}"
        )


class TestSecondStageParity:
    """Cross-language parity tests for SecondStage regression."""

    def test_second_stage_coefficients(self):
        """SecondStage coefficients match Python ↔ Julia."""
        np.random.seed(404)
        n = 1000

        # Simulated D̂ and Y
        d_hat = np.random.normal(0, 1, n)
        y = 2.5 * d_hat + 0.3 * np.random.normal(0, 1, n)

        # Python
        py_model = SecondStage()
        py_model.fit(y, d_hat)

        # Julia (will warn about naive SEs)
        jl_result = julia_second_stage(y, d_hat)

        # Compare coefficients
        assert np.allclose(py_model.coef_, jl_result["coef"], rtol=0.01), (
            f"Coefficient mismatch: Python={py_model.coef_}, Julia={jl_result['coef']}"
        )

    def test_second_stage_naive_se(self):
        """SecondStage naive SEs match Python ↔ Julia (both WRONG but consistent)."""
        np.random.seed(505)
        n = 500

        d_hat = np.random.normal(0, 1, n)
        y = 3.0 * d_hat + np.random.normal(0, 1, n)

        # Python
        py_model = SecondStage()
        py_model.fit(y, d_hat)

        # Julia
        jl_result = julia_second_stage(y, d_hat)

        # Compare naive SEs (both implementations should produce same WRONG SEs)
        assert np.allclose(py_model.se_naive_, jl_result["se_naive"], rtol=0.01), (
            f"Naive SE mismatch: Python={py_model.se_naive_}, Julia={jl_result['se_naive']}"
        )


class TestWaldIdentityParity:
    """Cross-language parity for Wald identity: γ = π × β."""

    def test_wald_identity_holds(self):
        """Wald identity γ = π × β holds in both Python and Julia."""
        np.random.seed(606)
        n = 2000

        # DGP: Z → D → Y
        true_pi = 0.8
        true_beta = 2.0

        z = np.random.normal(0, 1, n)
        d = true_pi * z + 0.5 * np.random.normal(0, 1, n)
        y = true_beta * d + np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python first stage
        py_first = FirstStage()
        py_first.fit(d, Z)
        py_pi = py_first.coef_[0]

        # Python reduced form
        py_reduced = ReducedForm()
        py_reduced.fit(y, Z)
        py_gamma = py_reduced.coef_[0]

        # Julia first stage
        jl_first = julia_first_stage(d, Z)
        jl_pi = jl_first["coef"][0]

        # Julia reduced form
        jl_reduced = julia_reduced_form(y, Z)
        jl_gamma = jl_reduced["coef"][0]

        # Wald identity: β = γ / π
        py_beta_wald = py_gamma / py_pi
        jl_beta_wald = jl_gamma / jl_pi

        # Both should recover similar structural effect
        assert np.isclose(py_beta_wald, jl_beta_wald, rtol=0.01), (
            f"Wald β mismatch: Python={py_beta_wald}, Julia={jl_beta_wald}"
        )

        # Both should be close to true effect
        assert abs(py_beta_wald - true_beta) < 0.15, (
            f"Python Wald β too far from truth: {py_beta_wald} vs {true_beta}"
        )
        assert abs(jl_beta_wald - true_beta) < 0.15, (
            f"Julia Wald β too far from truth: {jl_beta_wald} vs {true_beta}"
        )


class TestEndToEndParity:
    """End-to-end IV decomposition parity tests."""

    def test_manual_tsls_matches_tsls(self):
        """Manual 2-stage process matches TSLS in both languages."""
        np.random.seed(707)
        n = 1000

        # Generate IV data
        z = np.random.normal(0, 1, n)
        d = 0.7 * z + 0.5 * np.random.normal(0, 1, n)
        y = 2.0 * d + np.random.normal(0, 1, n)

        Z = z.reshape(-1, 1)

        # Python: manual two-stage
        py_first = FirstStage()
        py_first.fit(d, Z)
        d_hat_py = py_first.fitted_values_

        py_second = SecondStage()
        py_second.fit(y, d_hat_py)

        # Julia: manual two-stage
        jl_first = julia_first_stage(d, Z)
        d_hat_jl = jl_first["fitted_values"]

        jl_second = julia_second_stage(y, d_hat_jl)

        # Coefficients should match
        assert np.isclose(py_second.coef_[0], jl_second["coef"][0], rtol=0.01), (
            f"Manual 2SLS coef mismatch: Python={py_second.coef_[0]}, Julia={jl_second['coef'][0]}"
        )

        # Both should be close to true effect
        assert abs(py_second.coef_[0] - 2.0) < 0.15
        assert abs(jl_second["coef"][0] - 2.0) < 0.15
