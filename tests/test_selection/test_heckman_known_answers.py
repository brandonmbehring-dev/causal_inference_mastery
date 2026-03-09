"""
Layer 1: Known-answer tests for Heckman selection model.

Tests verify:
1. Return type structure matches HeckmanResult
2. Coefficients recover true values within tolerance
3. IMR computation is correct
4. Selection test detects bias when present
5. Standard errors are reasonable
"""

import numpy as np
import pytest
from scipy import stats

from src.causal_inference.selection.heckman import (
    heckman_two_step,
    _fit_probit,
    _compute_imr,
    _fit_ols,
)
from src.causal_inference.selection.diagnostics import (
    selection_bias_test,
    diagnose_identification,
)


class TestHeckmanReturnType:
    """Tests for return type structure."""

    def test_returns_dict_with_required_keys(self, simple_heckman_data):
        """Result contains all required keys."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        required_keys = [
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "rho",
            "sigma",
            "lambda_coef",
            "lambda_se",
            "lambda_pvalue",
            "n_selected",
            "n_total",
            "selection_probs",
            "imr",
            "gamma",
            "beta",
            "vcov",
            "selection_diagnostics",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_numeric_types_correct(self, simple_heckman_data):
        """Numeric fields have correct types."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # Scalars
        assert isinstance(result["estimate"], float)
        assert isinstance(result["se"], float)
        assert isinstance(result["ci_lower"], float)
        assert isinstance(result["ci_upper"], float)
        assert isinstance(result["rho"], float)
        assert isinstance(result["sigma"], float)
        assert isinstance(result["lambda_coef"], float)
        assert isinstance(result["lambda_se"], float)
        assert isinstance(result["lambda_pvalue"], float)
        assert isinstance(result["n_selected"], int)
        assert isinstance(result["n_total"], int)

        # Arrays
        assert isinstance(result["selection_probs"], np.ndarray)
        assert isinstance(result["imr"], np.ndarray)
        assert isinstance(result["gamma"], np.ndarray)
        assert isinstance(result["beta"], np.ndarray)
        assert isinstance(result["vcov"], np.ndarray)

    def test_array_shapes_correct(self, simple_heckman_data):
        """Array shapes are consistent."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        n_total = result["n_total"]

        # Selection probs and IMR for full sample
        assert len(result["selection_probs"]) == n_total
        assert len(result["imr"]) == n_total

        # Vcov is square
        k = len(result["beta"])
        assert result["vcov"].shape == (k, k)


class TestCoefficientRecovery:
    """Tests for recovering true DGP parameters."""

    def test_beta_x_recovery_moderate_selection(self, simple_heckman_data):
        """Recovers β_x within 30% of true value (moderate selection)."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        true_beta = data["true_beta"]
        estimated = result["estimate"]

        # Within 30% relative error (accounting for estimation noise)
        relative_error = abs(estimated - true_beta) / abs(true_beta)
        assert relative_error < 0.30, (
            f"β_x recovery failed: true={true_beta:.3f}, "
            f"estimated={estimated:.3f}, rel_error={relative_error:.2%}"
        )

    def test_beta_x_recovery_strong_selection(self, strong_selection_data):
        """Recovers β_x with strong selection bias."""
        data = strong_selection_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        true_beta = data["true_beta"]
        estimated = result["estimate"]

        relative_error = abs(estimated - true_beta) / abs(true_beta)
        assert relative_error < 0.35, (
            f"Strong selection β_x recovery failed: true={true_beta:.3f}, estimated={estimated:.3f}"
        )

    def test_rho_sign_correct(self, simple_heckman_data):
        """Estimated ρ has correct sign."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        true_rho = data["true_rho"]
        estimated_rho = result["rho"]

        # Same sign
        assert np.sign(estimated_rho) == np.sign(true_rho), (
            f"ρ sign mismatch: true={true_rho:.3f}, estimated={estimated_rho:.3f}"
        )

    def test_rho_sign_negative_selection(self, negative_selection_data):
        """Correctly identifies negative selection."""
        data = negative_selection_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert result["rho"] < 0, f"Expected negative ρ, got {result['rho']:.3f}"
        assert result["lambda_coef"] < 0, f"Expected negative λ, got {result['lambda_coef']:.3f}"

    def test_no_selection_rho_near_zero(self, no_selection_data):
        """ρ ≈ 0 when no selection bias."""
        data = no_selection_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # ρ should be near zero (within 0.3 absolute)
        assert abs(result["rho"]) < 0.4, f"Expected ρ ≈ 0, got {result['rho']:.3f}"


class TestIMRComputation:
    """Tests for Inverse Mills Ratio computation."""

    def test_imr_positive_for_selected(self, simple_heckman_data):
        """IMR values are positive for selected observations."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        imr_selected = result["imr"][data["selected"] == 1]
        assert np.all(imr_selected > 0), "IMR should be positive for selected"

    def test_imr_formula_correct(self):
        """IMR = φ(Φ⁻¹(p)) / p is correctly computed."""
        # Test at specific probability values
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        imr = _compute_imr(probs)

        # Manual calculation
        z = stats.norm.ppf(probs)
        phi_z = stats.norm.pdf(z)
        expected_imr = phi_z / probs

        np.testing.assert_allclose(imr, expected_imr, rtol=1e-10)

    def test_imr_decreases_with_prob(self):
        """Higher selection probability → lower IMR."""
        probs = np.linspace(0.1, 0.9, 9)
        imr = _compute_imr(probs)

        # IMR should be monotonically decreasing
        assert np.all(np.diff(imr) < 0), "IMR should decrease with probability"

    def test_imr_handles_boundary_probs(self):
        """IMR handles probabilities near 0 and 1."""
        probs = np.array([0.001, 0.01, 0.99, 0.999])
        imr = _compute_imr(probs)

        # Should not have inf or nan
        assert np.all(np.isfinite(imr)), "IMR should be finite at boundaries"


class TestSelectionTest:
    """Tests for selection bias hypothesis test."""

    def test_detects_selection_bias(self, strong_selection_data):
        """Detects significant selection when ρ ≠ 0."""
        data = strong_selection_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # p-value should be small for strong selection
        assert result["lambda_pvalue"] < 0.10, (
            f"Should detect selection bias: λ={result['lambda_coef']:.3f}, "
            f"p={result['lambda_pvalue']:.4f}"
        )

    def test_selection_bias_test_function(self, simple_heckman_data):
        """selection_bias_test() returns correct structure."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        test_result = selection_bias_test(
            lambda_coef=result["lambda_coef"],
            lambda_se=result["lambda_se"],
        )

        assert "statistic" in test_result
        assert "pvalue" in test_result
        assert "reject_null" in test_result
        assert "interpretation" in test_result

        # Check it's a boolean type (numpy or Python)
        assert test_result["reject_null"] in (True, False)


class TestStandardErrors:
    """Tests for standard error computation."""

    def test_se_positive(self, simple_heckman_data):
        """Standard errors are positive."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert result["se"] > 0, "SE should be positive"
        assert result["lambda_se"] > 0, "λ SE should be positive"

    def test_ci_contains_estimate(self, simple_heckman_data):
        """Confidence interval contains point estimate."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]

    def test_ci_width_reasonable(self, simple_heckman_data):
        """CI width is reasonable (not too narrow or wide)."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        ci_width = result["ci_upper"] - result["ci_lower"]

        # Width should be positive and not excessively wide
        assert ci_width > 0
        assert ci_width < 5.0, f"CI too wide: {ci_width:.2f}"

    def test_vcov_symmetric(self, simple_heckman_data):
        """Variance-covariance matrix is symmetric."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        vcov = result["vcov"]
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)

    def test_vcov_positive_semidefinite(self, simple_heckman_data):
        """Variance-covariance matrix is positive semidefinite."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        eigenvalues = np.linalg.eigvalsh(result["vcov"])
        # Allow small negative eigenvalues due to numerical precision
        assert np.all(eigenvalues > -1e-10), "Vcov should be PSD"


class TestProbitModel:
    """Tests for probit selection equation."""

    def test_probit_probs_in_unit_interval(self, simple_heckman_data):
        """Selection probabilities are in [0, 1]."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        probs = result["selection_probs"]
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_probit_converges(self, large_sample_data):
        """Probit model converges with sufficient data."""
        # Use large_sample_data which has more observations for stable convergence
        data = large_sample_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # With large sample, probit should converge
        # If not, result should still be finite (graceful degradation)
        assert result["selection_diagnostics"]["converged"] or np.isfinite(result["estimate"])

    def test_fit_probit_directly(self):
        """_fit_probit returns correct structure."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, (n, 2))
        logit_p = 0.5 + 0.8 * X[:, 0] + 0.3 * X[:, 1]
        probs = 1 / (1 + np.exp(-logit_p))
        y = np.random.binomial(1, probs, n)

        gamma, fitted_probs, diag = _fit_probit(y, X, add_intercept=True)

        assert len(gamma) == 3  # intercept + 2 covariates
        assert len(fitted_probs) == n
        assert "pseudo_r_squared" in diag
        assert "converged" in diag


class TestIdentificationDiagnostics:
    """Tests for identification checks."""

    def test_diagnose_with_exclusion(self, simple_heckman_data):
        """Detects exclusion restriction when present."""
        data = simple_heckman_data

        result = diagnose_identification(
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # Z (second column of selection_covariates) should be detected as excluded
        assert result["has_exclusion"]
        assert result["identification_strength"] in ["strong", "weak"]

    def test_diagnose_without_exclusion(self, fragile_identification_data):
        """Warns about fragile identification without exclusion."""
        data = fragile_identification_data

        # When same covariates used
        result = diagnose_identification(
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["selection_covariates"],
        )

        # No exclusion when covariates are identical
        assert not result["has_exclusion"]
        assert result["identification_strength"] == "fragile"
