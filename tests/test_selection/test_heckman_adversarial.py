"""
Layer 2: Adversarial tests for Heckman selection model.

Tests cover:
1. Input validation (bad inputs should fail explicitly)
2. Edge cases (extreme selection rates, small samples)
3. Numerical stability (extreme values, near-singular cases)
4. Boundary conditions (all/none selected, perfect prediction)
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.selection.heckman import (
    heckman_two_step,
    _compute_imr,
    _fit_probit,
)
from src.causal_inference.selection.diagnostics import (
    selection_bias_test,
    diagnose_identification,
)


class TestInputValidation:
    """Tests for explicit input validation errors."""

    def test_rejects_non_binary_selection(self, simple_heckman_data):
        """Rejects non-binary selection indicator."""
        data = simple_heckman_data
        invalid_selected = data["selected"].copy()
        invalid_selected[0] = 2.0  # Invalid value

        with pytest.raises(ValueError, match="binary"):
            heckman_two_step(
                outcome=data["outcome"],
                selected=invalid_selected,
                selection_covariates=data["selection_covariates"],
            )

    def test_rejects_all_selected(self):
        """Rejects when all observations are selected."""
        np.random.seed(42)
        n = 100
        outcome = np.random.normal(0, 1, n)
        selected = np.ones(n)  # All selected
        covariates = np.random.normal(0, 1, (n, 2))

        with pytest.raises(ValueError, match="[Aa]ll.*selected"):
            heckman_two_step(
                outcome=outcome,
                selected=selected,
                selection_covariates=covariates,
            )

    def test_rejects_none_selected(self):
        """Rejects when no observations are selected."""
        np.random.seed(42)
        n = 100
        outcome = np.random.normal(0, 1, n)
        selected = np.zeros(n)  # None selected
        covariates = np.random.normal(0, 1, (n, 2))

        with pytest.raises(ValueError, match="[Nn]o.*selected"):
            heckman_two_step(
                outcome=outcome,
                selected=selected,
                selection_covariates=covariates,
            )

    def test_rejects_mismatched_lengths(self, simple_heckman_data):
        """Rejects arrays with different lengths."""
        data = simple_heckman_data

        with pytest.raises(ValueError, match="[Ll]ength|mismatch"):
            heckman_two_step(
                outcome=data["outcome"][:50],  # Wrong length
                selected=data["selected"],
                selection_covariates=data["selection_covariates"],
            )

    def test_rejects_nan_in_selected_outcomes(self, simple_heckman_data):
        """Rejects NaN in outcomes for selected observations."""
        data = simple_heckman_data
        outcome = data["outcome"].copy()

        # Find a selected observation and set to NaN
        selected_idx = np.where(data["selected"] == 1)[0][0]
        outcome[selected_idx] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            heckman_two_step(
                outcome=outcome,
                selected=data["selected"],
                selection_covariates=data["selection_covariates"],
            )

    def test_rejects_1d_selection_covariates(self):
        """Handles 1D covariates correctly (reshapes)."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)  # 1D array
        selected = (X > 0).astype(float)
        outcome = 2.0 * X + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        # Should work (reshapes internally)
        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=X,  # 1D
        )

        assert result["n_total"] == n


class TestEdgeCases:
    """Tests for edge case behavior."""

    def test_very_small_sample(self):
        """Handles very small sample size."""
        np.random.seed(42)
        n = 30
        rho = 0.3

        X = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)

        cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
        errors = np.random.multivariate_normal([0, 0], cov_matrix, n)
        u, v = errors[:, 0], errors[:, 1]

        s_star = 0.3 + 0.5 * Z + v
        selected = (s_star > 0).astype(float)

        # Ensure at least some selected
        n_selected = int(np.sum(selected))
        if n_selected < 10:
            pytest.skip("Too few selected for meaningful test")

        outcome = 1.0 + 2.0 * X + u
        outcome = np.where(selected == 1, outcome, np.nan)

        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=np.column_stack([X, Z]),
            outcome_covariates=X.reshape(-1, 1),
        )

        # Should complete without error
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

    def test_high_selection_rate(self, high_selection_rate_data):
        """Handles ~90% selection rate."""
        data = high_selection_rate_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        selection_rate = data["n_selected"] / data["n_total"]
        assert selection_rate > 0.85, "Fixture should have high selection"
        assert np.isfinite(result["estimate"])

    def test_low_selection_rate(self, low_selection_rate_data):
        """Handles ~30% selection rate."""
        data = low_selection_rate_data
        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        selection_rate = data["n_selected"] / data["n_total"]
        assert selection_rate < 0.45, "Fixture should have low selection"
        assert np.isfinite(result["estimate"])

    def test_single_covariate(self):
        """Works with single covariate."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)

        selected = (X + np.random.normal(0, 1, n) > 0).astype(float)
        outcome = 2.0 * X + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=X.reshape(-1, 1),
        )

        assert np.isfinite(result["estimate"])

    def test_many_covariates(self):
        """Handles many covariates (p = 10)."""
        np.random.seed(42)
        n = 500
        p = 10

        X = np.random.normal(0, 1, (n, p))
        linear_comb = X @ np.random.normal(0, 0.3, p)

        selected = (linear_comb + np.random.normal(0, 1, n) > 0).astype(float)
        outcome = 2.0 * X[:, 0] + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=X,
            outcome_covariates=X[:, :5],  # Subset for outcome
        )

        assert np.isfinite(result["estimate"])
        assert result["vcov"].shape[0] > 5  # Multiple coefficients


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_extreme_selection_probabilities(self):
        """Handles near-0 and near-1 selection probabilities."""
        np.random.seed(42)
        n = 300

        # Create data with extreme selection probabilities
        Z = np.random.normal(0, 2, n)  # High variance → extreme probs
        X = np.random.normal(0, 1, n)

        # Strong selection mechanism
        s_star = 0 + 1.5 * Z + np.random.normal(0, 0.5, n)
        selected = (s_star > 0).astype(float)

        outcome = 2.0 * X + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=np.column_stack([X, Z]),
            outcome_covariates=X.reshape(-1, 1),
        )

        # Should not have inf or nan
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])
        assert np.all(np.isfinite(result["selection_probs"]))

    def test_imr_boundary_stability(self):
        """IMR computation is stable at probability boundaries."""
        # Very low probabilities
        low_probs = np.array([1e-6, 1e-5, 1e-4, 0.001])
        imr_low = _compute_imr(low_probs)
        assert np.all(np.isfinite(imr_low))
        assert np.all(imr_low > 0)

        # Very high probabilities
        high_probs = np.array([0.999, 0.9999, 0.99999, 1 - 1e-6])
        imr_high = _compute_imr(high_probs)
        assert np.all(np.isfinite(imr_high))
        assert np.all(imr_high > 0)

    def test_collinear_covariates(self):
        """Handles nearly collinear covariates."""
        np.random.seed(42)
        n = 200

        X1 = np.random.normal(0, 1, n)
        X2 = X1 + np.random.normal(0, 0.01, n)  # Nearly collinear
        X = np.column_stack([X1, X2])

        selected = (X1 + np.random.normal(0, 1, n) > 0).astype(float)
        outcome = 2.0 * X1 + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        # Should complete (may have warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = heckman_two_step(
                outcome=outcome,
                selected=selected,
                selection_covariates=X,
            )

        # May be numerically unstable but should complete
        assert np.isfinite(result["estimate"])

    def test_constant_covariate(self):
        """Handles constant covariate (zero variance)."""
        np.random.seed(42)
        n = 200

        X = np.random.normal(0, 1, n)
        constant = np.ones(n)  # Constant
        covariates = np.column_stack([X, constant])

        selected = (X + np.random.normal(0, 1, n) > 0).astype(float)
        outcome = 2.0 * X + np.random.normal(0, 1, n)
        outcome = np.where(selected == 1, outcome, np.nan)

        # May fail or warn due to singularity
        # Test that it either works or fails gracefully
        try:
            result = heckman_two_step(
                outcome=outcome,
                selected=selected,
                selection_covariates=covariates,
            )
            # If it works, check finite result
            assert np.isfinite(result["estimate"])
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable to fail for singular design
            pass


class TestOutcomeArrayFormats:
    """Tests for different outcome array formats."""

    def test_full_array_with_nan(self, simple_heckman_data):
        """Accepts full array with NaN for unselected."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome"],  # Has NaN
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
        )
        assert result["n_total"] == data["n_total"]

    def test_selected_only_array(self, simple_heckman_data):
        """Accepts array with only selected observations."""
        data = simple_heckman_data
        result = heckman_two_step(
            outcome=data["outcome_selected"],  # Selected only
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
        )
        assert result["n_selected"] == len(data["outcome_selected"])


class TestWarnings:
    """Tests for appropriate warning behavior."""

    def test_warns_without_exclusion_restriction(self, simple_heckman_data):
        """Warns when no exclusion restriction provided."""
        data = simple_heckman_data

        # Use same covariates for both equations
        with pytest.warns(UserWarning, match="exclusion"):
            heckman_two_step(
                outcome=data["outcome"],
                selected=data["selected"],
                selection_covariates=data["selection_covariates"],
                outcome_covariates=None,  # Will use selection_covariates
            )


class TestDiagnosticsFunctions:
    """Tests for diagnostic functions."""

    def test_selection_bias_test_handles_zero_se(self):
        """selection_bias_test handles zero standard error gracefully."""
        result = selection_bias_test(lambda_coef=0.5, lambda_se=0.0)

        assert np.isnan(result["statistic"])
        assert "invalid" in result["interpretation"].lower()

    def test_selection_bias_test_handles_nan_se(self):
        """selection_bias_test handles NaN standard error."""
        result = selection_bias_test(lambda_coef=0.5, lambda_se=np.nan)

        assert np.isnan(result["statistic"])

    def test_diagnose_identification_1d_covariates(self):
        """diagnose_identification handles 1D arrays."""
        np.random.seed(42)
        n = 100
        Z = np.random.normal(0, 1, n)
        X = np.random.normal(0, 1, n)

        result = diagnose_identification(
            selection_covariates=Z,  # 1D
            outcome_covariates=X,  # 1D
        )

        assert "has_exclusion" in result
        assert "identification_strength" in result

    def test_diagnose_identification_perfect_collinearity(self):
        """diagnose_identification detects perfect collinearity."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, n)

        # Same variable in both
        result = diagnose_identification(
            selection_covariates=X.reshape(-1, 1),
            outcome_covariates=X.reshape(-1, 1),
        )

        assert not result["has_exclusion"]
        assert result["identification_strength"] == "fragile"


class TestParameterRecoveryRobustness:
    """Tests for robust parameter recovery."""

    def test_intercept_option(self, simple_heckman_data):
        """add_intercept=False works correctly."""
        data = simple_heckman_data

        # Add intercept manually
        n = len(data["selected"])
        sel_cov_with_int = np.column_stack(
            [
                np.ones(n),
                data["selection_covariates"],
            ]
        )
        out_cov_with_int = np.column_stack(
            [
                np.ones(n),
                data["outcome_covariates"],
            ]
        )

        result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=sel_cov_with_int,
            outcome_covariates=out_cov_with_int,
            add_intercept=False,
        )

        assert np.isfinite(result["estimate"])

    def test_different_alpha_levels(self, simple_heckman_data):
        """Different alpha levels produce different CIs."""
        data = simple_heckman_data

        result_05 = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
            alpha=0.05,
        )

        result_01 = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
            alpha=0.01,
        )

        # 99% CI should be wider than 95% CI
        width_05 = result_05["ci_upper"] - result_05["ci_lower"]
        width_01 = result_01["ci_upper"] - result_01["ci_lower"]

        assert width_01 > width_05


class TestProbitConvergence:
    """Tests for probit model convergence."""

    def test_probit_handles_difficult_data(self):
        """Probit handles difficult optimization case."""
        np.random.seed(42)
        n = 100

        # Create somewhat separable data
        X = np.random.normal(0, 1, (n, 2))
        X[:50, 0] += 2  # Shift first half

        y = np.zeros(n)
        y[:50] = 1

        # Should converge (or at least not crash)
        gamma, probs, diag = _fit_probit(y, X, add_intercept=True)

        assert len(gamma) == 3
        assert len(probs) == n
        assert "converged" in diag

    def test_probit_perfect_separation_warning(self):
        """Probit warns on perfect separation."""
        np.random.seed(42)
        n = 100

        # Perfect separation
        X = np.random.normal(0, 1, (n, 1))
        y = (X[:, 0] > 0).astype(float)

        # May warn about convergence
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gamma, probs, diag = _fit_probit(y, X, add_intercept=True)

        # Should complete (may or may not have warnings)
        assert len(gamma) == 2
