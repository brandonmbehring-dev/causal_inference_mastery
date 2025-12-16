"""
Tests for SCM Diagnostics

Tests pre-treatment fit, covariate balance, and weight diagnostics.
"""

import numpy as np
import pytest

from causal_inference.scm.diagnostics import (
    check_pre_treatment_fit,
    check_covariate_balance,
    check_weight_properties,
    diagnose_scm_quality,
    compute_rmspe_ratio,
)


class TestPreTreatmentFit:
    """Tests for pre-treatment fit diagnostics."""

    def test_perfect_fit(self):
        """Perfect fit should have RMSE=0, R²=1."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        synthetic = treated.copy()

        result = check_pre_treatment_fit(treated, synthetic)

        assert np.isclose(result["rmse"], 0.0, atol=1e-10)
        assert np.isclose(result["r_squared"], 1.0, atol=1e-10)
        assert result["fit_quality"] == "excellent"

    def test_poor_fit(self):
        """Poor fit should have low R²."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        synthetic = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Inverted

        result = check_pre_treatment_fit(treated, synthetic)

        assert result["r_squared"] < 0
        assert result["fit_quality"] == "poor"

    def test_fit_metrics(self):
        """All fit metrics should be present."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        synthetic = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        result = check_pre_treatment_fit(treated, synthetic)

        assert "rmse" in result
        assert "r_squared" in result
        assert "mape" in result
        assert "max_gap" in result
        assert "mean_gap" in result
        assert "fit_quality" in result

    def test_fit_quality_categories(self):
        """Test different fit quality categories."""
        treated = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Excellent: R² > 0.99
        synthetic_excellent = treated + np.random.randn(5) * 0.1
        result = check_pre_treatment_fit(treated, synthetic_excellent)
        assert result["fit_quality"] in ["excellent", "good"]

        # Poor: large deviation
        synthetic_poor = treated + 20
        result = check_pre_treatment_fit(treated, synthetic_poor)
        assert result["fit_quality"] == "poor"


class TestCovariateBalance:
    """Tests for covariate balance checks."""

    def test_perfect_balance(self):
        """Identical covariates should show perfect balance."""
        treated_cov = np.array([1.0, 2.0, 3.0])
        synthetic_cov = treated_cov.copy()

        result = check_covariate_balance(treated_cov, synthetic_cov)

        for name, metrics in result.items():
            assert np.isclose(metrics["difference"], 0.0)
            assert metrics["balanced"]  # Use truthy check for numpy bool

    def test_imbalance_detection(self):
        """Should detect imbalance (>10% difference)."""
        treated_cov = np.array([10.0, 20.0])
        synthetic_cov = np.array([12.0, 20.0])  # 20% difference in first

        result = check_covariate_balance(treated_cov, synthetic_cov)

        assert not result["X0"]["balanced"]  # 20% diff
        assert result["X1"]["balanced"]  # 0% diff

    def test_custom_names(self):
        """Should use custom covariate names."""
        treated_cov = np.array([1.0, 2.0])
        synthetic_cov = np.array([1.0, 2.0])

        result = check_covariate_balance(
            treated_cov, synthetic_cov,
            covariate_names=["GDP", "Population"]
        )

        assert "GDP" in result
        assert "Population" in result

    def test_dimension_mismatch(self):
        """Should raise error on dimension mismatch."""
        with pytest.raises(ValueError, match="mismatch"):
            check_covariate_balance(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            )


class TestWeightProperties:
    """Tests for weight diagnostics."""

    def test_concentrated_weights(self):
        """Should detect weight concentration."""
        weights = np.array([0.95, 0.05, 0.0, 0.0, 0.0])

        result = check_weight_properties(weights)

        assert result["max_weight"] == 0.95
        assert result["n_nonzero"] == 2
        assert result["sparsity"] == 0.6  # 3/5 are zero

    def test_uniform_weights(self):
        """Uniform weights should have high effective N."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = check_weight_properties(weights)

        assert np.isclose(result["effective_n"], 5.0)
        assert np.isclose(result["hhi"], 0.2)  # 5 * 0.04 = 0.2

    def test_top_donors(self):
        """Should list top donors correctly."""
        weights = np.array([0.5, 0.3, 0.1, 0.1, 0.0])

        result = check_weight_properties(
            weights,
            control_labels=["A", "B", "C", "D", "E"]
        )

        # Top donor should be "A" with 0.5
        assert result["top_donors"][0] == ("A", 0.5)
        assert result["top_donors"][1] == ("B", 0.3)


class TestDiagnoseQuality:
    """Tests for comprehensive quality diagnostics."""

    def test_high_quality(self):
        """Good fit + balanced weights = high quality."""
        pre_fit = {"r_squared": 0.98, "rmse": 0.1, "mape": 1.0}
        weight_props = {"max_weight": 0.4, "effective_n": 3.5, "sparsity": 0.3}

        result = diagnose_scm_quality(pre_fit, weight_props, n_pre_periods=10)

        assert result["overall_quality"] in ["high", "medium"]
        assert len(result["warnings"]) == 0 or len(result["warnings"]) == 1

    def test_low_quality_poor_fit(self):
        """Poor fit should trigger warnings."""
        pre_fit = {"r_squared": 0.5, "rmse": 2.0, "mape": 10.0}
        weight_props = {"max_weight": 0.3, "effective_n": 4.0, "sparsity": 0.4}

        result = diagnose_scm_quality(pre_fit, weight_props, n_pre_periods=10)

        assert result["overall_quality"] == "low"
        assert len(result["warnings"]) > 0
        assert any("pre-treatment fit" in w.lower() for w in result["warnings"])

    def test_few_pre_periods_warning(self):
        """Few pre-periods should trigger warning."""
        pre_fit = {"r_squared": 0.95, "rmse": 0.2, "mape": 2.0}
        weight_props = {"max_weight": 0.5, "effective_n": 2.5, "sparsity": 0.3}

        result = diagnose_scm_quality(pre_fit, weight_props, n_pre_periods=3)

        assert any("pre-treatment periods" in w.lower() for w in result["warnings"])

    def test_concentrated_weight_warning(self):
        """High weight concentration should trigger warning."""
        pre_fit = {"r_squared": 0.95, "rmse": 0.2, "mape": 2.0}
        weight_props = {"max_weight": 0.95, "effective_n": 1.1, "sparsity": 0.8}

        result = diagnose_scm_quality(pre_fit, weight_props, n_pre_periods=10)

        assert any("concentration" in w.lower() for w in result["warnings"])


class TestRMSPERatio:
    """Tests for RMSPE ratio computation."""

    def test_rmspe_ratio(self):
        """Should compute pre/post RMSPE and ratio."""
        gap = np.array([0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 2.0])
        treatment_period = 4

        pre_rmspe, post_rmspe, ratio = compute_rmspe_ratio(gap, treatment_period)

        assert np.isclose(pre_rmspe, 0.1)
        assert np.isclose(post_rmspe, 2.0)
        assert np.isclose(ratio, 20.0)

    def test_zero_pre_rmspe(self):
        """Zero pre-RMSPE should give inf ratio."""
        gap = np.array([0, 0, 0, 0, 1.0, 1.0])
        treatment_period = 4

        pre_rmspe, post_rmspe, ratio = compute_rmspe_ratio(gap, treatment_period)

        assert ratio == np.inf or ratio > 1e6
