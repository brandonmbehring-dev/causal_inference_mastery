"""
Cross-Language Parity Tests: DML Continuous Treatment (Session 116)

Validates Python↔Julia parity for Double Machine Learning with continuous treatment.

Parity Requirements:
- ATE: rtol=0.01 (1%)
- SE: rtol=0.05 (5%)
- CATE: rtol=0.05 (5%)
"""

import numpy as np
import pytest

from causal_inference.cate import dml_continuous

from .julia_interface import is_julia_available, julia_dml_continuous


# Skip all tests if Julia is not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-language validation"
)


def generate_continuous_dgp(
    n: int = 500, p: int = 3, true_effect: float = 2.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for continuous treatment DML.

    Parameters
    ----------
    n : int
        Number of observations
    p : int
        Number of covariates
    true_effect : float
        True treatment effect dE[Y]/dD
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (Y, D, X) where Y is outcomes, D is continuous treatment, X is covariates
    """
    np.random.seed(random_state)
    X = np.random.randn(n, p)
    # Continuous treatment: D depends on first covariate + noise
    D = X[:, 0] + np.random.randn(n)
    # Outcome: includes confounding through X[:, 0]
    Y = 1.0 + X[:, 0] + true_effect * D + np.random.randn(n)
    return Y, D, X


class TestDMLContinuousParity:
    """Python ↔ Julia parity tests for DML Continuous."""

    def test_ate_parity(self):
        """ATE estimates match within rtol=0.01."""
        Y, D, X = generate_continuous_dgp(n=500, p=3, true_effect=2.0, random_state=42)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # ATE should match within 1%
        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.01), (
            f"ATE mismatch: Python={py_result.ate:.6f}, Julia={jl_result['ate']:.6f}"
        )

    def test_se_parity(self):
        """SE estimates match within rtol=0.05."""
        Y, D, X = generate_continuous_dgp(n=500, p=3, true_effect=2.0, random_state=42)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # SE should match within 5%
        assert np.isclose(py_result.ate_se, jl_result["ate_se"], rtol=0.05), (
            f"SE mismatch: Python={py_result.ate_se:.6f}, Julia={jl_result['ate_se']:.6f}"
        )

    def test_ci_parity(self):
        """Confidence intervals match within rtol=0.02."""
        Y, D, X = generate_continuous_dgp(n=500, p=3, true_effect=2.0, random_state=42)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # CI bounds should match within 2%
        assert np.isclose(py_result.ci_lower, jl_result["ci_lower"], rtol=0.02), (
            f"CI lower mismatch: Python={py_result.ci_lower:.6f}, Julia={jl_result['ci_lower']:.6f}"
        )
        assert np.isclose(py_result.ci_upper, jl_result["ci_upper"], rtol=0.02), (
            f"CI upper mismatch: Python={py_result.ci_upper:.6f}, Julia={jl_result['ci_upper']:.6f}"
        )

    def test_cate_parity(self):
        """CATE arrays match within rtol=0.05."""
        Y, D, X = generate_continuous_dgp(n=200, p=2, true_effect=2.0, random_state=123)

        py_result = dml_continuous(Y, D, X, n_folds=3, model="ols")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=3, model="ols")

        # CATE arrays should have same length
        assert len(py_result.cate) == len(jl_result["cate"])

        # CATE values should match within 5%
        # Use mean absolute difference relative to mean CATE
        py_cate_mean = np.abs(py_result.cate).mean()
        diff = np.abs(py_result.cate - jl_result["cate"]).mean()
        rel_diff = diff / py_cate_mean if py_cate_mean > 0 else diff

        assert rel_diff < 0.05, f"CATE mean relative difference {rel_diff:.4f} > 0.05"

    def test_diagnostics_parity(self):
        """Diagnostic fields (R², n, n_folds) match exactly."""
        Y, D, X = generate_continuous_dgp(n=300, p=3, true_effect=1.5, random_state=456)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # n and n_folds should match exactly
        assert py_result.n == jl_result["n"], (
            f"n mismatch: Python={py_result.n}, Julia={jl_result['n']}"
        )
        assert py_result.n_folds == jl_result["n_folds"], (
            f"n_folds mismatch: Python={py_result.n_folds}, Julia={jl_result['n_folds']}"
        )

        # R² should match within 5%
        assert np.isclose(py_result.outcome_r2, jl_result["outcome_r2"], rtol=0.05), (
            f"outcome_r2 mismatch: Python={py_result.outcome_r2:.4f}, Julia={jl_result['outcome_r2']:.4f}"
        )
        assert np.isclose(py_result.treatment_r2, jl_result["treatment_r2"], rtol=0.05), (
            f"treatment_r2 mismatch: Python={py_result.treatment_r2:.4f}, Julia={jl_result['treatment_r2']:.4f}"
        )


class TestDMLContinuousParityVariants:
    """Parity tests across different DML configurations."""

    def test_ols_model_parity(self):
        """OLS model parity."""
        Y, D, X = generate_continuous_dgp(n=400, p=2, true_effect=2.0, random_state=789)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="linear")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ols")

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.01)

    def test_ridge_model_parity(self):
        """Ridge model parity."""
        Y, D, X = generate_continuous_dgp(n=400, p=2, true_effect=2.0, random_state=101)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.01)

    def test_two_folds_parity(self):
        """Two-fold cross-fitting parity."""
        Y, D, X = generate_continuous_dgp(n=300, p=2, true_effect=2.0, random_state=202)

        py_result = dml_continuous(Y, D, X, n_folds=2, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=2, model="ridge")

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.02)  # Looser for fewer folds
        assert len(py_result.fold_estimates) == 2
        assert len(jl_result["fold_estimates"]) == 2

    def test_ten_folds_parity(self):
        """Ten-fold cross-fitting parity."""
        Y, D, X = generate_continuous_dgp(n=500, p=2, true_effect=2.0, random_state=303)

        py_result = dml_continuous(Y, D, X, n_folds=10, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=10, model="ridge")

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.01)
        assert len(py_result.fold_estimates) == 10
        assert len(jl_result["fold_estimates"]) == 10


class TestDMLContinuousParityEdgeCases:
    """Parity tests for edge cases."""

    def test_zero_effect_parity(self):
        """Zero effect parity."""
        np.random.seed(404)
        n = 400
        X = np.random.randn(n, 2)
        D = X[:, 0] + np.random.randn(n)
        Y = 1.0 + X[:, 0] + 0.0 * D + np.random.randn(n)  # True effect = 0

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # Both should be close to 0
        assert abs(py_result.ate) < 0.2
        assert abs(jl_result["ate"]) < 0.2
        # Should match each other
        assert np.isclose(py_result.ate, jl_result["ate"], atol=0.05)

    def test_negative_effect_parity(self):
        """Negative effect parity."""
        np.random.seed(505)
        n = 400
        X = np.random.randn(n, 2)
        D = X[:, 0] + np.random.randn(n)
        Y = 1.0 + X[:, 0] + (-1.5) * D + np.random.randn(n)  # Negative effect

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.01)

    def test_high_dimensional_parity(self):
        """High-dimensional covariates parity."""
        np.random.seed(606)
        n = 500
        p = 15  # High-dimensional
        X = np.random.randn(n, p)
        D = X[:, 0] + 0.5 * np.random.randn(n)
        Y = 1.0 + X[:, 0] + 2.0 * D + np.random.randn(n)

        py_result = dml_continuous(Y, D, X, n_folds=5, model="ridge")
        jl_result = julia_dml_continuous(Y, D, X, n_folds=5, model="ridge")

        # May have slightly larger tolerance for high-dim
        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.02)
