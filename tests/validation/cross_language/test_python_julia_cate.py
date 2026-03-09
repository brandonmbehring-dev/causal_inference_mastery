"""
Cross-language validation tests for CATE meta-learners.

Tests Python ↔ Julia parity for S, T, X, R-learner and Double ML.

Tolerance Strategy:
- ATE: rtol=0.15 (different propensity/outcome implementations)
- SE: rtol=0.30 (SE estimation varies by method)
- CATE correlation: r > 0.85 (individual effects should correlate)
"""

import numpy as np
import pytest

from src.causal_inference.cate import s_learner, t_learner, x_learner, r_learner, double_ml
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_s_learner,
    julia_t_learner,
    julia_x_learner,
    julia_r_learner,
    julia_double_ml,
)


pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-validation"
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_cate_dgp(
    n: int = 300,
    p: int = 2,
    true_ate: float = 2.0,
    effect_type: str = "constant",
    noise_sd: float = 1.0,
    seed: int = 42,
):
    """
    Generate data for CATE cross-language testing.

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of covariates
    true_ate : float
        True average treatment effect
    effect_type : str
        "constant" or "linear"
    noise_sd : float
        Standard deviation of noise
    seed : int
        Random seed

    Returns
    -------
    tuple
        (outcomes, treatment, covariates, true_cate)
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)
    T = (np.random.rand(n) > 0.5).astype(float)

    if effect_type == "constant":
        true_cate = np.full(n, true_ate)
    elif effect_type == "linear":
        true_cate = true_ate + X[:, 0]
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # DGP: Y = 1 + 0.5*X0 + tau(X)*T + eps
    Y = 1.0 + 0.5 * X[:, 0] + true_cate * T + noise_sd * np.random.randn(n)

    return Y, T, X, true_cate


# =============================================================================
# S-Learner Tests
# =============================================================================


class TestSLearnerParity:
    """Cross-language parity tests for S-Learner."""

    def test_s_learner_ate_parity(self):
        """S-Learner ATE estimates should be close."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = s_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_s_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate"], jl_result["ate"], rtol=0.15), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, Julia={jl_result['ate']:.4f}"
        )

    def test_s_learner_se_parity(self):
        """S-Learner SE estimates should be in same ballpark."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = s_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_s_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate_se"], jl_result["se"], rtol=0.30), (
            f"SE mismatch: Python={py_result['ate_se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_s_learner_cate_constant(self):
        """S-Learner with linear model produces constant CATE (expected behavior)."""
        Y, T, X, _ = generate_cate_dgp(n=400, effect_type="linear", seed=123)

        py_result = s_learner(Y, T, X, model="linear")
        jl_result = julia_s_learner(Y, T, X)

        # Linear S-learner produces constant CATE (treatment coefficient)
        # Both should have low variance in CATE
        py_cate_std = np.std(py_result["cate"])
        jl_cate_std = np.std(jl_result["cate"])

        # For constant CATE, std should be near 0
        assert py_cate_std < 0.01, f"Python CATE not constant: std={py_cate_std:.4f}"
        assert jl_cate_std < 0.01, f"Julia CATE not constant: std={jl_cate_std:.4f}"

        # Mean CATE should be close between implementations
        assert np.isclose(np.mean(py_result["cate"]), np.mean(jl_result["cate"]), rtol=0.15)


# =============================================================================
# T-Learner Tests
# =============================================================================


class TestTLearnerParity:
    """Cross-language parity tests for T-Learner."""

    def test_t_learner_ate_parity(self):
        """T-Learner ATE estimates should be close."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = t_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_t_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate"], jl_result["ate"], rtol=0.15), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, Julia={jl_result['ate']:.4f}"
        )

    def test_t_learner_se_parity(self):
        """T-Learner SE estimates should be in same ballpark."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = t_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_t_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate_se"], jl_result["se"], rtol=0.30), (
            f"SE mismatch: Python={py_result['ate_se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_t_learner_cate_correlation(self):
        """T-Learner CATE vectors should be highly correlated."""
        Y, T, X, _ = generate_cate_dgp(n=400, effect_type="linear", seed=123)

        py_result = t_learner(Y, T, X, model="linear")
        jl_result = julia_t_learner(Y, T, X)

        corr = np.corrcoef(py_result["cate"], jl_result["cate"])[0, 1]
        assert corr > 0.85, f"CATE correlation too low: {corr:.4f}"


# =============================================================================
# X-Learner Tests
# =============================================================================


class TestXLearnerParity:
    """Cross-language parity tests for X-Learner."""

    def test_x_learner_ate_parity(self):
        """X-Learner ATE estimates should be close."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        # Use linear model for fair comparison (Python default is random_forest)
        py_result = x_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_x_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate"], jl_result["ate"], rtol=0.15), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, Julia={jl_result['ate']:.4f}"
        )

    def test_x_learner_se_parity(self):
        """X-Learner SE estimates should be in same ballpark."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = x_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_x_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate_se"], jl_result["se"], rtol=0.30), (
            f"SE mismatch: Python={py_result['ate_se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_x_learner_cate_correlation(self):
        """X-Learner CATE vectors should be highly correlated."""
        Y, T, X, _ = generate_cate_dgp(n=400, effect_type="linear", seed=123)

        py_result = x_learner(Y, T, X, model="linear")
        jl_result = julia_x_learner(Y, T, X)

        corr = np.corrcoef(py_result["cate"], jl_result["cate"])[0, 1]
        assert corr > 0.85, f"CATE correlation too low: {corr:.4f}"


# =============================================================================
# R-Learner Tests
# =============================================================================


class TestRLearnerParity:
    """Cross-language parity tests for R-Learner."""

    def test_r_learner_ate_parity(self):
        """R-Learner ATE estimates should be close."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = r_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_r_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate"], jl_result["ate"], rtol=0.15), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, Julia={jl_result['ate']:.4f}"
        )

    def test_r_learner_se_parity(self):
        """R-Learner SE estimates should be in same ballpark."""
        Y, T, X, _ = generate_cate_dgp(n=300, true_ate=2.0, seed=42)

        py_result = r_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_r_learner(Y, T, X, alpha=0.05)

        assert np.isclose(py_result["ate_se"], jl_result["se"], rtol=0.30), (
            f"SE mismatch: Python={py_result['ate_se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_r_learner_ci_coverage(self):
        """Both R-Learner implementations should have CI containing true ATE."""
        Y, T, X, _ = generate_cate_dgp(n=400, true_ate=2.0, seed=999)

        py_result = r_learner(Y, T, X, model="linear", alpha=0.05)
        jl_result = julia_r_learner(Y, T, X, alpha=0.05)

        # Both CIs should contain true ATE
        assert py_result["ci_lower"] < 2.0 < py_result["ci_upper"], (
            f"Python CI [{py_result['ci_lower']:.2f}, {py_result['ci_upper']:.2f}] doesn't contain 2.0"
        )
        assert jl_result["ci_lower"] < 2.0 < jl_result["ci_upper"], (
            f"Julia CI [{jl_result['ci_lower']:.2f}, {jl_result['ci_upper']:.2f}] doesn't contain 2.0"
        )


# =============================================================================
# Double ML Tests
# =============================================================================


class TestDoubleMachineLearningParity:
    """Cross-language parity tests for Double Machine Learning."""

    def test_dml_ate_parity(self):
        """DML ATE estimates should be close."""
        Y, T, X, _ = generate_cate_dgp(n=400, true_ate=2.0, seed=42)

        py_result = double_ml(Y, T, X, n_folds=5, model="linear", alpha=0.05)
        jl_result = julia_double_ml(Y, T, X, n_folds=5, alpha=0.05)

        # DML has more randomness from cross-fitting, use wider tolerance
        assert np.isclose(py_result["ate"], jl_result["ate"], rtol=0.20), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, Julia={jl_result['ate']:.4f}"
        )

    def test_dml_se_parity(self):
        """DML SE estimates should be in same ballpark."""
        Y, T, X, _ = generate_cate_dgp(n=400, true_ate=2.0, seed=42)

        py_result = double_ml(Y, T, X, n_folds=5, model="linear", alpha=0.05)
        jl_result = julia_double_ml(Y, T, X, n_folds=5, alpha=0.05)

        # SE can vary more due to cross-fitting randomness
        assert np.isclose(py_result["ate_se"], jl_result["se"], rtol=0.40), (
            f"SE mismatch: Python={py_result['ate_se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_dml_ci_coverage(self):
        """Both DML implementations should have CI containing true ATE."""
        Y, T, X, _ = generate_cate_dgp(n=500, true_ate=2.0, seed=888)

        py_result = double_ml(Y, T, X, n_folds=5, model="linear", alpha=0.05)
        jl_result = julia_double_ml(Y, T, X, n_folds=5, alpha=0.05)

        # Both CIs should contain true ATE
        assert py_result["ci_lower"] < 2.0 < py_result["ci_upper"], (
            f"Python CI [{py_result['ci_lower']:.2f}, {py_result['ci_upper']:.2f}] doesn't contain 2.0"
        )
        assert jl_result["ci_lower"] < 2.0 < jl_result["ci_upper"], (
            f"Julia CI [{jl_result['ci_lower']:.2f}, {jl_result['ci_upper']:.2f}] doesn't contain 2.0"
        )
