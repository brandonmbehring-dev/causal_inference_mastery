"""
Cross-language validation: Python simple_ate vs Julia SimpleATE.

Validates that Python and Julia implementations produce identical results (rtol < 1e-10).
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators import simple_ate
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_simple_ate
)


pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-validation"
)


class TestPythonJuliaSimpleATE:
    """Cross-validate Python simple_ate against Julia SimpleATE."""

    def test_basic_case(self):
        """Basic 4-unit case."""
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        treatment = np.array([1, 1, 0, 0])

        python_result = simple_ate(outcomes, treatment)
        julia_result = julia_simple_ate(outcomes, treatment)

        # Validate estimate
        assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10), \
            f"Estimate mismatch: Python={python_result['estimate']}, Julia={julia_result['estimate']}"

        # Validate SE
        assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10), \
            f"SE mismatch: Python={python_result['se']}, Julia={julia_result['se']}"

        # Validate CI
        assert np.isclose(python_result["ci_lower"], julia_result["ci_lower"], rtol=1e-10)
        assert np.isclose(python_result["ci_upper"], julia_result["ci_upper"], rtol=1e-10)


    def test_balanced_rct(self):
        """Balanced RCT (n=100)."""
        np.random.seed(42)
        n = 100
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = np.where(treatment == 1,
                           np.random.normal(2.0, 1.0, n),
                           np.random.normal(0.0, 1.0, n))

        python_result = simple_ate(outcomes, treatment)
        julia_result = julia_simple_ate(outcomes, treatment)

        assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
        assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10)
        assert np.isclose(python_result["ci_lower"], julia_result["ci_lower"], rtol=1e-10)
        assert np.isclose(python_result["ci_upper"], julia_result["ci_upper"], rtol=1e-10)


    def test_small_sample(self):
        """Small sample (n=20) to test t-distribution."""
        np.random.seed(42)
        n = 20
        treatment = np.array([1] * 10 + [0] * 10)
        outcomes = np.where(treatment == 1,
                           np.random.normal(2.0, 1.0, n),
                           np.random.normal(0.0, 1.0, n))

        python_result = simple_ate(outcomes, treatment)
        julia_result = julia_simple_ate(outcomes, treatment)

        assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
        assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10)
        # t-distribution critical values should match
        assert np.isclose(python_result["ci_lower"], julia_result["ci_lower"], rtol=1e-10)
        assert np.isclose(python_result["ci_upper"], julia_result["ci_upper"], rtol=1e-10)


    def test_heteroskedastic(self):
        """Heteroskedastic errors (different variances)."""
        np.random.seed(42)
        n = 200
        treatment = np.array([1] * 100 + [0] * 100)
        outcomes = np.where(treatment == 1,
                           np.random.normal(2.0, 2.0, n),  # σ=2
                           np.random.normal(0.0, 1.0, n))  # σ=1

        python_result = simple_ate(outcomes, treatment)
        julia_result = julia_simple_ate(outcomes, treatment)

        # Neyman variance should match exactly
        assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
        assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10)


    def test_zero_effect(self):
        """Zero treatment effect."""
        np.random.seed(42)
        outcomes = np.random.normal(5.0, 1.0, 100)
        treatment = np.array([1] * 50 + [0] * 50)

        python_result = simple_ate(outcomes, treatment)
        julia_result = julia_simple_ate(outcomes, treatment)

        # ATE should be near zero
        assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
        assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10)


    def test_different_alpha_levels(self):
        """Test with different alpha levels (90%, 95%, 99% CI)."""
        outcomes = np.array([10.0, 8.0, 4.0, 2.0])
        treatment = np.array([1, 1, 0, 0])

        for alpha in [0.01, 0.05, 0.10]:
            python_result = simple_ate(outcomes, treatment, alpha=alpha)
            julia_result = julia_simple_ate(outcomes, treatment, alpha=alpha)

            # CI width should change with alpha, but results should match
            assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
            assert np.isclose(python_result["ci_lower"], julia_result["ci_lower"], rtol=1e-10)
            assert np.isclose(python_result["ci_upper"], julia_result["ci_upper"], rtol=1e-10)
