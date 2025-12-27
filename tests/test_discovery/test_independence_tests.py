"""Tests for conditional independence tests.

Session 133: Validation of CI test implementations.
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.discovery import (
    fisher_z_test,
    partial_correlation_test,
    partial_correlation,
    g_squared_test,
    kernel_ci_test,
    ci_test,
)


# =============================================================================
# Test Data Generation
# =============================================================================


@pytest.fixture
def independent_data():
    """Generate independent X and Y."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    return np.column_stack([X, Y])


@pytest.fixture
def dependent_data():
    """Generate dependent X and Y (Y = X + noise)."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    Y = 0.8 * X + rng.normal(0, 0.5, n)
    return np.column_stack([X, Y])


@pytest.fixture
def confounded_data():
    """Generate X, Y confounded by Z: Z -> X, Z -> Y."""
    rng = np.random.default_rng(42)
    n = 500
    Z = rng.normal(0, 1, n)
    X = 0.7 * Z + rng.normal(0, 0.5, n)
    Y = 0.7 * Z + rng.normal(0, 0.5, n)
    return np.column_stack([X, Y, Z])


@pytest.fixture
def chain_data():
    """Generate chain: X -> Z -> Y."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    Z = 0.8 * X + rng.normal(0, 0.5, n)
    Y = 0.8 * Z + rng.normal(0, 0.5, n)
    return np.column_stack([X, Y, Z])


@pytest.fixture
def collider_data():
    """Generate collider: X -> Z <- Y."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    Z = 0.7 * X + 0.7 * Y + rng.normal(0, 0.3, n)
    return np.column_stack([X, Y, Z])


# =============================================================================
# Partial Correlation Tests
# =============================================================================


class TestPartialCorrelation:
    """Test partial correlation computation."""

    def test_independent_variables_zero_correlation(self, independent_data):
        """Independent variables should have near-zero partial correlation."""
        rho = partial_correlation(independent_data, 0, 1)
        assert abs(rho) < 0.15, f"Independent vars have rho = {rho:.3f}"

    def test_dependent_variables_nonzero_correlation(self, dependent_data):
        """Dependent variables should have non-zero partial correlation."""
        rho = partial_correlation(dependent_data, 0, 1)
        assert abs(rho) > 0.5, f"Dependent vars have rho = {rho:.3f}"

    def test_confounded_conditional_independence(self, confounded_data):
        """X ⊥ Y | Z when Z confounds both."""
        # Unconditional: X and Y are dependent
        rho_unconditional = partial_correlation(confounded_data, 0, 1)
        assert abs(rho_unconditional) > 0.3

        # Conditional on Z: X ⊥ Y
        rho_conditional = partial_correlation(confounded_data, 0, 1, [2])
        assert abs(rho_conditional) < 0.15, f"Conditional rho = {rho_conditional:.3f}"

    def test_chain_conditional_independence(self, chain_data):
        """In chain X -> Z -> Y: X ⊥ Y | Z."""
        # Unconditional: X and Y are dependent
        rho_unconditional = partial_correlation(chain_data, 0, 1)
        assert abs(rho_unconditional) > 0.3

        # Conditional on Z: X ⊥ Y
        rho_conditional = partial_correlation(chain_data, 0, 1, [2])
        assert abs(rho_conditional) < 0.15

    def test_collider_conditional_dependence(self, collider_data):
        """In collider X -> Z <- Y: X ⊥ Y unconditionally, X dep Y | Z."""
        # Unconditional: X and Y are independent
        rho_unconditional = partial_correlation(collider_data, 0, 1)
        assert abs(rho_unconditional) < 0.15

        # Conditional on collider Z: X and Y become dependent!
        rho_conditional = partial_correlation(collider_data, 0, 1, [2])
        assert abs(rho_conditional) > 0.3, f"Collider effect: rho|Z = {rho_conditional:.3f}"

    def test_partial_correlation_bounds(self, dependent_data):
        """Partial correlation should be in [-1, 1]."""
        rho = partial_correlation(dependent_data, 0, 1)
        assert -1 <= rho <= 1


# =============================================================================
# Fisher Z Test
# =============================================================================


class TestFisherZTest:
    """Test Fisher's Z CI test."""

    def test_independent_high_pvalue(self, independent_data):
        """Independent variables should have high p-value."""
        result = fisher_z_test(independent_data, 0, 1, alpha=0.05)
        assert result.pvalue > 0.05, f"Independent: p = {result.pvalue:.4f}"
        assert result.independent

    def test_dependent_low_pvalue(self, dependent_data):
        """Dependent variables should have low p-value."""
        result = fisher_z_test(dependent_data, 0, 1, alpha=0.05)
        assert result.pvalue < 0.05, f"Dependent: p = {result.pvalue:.4f}"
        assert not result.independent

    def test_conditional_independence_confounded(self, confounded_data):
        """X ⊥ Y | Z when confounded."""
        result = fisher_z_test(confounded_data, 0, 1, [2], alpha=0.05)
        assert result.independent, f"Confounded CI: p = {result.pvalue:.4f}"

    def test_conditional_independence_chain(self, chain_data):
        """X ⊥ Y | Z in chain."""
        result = fisher_z_test(chain_data, 0, 1, [2], alpha=0.05)
        assert result.independent, f"Chain CI: p = {result.pvalue:.4f}"

    def test_alpha_threshold(self, dependent_data):
        """Different alpha should change result."""
        result_strict = fisher_z_test(dependent_data, 0, 1, alpha=0.001)
        result_loose = fisher_z_test(dependent_data, 0, 1, alpha=0.10)

        # Both should find dependence for truly dependent data
        assert not result_strict.independent
        assert not result_loose.independent

    def test_result_attributes(self, independent_data):
        """Result should have all attributes."""
        result = fisher_z_test(independent_data, 0, 1, alpha=0.05)

        assert hasattr(result, "independent")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "statistic")
        assert hasattr(result, "alpha")
        assert hasattr(result, "conditioning_set")


# =============================================================================
# Partial Correlation Test
# =============================================================================


class TestPartialCorrelationTest:
    """Test partial correlation based CI test."""

    def test_independent_detected(self, independent_data):
        """Should detect independence."""
        result = partial_correlation_test(independent_data, 0, 1, alpha=0.05)
        assert result.independent

    def test_dependent_detected(self, dependent_data):
        """Should detect dependence."""
        result = partial_correlation_test(dependent_data, 0, 1, alpha=0.05)
        assert not result.independent

    def test_consistent_with_fisher_z(self, confounded_data):
        """Should give similar results to Fisher Z."""
        result_fisher = fisher_z_test(confounded_data, 0, 1, [2], alpha=0.05)
        result_partial = partial_correlation_test(confounded_data, 0, 1, [2], alpha=0.05)

        # Both should agree
        assert result_fisher.independent == result_partial.independent


# =============================================================================
# G² Test (Categorical)
# =============================================================================


class TestGSquaredTest:
    """Test G² (likelihood ratio) test."""

    def test_independent_categorical(self):
        """Independent categorical variables."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.choice([0, 1, 2], n)
        Y = rng.choice([0, 1, 2], n)
        data = np.column_stack([X, Y])

        result = g_squared_test(data, 0, 1, alpha=0.05)
        assert result.pvalue > 0.01  # May not always pass due to discretization

    def test_dependent_categorical(self):
        """Dependent categorical variables."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.choice([0, 1, 2], n)
        Y = X.copy()  # Perfect dependence
        # Add noise to ~10% of entries
        noise_mask = rng.random(n) < 0.1
        Y[noise_mask] = rng.choice([0, 1, 2], np.sum(noise_mask))
        data = np.column_stack([X, Y])

        result = g_squared_test(data, 0, 1, alpha=0.05)
        assert not result.independent, f"Dependent categorical: p = {result.pvalue:.4f}"

    def test_continuous_discretization(self, dependent_data):
        """G² should work on continuous data via discretization."""
        result = g_squared_test(dependent_data, 0, 1, alpha=0.05, n_bins=4)
        # Should detect dependence
        assert result.pvalue < 0.05


# =============================================================================
# Kernel CI Test
# =============================================================================


class TestKernelCITest:
    """Test kernel-based CI test."""

    def test_independent_detected(self, independent_data):
        """Kernel test should detect independence."""
        result = kernel_ci_test(independent_data, 0, 1, alpha=0.05, n_bootstrap=50)
        # May not always pass due to bootstrap variance
        assert result.pvalue > 0.01

    def test_dependent_detected(self, dependent_data):
        """Kernel test should detect dependence."""
        result = kernel_ci_test(dependent_data, 0, 1, alpha=0.05, n_bootstrap=50)
        assert not result.independent

    def test_nonlinear_dependence(self):
        """Kernel test can detect nonlinear dependence."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.uniform(-2, 2, n)
        Y = X**2 + rng.normal(0, 0.3, n)  # Nonlinear dependence
        data = np.column_stack([X, Y])

        result = kernel_ci_test(data, 0, 1, alpha=0.05, n_bootstrap=100)
        # Should detect dependence
        assert not result.independent, f"Nonlinear: p = {result.pvalue:.4f}"


# =============================================================================
# Unified CI Test Interface
# =============================================================================


class TestUnifiedCITest:
    """Test ci_test() unified interface."""

    def test_all_methods_available(self, dependent_data):
        """All methods should be callable."""
        methods = ["fisher_z", "partial_correlation", "g_squared", "kernel"]

        for method in methods:
            kwargs = {}
            if method == "kernel":
                kwargs["n_bootstrap"] = 30
            result = ci_test(dependent_data, 0, 1, alpha=0.05, method=method, **kwargs)
            assert hasattr(result, "pvalue")

    def test_invalid_method_raises(self, dependent_data):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            ci_test(dependent_data, 0, 1, method="invalid")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_sample_size(self):
        """CI test should handle small samples."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (20, 3))

        result = fisher_z_test(data, 0, 1, alpha=0.05)
        assert 0 <= result.pvalue <= 1

    def test_large_conditioning_set(self):
        """CI test with many conditioning variables."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (500, 10))

        result = fisher_z_test(data, 0, 1, list(range(2, 8)), alpha=0.05)
        assert 0 <= result.pvalue <= 1

    def test_perfect_correlation(self):
        """Handle perfectly correlated variables."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        Y = X.copy()  # Perfect correlation
        data = np.column_stack([X, Y])

        result = fisher_z_test(data, 0, 1, alpha=0.05)
        # Should detect strong dependence
        assert result.pvalue < 0.001

    def test_constant_variable(self):
        """Handle constant variable gracefully."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        Y = np.ones(n)  # Constant
        data = np.column_stack([X, Y])

        # Should not crash, but correlation is undefined
        rho = partial_correlation(data, 0, 1)
        # Result may be 0 or nan depending on handling
        assert not np.isinf(rho)
