"""Triangulation tests: Python RCT estimators vs R reference implementations.

This module provides Layer 5 validation by comparing our Python RCT implementations
against R implementations using base R, sandwich (HC3), and coin packages.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established based on implementation differences):
- ATE estimate: rtol=0.05 (5% relative tolerance)
- Standard error: rtol=0.15 (15% relative, accounts for HC3 vs Neyman variance)
- CI bounds: rtol=0.10 (10% relative tolerance)

Run with: pytest tests/validation/r_triangulation/test_rct_vs_r.py -v

References:
- Imbens & Rubin (2015). Causal Inference for Statistics, Social, and Biomedical Sciences
- Long & Ervin (2000). Using heteroscedasticity consistent standard errors
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_rct_r_packages_installed,
    r_ipw_ate,
    r_permutation_test,
    r_regression_ate,
    r_simple_ate,
    r_stratified_ate,
)

# Lazy import Python implementations
try:
    from src.causal_inference.rct.simple_ate import simple_ate
    from src.causal_inference.rct.stratified_ate import stratified_ate
    from src.causal_inference.rct.regression_adjusted import regression_adjusted_ate
    from src.causal_inference.rct.permutation_test import permutation_test
    from src.causal_inference.observational.ipw import ipw_ate

    RCT_AVAILABLE = True
except ImportError:
    RCT_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_rct_python = pytest.mark.skipif(
    not RCT_AVAILABLE,
    reason="Python RCT module not available",
)

requires_rct_r_packages = pytest.mark.skipif(
    not check_rct_r_packages_installed(),
    reason="R packages 'sandwich' and 'coin' not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_rct_data(
    n: int = 100,
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    seed: int = 42,
    balanced: bool = True,
) -> dict:
    """Generate data from a simple RCT DGP.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.
    balanced : bool
        If True, equal treatment/control. If False, 30/70 split.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, true_ate arrays.
    """
    np.random.seed(seed)

    if balanced:
        treatment = np.array([1] * (n // 2) + [0] * (n - n // 2))
        np.random.shuffle(treatment)
    else:
        treatment = np.random.binomial(1, 0.3, n)

    # Simple linear DGP: Y = tau * T + epsilon
    outcomes = true_ate * treatment + noise_sd * np.random.randn(n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "true_ate": true_ate,
        "n": n,
    }


def generate_stratified_rct_data(
    n: int = 100,
    n_strata: int = 5,
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from a stratified RCT DGP.

    Parameters
    ----------
    n : int
        Sample size.
    n_strata : int
        Number of strata.
    true_ate : float
        True average treatment effect (constant across strata).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, strata, true_ate.
    """
    np.random.seed(seed)

    # Assign strata
    strata = np.repeat(np.arange(1, n_strata + 1), n // n_strata)
    if len(strata) < n:
        strata = np.concatenate([strata, np.array([n_strata] * (n - len(strata)))])

    # Randomize within strata
    treatment = np.zeros(n, dtype=int)
    for s in range(1, n_strata + 1):
        mask = strata == s
        n_s = mask.sum()
        t_s = np.array([1] * (n_s // 2) + [0] * (n_s - n_s // 2))
        np.random.shuffle(t_s)
        treatment[mask] = t_s

    # Stratum-specific baselines
    stratum_effects = np.random.randn(n_strata) * 2
    baseline = stratum_effects[strata - 1]

    # Outcome
    outcomes = baseline + true_ate * treatment + noise_sd * np.random.randn(n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "strata": strata,
        "true_ate": true_ate,
        "n": n,
    }


def generate_regression_rct_data(
    n: int = 100,
    true_ate: float = 2.0,
    covariate_effect: float = 0.5,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from an RCT with covariate adjustment.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    covariate_effect : float
        Effect of covariate on outcome.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    treatment = np.array([1] * (n // 2) + [0] * (n - n // 2))
    np.random.shuffle(treatment)

    # Covariate (prognostic, not confounding in RCT)
    covariate = np.random.randn(n)

    # Outcome: Y = tau * T + beta * X + epsilon
    outcomes = true_ate * treatment + covariate_effect * covariate + noise_sd * np.random.randn(n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariate.reshape(-1, 1),
        "true_ate": true_ate,
        "n": n,
    }


def generate_ipw_data(
    n: int = 100,
    true_ate: float = 2.0,
    propensity_range: tuple = (0.3, 0.7),
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data with known propensity scores for IPW.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    propensity_range : tuple
        Range of propensity scores (min, max).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, propensity, true_ate.
    """
    np.random.seed(seed)

    # Known propensity scores
    propensity = np.random.uniform(propensity_range[0], propensity_range[1], n)

    # Treatment assignment based on propensity
    treatment = np.random.binomial(1, propensity)

    # Outcome
    outcomes = true_ate * treatment + noise_sd * np.random.randn(n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "propensity": propensity,
        "true_ate": true_ate,
        "n": n,
    }


# =============================================================================
# Test Classes
# =============================================================================


@requires_rct_python
class TestSimpleATEVsR:
    """Compare Python simple_ate() to R's Welch t-test."""

    def test_balanced_design(self):
        """Python and R should agree on balanced RCT with rtol=0.05."""
        data = generate_rct_data(n=100, true_ate=2.0, seed=42, balanced=True)

        # Python result
        py_result = simple_ate(data["outcomes"], data["treatment"])

        # R result
        r_result = r_simple_ate(data["outcomes"], data["treatment"])

        if r_result is None:
            pytest.skip("R simple_ate unavailable")

        # Compare estimates
        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

        # Compare SE (may differ due to variance formula)
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.15), (
            f"SE mismatch: Python={py_result['se']:.4f}, R={r_result['se']:.4f}"
        )

    def test_unbalanced_design(self):
        """Python and R should agree on unbalanced RCT."""
        data = generate_rct_data(n=100, true_ate=3.0, seed=123, balanced=False)

        py_result = simple_ate(data["outcomes"], data["treatment"])
        r_result = r_simple_ate(data["outcomes"], data["treatment"])

        if r_result is None:
            pytest.skip("R simple_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

    def test_small_sample(self):
        """Python and R should agree on small sample (n=20)."""
        data = generate_rct_data(n=20, true_ate=5.0, seed=999, balanced=True)

        py_result = simple_ate(data["outcomes"], data["treatment"])
        r_result = r_simple_ate(data["outcomes"], data["treatment"])

        if r_result is None:
            pytest.skip("R simple_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )


@requires_rct_python
class TestStratifiedATEVsR:
    """Compare Python stratified_ate() to R precision-weighted estimator."""

    def test_five_strata(self):
        """Python and R should agree with 5 strata."""
        data = generate_stratified_rct_data(n=100, n_strata=5, true_ate=2.0, seed=42)

        py_result = stratified_ate(data["outcomes"], data["treatment"], data["strata"])
        r_result = r_stratified_ate(data["outcomes"], data["treatment"], data["strata"])

        if r_result is None:
            pytest.skip("R stratified_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

    def test_two_strata(self):
        """Python and R should agree with 2 strata (binary stratification)."""
        data = generate_stratified_rct_data(n=100, n_strata=2, true_ate=1.5, seed=123)

        py_result = stratified_ate(data["outcomes"], data["treatment"], data["strata"])
        r_result = r_stratified_ate(data["outcomes"], data["treatment"], data["strata"])

        if r_result is None:
            pytest.skip("R stratified_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )


@requires_rct_python
@requires_rct_r_packages
class TestRegressionATEVsR:
    """Compare Python regression_adjusted_ate() to R lm() + HC3."""

    def test_single_covariate(self):
        """Python and R should agree with single covariate adjustment."""
        data = generate_regression_rct_data(n=100, true_ate=2.0, seed=42)

        py_result = regression_adjusted_ate(data["outcomes"], data["treatment"], data["covariates"])
        r_result = r_regression_ate(data["outcomes"], data["treatment"], data["covariates"])

        if r_result is None:
            pytest.skip("R regression_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

        # SE comparison (HC3 implementations may vary slightly)
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.15), (
            f"SE mismatch: Python={py_result['se']:.4f}, R={r_result['se']:.4f}"
        )

    def test_multiple_covariates(self):
        """Python and R should agree with multiple covariates."""
        np.random.seed(42)
        n = 100
        treatment = np.array([1] * 50 + [0] * 50)
        np.random.shuffle(treatment)
        covariates = np.random.randn(n, 3)  # 3 covariates
        outcomes = 2.0 * treatment + covariates @ np.array([0.5, 0.3, 0.2]) + np.random.randn(n)

        py_result = regression_adjusted_ate(outcomes, treatment, covariates)
        r_result = r_regression_ate(outcomes, treatment, covariates)

        if r_result is None:
            pytest.skip("R regression_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )


@requires_rct_python
@requires_rct_r_packages
class TestPermutationTestVsR:
    """Compare Python permutation_test() to R coin package."""

    def test_small_sample_pvalue(self):
        """P-values should be similar (Monte Carlo variability expected)."""
        # Small sample for faster permutation
        np.random.seed(42)
        outcomes = np.array([10, 12, 11, 4, 5, 3], dtype=float)
        treatment = np.array([1, 1, 1, 0, 0, 0])

        py_result = permutation_test(outcomes, treatment, n_permutations=1000, seed=42)
        r_result = r_permutation_test(outcomes, treatment, n_permutations=1000, seed=42)

        if r_result is None:
            pytest.skip("R permutation_test unavailable")

        # Observed statistic should match exactly
        assert np.isclose(py_result["observed_diff"], r_result["observed_statistic"], rtol=1e-10), (
            f"Observed stat mismatch: Python={py_result['observed_diff']:.4f}, R={r_result['observed_statistic']:.4f}"
        )

        # P-values should be similar (allow for Monte Carlo variance)
        # With same seed and n_permutations, should be close
        assert np.isclose(py_result["p_value"], r_result["p_value"], atol=0.05), (
            f"P-value mismatch: Python={py_result['p_value']:.4f}, R={r_result['p_value']:.4f}"
        )

    def test_larger_sample(self):
        """Test with larger sample."""
        data = generate_rct_data(n=50, true_ate=3.0, seed=123)

        py_result = permutation_test(
            data["outcomes"], data["treatment"], n_permutations=500, seed=42
        )
        r_result = r_permutation_test(
            data["outcomes"], data["treatment"], n_permutations=500, seed=42
        )

        if r_result is None:
            pytest.skip("R permutation_test unavailable")

        # Observed statistics should match
        assert np.isclose(py_result["observed_diff"], r_result["observed_statistic"], rtol=0.05)


@requires_rct_python
class TestIPWATEVsR:
    """Compare Python ipw_ate() to R Horvitz-Thompson estimator."""

    def test_known_propensity(self):
        """Python and R should agree with known propensity scores."""
        data = generate_ipw_data(n=100, true_ate=2.0, seed=42)

        py_result = ipw_ate(data["outcomes"], data["treatment"], data["propensity"])
        r_result = r_ipw_ate(data["outcomes"], data["treatment"], data["propensity"])

        if r_result is None:
            pytest.skip("R ipw_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

    def test_constant_propensity(self):
        """Python and R should agree with constant propensity (equal to simple ATE)."""
        np.random.seed(42)
        n = 100
        treatment = np.random.binomial(1, 0.5, n)
        outcomes = 2.5 * treatment + np.random.randn(n)
        propensity = np.full(n, 0.5)  # Constant propensity

        py_result = ipw_ate(outcomes, treatment, propensity)
        r_result = r_ipw_ate(outcomes, treatment, propensity)

        if r_result is None:
            pytest.skip("R ipw_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )

    def test_varying_propensity(self):
        """Python and R should agree with varying propensity scores."""
        data = generate_ipw_data(n=100, true_ate=1.5, propensity_range=(0.2, 0.8), seed=999)

        py_result = ipw_ate(data["outcomes"], data["treatment"], data["propensity"])
        r_result = r_ipw_ate(data["outcomes"], data["treatment"], data["propensity"])

        if r_result is None:
            pytest.skip("R ipw_ate unavailable")

        assert np.isclose(py_result["ate"], r_result["estimate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result['ate']:.4f}, R={r_result['estimate']:.4f}"
        )


# =============================================================================
# Integration Tests
# =============================================================================


@requires_rct_python
class TestRCTEstimatorsConsistency:
    """Test that all estimators give similar results on the same RCT data."""

    def test_all_estimators_similar(self):
        """All estimators should give similar ATE estimates on clean RCT data."""
        np.random.seed(42)
        n = 200
        true_ate = 2.0

        # Generate data
        treatment = np.array([1] * 100 + [0] * 100)
        np.random.shuffle(treatment)
        covariate = np.random.randn(n)
        outcomes = true_ate * treatment + 0.3 * covariate + np.random.randn(n)
        strata = np.repeat([1, 2, 3, 4], 50)
        propensity = np.full(n, 0.5)

        # Collect all estimates
        estimates = []

        # Simple ATE
        py_simple = simple_ate(outcomes, treatment)
        estimates.append(("simple_ate", py_simple["ate"]))

        # Stratified ATE
        py_strat = stratified_ate(outcomes, treatment, strata)
        estimates.append(("stratified_ate", py_strat["ate"]))

        # Regression adjusted
        py_reg = regression_adjusted_ate(outcomes, treatment, covariate.reshape(-1, 1))
        estimates.append(("regression_ate", py_reg["ate"]))

        # IPW
        py_ipw = ipw_ate(outcomes, treatment, propensity)
        estimates.append(("ipw_ate", py_ipw["ate"]))

        # All estimates should be within 20% of each other
        ate_values = [e[1] for e in estimates]
        mean_ate = np.mean(ate_values)

        for name, ate in estimates:
            assert np.isclose(ate, mean_ate, rtol=0.20), (
                f"{name} ATE={ate:.4f} differs from mean={mean_ate:.4f}"
            )
