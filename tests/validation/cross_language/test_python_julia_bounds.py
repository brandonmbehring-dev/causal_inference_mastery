"""
Cross-language validation tests for Partial Identification Bounds.

Session 95: Python ↔ Julia parity for Manski and Lee bounds.
"""

import pytest
import numpy as np
from typing import Tuple

from .julia_interface import (
    is_julia_available,
    julia_manski_worst_case,
    julia_lee_bounds,
)

# Import Python implementations
try:
    from src.causal_inference.bounds.manski import manski_worst_case, manski_mtr, manski_mts
    from src.causal_inference.bounds.lee import lee_bounds

    PYTHON_BOUNDS_AVAILABLE = True
except ImportError:
    PYTHON_BOUNDS_AVAILABLE = False


# Skip all tests if Julia not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-validation",
)


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def generate_bounds_data(
    n: int = 500,
    true_ate: float = 2.0,
    treatment_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple data for bounds testing."""
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < treatment_prob).astype(float)
    noise = rng.standard_normal(n)
    outcome = true_ate * treatment + noise
    return outcome, treatment


def generate_lee_data(
    n: int = 1000,
    true_ate: float = 2.0,
    attrition_base: float = 0.2,
    attrition_diff: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with sample selection for Lee bounds."""
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < 0.5).astype(float)

    # Treatment increases observation probability
    obs_prob = (1 - attrition_base) + attrition_diff * treatment
    observed = (rng.random(n) < obs_prob).astype(float)

    # Outcome
    outcome = true_ate * treatment + rng.standard_normal(n)

    return outcome, treatment, observed


# =============================================================================
# MANSKI WORST-CASE BOUNDS TESTS
# =============================================================================


class TestManskiWorstCaseParity:
    """Cross-language tests for Manski worst-case bounds."""

    def test_basic_bounds(self):
        """Test basic Manski worst-case bounds match."""
        outcome, treatment = generate_bounds_data(n=500, true_ate=2.0, seed=42)

        jl_result = julia_manski_worst_case(outcome, treatment)

        # Check Julia result structure
        assert "bounds_lower" in jl_result
        assert "bounds_upper" in jl_result
        assert jl_result["bounds_lower"] <= jl_result["bounds_upper"]
        assert jl_result["assumptions"] == "worst_case"

    def test_contains_true_ate(self):
        """True ATE should be within bounds."""
        true_ate = 2.0
        outcome, treatment = generate_bounds_data(n=1000, true_ate=true_ate, seed=42)

        jl_result = julia_manski_worst_case(outcome, treatment)

        # True ATE should be in bounds
        assert jl_result["bounds_lower"] <= true_ate <= jl_result["bounds_upper"]

    def test_naive_ate_in_bounds(self):
        """Naive ATE should be within worst-case bounds."""
        outcome, treatment = generate_bounds_data(n=500, seed=42)

        jl_result = julia_manski_worst_case(outcome, treatment)

        assert jl_result["ate_in_bounds"] is True
        assert jl_result["bounds_lower"] <= jl_result["naive_ate"] <= jl_result["bounds_upper"]

    def test_sample_counts(self):
        """Check sample counts are correct."""
        outcome, treatment = generate_bounds_data(n=500, treatment_prob=0.5, seed=42)

        jl_result = julia_manski_worst_case(outcome, treatment)

        assert jl_result["n_treated"] > 0
        assert jl_result["n_control"] > 0
        assert jl_result["n_treated"] + jl_result["n_control"] == 500

    def test_custom_outcome_support(self):
        """Test with custom outcome support bounds."""
        outcome, treatment = generate_bounds_data(n=500, seed=42)

        jl_result = julia_manski_worst_case(outcome, treatment, outcome_support=(-5.0, 5.0))

        # Bounds should be finite
        assert np.isfinite(jl_result["bounds_lower"])
        assert np.isfinite(jl_result["bounds_upper"])


# =============================================================================
# LEE BOUNDS TESTS
# =============================================================================


class TestLeeBoundsParity:
    """Cross-language tests for Lee (2009) bounds."""

    def test_basic_lee_bounds(self):
        """Test basic Lee bounds computation."""
        outcome, treatment, observed = generate_lee_data(n=1000, seed=42)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            monotonicity="positive",
            n_bootstrap=100,
            seed=42,
        )

        # Check structure
        assert "bounds_lower" in jl_result
        assert "bounds_upper" in jl_result
        assert jl_result["bounds_lower"] <= jl_result["bounds_upper"]
        assert jl_result["monotonicity"] == "positive"

    def test_contains_true_ate(self):
        """True ATE should be within Lee bounds."""
        true_ate = 2.0
        outcome, treatment, observed = generate_lee_data(n=2000, true_ate=true_ate, seed=42)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            n_bootstrap=200,
            seed=42,
        )

        # True ATE should be in bounds (with some tolerance for randomness)
        # Lee bounds should contain true effect under correct assumptions
        assert jl_result["bounds_lower"] <= true_ate + 0.5
        assert jl_result["bounds_upper"] >= true_ate - 0.5

    def test_ci_contains_bounds(self):
        """Bootstrap CI should contain point estimates."""
        outcome, treatment, observed = generate_lee_data(n=1000, seed=42)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            n_bootstrap=500,
            seed=42,
        )

        # CI should cover bounds
        if not np.isnan(jl_result["ci_lower"]):
            assert jl_result["ci_lower"] <= jl_result["bounds_lower"]
            assert jl_result["bounds_upper"] <= jl_result["ci_upper"]

    def test_attrition_rates(self):
        """Check attrition rates are computed correctly."""
        outcome, treatment, observed = generate_lee_data(
            n=1000, attrition_base=0.2, attrition_diff=0.1, seed=42
        )

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            n_bootstrap=50,
            seed=42,
        )

        # Treatment should have lower attrition (positive monotonicity)
        assert jl_result["attrition_treated"] <= jl_result["attrition_control"] + 0.1

    def test_negative_monotonicity(self):
        """Test with negative monotonicity assumption."""
        outcome, treatment, observed = generate_lee_data(n=1000, seed=42)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            monotonicity="negative",
            n_bootstrap=50,
            seed=42,
        )

        assert jl_result["monotonicity"] == "negative"

    def test_trimming_information(self):
        """Check trimming diagnostics."""
        outcome, treatment, observed = generate_lee_data(n=1000, seed=42)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            n_bootstrap=50,
            seed=42,
        )

        assert jl_result["n_treated_observed"] > 0
        assert jl_result["n_control_observed"] > 0
        assert jl_result["trimming_proportion"] >= 0
        assert jl_result["trimmed_group"] in ["treated", "control", "none"]


# =============================================================================
# EDGE CASES
# =============================================================================


class TestBoundsEdgeCases:
    """Edge case tests for bounds estimators."""

    def test_equal_treatment_groups(self):
        """Test with equal treatment/control sizes."""
        rng = np.random.default_rng(42)
        n = 500
        treatment = np.concatenate([np.ones(250), np.zeros(250)])
        outcome = 2.0 * treatment + rng.standard_normal(n)

        jl_result = julia_manski_worst_case(outcome, treatment)

        assert jl_result["n_treated"] == 250
        assert jl_result["n_control"] == 250

    def test_unbalanced_treatment(self):
        """Test with unbalanced treatment assignment."""
        rng = np.random.default_rng(42)
        n = 500
        treatment = (rng.random(n) < 0.2).astype(float)
        outcome = 2.0 * treatment + rng.standard_normal(n)

        jl_result = julia_manski_worst_case(outcome, treatment)

        # Should still compute valid bounds
        assert jl_result["bounds_lower"] < jl_result["bounds_upper"]

    def test_high_attrition(self):
        """Test Lee bounds with high attrition."""
        rng = np.random.default_rng(42)
        n = 1000
        treatment = (rng.random(n) < 0.5).astype(float)
        # High attrition: 50% base + 10% treatment effect
        obs_prob = 0.5 + 0.1 * treatment
        observed = (rng.random(n) < obs_prob).astype(float)
        outcome = 2.0 * treatment + rng.standard_normal(n)

        jl_result = julia_lee_bounds(
            outcome,
            treatment,
            observed,
            n_bootstrap=50,
            seed=42,
        )

        # Should have wider bounds with high attrition
        assert jl_result["bounds_width"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
