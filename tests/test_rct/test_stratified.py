"""Test stratified ATE estimator.

Stratification (blocking) reduces variance by removing between-block variation.
Key principle: Treatment is randomized WITHIN strata, not overall.

Following test-first development: write tests before implementation.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_stratified import stratified_ate


class TestStratifiedATEKnownAnswers:
    """Test stratified_ate with hand-calculated values."""

    def test_stratified_reduces_variance(self):
        """
        Test that stratified estimator has lower SE than simple difference-in-means.

        Setup: Create data where strata have different baseline levels.
        - Stratum 1 (high baseline): Y(0) ~ 100, treatment effect = 5
        - Stratum 2 (low baseline): Y(0) ~ 10, treatment effect = 5

        Without stratification: High variance due to baseline differences
        With stratification: Lower variance (remove between-stratum variation)
        """
        # Create stratified data
        np.random.seed(42)

        # Stratum 1: High baseline, n=50
        stratum1 = np.ones(50)
        treatment1 = np.random.binomial(1, 0.5, 50)
        outcomes1 = 100 + treatment1 * 5 + np.random.normal(0, 2, 50)

        # Stratum 2: Low baseline, n=50
        stratum2 = np.ones(50) * 2
        treatment2 = np.random.binomial(1, 0.5, 50)
        outcomes2 = 10 + treatment2 * 5 + np.random.normal(0, 2, 50)

        # Combine
        strata = np.concatenate([stratum1, stratum2])
        treatment = np.concatenate([treatment1, treatment2])
        outcomes = np.concatenate([outcomes1, outcomes2])

        # Estimate both ways
        simple_result = simple_ate(outcomes, treatment)
        stratified_result = stratified_ate(outcomes, treatment, strata)

        # Both should estimate ~5.0 (generous tolerance for stochastic test)
        # Note: With seed=42, estimates may vary due to random treatment assignment
        assert np.isclose(simple_result["estimate"], 5.0, atol=2.5)
        assert np.isclose(stratified_result["estimate"], 5.0, atol=2.5)

        # KEY TEST: Stratified should have smaller SE (variance reduction)
        # This is the main point - stratification removes between-stratum variation
        assert stratified_result["se"] < simple_result["se"], (
            f"Stratified SE ({stratified_result['se']}) should be < simple SE ({simple_result['se']})"
        )

    def test_stratified_known_answer(self):
        """
        Test with hand-calculated stratified ATE.

        Data:
        - Stratum 1: Treatment [1,1], Control [0,0], Outcomes [10,8,4,2]
          - Treated mean: 9.0, Control mean: 3.0, ATE_1 = 6.0, n_1 = 4
        - Stratum 2: Treatment [1,1], Control [0,0], Outcomes [20,18,14,12]
          - Treated mean: 19.0, Control mean: 13.0, ATE_2 = 6.0, n_2 = 4

        Overall ATE (weighted): (4*6.0 + 4*6.0) / 8 = 6.0
        """
        strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        treatment = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        outcomes = np.array([10.0, 8.0, 4.0, 2.0, 20.0, 18.0, 14.0, 12.0])

        result = stratified_ate(outcomes, treatment, strata)

        # Both strata have ATE = 6.0, so weighted average = 6.0
        assert np.isclose(result["estimate"], 6.0), f"Expected ATE=6.0, got {result['estimate']}"

        # Check stratum-specific estimates
        assert "stratum_estimates" in result
        assert len(result["stratum_estimates"]) == 2
        assert np.isclose(result["stratum_estimates"][0], 6.0)
        assert np.isclose(result["stratum_estimates"][1], 6.0)

    def test_single_stratum_equals_simple_ate(self):
        """
        Test that stratified_ate with one stratum equals simple_ate.

        When there's only one stratum, stratification does nothing.
        """
        # Simple RCT data
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        strata = np.ones(4)  # All same stratum

        simple_result = simple_ate(outcomes, treatment)
        stratified_result = stratified_ate(outcomes, treatment, strata)

        # Should be identical
        assert np.isclose(simple_result["estimate"], stratified_result["estimate"])

    def test_heterogeneous_effects_across_strata(self):
        """
        Test stratified estimator with different treatment effects per stratum.

        Setup:
        - Stratum 1: ATE = 10.0
        - Stratum 2: ATE = 20.0
        - Equal sizes (n=4 each)
        - Overall weighted ATE = (4*10 + 4*20) / 8 = 15.0
        """
        strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        treatment = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        # Stratum 1: Treated [15, 13], Control [5, 3] → ATE = 14-4 = 10
        # Stratum 2: Treated [30, 28], Control [10, 8] → ATE = 29-9 = 20
        outcomes = np.array([15.0, 13.0, 5.0, 3.0, 30.0, 28.0, 10.0, 8.0])

        result = stratified_ate(outcomes, treatment, strata)

        # Weighted average: (4*10 + 4*20) / 8 = 15.0
        expected_ate = 15.0
        assert np.isclose(result["estimate"], expected_ate), (
            f"Expected ATE={expected_ate}, got {result['estimate']}"
        )

        # Check stratum-specific estimates
        assert "stratum_estimates" in result
        assert np.isclose(result["stratum_estimates"][0], 10.0)
        assert np.isclose(result["stratum_estimates"][1], 20.0)

    def test_many_strata(self):
        """
        Test stratified estimator with many strata (5 strata).

        Each stratum has constant ATE = 5, so overall ATE = 5.
        """
        n_strata = 5
        n_per_stratum = 4  # 2 treated, 2 control per stratum

        strata = np.repeat(np.arange(1, n_strata + 1), n_per_stratum)
        treatment = np.tile([1, 1, 0, 0], n_strata)

        # Each stratum has different baseline but same treatment effect = 5
        baselines = np.array([10, 20, 30, 40, 50])
        outcomes = (
            np.repeat(baselines, n_per_stratum)
            + treatment * 5
            + np.random.normal(0, 0.1, len(strata))
        )

        result = stratified_ate(outcomes, treatment, strata)

        # Should recover ATE ≈ 5 (constant across strata)
        expected_ate = 5.0
        assert np.isclose(result["estimate"], expected_ate, atol=0.5), (
            f"Expected ATE={expected_ate}, got {result['estimate']}"
        )

        # Should have 5 stratum-specific estimates
        assert "stratum_estimates" in result
        assert len(result["stratum_estimates"]) == n_strata


class TestStratifiedATEErrorHandling:
    """Test error handling for stratified_ate."""

    def test_mismatched_strata_length(self):
        """Test that mismatched strata length raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        strata = np.array([1, 1])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            stratified_ate(outcomes, treatment, strata)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "strata" in error_msg.lower()
        assert "length" in error_msg.lower()

    def test_stratum_with_no_treated(self):
        """Test that stratum with no treated units raises ValueError."""
        # Stratum 1 has no treated units
        strata = np.array([1, 1, 2, 2])
        treatment = np.array([0, 0, 1, 0])  # No treated in stratum 1
        outcomes = np.array([1.0, 2.0, 5.0, 3.0])

        with pytest.raises(ValueError) as exc_info:
            stratified_ate(outcomes, treatment, strata)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "stratum" in error_msg.lower()
        assert "no treated" in error_msg.lower() or "no variation" in error_msg.lower()

    def test_stratum_with_no_control(self):
        """Test that stratum with no control units raises ValueError."""
        # Stratum 2 has no control units
        strata = np.array([1, 1, 2, 2])
        treatment = np.array([1, 0, 1, 1])  # No control in stratum 2
        outcomes = np.array([7.0, 3.0, 10.0, 12.0])

        with pytest.raises(ValueError) as exc_info:
            stratified_ate(outcomes, treatment, strata)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "stratum" in error_msg.lower()
        assert "no control" in error_msg.lower() or "no variation" in error_msg.lower()


class TestStratifiedATEProperties:
    """Test statistical properties of stratified estimator."""

    def test_stratum_weights_sum_to_one(self):
        """Test that stratum weights are proportional to stratum sizes."""
        # Unequal stratum sizes
        np.random.seed(42)
        strata = np.array([1] * 30 + [2] * 70)  # 30 in stratum 1, 70 in stratum 2
        treatment = np.random.binomial(1, 0.5, 100)
        outcomes = np.random.normal(5, 2, 100)

        result = stratified_ate(outcomes, treatment, strata)

        # Weights should be proportional to n
        assert "stratum_weights" in result
        assert np.isclose(sum(result["stratum_weights"]), 1.0)
        assert np.isclose(result["stratum_weights"][0], 0.3, atol=0.01)  # 30/100
        assert np.isclose(result["stratum_weights"][1], 0.7, atol=0.01)  # 70/100

    def test_balanced_strata(self):
        """Test with perfectly balanced treatment within strata."""
        # Each stratum has exactly 50/50 treatment split
        strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        treatment = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0, 20.0, 18.0, 14.0, 12.0])

        result = stratified_ate(outcomes, treatment, strata)

        # Should handle balanced strata correctly
        assert "estimate" in result
        assert "se" in result
        assert result["se"] > 0  # Variance should be finite


# Marker for pytest
pytestmark = pytest.mark.unit
