"""
Adversarial tests for stratified_ate estimator.

Stratified-specific edge cases:
1. Single observation per stratum
2. Extreme number of strata (n=100 with 100 strata)
3. One stratum with all treated
4. Highly imbalanced strata sizes
5. Variance heterogeneity across strata
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators_stratified import stratified_ate


class TestStratifiedATEStratumEdgeCases:
    """Test stratified_ate with stratum-specific edge cases."""

    def test_many_strata_few_observations(self):
        """100 strata with 2 observations each (n=200 total)."""
        outcomes = []
        treatment = []
        strata = []

        for s in range(100):
            outcomes.extend([2.0, 0.0])  # ATE = 2 in each stratum
            treatment.extend([1, 0])
            strata.extend([s, s])

        result = stratified_ate(np.array(outcomes), np.array(treatment), np.array(strata))

        # Should compute ATE ≈ 2.0
        assert np.isclose(result["estimate"], 2.0)
        assert result["n_strata"] == 100

    def test_highly_imbalanced_strata_sizes(self):
        """One huge stratum (n=990), many tiny strata (n=2 each)."""
        outcomes = []
        treatment = []
        strata = []

        # Large stratum (s=0): n=990
        np.random.seed(42)
        t_large = np.array([1] * 495 + [0] * 495)
        y_large = np.where(
            t_large == 1, np.random.normal(2.0, 1.0, 990), np.random.normal(0.0, 1.0, 990)
        )
        outcomes.extend(y_large)
        treatment.extend(t_large)
        strata.extend([0] * 990)

        # 5 tiny strata (s=1-5): n=2 each
        for s in range(1, 6):
            outcomes.extend([2.0, 0.0])
            treatment.extend([1, 0])
            strata.extend([s, s])

        result = stratified_ate(np.array(outcomes), np.array(treatment), np.array(strata))

        # Should be dominated by large stratum
        assert 1.8 < result["estimate"] < 2.2
        assert result["n_strata"] == 6

    def test_extreme_variance_heterogeneity(self):
        """One stratum with σ²=0.001, another with σ²=1000."""
        np.random.seed(42)

        # Stratum 1: Low variance
        y1_low = 2.0 + np.random.normal(0, 0.01, 50)
        y0_low = 0.0 + np.random.normal(0, 0.01, 50)
        outcomes_low = np.concatenate([y1_low, y0_low])
        treatment_low = np.array([1] * 50 + [0] * 50)
        strata_low = np.array([0] * 100)

        # Stratum 2: High variance
        y1_high = 2.0 + np.random.normal(0, 100, 50)
        y0_high = 0.0 + np.random.normal(0, 100, 50)
        outcomes_high = np.concatenate([y1_high, y0_high])
        treatment_high = np.array([1] * 50 + [0] * 50)
        strata_high = np.array([1] * 100)

        outcomes = np.concatenate([outcomes_low, outcomes_high])
        treatment = np.concatenate([treatment_low, treatment_high])
        strata = np.concatenate([strata_low, strata_high])

        result = stratified_ate(outcomes, treatment, strata)

        # Should compute valid estimate
        assert np.isfinite(result["estimate"])
        # SE should be dominated by high-variance stratum
        assert result["se"] > 5.0

    def test_single_observation_per_group_per_stratum(self):
        """n1=1, n0=1 in each stratum (bug was fixed!)."""
        outcomes = np.array([10.0, 5.0, 20.0, 15.0, 30.0, 25.0])
        treatment = np.array([1, 0, 1, 0, 1, 0])
        strata = np.array([0, 0, 1, 1, 2, 2])

        result = stratified_ate(outcomes, treatment, strata)

        # ATE in each stratum: (10-5)=5, (20-15)=5, (30-25)=5
        # Overall ATE = 5
        assert result["estimate"] == 5.0
        # Variance should be zero in each stratum (n=1)
        assert result["se"] < 1e-10

    def test_perfect_stratification(self):
        """All outcomes identical within treatment-stratum groups."""
        # Stratum 0: 20 units (10 treated with Y=10, 10 control with Y=5)
        # Stratum 1: 20 units (10 treated with Y=20, 10 control with Y=15)
        outcomes = np.array(
            [10.0] * 10
            + [5.0] * 10  # Stratum 0
            + [20.0] * 10
            + [15.0] * 10  # Stratum 1
        )
        treatment = np.array(
            [1] * 10
            + [0] * 10  # Stratum 0
            + [1] * 10
            + [0] * 10  # Stratum 1
        )
        strata = np.array([0] * 20 + [1] * 20)

        result = stratified_ate(outcomes, treatment, strata)

        # ATE = 5 in both strata: (10-5)=5 in s0, (20-15)=5 in s1
        assert result["estimate"] == 5.0
        # Zero variance within treatment-stratum groups
        assert result["se"] < 1e-10


class TestStratifiedATEWeighting:
    """Test stratified_ate weighting is correct."""

    def test_unequal_stratum_weights(self):
        """Verify stratum weights sum to 1 and are proportional to size."""
        outcomes = np.array([2.0, 0.0] * 25 + [2.0, 0.0] * 75)  # s1: 50, s2: 150
        treatment = np.array([1, 0] * 25 + [1, 0] * 75)
        strata = np.array([0] * 50 + [1] * 150)

        result = stratified_ate(outcomes, treatment, strata)

        # Weights should be 50/200=0.25 and 150/200=0.75
        weights = result["stratum_weights"]
        assert np.isclose(weights[0], 0.25)
        assert np.isclose(weights[1], 0.75)
        assert np.isclose(sum(weights), 1.0)

    def test_weighted_average_correctness(self):
        """Verify weighted average is computed correctly."""
        # Stratum 1: ATE=10, n=50
        # Stratum 2: ATE=0, n=150
        # Overall ATE = 0.25*10 + 0.75*0 = 2.5
        outcomes = np.concatenate(
            [
                np.array([10.0] * 25 + [0.0] * 25),  # s1
                np.array([0.0] * 75 + [0.0] * 75),  # s2
            ]
        )
        treatment = np.array([1] * 25 + [0] * 25 + [1] * 75 + [0] * 75)
        strata = np.array([0] * 50 + [1] * 150)

        result = stratified_ate(outcomes, treatment, strata)

        assert np.isclose(result["estimate"], 2.5)


class TestStratifiedATEMissingStrata:
    """Test error handling for missing strata."""

    def test_empty_stratum_after_split(self):
        """All strata should have both treated and control."""
        # This should raise an error - tested in main test suite
        pass  # Covered by error handling tests


class TestStratifiedATENumericalStability:
    """Test stratified_ate with numerically challenging scenarios."""

    def test_extremely_large_number_of_strata(self):
        """
        Test with 500 strata (near-continuous stratification).

        Each stratum has n=2 (1 treated, 1 control).
        """
        np.random.seed(42)
        n_strata = 500
        outcomes = []
        treatment = []
        strata = []

        for s in range(n_strata):
            # Each stratum: 1 treated (Y=2), 1 control (Y=0) → ATE=2
            outcomes.extend([2.0, 0.0])
            treatment.extend([1, 0])
            strata.extend([s, s])

        result = stratified_ate(np.array(outcomes), np.array(treatment), np.array(strata))

        # Should compute ATE = 2.0
        assert np.isclose(result["estimate"], 2.0)
        assert result["n_strata"] == 500
        # SE should be zero (no variance within strata)
        assert result["se"] < 1e-10
