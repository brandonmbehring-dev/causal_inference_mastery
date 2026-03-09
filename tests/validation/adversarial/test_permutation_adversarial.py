"""
Adversarial tests for permutation_test estimator.

Permutation-specific edge cases:
1. Perfect separation (all treated high, all control low)
2. Tied outcomes (many identical values)
3. Very small sample (exact enumeration)
4. Large sample (Monte Carlo approximation)
5. Zero effect case
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators_permutation import permutation_test


class TestPermutationTestPerfectSeparation:
    """Test permutation_test with perfect separation."""

    def test_no_overlap_in_outcomes(self):
        """All treated > all control (perfect separation)."""
        outcomes = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = permutation_test(outcomes, treatment, n_permutations=None)  # Exact

        # Should have low p-value (relaxed to 0.15 for small sample discrete test)
        assert result["p_value"] < 0.15
        # Observed statistic should be extreme
        assert result["observed_statistic"] > 5.0

    def test_zero_effect_case(self):
        """All outcomes identical (no effect)."""
        outcomes = np.array([5.0] * 10)
        treatment = np.array([1] * 5 + [0] * 5)

        result = permutation_test(outcomes, treatment, n_permutations=100)

        # p-value should be 1.0 (or close with smoothing)
        assert result["p_value"] > 0.9
        # Observed statistic should be zero
        assert result["observed_statistic"] == 0.0


class TestPermutationTestTiedValues:
    """Test permutation_test with many tied values."""

    def test_many_ties(self):
        """Only 2 unique values (half ties)."""
        outcomes = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 10.0, 5.0])
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        result = permutation_test(outcomes, treatment, n_permutations=1000, random_seed=42)

        # Should compute valid p-value
        assert 0.0 < result["p_value"] < 1.0
        # Many permutations will produce same statistic due to ties
        unique_stats = len(np.unique(result["permutation_distribution"]))
        assert unique_stats < result["n_permutations"]  # Fewer unique values than permutations


class TestPermutationTestSampleSizes:
    """Test permutation_test with different sample sizes."""

    def test_minimum_exact_sample(self):
        """n=4 - exact enumeration (C(4,2)=6 permutations)."""
        outcomes = np.array([10.0, 8.0, 3.0, 1.0])
        treatment = np.array([1, 1, 0, 0])

        result = permutation_test(outcomes, treatment, n_permutations=None)

        # Should enumerate all 6 permutations
        assert result["n_permutations"] == 6
        assert len(result["permutation_distribution"]) == 6

    def test_large_sample_monte_carlo(self):
        """n=1000 - should use Monte Carlo (exact would be huge)."""
        np.random.seed(42)
        n = 1000
        treatment = np.array([1] * 500 + [0] * 500)
        outcomes = np.where(
            treatment == 1, np.random.normal(2.0, 1.0, n), np.random.normal(0.0, 1.0, n)
        )

        result = permutation_test(outcomes, treatment, n_permutations=1000, random_seed=42)

        # Should run 1000 permutations
        assert result["n_permutations"] == 1000
        # Should detect effect (p < 0.05)
        assert result["p_value"] < 0.05


class TestPermutationTestAlternatives:
    """Test different alternative hypotheses."""

    def test_one_sided_greater(self):
        """Test H1: treated > control."""
        outcomes = np.array([10.0, 9.0, 8.0, 2.0, 1.0, 0.0])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = permutation_test(
            outcomes, treatment, n_permutations=1000, alternative="greater", random_seed=42
        )

        # Should have low p-value for greater alternative
        assert result["p_value"] < 0.1

    def test_one_sided_less(self):
        """Test H1: treated < control."""
        outcomes = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = permutation_test(
            outcomes, treatment, n_permutations=1000, alternative="less", random_seed=42
        )

        # Should have low p-value for less alternative
        assert result["p_value"] < 0.1


class TestPermutationTestOutliers:
    """Test permutation_test with extreme outliers."""

    def test_single_extreme_outlier(self):
        """One extreme outlier in treated group."""
        outcomes = np.array([10000.0, 2.0, 2.1, 1.0, 1.1, 1.2])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = permutation_test(outcomes, treatment, n_permutations=1000, random_seed=42)

        # Outlier should make p-value small (relaxed to 0.15 for stochastic variation)
        assert result["p_value"] < 0.15
        # Observed statistic should be huge
        assert result["observed_statistic"] > 1000
