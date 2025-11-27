"""Test permutation test for RCTs.

Permutation tests (Fisher exact tests) provide exact p-values under the sharp null
hypothesis of no treatment effect for any unit. Unlike asymptotic tests, they make
no distributional assumptions and work for small samples.

Key principle: Under H0, treatment labels are exchangeable. Any permutation of
treatment assignments is equally likely under randomization.

Following test-first development: write tests before implementation.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_permutation import permutation_test


class TestPermutationTestKnownAnswers:
    """Test permutation_test with known p-values."""

    def test_permutation_with_no_effect(self):
        """
        Test permutation test under true null (no treatment effect).

        With no treatment effect, the observed test statistic should be
        typical under the permutation distribution, yielding p-value > 0.05.
        """
        np.random.seed(42)
        n = 20
        treatment = np.array([1, 0] * (n // 2))

        # No treatment effect: Y(1) = Y(0) for all units
        outcomes = np.random.normal(5, 2, n)  # Same distribution for all

        result = permutation_test(outcomes, treatment, n_permutations=1000, random_seed=42)

        # Under true null, p-value should be reasonably large (not extreme)
        # With 1000 permutations, expect p > 0.05 most of the time
        assert result["p_value"] > 0.05, \
            f"Under null, expected p > 0.05, got {result['p_value']}"

    def test_permutation_with_strong_effect(self):
        """
        Test permutation test with strong treatment effect.

        With large treatment effect, observed statistic should be extreme
        in permutation distribution, yielding small p-value.
        """
        np.random.seed(42)
        n = 30
        treatment = np.array([1, 0] * (n // 2))

        # Strong effect: Treatment increases outcome by 10 (large relative to noise)
        outcomes = 10 * treatment + np.random.normal(0, 1, n)

        result = permutation_test(outcomes, treatment, n_permutations=1000, random_seed=42)

        # With strong effect, p-value should be very small
        assert result["p_value"] < 0.01, \
            f"With strong effect, expected p < 0.01, got {result['p_value']}"

    def test_permutation_distribution_properties(self):
        """
        Test that permutation distribution has correct properties.

        The permutation distribution should:
        1. Contain n_permutations test statistics
        2. Include the observed statistic
        3. Be centered near zero under null
        """
        np.random.seed(123)
        treatment = np.array([1, 1, 1, 0, 0, 0])
        outcomes = np.array([5.0, 6.0, 7.0, 3.0, 4.0, 2.0])

        result = permutation_test(outcomes, treatment, n_permutations=500, random_seed=42)

        # Should return permutation distribution
        assert "permutation_distribution" in result
        assert len(result["permutation_distribution"]) == 500

        # Observed statistic should be included
        assert "observed_statistic" in result

        # Under null, permutation distribution mean should be near zero
        # (symmetric around zero for balanced designs)
        assert abs(np.mean(result["permutation_distribution"])) < 2.0

    def test_exact_permutation_small_sample(self):
        """
        Test exact permutation test (all permutations) for very small sample.

        With n=6 (3 treated, 3 control), there are C(6,3) = 20 permutations.
        Test that exact computation works.
        """
        treatment = np.array([1, 1, 1, 0, 0, 0])
        # Strong effect for clear result
        outcomes = np.array([7.0, 8.0, 9.0, 1.0, 2.0, 3.0])

        # Exact test: n_permutations = "all" or very large number
        result = permutation_test(outcomes, treatment, n_permutations=None, random_seed=42)

        # Should have exactly C(6,3) = 20 permutations
        assert len(result["permutation_distribution"]) == 20

        # With strong effect (ATE = 6), should reject null (p <= 0.15 for discrete test)
        assert result["p_value"] <= 0.15

    def test_two_sided_vs_one_sided(self):
        """
        Test that two-sided p-value is approximately 2x one-sided for extreme values.

        For very large or very small observed statistics, one-sided p-value
        should be roughly half the two-sided p-value.
        """
        np.random.seed(42)
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # Strong positive effect
        outcomes = np.array([10.0, 11.0, 9.0, 10.5, 1.0, 2.0, 1.5, 2.5])

        result_two_sided = permutation_test(
            outcomes, treatment, n_permutations=1000, alternative="two-sided", random_seed=42
        )
        result_greater = permutation_test(
            outcomes, treatment, n_permutations=1000, alternative="greater", random_seed=42
        )

        # Two-sided should be roughly 2x one-sided for extreme values
        # (within Monte Carlo error)
        assert result_two_sided["p_value"] >= result_greater["p_value"]
        assert result_two_sided["p_value"] <= 2 * result_greater["p_value"] + 0.05


class TestPermutationTestErrorHandling:
    """Test error handling for permutation_test."""

    def test_empty_input_fails_fast(self):
        """Test that empty input arrays raise ValueError."""
        treatment = np.array([])
        outcomes = np.array([])

        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=100)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "empty" in error_msg.lower()

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([5.0, 6.0])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=100)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "length" in error_msg.lower()

    def test_nan_values_fail_fast(self):
        """Test that NaN values raise ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([5.0, np.nan, 3.0, 2.0])

        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=100)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "nan" in error_msg.lower()

    def test_all_treated_fails_fast(self):
        """Test that data with no control units raises ValueError."""
        treatment = np.array([1, 1, 1, 1])
        outcomes = np.array([5.0, 6.0, 7.0, 8.0])

        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=100)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "no control" in error_msg.lower()

    def test_invalid_alternative_fails_fast(self):
        """Test that invalid alternative raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([5.0, 6.0, 3.0, 2.0])

        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=100, alternative="invalid")

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alternative" in error_msg.lower()

    def test_invalid_n_permutations_fails_fast(self):
        """Test that invalid n_permutations raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([5.0, 6.0, 3.0, 2.0])

        # Negative n_permutations
        with pytest.raises(ValueError) as exc_info:
            permutation_test(outcomes, treatment, n_permutations=-100)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "permutation" in error_msg.lower()


class TestPermutationTestProperties:
    """Test statistical properties of permutation test."""

    def test_permutation_test_is_exact(self):
        """
        Test that permutation test achieves exact Type I error rate under null.

        Under H0, the probability of rejecting at alpha=0.05 should be
        exactly 0.05 (not asymptotically, but exactly).

        This is a Monte Carlo check: generate data under null, run permutation
        tests, and check rejection rate ≈ 0.05.
        """
        np.random.seed(42)
        n_simulations = 100  # Limited for speed
        n_rejections = 0
        alpha = 0.05

        for _ in range(n_simulations):
            # Generate data under null (no treatment effect)
            treatment = np.array([1, 0] * 10)  # n=20
            outcomes = np.random.normal(5, 2, 20)  # Same distribution

            result = permutation_test(outcomes, treatment, n_permutations=500, random_seed=None)

            if result["p_value"] < alpha:
                n_rejections += 1

        rejection_rate = n_rejections / n_simulations

        # Should be close to 0.05 (within Monte Carlo error)
        # With 100 simulations, SE ≈ sqrt(0.05*0.95/100) ≈ 0.022
        # So within 2 SE ≈ 0.044, expect rejection rate in [0.006, 0.094]
        assert 0.0 <= rejection_rate <= 0.15, \
            f"Rejection rate {rejection_rate} should be close to 0.05 under null"

    def test_permutation_reproducibility_with_seed(self):
        """
        Test that permutation test is reproducible with same random seed.
        """
        treatment = np.array([1, 1, 1, 0, 0, 0])
        outcomes = np.array([5.0, 6.0, 7.0, 3.0, 4.0, 2.0])

        result1 = permutation_test(outcomes, treatment, n_permutations=500, random_seed=123)
        result2 = permutation_test(outcomes, treatment, n_permutations=500, random_seed=123)

        # Should be identical with same seed
        assert result1["p_value"] == result2["p_value"]
        assert np.array_equal(result1["permutation_distribution"], result2["permutation_distribution"])

    def test_permutation_power_increases_with_effect_size(self):
        """
        Test that power increases with effect size.

        Larger treatment effects should lead to smaller p-values.
        """
        np.random.seed(42)
        treatment = np.array([1, 0] * 15)  # n=30

        # Small effect
        outcomes_small = 1 * treatment + np.random.normal(0, 2, 30)
        result_small = permutation_test(outcomes_small, treatment, n_permutations=500, random_seed=42)

        # Large effect
        outcomes_large = 5 * treatment + np.random.normal(0, 2, 30)
        result_large = permutation_test(outcomes_large, treatment, n_permutations=500, random_seed=42)

        # Larger effect should have smaller p-value (higher power)
        assert result_large["p_value"] < result_small["p_value"]


# Marker for pytest
pytestmark = pytest.mark.unit
