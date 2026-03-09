"""
Layer 1: Unit Tests for PSM Balance Diagnostics.

Tests compute_smd(), compute_variance_ratio(), check_covariate_balance(),
and balance_summary() with hand-calculated known answers.

Coverage:
- SMD calculation (pooled and unpooled)
- Variance ratio calculation
- Edge cases (zero variance, perfect balance)
- Full covariate balance checking
- Balance summary statistics

References:
- src/causal_inference/psm/balance.py (Python implementation)
- julia/src/psm/balance.jl (Julia reference)
- Austin, P. C. (2009). Balance diagnostics. Statistics in Medicine, 28(25), 3083-3107.
"""

import numpy as np
import pytest

from src.causal_inference.psm.balance import (
    compute_smd,
    compute_variance_ratio,
    check_covariate_balance,
    balance_summary,
)


class TestComputeSMD:
    """Unit tests for compute_smd() with hand-calculated expected values."""

    def test_smd_perfect_balance(self):
        """
        SMD = 0 when groups have identical means.

        Hand calculation:
        - x_t = [1, 2, 3], mean = 2.0, var = 1.0
        - x_c = [1, 2, 3], mean = 2.0, var = 1.0
        - pooled_std = sqrt((1.0 + 1.0) / 2) = 1.0
        - SMD = (2.0 - 2.0) / 1.0 = 0.0
        """
        x_t = np.array([1.0, 2.0, 3.0])
        x_c = np.array([1.0, 2.0, 3.0])

        smd = compute_smd(x_t, x_c, pooled=True)

        assert abs(smd) < 1e-10, f"Expected SMD ≈ 0, got {smd}"

    def test_smd_known_value_pooled(self):
        """
        SMD with simple known answer (pooled).

        Hand calculation:
        - x_t = [4, 6], mean = 5.0, var = 2.0
        - x_c = [1, 3], mean = 2.0, var = 2.0
        - pooled_std = sqrt((2.0 + 2.0) / 2) = sqrt(2.0) = 1.414
        - SMD = (5.0 - 2.0) / 1.414 = 2.121
        """
        x_t = np.array([4.0, 6.0])
        x_c = np.array([1.0, 3.0])

        smd = compute_smd(x_t, x_c, pooled=True)

        expected_smd = 3.0 / np.sqrt(2.0)  # = 2.121
        assert abs(smd - expected_smd) < 1e-10, f"Expected SMD = {expected_smd:.4f}, got {smd:.4f}"

    def test_smd_known_value_unpooled(self):
        """
        SMD with simple known answer (unpooled, standardized by treated variance).

        Hand calculation:
        - x_t = [4, 6], mean = 5.0, var = 2.0
        - x_c = [1, 3], mean = 2.0, var = 2.0
        - std_t = sqrt(2.0) = 1.414
        - SMD = (5.0 - 2.0) / 1.414 = 2.121
        """
        x_t = np.array([4.0, 6.0])
        x_c = np.array([1.0, 3.0])

        smd = compute_smd(x_t, x_c, pooled=False)

        expected_smd = 3.0 / np.sqrt(2.0)  # = 2.121
        assert abs(smd - expected_smd) < 1e-10, f"Expected SMD = {expected_smd:.4f}, got {smd:.4f}"

    def test_smd_negative_value(self):
        """
        SMD < 0 when control group has higher mean.

        Hand calculation:
        - x_t = [1, 2], mean = 1.5, var = 0.5
        - x_c = [4, 5], mean = 4.5, var = 0.5
        - pooled_std = sqrt((0.5 + 0.5) / 2) = sqrt(0.5) = 0.707
        - SMD = (1.5 - 4.5) / 0.707 = -4.243
        """
        x_t = np.array([1.0, 2.0])
        x_c = np.array([4.0, 5.0])

        smd = compute_smd(x_t, x_c, pooled=True)

        expected_smd = -3.0 / np.sqrt(0.5)  # = -4.243
        assert abs(smd - expected_smd) < 1e-10, f"Expected SMD = {expected_smd:.4f}, got {smd:.4f}"

    def test_smd_both_zero_variance(self):
        """
        SMD = 0 when both groups have zero variance and same mean.

        Hand calculation:
        - x_t = [2, 2, 2], mean = 2.0, var = 0.0
        - x_c = [2, 2, 2], mean = 2.0, var = 0.0
        - pooled_std → 0, means equal → SMD = 0
        """
        x_t = np.array([2.0, 2.0, 2.0])
        x_c = np.array([2.0, 2.0, 2.0])

        smd = compute_smd(x_t, x_c, pooled=True)

        assert smd == 0.0, f"Expected SMD = 0 (identical constant values), got {smd}"

    def test_smd_zero_variance_different_means(self):
        """
        SMD → ∞ when both groups have zero variance but different means (perfect separation).

        Hand calculation:
        - x_t = [5, 5], mean = 5.0, var = 0.0
        - x_c = [2, 2], mean = 2.0, var = 0.0
        - pooled_std → 0, means differ → SMD = sign(5-2) * 1e6 = +1e6
        """
        x_t = np.array([5.0, 5.0])
        x_c = np.array([2.0, 2.0])

        smd = compute_smd(x_t, x_c, pooled=True)

        assert smd == 1e6, f"Expected SMD = 1e6 (proxy for +Inf), got {smd}"

    def test_smd_zero_variance_unpooled(self):
        """
        Unpooled SMD → ∞ when treated group has zero variance but different mean.

        Hand calculation:
        - x_t = [3, 3], mean = 3.0, var = 0.0
        - x_c = [1, 5], mean = 3.0, var = 8.0
        - Means equal but var_t = 0 → SMD = 0 (special case: identical means)
        """
        x_t = np.array([3.0, 3.0])
        x_c = np.array([1.0, 5.0])

        smd = compute_smd(x_t, x_c, pooled=False)

        assert smd == 0.0, f"Expected SMD = 0 (identical means, zero treated variance), got {smd}"


class TestComputeVarianceRatio:
    """Unit tests for compute_variance_ratio() with hand-calculated expected values."""

    def test_vr_equal_variances(self):
        """
        VR = 1.0 when groups have equal variances.

        Hand calculation:
        - x_t = [1, 3, 5], var = 4.0
        - x_c = [2, 4, 6], var = 4.0
        - VR = 4.0 / 4.0 = 1.0
        """
        x_t = np.array([1.0, 3.0, 5.0])
        x_c = np.array([2.0, 4.0, 6.0])

        vr = compute_variance_ratio(x_t, x_c)

        assert abs(vr - 1.0) < 1e-10, f"Expected VR = 1.0, got {vr}"

    def test_vr_known_value(self):
        """
        VR with simple known answer.

        Hand calculation:
        - x_t = [1, 5], mean = 3, var = 8.0
        - x_c = [2, 4], mean = 3, var = 2.0
        - VR = 8.0 / 2.0 = 4.0
        """
        x_t = np.array([1.0, 5.0])
        x_c = np.array([2.0, 4.0])

        vr = compute_variance_ratio(x_t, x_c)

        assert abs(vr - 4.0) < 1e-10, f"Expected VR = 4.0, got {vr}"

    def test_vr_both_zero_variance(self):
        """
        VR = 1.0 when both groups have zero variance.

        Hand calculation:
        - x_t = [3, 3, 3], var = 0.0
        - x_c = [7, 7, 7], var = 0.0
        - VR = 0.0 / 0.0 → 1.0 (convention)
        """
        x_t = np.array([3.0, 3.0, 3.0])
        x_c = np.array([7.0, 7.0, 7.0])

        vr = compute_variance_ratio(x_t, x_c)

        assert vr == 1.0, f"Expected VR = 1.0 (both zero variance), got {vr}"

    def test_vr_control_zero_variance(self):
        """
        VR → ∞ when only control group has zero variance.

        Hand calculation:
        - x_t = [1, 5], var = 8.0
        - x_c = [3, 3], var = 0.0
        - VR = 8.0 / 0.0 → ∞
        """
        x_t = np.array([1.0, 5.0])
        x_c = np.array([3.0, 3.0])

        vr = compute_variance_ratio(x_t, x_c)

        assert np.isinf(vr), f"Expected VR = ∞ (control zero variance), got {vr}"


class TestCheckCovariateBalance:
    """Unit tests for check_covariate_balance() with known examples."""

    def test_balance_perfect_matched(self):
        """
        Perfect balance after matching (all SMD = 0).

        Setup:
        - 6 units: T = [1,1,1,0,0,0]
        - 2 covariates: X1 = [1,2,3,1,2,3], X2 = [4,5,6,4,5,6]
        - Matched pairs: [(0,3), (1,4), (2,5)] (perfectly matched on both covariates)

        Expected:
        - balanced = True (all |SMD| < 0.1)
        - smd_after = [0, 0] (exact match)
        - vr_after = [1.0, 1.0] (equal variances)
        """
        covariates = np.array(
            [
                [1, 4],  # Treated 0
                [2, 5],  # Treated 1
                [3, 6],  # Treated 2
                [1, 4],  # Control 3
                [2, 5],  # Control 4
                [3, 6],  # Control 5
            ]
        )
        treatment = np.array([True, True, True, False, False, False])
        matched_indices = [(0, 3), (1, 4), (2, 5)]

        balanced, smd_after, vr_after, smd_before, vr_before = check_covariate_balance(
            covariates, treatment, matched_indices, threshold=0.1
        )

        assert balanced == True, "Expected perfect balance"
        assert np.allclose(smd_after, 0.0, atol=1e-10), f"Expected SMD = 0, got {smd_after}"
        assert np.allclose(vr_after, 1.0, atol=1e-10), f"Expected VR = 1, got {vr_after}"

    def test_balance_imbalanced(self):
        """
        Poor balance after matching (some SMD > threshold).

        Setup:
        - 6 units: T = [1,1,1,0,0,0]
        - 1 covariate: X = [5,6,7,1,2,3] (large initial imbalance)
        - Matched pairs: [(0,5), (1,4), (2,3)] (still imbalanced: SMD ≠ 0)

        Expected:
        - balanced = False (SMD > 0.1)
        - smd_after > 0.1 (residual imbalance remains)
        """
        covariates = np.array([5, 6, 7, 1, 2, 3]).reshape(-1, 1)
        treatment = np.array([True, True, True, False, False, False])
        matched_indices = [(0, 5), (1, 4), (2, 3)]  # Pairs: (5,3), (6,2), (7,1)

        balanced, smd_after, vr_after, smd_before, vr_before = check_covariate_balance(
            covariates, treatment, matched_indices, threshold=0.1
        )

        # After matching: x_t_matched = [5,6,7], x_c_matched = [3,2,1]
        # Mean difference = (5+6+7)/3 - (3+2+1)/3 = 6 - 2 = 4
        # This should have large SMD
        assert balanced == False, "Expected imbalance (SMD > 0.1)"
        assert abs(smd_after[0]) > 0.1, f"Expected |SMD| > 0.1, got {abs(smd_after[0])}"

    def test_balance_no_matches(self):
        """
        No matches → should return NaN for after-matching metrics.

        Setup:
        - 4 units with matched_indices = []

        Expected:
        - balanced = False
        - smd_after = [NaN]
        - vr_after = [NaN]
        - smd_before, vr_before computed normally
        """
        covariates = np.array([1, 2, 3, 4]).reshape(-1, 1)
        treatment = np.array([True, True, False, False])
        matched_indices = []

        balanced, smd_after, vr_after, smd_before, vr_before = check_covariate_balance(
            covariates, treatment, matched_indices, threshold=0.1
        )

        assert balanced is False
        assert np.all(np.isnan(smd_after)), f"Expected NaN for smd_after, got {smd_after}"
        assert np.all(np.isnan(vr_after)), f"Expected NaN for vr_after, got {vr_after}"
        assert np.all(np.isfinite(smd_before)), "Expected finite smd_before"


class TestBalanceSummary:
    """Unit tests for balance_summary() with known statistics."""

    def test_summary_all_balanced(self):
        """
        All covariates balanced (|SMD| < 0.1).

        Hand calculation:
        - smd_before = [0.5, 0.8, 1.2]
        - smd_after = [0.05, 0.08, 0.09]
        - All |smd_after| < 0.1 → all_balanced = True
        - n_balanced = 3, n_imbalanced = 0
        - mean_smd_before = (0.5 + 0.8 + 1.2) / 3 = 0.833
        - mean_smd_after = (0.05 + 0.08 + 0.09) / 3 = 0.073
        - improvement = (0.833 - 0.073) / 0.833 = 0.912 (91.2%)
        """
        smd_before = np.array([0.5, 0.8, 1.2])
        smd_after = np.array([0.05, 0.08, 0.09])
        vr_before = np.array([1.5, 2.0, 3.0])
        vr_after = np.array([1.1, 1.2, 1.3])

        summary = balance_summary(smd_after, vr_after, smd_before, vr_before, threshold=0.1)

        assert summary["n_covariates"] == 3
        assert summary["n_balanced"] == 3
        assert summary["n_imbalanced"] == 0
        assert summary["all_balanced"] is True
        assert abs(summary["mean_smd_before"] - 0.833333) < 1e-5
        assert abs(summary["mean_smd_after"] - 0.073333) < 1e-5
        assert abs(summary["improvement"] - 0.912) < 1e-3

    def test_summary_some_imbalanced(self):
        """
        Some covariates imbalanced.

        Hand calculation:
        - smd_before = [0.6, 0.9]
        - smd_after = [0.05, 0.15]  # Second covariate: |0.15| ≥ 0.1
        - n_balanced = 1, n_imbalanced = 1
        - all_balanced = False
        """
        smd_before = np.array([0.6, 0.9])
        smd_after = np.array([0.05, 0.15])
        vr_before = np.array([1.8, 2.2])
        vr_after = np.array([1.1, 1.4])

        summary = balance_summary(smd_after, vr_after, smd_before, vr_before, threshold=0.1)

        assert summary["n_balanced"] == 1
        assert summary["n_imbalanced"] == 1
        assert summary["all_balanced"] is False

    def test_summary_no_improvement(self):
        """
        Balance worsened after matching (improvement < 0).

        Hand calculation:
        - smd_before = [0.1, 0.2]
        - smd_after = [0.3, 0.4]  # Worse balance
        - mean_smd_before = 0.15, mean_smd_after = 0.35
        - improvement = (0.15 - 0.35) / 0.15 = -1.333 (negative improvement)
        """
        smd_before = np.array([0.1, 0.2])
        smd_after = np.array([0.3, 0.4])
        vr_before = np.array([1.1, 1.2])
        vr_after = np.array([1.5, 1.8])

        summary = balance_summary(smd_after, vr_after, smd_before, vr_before, threshold=0.1)

        assert summary["improvement"] < 0, "Expected negative improvement (balance worsened)"


class TestBalanceEdgeCases:
    """Edge cases and error handling for balance functions."""

    def test_check_balance_length_mismatch(self):
        """check_covariate_balance() should raise on length mismatch."""
        covariates = np.array([[1, 2], [3, 4]])
        treatment = np.array([True, False, True])  # Wrong length!
        matched_indices = [(0, 1)]

        with pytest.raises(ValueError, match="CRITICAL ERROR: Mismatched lengths"):
            check_covariate_balance(covariates, treatment, matched_indices)

    def test_check_balance_invalid_threshold(self):
        """check_covariate_balance() should raise on threshold ≤ 0."""
        covariates = np.array([[1], [2]])
        treatment = np.array([True, False])
        matched_indices = [(0, 1)]

        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid threshold"):
            check_covariate_balance(covariates, treatment, matched_indices, threshold=0.0)

        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid threshold"):
            check_covariate_balance(covariates, treatment, matched_indices, threshold=-0.1)
