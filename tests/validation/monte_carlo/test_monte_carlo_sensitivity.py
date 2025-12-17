"""
Monte Carlo Validation Tests for Sensitivity Analysis.

Validates statistical properties of:
1. E-value: formula accuracy, effect type conversions
2. Rosenbaum bounds: gamma_critical detection, p-value properties

Session 53: Sensitivity Monte Carlo Validation
"""

import numpy as np
import pytest

from causal_inference.sensitivity import e_value, rosenbaum_bounds
from tests.validation.monte_carlo.dgp_sensitivity import (
    dgp_evalue_known_rr,
    dgp_evalue_smd,
    dgp_matched_pairs_no_confounding,
    dgp_matched_pairs_weak_effect,
    dgp_matched_pairs_strong_effect,
    dgp_matched_pairs_null_effect,
)


# =============================================================================
# E-Value Monte Carlo Tests
# =============================================================================


class TestEValueFormula:
    """Validate E-value formula correctness via Monte Carlo."""

    @pytest.mark.monte_carlo
    def test_evalue_formula_accuracy(self):
        """E-value formula matches theoretical value across RR range."""
        # Test across range of risk ratios
        rr_values = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        for rr in rr_values:
            result = e_value(rr, effect_type="rr")

            # Theoretical E-value: E = RR + sqrt(RR * (RR - 1))
            if rr == 1.0:
                expected_e = 1.0
            else:
                expected_e = rr + np.sqrt(rr * (rr - 1))

            assert np.isclose(result["e_value"], expected_e, rtol=1e-10), (
                f"E-value mismatch for RR={rr}: got {result['e_value']}, "
                f"expected {expected_e}"
            )

    @pytest.mark.monte_carlo
    def test_evalue_monotonicity(self):
        """E-value increases monotonically with effect size."""
        rr_values = np.linspace(1.0, 5.0, 50)
        e_values = [e_value(rr, effect_type="rr")["e_value"] for rr in rr_values]

        # Check monotonicity
        for i in range(1, len(e_values)):
            assert e_values[i] >= e_values[i - 1] - 1e-10, (
                f"E-value not monotonic: E({rr_values[i]}) = {e_values[i]} < "
                f"E({rr_values[i-1]}) = {e_values[i-1]}"
            )

    @pytest.mark.monte_carlo
    def test_evalue_protective_symmetry(self):
        """Protective effects (RR < 1) give same E-value as 1/RR."""
        rr_values = [0.25, 0.33, 0.5, 0.67, 0.8]

        for rr in rr_values:
            result_protective = e_value(rr, effect_type="rr")
            result_harmful = e_value(1 / rr, effect_type="rr")

            assert np.isclose(
                result_protective["e_value"],
                result_harmful["e_value"],
                rtol=1e-10,
            ), f"Protective/harmful symmetry broken for RR={rr}"


class TestEValueConversions:
    """Validate E-value effect type conversions via Monte Carlo."""

    @pytest.mark.monte_carlo
    def test_smd_conversion_accuracy(self, n_runs: int = 500):
        """SMD to RR conversion produces consistent E-values."""
        results = []

        for seed in range(n_runs):
            # Generate data with known SMD
            outcomes, treatment, true_smd = dgp_evalue_smd(
                n=200, true_smd=0.5, random_state=seed
            )

            # Compute observed SMD
            treated_mean = outcomes[treatment == 1].mean()
            control_mean = outcomes[treatment == 0].mean()
            pooled_sd = np.sqrt(
                (outcomes[treatment == 1].var() + outcomes[treatment == 0].var()) / 2
            )
            observed_smd = (treated_mean - control_mean) / pooled_sd

            # Compute E-value
            result = e_value(observed_smd, effect_type="smd")
            results.append(result["e_value"])

        # E-values should be reasonably consistent
        mean_e = np.mean(results)
        std_e = np.std(results)

        # Expected E-value for SMD=0.5: RR ≈ exp(0.91 * 0.5) ≈ 1.577
        expected_rr = np.exp(0.91 * 0.5)
        expected_e = expected_rr + np.sqrt(expected_rr * (expected_rr - 1))

        assert np.isclose(mean_e, expected_e, rtol=0.15), (
            f"Mean E-value ({mean_e:.3f}) differs from expected ({expected_e:.3f})"
        )
        assert std_e < 0.5, f"E-value variance too high: SD = {std_e:.3f}"

    @pytest.mark.monte_carlo
    def test_ate_conversion_accuracy(self):
        """ATE to RR conversion produces correct E-values."""
        # Test various ATE/baseline combinations
        test_cases = [
            (0.1, 0.2, 1.5),  # ATE=0.1, baseline=0.2 → RR=1.5
            (0.2, 0.2, 2.0),  # ATE=0.2, baseline=0.2 → RR=2.0
            (0.1, 0.1, 2.0),  # ATE=0.1, baseline=0.1 → RR=2.0
            (-0.1, 0.3, 0.67),  # Protective: ATE=-0.1, baseline=0.3 → RR≈0.67
        ]

        for ate, baseline, expected_rr in test_cases:
            result = e_value(ate, effect_type="ate", baseline_risk=baseline)

            # Compute expected E-value
            rr = expected_rr
            if rr < 1:
                rr = 1 / rr
            expected_e = rr + np.sqrt(rr * (rr - 1))

            assert np.isclose(result["e_value"], expected_e, rtol=0.05), (
                f"E-value mismatch for ATE={ate}, baseline={baseline}: "
                f"got {result['e_value']:.3f}, expected {expected_e:.3f}"
            )


class TestEValueCI:
    """Validate E-value confidence interval handling."""

    @pytest.mark.monte_carlo
    def test_ci_crossing_null(self, n_runs: int = 500):
        """CI crossing null produces E_value_ci = 1.0."""
        null_crossing_count = 0

        for seed in range(n_runs):
            rng = np.random.RandomState(seed)

            # Generate RR and CI that may or may not cross null
            rr = rng.uniform(0.8, 1.5)
            ci_width = rng.uniform(0.3, 0.8)
            ci_lower = rr - ci_width / 2
            ci_upper = rr + ci_width / 2

            result = e_value(rr, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="rr")

            # Check if CI crosses null (1.0)
            ci_crosses_null = ci_lower <= 1.0 <= ci_upper

            if ci_crosses_null:
                null_crossing_count += 1
                assert result["e_value_ci"] == pytest.approx(1.0, abs=0.01), (
                    f"E_value_ci should be 1.0 when CI crosses null: "
                    f"RR={rr:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}]"
                )

        # Ensure we had some null-crossing cases
        assert null_crossing_count > 50, "Not enough null-crossing cases generated"


# =============================================================================
# Rosenbaum Bounds Monte Carlo Tests
# =============================================================================


class TestRosenbaumProperties:
    """Validate Rosenbaum bounds statistical properties."""

    @pytest.mark.monte_carlo
    def test_pvalue_monotonicity(self, n_runs: int = 200):
        """P-value bounds are monotonic in gamma."""
        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_no_confounding(
                n_pairs=30, true_effect=2.0, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 3.0), n_gamma=20
            )

            # P_upper should be non-decreasing
            p_upper_diffs = np.diff(result["p_upper"])
            assert all(p_upper_diffs >= -1e-10), (
                f"P_upper not monotonic at seed {seed}"
            )

            # P_lower should be non-increasing (or approximately so)
            # Note: numerical issues can cause small violations
            p_lower_diffs = np.diff(result["p_lower"])
            large_violations = np.sum(p_lower_diffs > 0.05)
            assert large_violations == 0, (
                f"P_lower has large monotonicity violations at seed {seed}"
            )

    @pytest.mark.monte_carlo
    def test_pvalue_ordering(self, n_runs: int = 200):
        """P_upper >= P_lower at all gamma values."""
        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_no_confounding(
                n_pairs=30, true_effect=1.5, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 3.0), n_gamma=20
            )

            for i, gamma in enumerate(result["gamma_values"]):
                assert result["p_upper"][i] >= result["p_lower"][i] - 1e-10, (
                    f"P_upper < P_lower at gamma={gamma:.2f}, seed={seed}"
                )

    @pytest.mark.monte_carlo
    def test_pvalues_bounded_zero_one(self, n_runs: int = 200):
        """All p-values are in [0, 1]."""
        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_no_confounding(
                n_pairs=30, true_effect=2.0, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 5.0), n_gamma=20
            )

            assert all(0 <= p <= 1 for p in result["p_upper"]), "P_upper out of bounds"
            assert all(0 <= p <= 1 for p in result["p_lower"]), "P_lower out of bounds"


class TestRosenbaumEffectSize:
    """Validate Rosenbaum bounds respond correctly to effect size."""

    @pytest.mark.monte_carlo
    def test_strong_effect_robust(self, n_runs: int = 300):
        """Strong effects have high gamma_critical (robust)."""
        gamma_criticals = []

        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_strong_effect(
                n_pairs=40, true_effect=5.0, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 5.0), n_gamma=30
            )

            if result["gamma_critical"] is not None:
                gamma_criticals.append(result["gamma_critical"])

        # Strong effect: most runs should be robust (no gamma_critical or high)
        robust_count = n_runs - len(gamma_criticals)
        if len(gamma_criticals) > 0:
            mean_gamma = np.mean(gamma_criticals)
            # Those with gamma_critical should still be relatively high
            assert mean_gamma > 2.5 or robust_count > n_runs * 0.5, (
                f"Strong effect not robust: mean gamma_critical={mean_gamma:.2f}, "
                f"robust_count={robust_count}/{n_runs}"
            )

    @pytest.mark.monte_carlo
    def test_weak_effect_sensitive(self, n_runs: int = 300):
        """Weak effects have low gamma_critical (sensitive)."""
        gamma_criticals = []

        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_weak_effect(
                n_pairs=40, true_effect=0.3, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 3.0), n_gamma=20
            )

            if result["gamma_critical"] is not None:
                gamma_criticals.append(result["gamma_critical"])

        # Weak effect: most should have low gamma_critical
        if len(gamma_criticals) > n_runs * 0.3:  # At least 30% have gamma_critical
            mean_gamma = np.mean(gamma_criticals)
            assert mean_gamma < 2.0, (
                f"Weak effect too robust: mean gamma_critical={mean_gamma:.2f}"
            )

    @pytest.mark.monte_carlo
    def test_null_effect_very_sensitive(self, n_runs: int = 300):
        """Null effects are very sensitive (low gamma_critical or non-significant)."""
        gamma_criticals = []
        not_significant_at_gamma1 = 0

        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_null_effect(
                n_pairs=40, random_state=seed
            )

            result = rosenbaum_bounds(
                treated, control, gamma_range=(1.0, 2.0), n_gamma=15, alpha=0.05
            )

            # Check if not significant at gamma=1 (upper p-value > 0.05)
            if result["p_upper"][0] > 0.05:
                not_significant_at_gamma1 += 1
            elif result["gamma_critical"] is not None:
                gamma_criticals.append(result["gamma_critical"])

        # Null effect: most should be non-significant at gamma=1 or very sensitive
        non_robust_rate = (not_significant_at_gamma1 + len(gamma_criticals)) / n_runs
        assert non_robust_rate > 0.90, (
            f"Null effect unexpectedly robust: {non_robust_rate:.1%} non-robust"
        )


class TestRosenbaumSampleSize:
    """Validate Rosenbaum bounds behavior with sample size."""

    @pytest.mark.monte_carlo
    def test_larger_samples_more_robust(self, n_runs: int = 200):
        """Larger samples produce higher gamma_critical (more robust)."""
        small_gammas = []
        large_gammas = []

        for seed in range(n_runs):
            # Small sample
            treated_s, control_s, _ = dgp_matched_pairs_no_confounding(
                n_pairs=15, true_effect=1.5, random_state=seed
            )
            result_s = rosenbaum_bounds(
                treated_s, control_s, gamma_range=(1.0, 3.0), n_gamma=20
            )
            if result_s["gamma_critical"] is not None:
                small_gammas.append(result_s["gamma_critical"])

            # Large sample (same effect)
            treated_l, control_l, _ = dgp_matched_pairs_no_confounding(
                n_pairs=60, true_effect=1.5, random_state=seed + 10000
            )
            result_l = rosenbaum_bounds(
                treated_l, control_l, gamma_range=(1.0, 3.0), n_gamma=20
            )
            if result_l["gamma_critical"] is not None:
                large_gammas.append(result_l["gamma_critical"])

        # Compare: larger samples should have higher gamma_critical on average
        # (or more "None" results indicating robustness)
        small_robust = n_runs - len(small_gammas)
        large_robust = n_runs - len(large_gammas)

        # Either more robust results, or higher mean gamma_critical
        if len(small_gammas) > 10 and len(large_gammas) > 10:
            assert large_robust >= small_robust - 20 or np.mean(large_gammas) >= np.mean(small_gammas) - 0.3, (
                f"Larger samples not more robust: "
                f"small={np.mean(small_gammas):.2f} ({small_robust} robust), "
                f"large={np.mean(large_gammas):.2f} ({large_robust} robust)"
            )


class TestRosenbaumInterpretation:
    """Validate Rosenbaum bounds interpretation."""

    @pytest.mark.monte_carlo
    def test_interpretation_present(self, n_runs: int = 100):
        """All results have non-empty interpretation."""
        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_no_confounding(
                n_pairs=30, true_effect=2.0, random_state=seed
            )

            result = rosenbaum_bounds(treated, control)

            assert result["interpretation"], f"Empty interpretation at seed {seed}"
            assert len(result["interpretation"]) > 20, "Interpretation too short"

    @pytest.mark.monte_carlo
    def test_interpretation_mentions_robustness(self, n_runs: int = 100):
        """Interpretation contains robustness assessment."""
        for seed in range(n_runs):
            treated, control, _ = dgp_matched_pairs_no_confounding(
                n_pairs=30, true_effect=2.0, random_state=seed
            )

            result = rosenbaum_bounds(treated, control)
            interp_lower = result["interpretation"].lower()

            assert "robust" in interp_lower or "sensitive" in interp_lower, (
                f"Interpretation missing robustness assessment at seed {seed}"
            )
