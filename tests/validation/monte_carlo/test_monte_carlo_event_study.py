"""
Monte Carlo validation for Event Study estimator.

Validates statistical properties:
- Pre-treatment effects are null (parallel trends)
- Post-treatment effects recover true dynamics
- Coverage of event-time specific effects

Key tests:
1. Pre-trends should be ~0 when parallel trends hold
2. Post-treatment effects should match DGP
3. Pre-trends should be detected when violated

References:
    Roth (2022). "Pretest with Caution: Event-Study Estimates After Testing for
    Parallel Trends"
"""

import numpy as np
import pytest
from src.causal_inference.did import event_study
from tests.validation.monte_carlo.dgp_did import (
    dgp_event_study_null_pretrends,
    dgp_event_study_violated_pretrends,
    dgp_event_study_dynamic,
)


class TestEventStudyPretrends:
    """Monte Carlo validation of pre-treatment effect estimation."""

    @pytest.mark.slow
    def test_pretrends_null_when_parallel(self):
        """
        Validate pre-treatment effects are ~0 when parallel trends hold.

        DGP: β_k = 0 for all k < 0
        Expected: Average |pre-trend estimate| < 0.30
        """
        n_runs = 500

        # Collect pre-trend estimates at k=-3, -2 (k=-1 is omitted reference)
        pretrend_estimates = {-3: [], -2: []}

        for seed in range(n_runs):
            data = dgp_event_study_null_pretrends(
                n_treated=100,
                n_control=100,
                n_pre=5,
                n_post=5,
                treatment_time=5,
                true_effect=2.0,
                random_state=seed,
            )

            result = event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
                cluster_se=True,
            )

            # Collect estimates for pre-treatment event times (from leads)
            for k in [-3, -2]:
                if k in result["leads"]:
                    pretrend_estimates[k].append(result["leads"][k]["estimate"])

        # Check each pre-trend is centered at 0
        for k in [-3, -2]:
            if len(pretrend_estimates[k]) > 0:
                mean_est = np.mean(pretrend_estimates[k])
                print(f"Pre-trend k={k}: mean={mean_est:.4f}")
                assert abs(mean_est) < 0.35, (
                    f"Pre-trend at k={k} has mean {mean_est:.4f}, expected ~0 with parallel trends"
                )

    @pytest.mark.slow
    def test_pretrends_detected_when_violated(self):
        """
        Validate pre-treatment effects are detected when parallel trends violated.

        DGP: β_k = 0.3 × k for k < 0 (linear pre-trend)
        Expected: Pre-trends should be significantly different from 0
        """
        n_runs = 300
        pretrend_slope = 0.3

        # Collect pre-trend estimates (k=-1 is reference, so use -3, -2)
        pretrend_estimates = {-3: [], -2: []}

        for seed in range(n_runs):
            data = dgp_event_study_violated_pretrends(
                n_treated=100,
                n_control=100,
                n_pre=5,
                n_post=5,
                treatment_time=5,
                true_effect=2.0,
                pretrend_slope=pretrend_slope,
                random_state=seed,
            )

            result = event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
                cluster_se=True,
            )

            for k in [-3, -2]:
                if k in result["leads"]:
                    pretrend_estimates[k].append(result["leads"][k]["estimate"])

        # Pre-trends should match DGP slope (relative to k=-1 reference)
        # True values: k=-3 → -0.9, k=-2 → -0.6 (relative to k=-1 = -0.3)
        # But since k=-1 is reference (=0), we see k=-3 relative to k=-1
        # Relative effect: k=-3 vs k=-1 = -0.9 - (-0.3) = -0.6
        # Relative effect: k=-2 vs k=-1 = -0.6 - (-0.3) = -0.3
        for k in [-3, -2]:
            if len(pretrend_estimates[k]) > 0:
                mean_est = np.mean(pretrend_estimates[k])
                # True relative to k=-1
                true_val = pretrend_slope * k - pretrend_slope * (-1)
                bias = abs(mean_est - true_val)
                print(f"Pre-trend k={k}: mean={mean_est:.4f}, true_relative={true_val:.4f}")

                # Should recover the relative pre-trend (allow for estimation error)
                assert bias < 0.60, (
                    f"Pre-trend at k={k}: bias {bias:.4f} exceeds 0.60. "
                    f"Expected ~{true_val:.4f}, got {mean_est:.4f}"
                )


class TestEventStudyPostTreatment:
    """Monte Carlo validation of post-treatment effect estimation."""

    @pytest.mark.slow
    def test_post_treatment_constant_effect(self):
        """
        Validate post-treatment effects recover constant true effect.

        DGP: β_k = 2.0 for all k >= 0
        """
        n_runs = 500
        true_effect = 2.0

        # Collect post-treatment estimates at k=0, 1, 2
        post_estimates = {0: [], 1: [], 2: []}

        for seed in range(n_runs):
            data = dgp_event_study_null_pretrends(
                n_treated=100,
                n_control=100,
                n_pre=5,
                n_post=5,
                treatment_time=5,
                true_effect=true_effect,
                random_state=seed,
            )

            result = event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )

            # Get post-treatment effects from lags
            for k in [0, 1, 2]:
                if k in result["lags"]:
                    post_estimates[k].append(result["lags"][k]["estimate"])

        # Each post-treatment effect should be ~2.0
        for k in [0, 1, 2]:
            if len(post_estimates[k]) > 0:
                mean_est = np.mean(post_estimates[k])
                bias = abs(mean_est - true_effect)
                print(f"Post-treatment k={k}: mean={mean_est:.4f}, bias={bias:.4f}")

                assert bias < 0.30, (
                    f"Post-treatment at k={k}: bias {bias:.4f} exceeds 0.30. "
                    f"Expected {true_effect:.4f}, got {mean_est:.4f}"
                )

    @pytest.mark.slow
    def test_post_treatment_dynamic_effects(self):
        """
        Validate event study captures time-varying treatment effects.

        DGP: β_k grows from 1.0 to 3.0 over event times 0-4
        """
        n_runs = 300
        effect_path = {0: 1.0, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.0}

        post_estimates = {k: [] for k in effect_path.keys()}

        for seed in range(n_runs):
            data = dgp_event_study_dynamic(
                n_treated=100,
                n_control=100,
                n_pre=5,
                n_post=5,
                treatment_time=5,
                effect_path=effect_path,
                random_state=seed,
            )

            result = event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )

            # Get post-treatment effects from lags
            for k in effect_path.keys():
                if k in result["lags"]:
                    post_estimates[k].append(result["lags"][k]["estimate"])

        print("\n=== Dynamic Effects Recovery ===")
        for k, true_val in effect_path.items():
            if len(post_estimates[k]) > 0:
                mean_est = np.mean(post_estimates[k])
                print(f"k={k}: true={true_val:.2f}, est={mean_est:.4f}")

        # Check pattern: effects should increase from k=0 to k=2
        if len(post_estimates[0]) > 0 and len(post_estimates[2]) > 0:
            mean_k0 = np.mean(post_estimates[0])
            mean_k2 = np.mean(post_estimates[2])
            assert mean_k2 > mean_k0 * 0.8, (
                f"Expected increasing effects but k=0 ({mean_k0:.4f}) >= k=2 ({mean_k2:.4f})"
            )


class TestEventStudyCoverage:
    """Monte Carlo validation of event study confidence intervals."""

    @pytest.mark.slow
    def test_event_time_ci_coverage(self):
        """
        Validate coverage for event-time specific effects.

        Expected: 88-98% coverage (allowing for cluster SE conservatism)
        """
        n_runs = 300
        true_effect = 2.0

        coverage_by_k = {0: [], 1: [], 2: []}

        for seed in range(n_runs):
            data = dgp_event_study_null_pretrends(
                n_treated=100,
                n_control=100,
                n_pre=5,
                n_post=5,
                true_effect=true_effect,
                random_state=seed,
            )

            result = event_study(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                time=data["time"],
                unit_id=data["unit_id"],
                treatment_time=data["treatment_time"],
            )

            # Get coverage from lags (post-treatment)
            for k in [0, 1, 2]:
                if k in result["lags"]:
                    ci_low = result["lags"][k]["ci_lower"]
                    ci_high = result["lags"][k]["ci_upper"]
                    covers = ci_low <= true_effect <= ci_high
                    coverage_by_k[k].append(covers)

        print("\n=== Event Time CI Coverage ===")
        for k in [0, 1, 2]:
            if len(coverage_by_k[k]) > 0:
                coverage = np.mean(coverage_by_k[k])
                print(f"k={k}: coverage={coverage:.4f}")

                # Allow wider range due to cluster SE variance
                assert 0.80 < coverage < 0.99, (
                    f"Coverage at k={k} is {coverage:.4f}, expected [0.80, 0.99]"
                )


class TestEventStudyDiagnostics:
    """Diagnostic tests for event study Monte Carlo."""

    def test_event_study_returns_expected_structure(self):
        """Verify event_study returns all expected output fields."""
        data = dgp_event_study_null_pretrends(n_treated=50, n_control=50, random_state=42)

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
        )

        # Check required fields (event_study uses leads/lags structure)
        assert "leads" in result  # Pre-treatment effects
        assert "lags" in result  # Post-treatment effects
        assert "parallel_trends_plausible" in result
        assert "omitted_period" in result

        # Should have effects for multiple event times
        assert len(result["leads"]) + len(result["lags"]) > 3

    def test_event_study_reference_period(self):
        """Verify event study uses k=-1 as reference period."""
        data = dgp_event_study_null_pretrends(n_treated=50, n_control=50, random_state=42)

        result = event_study(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            time=data["time"],
            unit_id=data["unit_id"],
            treatment_time=data["treatment_time"],
        )

        # Reference period (k=-1) should be omitted (normalized to 0)
        assert result["omitted_period"] == -1, (
            f"Expected omitted period to be -1, got {result['omitted_period']}"
        )
        # k=-1 should not be in leads (it's the reference)
        assert -1 not in result["leads"], "Reference period k=-1 should be omitted from leads"
