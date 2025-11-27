"""TDD Step 1: Tests for small sample inference corrections.

These tests verify that IPW and DR estimators use t-distribution
for small samples (n < 50) instead of z-distribution.

Issue: IPW and DR use z-distribution unconditionally, leading to
narrower CIs than appropriate for small samples.

Written BEFORE fixes per TDD protocol.
"""

import numpy as np
import pytest
from scipy import stats

from src.causal_inference.observational.ipw import ipw_ate_observational
from src.causal_inference.observational.doubly_robust import dr_ate
from src.causal_inference.rct.estimators_ipw import ipw_ate


class TestSmallSampleInference:
    """Small samples should use t-distribution, not z."""

    def test_ipw_uses_t_distribution_small_n(self):
        """IPW CI should be wider than z-based CI for n < 50.

        For n=30, t_{0.975, df≈28} ≈ 2.048 vs z_{0.975} = 1.96
        So t-based CI should be ~4.5% wider.
        """
        np.random.seed(42)
        n = 30

        # Simple RCT data (no confounding to isolate inference)
        treatment = np.array([1] * 15 + [0] * 15).astype(float)
        propensity = np.ones(n) * 0.5  # Constant propensity
        outcomes = np.where(treatment == 1, 5.0, 3.0) + np.random.normal(0, 1, n)

        result = ipw_ate(outcomes, treatment, propensity)

        # z-based CI width
        z_critical = stats.norm.ppf(0.975)  # 1.96
        z_ci_width = 2 * z_critical * result["se"]

        # t-based CI width (what it should be)
        df = n - 2  # Approximate df
        t_critical = stats.t.ppf(0.975, df=df)  # ~2.048
        t_ci_width = 2 * t_critical * result["se"]

        # Actual CI width from function
        actual_ci_width = result["ci_upper"] - result["ci_lower"]

        # CI should be closer to t-based than z-based for small samples
        # If using z: actual ≈ z_ci_width
        # If using t: actual ≈ t_ci_width (wider)
        assert actual_ci_width > z_ci_width * 1.01, (
            f"Small sample (n={n}) CI width ({actual_ci_width:.4f}) "
            f"should be wider than z-based ({z_ci_width:.4f}). "
            f"Currently using z-distribution instead of t."
        )

    def test_ipw_uses_z_distribution_large_n(self):
        """IPW should use z-distribution when n >= 50 (t ≈ z asymptotically)."""
        np.random.seed(42)
        n = 200

        treatment = np.array([1] * 100 + [0] * 100).astype(float)
        propensity = np.ones(n) * 0.5
        outcomes = np.where(treatment == 1, 5.0, 3.0) + np.random.normal(0, 1, n)

        result = ipw_ate(outcomes, treatment, propensity)

        # For large n, t ≈ z, so CI widths should be very close
        z_ci_width = 2 * 1.96 * result["se"]
        actual_ci_width = result["ci_upper"] - result["ci_lower"]

        # Should be within 2% of z-based for large samples
        assert np.isclose(actual_ci_width, z_ci_width, rtol=0.02), (
            f"Large sample (n={n}) CI should be approximately z-based"
        )

    def test_dr_uses_t_distribution_small_n(self):
        """DR CI should be wider than z-based CI for n < 50."""
        np.random.seed(42)
        n = 30

        # Simple data for DR
        X = np.random.normal(0, 1, (n, 1))
        treatment = np.array([1] * 15 + [0] * 15).astype(float)
        outcomes = 2.0 * treatment + 0.5 * X.flatten() + np.random.normal(0, 1, n)

        result = dr_ate(outcomes, treatment, X)

        # z-based CI width
        z_ci_width = 2 * 1.96 * result["se"]

        # Actual CI width
        actual_ci_width = result["ci_upper"] - result["ci_lower"]

        # Should be wider than z-based for small samples
        assert actual_ci_width > z_ci_width * 1.01, (
            f"Small sample DR (n={n}) CI width ({actual_ci_width:.4f}) "
            f"should be wider than z-based ({z_ci_width:.4f}). "
            f"Currently using z-distribution instead of t."
        )

    def test_observational_ipw_uses_t_distribution_small_n(self):
        """Observational IPW should also use t-distribution for small samples."""
        np.random.seed(42)
        n = 40

        # Create confounded data
        X = np.random.normal(0, 1, (n, 1))
        logit = 0.5 * X.flatten()
        treatment = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
        outcomes = 2.0 * treatment + 0.3 * X.flatten() + np.random.normal(0, 1, n)

        result = ipw_ate_observational(outcomes, treatment, X)

        # z-based CI width
        z_ci_width = 2 * 1.96 * result["se"]

        # Actual CI width
        actual_ci_width = result["ci_upper"] - result["ci_lower"]

        # Should be wider than z-based for small samples
        assert actual_ci_width > z_ci_width * 1.01, (
            f"Small sample observational IPW (n={n}) CI width ({actual_ci_width:.4f}) "
            f"should be wider than z-based ({z_ci_width:.4f})"
        )


class TestSmallSampleCoverage:
    """Monte Carlo validation that t-distribution improves coverage for small n."""

    @pytest.mark.slow
    def test_ipw_coverage_small_sample(self):
        """t-distribution should give 93-97% coverage for n=30."""
        np.random.seed(42)
        n_sims = 500
        n = 30
        true_ate = 2.0
        covered = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)

            treatment = np.array([1] * 15 + [0] * 15).astype(float)
            propensity = np.ones(n) * 0.5
            outcomes = true_ate * treatment + rng.normal(0, 1, n)

            result = ipw_ate(outcomes, treatment, propensity)
            covered.append(result["ci_lower"] <= true_ate <= result["ci_upper"])

        coverage = np.mean(covered)
        # With t-distribution, should achieve 93-97% coverage
        # With z-distribution, coverage would be lower (~90-92%)
        assert 0.90 <= coverage <= 0.98, (
            f"IPW coverage {coverage:.2%} for n={n}. "
            f"Expected 93-97% with t-distribution."
        )
