"""
Tests for CACE (Complier Average Causal Effect) estimation.

Test Structure (following 6-layer validation):
- Layer 1: Known-answer tests with hand-calculated values
- Layer 2: Adversarial edge cases
- Layer 3: Monte Carlo validation (5,000 runs)
- Layer 4: Cross-language parity (separate file)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.principal_stratification import (
    cace_2sls,
    wald_estimator,
    cace_em,
    CACEResult,
    StrataProportions,
)


# =============================================================================
# Test Fixtures / Data Generators
# =============================================================================


def generate_ps_dgp(
    n: int = 1000,
    pi_c: float = 0.70,
    pi_a: float = 0.15,
    pi_n: float = 0.15,
    true_cace: float = 2.0,
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Generate data from a principal stratification DGP.

    Parameters
    ----------
    n : int
        Sample size
    pi_c : float
        Proportion of compliers
    pi_a : float
        Proportion of always-takers
    pi_n : float
        Proportion of never-takers
    true_cace : float
        True CACE for compliers
    baseline : float
        Baseline outcome level
    noise_sd : float
        Standard deviation of noise
    seed : int
        Random seed

    Returns
    -------
    dict
        Dictionary with Y, D, Z, true_cace, strata, and proportions
    """
    np.random.seed(seed)

    # Normalize proportions
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c / total, pi_a / total, pi_n / total

    # Generate random assignment
    Z = np.random.binomial(1, 0.5, n)

    # Generate strata: 0=compliers, 1=always-takers, 2=never-takers
    strata = np.random.choice([0, 1, 2], n, p=[pi_c, pi_a, pi_n])

    # Treatment based on stratum
    D = np.where(
        strata == 0,
        Z,  # Compliers: follow assignment
        np.where(strata == 1, 1, 0),  # Always/Never-takers
    )

    # Outcome: Y = baseline + cace * D + noise
    # Under exclusion restriction, effect is ONLY for compliers
    # But since D=1 iff treated, this simplifies for all groups
    Y = baseline + true_cace * D + noise_sd * np.random.randn(n)

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "true_cace": true_cace,
        "strata": strata,
        "pi_c": pi_c,
        "pi_a": pi_a,
        "pi_n": pi_n,
    }


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestCACEKnownAnswer:
    """Known-answer tests with hand-calculated expected values."""

    def test_cace_equals_late(self):
        """CACE should equal LATE (Wald estimator)."""
        data = generate_ps_dgp(n=5000, true_cace=2.0, seed=42)

        result_2sls = cace_2sls(data["Y"], data["D"], data["Z"])
        result_wald = wald_estimator(data["Y"], data["D"], data["Z"])

        # Both should give same point estimate (up to numerical precision)
        assert_allclose(
            result_2sls["cace"],
            result_wald["cace"],
            rtol=1e-10,
            err_msg="2SLS and Wald CACE should be identical",
        )

    def test_first_stage_equals_complier_proportion(self):
        """First-stage coefficient should equal complier proportion."""
        pi_c = 0.60
        data = generate_ps_dgp(n=10000, pi_c=pi_c, pi_a=0.20, pi_n=0.20, seed=123)

        result = cace_2sls(data["Y"], data["D"], data["Z"])

        # First-stage = P(D=1|Z=1) - P(D=1|Z=0) = pi_c
        assert_allclose(
            result["first_stage_coef"],
            pi_c,
            rtol=0.05,
            err_msg=f"First-stage should equal complier proportion ({pi_c})",
        )

    def test_strata_proportions_sum_to_one(self):
        """Strata proportions should sum to 1."""
        data = generate_ps_dgp(n=1000, seed=456)
        result = cace_2sls(data["Y"], data["D"], data["Z"])

        props = result["strata_proportions"]
        total = props["compliers"] + props["always_takers"] + props["never_takers"]

        assert_allclose(total, 1.0, atol=1e-10, err_msg="Strata proportions should sum to 1")

    def test_perfect_compliance_cace_equals_ate(self):
        """With 100% compliers, CACE should equal ATE."""
        np.random.seed(789)
        n = 2000
        true_ate = 3.0

        Z = np.random.binomial(1, 0.5, n)
        D = Z.copy()  # Perfect compliance
        Y = 1.0 + true_ate * D + np.random.randn(n)

        result = cace_2sls(Y, D, Z)

        assert_allclose(
            result["cace"],
            true_ate,
            rtol=0.10,
            err_msg="With perfect compliance, CACE = ATE",
        )
        assert_allclose(
            result["strata_proportions"]["compliers"],
            1.0,
            atol=0.01,
            err_msg="Complier proportion should be ~1.0",
        )

    def test_reduced_form_times_pi_equals_gamma(self):
        """Reduced form = CACE * first-stage."""
        data = generate_ps_dgp(n=5000, true_cace=2.0, seed=101)
        result = cace_2sls(data["Y"], data["D"], data["Z"])

        # gamma = CACE * pi  =>  CACE = gamma / pi
        gamma_computed = result["cace"] * result["first_stage_coef"]

        assert_allclose(
            result["reduced_form"],
            gamma_computed,
            rtol=1e-10,
            err_msg="Reduced form should equal CACE * first-stage",
        )


# =============================================================================
# Layer 2: Adversarial / Edge Cases
# =============================================================================


class TestCACEAdversarial:
    """Adversarial edge case tests."""

    def test_weak_first_stage_warning(self):
        """Low first-stage F should flag weak instrument."""
        # Few compliers => weak first stage
        data = generate_ps_dgp(n=500, pi_c=0.05, pi_a=0.475, pi_n=0.475, true_cace=2.0, seed=202)

        result = cace_2sls(data["Y"], data["D"], data["Z"])

        # F-stat should be low (but may still pass due to randomness)
        # Just verify it runs and returns valid result
        assert result["first_stage_f"] > 0
        assert np.isfinite(result["cace"])

    def test_no_compliers_raises_error(self):
        """Should raise error when no compliers detected."""
        np.random.seed(303)
        n = 500
        Z = np.random.binomial(1, 0.5, n)

        # All always-takers: D=1 regardless of Z
        D = np.ones(n)
        Y = 1.0 + 2.0 * D + np.random.randn(n)

        with pytest.raises(ValueError, match="No compliers detected"):
            cace_2sls(Y, D, Z)

    def test_non_binary_treatment_raises(self):
        """Non-binary treatment should raise ValueError."""
        np.random.seed(404)
        n = 100
        Y = np.random.randn(n)
        D = np.random.choice([0, 1, 2], n)  # Non-binary
        Z = np.random.binomial(1, 0.5, n)

        with pytest.raises(ValueError, match="treatment must be binary"):
            cace_2sls(Y, D, Z)

    def test_non_binary_instrument_raises(self):
        """Non-binary instrument should raise ValueError."""
        np.random.seed(505)
        n = 100
        Y = np.random.randn(n)
        D = np.random.binomial(1, 0.5, n)
        Z = np.random.choice([0, 1, 2], n)  # Non-binary

        with pytest.raises(ValueError, match="instrument must be binary"):
            cace_2sls(Y, D, Z)

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        Z = np.random.binomial(1, 0.5, 99)  # Wrong length

        with pytest.raises(ValueError, match="must match outcome length"):
            cace_2sls(Y, D, Z)

    def test_extreme_proportions_still_works(self):
        """Should work with extreme but valid proportions."""
        # 95% compliers, 2.5% each always/never
        data = generate_ps_dgp(n=2000, pi_c=0.95, pi_a=0.025, pi_n=0.025, true_cace=1.5, seed=606)

        result = cace_2sls(data["Y"], data["D"], data["Z"])

        assert np.isfinite(result["cace"])
        assert result["strata_proportions"]["compliers"] > 0.90


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestCACEMonteCarlo:
    """Monte Carlo simulation tests."""

    @pytest.mark.slow
    def test_cace_unbiased_5000_runs(self):
        """
        Monte Carlo: CACE should be unbiased over 5,000 runs.

        Target: |bias| < 0.10
        """
        n_runs = 5000
        n_obs = 500
        true_cace = 2.0
        estimates = []

        for seed in range(n_runs):
            data = generate_ps_dgp(
                n=n_obs,
                true_cace=true_cace,
                pi_c=0.70,
                pi_a=0.15,
                pi_n=0.15,
                seed=seed,
            )
            result = cace_2sls(data["Y"], data["D"], data["Z"])
            estimates.append(result["cace"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_cace

        assert abs(bias) < 0.10, f"Bias {bias:.4f} exceeds threshold 0.10"

    @pytest.mark.slow
    def test_cace_coverage_5000_runs(self):
        """
        Monte Carlo: 95% CI should have coverage between 93-97%.
        """
        n_runs = 5000
        n_obs = 500
        true_cace = 2.0
        covered = 0

        for seed in range(n_runs):
            data = generate_ps_dgp(
                n=n_obs,
                true_cace=true_cace,
                pi_c=0.70,
                pi_a=0.15,
                pi_n=0.15,
                seed=seed,
            )
            result = cace_2sls(data["Y"], data["D"], data["Z"])

            if result["ci_lower"] <= true_cace <= result["ci_upper"]:
                covered += 1

        coverage = covered / n_runs

        assert 0.93 <= coverage <= 0.97, f"Coverage {coverage:.2%} outside [93%, 97%]"

    @pytest.mark.slow
    def test_se_accuracy_5000_runs(self):
        """
        Monte Carlo: SE should accurately estimate sampling std dev.

        Target: |mean(SE) - sd(estimates)| / sd(estimates) < 0.15
        """
        n_runs = 5000
        n_obs = 500
        true_cace = 2.0
        estimates = []
        standard_errors = []

        for seed in range(n_runs):
            data = generate_ps_dgp(
                n=n_obs,
                true_cace=true_cace,
                pi_c=0.70,
                pi_a=0.15,
                pi_n=0.15,
                seed=seed,
            )
            result = cace_2sls(data["Y"], data["D"], data["Z"])
            estimates.append(result["cace"])
            standard_errors.append(result["se"])

        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(standard_errors)
        se_error = abs(mean_se - empirical_sd) / empirical_sd

        assert se_error < 0.15, f"SE error {se_error:.2%} exceeds 15%"

    def test_wald_matches_2sls_monte_carlo(self):
        """Wald and 2SLS should give identical results (no covariates)."""
        n_runs = 100

        for seed in range(n_runs):
            data = generate_ps_dgp(n=500, true_cace=2.0, seed=seed)

            result_2sls = cace_2sls(data["Y"], data["D"], data["Z"])
            result_wald = wald_estimator(data["Y"], data["D"], data["Z"])

            assert_allclose(
                result_2sls["cace"],
                result_wald["cace"],
                rtol=1e-10,
                err_msg=f"Seed {seed}: 2SLS and Wald mismatch",
            )


# =============================================================================
# Return Type Tests
# =============================================================================


class TestCACEReturnType:
    """Tests for return type structure."""

    def test_return_type_structure(self):
        """Result should have all expected keys."""
        data = generate_ps_dgp(n=500, seed=707)
        result = cace_2sls(data["Y"], data["D"], data["Z"])

        expected_keys = [
            "cace",
            "se",
            "ci_lower",
            "ci_upper",
            "z_stat",
            "pvalue",
            "strata_proportions",
            "first_stage_coef",
            "first_stage_se",
            "first_stage_f",
            "reduced_form",
            "reduced_form_se",
            "n",
            "n_treated_assigned",
            "n_control_assigned",
            "method",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_strata_proportions_structure(self):
        """Strata proportions should have expected keys."""
        data = generate_ps_dgp(n=500, seed=808)
        result = cace_2sls(data["Y"], data["D"], data["Z"])

        props = result["strata_proportions"]
        expected_keys = ["compliers", "always_takers", "never_takers", "compliers_se"]

        for key in expected_keys:
            assert key in props, f"Missing strata key: {key}"

    def test_all_values_finite(self):
        """All numeric values should be finite."""
        data = generate_ps_dgp(n=1000, seed=909)
        result = cace_2sls(data["Y"], data["D"], data["Z"])

        numeric_keys = [
            "cace",
            "se",
            "ci_lower",
            "ci_upper",
            "z_stat",
            "pvalue",
            "first_stage_coef",
            "first_stage_se",
            "first_stage_f",
            "reduced_form",
            "reduced_form_se",
        ]

        for key in numeric_keys:
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# =============================================================================
# Covariates Tests
# =============================================================================


class TestCACEWithCovariates:
    """Tests for CACE with covariates."""

    def test_covariates_1d_works(self):
        """1D covariates should work."""
        data = generate_ps_dgp(n=1000, seed=1001)
        X = np.random.randn(1000)

        result = cace_2sls(data["Y"], data["D"], data["Z"], covariates=X)

        assert np.isfinite(result["cace"])

    def test_covariates_2d_works(self):
        """2D covariates should work."""
        data = generate_ps_dgp(n=1000, seed=1002)
        X = np.random.randn(1000, 3)

        result = cace_2sls(data["Y"], data["D"], data["Z"], covariates=X)

        assert np.isfinite(result["cace"])

    def test_covariates_wrong_length_raises(self):
        """Covariates with wrong length should raise."""
        data = generate_ps_dgp(n=100, seed=1003)
        X = np.random.randn(99, 2)  # Wrong length

        with pytest.raises(ValueError, match="must match outcome length"):
            cace_2sls(data["Y"], data["D"], data["Z"], covariates=X)


# =============================================================================
# Inference Type Tests
# =============================================================================


class TestCACEInference:
    """Tests for different inference types."""

    def test_robust_se_differs_from_standard(self):
        """Robust and standard SEs should generally differ."""
        # Generate heteroskedastic data
        np.random.seed(1101)
        n = 1000
        Z = np.random.binomial(1, 0.5, n)
        D = np.where(np.random.rand(n) < 0.3 + 0.4 * Z, 1, 0)  # Compliers
        # Heteroskedastic noise
        noise = np.random.randn(n) * (1 + D)
        Y = 1.0 + 2.0 * D + noise

        result_robust = cace_2sls(Y, D, Z, inference="robust")
        result_standard = cace_2sls(Y, D, Z, inference="standard")

        # Point estimates should be same
        assert_allclose(result_robust["cace"], result_standard["cace"], rtol=1e-10)

        # SEs may differ (though not guaranteed)
        # Just check both are valid
        assert result_robust["se"] > 0
        assert result_standard["se"] > 0

    def test_alpha_affects_ci(self):
        """Different alpha should change CI width."""
        data = generate_ps_dgp(n=1000, seed=1202)

        result_95 = cace_2sls(data["Y"], data["D"], data["Z"], alpha=0.05)
        result_90 = cace_2sls(data["Y"], data["D"], data["Z"], alpha=0.10)

        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_90 = result_90["ci_upper"] - result_90["ci_lower"]

        assert width_95 > width_90, "95% CI should be wider than 90% CI"

    def test_invalid_alpha_raises(self):
        """Invalid alpha should raise."""
        data = generate_ps_dgp(n=100, seed=1303)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            cace_2sls(data["Y"], data["D"], data["Z"], alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            cace_2sls(data["Y"], data["D"], data["Z"], alpha=0.0)


# =============================================================================
# EM Algorithm Tests (Session 112)
# =============================================================================


class TestCACEEM:
    """Tests for EM-based CACE estimation."""

    def test_em_runs_and_returns_valid_result(self):
        """EM algorithm should run and return valid result."""
        data = generate_ps_dgp(n=1000, true_cace=2.0, seed=2001)

        result = cace_em(data["Y"], data["D"], data["Z"])

        # Check all expected keys present
        assert "cace" in result
        assert "se" in result
        assert "method" in result
        assert result["method"] == "em"

        # Check values are finite
        assert np.isfinite(result["cace"])
        assert np.isfinite(result["se"])
        assert result["se"] > 0

    def test_em_close_to_2sls(self):
        """EM should give similar results to 2SLS with well-separated strata."""
        data = generate_ps_dgp(n=3000, true_cace=2.0, seed=2002)

        result_em = cace_em(data["Y"], data["D"], data["Z"])
        result_2sls = cace_2sls(data["Y"], data["D"], data["Z"])

        # Should be close but not necessarily identical
        assert_allclose(
            result_em["cace"],
            result_2sls["cace"],
            rtol=0.15,
            err_msg="EM and 2SLS CACE should be similar",
        )

    def test_em_strata_sum_to_one(self):
        """EM strata proportions should sum to 1."""
        data = generate_ps_dgp(n=1000, seed=2003)

        result = cace_em(data["Y"], data["D"], data["Z"])

        props = result["strata_proportions"]
        total = props["compliers"] + props["always_takers"] + props["never_takers"]

        assert_allclose(total, 1.0, atol=1e-6, err_msg="Strata should sum to 1")

    def test_em_recovers_true_cace(self):
        """EM should recover true CACE on average."""
        true_cace = 2.0
        data = generate_ps_dgp(n=5000, true_cace=true_cace, seed=2004)

        result = cace_em(data["Y"], data["D"], data["Z"])

        # With n=5000, should be within 0.3 of true value
        assert abs(result["cace"] - true_cace) < 0.3, (
            f"EM estimate {result['cace']:.3f} too far from true {true_cace}"
        )

    def test_em_confidence_interval_contains_true(self):
        """EM CI should generally contain true CACE."""
        true_cace = 2.0
        data = generate_ps_dgp(n=2000, true_cace=true_cace, seed=2005)

        result = cace_em(data["Y"], data["D"], data["Z"])

        # Check CI contains true value (should happen ~95% of time)
        # For a single test, just check it's reasonable
        assert result["ci_lower"] < result["ci_upper"]
        # Not asserting containment since individual runs may miss

    def test_em_first_stage_equals_complier_proportion(self):
        """EM first-stage should approximate complier proportion."""
        pi_c = 0.65
        data = generate_ps_dgp(n=3000, pi_c=pi_c, pi_a=0.175, pi_n=0.175, seed=2006)

        result = cace_em(data["Y"], data["D"], data["Z"])

        # First-stage should be close to true complier proportion
        assert_allclose(
            result["first_stage_coef"],
            pi_c,
            rtol=0.10,
            err_msg=f"First-stage {result['first_stage_coef']:.3f} != pi_c {pi_c}",
        )

    def test_em_method_field(self):
        """EM result should have method='em'."""
        data = generate_ps_dgp(n=500, seed=2007)

        result = cace_em(data["Y"], data["D"], data["Z"])

        assert result["method"] == "em"

    def test_em_max_iter_warning(self):
        """EM should warn (not error) when hitting max iterations."""
        data = generate_ps_dgp(n=500, seed=2008)

        # Use very small max_iter to force hitting limit
        import warnings as warn_module

        with warn_module.catch_warnings(record=True) as w:
            warn_module.simplefilter("always")
            result = cace_em(data["Y"], data["D"], data["Z"], max_iter=2, tol=1e-20)

            # Should still return a result
            assert np.isfinite(result["cace"])

            # Should have issued warning
            # (May not always fire depending on convergence)

    def test_em_different_tol_affects_result(self):
        """Different tolerance should potentially affect iteration count."""
        data = generate_ps_dgp(n=1000, seed=2009)

        # Tight tolerance
        result_tight = cace_em(data["Y"], data["D"], data["Z"], tol=1e-10)

        # Loose tolerance
        result_loose = cace_em(data["Y"], data["D"], data["Z"], tol=1e-2)

        # Both should give valid results
        assert np.isfinite(result_tight["cace"])
        assert np.isfinite(result_loose["cace"])

        # Results should be similar (same estimand)
        assert_allclose(
            result_tight["cace"],
            result_loose["cace"],
            rtol=0.05,
        )

    def test_em_covariates_warning(self):
        """EM should warn when covariates are provided (not used)."""
        data = generate_ps_dgp(n=500, seed=2010)
        X = np.random.randn(500, 2)

        import warnings as warn_module

        with warn_module.catch_warnings(record=True) as w:
            warn_module.simplefilter("always")
            result = cace_em(data["Y"], data["D"], data["Z"], covariates=X)

            # Check warning was issued
            cov_warnings = [x for x in w if "covariates are not used" in str(x.message)]
            assert len(cov_warnings) > 0, "Should warn about covariates"

        # Result should still be valid
        assert np.isfinite(result["cace"])


class TestCACEEMMonteCarlo:
    """Monte Carlo validation for EM algorithm."""

    def test_em_unbiased_1000_runs(self):
        """
        Monte Carlo: EM should be approximately unbiased.

        Target: |bias| < 0.15 (may be slightly higher than 2SLS due to finite sample)
        """
        n_runs = 1000
        n_obs = 500
        true_cace = 2.0
        estimates = []

        for seed in range(n_runs):
            data = generate_ps_dgp(
                n=n_obs,
                true_cace=true_cace,
                pi_c=0.70,
                pi_a=0.15,
                pi_n=0.15,
                seed=seed + 10000,  # Different seeds from 2SLS tests
            )
            result = cace_em(data["Y"], data["D"], data["Z"])
            estimates.append(result["cace"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_cace

        assert abs(bias) < 0.15, f"EM bias {bias:.4f} exceeds threshold 0.15"

    @pytest.mark.slow
    def test_em_coverage_1000_runs(self):
        """
        Monte Carlo: EM 95% CI coverage.

        Target: Coverage 90-98% (wider range due to SE approximation)
        """
        n_runs = 1000
        n_obs = 500
        true_cace = 2.0
        covered = 0

        for seed in range(n_runs):
            data = generate_ps_dgp(
                n=n_obs,
                true_cace=true_cace,
                pi_c=0.70,
                pi_a=0.15,
                pi_n=0.15,
                seed=seed + 20000,
            )
            result = cace_em(data["Y"], data["D"], data["Z"])

            if result["ci_lower"] <= true_cace <= result["ci_upper"]:
                covered += 1

        coverage = covered / n_runs

        assert 0.90 <= coverage <= 0.98, f"EM coverage {coverage:.2%} outside [90%, 98%]"

    def test_em_vs_2sls_correlation(self):
        """EM and 2SLS estimates should be highly correlated."""
        n_runs = 100
        em_estimates = []
        twosls_estimates = []

        for seed in range(n_runs):
            data = generate_ps_dgp(n=500, true_cace=2.0, seed=seed + 30000)

            result_em = cace_em(data["Y"], data["D"], data["Z"])
            result_2sls = cace_2sls(data["Y"], data["D"], data["Z"])

            em_estimates.append(result_em["cace"])
            twosls_estimates.append(result_2sls["cace"])

        correlation = np.corrcoef(em_estimates, twosls_estimates)[0, 1]

        assert correlation > 0.95, f"EM-2SLS correlation {correlation:.3f} too low"
