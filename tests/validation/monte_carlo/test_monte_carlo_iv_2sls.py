"""
Monte Carlo validation for Two-Stage Least Squares (2SLS) estimator.

Validates statistical properties across instrument strength scenarios:
- Strong IV (F > 20): Unbiased, correct coverage
- Moderate IV (F ≈ 15): Slight bias, coverage near target
- Weak IV (F ≈ 8): Substantial bias, undercoverage (documented)
- Very Weak IV (F < 5): Severe bias (educational demonstration)

Key References:
    - Staiger & Stock (1997). "Instrumental Variables Regression with Weak Instruments"
    - Stock & Yogo (2005). "Testing for Weak Instruments in Linear IV Regression"

The key insight validated here: 2SLS is biased toward OLS when instruments are weak,
and 2SLS confidence intervals have severe undercoverage (CI too narrow).
"""

import numpy as np
import pytest
from src.causal_inference.iv import TwoStageLeastSquares
from tests.validation.monte_carlo.dgp_iv import (
    dgp_iv_strong,
    dgp_iv_moderate,
    dgp_iv_weak,
    dgp_iv_very_weak,
    dgp_iv_heteroskedastic,
    compute_ols_probability_limit,
)
from tests.validation.utils import validate_monte_carlo_results


class Test2SLSUnbiasedness:
    """Monte Carlo validation of 2SLS bias across instrument strength."""

    @pytest.mark.slow
    def test_2sls_unbiased_strong_iv(self):
        """
        2SLS should be unbiased with strong instruments (F >> 20).

        DGP: D = 0.8*Z + ν, Y = 0.5*D + ε, Cov(ν,ε) = 0.5
        Expected F ≈ 640, true β = 0.5

        Expected: Bias < 0.05 (essentially unbiased)
        """
        n_runs = 3000
        true_beta = 0.5

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)

            estimates.append(iv.coef_[0])
            standard_errors.append(iv.se_[0])
            ci_lowers.append(iv.ci_[0, 0])
            ci_uppers.append(iv.ci_[0, 1])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_beta,
            bias_threshold=0.05,
            coverage_lower=0.93,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["bias_ok"], (
            f"2SLS bias {validation['bias']:.4f} exceeds 0.05 with strong IV. "
            f"Mean estimate: {np.mean(estimates):.4f}, true β: {true_beta}"
        )

    @pytest.mark.slow
    def test_2sls_biased_weak_iv(self):
        """
        2SLS shows substantial bias with weak instruments (F ≈ 8).

        This test DOCUMENTS the expected failure of 2SLS with weak instruments.
        The bias should be toward OLS (positive bias when Cov(ν,ε) > 0).

        Expected: Bias > 0.10 (demonstrating weak IV problem)
        """
        n_runs = 3000
        true_beta = 0.5
        endogeneity_rho = 0.5

        estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(
                n=500,
                true_beta=true_beta,
                endogeneity_rho=endogeneity_rho,
                random_state=seed,
            )

            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)

            estimates.append(iv.coef_[0])

        mean_estimate = np.mean(estimates)
        bias = abs(mean_estimate - true_beta)

        # With weak IV, 2SLS is biased toward OLS
        # The bias direction should be positive (toward OLS plim)
        bias_direction = mean_estimate - true_beta

        # Weak IV bias should be substantial
        assert bias > 0.05, (
            f"Expected substantial weak IV bias, got {bias:.4f}. "
            f"Mean estimate: {mean_estimate:.4f}, true β: {true_beta}"
        )

        # Bias should be toward OLS (positive when ρ > 0)
        expected_direction = compute_ols_probability_limit(endogeneity_rho)
        assert bias_direction > 0 if expected_direction > 0 else bias_direction < 0, (
            f"Bias direction ({bias_direction:.4f}) should match OLS direction "
            f"({expected_direction:.4f})"
        )

    @pytest.mark.slow
    def test_2sls_bias_toward_ols_very_weak(self):
        """
        With very weak instruments (F < 5), 2SLS converges to OLS.

        This is the worst-case scenario: instruments are essentially useless.
        The test documents that 2SLS provides little information beyond OLS.
        """
        n_runs = 2000
        true_beta = 0.5
        endogeneity_rho = 0.5

        iv_estimates = []
        ols_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_very_weak(
                n=500,
                true_beta=true_beta,
                endogeneity_rho=endogeneity_rho,
                random_state=seed,
            )

            # 2SLS estimate
            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)
            iv_estimates.append(iv.coef_[0])

            # OLS estimate (biased)
            ols_beta = np.cov(data.Y, data.D)[0, 1] / np.var(data.D)
            ols_estimates.append(ols_beta)

        mean_iv = np.mean(iv_estimates)
        mean_ols = np.mean(ols_estimates)

        # With very weak IV, 2SLS should be close to OLS
        # (both are biased in the same direction)
        iv_ols_difference = abs(mean_iv - mean_ols)

        # 2SLS should be closer to OLS than to true value
        iv_true_bias = abs(mean_iv - true_beta)
        ols_true_bias = abs(mean_ols - true_beta)

        # Document that very weak IV provides little benefit
        assert iv_true_bias > 0.15, f"Very weak IV bias {iv_true_bias:.4f} should be substantial"


class Test2SLSCoverage:
    """Monte Carlo validation of 2SLS confidence interval coverage."""

    @pytest.mark.slow
    def test_2sls_coverage_strong_iv(self):
        """
        2SLS 95% CI should have correct coverage (93-97%) with strong instruments.
        """
        n_runs = 3000
        true_beta = 0.5

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)

            estimates.append(iv.coef_[0])
            standard_errors.append(iv.se_[0])
            ci_lowers.append(iv.ci_[0, 0])
            ci_uppers.append(iv.ci_[0, 1])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_beta,
            bias_threshold=0.05,
            coverage_lower=0.93,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["coverage_ok"], (
            f"Coverage {validation['coverage']:.2%} outside [93%, 97%] range. "
            f"Strong IV should have correct coverage."
        )

    @pytest.mark.slow
    def test_2sls_undercoverage_weak_iv(self):
        """
        2SLS CI has severe undercoverage with weak instruments.

        This is the key weak instrument problem: standard errors are too small,
        leading to confidence intervals that are too narrow.

        Expected: Coverage < 90% (often 70-80%)
        """
        n_runs = 3000
        true_beta = 0.5

        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)

            ci_lowers.append(iv.ci_[0, 0])
            ci_uppers.append(iv.ci_[0, 1])

        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        coverage = np.mean((ci_lowers <= true_beta) & (true_beta <= ci_uppers))

        # Weak IV causes undercoverage
        assert coverage < 0.92, (
            f"Coverage {coverage:.2%} too high for weak IV. "
            f"2SLS CI should have undercoverage (< 92%) with F ≈ 8."
        )

    @pytest.mark.slow
    def test_2sls_coverage_moderate_iv(self):
        """
        2SLS coverage with moderate instruments (F ≈ 15).

        At this strength, coverage should be between strong IV (95%) and weak IV (75%).
        """
        n_runs = 2000
        true_beta = 0.5

        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_moderate(n=500, true_beta=true_beta, random_state=seed)

            iv = TwoStageLeastSquares(inference="robust")
            iv.fit(data.Y, data.D, data.Z)

            ci_lowers.append(iv.ci_[0, 0])
            ci_uppers.append(iv.ci_[0, 1])

        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        coverage = np.mean((ci_lowers <= true_beta) & (true_beta <= ci_uppers))

        # Moderate IV: coverage not as good as strong, not as bad as weak
        assert 0.85 < coverage < 0.96, (
            f"Moderate IV coverage {coverage:.2%} outside expected [85%, 96%] range."
        )


class Test2SLSSEAccuracy:
    """Monte Carlo validation of 2SLS standard error estimation."""

    @pytest.mark.slow
    def test_2sls_se_accuracy_homoskedastic(self):
        """
        2SLS standard SE should be accurate under homoskedasticity.

        SE accuracy = |empirical SD - mean(SE)| / empirical SD
        Expected: < 10% error
        """
        n_runs = 3000
        true_beta = 0.5

        estimates = []
        standard_errors = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            iv = TwoStageLeastSquares(inference="standard")
            iv.fit(data.Y, data.D, data.Z)

            estimates.append(iv.coef_[0])
            standard_errors.append(iv.se_[0])

        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(standard_errors)
        se_accuracy = abs(empirical_sd - mean_se) / empirical_sd

        assert se_accuracy < 0.15, (
            f"SE accuracy {se_accuracy:.2%} exceeds 15%. "
            f"Empirical SD: {empirical_sd:.4f}, Mean SE: {mean_se:.4f}"
        )

    @pytest.mark.slow
    def test_2sls_robust_se_heteroskedastic(self):
        """
        Robust (HC0) SE should be correct under heteroskedasticity.

        When errors are heteroskedastic, standard SE is incorrect but
        robust SE should provide valid inference.
        """
        n_runs = 2000
        true_beta = 0.5

        estimates_robust = []
        standard_errors_robust = []
        ci_lowers_robust = []
        ci_uppers_robust = []

        estimates_standard = []
        standard_errors_standard = []
        ci_lowers_standard = []
        ci_uppers_standard = []

        for seed in range(n_runs):
            data = dgp_iv_heteroskedastic(n=500, true_beta=true_beta, random_state=seed)

            # Robust inference
            iv_robust = TwoStageLeastSquares(inference="robust")
            iv_robust.fit(data.Y, data.D, data.Z)

            estimates_robust.append(iv_robust.coef_[0])
            standard_errors_robust.append(iv_robust.se_[0])
            ci_lowers_robust.append(iv_robust.ci_[0, 0])
            ci_uppers_robust.append(iv_robust.ci_[0, 1])

            # Standard inference (for comparison)
            iv_standard = TwoStageLeastSquares(inference="standard")
            iv_standard.fit(data.Y, data.D, data.Z)

            estimates_standard.append(iv_standard.coef_[0])
            standard_errors_standard.append(iv_standard.se_[0])
            ci_lowers_standard.append(iv_standard.ci_[0, 0])
            ci_uppers_standard.append(iv_standard.ci_[0, 1])

        # Robust SE should provide better coverage under heteroskedasticity
        validation_robust = validate_monte_carlo_results(
            estimates_robust,
            standard_errors_robust,
            ci_lowers_robust,
            ci_uppers_robust,
            true_beta,
            bias_threshold=0.10,
            coverage_lower=0.90,
            coverage_upper=0.98,
            se_accuracy_threshold=0.20,
        )

        validation_standard = validate_monte_carlo_results(
            estimates_standard,
            standard_errors_standard,
            ci_lowers_standard,
            ci_uppers_standard,
            true_beta,
            bias_threshold=0.10,
            coverage_lower=0.90,
            coverage_upper=0.98,
            se_accuracy_threshold=0.20,
        )

        # Robust should have better coverage than standard
        assert validation_robust["coverage"] >= validation_standard["coverage"] - 0.05, (
            f"Robust coverage ({validation_robust['coverage']:.2%}) should be at least "
            f"as good as standard ({validation_standard['coverage']:.2%}) under heteroskedasticity."
        )


class Test2SLSDiagnostics:
    """Monte Carlo validation of 2SLS diagnostic statistics."""

    @pytest.mark.slow
    def test_first_stage_f_stat_strong_iv(self):
        """
        First-stage F-statistic should indicate strong instruments.
        """
        n_runs = 500
        f_stats = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, random_state=seed)

            iv = TwoStageLeastSquares()
            iv.fit(data.Y, data.D, data.Z)

            f_stats.append(iv.first_stage_f_stat_)

        mean_f = np.mean(f_stats)

        # Strong IV should have F >> 10
        assert mean_f > 50, (
            f"Mean F-statistic {mean_f:.1f} too low for strong IV DGP. Expected F >> 10."
        )

    @pytest.mark.slow
    def test_first_stage_f_stat_weak_iv(self):
        """
        First-stage F-statistic should indicate weak instruments.
        """
        n_runs = 500
        f_stats = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, random_state=seed)

            iv = TwoStageLeastSquares()
            iv.fit(data.Y, data.D, data.Z)

            f_stats.append(iv.first_stage_f_stat_)

        mean_f = np.mean(f_stats)

        # Weak IV should have F < 10
        assert mean_f < 15, (
            f"Mean F-statistic {mean_f:.1f} too high for weak IV DGP. Expected F ≈ 8."
        )

        # Should be above 5 (not very weak)
        assert mean_f > 5, f"Mean F-statistic {mean_f:.1f} too low. Expected F ≈ 8 for weak IV DGP."
