"""
Monte Carlo validation for GMM (Generalized Method of Moments) estimator.

Key properties validated:
- One-step GMM ≡ 2SLS (identical point estimates)
- Two-step GMM more efficient than one-step with over-identification
- Hansen J-test Type I error ≈ 0.05 under valid restrictions
- Hansen J-test power to detect invalid instruments

Key References:
    - Hansen (1982). "Large Sample Properties of Generalized Method of Moments Estimators"
    - Newey & West (1987). "Hypothesis Testing with Efficient Method of Moments Estimation"

The key insight: Two-step GMM achieves efficiency gains by using optimal weighting,
and Hansen J-test provides a specification test for instrument validity.
"""

import numpy as np
import pytest
from src.causal_inference.iv import TwoStageLeastSquares, GMM
from tests.validation.monte_carlo.dgp_iv import (
    dgp_iv_strong,
    dgp_iv_over_identified,
    dgp_iv_invalid_instruments,
)
from tests.validation.utils import validate_monte_carlo_results


class TestGMMEquivalence:
    """Test GMM equivalence to 2SLS."""

    @pytest.mark.slow
    def test_gmm_1step_equals_2sls(self):
        """
        One-step GMM should produce identical results to 2SLS.

        GMM one-step uses W = (Z'Z)^(-1) which is equivalent to 2SLS.
        """
        n_runs = 500
        true_beta = 0.5

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500, true_beta=true_beta, n_instruments=3, random_state=seed
            )

            gmm = GMM(steps="one", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)

            # Point estimates should be identical (or very close)
            assert np.isclose(gmm.coef_[0], tsls.coef_[0], rtol=1e-6), (
                f"GMM one-step ({gmm.coef_[0]:.6f}) should equal 2SLS ({tsls.coef_[0]:.6f})"
            )

    @pytest.mark.slow
    def test_gmm_1step_equals_2sls_just_identified(self):
        """
        With just-identification (q=p), all IV estimators are equivalent.
        """
        n_runs = 500
        true_beta = 0.5

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            gmm = GMM(steps="one", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)

            assert np.isclose(gmm.coef_[0], tsls.coef_[0], rtol=1e-6), (
                f"Just-identified: GMM ({gmm.coef_[0]:.6f}) should equal 2SLS ({tsls.coef_[0]:.6f})"
            )


class TestGMMEfficiency:
    """Test GMM efficiency with over-identification."""

    @pytest.mark.slow
    def test_gmm_2step_lower_variance_than_1step(self):
        """
        Two-step GMM should have lower variance than one-step with over-identification.

        The optimal weighting matrix in two-step GMM achieves efficiency gains.
        """
        n_runs = 2000
        true_beta = 0.5

        gmm1_estimates = []
        gmm2_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500, true_beta=true_beta, n_instruments=3, random_state=seed
            )

            gmm1 = GMM(steps="one", inference="robust")
            gmm1.fit(data.Y, data.D, data.Z)
            gmm1_estimates.append(gmm1.coef_[0])

            gmm2 = GMM(steps="two", inference="robust")
            gmm2.fit(data.Y, data.D, data.Z)
            gmm2_estimates.append(gmm2.coef_[0])

        gmm1_var = np.var(gmm1_estimates)
        gmm2_var = np.var(gmm2_estimates)

        # Two-step should have lower variance (efficiency gain)
        assert gmm2_var <= gmm1_var * 1.05, (
            f"Two-step GMM variance ({gmm2_var:.4f}) should be <= "
            f"one-step variance ({gmm1_var:.4f}) with over-identification."
        )

    @pytest.mark.slow
    def test_gmm_2step_unbiased_over_identified(self):
        """
        Two-step GMM should be unbiased with over-identification.
        """
        n_runs = 2000
        true_beta = 0.5

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500, true_beta=true_beta, n_instruments=3, random_state=seed
            )

            gmm = GMM(steps="two", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            estimates.append(gmm.coef_[0])
            standard_errors.append(gmm.se_[0])
            ci_lowers.append(gmm.ci_[0, 0])
            ci_uppers.append(gmm.ci_[0, 1])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_beta,
            bias_threshold=0.05,
            coverage_lower=0.92,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["bias_ok"], (
            f"Two-step GMM bias {validation['bias']:.4f} exceeds threshold. "
            f"Mean estimate: {np.mean(estimates):.4f}, true β: {true_beta}"
        )


class TestHansenJTestSize:
    """Test Hansen J-test Type I error (size) under valid instruments."""

    @pytest.mark.slow
    def test_hansen_j_type1_error(self):
        """
        Hansen J-test should reject H₀ ≈ 5% of the time under valid instruments.

        H₀: All instruments are valid (satisfy exclusion restriction)
        Under H₀, reject rate should ≈ α (nominal level)
        """
        n_runs = 2000
        alpha = 0.05

        rejections = []

        for seed in range(n_runs):
            # Valid instruments (no exclusion violation)
            data = dgp_iv_over_identified(n=500, true_beta=0.5, n_instruments=3, random_state=seed)

            gmm = GMM(steps="two", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            # Check if J-test rejects
            rejected = gmm.j_pvalue_ < alpha
            rejections.append(rejected)

        rejection_rate = np.mean(rejections)

        # Type I error should be close to nominal level
        assert 0.03 < rejection_rate < 0.08, (
            f"Hansen J-test rejection rate {rejection_rate:.2%} too far from "
            f"nominal {alpha:.0%}. Expected ~5% under valid instruments."
        )

    @pytest.mark.slow
    def test_hansen_j_test_statistic_distribution(self):
        """
        Under H₀, J-statistic should follow χ²(q-p) distribution.
        """
        n_runs = 2000
        n_instruments = 3
        n_endogenous = 1
        df = n_instruments - n_endogenous  # χ²(2)

        j_stats = []

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500, true_beta=0.5, n_instruments=n_instruments, random_state=seed
            )

            gmm = GMM(steps="two", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            j_stats.append(gmm.j_statistic_)

        # Mean of χ²(df) = df
        expected_mean = df
        actual_mean = np.mean(j_stats)

        assert abs(actual_mean - expected_mean) < 0.5, (
            f"J-statistic mean {actual_mean:.2f} should be close to χ²({df}) mean = {df}"
        )


class TestHansenJTestPower:
    """Test Hansen J-test power to detect invalid instruments."""

    @pytest.mark.slow
    def test_hansen_j_detects_invalid_instruments(self):
        """
        Hansen J-test should reject with high probability when instruments are invalid.
        """
        n_runs = 1000
        alpha = 0.05

        rejections = []

        for seed in range(n_runs):
            # Invalid instruments (one instrument directly affects Y)
            data = dgp_iv_invalid_instruments(
                n=500,
                true_beta=0.5,
                violation_strength=0.3,  # Direct effect of Z₂ on Y
                random_state=seed,
            )

            gmm = GMM(steps="two", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            rejected = gmm.j_pvalue_ < alpha
            rejections.append(rejected)

        rejection_rate = np.mean(rejections)

        # With invalid instruments, rejection rate should be high
        assert rejection_rate > 0.50, (
            f"Hansen J-test power {rejection_rate:.2%} too low. "
            f"Should reject > 50% with invalid instruments."
        )

    @pytest.mark.slow
    def test_hansen_j_power_increases_with_violation(self):
        """
        J-test power should increase with stronger exclusion violations.
        """
        n_runs = 500
        alpha = 0.05

        violation_strengths = [0.1, 0.2, 0.3]
        rejection_rates = []

        for violation in violation_strengths:
            rejections = []
            for seed in range(n_runs):
                data = dgp_iv_invalid_instruments(
                    n=500,
                    true_beta=0.5,
                    violation_strength=violation,
                    random_state=seed,
                )

                gmm = GMM(steps="two", inference="robust")
                gmm.fit(data.Y, data.D, data.Z)

                rejected = gmm.j_pvalue_ < alpha
                rejections.append(rejected)

            rejection_rates.append(np.mean(rejections))

        # Power should increase with violation strength
        assert rejection_rates[1] >= rejection_rates[0] - 0.05, (
            f"Power should increase with violation strength. "
            f"Got {rejection_rates} for violations {violation_strengths}"
        )
        assert rejection_rates[2] >= rejection_rates[1] - 0.05, (
            f"Power should increase with violation strength. "
            f"Got {rejection_rates} for violations {violation_strengths}"
        )


class TestGMMCoverage:
    """Test GMM confidence interval coverage."""

    @pytest.mark.slow
    def test_gmm_2step_coverage_over_identified(self):
        """
        Two-step GMM should have correct coverage with over-identification.
        """
        n_runs = 2000
        true_beta = 0.5

        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500, true_beta=true_beta, n_instruments=3, random_state=seed
            )

            gmm = GMM(steps="two", inference="robust")
            gmm.fit(data.Y, data.D, data.Z)

            ci_lowers.append(gmm.ci_[0, 0])
            ci_uppers.append(gmm.ci_[0, 1])

        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        coverage = np.mean((ci_lowers <= true_beta) & (true_beta <= ci_uppers))

        assert 0.92 < coverage < 0.97, (
            f"GMM two-step coverage {coverage:.2%} outside expected [92%, 97%] range."
        )
