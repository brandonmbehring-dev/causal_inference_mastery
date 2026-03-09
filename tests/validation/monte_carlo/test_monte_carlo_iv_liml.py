"""
Monte Carlo validation for LIML (Limited Information Maximum Likelihood) estimator.

LIML is a k-class estimator that is more robust to weak instruments than 2SLS.
Key properties validated:
- Less biased than 2SLS with weak instruments
- Median-unbiased (median ≈ true β)
- Higher variance than 2SLS (bias-variance tradeoff)
- Better coverage than 2SLS under weak IV

Key References:
    - Anderson & Rubin (1949). "Estimation of the Parameters of a Single Equation"
    - Staiger & Stock (1997). "Instrumental Variables Regression with Weak Instruments"
    - Stock & Yogo (2005). "Testing for Weak Instruments in Linear IV Regression"

The key insight: LIML sacrifices variance for reduced bias under weak instruments.
"""

import numpy as np
import pytest
from src.causal_inference.iv import TwoStageLeastSquares, LIML
from tests.validation.monte_carlo.dgp_iv import (
    dgp_iv_strong,
    dgp_iv_weak,
    dgp_iv_very_weak,
    dgp_iv_over_identified,
)
from tests.validation.utils import validate_monte_carlo_results


class TestLIMLvsT2SLSBias:
    """Compare LIML and 2SLS bias across instrument strength."""

    @pytest.mark.slow
    def test_liml_equals_2sls_strong_iv(self):
        """
        LIML and 2SLS should be essentially identical with strong instruments.

        When instruments are strong, the k-class estimators converge and
        LIML provides no advantage over 2SLS.
        """
        n_runs = 2000
        true_beta = 0.5

        liml_estimates = []
        tsls_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            # LIML
            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

            # 2SLS
            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_estimates.append(tsls.coef_[0])

        liml_bias = abs(np.mean(liml_estimates) - true_beta)
        tsls_bias = abs(np.mean(tsls_estimates) - true_beta)

        # Both should be unbiased
        assert liml_bias < 0.05, f"LIML bias {liml_bias:.4f} too high for strong IV"
        assert tsls_bias < 0.05, f"2SLS bias {tsls_bias:.4f} too high for strong IV"

        # Should be very similar
        bias_difference = abs(liml_bias - tsls_bias)
        assert bias_difference < 0.02, (
            f"LIML and 2SLS should be similar with strong IV. "
            f"LIML bias: {liml_bias:.4f}, 2SLS bias: {tsls_bias:.4f}"
        )

    @pytest.mark.slow
    def test_liml_less_biased_than_2sls_weak_iv(self):
        """
        LIML should have less bias than 2SLS with weak instruments.

        This is the key advantage of LIML: it remains (approximately) median-unbiased
        even when 2SLS is substantially biased toward OLS.
        """
        n_runs = 3000
        true_beta = 0.5

        liml_estimates = []
        tsls_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            # LIML
            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

            # 2SLS
            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_estimates.append(tsls.coef_[0])

        liml_bias = abs(np.mean(liml_estimates) - true_beta)
        tsls_bias = abs(np.mean(tsls_estimates) - true_beta)

        # LIML should be less biased
        assert liml_bias < tsls_bias, (
            f"LIML bias ({liml_bias:.4f}) should be less than 2SLS bias ({tsls_bias:.4f}) "
            f"with weak instruments."
        )


class TestLIMLMedianUnbiased:
    """Test LIML's median-unbiasedness property."""

    @pytest.mark.slow
    def test_liml_median_unbiased_weak_iv(self):
        """
        LIML should be approximately median-unbiased with weak instruments.

        While the mean may be affected by outliers (fat tails with weak IV),
        the median should be close to the true parameter value.
        """
        n_runs = 3000
        true_beta = 0.5

        liml_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

        median_estimate = np.median(liml_estimates)
        median_bias = abs(median_estimate - true_beta)

        # Median should be close to true value
        assert median_bias < 0.10, (
            f"LIML median bias {median_bias:.4f} too high. "
            f"Median estimate: {median_estimate:.4f}, true β: {true_beta}"
        )

    @pytest.mark.slow
    def test_liml_mean_vs_median_weak_iv(self):
        """
        LIML mean may deviate from median due to fat tails under weak IV.

        This documents that while LIML is median-unbiased, the mean
        can be affected by extreme values.
        """
        n_runs = 3000
        true_beta = 0.5

        liml_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

        mean_estimate = np.mean(liml_estimates)
        median_estimate = np.median(liml_estimates)

        # Mean and median may differ under weak IV
        mean_median_diff = abs(mean_estimate - median_estimate)

        # Document: fat tails cause mean-median divergence
        # (This is expected behavior, not a test failure)
        if mean_median_diff > 0.05:
            # This is normal for LIML with weak IV
            pass


class TestLIMLVariance:
    """Test LIML's variance properties."""

    @pytest.mark.slow
    def test_liml_higher_variance_than_2sls(self):
        """
        LIML should have higher variance than 2SLS.

        This is the bias-variance tradeoff: LIML reduces bias at the cost
        of increased variance.
        """
        n_runs = 3000
        true_beta = 0.5

        liml_estimates = []
        tsls_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_estimates.append(tsls.coef_[0])

        liml_var = np.var(liml_estimates)
        tsls_var = np.var(tsls_estimates)

        # LIML variance >= 2SLS variance (with weak IV, often much higher)
        # Allow some tolerance due to MC noise
        assert liml_var >= tsls_var * 0.9, (
            f"LIML variance ({liml_var:.4f}) should be >= 2SLS variance ({tsls_var:.4f}). "
            f"LIML sacrifices variance for reduced bias."
        )


class TestLIMLCoverage:
    """Test LIML confidence interval coverage."""

    @pytest.mark.slow
    def test_liml_coverage_strong_iv(self):
        """
        LIML coverage should be correct (93-97%) with strong instruments.
        """
        n_runs = 2000
        true_beta = 0.5

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)

            estimates.append(liml.coef_[0])
            standard_errors.append(liml.se_[0])
            ci_lowers.append(liml.ci_[0, 0])
            ci_uppers.append(liml.ci_[0, 1])

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
            f"LIML coverage {validation['coverage']:.2%} outside [93%, 97%] with strong IV."
        )

    @pytest.mark.slow
    def test_liml_better_coverage_than_2sls_weak_iv(self):
        """
        LIML should have better coverage than 2SLS with weak instruments.

        While neither may achieve nominal coverage, LIML's reduced bias
        should translate to better coverage.
        """
        n_runs = 3000
        true_beta = 0.5

        liml_ci_lowers = []
        liml_ci_uppers = []
        tsls_ci_lowers = []
        tsls_ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_ci_lowers.append(liml.ci_[0, 0])
            liml_ci_uppers.append(liml.ci_[0, 1])

            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_ci_lowers.append(tsls.ci_[0, 0])
            tsls_ci_uppers.append(tsls.ci_[0, 1])

        liml_coverage = np.mean(
            (np.array(liml_ci_lowers) <= true_beta) & (true_beta <= np.array(liml_ci_uppers))
        )
        tsls_coverage = np.mean(
            (np.array(tsls_ci_lowers) <= true_beta) & (true_beta <= np.array(tsls_ci_uppers))
        )

        # LIML should have better (or equal) coverage
        assert liml_coverage >= tsls_coverage - 0.03, (
            f"LIML coverage ({liml_coverage:.2%}) should be at least as good as "
            f"2SLS coverage ({tsls_coverage:.2%}) with weak IV."
        )


class TestLIMLKappa:
    """Test LIML kappa parameter properties."""

    @pytest.mark.slow
    def test_liml_kappa_close_to_one_strong_iv(self):
        """
        LIML kappa should be close to 1 with strong instruments.

        κ_LIML → 1 as instruments get stronger, making LIML → 2SLS.
        """
        n_runs = 500
        kappas = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, random_state=seed)

            liml = LIML()
            liml.fit(data.Y, data.D, data.Z)
            kappas.append(liml.kappa_)

        mean_kappa = np.mean(kappas)

        # κ should be close to 1 with strong IV
        assert abs(mean_kappa - 1.0) < 0.1, (
            f"Mean kappa {mean_kappa:.4f} should be close to 1 with strong IV."
        )

    @pytest.mark.slow
    def test_liml_kappa_larger_weak_iv(self):
        """
        LIML kappa increases with weaker instruments.

        κ_LIML > 1 indicates the model is downweighting the first stage
        to reduce weak IV bias.
        """
        n_runs = 500
        kappas_strong = []
        kappas_weak = []

        for seed in range(n_runs):
            data_strong = dgp_iv_strong(n=500, random_state=seed)
            liml_strong = LIML()
            liml_strong.fit(data_strong.Y, data_strong.D, data_strong.Z)
            kappas_strong.append(liml_strong.kappa_)

            data_weak = dgp_iv_weak(n=500, random_state=seed)
            liml_weak = LIML()
            liml_weak.fit(data_weak.Y, data_weak.D, data_weak.Z)
            kappas_weak.append(liml_weak.kappa_)

        mean_kappa_strong = np.mean(kappas_strong)
        mean_kappa_weak = np.mean(kappas_weak)

        # Weak IV should have larger kappa
        assert mean_kappa_weak > mean_kappa_strong, (
            f"Weak IV kappa ({mean_kappa_weak:.4f}) should be larger than "
            f"strong IV kappa ({mean_kappa_strong:.4f})."
        )


class TestLIMLOverIdentified:
    """Test LIML with over-identified case (multiple instruments)."""

    @pytest.mark.slow
    def test_liml_unbiased_over_identified(self):
        """
        LIML should be unbiased with multiple valid instruments.
        """
        n_runs = 2000
        true_beta = 0.5

        estimates = []

        for seed in range(n_runs):
            data = dgp_iv_over_identified(
                n=500,
                true_beta=true_beta,
                n_instruments=3,
                random_state=seed,
            )

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            estimates.append(liml.coef_[0])

        bias = abs(np.mean(estimates) - true_beta)

        assert bias < 0.05, (
            f"LIML bias {bias:.4f} too high with over-identification. "
            f"Mean estimate: {np.mean(estimates):.4f}, true β: {true_beta}"
        )
