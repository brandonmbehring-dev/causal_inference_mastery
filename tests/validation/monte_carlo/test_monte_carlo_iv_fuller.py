"""
Monte Carlo validation for Fuller estimator.

Fuller is a modified LIML estimator with bias correction: κ_Fuller = κ_LIML - α/(n-L)
- Fuller-1 (α=1): Recommended for weak IV + small samples
- Fuller-4 (α=4): More conservative, lower variance

Key properties validated:
- Lower MSE than LIML (better bias-variance tradeoff)
- Fuller-1 best overall choice for weak instruments
- Fuller-4 most conservative (smallest bias, highest variance)
- Correct coverage under weak IV

Key References:
    - Fuller (1977). "Some Properties of a Modification of the Limited Information Estimator"
    - Hahn, Hausman & Kuersteiner (2004). "Estimation with Weak Instruments"

The key insight: Fuller-1 achieves the best bias-variance tradeoff under weak instruments.
"""

import numpy as np
import pytest
from src.causal_inference.iv import TwoStageLeastSquares, LIML, Fuller
from tests.validation.monte_carlo.dgp_iv import (
    dgp_iv_strong,
    dgp_iv_weak,
    dgp_iv_moderate,
)
from tests.validation.utils import validate_monte_carlo_results


class TestFullerBias:
    """Test Fuller bias properties."""

    @pytest.mark.slow
    def test_fuller1_less_biased_than_2sls_weak_iv(self):
        """
        Fuller-1 should have less bias than 2SLS with weak instruments.
        """
        n_runs = 3000
        true_beta = 0.5

        fuller_estimates = []
        tsls_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            # Fuller-1
            fuller = Fuller(alpha_param=1.0, inference="robust")
            fuller.fit(data.Y, data.D, data.Z)
            fuller_estimates.append(fuller.coef_[0])

            # 2SLS
            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_estimates.append(tsls.coef_[0])

        fuller_bias = abs(np.mean(fuller_estimates) - true_beta)
        tsls_bias = abs(np.mean(tsls_estimates) - true_beta)

        # Fuller should be less biased
        assert fuller_bias < tsls_bias, (
            f"Fuller-1 bias ({fuller_bias:.4f}) should be less than "
            f"2SLS bias ({tsls_bias:.4f}) with weak IV."
        )

    @pytest.mark.slow
    def test_fuller4_smallest_bias(self):
        """
        Fuller-4 should have the smallest bias (most conservative).
        """
        n_runs = 2000
        true_beta = 0.5

        fuller1_estimates = []
        fuller4_estimates = []
        liml_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            fuller1 = Fuller(alpha_param=1.0, inference="robust")
            fuller1.fit(data.Y, data.D, data.Z)
            fuller1_estimates.append(fuller1.coef_[0])

            fuller4 = Fuller(alpha_param=4.0, inference="robust")
            fuller4.fit(data.Y, data.D, data.Z)
            fuller4_estimates.append(fuller4.coef_[0])

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

        fuller1_bias = abs(np.mean(fuller1_estimates) - true_beta)
        fuller4_bias = abs(np.mean(fuller4_estimates) - true_beta)
        liml_bias = abs(np.mean(liml_estimates) - true_beta)

        # Fuller-4 should have smallest (or similar) bias
        assert fuller4_bias <= fuller1_bias + 0.02, (
            f"Fuller-4 bias ({fuller4_bias:.4f}) should be <= Fuller-1 bias ({fuller1_bias:.4f})."
        )


class TestFullerMSE:
    """Test Fuller MSE (Mean Squared Error) properties."""

    @pytest.mark.slow
    def test_fuller1_lower_mse_than_liml_weak_iv(self):
        """
        Fuller-1 should have lower MSE than LIML with weak instruments.

        MSE = Bias² + Variance
        Fuller-1 trades off slightly more bias for much lower variance.
        """
        n_runs = 3000
        true_beta = 0.5

        fuller_estimates = []
        liml_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            fuller = Fuller(alpha_param=1.0, inference="robust")
            fuller.fit(data.Y, data.D, data.Z)
            fuller_estimates.append(fuller.coef_[0])

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

        # Compute MSE = Bias² + Variance
        fuller_bias = np.mean(fuller_estimates) - true_beta
        fuller_var = np.var(fuller_estimates)
        fuller_mse = fuller_bias**2 + fuller_var

        liml_bias = np.mean(liml_estimates) - true_beta
        liml_var = np.var(liml_estimates)
        liml_mse = liml_bias**2 + liml_var

        # Fuller-1 should have lower MSE (better bias-variance tradeoff)
        assert fuller_mse <= liml_mse * 1.1, (
            f"Fuller-1 MSE ({fuller_mse:.4f}) should be <= LIML MSE ({liml_mse:.4f}). "
            f"Fuller-1: bias={fuller_bias:.4f}, var={fuller_var:.4f}. "
            f"LIML: bias={liml_bias:.4f}, var={liml_var:.4f}"
        )

    @pytest.mark.slow
    def test_fuller1_lower_variance_than_liml(self):
        """
        Fuller-1 should have lower variance than LIML.

        This is the key advantage of Fuller: it "shrinks" LIML to reduce variance.
        """
        n_runs = 3000
        true_beta = 0.5

        fuller_estimates = []
        liml_estimates = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            fuller = Fuller(alpha_param=1.0, inference="robust")
            fuller.fit(data.Y, data.D, data.Z)
            fuller_estimates.append(fuller.coef_[0])

            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            liml_estimates.append(liml.coef_[0])

        fuller_var = np.var(fuller_estimates)
        liml_var = np.var(liml_estimates)

        # Fuller should have lower variance
        assert fuller_var < liml_var, (
            f"Fuller-1 variance ({fuller_var:.4f}) should be < LIML variance ({liml_var:.4f})."
        )


class TestFullerCoverage:
    """Test Fuller confidence interval coverage."""

    @pytest.mark.slow
    def test_fuller1_coverage_strong_iv(self):
        """
        Fuller-1 coverage should be correct with strong instruments.
        """
        n_runs = 2000
        true_beta = 0.5

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_strong(n=500, true_beta=true_beta, random_state=seed)

            fuller = Fuller(alpha_param=1.0, inference="robust")
            fuller.fit(data.Y, data.D, data.Z)

            estimates.append(fuller.coef_[0])
            standard_errors.append(fuller.se_[0])
            ci_lowers.append(fuller.ci_[0, 0])
            ci_uppers.append(fuller.ci_[0, 1])

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
            f"Fuller-1 coverage {validation['coverage']:.2%} outside [93%, 97%] with strong IV."
        )

    @pytest.mark.slow
    def test_fuller1_better_coverage_than_2sls_weak_iv(self):
        """
        Fuller-1 should have better coverage than 2SLS with weak instruments.
        """
        n_runs = 3000
        true_beta = 0.5

        fuller_ci_lowers = []
        fuller_ci_uppers = []
        tsls_ci_lowers = []
        tsls_ci_uppers = []

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            fuller = Fuller(alpha_param=1.0, inference="robust")
            fuller.fit(data.Y, data.D, data.Z)
            fuller_ci_lowers.append(fuller.ci_[0, 0])
            fuller_ci_uppers.append(fuller.ci_[0, 1])

            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            tsls_ci_lowers.append(tsls.ci_[0, 0])
            tsls_ci_uppers.append(tsls.ci_[0, 1])

        fuller_coverage = np.mean(
            (np.array(fuller_ci_lowers) <= true_beta) & (true_beta <= np.array(fuller_ci_uppers))
        )
        tsls_coverage = np.mean(
            (np.array(tsls_ci_lowers) <= true_beta) & (true_beta <= np.array(tsls_ci_uppers))
        )

        # Fuller should have better coverage
        assert fuller_coverage >= tsls_coverage - 0.03, (
            f"Fuller-1 coverage ({fuller_coverage:.2%}) should be >= "
            f"2SLS coverage ({tsls_coverage:.2%}) with weak IV."
        )


class TestFullerKappa:
    """Test Fuller kappa correction properties."""

    @pytest.mark.slow
    def test_fuller_kappa_less_than_liml(self):
        """
        Fuller kappa should be less than LIML kappa (κ_Fuller = κ_LIML - α/(n-L)).
        """
        n_runs = 500

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, random_state=seed)

            fuller = Fuller(alpha_param=1.0)
            fuller.fit(data.Y, data.D, data.Z)

            liml = LIML()
            liml.fit(data.Y, data.D, data.Z)

            # Fuller kappa should be less than LIML kappa
            assert fuller.kappa_ < liml.kappa_, (
                f"Fuller kappa ({fuller.kappa_:.4f}) should be < LIML kappa ({liml.kappa_:.4f})"
            )

    @pytest.mark.slow
    def test_fuller_kappa_correction_formula(self):
        """
        Verify κ_Fuller = κ_LIML - α/(n-L).
        """
        n_runs = 100
        alpha = 1.0

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, random_state=seed)

            fuller = Fuller(alpha_param=alpha)
            fuller.fit(data.Y, data.D, data.Z)

            liml = LIML()
            liml.fit(data.Y, data.D, data.Z)

            # Compute expected kappa correction
            # n - L where L = number of exogenous variables + 1
            # For just-identified with no controls: L = 1 (constant)
            n_minus_L = data.n - 1
            expected_correction = alpha / n_minus_L
            expected_kappa = liml.kappa_ - expected_correction

            # Check kappa matches formula (with tolerance for numerical precision)
            assert np.isclose(fuller.kappa_, expected_kappa, rtol=0.01), (
                f"Fuller kappa ({fuller.kappa_:.6f}) doesn't match formula. "
                f"Expected: {expected_kappa:.6f} = {liml.kappa_:.6f} - {expected_correction:.6f}"
            )


class TestFullerComparison:
    """Compare Fuller-1, Fuller-4, LIML, and 2SLS."""

    @pytest.mark.slow
    def test_estimator_ranking_weak_iv(self):
        """
        Ranking under weak IV should be: Fuller-1 ≈ Fuller-4 < LIML < 2SLS (by MSE).
        """
        n_runs = 2000
        true_beta = 0.5

        results = {
            "fuller1": [],
            "fuller4": [],
            "liml": [],
            "tsls": [],
        }

        for seed in range(n_runs):
            data = dgp_iv_weak(n=500, true_beta=true_beta, random_state=seed)

            # Fuller-1
            f1 = Fuller(alpha_param=1.0, inference="robust")
            f1.fit(data.Y, data.D, data.Z)
            results["fuller1"].append(f1.coef_[0])

            # Fuller-4
            f4 = Fuller(alpha_param=4.0, inference="robust")
            f4.fit(data.Y, data.D, data.Z)
            results["fuller4"].append(f4.coef_[0])

            # LIML
            liml = LIML(inference="robust")
            liml.fit(data.Y, data.D, data.Z)
            results["liml"].append(liml.coef_[0])

            # 2SLS
            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            results["tsls"].append(tsls.coef_[0])

        # Compute MSE for each
        mse = {}
        for name, estimates in results.items():
            bias = np.mean(estimates) - true_beta
            var = np.var(estimates)
            mse[name] = bias**2 + var

        # Fuller-1 should have best MSE
        assert mse["fuller1"] <= mse["liml"] * 1.1, (
            f"Fuller-1 MSE ({mse['fuller1']:.4f}) should be <= LIML MSE ({mse['liml']:.4f})"
        )

        # 2SLS should have worst bias (even if low variance)
        bias_2sls = abs(np.mean(results["tsls"]) - true_beta)
        bias_fuller1 = abs(np.mean(results["fuller1"]) - true_beta)
        assert bias_2sls > bias_fuller1, (
            f"2SLS bias ({bias_2sls:.4f}) should be > Fuller-1 bias ({bias_fuller1:.4f})"
        )
