"""
Monte Carlo validation for classic 2×2 DiD estimator.

Validates statistical properties:
- Bias < 0.10 (observational standard)
- Coverage 93-97% (for 95% CI)
- SE accuracy within 15% (cluster-robust)

Key tests:
1. Unbiasedness with homoskedastic errors
2. Coverage with cluster-robust SEs
3. Robustness to heteroskedasticity
4. Serial correlation handling

References:
    Bertrand, Duflo, Mullainathan (2004). How much should we trust
    differences-in-differences estimates?
"""

import numpy as np
import pytest
from src.causal_inference.did import did_2x2
from tests.validation.monte_carlo.dgp_did import (
    dgp_did_2x2_simple,
    dgp_did_2x2_heteroskedastic,
    dgp_did_2x2_serial_correlation,
)
from tests.validation.utils import validate_monte_carlo_results


class TestDiD2x2MonteCarloUnbiasedness:
    """Monte Carlo validation of DiD unbiasedness."""

    @pytest.mark.slow
    def test_did_2x2_unbiased_simple(self):
        """
        Validate did_2x2 is unbiased with simple DGP.

        DGP: Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it
        True ATT = 2.0, n=200 (100 treated, 100 control)

        Expected: Bias < 0.10
        """
        n_runs = 2000
        true_att = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_did_2x2_simple(
                n_treated=100,
                n_control=100,
                n_pre=1,
                n_post=1,
                true_att=true_att,
                random_state=seed,
            )

            result = did_2x2(
                outcomes=data.outcomes,
                treatment=data.treatment,
                post=data.post,
                unit_id=data.unit_id,
                cluster_se=True,
            )

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_att,
            bias_threshold=0.10,
            coverage_lower=0.93,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["bias_ok"], (
            f"Bias {validation['bias']:.4f} exceeds 0.10. "
            f"Mean estimate: {np.mean(estimates):.4f}, true ATT: {true_att}"
        )

    @pytest.mark.slow
    def test_did_2x2_unbiased_larger_sample(self):
        """
        Validate did_2x2 with larger sample (n=400).

        Larger sample should yield smaller bias and tighter estimates.
        """
        n_runs = 1000
        true_att = 2.0

        estimates = []
        for seed in range(n_runs):
            data = dgp_did_2x2_simple(
                n_treated=200,
                n_control=200,
                true_att=true_att,
                random_state=seed,
            )
            result = did_2x2(data.outcomes, data.treatment, data.post, data.unit_id)
            estimates.append(result["estimate"])

        bias = abs(np.mean(estimates) - true_att)
        assert bias < 0.05, f"Bias {bias:.4f} exceeds 0.05 for n=400"


class TestDiD2x2MonteCarloCoverage:
    """Monte Carlo validation of confidence interval coverage."""

    @pytest.mark.slow
    def test_did_2x2_coverage_cluster_se(self):
        """
        Validate 95% CI coverage with cluster-robust SEs.

        Expected: Coverage 93-97% (allowing for Monte Carlo error)
        """
        n_runs = 2000
        true_att = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_did_2x2_simple(
                n_treated=100,
                n_control=100,
                true_att=true_att,
                random_state=seed,
            )

            result = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                cluster_se=True,
            )

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_att,
            coverage_lower=0.93,
            coverage_upper=0.97,
        )

        assert validation["coverage_ok"], (
            f"Coverage {validation['coverage']:.4f} outside [0.93, 0.97]. "
            f"This may indicate issues with cluster-robust SE computation."
        )

    @pytest.mark.slow
    def test_did_2x2_se_accuracy(self):
        """
        Validate SE accuracy: estimated SEs close to empirical SD.

        SE accuracy = |std(estimates) - mean(SE)| / std(estimates)
        Expected: < 15%
        """
        n_runs = 2000
        true_att = 2.0

        estimates = []
        standard_errors = []

        for seed in range(n_runs):
            data = dgp_did_2x2_simple(
                n_treated=100,
                n_control=100,
                true_att=true_att,
                random_state=seed,
            )

            result = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                cluster_se=True,
            )

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])

        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(standard_errors)
        se_accuracy = abs(empirical_sd - mean_se) / empirical_sd

        assert se_accuracy < 0.15, (
            f"SE accuracy {se_accuracy:.4f} exceeds 15%. "
            f"Empirical SD: {empirical_sd:.4f}, Mean SE: {mean_se:.4f}"
        )


class TestDiD2x2MonteCarloHeteroskedasticity:
    """Monte Carlo validation with heteroskedastic errors."""

    @pytest.mark.slow
    def test_did_2x2_heteroskedastic_unbiased(self):
        """
        Validate did_2x2 remains unbiased with heteroskedastic errors.

        DGP: σ_treated = 2.0, σ_control = 1.0
        """
        n_runs = 1500
        true_att = 2.0

        estimates = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_did_2x2_heteroskedastic(
                n_treated=100,
                n_control=100,
                true_att=true_att,
                sigma_treated=2.0,
                sigma_control=1.0,
                random_state=seed,
            )

            result = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                cluster_se=True,
            )

            estimates.append(result["estimate"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        bias = abs(np.mean(estimates) - true_att)
        assert bias < 0.10, f"Bias {bias:.4f} with heteroskedastic errors"

        # Coverage may be slightly lower with heteroskedasticity
        coverage = np.mean((np.array(ci_lowers) <= true_att) & (true_att <= np.array(ci_uppers)))
        assert coverage > 0.90, f"Coverage {coverage:.4f} too low with heteroskedastic errors"


class TestDiD2x2MonteCarloSerialCorrelation:
    """Monte Carlo validation with serial correlation."""

    @pytest.mark.slow
    def test_did_2x2_serial_correlation_cluster_se(self):
        """
        Validate cluster-robust SEs handle serial correlation.

        DGP: AR(1) errors with ρ=0.5, 10 periods
        Cluster-robust SEs should maintain valid coverage.
        Naive SEs would under-cover (over-reject).
        """
        n_runs = 1000
        true_att = 2.0

        estimates_cluster = []
        ci_lowers_cluster = []
        ci_uppers_cluster = []

        for seed in range(n_runs):
            data = dgp_did_2x2_serial_correlation(
                n_treated=50,
                n_control=50,
                n_pre=5,
                n_post=5,
                true_att=true_att,
                rho=0.5,
                random_state=seed,
            )

            result = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                cluster_se=True,
            )

            estimates_cluster.append(result["estimate"])
            ci_lowers_cluster.append(result["ci_lower"])
            ci_uppers_cluster.append(result["ci_upper"])

        bias = abs(np.mean(estimates_cluster) - true_att)
        assert bias < 0.15, f"Bias {bias:.4f} with serial correlation"

        coverage = np.mean(
            (np.array(ci_lowers_cluster) <= true_att) & (true_att <= np.array(ci_uppers_cluster))
        )
        # With serial correlation, cluster SE is critical
        # Coverage should still be in valid range
        assert coverage > 0.88, (
            f"Coverage {coverage:.4f} too low with AR(1) errors. "
            f"Cluster-robust SEs may not fully account for serial correlation."
        )

    @pytest.mark.slow
    def test_did_2x2_naive_se_undercoverage(self):
        """
        Demonstrate that naive SEs under-cover with serial correlation.

        This is an educational test showing WHY cluster SEs are needed.
        """
        n_runs = 500
        true_att = 2.0

        ci_covers_naive = []
        ci_covers_cluster = []

        for seed in range(n_runs):
            data = dgp_did_2x2_serial_correlation(
                n_treated=50,
                n_control=50,
                n_pre=5,
                n_post=5,
                true_att=true_att,
                rho=0.5,
                random_state=seed,
            )

            # Naive SEs (incorrect)
            result_naive = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                se_method="naive",
            )
            ci_covers_naive.append(result_naive["ci_lower"] <= true_att <= result_naive["ci_upper"])

            # Cluster SEs (correct)
            result_cluster = did_2x2(
                data.outcomes,
                data.treatment,
                data.post,
                data.unit_id,
                se_method="cluster",
            )
            ci_covers_cluster.append(
                result_cluster["ci_lower"] <= true_att <= result_cluster["ci_upper"]
            )

        coverage_naive = np.mean(ci_covers_naive)
        coverage_cluster = np.mean(ci_covers_cluster)

        # Naive SEs should under-cover (over-reject)
        # This demonstrates the Bertrand et al. (2004) problem
        assert coverage_naive < coverage_cluster, (
            f"Expected naive coverage ({coverage_naive:.4f}) < "
            f"cluster coverage ({coverage_cluster:.4f})"
        )

        # Naive coverage should be substantially below nominal
        # With ρ=0.5 and 10 periods, naive SEs are too small
        assert coverage_naive < 0.90, (
            f"Naive coverage {coverage_naive:.4f} not low enough to demonstrate "
            f"serial correlation problem. Expected < 0.90."
        )


class TestDiD2x2MonteCarloDiagnostics:
    """Diagnostic tests for Monte Carlo validation."""

    def test_dgp_generates_expected_structure(self):
        """Verify DGP generates data with expected structure."""
        data = dgp_did_2x2_simple(
            n_treated=50,
            n_control=50,
            n_pre=2,
            n_post=3,
            random_state=42,
        )

        # Check dimensions
        n_expected = 100 * 5  # 100 units × 5 periods
        assert len(data.outcomes) == n_expected
        assert len(data.treatment) == n_expected
        assert len(data.post) == n_expected
        assert len(data.unit_id) == n_expected

        # Check treatment/control split
        unique_units = np.unique(data.unit_id)
        n_treated_units = sum(data.treatment[data.unit_id == u][0] == 1 for u in unique_units)
        assert n_treated_units == 50

        # Check pre/post split
        n_post_obs = np.sum(data.post == 1)
        assert n_post_obs == 100 * 3  # 100 units × 3 post periods

    def test_estimate_distribution_centered(self):
        """Verify estimates are centered around true ATT."""
        n_runs = 300
        true_att = 2.0

        estimates = []
        for seed in range(n_runs):
            data = dgp_did_2x2_simple(
                n_treated=100, n_control=100, true_att=true_att, random_state=seed
            )
            result = did_2x2(data.outcomes, data.treatment, data.post, data.unit_id)
            estimates.append(result["estimate"])

        mean_est = np.mean(estimates)
        assert abs(mean_est - true_att) < 0.15, (
            f"Estimates not centered: mean={mean_est:.4f}, true={true_att}"
        )

        # Check reasonable variance
        std_est = np.std(estimates)
        assert 0.05 < std_est < 0.5, f"Unexpected estimate SD: {std_est:.4f}"
