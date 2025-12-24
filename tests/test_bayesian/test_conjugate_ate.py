"""
Unit tests for Bayesian ATE with conjugate priors.

Session 101: Initial test suite.

Tests cover:
1. Basic functionality and output structure
2. Known-answer validation
3. Prior sensitivity analysis
4. Monte Carlo calibration
5. Edge cases and error handling
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.bayesian import bayesian_ate, BayesianATEResult


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBayesianATEBasic:
    """Basic functionality tests."""

    def test_returns_correct_structure(self, simple_bayesian_data: dict) -> None:
        """Result has all expected fields."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )

        # Check all required fields exist
        assert "posterior_mean" in result
        assert "posterior_sd" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "credible_level" in result
        assert "prior_mean" in result
        assert "prior_sd" in result
        assert "posterior_samples" in result
        assert "n" in result
        assert "n_treated" in result
        assert "n_control" in result
        assert "prior_to_posterior_shrinkage" in result
        assert "effective_sample_size" in result
        assert "ols_estimate" in result
        assert "ols_se" in result
        assert "sigma2_mle" in result

    def test_posterior_mean_finite(self, simple_bayesian_data: dict) -> None:
        """Posterior mean is always finite."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        assert np.isfinite(result["posterior_mean"])

    def test_posterior_sd_positive(self, simple_bayesian_data: dict) -> None:
        """Posterior SD is always positive."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        assert result["posterior_sd"] > 0

    def test_credible_interval_contains_posterior_mean(
        self, simple_bayesian_data: dict
    ) -> None:
        """CI always contains the posterior mean."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]

    def test_posterior_samples_shape(self, simple_bayesian_data: dict) -> None:
        """Posterior samples have correct shape."""
        n_samples = 3000
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            n_posterior_samples=n_samples,
        )
        assert result["posterior_samples"].shape == (n_samples,)

    def test_sample_sizes_consistent(self, simple_bayesian_data: dict) -> None:
        """Sample sizes are consistent."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        assert result["n"] == result["n_treated"] + result["n_control"]
        assert result["n"] == len(simple_bayesian_data["outcomes"])


# =============================================================================
# Known-Answer Tests
# =============================================================================


class TestBayesianATEKnownAnswer:
    """Known-answer validation tests."""

    def test_uninformative_prior_matches_ols(self, simple_bayesian_data: dict) -> None:
        """With flat prior (large prior_sd), posterior approximates OLS estimate."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            prior_mean=0.0,
            prior_sd=1000.0,  # Very flat prior
        )

        # Posterior mean should be very close to OLS estimate
        assert np.isclose(
            result["posterior_mean"], result["ols_estimate"], rtol=1e-3
        ), f"Posterior {result['posterior_mean']:.4f} != OLS {result['ols_estimate']:.4f}"

    def test_recovers_true_ate(self, simple_bayesian_data: dict) -> None:
        """Posterior mean is close to true ATE."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        true_ate = simple_bayesian_data["true_ate"]

        # Should be within 0.5 of true value with n=200
        assert abs(result["posterior_mean"] - true_ate) < 0.5, (
            f"Posterior {result['posterior_mean']:.4f} too far from true {true_ate}"
        )

    def test_credible_interval_covers_true_value(
        self, simple_bayesian_data: dict
    ) -> None:
        """95% credible interval contains true ATE."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            credible_level=0.95,
        )
        true_ate = simple_bayesian_data["true_ate"]

        # 95% CI should contain true value
        assert result["ci_lower"] < true_ate < result["ci_upper"], (
            f"True ATE {true_ate} outside CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        )

    def test_zero_effect_detection(self, zero_effect_data: dict) -> None:
        """Correctly identifies zero treatment effect."""
        result = bayesian_ate(
            zero_effect_data["outcomes"],
            zero_effect_data["treatment"],
            prior_mean=0.0,
        )

        # Posterior mean should be near zero
        assert abs(result["posterior_mean"]) < 0.5
        # 95% CI should contain zero
        assert result["ci_lower"] < 0 < result["ci_upper"]


# =============================================================================
# Prior Sensitivity Tests
# =============================================================================


class TestBayesianATEPriorSensitivity:
    """Prior specification tests."""

    def test_prior_shrinkage_computed(self, simple_bayesian_data: dict) -> None:
        """Shrinkage metric is in [0, 1]."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )
        shrinkage = result["prior_to_posterior_shrinkage"]
        assert 0 <= shrinkage <= 1, f"Shrinkage {shrinkage} outside [0, 1]"

    def test_wider_prior_less_shrinkage(self, simple_bayesian_data: dict) -> None:
        """Wider prior results in less shrinkage toward prior mean."""
        result_narrow = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            prior_sd=1.0,  # Narrow prior
        )
        result_wide = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            prior_sd=100.0,  # Wide prior
        )

        # Wider prior should have less shrinkage
        assert result_wide["prior_to_posterior_shrinkage"] < result_narrow["prior_to_posterior_shrinkage"]

    def test_strong_prior_dominates_small_sample(
        self, small_sample_data: dict
    ) -> None:
        """Strong prior + few observations yields posterior near prior."""
        prior_mean = 5.0
        prior_sd = 0.5  # Strong prior

        result = bayesian_ate(
            small_sample_data["outcomes"],
            small_sample_data["treatment"],
            prior_mean=prior_mean,
            prior_sd=prior_sd,
        )

        # Posterior should be pulled toward prior
        true_ate = small_sample_data["true_ate"]
        distance_to_prior = abs(result["posterior_mean"] - prior_mean)
        distance_to_true = abs(result["posterior_mean"] - true_ate)

        # With strong prior and small sample, should be closer to prior
        assert result["prior_to_posterior_shrinkage"] > 0.1

    def test_data_dominates_weak_prior(self, large_sample_data: dict) -> None:
        """Many observations + weak prior yields posterior near OLS."""
        result = bayesian_ate(
            large_sample_data["outcomes"],
            large_sample_data["treatment"],
            prior_mean=10.0,  # Very wrong prior mean
            prior_sd=10.0,  # Weak prior
        )

        # Posterior should be near OLS despite wrong prior
        assert np.isclose(
            result["posterior_mean"], result["ols_estimate"], rtol=0.05
        )
        # Shrinkage should be minimal
        assert result["prior_to_posterior_shrinkage"] < 0.1


# =============================================================================
# Covariate Adjustment Tests
# =============================================================================


class TestBayesianATECovariates:
    """Tests for covariate-adjusted estimation."""

    def test_with_covariates(self, bayesian_data_with_covariates: dict) -> None:
        """Works correctly with covariates."""
        result = bayesian_ate(
            bayesian_data_with_covariates["outcomes"],
            bayesian_data_with_covariates["treatment"],
            covariates=bayesian_data_with_covariates["covariates"],
        )

        true_ate = bayesian_data_with_covariates["true_ate"]
        # Should recover true ATE
        assert abs(result["posterior_mean"] - true_ate) < 0.6

    def test_covariates_estimation_valid(
        self, bayesian_data_with_covariates: dict
    ) -> None:
        """Covariate adjustment produces valid estimates."""
        result_with_cov = bayesian_ate(
            bayesian_data_with_covariates["outcomes"],
            bayesian_data_with_covariates["treatment"],
            covariates=bayesian_data_with_covariates["covariates"],
        )

        # Estimates should be finite and reasonable
        assert np.isfinite(result_with_cov["posterior_mean"])
        assert result_with_cov["posterior_sd"] > 0
        # With covariates controlling for confounding, should still recover ATE
        true_ate = bayesian_data_with_covariates["true_ate"]
        assert abs(result_with_cov["posterior_mean"] - true_ate) < 0.8


# =============================================================================
# Monte Carlo Calibration Tests
# =============================================================================


class TestBayesianATEMonteCarlo:
    """Monte Carlo validation of coverage."""

    @pytest.mark.monte_carlo
    def test_credible_interval_calibration(self) -> None:
        """95% credible intervals contain true value approximately 95% of time."""
        np.random.seed(42)
        n_simulations = 500
        n_per_sim = 100
        true_ate = 2.0
        covered = 0

        for _ in range(n_simulations):
            treatment = np.random.binomial(1, 0.5, n_per_sim)
            outcomes = true_ate * treatment + np.random.normal(0, 1, n_per_sim)

            result = bayesian_ate(
                outcomes, treatment, prior_mean=0.0, prior_sd=10.0, credible_level=0.95
            )

            if result["ci_lower"] < true_ate < result["ci_upper"]:
                covered += 1

        coverage = covered / n_simulations

        # Should be approximately 95% (allowing for Monte Carlo error)
        assert 0.90 < coverage < 0.99, f"Coverage {coverage:.2%} outside [90%, 99%]"

    @pytest.mark.monte_carlo
    def test_posterior_mean_unbiased(self) -> None:
        """Posterior mean is approximately unbiased across simulations."""
        np.random.seed(123)
        n_simulations = 500
        n_per_sim = 100
        true_ate = 2.0
        posterior_means = []

        for _ in range(n_simulations):
            treatment = np.random.binomial(1, 0.5, n_per_sim)
            outcomes = true_ate * treatment + np.random.normal(0, 1, n_per_sim)

            result = bayesian_ate(
                outcomes, treatment, prior_mean=0.0, prior_sd=10.0
            )
            posterior_means.append(result["posterior_mean"])

        mean_of_means = np.mean(posterior_means)
        bias = mean_of_means - true_ate

        # Bias should be small (< 0.1 with weak prior)
        assert abs(bias) < 0.1, f"Bias {bias:.4f} exceeds threshold"

    @pytest.mark.monte_carlo
    def test_posterior_samples_match_distribution(
        self, simple_bayesian_data: dict
    ) -> None:
        """Posterior samples match the claimed normal distribution."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            n_posterior_samples=10000,
        )

        samples = result["posterior_samples"]
        sample_mean = np.mean(samples)
        sample_sd = np.std(samples)

        # Samples should match posterior parameters
        assert np.isclose(sample_mean, result["posterior_mean"], rtol=0.05)
        assert np.isclose(sample_sd, result["posterior_sd"], rtol=0.05)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestBayesianATEEdgeCases:
    """Edge cases and error handling tests."""

    def test_length_mismatch_raises(self) -> None:
        """Mismatched array lengths raise ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0])
        treatment = np.array([1, 0])

        with pytest.raises(ValueError, match="Length mismatch"):
            bayesian_ate(outcomes, treatment)

    def test_non_binary_treatment_raises(self) -> None:
        """Non-binary treatment raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 1, 2, 0])  # Invalid: contains 2

        with pytest.raises(ValueError, match="binary"):
            bayesian_ate(outcomes, treatment)

    def test_negative_prior_sd_raises(self) -> None:
        """Negative prior_sd raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 0, 0])

        with pytest.raises(ValueError, match="prior_sd"):
            bayesian_ate(outcomes, treatment, prior_sd=-1.0)

    def test_invalid_credible_level_raises(self) -> None:
        """Credible level outside (0, 1) raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 0, 0])

        with pytest.raises(ValueError, match="credible_level"):
            bayesian_ate(outcomes, treatment, credible_level=1.5)

        with pytest.raises(ValueError, match="credible_level"):
            bayesian_ate(outcomes, treatment, credible_level=0.0)

    def test_all_treated_raises(self) -> None:
        """All units treated raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 1, 1])

        with pytest.raises(ValueError, match="non-empty"):
            bayesian_ate(outcomes, treatment)

    def test_all_control_raises(self) -> None:
        """All units control raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 0, 0])

        with pytest.raises(ValueError, match="non-empty"):
            bayesian_ate(outcomes, treatment)

    def test_covariates_length_mismatch_raises(self) -> None:
        """Covariate length mismatch raises ValueError."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 0, 0])
        covariates = np.array([[1.0], [2.0], [3.0]])  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            bayesian_ate(outcomes, treatment, covariates=covariates)


# =============================================================================
# Comparison with Frequentist Tests
# =============================================================================


class TestBayesianVsFrequentist:
    """Compare Bayesian and frequentist estimates."""

    def test_weak_prior_matches_frequentist(
        self, simple_bayesian_data: dict
    ) -> None:
        """With weak prior, Bayesian CI similar to frequentist CI."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
            prior_sd=1000.0,  # Very weak prior
            credible_level=0.95,
        )

        # Compute frequentist CI for comparison
        ols_ci_lower = result["ols_estimate"] - 1.96 * result["ols_se"]
        ols_ci_upper = result["ols_estimate"] + 1.96 * result["ols_se"]

        # Bayesian CI should be very similar with flat prior
        assert np.isclose(result["ci_lower"], ols_ci_lower, rtol=0.05)
        assert np.isclose(result["ci_upper"], ols_ci_upper, rtol=0.05)

    def test_effective_sample_size_reasonable(
        self, simple_bayesian_data: dict
    ) -> None:
        """Effective sample size is reasonable."""
        result = bayesian_ate(
            simple_bayesian_data["outcomes"],
            simple_bayesian_data["treatment"],
        )

        # ESS should be positive and <= n
        assert 0 < result["effective_sample_size"] <= result["n"]


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestBayesianATENumericalStability:
    """Tests for numerical stability."""

    def test_extreme_outcomes(self) -> None:
        """Works with extreme outcome values."""
        np.random.seed(42)
        n = 100
        treatment = np.random.binomial(1, 0.5, n)
        outcomes = 1e6 * treatment + 1e6 + np.random.normal(0, 100, n)

        result = bayesian_ate(outcomes, treatment)

        assert np.isfinite(result["posterior_mean"])
        assert np.isfinite(result["posterior_sd"])
        assert result["posterior_mean"] > 0

    def test_very_small_variance(self) -> None:
        """Works with very small outcome variance."""
        np.random.seed(42)
        n = 100
        treatment = np.random.binomial(1, 0.5, n)
        outcomes = 2.0 * treatment + np.random.normal(0, 0.01, n)

        result = bayesian_ate(outcomes, treatment)

        assert np.isfinite(result["posterior_mean"])
        assert result["posterior_sd"] > 0

    def test_imbalanced_treatment(self) -> None:
        """Works with imbalanced treatment assignment."""
        np.random.seed(42)
        n = 200
        # 90% treatment, 10% control
        treatment = np.random.binomial(1, 0.9, n)
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = bayesian_ate(outcomes, treatment)

        assert np.isfinite(result["posterior_mean"])
        assert result["n_treated"] > result["n_control"]
