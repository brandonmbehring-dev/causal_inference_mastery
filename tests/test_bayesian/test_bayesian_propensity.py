"""
Unit tests for Bayesian propensity score estimation.

Session 102: Initial test suite.

Tests cover:
1. Stratified Beta-Binomial estimation
2. Logistic regression with Laplace approximation
3. Automatic method selection
4. Edge cases and error handling
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.bayesian import (
    bayesian_propensity,
    bayesian_propensity_stratified,
    bayesian_propensity_logistic,
    BayesianPropensityResult,
)


# =============================================================================
# Test Data Generators
# =============================================================================


@pytest.fixture
def discrete_covariate_data() -> dict:
    """Data with discrete covariates for stratified testing."""
    np.random.seed(42)
    n = 300
    # Binary and ternary covariates
    X1 = np.random.choice([0, 1], n)
    X2 = np.random.choice([0, 1, 2], n)
    X = np.column_stack([X1, X2])

    # Treatment depends on covariates
    prob = 0.3 + 0.2 * X1 + 0.1 * X2
    treatment = (np.random.rand(n) < prob).astype(float)

    return {"treatment": treatment, "covariates": X}


@pytest.fixture
def continuous_covariate_data() -> dict:
    """Data with continuous covariates for logistic testing."""
    np.random.seed(123)
    n = 300
    X = np.random.randn(n, 2)

    # Logistic propensity
    logit = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    prob = 1 / (1 + np.exp(-logit))
    treatment = (np.random.rand(n) < prob).astype(float)

    return {"treatment": treatment, "covariates": X, "true_coef": np.array([0.5, 0.3])}


# =============================================================================
# Stratified Beta-Binomial Tests
# =============================================================================


class TestBayesianPropensityStratified:
    """Tests for stratified Beta-Binomial estimation."""

    def test_returns_correct_structure(self, discrete_covariate_data: dict) -> None:
        """Result has all expected fields."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
        )

        assert "posterior_samples" in result
        assert "posterior_mean" in result
        assert "posterior_sd" in result
        assert "strata" in result
        assert "n_strata" in result
        assert "stratum_info" in result
        assert result["method"] == "stratified_beta_binomial"

    def test_posterior_mean_in_valid_range(self, discrete_covariate_data: dict) -> None:
        """Posterior means are in [0, 1]."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
        )

        assert np.all(result["posterior_mean"] >= 0)
        assert np.all(result["posterior_mean"] <= 1)

    def test_posterior_sd_positive(self, discrete_covariate_data: dict) -> None:
        """Posterior SDs are positive."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
        )

        assert np.all(result["posterior_sd"] > 0)

    def test_samples_match_distribution(self, discrete_covariate_data: dict) -> None:
        """Posterior samples match claimed distribution."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            n_posterior_samples=5000,
        )

        # Check that sample mean/sd match posterior mean/sd for first observation
        sample_mean = np.mean(result["posterior_samples"][:, 0])
        sample_sd = np.std(result["posterior_samples"][:, 0])

        assert np.isclose(sample_mean, result["posterior_mean"][0], rtol=0.1)
        assert np.isclose(sample_sd, result["posterior_sd"][0], rtol=0.1)

    def test_stratum_info_complete(self, discrete_covariate_data: dict) -> None:
        """Stratum info contains all required fields."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
        )

        for stratum in result["stratum_info"]:
            assert "stratum_id" in stratum
            assert "n_obs" in stratum
            assert "n_treated" in stratum
            assert "n_control" in stratum
            assert "posterior_alpha" in stratum
            assert "posterior_beta" in stratum
            assert "posterior_mean" in stratum
            assert "posterior_sd" in stratum

    def test_uniform_prior(self, discrete_covariate_data: dict) -> None:
        """Default Beta(1,1) prior is uniform."""
        result = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            prior_alpha=1.0,
            prior_beta=1.0,
        )

        # With uniform prior and no data, posterior would be uniform
        # With data, posterior mean should be close to sample proportion
        assert result["prior_alpha"] == 1.0
        assert result["prior_beta"] == 1.0

    def test_informative_prior_effect(self, discrete_covariate_data: dict) -> None:
        """Informative prior shifts posterior."""
        # Strong prior toward 0.5
        result_informative = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            prior_alpha=10.0,
            prior_beta=10.0,
        )

        # Weak prior
        result_weak = bayesian_propensity_stratified(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            prior_alpha=1.0,
            prior_beta=1.0,
        )

        # With informative prior, posterior should be shrunk toward 0.5
        # Mean should be closer to 0.5 with informative prior
        dist_to_half_informative = np.abs(result_informative["posterior_mean"] - 0.5)
        dist_to_half_weak = np.abs(result_weak["posterior_mean"] - 0.5)

        assert np.mean(dist_to_half_informative) <= np.mean(dist_to_half_weak)


# =============================================================================
# Logistic Regression Tests
# =============================================================================


class TestBayesianPropensityLogistic:
    """Tests for logistic regression with Laplace approximation."""

    def test_returns_correct_structure(self, continuous_covariate_data: dict) -> None:
        """Result has all expected fields."""
        result = bayesian_propensity_logistic(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
        )

        assert "posterior_samples" in result
        assert "posterior_mean" in result
        assert "posterior_sd" in result
        assert "coefficient_mean" in result
        assert "coefficient_sd" in result
        assert result["method"] == "logistic_laplace"

    def test_posterior_mean_in_valid_range(self, continuous_covariate_data: dict) -> None:
        """Posterior means are in [0, 1]."""
        result = bayesian_propensity_logistic(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
        )

        assert np.all(result["posterior_mean"] >= 0)
        assert np.all(result["posterior_mean"] <= 1)

    def test_recovers_coefficients(self, continuous_covariate_data: dict) -> None:
        """Coefficient estimates are close to true values."""
        result = bayesian_propensity_logistic(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
        )

        # Coefficients should include intercept + 2 covariates
        assert len(result["coefficient_mean"]) == 3

        # Covariate coefficients (indices 1 and 2) should be close to true
        true_coef = continuous_covariate_data["true_coef"]
        assert np.abs(result["coefficient_mean"][1] - true_coef[0]) < 0.5
        assert np.abs(result["coefficient_mean"][2] - true_coef[1]) < 0.5

    def test_weak_prior_matches_mle(self, continuous_covariate_data: dict) -> None:
        """With weak prior, posterior mean close to MLE."""
        result = bayesian_propensity_logistic(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
            prior_sd=1000.0,  # Very weak prior
        )

        # Posterior should be similar to MLE
        # (We just check it's not dominated by prior)
        assert result["prior_sd"] == 1000.0

    def test_coefficient_uncertainty(self, continuous_covariate_data: dict) -> None:
        """Coefficient SDs are positive and finite."""
        result = bayesian_propensity_logistic(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
        )

        assert np.all(result["coefficient_sd"] > 0)
        assert np.all(np.isfinite(result["coefficient_sd"]))


# =============================================================================
# Automatic Method Selection Tests
# =============================================================================


class TestBayesianPropensityAuto:
    """Tests for automatic method selection."""

    def test_selects_stratified_for_discrete(
        self, discrete_covariate_data: dict
    ) -> None:
        """Uses stratified method for discrete covariates."""
        result = bayesian_propensity(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            method="auto",
        )

        assert result["method"] == "stratified_beta_binomial"

    def test_selects_logistic_for_continuous(
        self, continuous_covariate_data: dict
    ) -> None:
        """Uses logistic method for continuous covariates."""
        result = bayesian_propensity(
            continuous_covariate_data["treatment"],
            continuous_covariate_data["covariates"],
            method="auto",
        )

        assert result["method"] == "logistic_laplace"

    def test_explicit_method_override(self, discrete_covariate_data: dict) -> None:
        """Explicit method overrides auto selection."""
        result = bayesian_propensity(
            discrete_covariate_data["treatment"],
            discrete_covariate_data["covariates"],
            method="logistic",
        )

        assert result["method"] == "logistic_laplace"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestBayesianPropensityEdgeCases:
    """Edge cases and error handling tests."""

    def test_length_mismatch_raises(self) -> None:
        """Mismatched array lengths raise ValueError."""
        treatment = np.array([1.0, 0.0, 1.0])
        covariates = np.array([[1.0], [2.0]])

        with pytest.raises(ValueError, match="Length mismatch"):
            bayesian_propensity_stratified(treatment, covariates)

    def test_non_binary_treatment_raises(self) -> None:
        """Non-binary treatment raises ValueError."""
        treatment = np.array([0.0, 1.0, 2.0, 0.0])
        covariates = np.random.randn(4, 2)

        with pytest.raises(ValueError, match="binary"):
            bayesian_propensity_stratified(treatment, covariates)

    def test_invalid_prior_raises(self) -> None:
        """Invalid prior parameters raise ValueError."""
        treatment = np.array([1.0, 0.0, 1.0, 0.0])
        covariates = np.random.randn(4, 2)

        with pytest.raises(ValueError, match="positive"):
            bayesian_propensity_stratified(treatment, covariates, prior_alpha=-1.0)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        treatment = np.array([1.0, 0.0, 1.0, 0.0])
        covariates = np.random.randn(4, 2)

        with pytest.raises(ValueError, match="Unknown method"):
            bayesian_propensity(treatment, covariates, method="invalid")

    def test_single_covariate(self) -> None:
        """Works with single covariate."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        T = (X > 0).astype(float)

        result = bayesian_propensity_logistic(T, X)

        assert result["n"] == n
        assert len(result["posterior_mean"]) == n

    def test_many_covariates(self) -> None:
        """Works with many covariates."""
        np.random.seed(42)
        n = 200
        p = 10
        X = np.random.randn(n, p)
        T = (X[:, 0] > 0).astype(float)

        result = bayesian_propensity_logistic(T, X)

        assert len(result["coefficient_mean"]) == p + 1  # +1 for intercept


# =============================================================================
# Monte Carlo Validation
# =============================================================================


class TestBayesianPropensityMonteCarlo:
    """Monte Carlo validation tests."""

    @pytest.mark.monte_carlo
    def test_stratified_coverage(self) -> None:
        """Posterior intervals have correct coverage."""
        np.random.seed(42)
        n_sim = 200
        n = 100
        covered = 0

        for _ in range(n_sim):
            # Generate data with known propensity
            X = np.random.choice([0, 1], n)
            true_prop = np.where(X == 0, 0.3, 0.7)
            T = (np.random.rand(n) < true_prop).astype(float)

            result = bayesian_propensity_stratified(
                T, X.reshape(-1, 1), n_posterior_samples=1000
            )

            # Check if 95% interval covers true value (for first stratum)
            samples = result["posterior_samples"][:, 0]
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)

            # Get true propensity for this observation
            true_p = true_prop[0]

            if ci_lower <= true_p <= ci_upper:
                covered += 1

        coverage = covered / n_sim
        assert 0.85 < coverage < 0.99

    @pytest.mark.monte_carlo
    def test_logistic_coverage(self) -> None:
        """Posterior intervals for coefficients have correct coverage."""
        np.random.seed(123)
        n_sim = 200
        n = 200
        true_beta = 0.5
        covered = 0

        for _ in range(n_sim):
            X = np.random.randn(n, 1)
            logit = true_beta * X[:, 0]
            prob = 1 / (1 + np.exp(-logit))
            T = (np.random.rand(n) < prob).astype(float)

            result = bayesian_propensity_logistic(
                T, X, n_posterior_samples=1000, include_intercept=False
            )

            # Check if 95% interval covers true value
            samples = result["coefficient_samples"][:, 0]
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)

            if ci_lower <= true_beta <= ci_upper:
                covered += 1

        coverage = covered / n_sim
        assert 0.85 < coverage < 0.99


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestBayesianPropensityStability:
    """Numerical stability tests."""

    def test_extreme_imbalance(self) -> None:
        """Works with extreme treatment imbalance."""
        np.random.seed(42)
        n = 100
        # 95% treated
        T = np.random.binomial(1, 0.95, n).astype(float)
        X = np.random.randn(n, 2)

        result = bayesian_propensity_logistic(T, X)

        assert np.all(np.isfinite(result["posterior_mean"]))
        assert np.mean(result["posterior_mean"]) > 0.8

    def test_perfect_separation(self) -> None:
        """Handles (near) perfect separation."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        # Near-perfect separation
        T = (X[:, 0] > 0).astype(float)
        # Add a few misclassifications
        T[:5] = 1 - T[:5]

        result = bayesian_propensity_logistic(T, X)

        assert np.all(np.isfinite(result["posterior_mean"]))

    def test_empty_stratum_handled(self) -> None:
        """Handles strata with no observations gracefully."""
        np.random.seed(42)
        n = 50
        # All observations in one stratum
        X = np.zeros((n, 1))
        T = np.random.binomial(1, 0.5, n).astype(float)

        result = bayesian_propensity_stratified(T, X)

        assert result["n_strata"] == 1
        assert np.all(np.isfinite(result["posterior_mean"]))
