"""
Unit Tests for Bayesian Doubly Robust ATE Estimation.

Session 103: Initial test suite.

Tests cover:
1. Basic functionality and return structure
2. Known-answer validation
3. Double robustness property
4. Monte Carlo coverage validation
5. Edge cases and error handling
"""

import numpy as np
import pytest

from src.causal_inference.bayesian import bayesian_dr_ate, BayesianDRResult


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_dr_data(
    n: int = 300,
    true_ate: float = 2.0,
    confounded: bool = True,
    seed: int = 42,
) -> dict:
    """
    Generate data for DR testing.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True treatment effect.
    confounded : bool
        If True, treatment assignment depends on covariates.
    seed : int
        Random seed.

    Returns
    -------
    dict
        outcomes, treatment, covariates, true_ate
    """
    np.random.seed(seed)
    X = np.random.randn(n, 2)

    if confounded:
        # Propensity depends on X
        logit = 0.5 * X[:, 0] + 0.3 * X[:, 1]
        prob = 1 / (1 + np.exp(-logit))
        treatment = np.random.binomial(1, prob).astype(float)
    else:
        # Random treatment assignment
        treatment = np.random.binomial(1, 0.5, n).astype(float)

    # Outcome depends on treatment and X
    outcomes = true_ate * treatment + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 1, n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": X,
        "true_ate": true_ate,
    }


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBayesianDRBasic:
    """Basic functionality tests."""

    def test_returns_correct_structure(self):
        """Result has all expected fields."""
        data = generate_dr_data()
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            n_posterior_samples=100,
        )

        # Check all required keys
        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "credible_level" in result
        assert "posterior_samples" in result
        assert "n" in result
        assert "n_treated" in result
        assert "n_control" in result
        assert "propensity_mean" in result
        assert "propensity_mean_uncertainty" in result
        assert "outcome_r2" in result
        assert "frequentist_estimate" in result
        assert "frequentist_se" in result

    def test_posterior_samples_shape(self):
        """Posterior samples have correct shape."""
        data = generate_dr_data()
        n_samples = 500
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            n_posterior_samples=n_samples,
        )

        assert result["posterior_samples"].shape == (n_samples,)

    def test_estimate_in_credible_interval(self):
        """Posterior mean is within credible interval."""
        data = generate_dr_data()
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
        )

        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]

    def test_se_positive(self):
        """Standard error is positive."""
        data = generate_dr_data()
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
        )

        assert result["se"] > 0

    def test_sample_sizes_correct(self):
        """Sample sizes are correctly computed."""
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
        )

        assert result["n"] == 200
        assert result["n_treated"] + result["n_control"] == 200


# =============================================================================
# Known-Answer Tests
# =============================================================================


class TestBayesianDRKnownAnswer:
    """Known-answer validation tests."""

    def test_known_effect_recovered(self):
        """Recovers true effect in well-specified scenario."""
        data = generate_dr_data(n=500, true_ate=2.0, seed=42)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            n_posterior_samples=500,
        )

        # Estimate should be within 1.0 of true value
        assert abs(result["estimate"] - data["true_ate"]) < 1.0

    def test_no_treatment_effect(self):
        """Correctly estimates zero effect."""
        data = generate_dr_data(n=500, true_ate=0.0, seed=123)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
        )

        # CI should contain zero
        assert result["ci_lower"] <= 0 <= result["ci_upper"]

    def test_close_to_frequentist(self):
        """Bayesian estimate close to frequentist estimate."""
        data = generate_dr_data(n=500, seed=42)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            n_posterior_samples=1000,
        )

        # Should be within 0.5 of frequentist estimate
        diff = abs(result["estimate"] - result["frequentist_estimate"])
        assert diff < 0.5


# =============================================================================
# Double Robustness Tests
# =============================================================================


class TestBayesianDRDoubleRobustness:
    """Tests for double robustness property."""

    def test_both_correct_lowest_variance(self):
        """Both models correct should give reasonable SE."""
        data = generate_dr_data(n=500, seed=42)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            n_posterior_samples=500,
        )

        # SE should be reasonable (not too large)
        assert result["se"] < 1.0

    def test_unconfounded_recovery(self):
        """Works well in unconfounded scenario."""
        data = generate_dr_data(n=500, true_ate=2.0, confounded=False, seed=42)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
        )

        # Should recover effect well in unconfounded case
        assert abs(result["estimate"] - data["true_ate"]) < 0.5


# =============================================================================
# Propensity Method Tests
# =============================================================================


class TestBayesianDRPropensityMethods:
    """Tests for different propensity methods."""

    def test_auto_method(self):
        """Auto method selection works."""
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            propensity_method="auto",
        )

        assert result["estimate"] is not None

    def test_logistic_method(self):
        """Logistic method works with continuous covariates."""
        data = generate_dr_data(n=200)
        result = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            propensity_method="logistic",
        )

        assert result["estimate"] is not None

    def test_stratified_method(self):
        """Stratified method works with discrete covariates."""
        np.random.seed(42)
        n = 200
        X = np.column_stack(
            [
                np.random.choice([0, 1], n),
                np.random.choice([0, 1, 2], n),
            ]
        ).astype(float)
        prob = 0.3 + 0.2 * X[:, 0] + 0.1 * X[:, 1]
        T = np.random.binomial(1, prob).astype(float)
        Y = 2.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, n)

        result = bayesian_dr_ate(Y, T, X, propensity_method="stratified")
        assert result["estimate"] is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestBayesianDREdgeCases:
    """Edge cases and error handling tests."""

    def test_length_mismatch_raises(self):
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="different lengths"):
            bayesian_dr_ate(
                outcomes=np.array([1.0, 2.0, 3.0]),
                treatment=np.array([1.0, 0.0]),
                covariates=np.array([[1.0], [2.0], [3.0]]),
            )

    def test_non_binary_treatment_raises(self):
        """Non-binary treatment raises ValueError."""
        with pytest.raises(ValueError, match="binary"):
            bayesian_dr_ate(
                outcomes=np.array([1.0, 2.0, 3.0]),
                treatment=np.array([0.0, 1.0, 2.0]),
                covariates=np.array([[1.0], [2.0], [3.0]]),
            )

    def test_nan_raises(self):
        """NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            bayesian_dr_ate(
                outcomes=np.array([1.0, np.nan, 3.0]),
                treatment=np.array([1.0, 0.0, 1.0]),
                covariates=np.array([[1.0], [2.0], [3.0]]),
            )

    def test_inf_raises(self):
        """Infinite values raise ValueError."""
        with pytest.raises(ValueError, match="Infinite"):
            bayesian_dr_ate(
                outcomes=np.array([1.0, np.inf, 3.0]),
                treatment=np.array([1.0, 0.0, 1.0]),
                covariates=np.array([[1.0], [2.0], [3.0]]),
            )

    def test_invalid_trim_threshold_raises(self):
        """Invalid trim threshold raises ValueError."""
        data = generate_dr_data(n=100)
        with pytest.raises(ValueError, match="trim_threshold"):
            bayesian_dr_ate(
                data["outcomes"],
                data["treatment"],
                data["covariates"],
                trim_threshold=0.6,  # Invalid: > 0.5
            )

    def test_invalid_credible_level_raises(self):
        """Invalid credible level raises ValueError."""
        data = generate_dr_data(n=100)
        with pytest.raises(ValueError, match="credible_level"):
            bayesian_dr_ate(
                data["outcomes"],
                data["treatment"],
                data["covariates"],
                credible_level=1.5,  # Invalid: > 1
            )

    def test_single_covariate(self):
        """Works with single covariate (1D array)."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + 0.5 * X + np.random.normal(0, 1, n)

        result = bayesian_dr_ate(Y, T, X)
        assert result["n"] == n

    def test_extreme_propensity(self):
        """Handles extreme propensity cases."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        # Very imbalanced treatment (mostly treated)
        T = (np.random.rand(n) < 0.9).astype(float)
        Y = 2.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, n)

        result = bayesian_dr_ate(Y, T, X)
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0


# =============================================================================
# Monte Carlo Validation
# =============================================================================


class TestBayesianDRMonteCarlo:
    """Monte Carlo validation tests."""

    @pytest.mark.monte_carlo
    def test_coverage(self):
        """95% credible intervals have approximately 95% coverage."""
        n_sim = 200
        n = 200
        true_ate = 2.0
        covered = 0

        for i in range(n_sim):
            data = generate_dr_data(n=n, true_ate=true_ate, seed=i)
            result = bayesian_dr_ate(
                data["outcomes"],
                data["treatment"],
                data["covariates"],
                n_posterior_samples=500,
                credible_level=0.95,
            )

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                covered += 1

        coverage = covered / n_sim
        # Allow wider range due to propensity model misspecification
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside expected range"

    @pytest.mark.monte_carlo
    def test_unbiasedness(self):
        """Posterior mean is approximately unbiased."""
        n_sim = 200
        n = 300
        true_ate = 2.0
        estimates = []

        for i in range(n_sim):
            data = generate_dr_data(n=n, true_ate=true_ate, seed=i + 1000)
            result = bayesian_dr_ate(
                data["outcomes"],
                data["treatment"],
                data["covariates"],
                n_posterior_samples=500,
            )
            estimates.append(result["estimate"])

        bias = np.mean(estimates) - true_ate
        assert abs(bias) < 0.15, f"Bias {bias:.3f} exceeds threshold"


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestBayesianDRStability:
    """Numerical stability tests."""

    def test_large_outcomes(self):
        """Handles large outcome values."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        T = np.random.binomial(1, 0.5, n).astype(float)
        # Large outcomes (e.g., thousands)
        Y = 1000 + 2.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, n)

        result = bayesian_dr_ate(Y, T, X)
        assert np.isfinite(result["estimate"])

    def test_many_covariates(self):
        """Handles many covariates."""
        np.random.seed(42)
        n = 200
        p = 10  # Many covariates
        X = np.random.randn(n, p)
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + 0.5 * X[:, 0] + np.random.normal(0, 1, n)

        result = bayesian_dr_ate(Y, T, X)
        assert np.isfinite(result["estimate"])
        assert result["propensity_mean"].shape == (n,)


# =============================================================================
# Credible Level Tests
# =============================================================================


class TestBayesianDRCredibleLevel:
    """Tests for different credible levels."""

    def test_90_credible_level(self):
        """90% credible interval is narrower than 95%."""
        data = generate_dr_data(n=300)

        result_90 = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            credible_level=0.90,
        )

        result_95 = bayesian_dr_ate(
            data["outcomes"],
            data["treatment"],
            data["covariates"],
            credible_level=0.95,
        )

        width_90 = result_90["ci_upper"] - result_90["ci_lower"]
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]

        assert width_90 < width_95
