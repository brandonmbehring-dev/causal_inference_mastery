"""
Unit Tests for Hierarchical Bayesian ATE Estimation.

Session 104: Initial test suite.

Tests cover:
1. Basic functionality and return structure
2. Known-answer validation
3. Partial pooling behavior
4. MCMC diagnostics
5. Edge cases and error handling
"""

import numpy as np
import pytest

# Skip all tests if PyMC is not installed
pytest.importorskip("pymc", reason="PyMC required for hierarchical tests")

from src.causal_inference.bayesian.hierarchical_ate import hierarchical_bayesian_ate


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_hierarchical_data(
    n_groups: int = 5,
    n_per_group: int = 50,
    population_ate: float = 2.0,
    tau: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Generate hierarchical data with known population ATE and heterogeneity.

    Parameters
    ----------
    n_groups : int
        Number of groups.
    n_per_group : int
        Observations per group.
    population_ate : float
        True population-level ATE.
    tau : float
        True between-group standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        outcomes, treatment, groups, true values
    """
    np.random.seed(seed)
    n = n_groups * n_per_group

    # Generate group assignments
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Generate group-specific effects
    true_group_effects = np.random.normal(population_ate, tau, n_groups)

    # Generate treatment (randomized within groups)
    treatment = np.random.binomial(1, 0.5, n).astype(float)

    # Generate outcomes
    group_effects = true_group_effects[groups]
    outcomes = group_effects * treatment + np.random.normal(0, 1, n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "groups": groups,
        "true_population_ate": population_ate,
        "true_tau": tau,
        "true_group_effects": true_group_effects,
        "n_groups": n_groups,
    }


def generate_homogeneous_data(
    n_groups: int = 5,
    n_per_group: int = 50,
    population_ate: float = 2.0,
    seed: int = 42,
) -> dict:
    """Generate data with no between-group heterogeneity (tau=0)."""
    np.random.seed(seed)
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    treatment = np.random.binomial(1, 0.5, n).astype(float)
    outcomes = population_ate * treatment + np.random.normal(0, 1, n)
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "groups": groups,
        "true_population_ate": population_ate,
        "true_tau": 0.0,
    }


def generate_heterogeneous_data(
    n_groups: int = 5,
    n_per_group: int = 50,
    seed: int = 42,
) -> dict:
    """Generate data with large between-group heterogeneity."""
    np.random.seed(seed)
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Large spread of group effects
    true_group_effects = np.linspace(-1, 5, n_groups)  # Range from -1 to 5
    treatment = np.random.binomial(1, 0.5, n).astype(float)
    group_effects = true_group_effects[groups]
    outcomes = group_effects * treatment + np.random.normal(0, 1, n)

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "groups": groups,
        "true_group_effects": true_group_effects,
        "true_population_ate": np.mean(true_group_effects),
    }


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestHierarchicalATEBasic:
    """Basic functionality tests."""

    @pytest.mark.slow
    def test_returns_correct_structure(self):
        """Test that result contains all expected fields."""
        data = generate_hierarchical_data(n_groups=3, n_per_group=30, seed=1)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=200,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        # Check all required fields
        assert "population_ate" in result
        assert "population_ate_se" in result
        assert "population_ate_ci_lower" in result
        assert "population_ate_ci_upper" in result
        assert "group_ates" in result
        assert "group_ate_ses" in result
        assert "group_ids" in result
        assert "tau" in result
        assert "tau_ci_lower" in result
        assert "tau_ci_upper" in result
        assert "posterior_samples" in result
        assert "n_groups" in result
        assert "n_obs" in result
        assert "credible_level" in result
        assert "rhat_max" in result
        assert "ess_min" in result
        assert "divergences" in result

    @pytest.mark.slow
    def test_posterior_samples_shape(self):
        """Test posterior samples have correct dimensions."""
        n_groups = 4
        n_samples = 300
        data = generate_hierarchical_data(n_groups=n_groups, n_per_group=25, seed=2)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=n_samples,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        # Check shapes
        total_samples = n_samples * 2  # 2 chains
        assert result["posterior_samples"]["mu"].shape == (total_samples,)
        assert result["posterior_samples"]["tau"].shape == (total_samples,)
        assert result["posterior_samples"]["theta"].shape == (total_samples, n_groups)

    @pytest.mark.slow
    def test_group_ates_match_groups(self):
        """Test that group_ates has one entry per group."""
        n_groups = 5
        data = generate_hierarchical_data(n_groups=n_groups, n_per_group=20, seed=3)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=200,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        assert len(result["group_ates"]) == n_groups
        assert len(result["group_ate_ses"]) == n_groups
        assert len(result["group_ids"]) == n_groups
        assert result["n_groups"] == n_groups

    @pytest.mark.slow
    def test_estimate_in_credible_interval(self):
        """Test that population ATE estimate is within its credible interval."""
        data = generate_hierarchical_data(seed=4)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=200,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        assert result["population_ate_ci_lower"] <= result["population_ate"]
        assert result["population_ate"] <= result["population_ate_ci_upper"]

    @pytest.mark.slow
    def test_se_positive(self):
        """Test that standard errors are positive."""
        data = generate_hierarchical_data(seed=5)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=200,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        assert result["population_ate_se"] > 0
        assert np.all(result["group_ate_ses"] > 0)


# =============================================================================
# Known-Answer Tests
# =============================================================================


class TestHierarchicalATEKnownAnswer:
    """Known-answer validation tests."""

    @pytest.mark.slow
    def test_population_ate_recovered(self):
        """Test that population ATE is close to true value."""
        data = generate_hierarchical_data(
            n_groups=5,
            n_per_group=100,
            population_ate=2.0,
            tau=0.3,
            seed=42,
        )
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=2,
            n_tune=200,
            progressbar=False,
        )

        # Should be within 0.5 of true value
        assert abs(result["population_ate"] - data["true_population_ate"]) < 0.5

    @pytest.mark.slow
    def test_homogeneous_groups_low_tau(self):
        """Test that tau is small when groups are homogeneous."""
        data = generate_homogeneous_data(n_groups=5, n_per_group=80, seed=43)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=2,
            n_tune=200,
            progressbar=False,
        )

        # Tau should be close to 0 (allow up to 0.5 due to noise)
        assert result["tau"] < 0.5

    @pytest.mark.slow
    def test_heterogeneous_groups_high_tau(self):
        """Test that tau captures heterogeneity."""
        data = generate_heterogeneous_data(n_groups=5, n_per_group=80, seed=44)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=2,
            n_tune=200,
            progressbar=False,
        )

        # Tau should be substantial (true effects range from -1 to 5)
        assert result["tau"] > 0.5


# =============================================================================
# Partial Pooling Tests
# =============================================================================


class TestPartialPooling:
    """Tests for partial pooling behavior."""

    @pytest.mark.slow
    def test_small_groups_shrink_more(self):
        """Test that small groups shrink toward population mean."""
        np.random.seed(45)

        # Create unbalanced groups
        n_per_group = [20, 20, 200, 200, 200]  # Groups 0,1 are small
        n_groups = len(n_per_group)
        n = sum(n_per_group)

        groups = np.concatenate([np.full(n, i) for i, n in enumerate(n_per_group)])
        treatment = np.random.binomial(1, 0.5, n).astype(float)

        # True effects: make small groups have extreme values
        true_effects = [5.0, 5.0, 2.0, 2.0, 2.0]  # Small groups have effect=5
        population_mean = np.average(true_effects, weights=n_per_group)

        outcomes = np.zeros(n)
        idx = 0
        for i, n_g in enumerate(n_per_group):
            outcomes[idx : idx + n_g] = true_effects[i] * treatment[
                idx : idx + n_g
            ] + np.random.randn(n_g)
            idx += n_g

        result = hierarchical_bayesian_ate(
            outcomes,
            treatment,
            groups,
            n_samples=500,
            n_chains=2,
            n_tune=200,
            progressbar=False,
        )

        # Small groups (0, 1) should shrink toward population mean
        # Their estimates should be between true effect (5.0) and population mean
        small_group_estimates = result["group_ates"][:2]
        large_group_estimates = result["group_ates"][2:]

        # Small groups should be pulled toward population mean (closer to 2.0)
        # while still being above the large group estimates
        avg_small = np.mean(small_group_estimates)
        avg_large = np.mean(large_group_estimates)

        # The small groups' estimates should be less extreme than 5.0
        assert avg_small < 4.5  # Shrunk from true 5.0

        # But still reflect some difference
        assert avg_small > avg_large  # Still higher than large groups


# =============================================================================
# MCMC Diagnostics Tests
# =============================================================================


class TestMCMCDiagnostics:
    """Tests for MCMC convergence diagnostics."""

    @pytest.mark.slow
    def test_rhat_acceptable(self):
        """Test that R-hat indicates convergence."""
        data = generate_hierarchical_data(n_groups=4, n_per_group=50, seed=46)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=4,
            n_tune=300,
            target_accept=0.95,
            progressbar=False,
        )

        # R-hat should be < 1.1 for convergence
        assert result["rhat_max"] < 1.1

    @pytest.mark.slow
    def test_ess_adequate(self):
        """Test that effective sample size is adequate."""
        data = generate_hierarchical_data(n_groups=4, n_per_group=50, seed=47)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=4,
            n_tune=300,
            progressbar=False,
        )

        # ESS should be > 100 at minimum
        assert result["ess_min"] > 100

    @pytest.mark.slow
    def test_no_divergences(self):
        """Test that sampling has no divergent transitions."""
        data = generate_hierarchical_data(n_groups=4, n_per_group=50, seed=48)
        result = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=500,
            n_chains=2,
            n_tune=300,
            target_accept=0.95,
            progressbar=False,
        )

        # Should have no divergences with high target_accept
        assert result["divergences"] == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_length_mismatch_outcomes_treatment(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 1.0]),  # Wrong length
                np.array([0, 0, 1]),
            )

    def test_length_mismatch_outcomes_groups(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0, 0]),  # Wrong length
            )

    def test_non_binary_treatment(self):
        """Test that non-binary treatment raises ValueError."""
        with pytest.raises(ValueError, match="binary"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.5, 1.0]),  # Not binary
                np.array([0, 0, 1]),
            )

    def test_single_group_raises(self):
        """Test that single group raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 groups"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0, 0, 0]),  # Only one group
            )

    def test_invalid_credible_level(self):
        """Test that invalid credible level raises ValueError."""
        with pytest.raises(ValueError, match="credible_level"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 0.0, 1.0]),
                np.array([0, 0, 1, 1]),
                credible_level=1.5,
            )

    def test_invalid_n_samples(self):
        """Test that invalid n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples"):
            hierarchical_bayesian_ate(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 0.0, 1.0]),
                np.array([0, 0, 1, 1]),
                n_samples=50,  # Too few
            )

    @pytest.mark.slow
    def test_string_group_ids(self):
        """Test that string group identifiers work."""
        np.random.seed(49)
        n = 100
        groups = np.array(["A", "B"] * 50)
        treatment = np.random.binomial(1, 0.5, n).astype(float)
        outcomes = 2.0 * treatment + np.random.randn(n)

        result = hierarchical_bayesian_ate(
            outcomes,
            treatment,
            groups,
            n_samples=200,
            n_chains=2,
            n_tune=100,
            progressbar=False,
        )

        assert result["n_groups"] == 2
        assert "A" in result["group_ids"]
        assert "B" in result["group_ids"]


# =============================================================================
# Credible Level Tests
# =============================================================================


class TestCredibleLevel:
    """Tests for credible interval behavior."""

    @pytest.mark.slow
    def test_90_narrower_than_95(self):
        """Test that 90% CI is narrower than 95% CI."""
        data = generate_hierarchical_data(n_groups=4, n_per_group=40, seed=50)

        result_90 = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=300,
            n_chains=2,
            n_tune=100,
            credible_level=0.90,
            random_seed=50,
            progressbar=False,
        )

        result_95 = hierarchical_bayesian_ate(
            data["outcomes"],
            data["treatment"],
            data["groups"],
            n_samples=300,
            n_chains=2,
            n_tune=100,
            credible_level=0.95,
            random_seed=50,
            progressbar=False,
        )

        width_90 = result_90["population_ate_ci_upper"] - result_90["population_ate_ci_lower"]
        width_95 = result_95["population_ate_ci_upper"] - result_95["population_ate_ci_lower"]

        assert width_90 < width_95


# =============================================================================
# Monte Carlo Validation (Slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestMonteCarloValidation:
    """Monte Carlo validation tests."""

    def test_coverage(self):
        """Test that credible intervals achieve nominal coverage."""
        n_sim = 30  # Reduced for speed
        true_ate = 2.0
        covered = 0

        for i in range(n_sim):
            data = generate_hierarchical_data(
                n_groups=4,
                n_per_group=50,
                population_ate=true_ate,
                tau=0.3,
                seed=1000 + i,
            )
            result = hierarchical_bayesian_ate(
                data["outcomes"],
                data["treatment"],
                data["groups"],
                n_samples=300,
                n_chains=2,
                n_tune=100,
                credible_level=0.95,
                progressbar=False,
            )

            if result["population_ate_ci_lower"] <= true_ate <= result["population_ate_ci_upper"]:
                covered += 1

        coverage = covered / n_sim
        # Allow 70-100% due to limited simulations
        assert 0.70 < coverage <= 1.0
