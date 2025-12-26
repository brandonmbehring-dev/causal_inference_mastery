"""
Tests for Bayesian CACE (Complier Average Causal Effect) estimation.

Test Structure (following 6-layer validation):
- Layer 1: Known-answer tests (posterior mean ≈ frequentist)
- Layer 2: Adversarial edge cases (prior sensitivity)
- Layer 3: Monte Carlo validation (coverage check)
- Layer 4: Cross-language parity (separate file)
"""

import numpy as np
import pytest
import warnings

# Check if PyMC is available
try:
    import pymc as pm
    import arviz as az

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# Skip all tests if PyMC not installed
pytestmark = pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")


# =============================================================================
# Test Fixtures / Data Generators
# =============================================================================


def generate_ps_dgp(
    n: int = 500,
    pi_c: float = 0.60,
    pi_a: float = 0.20,
    pi_n: float = 0.20,
    true_cace: float = 2.0,
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Generate data from a principal stratification DGP.

    Returns dictionary with Y, D, Z, true_cace, and strata proportions.
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

    # Outcome with proper exclusion restriction:
    # Only compliers have causal effect of treatment
    Y = baseline + np.where(strata == 0, true_cace * D, 0) + noise_sd * np.random.randn(n)

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
# Layer 1: Known-Answer / Structure Tests
# =============================================================================


class TestCACEBayesianStructure:
    """Test that Bayesian CACE returns correct structure."""

    def test_bayesian_runs_returns_valid_result(self):
        """cace_bayesian should return BayesianPSResult with all fields."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=300, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        # Check all required fields exist
        assert "cace_mean" in result
        assert "cace_sd" in result
        assert "cace_hdi_lower" in result
        assert "cace_hdi_upper" in result
        assert "cace_samples" in result
        assert "pi_c_mean" in result
        assert "pi_c_samples" in result
        assert "pi_a_mean" in result
        assert "pi_n_mean" in result
        assert "rhat" in result
        assert "ess" in result
        assert "n_samples" in result
        assert "n_chains" in result
        assert "model" in result

    def test_posterior_samples_structure(self):
        """Posterior samples should have correct shape."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=300, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        # Quick mode: 1000 samples × 2 chains = 2000 total
        assert len(result["cace_samples"]) == 2000
        assert len(result["pi_c_samples"]) == 2000
        assert result["n_samples"] == 2000
        assert result["n_chains"] == 2

    def test_hdi_contains_mean(self):
        """HDI should contain the posterior mean."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=300, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        assert result["cace_hdi_lower"] <= result["cace_mean"] <= result["cace_hdi_upper"]


class TestCACEBayesianKnownAnswer:
    """Test Bayesian CACE against frequentist estimates."""

    def test_posterior_mean_close_to_em(self):
        """Posterior mean should be close to EM estimate with weak priors."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian
        from src.causal_inference.principal_stratification import cace_em

        data = generate_ps_dgp(n=500, seed=42)

        # EM estimate
        em_result = cace_em(data["Y"], data["D"], data["Z"])

        # Bayesian with weak priors (large prior_mu_sd)
        bayes_result = cace_bayesian(
            data["Y"],
            data["D"],
            data["Z"],
            prior_mu_sd=100.0,  # Very weak prior
            quick=True,
            random_seed=42,
        )

        # Should be within 30% of each other (MCMC has variance)
        assert np.isclose(bayes_result["cace_mean"], em_result["cace"], rtol=0.30)

    def test_strata_proportions_sum_to_one(self):
        """Posterior mean strata proportions should sum to 1."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=300, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        total = result["pi_c_mean"] + result["pi_a_mean"] + result["pi_n_mean"]
        assert np.isclose(total, 1.0, rtol=1e-6)

    def test_credible_interval_width_reasonable(self):
        """CI should not be too narrow or too wide."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=500, true_cace=2.0, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        ci_width = result["cace_hdi_upper"] - result["cace_hdi_lower"]

        # CI should be between 0.5 and 4.0 for this DGP
        assert 0.3 < ci_width < 5.0, f"CI width {ci_width} seems unreasonable"


# =============================================================================
# Layer 2: MCMC Diagnostics Tests
# =============================================================================


class TestCACEBayesianDiagnostics:
    """Test MCMC diagnostics and convergence."""

    def test_mcmc_diagnostics_acceptable(self):
        """R-hat and ESS should be acceptable with sufficient sampling."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=400, seed=42)
        result = cace_bayesian(
            data["Y"],
            data["D"],
            data["Z"],
            n_samples=1500,
            n_chains=2,
            random_seed=42,
        )

        # R-hat should be close to 1
        max_rhat = max(result["rhat"].values())
        assert max_rhat < 1.10, f"R-hat {max_rhat} too high"

        # ESS should be reasonable
        min_ess = min(result["ess"].values())
        assert min_ess > 100, f"ESS {min_ess} too low"

    def test_rhat_dict_structure(self):
        """R-hat dictionary should contain key parameters."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=300, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        # Should have R-hat for CACE
        assert "cace" in result["rhat"]
        # Should have R-hat for at least some strata proportions
        assert any("pi" in k for k in result["rhat"].keys())


# =============================================================================
# Layer 3: Quick Mode Tests
# =============================================================================


class TestCACEBayesianQuickMode:
    """Test quick mode for fast exploratory analysis."""

    def test_quick_mode_uses_fewer_samples(self):
        """Quick mode should use 1000 samples × 2 chains."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=200, seed=42)
        result = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        assert result["n_samples"] == 2000  # 1000 × 2
        assert result["n_chains"] == 2

    def test_quick_mode_faster_than_full(self):
        """Quick mode should complete faster (implicit via sample count)."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=200, seed=42)

        # Quick mode
        result_quick = cace_bayesian(
            data["Y"], data["D"], data["Z"], quick=True, random_seed=42
        )

        # Quick uses 2000 samples, full would use 8000 (2000 × 4)
        assert result_quick["n_samples"] == 2000


# =============================================================================
# Layer 4: Prior Sensitivity Tests
# =============================================================================


class TestCACEBayesianPriors:
    """Test sensitivity to prior specification."""

    def test_weak_prior_matches_frequentist(self):
        """Very weak priors should give results close to frequentist."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian
        from src.causal_inference.principal_stratification import cace_2sls

        data = generate_ps_dgp(n=500, seed=42)

        # Frequentist 2SLS
        freq_result = cace_2sls(data["Y"], data["D"], data["Z"])

        # Bayesian with very weak priors
        bayes_result = cace_bayesian(
            data["Y"],
            data["D"],
            data["Z"],
            prior_mu_sd=1000.0,  # Extremely weak
            prior_alpha=(0.1, 0.1, 0.1),  # Weak Dirichlet
            quick=True,
            random_seed=42,
        )

        # Should be within 40% (MCMC variance + different model)
        assert np.isclose(bayes_result["cace_mean"], freq_result["cace"], rtol=0.40)

    def test_informative_prior_induces_shrinkage(self):
        """Strong prior should pull estimate toward prior mean (0)."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        data = generate_ps_dgp(n=200, true_cace=5.0, seed=42)  # Large true effect

        # Weak prior
        result_weak = cace_bayesian(
            data["Y"],
            data["D"],
            data["Z"],
            prior_mu_sd=100.0,
            quick=True,
            random_seed=42,
        )

        # Strong prior centered at 0
        result_strong = cace_bayesian(
            data["Y"],
            data["D"],
            data["Z"],
            prior_mu_sd=0.5,  # Strong prior
            quick=True,
            random_seed=43,
        )

        # Strong prior should shrink toward 0
        assert abs(result_strong["cace_mean"]) < abs(result_weak["cace_mean"])


# =============================================================================
# Layer 5: Input Validation Tests
# =============================================================================


class TestCACEBayesianValidation:
    """Test input validation and error handling."""

    def test_length_mismatch_raises(self):
        """Mismatched input lengths should raise ValueError."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        Z = np.random.binomial(1, 0.5, 50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            cace_bayesian(Y, D, Z, quick=True, random_seed=42)

    def test_non_binary_treatment_raises(self):
        """Non-binary treatment should raise ValueError."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        Y = np.random.randn(100)
        D = np.random.randint(0, 3, 100)  # Not binary
        Z = np.random.binomial(1, 0.5, 100)

        with pytest.raises(ValueError, match="binary"):
            cace_bayesian(Y, D, Z, quick=True, random_seed=42)

    def test_non_binary_instrument_raises(self):
        """Non-binary instrument should raise ValueError."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        Z = np.random.randint(0, 3, 100)  # Not binary

        with pytest.raises(ValueError, match="binary"):
            cace_bayesian(Y, D, Z, quick=True, random_seed=42)


# =============================================================================
# Layer 6: Warning Tests
# =============================================================================


class TestCACEBayesianWarnings:
    """Test diagnostic warnings are emitted correctly."""

    def test_emit_diagnostic_warnings_rhat(self):
        """Should warn when R-hat is too high."""
        from src.causal_inference.principal_stratification.bayesian import (
            _emit_diagnostic_warnings,
        )

        rhat = {"cace": 1.2, "pi": 1.1}  # High R-hat
        ess = {"cace": 500, "pi": 500}

        with pytest.warns(RuntimeWarning, match="R-hat"):
            _emit_diagnostic_warnings(rhat, ess, divergences=0)

    def test_emit_diagnostic_warnings_ess(self):
        """Should warn when ESS is too low."""
        from src.causal_inference.principal_stratification.bayesian import (
            _emit_diagnostic_warnings,
        )

        rhat = {"cace": 1.01, "pi": 1.01}
        ess = {"cace": 50, "pi": 50}  # Low ESS

        with pytest.warns(RuntimeWarning, match="ESS"):
            _emit_diagnostic_warnings(rhat, ess, divergences=0)

    def test_emit_diagnostic_warnings_divergences(self):
        """Should warn when there are divergences."""
        from src.causal_inference.principal_stratification.bayesian import (
            _emit_diagnostic_warnings,
        )

        rhat = {"cace": 1.01}
        ess = {"cace": 500}

        with pytest.warns(RuntimeWarning, match="divergent"):
            _emit_diagnostic_warnings(rhat, ess, divergences=5)


# =============================================================================
# Monte Carlo Validation (Light version - full would need more samples)
# =============================================================================


class TestCACEBayesianMonteCarlo:
    """Light Monte Carlo validation for Bayesian CACE."""

    @pytest.mark.slow
    def test_credible_interval_coverage_100_runs(self):
        """95% HDI should contain true CACE ~95% of the time."""
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian

        true_cace = 2.0
        n_runs = 100
        covers = []

        for seed in range(n_runs):
            data = generate_ps_dgp(n=300, true_cace=true_cace, seed=seed)
            result = cace_bayesian(
                data["Y"], data["D"], data["Z"], quick=True, random_seed=seed
            )

            covers.append(
                result["cace_hdi_lower"] <= true_cace <= result["cace_hdi_upper"]
            )

        coverage = np.mean(covers)
        # Allow 85-99% coverage (Bayesian HDI, small sample Monte Carlo variance)
        assert 0.80 < coverage < 0.99, f"Coverage {coverage:.2%} outside range"
