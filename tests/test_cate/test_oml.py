"""Tests for Orthogonal Machine Learning (OML) / Interactive Regression Model.

Tests are organized by layer:
1. Known-Answer: Verify IRM recovers known treatment effects
2. Adversarial: Edge cases and challenging scenarios
3. Monte Carlo: Statistical validation of bias, coverage, and double robustness
"""

import pytest
import numpy as np
from scipy import stats

from src.causal_inference.cate import irm_dml
from tests.test_cate.conftest import generate_cate_dgp


def generate_irm_dgp(
    n: int = 500,
    p: int = 2,
    true_ate: float = 2.0,
    confounding_strength: float = 0.5,
    noise_sd: float = 1.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for IRM testing.

    IRM DGP:
    - X ~ N(0, I_p)
    - e(X) = 1 / (1 + exp(-confounding_strength * X₁)) (propensity)
    - T ~ Bernoulli(e(X))
    - g0(X) = 1 + X₁ (control outcome)
    - g1(X) = 1 + X₁ + true_ate (treated outcome)
    - Y = T * g1(X) + (1-T) * g0(X) + ε

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of covariates.
    true_ate : float
        True average treatment effect.
    confounding_strength : float
        Strength of confounding (0 = RCT, higher = more confounding).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple
        (outcomes, treatment, covariates)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, p)

    # Propensity with confounding
    propensity = 1 / (1 + np.exp(-confounding_strength * X[:, 0]))
    T = np.random.binomial(1, propensity, n).astype(float)

    # Potential outcomes (IRM structure)
    g0 = 1 + X[:, 0]  # E[Y|T=0, X]
    g1 = 1 + X[:, 0] + true_ate  # E[Y|T=1, X]

    # Observed outcome
    noise = np.random.randn(n) * noise_sd
    Y = T * g1 + (1 - T) * g0 + noise

    return Y, T, X


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestIRMKnownAnswer:
    """Tests with known true treatment effects."""

    def test_irm_constant_effect(self):
        """IRM recovers constant ATE within tolerance."""
        Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

        result = irm_dml(Y, T, X, n_folds=5)

        # ATE should be close to true value
        assert abs(result["ate"] - 2.0) < 0.5, (
            f"IRM ATE {result['ate']:.3f} far from true 2.0"
        )
        assert result["method"] == "irm_dml"
        assert result["target"] == "ate"
        assert result["score_type"] == "irm"

    def test_irm_heterogeneous_effect(self):
        """IRM captures CATE shape correctly."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        T = (np.random.rand(n) > 0.5).astype(float)
        # Heterogeneous effect: τ(X) = 2 + X₁
        true_cate = 2 + X[:, 0]
        Y = 1 + X[:, 0] + true_cate * T + np.random.randn(n)

        result = irm_dml(Y, T, X, n_folds=5)

        # CATE should correlate with true CATE
        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        assert correlation > 0.3, (
            f"CATE correlation {correlation:.3f} too low for heterogeneous effect"
        )

    def test_irm_returns_valid_ci(self):
        """IRM returns valid confidence intervals."""
        Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

        result = irm_dml(Y, T, X, n_folds=5, alpha=0.05)

        # CI should contain ATE point estimate
        assert result["ci_lower"] < result["ate"] < result["ci_upper"]

        # CI width should be reasonable
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 0.1 < ci_width < 5.0, f"CI width {ci_width:.3f} seems unreasonable"

    def test_irm_se_positive(self):
        """Standard error should be positive and finite."""
        Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

        result = irm_dml(Y, T, X)

        assert result["ate_se"] > 0, "SE must be positive"
        assert np.isfinite(result["ate_se"]), "SE must be finite"

    def test_irm_different_fold_counts(self):
        """IRM works with different numbers of folds."""
        Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)
        true_ate = 2.0

        for n_folds in [2, 3, 5, 10]:
            result = irm_dml(Y, T, X, n_folds=n_folds)
            assert abs(result["ate"] - true_ate) < 1.0, (
                f"n_folds={n_folds}: ATE {result['ate']:.3f} far from {true_ate}"
            )

    def test_atte_equals_ate_under_rct(self):
        """ATTE equals ATE when treatment is randomized (no selection)."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        T = (np.random.rand(n) > 0.5).astype(float)  # Pure RCT
        Y = 1 + X[:, 0] + 2.0 * T + np.random.randn(n)

        result_ate = irm_dml(Y, T, X, target="ate")
        result_atte = irm_dml(Y, T, X, target="atte")

        # Under RCT, ATE and ATTE should be similar
        assert abs(result_ate["ate"] - result_atte["ate"]) < 0.5, (
            f"ATE {result_ate['ate']:.3f} and ATTE {result_atte['ate']:.3f} "
            f"differ too much under RCT"
        )

    def test_atte_target_parameter(self):
        """ATTE correctly estimates effect on treated."""
        Y, T, X = generate_irm_dgp(n=500, true_ate=2.0, seed=42)

        result = irm_dml(Y, T, X, target="atte")

        assert result["target"] == "atte"
        assert result["method"] == "irm_dml"
        # ATTE should be in reasonable range
        assert 0.5 < result["ate"] < 4.0


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestIRMAdversarial:
    """Adversarial tests for edge cases and robustness."""

    def test_irm_confounded_dgp(self):
        """IRM handles confounded treatment assignment."""
        # Strong confounding
        Y, T, X = generate_irm_dgp(
            n=500, true_ate=2.0, confounding_strength=1.0, seed=42
        )

        result = irm_dml(Y, T, X, n_folds=5)

        # Should still recover ATE (IRM is doubly robust)
        assert abs(result["ate"] - 2.0) < 1.0

    def test_irm_high_dimensional(self):
        """IRM works with many covariates."""
        np.random.seed(42)
        n = 500
        p = 15
        X = np.random.randn(n, p)
        T = (np.random.rand(n) > 0.5).astype(float)
        Y = 1 + X[:, 0] + 2.0 * T + np.random.randn(n)

        result = irm_dml(Y, T, X, n_folds=5, nuisance_model="ridge")

        assert np.isfinite(result["ate"])
        assert abs(result["ate"] - 2.0) < 1.5

    def test_irm_extreme_propensity(self):
        """IRM handles near-extreme propensity scores."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        # Extreme propensity: mostly treated or mostly control
        propensity = 1 / (1 + np.exp(-2.0 * X[:, 0]))
        T = np.random.binomial(1, propensity, n).astype(float)
        Y = 1 + X[:, 0] + 2.0 * T + np.random.randn(n)

        result = irm_dml(Y, T, X, n_folds=5)

        # Should still produce finite results
        assert np.isfinite(result["ate"])
        assert np.isfinite(result["ate_se"])

    def test_irm_unbalanced_treatment(self):
        """IRM handles severely imbalanced treatment groups."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        # 90% control, 10% treated
        T = (np.random.rand(n) > 0.9).astype(float)
        Y = 1 + X[:, 0] + 2.0 * T + np.random.randn(n)

        result = irm_dml(Y, T, X, n_folds=5)

        assert np.isfinite(result["ate"])

    def test_irm_small_sample(self):
        """IRM works with small samples."""
        Y, T, X = generate_irm_dgp(n=50, true_ate=2.0, seed=42)

        result = irm_dml(Y, T, X, n_folds=2)

        assert np.isfinite(result["ate"])


class TestIRMInputValidation:
    """Tests for input validation."""

    def test_irm_invalid_n_folds(self):
        """IRM raises error for n_folds < 2."""
        Y, T, X = generate_irm_dgp(n=100, seed=42)

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            irm_dml(Y, T, X, n_folds=1)

    def test_irm_invalid_target(self):
        """IRM raises error for invalid target."""
        Y, T, X = generate_irm_dgp(n=100, seed=42)

        with pytest.raises(ValueError, match="Invalid target"):
            irm_dml(Y, T, X, target="invalid")

    def test_irm_non_binary_treatment(self):
        """IRM raises error for non-binary treatment."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        T = np.random.randint(0, 3, n).astype(float)  # 0, 1, or 2
        Y = np.random.randn(n)

        with pytest.raises(ValueError, match="binary"):
            irm_dml(Y, T, X)

    def test_irm_length_mismatch(self):
        """IRM raises error for length mismatch."""
        np.random.seed(42)
        Y = np.random.randn(100)
        T = np.random.binomial(1, 0.5, 50).astype(float)
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="Length mismatch"):
            irm_dml(Y, T, X)


class TestIRMNuisanceModels:
    """Tests for different nuisance model configurations."""

    def test_irm_ridge_nuisance(self):
        """IRM with ridge nuisance model (default)."""
        Y, T, X = generate_irm_dgp(n=300, seed=42)

        result = irm_dml(Y, T, X, nuisance_model="ridge")

        assert np.isfinite(result["ate"])

    def test_irm_linear_nuisance(self):
        """IRM with linear nuisance model."""
        Y, T, X = generate_irm_dgp(n=300, seed=42)

        result = irm_dml(Y, T, X, nuisance_model="linear")

        assert np.isfinite(result["ate"])

    def test_irm_random_forest_nuisance(self):
        """IRM with random forest nuisance model."""
        Y, T, X = generate_irm_dgp(n=300, seed=42)

        result = irm_dml(Y, T, X, nuisance_model="random_forest")

        assert np.isfinite(result["ate"])


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestIRMMonteCarlo:
    """Monte Carlo validation of statistical properties."""

    @pytest.mark.slow
    def test_irm_unbiased_constant_effect(self):
        """Monte Carlo: IRM has low bias for constant effect.

        Target: Bias < 0.10
        """
        np.random.seed(42)
        n_sims = 200
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            Y, T, X = generate_irm_dgp(
                n=300, true_ate=true_ate, seed=sim * 1000 + 42
            )
            result = irm_dml(Y, T, X, n_folds=5)
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate
        assert abs(bias) < 0.10, f"Bias {bias:.4f} exceeds threshold 0.10"

    @pytest.mark.slow
    def test_irm_coverage(self):
        """Monte Carlo: IRM has correct coverage.

        Target: 93-97% for 95% CIs
        """
        np.random.seed(42)
        n_sims = 200
        true_ate = 2.0

        covers = []
        for sim in range(n_sims):
            Y, T, X = generate_irm_dgp(
                n=300, true_ate=true_ate, seed=sim * 1000 + 42
            )
            result = irm_dml(Y, T, X, n_folds=5, alpha=0.05)
            covers.append(result["ci_lower"] < true_ate < result["ci_upper"])

        coverage = np.mean(covers)
        assert 0.90 < coverage < 0.99, (
            f"Coverage {coverage:.2%} outside acceptable range [90%, 99%]"
        )

    @pytest.mark.slow
    def test_irm_doubly_robust_outcome_misspecified(self):
        """Monte Carlo: IRM consistent when outcome model misspecified.

        The doubly robust property means IRM should be consistent if
        propensity model is correct, even if outcome model is wrong.
        """
        np.random.seed(42)
        n_sims = 100
        true_ate = 2.0

        estimates = []
        for sim in range(n_sims):
            n = 400
            np.random.seed(sim * 1000 + 42)
            X = np.random.randn(n, 2)
            # Simple propensity (easy to fit)
            propensity = 1 / (1 + np.exp(-0.3 * X[:, 0]))
            T = np.random.binomial(1, propensity, n).astype(float)
            # Complex nonlinear outcome (hard to fit with linear model)
            g0 = np.sin(X[:, 0]) + X[:, 1] ** 2
            g1 = g0 + true_ate
            Y = T * g1 + (1 - T) * g0 + np.random.randn(n)

            # Use linear model (will be misspecified for outcome)
            result = irm_dml(Y, T, X, n_folds=5, nuisance_model="linear")
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate
        # DR property: should still have reasonable bias even with misspecified outcome
        assert abs(bias) < 0.50, (
            f"Bias {bias:.4f} too large despite propensity being correct"
        )

    @pytest.mark.slow
    def test_atte_unbiased(self):
        """Monte Carlo: ATTE estimate has low bias.

        Target: Bias < 0.15
        """
        np.random.seed(42)
        n_sims = 200
        true_ate = 2.0  # In this DGP, ATTE = ATE

        estimates = []
        for sim in range(n_sims):
            Y, T, X = generate_irm_dgp(
                n=300, true_ate=true_ate, seed=sim * 1000 + 42
            )
            result = irm_dml(Y, T, X, n_folds=5, target="atte")
            estimates.append(result["ate"])

        bias = np.mean(estimates) - true_ate
        # ATTE may have slightly more variance
        assert abs(bias) < 0.15, f"ATTE bias {bias:.4f} exceeds threshold 0.15"

    @pytest.mark.slow
    def test_irm_vs_dml_comparison(self):
        """Monte Carlo: IRM and DML similar on well-specified DGP.

        When the PLR model is correct, IRM and DML should give similar results.
        """
        from src.causal_inference.cate import double_ml

        np.random.seed(42)
        n_sims = 100
        true_ate = 2.0

        irm_estimates = []
        dml_estimates = []

        for sim in range(n_sims):
            # PLR-compatible DGP: Y = θT + g(X) + ε
            Y, T, X, _ = generate_cate_dgp(
                n=300, effect_type="constant", true_ate=true_ate, seed=sim * 1000 + 42
            )

            irm_result = irm_dml(Y, T, X, n_folds=5)
            dml_result = double_ml(Y, T, X, n_folds=5)

            irm_estimates.append(irm_result["ate"])
            dml_estimates.append(dml_result["ate"])

        irm_bias = np.mean(irm_estimates) - true_ate
        dml_bias = np.mean(dml_estimates) - true_ate

        # Both should have low bias
        assert abs(irm_bias) < 0.15, f"IRM bias {irm_bias:.4f}"
        assert abs(dml_bias) < 0.15, f"DML bias {dml_bias:.4f}"

        # They should give similar results
        mean_diff = abs(np.mean(irm_estimates) - np.mean(dml_estimates))
        assert mean_diff < 0.20, f"IRM and DML differ by {mean_diff:.4f}"
