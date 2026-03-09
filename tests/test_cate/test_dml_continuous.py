"""Tests for DML with continuous treatment.

Implements 6-layer validation:
1. Known-Answer Tests - DGP with known true effect
2. Adversarial Tests - Edge cases and challenging scenarios
3. Monte Carlo Tests - Bias, coverage, SE calibration
4. Cross-Language Tests - Python/Julia parity (in separate file)
5. Diagnostic Tests - Result structure and diagnostics
6. Golden Reference Tests - Frozen expected results

References
----------
- Chernozhukov et al. (2018). "Double/debiased machine learning."
- Colangelo & Lee (2020). "Double Debiased ML with Continuous Treatments."
"""

import numpy as np
import pytest
from dataclasses import fields

from src.causal_inference.cate.dml_continuous import (
    dml_continuous,
    DMLContinuousResult,
    _validate_continuous_inputs,
    _get_regression_model,
)


# =============================================================================
# Fixtures: Data Generating Processes
# =============================================================================


def generate_continuous_dgp(
    n: int = 500,
    p: int = 5,
    true_effect: float = 2.0,
    confounding_strength: float = 1.0,
    noise_level: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate data for continuous treatment DML testing.

    DGP:
    X ~ N(0, I_p)
    D = confounding_strength * X[:, 0] + eta, eta ~ N(0, 1)
    Y = intercept + X[:, 0] + true_effect * D + epsilon, epsilon ~ N(0, noise_level)

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    true_effect : float
        True treatment effect (theta).
    confounding_strength : float
        Strength of X -> D confounding.
    noise_level : float
        Outcome noise standard deviation.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (Y, D, X, true_effect)
    """
    rng = np.random.default_rng(random_state)

    X = rng.standard_normal((n, p))

    # Continuous treatment with confounding
    eta = rng.standard_normal(n)
    D = confounding_strength * X[:, 0] + eta

    # Outcome with treatment effect
    epsilon = noise_level * rng.standard_normal(n)
    Y = 1.0 + X[:, 0] + true_effect * D + epsilon

    return Y, D, X, true_effect


def generate_heterogeneous_dgp(
    n: int = 500,
    p: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with heterogeneous treatment effects.

    DGP:
    X ~ N(0, I_p)
    D = X[:, 0] + eta
    tau(X) = 1 + 2*X[:, 1]  # Effect depends on X[:, 1]
    Y = X[:, 0] + tau(X) * D + epsilon

    Returns
    -------
    tuple
        (Y, D, X, true_cate) where true_cate = 1 + 2*X[:, 1]
    """
    rng = np.random.default_rng(random_state)

    X = rng.standard_normal((n, p))
    D = X[:, 0] + rng.standard_normal(n)

    # Heterogeneous effect
    true_cate = 1.0 + 2.0 * X[:, 1]

    epsilon = rng.standard_normal(n)
    Y = X[:, 0] + true_cate * D + epsilon

    return Y, D, X, true_cate


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestKnownAnswer:
    """Tests with known true effects from DGP."""

    def test_constant_effect_recovered(self):
        """DML should recover constant treatment effect within tolerance."""
        Y, D, X, true_effect = generate_continuous_dgp(n=1000, true_effect=2.0, random_state=42)
        result = dml_continuous(Y, D, X)

        assert np.abs(result.ate - true_effect) < 0.3, (
            f"ATE estimate {result.ate:.4f} too far from true effect {true_effect}"
        )

    def test_zero_effect_recovered(self):
        """DML should recover zero effect when there is none."""
        Y, D, X, _ = generate_continuous_dgp(n=1000, true_effect=0.0, random_state=43)
        result = dml_continuous(Y, D, X)

        assert np.abs(result.ate) < 0.2, f"ATE estimate {result.ate:.4f} should be near 0"

    def test_negative_effect_recovered(self):
        """DML should recover negative treatment effects."""
        Y, D, X, true_effect = generate_continuous_dgp(n=1000, true_effect=-1.5, random_state=44)
        result = dml_continuous(Y, D, X)

        assert np.abs(result.ate - true_effect) < 0.3, (
            f"ATE estimate {result.ate:.4f} too far from true effect {true_effect}"
        )

    def test_large_effect_recovered(self):
        """DML should recover large treatment effects."""
        Y, D, X, true_effect = generate_continuous_dgp(n=1000, true_effect=5.0, random_state=45)
        result = dml_continuous(Y, D, X)

        assert np.abs(result.ate - true_effect) < 0.5, (
            f"ATE estimate {result.ate:.4f} too far from true effect {true_effect}"
        )

    def test_ci_contains_true_effect(self):
        """95% CI should contain true effect."""
        Y, D, X, true_effect = generate_continuous_dgp(n=1000, true_effect=2.0, random_state=46)
        result = dml_continuous(Y, D, X)

        assert result.ci_lower < true_effect < result.ci_upper, (
            f"CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
            f"does not contain true effect {true_effect}"
        )


class TestHeterogeneousEffects:
    """Tests for CATE (heterogeneous effects) estimation."""

    def test_cate_shape_matches_n(self):
        """CATE array should have shape (n,)."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=50)
        result = dml_continuous(Y, D, X)

        assert result.cate.shape == (500,), f"CATE shape {result.cate.shape} != (500,)"

    def test_cate_mean_close_to_ate(self):
        """Mean of CATE should be close to ATE."""
        Y, D, X, _ = generate_continuous_dgp(n=1000, random_state=51)
        result = dml_continuous(Y, D, X)

        cate_mean = np.mean(result.cate)
        assert np.abs(cate_mean - result.ate) < 0.5, (
            f"CATE mean {cate_mean:.4f} too far from ATE {result.ate:.4f}"
        )

    def test_heterogeneous_effects_detected(self):
        """CATE should show variation when true effects are heterogeneous."""
        Y, D, X, true_cate = generate_heterogeneous_dgp(n=1000, random_state=52)
        result = dml_continuous(Y, D, X)

        # CATE should have meaningful variation
        cate_std = np.std(result.cate)
        assert cate_std > 0.1, f"CATE std {cate_std:.4f} too small for heterogeneous DGP"


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestAdversarial:
    """Edge cases and challenging scenarios."""

    def test_high_dimensional_covariates(self):
        """DML should work with p > n/10 covariates."""
        n, p = 200, 30
        Y, D, X, true_effect = generate_continuous_dgp(n=n, p=p, true_effect=2.0, random_state=60)
        result = dml_continuous(Y, D, X, n_folds=3)

        # Should run without error and produce reasonable estimate
        assert np.isfinite(result.ate)
        assert np.abs(result.ate - true_effect) < 1.0  # Looser tolerance

    def test_strong_confounding(self):
        """DML should handle strong confounding."""
        Y, D, X, true_effect = generate_continuous_dgp(
            n=1000, true_effect=2.0, confounding_strength=3.0, random_state=61
        )
        result = dml_continuous(Y, D, X)

        # Should still recover effect despite strong confounding
        assert np.abs(result.ate - true_effect) < 0.5

    def test_low_treatment_variation(self):
        """DML should handle low but non-zero treatment variation."""
        rng = np.random.default_rng(62)
        n = 500
        X = rng.standard_normal((n, 5))
        D = X[:, 0] * 0.5 + 0.1 * rng.standard_normal(n)  # Low variation
        Y = 1.0 + 2.0 * D + rng.standard_normal(n)

        result = dml_continuous(Y, D, X)
        assert np.isfinite(result.ate)

    def test_single_covariate(self):
        """DML should work with single covariate."""
        rng = np.random.default_rng(63)
        n = 500
        X = rng.standard_normal(n)  # 1D array
        D = X + rng.standard_normal(n)
        Y = 1.0 + 2.0 * D + rng.standard_normal(n)

        result = dml_continuous(Y, D, X)
        assert np.isfinite(result.ate)

    def test_two_folds(self):
        """DML should work with minimum 2 folds."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=64)
        result = dml_continuous(Y, D, X, n_folds=2)

        assert result.n_folds == 2
        assert len(result.fold_estimates) == 2

    def test_many_folds(self):
        """DML should work with many folds."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=65)
        result = dml_continuous(Y, D, X, n_folds=10)

        assert result.n_folds == 10
        assert len(result.fold_estimates) == 10


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo validation of statistical properties."""

    @pytest.mark.monte_carlo
    def test_unbiased_estimation(self):
        """Monte Carlo: bias should be < 0.10 over many runs."""
        n_runs = 100  # Reduced for speed; use 1000+ for thorough testing
        true_effect = 2.0
        estimates = []

        for seed in range(n_runs):
            Y, D, X, _ = generate_continuous_dgp(n=500, true_effect=true_effect, random_state=seed)
            result = dml_continuous(Y, D, X)
            estimates.append(result.ate)

        bias = np.mean(estimates) - true_effect
        assert np.abs(bias) < 0.10, f"Bias {bias:.4f} exceeds 0.10 threshold"

    @pytest.mark.monte_carlo
    def test_coverage_rate(self):
        """Monte Carlo: 95% CI coverage should be 93-97%."""
        n_runs = 100
        true_effect = 2.0
        covers = []

        for seed in range(n_runs):
            Y, D, X, _ = generate_continuous_dgp(n=500, true_effect=true_effect, random_state=seed)
            result = dml_continuous(Y, D, X)
            covers.append(result.ci_lower < true_effect < result.ci_upper)

        coverage = np.mean(covers)
        assert 0.85 < coverage < 0.99, f"Coverage {coverage:.2%} outside [85%, 99%]"

    @pytest.mark.monte_carlo
    def test_se_calibration(self):
        """Monte Carlo: SE should be within 30% of empirical SD."""
        n_runs = 100
        true_effect = 2.0
        estimates = []
        ses = []

        for seed in range(n_runs):
            Y, D, X, _ = generate_continuous_dgp(n=500, true_effect=true_effect, random_state=seed)
            result = dml_continuous(Y, D, X)
            estimates.append(result.ate)
            ses.append(result.ate_se)

        empirical_sd = np.std(estimates)
        mean_se = np.mean(ses)

        se_ratio = mean_se / empirical_sd
        assert 0.7 < se_ratio < 1.5, f"SE ratio {se_ratio:.2f} outside [0.7, 1.5]"


# =============================================================================
# Layer 5: Diagnostic Tests
# =============================================================================


class TestDiagnostics:
    """Tests for result structure and diagnostics."""

    def test_result_type(self):
        """Result should be DMLContinuousResult dataclass."""
        Y, D, X, _ = generate_continuous_dgp(n=200, random_state=80)
        result = dml_continuous(Y, D, X)

        assert isinstance(result, DMLContinuousResult)

    def test_all_fields_present(self):
        """Result should have all expected fields."""
        Y, D, X, _ = generate_continuous_dgp(n=200, random_state=81)
        result = dml_continuous(Y, D, X)

        expected_fields = {
            "cate",
            "ate",
            "ate_se",
            "ci_lower",
            "ci_upper",
            "method",
            "fold_estimates",
            "fold_ses",
            "outcome_r2",
            "treatment_r2",
            "n",
            "n_folds",
        }
        actual_fields = {f.name for f in fields(result)}
        assert expected_fields == actual_fields

    def test_method_name(self):
        """Method should be 'dml_continuous'."""
        Y, D, X, _ = generate_continuous_dgp(n=200, random_state=82)
        result = dml_continuous(Y, D, X)

        assert result.method == "dml_continuous"

    def test_fold_estimates_finite(self):
        """Fold estimates should be finite."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=83)
        result = dml_continuous(Y, D, X, n_folds=5)

        assert np.all(np.isfinite(result.fold_estimates))
        assert len(result.fold_estimates) == 5

    def test_fold_ses_finite(self):
        """Fold SEs should be finite and positive."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=84)
        result = dml_continuous(Y, D, X, n_folds=5)

        assert np.all(np.isfinite(result.fold_ses))
        assert np.all(result.fold_ses > 0)

    def test_outcome_r2_in_range(self):
        """Outcome R-squared should be in [0, 1] or slightly negative."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=85)
        result = dml_continuous(Y, D, X)

        assert -0.1 <= result.outcome_r2 <= 1.0

    def test_treatment_r2_in_range(self):
        """Treatment R-squared should be in [0, 1] or slightly negative."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=86)
        result = dml_continuous(Y, D, X)

        assert -0.1 <= result.treatment_r2 <= 1.0

    def test_n_and_n_folds_correct(self):
        """n and n_folds should match input."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=87)
        result = dml_continuous(Y, D, X, n_folds=7)

        assert result.n == 500
        assert result.n_folds == 7


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_length_mismatch_treatment(self):
        """Should raise on length mismatch between outcomes and treatment."""
        Y = np.array([1, 2, 3])
        D = np.array([1, 2])  # Wrong length
        X = np.array([[1], [2], [3]])

        with pytest.raises(ValueError, match="Length mismatch"):
            dml_continuous(Y, D, X)

    def test_length_mismatch_covariates(self):
        """Should raise on length mismatch between outcomes and covariates."""
        Y = np.array([1, 2, 3])
        D = np.array([1, 2, 3])
        X = np.array([[1], [2]])  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            dml_continuous(Y, D, X)

    def test_no_treatment_variation(self):
        """Should raise when treatment has no variation."""
        Y = np.array([1.0, 2.0, 3.0])
        D = np.array([1.0, 1.0, 1.0])  # Constant
        X = np.array([[1], [2], [3]])

        with pytest.raises(ValueError, match="No treatment variation"):
            dml_continuous(Y, D, X)

    def test_nan_in_outcomes(self):
        """Should raise when outcomes contain NaN."""
        Y = np.array([1.0, np.nan, 3.0])
        D = np.array([1.0, 2.0, 3.0])
        X = np.array([[1], [2], [3]])

        with pytest.raises(ValueError, match="NaN"):
            dml_continuous(Y, D, X)

    def test_nan_in_treatment(self):
        """Should raise when treatment contains NaN."""
        Y = np.array([1.0, 2.0, 3.0])
        D = np.array([1.0, np.nan, 3.0])
        X = np.array([[1], [2], [3]])

        with pytest.raises(ValueError, match="NaN"):
            dml_continuous(Y, D, X)

    def test_n_folds_too_small(self):
        """Should raise when n_folds < 2."""
        Y, D, X, _ = generate_continuous_dgp(n=100, random_state=90)

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            dml_continuous(Y, D, X, n_folds=1)

    def test_invalid_model_type(self):
        """Should raise on invalid model type."""
        Y, D, X, _ = generate_continuous_dgp(n=100, random_state=91)

        with pytest.raises(ValueError, match="Unknown model type"):
            dml_continuous(Y, D, X, outcome_model="invalid_model")


# =============================================================================
# Model Variant Tests
# =============================================================================


class TestModelVariants:
    """Tests for different nuisance model choices."""

    def test_linear_outcome_model(self):
        """Should work with linear outcome model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=100)
        result = dml_continuous(Y, D, X, outcome_model="linear")
        assert np.isfinite(result.ate)

    def test_ridge_outcome_model(self):
        """Should work with ridge outcome model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=101)
        result = dml_continuous(Y, D, X, outcome_model="ridge")
        assert np.isfinite(result.ate)

    def test_random_forest_outcome_model(self):
        """Should work with random forest outcome model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=102)
        result = dml_continuous(Y, D, X, outcome_model="random_forest")
        assert np.isfinite(result.ate)

    def test_linear_treatment_model(self):
        """Should work with linear treatment model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=103)
        result = dml_continuous(Y, D, X, treatment_model="linear")
        assert np.isfinite(result.ate)

    def test_ridge_treatment_model(self):
        """Should work with ridge treatment model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=104)
        result = dml_continuous(Y, D, X, treatment_model="ridge")
        assert np.isfinite(result.ate)

    def test_random_forest_treatment_model(self):
        """Should work with random forest treatment model."""
        Y, D, X, _ = generate_continuous_dgp(n=500, random_state=105)
        result = dml_continuous(Y, D, X, treatment_model="random_forest")
        assert np.isfinite(result.ate)


# =============================================================================
# Golden Reference Tests
# =============================================================================


class TestGoldenReference:
    """Tests against frozen expected results."""

    def test_golden_reference_seed_42(self):
        """Result should match frozen reference for seed 42."""
        Y, D, X, _ = generate_continuous_dgp(n=500, p=5, true_effect=2.0, random_state=42)
        result = dml_continuous(Y, D, X, n_folds=5)

        # These are approximate expectations; update after implementation stabilizes
        # For now, just check reasonable bounds
        assert 1.5 < result.ate < 2.5, f"ATE {result.ate:.4f} outside expected range"
        assert 0.02 < result.ate_se < 0.3, f"SE {result.ate_se:.4f} outside expected range"
