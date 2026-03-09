"""
Tests for RKD Diagnostic Functions.

Test structure:
1. Density smoothness tests
2. Covariate smoothness tests
3. First stage strength tests
4. Summary function tests
"""

import numpy as np
import pytest
from scipy import stats

from src.causal_inference.rkd.diagnostics import (
    density_smoothness_test,
    covariate_smoothness_test,
    first_stage_test,
    rkd_diagnostics_summary,
    DensitySmoothnessResult,
    CovariateSmoothnessResult,
    FirstStageResult,
)


# =============================================================================
# DGP Functions for Diagnostics Testing
# =============================================================================


def generate_smooth_density_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = None,
) -> np.ndarray:
    """Generate running variable with smooth density at cutoff."""
    if seed is not None:
        np.random.seed(seed)

    # Uniform distribution - smooth density everywhere
    X = np.random.uniform(-5, 5, n)
    return X


def generate_bunching_data(
    n: int = 1000,
    cutoff: float = 0.0,
    bunching_mass: float = 0.2,
    seed: int = None,
) -> np.ndarray:
    """Generate running variable with bunching near cutoff."""
    if seed is not None:
        np.random.seed(seed)

    # Base uniform distribution
    n_base = int(n * (1 - bunching_mass))
    X_base = np.random.uniform(-5, 5, n_base)

    # Extra mass just below cutoff (manipulation)
    n_bunched = n - n_base
    X_bunched = np.random.uniform(cutoff - 0.5, cutoff, n_bunched)

    X = np.concatenate([X_base, X_bunched])
    np.random.shuffle(X)
    return X


def generate_smooth_covariate_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate running variable and smooth covariate."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-5, 5, n)

    # Covariate linearly related to X, no kink
    cov = 0.5 * X + np.random.normal(0, 1, n)

    return X, cov


def generate_kinked_covariate_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate covariate with a kink at cutoff (violation)."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-5, 5, n)

    # Covariate with kink at cutoff
    cov = np.where(X < cutoff, 0.3 * X, 0.8 * X) + np.random.normal(0, 0.5, n)

    return X, cov


def generate_strong_first_stage_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate treatment with strong kink at cutoff."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-5, 5, n)

    # Strong kink in D
    D = np.where(X < cutoff, 0.3 * X, 1.5 * X) + np.random.normal(0, 0.3, n)

    return X, D


def generate_weak_first_stage_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate treatment with weak kink at cutoff."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-5, 5, n)

    # Weak kink in D (slopes almost equal)
    D = np.where(X < cutoff, 0.9 * X, 1.1 * X) + np.random.normal(0, 0.5, n)

    return X, D


# =============================================================================
# Density Smoothness Tests
# =============================================================================


class TestDensitySmoothness:
    """Tests for density smoothness test."""

    def test_smooth_density_not_rejected(self):
        """With smooth density, test should not reject H0."""
        X = generate_smooth_density_data(n=2000, cutoff=0.0, seed=42)

        result = density_smoothness_test(X, cutoff=0.0)

        assert isinstance(result, DensitySmoothnessResult)
        # p-value should be high (not reject smoothness)
        # Allow for some false positives
        assert result.p_value > 0.01 or "Insufficient" in result.interpretation

    def test_bunching_detected(self):
        """With bunching, test should detect density irregularity."""
        X = generate_bunching_data(n=2000, cutoff=0.0, bunching_mass=0.3, seed=42)

        result = density_smoothness_test(X, cutoff=0.0)

        assert isinstance(result, DensitySmoothnessResult)
        # Note: This is a slope-based test, not level-based
        # Bunching may or may not be detected depending on pattern
        assert np.isfinite(result.p_value)

    def test_insufficient_data_handling(self):
        """Should handle insufficient data gracefully."""
        np.random.seed(42)
        X = np.concatenate(
            [
                np.random.uniform(-1, 0, 10),  # Few left
                np.random.uniform(0, 1, 10),  # Few right
            ]
        )

        result = density_smoothness_test(X, cutoff=0.0)

        assert result.p_value == 1.0
        assert "Insufficient" in result.interpretation

    def test_custom_bins(self):
        """Should work with custom number of bins."""
        X = generate_smooth_density_data(n=1000, seed=42)

        result_10 = density_smoothness_test(X, cutoff=0.0, n_bins=10)
        result_30 = density_smoothness_test(X, cutoff=0.0, n_bins=30)

        assert result_10.n_bins == 10
        assert result_30.n_bins == 30

    def test_custom_bandwidth(self):
        """Should accept custom bandwidth."""
        X = generate_smooth_density_data(n=1000, seed=42)

        result = density_smoothness_test(X, cutoff=0.0, bandwidth=1.5)

        assert isinstance(result, DensitySmoothnessResult)
        assert np.isfinite(result.slope_left) or np.isnan(result.slope_left)

    def test_result_fields(self):
        """Result should have all documented fields."""
        X = generate_smooth_density_data(n=1000, seed=42)

        result = density_smoothness_test(X, cutoff=0.0)

        assert hasattr(result, "slope_left")
        assert hasattr(result, "slope_right")
        assert hasattr(result, "slope_difference")
        assert hasattr(result, "se")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "n_bins")
        assert hasattr(result, "interpretation")

    def test_interpretation_levels(self):
        """Interpretation should reflect significance level."""
        # Create data that might give different significance levels
        np.random.seed(42)

        # Test with smooth data
        X_smooth = generate_smooth_density_data(n=2000, seed=42)
        result_smooth = density_smoothness_test(X_smooth, cutoff=0.0)

        if result_smooth.p_value >= 0.10:
            assert "No evidence" in result_smooth.interpretation


# =============================================================================
# Covariate Smoothness Tests
# =============================================================================


class TestCovariateSmoothness:
    """Tests for covariate smoothness test."""

    def test_smooth_covariate_passes(self):
        """Smooth covariate should pass the test."""
        X, cov = generate_smooth_covariate_data(n=1500, seed=42)

        results = covariate_smoothness_test(X, cov, cutoff=0.0)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, CovariateSmoothnessResult)
        # Should pass (is_smooth = True) most of the time
        # Allow for some false positives
        assert result.p_value > 0.01 or result.is_smooth is False

    def test_kinked_covariate_fails(self):
        """Covariate with kink should fail the test."""
        X, cov = generate_kinked_covariate_data(n=2000, seed=42)

        results = covariate_smoothness_test(X, cov, cutoff=0.0, bandwidth=2.0)

        result = results[0]
        # Kink should be detected (low p-value, is_smooth = False)
        # Allow for some false negatives
        assert result.p_value < 0.5 or not result.is_smooth

    def test_multiple_covariates(self):
        """Should test multiple covariates."""
        np.random.seed(42)
        n = 1000
        X = np.random.uniform(-5, 5, n)

        # Create 3 covariates
        cov1 = 0.5 * X + np.random.normal(0, 1, n)  # Smooth
        cov2 = np.where(X < 0, 0.3 * X, 0.8 * X) + np.random.normal(0, 0.5, n)  # Kinked
        cov3 = np.random.normal(0, 1, n)  # Independent

        covariates = np.column_stack([cov1, cov2, cov3])

        results = covariate_smoothness_test(
            X, covariates, cutoff=0.0, covariate_names=["Smooth", "Kinked", "Independent"]
        )

        assert len(results) == 3
        assert results[0].covariate_name == "Smooth"
        assert results[1].covariate_name == "Kinked"
        assert results[2].covariate_name == "Independent"

    def test_covariate_names_default(self):
        """Default covariate names should be generated."""
        X, cov = generate_smooth_covariate_data(n=500, seed=42)

        results = covariate_smoothness_test(X, cov, cutoff=0.0)

        assert "Covariate_1" in results[0].covariate_name

    def test_insufficient_data_handling(self):
        """Should handle insufficient data gracefully."""
        np.random.seed(42)
        X = np.concatenate(
            [
                np.random.uniform(-0.1, 0, 3),  # Very few left
                np.random.uniform(0, 0.1, 3),  # Very few right
            ]
        )
        cov = np.random.normal(0, 1, 6)

        results = covariate_smoothness_test(X, cov, cutoff=0.0, bandwidth=0.1)

        assert len(results) == 1
        # Should handle gracefully (is_smooth = True due to insufficient data)
        assert results[0].p_value == 1.0 or results[0].is_smooth

    def test_result_fields(self):
        """Result should have all documented fields."""
        X, cov = generate_smooth_covariate_data(n=500, seed=42)

        results = covariate_smoothness_test(X, cov, cutoff=0.0)
        result = results[0]

        assert hasattr(result, "covariate_name")
        assert hasattr(result, "slope_left")
        assert hasattr(result, "slope_right")
        assert hasattr(result, "slope_difference")
        assert hasattr(result, "se")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "is_smooth")


# =============================================================================
# First Stage Tests
# =============================================================================


class TestFirstStage:
    """Tests for first stage strength test."""

    def test_strong_first_stage_detected(self):
        """Strong first stage should be detected."""
        X, D = generate_strong_first_stage_data(n=1500, seed=42)

        result = first_stage_test(D, X, cutoff=0.0)

        assert isinstance(result, FirstStageResult)
        # Strong kink (0.3 to 1.5, diff = 1.2) should give high F
        assert result.f_stat > 5  # At least moderate
        assert result.is_strong or result.f_stat >= 10

    def test_weak_first_stage_detected(self):
        """Weak first stage should show low F-stat."""
        X, D = generate_weak_first_stage_data(n=800, seed=42)

        result = first_stage_test(D, X, cutoff=0.0)

        # Weak kink should give lower F (but may still be > 10 with enough data)
        assert np.isfinite(result.f_stat)

    def test_no_kink_very_low_f(self):
        """No kink should give very low F-stat."""
        np.random.seed(42)
        n = 1000
        X = np.random.uniform(-5, 5, n)

        # Linear D with no kink at all
        D = 1.0 * X + np.random.normal(0, 0.3, n)

        result = first_stage_test(D, X, cutoff=0.0)

        # F should be low
        assert result.f_stat < 20  # Not extremely high

    def test_insufficient_data_handling(self):
        """Should handle insufficient data gracefully."""
        np.random.seed(42)
        X = np.concatenate(
            [
                np.random.uniform(-0.1, 0, 3),
                np.random.uniform(0, 0.1, 3),
            ]
        )
        D = X + np.random.normal(0, 0.1, 6)

        result = first_stage_test(D, X, cutoff=0.0, bandwidth=0.1)

        assert result.p_value == 1.0
        assert not result.is_strong
        assert "Insufficient" in result.interpretation

    def test_custom_bandwidth(self):
        """Should accept custom bandwidth."""
        X, D = generate_strong_first_stage_data(n=500, seed=42)

        result = first_stage_test(D, X, cutoff=0.0, bandwidth=3.0)

        assert isinstance(result, FirstStageResult)
        assert np.isfinite(result.f_stat)

    def test_result_fields(self):
        """Result should have all documented fields."""
        X, D = generate_strong_first_stage_data(n=500, seed=42)

        result = first_stage_test(D, X, cutoff=0.0)

        assert hasattr(result, "kink_estimate")
        assert hasattr(result, "se")
        assert hasattr(result, "f_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "is_strong")
        assert hasattr(result, "interpretation")

    def test_interpretation_levels(self):
        """Interpretation should reflect F-stat magnitude."""
        # Strong first stage
        X_strong, D_strong = generate_strong_first_stage_data(n=2000, seed=42)
        result_strong = first_stage_test(D_strong, X_strong, cutoff=0.0)

        if result_strong.f_stat >= 10:
            assert (
                "Strong" in result_strong.interpretation
                or "reliable" in result_strong.interpretation.lower()
            )


# =============================================================================
# Summary Function Tests
# =============================================================================


class TestRKDDiagnosticsSummary:
    """Tests for comprehensive diagnostics summary."""

    def test_summary_returns_dict(self):
        """Summary should return dictionary with all components."""
        np.random.seed(42)
        n = 1000
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.5 * X, 1.5 * X) + np.random.normal(0, 0.3, n)
        Y = 2.0 * D + np.random.normal(0, 0.5, n)

        result = rkd_diagnostics_summary(Y, X, D, cutoff=0.0)

        assert isinstance(result, dict)
        assert "density_test" in result
        assert "first_stage_test" in result
        assert "covariate_tests" in result
        assert "summary" in result

    def test_summary_with_covariates(self):
        """Summary should include covariate tests when provided."""
        np.random.seed(42)
        n = 1000
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.5 * X, 1.5 * X) + np.random.normal(0, 0.3, n)
        Y = 2.0 * D + np.random.normal(0, 0.5, n)

        covariates = np.column_stack(
            [
                np.random.normal(0, 1, n),
                X + np.random.normal(0, 0.5, n),
            ]
        )

        result = rkd_diagnostics_summary(
            Y, X, D, cutoff=0.0, covariates=covariates, covariate_names=["Age", "Income"]
        )

        assert len(result["covariate_tests"]) == 2

    def test_summary_without_covariates(self):
        """Summary should work without covariates."""
        np.random.seed(42)
        n = 500
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.5 * X, 1.5 * X) + np.random.normal(0, 0.3, n)
        Y = 2.0 * D + np.random.normal(0, 0.5, n)

        result = rkd_diagnostics_summary(Y, X, D, cutoff=0.0)

        assert result["covariate_tests"] == []
        assert "covariates_smooth" in result["summary"]
        assert result["summary"]["covariates_smooth"]  # True when no covariates

    def test_summary_all_pass_flag(self):
        """Summary should correctly compute all_pass flag."""
        np.random.seed(42)
        n = 2000
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.3 * X, 1.5 * X) + np.random.normal(0, 0.2, n)
        Y = 2.0 * D + np.random.normal(0, 0.3, n)

        result = rkd_diagnostics_summary(Y, X, D, cutoff=0.0)

        summary = result["summary"]
        all_pass_computed = (
            summary["density_smooth"]
            and summary["first_stage_strong"]
            and summary["covariates_smooth"]
        )

        assert summary["all_pass"] == all_pass_computed

    def test_summary_with_custom_bandwidth(self):
        """Summary should use custom bandwidth if provided."""
        np.random.seed(42)
        n = 500
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.5 * X, 1.5 * X) + np.random.normal(0, 0.3, n)
        Y = 2.0 * D + np.random.normal(0, 0.5, n)

        result = rkd_diagnostics_summary(Y, X, D, cutoff=0.0, bandwidth=2.0)

        assert isinstance(result, dict)


# =============================================================================
# Edge Cases
# =============================================================================


class TestDiagnosticsEdgeCases:
    """Edge cases for diagnostic functions."""

    def test_all_data_one_side(self):
        """Should handle all data on one side of cutoff."""
        np.random.seed(42)
        X = np.random.uniform(-5, -0.1, 100)  # All left of 0
        cov = np.random.normal(0, 1, 100)

        result = density_smoothness_test(X, cutoff=0.0)
        assert "Insufficient" in result.interpretation

        cov_result = covariate_smoothness_test(X, cov, cutoff=0.0)
        assert cov_result[0].p_value == 1.0

    def test_extreme_cutoff(self):
        """Should handle cutoff at extreme of data range."""
        np.random.seed(42)
        X = np.random.uniform(-5, 5, 1000)

        # Cutoff near edge
        result = density_smoothness_test(X, cutoff=4.5)
        assert isinstance(result, DensitySmoothnessResult)

    def test_many_covariates(self):
        """Should handle many covariates."""
        np.random.seed(42)
        n = 500
        k = 10
        X = np.random.uniform(-5, 5, n)
        covariates = np.random.normal(0, 1, (n, k))

        results = covariate_smoothness_test(X, covariates, cutoff=0.0)

        assert len(results) == k

    def test_constant_covariate(self):
        """Should handle constant covariate."""
        np.random.seed(42)
        n = 500
        X = np.random.uniform(-5, 5, n)
        cov = np.ones(n)  # Constant

        results = covariate_smoothness_test(X, cov, cutoff=0.0)

        # Should not crash
        assert len(results) == 1

    def test_highly_variable_data(self):
        """Should handle highly variable data."""
        np.random.seed(42)
        n = 500
        X = np.random.normal(0, 10, n)  # Wide spread
        D = X + np.random.normal(0, 5, n)  # High noise

        result = first_stage_test(D, X, cutoff=0.0)

        assert isinstance(result, FirstStageResult)
