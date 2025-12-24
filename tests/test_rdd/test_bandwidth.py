"""
Layer 1: Unit Tests for RDD Bandwidth Selection.

Tests imbens_kalyanaraman_bandwidth(), cct_bandwidth(), and
cross_validation_bandwidth() with known properties and edge cases.

Coverage:
- IK bandwidth formula (kernel constants, regularization)
- CCT bias correction ratio (h_bias ≈ 1.5 * h_main)
- Cross-validation bandwidth selection
- Edge cases (small samples, zero variance, regularization)

References:
- src/causal_inference/rdd/bandwidth.py
- Imbens & Kalyanaraman (2012)
- Calonico, Cattaneo & Titiunik (2014)
"""

import numpy as np
import pytest

from src.causal_inference.rdd.bandwidth import (
    imbens_kalyanaraman_bandwidth,
    cct_bandwidth,
    cross_validation_bandwidth,
)


class TestImbensKalyanaramanBandwidth:
    """Test IK bandwidth with known properties."""

    def test_ik_triangular_vs_rectangular(self):
        """
        Triangular kernel should give smaller bandwidth than rectangular.

        Kernel constants (from IK 2012, Table 1):
        - Triangular: C1 = 3.43754
        - Rectangular: C1 = 5.40554

        Since h_IK ∝ C1, rectangular kernel → larger bandwidth.
        """
        np.random.seed(42)
        X = np.random.uniform(-5, 5, 200)
        Y = X + 2 * (X >= 0) + np.random.normal(0, 1, 200)

        h_tri = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0, kernel="triangular")
        h_rect = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0, kernel="rectangular")

        assert h_rect > h_tri, \
            f"Rectangular kernel should have larger bandwidth: h_rect={h_rect:.3f}, h_tri={h_tri:.3f}"

        # Ratio should be approximately C1_rect / C1_tri = 5.40554 / 3.43754 ≈ 1.57
        ratio = h_rect / h_tri
        expected_ratio = 5.40554 / 3.43754
        assert abs(ratio - expected_ratio) < 0.3, \
            f"Expected ratio ≈ {expected_ratio:.2f}, got {ratio:.2f}"

    def test_ik_positive_bandwidth(self):
        """IK bandwidth should always be positive."""
        np.random.seed(123)
        X = np.random.uniform(-2, 2, 100)
        Y = 0.5 * X + np.random.normal(0, 0.5, 100)

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)

        assert h > 0, f"Bandwidth should be positive, got {h}"
        assert np.isfinite(h), f"Bandwidth should be finite, got {h}"

    def test_ik_regularization_bounds(self):
        """
        IK bandwidth should be regularized within [0.1*sd(X), 2*sd(X)].

        This prevents extreme bandwidths from numerical instability.
        """
        np.random.seed(456)
        X = np.random.normal(0, 2, 150)
        Y = X**2 + np.random.normal(0, 1, 150)

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)

        x_sd = np.std(X)
        h_min = 0.1 * x_sd
        h_max = 2.0 * x_sd

        assert h >= h_min, f"Bandwidth {h:.3f} below minimum {h_min:.3f}"
        assert h <= h_max, f"Bandwidth {h:.3f} above maximum {h_max:.3f}"

    def test_ik_linear_dgp_smooth(self):
        """
        For linear DGP (Y = aX + b), second derivative m'' ≈ 0.

        IK formula: h ∝ [σ² / (n * m''²)]^(1/5)
        With m'' → 0, formula uses m'' = 1e-6 floor.

        Bandwidth will be finite and within regularization bounds.
        """
        np.random.seed(789)
        n = 200
        X = np.random.uniform(-3, 3, n)
        Y = 2.0 * X + 5.0 + np.random.normal(0, 0.5, n)  # Linear, no RD jump

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)

        # Linear function → should still yield finite positive bandwidth
        assert h > 0, f"Bandwidth should be positive, got {h}"
        assert np.isfinite(h), f"Bandwidth should be finite, got {h}"

        # Should be within regularization bounds
        x_sd = np.std(X)
        h_min = 0.1 * x_sd
        h_max = 2.0 * x_sd
        assert h_min <= h <= h_max, \
            f"Bandwidth {h:.3f} should be in [{h_min:.3f}, {h_max:.3f}]"

    def test_ik_invalid_kernel(self):
        """Should raise ValueError for unsupported kernel."""
        X = np.array([1, 2, 3, 4])
        Y = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="Unknown kernel"):
            imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0, kernel="epanechnikov")


class TestCCTBandwidth:
    """Test CCT bandwidth approximation (BUG-2 fix validation)."""

    def test_cct_emits_approximation_warning(self):
        """
        BUG-2 FIX: cct_bandwidth() should warn users it's an approximation.

        The function is NOT true CCT - it uses IK bandwidth with 1.5× scaling.
        Users should be explicitly warned about this limitation.
        """
        import warnings

        np.random.seed(42)
        X = np.random.uniform(-4, 4, 200)
        Y = X + 3 * (X >= 0) + np.random.normal(0, 1, 200)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cct_bandwidth(Y, X, cutoff=0.0)

            # Check warning was emitted
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert issubclass(w[0].category, UserWarning)
            assert "APPROXIMATION" in str(w[0].message)
            assert "rdrobust" in str(w[0].message).lower()

    def test_cct_bias_correction_ratio(self):
        """
        CCT bias correction bandwidth should be ≈ 1.5 * main bandwidth.

        From implementation: h_bias = 1.5 * h_main (ad-hoc approximation)
        """
        import warnings

        np.random.seed(42)
        X = np.random.uniform(-4, 4, 200)
        Y = X + 3 * (X >= 0) + np.random.normal(0, 1, 200)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress approximation warning
            h_main, h_bias = cct_bandwidth(Y, X, cutoff=0.0, bias_correction=True)

        # Check ratio
        ratio = h_bias / h_main
        assert abs(ratio - 1.5) < 0.01, \
            f"Expected h_bias ≈ 1.5 * h_main, got ratio = {ratio:.3f}"

    def test_cct_no_bias_correction(self):
        """
        With bias_correction=False, h_main == h_bias.
        """
        import warnings

        np.random.seed(123)
        X = np.random.uniform(-3, 3, 150)
        Y = 0.5 * X**2 + np.random.normal(0, 0.8, 150)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_main, h_bias = cct_bandwidth(Y, X, cutoff=0.0, bias_correction=False)

        assert h_main == h_bias, \
            f"With bias_correction=False, should have h_main == h_bias, got {h_main:.3f} vs {h_bias:.3f}"

    def test_cct_uses_ik_approximation(self):
        """
        CCT implementation uses IK bandwidth as approximation.

        BUG-2 DOCUMENTATION: This test documents the known limitation that
        CCT h_main equals IK bandwidth (not true CCT bandwidth).
        """
        import warnings

        np.random.seed(456)
        X = np.random.uniform(-5, 5, 200)
        Y = np.sin(X) + 2 * (X >= 0) + np.random.normal(0, 1, 200)

        h_ik = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_main, h_bias = cct_bandwidth(Y, X, cutoff=0.0)

        assert h_main == h_ik, \
            f"CCT main bandwidth should equal IK bandwidth, got {h_main:.3f} vs {h_ik:.3f}"


class TestCrossValidationBandwidth:
    """Test CV bandwidth selection."""

    def test_cv_selects_from_grid(self):
        """
        CV bandwidth should be one of the values in the provided grid.
        """
        np.random.seed(42)
        X = np.random.uniform(-2, 2, 100)
        Y = X + 1.5 * (X >= 0) + np.random.normal(0, 0.5, 100)

        h_grid = np.array([0.5, 1.0, 1.5, 2.0])
        h_cv = cross_validation_bandwidth(Y, X, cutoff=0.0, h_grid=h_grid)

        assert h_cv in h_grid, \
            f"CV bandwidth {h_cv} should be in grid {h_grid}"

    def test_cv_default_grid_range(self):
        """
        Default grid should span [0.5*sd(X), 2.0*sd(X)].
        """
        np.random.seed(123)
        X = np.random.normal(0, 1.5, 120)
        Y = X + 2 * (X >= 0) + np.random.normal(0, 0.8, 120)

        h_cv = cross_validation_bandwidth(Y, X, cutoff=0.0)

        x_sd = np.std(X)
        h_min = 0.5 * x_sd
        h_max = 2.0 * x_sd

        assert h_min <= h_cv <= h_max, \
            f"CV bandwidth {h_cv:.3f} should be in [{h_min:.3f}, {h_max:.3f}]"

    def test_cv_positive_finite(self):
        """CV bandwidth should be positive and finite."""
        np.random.seed(789)
        X = np.random.uniform(-3, 3, 80)
        Y = 0.8 * X + np.random.normal(0, 1, 80)

        h_cv = cross_validation_bandwidth(Y, X, cutoff=0.0)

        assert h_cv > 0, f"CV bandwidth should be positive, got {h_cv}"
        assert np.isfinite(h_cv), f"CV bandwidth should be finite, got {h_cv}"


class TestBandwidthEdgeCases:
    """Edge cases for bandwidth selection."""

    def test_ik_small_sample(self):
        """IK bandwidth with small sample (n < 30)."""
        np.random.seed(42)
        X = np.random.uniform(-2, 2, 20)  # Small sample
        Y = X + 1 * (X >= 0) + np.random.normal(0, 0.5, 20)

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)

        # Should still return finite positive bandwidth
        assert h > 0
        assert np.isfinite(h)

    def test_ik_perfect_separation(self):
        """IK bandwidth when treatment/control perfectly separated."""
        # All treated on right, all control on left
        X = np.array([-2, -1.5, -1, 0.5, 1, 1.5])
        Y = np.array([1, 1.2, 1.1, 3.5, 3.8, 3.6])  # Sharp jump at 0

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)

        # Should still compute bandwidth
        assert h > 0
        assert np.isfinite(h)

    def test_cct_returns_two_values(self):
        """CCT should always return tuple of (h_main, h_bias)."""
        import warnings

        np.random.seed(42)
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([1, 2, 3, 5, 6])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cct_bandwidth(Y, X, cutoff=3.0)

        assert isinstance(result, tuple), "CCT should return tuple"
        assert len(result) == 2, "CCT should return (h_main, h_bias)"

        h_main, h_bias = result
        assert h_main > 0
        assert h_bias > 0
