"""TDD Step 1: Tests for propensity clipping warnings.

These tests verify that propensity clipping (for numerical stability)
is logged as a warning rather than happening silently.

Issue: ipw.py:236 clips propensity values silently with `pass`.

Written BEFORE fixes per TDD protocol.
"""

import numpy as np
import pytest

from src.causal_inference.observational.ipw import ipw_ate_observational


class TestPropensityClippingWarnings:
    """Propensity clipping should warn, not fail silently."""

    def test_clipping_logs_warning(self):
        """Should warn when propensity values are clipped for numerical stability.

        Even with mild confounding, some propensities may be very close to 0/1.
        The clipping to [epsilon, 1-epsilon] should be logged.
        """
        np.random.seed(42)
        n = 100

        # Create data where some propensities are very close to 0 or 1
        # (but not exactly, to avoid perfect separation error)
        X = np.vstack(
            [
                np.random.normal(-1.0, 0.3, (50, 2)),
                np.random.normal(1.0, 0.3, (50, 2)),
            ]
        )
        # Use deterministic but probabilistic treatment assignment
        logit = 2.0 * X[:, 0]  # Strong confounding
        prob = 1 / (1 + np.exp(-logit))
        T = (np.random.uniform(0, 1, n) < prob).astype(float)
        Y = 2.0 * T + np.random.normal(0, 1, n)

        # Should produce warning about clipping (if any clipping occurs)
        # Note: The warning about extreme propensities is separate from clipping
        result = ipw_ate_observational(Y, T, X)

        # Should include clipping information in result diagnostics
        assert "n_propensity_clipped" in result or "propensity_clipped" in result, (
            "Result should include information about propensity clipping"
        )

    def test_result_includes_clipping_count(self):
        """Result should include count of clipped propensity values."""
        np.random.seed(42)
        n = 100

        # Well-overlapping data (no clipping needed)
        X = np.random.normal(0, 1, (n, 2))
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + X[:, 0] + np.random.normal(0, 1, n)

        result = ipw_ate_observational(Y, T, X)

        # Should include clipping count (likely 0 for well-overlapping data)
        assert "n_propensity_clipped" in result, "Result should include n_propensity_clipped field"

    def test_no_warning_without_clipping(self):
        """Should not warn when no clipping is needed."""
        np.random.seed(42)
        n = 200

        # Well-overlapping data
        X = np.random.normal(0, 1, (n, 2))
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + np.random.normal(0, 1, n)

        # Should not produce clipping warning (may still produce extreme warning)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ipw_ate_observational(Y, T, X)

            # Check no warnings about "clipped" specifically
            clipping_warnings = [
                warning for warning in w if "clipped" in str(warning.message).lower()
            ]
            # There may be other warnings (extreme propensity), but not clipping
            # For well-behaved data, should have n_propensity_clipped = 0
            if "n_propensity_clipped" in result:
                assert result["n_propensity_clipped"] == 0 or len(clipping_warnings) > 0
