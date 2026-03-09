"""TDD Step 1: Tests for perfect separation detection.

These tests verify that propensity estimation detects perfect separation
(where treatment is perfectly predicted by covariates) and raises an error.

Issue: sklearn's LogisticRegression may silently return propensity = 0/1
when there's perfect separation, leading to infinite IPW weights.

Written BEFORE fixes per TDD protocol.
"""

import numpy as np
import pytest

from src.causal_inference.observational.ipw import ipw_ate_observational
from src.causal_inference.observational.propensity import estimate_propensity


class TestPerfectSeparation:
    """Perfect separation in propensity model should raise error."""

    def test_detects_perfect_separation(self):
        """Should raise ValueError when propensity hits exactly 0 or 1.

        Perfect separation: Treatment can be perfectly predicted by X.
        This makes IPW weights infinite (1/0 or 1/1-1).
        """
        np.random.seed(42)
        n = 100

        # Create perfectly separable data
        # X[:,0] < 0 → T=0; X[:,0] >= 0 → T=1
        X = np.vstack(
            [
                np.random.uniform(-2, -0.1, (50, 2)),  # All X[:,0] < 0
                np.random.uniform(0.1, 2, (50, 2)),  # All X[:,0] > 0
            ]
        )
        T = np.array([0] * 50 + [1] * 50).astype(float)  # Perfectly separable
        Y = 2.0 * T + np.random.normal(0, 1, n)

        # Should raise ValueError mentioning perfect separation or extreme propensity
        with pytest.raises(ValueError, match="(?i)perfect separation|propensity.*0 or 1|extreme"):
            ipw_ate_observational(Y, T, X)

    def test_detects_near_separation(self):
        """Should warn when propensity is extreme (< 0.01 or > 0.99).

        Near-perfect separation: Some propensities are extreme but not exactly 0/1.
        This should trigger a warning about potential positivity violation.
        """
        np.random.seed(42)
        n = 100

        # Create near-perfectly separable data
        # Groups barely overlap
        X = np.vstack(
            [
                np.random.normal(-1.5, 0.2, (50, 2)),  # Control group, tight cluster
                np.random.normal(1.5, 0.2, (50, 2)),  # Treatment group, tight cluster
            ]
        )
        T = np.array([0] * 50 + [1] * 50).astype(float)
        Y = 2.0 * T + np.random.normal(0, 1, n)

        # Should warn about extreme propensities / positivity violation
        with pytest.warns(UserWarning, match="extreme propensity|positivity|overlap"):
            ipw_ate_observational(Y, T, X)

    def test_propensity_estimation_detects_separation(self):
        """Direct test: estimate_propensity should detect perfect separation."""
        np.random.seed(42)

        # Perfectly separable data with variation in both columns
        X = np.array(
            [
                [-1.0, 0.1],
                [-2.0, 0.2],
                [-3.0, 0.3],  # Control group
                [1.0, 0.4],
                [2.0, 0.5],
                [3.0, 0.6],  # Treatment group (separable by X[:,0])
            ]
        )
        T = np.array([0, 0, 0, 1, 1, 1]).astype(float)

        # Should raise ValueError about perfect separation or extreme propensity
        with pytest.raises(ValueError, match="(?i)perfect separation|propensity.*0 or 1|extreme"):
            estimate_propensity(T, X)


class TestPerfectSeparationEdgeCases:
    """Edge cases for separation detection."""

    def test_barely_overlapping_passes(self):
        """Data with some overlap (not perfect separation) should work."""
        np.random.seed(42)
        n = 100

        # Some overlap between groups
        X = np.vstack(
            [
                np.random.normal(-0.5, 0.5, (50, 2)),
                np.random.normal(0.5, 0.5, (50, 2)),
            ]
        )
        T = np.array([0] * 50 + [1] * 50).astype(float)
        Y = 2.0 * T + np.random.normal(0, 1, n)

        # Should NOT raise - there is overlap
        result = ipw_ate_observational(Y, T, X)

        # Should produce finite estimate
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

    def test_random_assignment_no_separation(self):
        """Random treatment assignment should never trigger separation warning."""
        np.random.seed(42)
        n = 100

        # Random assignment independent of X
        X = np.random.normal(0, 1, (n, 2))
        T = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + X[:, 0] + np.random.normal(0, 1, n)

        # Should work without warnings
        result = ipw_ate_observational(Y, T, X)

        assert np.isfinite(result["estimate"])
