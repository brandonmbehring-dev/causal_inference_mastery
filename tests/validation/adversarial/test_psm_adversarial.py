"""
Adversarial tests for PSM estimator (Layer 2).

These tests document edge cases discovered during Session 2 implementation.

Categories:
1. Extreme propensity scores (near 0 or 1)
2. No common support (complete separation)
3. Perfect separation in outcomes
4. Insufficient controls for matching
5. All propensities identical (randomized)
6. Caliper too restrictive (no matches)
7. High-dimensional curse (p > n)
8. Extreme imbalance (n_treated << n_control)
9. Tied propensity scores (discrete covariates)
10. Outliers in outcomes
11. Outliers in covariates
12. Zero variance in covariates

Status: Discovered during Session 2 implementation
"""

import numpy as np
import pytest
from src.causal_inference.psm import psm_ate


class TestPSMExtremePropensities:
    """Test PSM with extreme propensity scores."""

    def test_near_zero_propensity(self):
        """
        Propensity very close to 0 for some units.

        Expected:
        - Should compute estimate (propensity clamped to eps=1e-10)
        - SE > 0 and finite
        - Warns on perfect separation (>10% units with extreme propensity)
        """
        np.random.seed(42)
        n = 100

        # Treated units have very low propensity (should have been control)
        # Need ≥10 extreme units to trigger 10% warning threshold
        # Strategy: Most treated have high X, control has low X
        # Then some treated with low X will have propensity near 0
        treatment = np.array([1] * 35 + [1] * 15 + [0] * 50)
        covariates = np.concatenate(
            [
                np.random.normal(5, 1, (35, 2)),  # Normal treated (high X)
                np.random.normal(
                    -10, 1, (15, 2)
                ),  # Treated with control-like X (near 0 propensity) - 15%
                np.random.normal(-10, 1, (50, 2)),  # Control (low X)
            ]
        )
        outcomes = 2.0 * treatment + 0.5 * covariates[:, 0] + np.random.normal(0, 1, n)

        # Should succeed with warning
        with pytest.warns(RuntimeWarning, match="perfect separation"):
            result = psm_ate(outcomes, treatment, covariates, M=1, caliper=0.5)

        assert np.isfinite(result["estimate"])
        assert result["se"] > 0
        assert np.isfinite(result["se"])

    def test_near_one_propensity(self):
        """
        Propensity very close to 1 for some units.

        Expected:
        - Should compute estimate (propensity clamped to 1-eps=0.9999999999)
        - SE > 0 and finite
        - Warns on perfect separation
        """
        np.random.seed(123)
        n = 100

        # Need ≥10 extreme units to trigger 10% warning threshold
        # Strategy: Most treated have high X, control has low X
        # Then some control with high X will have propensity near 1
        treatment = np.array([1] * 50 + [0] * 35 + [0] * 15)
        covariates = np.concatenate(
            [
                np.random.normal(10, 1, (50, 2)),  # Treated (high X)
                np.random.normal(-5, 1, (35, 2)),  # Normal control (low X)
                np.random.normal(
                    10, 1, (15, 2)
                ),  # Control with treated-like X (near 1 propensity) - 15%
            ]
        )
        outcomes = 2.0 * treatment + 0.5 * covariates[:, 0] + np.random.normal(0, 1, n)

        with pytest.warns(RuntimeWarning, match="perfect separation"):
            result = psm_ate(outcomes, treatment, covariates, M=1, caliper=0.5)

        assert np.isfinite(result["estimate"])
        assert result["se"] > 0


class TestPSMCommonSupport:
    """Test PSM with limited/no common support."""

    def test_complete_separation(self):
        """
        Complete separation: treated/control have no overlap in covariates.

        Expected:
        - Should fail with "No common support" error (when min_overlap enforced strictly)
        - OR warn and proceed with small overlap region
        """
        np.random.seed(456)

        # Complete separation
        treatment = np.array([1] * 50 + [0] * 50)
        covariates = np.concatenate(
            [
                np.random.normal(5, 1, (50, 1)),  # Treated: X ~ N(5, 1)
                np.random.normal(-5, 1, (50, 1)),  # Control: X ~ N(-5, 1)
            ]
        )
        outcomes = 2.0 * treatment + np.random.normal(0, 1, 100)

        # With current min_overlap=0.001, this should pass
        # Use large caliper to allow matches despite separation
        result = psm_ate(outcomes, treatment, covariates, M=1, caliper=np.inf)

        # Most units outside support
        assert result["convergence_status"]["n_outside_support"] > 80

    def test_no_matches_strict_caliper(self):
        """
        Caliper too restrictive → no matches found.

        Expected:
        - Should fail with "No matches found" error
        """
        np.random.seed(789)

        # Use complete separation to guarantee propensities far apart
        treatment = np.array([1] * 50 + [0] * 50)
        covariates = np.concatenate(
            [
                np.random.normal(10, 0.5, (50, 1)),  # Treated: X ~ N(10, 0.5)
                np.random.normal(-10, 0.5, (50, 1)),  # Control: X ~ N(-10, 0.5)
            ]
        )
        outcomes = 2.0 * treatment + np.random.normal(0, 1, 100)

        # With complete separation, propensities are ~0 and ~1 (distance ~1)
        # Caliper 0.01 guarantees no matches
        with pytest.raises(ValueError, match="No matches found"):
            psm_ate(outcomes, treatment, covariates, M=1, caliper=0.01)


class TestPSMMatchingEdgeCases:
    """Test PSM matching edge cases."""

    def test_insufficient_controls_without_replacement(self):
        """
        M=5 but only 3 controls → should fail.

        Expected:
        - Should fail with "Insufficient controls" error
        """
        np.random.seed(321)

        treatment = np.array([1] * 10 + [0] * 3)
        covariates = np.random.normal(0, 1, (13, 2))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, 13)

        with pytest.raises(ValueError, match="Insufficient controls"):
            psm_ate(outcomes, treatment, covariates, M=5, with_replacement=False)

    def test_with_replacement_matches(self):
        """
        M:1 matching with replacement → controls can be reused.

        Expected:
        - Should succeed
        - n_matched = n_treated (all treated units matched)
        - Some controls used multiple times
        """
        np.random.seed(654)
        n = 60

        treatment = np.array([1] * 40 + [0] * 20)  # More treated than control
        covariates = np.random.normal(0, 1, (n, 2))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=2, with_replacement=True)

        # All treated matched (with replacement)
        assert result["n_matched"] == 40

    def test_constant_propensity(self):
        """
        All propensities identical (randomized experiment).

        Expected:
        - Should succeed
        - All units within common support
        - Most units matched
        """
        np.random.seed(987)
        n = 80

        # Constant covariate → constant propensity ≈ 0.5
        treatment = np.random.binomial(1, 0.5, n).astype(bool)
        covariates = np.ones((n, 1))  # Constant X
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=1)

        # All units have same propensity
        propensity = result["propensity_scores"]
        assert np.std(propensity) < 0.05  # Nearly constant

        # Most units matched (ties handled randomly)
        n_treated = np.sum(treatment)
        assert result["n_matched"] >= 0.7 * n_treated


class TestPSMHighDimensional:
    """Test PSM with many covariates."""

    def test_curse_of_dimensionality(self):
        """
        High-dimensional: p=20, n=50 (p/n=0.4).

        Expected:
        - Propensity estimation may fail to converge
        - OR succeeds but with poor matches (high SE)
        - Warns on non-convergence
        """
        np.random.seed(147)
        n = 50
        p = 20

        treatment = np.random.binomial(1, 0.5, n).astype(bool)
        covariates = np.random.normal(0, 1, (n, p))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        # May not converge
        result = psm_ate(outcomes, treatment, covariates, M=1)

        # Check if converged
        if not result["convergence_status"]["propensity_converged"]:
            # Expected - p too large for n
            pass
        else:
            # If converged, SE should be large (poor estimation)
            assert result["se"] > 0.5


class TestPSMExtremeImbalance:
    """Test PSM with extreme treatment imbalance."""

    def test_single_treated_unit(self):
        """
        Only 1 treated unit.

        Expected:
        - Should succeed
        - n_matched = 1
        - SE very large (single observation)
        """
        np.random.seed(258)
        n = 100

        treatment = np.array([1] + [0] * 99)
        covariates = np.random.normal(0, 1, (n, 2))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=3)

        assert result["n_matched"] == 1
        assert result["se"] > 1.0  # Very large SE

    def test_extreme_imbalance(self):
        """
        Extreme imbalance: 5 treated, 95 control.

        Expected:
        - Should succeed
        - All 5 treated matched
        - SE moderate
        """
        np.random.seed(369)
        n = 100

        treatment = np.array([1] * 5 + [0] * 95)
        covariates = np.random.normal(0, 1, (n, 2))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=5)

        assert result["n_matched"] == 5
        assert result["se"] > 0


class TestPSMOutliersAndVariance:
    """Test PSM with outliers and variance issues."""

    def test_extreme_outcome_outlier(self):
        """
        One unit has extreme outcome (1000x larger).

        Expected:
        - Should succeed
        - ATE may be biased if outlier matched
        - SE very large
        """
        np.random.seed(741)
        n = 100

        treatment = np.random.binomial(1, 0.5, n).astype(bool)
        covariates = np.random.normal(0, 1, (n, 2))
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)
        outcomes[0] = 1000.0  # Extreme outlier

        result = psm_ate(outcomes, treatment, covariates, M=1)

        # Should compute (may be biased)
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0

    def test_zero_variance_covariate(self):
        """
        One covariate has zero variance (constant).

        Expected:
        - Should succeed
        - Logistic regression handles constant covariate
        """
        np.random.seed(852)
        n = 100

        treatment = np.random.binomial(1, 0.5, n).astype(bool)
        covariates = np.column_stack(
            [
                np.random.normal(0, 1, n),
                np.ones(n),  # Zero variance
            ]
        )
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=1)

        assert np.isfinite(result["estimate"])
        assert result["se"] > 0


class TestPSMTiedPropensities:
    """Test PSM with many tied propensity scores."""

    def test_discrete_covariates_many_ties(self):
        """
        Discrete covariates → many identical propensity scores.

        Expected:
        - Should succeed
        - Tie-breaking handled (random selection among ties)
        - Most units matched
        """
        np.random.seed(963)
        n = 120

        # Binary covariates → 4 possible propensity values
        treatment = np.random.binomial(1, 0.5, n).astype(bool)
        covariates = np.column_stack(
            [
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n),
            ]
        ).astype(float)
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = psm_ate(outcomes, treatment, covariates, M=1)

        # Should succeed
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0

        # Most units matched
        n_treated = np.sum(treatment)
        assert result["n_matched"] >= 0.7 * n_treated
