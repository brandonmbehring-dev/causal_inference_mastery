"""
Adversarial tests for ipw_ate estimator.

IPW-specific edge cases:
1. Propensities near 0 or 1 (extreme weights)
2. Propensity exactly 0.5 (no variation)
3. One unit with propensity 0.001, rest normal
4. Extreme weight instability
5. Propensity perfectly separates treated/control
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators_ipw import ipw_ate


class TestIPWATEExtremePropensities:
    """Test ipw_ate with extreme propensity scores."""

    def test_near_zero_propensity(self):
        """Propensity very close to 0 (but not exactly 0)."""
        outcomes = np.array([10.0] + [2.0] * 99)
        treatment = np.array([1] + [1] * 49 + [0] * 50)
        propensity = np.array([0.001] + [0.5] * 99)  # Near-zero for first unit

        result = ipw_ate(outcomes, treatment, propensity)

        # Should compute estimate but SE may be large
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0

    def test_near_one_propensity(self):
        """Propensity very close to 1.

        Note: All treated have Y=2.0, all control have Y=0.0 (zero variance within groups).
        SE=0 is mathematically correct in this case. With weight normalization, extreme
        propensity (0.999) doesn't cause numerical issues.
        """
        outcomes = np.array([2.0] * 50 + [0.0] + [0.0] * 49)
        treatment = np.array([1] * 50 + [0] + [0] * 49)
        propensity = np.array([0.5] * 50 + [0.999] + [0.5] * 49)  # Near-one for control unit

        result = ipw_ate(outcomes, treatment, propensity)

        assert np.isfinite(result["estimate"])
        assert np.isclose(result["estimate"], 2.0)  # True ATE = 2.0
        # SE=0 is correct (zero variance within groups)
        assert result["se"] < 0.01  # Expect near-zero SE
        # Weight normalization prevents numerical issues
        assert np.isfinite(result["se"])

    def test_extreme_weight_variability(self):
        """Mix of propensities from 0.01 to 0.99.

        Note: With weight normalization, SE is reduced compared to raw IPW.
        Threshold relaxed to 0.2 to account for normalization reducing variance.
        """
        np.random.seed(42)
        n = 100
        propensity = np.linspace(0.01, 0.99, n)
        treatment = (np.random.uniform(0, 1, n) < propensity).astype(float)
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        result = ipw_ate(outcomes, treatment, propensity)

        # Should compute estimate
        assert np.isfinite(result["estimate"])
        # SE should be moderate (weight normalization reduces variance)
        assert result["se"] > 0.2
        assert np.isfinite(result["se"])


class TestIPWATEConstantPropensity:
    """Test ipw_ate with constant propensity."""

    def test_all_propensities_equal(self):
        """All propensities = 0.5 (randomized experiment)."""
        np.random.seed(42)
        n = 100
        propensity = np.ones(n) * 0.5
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = np.where(treatment == 1,
                           np.random.normal(2.0, 1.0, n),
                           np.random.normal(0.0, 1.0, n))

        result = ipw_ate(outcomes, treatment, propensity)

        # Should match simple ATE (since weights all equal)
        assert 1.5 < result["estimate"] < 2.5
        assert result["se"] < 0.3


class TestIPWATEOutliers:
    """Test ipw_ate with extreme outcome values."""

    def test_extreme_outcome_with_low_propensity(self):
        """Extreme outcome for unit with propensity=0.01."""
        outcomes = np.array([10000.0] + [2.0] * 99)
        treatment = np.array([1] + [1] * 49 + [0] * 50)
        propensity = np.array([0.01] + [0.5] * 99)

        result = ipw_ate(outcomes, treatment, propensity)

        # Extreme weight (1/0.01=100) on extreme outcome → huge estimate
        assert result["estimate"] > 100
        assert np.isfinite(result["estimate"])

    def test_perfect_balance_despite_varying_propensity(self):
        """Varying propensities but perfectly balanced outcomes.

        Fixed test data: Ensure treated always have Y=2.0, control always have Y=0.0.
        """
        # Create data where treated have Y=2.0, control have Y=0.0
        n = 100
        treatment = np.array([1] * 50 + [0] * 50)  # First 50 treated, last 50 control
        outcomes = np.where(treatment == 1, 2.0, 0.0)  # Perfect separation by treatment
        propensity = np.tile([0.1, 0.3, 0.5, 0.7, 0.9], 20)  # Varying propensities

        result = ipw_ate(outcomes, treatment, propensity)

        # ATE should be 2.0
        assert np.isclose(result["estimate"], 2.0, rtol=0.1)
        # SE should be near zero (no variance within groups)
        assert result["se"] < 0.01


class TestIPWATEPerfectSeparation:
    """Test ipw_ate with perfect propensity separation."""

    def test_propensity_perfectly_predicts_treatment(self):
        """
        Propensities near 0 for control, near 1 for treated (perfect separation).

        This is extreme but can occur in observational studies when treatment
        is nearly deterministic given X.
        """
        n = 100
        # Treated have propensity near 1, control have propensity near 0
        treatment = np.array([1] * 50 + [0] * 50)
        propensity = np.concatenate([
            np.ones(50) * 0.99,   # Treated
            np.ones(50) * 0.01    # Control
        ])
        outcomes = np.where(treatment == 1,
                           np.random.normal(2.0, 1.0, n),
                           np.random.normal(0.0, 1.0, n))
        np.random.seed(42)

        result = ipw_ate(outcomes, treatment, propensity)

        # Should compute estimate despite extreme separation
        assert np.isfinite(result["estimate"])
        assert result["estimate"] > 0  # True ATE = 2.0
        # SE should be positive and finite (weight normalization reduces variance)
        assert result["se"] > 0
        assert np.isfinite(result["se"])

    def test_all_identical_propensities_by_group(self):
        """
        All treated have same propensity, all control have same propensity.

        Treated: P(T=1|X) = 0.8
        Control: P(T=1|X) = 0.2
        """
        treatment = np.array([1] * 50 + [0] * 50)
        propensity = np.array([0.8] * 50 + [0.2] * 50)
        outcomes = np.array([2.0] * 50 + [0.0] * 50)  # Perfect separation by treatment

        result = ipw_ate(outcomes, treatment, propensity)

        # ATE = 2.0
        assert np.isclose(result["estimate"], 2.0)
        # SE should be near zero (no variance within groups)
        assert result["se"] < 0.01
