"""Test RCT estimators with hand-calculated known answers.

These tests use simple data where we can manually calculate the expected
results. This validates that our implementation is mathematically correct.

Following annuity_forecasting pattern: every metric gets known-answer validation.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate


class TestSimpleATEKnownAnswers:
    """Test simple_ate with hand-calculated expected values."""

    def test_simple_ate_basic_case(self):
        """
        Test ATE calculation with simplest possible data.

        Data:
        - Treatment: [1, 1, 0, 0]
        - Outcomes:  [7, 5, 3, 1]

        Hand calculation:
        - Treated mean: (7.0 + 5.0) / 2 = 6.0
        - Control mean: (3.0 + 1.0) / 2 = 2.0
        - ATE: 6.0 - 2.0 = 4.0

        Expected:
        - estimate = 4.0
        - n_treated = 2
        - n_control = 2
        """
        # Inline deterministic data matching docstring (known-answer test)
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        # Check point estimate
        assert np.isclose(result["estimate"], 4.0), f"Expected ATE=4.0, got {result['estimate']}"

        # Check sample sizes
        assert result["n_treated"] == 2
        assert result["n_control"] == 2

    def test_variance_calculation(self, balanced_rct_data):
        """
        Test standard error calculation with known variance.

        Data:
        - Treatment: [1, 1, 1, 0, 0, 0]
        - Outcomes:  [3, 5, 7, 1, 2, 3]

        Hand calculation:
        - Treated: mean=5.0, var=4.0, n=3
        - Control: mean=2.0, var=1.0, n=3
        - ATE: 5.0 - 2.0 = 3.0
        - Var(ATE): 4.0/3 + 1.0/3 = 5/3
        - SE(ATE): sqrt(5/3) ≈ 1.291
        """
        result = simple_ate(
            outcomes=balanced_rct_data["outcomes"], treatment=balanced_rct_data["treatment"]
        )

        expected_se = balanced_rct_data["expected_se"]

        assert np.isclose(result["se"], expected_se, rtol=1e-6), (
            f"Expected SE={expected_se:.6f}, got {result['se']:.6f}"
        )

    def test_confidence_interval_construction(self, simple_rct_data, alpha_standard):
        """
        Test 95% confidence interval calculation.

        Using simple_rct_data:
        - ATE = 4.0
        - SE = (to be calculated from data)
        - n=4 (2 treated, 2 control), df=2
        - 95% CI: ATE ± t_0.025,df=2 * SE (uses t-distribution, not z)

        Validates that CI is correctly constructed using t-distribution.
        """
        from scipy import stats

        result = simple_ate(
            outcomes=simple_rct_data["outcomes"],
            treatment=simple_rct_data["treatment"],
            alpha=alpha_standard,
        )

        # CI should be symmetric around estimate
        ci_lower = result["ci_lower"]
        ci_upper = result["ci_upper"]
        estimate = result["estimate"]
        se = result["se"]

        # Check symmetry
        assert np.isclose(estimate - ci_lower, ci_upper - estimate, rtol=1e-6)

        # Calculate degrees of freedom (Satterthwaite)
        # For n1=2, n0=2: df ≈ 2 (conservative)
        n1 = np.sum(simple_rct_data["treatment"] == 1)
        n0 = np.sum(simple_rct_data["treatment"] == 0)
        df = n1 + n0 - 2  # Conservative df for small sample

        # Get t critical value for 95% CI
        t_crit = stats.t.ppf(1 - alpha_standard / 2, df)

        # Check width (should be 2 * t_crit * SE, not 1.96 * SE)
        expected_width = 2 * t_crit * se
        actual_width = ci_upper - ci_lower
        assert np.isclose(actual_width, expected_width, rtol=1e-4)

    def test_zero_effect_case(self):
        """
        Test when there's no treatment effect.

        Data designed so treated and control have same mean.
        """
        # Create data with zero effect
        outcomes = np.array([3.0, 5.0, 3.0, 5.0])
        treatment = np.array([1, 1, 0, 0])

        # Both groups have mean = 4.0
        # Expected ATE = 0.0

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        assert np.isclose(result["estimate"], 0.0, atol=1e-10), (
            f"Expected ATE=0.0 with no effect, got {result['estimate']}"
        )

    def test_negative_effect_case(self):
        """
        Test when treatment has negative effect.

        Ensures estimator handles negative effects correctly.
        """
        # Create data where treatment decreases outcome
        outcomes = np.array([2.0, 4.0, 6.0, 8.0])
        treatment = np.array([1, 1, 0, 0])

        # Treated mean: 3.0, Control mean: 7.0
        # Expected ATE: 3.0 - 7.0 = -4.0

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        assert np.isclose(result["estimate"], -4.0), f"Expected ATE=-4.0, got {result['estimate']}"


class TestSimpleATEEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_unit_per_group(self):
        """
        Test with minimal data (n=1 per group).

        Should still work mathematically but SE will be large.
        """
        outcomes = np.array([5.0, 3.0])
        treatment = np.array([1, 0])

        # Expected ATE: 5.0 - 3.0 = 2.0

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        assert np.isclose(result["estimate"], 2.0)
        assert result["n_treated"] == 1
        assert result["n_control"] == 1

    def test_unbalanced_groups(self):
        """
        Test with unbalanced group sizes (70/30 split).

        Estimator should still be unbiased.
        """
        # 7 treated, 3 control
        outcomes = np.array([5, 6, 7, 8, 9, 10, 11, 2, 3, 4])
        treatment = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

        # Treated mean: 8.0, Control mean: 3.0
        # Expected ATE: 5.0

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        assert np.isclose(result["estimate"], 5.0)
        assert result["n_treated"] == 7
        assert result["n_control"] == 3

    def test_large_variance_case(self):
        """
        Test with high-variance outcomes.

        Should handle large variance correctly in SE calculation.
        """
        # High variance outcomes
        np.random.seed(42)
        outcomes = np.random.normal(loc=5, scale=100, size=100)
        treatment = np.random.binomial(n=1, p=0.5, size=100)

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        # SE should be large due to high variance
        assert result["se"] > 1.0, "SE should be large with high variance data"


# Marker for pytest to identify these as known-answer tests
pytestmark = pytest.mark.unit
