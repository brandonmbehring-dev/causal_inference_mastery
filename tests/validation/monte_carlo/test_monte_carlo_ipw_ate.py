"""
Monte Carlo validation for ipw_ate estimator.

Validates IPW handles non-constant propensity scores correctly.
"""

import numpy as np
from src.causal_inference.rct.estimators_ipw import ipw_ate
from tests.validation.monte_carlo.dgp_generators import dgp_ipw_rct
from tests.validation.utils import validate_monte_carlo_results


class TestIPWATEMonteCarlo:
    """Monte Carlo validation for ipw_ate."""

    def test_ipw_rct(self):
        """Validate ipw_ate with varying propensity scores (1000 runs)."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment, propensity = dgp_ipw_rct(
                n=100, true_ate=true_ate, random_state=seed
            )
            result = ipw_ate(outcomes, treatment, propensity)

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        # Validate
        validation = validate_monte_carlo_results(
            estimates, standard_errors, ci_lowers, ci_uppers, true_ate
        )

        assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
        assert validation["coverage_ok"], f"Coverage {validation['coverage']:.4f} outside [0.93, 0.97]"
        # IPW can have higher SE variability
        assert validation["se_accuracy"] < 0.20, f"SE accuracy {validation['se_accuracy']:.4f} exceeds 20%"


    def test_ipw_constant_propensity(self):
        """When propensity is constant, IPW should match simple ATE."""
        from src.causal_inference.rct.estimators import simple_ate

        n_runs = 500
        true_ate = 2.0

        differences = []

        for seed in range(n_runs):
            np.random.seed(seed)
            n = 100
            treatment = np.array([1] * 50 + [0] * 50)
            np.random.shuffle(treatment)
            outcomes = 2.0 * treatment + np.random.normal(0, 1, n)
            propensity = np.ones(n) * 0.5  # Constant propensity

            # IPW estimate
            result_ipw = ipw_ate(outcomes, treatment, propensity)

            # Simple ATE estimate
            result_simple = simple_ate(outcomes, treatment)

            # Should be very close
            differences.append(abs(result_ipw["estimate"] - result_simple["estimate"]))

        # Mean difference should be tiny
        mean_diff = np.mean(differences)
        assert mean_diff < 0.05, f"IPW differs from simple ATE by {mean_diff:.4f}"


    def test_ipw_propensity_variation(self):
        """Verify IPW handles propensity variation correctly."""
        n_runs = 500
        true_ate = 2.0

        estimates = []

        for seed in range(n_runs):
            outcomes, treatment, propensity = dgp_ipw_rct(
                n=200, true_ate=true_ate, random_state=seed
            )
            result = ipw_ate(outcomes, treatment, propensity)
            estimates.append(result["estimate"])

        # Despite varying propensities, estimates should center around true ATE
        mean_estimate = np.mean(estimates)
        assert abs(mean_estimate - true_ate) < 0.1, \
            f"Mean IPW estimate ({mean_estimate:.4f}) far from true ATE ({true_ate})"
