"""
Monte Carlo validation for stratified_ate estimator.

Validates stratified estimator reduces variance compared to simple estimator.
"""

import numpy as np
from src.causal_inference.rct.estimators_stratified import stratified_ate
from tests.validation.monte_carlo.dgp_generators import dgp_stratified_rct
from tests.validation.utils import validate_monte_carlo_results


class TestStratifiedATEMonteCarlo:
    """Monte Carlo validation for stratified_ate."""

    def test_stratified_rct(self):
        """Validate stratified_ate on stratified RCT (3 strata, 1000 runs)."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment, strata = dgp_stratified_rct(
                n_per_stratum=40, n_strata=3, true_ate=true_ate, random_state=seed
            )
            result = stratified_ate(outcomes, treatment, strata)

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
        assert validation["se_accuracy_ok"], f"SE accuracy {validation['se_accuracy']:.4f} exceeds 10%"


    def test_stratified_variance_reduction(self):
        """Verify stratified estimator has lower variance than simple estimator."""
        from src.causal_inference.rct.estimators import simple_ate

        n_runs = 500
        true_ate = 2.0

        stratified_ses = []
        simple_ses = []

        for seed in range(n_runs):
            outcomes, treatment, strata = dgp_stratified_rct(
                n_per_stratum=40, n_strata=3, true_ate=true_ate, random_state=seed
            )

            # Stratified estimate
            result_stratified = stratified_ate(outcomes, treatment, strata)
            stratified_ses.append(result_stratified["se"])

            # Simple estimate (ignoring strata)
            result_simple = simple_ate(outcomes, treatment)
            simple_ses.append(result_simple["se"])

        # Stratified should have lower average SE (variance reduction)
        mean_stratified_se = np.mean(stratified_ses)
        mean_simple_se = np.mean(simple_ses)

        assert mean_stratified_se < mean_simple_se, \
            f"Stratified SE ({mean_stratified_se:.4f}) not smaller than simple SE ({mean_simple_se:.4f})"
