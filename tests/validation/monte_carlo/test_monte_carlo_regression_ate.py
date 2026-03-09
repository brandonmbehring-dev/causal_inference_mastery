"""
Monte Carlo validation for regression_adjusted_ate estimator.

Validates regression adjustment reduces variance when covariates predict outcome.
"""

import numpy as np
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate
from tests.validation.monte_carlo.dgp_generators import dgp_regression_rct
from tests.validation.utils import validate_monte_carlo_results


class TestRegressionATEMonteCarlo:
    """Monte Carlo validation for regression_adjusted_ate."""

    def test_regression_rct(self):
        """Validate regression_adjusted_ate (n=100, 1000 runs)."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment, X = dgp_regression_rct(
                n=100, true_ate=true_ate, covariate_effect=3.0, random_state=seed
            )
            result = regression_adjusted_ate(outcomes, treatment, X)

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        # Validate
        validation = validate_monte_carlo_results(
            estimates, standard_errors, ci_lowers, ci_uppers, true_ate
        )

        assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
        assert validation["coverage_ok"], (
            f"Coverage {validation['coverage']:.4f} outside [0.93, 0.97]"
        )
        assert validation["se_accuracy_ok"], (
            f"SE accuracy {validation['se_accuracy']:.4f} exceeds 10%"
        )

    def test_regression_variance_reduction(self):
        """Verify regression adjustment reduces variance vs simple estimator."""
        from src.causal_inference.rct.estimators import simple_ate

        n_runs = 500
        true_ate = 2.0

        regression_ses = []
        simple_ses = []

        for seed in range(n_runs):
            outcomes, treatment, X = dgp_regression_rct(
                n=100, true_ate=true_ate, covariate_effect=3.0, random_state=seed
            )

            # Regression-adjusted estimate
            result_regression = regression_adjusted_ate(outcomes, treatment, X)
            regression_ses.append(result_regression["se"])

            # Simple estimate (ignoring covariate)
            result_simple = simple_ate(outcomes, treatment)
            simple_ses.append(result_simple["se"])

        # Regression should have lower average SE (variance reduction)
        mean_regression_se = np.mean(regression_ses)
        mean_simple_se = np.mean(simple_ses)

        assert mean_regression_se < mean_simple_se, (
            f"Regression SE ({mean_regression_se:.4f}) not smaller than simple SE ({mean_simple_se:.4f})"
        )

    def test_r_squared_diagnostic(self):
        """Verify R² is high when covariate strongly predicts outcome."""
        n_runs = 100
        true_ate = 2.0

        r_squareds = []

        for seed in range(n_runs):
            outcomes, treatment, X = dgp_regression_rct(
                n=200, true_ate=true_ate, covariate_effect=3.0, random_state=seed
            )
            result = regression_adjusted_ate(outcomes, treatment, X)
            r_squareds.append(result["r_squared"])

        # R² should be high (covariate effect is 3.0, strong)
        mean_r_squared = np.mean(r_squareds)
        assert mean_r_squared > 0.7, f"Mean R² ({mean_r_squared:.4f}) too low"
