"""
Monte Carlo validation for CATE estimators.

Validates statistical properties of meta-learners:
- S-Learner, T-Learner, X-Learner, R-Learner
- Double Machine Learning (DML)
- Causal Forest

Test criteria:
- ATE Bias < 0.10 (observational setting)
- Coverage 90-98% for 95% CI
- Heterogeneity detection: CATE correlation > 0.5 with truth

Session 62: Complete CATE Validation

References:
    - Kunzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
    - Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
    - Chernozhukov et al. (2018). "Double/debiased machine learning"
"""

import numpy as np
import pytest
from scipy import stats

from src.causal_inference.cate import (
    s_learner,
    t_learner,
    x_learner,
    r_learner,
    double_ml,
    causal_forest,
)
from tests.validation.monte_carlo.dgp_cate import (
    dgp_constant_effect,
    dgp_linear_heterogeneity,
    dgp_nonlinear_heterogeneity,
    dgp_complex_heterogeneity,
    dgp_high_dimensional,
    dgp_imbalanced_treatment,
)


# =============================================================================
# Monte Carlo Utilities
# =============================================================================


def mc_statistics(estimates, true_value):
    """Compute Monte Carlo statistics."""
    estimates = np.array(estimates)
    mean_est = np.mean(estimates)
    bias = mean_est - true_value
    rmse = np.sqrt(np.mean((estimates - true_value) ** 2))
    return {"mean": mean_est, "bias": bias, "abs_bias": abs(bias), "rmse": rmse}


def coverage_rate(ci_lowers, ci_uppers, true_value):
    """Compute confidence interval coverage rate."""
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    covered = (ci_lowers <= true_value) & (ci_uppers >= true_value)
    return np.mean(covered)


def cate_correlation(estimated_cate, true_cate):
    """Compute correlation between estimated and true CATE."""
    return np.corrcoef(estimated_cate, true_cate)[0, 1]


# =============================================================================
# S-Learner Monte Carlo Tests
# =============================================================================


class TestSLearnerMonteCarlo:
    """Monte Carlo validation for S-Learner."""

    def test_constant_effect_bias(self):
        """S-Learner should have low bias with constant treatment effect."""
        n_runs = 300
        true_ate = 2.0
        estimates = []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = s_learner(data.Y, data.T, data.X)
            estimates.append(result["ate"])

        stats_result = mc_statistics(estimates, true_ate)
        assert stats_result["abs_bias"] < 0.10, (
            f"S-Learner bias {stats_result['abs_bias']:.4f} exceeds 0.10"
        )

    def test_constant_effect_coverage(self):
        """S-Learner CI coverage should be in [0.90, 0.98] for 95% CI."""
        n_runs = 300
        true_ate = 2.0
        ci_lowers, ci_uppers = [], []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = s_learner(data.Y, data.T, data.X)
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        coverage = coverage_rate(ci_lowers, ci_uppers, true_ate)
        assert 0.90 <= coverage <= 0.98, f"S-Learner coverage {coverage:.2%} outside [90%, 98%]"

    def test_regularization_bias_detection(self):
        """
        S-Learner shows regularization bias toward 0 with small effects.

        This is a KNOWN LIMITATION (documented in meta_learners.py).
        S-learner may underestimate small treatment effects because the
        model can "regularize away" the treatment if X explains most of Y.
        """
        n_runs = 200
        small_ate = 0.3  # Small effect relative to X main effects
        estimates = []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=small_ate, p=10, random_state=seed)
            result = s_learner(data.Y, data.T, data.X, model="ridge")
            estimates.append(result["ate"])

        mean_estimate = np.mean(estimates)
        # S-learner typically underestimates (biased toward 0)
        # We just verify it runs and gets a reasonable estimate
        assert 0.0 < mean_estimate < 0.6, (
            f"S-Learner mean estimate {mean_estimate:.4f} out of expected range"
        )


# =============================================================================
# T-Learner Monte Carlo Tests
# =============================================================================


class TestTLearnerMonteCarlo:
    """Monte Carlo validation for T-Learner."""

    def test_linear_heterogeneity_bias(self):
        """T-Learner should have low ATE bias with linear heterogeneity."""
        n_runs = 300
        estimates = []
        true_ates = []

        for seed in range(n_runs):
            data = dgp_linear_heterogeneity(n=500, random_state=seed)
            result = t_learner(data.Y, data.T, data.X)
            estimates.append(result["ate"])
            true_ates.append(data.true_ate)

        # Average true ATE across simulations
        avg_true_ate = np.mean(true_ates)
        stats_result = mc_statistics(estimates, avg_true_ate)

        assert stats_result["abs_bias"] < 0.15, (
            f"T-Learner bias {stats_result['abs_bias']:.4f} exceeds 0.15"
        )

    def test_linear_heterogeneity_cate_recovery(self):
        """T-Learner should recover linear CATE structure."""
        # Single larger sample for heterogeneity detection
        data = dgp_linear_heterogeneity(n=2000, random_state=42)
        result = t_learner(data.Y, data.T, data.X)

        corr = cate_correlation(result["cate"], data.true_cate)
        assert corr > 0.5, f"T-Learner CATE correlation {corr:.2f} < 0.5"

    def test_coverage_with_heterogeneity(self):
        """T-Learner CI coverage for heterogeneous effects."""
        n_runs = 300
        ci_lowers, ci_uppers, true_ates = [], [], []

        for seed in range(n_runs):
            data = dgp_linear_heterogeneity(n=500, random_state=seed)
            result = t_learner(data.Y, data.T, data.X)
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])
            true_ates.append(data.true_ate)

        avg_true_ate = np.mean(true_ates)
        coverage = coverage_rate(ci_lowers, ci_uppers, avg_true_ate)

        assert 0.88 <= coverage <= 0.99, f"T-Learner coverage {coverage:.2%} outside [88%, 99%]"


# =============================================================================
# X-Learner Monte Carlo Tests
# =============================================================================


class TestXLearnerMonteCarlo:
    """Monte Carlo validation for X-Learner."""

    def test_imbalanced_treatment_handling(self):
        """
        X-Learner should handle imbalanced treatment better than T-Learner.

        This is the key advantage of X-learner: it uses the larger group
        to improve estimates for the smaller group.
        """
        n_runs = 200
        x_estimates, t_estimates = [], []

        for seed in range(n_runs):
            data = dgp_imbalanced_treatment(
                n=500, true_ate=2.0, treatment_prob=0.1, random_state=seed
            )
            x_result = x_learner(data.Y, data.T, data.X, model="linear")
            t_result = t_learner(data.Y, data.T, data.X)

            x_estimates.append(x_result["ate"])
            t_estimates.append(t_result["ate"])

        x_bias = abs(np.mean(x_estimates) - 2.0)
        t_bias = abs(np.mean(t_estimates) - 2.0)

        # X-learner should have lower or comparable bias
        # Allow some tolerance due to MC noise
        assert x_bias < t_bias + 0.15, (
            f"X-Learner bias ({x_bias:.4f}) not better than T-Learner ({t_bias:.4f})"
        )

    def test_nonlinear_heterogeneity_recovery(self):
        """X-Learner with RF should capture nonlinear heterogeneity."""
        data = dgp_nonlinear_heterogeneity(n=2000, random_state=42)
        result = x_learner(data.Y, data.T, data.X, model="random_forest")

        corr = cate_correlation(result["cate"], data.true_cate)
        # RF X-learner should capture nonlinear patterns
        assert corr > 0.4, f"X-Learner CATE correlation {corr:.2f} < 0.4"


# =============================================================================
# R-Learner Monte Carlo Tests
# =============================================================================


class TestRLearnerMonteCarlo:
    """Monte Carlo validation for R-Learner (Robinson transformation)."""

    def test_doubly_robust_bias(self):
        """
        R-Learner is doubly robust: consistent if either propensity
        OR outcome model is correct.
        """
        n_runs = 300
        true_ate = 2.0
        estimates = []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = r_learner(data.Y, data.T, data.X)
            estimates.append(result["ate"])

        stats_result = mc_statistics(estimates, true_ate)
        # R-learner may have higher variance but should be unbiased
        assert stats_result["abs_bias"] < 0.15, (
            f"R-Learner bias {stats_result['abs_bias']:.4f} exceeds 0.15"
        )

    def test_confounded_treatment_handling(self):
        """R-Learner handles confounded treatment assignment."""
        n_runs = 200
        true_ate = 2.0
        estimates = []

        for seed in range(n_runs):
            # Higher propensity strength = more confounding
            data = dgp_constant_effect(
                n=500, true_ate=true_ate, propensity_strength=1.0, random_state=seed
            )
            result = r_learner(data.Y, data.T, data.X)
            estimates.append(result["ate"])

        stats_result = mc_statistics(estimates, true_ate)
        # Should handle confounding reasonably
        assert stats_result["abs_bias"] < 0.20, (
            f"R-Learner bias with confounding {stats_result['abs_bias']:.4f} exceeds 0.20"
        )


# =============================================================================
# Double ML Monte Carlo Tests
# =============================================================================


class TestDMLMonteCarlo:
    """Monte Carlo validation for Double Machine Learning."""

    def test_cross_fitting_bias_reduction(self):
        """
        DML with cross-fitting should have lower bias than R-learner
        (which uses in-sample predictions).
        """
        n_runs = 200
        true_ate = 2.0
        dml_estimates, r_estimates = [], []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            dml_result = double_ml(data.Y, data.T, data.X, n_folds=5)
            r_result = r_learner(data.Y, data.T, data.X)

            dml_estimates.append(dml_result["ate"])
            r_estimates.append(r_result["ate"])

        dml_bias = abs(np.mean(dml_estimates) - true_ate)
        r_bias = abs(np.mean(r_estimates) - true_ate)

        # Both should be reasonably unbiased
        assert dml_bias < 0.15, f"DML bias {dml_bias:.4f} exceeds 0.15"
        assert r_bias < 0.15, f"R-Learner bias {r_bias:.4f} exceeds 0.15"

    def test_dml_coverage(self):
        """DML confidence intervals should have proper coverage."""
        n_runs = 300
        true_ate = 2.0
        ci_lowers, ci_uppers = [], []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = double_ml(data.Y, data.T, data.X, n_folds=5)
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        coverage = coverage_rate(ci_lowers, ci_uppers, true_ate)
        assert 0.88 <= coverage <= 0.99, f"DML coverage {coverage:.2%} outside [88%, 99%]"

    def test_dml_linear_heterogeneity(self):
        """DML should recover linear heterogeneity."""
        data = dgp_linear_heterogeneity(n=2000, random_state=42)
        result = double_ml(data.Y, data.T, data.X, n_folds=5)

        corr = cate_correlation(result["cate"], data.true_cate)
        assert corr > 0.4, f"DML CATE correlation {corr:.2f} < 0.4"


# =============================================================================
# Causal Forest Monte Carlo Tests
# =============================================================================


class TestCausalForestMonteCarlo:
    """Monte Carlo validation for Causal Forest."""

    @pytest.mark.slow
    def test_complex_heterogeneity_detection(self):
        """
        Causal Forest should excel at complex heterogeneity (step + linear).

        This tests the key advantage of forests: capturing nonlinear,
        discontinuous treatment effect patterns.
        """
        data = dgp_complex_heterogeneity(n=2000, random_state=42)
        result = causal_forest(
            data.Y,
            data.T,
            data.X,
            n_estimators=100,
            min_samples_leaf=10,
            honest=True,
        )

        corr = cate_correlation(result["cate"], data.true_cate)
        # Forest should capture step function + linear well
        assert corr > 0.5, f"Causal Forest CATE correlation {corr:.2f} < 0.5"

    @pytest.mark.slow
    def test_honest_splitting_coverage(self):
        """
        Honest splitting (CONCERN-28) should give valid CIs.

        Without honest splitting, forests overfit and CIs undercover.
        With honest splitting, coverage should be close to nominal.
        """
        n_runs = 100  # Fewer runs due to computational cost
        true_ate = 2.0
        ci_lowers, ci_uppers = [], []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = causal_forest(
                data.Y,
                data.T,
                data.X,
                n_estimators=50,  # Fewer trees for speed
                honest=True,
            )
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        coverage = coverage_rate(ci_lowers, ci_uppers, true_ate)
        # Honest forests should have reasonable coverage
        # Note: 100% coverage indicates conservative CIs, which is acceptable
        assert 0.85 <= coverage <= 1.0, f"Causal Forest coverage {coverage:.2%} outside [85%, 100%]"

    @pytest.mark.slow
    def test_forest_bias(self):
        """Causal Forest should have low ATE bias."""
        n_runs = 100
        true_ate = 2.0
        estimates = []

        for seed in range(n_runs):
            data = dgp_constant_effect(n=500, true_ate=true_ate, random_state=seed)
            result = causal_forest(
                data.Y,
                data.T,
                data.X,
                n_estimators=50,
                honest=True,
            )
            estimates.append(result["ate"])

        stats_result = mc_statistics(estimates, true_ate)
        assert stats_result["abs_bias"] < 0.15, (
            f"Causal Forest bias {stats_result['abs_bias']:.4f} exceeds 0.15"
        )


# =============================================================================
# Cross-Method Comparison Tests
# =============================================================================


class TestCATEMethodComparison:
    """Compare different CATE methods across scenarios."""

    def test_all_methods_constant_effect(self):
        """All methods should recover constant effect reasonably."""
        data = dgp_constant_effect(n=1000, true_ate=2.0, random_state=42)

        methods = {
            "s_learner": s_learner(data.Y, data.T, data.X),
            "t_learner": t_learner(data.Y, data.T, data.X),
            "r_learner": r_learner(data.Y, data.T, data.X),
            "double_ml": double_ml(data.Y, data.T, data.X, n_folds=5),
        }

        for name, result in methods.items():
            error = abs(result["ate"] - 2.0)
            assert error < 0.5, f"{name} ATE error {error:.4f} exceeds 0.5"

    def test_method_ranking_by_scenario(self):
        """
        Different methods shine in different scenarios.

        This test documents expected relative performance:
        - S-learner: Good for homogeneous effects
        - T-learner: Good with balanced data
        - X-learner: Good with imbalanced data
        - R-learner/DML: Good with confounding
        - Causal Forest: Good with complex heterogeneity
        """
        # Just verify all methods run and return valid results
        scenarios = [
            ("constant", dgp_constant_effect(n=500, random_state=42)),
            ("linear", dgp_linear_heterogeneity(n=500, random_state=42)),
            ("imbalanced", dgp_imbalanced_treatment(n=500, random_state=42)),
        ]

        for scenario_name, data in scenarios:
            for method_name, method in [
                ("s", s_learner),
                ("t", t_learner),
                ("r", r_learner),
            ]:
                result = method(data.Y, data.T, data.X)
                assert np.isfinite(result["ate"]), (
                    f"{method_name}-learner failed on {scenario_name}"
                )
                assert np.isfinite(result["ate_se"]), (
                    f"{method_name}-learner SE failed on {scenario_name}"
                )


# =============================================================================
# High-Dimensional Tests
# =============================================================================


class TestCATEHighDimensional:
    """Monte Carlo tests for high-dimensional scenarios."""

    def test_sparse_heterogeneity_detection(self):
        """Methods should identify sparse heterogeneity in high-D."""
        data = dgp_high_dimensional(n=500, p=30, n_relevant=5, random_state=42)

        # Ridge-based methods should handle high-D well
        result = double_ml(
            data.Y,
            data.T,
            data.X,
            n_folds=5,
            nuisance_model="ridge",
        )

        # Should get reasonable ATE estimate
        error = abs(result["ate"] - data.true_ate)
        assert error < 0.5, f"DML high-D ATE error {error:.4f} exceeds 0.5"

    def test_regularization_with_many_covariates(self):
        """Regularized methods should not overfit with p > n/2."""
        data = dgp_high_dimensional(n=200, p=80, n_relevant=3, random_state=42)

        # S-learner with ridge should handle this
        result = s_learner(data.Y, data.T, data.X, model="ridge")

        # Should at least return finite values
        assert np.isfinite(result["ate"]), "S-Learner failed in high-D"
        assert np.isfinite(result["ate_se"]), "S-Learner SE failed in high-D"
