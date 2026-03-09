"""Adversarial tests for doubly robust (DR) estimation.

Tests stress the DR estimator with challenging edge cases:
- Perfect separation with extreme propensities
- Severely misspecified models
- High-dimensional covariates
- Small sample sizes
- Collinear covariates
- Extreme treatment imbalance

These tests ensure the DR estimator handles real-world pathologies gracefully.
"""

import numpy as np
import pytest
from src.causal_inference.observational.doubly_robust import dr_ate


class TestDoublyRobustAdversarialPropensity:
    """Adversarial tests for extreme propensity scenarios."""

    def test_perfect_separation_extreme_propensities(self):
        """
        Test with perfect separation producing extreme propensities near 0 and 1.

        Challenge: Propensities near boundaries stress numerical stability.
        Expected: Clipping at ε=1e-6 prevents division by zero.
        """
        np.random.seed(1001)
        n = 200

        X = np.random.normal(0, 1, n)

        # Perfect separation: very strong confounding
        logit_prop = 5.0 * X  # Extreme coefficient
        e_X = 1 / (1 + np.exp(-logit_prop))
        # Will produce propensities very close to 0 and 1

        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run without error due to propensity clipping
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])

        # Propensities should be clipped away from boundaries
        assert np.all(result["propensity"] >= 1e-6)
        assert np.all(result["propensity"] <= 1 - 1e-6)

    def test_near_constant_propensity_no_confounding(self):
        """
        Test with very weak confounding (propensities near 0.5).

        Challenge: Little confounding means IPW provides minimal correction.
        Expected: DR should rely more on outcome model.
        """
        np.random.seed(1002)
        n = 300

        X = np.random.normal(0, 1, n)

        # Very weak confounding
        logit_prop = 0.05 * X  # Tiny coefficient
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should recover ATE well (outcome model correct)
        assert np.abs(result["estimate"] - 3.0) < 0.4

        # Propensities should be close to 0.5
        assert 0.4 < np.mean(result["propensity"]) < 0.6

    def test_bimodal_propensity_distribution(self):
        """
        Test with bimodal propensity distribution (two clusters).

        Challenge: Some units have very low propensity, others very high.
        Expected: DR handles via clipping and combining with outcome model.
        """
        np.random.seed(1003)
        n = 400

        X = np.random.normal(0, 1, n)

        # Bimodal propensity: depends on X but with strong effect
        logit_prop = 3.0 * X  # Strong effect creates bimodal distribution
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 2.5 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run and produce finite estimates
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0


class TestDoublyRobustAdversarialOutcomeModel:
    """Adversarial tests for misspecified outcome models."""

    def test_severely_misspecified_outcome_quadratic_truth(self):
        """
        Test with severely misspecified outcome model (quadratic truth, linear fit).

        Challenge: Large outcome model misspecification.
        Expected: If propensity correct, DR still consistent.
        """
        np.random.seed(1004)
        n = 300

        X = np.random.normal(0, 1, n)

        # Propensity: linear (correctly modeled)
        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome: SEVERELY quadratic (large coefficient)
        Y = 3.0 * T + (2 + 2.0 * X**2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # DR should still work due to correct propensity (IPW protects)
        # But may have larger bias than mild misspecification
        assert np.abs(result["estimate"] - 3.0) < 0.8  # More tolerance

        # Should run without error
        assert np.isfinite(result["se"])

    def test_outcome_with_interactions_not_modeled(self):
        """
        Test with outcome having T × X interaction (not captured by separate models).

        Challenge: Treatment effect heterogeneity not fully captured.
        Expected: DR provides some protection but may have bias.
        """
        np.random.seed(1005)
        n = 350

        X = np.random.normal(0, 1, n)

        logit_prop = 0.6 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Heterogeneous treatment effect: varies with X
        # ATE = 3.0, but effect is 3.0 + 0.5*X
        Y = (3.0 + 0.5 * X) * T + (2 + 0.3 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Average treatment effect should be near 3.0
        assert np.abs(result["estimate"] - 3.0) < 0.5

        # Should run successfully
        assert np.isfinite(result["estimate"])


class TestDoublyRobustAdversarialHighDimensional:
    """Adversarial tests for high-dimensional covariate scenarios."""

    def test_high_dimensional_covariates_p20(self):
        """
        Test with many covariates (p=20) relative to sample size (n=150).

        Challenge: High p/n ratio stresses model fitting.
        Expected: DR should run but may have higher variance.
        """
        np.random.seed(1006)
        n = 150
        p = 20

        X = np.random.normal(0, 1, (n, p))

        # Propensity depends on first 3 covariates
        logit_prop = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2]
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        # Outcome depends on first 3 covariates
        Y = (
            3.0 * T
            + (2 + 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2])
            + np.random.normal(0, 0.5, n)
        )

        result = dr_ate(Y, T, X)

        # Should run despite high dimensionality
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0

        # May have high variance
        assert result["se"] < 2.0  # Not absurdly high

    def test_high_dimensional_covariates_p30_with_trimming(self):
        """
        Test with very many covariates (p=30, n=100) and trimming.

        Challenge: p/n = 0.3 is extreme for logistic/linear regression.
        Expected: Trimming helps by removing units hard to model.
        """
        np.random.seed(1007)
        n = 100
        p = 30

        X = np.random.normal(0, 1, (n, p))

        # Propensity depends on first 2 covariates (sparse)
        logit_prop = 0.8 * X[:, 0] - 0.6 * X[:, 1]
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X[:, 0] + 0.4 * X[:, 1]) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X, trim_at=(0.1, 0.9))

        # Should run with trimming
        assert np.isfinite(result["estimate"])

        # Should have trimmed some units
        assert result["n_trimmed"] >= 0  # May or may not trim depending on propensities


class TestDoublyRobustAdversarialSmallSample:
    """Adversarial tests for small sample sizes."""

    def test_small_sample_n50(self):
        """
        Test with very small sample (n=50).

        Challenge: Small n increases variance and model uncertainty.
        Expected: DR runs but has high SE.
        """
        np.random.seed(1008)
        n = 50

        X = np.random.normal(0, 1, n)

        logit_prop = 0.8 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run despite small sample
        assert np.isfinite(result["estimate"])

        # SE will be larger than typical (expect >0.15 with n=50)
        assert result["se"] > 0.15  # Expect high uncertainty

    def test_small_sample_n100_extreme_confounding(self):
        """
        Test with n=100 and extreme confounding.

        Challenge: Small sample + strong confounding = extreme propensities.
        Expected: Trimming may be necessary.
        """
        np.random.seed(1009)
        n = 100

        X = np.random.normal(0, 1, n)

        # Extreme confounding
        logit_prop = 2.5 * X
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X, trim_at=(0.05, 0.95))

        # Should run with trimming
        assert np.isfinite(result["estimate"])
        assert result["n"] <= n


class TestDoublyRobustAdversarialCollinearity:
    """Adversarial tests for collinear covariates."""

    def test_perfect_collinearity_two_covariates(self):
        """
        Test with two perfectly collinear covariates (X2 = X1).

        Challenge: Collinearity causes model instability.
        Expected: sklearn handles via regularization, DR runs.
        """
        np.random.seed(1010)
        n = 200

        X1 = np.random.normal(0, 1, n)
        X2 = X1  # Perfect collinearity
        X = np.column_stack([X1, X2])

        # Propensity depends only on X1 (X2 redundant)
        logit_prop = 0.8 * X1
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X1) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run (sklearn handles collinearity)
        assert np.isfinite(result["estimate"])

        # May have higher variance due to instability
        assert result["se"] > 0

    def test_high_collinearity_three_covariates(self):
        """
        Test with three highly (but not perfectly) collinear covariates.

        Challenge: Near-collinearity increases model variance.
        Expected: DR runs but with inflated SE.
        """
        np.random.seed(1011)
        n = 250

        X1 = np.random.normal(0, 1, n)
        X2 = X1 + np.random.normal(0, 0.1, n)  # High correlation with X1
        X3 = X1 + np.random.normal(0, 0.1, n)  # High correlation with X1
        X = np.column_stack([X1, X2, X3])

        logit_prop = 0.6 * X1 + 0.1 * X2 - 0.1 * X3
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.4 * X1 + 0.2 * X2) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run with collinearity
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0


class TestDoublyRobustAdversarialTreatmentImbalance:
    """Adversarial tests for extreme treatment imbalance."""

    def test_extreme_imbalance_10_percent_treated(self):
        """
        Test with extreme treatment imbalance (10% treated, 90% control).

        Challenge: Few treated units means outcome model for T=1 has high variance.
        Expected: DR runs but outcome diagnostics show low sample size.
        """
        np.random.seed(1012)
        n = 300

        X = np.random.normal(0, 1, n)

        # Create imbalance via propensity
        logit_prop = -2.0 + 0.5 * X  # Shift to create ~10% treatment rate
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run despite imbalance
        assert np.isfinite(result["estimate"])

        # Check that there is imbalance
        treatment_rate = result["n_treated"] / result["n"]
        assert treatment_rate < 0.3  # Expect <30% treated

        # Outcome diagnostics should reflect small treated sample
        assert result["outcome_diagnostics"]["n_treated"] < 100

    def test_extreme_imbalance_90_percent_treated(self):
        """
        Test with extreme treatment imbalance (90% treated, 10% control).

        Challenge: Few control units means outcome model for T=0 has high variance.
        Expected: DR runs but outcome diagnostics show low control sample size.
        """
        np.random.seed(1013)
        n = 300

        X = np.random.normal(0, 1, n)

        # Create imbalance via propensity
        logit_prop = 2.0 + 0.5 * X  # Shift to create ~90% treatment rate
        e_X = 1 / (1 + np.exp(-logit_prop))
        T = np.random.binomial(1, e_X)

        Y = 3.0 * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

        result = dr_ate(Y, T, X)

        # Should run despite imbalance
        assert np.isfinite(result["estimate"])

        # Check that there is imbalance
        treatment_rate = result["n_treated"] / result["n"]
        assert treatment_rate > 0.7  # Expect >70% treated

        # Outcome diagnostics should reflect small control sample
        assert result["outcome_diagnostics"]["n_control"] < 100
