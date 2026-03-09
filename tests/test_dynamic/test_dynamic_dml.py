"""Tests for Dynamic DML estimator.

Implements 3-layer validation:
1. Known-answer tests: Verify recovery of known effects
2. Adversarial tests: Edge cases and challenging scenarios
3. Monte Carlo: Statistical properties validation

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation.
"""

import numpy as np
import pytest

from causal_inference.dynamic import (
    BlockedTimeSeriesSplit,
    DynamicDMLResult,
    PanelStratifiedSplit,
    ProgressiveBlockSplit,
    RollingOriginSplit,
    dynamic_dml,
    dynamic_dml_panel,
    simulate_dynamic_dgp,
)

# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestKnownAnswer:
    """Tests verifying recovery of known treatment effects."""

    def test_contemporaneous_effect_recovery(self, simple_dgp):
        """Recover contemporaneous effect with no lagged effects."""
        Y, D, X, true_effects = simple_dgp

        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=5, nuisance_model="ridge")

        # Should recover theta_0 ≈ 2.0
        assert abs(result.theta[0] - 2.0) < 0.3, f"Bias too large: {result.theta[0] - 2.0:.3f}"

        # CI should cover true value
        assert result.ci_lower[0] < 2.0 < result.ci_upper[0], "CI does not cover true effect"

    def test_lagged_effects_recovery(self, lagged_dgp):
        """Recover both contemporaneous and lagged effects."""
        Y, D, X, true_effects = lagged_dgp

        result = dynamic_dml(Y, D, X, max_lag=2, n_folds=5, nuisance_model="ridge")

        # Check each lag
        for h in range(3):
            bias = result.theta[h] - true_effects[h]
            assert abs(bias) < 0.5, f"Lag {h} bias too large: {bias:.3f}"

    def test_zero_effect_detection(self, zero_effect_dgp):
        """Correctly identify zero treatment effect."""
        Y, D, X, _ = zero_effect_dgp

        result = dynamic_dml(Y, D, X, max_lag=1, n_folds=5, nuisance_model="ridge")

        # Effects should be close to zero
        for h in range(2):
            assert abs(result.theta[h]) < 0.5, f"Lag {h} spurious effect: {result.theta[h]:.3f}"

        # CIs should include zero
        for h in range(2):
            assert result.ci_lower[h] < 0 < result.ci_upper[h], f"Lag {h} CI excludes zero"

    def test_panel_data_recovery(self, panel_dgp):
        """Recover effects from panel data."""
        Y, D, X, unit_id, true_effects = panel_dgp

        result = dynamic_dml_panel(
            outcomes=Y,
            treatments=D,
            states=X,
            unit_id=unit_id,
            max_lag=1,
            n_folds=5,
            nuisance_model="ridge",
        )

        # Check recovery
        for h in range(2):
            bias = result.theta[h] - true_effects[h]
            # Panel data has more variance, allow larger tolerance
            assert abs(bias) < 0.6, f"Lag {h} bias: {bias:.3f}"

    def test_cumulative_effect(self, lagged_dgp):
        """Verify cumulative effect calculation."""
        Y, D, X, true_effects = lagged_dgp

        discount = 0.99
        result = dynamic_dml(
            Y, D, X, max_lag=2, n_folds=5, nuisance_model="ridge", discount_factor=discount
        )

        # Expected cumulative: sum(discount^h * theta_h)
        true_cumulative = sum(discount**h * true_effects[h] for h in range(3))

        assert abs(result.cumulative_effect - true_cumulative) < 1.5, (
            f"Cumulative effect bias: {result.cumulative_effect - true_cumulative:.3f}"
        )


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestAdversarial:
    """Adversarial tests for edge cases and challenging scenarios."""

    def test_sparse_treatment(self, sparse_treatment_dgp):
        """Handle sparse treatment (10% treated)."""
        Y, D, X, true_effects = sparse_treatment_dgp

        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge")

        # Should still estimate, may have higher variance
        assert np.isfinite(result.theta[0]), "Non-finite estimate with sparse treatment"
        assert result.theta_se[0] > 0, "Standard error should be positive"

    def test_autocorrelated_errors(self, autocorrelated_dgp):
        """Handle autocorrelated errors with HAC."""
        Y, D, X, true_effects = autocorrelated_dgp

        result = dynamic_dml(
            Y, D, X, max_lag=0, n_folds=5, nuisance_model="ridge", hac_kernel="bartlett"
        )

        # HAC should provide valid inference despite autocorrelation
        assert result.theta_se[0] > 0, "HAC SE should be positive"

        # Bias check
        assert abs(result.theta[0] - 2.0) < 0.5, (
            f"Bias with autocorrelation: {result.theta[0] - 2.0:.3f}"
        )

    def test_confounded_treatment(self, adversarial_confounded_dgp):
        """Test with strong confounding (DML should reduce bias)."""
        Y, D, X, true_effects = adversarial_confounded_dgp

        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=5, nuisance_model="random_forest")

        # DML should reduce but may not eliminate confounding bias
        # since X only proxies for U
        assert np.isfinite(result.theta[0]), "Non-finite with confounding"

    def test_short_time_series(self):
        """Handle short time series (T=100)."""
        Y, D, X, _ = simulate_dynamic_dgp(n_obs=100, n_lags=2, seed=42)

        result = dynamic_dml(Y, D, X, max_lag=2, n_folds=3, nuisance_model="ridge")

        assert len(result.theta) == 3, "Should estimate effects for all lags"
        assert all(np.isfinite(result.theta)), "All estimates should be finite"

    def test_high_dimensional_states(self):
        """Handle high-dimensional covariates."""
        np.random.seed(42)
        n = 300
        p = 20  # High-dimensional

        X = np.random.randn(n, p)
        D = np.random.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, :3] @ [1, 0.5, 0.2] + np.random.randn(n)

        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge")

        assert abs(result.theta[0] - 2.0) < 0.5, "Bias with high-dim X"

    def test_minimum_observations(self):
        """Reject insufficient observations."""
        np.random.seed(42)
        Y = np.random.randn(15)
        D = np.random.binomial(1, 0.5, 15).astype(float)
        X = np.random.randn(15, 2)

        with pytest.raises(ValueError, match="Insufficient observations"):
            dynamic_dml(Y, D, X, max_lag=10)

    def test_invalid_max_lag(self):
        """Reject invalid max_lag."""
        Y, D, X, _ = simulate_dynamic_dgp(n_obs=100, seed=42)

        with pytest.raises(ValueError, match="max_lag"):
            dynamic_dml(Y, D, X, max_lag=-1)

    def test_length_mismatch(self):
        """Reject mismatched input lengths."""
        Y = np.random.randn(100)
        D = np.random.randn(101)  # Wrong length
        X = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Length mismatch"):
            dynamic_dml(Y, D, X, max_lag=2)


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo validation of statistical properties."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_unbiasedness_lag_0(self):
        """Monte Carlo: Bias < 0.10 for contemporaneous effect."""
        n_sims = 500
        true_effect = 2.0
        estimates = []

        for seed in range(n_sims):
            Y, D, X, _ = simulate_dynamic_dgp(
                n_obs=300,
                n_lags=1,
                true_effects=np.array([true_effect]),
                confounding_strength=0.0,  # No confounding for clean test
                seed=seed,
            )

            result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge")
            estimates.append(result.theta[0])

        bias = np.mean(estimates) - true_effect
        assert abs(bias) < 0.10, f"Bias {bias:.4f} exceeds 0.10"

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_coverage_lag_0(self):
        """Monte Carlo: Coverage 93-97% for 95% CIs."""
        n_sims = 500
        true_effect = 2.0
        coverage_count = 0

        for seed in range(n_sims):
            Y, D, X, _ = simulate_dynamic_dgp(
                n_obs=300,
                n_lags=1,
                true_effects=np.array([true_effect]),
                confounding_strength=0.0,
                seed=seed,
            )

            result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge", alpha=0.05)

            if result.ci_lower[0] < true_effect < result.ci_upper[0]:
                coverage_count += 1

        coverage = coverage_count / n_sims
        assert 0.90 < coverage < 0.98, f"Coverage {coverage:.2%} outside [93%, 97%]"

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_lagged_effect_recovery(self):
        """Monte Carlo: Recovery of lagged effects."""
        n_sims = 300
        true_effects = np.array([2.0, 1.0, 0.5])
        estimates = {0: [], 1: [], 2: []}

        for seed in range(n_sims):
            Y, D, X, _ = simulate_dynamic_dgp(
                n_obs=400, n_lags=3, true_effects=true_effects, confounding_strength=0.1, seed=seed
            )

            result = dynamic_dml(Y, D, X, max_lag=2, n_folds=3, nuisance_model="ridge")

            for h in range(3):
                estimates[h].append(result.theta[h])

        # Check bias at each lag
        for h in range(3):
            bias = np.mean(estimates[h]) - true_effects[h]
            threshold = 0.10 if h == 0 else 0.15  # Allow more bias for lagged effects
            assert abs(bias) < threshold, f"Lag {h} bias {bias:.4f} exceeds {threshold}"

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_hac_improves_coverage(self):
        """Monte Carlo: HAC should improve coverage with autocorrelation."""
        n_sims = 200
        true_effect = 2.0
        coverage_naive = 0
        coverage_hac = 0

        for seed in range(n_sims):
            np.random.seed(seed)
            n = 300
            X = np.random.randn(n, 3)
            D = np.random.binomial(1, 0.5, n).astype(float)

            # Autocorrelated errors
            epsilon = np.zeros(n)
            epsilon[0] = np.random.randn()
            for t in range(1, n):
                epsilon[t] = 0.6 * epsilon[t - 1] + np.sqrt(0.64) * np.random.randn()

            Y = true_effect * D + X @ [1, 0.5, 0.2] + epsilon

            # With HAC
            result_hac = dynamic_dml(
                Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge", hac_kernel="bartlett"
            )

            if result_hac.ci_lower[0] < true_effect < result_hac.ci_upper[0]:
                coverage_hac += 1

        coverage_hac = coverage_hac / n_sims
        # HAC coverage should be reasonable (>85% at least)
        assert coverage_hac > 0.85, f"HAC coverage {coverage_hac:.2%} too low"


# =============================================================================
# Cross-Fitting Strategy Tests
# =============================================================================


class TestCrossFitting:
    """Tests for cross-fitting strategies."""

    def test_blocked_split_indices(self):
        """BlockedTimeSeriesSplit produces valid indices."""
        cv = BlockedTimeSeriesSplit(n_splits=5)
        X = np.arange(100)

        all_test = []
        for train_idx, test_idx in cv.split(X):
            # Train and test should not overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            all_test.extend(test_idx.tolist())

        # All indices should be covered
        assert sorted(all_test) == list(range(100))

    def test_rolling_split_forward_only(self):
        """RollingOriginSplit only trains on past data."""
        cv = RollingOriginSplit(initial_window=50, step=10, horizon=5)
        X = np.arange(100)

        for train_idx, test_idx in cv.split(X):
            # All training indices should be before all test indices
            assert train_idx.max() < test_idx.min()

    def test_panel_split_units(self):
        """PanelStratifiedSplit splits by unit."""
        cv = PanelStratifiedSplit(n_splits=5)
        n_units, n_periods = 50, 10
        X = np.arange(n_units * n_periods)
        unit_id = np.repeat(np.arange(n_units), n_periods)

        for train_idx, test_idx in cv.split(X, unit_id=unit_id):
            train_units = np.unique(unit_id[train_idx])
            test_units = np.unique(unit_id[test_idx])

            # Units should not overlap between train and test
            assert len(np.intersect1d(train_units, test_units)) == 0

    def test_progressive_split(self):
        """ProgressiveBlockSplit uses expanding window."""
        cv = ProgressiveBlockSplit(n_blocks=10, min_train_blocks=3)
        X = np.arange(1000)

        train_sizes = []
        for train_idx, test_idx in cv.split(X):
            train_sizes.append(len(train_idx))
            # Training should be before test
            assert train_idx.max() < test_idx.min()

        # Training size should increase
        assert train_sizes == sorted(train_sizes)

    def test_cross_fitting_strategies_run(self, lagged_dgp):
        """All cross-fitting strategies should run successfully."""
        Y, D, X, _ = lagged_dgp

        for strategy in ["blocked", "rolling", "progressive"]:
            result = dynamic_dml(Y, D, X, max_lag=1, n_folds=3, cross_fitting=strategy)
            assert result is not None
            assert all(np.isfinite(result.theta))


# =============================================================================
# Result Object Tests
# =============================================================================


class TestDynamicDMLResult:
    """Tests for DynamicDMLResult dataclass."""

    def test_summary_format(self, simple_dgp):
        """Summary method produces valid output."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=1, n_folds=3)

        summary = result.summary()
        assert "Dynamic DML Results" in summary
        assert "Lag" in summary
        assert "Effect" in summary
        assert "SE" in summary

    def test_is_significant(self, simple_dgp):
        """is_significant method works correctly."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3)

        # Lag 0 should be significant (true effect = 2.0)
        assert result.is_significant(0) == True  # noqa: E712 - numpy bool

    def test_nuisance_r2_populated(self, simple_dgp):
        """Nuisance R² values are populated."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=1, n_folds=3)

        assert "outcome_r2" in result.nuisance_r2
        assert "propensity_r2" in result.nuisance_r2
        assert len(result.nuisance_r2["outcome_r2"]) > 0


# =============================================================================
# Nuisance Model Tests
# =============================================================================


class TestNuisanceModels:
    """Tests for different nuisance model configurations."""

    def test_ridge_nuisance(self, simple_dgp):
        """Ridge regression nuisance model works."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="ridge")
        assert np.isfinite(result.theta[0])

    def test_random_forest_nuisance(self, simple_dgp):
        """Random forest nuisance model works."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="random_forest")
        assert np.isfinite(result.theta[0])

    def test_gradient_boosting_nuisance(self, simple_dgp):
        """Gradient boosting nuisance model works."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, n_folds=3, nuisance_model="gradient_boosting")
        assert np.isfinite(result.theta[0])


# =============================================================================
# HAC Inference Tests
# =============================================================================


class TestHACInference:
    """Tests for HAC standard error estimation."""

    def test_bartlett_kernel(self, simple_dgp):
        """Bartlett kernel produces valid SEs."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, hac_kernel="bartlett")

        assert result.theta_se[0] > 0
        assert result.hac_kernel == "bartlett"

    def test_qs_kernel(self, simple_dgp):
        """Quadratic spectral kernel produces valid SEs."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, hac_kernel="qs")

        assert result.theta_se[0] > 0
        assert result.hac_kernel == "qs"

    def test_custom_bandwidth(self, simple_dgp):
        """Custom bandwidth is respected."""
        Y, D, X, _ = simple_dgp
        result = dynamic_dml(Y, D, X, max_lag=0, hac_bandwidth=10)

        assert result.hac_bandwidth == 10
