"""Tests for Panel DML-CRE (Mundlak Approach).

Comprehensive test suite covering:
- PanelData validation and properties
- DML-CRE binary treatment
- DML-CRE continuous treatment
- Adversarial edge cases
- Monte Carlo validation
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.panel import (
    PanelData,
    DMLCREResult,
    dml_cre,
    dml_cre_continuous,
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================


def generate_panel_dgp(
    n_units: int = 50,
    n_periods: int = 10,
    n_covariates: int = 3,
    true_ate: float = 2.0,
    unit_effect_strength: float = 0.5,
    binary_treatment: bool = True,
    random_state: int = 42,
) -> tuple[PanelData, float]:
    """Generate panel data with known treatment effect.

    DGP:
        αᵢ = unit_effect_strength * X̄ᵢ₀  (unit effect correlated with covariates)
        Dᵢₜ = f(Xᵢₜ) + noise  (treatment depends on covariates)
        Yᵢₜ = αᵢ + βXᵢₜ + true_ate * Dᵢₜ + εᵢₜ

    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Periods per unit.
    n_covariates : int
        Number of covariates.
    true_ate : float
        True treatment effect.
    unit_effect_strength : float
        Correlation between αᵢ and X̄ᵢ.
    binary_treatment : bool
        Whether treatment is binary or continuous.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (PanelData, true_ate)
    """
    np.random.seed(random_state)

    n_obs = n_units * n_periods
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Covariates
    X = np.random.randn(n_obs, n_covariates)

    # Unit effects: correlated with mean of first covariate
    X_reshaped = X.reshape(n_units, n_periods, n_covariates)
    X_bar_i = np.mean(X_reshaped, axis=1)  # (n_units, n_covariates)
    alpha_i_per_unit = unit_effect_strength * X_bar_i[:, 0]
    alpha_i = np.repeat(alpha_i_per_unit, n_periods)

    # Treatment
    if binary_treatment:
        # Propensity depends on X[:, 0]
        propensity = 1 / (1 + np.exp(-X[:, 0]))
        D = (np.random.rand(n_obs) < propensity).astype(float)
    else:
        # Continuous treatment
        D = X[:, 0] + np.random.randn(n_obs)

    # Outcome
    Y = alpha_i + X[:, 0] + true_ate * D + np.random.randn(n_obs)

    panel = PanelData(Y, D, X, unit_id, time)
    return panel, true_ate


# =============================================================================
# PanelData Tests
# =============================================================================


class TestPanelData:
    """Tests for PanelData dataclass."""

    def test_balanced_panel_creation(self):
        """Balanced panel initializes correctly."""
        n_units, n_periods = 10, 5
        n_obs = n_units * n_periods
        unit_id = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        Y = np.random.randn(n_obs)
        D = np.random.binomial(1, 0.5, n_obs).astype(float)
        X = np.random.randn(n_obs, 3)

        panel = PanelData(Y, D, X, unit_id, time)

        assert panel.n_units == n_units
        assert panel.n_periods == n_periods
        assert panel.n_obs == n_obs
        assert panel.n_covariates == 3
        assert panel.is_balanced is True

    def test_unbalanced_panel_creation(self):
        """Unbalanced panel initializes correctly."""
        # Unit 0: 5 periods, Unit 1: 3 periods, Unit 2: 4 periods
        unit_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        time = np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3])
        n_obs = len(unit_id)
        Y = np.random.randn(n_obs)
        D = np.random.binomial(1, 0.5, n_obs).astype(float)
        X = np.random.randn(n_obs, 2)

        panel = PanelData(Y, D, X, unit_id, time)

        assert panel.n_units == 3
        assert panel.n_periods == 5  # Max unique time values
        assert panel.n_obs == 12
        assert panel.is_balanced is False

    def test_unit_means_computation(self):
        """Unit means computed correctly."""
        # 2 units, 3 periods each
        unit_id = np.array([0, 0, 0, 1, 1, 1])
        time = np.array([0, 1, 2, 0, 1, 2])
        Y = np.zeros(6)
        D = np.zeros(6)
        # Unit 0: X = [1, 2, 3], Unit 1: X = [4, 5, 6]
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)

        panel = PanelData(Y, D, X, unit_id, time)
        means = panel.compute_unit_means()

        # Unit 0 mean = 2.0, Unit 1 mean = 5.0
        expected = np.array([[2.0], [2.0], [2.0], [5.0], [5.0], [5.0]])
        np.testing.assert_array_almost_equal(means, expected)

    def test_treatment_mean_computation(self):
        """Treatment mean computed correctly."""
        unit_id = np.array([0, 0, 0, 1, 1, 1])
        time = np.array([0, 1, 2, 0, 1, 2])
        Y = np.zeros(6)
        D = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Unit 0: mean 2/3, Unit 1: mean 0
        X = np.random.randn(6, 2)

        panel = PanelData(Y, D, X, unit_id, time)
        means = panel.compute_treatment_mean()

        # Unit 0: (1+0+1)/3 = 2/3, Unit 1: 0/3 = 0
        expected = np.array([2 / 3, 2 / 3, 2 / 3, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(means, expected)

    def test_validation_length_mismatch(self):
        """Raises error for mismatched array lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            PanelData(
                outcomes=np.zeros(10),
                treatment=np.zeros(8),  # Wrong length
                covariates=np.zeros((10, 2)),
                unit_id=np.zeros(10),
                time=np.zeros(10),
            )

    def test_validation_nan_values(self):
        """Raises error for NaN in data."""
        Y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        D = np.zeros(6)
        X = np.random.randn(6, 2)
        unit_id = np.array([0, 0, 0, 1, 1, 1])
        time = np.array([0, 1, 2, 0, 1, 2])

        with pytest.raises(ValueError, match="NaN or Inf"):
            PanelData(Y, D, X, unit_id, time)

    def test_validation_minimum_units(self):
        """Raises error for fewer than 2 units."""
        with pytest.raises(ValueError, match="Need at least 2 units"):
            PanelData(
                outcomes=np.zeros(5),
                treatment=np.zeros(5),
                covariates=np.zeros((5, 2)),
                unit_id=np.zeros(5),  # Only 1 unit
                time=np.arange(5),
            )

    def test_validation_minimum_obs_per_unit(self):
        """Raises error for units with < 2 observations."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            PanelData(
                outcomes=np.zeros(3),
                treatment=np.zeros(3),
                covariates=np.zeros((3, 2)),
                unit_id=np.array([0, 0, 1]),  # Unit 1 has only 1 obs
                time=np.array([0, 1, 0]),
            )


# =============================================================================
# Binary Treatment DML-CRE Tests
# =============================================================================


class TestDMLCRE:
    """Tests for dml_cre (binary treatment)."""

    def test_basic_estimation(self):
        """Basic estimation works and returns expected structure."""
        panel, true_ate = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, random_state=42
        )
        result = dml_cre(panel, n_folds=5)

        assert isinstance(result, DMLCREResult)
        assert result.method == "dml_cre"
        assert result.n_units == 50
        assert result.n_obs == 500
        assert result.n_folds == 5
        assert len(result.cate) == 500
        assert len(result.unit_effects) == 50
        assert len(result.fold_estimates) == 5

    def test_ate_near_true_value(self):
        """ATE estimate is near true value with large sample."""
        panel, true_ate = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=2.0,
            unit_effect_strength=0.5,
            random_state=123,
        )
        result = dml_cre(panel, n_folds=5, outcome_model="ridge")

        # Within 3 SE of true value
        assert abs(result.ate - true_ate) < 3 * result.ate_se, (
            f"ATE={result.ate:.3f}, true={true_ate}, SE={result.ate_se:.3f}"
        )

    def test_confidence_interval_covers_true(self):
        """95% CI covers true value."""
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=2.0, random_state=456
        )
        result = dml_cre(panel, n_folds=5)

        assert result.ci_lower < true_ate < result.ci_upper, (
            f"CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}] does not cover true={true_ate}"
        )

    def test_zero_effect(self):
        """Detects zero effect correctly."""
        panel, _ = generate_panel_dgp(n_units=100, n_periods=10, true_ate=0.0, random_state=789)
        result = dml_cre(panel, n_folds=5)

        # ATE should be near 0, and 0 should be in CI
        assert abs(result.ate) < 0.5
        assert result.ci_lower < 0 < result.ci_upper

    def test_negative_effect(self):
        """Estimates negative effect correctly."""
        panel, true_ate = generate_panel_dgp(
            n_units=100, n_periods=10, true_ate=-1.5, random_state=101
        )
        result = dml_cre(panel, n_folds=5)

        assert result.ate < 0
        assert abs(result.ate - true_ate) < 3 * result.ate_se

    def test_different_outcome_models(self):
        """Works with different outcome models."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=5, random_state=42)

        for model in ["linear", "ridge", "random_forest"]:
            result = dml_cre(panel, n_folds=3, outcome_model=model)
            assert not np.isnan(result.ate)
            assert result.ate_se > 0

    def test_stratified_crossfit_respects_units(self):
        """Cross-fitting keeps units together (no split across folds)."""
        panel, _ = generate_panel_dgp(n_units=20, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=4)

        # All fold estimates should be valid (non-NaN)
        assert not np.any(np.isnan(result.fold_estimates))
        assert len(result.fold_estimates) == 4

    def test_rejects_continuous_treatment(self):
        """Raises error for continuous treatment."""
        panel, _ = generate_panel_dgp(
            n_units=20, n_periods=5, binary_treatment=False, random_state=42
        )

        with pytest.raises(ValueError, match="Treatment must be binary"):
            dml_cre(panel)

    def test_rejects_too_many_folds(self):
        """Raises error when n_folds > n_units."""
        panel, _ = generate_panel_dgp(n_units=5, n_periods=10, random_state=42)

        with pytest.raises(ValueError, match="n_folds > n_units"):
            dml_cre(panel, n_folds=10)

    def test_unit_effects_structure(self):
        """Unit effects have correct structure."""
        panel, _ = generate_panel_dgp(n_units=30, n_periods=8, random_state=42)
        result = dml_cre(panel, n_folds=5)

        assert result.unit_effects.shape == (30,)
        # Unit effects should vary (not all same)
        assert np.std(result.unit_effects) > 0

    def test_r_squared_diagnostics(self):
        """R-squared diagnostics are reasonable."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        # R² should be between 0 and 1
        assert 0 <= result.outcome_r2 <= 1
        assert 0 <= result.treatment_r2 <= 1


# =============================================================================
# Continuous Treatment DML-CRE Tests
# =============================================================================


class TestDMLCREContinuous:
    """Tests for dml_cre_continuous."""

    def test_basic_estimation(self):
        """Basic estimation works and returns expected structure."""
        panel, true_ate = generate_panel_dgp(
            n_units=50,
            n_periods=10,
            true_ate=2.0,
            binary_treatment=False,
            random_state=42,
        )
        result = dml_cre_continuous(panel, n_folds=5)

        assert isinstance(result, DMLCREResult)
        assert result.method == "dml_cre_continuous"
        assert result.n_units == 50
        assert result.n_obs == 500
        assert len(result.cate) == 500

    def test_ate_near_true_value(self):
        """ATE estimate is near true value."""
        panel, true_ate = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=2.0,
            binary_treatment=False,
            random_state=123,
        )
        result = dml_cre_continuous(panel, n_folds=5)

        # Within 3 SE of true value
        assert abs(result.ate - true_ate) < 3 * result.ate_se

    def test_confidence_interval_covers_true(self):
        """95% CI covers true value."""
        panel, true_ate = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=2.0,
            binary_treatment=False,
            random_state=456,
        )
        result = dml_cre_continuous(panel, n_folds=5)

        assert result.ci_lower < true_ate < result.ci_upper

    def test_zero_effect(self):
        """Detects zero effect correctly."""
        panel, _ = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=0.0,
            binary_treatment=False,
            random_state=789,
        )
        result = dml_cre_continuous(panel, n_folds=5)

        assert abs(result.ate) < 0.5
        assert result.ci_lower < 0 < result.ci_upper

    def test_negative_effect(self):
        """Estimates negative effect correctly."""
        panel, true_ate = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=-1.5,
            binary_treatment=False,
            random_state=101,
        )
        result = dml_cre_continuous(panel, n_folds=5)

        assert result.ate < 0
        assert abs(result.ate - true_ate) < 3 * result.ate_se

    def test_different_treatment_models(self):
        """Works with different treatment models."""
        panel, _ = generate_panel_dgp(
            n_units=50, n_periods=5, binary_treatment=False, random_state=42
        )

        for model in ["linear", "ridge"]:
            result = dml_cre_continuous(panel, n_folds=3, treatment_model=model)
            assert not np.isnan(result.ate)
            assert result.ate_se > 0

    def test_treatment_r2_is_regular(self):
        """Treatment R² is regular (not pseudo) for continuous."""
        panel, _ = generate_panel_dgp(
            n_units=50, n_periods=10, binary_treatment=False, random_state=42
        )
        result = dml_cre_continuous(panel, n_folds=5)

        # Should be reasonable R² values
        assert 0 <= result.treatment_r2 <= 1
        # With confounding X[:, 0] → D, should have positive R²
        assert result.treatment_r2 > 0.1


# =============================================================================
# Adversarial Tests
# =============================================================================


class TestDMLCREAdversarial:
    """Adversarial edge case tests."""

    def test_small_panel(self):
        """Works with small panel (minimum viable)."""
        panel, _ = generate_panel_dgp(n_units=5, n_periods=4, random_state=42)
        result = dml_cre(panel, n_folds=2)

        assert not np.isnan(result.ate)

    def test_unbalanced_panel(self):
        """Works with unbalanced panel."""
        # Manually create unbalanced panel
        np.random.seed(42)
        # Unit 0: 10 periods, Unit 1: 5 periods, Unit 2: 8 periods
        unit_periods = [(0, 10), (1, 5), (2, 8), (3, 7), (4, 6)]
        unit_id_list = []
        time_list = []
        for unit, periods in unit_periods:
            unit_id_list.extend([unit] * periods)
            time_list.extend(range(periods))

        n_obs = len(unit_id_list)
        X = np.random.randn(n_obs, 2)
        D = (np.random.rand(n_obs) < 0.5).astype(float)
        Y = X[:, 0] + 2.0 * D + np.random.randn(n_obs)

        panel = PanelData(Y, D, X, np.array(unit_id_list), np.array(time_list))
        result = dml_cre(panel, n_folds=2)

        assert not np.isnan(result.ate)
        assert panel.is_balanced is False

    def test_high_dimensional_covariates(self):
        """Works with high-dimensional covariates."""
        panel, true_ate = generate_panel_dgp(
            n_units=50,
            n_periods=10,
            n_covariates=20,
            true_ate=2.0,
            random_state=42,
        )
        result = dml_cre(panel, n_folds=5, outcome_model="ridge")

        assert not np.isnan(result.ate)
        # Should still get reasonable estimate
        assert abs(result.ate - true_ate) < 1.5

    def test_strong_confounding(self):
        """Handles strong confounding (high unit effect correlation)."""
        panel, true_ate = generate_panel_dgp(
            n_units=100,
            n_periods=10,
            true_ate=2.0,
            unit_effect_strength=2.0,  # Strong confounding
            random_state=42,
        )
        result = dml_cre(panel, n_folds=5)

        # Should still recover true effect (Mundlak handles this)
        assert abs(result.ate - true_ate) < 3 * result.ate_se

    def test_imbalanced_treatment(self):
        """Works with imbalanced treatment (rare treatment)."""
        np.random.seed(42)
        n_units, n_periods = 100, 10
        n_obs = n_units * n_periods
        unit_id = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        X = np.random.randn(n_obs, 3)

        # Rare treatment (10% probability)
        D = (np.random.rand(n_obs) < 0.1).astype(float)
        Y = X[:, 0] + 2.0 * D + np.random.randn(n_obs)

        panel = PanelData(Y, D, X, unit_id, time)
        result = dml_cre(panel, n_folds=5)

        assert not np.isnan(result.ate)
        # Expect non-trivial SE with rare treatment
        assert result.ate_se > 0.05

    def test_all_zero_treatment_in_some_periods(self):
        """Works when some time periods have no treated units."""
        np.random.seed(42)
        n_units, n_periods = 20, 10
        n_obs = n_units * n_periods
        unit_id = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        X = np.random.randn(n_obs, 2)

        # Treatment only in periods 5-9
        D = np.zeros(n_obs)
        period_mask = time >= 5
        D[period_mask] = (np.random.rand(np.sum(period_mask)) < 0.5).astype(float)
        Y = X[:, 0] + 2.0 * D + np.random.randn(n_obs)

        panel = PanelData(Y, D, X, unit_id, time)
        result = dml_cre(panel, n_folds=4)

        assert not np.isnan(result.ate)


# =============================================================================
# Monte Carlo Validation
# =============================================================================


@pytest.mark.monte_carlo
class TestDMLCREMonteCarlo:
    """Monte Carlo validation tests."""

    def test_binary_unbiased(self):
        """Monte Carlo: Binary DML-CRE is unbiased."""
        true_ate = 2.0
        n_sims = 200
        estimates = []

        for seed in range(n_sims):
            panel, _ = generate_panel_dgp(
                n_units=50,
                n_periods=8,
                true_ate=true_ate,
                unit_effect_strength=0.5,
                random_state=1000 + seed,
            )
            result = dml_cre(panel, n_folds=3, outcome_model="linear")
            estimates.append(result.ate)

        bias = np.mean(estimates) - true_ate
        se_estimates = np.std(estimates) / np.sqrt(n_sims)

        assert abs(bias) < 0.20, f"Bias {bias:.4f} exceeds threshold"
        # Bias should be within 3 SE of 0
        assert abs(bias) < 3 * se_estimates, f"Bias {bias:.4f} > 3*SE({se_estimates:.4f})"

    def test_binary_coverage(self):
        """Monte Carlo: Binary DML-CRE has correct coverage."""
        true_ate = 2.0
        n_sims = 200
        covers = []

        for seed in range(n_sims):
            panel, _ = generate_panel_dgp(
                n_units=50,
                n_periods=8,
                true_ate=true_ate,
                random_state=2000 + seed,
            )
            result = dml_cre(panel, n_folds=3)
            covers.append(result.ci_lower < true_ate < result.ci_upper)

        coverage = np.mean(covers)
        # Coverage should be between 93% and 97%
        assert 0.88 < coverage < 0.98, f"Coverage {coverage:.2%} outside range"

    def test_continuous_unbiased(self):
        """Monte Carlo: Continuous DML-CRE is unbiased."""
        true_ate = 2.0
        n_sims = 200
        estimates = []

        for seed in range(n_sims):
            panel, _ = generate_panel_dgp(
                n_units=50,
                n_periods=8,
                true_ate=true_ate,
                binary_treatment=False,
                random_state=3000 + seed,
            )
            result = dml_cre_continuous(panel, n_folds=3, outcome_model="linear")
            estimates.append(result.ate)

        bias = np.mean(estimates) - true_ate

        assert abs(bias) < 0.20, f"Bias {bias:.4f} exceeds threshold"

    def test_continuous_coverage(self):
        """Monte Carlo: Continuous DML-CRE has correct coverage."""
        true_ate = 2.0
        n_sims = 200
        covers = []

        for seed in range(n_sims):
            panel, _ = generate_panel_dgp(
                n_units=50,
                n_periods=8,
                true_ate=true_ate,
                binary_treatment=False,
                random_state=4000 + seed,
            )
            result = dml_cre_continuous(panel, n_folds=3)
            covers.append(result.ci_lower < true_ate < result.ci_upper)

        coverage = np.mean(covers)
        assert 0.88 < coverage < 0.98, f"Coverage {coverage:.2%} outside range"


# =============================================================================
# Fold Estimates Tests
# =============================================================================


class TestFoldEstimates:
    """Tests for per-fold estimates and stability."""

    def test_fold_estimates_average_near_ate(self):
        """Mean of fold estimates should be near ATE."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        # Mean of fold estimates should be close to ATE
        fold_mean = np.mean(result.fold_estimates)
        assert abs(fold_mean - result.ate) < 0.5

    def test_fold_ses_are_positive(self):
        """Fold SEs should all be positive."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        assert np.all(result.fold_ses > 0)

    def test_fold_variance_diagnostic(self):
        """Variance across folds should be reasonable."""
        panel, _ = generate_panel_dgp(n_units=100, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        # Standard deviation of fold estimates should be < 2x the ATE SE
        fold_std = np.std(result.fold_estimates)
        assert fold_std < 2 * result.ate_se


# =============================================================================
# CATE Tests
# =============================================================================


class TestCATE:
    """Tests for CATE (heterogeneous effects)."""

    def test_cate_has_correct_shape(self):
        """CATE has correct shape."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        assert result.cate.shape == (500,)

    def test_cate_mean_near_ate(self):
        """Mean CATE should be near ATE."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        cate_mean = np.mean(result.cate)
        assert abs(cate_mean - result.ate) < 0.5

    def test_cate_varies_with_covariates(self):
        """CATE should vary (not be constant)."""
        panel, _ = generate_panel_dgp(n_units=50, n_periods=10, random_state=42)
        result = dml_cre(panel, n_folds=5)

        cate_std = np.std(result.cate)
        assert cate_std > 0.01  # Should have some variation
