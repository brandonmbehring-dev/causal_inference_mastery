"""
Cross-Language Parity Tests: Panel DML-CRE (Session 117)

Validates Python↔Julia parity for Panel DML with Correlated Random Effects.

Parity Requirements:
- ATE: rtol=0.05 (5%) - Looser due to different propensity model implementations
- SE: rtol=0.20 (20%) - Clustered SE can vary more
- n_obs, n_units, n_folds: Exact match
"""

import numpy as np
import pytest

from causal_inference.panel import PanelData, dml_cre, dml_cre_continuous

from .julia_interface import is_julia_available, julia_dml_cre, julia_dml_cre_continuous


# Skip all tests if Julia is not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-language validation"
)


def generate_panel_dgp(
    n_units: int = 50,
    n_periods: int = 10,
    n_covariates: int = 3,
    true_ate: float = 2.0,
    unit_effect_strength: float = 0.5,
    binary_treatment: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate panel test data.

    Returns
    -------
    tuple
        (Y, D, X, unit_id, time)
    """
    np.random.seed(random_state)

    n_obs = n_units * n_periods
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Covariates
    X = np.random.randn(n_obs, n_covariates)

    # Unit effects
    X_reshaped = X.reshape(n_units, n_periods, n_covariates)
    X_bar_i = np.mean(X_reshaped, axis=1)
    alpha_i_per_unit = unit_effect_strength * X_bar_i[:, 0]
    alpha_i = np.repeat(alpha_i_per_unit, n_periods)

    # Treatment
    if binary_treatment:
        propensity = 1 / (1 + np.exp(-X[:, 0]))
        D = (np.random.rand(n_obs) < propensity).astype(float)
    else:
        D = X[:, 0] + np.random.randn(n_obs)

    # Outcome
    Y = alpha_i + X[:, 0] + true_ate * D + np.random.randn(n_obs)

    return Y, D, X, unit_id, time


class TestDMLCREBinaryParity:
    """Python ↔ Julia parity tests for DML-CRE Binary Treatment."""

    def test_ate_parity(self):
        """ATE estimates match within rtol=0.05."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, random_state=42
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=5)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=5)

        # ATE should match within 5%
        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.05), (
            f"ATE mismatch: Python={py_result.ate:.6f}, Julia={jl_result['ate']:.6f}"
        )

    def test_se_parity(self):
        """SE estimates match within rtol=0.20."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, random_state=42
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=5)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=5)

        # SE should match within 20% (clustered SE can vary)
        assert np.isclose(py_result.ate_se, jl_result["ate_se"], rtol=0.20), (
            f"SE mismatch: Python={py_result.ate_se:.6f}, Julia={jl_result['ate_se']:.6f}"
        )

    def test_dimensions_parity(self):
        """n_obs, n_units, n_folds match exactly."""
        Y, D, X, unit_id, time = generate_panel_dgp(n_units=30, n_periods=8, random_state=123)

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=5)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=5)

        assert py_result.n_obs == jl_result["n_obs"]
        assert py_result.n_units == jl_result["n_units"]
        assert py_result.n_folds == jl_result["n_folds"]

    def test_cate_length_parity(self):
        """CATE arrays have same length."""
        Y, D, X, unit_id, time = generate_panel_dgp(n_units=20, n_periods=5, random_state=456)

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=3)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=3)

        assert len(py_result.cate) == len(jl_result["cate"])

    def test_unit_effects_length_parity(self):
        """Unit effects arrays have same length."""
        Y, D, X, unit_id, time = generate_panel_dgp(n_units=25, n_periods=6, random_state=789)

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=4)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=4)

        assert len(py_result.unit_effects) == len(jl_result["unit_effects"])


class TestDMLCREContinuousParity:
    """Python ↔ Julia parity tests for DML-CRE Continuous Treatment."""

    def test_ate_parity(self):
        """ATE estimates match within rtol=0.02."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, binary_treatment=False, random_state=42
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=5)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=5)

        # ATE should match within 2% for continuous treatment
        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.02), (
            f"ATE mismatch: Python={py_result.ate:.6f}, Julia={jl_result['ate']:.6f}"
        )

    def test_se_parity(self):
        """SE estimates match within rtol=0.15."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=10, true_ate=2.0, binary_treatment=False, random_state=42
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=5)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=5)

        # SE should match within 15%
        assert np.isclose(py_result.ate_se, jl_result["ate_se"], rtol=0.15), (
            f"SE mismatch: Python={py_result.ate_se:.6f}, Julia={jl_result['ate_se']:.6f}"
        )

    def test_dimensions_parity(self):
        """n_obs, n_units, n_folds match exactly."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=40, n_periods=8, binary_treatment=False, random_state=123
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=5)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=5)

        assert py_result.n_obs == jl_result["n_obs"]
        assert py_result.n_units == jl_result["n_units"]
        assert py_result.n_folds == jl_result["n_folds"]

    def test_r2_diagnostics_parity(self):
        """R² diagnostics match within rtol=0.10."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=10, binary_treatment=False, random_state=456
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=5)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=5)

        # R² should match within 10%
        assert np.isclose(py_result.outcome_r2, jl_result["outcome_r2"], rtol=0.10), (
            f"outcome_r2 mismatch: Python={py_result.outcome_r2:.4f}, "
            f"Julia={jl_result['outcome_r2']:.4f}"
        )
        assert np.isclose(py_result.treatment_r2, jl_result["treatment_r2"], rtol=0.10), (
            f"treatment_r2 mismatch: Python={py_result.treatment_r2:.4f}, "
            f"Julia={jl_result['treatment_r2']:.4f}"
        )


class TestDMLCREParityVariants:
    """Parity tests across different configurations."""

    def test_small_panel_parity(self):
        """Small panel (5 units) parity."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=5, n_periods=10, true_ate=2.0, random_state=101
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre(py_panel, n_folds=2)

        jl_result = julia_dml_cre(Y, D, X, unit_id, time, n_folds=2)

        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.10)

    def test_two_folds_parity(self):
        """Two-fold cross-fitting parity."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=30, n_periods=5, binary_treatment=False, random_state=202
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=2)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=2)

        assert len(py_result.fold_estimates) == 2
        assert len(jl_result["fold_estimates"]) == 2

    def test_high_dimensional_parity(self):
        """High-dimensional covariates parity."""
        Y, D, X, unit_id, time = generate_panel_dgp(
            n_units=50, n_periods=8, n_covariates=15, binary_treatment=False, random_state=303
        )

        py_panel = PanelData(Y, D, X, unit_id, time)
        py_result = dml_cre_continuous(py_panel, n_folds=5)

        jl_result = julia_dml_cre_continuous(Y, D, X, unit_id, time, n_folds=5)

        # May have slightly larger tolerance for high-dim
        assert np.isclose(py_result.ate, jl_result["ate"], rtol=0.05)
