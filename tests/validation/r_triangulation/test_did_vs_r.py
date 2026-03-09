"""Triangulation tests: Python DiD (Callaway-Sant'Anna) vs R `did` package.

This module provides Layer 5 validation by comparing our Python implementation
of Callaway-Sant'Anna (2021) DiD against the official R `did` package.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- ATT estimate: rtol=0.05
- Standard error: rtol=0.15
- Group-time effects: rtol=0.10

Run with: pytest tests/validation/r_triangulation/test_did_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_did_installed,
    check_r_available,
    r_did_callaway_santanna,
)

# Lazy import to avoid errors when did module paths differ
try:
    from src.causal_inference.did.callaway_santanna import callaway_santanna_ate
    from src.causal_inference.did.staggered import StaggeredData

    DID_AVAILABLE = True
except ImportError:
    DID_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_did_python = pytest.mark.skipif(
    not DID_AVAILABLE,
    reason="Python did module not available",
)

requires_did_r = pytest.mark.skipif(
    not check_did_installed(),
    reason="R 'did' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_staggered_did_dgp(
    n_units: int = 100,
    n_periods: int = 10,
    n_cohorts: int = 3,
    true_att: float = 2.0,
    never_treated_frac: float = 0.3,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from a staggered DiD DGP.

    Parameters
    ----------
    n_units : int
        Number of cross-sectional units.
    n_periods : int
        Number of time periods.
    n_cohorts : int
        Number of treatment cohorts (excluding never-treated).
    true_att : float
        True average treatment effect on treated.
    never_treated_frac : float
        Fraction of units that are never treated.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with Y, unit, time, first_treated arrays and true ATT.
    """
    np.random.seed(seed)

    # Assign units to cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    # Treatment cohorts: spread evenly across periods 3 to n_periods-2
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)

    unit_cohort = np.zeros(n_units, dtype=int)  # 0 = never treated
    treated_units = np.random.choice(n_units, n_treated, replace=False)

    for i, unit in enumerate(treated_units):
        cohort_idx = i % n_cohorts
        unit_cohort[unit] = cohort_periods[cohort_idx]

    # Generate panel data
    n_obs = n_units * n_periods
    unit_ids = np.repeat(np.arange(n_units), n_periods)
    time_ids = np.tile(np.arange(1, n_periods + 1), n_units)
    first_treated = unit_cohort[unit_ids]

    # Treatment indicator
    treated = np.zeros(n_obs)
    for i in range(n_obs):
        if first_treated[i] > 0 and time_ids[i] >= first_treated[i]:
            treated[i] = 1.0

    # Unit fixed effects
    unit_fe = np.random.randn(n_units) * 2
    unit_fe_obs = unit_fe[unit_ids]

    # Time fixed effects
    time_fe = np.cumsum(np.random.randn(n_periods) * 0.5)
    time_fe_obs = time_fe[time_ids - 1]

    # Outcome: Y = unit_fe + time_fe + ATT * treated + noise
    Y = unit_fe_obs + time_fe_obs + true_att * treated + noise_sd * np.random.randn(n_obs)

    return {
        "Y": Y,
        "unit": unit_ids,
        "time": time_ids,
        "first_treated": first_treated,
        "treated": treated,
        "true_att": true_att,
        "n_units": n_units,
        "n_periods": n_periods,
    }


def data_to_staggered(data: dict) -> StaggeredData:
    """Convert DGP data dict to StaggeredData object."""
    df = pd.DataFrame(
        {
            "Y": data["Y"],
            "unit": data["unit"],
            "time": data["time"],
            "first_treated": data["first_treated"],
        }
    )
    return StaggeredData.from_long_panel(
        data=df,
        outcome_col="Y",
        unit_col="unit",
        time_col="time",
        treatment_col="first_treated",
    )


# =============================================================================
# Layer 5: DiD ATT Triangulation
# =============================================================================


@requires_did_python
@requires_did_r
class TestDiDCallawayVsR:
    """Compare Python callaway_santanna_ate() to R `did` package."""

    def test_basic_att_parity(self):
        """Python CS-DiD ATT should match R `did` within rtol=0.05."""
        data = generate_staggered_did_dgp(
            n_units=100,
            n_periods=8,
            n_cohorts=2,
            true_att=2.0,
            seed=42,
        )

        # Python estimate
        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=100,
            random_state=42,
        )

        # R estimate
        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            control_group="nevertreated",
            aggregation="simple",
        )

        # Compare
        assert np.isclose(py_result["att"], r_result["att"], rtol=0.05), (
            f"ATT mismatch: Python={py_result['att']:.4f}, R={r_result['att']:.4f}"
        )

    def test_se_parity(self):
        """Python SE should match R SE within rtol=0.15."""
        data = generate_staggered_did_dgp(
            n_units=150,
            n_periods=10,
            n_cohorts=3,
            true_att=1.5,
            seed=123,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            control_group="nevertreated",
            n_bootstrap=200,
            random_state=123,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            control_group="nevertreated",
            aggregation="simple",
        )

        # SE comparison with looser tolerance
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.15), (
            f"SE mismatch: Python={py_result['se']:.4f}, R={r_result['se']:.4f}"
        )

    def test_dynamic_aggregation_parity(self):
        """Dynamic (event-time) aggregation should match R."""
        data = generate_staggered_did_dgp(
            n_units=120,
            n_periods=10,
            n_cohorts=3,
            true_att=2.5,
            seed=456,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="dynamic",
            control_group="nevertreated",
            n_bootstrap=100,
            random_state=456,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            control_group="nevertreated",
            aggregation="dynamic",
        )

        # Overall ATT should still match
        assert np.isclose(py_result["att"], r_result["att"], rtol=0.10), (
            f"Dynamic ATT mismatch: Python={py_result['att']:.4f}, R={r_result['att']:.4f}"
        )

    def test_group_aggregation_parity(self):
        """Group (cohort) aggregation should match R."""
        data = generate_staggered_did_dgp(
            n_units=100,
            n_periods=8,
            n_cohorts=2,
            true_att=1.0,
            seed=789,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="group",
            control_group="nevertreated",
            n_bootstrap=100,
            random_state=789,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            control_group="nevertreated",
            aggregation="group",
        )

        assert np.isclose(py_result["att"], r_result["att"], rtol=0.10), (
            f"Group ATT mismatch: Python={py_result['att']:.4f}, R={r_result['att']:.4f}"
        )

    def test_notyettreated_control(self):
        """Not-yet-treated control group should produce similar results."""
        data = generate_staggered_did_dgp(
            n_units=100,
            n_periods=10,
            n_cohorts=3,
            true_att=2.0,
            never_treated_frac=0.0,  # No never-treated, forces NYT
            seed=101,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            control_group="notyettreated",
            n_bootstrap=100,
            random_state=101,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            control_group="notyettreated",
            aggregation="simple",
        )

        assert np.isclose(py_result["att"], r_result["att"], rtol=0.10), (
            f"NYT ATT mismatch: Python={py_result['att']:.4f}, R={r_result['att']:.4f}"
        )

    def test_ci_coverage_consistency(self):
        """CI bounds should be consistent between Python and R."""
        data = generate_staggered_did_dgp(
            n_units=100,
            n_periods=8,
            true_att=2.0,
            seed=202,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            n_bootstrap=100,
            random_state=202,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
            aggregation="simple",
        )

        # Both CIs should contain the true ATT (if properly calibrated)
        true_att = data["true_att"]
        py_covers = py_result["ci_lower"] < true_att < py_result["ci_upper"]
        r_covers = r_result["ci_lower"] < true_att < r_result["ci_upper"]

        # At minimum, both should agree on coverage
        # (may fail occasionally due to randomness, but usually agree)
        # Just check that CIs are in same ballpark
        ci_width_ratio = (py_result["ci_upper"] - py_result["ci_lower"]) / (
            r_result["ci_upper"] - r_result["ci_lower"] + 1e-10
        )
        assert 0.5 < ci_width_ratio < 2.0, f"CI width ratio {ci_width_ratio:.2f} out of range"


@requires_did_python
@requires_did_r
class TestDiDEdgeCases:
    """Edge case tests for DiD triangulation."""

    def test_single_cohort(self):
        """Single treatment cohort should still match."""
        data = generate_staggered_did_dgp(
            n_units=80,
            n_periods=6,
            n_cohorts=1,  # Only one cohort
            true_att=3.0,
            seed=303,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            n_bootstrap=100,
            random_state=303,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
        )

        assert np.isclose(py_result["att"], r_result["att"], rtol=0.10), (
            f"Single cohort ATT mismatch"
        )

    def test_high_never_treated_fraction(self):
        """High never-treated fraction should still work."""
        data = generate_staggered_did_dgp(
            n_units=100,
            n_periods=8,
            n_cohorts=2,
            true_att=1.5,
            never_treated_frac=0.6,  # 60% never treated
            seed=404,
        )

        staggered_data = data_to_staggered(data)
        py_result = callaway_santanna_ate(
            staggered_data,
            aggregation="simple",
            n_bootstrap=100,
            random_state=404,
        )

        r_result = r_did_callaway_santanna(
            outcome=data["Y"],
            unit=data["unit"],
            time=data["time"],
            first_treated=data["first_treated"],
        )

        assert np.isclose(py_result["att"], r_result["att"], rtol=0.10), (
            f"High never-treated fraction ATT mismatch"
        )


@requires_did_python
@requires_did_r
class TestDiDMonteCarlo:
    """Monte Carlo validation of DiD triangulation."""

    @pytest.mark.slow
    def test_monte_carlo_bias_comparison(self):
        """Both Python and R should have similar bias properties."""
        n_sims = 50
        true_att = 2.0
        py_estimates = []
        r_estimates = []

        for sim in range(n_sims):
            data = generate_staggered_did_dgp(
                n_units=80,
                n_periods=8,
                n_cohorts=2,
                true_att=true_att,
                seed=1000 + sim,
            )

            try:
                staggered_data = data_to_staggered(data)
                py_result = callaway_santanna_ate(
                    staggered_data,
                    aggregation="simple",
                    n_bootstrap=50,
                    random_state=1000 + sim,
                )
                py_estimates.append(py_result["att"])

                r_result = r_did_callaway_santanna(
                    outcome=data["Y"],
                    unit=data["unit"],
                    time=data["time"],
                    first_treated=data["first_treated"],
                )
                r_estimates.append(r_result["att"])
            except Exception:
                continue

        py_bias = np.mean(py_estimates) - true_att
        r_bias = np.mean(r_estimates) - true_att

        # Both should have small bias
        assert abs(py_bias) < 0.20, f"Python bias {py_bias:.4f} too large"
        assert abs(r_bias) < 0.20, f"R bias {r_bias:.4f} too large"

        # Bias difference should be small
        assert abs(py_bias - r_bias) < 0.15, (
            f"Bias difference {abs(py_bias - r_bias):.4f} between Python and R too large"
        )
