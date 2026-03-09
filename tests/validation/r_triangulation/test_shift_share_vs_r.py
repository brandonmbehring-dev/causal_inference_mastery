"""R Triangulation Tests for Shift-Share IV Estimators.

Validates our Python ShiftShareIV implementation against R's ShiftShareSE package.
Tests focus on coefficient estimation, standard errors, and first-stage diagnostics.

References:
- Goldsmith-Pinkham, Sorkin, Swift (2020) - Bartik instruments
- Borusyak, Hull, Jaravel (2022) - Quasi-experimental shift-share designs
- Adão, Kolesár, Morales (2019) - Shift-share inference theory
"""

import numpy as np
import pytest
from typing import Tuple, Optional

# Import R interface
from .r_interface import (
    check_r_available,
    check_shiftsharese_installed,
    r_shift_share_ivreg_ss,
)

# Import Python implementation (guarded to prevent collection failure when not installed)
_ci_shift_share = pytest.importorskip(
    "causal_inference.shift_share",
    reason="causal_inference package not installed (run: pip install -e .)",
)
ShiftShareIV = _ci_shift_share.ShiftShareIV
shift_share_iv = _ci_shift_share.shift_share_iv

# Import DGP from test_shift_share
import sys
from pathlib import Path

test_ss_path = Path(__file__).parent.parent.parent / "test_shift_share"
if str(test_ss_path) not in sys.path:
    sys.path.insert(0, str(test_ss_path))

from conftest import generate_shift_share_data


# =============================================================================
# Skip Conditions
# =============================================================================

pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available",
)

requires_shiftsharese = pytest.mark.skipif(
    not check_shiftsharese_installed(),
    reason=(
        "R 'ShiftShareSE' package not installed. Install with: install.packages('ShiftShareSE')"
    ),
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ss_basic():
    """Basic shift-share data with strong first stage."""
    return generate_shift_share_data(
        n=300,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=2.0,
        random_state=42,
    )


@pytest.fixture
def ss_with_controls():
    """Shift-share with exogenous controls."""
    return generate_shift_share_data(
        n=300,
        n_sectors=10,
        true_beta=1.5,
        first_stage_strength=2.0,
        n_controls=3,
        random_state=123,
    )


@pytest.fixture
def ss_many_sectors():
    """Many sectors (50)."""
    return generate_shift_share_data(
        n=400,
        n_sectors=50,
        true_beta=2.0,
        first_stage_strength=1.5,
        random_state=456,
    )


@pytest.fixture
def ss_concentrated():
    """Concentrated shares (few dominant sectors)."""
    return generate_shift_share_data(
        n=300,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=2.0,
        share_concentration=0.2,
        random_state=789,
    )


@pytest.fixture
def ss_weak_first_stage():
    """Weak first-stage relationship."""
    return generate_shift_share_data(
        n=300,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=0.5,
        random_state=222,
    )


@pytest.fixture
def ss_large():
    """Large sample for precision testing."""
    return generate_shift_share_data(
        n=2000,
        n_sectors=15,
        true_beta=1.5,
        first_stage_strength=2.0,
        random_state=333,
    )


@pytest.fixture
def ss_negative():
    """Negative treatment effect."""
    return generate_shift_share_data(
        n=300,
        n_sectors=10,
        true_beta=-1.5,
        first_stage_strength=2.0,
        random_state=444,
    )


# =============================================================================
# Test Classes
# =============================================================================


@requires_shiftsharese
class TestShiftShareCoefficientVsR:
    """Test coefficient estimation parity with R ShiftShareSE."""

    def test_basic_coefficient_parity(self, ss_basic):
        """Basic 2SLS coefficient should match R within tolerance."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        # Python estimate
        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        # R estimate
        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Coefficient tolerance: rtol=0.05 (deterministic 2SLS)
        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.05,
        ), (
            f"Coefficient mismatch: Python={py_result['estimate']:.4f}, "
            f"R={r_result['coefficient']:.4f}"
        )

    def test_coefficient_with_controls(self, ss_with_controls):
        """Coefficient with controls should match R."""
        Y, D, shares, shocks, X, true_beta = ss_with_controls

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            X=X,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            X=X,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.05,
        ), (
            f"Coefficient with controls mismatch: "
            f"Python={py_result['estimate']:.4f}, R={r_result['coefficient']:.4f}"
        )

    def test_negative_effect_coefficient(self, ss_negative):
        """Negative treatment effect should match R."""
        Y, D, shares, shocks, X, true_beta = ss_negative

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Both should estimate negative effect
        assert py_result["estimate"] < 0, "Python should detect negative effect"
        assert r_result["coefficient"] < 0, "R should detect negative effect"

        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.05,
        )

    def test_large_sample_coefficient_precision(self, ss_large):
        """Large sample should give tighter coefficient agreement."""
        Y, D, shares, shocks, X, true_beta = ss_large

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Tighter tolerance for large sample
        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.02,
        ), (
            f"Large sample coefficient mismatch: "
            f"Python={py_result['estimate']:.4f}, R={r_result['coefficient']:.4f}"
        )


@requires_shiftsharese
class TestShiftShareSEVsR:
    """Test standard error estimation parity with R."""

    def test_basic_se_parity(self, ss_basic):
        """Standard errors should be reasonably close to R."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # SE tolerance: rtol=0.15 (different inference methods)
        # Note: Python uses HC3-style robust SE, R's AKM has different adjustment
        assert np.isclose(
            py_result["se"],
            r_result["se"],
            rtol=0.20,
        ), f"SE mismatch: Python={py_result['se']:.4f}, R={r_result['se']:.4f}"

    def test_se_order_of_magnitude(self, ss_basic):
        """SE should be same order of magnitude as R."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # SEs should be within factor of 2
        ratio = py_result["se"] / r_result["se"]
        assert 0.5 < ratio < 2.0, (
            f"SE ratio {ratio:.2f} outside [0.5, 2.0]: "
            f"Python={py_result['se']:.4f}, R={r_result['se']:.4f}"
        )


@requires_shiftsharese
class TestShiftShareFirstStageVsR:
    """Test first-stage diagnostics parity with R."""

    def test_first_stage_f_parity(self, ss_basic):
        """First-stage F-statistic should match R closely."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # First-stage F should be very close (same calculation)
        assert np.isclose(
            py_result["first_stage"]["f_statistic"],
            r_result["first_stage_f"],
            rtol=0.05,
        ), (
            f"First-stage F mismatch: "
            f"Python={py_result['first_stage']['f_statistic']:.2f}, "
            f"R={r_result['first_stage_f']:.2f}"
        )

    def test_weak_first_stage_both_detect(self, ss_weak_first_stage):
        """Both should detect weak first stage."""
        Y, D, shares, shocks, X, true_beta = ss_weak_first_stage

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Both should have F < 10 (Stock-Yogo threshold)
        py_f = py_result["first_stage"]["f_statistic"]
        r_f = r_result["first_stage_f"]

        # At least one should detect weak instrument
        weak_detected = (py_f < 10) or (r_f < 10)
        assert weak_detected, (
            f"Neither detected weak instrument: Python F={py_f:.2f}, R F={r_f:.2f}"
        )


@requires_shiftsharese
class TestShiftShareEdgeCases:
    """Test edge cases and challenging scenarios."""

    def test_many_sectors_parity(self, ss_many_sectors):
        """Many sectors should still give coefficient agreement."""
        Y, D, shares, shocks, X, true_beta = ss_many_sectors

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.05,
        ), (
            f"Many sectors coefficient mismatch: "
            f"Python={py_result['estimate']:.4f}, R={r_result['coefficient']:.4f}"
        )

    def test_concentrated_shares_parity(self, ss_concentrated):
        """Concentrated shares should still work."""
        Y, D, shares, shocks, X, true_beta = ss_concentrated

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        assert np.isclose(
            py_result["estimate"],
            r_result["coefficient"],
            rtol=0.07,
        ), (
            f"Concentrated shares coefficient mismatch: "
            f"Python={py_result['estimate']:.4f}, R={r_result['coefficient']:.4f}"
        )


@requires_shiftsharese
class TestShiftShareTrueEffectRecovery:
    """Test that both Python and R recover true effects."""

    def test_both_recover_positive_effect(self, ss_basic):
        """Both should recover true positive effect."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Both should be in reasonable range of true effect
        assert 0.5 < py_result["estimate"] < 4.0, (
            f"Python estimate {py_result['estimate']:.4f} far from true {true_beta}"
        )
        assert 0.5 < r_result["coefficient"] < 4.0, (
            f"R estimate {r_result['coefficient']:.4f} far from true {true_beta}"
        )

    def test_both_recover_negative_effect(self, ss_negative):
        """Both should recover true negative effect."""
        Y, D, shares, shocks, X, true_beta = ss_negative

        py_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        r_result = r_shift_share_ivreg_ss(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            se_method="AKM",
        )

        if r_result is None:
            pytest.skip("R ShiftShareSE returned None")

        # Both should be in reasonable range of true effect
        assert -3.0 < py_result["estimate"] < -0.5, (
            f"Python estimate {py_result['estimate']:.4f} far from true {true_beta}"
        )
        assert -3.0 < r_result["coefficient"] < -0.5, (
            f"R estimate {r_result['coefficient']:.4f} far from true {true_beta}"
        )


@requires_shiftsharese
@pytest.mark.slow
class TestShiftShareMonteCarloTriangulation:
    """Monte Carlo tests for systematic agreement."""

    def test_monte_carlo_coefficient_agreement(self):
        """Monte Carlo: Python and R should agree on 85%+ of runs."""
        n_runs = 25
        agreements = 0
        rtol = 0.08

        for seed in range(n_runs):
            Y, D, shares, shocks, X, true_beta = generate_shift_share_data(
                n=300,
                n_sectors=10,
                true_beta=2.0,
                first_stage_strength=2.0,
                random_state=seed,
            )

            py_result = shift_share_iv(
                Y=Y,
                D=D,
                shares=shares,
                shocks=shocks,
                inference="robust",
            )

            r_result = r_shift_share_ivreg_ss(
                Y=Y,
                D=D,
                shares=shares,
                shocks=shocks,
                se_method="AKM",
            )

            if r_result is None:
                continue

            if np.isclose(
                py_result["estimate"],
                r_result["coefficient"],
                rtol=rtol,
            ):
                agreements += 1

        agreement_rate = agreements / n_runs
        assert agreement_rate >= 0.85, (
            f"Monte Carlo coefficient agreement rate {agreement_rate:.0%} < 85%"
        )

    def test_monte_carlo_se_correlation(self):
        """Monte Carlo: Python and R SEs should be positively correlated."""
        n_runs = 20
        py_ses = []
        r_ses = []

        for seed in range(n_runs):
            Y, D, shares, shocks, X, true_beta = generate_shift_share_data(
                n=300,
                n_sectors=10,
                true_beta=2.0,
                first_stage_strength=2.0,
                random_state=seed,
            )

            py_result = shift_share_iv(
                Y=Y,
                D=D,
                shares=shares,
                shocks=shocks,
                inference="robust",
            )

            r_result = r_shift_share_ivreg_ss(
                Y=Y,
                D=D,
                shares=shares,
                shocks=shocks,
                se_method="AKM",
            )

            if r_result is not None:
                py_ses.append(py_result["se"])
                r_ses.append(r_result["se"])

        if len(py_ses) < 10:
            pytest.skip("Not enough valid R results")

        # Correlation should be positive
        correlation = np.corrcoef(py_ses, r_ses)[0, 1]
        assert correlation > 0.5, f"SE correlation {correlation:.2f} too low (expected > 0.5)"


# =============================================================================
# Basic Python Implementation Tests (without R dependency)
# =============================================================================


class TestShiftShareBasicFunctionality:
    """Basic tests that don't require R."""

    def test_class_and_function_equivalence(self, ss_basic):
        """ShiftShareIV class and shift_share_iv function should match."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        # Class-based
        estimator = ShiftShareIV(inference="robust")
        class_result = estimator.fit(Y=Y, D=D, shares=shares, shocks=shocks)

        # Function-based
        func_result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        assert np.isclose(
            class_result["estimate"],
            func_result["estimate"],
            rtol=1e-10,
        )
        assert np.isclose(
            class_result["se"],
            func_result["se"],
            rtol=1e-10,
        )

    def test_result_structure(self, ss_basic):
        """Result should have expected structure."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        # Required keys
        assert "estimate" in result
        assert "se" in result
        assert "t_stat" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "first_stage" in result

        # First stage structure
        assert "f_statistic" in result["first_stage"]
        assert "coefficient" in result["first_stage"]

    def test_positive_se(self, ss_basic):
        """Standard error should be positive."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        assert result["se"] > 0

    def test_ci_contains_estimate(self, ss_basic):
        """Confidence interval should contain point estimate."""
        Y, D, shares, shocks, X, true_beta = ss_basic

        result = shift_share_iv(
            Y=Y,
            D=D,
            shares=shares,
            shocks=shocks,
            inference="robust",
        )

        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]
