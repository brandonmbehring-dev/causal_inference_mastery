"""R Triangulation Tests for Bunching Estimators.

Validates our Python bunching implementation against R's bunchr package.
Tests focus on excess mass estimation, elasticity calculation, and
counterfactual polynomial fitting.

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Optimization frictions
- Kleven (2016) - Bunching estimation review
"""

import numpy as np
import pytest
from typing import Tuple

# Import R interface
from .r_interface import (
    check_r_available,
    check_bunchr_installed,
    r_bunching_estimate,
    r_bunching_elasticity,
)

# Import Python implementation
from causal_inference.bunching import (
    bunching_estimator,
    compute_excess_mass,
    compute_elasticity,
    estimate_counterfactual,
)

# Import DGP functions
import sys
from pathlib import Path

# Add monte_carlo directory to path for DGP imports
monte_carlo_path = Path(__file__).parent.parent / "monte_carlo"
if str(monte_carlo_path) not in sys.path:
    sys.path.insert(0, str(monte_carlo_path))

from dgp_bunching import (
    dgp_bunching_simple,
    dgp_bunching_uniform_counterfactual,
    dgp_bunching_no_effect,
    dgp_bunching_with_elasticity,
    dgp_bunching_diffuse,
    dgp_bunching_large_sample,
)


# =============================================================================
# Skip Conditions
# =============================================================================

pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available",
)

requires_bunchr = pytest.mark.skipif(
    not check_bunchr_installed(),
    reason="R 'bunchr' package not installed. Install with: install.packages('bunchr')",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bunching_basic() -> Tuple[np.ndarray, float, float, float]:
    """Basic bunching data with moderate excess mass."""
    return dgp_bunching_simple(
        n=2000,
        kink_point=50.0,
        true_excess_mass=2.0,
        counterfactual_std=15.0,
        bunching_std=1.0,
        random_state=42,
    )


@pytest.fixture
def bunching_uniform() -> Tuple[np.ndarray, float, float, float]:
    """Bunching with uniform counterfactual (simpler h0)."""
    return dgp_bunching_uniform_counterfactual(
        n=2000,
        kink_point=50.0,
        data_range=(20.0, 80.0),
        buncher_fraction=0.15,
        bunching_std=1.0,
        random_state=42,
    )


@pytest.fixture
def bunching_null() -> Tuple[np.ndarray, float, float, float]:
    """No bunching (null effect for Type I error testing)."""
    return dgp_bunching_no_effect(
        n=2000,
        kink_point=50.0,
        counterfactual_std=15.0,
        random_state=42,
    )


@pytest.fixture
def bunching_with_rates() -> Tuple[np.ndarray, float, float, float, float, float]:
    """Bunching data with tax rates for elasticity estimation."""
    return dgp_bunching_with_elasticity(
        n=2000,
        kink_point=50000.0,
        t1_rate=0.20,
        t2_rate=0.30,
        true_elasticity=0.25,
        counterfactual_std=12000.0,
        random_state=42,
    )


@pytest.fixture
def bunching_large() -> Tuple[np.ndarray, float, float, float]:
    """Large sample for precision testing."""
    return dgp_bunching_large_sample(
        n=10000,
        kink_point=50.0,
        true_excess_mass=1.5,
        counterfactual_std=15.0,
        bunching_std=1.0,
        random_state=42,
    )


@pytest.fixture
def bunching_diffuse() -> Tuple[np.ndarray, float, float, float]:
    """Diffuse bunching (optimization frictions)."""
    return dgp_bunching_diffuse(
        n=2000,
        kink_point=50.0,
        buncher_fraction=0.15,
        bunching_std=5.0,
        counterfactual_std=15.0,
        random_state=42,
    )


# =============================================================================
# Test Classes
# =============================================================================


@requires_bunchr
class TestBunchingExcessMassVsR:
    """Test excess mass estimation parity with R bunchr."""

    def test_basic_excess_mass_parity(self, bunching_basic):
        """Basic excess mass estimate should match R within tolerance."""
        data, true_b, kink, bw = bunching_basic
        bin_width = 2.0

        # Python estimate
        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=50,
        )

        # R estimate
        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=7,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=50,
            seed=42,
        )

        if r_result is None:
            pytest.skip("R bunchr returned None (package issue)")

        # Tolerance: rtol=0.15 due to different binning edge handling
        assert py_result["excess_mass"] is not None
        assert r_result["excess_mass"] is not None
        assert np.isclose(
            py_result["excess_mass"],
            r_result["excess_mass"],
            rtol=0.15,
        ), (
            f"Excess mass mismatch: Python={py_result['excess_mass']:.4f}, "
            f"R={r_result['excess_mass']:.4f}"
        )

    def test_uniform_counterfactual_parity(self, bunching_uniform):
        """Uniform counterfactual should give closer agreement."""
        data, true_b, kink, bw = bunching_uniform
        bin_width = 2.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=5,  # Lower order for uniform
            n_bootstrap=50,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=5,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=50,
            seed=42,
        )

        if r_result is None:
            pytest.skip("R bunchr returned None")

        # Tighter tolerance for uniform counterfactual
        assert np.isclose(
            py_result["excess_mass"],
            r_result["excess_mass"],
            rtol=0.12,
        ), (
            f"Uniform excess mass mismatch: Python={py_result['excess_mass']:.4f}, "
            f"R={r_result['excess_mass']:.4f}"
        )

    def test_large_sample_parity(self, bunching_large):
        """Large sample should give tighter agreement."""
        data, true_b, kink, bw = bunching_large
        bin_width = 2.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=30,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=7,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=30,
            seed=42,
        )

        if r_result is None:
            pytest.skip("R bunchr returned None")

        # Tighter tolerance for large sample
        assert np.isclose(
            py_result["excess_mass"],
            r_result["excess_mass"],
            rtol=0.10,
        ), (
            f"Large sample excess mass mismatch: Python={py_result['excess_mass']:.4f}, "
            f"R={r_result['excess_mass']:.4f}"
        )

    def test_null_effect_both_detect_no_bunching(self, bunching_null):
        """Both should return small excess mass when no bunching exists."""
        data, true_b, kink, bw = bunching_null
        bin_width = 2.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=50,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=7,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=50,
            seed=42,
        )

        if r_result is None:
            pytest.skip("R bunchr returned None")

        # Both should be close to 0
        assert abs(py_result["excess_mass"]) < 1.0, (
            f"Python excess mass too large for null: {py_result['excess_mass']:.4f}"
        )
        if r_result["excess_mass"] is not None:
            assert abs(r_result["excess_mass"]) < 1.0, (
                f"R excess mass too large for null: {r_result['excess_mass']:.4f}"
            )


@requires_bunchr
class TestBunchingElasticityVsR:
    """Test elasticity estimation parity with R."""

    def test_elasticity_parity(self, bunching_with_rates):
        """Elasticity estimate should match R within tolerance."""
        data, true_e, true_b, kink, t1, t2 = bunching_with_rates
        bin_width = 1000.0  # Larger bins for tax data

        # Python: estimate excess mass then elasticity
        py_bunch = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=3000.0,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=50,
        )
        py_elasticity = compute_elasticity(
            excess_mass=py_bunch["excess_mass"],
            t1_rate=t1,
            t2_rate=t2,
        )

        # R elasticity
        r_result = r_bunching_elasticity(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            t1_rate=t1,
            t2_rate=t2,
            poly_order=7,
            n_bootstrap=50,
            seed=42,
        )

        if r_result is None:
            pytest.skip("R bunching elasticity returned None")

        # Tolerance: rtol=0.20 due to excess mass differences propagating
        assert np.isclose(
            py_elasticity,
            r_result["elasticity"],
            rtol=0.20,
        ), f"Elasticity mismatch: Python={py_elasticity:.4f}, R={r_result['elasticity']:.4f}"

    def test_elasticity_both_recover_true_value(self, bunching_with_rates):
        """Both Python and R should recover true elasticity approximately."""
        data, true_e, true_b, kink, t1, t2 = bunching_with_rates
        bin_width = 1000.0

        py_bunch = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=3000.0,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=50,
        )
        py_elasticity = compute_elasticity(
            excess_mass=py_bunch["excess_mass"],
            t1_rate=t1,
            t2_rate=t2,
        )

        r_result = r_bunching_elasticity(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            t1_rate=t1,
            t2_rate=t2,
            poly_order=7,
            n_bootstrap=50,
            seed=42,
        )

        # Both should be in reasonable range of true elasticity
        # (DGP has noise, so 50% tolerance)
        assert 0.05 < py_elasticity < 0.50, (
            f"Python elasticity {py_elasticity:.4f} outside plausible range [0.05, 0.50]"
        )
        if r_result is not None:
            assert 0.05 < r_result["elasticity"] < 0.50, (
                f"R elasticity {r_result['elasticity']:.4f} outside plausible range"
            )


@requires_bunchr
class TestBunchingEdgeCases:
    """Test edge cases and challenging scenarios."""

    def test_diffuse_bunching_both_handle(self, bunching_diffuse):
        """Diffuse bunching (frictions) should be handled by both."""
        data, true_b, kink, bw = bunching_diffuse
        bin_width = 3.0  # Wider bins for diffuse bunching

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=5,
            n_bootstrap=50,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=5,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=50,
            seed=42,
        )

        # Both should detect positive excess mass
        assert py_result["excess_mass"] > 0, "Python should detect positive excess mass"
        if r_result is not None and r_result["excess_mass"] is not None:
            assert r_result["excess_mass"] > 0, "R should detect positive excess mass"

    def test_narrow_bunching_region(self, bunching_basic):
        """Narrow bunching region should still work."""
        data, true_b, kink, bw = bunching_basic
        narrow_bw = bw * 0.5  # Half the recommended width
        bin_width = 1.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=narrow_bw,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=30,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=7,
            excluded_lower=kink - narrow_bw,
            excluded_upper=kink + narrow_bw,
            n_bootstrap=30,
            seed=42,
        )

        # Both should still produce estimates
        assert py_result["excess_mass"] is not None
        if r_result is not None:
            assert r_result["excess_mass"] is not None

    def test_wide_bunching_region(self, bunching_basic):
        """Wide bunching region should still work."""
        data, true_b, kink, bw = bunching_basic
        wide_bw = bw * 2.0  # Double the recommended width
        bin_width = 3.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=wide_bw,
            bin_width=bin_width,
            poly_order=7,
            n_bootstrap=30,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=7,
            excluded_lower=kink - wide_bw,
            excluded_upper=kink + wide_bw,
            n_bootstrap=30,
            seed=42,
        )

        # Both should still produce estimates
        assert py_result["excess_mass"] is not None
        if r_result is not None:
            assert r_result["excess_mass"] is not None

    def test_high_polynomial_order(self, bunching_basic):
        """High polynomial order should work."""
        data, true_b, kink, bw = bunching_basic
        bin_width = 2.0

        py_result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=10,  # Higher order
            n_bootstrap=30,
        )

        r_result = r_bunching_estimate(
            data=data,
            kink_point=kink,
            bin_width=bin_width,
            poly_order=10,
            excluded_lower=kink - bw,
            excluded_upper=kink + bw,
            n_bootstrap=30,
            seed=42,
        )

        # Both should handle high polynomial order
        assert py_result["excess_mass"] is not None
        if r_result is not None:
            assert r_result["excess_mass"] is not None


@requires_bunchr
@pytest.mark.slow
class TestBunchingMonteCarloTriangulation:
    """Monte Carlo tests for systematic agreement."""

    def test_monte_carlo_excess_mass_agreement(self):
        """Monte Carlo: Python and R should agree on 80%+ of runs."""
        n_runs = 20
        agreements = 0
        rtol = 0.20  # Relative tolerance for agreement

        for seed in range(n_runs):
            data, true_b, kink, bw = dgp_bunching_simple(
                n=1500,
                kink_point=50.0,
                true_excess_mass=2.0,
                counterfactual_std=15.0,
                bunching_std=1.0,
                random_state=seed,
            )
            bin_width = 2.0

            py_result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=bw,
                bin_width=bin_width,
                poly_order=7,
                n_bootstrap=20,
            )

            r_result = r_bunching_estimate(
                data=data,
                kink_point=kink,
                bin_width=bin_width,
                poly_order=7,
                excluded_lower=kink - bw,
                excluded_upper=kink + bw,
                n_bootstrap=20,
                seed=seed,
            )

            if r_result is None or r_result["excess_mass"] is None:
                continue

            if np.isclose(
                py_result["excess_mass"],
                r_result["excess_mass"],
                rtol=rtol,
            ):
                agreements += 1

        agreement_rate = agreements / n_runs
        assert agreement_rate >= 0.80, f"Monte Carlo agreement rate {agreement_rate:.0%} < 80%"

    def test_monte_carlo_null_effect_type_i(self):
        """Monte Carlo: Both should have similar false positive rates under null."""
        n_runs = 30
        py_false_positives = 0
        r_false_positives = 0
        threshold = 1.5  # Excess mass > 1.5 considered "detection"

        for seed in range(n_runs):
            data, true_b, kink, bw = dgp_bunching_no_effect(
                n=1500,
                kink_point=50.0,
                counterfactual_std=15.0,
                random_state=seed,
            )
            bin_width = 2.0

            py_result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=bw,
                bin_width=bin_width,
                poly_order=7,
                n_bootstrap=20,
            )

            r_result = r_bunching_estimate(
                data=data,
                kink_point=kink,
                bin_width=bin_width,
                poly_order=7,
                excluded_lower=kink - bw,
                excluded_upper=kink + bw,
                n_bootstrap=20,
                seed=seed,
            )

            if abs(py_result["excess_mass"]) > threshold:
                py_false_positives += 1

            if r_result is not None and r_result["excess_mass"] is not None:
                if abs(r_result["excess_mass"]) > threshold:
                    r_false_positives += 1

        py_fp_rate = py_false_positives / n_runs
        r_fp_rate = r_false_positives / n_runs

        # Both should have low false positive rates
        assert py_fp_rate < 0.20, f"Python FP rate {py_fp_rate:.0%} too high"
        # R rate check only if we got results
        if r_false_positives > 0:
            assert r_fp_rate < 0.25, f"R FP rate {r_fp_rate:.0%} too high"


# =============================================================================
# Counterfactual Comparison (without bunchr dependency)
# =============================================================================


class TestCounterfactualParity:
    """Test counterfactual estimation (works without bunchr)."""

    def test_counterfactual_polynomial_fit(self, bunching_basic):
        """Counterfactual polynomial fit should be reasonable."""
        data, true_b, kink, bw = bunching_basic
        bin_width = 2.0

        result = estimate_counterfactual(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=7,
        )

        # Should have reasonable R-squared
        assert result["r_squared"] > 0.80, f"Counterfactual R² = {result['r_squared']:.4f} too low"

        # Counterfactual should be positive
        assert np.all(np.array(result["counterfactual_counts"]) >= 0), (
            "Counterfactual has negative counts"
        )

    def test_counterfactual_uniform_high_fit(self, bunching_uniform):
        """Uniform counterfactual should have high R-squared."""
        data, true_b, kink, bw = bunching_uniform
        bin_width = 2.0

        result = estimate_counterfactual(
            data=data,
            kink_point=kink,
            bunching_width=bw,
            bin_width=bin_width,
            poly_order=3,  # Low order sufficient for uniform
        )

        # Uniform should fit very well
        assert result["r_squared"] > 0.85, (
            f"Uniform counterfactual R² = {result['r_squared']:.4f} too low"
        )
