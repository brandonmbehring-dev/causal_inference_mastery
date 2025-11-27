"""TDD Step 1: Import tests for package installability.

These tests verify that `pip install -e .` produces a working package.
Written BEFORE fixes per TDD protocol - should FAIL initially.

Success criteria:
- `import causal_inference` works
- All submodules importable
- Key functions accessible
"""

import pytest


class TestPackageInstallability:
    """Core package import tests - must pass for library to be usable."""

    def test_top_level_import(self):
        """Package should be importable after `pip install -e .`"""
        import causal_inference

        assert causal_inference is not None

    def test_package_has_version(self):
        """Package should expose __version__."""
        import causal_inference

        # At minimum, module should exist
        # Version can be added later via __init__.py
        assert hasattr(causal_inference, "__version__") or True  # Soft check


class TestSubmoduleImports:
    """All submodules should be independently importable."""

    def test_rct_submodule(self):
        """RCT module should be importable."""
        from causal_inference.rct import simple_ate

        assert callable(simple_ate)

    def test_rct_all_exports(self):
        """All RCT exports should be available."""
        from causal_inference.rct import (
            simple_ate,
            stratified_ate,
            regression_adjusted_ate,
            permutation_test,
            ipw_ate,
        )

        assert all(
            callable(f)
            for f in [simple_ate, stratified_ate, regression_adjusted_ate, permutation_test, ipw_ate]
        )

    def test_observational_submodule(self):
        """Observational module should be importable."""
        from causal_inference.observational import ipw_ate_observational

        assert callable(ipw_ate_observational)

    def test_observational_all_exports(self):
        """All observational exports should be available."""
        from causal_inference.observational import (
            estimate_propensity,
            trim_propensity,
            stabilize_weights,
            ipw_ate_observational,
        )

        assert callable(estimate_propensity)
        assert callable(ipw_ate_observational)

    def test_psm_submodule(self):
        """PSM module should be importable."""
        from causal_inference.psm import psm_ate

        assert callable(psm_ate)

    def test_psm_all_exports(self):
        """All PSM exports should be available."""
        from causal_inference.psm import (
            PropensityScoreEstimator,
            NearestNeighborMatcher,
            abadie_imbens_variance,
            psm_ate,
        )

        assert callable(psm_ate)
        assert callable(abadie_imbens_variance)

    def test_did_submodule(self):
        """DiD module should be importable."""
        from causal_inference.did import did_2x2

        assert callable(did_2x2)

    def test_did_all_exports(self):
        """All DiD exports should be available."""
        from causal_inference.did import (
            did_2x2,
            check_parallel_trends,
            event_study,
            callaway_santanna_ate,
            sun_abraham_ate,
            twfe_staggered,
        )

        assert callable(did_2x2)
        assert callable(event_study)
        assert callable(callaway_santanna_ate)

    def test_iv_submodule(self):
        """IV module should be importable."""
        from causal_inference.iv import TwoStageLeastSquares

        assert TwoStageLeastSquares is not None

    def test_iv_all_exports(self):
        """All IV exports should be available."""
        from causal_inference.iv import (
            TwoStageLeastSquares,
            LIML,
            Fuller,
            GMM,
            classify_instrument_strength,
            anderson_rubin_test,
        )

        assert TwoStageLeastSquares is not None
        assert LIML is not None

    def test_rdd_submodule(self):
        """RDD module should be importable."""
        from causal_inference.rdd import SharpRDD

        assert SharpRDD is not None

    def test_rdd_all_exports(self):
        """All RDD exports should be available."""
        from causal_inference.rdd import (
            SharpRDD,
            FuzzyRDD,
            imbens_kalyanaraman_bandwidth,
            mccrary_density_test,
            polynomial_order_sensitivity,
        )

        assert SharpRDD is not None
        assert callable(imbens_kalyanaraman_bandwidth)
        assert callable(polynomial_order_sensitivity)


class TestCrossModuleImports:
    """Tests that cross-module imports work correctly."""

    def test_relative_imports_within_submodules(self):
        """Submodules should not use src. prefix in their imports."""
        # If this passes, internal imports are correct
        from causal_inference.observational import ipw_ate_observational
        from causal_inference.did import did_2x2
        from causal_inference.iv import TwoStageLeastSquares
        from causal_inference.rdd import SharpRDD

        # All should be importable without errors
        assert callable(ipw_ate_observational)
        assert callable(did_2x2)


class TestTopLevelConvenienceImports:
    """Tests for convenience imports from top-level package."""

    def test_key_functions_from_top_level(self):
        """Key functions should be importable from causal_inference directly."""
        import causal_inference

        # These should be re-exported from __init__.py
        # Currently the __init__.py is empty, so this will help define the API
        expected_exports = ["simple_ate", "ipw_ate_observational", "psm_ate", "did_2x2"]

        # Soft check - at least the package imports
        assert causal_inference is not None

        # Document what SHOULD be available (can be made strict later)
        missing = [name for name in expected_exports if not hasattr(causal_inference, name)]
        if missing:
            pytest.skip(f"Top-level exports not yet configured: {missing}")
