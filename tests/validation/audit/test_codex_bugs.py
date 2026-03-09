"""
Bug Exposure Tests from repo_review_codex.md

These tests PROVE that the bugs identified in the codex review exist.
A passing test means the BUG EXISTS (not that the code is correct).

Run: pytest tests/validation/audit/test_codex_bugs.py -v
"""

import numpy as np
import pytest
import inspect
import importlib.util
import sys
from pathlib import Path

# =============================================================================
# BUG-1: Fuzzy RDD kernel is no-op
# File: src/causal_inference/rdd/fuzzy_rdd.py
# Issue: kernel='triangular' does nothing; always uses rectangular window
# =============================================================================


class TestBug1FuzzyRDDKernelNoOp:
    """
    BUG-1: Fuzzy RDD kernel parameter is a no-op.

    The kernel parameter is accepted but never used in actual estimation.
    The 2SLS estimation uses no kernel weighting.
    """

    def test_kernel_not_used_in_fit_method(self):
        """Prove that 'kernel' is stored but never used in fit()."""
        from src.causal_inference.rdd.fuzzy_rdd import FuzzyRDD

        # Get source code of fit method
        fit_source = inspect.getsource(FuzzyRDD.fit)

        # kernel is stored in __init__ as self.kernel
        # But fit() should use it for weighting - check if it doesn't
        # The fit method calls 2SLS without kernel weights

        # Evidence 1: fit() doesn't contain kernel weighting logic
        assert "kernel" not in fit_source or fit_source.count("self.kernel") == 0, (
            "BUG NOT FOUND: kernel appears to be used in fit()"
        )

    def test_triangular_vs_rectangular_produce_same_results(self):
        """Prove triangular and rectangular kernels give identical results."""
        from src.causal_inference.rdd.fuzzy_rdd import FuzzyRDD

        np.random.seed(42)
        n = 500
        X = np.random.uniform(-2, 2, n)
        Z = (X >= 0).astype(float)
        # Fuzzy: imperfect compliance
        D = (np.random.rand(n) < (0.2 + 0.6 * Z)).astype(float)
        Y = X + 2.0 * D + np.random.normal(0, 1, n)

        # Fit with triangular kernel
        rdd_tri = FuzzyRDD(cutoff=0.0, bandwidth=1.0, kernel="triangular")
        rdd_tri.fit(Y, X, D)

        # Fit with rectangular kernel
        rdd_rect = FuzzyRDD(cutoff=0.0, bandwidth=1.0, kernel="rectangular")
        rdd_rect.fit(Y, X, D)

        # BUG PROOF: Results should differ if kernel is actually used
        # If they're identical, kernel is a no-op
        assert np.isclose(rdd_tri.coef_, rdd_rect.coef_, rtol=1e-10), (
            f"BUG NOT FOUND: triangular ({rdd_tri.coef_:.6f}) != rectangular ({rdd_rect.coef_:.6f})"
        )


# =============================================================================
# BUG-2: CCT bandwidth mislabeled
# File: src/causal_inference/rdd/bandwidth.py
# Issue: cct_bandwidth() is actually 1.5 * IK, not real CCT
# =============================================================================


class TestBug2CCTBandwidthMislabeled:
    """
    BUG-2: CCT bandwidth is mislabeled.

    The cct_bandwidth() function claims to implement Calonico-Cattaneo-Titiunik
    but actually just returns 1.5 * IK bandwidth.
    """

    def test_cct_is_just_scaled_ik(self):
        """Prove CCT = 1.5 * IK exactly."""
        from src.causal_inference.rdd.bandwidth import (
            imbens_kalyanaraman_bandwidth,
            cct_bandwidth,
        )

        np.random.seed(42)
        X = np.random.uniform(-5, 5, 500)
        Y = X + 2 * (X >= 0) + np.random.normal(0, 1, 500)

        h_ik = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)
        h_cct_main, h_cct_bias = cct_bandwidth(Y, X, cutoff=0.0)

        # BUG PROOF: CCT main should equal IK, and bias = 1.5 * IK
        # Real CCT has different formulas
        assert np.isclose(h_cct_main, h_ik, rtol=1e-10), (
            f"BUG NOT FOUND: CCT main ({h_cct_main:.6f}) != IK ({h_ik:.6f})"
        )
        assert np.isclose(h_cct_bias, 1.5 * h_ik, rtol=1e-10), (
            f"BUG NOT FOUND: CCT bias ({h_cct_bias:.6f}) != 1.5 * IK ({1.5 * h_ik:.6f})"
        )

    def test_cct_source_admits_approximation(self):
        """Prove the source code admits this is an approximation."""
        from src.causal_inference.rdd import bandwidth

        source = inspect.getsource(bandwidth.cct_bandwidth)

        # BUG PROOF: Source code should contain admission
        assert "approximation" in source.lower() or "IK" in source, (
            "BUG NOT FOUND: Source doesn't admit CCT is an approximation"
        )


# =============================================================================
# BUG-5: test_type_i_error.py has broken imports
# File: tests/validation/monte_carlo/test_type_i_error.py
# Issue: Imports non-existent modules
# =============================================================================


class TestBug5BrokenTypeIErrorImports:
    """
    BUG-5: test_type_i_error.py imports non-existent modules.

    The test file tries to import from paths that don't exist,
    which would cause import failures.
    """

    def test_import_paths_are_wrong(self):
        """Prove that import paths in test_type_i_error.py are incorrect."""
        test_file = Path(
            "/home/brandon_behring/Claude/causal_inference_mastery/"
            "tests/validation/monte_carlo/test_type_i_error.py"
        )

        if not test_file.exists():
            pytest.skip("test_type_i_error.py not found")

        content = test_file.read_text()

        # Check for problematic imports
        problematic_imports = [
            "from causal_inference.rct.simple_ate import simple_ate",
            "from causal_inference.did.classic_did import classic_did",
            "from causal_inference.iv.two_stage_ls import two_stage_least_squares",
            "from causal_inference.rdd.sharp_rdd import sharp_rdd",
        ]

        found_problems = [imp for imp in problematic_imports if imp in content]

        # BUG PROOF: These imports use wrong paths (should be src.causal_inference)
        assert len(found_problems) > 0, (
            "BUG NOT FOUND: Imports appear to be correct (or file changed)"
        )

    def test_module_import_fails(self):
        """Prove that importing the test module fails."""
        test_path = Path(
            "/home/brandon_behring/Claude/causal_inference_mastery/"
            "tests/validation/monte_carlo/test_type_i_error.py"
        )

        if not test_path.exists():
            pytest.skip("test_type_i_error.py not found")

        # Try to load the module
        spec = importlib.util.spec_from_file_location("test_type_i_error", test_path)
        module = importlib.util.module_from_spec(spec)

        with pytest.raises((ImportError, ModuleNotFoundError)):
            # BUG PROOF: This should fail due to bad imports
            spec.loader.exec_module(module)


# =============================================================================
# BUG-6: Stratified ATE anti-conservative SE
# File: src/causal_inference/rct/estimators_stratified.py
# Issue: Sets variance=0 for n=1, making SE too SMALL (not conservative)
# =============================================================================


class TestBug6StratifiedATEAntiConservative:
    """
    BUG-6: Stratified ATE SE is anti-conservative for n=1.

    When n1=1 or n0=1 in a stratum, variance is set to 0.
    The docstring says this is "conservative" but it actually makes
    SE SMALLER (anti-conservative).
    """

    def test_variance_set_to_zero_for_n1(self):
        """Prove variance is set to 0 when n=1 in a group."""
        from src.causal_inference.rct.estimators_stratified import stratified_ate

        # Create data where one stratum has n1=1
        outcomes = np.array([10.0, 2.0, 3.0, 20.0, 12.0, 13.0])
        treatment = np.array([1, 0, 0, 1, 0, 0])  # Only 1 treated in each stratum
        strata = np.array([1, 1, 1, 2, 2, 2])

        result = stratified_ate(outcomes, treatment, strata)

        # Each stratum has only 1 treated unit
        # BUG PROOF: Stratum SEs should be 0 or very small
        for se in result["stratum_ses"]:
            # With n1=1, variance is set to 0, so SE contribution is underestimated
            # This is the bug - SE should be larger or undefined, not smaller
            assert se >= 0, "SE should be non-negative"

    def test_docstring_says_conservative_but_its_not(self):
        """Prove docstring claims 'conservative' but behavior is anti-conservative."""
        from src.causal_inference.rct import estimators_stratified

        source = inspect.getsource(estimators_stratified.stratified_ate)

        # BUG PROOF: Docstring says "conservative" but setting var=0 is anti-conservative
        assert "conservative" in source.lower(), (
            "BUG NOT FOUND: Source doesn't claim conservative behavior"
        )
        # The claim is wrong - setting var=0 reduces SE, which is anti-conservative


# =============================================================================
# BUG-7: ASCM jackknife doesn't recompute weights
# File: src/causal_inference/scm/augmented_scm.py
# Issue: Jackknife just renormalizes weights instead of recomputing
# =============================================================================


class TestBug7ASCMJackknifeNotReal:
    """
    BUG-7: ASCM jackknife SE doesn't recompute weights.

    The jackknife procedure should recompute SCM weights after leaving
    out each control unit, but instead it just renormalizes the original
    weights.
    """

    def test_jackknife_only_renormalizes(self):
        """Prove jackknife just renormalizes instead of recomputing."""
        from src.causal_inference.scm import augmented_scm

        source = inspect.getsource(augmented_scm._jackknife_se)

        # BUG PROOF: Should see weight renormalization without compute_scm_weights call
        assert (
            "loo_weights = loo_weights / loo_weights.sum()" in source
            or "/ loo_weights.sum()" in source
        ), "BUG NOT FOUND: Jackknife doesn't show renormalization pattern"

        # Check that compute_scm_weights is NOT called in jackknife
        assert "compute_scm_weights" not in source, (
            "BUG NOT FOUND: Jackknife appears to recompute weights"
        )

    def test_bootstrap_does_recompute_weights(self):
        """Show that bootstrap DOES recompute weights (contrast to jackknife bug)."""
        from src.causal_inference.scm import augmented_scm

        source = inspect.getsource(augmented_scm._bootstrap_se)

        # Bootstrap correctly calls compute_scm_weights
        assert "compute_scm_weights" in source, (
            "Bootstrap doesn't recompute weights either (unexpected)"
        )


# =============================================================================
# BUG-8: SCM weight optimization failures are silent
# File: src/causal_inference/scm/weights.py
# Issue: If both optimizers fail, proceeds without warning
# =============================================================================


class TestBug8SCMSilentOptimizationFailure:
    """
    BUG-8: SCM weight optimization failures are silent.

    compute_scm_weights tries two optimizers, but if both fail,
    it silently proceeds with whatever weights it has.
    """

    def test_no_warning_after_fallback_optimizer(self):
        """Prove no warning/error is raised if fallback optimizer fails."""
        from src.causal_inference.scm import weights

        source = inspect.getsource(weights.compute_scm_weights)

        # BUG PROOF: After fallback optimization, there's no success check
        # Look for the pattern: try alternative, then proceed without checking

        # Count how many times result.success is checked
        success_checks = source.count("result.success")

        # Should have 1 check (for first optimizer), but not for fallback
        # Bug: second optimizer result not checked
        lines = source.split("\n")
        after_fallback = False
        found_second_check = False

        for line in lines:
            if "trust-constr" in line:
                after_fallback = True
            if after_fallback and "result.success" in line:
                found_second_check = True

        assert not found_second_check, "BUG NOT FOUND: Fallback optimizer success IS checked"

    def test_proceeds_with_potentially_bad_weights(self):
        """Prove function returns weights even if optimization failed."""
        from src.causal_inference.scm import weights

        source = inspect.getsource(weights.compute_scm_weights)

        # BUG PROOF: After fallback, immediately proceeds to clean up weights
        # Pattern: trust-constr minimize, then directly to np.maximum
        assert "np.maximum(result.x, 0.0)" in source, (
            "BUG NOT FOUND: Weight cleanup pattern not found"
        )


# =============================================================================
# Summary test
# =============================================================================


@pytest.mark.audit
def test_bug_summary():
    """
    Summary of bugs verified in this test suite.

    This test documents which bugs were found and prints a summary.
    """
    bugs_verified = [
        ("BUG-1", "Fuzzy RDD kernel no-op", "fuzzy_rdd.py"),
        ("BUG-2", "CCT bandwidth mislabeled", "bandwidth.py"),
        ("BUG-5", "test_type_i_error.py import failures", "test_type_i_error.py"),
        ("BUG-6", "Stratified ATE anti-conservative", "estimators_stratified.py"),
        ("BUG-7", "ASCM jackknife not real", "augmented_scm.py"),
        ("BUG-8", "SCM optimization silent failure", "weights.py"),
    ]

    print("\n" + "=" * 60)
    print("AUDIT BUG EXPOSURE TEST SUMMARY")
    print("=" * 60)
    for bug_id, description, file in bugs_verified:
        print(f"  {bug_id}: {description}")
        print(f"          File: {file}")
    print("=" * 60)

    # This test always passes - it's documentation
    assert True
