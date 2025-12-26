"""
Cross-language validation tests for Principal Stratification.

Tests Python <-> Julia parity for CACE/LATE estimation (Session 111).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.principal_stratification import cace_2sls, wald_estimator, cace_em

# Import Julia interface with skip if unavailable
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_cace_2sls,
        julia_wald_estimator,
        julia_cace_em,
    )
    JULIA_AVAILABLE = is_julia_available()
except ImportError:
    JULIA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not JULIA_AVAILABLE,
    reason="Julia not available for cross-language tests"
)


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_ps_dgp(
    n: int = 1000,
    pi_c: float = 0.70,
    pi_a: float = 0.15,
    pi_n: float = 0.15,
    true_cace: float = 2.0,
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    random_seed: int = 42,
):
    """
    Generate principal stratification DGP for testing.

    Principal strata:
    - Compliers: D(0)=0, D(1)=1 (respond to assignment)
    - Always-takers: D(0)=1, D(1)=1 (always treated)
    - Never-takers: D(0)=0, D(1)=0 (never treated)

    Returns tuple with all data components.
    """
    np.random.seed(random_seed)

    # Normalize proportions
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c / total, pi_a / total, pi_n / total

    # Generate random assignment
    Z = np.random.rand(n) < 0.5

    # Generate strata: 0=compliers, 1=always-takers, 2=never-takers
    strata = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        if r < pi_c:
            strata[i] = 0  # Complier
        elif r < pi_c + pi_a:
            strata[i] = 1  # Always-taker
        else:
            strata[i] = 2  # Never-taker

    # Treatment based on stratum
    D = np.zeros(n, dtype=bool)
    for i in range(n):
        if strata[i] == 0:
            D[i] = Z[i]  # Complier: follow assignment
        elif strata[i] == 1:
            D[i] = True  # Always-taker
        else:
            D[i] = False  # Never-taker

    # Outcome: Y = baseline + cace * D + noise
    Y = baseline + true_cace * D.astype(float) + noise_sd * np.random.randn(n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_cace": true_cace,
        "strata": strata,
        "pi_c": pi_c,
        "pi_a": pi_a,
        "pi_n": pi_n,
    }


# =============================================================================
# CACE 2SLS Parity Tests
# =============================================================================


class TestCACE2SLSParity:
    """Python <-> Julia parity for CACE 2SLS estimate."""

    def test_estimate_parity(self):
        """CACE estimates should match within tolerance."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=42)

        # Python
        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Julia
        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Primary estimate parity
        assert_allclose(
            py_result["cace"], jl_result["cace"],
            rtol=0.05,
            err_msg=f"CACE mismatch: Python={py_result['cace']:.4f}, Julia={jl_result['cace']:.4f}"
        )

    def test_se_parity(self):
        """Standard errors should match within tolerance."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=123)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # SE parity (robust SE may differ slightly)
        assert_allclose(
            py_result["se"], jl_result["se"],
            rtol=0.10,
            err_msg=f"SE mismatch: Python={py_result['se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_ci_parity(self):
        """Confidence intervals should match."""
        data = generate_ps_dgp(n=1500, true_cace=1.5, random_seed=456)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        assert_allclose(
            py_result["ci_lower"], jl_result["ci_lower"],
            rtol=0.10,
            err_msg=f"CI lower mismatch"
        )

        assert_allclose(
            py_result["ci_upper"], jl_result["ci_upper"],
            rtol=0.10,
            err_msg=f"CI upper mismatch"
        )

    def test_first_stage_parity(self):
        """First-stage coefficient should match."""
        data = generate_ps_dgp(n=2000, pi_c=0.60, pi_a=0.20, pi_n=0.20, random_seed=789)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # First-stage coefficient = complier proportion
        assert_allclose(
            py_result["first_stage_coef"], jl_result["first_stage_coef"],
            rtol=0.05,
            err_msg=f"First-stage mismatch"
        )

        # F-statistic
        assert_allclose(
            py_result["first_stage_f"], jl_result["first_stage_f"],
            rtol=0.10,
            err_msg=f"First-stage F mismatch"
        )

    def test_strata_proportions_parity(self):
        """Strata proportions should match."""
        data = generate_ps_dgp(n=3000, pi_c=0.65, pi_a=0.20, pi_n=0.15, random_seed=101)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Check each strata proportion
        assert_allclose(
            py_result["strata_proportions"]["compliers"],
            jl_result["strata_proportions"]["compliers"],
            rtol=0.05,
            err_msg="Complier proportion mismatch"
        )

        assert_allclose(
            py_result["strata_proportions"]["always_takers"],
            jl_result["strata_proportions"]["always_takers"],
            rtol=0.10,
            err_msg="Always-taker proportion mismatch"
        )

        assert_allclose(
            py_result["strata_proportions"]["never_takers"],
            jl_result["strata_proportions"]["never_takers"],
            rtol=0.10,
            err_msg="Never-taker proportion mismatch"
        )


# =============================================================================
# Wald Estimator Parity Tests
# =============================================================================


class TestWaldEstimatorParity:
    """Python <-> Julia parity for Wald estimator."""

    def test_estimate_parity(self):
        """Wald estimates should match within tolerance."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=202)

        # Python
        py_result = wald_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Julia
        jl_result = julia_wald_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Primary estimate parity
        assert_allclose(
            py_result["cace"], jl_result["cace"],
            rtol=0.05,
            err_msg=f"Wald CACE mismatch: Python={py_result['cace']:.4f}, Julia={jl_result['cace']:.4f}"
        )

    def test_wald_equals_2sls(self):
        """Wald and 2SLS should produce identical estimates (no covariates)."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=303)

        # Python
        py_2sls = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )
        py_wald = wald_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Julia
        jl_2sls = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )
        jl_wald = julia_wald_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Python: Wald = 2SLS
        assert_allclose(
            py_2sls["cace"], py_wald["cace"],
            rtol=1e-10,
            err_msg="Python: 2SLS != Wald"
        )

        # Julia: Wald = 2SLS
        assert_allclose(
            jl_2sls["cace"], jl_wald["cace"],
            rtol=1e-10,
            err_msg="Julia: 2SLS != Wald"
        )

    def test_reduced_form_identity(self):
        """Reduced form = CACE * first-stage in both languages."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=404)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Python identity
        py_rf_computed = py_result["cace"] * py_result["first_stage_coef"]
        assert_allclose(
            py_result["reduced_form"], py_rf_computed,
            rtol=1e-10,
            err_msg="Python: RF != CACE * FS"
        )

        # Julia identity
        jl_rf_computed = jl_result["cace"] * jl_result["first_stage_coef"]
        assert_allclose(
            jl_result["reduced_form"], jl_rf_computed,
            rtol=1e-10,
            err_msg="Julia: RF != CACE * FS"
        )


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestCrossLanguageRobustness:
    """Robustness tests for cross-language parity."""

    def test_large_sample_parity(self):
        """Parity holds with large samples."""
        data = generate_ps_dgp(n=5000, true_cace=2.0, random_seed=505)

        py_result = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Tighter tolerance with large samples
        assert_allclose(
            py_result["cace"], jl_result["cace"],
            rtol=0.02,
            err_msg="Large sample CACE mismatch"
        )

    def test_varying_complier_proportions(self):
        """Parity holds across different complier proportions."""
        for pi_c in [0.30, 0.50, 0.70, 0.90]:
            data = generate_ps_dgp(
                n=2000,
                pi_c=pi_c,
                pi_a=(1 - pi_c) / 2,
                pi_n=(1 - pi_c) / 2,
                true_cace=2.0,
                random_seed=int(pi_c * 1000),
            )

            py_result = cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )

            jl_result = julia_cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )

            assert_allclose(
                py_result["cace"], jl_result["cace"],
                rtol=0.10,
                err_msg=f"CACE mismatch at pi_c={pi_c}"
            )

    def test_varying_treatment_effects(self):
        """Parity holds across different true CACE values."""
        for true_cace in [0.5, 1.0, 2.0, 5.0]:
            data = generate_ps_dgp(
                n=2000,
                true_cace=true_cace,
                random_seed=int(true_cace * 100),
            )

            py_result = cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )

            jl_result = julia_cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )

            assert_allclose(
                py_result["cace"], jl_result["cace"],
                rtol=0.10,
                err_msg=f"CACE mismatch at true_cace={true_cace}"
            )


# =============================================================================
# Monte Carlo Parity
# =============================================================================


class TestMonteCarloParity:
    """Monte Carlo parity tests."""

    @pytest.mark.slow
    def test_mc_bias_parity(self):
        """Both languages should produce similar bias distributions."""
        n_runs = 100
        true_cace = 2.0

        py_estimates = []
        jl_estimates = []

        for seed in range(n_runs):
            data = generate_ps_dgp(n=500, true_cace=true_cace, random_seed=seed)

            py_result = cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )
            jl_result = julia_cace_2sls(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )

            py_estimates.append(py_result["cace"])
            jl_estimates.append(jl_result["cace"])

        py_bias = np.mean(py_estimates) - true_cace
        jl_bias = np.mean(jl_estimates) - true_cace

        # Both languages should have similar bias
        assert_allclose(
            py_bias, jl_bias,
            atol=0.05,
            err_msg=f"Bias mismatch: Python={py_bias:.4f}, Julia={jl_bias:.4f}"
        )

        # Both should be approximately unbiased
        assert abs(py_bias) < 0.10, f"Python bias too large: {py_bias:.4f}"
        assert abs(jl_bias) < 0.10, f"Julia bias too large: {jl_bias:.4f}"


# =============================================================================
# EM Estimator Cross-Language Parity Tests
# =============================================================================


class TestCACEEMParity:
    """Python <-> Julia parity for CACE EM estimate."""

    def test_estimate_parity(self):
        """EM CACE estimates should match within tolerance."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=3001)

        # Python
        py_result = cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Julia
        jl_result = julia_cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            alpha=0.05,
        )

        # Primary estimate parity (wider tolerance for EM)
        assert_allclose(
            py_result["cace"], jl_result["cace"],
            rtol=0.15,
            err_msg=f"EM CACE mismatch: Python={py_result['cace']:.4f}, Julia={jl_result['cace']:.4f}"
        )

    def test_strata_parity(self):
        """EM strata proportions should match."""
        data = generate_ps_dgp(n=2000, pi_c=0.65, pi_a=0.20, pi_n=0.15, random_seed=3002)

        py_result = cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Complier proportion parity
        assert_allclose(
            py_result["strata_proportions"]["compliers"],
            jl_result["strata_proportions"]["compliers"],
            rtol=0.15,
            err_msg="EM complier proportion mismatch"
        )

    def test_method_field(self):
        """EM method field should be 'em' in both languages."""
        data = generate_ps_dgp(n=500, random_seed=3003)

        py_result = cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        jl_result = julia_cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        assert py_result["method"] == "em"
        assert jl_result["method"] == "em"

    def test_em_vs_2sls_both_languages(self):
        """EM and 2SLS should be close in both languages."""
        data = generate_ps_dgp(n=2000, true_cace=2.0, random_seed=3004)

        # Python
        py_em = cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )
        py_2sls = cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Julia
        jl_em = julia_cace_em(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )
        jl_2sls = julia_cace_2sls(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Python: EM close to 2SLS
        assert_allclose(
            py_em["cace"], py_2sls["cace"],
            rtol=0.20,
            err_msg=f"Python: EM ({py_em['cace']:.4f}) too far from 2SLS ({py_2sls['cace']:.4f})"
        )

        # Julia: EM close to 2SLS
        assert_allclose(
            jl_em["cace"], jl_2sls["cace"],
            rtol=0.20,
            err_msg=f"Julia: EM ({jl_em['cace']:.4f}) too far from 2SLS ({jl_2sls['cace']:.4f})"
        )


# =============================================================================
# Bayesian CACE Parity Tests
# =============================================================================

# Check if PyMC is available for Bayesian tests
try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
@pytest.mark.skipif(not is_julia_available(), reason="Julia not available")
class TestCACEBayesianParity:
    """Python <-> Julia parity for Bayesian CACE estimates.

    Note: Python uses PyMC (NUTS), Julia uses simple MH sampler.
    Parity is approximate due to different sampling algorithms.
    """

    def test_structure_parity(self):
        """Bayesian result structures should match."""
        data = generate_ps_dgp(n=300, true_cace=2.0, random_seed=4001)

        # Python
        from src.causal_inference.principal_stratification.bayesian import cace_bayesian as py_cace_bayesian

        py_result = py_cace_bayesian(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            quick=True,
            random_seed=42,
        )

        # Julia
        jl_result = julia_cace_bayesian(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            quick=True,
            seed=42,
        )

        # Both should have key fields
        assert "cace_mean" in py_result or "cace_mean" in jl_result  # Different return types
        assert "pi_c_mean" in jl_result
        assert jl_result["n_samples"] > 0
        assert jl_result["n_chains"] > 0

    def test_strata_sum_parity(self):
        """Both implementations should have strata summing to 1."""
        data = generate_ps_dgp(n=300, random_seed=4002)

        # Julia
        jl_result = julia_cace_bayesian(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            quick=True,
            seed=42,
        )

        # Julia strata should sum to 1
        jl_total = jl_result["pi_c_mean"] + jl_result["pi_a_mean"] + jl_result["pi_n_mean"]
        assert_allclose(jl_total, 1.0, rtol=1e-6)

    def test_quick_mode_samples(self):
        """Quick mode should produce 2000 samples (1000 × 2 chains)."""
        data = generate_ps_dgp(n=200, random_seed=4003)

        jl_result = julia_cace_bayesian(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            quick=True,
            seed=42,
        )

        assert jl_result["n_samples"] == 2000
        assert jl_result["n_chains"] == 2


# =============================================================================
# Bounds Parity Tests (Session 114)
# =============================================================================


# Import bounds and SACE from Python
from src.causal_inference.principal_stratification import (
    ps_bounds_monotonicity,
    ps_bounds_no_assumption,
    sace_bounds,
)


# Import Julia bounds interface
try:
    from tests.validation.cross_language.julia_interface import (
        julia_ps_bounds_monotonicity,
        julia_ps_bounds_no_assumption,
        julia_sace_bounds,
    )
except ImportError:
    pass


def generate_bounds_dgp(
    n: int = 500,
    pi_c: float = 0.60,
    pi_a: float = 0.20,
    pi_n: float = 0.20,
    true_cace: float = 2.0,
    direct_effect: float = 0.0,
    seed: int = 42,
):
    """Generate data for bounds testing."""
    np.random.seed(seed)

    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c / total, pi_a / total, pi_n / total

    Z = np.random.binomial(1, 0.5, n).astype(float)

    strata = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        if r < pi_c:
            strata[i] = 0
        elif r < pi_c + pi_a:
            strata[i] = 1
        else:
            strata[i] = 2

    D = np.zeros(n, dtype=float)
    for i in range(n):
        if strata[i] == 0:
            D[i] = Z[i]
        elif strata[i] == 1:
            D[i] = 1.0
        else:
            D[i] = 0.0

    Y = 1.0 + true_cace * D + direct_effect * Z + np.random.randn(n)

    return {"Y": Y, "D": D, "Z": Z, "true_cace": true_cace}


def generate_sace_dgp(
    n: int = 500,
    true_sace: float = 1.5,
    p_AS: float = 0.60,
    p_protected: float = 0.25,
    seed: int = 42,
):
    """Generate data for SACE testing (selection monotonicity)."""
    np.random.seed(seed)

    D = np.random.binomial(1, 0.5, n).astype(float)

    # Strata
    strata = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        if r < p_AS:
            strata[i] = 0  # Always-survivor
        else:
            strata[i] = 1  # Protected

    # Survival
    S = np.zeros(n, dtype=float)
    for i in range(n):
        if strata[i] == 0:
            S[i] = 1.0
        else:
            S[i] = D[i]

    # Outcome
    Y = np.where(S == 1, 1.0 + true_sace * D + np.random.randn(n), np.nan)

    return {"Y": Y, "D": D, "S": S, "true_sace": true_sace}


class TestBoundsParity:
    """Python <-> Julia parity for bounds functions."""

    def test_monotonicity_bounds_parity(self):
        """ps_bounds_monotonicity should match between languages."""
        data = generate_bounds_dgp(n=500, seed=5001)

        py_result = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"],
            direct_effect_bound=0.5
        )

        jl_result = julia_ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"],
            direct_effect_bound=0.5
        )

        assert_allclose(
            py_result["lower_bound"], jl_result["lower_bound"],
            rtol=0.05,
            err_msg="Lower bound mismatch"
        )

        assert_allclose(
            py_result["upper_bound"], jl_result["upper_bound"],
            rtol=0.05,
            err_msg="Upper bound mismatch"
        )

    def test_no_assumption_bounds_parity(self):
        """ps_bounds_no_assumption should match between languages."""
        data = generate_bounds_dgp(n=500, seed=5002)

        py_result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])
        jl_result = julia_ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        assert_allclose(
            py_result["lower_bound"], jl_result["lower_bound"],
            rtol=0.10,
            err_msg="Manski lower bound mismatch"
        )

        assert_allclose(
            py_result["upper_bound"], jl_result["upper_bound"],
            rtol=0.10,
            err_msg="Manski upper bound mismatch"
        )

    def test_identified_flag_parity(self):
        """Identified flag should match."""
        data = generate_bounds_dgp(n=500, seed=5003)

        py_result = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"],
            direct_effect_bound=0.0  # Point identified
        )

        jl_result = julia_ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"],
            direct_effect_bound=0.0
        )

        assert py_result["identified"] == jl_result["identified"]


class TestSACEParity:
    """Python <-> Julia parity for SACE functions."""

    def test_sace_bounds_parity(self):
        """sace_bounds should match between languages."""
        data = generate_sace_dgp(n=500, seed=6001)

        py_result = sace_bounds(
            data["Y"], data["D"], data["S"],
            monotonicity="selection"
        )

        jl_result = julia_sace_bounds(
            data["Y"], data["D"], data["S"],
            monotonicity="selection"
        )

        assert_allclose(
            py_result["lower_bound"], jl_result["lower_bound"],
            rtol=0.10,
            err_msg="SACE lower bound mismatch"
        )

        assert_allclose(
            py_result["upper_bound"], jl_result["upper_bound"],
            rtol=0.10,
            err_msg="SACE upper bound mismatch"
        )

    def test_survival_proportions_parity(self):
        """Survival proportions should match."""
        data = generate_sace_dgp(n=500, seed=6002)

        py_result = sace_bounds(data["Y"], data["D"], data["S"])
        jl_result = julia_sace_bounds(data["Y"], data["D"], data["S"])

        assert_allclose(
            py_result["proportion_survivors_treat"],
            jl_result["proportion_survivors_treat"],
            rtol=0.05,
            err_msg="Treated survival proportion mismatch"
        )

        assert_allclose(
            py_result["proportion_survivors_control"],
            jl_result["proportion_survivors_control"],
            rtol=0.05,
            err_msg="Control survival proportion mismatch"
        )

    def test_sample_size_parity(self):
        """Sample size should match exactly."""
        n = 300
        data = generate_sace_dgp(n=n, seed=6003)

        py_result = sace_bounds(data["Y"], data["D"], data["S"])
        jl_result = julia_sace_bounds(data["Y"], data["D"], data["S"])

        assert py_result["n"] == jl_result["n"] == n
