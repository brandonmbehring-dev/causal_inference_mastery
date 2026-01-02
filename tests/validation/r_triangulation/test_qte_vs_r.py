"""
R Triangulation Tests for Quantile Treatment Effects (QTE).

Compares Python QTE implementation against R quantreg package.

Tests verify:
1. Conditional QTE at standard quantiles (0.25, 0.50, 0.75)
2. Unconditional QTE via RIF regression
3. QTE process across quantile grid
4. Monte Carlo stability

Tolerance Standards
-------------------
| Metric           | Tolerance | Rationale                     |
|------------------|-----------|-------------------------------|
| Conditional QTE  | rtol=0.10 | Same quantreg algorithm       |
| Unconditional QTE| rtol=0.15 | RIF estimation differs        |
| Extreme quantiles| rtol=0.20 | Sparse data at tails          |
| QTE process      | rtol=0.12 | Fine grid adds variance       |

References
----------
- Koenker, R. (2005). Quantile Regression. Cambridge University Press.
- Firpo, S., Fortin, N.M., & Lemieux, T. (2009). Unconditional Quantile
  Regressions. Econometrica, 77(3), 953-973.
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.qte import (
    conditional_qte,
    unconditional_qte,
    rif_qte,
    conditional_qte_band,
)

# Import R interface
try:
    from tests.validation.r_triangulation.r_interface import (
        check_quantreg_installed,
        r_conditional_qte,
        r_unconditional_qte,
        r_qte_process,
    )
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


def check_r_available():
    """Check if R and quantreg are available."""
    if not R_AVAILABLE:
        return False
    try:
        return check_quantreg_installed()
    except Exception:
        return False


# Skip if R/quantreg not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R or quantreg package not available"
)


# =============================================================================
# Data Generation Functions
# =============================================================================


def generate_qte_dgp(
    n: int = 2000,
    seed: int = 42,
    effect_type: str = "homogeneous",
    error_dist: str = "normal",
) -> dict:
    """Generate data for QTE estimation.

    Parameters
    ----------
    n : int
        Sample size.
    seed : int
        Random seed.
    effect_type : str
        Type of treatment effect heterogeneity:
        - "homogeneous": Constant effect across quantiles
        - "location": Location shift only (same for all quantiles)
        - "scale": Treatment affects variance (different QTE at tails)
        - "distributional": Full distribution change
    error_dist : str
        Error distribution: "normal", "heavy_tail", "skewed"

    Returns
    -------
    dict
        Dictionary with:
        - outcome: np.ndarray (n,)
        - treatment: np.ndarray (n,) binary
        - covariates: np.ndarray (n, k)
        - true_qte: Dict[float, float] - true QTE at select quantiles
    """
    np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 2)
    X[:, 0] = X[:, 0] * 0.5 + 1.0

    # Treatment assignment (randomized)
    D = np.random.binomial(1, 0.5, n).astype(float)

    # Generate errors based on distribution
    if error_dist == "normal":
        epsilon = np.random.randn(n)
    elif error_dist == "heavy_tail":
        epsilon = stats.t.rvs(df=3, size=n)  # t-distribution with df=3
    elif error_dist == "skewed":
        epsilon = stats.skewnorm.rvs(a=4, size=n)  # Skewed normal
    else:
        epsilon = np.random.randn(n)

    # Generate potential outcomes based on effect type
    Y0 = 1.0 + 0.5 * X[:, 0] + epsilon

    if effect_type == "homogeneous":
        # Constant QTE = 2.0 at all quantiles
        Y1 = Y0 + 2.0
        true_qte = {0.25: 2.0, 0.50: 2.0, 0.75: 2.0}

    elif effect_type == "location":
        # Location shift (same as homogeneous for QTE)
        Y1 = Y0 + 2.0
        true_qte = {0.25: 2.0, 0.50: 2.0, 0.75: 2.0}

    elif effect_type == "scale":
        # Treatment increases variance: Y(1) has larger spread
        # Y(1) = Y(0) + 1.5 * epsilon (increases scale)
        # QTE varies: larger at tails
        scale_factor = 1.5
        Y1 = 1.0 + 0.5 * X[:, 0] + 2.0 + scale_factor * epsilon
        # True QTE at quantiles (approximate)
        q_0 = stats.norm.ppf([0.25, 0.50, 0.75])
        true_qte = {
            0.25: 2.0 + (scale_factor - 1) * q_0[0],
            0.50: 2.0,
            0.75: 2.0 + (scale_factor - 1) * q_0[2],
        }

    elif effect_type == "distributional":
        # Full distributional change
        # Y(1) from different distribution
        epsilon1 = 0.5 * np.random.randn(n) + 0.5 * np.random.exponential(1, n)
        Y1 = 1.0 + 0.5 * X[:, 0] + 3.0 + epsilon1
        # True QTE is complex - estimate empirically
        q0 = np.quantile(Y0, [0.25, 0.50, 0.75])
        q1 = np.quantile(Y1, [0.25, 0.50, 0.75])
        true_qte = dict(zip([0.25, 0.50, 0.75], q1 - q0))

    else:
        Y1 = Y0 + 2.0
        true_qte = {0.25: 2.0, 0.50: 2.0, 0.75: 2.0}

    # Observed outcome
    Y = D * Y1 + (1 - D) * Y0

    return {
        "outcome": Y,
        "treatment": D,
        "covariates": X,
        "true_qte": true_qte,
        "Y0": Y0,
        "Y1": Y1,
    }


def generate_extreme_quantile_dgp(
    n: int = 5000,
    seed: int = 42,
) -> dict:
    """Generate data for testing extreme quantiles (0.05, 0.95).

    Uses larger sample size for stable tail estimation.
    """
    np.random.seed(seed)

    D = np.random.binomial(1, 0.5, n).astype(float)
    epsilon = np.random.randn(n)

    Y0 = 1.0 + epsilon
    Y1 = Y0 + 2.0  # Constant treatment effect

    Y = D * Y1 + (1 - D) * Y0

    return {
        "outcome": Y,
        "treatment": D,
        "true_qte": {0.05: 2.0, 0.95: 2.0},
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestConditionalQTEVsR:
    """Compare conditional QTE between Python and R."""

    def test_median_effect_parity(self):
        """Test median (0.50) QTE estimation parity."""
        data = generate_qte_dgp(n=2000, seed=42, effect_type="homogeneous")

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=np.array([0.50]),
        )
        py_qte_50 = py_result["quantile_effects"][0.50]

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=np.array([0.50]),
        )

        assert r_result is not None, "R conditional QTE failed"
        r_qte_50 = r_result["quantile_effects"][0.50]

        # Should match closely - same algorithm
        assert np.isclose(py_qte_50, r_qte_50, rtol=0.08), (
            f"Median QTE mismatch: Python={py_qte_50:.3f}, R={r_qte_50:.3f}"
        )

        # Both should be near true value of 2.0
        assert np.isclose(py_qte_50, 2.0, rtol=0.15), (
            f"Python median QTE {py_qte_50:.3f} far from true value 2.0"
        )

    def test_quartile_effects_parity(self):
        """Test QTE at 0.25, 0.50, 0.75 quantiles."""
        data = generate_qte_dgp(n=2500, seed=123, effect_type="homogeneous")
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R conditional QTE failed"

        # Compare at each quantile
        for tau in quantiles:
            py_qte = py_result["quantile_effects"][tau]
            r_qte = r_result["quantile_effects"][tau]

            assert np.isclose(py_qte, r_qte, rtol=0.12), (
                f"QTE({tau}) mismatch: Python={py_qte:.3f}, R={r_qte:.3f}"
            )

    def test_qte_with_covariates(self):
        """Test conditional QTE with covariates."""
        data = generate_qte_dgp(n=2000, seed=456, effect_type="homogeneous")
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python with covariates
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
            covariates=data["covariates"],
        )

        # R with covariates
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
            covariates=data["covariates"],
        )

        assert r_result is not None, "R conditional QTE with covariates failed"

        # Compare median (less sensitive to covariate specification)
        py_qte_50 = py_result["quantile_effects"][0.50]
        r_qte_50 = r_result["quantile_effects"][0.50]

        assert np.isclose(py_qte_50, r_qte_50, rtol=0.15), (
            f"Conditional QTE(0.50) with covariates mismatch: "
            f"Python={py_qte_50:.3f}, R={r_qte_50:.3f}"
        )

    def test_qte_heterogeneous_effects(self):
        """Test QTE detection of scale effects (heterogeneous across quantiles)."""
        data = generate_qte_dgp(n=3000, seed=789, effect_type="scale")
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R QTE for scale effects failed"

        # For scale effects: QTE(0.75) > QTE(0.50) > QTE(0.25)
        py_order = (
            py_result["quantile_effects"][0.25] <
            py_result["quantile_effects"][0.50] <
            py_result["quantile_effects"][0.75]
        )
        r_order = (
            r_result["quantile_effects"][0.25] <
            r_result["quantile_effects"][0.50] <
            r_result["quantile_effects"][0.75]
        )

        # Both should detect the ordering
        assert py_order, "Python didn't detect scale effect ordering"
        assert r_order, "R didn't detect scale effect ordering"


class TestUnconditionalQTEVsR:
    """Compare unconditional QTE (RIF regression) between Python and R."""

    def test_uqte_median_parity(self):
        """Test unconditional QTE at median."""
        data = generate_qte_dgp(n=2500, seed=42, effect_type="homogeneous")

        # Python
        py_result = unconditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=np.array([0.50]),
        )
        py_uqte = py_result["quantile_effects"][0.50]

        # R
        r_result = r_unconditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=np.array([0.50]),
        )

        assert r_result is not None, "R unconditional QTE failed"
        r_uqte = r_result["uqte"][0.50]

        # Unconditional QTE may differ more due to RIF implementation
        assert np.isclose(py_uqte, r_uqte, rtol=0.18), (
            f"Unconditional QTE(0.50) mismatch: Python={py_uqte:.3f}, R={r_uqte:.3f}"
        )

    def test_rif_qte_parity(self):
        """Test RIF-based QTE estimation."""
        data = generate_qte_dgp(n=2500, seed=123, effect_type="homogeneous")
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python RIF-QTE
        py_result = rif_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R unconditional QTE (uses RIF internally)
        r_result = r_unconditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R RIF QTE failed"

        # Compare at each quantile with wider tolerance
        for tau in quantiles:
            py_qte = py_result["quantile_effects"][tau]
            r_qte = r_result["uqte"][tau]

            assert np.isclose(py_qte, r_qte, rtol=0.20), (
                f"RIF QTE({tau}) mismatch: Python={py_qte:.3f}, R={r_qte:.3f}"
            )

    def test_unconditional_vs_conditional(self):
        """Test that unconditional and conditional QTE are similar for RCT."""
        data = generate_qte_dgp(n=3000, seed=456, effect_type="homogeneous")
        quantiles = np.array([0.50])

        # Conditional
        cond_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # Unconditional
        uncond_result = unconditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # For RCT with no covariates, should be similar
        cond_qte = cond_result["quantile_effects"][0.50]
        uncond_qte = uncond_result["quantile_effects"][0.50]

        assert np.isclose(cond_qte, uncond_qte, rtol=0.20), (
            f"Conditional ({cond_qte:.3f}) and Unconditional ({uncond_qte:.3f}) "
            f"QTE differ too much for RCT"
        )


class TestQTEEdgeCases:
    """Test edge cases for QTE estimation."""

    def test_extreme_quantiles(self):
        """Test QTE at extreme quantiles (0.05, 0.95)."""
        data = generate_extreme_quantile_dgp(n=5000, seed=42)
        quantiles = np.array([0.05, 0.95])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R extreme quantile estimation failed"

        # Wider tolerance for extreme quantiles
        for tau in quantiles:
            py_qte = py_result["quantile_effects"][tau]
            r_qte = r_result["quantile_effects"][tau]

            assert np.isclose(py_qte, r_qte, rtol=0.25), (
                f"Extreme QTE({tau}) mismatch: Python={py_qte:.3f}, R={r_qte:.3f}"
            )

            # Should still be near true value of 2.0
            assert np.isclose(py_qte, 2.0, rtol=0.35), (
                f"Extreme QTE({tau})={py_qte:.3f} far from true 2.0"
            )

    def test_heavy_tailed_distribution(self):
        """Test QTE with heavy-tailed errors (t-distribution)."""
        data = generate_qte_dgp(
            n=3000, seed=42, effect_type="homogeneous", error_dist="heavy_tail"
        )
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R QTE with heavy tails failed"

        # Compare median (robust to heavy tails)
        py_qte_50 = py_result["quantile_effects"][0.50]
        r_qte_50 = r_result["quantile_effects"][0.50]

        assert np.isclose(py_qte_50, r_qte_50, rtol=0.15), (
            f"Heavy-tail QTE(0.50) mismatch: Python={py_qte_50:.3f}, R={r_qte_50:.3f}"
        )

    def test_skewed_distribution(self):
        """Test QTE with skewed error distribution."""
        data = generate_qte_dgp(
            n=3000, seed=42, effect_type="homogeneous", error_dist="skewed"
        )
        quantiles = np.array([0.25, 0.50, 0.75])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # Verify estimation didn't fail
        assert all(
            not np.isnan(py_result["quantile_effects"][tau])
            for tau in quantiles
        ), "Python QTE returned NaN for skewed distribution"

    def test_qte_process_full_grid(self):
        """Test QTE process estimation across full quantile grid."""
        data = generate_qte_dgp(n=3000, seed=42, effect_type="homogeneous")

        # R QTE process
        r_result = r_qte_process(
            outcome=data["outcome"],
            treatment=data["treatment"],
            n_quantiles=19,
        )

        assert r_result is not None, "R QTE process failed"

        # Check process is smooth (no wild jumps)
        qte_process = r_result["qte_process"]
        max_jump = np.max(np.abs(np.diff(qte_process)))

        assert max_jump < 1.0, f"QTE process has large jump: {max_jump:.3f}"

        # Check average is near true value
        assert np.isclose(np.mean(qte_process), 2.0, rtol=0.20), (
            f"QTE process mean {np.mean(qte_process):.3f} far from true 2.0"
        )


class TestQTEStandardErrors:
    """Test QTE standard error estimation."""

    def test_se_magnitude_reasonable(self):
        """Test that QTE standard errors are reasonable magnitude."""
        data = generate_qte_dgp(n=2000, seed=42, effect_type="homogeneous")
        quantiles = np.array([0.50])

        # Python
        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        # R
        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R QTE SE estimation failed"

        # SE should be in reasonable range
        py_se = py_result["quantile_se"][0.50]
        r_se = r_result["quantile_se"][0.50]

        assert 0.01 < py_se < 0.5, f"Python SE {py_se:.4f} out of range"
        assert 0.01 < r_se < 0.5, f"R SE {r_se:.4f} out of range"

    def test_se_parity(self):
        """Test that Python and R SEs are similar order of magnitude."""
        data = generate_qte_dgp(n=2000, seed=42, effect_type="homogeneous")
        quantiles = np.array([0.25, 0.50, 0.75])

        py_result = conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        r_result = r_conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            quantiles=quantiles,
        )

        assert r_result is not None, "R QTE failed"

        for tau in quantiles:
            py_se = py_result["quantile_se"][tau]
            r_se = r_result["quantile_se"][tau]

            # SE estimation can vary, allow 50% difference
            ratio = py_se / r_se if r_se > 0 else float('inf')
            assert 0.5 < ratio < 2.0, (
                f"SE({tau}) ratio {ratio:.2f} out of range: "
                f"Python={py_se:.4f}, R={r_se:.4f}"
            )


class TestQTEMonteCarloTriangulation:
    """Monte Carlo tests for QTE estimation consistency."""

    @pytest.mark.slow
    def test_monte_carlo_median_qte_20_runs(self):
        """Monte Carlo: Median QTE estimation across 20 runs."""
        n_runs = 20
        n_per_run = 2000
        true_qte = 2.0

        py_qtes = []
        r_qtes = []

        for run in range(n_runs):
            data = generate_qte_dgp(
                n=n_per_run,
                seed=1000 + run,
                effect_type="homogeneous",
            )

            # Python
            py_result = conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=np.array([0.50]),
            )
            py_qtes.append(py_result["quantile_effects"][0.50])

            # R
            r_result = r_conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=np.array([0.50]),
            )
            if r_result is not None:
                r_qtes.append(r_result["quantile_effects"][0.50])

        # Check bias
        py_bias = np.mean(py_qtes) - true_qte
        r_bias = np.mean(r_qtes) - true_qte if r_qtes else np.nan

        assert abs(py_bias) < 0.20, f"Python median QTE bias {py_bias:.3f} too large"
        if r_qtes:
            assert abs(r_bias) < 0.20, f"R median QTE bias {r_bias:.3f} too large"

        # Check variance
        py_std = np.std(py_qtes)
        assert py_std < 0.3, f"Python QTE std {py_std:.3f} too large"

        # Check correlation between Python and R
        if len(r_qtes) == n_runs:
            correlation = np.corrcoef(py_qtes, r_qtes)[0, 1]
            assert correlation > 0.7, (
                f"Low correlation ({correlation:.3f}) between Python and R QTE"
            )

    @pytest.mark.slow
    def test_monte_carlo_quartiles_15_runs(self):
        """Monte Carlo: Quartile QTE estimation across 15 runs."""
        n_runs = 15
        n_per_run = 2500
        quantiles = np.array([0.25, 0.50, 0.75])

        py_results = {tau: [] for tau in quantiles}
        r_results = {tau: [] for tau in quantiles}

        for run in range(n_runs):
            data = generate_qte_dgp(
                n=n_per_run,
                seed=2000 + run,
                effect_type="homogeneous",
            )

            py_result = conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=quantiles,
            )
            for tau in quantiles:
                py_results[tau].append(py_result["quantile_effects"][tau])

            r_result = r_conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=quantiles,
            )
            if r_result is not None:
                for tau in quantiles:
                    r_results[tau].append(r_result["quantile_effects"][tau])

        # Check bias at each quantile
        for tau in quantiles:
            py_bias = np.mean(py_results[tau]) - 2.0
            assert abs(py_bias) < 0.25, (
                f"Python QTE({tau}) bias {py_bias:.3f} too large"
            )

    @pytest.mark.slow
    def test_monte_carlo_heterogeneous_detection(self):
        """Monte Carlo: Detection of heterogeneous effects."""
        n_runs = 10
        n_per_run = 3000

        detected_py = 0
        detected_r = 0

        for run in range(n_runs):
            data = generate_qte_dgp(
                n=n_per_run,
                seed=3000 + run,
                effect_type="scale",  # True heterogeneity
            )
            quantiles = np.array([0.25, 0.75])

            py_result = conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=quantiles,
            )

            # Detect if QTE(0.75) > QTE(0.25)
            py_diff = (
                py_result["quantile_effects"][0.75] -
                py_result["quantile_effects"][0.25]
            )
            if py_diff > 0.1:  # Meaningful difference
                detected_py += 1

            r_result = r_conditional_qte(
                outcome=data["outcome"],
                treatment=data["treatment"],
                quantiles=quantiles,
            )
            if r_result is not None:
                r_diff = (
                    r_result["quantile_effects"][0.75] -
                    r_result["quantile_effects"][0.25]
                )
                if r_diff > 0.1:
                    detected_r += 1

        # Should detect heterogeneity in most runs
        assert detected_py >= 6, (
            f"Python only detected heterogeneity in {detected_py}/{n_runs} runs"
        )
        assert detected_r >= 6, (
            f"R only detected heterogeneity in {detected_r}/{n_runs} runs"
        )


# =============================================================================
# Test Summary
# =============================================================================

"""
Test Summary for QTE R Triangulation
====================================

Classes:
- TestConditionalQTEVsR: 4 tests for conditional QTE
- TestUnconditionalQTEVsR: 3 tests for unconditional/RIF QTE
- TestQTEEdgeCases: 4 tests for edge cases
- TestQTEStandardErrors: 2 tests for SE estimation
- TestQTEMonteCarloTriangulation: 3 slow Monte Carlo tests

Total: 16 tests

Tolerance Standards:
- Median QTE: rtol=0.08-0.10 (same algorithm)
- Quartile QTE: rtol=0.12 (quantile estimation variance)
- Unconditional QTE: rtol=0.15-0.20 (RIF differences)
- Extreme quantiles: rtol=0.25 (sparse tails)
- SE comparison: 0.5-2.0 ratio (different estimation methods)

Skip Conditions:
- R not available
- quantreg package not installed
"""
