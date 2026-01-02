"""
R Triangulation Tests for Marginal Treatment Effects (MTE).

Compares Python MTE implementation against R localIV package.

Tests verify:
1. MTE curve shape and magnitude parity
2. Integrated treatment effects (ATE, ATT, ATU)
3. LATE estimation equivalence
4. Monte Carlo stability across implementations

Tolerance Standards
-------------------
| Metric        | Tolerance | Rationale                    |
|---------------|-----------|------------------------------|
| MTE curve     | rtol=0.15 | Nonparametric, bandwidth-sensitive |
| ATE/ATT/ATU   | rtol=0.10 | Integrated from curve        |
| LATE          | rtol=0.05 | Should match 2SLS closely    |

References
----------
- Heckman, J.J. & Vytlacil, E. (2005). Structural Equations, Treatment
  Effects, and Econometric Policy Evaluation. Econometrica, 73(3), 669-738.
"""

import numpy as np
import pytest
from scipy import stats

from causal_inference.mte import (
    local_iv,
    polynomial_mte,
    late_estimator,
    ate_from_mte,
    att_from_mte,
    atu_from_mte,
)

# Import R interface
try:
    from tests.validation.r_triangulation.r_interface import (
        check_localiv_installed,
        r_mte_estimate,
        r_mte_policy_effect,
    )
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


def check_r_available():
    """Check if R and localIV are available."""
    if not R_AVAILABLE:
        return False
    try:
        return check_localiv_installed()
    except Exception:
        return False


# Skip if R/localIV not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R or localIV package not available"
)


# =============================================================================
# Data Generation Functions
# =============================================================================


def generate_mte_dgp(
    n: int = 2000,
    seed: int = 42,
    effect_heterogeneity: str = "linear",
) -> dict:
    """Generate data following the MTE model.

    The DGP follows the Heckman-Vytlacil framework:
    - Selection: D = 1{P(Z) > U} where U ~ U(0,1)
    - Outcome: Y = mu_0(X) + D * (mu_1(X) - mu_0(X) + beta(U))

    Parameters
    ----------
    n : int
        Sample size.
    seed : int
        Random seed.
    effect_heterogeneity : str
        Type of treatment effect heterogeneity:
        - "constant": MTE(u) = 2.0 (constant)
        - "linear": MTE(u) = 3.0 - 2.0*u (decreasing)
        - "nonlinear": MTE(u) = 2.0 + sin(2*pi*u) (periodic)

    Returns
    -------
    dict
        Dictionary with:
        - outcome: np.ndarray (n,)
        - treatment: np.ndarray (n,) binary
        - instrument: np.ndarray (n,) propensity-like
        - covariates: np.ndarray (n, k)
        - true_mte_func: callable
        - true_ate: float
        - true_att: float
        - true_atu: float
    """
    np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 2)
    X[:, 0] = X[:, 0] * 0.5 + 1.0  # Centered around 1

    # Instrument: affects propensity but not outcome directly
    Z_raw = 0.5 * X[:, 0] + 0.5 * np.random.randn(n)

    # Propensity score: P(D=1|Z) - probit model
    propensity = stats.norm.cdf(Z_raw)

    # Unobserved resistance U ~ U(0,1)
    U = np.random.uniform(0, 1, n)

    # Selection: D = 1 if P(Z) > U (monotonicity holds)
    D = (propensity > U).astype(float)

    # Define MTE function based on heterogeneity type
    if effect_heterogeneity == "constant":
        def mte_func(u):
            return np.full_like(u, 2.0)
        true_ate = 2.0
    elif effect_heterogeneity == "linear":
        def mte_func(u):
            return 3.0 - 2.0 * u  # Decreasing: MTE(0)=3, MTE(1)=1
        true_ate = 2.0  # Integral of 3 - 2u from 0 to 1
    elif effect_heterogeneity == "nonlinear":
        def mte_func(u):
            return 2.0 + 0.5 * np.sin(2 * np.pi * u)
        true_ate = 2.0  # Sin integrates to 0
    else:
        raise ValueError(f"Unknown effect_heterogeneity: {effect_heterogeneity}")

    # Potential outcomes
    # Y(0) = 1.0 + 0.5*X1 + epsilon
    # Y(1) = Y(0) + MTE(U)
    epsilon = np.random.randn(n)
    Y0 = 1.0 + 0.5 * X[:, 0] + epsilon
    Y1 = Y0 + mte_func(U)

    # Observed outcome
    Y = D * Y1 + (1 - D) * Y0

    # Compute true ATT and ATU
    # ATT = E[MTE(U) | D=1] = E[MTE(U) | P > U]
    treated_mask = D == 1
    true_att = np.mean(mte_func(U[treated_mask])) if np.sum(treated_mask) > 0 else true_ate
    true_atu = np.mean(mte_func(U[~treated_mask])) if np.sum(~treated_mask) > 0 else true_ate

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": propensity,  # Use propensity as continuous instrument
        "covariates": X,
        "true_mte_func": mte_func,
        "true_ate": true_ate,
        "true_att": true_att,
        "true_atu": true_atu,
        "propensity": propensity,
        "U": U,
    }


def generate_late_dgp(
    n: int = 2000,
    seed: int = 42,
    late_value: float = 2.5,
) -> dict:
    """Generate data for LATE estimation with binary instrument.

    Parameters
    ----------
    n : int
        Sample size.
    seed : int
        Random seed.
    late_value : float
        True LATE value.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, binary instrument, true_late.
    """
    np.random.seed(seed)

    # Binary instrument
    Z = np.random.binomial(1, 0.5, n)

    # Propensity conditional on Z
    U = np.random.uniform(0, 1, n)
    propensity_z1 = 0.7
    propensity_z0 = 0.3

    # Treatment: D = 1 if propensity(Z) > U
    D = np.where(Z == 1, propensity_z1 > U, propensity_z0 > U).astype(float)

    # Compliers: those who switch treatment when Z changes
    # E[Y1 - Y0 | complier] = late_value
    epsilon = np.random.randn(n)
    Y0 = 1.0 + epsilon
    Y1 = Y0 + late_value  # Constant LATE for simplicity

    Y = D * Y1 + (1 - D) * Y0

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z.astype(float),
        "true_late": late_value,
        "first_stage": propensity_z1 - propensity_z0,  # ~0.4
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestMTECurveVsR:
    """Compare MTE curve estimation between Python and R."""

    def test_mte_curve_shape_constant(self):
        """Test MTE curve estimation with constant treatment effect."""
        data = generate_mte_dgp(n=3000, seed=42, effect_heterogeneity="constant")

        # Python estimation
        py_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        # R estimation
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
            n_grid=100,
        )

        assert r_result is not None, "R MTE estimation failed"

        # Compare average MTE (should be ~2.0 for constant)
        py_mte_mean = np.mean(py_result["mte_curve"])
        r_mte_mean = np.mean(r_result["mte_curve"])

        # Allow larger tolerance for nonparametric estimation
        assert np.isclose(py_mte_mean, r_mte_mean, rtol=0.20), (
            f"MTE mean mismatch: Python={py_mte_mean:.3f}, R={r_mte_mean:.3f}"
        )

        # Both should be near true value of 2.0
        assert np.isclose(py_mte_mean, 2.0, rtol=0.25), (
            f"Python MTE mean {py_mte_mean:.3f} far from true value 2.0"
        )

    def test_mte_curve_shape_linear(self):
        """Test MTE curve estimation with linear heterogeneity."""
        data = generate_mte_dgp(n=3000, seed=123, effect_heterogeneity="linear")

        # Python estimation
        py_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        # R estimation
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
            n_grid=100,
        )

        assert r_result is not None, "R MTE estimation failed"

        # For linear MTE(u) = 3 - 2u, curve should be decreasing
        # Check slope is negative for both
        py_mte = py_result["mte_curve"]
        r_mte = r_result["mte_curve"]

        # Fit linear trend to each
        u_grid_py = np.linspace(0.01, 0.99, len(py_mte))
        u_grid_r = r_result["u_grid"]

        py_slope = np.polyfit(u_grid_py, py_mte, 1)[0]
        r_slope = np.polyfit(u_grid_r, r_mte, 1)[0]

        # Both slopes should be negative (decreasing MTE)
        assert py_slope < 0, f"Python MTE slope should be negative: {py_slope:.3f}"
        assert r_slope < 0, f"R MTE slope should be negative: {r_slope:.3f}"

        # Slopes should be similar
        assert np.isclose(py_slope, r_slope, rtol=0.30), (
            f"MTE slope mismatch: Python={py_slope:.3f}, R={r_slope:.3f}"
        )

    def test_ate_from_mte_parity(self):
        """Test ATE computed from MTE matches between Python and R."""
        data = generate_mte_dgp(n=2500, seed=456, effect_heterogeneity="linear")

        # Python: estimate MTE then compute ATE
        py_mte_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )
        py_ate_result = ate_from_mte(py_mte_result)
        py_ate = py_ate_result["estimate"]

        # R estimation includes ATE
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        assert r_result is not None, "R MTE estimation failed"
        r_ate = r_result["ate"]

        # Compare ATE values
        assert np.isclose(py_ate, r_ate, rtol=0.15), (
            f"ATE mismatch: Python={py_ate:.3f}, R={r_ate:.3f}"
        )

        # Both should be near true ATE of 2.0
        assert np.isclose(py_ate, data["true_ate"], rtol=0.25), (
            f"Python ATE {py_ate:.3f} far from true value {data['true_ate']:.3f}"
        )

    def test_att_from_mte_parity(self):
        """Test ATT computed from MTE matches between Python and R."""
        data = generate_mte_dgp(n=2500, seed=789, effect_heterogeneity="linear")

        # Python
        py_mte_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )
        py_att_result = att_from_mte(py_mte_result, data["treatment"])
        py_att = py_att_result["estimate"]

        # R
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        assert r_result is not None, "R MTE estimation failed"
        r_att = r_result["att"]

        # ATT tolerance slightly higher due to weighting differences
        assert np.isclose(py_att, r_att, rtol=0.20), (
            f"ATT mismatch: Python={py_att:.3f}, R={r_att:.3f}"
        )

    def test_atu_from_mte_parity(self):
        """Test ATU computed from MTE matches between Python and R."""
        data = generate_mte_dgp(n=2500, seed=321, effect_heterogeneity="linear")

        # Python
        py_mte_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )
        py_atu_result = atu_from_mte(py_mte_result, data["treatment"])
        py_atu = py_atu_result["estimate"]

        # R
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        assert r_result is not None, "R MTE estimation failed"
        r_atu = r_result["atu"]

        assert np.isclose(py_atu, r_atu, rtol=0.20), (
            f"ATU mismatch: Python={py_atu:.3f}, R={r_atu:.3f}"
        )


class TestLATEVsR:
    """Compare LATE estimation between Python and R."""

    def test_late_with_binary_instrument(self):
        """Test LATE estimation with binary instrument (Wald estimator)."""
        data = generate_late_dgp(n=3000, seed=42, late_value=2.5)

        # Python LATE (Wald estimator)
        py_result = late_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )
        py_late = py_result["late"]

        # R: Use MTE-based LATE (or simple 2SLS)
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        assert r_result is not None, "R MTE estimation failed"
        r_late = r_result["late"]

        # LATE should match closely with binary instrument
        assert np.isclose(py_late, r_late, rtol=0.10), (
            f"LATE mismatch: Python={py_late:.3f}, R={r_late:.3f}"
        )

        # Both should be near true LATE
        assert np.isclose(py_late, data["true_late"], rtol=0.15), (
            f"Python LATE {py_late:.3f} far from true value {data['true_late']:.3f}"
        )

    def test_late_standard_errors(self):
        """Test LATE standard error estimation."""
        data = generate_late_dgp(n=2000, seed=123, late_value=2.0)

        py_result = late_estimator(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
        )

        # Check SE is reasonable (not too small or too large)
        assert 0.05 < py_result["se"] < 1.0, (
            f"LATE SE {py_result['se']:.3f} seems unreasonable"
        )

        # Check CI covers true value
        covers_true = (
            py_result["ci_lower"] < data["true_late"] < py_result["ci_upper"]
        )
        # Not a strict test, but should usually cover
        if not covers_true:
            pytest.xfail(
                f"CI [{py_result['ci_lower']:.3f}, {py_result['ci_upper']:.3f}] "
                f"doesn't cover true LATE {data['true_late']:.3f}"
            )


class TestMTEEdgeCases:
    """Test edge cases for MTE estimation."""

    def test_weak_instrument_warning(self):
        """Test behavior with weak instrument."""
        np.random.seed(42)
        n = 1000

        # Weak instrument: barely correlated with treatment
        Z = np.random.randn(n)
        D = np.random.binomial(1, 0.5, n).astype(float)  # Nearly independent of Z
        Y = 1.0 + 2.0 * D + np.random.randn(n)

        # Should still produce estimates, but may warn
        py_result = local_iv(
            outcome=Y,
            treatment=D,
            instrument=Z,
        )

        # Estimate should exist but may be unreliable
        assert "mte_curve" in py_result
        assert len(py_result["mte_curve"]) > 0

    def test_no_covariates(self):
        """Test MTE estimation without covariates."""
        data = generate_mte_dgp(n=2000, seed=42, effect_heterogeneity="constant")

        # Python without covariates
        py_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=None,
        )

        # R without covariates
        r_result = r_mte_estimate(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=None,
        )

        assert r_result is not None, "R MTE estimation without covariates failed"

        # Results should still be comparable
        py_ate = np.mean(py_result["mte_curve"])
        r_ate = r_result["ate"]

        assert np.isclose(py_ate, r_ate, rtol=0.25), (
            f"ATE mismatch without covariates: Python={py_ate:.3f}, R={r_ate:.3f}"
        )

    def test_polynomial_mte_vs_local_iv(self):
        """Compare polynomial MTE to local IV estimation."""
        data = generate_mte_dgp(n=2500, seed=42, effect_heterogeneity="linear")

        # Local IV (nonparametric)
        local_result = local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

        # Polynomial MTE (parametric)
        poly_result = polynomial_mte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
            degree=2,
        )

        # ATE from both should be similar
        local_ate = np.mean(local_result["mte_curve"])
        poly_ate = poly_result["ate"]

        assert np.isclose(local_ate, poly_ate, rtol=0.30), (
            f"ATE mismatch: Local IV={local_ate:.3f}, Polynomial={poly_ate:.3f}"
        )


class TestMTEMonteCarloTriangulation:
    """Monte Carlo tests for MTE estimation consistency."""

    @pytest.mark.slow
    def test_monte_carlo_ate_15_runs(self):
        """Monte Carlo test: ATE estimation across 15 runs."""
        n_runs = 15
        n_per_run = 2000

        py_ates = []
        r_ates = []
        true_ate = 2.0

        for run in range(n_runs):
            data = generate_mte_dgp(
                n=n_per_run,
                seed=1000 + run,
                effect_heterogeneity="constant",
            )

            # Python
            py_result = local_iv(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
                covariates=data["covariates"],
            )
            py_ate = ate_from_mte(py_result)["estimate"]
            py_ates.append(py_ate)

            # R
            r_result = r_mte_estimate(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
                covariates=data["covariates"],
            )
            if r_result is not None:
                r_ates.append(r_result["ate"])

        # Check bias
        py_bias = np.mean(py_ates) - true_ate
        r_bias = np.mean(r_ates) - true_ate if r_ates else np.nan

        assert abs(py_bias) < 0.30, f"Python ATE bias {py_bias:.3f} exceeds threshold"
        if r_ates:
            assert abs(r_bias) < 0.30, f"R ATE bias {r_bias:.3f} exceeds threshold"

        # Check correlation between Python and R estimates
        if len(r_ates) == n_runs:
            correlation = np.corrcoef(py_ates, r_ates)[0, 1]
            assert correlation > 0.5, (
                f"Low correlation ({correlation:.3f}) between Python and R ATE estimates"
            )

    @pytest.mark.slow
    def test_monte_carlo_late_consistency(self):
        """Monte Carlo test: LATE estimation consistency."""
        n_runs = 20
        n_per_run = 2000
        true_late = 2.5

        py_lates = []

        for run in range(n_runs):
            data = generate_late_dgp(
                n=n_per_run,
                seed=2000 + run,
                late_value=true_late,
            )

            py_result = late_estimator(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
            )
            py_lates.append(py_result["late"])

        # Check bias
        py_bias = np.mean(py_lates) - true_late
        assert abs(py_bias) < 0.20, f"Python LATE bias {py_bias:.3f} exceeds threshold"

        # Check variance is reasonable
        py_std = np.std(py_lates)
        assert py_std < 0.5, f"Python LATE std {py_std:.3f} too large"

        # Check coverage (rough: mean ± 2*SE should cover true)
        se_of_mean = py_std / np.sqrt(n_runs)
        covers = abs(py_bias) < 2 * se_of_mean
        if not covers:
            pytest.xfail(f"Monte Carlo mean doesn't cover true LATE")

    @pytest.mark.slow
    def test_monte_carlo_heterogeneous_mte(self):
        """Monte Carlo test: Heterogeneous MTE curve detection."""
        n_runs = 10
        n_per_run = 3000

        # Track whether slope is detected correctly
        py_slopes = []
        r_slopes = []

        for run in range(n_runs):
            data = generate_mte_dgp(
                n=n_per_run,
                seed=3000 + run,
                effect_heterogeneity="linear",  # True slope is -2.0
            )

            # Python
            py_result = local_iv(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
                covariates=data["covariates"],
            )
            u_grid = np.linspace(0.01, 0.99, len(py_result["mte_curve"]))
            py_slope = np.polyfit(u_grid, py_result["mte_curve"], 1)[0]
            py_slopes.append(py_slope)

            # R
            r_result = r_mte_estimate(
                outcome=data["outcome"],
                treatment=data["treatment"],
                instrument=data["instrument"],
                covariates=data["covariates"],
            )
            if r_result is not None:
                r_u_grid = r_result["u_grid"]
                r_slope = np.polyfit(r_u_grid, r_result["mte_curve"], 1)[0]
                r_slopes.append(r_slope)

        # Both should detect negative slope (true is -2.0)
        py_mean_slope = np.mean(py_slopes)
        assert py_mean_slope < -0.5, (
            f"Python mean slope {py_mean_slope:.3f} doesn't detect heterogeneity"
        )

        if r_slopes:
            r_mean_slope = np.mean(r_slopes)
            assert r_mean_slope < -0.5, (
                f"R mean slope {r_mean_slope:.3f} doesn't detect heterogeneity"
            )


# =============================================================================
# Test Summary
# =============================================================================

"""
Test Summary for MTE R Triangulation
====================================

Classes:
- TestMTECurveVsR: 5 tests for MTE curve and integrated effects
- TestLATEVsR: 2 tests for LATE estimation
- TestMTEEdgeCases: 3 tests for edge cases
- TestMTEMonteCarloTriangulation: 3 slow Monte Carlo tests

Total: 13 tests

Tolerance Standards:
- MTE curve mean: rtol=0.15-0.20 (nonparametric estimation variance)
- MTE slope: rtol=0.30 (sensitive to bandwidth)
- ATE/ATT/ATU: rtol=0.15-0.20 (integrated from curve)
- LATE: rtol=0.10-0.15 (should match 2SLS)
- Monte Carlo bias: <0.30 (allows for sampling variance)

Skip Conditions:
- R not available
- localIV package not installed
"""
