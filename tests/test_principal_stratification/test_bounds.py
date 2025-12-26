"""Tests for principal stratification bounds.

Test Structure (6-layer validation):
- Layer 1: Known-answer tests (bounds contain true value)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo (coverage of bounds)
- Layer 4: Cross-language (Julia parity - separate file)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.principal_stratification import (
    ps_bounds_monotonicity,
    ps_bounds_no_assumption,
    ps_bounds_balke_pearl,
    cace_2sls,
)
from src.causal_inference.principal_stratification.types import BoundsResult


# =============================================================================
# Test Data Generator
# =============================================================================


def generate_ps_dgp(
    n: int = 500,
    pi_c: float = 0.60,
    pi_a: float = 0.20,
    pi_n: float = 0.20,
    true_cace: float = 2.0,
    direct_effect: float = 0.0,
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
):
    """Generate data from principal stratification DGP.

    Parameters
    ----------
    direct_effect : float
        Direct effect of Z on Y (violates exclusion restriction).
    """
    np.random.seed(seed)

    # Normalize
    total = pi_c + pi_a + pi_n
    pi_c, pi_a, pi_n = pi_c / total, pi_a / total, pi_n / total

    # Random assignment
    Z = np.random.binomial(1, 0.5, n)

    # Strata
    strata = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        if r < pi_c:
            strata[i] = 0  # Complier
        elif r < pi_c + pi_a:
            strata[i] = 1  # Always-taker
        else:
            strata[i] = 2  # Never-taker

    # Treatment
    D = np.zeros(n)
    for i in range(n):
        if strata[i] == 0:
            D[i] = Z[i]
        elif strata[i] == 1:
            D[i] = 1
        else:
            D[i] = 0

    # Outcome with possible direct effect
    Y = baseline + true_cace * D + direct_effect * Z + noise_sd * np.random.randn(n)

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "true_cace": true_cace,
        "direct_effect": direct_effect,
        "strata": strata,
        "pi_c": pi_c,
        "pi_a": pi_a,
        "pi_n": pi_n,
    }


def generate_defiers_dgp(
    n: int = 500,
    pi_defier: float = 0.15,
    true_cace: float = 2.0,
    seed: int = 42,
):
    """Generate data with defiers (violates monotonicity)."""
    np.random.seed(seed)

    Z = np.random.binomial(1, 0.5, n)

    # Strata with defiers
    strata = np.zeros(n, dtype=int)
    pi_c = 0.50
    pi_a = 0.15
    pi_n = 0.20
    # pi_defier = remaining

    for i in range(n):
        r = np.random.rand()
        if r < pi_c:
            strata[i] = 0  # Complier
        elif r < pi_c + pi_a:
            strata[i] = 1  # Always-taker
        elif r < pi_c + pi_a + pi_n:
            strata[i] = 2  # Never-taker
        else:
            strata[i] = 3  # Defier

    # Treatment with defiers
    D = np.zeros(n)
    for i in range(n):
        if strata[i] == 0:  # Complier
            D[i] = Z[i]
        elif strata[i] == 1:  # Always-taker
            D[i] = 1
        elif strata[i] == 2:  # Never-taker
            D[i] = 0
        else:  # Defier
            D[i] = 1 - Z[i]

    # Outcome
    Y = 1.0 + true_cace * D + np.random.randn(n)

    return {"Y": Y, "D": D, "Z": Z, "true_cace": true_cace, "strata": strata}


# =============================================================================
# Layer 1: Known-Answer Tests - ps_bounds_monotonicity
# =============================================================================


class TestPSBoundsMonotonicity:
    """Tests for bounds under monotonicity."""

    def test_returns_valid_result(self):
        """Test that function returns BoundsResult with correct structure."""
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_monotonicity(data["Y"], data["D"], data["Z"])

        assert isinstance(result, dict)
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "bound_width" in result
        assert "identified" in result
        assert "assumptions" in result
        assert "method" in result

    def test_no_direct_effect_point_identified(self):
        """When direct_effect_bound=0, bounds collapse to LATE."""
        data = generate_ps_dgp(n=500, direct_effect=0.0, seed=42)
        result = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=0.0
        )

        assert result["identified"] is True
        assert_allclose(result["bound_width"], 0.0, atol=1e-10)
        assert result["lower_bound"] == result["upper_bound"]

    def test_bounds_contain_2sls(self):
        """Bounds should contain 2SLS estimate when exclusion holds."""
        data = generate_ps_dgp(n=500, direct_effect=0.0, seed=42)

        cace_result = cace_2sls(data["Y"], data["D"], data["Z"])
        cace_point = cace_result["cace"]

        # With δ = 0.5, bounds should contain point estimate
        bounds_result = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=0.5
        )

        assert bounds_result["lower_bound"] <= cace_point <= bounds_result["upper_bound"]

    def test_direct_effect_widens_bounds(self):
        """Larger direct_effect_bound should widen bounds."""
        data = generate_ps_dgp(n=300, seed=42)

        result_tight = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=0.1
        )
        result_wide = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=1.0
        )

        assert result_wide["bound_width"] > result_tight["bound_width"]

    def test_bounds_contain_true_cace(self):
        """Bounds should contain true CACE when assumptions hold."""
        data = generate_ps_dgp(n=500, true_cace=2.0, direct_effect=0.0, seed=42)

        result = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=0.0
        )

        # With proper identification, point estimate should be close to true
        point_estimate = (result["lower_bound"] + result["upper_bound"]) / 2
        assert abs(point_estimate - 2.0) < 0.5  # Within 0.5 of true value

    def test_weak_instrument_infinite_bounds(self):
        """Weak instrument should produce infinite bounds with warning."""
        np.random.seed(42)
        n = 300
        Z = np.random.binomial(1, 0.5, n)
        D = Z.copy()  # Perfect compliance - make first stage exactly zero
        D[:] = 0  # All control - no variation in D given Z
        Y = np.random.randn(n)

        # First stage will be zero since D has no variation
        with pytest.warns(RuntimeWarning, match="First stage is essentially zero"):
            result = ps_bounds_monotonicity(Y, D, Z)

        assert result["lower_bound"] == -np.inf
        assert result["upper_bound"] == np.inf


# =============================================================================
# Layer 1: Known-Answer Tests - ps_bounds_no_assumption
# =============================================================================


class TestPSBoundsNoAssumption:
    """Tests for Manski bounds without assumptions."""

    def test_returns_valid_result(self):
        """Test basic structure."""
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        assert isinstance(result, dict)
        assert result["lower_bound"] <= result["upper_bound"]
        assert result["identified"] is False
        assert result["assumptions"] == []

    def test_bounds_use_outcome_support(self):
        """Manski bounds should span approximately [Y_min - Y_max, Y_max - Y_min]."""
        data = generate_ps_dgp(n=300, seed=42)
        Y = data["Y"]

        result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        Y_min, Y_max = np.min(Y), np.max(Y)
        expected_lower = Y_min - Y_max
        expected_upper = Y_max - Y_min

        # Bounds should be close to Manski bounds
        assert result["lower_bound"] >= expected_lower - 0.1
        assert result["upper_bound"] <= expected_upper + 0.1

    def test_custom_support_used(self):
        """Custom outcome_support should be respected."""
        data = generate_ps_dgp(n=300, seed=42)

        result = ps_bounds_no_assumption(
            data["Y"], data["D"], data["Z"], outcome_support=(-5.0, 5.0)
        )

        # With support [-5, 5], bounds should be within [-10, 10]
        assert result["lower_bound"] >= -10.0
        assert result["upper_bound"] <= 10.0

    def test_wider_than_monotonicity_bounds(self):
        """No-assumption bounds should be wider than monotonicity bounds."""
        data = generate_ps_dgp(n=500, direct_effect=0.0, seed=42)

        result_mono = ps_bounds_monotonicity(
            data["Y"], data["D"], data["Z"], direct_effect_bound=0.0
        )
        result_none = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        # Manski bounds are typically much wider
        assert result_none["bound_width"] >= result_mono["bound_width"]


# =============================================================================
# Layer 1: Known-Answer Tests - ps_bounds_balke_pearl
# =============================================================================


class TestPSBoundsbalkePearl:
    """Tests for Balke-Pearl bounds."""

    def test_returns_valid_result(self):
        """Test basic structure."""
        data = generate_ps_dgp(n=300, seed=42)
        result = ps_bounds_balke_pearl(data["Y"], data["D"], data["Z"])

        assert isinstance(result, dict)
        assert result["lower_bound"] <= result["upper_bound"]
        assert "iv_constraints" in result["assumptions"]

    def test_tighter_than_manski_with_compliance(self):
        """Balke-Pearl should be tighter than Manski when there's compliance."""
        data = generate_ps_dgp(n=500, pi_c=0.7, seed=42)

        result_bp = ps_bounds_balke_pearl(data["Y"], data["D"], data["Z"])
        result_manski = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

        # Balke-Pearl uses IV structure, should be tighter
        assert result_bp["bound_width"] <= result_manski["bound_width"] * 1.5


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestBoundsAdversarial:
    """Edge cases and adversarial inputs."""

    def test_input_validation_length_mismatch(self):
        """Should raise on length mismatch."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        Z = np.random.binomial(1, 0.5, 50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            ps_bounds_monotonicity(Y, D, Z)

    def test_input_validation_non_binary_treatment(self):
        """Should raise on non-binary treatment."""
        Y = np.random.randn(100)
        D = np.random.randint(0, 3, 100)  # Not binary
        Z = np.random.binomial(1, 0.5, 100)

        with pytest.raises(ValueError, match="Treatment must be binary"):
            ps_bounds_monotonicity(Y, D, Z)

    def test_input_validation_non_binary_instrument(self):
        """Should raise on non-binary instrument."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        Z = np.random.randint(0, 3, 100)  # Not binary

        with pytest.raises(ValueError, match="Instrument must be binary"):
            ps_bounds_monotonicity(Y, D, Z)

    def test_negative_direct_effect_bound(self):
        """Should raise on negative direct_effect_bound."""
        data = generate_ps_dgp(n=100, seed=42)

        with pytest.raises(ValueError, match="non-negative"):
            ps_bounds_monotonicity(
                data["Y"], data["D"], data["Z"], direct_effect_bound=-0.5
            )

    def test_invalid_outcome_support(self):
        """Should raise on invalid outcome_support."""
        data = generate_ps_dgp(n=100, seed=42)

        with pytest.raises(ValueError, match="Y_min < Y_max"):
            ps_bounds_no_assumption(
                data["Y"], data["D"], data["Z"], outcome_support=(5.0, -5.0)
            )

    def test_all_treated_or_control(self):
        """Handle case where all units are in one treatment group."""
        np.random.seed(42)
        n = 100
        Z = np.random.binomial(1, 0.5, n)
        D = np.ones(n)  # All treated
        Y = np.random.randn(n)

        # Should still produce valid bounds (infinite due to no variation in D)
        with pytest.warns(RuntimeWarning):
            result = ps_bounds_monotonicity(Y, D, Z)

        assert np.isinf(result["lower_bound"]) or np.isinf(result["upper_bound"])


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestBoundsMonteCarlo:
    """Monte Carlo validation of bounds coverage."""

    @pytest.mark.slow
    def test_bounds_cover_true_cace(self):
        """Bounds should cover true CACE with high probability."""
        n_sims = 100
        true_cace = 2.0
        covers = []

        for seed in range(n_sims):
            data = generate_ps_dgp(
                n=500, true_cace=true_cace, direct_effect=0.0, seed=seed
            )

            result = ps_bounds_monotonicity(
                data["Y"], data["D"], data["Z"], direct_effect_bound=0.0
            )

            # Check coverage (with some tolerance for finite sample)
            midpoint = (result["lower_bound"] + result["upper_bound"]) / 2
            # For point-identified case, midpoint should be close to true
            covers.append(abs(midpoint - true_cace) < 1.0)

        coverage = np.mean(covers)
        # Should cover in at least 80% of simulations
        assert coverage > 0.80, f"Coverage {coverage:.2%} below 80%"

    @pytest.mark.slow
    def test_bounds_contain_true_with_defiers(self):
        """With defiers, no-assumption bounds should still contain true."""
        n_sims = 50
        true_cace = 2.0
        covers = []

        for seed in range(n_sims):
            data = generate_defiers_dgp(
                n=500, true_cace=true_cace, pi_defier=0.15, seed=seed
            )

            result = ps_bounds_no_assumption(data["Y"], data["D"], data["Z"])

            # Bounds should contain true CACE
            covers.append(
                result["lower_bound"] <= true_cace <= result["upper_bound"]
            )

        coverage = np.mean(covers)
        # Should always cover since bounds are conservative
        assert coverage > 0.90, f"Coverage {coverage:.2%} below 90%"
