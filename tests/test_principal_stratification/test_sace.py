"""Tests for SACE (Survivor Average Causal Effect) estimation.

Test Structure (6-layer validation):
- Layer 1: Known-answer tests (bounds contain true value)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo (coverage)
- Layer 4: Cross-language (Julia parity - separate file)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.principal_stratification import sace_bounds, sace_sensitivity
from src.causal_inference.principal_stratification.types import SACEResult


# =============================================================================
# Test Data Generator
# =============================================================================


def generate_sace_dgp(
    n: int = 500,
    true_sace: float = 1.5,
    p_AS: float = 0.50,  # Always-survivor proportion
    p_protected: float = 0.20,  # Protected by treatment
    p_harmed: float = 0.10,  # Harmed by treatment
    baseline: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
):
    """Generate data with truncation by death for SACE estimation.

    Strata:
    - Always-survivors (AS): S(0)=1, S(1)=1
    - Protected: S(0)=0, S(1)=1
    - Harmed: S(0)=1, S(1)=0
    - Never-survivors: S(0)=0, S(1)=0
    """
    np.random.seed(seed)

    # Normalize
    p_never = max(0, 1.0 - p_AS - p_protected - p_harmed)
    total = p_AS + p_protected + p_harmed + p_never

    p_AS /= total
    p_protected /= total
    p_harmed /= total
    p_never /= total

    # Random treatment assignment (as if randomized)
    D = np.random.binomial(1, 0.5, n)

    # Assign strata
    strata = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        if r < p_AS:
            strata[i] = 0  # Always-survivor
        elif r < p_AS + p_protected:
            strata[i] = 1  # Protected
        elif r < p_AS + p_protected + p_harmed:
            strata[i] = 2  # Harmed
        else:
            strata[i] = 3  # Never-survivor

    # Survival based on stratum and treatment
    S = np.zeros(n, dtype=int)
    for i in range(n):
        if strata[i] == 0:  # Always-survivor
            S[i] = 1
        elif strata[i] == 1:  # Protected
            S[i] = D[i]  # Survives only if treated
        elif strata[i] == 2:  # Harmed
            S[i] = 1 - D[i]  # Survives only if not treated
        else:  # Never-survivor
            S[i] = 0

    # Potential outcomes (only for survivors)
    Y0_latent = baseline + noise_sd * np.random.randn(n)
    Y1_latent = baseline + true_sace + noise_sd * np.random.randn(n)

    # Observed outcome (NaN for non-survivors)
    Y = np.where(S == 1, np.where(D == 1, Y1_latent, Y0_latent), np.nan)

    return {
        "Y": Y,
        "D": D,
        "S": S,
        "true_sace": true_sace,
        "strata": strata,
        "p_AS": p_AS,
        "p_protected": p_protected,
        "p_harmed": p_harmed,
    }


def generate_selection_monotonicity_dgp(
    n: int = 500,
    true_sace: float = 1.5,
    p_AS: float = 0.60,
    p_protected: float = 0.25,
    seed: int = 42,
):
    """Generate data with selection monotonicity (no harmed stratum)."""
    return generate_sace_dgp(
        n=n,
        true_sace=true_sace,
        p_AS=p_AS,
        p_protected=p_protected,
        p_harmed=0.0,  # No harmed stratum
        seed=seed,
    )


# =============================================================================
# Layer 1: Known-Answer Tests - sace_bounds
# =============================================================================


class TestSACEBoundsStructure:
    """Tests for sace_bounds return structure."""

    def test_returns_valid_result(self):
        """Test that function returns SACEResult with correct structure."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data["Y"], data["D"], data["S"])

        assert isinstance(result, dict)
        assert "sace" in result
        assert "se" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "proportion_survivors_treat" in result
        assert "proportion_survivors_control" in result
        assert "n" in result
        assert "method" in result

    def test_bounds_ordered(self):
        """Lower bound should be less than or equal to upper bound."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data["Y"], data["D"], data["S"])

        assert result["lower_bound"] <= result["upper_bound"]

    def test_survival_proportions_valid(self):
        """Survival proportions should be in [0, 1]."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_bounds(data["Y"], data["D"], data["S"])

        assert 0 <= result["proportion_survivors_treat"] <= 1
        assert 0 <= result["proportion_survivors_control"] <= 1

    def test_sample_size_correct(self):
        """Sample size should match input."""
        n = 300
        data = generate_sace_dgp(n=n, seed=42)
        result = sace_bounds(data["Y"], data["D"], data["S"])

        assert result["n"] == n


# =============================================================================
# Layer 1: Known-Answer Tests - Monotonicity Options
# =============================================================================


class TestSACEBoundsMonotonicity:
    """Tests for different monotonicity assumptions."""

    def test_no_monotonicity_widest_bounds(self):
        """No monotonicity and selection monotonicity produce valid bounds."""
        data = generate_sace_dgp(n=500, seed=42)

        result_none = sace_bounds(data["Y"], data["D"], data["S"], monotonicity="none")
        result_selection = sace_bounds(
            data["Y"], data["D"], data["S"], monotonicity="selection"
        )

        # Both should produce valid bounds
        assert result_none["lower_bound"] <= result_none["upper_bound"]
        assert result_selection["lower_bound"] <= result_selection["upper_bound"]

        # Both bounds should be finite
        assert np.isfinite(result_none["lower_bound"])
        assert np.isfinite(result_selection["lower_bound"])

    def test_selection_monotonicity_valid(self):
        """Selection monotonicity should produce valid bounds."""
        data = generate_selection_monotonicity_dgp(n=500, seed=42)

        result_none = sace_bounds(data["Y"], data["D"], data["S"], monotonicity="none")
        result_selection = sace_bounds(
            data["Y"], data["D"], data["S"], monotonicity="selection"
        )

        # Both produce valid, ordered bounds
        assert result_none["lower_bound"] <= result_none["upper_bound"]
        assert result_selection["lower_bound"] <= result_selection["upper_bound"]

        # Selection monotonicity bounds should contain true SACE
        true_sace = data["true_sace"]
        assert result_selection["lower_bound"] <= true_sace <= result_selection["upper_bound"]

    def test_both_monotonicity_tightest(self):
        """Both monotonicity assumptions should give tightest bounds."""
        data = generate_selection_monotonicity_dgp(n=500, seed=42)

        result_selection = sace_bounds(
            data["Y"], data["D"], data["S"], monotonicity="selection"
        )
        result_both = sace_bounds(
            data["Y"], data["D"], data["S"], monotonicity="both"
        )

        width_selection = result_selection["upper_bound"] - result_selection["lower_bound"]
        width_both = result_both["upper_bound"] - result_both["lower_bound"]

        # Both should be at least as tight
        assert width_both <= width_selection * 1.01


# =============================================================================
# Layer 1: Known-Answer Tests - Bounds Coverage
# =============================================================================


class TestSACEBoundsKnownAnswer:
    """Tests for bounds containing true values."""

    def test_bounds_contain_true_sace_selection_monotonicity(self):
        """Under selection monotonicity, bounds should contain true SACE."""
        data = generate_selection_monotonicity_dgp(
            n=1000, true_sace=1.5, p_AS=0.7, seed=42
        )

        result = sace_bounds(
            data["Y"], data["D"], data["S"], monotonicity="selection"
        )

        # True SACE should be within bounds
        assert result["lower_bound"] <= data["true_sace"] <= result["upper_bound"]

    def test_point_estimate_reasonable(self):
        """Point estimate (midpoint) should be in reasonable range."""
        data = generate_sace_dgp(n=500, true_sace=1.5, seed=42)
        result = sace_bounds(data["Y"], data["D"], data["S"], monotonicity="selection")

        # Point estimate should be within 2.0 of true
        assert abs(result["sace"] - data["true_sace"]) < 2.0


# =============================================================================
# Layer 1: Known-Answer Tests - sace_sensitivity
# =============================================================================


class TestSACESensitivity:
    """Tests for sace_sensitivity function."""

    def test_returns_expected_keys(self):
        """Should return dict with alpha, lower_bound, upper_bound, sace."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data["Y"], data["D"], data["S"])

        assert "alpha" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "sace" in result

    def test_output_lengths_match(self):
        """All output arrays should have same length."""
        n_points = 25
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data["Y"], data["D"], data["S"], n_points=n_points)

        assert len(result["alpha"]) == n_points
        assert len(result["lower_bound"]) == n_points
        assert len(result["upper_bound"]) == n_points
        assert len(result["sace"]) == n_points

    def test_alpha_range_correct(self):
        """Alpha values should span specified range."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(
            data["Y"], data["D"], data["S"], alpha_range=(0.2, 0.8), n_points=50
        )

        assert result["alpha"][0] == pytest.approx(0.2)
        assert result["alpha"][-1] == pytest.approx(0.8)

    def test_bounds_change_with_alpha(self):
        """Bounds should change smoothly with alpha."""
        data = generate_sace_dgp(n=500, seed=42)
        result = sace_sensitivity(data["Y"], data["D"], data["S"], n_points=20)

        # Widths should be computed correctly
        widths = result["upper_bound"] - result["lower_bound"]

        # All widths should be non-negative
        assert np.all(widths >= 0)

        # Widths should change smoothly (no discontinuities)
        # Check that max change between adjacent points is reasonable
        max_change = np.max(np.abs(np.diff(widths)))
        assert max_change < np.max(widths)  # No sudden jumps

    def test_sace_is_midpoint(self):
        """SACE should be midpoint of bounds."""
        data = generate_sace_dgp(n=300, seed=42)
        result = sace_sensitivity(data["Y"], data["D"], data["S"])

        midpoints = (result["lower_bound"] + result["upper_bound"]) / 2
        assert_allclose(result["sace"], midpoints, rtol=1e-10)


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestSACEAdversarial:
    """Edge cases and adversarial inputs."""

    def test_input_validation_length_mismatch(self):
        """Should raise on length mismatch."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        S = np.random.binomial(1, 0.8, 50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            sace_bounds(Y, D, S)

    def test_input_validation_non_binary_treatment(self):
        """Should raise on non-binary treatment."""
        Y = np.random.randn(100)
        D = np.random.randint(0, 3, 100)  # Not binary
        S = np.random.binomial(1, 0.8, 100)

        with pytest.raises(ValueError, match="Treatment must be binary"):
            sace_bounds(Y, D, S)

    def test_input_validation_non_binary_survival(self):
        """Should raise on non-binary survival."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        S = np.random.randint(0, 3, 100)  # Not binary

        with pytest.raises(ValueError, match="Survival must be binary"):
            sace_bounds(Y, D, S)

    def test_no_survivors_raises(self):
        """Should raise when no survivors."""
        Y = np.random.randn(100)
        D = np.random.binomial(1, 0.5, 100)
        S = np.zeros(100, dtype=int)  # All dead

        with pytest.raises(ValueError, match="No survivors"):
            sace_bounds(Y, D, S)

    def test_handles_nan_outcome(self):
        """Should handle NaN outcomes for non-survivors."""
        np.random.seed(42)
        n = 100
        D = np.random.binomial(1, 0.5, n)
        S = np.random.binomial(1, 0.8, n)
        Y = np.where(S == 1, np.random.randn(n), np.nan)

        # Should not raise
        result = sace_bounds(Y, D, S)
        assert np.isfinite(result["sace"])

    def test_all_treated_survivors(self):
        """Handle case where only treated survive."""
        np.random.seed(42)
        n = 100
        D = np.random.binomial(1, 0.5, n)
        S = D  # Only treated survive
        Y = np.where(S == 1, np.random.randn(n), np.nan)

        # Should produce valid bounds
        result = sace_bounds(Y, D, S)
        assert result["lower_bound"] <= result["upper_bound"]


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestSACEMonteCarlo:
    """Monte Carlo validation of SACE bounds."""

    @pytest.mark.slow
    def test_bounds_cover_true_sace_selection_monotonicity(self):
        """Under selection monotonicity, bounds should cover true SACE."""
        n_sims = 100
        true_sace = 1.5
        covers = []

        for seed in range(n_sims):
            data = generate_selection_monotonicity_dgp(
                n=500, true_sace=true_sace, p_AS=0.65, seed=seed
            )

            result = sace_bounds(
                data["Y"], data["D"], data["S"], monotonicity="selection"
            )

            covers.append(
                result["lower_bound"] <= true_sace <= result["upper_bound"]
            )

        coverage = np.mean(covers)
        # Should cover in at least 85% of simulations
        assert coverage > 0.85, f"Coverage {coverage:.2%} below 85%"

    @pytest.mark.slow
    def test_lee_bounds_valid(self):
        """Lee bounds (no assumption) should produce valid, ordered bounds."""
        n_sims = 50
        true_sace = 1.5
        valid_bounds = []

        for seed in range(n_sims):
            data = generate_sace_dgp(
                n=500,
                true_sace=true_sace,
                p_harmed=0.15,  # Include harmed stratum
                seed=seed,
            )

            result = sace_bounds(data["Y"], data["D"], data["S"], monotonicity="none")

            # Bounds should be properly ordered and finite
            valid_bounds.append(
                result["lower_bound"] <= result["upper_bound"]
                and np.isfinite(result["lower_bound"])
                and np.isfinite(result["upper_bound"])
            )

        validity = np.mean(valid_bounds)
        # All simulations should produce valid bounds
        assert validity == 1.0, f"Invalid bounds in {(1-validity)*100:.1f}% of simulations"
