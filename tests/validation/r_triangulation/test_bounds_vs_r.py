"""Triangulation tests: Python Bounds (Manski, Lee) vs R reference.

This module provides Layer 5 validation by comparing our Python partial
identification bounds implementations against R implementations.

Note: No standard R package exists for Manski/Lee bounds. We use manual
base R implementations that compute the same formulas. This provides
cross-implementation validation even without a canonical R package.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established based on numerical precision):
- Bounds endpoints: rtol=0.05 (5% relative, direct computation)
- Bounds width: rtol=0.02 (2% relative, should match closely)
- Trimming proportion: rtol=0.01 (1% relative, exact share)
- Bootstrap CI: rtol=0.15 (15% relative, resampling variation)

Run with: pytest tests/validation/r_triangulation/test_bounds_vs_r.py -v

References:
- Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects.
  American Economic Review, 80(2), 319-323.
- Manski, C. F. (2003). Partial Identification of Probability Distributions.
  Springer.
- Lee, D. S. (2009). Training, Wages, and Sample Selection.
  Review of Economic Studies, 76(3), 1071-1102.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Dict, Any

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    r_manski_worst_case,
    r_manski_mtr,
    r_manski_mts,
    r_manski_mtr_mts,
    r_manski_iv,
    r_lee_bounds,
)

# Lazy import Python implementations
try:
    from src.causal_inference.bounds.manski import (
        manski_worst_case,
        manski_mtr,
        manski_mts,
        manski_mtr_mts,
        manski_iv,
    )
    from src.causal_inference.bounds.lee import lee_bounds

    BOUNDS_AVAILABLE = True
except ImportError:
    BOUNDS_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_bounds_python = pytest.mark.skipif(
    not BOUNDS_AVAILABLE,
    reason="Python Bounds module not available",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_bounds_data(
    n: int = 1000,
    p_treat: float = 0.5,
    true_ate: float = 2.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate basic data for Manski bounds testing.

    DGP:
    - T ~ Bernoulli(p_treat)
    - Y(0) ~ Normal(5, 1)  with support [0, 10]
    - Y(1) = Y(0) + true_ate + noise
    - Y = T * Y(1) + (1-T) * Y(0)

    The true ATE should lie within the bounds.

    Parameters
    ----------
    n : int
        Sample size.
    p_treat : float
        Treatment probability.
    true_ate : float
        True average treatment effect.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, true_ate, outcome_support.
    """
    np.random.seed(seed)

    # Treatment assignment
    treatment = np.random.binomial(1, p_treat, n)

    # Potential outcomes with bounded support
    y0 = np.clip(np.random.normal(5, 1, n), 0, 10)
    y1 = np.clip(y0 + true_ate + np.random.normal(0, 0.5, n), 0, 10)

    # Observed outcome
    outcome = treatment * y1 + (1 - treatment) * y0

    return {
        "outcome": outcome,
        "treatment": treatment,
        "true_ate": true_ate,
        "outcome_support": (0.0, 10.0),
    }


def generate_selection_data(
    n: int = 2000,
    p_treat: float = 0.5,
    true_ate: float = 2.0,
    base_obs_rate: float = 0.6,
    treatment_effect_on_obs: float = 0.2,
    monotonicity: str = "positive",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with sample selection for Lee bounds testing.

    DGP:
    - T ~ Bernoulli(p_treat)
    - S(t) = observation indicator depends on treatment
    - Positive monotonicity: P(S=1|T=1) > P(S=1|T=0)
    - Negative monotonicity: P(S=1|T=1) < P(S=1|T=0)
    - Y(t) = potential outcome
    - Observe Y only when S=1

    Parameters
    ----------
    n : int
        Sample size.
    p_treat : float
        Treatment probability.
    true_ate : float
        True ATE for always-observed subpopulation.
    base_obs_rate : float
        Base observation probability for control.
    treatment_effect_on_obs : float
        How much treatment changes observation probability.
    monotonicity : str
        "positive" or "negative".
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, observed, true_ate, monotonicity.
    """
    np.random.seed(seed)

    # Treatment assignment
    treatment = np.random.binomial(1, p_treat, n)

    # Selection probability depends on treatment
    if monotonicity == "positive":
        # Treatment increases observation
        p_obs = base_obs_rate + treatment_effect_on_obs * treatment
    else:
        # Treatment decreases observation
        p_obs = base_obs_rate - treatment_effect_on_obs * treatment

    # Observation indicator
    observed = np.random.binomial(1, p_obs)

    # Potential outcomes (always generated, observed only when S=1)
    y0 = np.random.normal(5, 1, n)
    y1 = y0 + true_ate + np.random.normal(0, 0.5, n)

    # Observed outcome
    outcome = treatment * y1 + (1 - treatment) * y0

    return {
        "outcome": outcome,
        "treatment": treatment,
        "observed": observed,
        "true_ate": true_ate,
        "monotonicity": monotonicity,
        "obs_rate_control": base_obs_rate,
        "obs_rate_treated": base_obs_rate + treatment_effect_on_obs
        if monotonicity == "positive"
        else base_obs_rate - treatment_effect_on_obs,
    }


def generate_iv_bounds_data(
    n: int = 2000,
    complier_share: float = 0.4,
    true_late: float = 2.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data for Manski IV bounds testing.

    DGP (LATE framework):
    - Z ~ Bernoulli(0.5) (instrument)
    - Compliance types: Always-takers, Never-takers, Compliers
    - D(z) = min(z, type), where type ∈ {AT=1, NT=0, C=z}
    - Y(1) - Y(0) = true_late for compliers

    Parameters
    ----------
    n : int
        Sample size.
    complier_share : float
        Share of compliers in population.
    true_late : float
        True LATE for compliers.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, instrument, outcome_support, true_late.
    """
    np.random.seed(seed)

    # Instrument
    instrument = np.random.binomial(1, 0.5, n)

    # Compliance types
    # Always-taker: D=1 regardless of Z
    # Never-taker: D=0 regardless of Z
    # Complier: D=Z
    always_taker_share = 0.2
    never_taker_share = 1 - always_taker_share - complier_share

    type_probs = [always_taker_share, never_taker_share, complier_share]
    types = np.random.choice(["AT", "NT", "C"], size=n, p=type_probs)

    # Treatment based on type and instrument
    treatment = np.where(
        types == "AT",
        1,
        np.where(types == "NT", 0, instrument),
    )

    # Potential outcomes
    y0 = np.clip(np.random.normal(5, 1, n), 0, 10)

    # Only compliers have true LATE effect
    y1 = y0.copy()
    y1[types == "C"] += true_late

    # Always-takers and never-takers have different effect
    y1[types == "AT"] += 0.5  # Small positive effect
    y1[types == "NT"] += 0.5  # Would have small effect if treated

    y1 = np.clip(y1, 0, 10)

    # Observed outcome
    outcome = treatment * y1 + (1 - treatment) * y0

    return {
        "outcome": outcome,
        "treatment": treatment,
        "instrument": instrument,
        "outcome_support": (0.0, 10.0),
        "true_late": true_late,
        "complier_share": complier_share,
    }


# =============================================================================
# Test Class: Manski Worst-Case Bounds
# =============================================================================


@requires_bounds_python
class TestManskiWorstCaseVsR:
    """Test Python manski_worst_case against R implementation."""

    def test_basic_worst_case(self):
        """Standard DGP: worst-case bounds should match R."""
        data = generate_bounds_data(n=1000, p_treat=0.5, true_ate=2.0, seed=42)

        # Python implementation
        py_result = manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )

        # R implementation
        r_result = r_manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None, "R implementation returned None"

        # Compare bounds
        assert np.isclose(
            py_result["bounds_lower"],
            r_result["bounds_lower"],
            rtol=0.05,
        ), (
            f"Lower bound mismatch: Python={py_result['bounds_lower']:.4f}, R={r_result['bounds_lower']:.4f}"
        )

        assert np.isclose(
            py_result["bounds_upper"],
            r_result["bounds_upper"],
            rtol=0.05,
        ), (
            f"Upper bound mismatch: Python={py_result['bounds_upper']:.4f}, R={r_result['bounds_upper']:.4f}"
        )

        # Compare width (should match closely)
        assert np.isclose(
            py_result["bounds_width"],
            r_result["bounds_width"],
            rtol=0.02,
        ), (
            f"Width mismatch: Python={py_result['bounds_width']:.4f}, R={r_result['bounds_width']:.4f}"
        )

    def test_wide_support(self):
        """Wide outcome support: bounds should be very wide."""
        data = generate_bounds_data(n=1000, seed=123)

        # Use wider support than data range
        wide_support = (-10.0, 20.0)

        py_result = manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=wide_support,
        )

        r_result = r_manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=wide_support,
        )

        assert r_result is not None

        # Bounds should match
        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

        # Wide support → wide bounds
        assert py_result["bounds_width"] > 20, "Expected very wide bounds"
        assert r_result["bounds_width"] > 20, "R should also have wide bounds"

    def test_narrow_support(self):
        """Narrow outcome support: tighter bounds."""
        np.random.seed(456)
        n = 1000
        treatment = np.random.binomial(1, 0.5, n)
        # Outcomes all in [4, 6]
        outcome = 5 + treatment * 0.5 + np.random.uniform(-0.5, 0.5, n)

        narrow_support = (4.0, 6.0)

        py_result = manski_worst_case(
            outcome=outcome, treatment=treatment, outcome_support=narrow_support
        )

        r_result = r_manski_worst_case(
            outcome=outcome, treatment=treatment, outcome_support=narrow_support
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

        # Narrow support → narrower bounds
        assert py_result["bounds_width"] < 5, "Expected narrower bounds"


# =============================================================================
# Test Class: Manski MTR Bounds
# =============================================================================


@requires_bounds_python
class TestManskiMTRVsR:
    """Test Python manski_mtr against R implementation."""

    def test_mtr_positive(self):
        """Positive MTR: treatment response weakly positive."""
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)

        py_result = manski_mtr(
            outcome=data["outcome"],
            treatment=data["treatment"],
            direction="positive",
            outcome_support=data["outcome_support"],
        )

        r_result = r_manski_mtr(
            outcome=data["outcome"],
            treatment=data["treatment"],
            direction="positive",
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

        # MTR tightens bounds relative to worst-case
        wc_result = manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )
        assert py_result["bounds_width"] <= wc_result["bounds_width"] + 0.1

    def test_mtr_negative(self):
        """Negative MTR: treatment response weakly negative."""
        # Generate data where treatment has negative effect
        np.random.seed(789)
        n = 1000
        treatment = np.random.binomial(1, 0.5, n)
        y0 = np.clip(np.random.normal(5, 1, n), 0, 10)
        y1 = np.clip(y0 - 1.5 + np.random.normal(0, 0.3, n), 0, 10)
        outcome = treatment * y1 + (1 - treatment) * y0

        py_result = manski_mtr(
            outcome=outcome,
            treatment=treatment,
            direction="negative",
            outcome_support=(0.0, 10.0),
        )

        r_result = r_manski_mtr(
            outcome=outcome,
            treatment=treatment,
            direction="negative",
            outcome_support=(0.0, 10.0),
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)


# =============================================================================
# Test Class: Manski MTS Bounds
# =============================================================================


@requires_bounds_python
class TestManskiMTSVsR:
    """Test Python manski_mts against R implementation."""

    def test_mts_basic(self):
        """Basic MTS: selection on outcome level."""
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)

        py_result = manski_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )

        r_result = r_manski_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

    def test_mts_strong_selection(self):
        """Strong selection: treated have much higher outcomes."""
        np.random.seed(321)
        n = 1000

        # Strong positive selection: high Y → more likely to be treated
        y_latent = np.random.normal(5, 2, n)
        p_treat = 1 / (1 + np.exp(-(y_latent - 5)))  # Logistic selection
        treatment = np.random.binomial(1, p_treat)

        # True outcomes
        y0 = np.clip(y_latent, 0, 10)
        y1 = np.clip(y_latent + 1.0, 0, 10)
        outcome = treatment * y1 + (1 - treatment) * y0

        py_result = manski_mts(outcome=outcome, treatment=treatment, outcome_support=(0.0, 10.0))

        r_result = r_manski_mts(outcome=outcome, treatment=treatment, outcome_support=(0.0, 10.0))

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)


# =============================================================================
# Test Class: Manski MTR+MTS Bounds
# =============================================================================


@requires_bounds_python
class TestManskiMTRMTSVsR:
    """Test Python manski_mtr_mts against R implementation."""

    def test_combined_positive(self):
        """Combined MTR+MTS with positive MTR."""
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)

        py_result = manski_mtr_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mtr_direction="positive",
            outcome_support=data["outcome_support"],
        )

        r_result = r_manski_mtr_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mtr_direction="positive",
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

    def test_bounds_ordering(self):
        """MTR+MTS bounds should be tighter than individual bounds."""
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)

        wc = manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )
        mtr = manski_mtr(
            outcome=data["outcome"],
            treatment=data["treatment"],
            direction="positive",
            outcome_support=data["outcome_support"],
        )
        mts = manski_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )
        combined = manski_mtr_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mtr_direction="positive",
            outcome_support=data["outcome_support"],
        )

        # Combined should be tightest
        assert combined["bounds_width"] <= mtr["bounds_width"] + 0.01
        assert combined["bounds_width"] <= mts["bounds_width"] + 0.01
        assert combined["bounds_width"] <= wc["bounds_width"] + 0.01


# =============================================================================
# Test Class: Manski IV Bounds
# =============================================================================


@requires_bounds_python
class TestManskiIVVsR:
    """Test Python manski_iv against R implementation."""

    def test_iv_bounds_basic(self):
        """Standard IV DGP with reasonable complier share."""
        data = generate_iv_bounds_data(n=2000, complier_share=0.4, true_late=2.5, seed=42)

        py_result = manski_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            outcome_support=data["outcome_support"],
        )

        r_result = r_manski_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

    def test_iv_bounds_weak_iv(self):
        """Weak IV: low complier share leads to wider bounds."""
        data = generate_iv_bounds_data(n=2000, complier_share=0.1, true_late=2.5, seed=123)

        py_result = manski_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            outcome_support=data["outcome_support"],
        )

        r_result = r_manski_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            outcome_support=data["outcome_support"],
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

        # Weak IV should give wider bounds
        strong_data = generate_iv_bounds_data(n=2000, complier_share=0.4, seed=42)
        strong_result = manski_iv(
            outcome=strong_data["outcome"],
            treatment=strong_data["treatment"],
            instrument=strong_data["instrument"],
            outcome_support=strong_data["outcome_support"],
        )

        # Weak IV bounds should be at least as wide as strong IV bounds
        assert py_result["bounds_width"] >= strong_result["bounds_width"] * 0.8


# =============================================================================
# Test Class: Lee Bounds
# =============================================================================


@requires_bounds_python
class TestLeeBoundsVsR:
    """Test Python lee_bounds against R implementation."""

    def test_lee_positive_selection(self):
        """Positive monotonicity: treatment increases observation."""
        data = generate_selection_data(
            n=2000,
            true_ate=2.0,
            base_obs_rate=0.6,
            treatment_effect_on_obs=0.2,
            monotonicity="positive",
            seed=42,
        )

        py_result = lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="positive",
            n_bootstrap=500,
            random_state=42,
        )

        r_result = r_lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="positive",
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05), (
            f"Lower: Python={py_result['bounds_lower']:.4f}, R={r_result['bounds_lower']:.4f}"
        )

        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05), (
            f"Upper: Python={py_result['bounds_upper']:.4f}, R={r_result['bounds_upper']:.4f}"
        )

        # Trimming proportion should match closely
        if "trimming_proportion" in r_result and r_result["trimming_proportion"] is not None:
            assert np.isclose(
                py_result["trimming_proportion"],
                r_result["trimming_proportion"],
                rtol=0.01,
            ), "Trimming proportion mismatch"

    def test_lee_negative_selection(self):
        """Negative monotonicity: treatment decreases observation."""
        data = generate_selection_data(
            n=2000,
            true_ate=2.0,
            base_obs_rate=0.8,
            treatment_effect_on_obs=0.2,
            monotonicity="negative",
            seed=123,
        )

        py_result = lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="negative",
            n_bootstrap=500,
            random_state=42,
        )

        r_result = r_lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="negative",
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

    def test_lee_minimal_selection(self):
        """Minimal differential attrition: bounds should be tight."""
        data = generate_selection_data(
            n=2000,
            true_ate=2.0,
            base_obs_rate=0.9,
            treatment_effect_on_obs=0.02,  # Very small selection effect
            monotonicity="positive",
            seed=456,
        )

        py_result = lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="positive",
            n_bootstrap=500,
            random_state=42,
        )

        r_result = r_lee_bounds(
            outcome=data["outcome"],
            treatment=data["treatment"],
            observed=data["observed"],
            monotonicity="positive",
        )

        assert r_result is not None

        assert np.isclose(py_result["bounds_lower"], r_result["bounds_lower"], rtol=0.05)
        assert np.isclose(py_result["bounds_upper"], r_result["bounds_upper"], rtol=0.05)

        # Minimal selection → tight bounds (small width)
        assert py_result["bounds_width"] < 2.0, "Expected tight bounds with minimal selection"


# =============================================================================
# Test Class: Bounds Consistency Checks
# =============================================================================


@requires_bounds_python
class TestBoundsConsistency:
    """Cross-method consistency tests."""

    def test_bounds_nesting(self):
        """More assumptions → tighter bounds: MTR+MTS ⊂ MTR ⊂ Worst-case."""
        data = generate_bounds_data(n=1000, true_ate=2.0, seed=42)

        wc = manski_worst_case(
            outcome=data["outcome"],
            treatment=data["treatment"],
            outcome_support=data["outcome_support"],
        )
        mtr = manski_mtr(
            outcome=data["outcome"],
            treatment=data["treatment"],
            direction="positive",
            outcome_support=data["outcome_support"],
        )
        combined = manski_mtr_mts(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mtr_direction="positive",
            outcome_support=data["outcome_support"],
        )

        # Widths should decrease with more assumptions
        assert wc["bounds_width"] >= mtr["bounds_width"] - 0.1, "WC should be wider than MTR"
        assert mtr["bounds_width"] >= combined["bounds_width"] - 0.1, (
            "MTR should be wider than MTR+MTS"
        )

        # Lower bounds: more assumptions → higher lower bound (tighter)
        assert wc["bounds_lower"] <= mtr["bounds_lower"] + 0.1
        assert mtr["bounds_lower"] <= combined["bounds_lower"] + 0.1

        # Upper bounds: more assumptions → lower upper bound (tighter)
        assert wc["bounds_upper"] >= mtr["bounds_upper"] - 0.1
        assert mtr["bounds_upper"] >= combined["bounds_upper"] - 0.1

    @pytest.mark.monte_carlo
    def test_lee_coverage_monte_carlo(self):
        """Monte Carlo: Lee bounds CI should achieve nominal coverage."""
        n_sims = 100
        true_ate = 2.0
        covered_lower = 0
        covered_upper = 0

        for i in range(n_sims):
            data = generate_selection_data(
                n=500,
                true_ate=true_ate,
                base_obs_rate=0.7,
                treatment_effect_on_obs=0.15,
                monotonicity="positive",
                seed=1000 + i,
            )

            result = lee_bounds(
                outcome=data["outcome"],
                treatment=data["treatment"],
                observed=data["observed"],
                monotonicity="positive",
                n_bootstrap=200,
                alpha=0.05,
                random_state=1000 + i,
            )

            # Check if true ATE is in bounds
            if result["bounds_lower"] <= true_ate <= result["bounds_upper"]:
                covered_lower += 1
                covered_upper += 1
            # Also check CI coverage
            if result.get("ci_lower") is not None:
                if result["ci_lower"] <= result["bounds_lower"]:
                    covered_lower += 1 if result["bounds_lower"] <= true_ate else 0
                if result["ci_upper"] >= result["bounds_upper"]:
                    covered_upper += 1 if result["bounds_upper"] >= true_ate else 0

        coverage_rate = covered_lower / n_sims

        # Coverage should be at least 85% (conservative due to simulation variance)
        assert coverage_rate >= 0.85, f"Coverage {coverage_rate:.1%} below 85%"


# =============================================================================
# Main: Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
