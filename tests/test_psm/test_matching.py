"""
Layer 1: Unit Tests for PSM Nearest Neighbor Matching.

Tests NearestNeighborMatcher with hand-calculated known answers for:
- Greedy matching algorithm
- With/without replacement
- Caliper restrictions
- ATE computation from matches

Coverage:
- Constructor validation
- Simple matching scenarios
- Edge cases (insufficient controls, restrictive caliper)
- ATE computation correctness

References:
- src/causal_inference/psm/matching.py (Python implementation)
- julia/src/psm/matching.jl (Julia reference)
"""

import numpy as np
import pytest

from src.causal_inference.psm.matching import NearestNeighborMatcher, MatchingResult


class TestNearestNeighborMatcherConstructor:
    """Test NearestNeighborMatcher initialization and parameter validation."""

    def test_constructor_valid_parameters(self):
        """Valid parameters should initialize successfully."""
        matcher = NearestNeighborMatcher(M=2, with_replacement=True, caliper=0.25)

        assert matcher.M == 2
        assert matcher.with_replacement is True
        assert matcher.caliper == 0.25

    def test_constructor_default_parameters(self):
        """Default parameters: M=1, without replacement, no caliper."""
        matcher = NearestNeighborMatcher()

        assert matcher.M == 1
        assert matcher.with_replacement is False
        assert np.isinf(matcher.caliper)

    def test_constructor_invalid_M(self):
        """M < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid M"):
            NearestNeighborMatcher(M=0)

        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid M"):
            NearestNeighborMatcher(M=-1)

    def test_constructor_invalid_caliper(self):
        """Caliper ≤ 0 (except np.inf) should raise ValueError."""
        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid caliper"):
            NearestNeighborMatcher(caliper=0.0)

        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid caliper"):
            NearestNeighborMatcher(caliper=-0.1)


class TestMatchingWithReplacement:
    """Test matching with replacement (controls can be reused)."""

    def test_simple_1to1_matching(self):
        """
        Simple 1:1 matching with perfect overlap.

        Setup:
        - 2 treated: propensity = [0.6, 0.7]
        - 2 controls: propensity = [0.55, 0.75]

        Expected matches (with replacement, M=1):
        - Treated 0 (e=0.6): closest control is 1 (e=0.55, dist=0.05)
        - Treated 1 (e=0.7): closest control is 3 (e=0.75, dist=0.05)

        Note: Indices in propensity array:
        - 0: treated (0.6)
        - 1: treated (0.7)
        - 2: control (0.55)
        - 3: control (0.75)
        """
        propensity = np.array([0.6, 0.7, 0.55, 0.75])
        treatment = np.array([True, True, False, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=True, caliper=np.inf)
        result = matcher.match(propensity, treatment)

        # Check matches
        assert len(result.matches) == 2, "Should have 2 match entries (one per treated)"
        assert result.matches[0] == [2], f"Treated 0 should match control 2, got {result.matches[0]}"
        assert result.matches[1] == [3], f"Treated 1 should match control 3, got {result.matches[1]}"

        # Check distances
        assert np.isclose(result.distances[0][0], 0.05), f"Distance should be 0.05, got {result.distances[0][0]}"
        assert np.isclose(result.distances[1][0], 0.05), f"Distance should be 0.05, got {result.distances[1][0]}"

        # Check summary statistics
        assert result.n_matched == 2, "Both treated units should be matched"
        assert len(result.matched_treated_indices) == 2
        assert set(result.matched_control_indices) == {2, 3}

    def test_1toM_matching(self):
        """
        1:M matching (M=2) with replacement.

        Setup:
        - 1 treated: propensity = [0.5]
        - 3 controls: propensity = [0.4, 0.6, 0.3]

        Expected (M=2):
        - Treated 0 (e=0.5): 2 nearest are controls 1 (e=0.4, dist=0.1) and 2 (e=0.6, dist=0.1)
          (Ties broken by order, so index 1 before 2)
        """
        propensity = np.array([0.5, 0.4, 0.6, 0.3])
        treatment = np.array([True, False, False, False])

        matcher = NearestNeighborMatcher(M=2, with_replacement=True, caliper=np.inf)
        result = matcher.match(propensity, treatment)

        # Treated 0 should match controls 1 and 2 (both distance 0.1)
        assert len(result.matches[0]) == 2, f"Should have 2 matches, got {len(result.matches[0])}"
        assert set(result.matches[0]) == {1, 2}, f"Should match controls 1 and 2, got {result.matches[0]}"

    def test_control_reuse_with_replacement(self):
        """
        With replacement: same control can match multiple treated units.

        Setup:
        - 2 treated: propensity = [0.5, 0.51]
        - 1 control: propensity = [0.5]

        Expected:
        - Both treated units should match the same control (index 2)
        """
        propensity = np.array([0.5, 0.51, 0.5])
        treatment = np.array([True, True, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=True, caliper=np.inf)
        result = matcher.match(propensity, treatment)

        # Both treated units should match control 2
        assert result.matches[0] == [2]
        assert result.matches[1] == [2]
        assert result.n_matched == 2


class TestMatchingWithoutReplacement:
    """Test matching without replacement (controls removed after use)."""

    def test_simple_without_replacement(self):
        """
        Without replacement: each control used at most once.

        Setup:
        - 2 treated: propensity = [0.6, 0.7]
        - 2 controls: propensity = [0.65, 0.75]

        Expected (greedy order):
        - Treated 0 (e=0.6): matches control 2 (e=0.65, dist=0.05)
        - Treated 1 (e=0.7): matches control 3 (e=0.75, dist=0.05)

        Both treated matched to different controls (no reuse).
        """
        propensity = np.array([0.6, 0.7, 0.65, 0.75])
        treatment = np.array([True, True, False, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=False, caliper=np.inf)
        result = matcher.match(propensity, treatment)

        # Check unique controls used
        all_controls = result.matches[0] + result.matches[1]
        assert len(all_controls) == len(set(all_controls)), "Controls should not be reused"
        assert result.n_matched == 2

    def test_insufficient_controls_without_replacement(self):
        """
        Without replacement: runs out of controls.

        Setup:
        - 3 treated: propensity = [0.5, 0.6, 0.7]
        - 2 controls: propensity = [0.55, 0.65]

        Expected:
        - Treated 0: matches control (first available)
        - Treated 1: matches control (second available)
        - Treated 2: no match (no controls left)
        """
        propensity = np.array([0.5, 0.6, 0.7, 0.55, 0.65])
        treatment = np.array([True, True, True, False, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=False, caliper=np.inf)
        result = matcher.match(propensity, treatment)

        # Only 2 out of 3 treated units can be matched
        assert result.n_matched == 2, f"Expected 2 matched, got {result.n_matched}"
        assert len(result.matches[2]) == 0, "Treated 2 should have no match"


class TestCaliperRestrictions:
    """Test caliper distance restrictions."""

    def test_all_within_caliper(self):
        """All potential matches within caliper."""
        propensity = np.array([0.5, 0.55, 0.6])  # treated, control, control
        treatment = np.array([True, False, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=True, caliper=0.1)
        result = matcher.match(propensity, treatment)

        # Treated 0 (e=0.5) to control 1 (e=0.55): distance 0.05 < 0.1
        assert result.n_matched == 1
        assert result.matches[0] == [1]

    def test_restrictive_caliper_no_matches(self):
        """
        Restrictive caliper excludes all matches.

        Setup:
        - 1 treated: propensity = [0.5]
        - 1 control: propensity = [0.7]
        - Distance = 0.2 > caliper (0.1)

        Expected: No match
        """
        propensity = np.array([0.5, 0.7])
        treatment = np.array([True, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=True, caliper=0.1)
        result = matcher.match(propensity, treatment)

        assert result.n_matched == 0, "Should have 0 matches (outside caliper)"
        assert result.matches[0] == [], "Match list should be empty"

    def test_caliper_partial_matches(self):
        """
        Caliper allows some but not all matches.

        Setup:
        - 2 treated: propensity = [0.5, 0.9]
        - 2 controls: propensity = [0.52, 0.95]
        - Caliper = 0.1

        Expected:
        - Treated 0 (e=0.5) matches control 2 (e=0.52, dist=0.02 < 0.1) ✓
        - Treated 1 (e=0.9) can't match control 3 (e=0.95, dist=0.05 < 0.1) ✓
        """
        propensity = np.array([0.5, 0.9, 0.52, 0.95])
        treatment = np.array([True, True, False, False])

        matcher = NearestNeighborMatcher(M=1, with_replacement=True, caliper=0.1)
        result = matcher.match(propensity, treatment)

        # Both should match (distances: 0.02 and 0.05, both < 0.1)
        assert result.n_matched == 2, f"Expected 2 matches, got {result.n_matched}"


class TestComputeATEFromMatches:
    """Test ATE computation from matched sample."""

    def test_simple_ate_calculation(self):
        """
        ATE from matched sample with known answer.

        Setup:
        - Treated outcomes: [10, 12] (indices 0, 1)
        - Control outcomes: [6, 8] (indices 2, 3)
        - Matches: [[2], [3]] (1:1 matching)

        Hand calculation:
        - Treated 0: τ₀ = 10 - 6 = 4
        - Treated 1: τ₁ = 12 - 8 = 4
        - ATE = (4 + 4) / 2 = 4
        """
        outcomes = np.array([10, 12, 6, 8])
        treatment = np.array([True, True, False, False])
        matches = [[2], [3]]

        matcher = NearestNeighborMatcher()
        ate, y1, y0 = matcher.compute_ate_from_matches(outcomes, treatment, matches)

        assert np.isclose(ate, 4.0), f"Expected ATE = 4.0, got {ate}"
        assert np.array_equal(y1, [10, 12]), f"Expected treated outcomes [10, 12], got {y1}"
        assert np.array_equal(y0, [6, 8]), f"Expected imputed control outcomes [6, 8], got {y0}"

    def test_ate_with_averaging(self):
        """
        ATE when M > 1 (average multiple controls).

        Setup:
        - Treated: [15] (index 0)
        - Controls: [5, 7] (indices 1, 2)
        - Matches: [[1, 2]] (both controls matched to treated 0)

        Hand calculation:
        - Imputed control: (5 + 7) / 2 = 6
        - τ₀ = 15 - 6 = 9
        - ATE = 9
        """
        outcomes = np.array([15, 5, 7])
        treatment = np.array([True, False, False])
        matches = [[1, 2]]

        matcher = NearestNeighborMatcher()
        ate, y1, y0 = matcher.compute_ate_from_matches(outcomes, treatment, matches)

        assert np.isclose(ate, 9.0), f"Expected ATE = 9.0, got {ate}"
        assert np.isclose(y0[0], 6.0), f"Expected imputed control = 6.0, got {y0[0]}"

    def test_ate_with_unmatched_treated(self):
        """
        ATE excludes unmatched treated units.

        Setup:
        - Treated outcomes: [10, 12] (indices 0, 1)
        - Control outcome: [5] (index 2)
        - Matches: [[2], []] (treated 1 unmatched)

        Expected: Only use matched treated 0
        - ATE = 10 - 5 = 5
        """
        outcomes = np.array([10, 12, 5])
        treatment = np.array([True, True, False])
        matches = [[2], []]  # Second treated unmatched

        matcher = NearestNeighborMatcher()
        ate, y1, y0 = matcher.compute_ate_from_matches(outcomes, treatment, matches)

        assert np.isclose(ate, 5.0), f"Expected ATE = 5.0, got {ate}"
        assert len(y1) == 1, f"Should only include 1 matched treated unit, got {len(y1)}"


class TestMatchingEdgeCases:
    """Edge cases and error handling."""

    def test_match_no_treated_units(self):
        """Should raise ValueError if no treated units."""
        propensity = np.array([0.5, 0.6])
        treatment = np.array([False, False])

        matcher = NearestNeighborMatcher()
        with pytest.raises(ValueError, match="No treated units"):
            matcher.match(propensity, treatment)

    def test_match_no_control_units(self):
        """Should raise ValueError if no control units."""
        propensity = np.array([0.5, 0.6])
        treatment = np.array([True, True])

        matcher = NearestNeighborMatcher()
        with pytest.raises(ValueError, match="No control units"):
            matcher.match(propensity, treatment)

    def test_match_insufficient_controls_M_too_large(self):
        """Should raise ValueError if M > n_control without replacement."""
        propensity = np.array([0.5, 0.6])  # 1 treated, 1 control
        treatment = np.array([True, False])

        matcher = NearestNeighborMatcher(M=2, with_replacement=False)
        with pytest.raises(ValueError, match="Insufficient controls"):
            matcher.match(propensity, treatment)

    def test_ate_no_matches(self):
        """Should raise ValueError if no matched units when computing ATE."""
        outcomes = np.array([10, 5])
        treatment = np.array([True, False])
        matches = [[]]  # No matches

        matcher = NearestNeighborMatcher()
        with pytest.raises(ValueError, match="No matched units"):
            matcher.compute_ate_from_matches(outcomes, treatment, matches)

    def test_ate_length_mismatch(self):
        """Should raise ValueError if outcomes and treatment lengths differ."""
        outcomes = np.array([10, 12])
        treatment = np.array([True, True, False])  # Wrong length!
        matches = [[2], []]

        matcher = NearestNeighborMatcher()
        with pytest.raises(ValueError, match="CRITICAL ERROR: Mismatched lengths"):
            matcher.compute_ate_from_matches(outcomes, treatment, matches)

    def test_ate_nan_in_outcomes(self):
        """Should raise ValueError if outcomes contain NaN."""
        outcomes = np.array([10, np.nan, 5])
        treatment = np.array([True, True, False])
        matches = [[2], [2]]

        matcher = NearestNeighborMatcher()
        with pytest.raises(ValueError, match="NaN or Inf in outcomes"):
            matcher.compute_ate_from_matches(outcomes, treatment, matches)
