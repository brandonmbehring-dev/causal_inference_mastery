"""
Nearest neighbor propensity score matching.

Implements greedy nearest neighbor matching algorithm from Julia reference (matching.jl).

Design:
- Greedy algorithm: process treated units sequentially
- For each treated: find M nearest controls by |e(Xᵢ) - e(Xⱼ)|
- With replacement: all controls always available
- Without replacement: remove controls after use
- Caliper: maximum propensity distance allowed

Author: Brandon Behring
Date: 2025-11-21
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
    validate_not_empty,
    validate_in_range,
)


@dataclass
class MatchingResult:
    """
    Result from nearest neighbor matching.

    Attributes:
        matches: List of matched control indices for each treated unit
                 matches[i] = [j1, j2, ..., jM] where j are control indices
        distances: Propensity score distances for each match
                   distances[i] = [d1, d2, ..., dM] where d = |e_i - e_j|
        n_matched: Number of treated units successfully matched (may be < n_treated if caliper restrictive)
        matched_treated_indices: Indices of treated units that were matched
        matched_control_indices: Indices of control units that were used (unique if without replacement)
    """

    matches: List[List[int]]
    distances: List[List[float]]
    n_matched: int
    matched_treated_indices: List[int]
    matched_control_indices: List[int]


class NearestNeighborMatcher:
    """
    Nearest neighbor propensity score matching.

    Matches each treated unit to M nearest control units by propensity score distance.

    Algorithm:
    1. For each treated unit i (in order):
       a. Compute distances |e(Xᵢ) - e(Xⱼ)| to all available controls
       b. Filter controls within caliper distance
       c. Select M nearest controls (by distance)
       d. Store matches and distances
       e. If without replacement: remove matched controls from pool

    Methods:
        match(propensity, treatment, ...): Find nearest neighbor matches

    Example:
        >>> matcher = NearestNeighborMatcher(M=1, with_replacement=False, caliper=0.25)
        >>> result = matcher.match(propensity, treatment)
        >>> print(f"Matched {result.n_matched}/{np.sum(treatment)} treated units")
    """

    def __init__(
        self,
        M: int = 1,
        with_replacement: bool = False,
        caliper: float = np.inf,
    ):
        """
        Initialize nearest neighbor matcher.

        Args:
            M: Number of matches per treated unit (default: 1)
            with_replacement: Allow reusing control units (default: False)
            caliper: Maximum propensity distance allowed (default: np.inf, no restriction)

        Raises:
            ValueError: If M < 1 or caliper invalid
        """
        if M < 1:
            raise ValueError(
                f"CRITICAL ERROR: Invalid M.\n"
                f"Function: NearestNeighborMatcher.__init__\n"
                f"M must be ≥ 1, got M = {M}\n"
                f"M is the number of matches per treated unit."
            )

        if caliper <= 0 and caliper != np.inf:
            raise ValueError(
                f"CRITICAL ERROR: Invalid caliper.\n"
                f"Function: NearestNeighborMatcher.__init__\n"
                f"caliper must be > 0 or np.inf, got caliper = {caliper}\n"
                f"Typical values: 0.1 (strict), 0.25 (moderate), np.inf (no restriction)."
            )

        self.M = M
        self.with_replacement = with_replacement
        self.caliper = caliper

    def match(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray,
    ) -> MatchingResult:
        """
        Find nearest neighbor matches for treated units.

        Implements greedy matching algorithm from Julia matching.jl (lines 56-217).

        Args:
            propensity: Propensity scores for all units (n,)
            treatment: Binary treatment indicator (n,) with {0, 1} or {False, True}

        Returns:
            MatchingResult with matches, distances, and diagnostics

        Raises:
            ValueError: If inputs invalid (wrong shapes, no variation, NaN/Inf)

        Algorithm:
        1. Split into treated and control propensities
        2. Initialize available control pool
        3. For each treated unit:
           a. Compute distances to available controls
           b. Filter by caliper
           c. Select M nearest
           d. Store matches
           e. Update available pool (if without replacement)
        4. Return matches and diagnostics

        Example:
            >>> matcher = NearestNeighborMatcher(M=2, with_replacement=True, caliper=0.2)
            >>> result = matcher.match(propensity, treatment)
            >>> # result.matches[i] contains indices of 2 matched controls for treated unit i
        """
        # ====================================================================
        # Input Validation (using shared utilities)
        # ====================================================================

        propensity = np.asarray(propensity)
        treatment = np.asarray(treatment).astype(bool)

        n = len(propensity)

        # Shared validations
        validate_not_empty(propensity, "propensity")
        validate_finite(propensity, "propensity")
        validate_arrays_same_length(propensity=propensity, treatment=treatment)
        validate_in_range(propensity, "propensity", 0.0, 1.0, inclusive=True)

        # PSM-specific: Check both treated and control groups present
        n_treated = np.sum(treatment)
        n_control = n - n_treated

        if n_treated == 0:
            raise ValueError(
                f"No treated units. All {n} units are in control group. "
                f"Cannot match without treated units."
            )

        if n_control == 0:
            raise ValueError(
                f"No control units. All {n} units are treated. Cannot match without control units."
            )

        # PSM-specific: Check sufficient controls for matching without replacement
        if not self.with_replacement and n_control < self.M:
            raise ValueError(
                f"Insufficient controls for matching without replacement. "
                f"M={self.M} matches requested, but only {n_control} controls available. "
                f"Solutions: (1) Use with_replacement=True, (2) Reduce M, (3) Increase sample size."
            )

        # ====================================================================
        # Extract Treated and Control Indices
        # ====================================================================

        indices_treated = np.where(treatment)[0]
        indices_control = np.where(~treatment)[0]

        propensity_treated = propensity[treatment]
        propensity_control = propensity[~treatment]

        # ====================================================================
        # Initialize Matching Data Structures
        # ====================================================================

        matches: List[List[int]] = []  # matches[i] = list of control indices for treated i
        distances: List[List[float]] = []  # distances[i] = propensity distances
        matched_treated_indices: List[int] = []  # Treated units successfully matched

        # Available controls pool (changes if without replacement)
        if self.with_replacement:
            # With replacement: all controls always available
            available_control_mask = np.ones(n_control, dtype=bool)
        else:
            # Without replacement: track which controls have been used
            available_control_mask = np.ones(n_control, dtype=bool)

        # ====================================================================
        # Greedy Matching Algorithm (Julia matching.jl lines 167-214)
        # ====================================================================

        for i in range(n_treated):
            e_i = propensity_treated[i]  # Propensity for this treated unit

            # Get available control propensities
            if self.with_replacement:
                # All controls always available
                available_propensities = propensity_control
                available_indices = np.arange(n_control)
            else:
                # Only unused controls available
                available_propensities = propensity_control[available_control_mask]
                available_indices = np.where(available_control_mask)[0]

            # Check if any controls available
            if len(available_propensities) == 0:
                # No more controls available (can happen without replacement)
                # This treated unit goes unmatched
                matches.append([])
                distances.append([])
                continue

            # Compute distances to available controls
            dists = np.abs(available_propensities - e_i)

            # Apply caliper filter
            within_caliper = dists <= self.caliper
            valid_controls = available_indices[within_caliper]
            valid_dists = dists[within_caliper]

            if len(valid_controls) == 0:
                # No controls within caliper for this treated unit
                matches.append([])
                distances.append([])
                continue

            # Find M nearest controls (or fewer if not enough available)
            n_matches = min(self.M, len(valid_controls))

            # Sort by distance and take M nearest
            sorted_idx = np.argsort(valid_dists)
            nearest_idx = sorted_idx[:n_matches]

            matched_control_idx = valid_controls[nearest_idx]
            matched_dists = valid_dists[nearest_idx]

            # Store matches (convert to original control indices)
            matched_control_indices_original = indices_control[matched_control_idx]
            matches.append(matched_control_indices_original.tolist())
            distances.append(matched_dists.tolist())
            matched_treated_indices.append(indices_treated[i])

            # Remove matched controls if without replacement
            if not self.with_replacement:
                available_control_mask[matched_control_idx] = False

        # ====================================================================
        # Compute Summary Statistics
        # ====================================================================

        # Count successfully matched treated units
        n_matched = sum(1 for m in matches if len(m) > 0)

        # Get unique matched control indices
        all_matched_controls = []
        for match_list in matches:
            all_matched_controls.extend(match_list)
        matched_control_indices_unique = sorted(set(all_matched_controls))

        # ====================================================================
        # Return Result
        # ====================================================================

        return MatchingResult(
            matches=matches,
            distances=distances,
            n_matched=n_matched,
            matched_treated_indices=matched_treated_indices,
            matched_control_indices=matched_control_indices_unique,
        )

    def compute_ate_from_matches(
        self,
        outcomes: np.ndarray,
        treatment: np.ndarray,
        matches: List[List[int]],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute ATE from matched sample.

        Algorithm:
        1. For each matched treated unit i:
           a. Get treated outcome Y_i
           b. Get control outcomes Y_j for j ∈ matches[i]
           c. Compute imputed control outcome: Ŷᵢ(0) = mean(Y_j)
           d. Individual treatment effect: τᵢ = Y_i - Ŷᵢ(0)
        2. ATE = mean(τᵢ) over all matched treated units

        Args:
            outcomes: Observed outcomes (n,)
            treatment: Binary treatment indicator (n,)
            matches: Match lists from MatchingResult.matches

        Returns:
            Tuple of:
            - ate: Average treatment effect
            - treated_outcomes: Outcomes for matched treated units (n_matched,)
            - imputed_control_outcomes: Imputed control outcomes (n_matched,)

        Raises:
            ValueError: If inputs invalid or no matches

        Example:
            >>> result = matcher.match(propensity, treatment)
            >>> ate, y1, y0 = matcher.compute_ate_from_matches(outcomes, treatment, result.matches)
        """
        # ====================================================================
        # Input Validation
        # ====================================================================

        outcomes = np.asarray(outcomes)
        treatment = np.asarray(treatment).astype(bool)

        n = len(outcomes)

        if len(treatment) != n:
            raise ValueError(
                f"CRITICAL ERROR: Mismatched lengths.\n"
                f"Function: compute_ate_from_matches\n"
                f"outcomes has length {n}, treatment has length {len(treatment)}\n"
                f"All inputs must have same length."
            )

        if np.any(np.isnan(outcomes)) or np.any(np.isinf(outcomes)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in outcomes.\n"
                f"Function: compute_ate_from_matches\n"
                f"Outcomes contain {np.sum(np.isnan(outcomes))} NaN "
                f"and {np.sum(np.isinf(outcomes))} Inf values."
            )

        # Count matched treated units
        n_matched = sum(1 for m in matches if len(m) > 0)

        if n_matched == 0:
            raise ValueError(
                f"CRITICAL ERROR: No matched units.\n"
                f"Function: compute_ate_from_matches\n"
                f"Cannot compute ATE with zero matched pairs.\n"
                f"Check matching results - may need to relax caliper or increase sample size."
            )

        # ====================================================================
        # Compute ATE
        # ====================================================================

        treated_outcomes_list = []
        imputed_control_outcomes_list = []

        indices_treated = np.where(treatment)[0]

        for i, match_list in enumerate(matches):
            if len(match_list) == 0:
                # This treated unit was not matched - skip
                continue

            # Treated outcome (observed)
            treated_idx = indices_treated[i]
            y_treated = outcomes[treated_idx]

            # Imputed control outcome (average of matched controls)
            control_outcomes = outcomes[match_list]
            y_control_imputed = np.mean(control_outcomes)

            treated_outcomes_list.append(y_treated)
            imputed_control_outcomes_list.append(y_control_imputed)

        # Convert to arrays
        treated_outcomes_matched = np.array(treated_outcomes_list)
        imputed_control_outcomes = np.array(imputed_control_outcomes_list)

        # Compute ATE
        individual_effects = treated_outcomes_matched - imputed_control_outcomes
        ate = np.mean(individual_effects)

        return ate, treated_outcomes_matched, imputed_control_outcomes
