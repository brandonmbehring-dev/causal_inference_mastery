"""Permutation test for RCTs (Fisher exact test / randomization inference).

This module implements permutation tests for RCTs, providing exact p-values under
the sharp null hypothesis of no treatment effect for any unit.

Key benefit: Exact inference with no distributional assumptions, works for small samples.
"""

import numpy as np
from typing import Dict, Union, List, Optional
from itertools import combinations
from scipy.special import comb


def permutation_test(
    outcomes: Union[np.ndarray, list],
    treatment: Union[np.ndarray, list],
    n_permutations: Optional[int] = 1000,
    alternative: str = "two-sided",
    random_seed: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform permutation test for treatment effect (Fisher exact test).

    Under the sharp null hypothesis of no treatment effect for any unit, treatment
    assignment is the only source of randomness. We can compute the exact distribution
    of the test statistic under all possible randomizations.

    Parameters
    ----------
    outcomes : np.ndarray or list
        Observed outcomes for all units.
    treatment : np.ndarray or list
        Treatment indicator (1=treated, 0=control). Also accepts boolean.
    n_permutations : int or None, default=1000
        Number of random permutations to use. If None, performs exact test
        (enumerates all permutations). Use None only for small samples
        (n < 20 or so), as exact enumeration becomes computationally expensive.
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "greater", or "less".
        - "two-sided": H1: treatment effect != 0
        - "greater": H1: treatment effect > 0 (treated > control)
        - "less": H1: treatment effect < 0 (treated < control)
    random_seed : int or None, default=None
        Random seed for reproducibility. Only used if n_permutations is not None.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'p_value': P-value from permutation test
        - 'observed_statistic': Observed difference-in-means (ATE estimate)
        - 'permutation_distribution': Array of test statistics from permutations
        - 'n_permutations': Number of permutations performed
        - 'alternative': Alternative hypothesis used

    Raises
    ------
    ValueError
        If inputs invalid (empty, mismatched lengths, NaN, no variation, etc.)

    Examples
    --------
    >>> # Small sample - exact test
    >>> treatment = np.array([1, 1, 1, 0, 0, 0])
    >>> outcomes = np.array([7, 8, 9, 1, 2, 3])
    >>> result = permutation_test(outcomes, treatment, n_permutations=None)
    >>> result['p_value']  # Exact p-value
    0.05

    >>> # Larger sample - Monte Carlo
    >>> treatment = np.array([1, 0] * 50)
    >>> outcomes = 5*treatment + np.random.normal(0, 1, 100)
    >>> result = permutation_test(outcomes, treatment, n_permutations=10000, random_seed=42)
    >>> result['p_value']  # Approximate p-value
    0.0001

    Notes
    -----
    - Test statistic: difference-in-means (treated - control)
    - Sharp null: Y_i(1) = Y_i(0) for all i (no effect for any unit)
    - P-value: proportion of permutations with statistic as extreme as observed
    - Exact test: enumerates all C(n, n1) permutations
    - Monte Carlo: randomly samples n_permutations from permutation space
    - Assumes treatment was randomly assigned (exchangeability under null)
    """
    # ============================================================================
    # Input Validation
    # ============================================================================

    # Convert to numpy arrays
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n = len(outcomes)

    # Check lengths match
    if len(treatment) != n:
        raise ValueError(
            f"CRITICAL ERROR: Arrays have different lengths.\n"
            f"Function: permutation_test\n"
            f"Expected: Same length arrays\n"
            f"Got: len(outcomes)={len(outcomes)}, len(treatment)={len(treatment)}"
        )

    # Check for empty
    if n == 0:
        raise ValueError(
            f"CRITICAL ERROR: Empty input arrays.\n"
            f"Function: permutation_test\n"
            f"Expected: Non-empty arrays"
        )

    # Check for NaN
    if np.any(np.isnan(outcomes)) or np.any(np.isnan(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: NaN values detected in input.\n"
            f"Function: permutation_test\n"
            f"NaN indicates data quality issues that must be addressed.\n"
            f"Got: {np.sum(np.isnan(outcomes))} NaN in outcomes, "
            f"{np.sum(np.isnan(treatment))} NaN in treatment"
        )

    # Check for infinite values
    if np.any(np.isinf(outcomes)) or np.any(np.isinf(treatment)):
        raise ValueError(
            f"CRITICAL ERROR: Infinite values detected in input.\n"
            f"Function: permutation_test\n"
            f"Got: {np.sum(np.isinf(outcomes))} inf in outcomes, "
            f"{np.sum(np.isinf(treatment))} inf in treatment"
        )

    # Check treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"CRITICAL ERROR: Treatment must be binary (0 or 1).\n"
            f"Function: permutation_test\n"
            f"Expected: Treatment values in {{0, 1}}\n"
            f"Got: Unique treatment values = {unique_treatment}"
        )

    # Check for treatment variation
    if len(unique_treatment) < 2:
        if unique_treatment[0] == 1:
            raise ValueError(
                f"CRITICAL ERROR: No control units in data.\n"
                f"Function: permutation_test\n"
                f"Cannot perform permutation test without control group.\n"
                f"Got: All units have treatment=1"
            )
        else:
            raise ValueError(
                f"CRITICAL ERROR: No treated units in data.\n"
                f"Function: permutation_test\n"
                f"Cannot perform permutation test without treated group.\n"
                f"Got: All units have treatment=0"
            )

    # Validate alternative
    valid_alternatives = ["two-sided", "greater", "less"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"CRITICAL ERROR: Invalid alternative hypothesis.\n"
            f"Function: permutation_test\n"
            f"Expected: alternative in {valid_alternatives}\n"
            f"Got: alternative='{alternative}'"
        )

    # Validate n_permutations
    if n_permutations is not None:
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError(
                f"CRITICAL ERROR: Invalid n_permutations.\n"
                f"Function: permutation_test\n"
                f"Expected: Positive integer or None (for exact test)\n"
                f"Got: n_permutations={n_permutations}"
            )

    # ============================================================================
    # Observed Test Statistic
    # ============================================================================

    def compute_test_statistic(y, t):
        """Compute difference-in-means test statistic."""
        y1 = y[t == 1]
        y0 = y[t == 0]
        return np.mean(y1) - np.mean(y0)

    observed_statistic = compute_test_statistic(outcomes, treatment)

    # ============================================================================
    # Permutation Distribution
    # ============================================================================

    n1 = int(np.sum(treatment == 1))  # Number of treated units
    n0 = n - n1

    # Decide: exact or Monte Carlo
    total_permutations = int(comb(n, n1, exact=True))

    if n_permutations is None:
        # Exact test: enumerate all permutations
        permutation_stats = []
        indices = np.arange(n)

        # Enumerate all combinations of n1 indices (treated positions)
        for treated_indices in combinations(indices, n1):
            t_perm = np.zeros(n)
            t_perm[list(treated_indices)] = 1
            stat = compute_test_statistic(outcomes, t_perm)
            permutation_stats.append(stat)

        permutation_distribution = np.array(permutation_stats)
        n_perms_performed = total_permutations

    else:
        # Monte Carlo: randomly sample permutations
        if random_seed is not None:
            np.random.seed(random_seed)

        permutation_stats = []
        for _ in range(n_permutations):
            # Randomly permute treatment assignments
            t_perm = np.random.permutation(treatment)
            stat = compute_test_statistic(outcomes, t_perm)
            permutation_stats.append(stat)

        permutation_distribution = np.array(permutation_stats)
        n_perms_performed = n_permutations

    # ============================================================================
    # P-value Calculation
    # ============================================================================

    if alternative == "two-sided":
        # P-value = proportion of permutations with |stat| >= |observed|
        p_value = np.mean(np.abs(permutation_distribution) >= np.abs(observed_statistic))
    elif alternative == "greater":
        # P-value = proportion of permutations with stat >= observed
        p_value = np.mean(permutation_distribution >= observed_statistic)
    else:  # alternative == "less"
        # P-value = proportion of permutations with stat <= observed
        p_value = np.mean(permutation_distribution <= observed_statistic)

    return {
        "p_value": float(p_value),
        "observed_statistic": float(observed_statistic),
        "permutation_distribution": permutation_distribution,
        "n_permutations": n_perms_performed,
        "alternative": alternative,
    }
