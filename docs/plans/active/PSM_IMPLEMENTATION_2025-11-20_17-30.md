# Plan: Propensity Score Matching (PSM) Implementation

**Created**: 2025-11-20 17:30
**Status**: NOT_STARTED
**Estimated Time**: 10-15 hours
**Purpose**: Implement nearest neighbor PSM with six-layer validation following RCT validation architecture

---

## Objective

Implement propensity score matching (PSM) estimator for Python with research-grade quality:
1. Nearest neighbor matching (1:1, M:1, with/without replacement, caliper)
2. Propensity score estimation via logistic regression
3. ATE/ATT estimation
4. Abadie-Imbens variance estimation
5. Comprehensive six-layer validation

---

## Current State

**Directories**:
- `src/causal_inference/psm/` - empty
- `tests/test_psm/` - empty
- `tests/validation/adversarial/` - no PSM tests
- `tests/validation/monte_carlo/` - no PSM tests

**Julia Implementation** (reference):
- `julia/src/estimators/psm/` - 6 files (~60KB total)
- Files: matching.jl, nearest_neighbor.jl, propensity.jl, variance.jl, balance.jl, problem.jl
- Fully tested with comprehensive validation

**Decision**: Implement Python version matching Julia's functionality

---

## Target State

**Python Implementation**:
- `src/causal_inference/psm/nearest_neighbor_psm.py` - main estimator (~300 lines)
- `src/causal_inference/psm/matching.py` - matching algorithms (~200 lines)
- `src/causal_inference/psm/propensity.py` - propensity score estimation (~150 lines)
- `src/causal_inference/psm/variance.py` - Abadie-Imbens variance (~200 lines)

**Validation**:
- Layer 1: Known-answer tests (10+ tests, >90% coverage)
- Layer 2: Adversarial tests (15+ tests, method-specific edge cases)
- Layer 3: Monte Carlo validation (5+ tests, 1000 runs each)
- Layer 4: Cross-language validation (Python vs Julia)
- Layer 5: R triangulation (deferred)
- Layer 6: Golden reference (JSON test fixtures)

---

## Detailed Plan

### Phase 1: Infrastructure Setup (1-2 hours)

**Step 1.1: Create module structure** (30 min)
- [ ] Create `src/causal_inference/psm/__init__.py`
- [ ] Create `src/causal_inference/psm/nearest_neighbor_psm.py` (empty skeleton)
- [ ] Create `src/causal_inference/psm/matching.py` (empty skeleton)
- [ ] Create `src/causal_inference/psm/propensity.py` (empty skeleton)
- [ ] Create `src/causal_inference/psm/variance.py` (empty skeleton)

**Step 1.2: Define interfaces** (30 min)
- [ ] Define function signature for `nearest_neighbor_psm_ate()`
- [ ] Define return dict structure (matching RCT estimators)
- [ ] Document parameters and return values

**Step 1.3: Test infrastructure** (30 min)
- [ ] Create `tests/test_psm/conftest.py` with fixtures
- [ ] Create `tests/test_psm/test_nearest_neighbor_psm.py` (empty)
- [ ] Create `tests/validation/adversarial/test_psm_adversarial.py` (empty)
- [ ] Create `tests/validation/monte_carlo/test_monte_carlo_psm.py` (empty)

---

### Phase 2: Propensity Score Estimation (2-3 hours)

**Step 2.1: Implement propensity score estimation** (1 hour)
- [ ] Implement logistic regression via sklearn
- [ ] Input: outcomes, treatment, covariates
- [ ] Output: propensity scores P(T=1|X) for each unit
- [ ] Validation: propensity scores in (0,1), no NaN

**Implementation**:
```python
from sklearn.linear_model import LogisticRegression

def estimate_propensity_scores(treatment, covariates, **kwargs):
    """
    Estimate propensity scores P(T=1|X) via logistic regression.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (0 or 1).
    covariates : np.ndarray
        Covariate matrix (n x p).
    **kwargs
        Additional arguments passed to LogisticRegression.

    Returns
    -------
    propensity_scores : np.ndarray
        P(T=1|X) for each unit.
    model : LogisticRegression
        Fitted logistic regression model.
    """
    # Fit logistic regression
    model = LogisticRegression(**kwargs)
    model.fit(covariates, treatment)

    # Predict propensity scores
    propensity_scores = model.predict_proba(covariates)[:, 1]

    return propensity_scores, model
```

**Step 2.2: Test propensity estimation** (1 hour)
- [ ] Test basic case (simple covariates)
- [ ] Test with perfect separation (propensity → 0 or 1)
- [ ] Test with multicollinearity
- [ ] Test with singular covariate matrix

**Step 2.3: Documentation** (30 min)
- [ ] Docstring with references (Rosenbaum & Rubin 1983)
- [ ] Examples in docstring
- [ ] Notes on assumptions

---

### Phase 3: Matching Algorithm (2-3 hours)

**Step 3.1: Implement nearest neighbor matching** (1.5 hours)
- [ ] Implement matching function
- [ ] Parameters: M (matches per treated), with_replacement, caliper
- [ ] Return: matches dict, distances, n_matched

**Implementation**:
```python
def nearest_neighbor_match(
    propensity_treated,
    propensity_control,
    indices_treated,
    indices_control,
    M=1,
    with_replacement=False,
    caliper=np.inf
):
    """
    Find nearest neighbor matches for treated units.

    Parameters
    ----------
    propensity_treated : np.ndarray
        Propensity scores for treated units.
    propensity_control : np.ndarray
        Propensity scores for control units.
    indices_treated : np.ndarray
        Original indices of treated units.
    indices_control : np.ndarray
        Original indices of control units.
    M : int, default=1
        Number of matches per treated unit.
    with_replacement : bool, default=False
        Allow reusing control units.
    caliper : float, default=np.inf
        Maximum propensity score distance.

    Returns
    -------
    matches : dict
        {treated_idx: [control_idx_1, ..., control_idx_M]}
    distances : dict
        {treated_idx: [dist_1, ..., dist_M]}
    n_matched : int
        Number of successfully matched treated units.
    """
    matches = {}
    distances = {}
    available_controls = set(range(len(propensity_control)))

    for i, e_treated in enumerate(propensity_treated):
        # Compute distances to all available controls
        dists = np.abs(e_treated - propensity_control[list(available_controls)])

        # Find M nearest within caliper
        within_caliper = dists <= caliper
        if np.sum(within_caliper) < M:
            continue  # Skip if not enough matches within caliper

        # Get M nearest
        sorted_idx = np.argsort(dists)
        matched_controls = sorted_idx[:M]

        # Store matches
        matches[indices_treated[i]] = [indices_control[j] for j in matched_controls]
        distances[indices_treated[i]] = [dists[j] for j in matched_controls]

        # Remove from pool if without replacement
        if not with_replacement:
            for j in matched_controls:
                available_controls.remove(j)

    return matches, distances, len(matches)
```

**Step 3.2: Test matching algorithm** (1 hour)
- [ ] Test 1:1 matching without replacement
- [ ] Test M:1 matching (M=2, M=3)
- [ ] Test with replacement
- [ ] Test caliper matching (drop units outside caliper)
- [ ] Test edge case: no matches within caliper
- [ ] Test edge case: more treated than control

**Step 3.3: Documentation** (30 min)
- [ ] Docstring with matching algorithm description
- [ ] References (Abadie & Imbens 2006, Rubin 1973)
- [ ] Examples

---

### Phase 4: ATE Estimation (1-2 hours)

**Step 4.1: Implement PSM ATE estimator** (1 hour)
- [ ] Use matching to pair treated and control units
- [ ] Compute ATE as average difference
- [ ] Handle unmatched units (dropped from analysis)

**Implementation**:
```python
def nearest_neighbor_psm_ate(
    outcomes,
    treatment,
    covariates,
    M=1,
    with_replacement=False,
    caliper=np.inf,
    alpha=0.05
):
    """
    Estimate ATE using nearest neighbor propensity score matching.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes.
    treatment : np.ndarray
        Binary treatment indicator.
    covariates : np.ndarray
        Covariate matrix (n x p).
    M : int, default=1
        Number of matches per treated unit.
    with_replacement : bool, default=False
        Allow reusing control units.
    caliper : float, default=np.inf
        Maximum propensity score distance.
    alpha : float, default=0.05
        Significance level for CI.

    Returns
    -------
    dict
        - 'estimate': ATE estimate
        - 'se': Standard error (Abadie-Imbens)
        - 'ci_lower': Lower CI bound
        - 'ci_upper': Upper CI bound
        - 'n_treated': Number of treated units
        - 'n_matched': Number of matched treated units
        - 'n_control': Number of control units
        - 'propensity_scores': Propensity scores for all units
        - 'match_quality': Matching diagnostics
    """
    # Step 1: Estimate propensity scores
    propensity_scores, model = estimate_propensity_scores(treatment, covariates)

    # Step 2: Separate treated and control
    treated_mask = treatment == 1
    control_mask = treatment == 0

    indices_treated = np.where(treated_mask)[0]
    indices_control = np.where(control_mask)[0]

    # Step 3: Match
    matches, distances, n_matched = nearest_neighbor_match(
        propensity_scores[treated_mask],
        propensity_scores[control_mask],
        indices_treated,
        indices_control,
        M=M,
        with_replacement=with_replacement,
        caliper=caliper
    )

    # Step 4: Compute ATE
    ate_sum = 0.0
    for treated_idx, control_indices in matches.items():
        y_treated = outcomes[treated_idx]
        y_control_mean = np.mean([outcomes[j] for j in control_indices])
        ate_sum += (y_treated - y_control_mean)

    ate = ate_sum / n_matched

    # Step 5: Compute variance (Abadie-Imbens)
    se, var_components = compute_abadie_imbens_variance(
        outcomes, treatment, covariates, propensity_scores,
        matches, M, with_replacement
    )

    # Step 6: Confidence interval
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z_critical * se
    ci_upper = ate + z_critical * se

    # Step 7: Match quality diagnostics
    match_quality = compute_match_quality(
        propensity_scores, treatment, matches, distances
    )

    return {
        "estimate": float(ate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_treated": int(np.sum(treatment == 1)),
        "n_matched": n_matched,
        "n_control": int(np.sum(treatment == 0)),
        "propensity_scores": propensity_scores,
        "match_quality": match_quality
    }
```

**Step 4.2: Test ATE estimation** (1 hour)
- [ ] Test with known ATE (hand-calculated)
- [ ] Test 1:1 matching recovers simple difference-in-means (when propensity = 0.5)
- [ ] Test M:1 matching
- [ ] Test with caliper (verify dropped units)

---

### Phase 5: Abadie-Imbens Variance (2-3 hours)

**Step 5.1: Implement Abadie-Imbens variance** (2 hours)
- [ ] Implement analytic variance formula
- [ ] Account for matching uncertainty
- [ ] Handle with_replacement case

**References**:
- Abadie & Imbens (2006) - variance formula
- Abadie & Imbens (2008) - bootstrap failure with replacement

**Implementation**: Complex formula from Abadie & Imbens (2006), ~100 lines

**Step 5.2: Test variance estimation** (1 hour)
- [ ] Test variance > 0
- [ ] Test SE decreases with larger M
- [ ] Monte Carlo validation (empirical SE ≈ estimated SE)

---

### Phase 6: Layer 1 - Known-Answer Tests (1-2 hours)

**Minimum**: 10 tests, >90% coverage

**Tests**:
- [ ] test_basic_case - hand-calculated with simple data
- [ ] test_zero_effect - ATE = 0
- [ ] test_negative_effect - ATE < 0
- [ ] test_1to1_matching - M=1 without replacement
- [ ] test_Mto1_matching - M=2, M=3
- [ ] test_with_replacement - M=1 with replacement
- [ ] test_caliper_matching - caliper=0.1
- [ ] test_no_matches_within_caliper - all dropped
- [ ] test_balanced_propensities - propensity ≈ 0.5 for all
- [ ] test_extreme_propensities - propensity near 0 or 1

**Expected Runtime**: <5 seconds

---

### Phase 7: Layer 2 - Adversarial Tests (2-3 hours)

**Minimum**: 15 tests, PSM-specific edge cases

**Categories**:

**Matching Edge Cases**:
- [ ] test_no_matches_within_caliper - caliper too strict, all dropped
- [ ] test_more_treated_than_control - n1=100, n0=10
- [ ] test_perfect_propensity_overlap - all propensities identical
- [ ] test_no_common_support - treated propensity > max control propensity

**Propensity Estimation Edge Cases**:
- [ ] test_perfect_separation - covariates perfectly predict treatment
- [ ] test_constant_covariate - zero variance covariate
- [ ] test_multicollinearity - highly correlated covariates

**Sample Size Edge Cases**:
- [ ] test_minimum_sample - n1=5, n0=5
- [ ] test_extreme_imbalance - n1=1, n0=999

**Numerical Stability**:
- [ ] test_extreme_outcome_values - Y ∈ [1e6, 1e7]
- [ ] test_propensity_near_boundary - propensity ∈ {0.001, 0.999}

**Data Quality**:
- [ ] test_nan_outcomes - should error cleanly
- [ ] test_nan_covariates - should error cleanly
- [ ] test_singular_covariate_matrix - should error or warn

**Expected Runtime**: ~10 seconds

---

### Phase 8: Layer 3 - Monte Carlo Validation (2-3 hours)

**Minimum**: 5 tests, 1000 runs each

**DGP Scenarios**:

1. **Simple Confounding**:
   - X ~ N(0, 1)
   - T ~ Bernoulli(logit(0.5 + 0.5*X))
   - Y = 2*T + X + ε
   - True ATE = 2.0

2. **Strong Confounding**:
   - X ~ N(0, 1)
   - T ~ Bernoulli(logit(0.5 + 2*X))
   - Y = 2*T + 2*X + ε
   - True ATE = 2.0

3. **Multiple Covariates**:
   - X1, X2, X3 ~ N(0, 1)
   - T ~ Bernoulli(logit(0.5 + 0.5*(X1+X2+X3)))
   - Y = 2*T + X1 + X2 + X3 + ε
   - True ATE = 2.0

4. **Weak Overlap**:
   - X ~ N(0, 1)
   - T ~ Bernoulli(logit(1.5 + X))
   - Y = 2*T + X + ε
   - True ATE = 2.0
   - Many treated with propensity > 0.9

5. **Heterogeneous Treatment Effects**:
   - X ~ N(0, 1)
   - T ~ Bernoulli(logit(0.5 + 0.5*X))
   - Y = (2 + 0.5*X)*T + X + ε
   - ATE varies with X

**Validation**:
- [ ] Bias < 0.1 (PSM has some bias unlike RCTs)
- [ ] Coverage ∈ [90%, 97%] (93-95% ideal)
- [ ] SE accuracy < 20% (empirical vs estimated)
- [ ] Performance: All tests < 60 seconds

---

### Phase 9: Layer 4 - Cross-Language Validation (1-2 hours)

**Goal**: Verify Python matches Julia implementation

**Tests**:
- [ ] test_python_julia_basic_case
- [ ] test_python_julia_1to1_matching
- [ ] test_python_julia_Mto1_matching
- [ ] test_python_julia_caliper_matching
- [ ] test_python_julia_with_replacement

**Method**:
- Generate test data in Python
- Run Python PSM estimator
- Call Julia PSM estimator via juliacall
- Compare: |ATE_python - ATE_julia| < 1e-6

**Expected Runtime**: ~5 seconds (if juliacall works)

---

### Phase 10: Documentation & Integration (1-2 hours)

- [ ] Update `README.md` with PSM status
- [ ] Update `docs/PYTHON_VALIDATION_ARCHITECTURE.md`
- [ ] Create `docs/PSM_IMPLEMENTATION_SUMMARY_2025-11-20.md`
- [ ] Update `CURRENT_WORK.md`
- [ ] Move plan to `docs/plans/implemented/`

---

## PSM-Specific Edge Cases

**Identified for adversarial tests**:

1. **No matches within caliper** - all units dropped, n_matched=0
2. **More treated than control** - some treated units unmatched
3. **Perfect propensity overlap** - all propensities identical (matching arbitrary)
4. **No common support** - treated propensities > all control propensities
5. **Perfect separation in covariates** - logistic regression fails to converge
6. **Singular covariate matrix** - logistic regression fails
7. **Propensity near 0 or 1** - numerical instability
8. **Extreme M** - M > n_control (impossible to match all)

---

## Success Criteria

- [ ] All Layer 1 tests passing (10+ tests, >90% coverage)
- [ ] All Layer 2 tests passing (15+ tests)
- [ ] All Layer 3 tests passing (5+ tests, bias < 0.1, coverage 90-97%)
- [ ] Layer 4 tests passing (if juliacall works) OR deferred with justification
- [ ] Documentation complete
- [ ] Test runtime < 90 seconds (all layers combined)
- [ ] No xfail tests (all issues resolved)

---

## Time Estimates

| Phase | Description | Estimate |
|-------|-------------|----------|
| 1 | Infrastructure setup | 1-2 hours |
| 2 | Propensity estimation | 2-3 hours |
| 3 | Matching algorithm | 2-3 hours |
| 4 | ATE estimation | 1-2 hours |
| 5 | Abadie-Imbens variance | 2-3 hours |
| 6 | Layer 1 tests | 1-2 hours |
| 7 | Layer 2 tests | 2-3 hours |
| 8 | Layer 3 tests | 2-3 hours |
| 9 | Layer 4 tests | 1-2 hours |
| 10 | Documentation | 1-2 hours |
| **Total** | | **15-25 hours** |

**Realistic Estimate**: 18-20 hours (mid-range)

---

## References

**Core Papers**:
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching estimators for average treatment effects. *Econometrica*, 74(1), 235-267.
- Abadie, A., & Imbens, G. W. (2008). On the failure of the bootstrap for matching estimators. *Econometrica*, 76(6), 1537-1557.
- Rubin, D. B. (1973). Matching to remove bias in observational studies. *Biometrics*, 159-183.

**Julia Reference Implementation**:
- `julia/src/estimators/psm/` - 6 files, ~60KB, fully tested

---

## Notes

**Bootstrap Warning**: Abadie & Imbens (2008) prove bootstrap FAILS for matching with replacement. Must use Abadie-Imbens analytic variance.

**Matching Order**: Greedy matching is order-dependent. Consider randomizing order for robustness (but reduces reproducibility).

**Common Support**: Caliper matching enforces common support by dropping units outside overlap region. This is good practice.

**M Selection**: Larger M reduces variance but may increase bias. M=1 is typical. M=3-5 can improve efficiency if bias is low.

---

**Status**: READY TO IMPLEMENT
**Next**: Phase 1 - Infrastructure Setup
