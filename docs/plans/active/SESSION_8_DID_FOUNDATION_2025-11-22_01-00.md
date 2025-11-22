# Session 8: DiD Foundation

**Created**: 2025-11-22 01:00
**Updated**: 2025-11-22 01:00
**Status**: NOT_STARTED
**Estimated Duration**: 10-12 hours
**Estimated Lines**: ~600 lines (estimator ~250 lines, tests ~350 lines)

---

## Objective

Implement classic 2×2 Difference-in-Differences (DiD) estimator with parallel trends testing, robust standard errors (cluster at unit level), and comprehensive validation. This is the foundation for Phase 3, establishing core DiD functionality before moving to event studies and modern heterogeneity-robust methods.

---

## Current State

**Files**:
- No DiD implementation exists yet in Python
- Julia Phase 3 has complete DiD implementation (reference for cross-validation)

**Capabilities**:
- ✅ RCT estimators complete (Session 4)
- ✅ Observational methods complete (Sessions 5-6: IPW, DR)
- ✅ PSM complete with Monte Carlo (Sessions 1-3, 7)
- ✅ IV complete (Sessions 11-13: 2SLS, LIML, Fuller, GMM)
- ❌ No DiD implementation

**References Available**:
- Angrist & Pischke (2009) - Mostly Harmless Econometrics, Chapter 5
- Cunningham (2021) - Causal Inference: The Mixtape, Chapter 9
- Julia Phase 3 DiD implementation for cross-validation

---

## Target State

**File to Create**:
- `src/causal_inference/did/did_estimator.py` (~250 lines)

**Expected Features**:
- Classic 2×2 DiD estimator (treatment vs control, pre vs post)
- Robust SE with cluster at unit level
- Parallel trends testing (pre-treatment slope comparison)
- Panel data structure support (long format)
- Three inference modes: standard, robust, clustered
- Summary table with diagnostics

**Test Coverage**:
- Layer 1 (Known-Answer): 10-12 tests
- Layer 2 (Adversarial): 8-10 tests (parallel trends violations, unbalanced panel)
- Layer 3: Deferred to Session 10 (combine with modern methods)
- Layer 4: Julia cross-validation

---

## Detailed Plan

### Phase 0: Plan Document Creation (10 minutes)
**Status**: ✅ COMPLETE (01:00)
**Tasks**:
- [x] Create plan document
- [x] Define objective and scope
- [x] Identify implementation approach

### Phase 1: DiD Estimator Implementation (4-5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 01:10
**Estimated Completion**: 06:10

**Tasks**:
- [ ] Create `src/causal_inference/did/` directory
- [ ] Create `src/causal_inference/did/__init__.py`
- [ ] Implement `did_estimator.py`:
  - [ ] DifferenceinDifferences class with .fit() method
  - [ ] Classic 2×2 DiD formula: (Y_post_treat - Y_pre_treat) - (Y_post_control - Y_pre_control)
  - [ ] Regression-based DiD: Y = β₀ + β₁*Post + β₂*Treat + β₃*Post×Treat + ε
    - β₃ is the DiD estimate
  - [ ] Three inference modes: standard, robust, clustered
  - [ ] Cluster-robust SE at unit level (default)
  - [ ] Summary table generation
- [ ] Implement parallel trends testing:
  - [ ] Pre-treatment slope comparison (treatment vs control)
  - [ ] Test H₀: β_trend_treat = β_trend_control
  - [ ] Return p-value and interpretation
- [ ] Input validation:
  - [ ] Check panel structure (unit ID, time period)
  - [ ] Verify treatment assignment consistency
  - [ ] Validate exactly 2 periods (pre, post)

**Expected Structure**:
```python
class DifferenceinDifferences:
    """
    Classic 2×2 Difference-in-Differences estimator.

    Model:
        Y = β₀ + β₁*Post + β₂*Treat + β₃*Post×Treat + ε

    DiD estimate: β₃ (interaction term)

    Parameters
    ----------
    inference : str, default='clustered'
        'standard', 'robust', or 'clustered' (cluster at unit level)
    alpha : float, default=0.05
        Significance level for confidence intervals

    Attributes (after .fit())
    --------------------------
    coef_ : float
        DiD estimate (β₃)
    se_ : float
        Standard error (cluster-robust by default)
    t_stat_ : float
        T-statistic
    p_value_ : float
        P-value (two-sided)
    ci_ : tuple
        (lower, upper) confidence interval
    parallel_trends_pvalue_ : float
        P-value for parallel trends test (pre-treatment)
    n_obs_ : int
        Sample size
    n_treated_ : int
        Number of treated units
    n_control_ : int
        Number of control units
    """

    def __init__(self, inference='clustered', alpha=0.05):
        pass

    def fit(self, Y, treat, post, unit_id=None, X=None):
        """
        Fit DiD estimator.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable
        treat : array-like, shape (n,)
            Treatment indicator (0/1)
        post : array-like, shape (n,)
            Post-treatment indicator (0/1)
        unit_id : array-like, shape (n,), optional
            Unit identifiers for clustering (required if inference='clustered')
        X : array-like, shape (n, k), optional
            Covariates to include in regression

        Returns
        -------
        self : DifferenceinDifferences
            Fitted estimator
        """
        pass

    def summary(self):
        """Return formatted results table."""
        pass

    def _parallel_trends_test(self, Y, treat, post, unit_id):
        """Test parallel trends assumption using pre-treatment data."""
        pass
```

### Phase 2: Test Suite Implementation (4-5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 06:10
**Estimated Completion**: 11:10

**Tasks**:
- [ ] Create `tests/test_did/` directory
- [ ] Create `tests/test_did/__init__.py`
- [ ] Test Class 1: TestDiDKnownAnswers (10-12 tests)
  - [ ] test_did_simple_2x2 - Known DiD effect
  - [ ] test_did_zero_effect - No treatment effect
  - [ ] test_did_with_covariates - DiD with controls
  - [ ] test_did_balanced_panel - Same units pre/post
  - [ ] test_did_different_sample_sizes - Unequal group sizes
  - [ ] test_did_agrees_with_manual - Compare to hand-calculated DiD
  - [ ] test_parallel_trends_satisfied - Pre-treatment slopes equal
  - [ ] test_inference_modes - Standard vs robust vs clustered SE
  - [ ] test_summary_table - Summary generation
  - [ ] test_confidence_intervals - CI coverage
- [ ] Test Class 2: TestDiDAdversarial (8-10 tests)
  - [ ] test_parallel_trends_violation - Detect when slopes differ
  - [ ] test_unbalanced_panel - Missing observations
  - [ ] test_single_treated_unit - Edge case
  - [ ] test_single_control_unit - Edge case
  - [ ] test_perfect_treatment_effect - Y_post = Y_pre + constant
  - [ ] test_time_varying_covariates - X changes over time
  - [ ] test_heteroskedasticity - Unequal variances
  - [ ] test_extreme_outliers - Robust to outliers?
  - [ ] test_invalid_inputs - Error handling
- [ ] Fixtures:
  - [ ] Create balanced panel DGP (same units pre/post)
  - [ ] Create unbalanced panel DGP (some attrition)
  - [ ] Create parallel trends violation DGP
  - [ ] Create heterogeneous treatment effects DGP

**Expected DGP Structure**:
```python
def generate_did_dgp(
    n_treated=100,
    n_control=100,
    true_did=2.0,
    parallel_trends=True,
    seed=None
):
    """
    Generate 2×2 DiD data.

    DGP:
    - Treatment assigned to n_treated units
    - Two periods: t=0 (pre), t=1 (post)
    - Parallel trends if parallel_trends=True:
      - Y₀ᵢₜ = αᵢ + λₜ + εᵢₜ (common time trend λₜ)
      - Y₁ᵢₜ = Y₀ᵢₜ + τ×Post (treatment effect τ only in post period)
    - Violation if parallel_trends=False:
      - Treatment group has different time trend

    Returns
    -------
    Y : array, shape (2n,)
        Outcome (stacked: pre-period, post-period)
    treat : array, shape (2n,)
        Treatment indicator
    post : array, shape (2n,)
        Post-treatment indicator
    unit_id : array, shape (2n,)
        Unit identifiers
    true_did : float
        True DiD effect
    """
    pass
```

### Phase 3: Julia Cross-Validation (1 hour)
**Status**: NOT_STARTED
**Estimated Start**: 11:10
**Estimated Completion**: 12:10

**Tasks**:
- [ ] Create `tests/validation/cross_language/test_did_julia_cross_validation.py`
- [ ] Test 1: Compare DiD estimates (Python vs Julia)
  - [ ] Same DGP → same DiD estimate (< 0.01 difference)
- [ ] Test 2: Compare standard errors
  - [ ] Standard SE match
  - [ ] Cluster-robust SE match
- [ ] Test 3: Parallel trends test consistency
  - [ ] Same p-value for parallel trends test

**Expected Structure**:
```python
import pytest
from julia import Main as jl

@pytest.mark.julia
def test_did_cross_validation_basic():
    """Test Python DiD matches Julia DiD."""
    # Generate data in Python
    Y, treat, post, unit_id, true_did = generate_did_dgp(
        n_treated=100, n_control=100, true_did=2.0, seed=42
    )

    # Python estimate
    did_py = DifferenceinDifferences(inference='robust')
    did_py.fit(Y, treat, post, unit_id)

    # Julia estimate (via PyCall)
    jl.eval('using CausalInferenceMastery')
    did_jl = jl.did_estimator(Y, treat, post, unit_id)

    # Compare
    assert abs(did_py.coef_ - did_jl['estimate']) < 0.01
    assert abs(did_py.se_ - did_jl['se']) < 0.01
```

### Phase 4: Documentation (1-1.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 12:10
**Estimated Completion**: 13:40

**Tasks**:
- [ ] Create `docs/SESSION_8_DID_FOUNDATION_2025-11-22.md`
- [ ] Document DiD implementation approach
- [ ] Document test results (pass rate, coverage)
- [ ] Document parallel trends testing
- [ ] Document Julia cross-validation results
- [ ] Update `src/causal_inference/did/README.md` with:
  - [ ] DiD overview
  - [ ] Use cases and examples
  - [ ] When to use DiD vs other methods
  - [ ] Limitations and assumptions
- [ ] Update `docs/ROADMAP.md`: Mark Session 8 as COMPLETE

### Phase 5: Commit and Wrap-up (10 minutes)
**Status**: NOT_STARTED
**Estimated Start**: 13:40
**Estimated Completion**: 13:50

**Tasks**:
- [ ] Run full test suite: `pytest tests/test_did/ tests/validation/cross_language/ -v`
- [ ] Verify all tests passing
- [ ] Git add all files
- [ ] Git commit with comprehensive message
- [ ] Move plan to docs/plans/implemented/

---

## Decisions Made

*To be filled during implementation*

**Implementation Decisions**:
- (Will document: Regression-based DiD vs simple means, inference mode defaults)

**Testing Decisions**:
- (Will document: DGP design, parallel trends violation approach)

**Cross-Validation Decisions**:
- (Will document: Julia compatibility, data sharing approach)

---

## Testing Strategy

**Coverage Goals**:
- DiD estimator: 90%+ (core implementation)
- Parallel trends test: 100% (critical assumption)
- Inference modes: 100% (standard, robust, clustered)

**Validation Approach**:
- Layer 1 (Known-Answer): Compare to manual DiD calculation
- Layer 2 (Adversarial): Test edge cases and assumption violations
- Layer 3: Deferred to Session 10 (Monte Carlo with modern methods)
- Layer 4 (Julia): Cross-validate estimates and SEs

**Key Tests**:
1. DiD estimate matches hand calculation
2. Parallel trends test detects violations
3. Cluster-robust SE larger than standard SE (typical)
4. Balanced vs unbalanced panel handling
5. Julia cross-validation matches

---

## Success Criteria

- [ ] DiD estimator implemented and tested
- [ ] Parallel trends test functional
- [ ] Three inference modes working
- [ ] 18-22 tests passing (10-12 known-answer + 8-10 adversarial)
- [ ] Julia cross-validation passing
- [ ] Coverage ≥ 90%
- [ ] Session summary document created
- [ ] ROADMAP updated
- [ ] All work committed to git

---

## Notes

**Time Estimate Justification**:
- DiD estimator implementation: 4-5 hours (regression-based, parallel trends, 3 inference modes)
- Test suite: 4-5 hours (18-22 tests, DGP creation)
- Julia cross-validation: 1 hour (PyCall setup, comparison tests)
- Documentation: 1-1.5 hours (session summary, README, roadmap update)
- Total: 10-12 hours

**Risks**:
- Julia cross-validation may require PyCall troubleshooting
- Parallel trends test may need careful pre-treatment data handling
- Cluster-robust SE may require specialized variance calculation

**Mitigation**:
- Review Julia implementation for reference
- Test parallel trends logic carefully with violation DGP
- Use statsmodels or linearmodels for cluster-robust SE

**Defer to Future Sessions**:
- Event study plots (Session 9)
- Dynamic treatment effects (Session 9)
- Staggered adoption (Session 10)
- Modern heterogeneity-robust methods (Session 10)

---

## Methodological Concerns Addressed

**From METHODOLOGICAL_CONCERNS.md**:

- **CONCERN-11**: TWFE bias with staggered adoption
  - Status: NOT YET (deferred to Session 10)
  - This session: Classic 2×2 DiD only (no staggered adoption)

- **CONCERN-12**: Pre-trends testing
  - Status: ✅ ADDRESSED in this session
  - Implementation: Parallel trends test using pre-treatment slopes

- **CONCERN-13**: Cluster-robust SEs
  - Status: ✅ ADDRESSED in this session
  - Implementation: Cluster at unit level (default)

**Note**: Session 8 focuses on classic 2×2 DiD where TWFE bias is not an issue (single treatment cohort, single period). CONCERN-11 will be addressed in Session 10 with modern DiD methods.

---

## References

**Papers**:
- Angrist & Pischke (2009) - Mostly Harmless Econometrics, Chapter 5
- Card & Krueger (1994) - Minimum wage and employment (classic DiD application)

**Textbooks**:
- Cunningham (2021) - Causal Inference: The Mixtape, Chapter 9
- Huntington-Klein (2021) - The Effect, Chapter 18

**Julia Reference**:
- Julia Phase 3 DiD implementation for cross-validation
