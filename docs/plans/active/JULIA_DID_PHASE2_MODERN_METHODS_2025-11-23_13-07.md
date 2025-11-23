# Julia DiD Phase 2: Modern Staggered DiD Methods

**Created**: 2025-11-23 13:07
**Updated**: 2025-11-23 13:07
**Status**: NOT_STARTED
**Estimated Duration**: 12-14 hours

## Objective

Achieve Python-Julia parity for DiD by implementing modern staggered DiD methods in Julia, addressing TWFE bias concerns (CONCERN-11) and completing the causal inference framework.

## Current State

### Julia DiD Implementation
- **Files**:
  - `julia/src/did/types.jl` (356 lines)
  - `julia/src/did/classic_did.jl` (632 lines)
  - `julia/src/did/event_study.jl` (676 lines)
- **Total Lines**: 1,664 lines
- **Estimators**: Classic 2×2 DiD, Event Study
- **Tests**: 174/184 passing (94.6%)
- **Last Session**: 2025-11-22 (Session 17 - Phase 1 completion)

### Python DiD Implementation (Reference)
- **Files**:
  - `src/causal_inference/did/staggered.py` (459 lines)
  - `src/causal_inference/did/callaway_santanna.py` (452 lines)
  - `src/causal_inference/did/sun_abraham.py` (384 lines)
- **Total Lines**: 2,582 lines (includes all estimators)
- **Estimators**: Classic 2×2, Event Study, TWFE, Callaway-Sant'Anna, Sun-Abraham
- **Tests**: 104/108 passing (96.3%)
- **Status**: Complete (Sessions 8-10)

### Gap Analysis
- **Missing in Julia**: ~1,300 lines of staggered DiD infrastructure and modern methods
- **Parity Inversion**: Python is ahead of Julia for DiD (opposite of project goal)

## Target State

### New Files to Create
1. `julia/src/did/staggered.jl` (~500 lines)
   - StaggeredDiDProblem type
   - Cohort identification and validation
   - TWFE estimator with bias warning
   - Helper functions for treatment timing

2. `julia/src/did/callaway_santanna.jl` (~450 lines)
   - CallawaySantAnna estimator type
   - Group-time ATT computation
   - Three aggregation schemes (simple/dynamic/group)
   - Bootstrap inference (unit-level resampling)

3. `julia/src/did/sun_abraham.jl` (~350 lines)
   - SunAbraham estimator type
   - Interaction-weighted regression
   - Cohort × event time interaction indicators
   - Delta method variance estimation

### Test Files to Create
4. `julia/test/did/test_staggered.jl` (~600 lines)
   - Layer 1: Known-answer tests (homogeneous/heterogeneous effects)
   - Layer 2: Edge cases (many cohorts, small cohorts, single cohort)
   - TWFE bias demonstration

5. `julia/test/did/test_callaway_santanna.jl` (~400 lines)
   - Layer 1: Known effects with simple aggregation
   - Dynamic effects aggregation
   - Control group selection (nevertreated vs notyettreated)
   - Bootstrap stability

6. `julia/test/did/test_sun_abraham.jl` (~300 lines)
   - Layer 1: Known effects with never-treated controls
   - Event time coefficient recovery
   - Variance estimation

7. `julia/test/did/test_pycall_staggered.jl` (~200 lines)
   - Cross-validation: Julia ↔ Python for all 3 new estimators
   - Target precision: estimates <1e-10, SEs <1e-6

### Expected Additions
- **Total New Lines**: ~2,800 lines (implementation + tests)
- **New Estimators**: 3 (Staggered TWFE, Callaway-Sant'Anna, Sun-Abraham)
- **New Tests**: ~100 test functions across 4 files
- **Features Added**:
  - Staggered treatment timing support
  - TWFE bias mitigation (modern methods)
  - Bootstrap inference
  - Multiple aggregation schemes
  - Delta method variance

## Detailed Implementation Plan

### Phase 1: Staggered DiD Infrastructure (Est: 2 hours)
**Start**: [Not started]

#### Task 1.1: Create StaggeredDiDProblem Type (30 min)
- [ ] Define `StaggeredDiDProblem{T,P} <: AbstractDiDProblem{T,P}`
- [ ] Add fields: outcomes, unit_id, time, treatment_time (cohort assignment)
- [ ] Constructor with comprehensive validation:
  - [ ] Time-varying treatment allowed (unlike Classic DiD)
  - [ ] Treatment timing must be unit-level (constant within units)
  - [ ] All units must eventually be treated OR never treated
  - [ ] No "switchers" (treatment then control)
- [ ] Reference: Python `StaggeredData` dataclass (`staggered.py:26-144`)

#### Task 1.2: Cohort Identification Functions (30 min)
- [ ] `identify_cohorts(unit_id, time, treatment_time)` → cohort labels
- [ ] `get_cohort_sizes(cohorts)` → Dict with cohort counts
- [ ] `validate_cohort_variation(cohorts)` → check ≥2 cohorts
- [ ] Reference: Python `_identify_cohorts()` (`staggered.py:147-186`)

#### Task 1.3: TWFE Estimator (45 min)
- [ ] Create `StaggeredTWFE <: AbstractDiDEstimator` struct
- [ ] Implement `solve(problem::StaggeredDiDProblem, ::StaggeredTWFE)`
- [ ] TWFE regression: Y_it = α_i + λ_t + τ·D_it + ε_it
- [ ] Cluster-robust SEs by unit
- [ ] **Add explicit bias warning** when heterogeneous effects suspected
- [ ] Reference: Python `twfe_staggered()` (`staggered.py:189-334`)

#### Task 1.4: Basic Tests (15 min)
- [ ] Test cohort identification with 3 cohorts
- [ ] Test TWFE with known homogeneous effect
- [ ] Test bias warning is displayed

**Completed**: [Not started]

---

### Phase 2: Callaway-Sant'Anna Estimator (Est: 4-5 hours)
**Start**: [Not started]

#### Task 2.1: CallawaySantAnna Type and Structure (30 min)
- [ ] Define `CallawaySantAnna <: AbstractDiDEstimator`
- [ ] Fields: n_bootstrap, control_group (:nevertreated or :notyettreated), aggregation (:simple, :dynamic, :group)
- [ ] Define `CallawaySantAnnaSolution <: AbstractDiDSolution`
- [ ] Fields for group-time ATTs: atts, group_labels, time_labels, aggregated_att
- [ ] Reference: Python class structure (`callaway_santanna.py:28-65`)

#### Task 2.2: Group-Time ATT Computation (90 min)
- [ ] Implement `_compute_group_time_atts(problem, control_group)`
- [ ] For each (group g, time t) pair:
  - [ ] Select comparison units based on control_group strategy
  - [ ] Compute ATT(g,t) using outcome regression or IPW
  - [ ] Store in matrix: rows=groups, cols=times
- [ ] Handle edge cases:
  - [ ] No valid comparison units → NaN
  - [ ] Pre-treatment periods → should be ≈0
- [ ] Reference: Python `_compute_group_time_atts()` (`callaway_santanna.py:271-378`)

#### Task 2.3: Aggregation Schemes (60 min)
- [ ] **Simple aggregation**: Average all post-treatment ATT(g,t)
  - [ ] Weight by group size
  - [ ] Exclude pre-treatment periods
- [ ] **Dynamic aggregation**: Average by event time
  - [ ] Compute event_time = t - g for each (g,t)
  - [ ] Average ATT(g,t) for same event_time
  - [ ] Return time series of effects
- [ ] **Group aggregation**: Average by cohort
  - [ ] Average ATT(g,t) over t for each group g
  - [ ] Return cohort-specific effects
- [ ] Reference: Python aggregation functions (`callaway_santanna.py:381-465`)

#### Task 2.4: Bootstrap Inference (90 min)
- [ ] Implement `_bootstrap_se(problem, estimator, n_bootstrap=250)`
- [ ] Bootstrap procedure:
  - [ ] Resample units (with replacement)
  - [ ] Recompute group-time ATTs on bootstrap sample
  - [ ] Apply same aggregation scheme
  - [ ] Store bootstrap estimate
- [ ] Compute SE as std dev of bootstrap distribution
- [ ] Handle numerical issues (NaN bootstrap samples)
- [ ] Reference: Python `_bootstrap_se()` (`callaway_santanna.py:188-268`)

#### Task 2.5: Main Solve Function (30 min)
- [ ] Implement `solve(problem::StaggeredDiDProblem, est::CallawaySantAnna)`
- [ ] Call group-time ATT computation
- [ ] Apply aggregation scheme
- [ ] Run bootstrap for SE estimation
- [ ] Construct CallawaySantAnnaSolution
- [ ] Return with appropriate retcode
- [ ] Reference: Python `callaway_santanna()` (`callaway_santanna.py:68-185`)

#### Task 2.6: Tests (60 min)
- [ ] Test homogeneous treatment effects (all ATT(g,t) equal)
- [ ] Test heterogeneous effects by cohort
- [ ] Test heterogeneous effects by time (dynamic)
- [ ] Test nevertreated vs notyettreated control group
- [ ] Test all 3 aggregation schemes
- [ ] Test bootstrap stability (SE > 0, reasonable magnitude)

**Completed**: [Not started]

---

### Phase 3: Sun-Abraham Estimator (Est: 3-4 hours)
**Start**: [Not started]

#### Task 3.1: SunAbraham Type and Structure (20 min)
- [ ] Define `SunAbraham <: AbstractDiDEstimator`
- [ ] Fields: omit_period (default: -1), cluster_se (default: true)
- [ ] Define `SunAbrahamSolution <: AbstractDiDSolution`
- [ ] Fields: event_time_atts (event time → ATT), overall_att, ses, cis
- [ ] Reference: Python class structure (`sun_abraham.py:23-56`)

#### Task 3.2: Cohort × Event Time Interaction Indicators (60 min)
- [ ] Implement `_create_interaction_indicators(unit_id, time, treatment_time)`
- [ ] Create indicator for each (cohort c, event_time e) combination
- [ ] Event time: e = time - treatment_time[unit]
- [ ] Exclude omit_period (default: e = -1 for normalization)
- [ ] Only include event times with data (not all cohorts have all event times)
- [ ] Handle never-treated cohort separately (cohort = ∞)
- [ ] Reference: Python `_create_interaction_indicators()` (`sun_abraham.py:261-340`)

#### Task 3.3: Interaction-Weighted Regression (60 min)
- [ ] Implement main solve function
- [ ] Regression: Y_it = α_i + λ_t + Σ_{c,e} β_{c,e}·I(cohort=c, event_time=e) + ε_it
- [ ] Include unit and time fixed effects (TWFE)
- [ ] Use only never-treated units as controls (cohort = ∞)
- [ ] Extract coefficients β_{c,e}
- [ ] Reference: Python `sun_abraham()` (`sun_abraham.py:59-189`)

#### Task 3.4: Delta Method Variance (60 min)
- [ ] Compute variance-covariance matrix of β_{c,e} coefficients
- [ ] For each event time e:
  - [ ] Identify all (c,e) interactions
  - [ ] Weight β_{c,e} by cohort size
  - [ ] Compute weighted average = ATT(e)
- [ ] Delta method: Var(ATT(e)) = w'·Σ·w where w = cohort size weights
- [ ] Cluster-robust covariance if cluster_se=true
- [ ] Reference: Python `_aggregate_by_event_time()` (`sun_abraham.py:192-258`)

#### Task 3.5: Overall ATT Aggregation (20 min)
- [ ] Average event-time ATTs over post-treatment periods
- [ ] Weight by number of observations at each event time
- [ ] Compute SE using delta method with full covariance
- [ ] Reference: Python `_aggregate_overall()` (`sun_abraham.py:343-372`)

#### Task 3.6: Tests (40 min)
- [ ] Test known constant effect (all event times equal)
- [ ] Test dynamic effects (increasing/decreasing over time)
- [ ] Test anticipation effects (pre-treatment non-zero)
- [ ] Test never-treated control requirement (error if no never-treated)
- [ ] Test event time coefficient recovery

**Completed**: [Not started]

---

### Phase 4: Tests and Cross-Validation (Est: 3-4 hours)
**Start**: [Not started]

#### Task 4.1: Comprehensive Staggered DiD Tests (90 min)
- [ ] **File**: `julia/test/did/test_staggered.jl`
- [ ] Layer 1 Known-Answer Tests:
  - [ ] Homogeneous treatment effects (3 cohorts, constant τ=2)
  - [ ] Heterogeneous by cohort (early=3, middle=2, late=1)
  - [ ] Heterogeneous by time (increasing τ(t) = t/2)
  - [ ] TWFE bias demonstration (compare TWFE vs CS vs SA)
- [ ] Layer 2 Edge Cases:
  - [ ] Many cohorts (10 cohorts)
  - [ ] Small cohorts (n=5 per cohort)
  - [ ] Single large cohort + never-treated
  - [ ] Unbalanced panel (missing observations)
- [ ] Constructor validation tests
- [ ] Target: 25-30 tests, ≥90% passing

#### Task 4.2: Callaway-Sant'Anna Tests (60 min)
- [ ] **File**: `julia/test/did/test_callaway_santanna.jl`
- [ ] Simple aggregation with known effect
- [ ] Dynamic aggregation with time-varying effects
- [ ] Group aggregation with cohort-varying effects
- [ ] Control group comparison (nevertreated vs notyettreated)
- [ ] Bootstrap SE stability (run 5 times, SE should be similar)
- [ ] Edge cases: single cohort, all treated same time
- [ ] Target: 15-20 tests, ≥90% passing

#### Task 4.3: Sun-Abraham Tests (45 min)
- [ ] **File**: `julia/test/did/test_sun_abraham.jl`
- [ ] Known constant effects (all event times equal)
- [ ] Dynamic effects (linear/quadratic time trends)
- [ ] Anticipation effects (pre-treatment non-zero)
- [ ] Omit period selection (test -1, -2, 0)
- [ ] Never-treated requirement (error if missing)
- [ ] Target: 12-15 tests, ≥90% passing

#### Task 4.4: PyCall Cross-Validation (90 min)
- [ ] **File**: `julia/test/did/test_pycall_staggered.jl`
- [ ] Import Python staggered DiD functions
- [ ] Test 1: TWFE comparison (Julia vs Python)
  - [ ] Same data, estimates match <1e-10, SEs <1e-6
- [ ] Test 2: Callaway-Sant'Anna comparison
  - [ ] Simple aggregation: estimates <1e-10
  - [ ] Dynamic aggregation: all event time ATTs match
  - [ ] Bootstrap SEs: within 10% (random variation acceptable)
- [ ] Test 3: Sun-Abraham comparison
  - [ ] Event time ATTs match <1e-10
  - [ ] Delta method SEs match <1e-6
- [ ] Test 4: All 3 methods on same data
  - [ ] Verify rank order: TWFE vs CS vs SA (expected patterns)
- [ ] Target: 10-12 tests, 100% passing (cross-validation critical)

**Completed**: [Not started]

---

## Completion Criteria

### Implementation
- [ ] All 3 new estimator files created (staggered.jl, callaway_santanna.jl, sun_abraham.jl)
- [ ] All 4 test files created
- [ ] Types exported from main module
- [ ] No TODOs or FIXMEs in implementation code

### Testing
- [ ] ≥90% test pass rate for new estimators (target: 52-57 new tests)
- [ ] All PyCall cross-validation tests passing (10-12 tests)
- [ ] Bootstrap inference produces reasonable SEs (not NaN, not zero)
- [ ] Edge cases handled gracefully (errors with informative messages)

### Documentation
- [ ] Comprehensive docstrings for all new types and functions
- [ ] Mathematical formulas included (using Julia LaTeX syntax)
- [ ] Examples in docstrings showing usage
- [ ] References to academic papers (Callaway-Sant'Anna 2021, Sun-Abraham 2021)

### Integration
- [ ] No breaking changes to existing Classic DiD or Event Study
- [ ] All 174 existing tests still passing
- [ ] Module loads without errors
- [ ] New estimators accessible via `solve()` interface

### Validation
- [ ] Python-Julia parity achieved (same estimators in both languages)
- [ ] CONCERN-11 (TWFE bias) addressed with modern methods
- [ ] Cross-validation confirms numerical agreement <1e-6

### Git
- [ ] Plan committed before coding starts
- [ ] Implementation commits reference plan
- [ ] No STATUS: INCOMPLETE markers
- [ ] Final commit message includes "Generated with Claude Code" attribution

## Progress Log

- 2025-11-23 13:07: Plan created (comprehensive audit completed)
- [To be updated as work progresses]

## Dependencies

### Blocking Dependencies (Must be available)
- ✅ Julia Classic DiD implementation (baseline for shared utilities)
- ✅ Python staggered DiD implementation (reference for PyCall validation)
- ✅ PyCall.jl configured and working (validated in Session 17)
- ✅ Statistics, LinearAlgebra packages (already used)

### Optional Dependencies (Can work around)
- ⏸️ Bootstrap.jl (will implement custom unit-level bootstrap)
- ⏸️ Distributions.jl (using approximations for statistical functions)

### Parallel Work Possible
- Phase 3 (Edge Cases + Docs) can start after Phase 2 completion
- No need to wait for full Phase 2 before beginning tests

## Technical Decisions

### Design Choices (Following Python Implementation)
1. **Bootstrap**: Unit-level resampling (not block bootstrap) for Callaway-Sant'Anna
2. **Aggregation**: Default to simple aggregation (easiest to interpret)
3. **Control Group**: Default to nevertreated (more conservative than notyettreated)
4. **Omit Period**: Default to -1 (period before treatment) for Sun-Abraham
5. **Cluster SEs**: Always cluster by unit_id (panel data standard)

### Deviations from Python (If needed)
- Julia will use SciML Problem-Estimator-Solution pattern (Python uses functions)
- Julia may use different bootstrap implementation (no dependence on sklearn)
- Julia will prioritize performance (pre-allocation, in-place operations where safe)

### Testing Strategy
- Layer 1 first (known-answer validation)
- Then Layer 2 (edge cases)
- PyCall validation last (ensures parity)
- Target 90% pass rate (not 100% - some edge cases may be challenging)

## References

### Academic Papers
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

### Python Implementation Files (Reference)
- `src/causal_inference/did/staggered.py` (459 lines)
- `src/causal_inference/did/callaway_santanna.py` (452 lines)
- `src/causal_inference/did/sun_abraham.py` (384 lines)
- `src/causal_inference/did/comparison.py` (383 lines)

### Julia Implementation Files (Current)
- `julia/src/did/types.jl` (356 lines)
- `julia/src/did/classic_did.jl` (632 lines)
- `julia/src/did/event_study.jl` (676 lines)

### Prior Session Summaries
- SESSION_17_JULIA_DID_PHASE_1_2025-11-22.md (Phase 1 completion)
- SESSION_8_DID_FOUNDATION_2025-11-21.md (Python Classic DiD)
- SESSION_9_EVENT_STUDY_2025-11-21.md (Python Event Study)
- SESSION_10_MODERN_DID_2025-11-21.md (Python Staggered DiD)

## Risks and Mitigation

### High Risk
1. **Bootstrap Convergence**: Bootstrap may be slow or unstable
   - Mitigation: Start with n_bootstrap=50 for testing, increase to 250 for final
   - Fallback: Asymptotic variance if bootstrap fails

2. **Group-Time ATT Complexity**: Many (g,t) pairs with small samples
   - Mitigation: Filter out pairs with <5 observations
   - Document which pairs are excluded

### Medium Risk
3. **PyCall Validation Failures**: Julia-Python may disagree on edge cases
   - Mitigation: Relax tolerances if algorithmic differences identified
   - Document any known discrepancies

4. **Test Pass Rate Below 90%**: Edge cases harder than expected
   - Mitigation: Focus on Layer 1 first (known-answer tests)
   - Defer challenging edge cases to Phase 3

### Low Risk
5. **Performance Issues**: Staggered DiD may be slow for large datasets
   - Mitigation: Profile if needed, optimize later
   - Document performance characteristics

## Success Metrics

### Quantitative
- New lines of code: 2,000-2,800 (target: 2,400)
- New tests: 50-60 (target: 55)
- Test pass rate: ≥90% (target: 95%)
- PyCall validation: 100% passing (mandatory)
- Time to completion: 12-14 hours (target: 13 hours)

### Qualitative
- Code quality: Readable, well-documented, follows SciML patterns
- Test coverage: Comprehensive Layer 1 + reasonable Layer 2
- Python parity: All estimators available in both languages
- Methodological rigor: Proper implementation of modern DiD methods

## Post-Completion Next Steps

After Phase 2 completion:
1. **Phase 3**: Fix Event Study edge cases (8 failing tests → 0)
2. **Phase 4**: Extended cross-validation and integration testing
3. **Optional**: Performance optimization if needed
4. **Optional**: Enhanced documentation and usage examples

This plan will be marked COMPLETED and moved to `docs/plans/implemented/` when all completion criteria are met.
