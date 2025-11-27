# Session 10: Modern DiD Methods - Implementation Plan

**Created**: 2025-11-21
**Updated**: 2025-11-21
**STATUS**: NOT_STARTED
**Estimated Duration**: 12-14 hours

---

## Objective

Implement modern Difference-in-Differences estimators (Callaway-Sant'Anna and Sun-Abraham) that address TWFE bias in staggered adoption settings with heterogeneous treatment effects.

**Key Deliverables**:
1. Staggered DiD implementation (time-varying treatment timing)
2. TWFE bias demonstration with simulation
3. Callaway-Sant'Anna estimator (group-time ATTs, then aggregation)
4. Sun-Abraham estimator (interaction-weighted estimator)
5. Comparison function (TWFE vs CS vs SA)

**Methodological Concerns Addressed**:
- CONCERN-11: TWFE bias with heterogeneous treatment effects + staggered adoption

---

## Current State

**Completed**:
- ✅ Session 8: DiD Foundation (2×2 DiD, cluster SEs, parallel trends)
- ✅ Session 9: Event Study Design (leads/lags, joint F-test, dynamic effects)
- ✅ 78 tests passing (27 + 41 from Session 8, 25 + 12 from Session 9)
- ✅ CONCERN-12 addressed (event study pre-trends diagnostics)
- ✅ CONCERN-13 addressed (cluster-robust SEs)

**Current Files**:
- `src/causal_inference/did/did_estimator.py` (410 lines) - 2×2 DiD
- `src/causal_inference/did/event_study.py` (525 lines) - Event study
- `tests/test_did/conftest.py` (521 lines, 9 fixtures)
- `tests/test_did/test_did_known_answers.py` (509 lines, 27 tests)
- `tests/test_did/test_event_study.py` (702 lines, 25 tests)
- `tests/validation/adversarial/test_did_adversarial.py` (500 lines, 14 tests)
- `tests/validation/adversarial/test_event_study_adversarial.py` (538 lines, 12 tests)

**Current Capabilities**:
- Classic 2×2 DiD (single treatment time for all treated units)
- Event study design (dynamic effects, single treatment time)
- Cluster-robust standard errors
- Pre-trends diagnostics (joint F-test, individual leads)

**Limitations (CONCERN-11)**:
- **TWFE biased with staggered adoption + heterogeneous effects**
  - Units treated at different times (e.g., some at t=3, others at t=5)
  - Treatment effects vary across groups or over time
  - TWFE uses "already treated" units as controls (forbidden comparison)
  - Results in negative weight problem and biased estimates

---

## Target State

**New Files to Create**:

1. **`src/causal_inference/did/staggered.py`** (~400 lines)
   - Data structures for staggered adoption
   - Utility functions for group/cohort identification
   - TWFE with staggered timing (to demonstrate bias)
   - Validation and input checks

2. **`src/causal_inference/did/callaway_santanna.py`** (~350 lines)
   - `callaway_santanna_ate()` - Main estimator
   - Group-time ATT estimation (cohort × time)
   - Aggregation functions (simple, dynamic, group-specific)
   - Bootstrap inference

3. **`src/causal_inference/did/sun_abraham.py`** (~300 lines)
   - `sun_abraham_ate()` - Main estimator
   - Interaction-weighted estimator
   - Never-treated and not-yet-treated controls
   - Variance estimation

4. **`src/causal_inference/did/comparison.py`** (~200 lines)
   - `compare_did_methods()` - Compare TWFE, CS, SA
   - Bias simulation demonstration
   - Visualization functions

5. **`tests/test_did/test_staggered.py`** (~500 lines)
   - Layer 1: Known-answer tests for CS and SA
   - TWFE bias demonstration tests
   - Input validation tests
   - Aggregation tests

6. **`tests/validation/adversarial/test_modern_did_adversarial.py`** (~200 lines)
   - Layer 2: Edge cases for CS and SA
   - Extreme staggering patterns
   - Small cohorts
   - Many cohorts

7. **`docs/SESSION_10_MODERN_DID_2025-11-21.md`** (comprehensive session summary)

**Expected Total**: ~1,950 lines of new code + documentation

**New Capabilities**:
- Staggered DiD with heterogeneous treatment effects (unbiased)
- Group-time ATTs with multiple aggregations
- Comparison across methods (TWFE vs CS vs SA)
- TWFE bias demonstration with simulations
- Bootstrap standard errors for CS and SA

---

## Detailed Plan

### Phase 1: Staggered DiD Implementation (~3-4 hours)

**Start**: [TBD]
**End**: [TBD]

#### 1.1 Create `src/causal_inference/did/staggered.py`

**Core data structures**:
```python
@dataclass
class StaggeredData:
    """Container for staggered DiD data."""
    outcomes: np.ndarray
    treatment: np.ndarray  # 0/1 for each unit-time
    time: np.ndarray
    unit_id: np.ndarray
    treatment_time: np.ndarray  # Treatment time for each unit (inf if never treated)

    @property
    def cohorts(self) -> np.ndarray:
        """Unique treatment times (cohorts/groups)."""
        return np.unique(self.treatment_time[np.isfinite(self.treatment_time)])

    @property
    def never_treated(self) -> np.ndarray:
        """Units never treated."""
        return np.isinf(self.treatment_time)
```

**Key functions**:
```python
def create_staggered_data(...) -> StaggeredData:
    """Create StaggeredData from arrays."""

def identify_cohorts(treatment_time: np.ndarray) -> Dict[int, np.ndarray]:
    """Map cohort (treatment time) to unit IDs."""

def twfe_staggered(data: StaggeredData, cluster_se: bool = True) -> Dict[str, Any]:
    """
    TWFE with staggered adoption (biased estimator for comparison).

    Y_it = α_i + λ_t + β·D_it + ε_it

    WARNING: This estimator is BIASED with heterogeneous effects.
    Use Callaway-Sant'Anna or Sun-Abraham instead.
    """
```

**Implementation decisions**:
- Use `np.inf` to represent never-treated units
- Store treatment_time per unit (not time-varying)
- Validate staggered structure (some units treated at different times)

---

### Phase 2: Callaway-Sant'Anna Estimator (~3-4 hours)

**Start**: [TBD]
**End**: [TBD]

#### 2.1 Create `src/causal_inference/did/callaway_santanna.py`

**Core estimator**:
```python
def callaway_santanna_ate(
    data: StaggeredData,
    aggregation: str = "simple",  # "simple", "dynamic", "group"
    control_group: str = "nevertreated",  # "nevertreated" or "notyettreated"
    alpha: float = 0.05,
    n_bootstrap: int = 250,
) -> Dict[str, Any]:
    """
    Callaway-Sant'Anna (2021) group-time ATT estimator.

    Two-step procedure:
    1. Estimate ATT(g,t) for each cohort g and time t
       - ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]
       - C: Control group (never-treated or not-yet-treated)

    2. Aggregate ATT(g,t) to summary estimand
       - Simple: Average over all (g,t)
       - Dynamic: Average by event time (t-g)
       - Group: Average by cohort g

    Returns:
        - att: Overall ATT (if aggregation="simple")
        - se: Bootstrap standard error
        - att_gt: Matrix of ATT(g,t) estimates
        - aggregated: Dict with aggregation results
    """
```

**Key implementation steps**:
1. Compute ATT(g,t) for each cohort × time cell
2. Use never-treated or not-yet-treated as control group
3. Aggregate ATT(g,t) using weighted average
4. Bootstrap for standard errors (resample units)

**Aggregations**:
- **Simple**: Average over all ATT(g,t)
- **Dynamic**: ATT by event time (k = t - g)
- **Group**: ATT by cohort g (treatment effect for each cohort)

**Reference**: Callaway & Sant'Anna (2021), Journal of Econometrics

---

### Phase 3: Sun-Abraham Estimator (~2-3 hours)

**Start**: [TBD]
**End**: [TBD]

#### 3.1 Create `src/causal_inference/did/sun_abraham.py`

**Core estimator**:
```python
def sun_abraham_ate(
    data: StaggeredData,
    alpha: float = 0.05,
    cluster_se: bool = True,
) -> Dict[str, Any]:
    """
    Sun-Abraham (2021) interaction-weighted estimator.

    Regression:
        Y_it = α_i + λ_t + Σ_{g,l} β_{g,l}·D_{it}^{g,l} + ε_it

    where:
        - D_{it}^{g,l} = 1{G_i = g}·1{t - G_i = l}
        - g: Cohort (treatment time)
        - l: Event time (periods since treatment)

    Then aggregate:
        ATT = Σ_{g,l} w_{g,l}·β_{g,l}

    where w_{g,l} are share of treated observations in cohort g at event time l.

    Returns:
        - att: Weighted average treatment effect
        - se: Cluster-robust standard error
        - cohort_effects: β_{g,l} for each cohort × event time
        - weights: w_{g,l} weights used
    """
```

**Key differences from TWFE**:
- Saturated model with cohort × event time interactions
- Never-treated and not-yet-treated as clean control group
- Weights based on sample composition (no negative weights)

**Reference**: Sun & Abraham (2021), Journal of Econometrics

---

### Phase 4: TWFE Bias Demonstration (~2 hours)

**Start**: [TBD]
**End**: [TBD]

#### 4.1 Create `src/causal_inference/did/comparison.py`

**Simulation for bias demonstration**:
```python
def demonstrate_twfe_bias(
    n_units: int = 300,
    n_periods: int = 10,
    cohorts: List[int] = [5, 7],
    true_effects: Dict[int, float] = {5: 2.0, 7: 4.0},
    n_sims: int = 1000,
) -> pd.DataFrame:
    """
    Demonstrate TWFE bias with staggered adoption.

    DGP:
    - Cohort 5 treated at t=5 with effect = 2.0
    - Cohort 7 treated at t=7 with effect = 4.0
    - Heterogeneous effects across cohorts

    Returns DataFrame with:
    - Method: "TWFE", "Callaway-Sant'Anna", "Sun-Abraham"
    - Mean estimate over 1000 simulations
    - Bias: estimate - true
    - RMSE, Coverage
    """
```

**Comparison function**:
```python
def compare_did_methods(
    data: StaggeredData,
    true_effect: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compare TWFE, Callaway-Sant'Anna, Sun-Abraham on same data.

    Returns DataFrame with columns:
    - method: "TWFE", "CS", "SA"
    - att: Estimated ATT
    - se: Standard error
    - ci_lower, ci_upper: 95% confidence interval
    - bias: If true_effect provided
    """
```

---

### Phase 5: Layer 1 Tests (~2-3 hours)

**Start**: [TBD]
**End**: [TBD]

**File**: `tests/test_did/test_staggered.py`

**Test fixtures** (add to conftest.py):
1. `staggered_homogeneous_data` - Same effect across cohorts (TWFE unbiased)
2. `staggered_heterogeneous_data` - Different effects (2.0 vs 4.0) (TWFE biased)
3. `staggered_dynamic_data` - Effects vary over time
4. `staggered_never_treated_data` - Some units never treated

**Test cases** (~25 tests):

**TWFE Bias Tests** (5):
- TWFE unbiased with homogeneous effects
- TWFE biased with heterogeneous effects (detect |bias| > 0.5)
- TWFE biased with 3+ cohorts
- TWFE uses forbidden comparisons (already-treated as controls)
- Negative weights demonstration

**Callaway-Sant'Anna Tests** (8):
- Known-answer: Homogeneous effects
- Known-answer: Heterogeneous effects
- Known-answer: Dynamic effects
- Aggregation: Simple vs dynamic vs group
- Control group: Never-treated vs not-yet-treated
- Bootstrap SE positive and reasonable
- No pre-treatment effects (validation)
- Never-treated units used correctly

**Sun-Abraham Tests** (7):
- Known-answer: Homogeneous effects
- Known-answer: Heterogeneous effects
- Cohort-event interactions estimated correctly
- Weights sum to 1.0
- ATT = weighted average of cohort effects
- Cluster SE positive
- Never-treated as clean control

**Input Validation Tests** (5):
- No treated units (error)
- No variation in treatment timing (error)
- Treatment time before first period (error)
- Treatment time after last period (warning)
- Missing never-treated units (warning if control_group="nevertreated")

---

### Phase 6: Layer 2 Adversarial Tests (~1.5-2 hours)

**Start**: [TBD]
**End**: [TBD]

**File**: `tests/validation/adversarial/test_modern_did_adversarial.py`

**Edge cases** (~10 tests):
- Many cohorts (10+ treatment times)
- Few cohorts (2 treatment times only)
- Small cohorts (n=5 per cohort)
- Unbalanced cohorts (90% in one cohort, 10% spread)
- No never-treated units (only not-yet-treated as control)
- Treatment in first period (t=0)
- Treatment in last period (no post-treatment data)
- Single post-treatment period per cohort
- Extreme heterogeneity (effect 1.0 vs 100.0)
- High variance in outcomes

---

### Phase 7: Documentation (~1 hour)

**Start**: [TBD]
**End**: [TBD]

**File**: `docs/SESSION_10_MODERN_DID_2025-11-21.md`

**Contents**:
1. Summary (what was completed, test results)
2. What Was Completed (each phase with details)
3. Key Implementation Decisions
   - Why CS and SA address TWFE bias
   - Group-time ATTs vs interaction-weighted
   - Bootstrap vs analytical SEs
   - Control group choices
4. TWFE Bias Explanation
   - Forbidden comparisons problem
   - Negative weights problem
   - When TWFE is still valid (homogeneous effects)
5. Lessons Learned
6. Files Created/Modified (with line counts)
7. Methodological Concerns Addressed (CONCERN-11)
8. Next Steps (Session 11: IV Foundation)
9. References (Callaway-Sant'Anna 2021, Sun-Abraham 2021, Goodman-Bacon 2021)

**Also update**:
- `docs/CURRENT_WORK.md` - Mark Session 10 complete
- `docs/METHODOLOGICAL_CONCERNS.md` - Mark CONCERN-11 as addressed
- Move this plan to `docs/plans/implemented/`

---

## Implementation Notes

### TWFE Bias Explained

**Problem**: TWFE with staggered adoption uses "already treated" units as implicit controls.

**Example**:
- Cohort A treated at t=5 (effect = 2.0)
- Cohort B treated at t=7 (effect = 4.0)

**At t=6**:
- Cohort A: Post-treatment (Y includes effect 2.0)
- Cohort B: Pre-treatment (Y does not include effect yet)
- DiD comparison: Cohort A (treated) vs Cohort B (control) ✓

**At t=8**:
- Cohort A: Post-treatment (Y includes effect 2.0)
- Cohort B: Post-treatment (Y includes effect 4.0)
- DiD comparison: Cohort B (newly treated) vs Cohort A ("control") ✗
- **Problem**: Cohort A is already treated! Using treated units as controls is forbidden.

**TWFE with heterogeneous effects**:
- Negative weights on some cohort × time cells
- Estimates contaminated by forbidden comparisons
- Can show negative ATT when all true effects are positive

**Reference**: Goodman-Bacon (2021), Journal of Econometrics

### Callaway-Sant'Anna Approach

**Key insight**: Only use valid control groups (never-treated or not-yet-treated).

**Step 1**: Compute ATT(g,t) for each cohort g and time t
- Use only never-treated units as controls (if available)
- Or use not-yet-treated units (must be pre-treatment at time t)

**Step 2**: Aggregate ATT(g,t) to summary estimand
- Simple: Average over all (g,t)
- Dynamic: Average by event time (k = t-g)
- Group: Average by cohort g

**Advantages**:
- No forbidden comparisons
- All weights non-negative
- Identifies clean ATT parameters

**Disadvantage**: More complex than TWFE, requires bootstrap inference

### Sun-Abraham Approach

**Key insight**: Saturated model with cohort × event time interactions, then aggregate.

**Step 1**: Estimate fully saturated model
- One coefficient β_{g,l} for each cohort g and event time l
- Include unit and time fixed effects
- Never-treated as omitted category

**Step 2**: Aggregate β_{g,l} using sample shares
- Weight each β_{g,l} by proportion of treated observations

**Advantages**:
- Implemented as single regression (easier than CS)
- Cluster-robust SEs via statsmodels
- Clean interpretation

**Disadvantage**: Large number of interaction terms with many cohorts/periods

---

## Quality Standards

**Code**:
- Type hints on all functions
- Comprehensive docstrings with math notation
- Black formatted (100-char lines)
- No silent failures (explicit errors)
- Academic references in docstrings

**Tests**:
- 100% pass rate before moving to next phase
- Real assertions (not just "doesn't crash")
- Known-answer validation with hand calculations
- TWFE bias demonstrated empirically (|bias| > 0.5)
- Clear test names and docstrings

**Documentation**:
- Mathematical notation for CS and SA estimators
- Intuitive explanation of TWFE bias problem
- Examples in docstrings
- Session summary with lessons learned
- Academic references

**Statistical Validation**:
- CS and SA unbiased (bias < 0.10) even with heterogeneous effects
- TWFE biased (|bias| > 0.5) with heterogeneous effects
- Coverage 93-97% for CS and SA
- Bootstrap SEs reasonable (not too conservative or liberal)

---

## Risk Assessment

**Low Risk**:
- Building on Session 8-9 DiD foundation
- Well-established modern DiD methods
- Clear academic references for implementation

**Medium Risk**:
- Callaway-Sant'Anna more complex than previous estimators
- Bootstrap inference may be slow (250 bootstrap samples)
- Sun-Abraham requires many interaction terms (can be memory-intensive)

**High Risk**:
- None identified

**Mitigation**:
- Start with simple 2-cohort example for validation
- Reference CS and SA papers closely for correct specification
- Use efficient numpy operations for bootstrap
- Test with small data first, then scale up

---

## Dependencies

**Python packages**:
- numpy, pandas (already installed)
- statsmodels (already installed)
- scipy (already installed)
- matplotlib (for comparison plots)

**Academic papers** (for reference):
1. Callaway & Sant'Anna (2021). "Difference-in-Differences with Multiple Time Periods." Journal of Econometrics 225(2): 200-230.
2. Sun & Abraham (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.
3. Goodman-Bacon (2021). "Difference-in-Differences with Variation in Treatment Timing." Journal of Econometrics 225(2): 254-277.

---

## Timeline

**Optimistic**: 11 hours (if no major issues)
**Realistic**: 12-13 hours
**Pessimistic**: 15 hours (if CS/SA implementation tricky)

**Phase breakdown**:
- Phase 1: 3-4 hours (staggered DiD infrastructure)
- Phase 2: 3-4 hours (Callaway-Sant'Anna)
- Phase 3: 2-3 hours (Sun-Abraham)
- Phase 4: 2 hours (TWFE bias demonstration)
- Phase 5: 2-3 hours (Layer 1 tests)
- Phase 6: 1.5-2 hours (Layer 2 tests)
- Phase 7: 1 hour (documentation)

**Checkpoints**:
- After Phase 1: Staggered data structure works, TWFE biased on simple example
- After Phase 2: CS estimator works on 2-cohort example
- After Phase 3: SA estimator works on 2-cohort example
- After Phase 4: TWFE bias demonstrated empirically
- After Phase 5: All Layer 1 tests passing
- After Phase 6: All Layer 2 tests passing
- After Phase 7: Documentation complete, plan moved to implemented/

---

## Success Criteria

- [ ] Staggered DiD data structure implemented
- [ ] TWFE with staggered timing implemented (for comparison)
- [ ] Callaway-Sant'Anna estimator implemented with 3 aggregations
- [ ] Sun-Abraham estimator implemented
- [ ] TWFE bias demonstrated empirically (|bias| > 0.5 with heterogeneous effects)
- [ ] CS and SA unbiased (bias < 0.10) with heterogeneous effects
- [ ] ≥25 Layer 1 tests, 100% passing
- [ ] ≥10 Layer 2 tests, 100% passing
- [ ] CONCERN-11 addressed (TWFE bias with staggered adoption)
- [ ] Comprehensive documentation created
- [ ] Total time ≤15 hours

---

**STATUS**: NOT_STARTED
**Next Phase**: Phase 1 - Staggered DiD Implementation
**Last Updated**: 2025-11-21
