# Session 9: Event Study Design - Implementation Plan

**Created**: 2025-11-21
**Updated**: 2025-11-21 (COMPLETED)
**STATUS**: COMPLETED
**Estimated Duration**: 10-12 hours
**Actual Duration**: ~5-6 hours

---

## Objective

Implement event study design for Difference-in-Differences analysis with dynamic treatment effects, enabling analysis of treatment effect heterogeneity over time and improved pre-trends diagnostics.

**Key Deliverables**:
1. Event study estimator with leads/lags coefficients
2. Multiple pre-period testing (test each pre-period separately)
3. Post-treatment dynamics visualization
4. Layer 3 Monte Carlo validation for DiD (5,000 runs × 5 DGPs)

**Methodological Concerns Addressed**:
- CONCERN-12: Pre-trends testing (event study provides better diagnostic than single parallel trends test)

---

## Current State

**Completed**:
- ✅ Session 8: DiD Foundation (2×2 DiD, cluster SEs, parallel trends)
- ✅ 41 tests passing (27 Layer 1 + 14 Layer 2)
- ✅ CONCERN-13 addressed (cluster-robust SEs implemented)

**Current Files**:
- `src/causal_inference/did/did_estimator.py` (410 lines)
- `tests/test_did/conftest.py` (265 lines, 5 fixtures)
- `tests/test_did/test_did_known_answers.py` (509 lines, 27 tests)
- `tests/validation/adversarial/test_did_adversarial.py` (500 lines, 14 tests)

**Current Capabilities**:
- Classic 2×2 DiD with single treatment effect estimate
- Parallel trends test (aggregate pre-treatment differential trend)
- Cluster-robust standard errors

**Limitations**:
- No dynamic treatment effects (assumes constant effect over time)
- Limited pre-trends diagnostics (single aggregate test, not period-by-period)
- No visualization of treatment effect evolution

---

## Target State

**New Files to Create**:
1. `src/causal_inference/did/event_study.py` (~300 lines)
   - `event_study()` function - main estimator
   - `plot_event_study()` function - visualization
   - Helper functions for coefficient extraction

2. `tests/test_did/test_event_study.py` (~400 lines)
   - Layer 1: Known-answer tests with hand-calculated leads/lags
   - Input validation tests
   - Plotting tests

3. `tests/validation/adversarial/test_event_study_adversarial.py` (~200 lines)
   - Layer 2: Edge cases (min periods, extreme imbalance, many leads/lags)

4. `tests/validation/monte_carlo/test_monte_carlo_did.py` (~350 lines)
   - Layer 3: Monte Carlo validation (5,000 runs × 5 DGPs)
   - Bias, coverage, SE accuracy validation

5. `docs/SESSION_9_EVENT_STUDY_2025-11-21.md` (comprehensive session summary)

**Expected Total**: ~1,250 lines of new code + documentation

**New Capabilities**:
- Dynamic treatment effects (separate coefficient for each time period relative to treatment)
- Period-by-period pre-trends testing (test each pre-period lead coefficient = 0)
- Event study plots (coefficients + confidence intervals over time)
- Monte Carlo validation of DiD properties (bias < 0.10, coverage 93-97%)

---

## Detailed Plan

### Phase 1: Event Study Estimator Implementation (~3-4 hours)

**Start**: [TBD]
**End**: [TBD]

#### 1.1 Create `src/causal_inference/did/event_study.py`

**Core function signature**:
```python
def event_study(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: int,
    n_leads: int = None,
    n_lags: int = None,
    alpha: float = 0.05,
    cluster_se: bool = True,
    omit_period: int = -1,  # Period to omit (reference period)
) -> Dict[str, Any]:
    """
    Event study design for DiD with dynamic treatment effects.

    Estimates separate treatment effects for each time period relative to treatment:
        Y_it = α_i + λ_t + Σ_{k≠-1} β_k·D_i·1{t - T_i = k} + ε_it

    where:
        - α_i: Unit fixed effects
        - λ_t: Time fixed effects
        - β_k: Treatment effect k periods relative to treatment
        - D_i: Treatment indicator (1 if unit i ever treated)
        - T_i: Treatment time for unit i
        - k: Time relative to treatment (k<0 = leads, k≥0 = lags)
        - Omit k=-1 to avoid collinearity (reference period)

    Returns coefficients for leads (pre-treatment periods) and lags (post-treatment periods).
    """
```

**Implementation steps**:
- [x] Input validation (same checks as did_2x2 + treatment_time bounds)
- [x] Auto-detect n_leads and n_lags if not provided
- [x] Create relative time indicators (time - treatment_time for each unit)
- [x] Generate dummy variables for each lead/lag (omitting reference period)
- [x] Add unit and time fixed effects
- [x] Fit OLS with cluster-robust SEs (statsmodels)
- [x] Extract lead/lag coefficients with SEs, p-values, CIs
- [x] Return structured output (leads dict, lags dict, diagnostics)

**Key decisions**:
- Default omit_period = -1 (period immediately before treatment)
- Use pandas for fixed effects creation (pd.get_dummies)
- Cluster SE at unit level (same as did_2x2)

#### 1.2 Create plotting function

```python
def plot_event_study(
    result: Dict[str, Any],
    title: str = "Event Study",
    xlabel: str = "Periods Relative to Treatment",
    ylabel: str = "Treatment Effect",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Create event study plot with leads/lags coefficients and confidence intervals.
    """
```

**Implementation**:
- Plot lead coefficients (k < 0) and lag coefficients (k ≥ 0)
- Add confidence intervals (shaded region or error bars)
- Add vertical line at treatment time (k=0)
- Add horizontal line at zero (null hypothesis)
- Mark omitted period clearly
- Optional save to file

---

### Phase 2: Layer 1 Known-Answer Tests (~2-3 hours)

**Start**: [TBD]
**End**: [TBD]

**File**: `tests/test_did/test_event_study.py`

**Test fixtures** (add to conftest.py):
1. `event_study_constant_effect_data` - Same effect in all post-periods
2. `event_study_dynamic_effect_data` - Increasing effect over time (1, 2, 3, 4)
3. `event_study_anticipation_data` - Pre-treatment effects (violation)
4. `event_study_many_periods_data` - 10 pre, 10 post periods

**Test cases** (~20 tests):

**Known-Answer Tests** (8):
- Zero treatment effect (all leads/lags = 0)
- Constant treatment effect (all lags = 2.0, all leads = 0)
- Dynamic increasing effects (lags = [1, 2, 3, 4])
- Dynamic decreasing effects (lags = [4, 3, 2, 1])
- Negative dynamic effects
- Hand-calculated with 3 pre, 3 post periods
- Many periods (10 pre, 10 post)
- Leads all zero (parallel trends hold)

**Input Validation Tests** (6):
- Invalid treatment_time (outside time range)
- Invalid omit_period (outside lead/lag range)
- Invalid n_leads/n_lags (negative, too large)
- Time-varying treatment (should error)
- No pre-treatment periods
- No post-treatment periods

**Pre-Trends Tests** (3):
- All lead coefficients insignificant (parallel trends plausible)
- Some lead coefficients significant (violation detected)
- Joint F-test for all leads = 0

**Output Structure Tests** (3):
- Result dict has required keys (leads, lags, diagnostics)
- Lead/lag dicts have period keys
- Each period has (estimate, se, t_stat, p_value, ci_lower, ci_upper)

---

### Phase 3: Layer 2 Adversarial Tests (~1.5-2 hours)

**Start**: [TBD]
**End**: [TBD]

**File**: `tests/validation/adversarial/test_event_study_adversarial.py`

**Edge cases** (~10 tests):
- Minimum periods (2 pre, 1 post)
- Maximum periods (20 pre, 20 post)
- Extreme imbalance (90/10, 10/90)
- High variance in outcomes
- Perfect separation (100x baseline difference)
- Unbalanced panel (missing observations)
- Negative outcomes
- All lags identical (constant effect)
- Treatment at first period (T=0, no pre-periods available)
- Treatment at last period (T=T_max, no post-periods)

---

### Phase 4: Layer 3 Monte Carlo Validation (~2-3 hours)

**Start**: [TBD]
**End**: [TBD]

**File**: `tests/validation/monte_carlo/test_monte_carlo_did.py`

**Purpose**: Validate DiD estimator statistical properties (bias, coverage, SE accuracy)

**Data Generating Processes** (5 DGPs):

1. **DGP 1: Zero effect, parallel trends**
   - True effect = 0 in all periods
   - Common time trend only
   - Expected: Bias < 0.05, Coverage 93-97%

2. **DGP 2: Constant treatment effect, parallel trends**
   - True effect = 2.0 in all post-periods
   - No anticipation effects
   - Expected: Bias < 0.10, Coverage 93-97%

3. **DGP 3: Dynamic increasing effects**
   - True effects = [1, 2, 3, 4] in periods 0, 1, 2, 3 post-treatment
   - Expected: Each lag estimate recovers true effect within 0.10

4. **DGP 4: High confounding**
   - Baseline differences = 50 between treated/control
   - Strong common trends
   - Expected: DiD differences out confounding, bias < 0.10

5. **DGP 5: Unbalanced panel**
   - Missing observations (~10% of data)
   - Expected: Unbiased but wider CIs, coverage maintained

**Simulation parameters**:
- n_sims = 5,000 per DGP
- n_units = 100 (50 treated, 50 control)
- n_periods = 10 (5 pre, 5 post)
- cluster SE = True

**Validation metrics**:
- Bias: mean(estimate - true) < 0.10
- Coverage: proportion of CIs containing true value ∈ [93%, 97%]
- SE accuracy: std(estimates) / mean(SE estimates) ∈ [0.90, 1.10]

**Tests** (~10 total):
- One test per DGP for bias
- One test per DGP for coverage
- Joint test for SE accuracy across all DGPs

---

### Phase 5: Documentation (~1 hour)

**Start**: [TBD]
**End**: [TBD]

**File**: `docs/SESSION_9_EVENT_STUDY_2025-11-21.md`

**Contents**:
1. Summary (what was completed, test results)
2. What Was Completed (each phase with details)
3. Test Results (total tests, pass rate, coverage)
4. Key Implementation Decisions
   - Why omit k=-1 period
   - Fixed effects vs first differences
   - Joint F-test for pre-trends
5. Lessons Learned
6. Files Created/Modified (with line counts)
7. Methodological Concerns Addressed (CONCERN-12)
8. Next Steps (Session 10: Modern DiD methods)
9. Time Summary (estimated vs actual)
10. Quality Metrics
11. References

**Also update**:
- `docs/CURRENT_WORK.md` - Mark Session 9 complete, update next steps
- Move this plan to `docs/plans/implemented/`

---

## Implementation Notes

### Event Study Regression Specification

**Two-way fixed effects (TWFE) with event time dummies**:

Y_{it} = α_i + λ_t + Σ_{k∈K, k≠-1} β_k · D_i · 1{t - T_i = k} + ε_{it}

Where:
- α_i: Unit fixed effects (differences out time-invariant unit characteristics)
- λ_t: Time fixed effects (differences out common time trends)
- β_k: Treatment effect k periods relative to treatment
- K: Set of all relative periods (e.g., {-5, -4, -3, -2, 0, 1, 2, 3, 4})
- Omit k=-1 to avoid perfect collinearity

**Why TWFE**:
- Cleanly separates unit effects, time effects, and treatment effects
- Standard in event study literature
- Easy to implement with pandas dummies + statsmodels

**Caution** (for Session 10):
- TWFE with staggered treatment timing has bias issues (CONCERN-11)
- Session 9 focuses on single treatment time (classic 2×2 extended to multiple periods)
- Session 10 will implement Callaway-Sant'Anna / Sun-Abraham for staggered timing

### Pre-Trends Testing Approaches

1. **Individual lead tests** (simple):
   - Test each β_k = 0 for k < 0 separately
   - Multiple testing problem (inflate false positive rate)
   - Solution: Bonferroni correction or report unadjusted

2. **Joint F-test** (recommended):
   - H₀: β_{-5} = β_{-4} = ... = β_{-1} = 0
   - Single test for all pre-trends jointly
   - More powerful than individual tests with correction
   - Implementation: Use statsmodels `wald_test()` method

**Include both approaches** in event_study() output:
- Individual lead p-values (for plotting)
- Joint F-test p-value (for formal test)

### Dependencies

- statsmodels (already installed in Session 8)
- matplotlib (for plotting)
- scipy (for stats functions)
- numpy, pandas (already used)

---

## Quality Standards

**Code**:
- Type hints on all functions
- Comprehensive docstrings with math notation
- Black formatted (100-char lines)
- No silent failures (explicit errors)

**Tests**:
- 100% pass rate before moving to next phase
- Real assertions (not just "doesn't crash")
- Known-answer validation with hand calculations
- Clear test names and docstrings

**Documentation**:
- Mathematical notation in docstrings
- Academic references (Angrist & Pischke, Cunningham, Roth 2022)
- Examples in docstrings
- Session summary with lessons learned

**Statistical Validation**:
- Bias < 0.10 for all DGPs
- Coverage 93-97% (nominal 95%)
- SE accuracy within 10% of empirical SD

---

## Risk Assessment

**Low Risk**:
- Building on proven Session 8 foundation
- Well-established event study methodology
- Similar test infrastructure already exists

**Medium Risk**:
- Monte Carlo validation may reveal unexpected biases
- Plotting function may need iteration for clarity
- Joint F-test implementation may be tricky

**High Risk**:
- None identified

**Mitigation**:
- Use statsmodels for TWFE (avoid manual implementation)
- Start with simple 3-period example for validation
- Reference academic papers for correct specification

---

## Timeline

**Optimistic**: 9 hours (if no major issues)
**Realistic**: 10-11 hours
**Pessimistic**: 13 hours (if Monte Carlo reveals issues)

**Phase breakdown**:
- Phase 1: 3-4 hours (estimator + plotting)
- Phase 2: 2-3 hours (Layer 1 tests)
- Phase 3: 1.5-2 hours (Layer 2 tests)
- Phase 4: 2-3 hours (Monte Carlo validation)
- Phase 5: 1 hour (documentation)

**Checkpoints**:
- After Phase 1: Estimator works on simple example
- After Phase 2: All Layer 1 tests passing
- After Phase 3: All Layer 2 tests passing
- After Phase 4: Monte Carlo validation confirms unbiasedness
- After Phase 5: Documentation complete, plan moved to implemented/

---

## Success Criteria

- [x] Event study estimator implemented with leads/lags coefficients
- [x] Cluster-robust standard errors for all coefficients
- [x] Joint F-test for pre-trends implemented
- [x] Event study plots with CIs
- [x] ≥20 Layer 1 tests, 100% passing
- [x] ≥10 Layer 2 tests, 100% passing
- [x] Monte Carlo validation: Bias < 0.10, Coverage 93-97% for all DGPs
- [x] CONCERN-12 addressed (improved pre-trends diagnostics)
- [x] Comprehensive documentation created
- [x] Total time ≤13 hours

---

**STATUS**: IN_PROGRESS
**Phase**: 1 - Event Study Estimator Implementation
**Last Updated**: 2025-11-21
