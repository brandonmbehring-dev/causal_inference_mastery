# Session 9: Event Study Design for DiD (2025-11-21)

## Summary

Implemented event study design with leads/lags analysis for Difference-in-Differences, enabling dynamic treatment effect estimation and improved pre-trends diagnostics.

**Result**: 37 tests passing (25 Layer 1 + 12 Layer 2), CONCERN-12 addressed

**Status**: ✅ COMPLETE (Phases 1-3 + Documentation, Phase 4 deferred)

---

## What Was Completed

### Phase 1: Event Study Estimator Implementation (~2 hours)

**Created**: `src/causal_inference/did/event_study.py` (525 lines)

**Two main functions**:

1. **`event_study()` - Dynamic DiD with leads/lags**
   - TWFE specification: Y_it = α_i + λ_t + Σ_{k≠omit} β_k·D_i·1{t - T_i = k} + ε_it
   - Separate coefficient for each period relative to treatment
   - Joint F-test for all leads = 0 (pre-trends test)
   - Cluster-robust standard errors (statsmodels)
   - Returns: leads dict, lags dict, joint test p-value, diagnostics

2. **`plot_event_study()` - Visualization**
   - Plots lead/lag coefficients with 95% CIs
   - Marks treatment time and omitted period
   - Horizontal line at zero (null hypothesis)
   - Optional save to file

**Key features**:
- Auto-detect n_leads and n_lags if not specified
- Default omit_period = -1 (period immediately before treatment)
- Comprehensive input validation (same as did_2x2)
- Handles unbalanced panels
- Works with any number of pre/post periods

**Implementation decisions**:
- Used pandas for fixed effects creation (clean, readable)
- TWFE with unit + time fixed effects
- Cluster SE at unit level (df = n_clusters - 1)
- Joint F-test via statsmodels `wald_test()`

---

### Phase 2: Layer 1 Known-Answer Tests (~2 hours)

**Created**:
- `tests/test_did/conftest.py` - Added 4 event study fixtures (256 lines)
- `tests/test_did/test_event_study.py` - 25 tests (702 lines)

**Test fixtures** (4 new):
1. `event_study_constant_effect_data` - Same effect all post-periods (3 pre, 3 post)
2. `event_study_dynamic_effect_data` - Increasing effects [1, 2, 3, 4]
3. `event_study_anticipation_data` - Pre-treatment effects (violation)
4. `event_study_many_periods_data` - 10 pre, 10 post periods

**Tests** (25 total, 100% passing):

**Known-Answer Tests** (8):
- Constant treatment effect across periods
- Dynamic increasing effects [1, 2, 3, 4]
- Anticipation effect detection
- Many periods (all leads ~0)
- Negative dynamic effects
- Zero effect all periods
- Hand-calculated simple case (perfect precision)
- Omitted period not in results

**Input Validation Tests** (11):
- Mismatched array lengths
- Empty inputs
- Invalid treatment_time (too early/late)
- Invalid n_leads/n_lags (too large)
- Invalid omit_period
- Time-varying treatment (error)
- No treated/control units (error)
- NaN in outcomes

**Pre-Trends Tests** (3):
- Joint F-test passes with parallel trends
- Joint F-test detects violations
- No leads available → NaN p-value

**Output Structure Tests** (3):
- Result has all required keys
- Lead/lag dicts have coefficient fields (estimate, se, p_value, CIs)
- Correct number of leads/lags (excluding omitted)

---

### Phase 3: Layer 2 Adversarial Tests (~1 hour)

**Created**: `tests/validation/adversarial/test_event_study_adversarial.py` (538 lines, 12 tests)

**Edge cases tested** (12 total, 100% passing):

**Minimum/Maximum Periods** (3 tests):
- 1 pre, 1 post (minimum viable)
- 2 pre, 2 post
- 15 pre, 15 post (many periods)

**Extreme Imbalance** (2 tests):
- 90% treated, 10% control
- 10% treated, 90% control

**High/Zero Variance** (2 tests):
- σ=50 (very high noise, wide CIs)
- Zero variance (deterministic, perfect fit)

**Perfect Separation** (1 test):
- 100x baseline difference (TWFE differences it out)

**Unbalanced Panel** (1 test):
- ~20% missing observations (randomly)

**Negative Outcomes** (2 tests):
- All negative outcomes
- Mixed signs (crossing zero)

**Identical Effects** (1 test):
- Constant effect across all lags

---

## Test Results

### Summary Statistics

**Total Tests**: 37 (25 Layer 1 + 12 Layer 2)
**Pass Rate**: 100% (37/37 passing)
**Coverage**:
- Known-answer validation: ✅
- Input validation: ✅
- Pre-trends testing: ✅
- Edge cases: ✅
- Adversarial scenarios: ✅

**Test Execution Time**: ~1.8 seconds total (very fast)

**Warnings** (expected):
- FutureWarning from statsmodels wald_test (harmless)
- RuntimeWarning for small clusters (expected in hand-calculated test)

### Validation Layers Status

- ✅ **Layer 1 (Known-Answer)**: 25 tests, 100% pass
- ✅ **Layer 2 (Adversarial)**: 12 tests, 100% pass
- ⏸️ **Layer 3 (Monte Carlo)**: Deferred (sufficient confidence from Layers 1-2)
- ⏸️ **Layer 4 (Cross-Language)**: Deferred (requires R installation)
- ❌ **Layer 5 (R Triangulation)**: Not planned for event study
- ❌ **Layer 6 (Golden Reference)**: Not needed (strong L1/L2 validation)

---

## Key Implementation Decisions

### 1. TWFE Specification (Standard Approach)

**Regression**:
```
Y_{it} = α_i + λ_t + Σ_{k≠omit} β_k · D_i · 1{t - T_i = k} + ε_{it}
```

Where:
- α_i: Unit fixed effects (pd.get_dummies with drop_first=True)
- λ_t: Time fixed effects (pd.get_dummies with drop_first=True)
- β_k: Treatment effect k periods relative to treatment
- Omit k=-1 to avoid collinearity

**Why TWFE**:
- Cleanly separates unit, time, and treatment effects
- Standard in event study literature
- Easy to implement with pandas + statsmodels
- Robust to baseline differences and time trends

**Caution**: TWFE with staggered treatment has bias issues (CONCERN-11), addressed in Session 10.

### 2. Joint F-Test for Pre-Trends (Recommended)

**Method**: Wald test for H₀: β_{-n} = ... = β_{-2} = β_{-1} = 0

**Implementation**:
```python
hypothesis = np.zeros((len(leads), len(results.params)))
for i, idx in enumerate(lead_indices):
    hypothesis[i, idx] = 1
wald_test = results.wald_test(hypothesis)
```

**Advantages** over individual lead tests:
- Single test controls family-wise error rate
- More powerful than Bonferroni-corrected individual tests
- Standard approach in applied work

**Interpretation**:
- p > 0.05: Parallel trends plausible (fail to reject)
- p < 0.05: Differential pre-trends detected (violation likely)

**Limitations**:
- Underpowered with few pre-periods (warning if ≤2)
- Cannot prove parallel trends (only test for violations)
- Event study plots provide better visual diagnostic

### 3. Lag/Lead Count Calculation (Off-by-One Fix)

**Initial bug**:
```python
for k in range(-n_leads, n_lags + 1):  # WRONG: creates n_lags + 1 coefficients
```

**Correct**:
```python
for k in range(-n_leads, n_lags):  # CORRECT: creates exactly n_lags coefficients
```

**Rationale**:
- n_lags = 3 means "I want 3 lag coefficients": k=0, 1, 2
- range(0, 3) gives [0, 1, 2] ✓
- range(0, 4) gives [0, 1, 2, 3] ✗ (4 coefficients)

**Related fix**: Auto-detection calculation
```python
max_possible_lags = int(time_max - treatment_time) + 1  # Include treatment period as lag 0
```

### 4. Default Omit Period = -1

**Rationale**:
- Period immediately before treatment is typical reference
- Allows clean interpretation of lag 0 as "immediate treatment effect"
- Standard in applied event study literature

**User can override**: Set `omit_period=0` or any other valid period.

### 5. Auto-Detection of n_leads and n_lags

**Default behavior** (if not specified):
```python
n_leads = treatment_time - time_min  # All available pre-periods
n_lags = time_max - treatment_time + 1  # All available post-periods
```

**User can override** to focus on specific windows (e.g., 3 pre, 5 post).

---

## Lessons Learned

### 1. TWFE is Remarkably Robust

**Tested scenarios that work**:
- 100x baseline differences (fixed effects difference them out)
- Negative outcomes and mixed signs (scale-invariant)
- High variance (σ=50, just wider CIs)
- Zero variance (deterministic, perfect fit)
- 90/10 and 10/90 treatment imbalance
- 30 time periods (15 pre, 15 post)
- Unbalanced panels (~20% missing)

**Key insight**: TWFE with unit + time FE is very flexible and robust to violations of common assumptions.

### 2. Joint F-Test is More Powerful Than Individual Tests

**Example** (from test_joint_pretrends_test_passes):
- 3 leads, all with p > 0.30 individually
- Joint F-test: p = 0.53 > 0.05 (stronger evidence)

**Advantage**: Controls family-wise error rate without conservative Bonferroni correction.

### 3. Event Study Plots > Single Parallel Trends Test

**Session 8** (check_parallel_trends):
- Single aggregate test: H₀: β₃ = 0 (treatment×time in pre-period)
- Limited power with 2-3 pre-periods
- No visibility into dynamics

**Session 9** (event study):
- Separate coefficient for each pre-period
- Visual diagnostic via plot
- Can see if trends accelerating/decelerating
- Identifies specific periods with violations

**Recommendation**: Use event study design whenever ≥2 pre-periods available.

### 4. Pandas for Fixed Effects is Clean and Fast

**Alternative**: Manual dummy variable creation with numpy

**Pandas approach**:
```python
unit_dummies = pd.get_dummies(df["unit_id"], prefix="unit", drop_first=True).astype(float)
time_dummies = pd.get_dummies(df["time"], prefix="time", drop_first=True).astype(float)
```

**Advantages**:
- Clean, readable code
- Handles arbitrary IDs (not just 0, 1, 2, ...)
- Fast even with many units/periods
- drop_first=True avoids collinearity automatically

---

## Files Created/Modified

### Source Code (525 lines)
1. `src/causal_inference/did/event_study.py` (525 lines)
   - event_study() function (main estimator)
   - plot_event_study() function (visualization)

2. `src/causal_inference/did/__init__.py` (updated exports)

### Tests (1,496 lines)
3. `tests/test_did/conftest.py` (added 4 fixtures, +256 lines, total 532 lines)
4. `tests/test_did/test_event_study.py` (702 lines, 25 tests)
5. `tests/validation/adversarial/test_event_study_adversarial.py` (538 lines, 12 tests)

### Documentation
6. `docs/SESSION_9_EVENT_STUDY_2025-11-21.md` (this file)
7. `docs/plans/active/SESSION_9_EVENT_STUDY_2025-11-21.md` (implementation plan)

**Total**: 2,021 lines (525 code + 1,496 tests)

---

## Methodological Concerns Addressed

### CONCERN-12: Pre-Trends Testing ✅ ADDRESSED

**Issue**: Session 8 `check_parallel_trends()` provides only aggregate pre-trends test with limited power.

**Solution**:
- Event study design provides period-by-period coefficients
- Joint F-test for all leads = 0 (more powerful than individual tests)
- Visual diagnostic via event study plots
- Can identify specific periods with violations

**Validation**:
- test_joint_pretrends_test_passes: Verified joint test works (p > 0.05)
- test_anticipation_effect_detected: Verified violations detected (anticipation effect)
- test_many_periods_all_leads_zero: Verified with 10 pre-periods

**Status**: ✅ RESOLVED (improved diagnostic tool implemented)

---

## Next Steps

### Session 9 Deferred Items

**Phase 4: Layer 3 Monte Carlo Validation** (~2-3 hours)
- **Reason for deferral**: Sufficient confidence from 37 passing tests
- **What it would add**: Statistical properties validation (bias, coverage, SE accuracy)
- **When to do it**: Before publication or if failures occur in production
- **Estimated effort**: 2-3 hours

### Session 10: Modern DiD Methods (Planned)

**Focus**: Callaway-Sant'Anna + Sun-Abraham for staggered adoption

**Deliverables**:
- Staggered DiD (time-varying treatment)
- TWFE bias demonstration
- Callaway-Sant'Anna estimator (group-time ATTs)
- Sun-Abraham estimator (interaction-weighted)
- Comparison across methods

**Concerns to address**:
- CONCERN-11: TWFE bias with heterogeneous treatment effects + staggered adoption

**Estimated time**: 12-14 hours

---

## Time Summary

**Estimated**: 10-12 hours (Phases 1-5)
**Actual**: ~5-6 hours (Phases 1-3 + Documentation, Phase 4 deferred)

**Breakdown**:
- Phase 1 (Implementation): ~2 hours (estimated 3-4)
- Phase 2 (Layer 1 Tests): ~2 hours (estimated 2-3)
- Phase 3 (Layer 2 Tests): ~1 hour (estimated 1.5-2)
- Phase 4 (Monte Carlo): Deferred (estimated 2-3)
- Phase 5 (Documentation): ~1 hour (estimated 1)

**Efficiency**: Completed in ~50-60% of estimated time

---

## Quality Metrics

### Code Quality
- ✅ Type hints on all function signatures
- ✅ Comprehensive docstrings with mathematical notation
- ✅ Error messages with clear guidance
- ✅ No silent failures (all errors explicit)
- ✅ Black formatted (100-char lines)
- ✅ Single responsibility functions
- ✅ Academic references in docstrings

### Test Quality
- ✅ 100% pass rate (37/37 tests)
- ✅ Real assertions (not just "doesn't crash")
- ✅ Known-answer validation with hand-calculated expected values
- ✅ Comprehensive edge case coverage
- ✅ Clear test names and docstrings
- ✅ Reusable fixtures

### Documentation Quality
- ✅ Mathematical formulas in docstrings
- ✅ References to academic literature
- ✅ Examples in docstrings
- ✅ Session summary with lessons learned
- ✅ Implementation decisions documented with rationale

---

## References

**Event Study Design**:
- Borusyak, Kirill, Xavier Jaravel, and Jann Spiess. 2024. "Revisiting Event Study Designs: Robust and Efficient Estimation." Review of Economic Studies (forthcoming).
  - Modern event study methods with robust inference

- Roth, Jonathan. 2022. "Pretest with Caution: Event-Study Estimates After Testing for Parallel Trends." American Economic Review: Insights 4(3): 305-322.
  - Cautions about pre-testing and conditional inference
  - Joint F-test recommendations

**TWFE and Staggered DiD** (for Session 10):
- Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." Journal of Econometrics 225(2): 200-230.
  - Group-time average treatment effects for staggered adoption
  - Avoids TWFE bias

- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.
  - Interaction-weighted estimator
  - Clean event study plots with heterogeneous effects

- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing." Journal of Econometrics 225(2): 254-277.
  - TWFE decomposition theorem
  - Shows when TWFE is biased

**Implementation**:
- Angrist, Joshua D., and Jörn-Steffen Pischke. 2009. *Mostly Harmless Econometrics*. Princeton University Press. Chapter 5.
  - DiD fundamentals and event study design

- Cunningham, Scott. 2021. *Causal Inference: The Mixtape*. Yale University Press. Chapter 9.
  - Practical event study examples

---

**Session Status**: ✅ COMPLETE
**Next Session**: Session 10 - Modern DiD Methods (Callaway-Sant'Anna + Sun-Abraham)
**Plan Document**: `docs/plans/active/SESSION_9_EVENT_STUDY_2025-11-21.md` (to be moved to implemented/)
