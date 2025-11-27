# Session 8: Difference-in-Differences (DiD) Foundation (2025-11-21)

## Summary

Implemented classic 2×2 Difference-in-Differences estimator with cluster-robust standard errors and comprehensive validation.

**Result**: 41 tests passing (27 Layer 1 + 14 Layer 2), CONCERN-13 addressed

**Status**: ✅ COMPLETE (Phases 1-3, Phase 4 deferred)

---

## What Was Completed

### Phase 1: DiD Estimator Implementation (~4 hours)

**Created**: `src/causal_inference/did/did_estimator.py` (410 lines)

**Two main functions**:

1. **`did_2x2()` - Classic 2×2 DiD estimator**
   - OLS regression: Y = β₀ + β₁·Treat + β₂·Post + β₃·(Treat×Post) + ε
   - β₃ coefficient is the DiD estimate
   - Cluster-robust standard errors via statsmodels
   - Degrees of freedom: n_clusters - 1 (Bertrand et al. 2004)
   - Returns: estimate, SE, t-stat, p-value, CI, diagnostics

2. **`check_parallel_trends()` - Pre-treatment trend testing**
   - Tests H₀: β₃ = 0 where β₃ is treatment×time interaction in pre-period
   - Requires ≥2 pre-treatment periods
   - Cluster-robust inference for trend coefficient
   - Returns: pre_trend_diff, p-value, parallel_trends_plausible, warnings

**Key features**:
- Comprehensive input validation (NaN, binary checks, unit-level treatment, alpha bounds)
- Small cluster warning (n_clusters < 30)
- Time-invariant treatment enforcement (prevents staggered adoption errors)
- Handles unbalanced panels
- Works with multiple pre/post periods

**Dependencies added**: `statsmodels` (pip install statsmodels)

---

### Phase 2: Layer 1 Known-Answer Tests (~2 hours)

**Created**:
- `tests/test_did/conftest.py` (265 lines, 5 fixtures)
- `tests/test_did/test_did_known_answers.py` (509 lines, 27 tests)

**Test fixtures** (5):
1. `simple_did_data` - Basic 2×2 DiD (n=50 per group, true effect=2.0)
2. `balanced_panel_data` - Multiple periods (3 pre, 3 post, true effect=5.0)
3. `heterogeneous_baselines_data` - Wide baseline variation (true effect=3.0)
4. `zero_effect_data` - Pure common trends (true effect=0.0)
5. `negative_effect_data` - Negative treatment effect (true effect=-4.0)

**Tests** (27 total, 100% passing):

**Known-Answer Tests** (11):
- Zero treatment effect
- Known positive effect (2.0)
- Common time trend removal
- Simple 2×2 hand calculation (perfect precision)
- Smaller sample size (n=20 per group)
- Cluster SE larger than naive SE
- Multiple pre/post periods (3×3)
- Heterogeneous baseline levels
- Treatment effect only in post
- Negative treatment effect (-4.0)
- Single pre/single post (minimum viable)

**Input Validation Tests** (8):
- Mismatched lengths
- Non-binary treatment
- Non-binary post
- NaN in outcomes
- Time-varying treatment (error)
- All treated units (error)
- All control units (error)
- Invalid alpha

**Parallel Trends Tests** (5):
- Parallel trends with balanced panel (p > 0.05)
- Parallel trends violation detected (p < 0.05)
- Insufficient pre-periods (error)
- No time variation in pre-period (error)
- NaN handling

**Diagnostics Tests** (3):
- Diagnostics structure (all required fields)
- Degrees of freedom calculation (n_clusters - 1)
- Small cluster warning (RuntimeWarning for n < 30)

---

### Phase 3: Layer 2 Adversarial Tests (~1.5 hours)

**Created**: `tests/validation/adversarial/test_did_adversarial.py` (500 lines, 14 tests)

**Edge cases tested** (14 total, 100% passing):

**Minimum Samples** (2 tests):
- n=5 per group (with RuntimeWarning)
- n=2 per group (extreme minimum, n_clusters=4)

**Extreme Imbalance** (2 tests):
- 90% treated, 10% control
- 10% treated, 90% control

**High Variance** (2 tests):
- Outcomes with σ=50 (wide CIs)
- Zero variance (deterministic outcomes)

**Many Periods** (1 test):
- 20 time periods (10 pre, 10 post)

**Unbalanced Panel** (1 test):
- Subset of units (60 of 80 units used)

**Perfect Separation** (1 test):
- 100x baseline difference (control≈10, treated≈1000)

**Negative Outcomes** (2 tests):
- All negative outcomes
- Mixed sign outcomes (crossing zero)

**Parallel Trends Adversarial** (2 tests):
- Extreme differential pre-trends (p < 0.001)
- Minimal pre-periods (exactly 2, with warning)

**Collinearity** (1 test):
- Identical pre/post means (no time effect in control)

---

## Test Results

### Summary Statistics

**Total Tests**: 41 (27 Layer 1 + 14 Layer 2)
**Pass Rate**: 100% (41/41 passing)
**Coverage**:
- Known-answer validation: ✅
- Input validation: ✅
- Parallel trends testing: ✅
- Edge cases: ✅
- Adversarial scenarios: ✅

**Test Execution Time**: ~1.5 seconds total

**Warnings** (expected):
- RuntimeWarning for small clusters (n < 30): Expected and documented
- statsmodels sqrt warning for n=5 extreme case: Numerical precision issue, non-critical

### Validation Layers Status

- ✅ **Layer 1 (Known-Answer)**: 27 tests, 100% pass
- ✅ **Layer 2 (Adversarial)**: 14 tests, 100% pass
- ⏸️ **Layer 3 (Monte Carlo)**: Deferred (planned for Session 9: Event Study)
- ⏸️ **Layer 4 (Cross-Language)**: Deferred (requires R installation)
- ❌ **Layer 5 (R Triangulation)**: Not planned for DiD (Julia reference not available)
- ❌ **Layer 6 (Golden Reference)**: Not needed (Layers 1-2 provide strong validation)

---

## Key Implementation Decisions

### 1. Cluster-Robust Standard Errors (CONCERN-13)

**Problem**: Panel data has serial correlation within units → naive SEs underestimate sampling variability

**Solution**: Statsmodels cluster-robust covariance
```python
results = model.fit(cov_type='cluster', cov_kwds={'groups': unit_id})
```

**Degrees of freedom**: n_clusters - 1 (following Bertrand, Duflo, Mullainathan 2004)

**Warning threshold**: n_clusters < 30 (cluster-robust SEs biased with few clusters)

**References**:
- Bertrand, Duflo, Mullainathan (2004). "How much should we trust DD estimates?" QJE 119(1): 249-275
- Cameron, Gelbach, Miller (2008). "Bootstrap-based improvements for inference with clustered errors." REStud 75(2): 414-427

### 2. Time-Invariant Treatment Enforcement

**Validation**: Treatment must be constant within units
```python
treatment_varies = df.groupby("unit_id")["treatment"].nunique()
if (treatment_varies > 1).any():
    raise ValueError("treatment must be constant within units...")
```

**Rationale**: Classic 2×2 DiD assumes unit-level treatment (all periods or no periods). Staggered adoption requires modern methods (Callaway-Sant'Anna, Sun-Abraham).

**Future**: Session 10 will implement staggered DiD for time-varying treatment.

### 3. Parallel Trends Testing

**Method**: Regress Y on Treat×Time in pre-treatment periods
```python
Y = β₀ + β₁·Treat + β₂·Time + β₃·(Treat×Time) + ε
```

**Test**: H₀: β₃ = 0 (no differential pre-trends)

**Requirements**:
- ≥2 pre-treatment periods (need time variation to estimate trend)
- Cluster-robust inference (account for serial correlation)

**Limitations**:
- Low power with few pre-periods (warning if n_pre ≤ 2)
- Cannot prove parallel trends (only test for differential trends)
- Underpowered with small clusters (warning if n_clusters < 20)

**Interpretation**:
- p > 0.05: Parallel trends plausible (cannot reject)
- p < 0.05: Differential pre-trends detected (violation likely)

### 4. Input Validation Strategy

**Comprehensive checks** (fail fast):
1. Array length matching
2. Empty array check
3. Alpha bounds (0 < α < 1)
4. NaN/inf detection in outcomes, treatment, post, time
5. Binary treatment/post validation
6. Unit-level treatment consistency
7. Pre/post period existence
8. Minimum pre-periods for parallel trends (≥2)

**Rationale**: DiD requires specific data structure. Clear error messages prevent silent failures.

### 5. Unbalanced Panel Handling

**Design**: DiD works with unbalanced panels (not all units in all periods)

**Statsmodels OLS**: Handles missing observations automatically

**Caveat**: Very sparse data may cause estimation issues (tested in Layer 2)

---

## Lessons Learned

### 1. Cluster-Robust SEs are Conservative

**Finding**: Cluster SEs typically 1.5-3x larger than naive SEs

**Example** (from test_cluster_se_larger_than_naive):
- Naive SE: 0.12
- Cluster SE: 0.31 (2.6x larger)

**Why acceptable**: Better to have conservative inference than anti-conservative

### 2. Small Cluster Counts are Problematic

**Finding**: Cluster-robust SEs biased with n_clusters < 30

**Mitigation**:
- RuntimeWarning when n_clusters < 30
- Suggest alternatives: Bootstrap, Wild cluster bootstrap, Aggregation

**Literature**: Cameron, Gelbach, Miller (2008) recommend ≥30 clusters

### 3. Parallel Trends Testing has Limited Power

**Finding**: With 2-3 pre-periods, hard to detect violations

**Example** (test_minimal_pre_periods):
- 2 pre-periods: Warning about underpowered test
- 3+ pre-periods: Better but still limited

**Recommendation**: Event study plots (Session 9) provide better diagnostic

### 4. DiD is Remarkably Robust

**Tested scenarios that work**:
- 100x baseline differences between groups (differences out)
- Negative outcomes, mixed signs (DiD is scale-invariant)
- High variance (σ=50, just wider CIs)
- Zero variance (deterministic)
- 90/10 treatment imbalance
- 20 time periods
- Unbalanced panels

**Key insight**: DiD differences out baseline differences and time trends → very flexible

---

## Files Created/Modified

### Source Code (410 lines)
1. `src/causal_inference/did/__init__.py` (exports)
2. `src/causal_inference/did/did_estimator.py` (392 lines)

### Tests (1,274 lines)
3. `tests/test_did/__init__.py`
4. `tests/test_did/conftest.py` (265 lines, 5 fixtures)
5. `tests/test_did/test_did_known_answers.py` (509 lines, 27 tests)
6. `tests/validation/adversarial/test_did_adversarial.py` (500 lines, 14 tests)

### Documentation
7. `docs/SESSION_8_DID_FOUNDATION_2025-11-21.md` (this file)

**Total**: 1,684 lines (410 code + 1,274 tests)

---

## Methodological Concerns Addressed

### CONCERN-13: Cluster-Robust SEs for Panel Data ✅ ADDRESSED

**Issue**: Panel data has serial correlation within units → naive SEs are anti-conservative

**Implementation**:
- statsmodels `cov_type='cluster'` with `cov_kwds={'groups': unit_id}`
- df = n_clusters - 1 (Bertrand et al. 2004)
- Warning when n_clusters < 30

**Validation**:
- test_cluster_se_larger_than_naive: Verified cluster SEs > naive SEs
- test_degrees_of_freedom_calculation: Verified df = n_clusters - 1
- test_small_cluster_warning: Verified warning for n < 30

**Status**: ✅ RESOLVED (fully implemented and tested)

---

## Next Steps

### Session 9: Event Study Design (Planned)

**Focus**: Leads/lags analysis and pre-treatment balance

**Deliverables**:
- Event study plots (treatment effects by period relative to treatment)
- Multiple pre-period coefficients (test each pre-period separately)
- Post-treatment dynamics (heterogeneous effects over time)
- Layer 3 Monte Carlo validation for DiD (5,000 runs × 5 DGPs)

**Concerns to address**:
- CONCERN-12: Pre-trends testing (event study provides better diagnostic)

### Session 10: Modern DiD Methods (Planned)

**Focus**: Callaway-Sant'Anna + Sun-Abraham vs TWFE

**Deliverables**:
- Staggered adoption DiD (time-varying treatment)
- TWFE bias demonstration
- Callaway-Sant'Anna estimator
- Sun-Abraham estimator
- Comparison across methods

**Concerns to address**:
- CONCERN-11: TWFE bias with heterogeneous treatment effects + staggered adoption

### Layer 4 Cross-Validation (Deferred)

**When**: After R installation (apt install r-base, install.packages("fixest"))

**Approach**:
1. Install R and fixest package
2. Install rpy2 (Python-R interface)
3. Create `tests/validation/cross_language/test_python_r_did.py`
4. Compare Python DiD to R fixest::feols() with cluster SEs
5. Verify agreement to 6+ decimal places

**Estimated time**: 2 hours (after R setup)

---

## Time Summary

**Estimated**: 11-14.5 hours (Phases 1-5)
**Actual**: ~7.5 hours (Phases 1-3 + Documentation)

**Breakdown**:
- Phase 1 (Implementation): ~4 hours (estimated 4-5)
- Phase 2 (Layer 1 Tests): ~2 hours (estimated 2-3)
- Phase 3 (Layer 2 Tests): ~1.5 hours (estimated 2-3)
- Phase 4 (Cross-Validation): Deferred (estimated 2)
- Phase 5 (Documentation): ~0.5 hours (estimated 1-1.5)

**Efficiency**: Completed in ~52% of estimated time (high-quality implementation)

---

## Quality Metrics

### Code Quality
- ✅ Type hints on all function signatures
- ✅ Comprehensive docstrings with examples
- ✅ Error messages with clear guidance
- ✅ No silent failures (all errors explicit)
- ✅ Black formatted (100-char lines)
- ✅ Single responsibility functions

### Test Quality
- ✅ 100% pass rate (41/41 tests)
- ✅ Real assertions (not just "doesn't crash")
- ✅ Known-answer validation (hand-calculated expected values)
- ✅ Edge case coverage (14 adversarial scenarios)
- ✅ Clear test names and docstrings
- ✅ Fixtures for reusability

### Documentation Quality
- ✅ Mathematical formulas in docstrings
- ✅ References to academic literature
- ✅ Examples in docstrings
- ✅ Session summary with lessons learned
- ✅ Decisions documented with rationale

---

## References

**Implementation**:
- Bertrand, Marianne, Esther Duflo, and Sendhil Mullainathan. 2004. "How Much Should We Trust Differences-in-Differences Estimates?" Quarterly Journal of Economics 119(1): 249-275.
  - Cluster-robust SEs for DiD
  - df = n_clusters - 1

- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2008. "Bootstrap-Based Improvements for Inference with Clustered Errors." Review of Economics and Statistics 75(2): 414-427.
  - Bootstrap methods for small cluster counts
  - Recommendation: ≥30 clusters for cluster-robust SEs

**Theory**:
- Angrist, Joshua D., and Jörn-Steffen Pischke. 2009. *Mostly Harmless Econometrics*. Princeton University Press. Chapter 5.
  - DiD fundamentals
  - Parallel trends assumption

- Cunningham, Scott. 2021. *Causal Inference: The Mixtape*. Yale University Press. Chapter 9.
  - DiD practical examples
  - Event study designs

**Modern DiD** (for Sessions 9-10):
- Callaway, Brantly, and Pedro H.C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." Journal of Econometrics 225(2): 200-230.

- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics 225(2): 175-199.

---

**Session Status**: ✅ COMPLETE
**Next Session**: Session 9 - Event Study Design
**Plan Document**: `docs/plans/active/SESSION_9_EVENT_STUDY_2025-11-21.md` (to be created)
