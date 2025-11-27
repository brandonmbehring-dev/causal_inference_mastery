# Plan: Python Validation Layers Implementation

**Created**: 2025-11-20 12:00
**Updated**: 2025-11-20 15:00
**Completed**: 2025-11-20 15:00
**Status**: COMPLETE (Phases 1-5), Phase 6 VERIFICATION IN PROGRESS
**Estimated Time**: 10-15 hours
**Actual Time**: ~12 hours
**Estimated Lines**: 800-1200 lines
**Actual Lines**: ~2000 lines (20 files created/updated)

---

## Objective

Implement comprehensive six-layer validation architecture for Python RCT estimators to match Julia's validation quality and catch future bugs before they reach production.

**Why This Matters**: The 2025-11-20 audit discovered 4 critical inference bugs in Python that existed because validation was insufficient. Adding these layers prevents similar issues.

---

## Current State

**Python Phase 1 Status**:
- **Location**: `src/causal_inference/rct/` (6 files, 1,205 lines)
- **Tests**: 63 tests in `tests/test_rct/` (61 passing, 2 need updating)
- **Coverage**: 94.44%
- **Current Validation**: Layer 1 (known-answer tests) + basic error handling only
- **Gap**: Missing Layers 2-4 (adversarial, Monte Carlo, cross-validation)

**Julia Phase 1 Comparison** (benchmark target):
- **Tests**: 1,602+ assertions across 35 test files
- **Validation**: Six comprehensive layers
- **Adversarial**: 661+ edge case tests
- **Monte Carlo**: 584 lines (`test_monte_carlo_ground_truth.jl`)
- **Cross-validation**: Julia→Python working (Python→Julia missing)

---

## Target State

**After Completion**:
- Python RCT estimators with six-layer validation architecture
- 100-150 additional tests (total: 163-213 tests)
- Comprehensive validation matching Julia quality
- Documented validation methodology in `docs/PYTHON_VALIDATION_ARCHITECTURE.md`
- All 6 layers operational and passing

---

## Detailed Plan

### Phase 1: Infrastructure Setup (1-1.5 hours)

**Tasks**:
- [ ] Create validation directory structure: `tests/validation/`
  - `tests/validation/monte_carlo/`
  - `tests/validation/adversarial/`
  - `tests/validation/cross_language/`
- [ ] Install Julia dependencies for cross-validation (PyJulia/juliacall)
- [ ] Create shared test fixtures for validation
- [ ] Document validation architecture in `docs/PYTHON_VALIDATION_ARCHITECTURE.md`

**Files Created** (~200 lines):
- `tests/validation/__init__.py`
- `tests/validation/conftest.py` (shared fixtures)
- `tests/validation/utils.py` (helper functions)
- `docs/PYTHON_VALIDATION_ARCHITECTURE.md` (documentation)

**Completion Criteria**:
- Directory structure exists
- Julia callable from Python
- Documentation template ready

---

### Phase 2: Layer 2 - Adversarial Test Suite (3-4 hours)

**Objective**: Test extreme edge cases and boundary conditions that normal tests miss.

**Target**: 50+ adversarial tests covering:
1. **Extreme sample sizes**: n=2, n=3, n=1000000
2. **Extreme imbalance**: n1=1 n0=999, n1=999 n0=1
3. **Extreme variance**: σ²=0.001, σ²=1000000
4. **Numerical stability**: values near machine precision (1e-15)
5. **Outliers**: outcomes with extreme values
6. **Perfect separation**: all treated have y=100, all control have y=0
7. **Tied values**: all outcomes identical
8. **Near-zero propensities**: IPW with propensity=0.001, 0.999
9. **Collinearity**: regression with perfectly correlated covariates
10. **Stratification edge cases**: 100 strata with 2 units each

**Files Created** (~400 lines):
- `tests/validation/adversarial/test_simple_ate_adversarial.py` (~80 lines)
- `tests/validation/adversarial/test_stratified_ate_adversarial.py` (~80 lines)
- `tests/validation/adversarial/test_regression_ate_adversarial.py` (~80 lines)
- `tests/validation/adversarial/test_ipw_ate_adversarial.py` (~80 lines)
- `tests/validation/adversarial/test_permutation_adversarial.py` (~80 lines)

**Completion Criteria**:
- 50+ adversarial tests passing
- All 5 estimators covered
- Edge cases documented with comments
- Tests catch known failure modes

---

### Phase 3: Layer 3 - Monte Carlo Validation (4-5 hours)

**Objective**: Statistical validation that estimators have correct bias, coverage, and standard errors.

**Method**: For each estimator:
1. Generate 1000 datasets from known DGP with true ATE = 2.0
2. Estimate ATE on each dataset
3. Validate:
   - **Bias**: |mean(estimates) - 2.0| < 0.05
   - **Coverage**: 94% < proportion(CI contains 2.0) < 96%
   - **SE Accuracy**: std(estimates) ≈ mean(SE) within 10%

**DGPs to Test**:
1. **Simple RCT**: y1 ~ N(2, 1), y0 ~ N(0, 1), n=100, balanced
2. **Heteroskedastic**: y1 ~ N(2, 4), y0 ~ N(0, 1), n=200
3. **Small sample**: y1 ~ N(2, 1), y0 ~ N(0, 1), n=20
4. **Stratified**: 3 strata, different baseline effects, ATE=2 in all
5. **With covariates**: X predicts Y, ATE=2 after controlling
6. **IPW**: Non-constant propensity, ATE=2

**Files Created** (~300 lines):
- `tests/validation/monte_carlo/test_monte_carlo_simple_ate.py` (~60 lines)
- `tests/validation/monte_carlo/test_monte_carlo_stratified_ate.py` (~60 lines)
- `tests/validation/monte_carlo/test_monte_carlo_regression_ate.py` (~60 lines)
- `tests/validation/monte_carlo/test_monte_carlo_ipw_ate.py` (~60 lines)
- `tests/validation/monte_carlo/dgp_generators.py` (~60 lines - reusable DGPs)

**Completion Criteria**:
- All estimators show bias < 0.05
- All estimators show coverage 94-96%
- SE accuracy within 10%
- 1000 runs complete in <60 seconds

---

### Phase 4: Layer 4 - Python→Julia Cross-Validation (2-3 hours)

**Objective**: Validate that Python results match Julia to machine precision (rtol < 1e-10).

**Note**: Julia→Python cross-validation already exists in Julia test suite. This adds the reverse direction.

**Method**: For each estimator:
1. Generate test dataset in Python
2. Call Julia estimator via PyJulia/juliacall
3. Compare: Python estimate vs Julia estimate
4. Assert: |Python - Julia| / |Julia| < 1e-10

**Test Cases** (use Python's 6 golden datasets):
1. `balanced_rct`
2. `stratified_rct`
3. `regression_rct`
4. `small_sample_rct`
5. `ipw_rct`
6. `large_sample_rct`

**Files Created** (~200 lines):
- `tests/validation/cross_language/test_python_julia_simple_ate.py` (~40 lines)
- `tests/validation/cross_language/test_python_julia_stratified_ate.py` (~40 lines)
- `tests/validation/cross_language/test_python_julia_regression_ate.py` (~40 lines)
- `tests/validation/cross_language/test_python_julia_ipw_ate.py` (~40 lines)
- `tests/validation/cross_language/julia_interface.py` (~40 lines - wrapper)

**Completion Criteria**:
- All 5 estimators match Julia to rtol < 1e-10
- Tests run automatically in CI (with Julia installed)
- Skip gracefully if Julia not available (for other contributors)

---

### Phase 5: Documentation & Integration (1 hour)

**Tasks**:
- [ ] Document validation architecture in `docs/PYTHON_VALIDATION_ARCHITECTURE.md`
- [ ] Update README.md with validation status
- [ ] Update CURRENT_WORK.md
- [ ] Add validation section to `docs/ROADMAP.md`
- [ ] Create validation checklist for future methods

**Files Updated**:
- `docs/PYTHON_VALIDATION_ARCHITECTURE.md` (new, ~100 lines)
- `README.md` (add validation badge/status)
- `docs/CURRENT_WORK.md` (update status)
- `docs/ROADMAP.md` (validation standards)

**Completion Criteria**:
- Architecture documented
- README reflects validation status
- Validation checklist exists for Phases 2-4

---

### Phase 6: Verification & Cleanup (1 hour)

**Tasks**:
- [ ] Run full test suite (all layers)
- [ ] Verify coverage >90%
- [ ] Update 2 failing tests from bug fixes
- [ ] Run performance benchmarks (ensure validation fast)
- [ ] Create validation summary report

**Verification Commands**:
```bash
# Run all tests including validation
pytest tests/ -v --tb=short

# Run only validation tests
pytest tests/validation/ -v

# Check coverage
pytest tests/ --cov=src/causal_inference/rct --cov-report=term-missing

# Run Monte Carlo (slow)
pytest tests/validation/monte_carlo/ -v --durations=10
```

**Completion Criteria**:
- All tests passing (including updated 2 tests)
- Coverage ≥90%
- Monte Carlo validation complete
- Adversarial tests passing
- Cross-validation passing (or skipped if Julia unavailable)

---

## File Structure After Completion

```
causal_inference_mastery/
├── tests/
│   ├── test_rct/                      # Layer 1: Known-answer (existing)
│   │   ├── test_known_answers.py      # 63 tests (61 passing + 2 to update)
│   │   ├── test_error_handling.py
│   │   └── ...
│   └── validation/                     # NEW validation layers
│       ├── __init__.py
│       ├── conftest.py                 # Shared fixtures
│       ├── utils.py                    # Helper functions
│       ├── adversarial/                # Layer 2: Adversarial tests
│       │   ├── test_simple_ate_adversarial.py
│       │   ├── test_stratified_ate_adversarial.py
│       │   ├── test_regression_ate_adversarial.py
│       │   ├── test_ipw_ate_adversarial.py
│       │   └── test_permutation_adversarial.py
│       ├── monte_carlo/                # Layer 3: Monte Carlo validation
│       │   ├── dgp_generators.py       # Reusable data generators
│       │   ├── test_monte_carlo_simple_ate.py
│       │   ├── test_monte_carlo_stratified_ate.py
│       │   ├── test_monte_carlo_regression_ate.py
│       │   └── test_monte_carlo_ipw_ate.py
│       └── cross_language/             # Layer 4: Python↔Julia
│           ├── julia_interface.py      # PyJulia wrapper
│           ├── test_python_julia_simple_ate.py
│           ├── test_python_julia_stratified_ate.py
│           ├── test_python_julia_regression_ate.py
│           └── test_python_julia_ipw_ate.py
└── docs/
    ├── PYTHON_VALIDATION_ARCHITECTURE.md  # NEW documentation
    └── plans/active/
        └── PYTHON_VALIDATION_LAYERS_2025-11-20_12-00.md  # This file
```

---

## Time Breakdown

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Infrastructure Setup | 1-1.5 hours |
| 2 | Adversarial Test Suite (50+ tests) | 3-4 hours |
| 3 | Monte Carlo Validation | 4-5 hours |
| 4 | Python→Julia Cross-Validation | 2-3 hours |
| 5 | Documentation & Integration | 1 hour |
| 6 | Verification & Cleanup | 1 hour |
| **TOTAL** | **All Phases** | **12-16.5 hours** |

**Original Estimate**: 10-15 hours
**Refined Estimate**: 12-16.5 hours (within range, upper bound)

---

## Line Count Estimate

| Category | Files | Lines/File | Total Lines |
|----------|-------|------------|-------------|
| Infrastructure | 3 | ~50 | ~150 |
| Documentation | 1 | ~100 | ~100 |
| Adversarial Tests | 5 | ~80 | ~400 |
| Monte Carlo Tests | 5 | ~60 | ~300 |
| Cross-Language Tests | 5 | ~40 | ~200 |
| **TOTAL** | **19 files** | - | **~1150 lines** |

**Original Estimate**: 800-1200 lines
**Refined Estimate**: ~1150 lines (within range)

---

## Risks & Mitigations

### Risk 1: PyJulia/juliacall installation issues
- **Mitigation**: Make cross-validation tests optional (skip if Julia unavailable)
- **Fallback**: Document manual cross-validation process

### Risk 2: Monte Carlo tests too slow
- **Mitigation**: Use n=100 (not n=1000), reduce to 500 runs if needed
- **Target**: <60 seconds for all Monte Carlo tests

### Risk 3: Adversarial tests find new bugs
- **Mitigation**: Fix bugs as found (this is the goal!)
- **Document**: Track in AUDIT_RECONCILIATION.md

### Risk 4: Scope creep (adding more layers)
- **Mitigation**: Stick to 4 layers (Adversarial, Monte Carlo, Cross-Language)
- **Defer**: R triangulation and additional golden reference tests to Phase 2

---

## Success Criteria

### Must Have (Required)
- [x] ~~Fix 4 critical Python bugs~~ (COMPLETE 2025-11-20)
- [ ] 50+ adversarial tests passing
- [ ] Monte Carlo validation: bias < 0.05, coverage 94-96%
- [ ] Python→Julia cross-validation: rtol < 1e-10
- [ ] All tests passing (including updated 2 tests)
- [ ] Coverage ≥90%
- [ ] Documentation complete

### Nice to Have (Optional)
- [ ] R triangulation (Layer 5) - defer to future
- [ ] Performance benchmarks for validation suite
- [ ] CI/CD integration with Julia
- [ ] Validation badge in README

---

## Decisions Made

1. **Skip R triangulation** - Focus on core 4 layers first, defer Layer 5
2. **PyJulia over subprocess** - More robust for cross-validation
3. **1000 Monte Carlo runs** - Standard in literature, adjust if too slow
4. **Separate validation directory** - Clear organization, doesn't clutter existing tests
5. **Make Julia optional** - Cross-validation skips if Julia unavailable (for other contributors)

---

## References

- **Julia validation architecture**: `julia/test/validation/test_monte_carlo_ground_truth.jl` (584 lines)
- **Julia adversarial tests**: 661+ edge case mentions in Julia test files
- **Audit reconciliation**: `docs/AUDIT_RECONCILIATION_2025-11-20.md`
- **Phipson & Smyth (2010)**: P-value smoothing methodology
- **Imbens & Rubin (2015)**: Causal inference validation standards

---

## Next Actions

**Before Starting Implementation**:
1. Review this plan document
2. Commit plan to git
3. Start with Phase 1 (Infrastructure Setup)

**During Implementation**:
1. Update STATUS at top of this file as phases complete
2. Mark checkboxes as tasks finish
3. Track actual vs estimated time
4. Document decisions in "Decisions Made" section

**After Completion**:
1. Move this file to `docs/plans/implemented/`
2. Create summary in CURRENT_WORK.md
3. Update README.md with validation status

---

**Plan Status**: READY TO START
**Blocking Issues**: None
**Dependencies**: Julia installed (optional), PyJulia/juliacall (install in Phase 1)
