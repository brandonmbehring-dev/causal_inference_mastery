# Known Bugs

**Last Updated**: 2025-12-24 (Session 108)
**Source**: `repo_review_codex.md` + verification tests

This document tracks known correctness and methodological bugs. Each bug has been verified with automated tests in `tests/validation/audit/test_codex_bugs.py`.

---

## FIXED (Sessions 106-108)

### ✅ BUG-8: SCM Optimization Silent Failure — **FIXED in Session 106**

**File**: `src/causal_inference/scm/weights.py`
**Fix**: Added `result.success` check after fallback optimizer. Now raises `ValueError` with diagnostic message.

### ✅ BUG-5: test_type_i_error.py Has Broken Imports — **FIXED in Session 106**

**File**: `tests/validation/monte_carlo/test_type_i_error.py` + bayesian module
**Fix**: Converted to relative imports in bayesian module. Both `pip install -e .` and direct import work.

### ✅ BUG-6: Stratified ATE Anti-Conservative SE — **FIXED in Session 106**

**File**: `src/causal_inference/rct/estimators_stratified.py`
**Fix**: When n₁=1 or n₀=1 in stratum, uses pooled variance from all strata (conservative estimate).

### ✅ BUG-7: ASCM Jackknife Not Real Jackknife — **FIXED in Session 107**

**File**: `src/causal_inference/scm/augmented_scm.py`
**Fix**: Replaced weight renormalization with `compute_scm_weights()` call in LOO loop. Now properly recomputes weights for each LOO configuration.

### ✅ BUG-1: Fuzzy RDD Kernel is No-Op — **FIXED in Session 108**

**File**: `src/causal_inference/rdd/fuzzy_rdd.py`
**Fix**: Implemented weighted 2SLS with kernel weights. Added `_compute_kernel_weights()` for triangular/rectangular/epanechnikov kernels. Added `_weighted_2sls()` with sandwich variance estimator. Kernel weights now properly applied in both first and second stage.

### ✅ BUG-2: CCT Bandwidth Mislabeled — **FIXED in Session 109**

**File**: `src/causal_inference/rdd/bandwidth.py`
**Fix**: Clarified that `cct_bandwidth()` is an approximation, not true CCT:
- Updated docstring with clear warning that function uses IK bandwidth with 1.5× scaling
- Added UserWarning at runtime recommending `rdrobust` for production use
- Updated API docs in `sharp_rdd.py`, `fuzzy_rdd.py`, and `__init__.py`
- Function kept for backward compatibility but limitation now transparent

---

## MEDIUM Severity (Documented Limitations)

### BUG-3: RKD SE Underestimation

**File**: `src/causal_inference/rkd/sharp_rkd.py`, `fuzzy_rkd.py`

**Issue**:
- Sharp RKD ignores variance of treatment slope (treats as known)
- Fuzzy RKD omits covariance between first-stage and reduced-form kinks

**Impact**: Standard errors are systematically too small.

**Status**: Documented in test comments but not in API docs.

---

### BUG-4: AR Test Wrong With Controls

**File**: `src/causal_inference/iv/diagnostics.py`

**Issue**: Anderson-Rubin test uses projection on Z only, even when controls X are present. Should residualize Z on X.

**Impact**: Invalid test statistic when covariates are included.

---

### BUG-9: Event Study Allows Staggered Misuse

**File**: `src/causal_inference/did/event_study.py`

**Issue**: `event_study()` accepts scalar `treatment_time` but doesn't validate that all treated units share that timing. Using with staggered adoption produces biased estimates.

**Impact**: Silent misuse with staggered adoption.

---

### BUG-10: Paired Variance Allowed With Replacement

**File**: `src/causal_inference/psm/psm_estimator.py`

**Issue**: `variance_method='paired'` is allowed even when `with_replacement=True`, but paired variance formula assumes no replacement.

**Impact**: Invalid variance estimate for matched samples with replacement.

---

## LOW Severity (Documentation/Style)

### DOC-1: Import Path Ambiguity

**Files**: Various

**Issue**: Mixed `src.causal_inference.*` and `causal_inference.*` imports. Editable installs work, normal installs may break.

---

### DOC-2: Stale Docstrings

**File**: `src/causal_inference/psm/psm_estimator.py`

**Issue**: Docstring says balance diagnostics "not yet implemented" but they are fully implemented.

---

## Verification

All HIGH-severity bugs have automated verification tests:

```bash
pytest tests/validation/audit/test_codex_bugs.py -v
```

Expected output: All tests **PASS** (tests prove bugs exist, not that code is correct).

---

## Bug Fix Tracking

| Bug | Priority | Session | Status |
|-----|----------|---------|--------|
| BUG-8 | HIGH | 106 | ✅ FIXED |
| BUG-5 | HIGH | 106 | ✅ FIXED |
| BUG-6 | HIGH | 106 | ✅ FIXED |
| BUG-7 | HIGH | 107 | ✅ FIXED |
| BUG-1 | HIGH | 108 | ✅ FIXED |
| BUG-2 | HIGH | 109 | ✅ FIXED |
| BUG-3 | MEDIUM | 110 | Scheduled |
| BUG-4 | MEDIUM | 110 | Scheduled |
| BUG-9 | MEDIUM | 110 | Scheduled |
| BUG-10 | MEDIUM | 110 | Scheduled |

---

**Last Audit**: Session 83 (2025-12-19)
**Last Fix Session**: 109 (2025-12-24)
