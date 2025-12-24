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

---

## HIGH Severity (Correctness Issues)

### BUG-2: CCT Bandwidth Mislabeled

**File**: `src/causal_inference/rdd/bandwidth.py`

**Issue**: `cct_bandwidth()` claims to implement Calonico-Cattaneo-Titiunik but actually returns:
- Main bandwidth: IK bandwidth (not CCT)
- Bias bandwidth: 1.5 × IK (ad-hoc scaling)

**Evidence**:
```python
h_ik = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)
h_cct_main, h_cct_bias = cct_bandwidth(Y, X, cutoff=0.0)
# h_cct_main == h_ik (exactly)
# h_cct_bias == 1.5 * h_ik (exactly)
```

**Impact**: Users expecting CCT properties (robust bias correction) get IK with arbitrary scaling.

**Remediation Options**:
1. Implement real CCT bandwidth (Cattaneo et al. 2014, 2020)
2. Rename to `cct_approx()` and document approximation
3. Delegate to `rdrobust` or equivalent external library

**Test**: `tests/validation/audit/test_codex_bugs.py::TestBug2CCTBandwidthMislabeled`

---

### BUG-5: test_type_i_error.py Has Broken Imports

**File**: `tests/validation/monte_carlo/test_type_i_error.py`

**Issue**: Uses `causal_inference.*` imports instead of `src.causal_inference.*`, causing ImportError when run.

**Evidence**:
```python
# These imports fail:
from causal_inference.rct.simple_ate import simple_ate
from causal_inference.did.classic_did import classic_did
# etc.
```

**Impact**: Type I error validation tests cannot run. Claims of Type I error control are unverified.

**Remediation**:
1. Fix imports to use `src.causal_inference.*`
2. Or fix package structure so both import styles work

**Test**: `tests/validation/audit/test_codex_bugs.py::TestBug5BrokenTypeIErrorImports`

---

### BUG-6: Stratified ATE Anti-Conservative SE

**File**: `src/causal_inference/rct/estimators_stratified.py`

**Issue**: When n₁=1 or n₀=1 in a stratum, variance is set to 0. Docstring says "conservative" but setting variance to 0 makes SE **smaller** (anti-conservative).

**Evidence**:
```python
# Line 211-212:
var1 = np.var(y1, ddof=1) if n1 > 1 else 0  # SE too small!
```

**Impact**: Confidence intervals are too narrow when any stratum has single-unit groups.

**Remediation Options**:
1. Use pooled variance from other strata
2. Return `np.nan` and document limitation
3. Implement conservative bound (largest observed variance)

**Test**: `tests/validation/audit/test_codex_bugs.py::TestBug6StratifiedATEAntiConservative`

---

### BUG-7: ASCM Jackknife Not Real Jackknife

**File**: `src/causal_inference/scm/augmented_scm.py`

**Issue**: `_jackknife_se()` renormalizes existing weights after dropping a donor instead of recomputing SCM weights. This is not a proper leave-one-out jackknife.

**Evidence**:
```python
# Line ~390-391:
loo_weights = loo_weights / loo_weights.sum()  # Just renormalize!
# Should instead call: compute_scm_weights(...)
```

**Impact**: Jackknife SE underestimates true uncertainty because it doesn't account for how dropping a donor changes optimal weights.

**Remediation**:
1. Recompute weights inside LOO loop (correct but slower)
2. Document limitation and recommend bootstrap SE instead

**Test**: `tests/validation/audit/test_codex_bugs.py::TestBug7ASCMJackknifeNotReal`

---

### BUG-8: SCM Optimization Silent Failure

**File**: `src/causal_inference/scm/weights.py`

**Issue**: `compute_scm_weights()` tries two optimizers (SLSQP, trust-constr) but if both fail, proceeds silently with whatever weights result, without warning.

**Evidence**:
```python
# After first optimizer, checks result.success
# After fallback optimizer, NO success check - just proceeds
```

**Impact**: Users may get invalid SCM weights without any warning.

**Remediation**:
1. Add `result.success` check after fallback optimizer
2. Raise warning or error if both optimizers fail
3. Return success flag in result object

**Test**: `tests/validation/audit/test_codex_bugs.py::TestBug8SCMSilentOptimizationFailure`

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
| BUG-2 | HIGH | 109 | Scheduled |
| BUG-3 | MEDIUM | 110 | Scheduled |
| BUG-4 | MEDIUM | 110 | Scheduled |
| BUG-9 | MEDIUM | 110 | Scheduled |
| BUG-10 | MEDIUM | 110 | Scheduled |

---

**Last Audit**: Session 83 (2025-12-19)
**Last Fix Session**: 108 (2025-12-24)
