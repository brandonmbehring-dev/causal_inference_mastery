# Known Limitations

Test xfails, expected limitations, and edge cases in causal_inference_mastery.

**See also**: [METHODOLOGICAL_CONCERNS.md](METHODOLOGICAL_CONCERNS.md) for addressed methodological issues.

---

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| xfail tests | 6 | Expected |
| skip tests | 2 | Deferred |
| Edge cases | 5 | Documented |

---

## xfail Tests (Expected Failures)

### RCT: Single Observation Groups

**Location**: `tests/validation/adversarial/test_simple_ate_adversarial.py`

**Tests** (6 xfails):
- `test_single_treated_unit_*`
- `test_single_control_unit_*`

**Issue**: When n1=1 or n0=1, variance calculation produces NaN due to ddof causing df≤0.

**Why xfail**: This is a fundamental statistical limitation. With a single observation, sample variance is undefined. The estimator correctly computes the point estimate, but SE is meaningless.

**Recommendation**: Use at least n=2 per group for valid inference.

---

### PSM: Limited Overlap

**Location**: `tests/validation/monte_carlo/test_monte_carlo_psm.py`

**Test**: `test_psm_limited_overlap_dgp`

**Issue**: With extreme propensity separation, PSM coverage drops to 31%. CIs are underconservative.

**Why xfail**: PSM fundamentally relies on overlap. When propensity scores are highly separated, matching becomes unreliable. This is expected behavior, not a bug.

**Reference**: Decision documented in Session 7 (PSM Monte Carlo).

---

### RDD: McCrary Density Test

**Location**: `tests/validation/monte_carlo/test_monte_carlo_rdd_diagnostics.py`

**Tests** (2 xfails):
- `test_mccrary_correct_type_1_*`

**Issue**: McCrary implementation has inflated Type I error (CONCERN-22).

**Why xfail**: The McCrary test implementation needs refinement. The test correctly identifies that manipulation is absent, but rejects at higher than nominal rate.

**Tracking**: CONCERN-22 in METHODOLOGICAL_CONCERNS.md

---

## Skipped Tests

### IV: AR Test Over-Identified

**Location**: `tests/test_iv/test_diagnostics.py`

**Test**: `test_ar_over_identified`

**Reason**: AR test for over-identified case (q>1) needs refinement. The formula for combining multiple instrument moments is complex.

**Status**: Deferred to future enhancement.

---

### Cross-Language: Fuller Modification

**Location**: `tests/validation/cross_language/test_python_julia_iv.py`

**Test**: `test_fuller_parity`

**Reason**: Python LIML doesn't support Fuller modification. Julia has Fuller implemented.

**Status**: Python enhancement pending.

---

## Edge Cases (Documented Behavior)

### Perfect Separation in Propensity Model

**Behavior**: Raises `ValueError("Perfect separation detected")`

**Why**: When treatment perfectly predicted by covariates, propensity scores become 0 or 1. IPW weights become infinite. Estimation is impossible.

**Test**: `tests/validation/adversarial/test_ipw_observational_adversarial.py::TestPerfectConfounding`

---

### High-Dimensional Covariates

**Behavior**: May trigger perfect separation or overfitting

**Why**: With p/n > 0.2, logistic regression can overfit. Regularization helps but doesn't eliminate the issue.

**Test**: `tests/validation/adversarial/test_ipw_observational_adversarial.py::TestHighDimensionalCovariates`

**Recommendation**: Keep p/n < 0.2 or use regularized propensity estimation.

---

### Cluster SE with Few Clusters

**Behavior**: Warning issued when clusters < 30

**Why**: Cluster-robust SE requires many clusters for reliable inference. With few clusters, SE is biased downward.

**Tracking**: CONCERN-13 in METHODOLOGICAL_CONCERNS.md

---

### TWFE with Staggered Adoption

**Behavior**: Warning issued when using TWFE with heterogeneous treatment timing

**Why**: TWFE produces biased estimates when treatment effects vary and timing is staggered. Uses Callaway-Sant'Anna or Sun-Abraham instead.

**Tracking**: CONCERN-11 (CRITICAL) in METHODOLOGICAL_CONCERNS.md

---

### Weak Instruments

**Behavior**: Warning when first-stage F < 10 (rule of thumb)

**Why**: Weak instruments cause biased 2SLS estimates toward OLS. LIML/Fuller are more robust but not immune.

**Tracking**: CONCERN-16 (CRITICAL) in METHODOLOGICAL_CONCERNS.md

---

## Cross-Language Skip Conditions

All cross-language tests have `skipif` conditions:
- Skip if `pyjulia` not installed
- Skip if Julia environment not configured

This allows Python tests to run independently of Julia.

---

## Test Counts by Category

| Suite | Total | Pass | xfail | Skip |
|-------|-------|------|-------|------|
| Python (non-MC) | 806 | 800 | 6 | 0 |
| Python Cross-Lang | 79 | 78 | 0 | 1 |
| Julia | 355 | 354 | 0 | 1 |

---

*Last updated: 2025-12-16 (Session 37.5)*
