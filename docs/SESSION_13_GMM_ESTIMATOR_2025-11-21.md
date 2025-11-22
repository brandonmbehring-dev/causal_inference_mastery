# Session 13: GMM Estimator (2025-11-21)

**Status**: ✅ **Production Ready**
**Duration**: ~3 hours (plan → implementation → tests → docs)
**Tests**: 116 passing, 1 skipped (117 total, 99.1% pass rate)
**Version**: 0.3.0

---

## Summary

Completed Session 13 by implementing Generalized Method of Moments (GMM) estimator for instrumental variables, providing efficient estimation with many instruments and formal overidentification testing via Hansen J-statistic. All estimators now production-ready with comprehensive test coverage.

---

## Deliverables

### Phase 0: Plan Document Creation (10 minutes)
✅ Created plan document before implementation
✅ Defined scope and success criteria
✅ Committed plan (commit 423622b)

### Phase 1: GMM Estimator Implementation (1.5 hours)
**File Created**:
- `src/causal_inference/iv/gmm.py` (471 lines)

**Features Implemented**:
- One-step GMM with W = (ZX'ZX)^(-1) (equivalent to 2SLS)
- Two-step efficient GMM with optimal weighting matrix
- Hansen J-test for overidentifying restrictions: J ~ χ²(q-p)
- Standard and robust inference
- Input validation and error handling

**Mathematical Details**:
```
GMM Objective: min_θ Q(θ) = g(θ)'Wg(θ)

where:
- θ = [β, γ]' (effects of D and X)
- g(θ) = (1/n)[Z,X]'(Y - D'β - X'γ) (moment conditions)

One-step: W = ([Z,X]'[Z,X])^(-1)
Two-step:
  1. Estimate θ₁ with W₁ = ([Z,X]'[Z,X])^(-1)
  2. Compute Ω = (1/n)[Z,X]'diag(û²)[Z,X] where û = Y - [D,X]'θ₁
  3. Estimate θ₂ with W₂ = Ω^(-1) (optimal)

Hansen J-test:
- J = n * g(θ_GMM)' W g(θ_GMM) ~ χ²(q-p) under H₀
- Reject if J > χ²_{q-p,α} → overidentifying restrictions invalid
```

**Bugs Fixed During Implementation**:
1. **Residual calculation error**: Initially computed residuals as `Y - D @ β` (missing X terms)
   - Fixed by estimating full model: `Y = [D,X]'θ + ε` where θ = [β, γ]'
   - Residuals: `û = Y - [D,X] @ θ`

2. **Parameter dimension mismatch**: GMM originally estimated only β, not [β, γ]
   - Fixed by creating combined regressor matrix: DX = [D, X_with_intercept]
   - Now estimates full parameter vector like 2SLS

### Phase 2: Test Suite (1 hour)
**File Created**:
- `tests/test_iv/test_gmm.py` (233 lines, 18 tests)

**Test Classes**:
1. **TestGMMBasicFunctionality** (4 tests):
   - GMM with strong instruments
   - GMM vs 2SLS agreement (one-step GMM = 2SLS)
   - One-step vs two-step comparison
   - Summary table generation

2. **TestHansenJTest** (3 tests):
   - J-statistic = 0 for just-identified models (q=p)
   - J-test with valid overidentifying restrictions
   - J-test detects invalid instruments

3. **TestGMMInputValidation** (4 tests):
   - Rejects underidentified models (q<p)
   - Summary raises error if not fitted
   - Rejects invalid `steps` parameter
   - Rejects invalid `inference` parameter

4. **TestGMMEdgeCases** (3 tests):
   - Over-identified models (q>p)
   - Just-identified models (q=p)
   - Weak instruments

5. **TestGMMInference** (2 tests):
   - Robust vs standard SEs
   - Confidence intervals

6. **TestGMMSpecialCases** (2 tests):
   - Two-step more efficient than one-step
   - J-test with increasing invalid instruments

**Test Results**: 18/18 passing (100%)

### Phase 3: Documentation (30 minutes)
**Files Updated**:
- `src/causal_inference/iv/__init__.py`: Added GMM export, bumped version to 0.3.0
- `src/causal_inference/iv/README.md`: Added Use Case 9 (GMM with Overidentification Test)
- Created `docs/SESSION_13_GMM_ESTIMATOR_2025-11-21.md`: This document

**README Additions**:
- GMM description in Core Estimators section
- Use Case 9: GMM with Overidentification Test (example code)
- "When to Use GMM" section with decision criteria
- Updated version to 0.3.0

---

## Test Coverage Summary

| Component | Tests | Passing | Coverage |
|-----------|-------|---------|----------|
| TwoStageLeastSquares (Session 11) | 28 | 28 | 100% |
| Three Stages (Session 11) | 18 | 18 | 100% |
| Weak Instrument Diagnostics (Session 11) | 18 | 17 | 94% (1 skipped) |
| LIML (Session 12) | 17 | 17 | 100% |
| Fuller (Session 12) | 18 | 18 | 100% |
| **GMM (Session 13)** | **18** | **18** | **100%** |
| **Total** | **117** | **116** | **99.1%** |

**Skipped Tests** (1):
- `test_anderson_rubin_with_multiple_instruments`: AR test for q>1 (documented limitation from Session 11)

---

## Key Technical Achievements

### 1. GMM Implementation
**Why GMM Matters**:
- 2SLS is efficient with just-identified models (q=p)
- GMM is asymptotically more efficient with many instruments (q>p)
- Two-step GMM accounts for heteroskedasticity automatically
- Provides formal test of overidentifying restrictions

**One-Step vs Two-Step**:
- One-step: W = (ZX'ZX)^(-1) → exactly equivalent to 2SLS
- Two-step: W = Ω^(-1) where Ω = (1/n)ZX'diag(û²)ZX → asymptotically optimal
- Two-step typically has smaller SEs than one-step (efficiency gain)

**Hansen J-Test**:
- Tests H₀: all moment conditions E[Z'ε] = 0 are valid
- J = n * g'Wg ~ χ²(q-p) under H₀
- Just-identified (q=p): J = 0 exactly (no degrees of freedom to test)
- Over-identified (q>p): J > 0, reject if too large

### 2. Full Model Estimation
**Model Structure**:
- Structural equation: Y = D'β + X'γ + ε
- Instruments: Z for D, X for itself
- Parameters: θ = [β, γ]' (effects of endogenous D and exogenous X)

**GMM Estimates Full θ**:
- Unlike simple IV that estimates only β, GMM estimates [β, γ]'
- This matches 2SLS structure (which also estimates both)
- Allows correct residual computation: û = Y - [D,X]'θ

### 3. Optimal Weighting Matrix
**Efficiency**:
- Two-step GMM uses Ω^(-1) as weighting matrix
- Ω estimates the covariance structure of moment conditions
- Accounts for heteroskedasticity automatically (robust)
- Asymptotically efficient among all IV estimators

**Practical Impact**:
- With many instruments, GMM often has smaller SEs than 2SLS
- With few instruments or weak IV, difference is small
- Most beneficial when q >> p (many instruments)

---

## Estimator Comparison Table (Updated)

| Property | 2SLS | LIML | Fuller-1 | GMM (two-step) |
|----------|------|------|----------|----------------|
| **Bias (Strong IV)** | Low | Low | Low | Low |
| **Bias (Weak IV)** | High | Lower | Lower | High |
| **Variance (Strong IV)** | Low | Low | Low | **Lowest** (many IV) |
| **Variance (Weak IV)** | Low | Higher | Moderate | Low |
| **MSE (Weak IV)** | High | Moderate | **Lowest** | High |
| **Recommended when** | F>20, q≈p | F<10 | F 5-20 | **q>>p**, F>20 |
| **Small sample (n<500)** | OK if F>20 | Unstable | **Best** | OK |
| **Large sample (n>1000)** | **Best** | Good | Good | **Best (many IV)** |
| **Overid test** | No | No | No | **Yes (J-test)** |

**Decision Tree** (Updated):
1. **F > 20, q ≈ p**: Use 2SLS (simplest, standard)
2. **F > 20, q >> p**: Use GMM (most efficient, J-test)
3. **10 < F ≤ 20**: Use Fuller-1 (best bias-variance tradeoff)
4. **5 < F ≤ 10**: Use Fuller-1 or LIML + AR CI
5. **F ≤ 5**: Use AR CI only, or find better instruments

---

## Known Limitations

1. **Anderson-Rubin test for q>1** (inherited from Session 11):
   - Just-identified (q=1, p=1) works correctly
   - Over-identified case needs normalization
   - Deferred to future enhancement

2. **Multivariate endogenous variables** (inherited):
   - Current focus on p=1 (single endogenous)
   - Extension to p>1 straightforward but not implemented

3. **GMM Asymptotic Assumptions**:
   - GMM efficiency requires large n (asymptotic theory)
   - Two-step may be less stable than one-step in small samples
   - J-test power depends on sample size

---

## Production Readiness Checklist

✅ **Core Functionality**:
- [x] 2SLS with correct standard errors (Session 11)
- [x] LIML estimator (Session 12)
- [x] Fuller estimator (Session 12)
- [x] **GMM (one-step + two-step) (Session 13)**
- [x] **Hansen J-test (Session 13)**
- [x] Three inference methods (standard, robust)
- [x] Input validation with clear error messages

✅ **Weak Instrument Diagnostics**:
- [x] Stock-Yogo classification
- [x] Cragg-Donald statistic
- [x] Anderson-Rubin CI (q=1)
- [x] Comprehensive diagnostic summary

✅ **Testing**:
- [x] 99.1% test pass rate (116/117)
- [x] Known-answer fixtures
- [x] Edge case validation
- [x] Documented limitations

✅ **Documentation**:
- [x] Comprehensive README with 9 use cases
- [x] Session summaries (11 + 12 + 13)
- [x] API reference
- [x] Estimator selection guidance

✅ **Code Quality**:
- [x] Type hints
- [x] Docstrings with examples
- [x] Error messages with context
- [x] Fail-fast validation

---

## Session Metrics

**Time Breakdown**:
- Phase 0 (plan): 10 minutes
- Phase 1 (GMM implementation): 1.5 hours
- Phase 2 (test suite): 1 hour
- Phase 3 (documentation): 30 minutes
- **Total**: ~3 hours (on target)

**Code Statistics**:
- Source code: 471 lines (gmm.py)
- Test code: 233 lines (test_gmm.py)
- Documentation: Updated README + session summary
- **Total new code**: ~704 lines

**Test Coverage**:
- Tests added: 18 (GMM)
- Tests passing: 116/117 (99.1%)
- Tests skipped: 1 (documented)
- **Total IV tests**: 117

**Token Usage**:
- Session 13: ~30K tokens (GMM + tests + docs)
- Remaining: ~70K tokens (35% of budget)
- Efficiency: Excellent (under estimate, no wasted iterations)

---

## References

### Papers Implemented

- **Hansen, L. P. (1982)**. Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029-1054.
  - **Implemented**: Two-step efficient GMM, optimal weighting matrix

- **Hansen, L. P. (1982)**. Overidentification test. *Econometrica*.
  - **Implemented**: Hansen J-test statistic and inference

### References Consulted

- **Hayashi, F. (2000)**. *Econometrics*, Chapter 3.
  - Reference for GMM formulas and J-test

- **Wooldridge, J. M. (2010)**. *Econometric Analysis of Cross Section and Panel Data*, Chapter 8.
  - Reference for GMM standard errors

---

## Lessons Learned

### 1. Estimate Full Model, Not Just β
**Challenge**: Initial GMM implementation only estimated β (effect of D), not [β, γ]'.

**Solution**: Estimate full parameter vector θ = [β, γ]' like 2SLS does.

**Benefit**: Correct residuals, proper SE computation, consistency with 2SLS.

### 2. One-Step GMM = 2SLS Exactly
**Observation**: With W = (ZX'ZX)^(-1), GMM is algebraically identical to 2SLS.

**Benefit**: Provides verification - one-step GMM coefficients match 2SLS exactly.

### 3. J-Test is Zero for Just-Identified
**Mathematical property**: When q = p, J-statistic is exactly zero (no overidentification to test).

**Implementation**: Check df = q - p, return J=0 if df=0.

**Benefit**: Avoids numerical issues, matches theory.

### 4. Test Organization Reused
**Approach**: Followed Session 11/12 pattern (separate test file, organized by functionality).

**Benefit**: Consistent structure, easy navigation, clear coverage tracking.

---

## Next Steps (Session 14)

### Potential Enhancements

**Priority 1** (Nice to Have):
1. **Anderson-Rubin test for q>1** (~1-2 hours)
   - Implement proper normalization for over-identified case
   - Research additional references
   - ~30 lines modification, 2-3 tests

**Priority 2** (Research/Optional):
2. **Multivariate IV (p>1)** (~2-3 hours)
   - Extend to multiple endogenous variables
   - Test with multivariate fixture
   - ~50 lines modification, 5-6 tests

3. **Monte Carlo validation** (~2-3 hours)
   - Verify finite-sample properties of GMM
   - Compare bias/variance/MSE across estimators
   - CI coverage rates
   - ~300 lines, 5-6 tests (optional)

**Priority 3** (Advanced Features):
4. **Continuous updating GMM** (~2-3 hours)
   - CUE-GMM for better finite-sample properties
   - Iterative reweighting

5. **Bootstrap inference** (~2-3 hours)
   - Bootstrap SEs and CIs
   - Pairs bootstrap for IV

**Deferred**:
- Spatial/Panel IV - future sessions after core complete
- Non-linear GMM - different framework

---

## Conclusion

Session 13 successfully implemented Generalized Method of Moments (GMM) estimator, completing the core IV estimator suite. Combined with Sessions 11-12, the module now offers:

- ✅ **5 estimators**: 2SLS, LIML, Fuller-1/4, GMM (one-step/two-step)
- ✅ **Hansen J-test**: Formal overidentification testing
- ✅ **117 tests**: 99.1% pass rate
- ✅ **Production-ready**: Complete documentation, error handling, validation

**Ready for real-world use** with clearly documented limitations and well-tested estimators optimized for different scenarios (strong IV, weak IV, many instruments).

---

**Session 13 Status**: ✅ **COMPLETE** (2025-11-21)
**Next Session**: Session 14 (optional enhancements)
**Module Version**: 0.3.0
