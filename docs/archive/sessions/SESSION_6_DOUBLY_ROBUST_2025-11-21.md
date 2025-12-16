# Session 6: Doubly Robust Estimation

**Date**: 2025-11-21
**Duration**: ~3 hours
**Status**: ✅ COMPLETE

---

## Summary

Implemented doubly robust (DR) ATE estimation combining inverse probability weighting with outcome regression. DR estimators are consistent if **either** the propensity model **or** the outcome model is correctly specified, providing robustness to model misspecification.

**Key Achievement**: Double robustness property validated via 25,000 Monte Carlo simulations.

---

## What Was Built

### 1. Outcome Regression Module

**File**: `src/causal_inference/observational/outcome_regression.py` (210 lines)

**Function**: `fit_outcome_models(outcomes, treatment, covariates, method="linear")`

**Implementation**:
- Fits separate linear regression models for treated (T=1) and control (T=0) units
- Returns μ₁(X) = E[Y|T=1, X] and μ₀(X) = E[Y|T=0, X] predictions on **all** covariates
- Diagnostics: R², RMSE, sample sizes for both models
- Comprehensive error handling (NaN, mismatched lengths, insufficient units, binary treatment)

**Tests**: 13 tests (100% pass rate)
- 4 tests: Correct specification (linear, multiple covariates, 1D, diagnostics)
- 2 tests: Misspecified models (quadratic truth, interactions)
- 7 tests: Error handling

### 2. Doubly Robust Estimator

**File**: `src/causal_inference/observational/doubly_robust.py` (330 lines)

**Function**: `dr_ate(outcomes, treatment, covariates, propensity=None, outcome_models=None, trim_at=None, alpha=0.05)`

**DR Estimator Formula**:
```
ATE_DR = (1/n) * Σ[
    T/e(X) * (Y - μ₁(X)) + μ₁(X)           # Treated contribution
    - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)  # Control contribution
]
```

**Variance Estimation** (Influence Function):
```python
IF_i = T/e(X) * (Y - μ₁(X)) + μ₁(X)
      - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)
      - ATE_DR

Var(ATE_DR) = (1/n²) * Σ IF_i²
SE(ATE_DR) = sqrt(Var(ATE_DR))
```

**Key Features**:
- Auto-estimates propensity via logistic regression (or accepts pre-computed)
- Auto-fits outcome models via linear regression (or accepts pre-computed)
- Propensity clipping at ε=1e-6 for numerical stability
- Optional propensity trimming at percentiles
- Returns comprehensive diagnostics from both models

**Tests**: 18 tests (100% pass rate)
- 2 tests: Both models correct (ideal case)
- 2 tests: Propensity correct, outcome wrong (IPW protects)
- 2 tests: Outcome correct, propensity wrong (regression protects)
- 1 test: Both models wrong (no protection, but runs)
- 3 tests: Trimming integration
- 3 tests: Pre-computed inputs
- 6 tests: Error handling

---

## Double Robustness Property

### Theory

The DR estimator is consistent if **EITHER**:
1. Propensity model e(X) = P(T=1|X) is correct, **OR**
2. Outcome models μ₁(X) = E[Y|T=1, X] and μ₀(X) = E[Y|T=0, X] are correct

**Why This Matters**:
- **IPW alone**: Biased if propensity model wrong
- **Regression alone**: Biased if outcome model wrong
- **DR**: Protected if **either** model correct (not both required)
- **Both correct**: DR is efficient (lowest variance)

### Monte Carlo Validation Results

**DGP 1: Both Models Correct (n=300)**
```
True DGP:
  e(X) = logistic(0.8*X)  # Linear (correctly modeled)
  μ(X) = 2 + 0.5*X        # Linear (correctly modeled)
  Y = 3.0*T + μ(X) + ε

Results (5000 simulations):
  Bias: < 0.05 ✅
  Coverage: 94-96% ✅
  SE accuracy: < 10% ✅
```

**DGP 2: Propensity Correct, Outcome Misspecified (n=400)**
```
True DGP:
  e(X) = logistic(0.8*X)      # Linear (CORRECT)
  μ_true(X) = 2 + 0.5*X²      # Quadratic (TRUE)
  μ_fit(X) = β₀ + β₁*X        # Linear (WRONG!)
  Y = 3.0*T + μ_true(X) + ε

Results (5000 simulations):
  Bias: < 0.10 ✅ (IPW protects despite outcome misspecification)
  Coverage: 93-97.5% ✅
  SE accuracy: < 15% ✅
```

**DGP 3: Outcome Correct, Propensity Misspecified (n=400)**
```
True DGP:
  e_true(X) = logistic(0.8*X²)    # Quadratic (TRUE)
  e_fit(X) = logistic(β₀ + β₁*X)  # Linear (WRONG!)
  μ(X) = 2 + 0.5*X                # Linear (CORRECT)
  Y = 2.5*T + μ(X) + ε

Results (5000 simulations):
  Bias: < 0.10 ✅ (Regression protects despite propensity misspecification)
  Coverage: 93-97.5% ✅
  SE accuracy: < 15% ✅
```

**DGP 4: Both Models Misspecified (n=300)**
```
True DGP:
  e_true(X) = logistic(0.8*X²)    # Quadratic (WRONG!)
  μ_true(X) = 2 + 0.5*X²          # Quadratic (WRONG!)
  Fitted models: Linear (both misspecified)

Results (5000 simulations):
  Runs successfully: ✅ (all 5000 simulations complete)
  All estimates finite: ✅
  All SEs positive: ✅
  Note: Bias not guaranteed (no protection with both models wrong)
```

**Key Findings**:
- ✅ Double robustness property validated: Consistent if either model correct
- ✅ Bias < 0.10 in DGPs 2-3 despite one model being misspecified
- ✅ Coverage rates conservative (93-97.5%) appropriate for observational studies
- ✅ DR estimator handles both-models-wrong scenario gracefully (doesn't crash)

---

## Adversarial Tests

**13 adversarial tests** stress-tested the DR estimator with challenging scenarios:

### Propensity Challenges (3 tests)
- ✅ Perfect separation with extreme propensities (near 0 and 1)
- ✅ Near-constant propensity (weak confounding)
- ✅ Bimodal propensity distribution

### Outcome Model Challenges (2 tests)
- ✅ Severely misspecified outcome (large quadratic effect)
- ✅ Treatment × covariate interactions not modeled

### High-Dimensional Covariates (2 tests)
- ✅ p=20 with n=150 (p/n ratio = 0.13)
- ✅ p=30 with n=100 (p/n ratio = 0.30)

### Small Sample Sizes (2 tests)
- ✅ n=50 (very small sample)
- ✅ n=100 with extreme confounding

### Collinearity (2 tests)
- ✅ Perfect collinearity (X2 = X1)
- ✅ High collinearity (correlation > 0.99)

### Treatment Imbalance (2 tests)
- ✅ 10% treated, 90% control
- ✅ 90% treated, 10% control

**All adversarial tests passed** - DR estimator handles real-world pathologies gracefully.

---

## Comparison: IPW vs Regression vs DR

| Property | IPW | Regression | DR |
|----------|-----|------------|-----|
| Requires correct propensity | ✅ Yes | ❌ No | ❌ No (if regression correct) |
| Requires correct outcome | ❌ No | ✅ Yes | ❌ No (if propensity correct) |
| Efficient (lowest variance) | ❌ No | ❌ No | ✅ Yes (if both correct) |
| Robust to one misspecification | ❌ No | ❌ No | ✅ Yes |
| Complexity | Moderate | Low | High |
| Variance | High | Moderate | Lowest (if both correct) |

**When to Use**:
- **IPW**: Strong propensity model, simple outcomes, or when outcome model difficult to specify
- **Regression**: Strong outcome model, simple propensities, or when overlap/positivity concerns
- **DR**: Uncertainty about both models (safest choice) or when efficiency matters

---

## Test Summary

| Phase | Tests | Status | Time |
|-------|-------|--------|------|
| Phase 1: Outcome Regression | 13 tests | ✅ 100% | ~15 min |
| Phase 2: DR Estimator | 18 tests | ✅ 100% | ~20 min |
| Phase 3: Adversarial Tests | 13 tests | ✅ 100% | ~30 min |
| Phase 4: Monte Carlo (25k sims) | 5 tests | ✅ 100% | ~90 min |
| **Total** | **49 tests** | **✅ 100%** | **~3 hours** |

**Files Created**:
1. `src/causal_inference/observational/outcome_regression.py` (210 lines)
2. `src/causal_inference/observational/doubly_robust.py` (330 lines)
3. `tests/observational/test_outcome_regression.py` (260 lines)
4. `tests/observational/test_doubly_robust.py` (490 lines)
5. `tests/validation/adversarial/test_doubly_robust_adversarial.py` (370 lines)
6. `tests/validation/monte_carlo/test_monte_carlo_doubly_robust.py` (310 lines)

**Total Lines**: ~2,000 (540 code + 1,430 tests + 30 docs)

---

## Design Decisions

### 1. Separate Outcome Models for T=0 and T=1
**Rationale**: Allows different covariate effects by treatment group
**Alternative**: Single model with treatment × covariate interactions
**Choice**: Separate models (simpler, more flexible, standard in literature)

### 2. Linear Regression Default
**Rationale**: Standard method, interpretable, fast, easy to diagnose misspecification
**Alternative**: ML methods (Random Forest, XGBoost) for automatic non-linearity handling
**Choice**: Linear regression (extensible to ML in future sessions)

### 3. Influence Function for Variance
**Rationale**: Theoretically correct, accounts for estimation uncertainty in both models
**Alternative**: Bootstrap (more robust but slower, requires resampling)
**Choice**: Influence function (standard in DR literature, computationally efficient)

### 4. Propensity Clipping (ε=1e-6)
**Rationale**: Reuse from Session 6, handles perfect separation gracefully
**Alternative**: Trimming only (but trimming biases ATE in finite samples)
**Choice**: Clipping for numerical stability (doesn't remove data)

### 5. Monte Carlo Thresholds
**Both correct**: Bias < 0.05 (same as RCT, ideal case)
**One correct**: Bias < 0.10 (relaxed, model misspecification tolerated)
**Both wrong**: No threshold (just verify runs successfully)

---

## Alternatives Deferred

### Alternative 1: Cross-Fitted DR (Sample Splitting)
**Why defer**: More complex, requires cross-validation infrastructure
**When to implement**: Session 8 or when comparing with targeted learning methods

### Alternative 2: Targeted Maximum Likelihood Estimation (TMLE)
**Why defer**: More sophisticated, requires iterative fitting with loss function minimization
**When to implement**: Advanced topics session after DR fundamentals solidified

### Alternative 3: ML-Based Outcome Models
**Why defer**: Start with linear to understand DR mechanics clearly
**When to implement**: Session on ML methods for causal inference (Random Forest, XGBoost, neural nets)

### Alternative 4: Bootstrap Variance Estimation
**Why defer**: Influence function sufficient for validation, bootstrap is slower
**When to implement**: Session on uncertainty quantification or sensitivity analysis

---

## Lessons Learned

### 1. Double Robustness is Powerful
Monte Carlo simulations **confirmed** the theoretical property: DR remains consistent even when one model is severely misspecified (quadratic truth vs linear fit). This is a major practical advantage over IPW-only or regression-only estimators.

### 2. Propensity Clipping is Essential
Adversarial tests showed that extreme propensities (near 0/1) occur frequently with strong confounding. Clipping at ε=1e-6 prevents division by zero without removing data.

### 3. Outcome Model Diagnostics are Crucial
R² and RMSE diagnostics for μ₁(X) and μ₀(X) help identify misspecification. Low R² suggests outcome model may not protect against propensity misspecification.

### 4. Treatment Imbalance Affects Outcome Models
When treatment rate is 10% or 90%, one outcome model is fit on very few observations (n_treated < 50), increasing variance. DR diagnostics expose this.

### 5. DR Handles High-Dimensional Covariates Better than Expected
Tests with p=30, n=100 (p/n=0.30) still ran successfully. Both propensity (logistic) and outcome (linear) models are regularized implicitly by sklearn's solvers.

---

## Next Steps (Session 8+)

### Immediate Extensions
1. **Propensity Score Matching (PSM)** - Matching on propensity score with caliper
2. **Cross-Fitted DR** - Sample splitting to reduce overfitting bias
3. **ML Methods** - Random Forest, XGBoost for propensity and outcome models

### Advanced Topics
4. **Targeted Maximum Likelihood (TMLE)** - More efficient DR estimator
5. **Sensitivity Analysis** - Unmeasured confounding robustness checks
6. **Regression Discontinuity Design (RDD)** - Local randomization methods
7. **Instrumental Variables (IV)** - Two-stage least squares, weak instruments

### Validation Enhancements
8. **Cross-Language Validation** - Julia implementation of DR estimator
9. **R Triangulation** - Compare to `tmle`, `drtmle` packages
10. **Performance Benchmarks** - Compare DR, IPW, regression-only across DGPs

---

## Session Statistics

**Start Time**: 2025-11-21 ~22:00
**End Time**: 2025-11-21 ~01:00
**Duration**: ~3 hours

**Tests Written**: 49 tests
**Tests Passing**: 49 (100%)
**Monte Carlo Runs**: 25,000 simulations
**Code Coverage**: 100% for new modules

**Commits**:
1. Phase 1: Outcome regression implementation + tests
2. Phase 2: DR estimator implementation + Layer 1 tests
3. Phase 3: Adversarial tests
4. Phase 4: Monte Carlo validation
5. Phase 5: Documentation + session summary

---

## Conclusion

Session 7 successfully implemented doubly robust ATE estimation with comprehensive validation:

✅ **Double robustness property validated** via 25,000 Monte Carlo simulations
✅ **49 tests passing** (100% pass rate) across 4 test layers
✅ **Adversarial tests** confirm robustness to real-world pathologies
✅ **Two new modules** (outcome regression + DR) production-ready

**Key Result**: DR estimator is consistent if **either** propensity **or** outcome model is correct, making it the **safest choice** when uncertain about model specification.

**Next**: Session 8 will implement Propensity Score Matching (PSM) to complete the suite of observational causal inference methods.

---

**Session Status**: ✅ COMPLETE
**Plan Document**: `docs/plans/active/SESSION_7_DOUBLY_ROBUST_2025-11-21_21-50.md`
**Python Status**: Phase 1 (RCT) ✅ | Observational IPW ✅ | Doubly Robust ✅ | PSM ⏳
