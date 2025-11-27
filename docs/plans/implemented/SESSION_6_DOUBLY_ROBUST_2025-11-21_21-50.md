# Session 6: Doubly Robust Estimation

**Created**: 2025-11-21 21:50
**Completed**: 2025-11-21 ~01:00
**Estimated Duration**: 4-5 hours
**Actual Duration**: ~3 hours
**Status**: COMPLETED

---

## Objective

Implement doubly robust (DR) estimation that combines IPW with outcome regression. DR estimators are consistent if **either** the propensity model OR the outcome model is correct (hence "doubly robust").

---

## Current State

### What Exists
- **RCT IPW**: Full implementation with robust variance
- **Observational IPW**: Propensity estimation + IPW (Session 6)
  - `estimate_propensity()`, `trim_propensity()`, `stabilize_weights()`
  - `ipw_ate_observational()`: End-to-end observational IPW
  - 55 tests, 100% pass rate

### What's Missing
- Outcome regression model
- Doubly robust estimator combining both approaches
- Validation with model misspecification scenarios

---

## Target State

### New Module: `src/causal_inference/observational/doubly_robust.py`

**Function**: `dr_ate(outcomes, treatment, covariates, **options)`

**Doubly Robust Estimator**:

```python
# DR estimator formula
ATE_DR = (1/n) * Σ[
    T/e(X) * (Y - μ₁(X)) + μ₁(X)           # Treated contribution
    - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)  # Control contribution
]

Where:
- e(X) = P(T=1|X) (propensity score)
- μ₁(X) = E[Y|T=1, X] (outcome model for treated)
- μ₀(X) = E[Y|T=0, X] (outcome model for control)
```

**Key Property**: Consistent if **either** e(X) OR μ(X) is correct
- If propensity correct but outcome wrong → IPW part protects
- If outcome correct but propensity wrong → Regression part protects
- If both correct → Efficient (lowest variance)

**Expected Lines**: ~200-250

---

## Detailed Plan

### Phase 1: Outcome Regression Module (1.5 hours)

#### 1.1 Create `src/causal_inference/observational/outcome_regression.py`

```python
def fit_outcome_models(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    method: str = "linear"
) -> Dict[str, Any]:
    """
    Fit separate outcome models for treated and control units.

    Returns
    -------
    dict with keys:
        - mu0_model: Fitted model for control (T=0)
        - mu1_model: Fitted model for treated (T=1)
        - mu0_predictions: E[Y|T=0, X] for all X
        - mu1_predictions: E[Y|T=1, X] for all X
        - diagnostics: R², RMSE, convergence
    """
```

**Implementation**:
- Use `sklearn.linear_model.LinearRegression` (default)
- Fit separate models for T=0 and T=1
- Predict on all covariates (not just own treatment group)
- Return both models + predictions + diagnostics

**Tests** (Layer 1):
- Test linear outcome model (true DGP)
- Test quadratic outcome model (misspecified as linear)
- Test with single covariate
- Test with multiple covariates
- Test error handling (NaN, mismatched lengths, no variation)

**Expected**: ~80 lines code, ~120 lines tests

---

### Phase 2: Doubly Robust Estimator (1.5 hours)

#### 2.1 Create `src/causal_inference/observational/doubly_robust.py`

```python
def dr_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    propensity: Optional[np.ndarray] = None,
    outcome_models: Optional[Dict] = None,
    trim_at: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Doubly robust ATE estimator.

    Workflow:
    1. Estimate propensity e(X) (or use provided)
    2. Fit outcome models μ₁(X), μ₀(X) (or use provided)
    3. Clip propensities for numerical stability
    4. Compute DR estimate
    5. Compute robust SE and CI

    Returns
    -------
    dict with keys:
        - estimate: DR ATE estimate
        - se: Robust standard error
        - ci_lower, ci_upper: Confidence interval
        - propensity_diagnostics: AUC, pseudo-R²
        - outcome_diagnostics: R², RMSE for μ₁ and μ₀
        - n_treated, n_control, n_trimmed
    """
```

**Variance Estimation**: Use influence function approach
```python
# Influence function for DR estimator
IF_i = T/e(X) * (Y - μ₁(X)) + μ₁(X)
      - (1-T)/(1-e(X)) * (Y - μ₀(X)) - μ₀(X)
      - ATE_DR

Var(ATE_DR) = (1/n²) * Σ IF_i²
SE(ATE_DR) = sqrt(Var(ATE_DR))
```

**Tests** (Layer 1):
- Test with both models correct (lowest variance)
- Test with propensity correct, outcome wrong (IPW protects)
- Test with outcome correct, propensity wrong (regression protects)
- Test with both wrong (biased, but test runs)
- Test trimming integration
- Test with pre-computed propensity/outcomes
- Test error handling

**Expected**: ~150 lines code, ~200 lines tests

---

### Phase 3: Layer 2 Adversarial Tests (1 hour)

**Edge Cases**:
- Perfect separation in propensity (extreme e(X))
- Severely misspecified outcome model (linear vs quadratic truth)
- High-dimensional covariates (p close to n)
- Small sample sizes (n=50, n=100)
- Collinear covariates affecting both models
- Extreme treatment imbalance (10% treated vs 90% control)

**Expected**: ~150 lines tests (6-8 tests)

---

### Phase 4: Layer 3 Monte Carlo Validation (1 hour)

#### DGPs Testing Double Robustness

**DGP 1: Both Models Correct**
```python
X ~ N(0, 1)
e(X) = logistic(0.8*X)      # Propensity model
μ(X) = 2 + 0.5*X            # Outcome model (linear)
Y = τ*T + μ(X) + ε
```
Expected: Lowest variance, bias < 0.05

**DGP 2: Propensity Correct, Outcome Misspecified**
```python
X ~ N(0, 1)
e(X) = logistic(0.8*X)      # Correct
μ_true(X) = 2 + 0.5*X²      # True (quadratic)
μ_fit(X) = β₀ + β₁*X        # Fitted (linear, wrong!)
Y = τ*T + μ_true(X) + ε
```
Expected: DR still consistent (IPW protects), bias < 0.10

**DGP 3: Outcome Correct, Propensity Misspecified**
```python
X ~ N(0, 1)
e_true(X) = logistic(0.8*X²)   # True (quadratic in X)
e_fit(X) = logistic(β₀ + β₁*X) # Fitted (linear, wrong!)
μ(X) = 2 + 0.5*X               # Correct
Y = τ*T + μ(X) + ε
```
Expected: DR still consistent (regression protects), bias < 0.10

**DGP 4: Both Models Misspecified**
```python
e_true(X) = logistic(0.8*X²)
e_fit(X) = logistic(β₀ + β₁*X)
μ_true(X) = 2 + 0.5*X²
μ_fit(X) = β₀ + β₁*X
```
Expected: Biased (but should run without error)

**Tests** (5000 runs each):
- Validate bias/coverage/SE for DGPs 1-3
- DGP 4 runs successfully (no crash test)
- Compare DR vs IPW-only vs Regression-only

**Expected**: ~250 lines tests (4 DGP classes + comparison)

---

### Phase 5: Documentation (30 min)

#### Session Summary Should Include
- DR estimator theory and double robustness property
- Monte Carlo results comparing DR vs IPW vs Regression
- When to use DR vs IPW alone
- Misspecification scenarios and robustness
- Variance comparison across estimators

---

## Success Criteria

✅ **Both Models Correct** (DGP 1):
- Bias < 0.05
- Coverage 94-96%
- SE accuracy < 10%
- Variance lower than IPW-only

✅ **One Model Correct** (DGPs 2-3):
- Bias < 0.10 (DR still consistent despite misspecification)
- Coverage 93-97%
- SE accuracy < 12%

✅ **Both Models Wrong** (DGP 4):
- Runs without error (may be biased)
- Returns finite estimates

✅ **Tests**:
- Layer 1: ≥12 functional tests
- Layer 2: ≥6 adversarial tests
- Layer 3: ≥4 Monte Carlo tests (20k runs total)

---

## Files to Create

**Source Code**:
1. `src/causal_inference/observational/outcome_regression.py` (~80 lines)
2. `src/causal_inference/observational/doubly_robust.py` (~150 lines)

**Tests**:
3. `tests/observational/test_outcome_regression.py` (~120 lines)
4. `tests/observational/test_doubly_robust.py` (~200 lines)
5. `tests/validation/adversarial/test_doubly_robust_adversarial.py` (~150 lines)
6. `tests/validation/monte_carlo/test_monte_carlo_doubly_robust.py` (~300 lines)

**Documentation**:
7. `docs/SESSION_7_DOUBLY_ROBUST_2025-11-21.md`

**Total Expected**: ~1,200 lines (230 code + 970 tests/docs)

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1 | 1.5 hr | Outcome regression + tests |
| 2 | 1.5 hr | DR estimator + Layer 1 tests |
| 3 | 1.0 hr | Layer 2 adversarial tests |
| 4 | 1.0 hr | Layer 3 Monte Carlo tests |
| 5 | 0.5 hr | Documentation |
| **Total** | **5.5 hr** | Complete doubly robust estimation |

---

## Decisions Made

### Design Decisions

1. **Separate Outcome Models for T=0 and T=1**
   - Rationale: Allows different covariate effects by treatment group
   - Alternative: Single model with treatment × covariate interactions
   - Choice: Separate models (simpler, more flexible)

2. **Linear Regression Default**
   - Rationale: Standard method, interpretable, fast
   - Alternative: ML methods (Random Forest, XGBoost)
   - Choice: Start with linear, can extend later

3. **Influence Function for Variance**
   - Rationale: Theoretically correct, accounts for estimation uncertainty
   - Alternative: Bootstrap (slower, more complex)
   - Choice: Influence function (standard in literature)

4. **Propensity Clipping (ε=1e-6)**
   - Rationale: Reuse from Session 6, handles perfect separation
   - Consistent with observational IPW behavior

5. **Monte Carlo Thresholds**
   - Both correct: Bias < 0.05 (same as RCT)
   - One correct: Bias < 0.10 (relaxed, model misspecification)
   - Both wrong: No threshold (just runs)

### Alternative Approaches Considered

**Alternative 1**: Cross-fitted DR (sample splitting)
- **Deferred**: More complex, requires cross-validation infrastructure
- Standard DR sufficient for Session 7

**Alternative 2**: Targeted Maximum Likelihood (TMLE)
- **Deferred**: More sophisticated, requires iterative fitting
- DR is simpler and demonstrates core concept

**Alternative 3**: ML-based outcome models
- **Deferred**: Start with linear, extend in future sessions

---

## Risks and Mitigation

### Risk 1: Outcome Model Overfitting
**Impact**: Poor predictions, increased variance
**Mitigation**:
- Diagnostics (R², RMSE on held-out data)
- Document when to use regularization
- Test with high-dimensional covariates

### Risk 2: Variance Estimation Complexity
**Impact**: Incorrect SE, poor coverage
**Mitigation**:
- Use established influence function formulas
- Monte Carlo validation will catch issues
- Compare to bootstrap (sanity check)

### Risk 3: Both Models Misspecified
**Impact**: DR not consistent, biased estimates
**Mitigation**:
- Document limitations clearly
- Monte Carlo DGP 4 demonstrates failure mode
- Emphasize need for at least one correct model

---

## Theory: Double Robustness Property

### Why DR Works

DR estimator has form:
```
ATE_DR = IPW_component + Regression_correction

IPW_component = (1/n) Σ [T*Y/e(X) - (1-T)*Y/(1-e(X))]
Regression_correction = (1/n) Σ [μ₁(X) - μ₀(X) - T*(μ₁(X) - μ(X))/e(X) + (1-T)*(μ₀(X) - μ(X))/(1-e(X))]
```

**Case 1: e(X) correct, μ(X) wrong**
- IPW component is consistent
- Regression correction → 0 in expectation
- Result: DR consistent

**Case 2: μ(X) correct, e(X) wrong**
- Regression correction is consistent
- IPW component noise cancels with correction
- Result: DR consistent

**Case 3: Both correct**
- Both components contribute
- Result: DR consistent AND efficient (lowest variance)

**Case 4: Both wrong**
- No protection
- Result: DR biased

---

## Comparison: IPW vs Regression vs DR

| Property | IPW | Regression | DR |
|----------|-----|------------|-----|
| Requires correct propensity | Yes | No | No (if regression correct) |
| Requires correct outcome | No | Yes | No (if propensity correct) |
| Efficient (lowest variance) | No | No | Yes (if both correct) |
| Robust to one misspecification | No | No | Yes |
| Complexity | Moderate | Low | High |

**When to use**:
- **IPW**: Strong propensity model, simple outcomes
- **Regression**: Strong outcome model, simple propensities
- **DR**: Uncertainty about both models (safest choice)

---

## Execution Summary

**All Phases Completed**:
- ✅ Phase 1: Outcome regression (13 tests, 100% pass)
- ✅ Phase 2: DR estimator (18 tests, 100% pass)
- ✅ Phase 3: Adversarial tests (13 tests, 100% pass)
- ✅ Phase 4: Monte Carlo validation (5 tests, 25k sims, 100% pass)
- ✅ Phase 5: Documentation (session summary, README update)

**Results**:
- 49 tests total, 100% pass rate
- Double robustness property validated via Monte Carlo
- Bias < 0.05 (both correct), < 0.10 (one correct)
- Coverage 93-97.5% across all DGPs
- Session completed in ~3 hours (under 4-5 hour estimate)

**Plan Status**: ✅ COMPLETED
**Session Summary**: `docs/SESSION_7_DOUBLY_ROBUST_2025-11-21.md`
