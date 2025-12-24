# Gap Analysis: Missing Causal Inference Methods

**Last Updated**: 2025-12-24 (Session 105)
**Previous Audit**: 2025-12-19 (Session 83)
**Status**: ✅ **ALL PRIORITY METHODS COMPLETE** - 23 method families implemented

---

## Session 98: Consolidation & Documentation

Rather than adding new methods (TMLE, DTR, etc.), Session 98 focused on:

1. **Updated METHOD_SELECTION.md** - Extended decision tree from 5 to 12 steps
2. **Updated QUICK_REFERENCE.md** - Full 21-method API reference
3. **Created 4 Tutorial Notebooks** in `docs/examples/`:
   - `example_policy_evaluation.ipynb` - DiD + Event Study + SCM
   - `example_rdd_analysis.ipynb` - Sharp RDD with diagnostics
   - `example_iv_weak_instruments.ipynb` - 2SLS vs LIML vs Fuller
   - `example_new_methods_showcase.ipynb` - Sessions 85-97 methods

**Rationale**: With 21 method families achieving Python-Julia parity and ~52,500 lines of code, the project benefits more from documentation depth than additional method breadth.

---

## Current Implementation Status

### Implemented Methods (21 Families)

| Family | Python | Julia | Estimators | Session |
|--------|--------|-------|------------|---------|
| RCT | ✅ | ✅ | SimpleATE, Stratified, IPW, Regression, Permutation | 1-6 |
| Observational | ✅ | ✅ | IPW, Doubly Robust, Outcome Regression | 5-6 |
| PSM | ✅ | ✅ | Nearest Neighbor, Caliper, Optimal | 7 |
| DiD | ✅ | ✅ | Classic, Event Study, Callaway-Sant'Anna, Sun-Abraham | 8-10 |
| IV | ✅ | ✅ | 2SLS, GMM, LIML, Fuller, CLR, Diagnostics | 11-13, 55-58, 70 |
| RDD | ✅ | ✅ | Sharp, Fuzzy, McCrary, Sensitivity | 14-16 |
| SCM | ✅ | ✅ | Basic, Augmented, Placebo Inference | 46-49 |
| CATE | ✅ | ✅ | S/T/X/R-Learners, DML, Causal Forest | 42-45 |
| Sensitivity | ✅ | ✅ | Rosenbaum Bounds, E-values | 43, 51-53 |
| RKD | ✅ | ✅ | Sharp, Fuzzy, Diagnostics | 72-75 |
| Bunching | ✅ | ✅ | Saez Excess Mass, Counterfactual | 76-81 |
| **Selection** | ✅ | ✅ | Heckman Two-Stage | 85 |
| **Bounds** | ✅ | ✅ | Manski (5 variants), Lee | 86-88 |
| **QTE** | ✅ | ✅ | Unconditional, Conditional, RIF | 88-89 |
| **MTE** | ✅ | ✅ | Local IV, Policy, LATE | 90-91 |
| **Mediation** | ✅ | ✅ | Baron-Kenny, NDE/NIE, CDE, Sensitivity | 92 |
| **Control Function** | ✅ | ✅ | Linear, Nonlinear (Probit/Logit) | 93 |
| **Shift-Share** | ✅ | ✅ | Bartik Instruments, Rotemberg Weights | 94, 97 |
| **TMLE** | ✅ | ✅ | Targeted Maximum Likelihood Estimation | 99-100 |
| **Bayesian** | ✅ | ✅ | Conjugate ATE, Propensity, DR, Hierarchical | 101-104 |

**Bold** = NEW since Session 83 audit

---

## Changes Since Session 83

### Methods Implemented (Sessions 85-95)

All TIER 1 gaps from Session 83 are now closed:

| Original Gap | Status | Session |
|--------------|--------|---------|
| 🔴 Heckman Selection | ✅ Complete | 85 |
| 🔴 Manski Bounds | ✅ Complete | 86-88 |
| 🔴 Lee Bounds | ✅ Complete | 87-88 |
| 🔴 QTE | ✅ Complete | 88-89 |
| 🟡 MTE | ✅ Complete | 90-91 |
| 🟡 Mediation | ✅ Complete | 92 |
| 🟡 Control Function | ✅ Complete | 93 |
| 🟡 Shift-Share | ⚠️ Python only | 94 |
| 🟢 GRF | ✅ Was already complete | 42 |

---

## Remaining Gaps

### TIER 1: High Priority

*All TIER 1 gaps closed as of Session 97.*

---

### TIER 2: Medium Priority

#### 2. TMLE (Targeted Maximum Likelihood Estimation) ✅ COMPLETE

**Priority**: ✅ DONE (Session 99)
**Use case**: Doubly robust estimation with formal statistical guarantees

**What it is**: Fluctuation-based approach to doubly robust estimation with better finite-sample properties.

**Implemented features**:
- ✅ Targeting step (linear fluctuation)
- ✅ Clever covariate construction: H = T/g - (1-T)/(1-g)
- ✅ Influence function-based inference (EIF)
- ✅ Convergence detection
- ✅ Double robustness validation

**Files created**:
- `src/causal_inference/observational/tmle.py` (~370 lines)
- `src/causal_inference/observational/tmle_helpers.py` (~220 lines)
- `tests/observational/test_tmle.py` (25 tests)
- `tests/validation/monte_carlo/test_monte_carlo_tmle.py` (7 MC tests)

**Validation results**:
- Bias: < 0.05 (both models correct)
- Coverage: 91-99%
- SE calibration: within 20%
- Double robustness: verified (one model wrong still works)

**References**:
- van der Laan & Rose (2011). Targeted Learning

---

### TIER 3: Lower Priority (Optional)

#### 3. Principal Stratification

**Use case**: Noncompliance beyond LATE, post-treatment confounding

**Estimated effort**: 3 sessions

---

#### 4. Dynamic Treatment Regimes (DTR)

**Use case**: Sequential treatment decisions over time

**Estimated effort**: 4 sessions

---

#### 5. Bayesian Causal Inference ✅ COMPLETE (Sessions 101-104)

**Priority**: ✅ DONE (Sessions 101-104)
**Use case**: Incorporating prior information, posterior distributions, hierarchical models

**Implemented features**:

*Session 101: Conjugate ATE*
- ✅ Conjugate normal-normal model (closed-form posterior)
- ✅ `bayesian_ate()` estimator (Python + Julia)
- ✅ Prior sensitivity diagnostics (shrinkage metric)
- ✅ Posterior samples for uncertainty propagation
- ✅ Credible intervals with probability interpretation

*Session 102: Bayesian Propensity Scores*
- ✅ Stratified Beta-Binomial (discrete covariates)
- ✅ Logistic Laplace approximation (continuous covariates)
- ✅ `bayesian_propensity()` with automatic method selection

*Session 103: Bayesian Doubly Robust*
- ✅ Combines Bayesian propensity with frequentist outcome models
- ✅ Propagates propensity uncertainty through AIPW formula
- ✅ Variance inflation using influence function

*Session 104: Hierarchical Bayesian ATE (MCMC)*
- ✅ Partial pooling across groups/sites
- ✅ Non-centered parameterization for stable MCMC
- ✅ PyMC (Python) / Turing.jl (Julia) integration
- ✅ MCMC diagnostics: R-hat, ESS, divergences

**Files created**:
- Python: `src/causal_inference/bayesian/` (~1,200 lines, 83+ tests)
- Julia: `julia/src/bayesian/` (~1,100 lines, 163+ tests)

**Dependencies** (optional):
- Python: `pymc>=5.10`, `arviz>=0.18` (for hierarchical)
- Julia: `Turing.jl`, `MCMCDiagnosticTools` (for hierarchical)

---

## Updated Roadmap

### Phase 14: Julia Parity (Sessions 96-97) ✅ COMPLETE

```
└── Session 97: Shift-Share Julia implementation ✅
```

### Phase 15: Advanced Methods (Sessions 98-100)

```
├── Session 98-99: TMLE (Python + Julia)
└── Session 100: Consolidation & Documentation
```

### Phase 16: Optional Extensions (Sessions 101+)

```
├── Principal Stratification (if needed)
├── Dynamic Treatment Regimes (if needed)
└── Bayesian Methods (if needed)
```

---

## Priority Matrix (Updated)

| Method | Interview Freq | Research Use | Difficulty | Priority |
|--------|---------------|--------------|------------|----------|
| ~~Shift-Share (Julia)~~ | ~~N/A~~ | ~~Medium~~ | ~~Low~~ | ✅ DONE |
| ~~TMLE~~ | ~~10%~~ | ~~Medium~~ | ~~Medium~~ | ✅ DONE (Sessions 99-100) |
| ~~Bayesian~~ | ~~5%~~ | ~~Medium~~ | ~~Medium~~ | ✅ DONE (Sessions 101-104) |
| Principal Strat | 5% | Low | High | 🟢 Optional |
| DTR | 5% | Low | High | 🟢 Optional |

---

## Summary

**Since Session 83**:
- Method families: 14 → 23 (+9)
- Lines (Python): ~22K → ~29K
- Lines (Julia): ~23K → ~27K
- Tests: ~7K → ~9K

**Bayesian Module (Sessions 101-104)**:
- Conjugate ATE with prior sensitivity
- Bayesian propensity scores (Beta-Binomial, Logistic Laplace)
- Bayesian Doubly Robust (uncertainty propagation)
- Hierarchical ATE with MCMC (PyMC/Turing.jl)

**Remaining work**:
- 0 HIGH priority (all closed)
- 0 MEDIUM priority (all closed)
- 2 LOW priority (optional: Principal Stratification, DTR)

**Total estimated sessions for gaps**: 0 sessions required (optional methods only)

---

**Generated**: Session 105 Documentation Update
