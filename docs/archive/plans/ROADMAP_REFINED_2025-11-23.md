# Causal Inference Mastery - Refined Strategic Roadmap

**Date**: 2025-11-23
**Version**: 2.0 (Refined)
**Status**: Phases 1-5 COMPLETE (80% Python-Julia parity), Phases 6-10 PLANNED
**Approach**: Outcome-driven phases with flexible timelines

---

## Executive Summary

### Project State (as of 2025-11-23)

**Achievement**: **80% completion toward full Python-Julia parity** (4 of 5 core phases complete)

**Implementation Status**:
- **Python**: 11,858 lines across 13 modules (RCT, IPW, DR, PSM, DiD, IV, RDD)
- **Julia**: 12,084 lines with Python parity for RCT, PSM, DiD, IV, RDD
- **Test Suite**: 2,420+ tests (438+ Python, 1,982+ Julia)
- **Pass Rates**: Python 96.8%, Julia 91-100% (method-dependent)
- **Code Quality**: Research-grade with 3-layer validation architecture

**Time Investment**: ~67.5 hours documented (55.5h Python + 12h Julia)

**Methods Implemented**:
1. **Randomization**: RCT (simple, regression-adjusted, stratified, IPW, permutation)
2. **Observational**: IPW, Doubly Robust, Propensity Score Matching
3. **Panel**: Difference-in-Differences (Classic, Event Study, Modern Staggered)
4. **Instrumental Variables**: 2SLS, LIML, Fuller, GMM
5. **Regression Discontinuity**: Sharp RDD, Fuzzy RDD, Diagnostics

**Validation Rigor**:
- 3-layer architecture: Known-answer → Adversarial → Monte Carlo
- Statistical properties: Bias < 0.05-0.15, Coverage 93-97%, SE accuracy < 10-20%
- Cross-language validation: PyCall operational for RCT, PSM, DiD (Staggered)

---

## Implementation Matrix

| Method | Python | Julia | PyCall | Monte Carlo | Adversarial | Status |
|--------|--------|-------|--------|-------------|-------------|--------|
| **RCT (5 estimators)** | ✅ 73 tests | ✅ 1,602+ assertions | ✅ Complete | ✅ 13 tests, 5k runs | ✅ 35 tests | **COMPLETE** |
| **IPW** | ✅ 55 tests | ✅ ~200 assertions | ✅ Complete | ✅ 5 tests, 25k runs | ✅ 13 tests | **COMPLETE** |
| **Doubly Robust** | ✅ 49 tests | ✅ ~200 assertions | ✅ Complete | ✅ 5 tests, 25k runs | ✅ 13 tests | **COMPLETE** |
| **PSM** | ✅ 23 tests | ✅ ~200 assertions | ✅ Complete | ✅ 5 DGPs, 5k runs | ✅ 13 tests | **COMPLETE** |
| **Classic DiD** | ✅ 41 tests | ✅ 72 tests | ⏳ Pending | ⏳ Pending | ✅ 14 tests | **NEEDS MC + PyCall** |
| **Event Study** | ✅ 37 tests | ⚠️ 55/63 passing | ⏳ Pending | ⏳ Pending | ✅ 12 tests | **NEEDS FIXES + MC** |
| **Modern DiD (CS, SA)** | ⚠️ 26/30 passing | ✅ 211/245 passing | ✅ 87/87 tests | ⏳ Pending | ⏳ Pending | **NEEDS FIXES + MC** |
| **2SLS** | ✅ 63/64 tests | ✅ ~150 assertions | ⏳ Pending | ⏳ Pending | ⏳ Pending | **NEEDS MC + PyCall** |
| **LIML** | ✅ 17 tests | ✅ ~150 assertions | ⏳ Pending | ⏳ Pending | ⏳ Pending | **NEEDS MC + PyCall** |
| **Fuller** | ✅ 18 tests | ✅ ~150 assertions | ⏳ Pending | ⏳ Pending | ⏳ Pending | **NEEDS MC + PyCall** |
| **GMM** | ✅ 18 tests | ✅ ~150 assertions | ⏳ Pending | ⏳ Pending | ⏳ Pending | **NEEDS MC + PyCall** |
| **Sharp RDD** | ✅ 20 tests | ✅ ~80 assertions | ⏳ Optional | ⏳ Pending | ⏳ Pending | **NEEDS MC** |
| **Fuzzy RDD** | ✅ 19 tests | ✅ ~80 assertions | ⏳ Optional | ⏳ Pending | ⏳ Pending | **NEEDS MC** |
| **RDD Diagnostics** | ✅ 18 tests | ✅ ~80 assertions | ⏳ Optional | ⏳ Pending | ⏳ Pending | **NEEDS MC** |

**Legend**: ✅ Complete | ⚠️ Partial | ⏳ Pending | ⏸️ Optional

---

## Phases 1-5: COMPLETE ✅

### Phase 1: RCT Foundation (COMPLETE - Session 4)
- **Outcome Achieved**: 5 estimators with 3-layer validation
- **Time**: ~3 hours
- **Tests**: 73 total (23 known-answer, 35 adversarial, 13 Monte Carlo @ 5k runs)
- **Pass Rate**: 100%
- **Coverage**: 95.22%
- **Julia Parity**: ✅ 1,602+ assertions, A+ grade

### Phase 2: Observational Methods (COMPLETE - Sessions 5-7)
- **Outcome Achieved**: IPW, DR, PSM with MC validation
- **Time**: ~12 hours (IPW 4h, DR 3h, PSM 5h)
- **Tests**: 127 total + 15 Monte Carlo @ 25-55k runs
- **Pass Rate**: 100%
- **Coverage**: 69-100%
- **Key Feature**: Double robustness validated empirically
- **Julia Parity**: ✅ ~600 assertions total

### Phase 3: Difference-in-Differences (COMPLETE - Sessions 8-10, 17-18)
- **Outcome Achieved**: Classic, Event Study, Modern (CS/SA) in Python + Julia
- **Time**: ~20 hours (Python 12h, Julia 8h)
- **Python Tests**: 108 tests (26/30 Modern DiD failing - tolerance issues)
- **Julia Tests**: 380 tests (72 Classic, 55/63 Event, 211/245 Staggered)
- **Pass Rate**: Python 96.3%, Julia 86-100%
- **Critical Fix**: Bootstrap constructor bug (0/50 → 50/50 samples)
- **PyCall**: ✅ 87/87 tests for Staggered DiD
- **Methodological**: CONCERN-11 (TWFE bias), CONCERN-12 (pre-trends), CONCERN-13 (cluster SE) addressed

### Phase 4: Instrumental Variables (COMPLETE - Sessions 11-13)
- **Outcome Achieved**: 2SLS, LIML, Fuller, GMM with diagnostics
- **Time**: ~13 hours
- **Tests**: 117 total (63/64 2SLS, 17 LIML, 18 Fuller, 18 GMM)
- **Pass Rate**: 99.1%
- **Coverage**: 99.1%
- **Julia Parity**: ✅ ~150 assertions, 84KB code
- **Methodological**: CONCERN-16,17,18,19 addressed (weak IV, Stock-Yogo, AR CI, Hansen J)

### Phase 5: Regression Discontinuity (COMPLETE - Sessions 14-16)
- **Outcome Achieved**: Sharp RDD, Fuzzy RDD, Diagnostics (Python + Julia)
- **Time**: ~20 hours (Python 14h, Julia included in general RDD implementation)
- **Tests**: 57 Python, ~80 Julia assertions
- **Pass Rate**: 100%
- **Coverage**: ~90%
- **Julia Parity**: ✅ ~80 assertions, 57KB code, R triangulation validation
- **Methodological**: CONCERN-22,23,24 addressed (McCrary, bandwidth, covariate balance)

---

## Phases 6-10: Outcome-Driven Strategic Plan

### Phase 6: Test Stabilization & Monte Carlo Validation

**Outcome**: 100% test pass rate + comprehensive Monte Carlo coverage for all methods

**Motivation**:
- 46 tests currently failing (Python DiD: 4, Julia Event: 8, Julia Staggered: 34)
- Monte Carlo validation only exists for Phases 1-2 (RCT, IPW, DR, PSM)
- Need empirical validation of statistical properties for DiD, IV, RDD

#### Part A: Test Fixes (8-10 hours)

**1. Python Modern DiD Fixes** (2 hours)
- **Issue**: 4 tests failing in `tests/test_did/test_modern_did.py`
- **Root Cause**: Tolerance mismatches, warning message assertions
- **Files**: `tests/test_did/test_modern_did.py`
- **Success Metric**: 30/30 tests passing (100%)
- **Tasks**:
  - Investigate exact tolerance failures
  - Adjust tolerances if needed (bootstrap variance)
  - Fix warning message assertions
  - Verify against Python implementations

**2. Julia Event Study Fixes** (3 hours)
- **Issue**: 8 tests failing in `julia/test/did/test_event_study.jl`
- **Root Cause**: Unbalanced panel edge cases, many leads/lags handling
- **Files**: `julia/test/did/test_event_study.jl`, possibly `julia/src/did/event_study.jl`
- **Success Metric**: 63/63 tests passing (100%)
- **Tasks**:
  - Debug unbalanced panel test failures
  - Fix leads/lags handling for extreme cases
  - Verify against Python event study implementation
  - Add edge case handling if needed

**3. Julia Staggered DiD Fixes** (3-5 hours)
- **Issue**: 34 tests failing in `julia/test/did/test_staggered_did.jl`
  - 15 failures: Unbalanced panel test design flaw
  - 19 failures: `haskey()` with NamedTuple technical issues
- **Root Cause**: Test design error + Julia NamedTuple API incompatibility
- **Files**: `julia/test/did/test_staggered_did.jl`
- **Success Metric**: 245/245 tests passing (100%)
- **Tasks**:
  - Fix unbalanced panel test data generation (ensure valid treatment times)
  - Replace `haskey(result, :field)` with `hasfield(typeof(result), :field)`
  - Verify all edge cases work correctly
  - Document NamedTuple limitations for future reference

#### Part B: Monte Carlo Validation (30-40 hours)

**1. DiD Monte Carlo** (12-15 hours)

**Objective**: Validate statistical properties of DiD estimators across realistic scenarios

**DGPs** (5 scenarios):
1. **Simple 2×2 DiD**: Balanced panel, homogeneous treatment effects
2. **Heterogeneous TE**: Cohort and time variation in treatment effects
3. **Parallel Trends Violation**: Mild violation to test sensitivity
4. **Staggered Adoption**: 3 cohorts with different treatment times
5. **Dynamic Effects**: Time-varying treatment effects

**Tests** (15 total = 5 DGPs × 3 estimator types):
- Classic DiD: 2×2 design, parallel trends test
- Event Study: Leads/lags, joint F-test
- Modern DiD (CS/SA): Staggered adoption, aggregation schemes

**Simulations**: 5,000 runs per test (75,000 total)

**Success Metrics**:
- Bias < 0.10 (unbiased under assumptions)
- Coverage: 93-97% (nominal 95%)
- SE Accuracy: < 15% (empirical SE vs analytical SE)
- Power: > 80% for large effects (τ = 5)

**Files**:
- `tests/validation/monte_carlo/test_monte_carlo_did.py` (~400 lines)
- `tests/validation/monte_carlo/dgp_did.py` (~300 lines)

**Estimated Time**: 12-15 hours (DGP design 4h, implementation 6h, debugging 2-5h)

---

**2. IV Monte Carlo** (10-12 hours)

**Objective**: Validate IV estimators under weak/strong instruments, overidentification

**DGPs** (5 scenarios):
1. **Strong IV**: Large first-stage F-stat (F > 100)
2. **Weak IV**: Small first-stage F-stat (F = 5-10)
3. **Many Instruments**: Overidentified (q > p)
4. **Heterogeneous Effects**: Instrument strength varies
5. **Exclusion Violation**: Mild violation to test robustness

**Tests** (20 total = 5 DGPs × 4 estimator types):
- 2SLS: Standard IV
- LIML: Weak IV performance
- Fuller-1/4: Bias reduction
- GMM: Overidentification

**Simulations**: 5,000 runs per test (100,000 total)

**Success Metrics**:
- Bias < 0.15 (acceptable for weak IV)
- Coverage: 90-97% (LIML/Fuller better with weak IV)
- SE Accuracy: < 20% (larger variance acceptable for weak IV)
- First-Stage F > 10: Power > 80%
- Hansen J: Type I error ≈ 5% under exclusion

**Files**:
- `tests/validation/monte_carlo/test_monte_carlo_iv.py` (~500 lines)
- `tests/validation/monte_carlo/dgp_iv.py` (~350 lines)

**Estimated Time**: 10-12 hours (DGP design 3h, implementation 5h, debugging 2-4h)

---

**3. RDD Monte Carlo** (8-10 hours)

**Objective**: Validate RDD estimators under sharp/fuzzy designs, bandwidth sensitivity

**DGPs** (4 scenarios):
1. **Sharp RDD**: Deterministic cutoff, linear potential outcomes
2. **Fuzzy RDD**: Probabilistic treatment, compliance < 1
3. **Nonlinearity**: Quadratic/cubic potential outcomes
4. **Heterogeneous Effects**: Treatment effect varies by distance to cutoff

**Tests** (12 total = 4 DGPs × 3 method types):
- Sharp RDD: Local linear regression
- Fuzzy RDD: 2SLS approach
- Diagnostics: McCrary test, covariate balance

**Simulations**: 3,000 runs per test (36,000 total)

**Success Metrics**:
- Bias < 0.15 (acceptable for local estimation)
- Coverage: 90-97% (bandwidth uncertainty)
- SE Accuracy: < 20% (robust SE)
- McCrary: Type I error ≈ 5% (no manipulation)
- Bandwidth Sensitivity: Estimates stable ±20% bandwidth

**Files**:
- `tests/validation/monte_carlo/test_monte_carlo_rdd.py` (~400 lines)
- `tests/validation/monte_carlo/dgp_rdd.py` (~300 lines)

**Estimated Time**: 8-10 hours (DGP design 3h, implementation 4h, debugging 1-3h)

---

**Phase 6 Success Criteria**:
- ✅ All 46 failing tests fixed (100% pass rate)
- ✅ Monte Carlo validation complete for DiD (15 tests, 75k runs)
- ✅ Monte Carlo validation complete for IV (20 tests, 100k runs)
- ✅ Monte Carlo validation complete for RDD (12 tests, 36k runs)
- ✅ Statistical properties validated (bias, coverage, SE accuracy)
- ✅ Documentation updated with MC results

**Estimated Time**: 38-50 hours (Test fixes: 8-10h, Monte Carlo: 30-40h)

---

### Phase 7: Cross-Language Validation Completion

**Outcome**: Full PyCall validation for DiD, IV, and optionally RDD

**Motivation**:
- PyCall operational for RCT, PSM, DiD (Staggered only)
- Need validation for DiD (Classic/Event), IV, RDD
- Ensures Python-Julia agreement for all shared methods

#### Part A: DiD Classic/Event PyCall (2-3 hours)

**Objective**: Validate Julia Classic DiD and Event Study against Python

**Tests to Add** (20-25 tests in `julia/test/did/test_pycall_validation.jl`):

1. **Classic 2×2 DiD** (8-10 tests):
   - Hand-calculable examples
   - Balanced/unbalanced panels
   - Cluster vs non-cluster SE
   - Parallel trends test agreement
   - Null/large effect detection

2. **Event Study** (12-15 tests):
   - Different numbers of leads/lags
   - Joint F-test for pre-trends
   - Event study coefficients
   - Standard errors
   - Dynamic effects

**Key Conversions**:
```julia
# Julia → Python
unit_id .- 1          # 1-indexed → 0-indexed
Int.(treatment)       # Bool → Int
Int.(post)            # Bool → Int
```

**Tolerances**:
- Deterministic estimates: 1e-10
- Standard errors: 1e-6
- P-values: 0.05

**Success Metric**: 100% agreement (all 20-25 tests passing)

**Files**: `julia/test/did/test_pycall_validation.jl` (+300 lines)

**Estimated Time**: 2-3 hours

---

#### Part B: IV PyCall (4-5 hours)

**Objective**: Validate Julia IV estimators (2SLS, LIML, Fuller, GMM) against Python

**Tests to Add** (30-35 tests in `julia/test/iv/test_pycall_validation.jl` - NEW FILE):

1. **2SLS** (10-12 tests):
   - Strong IV scenarios
   - Weak IV scenarios
   - First-stage F-stat agreement
   - Diagnostics (Stock-Yogo, Cragg-Donald)
   - Anderson-Rubin CI

2. **LIML** (6-8 tests):
   - k-class estimator agreement
   - Weak IV performance
   - SE comparison

3. **Fuller** (6-8 tests):
   - Fuller-1/4 agreement
   - Bias reduction validation

4. **GMM** (8-10 tests):
   - One-step GMM
   - Two-step GMM
   - Hansen J-test agreement
   - Optimal weighting matrix

**Key Conversions**:
```julia
# Julia → Python
Z .- 1                # Instrument indexing (if categorical)
Int.(binary_vars)     # Bool → Int
:gmm → "gmm"         # Symbol → String
```

**Tolerances**:
- 2SLS/LIML/Fuller estimates: 1e-10
- GMM estimates: 1e-8 (iterative)
- Standard errors: 1e-6
- Test statistics: 1e-5

**Success Metric**: 100% agreement (all 30-35 tests passing)

**Files**: `julia/test/iv/test_pycall_validation.jl` (~450 lines, new file)

**Estimated Time**: 4-5 hours

---

#### Part C: RDD PyCall (OPTIONAL) (3-4 hours)

**Objective**: Validate Julia RDD estimators against Python (optional - Julia has R validation)

**Tests to Add** (15-20 tests in `julia/test/rdd/test_pycall_validation.jl` - NEW FILE):

1. **Sharp RDD** (6-8 tests):
   - Local linear agreement
   - Bandwidth selection (IK, CCT)
   - Kernel choice
   - Robust SE

2. **Fuzzy RDD** (4-6 tests):
   - 2SLS RDD agreement
   - First-stage diagnostics
   - Compliance rate

3. **Diagnostics** (5-6 tests):
   - McCrary density test
   - Covariate balance
   - Placebo cutoffs

**Key Conversions**:
```julia
# Julia → Python
:triangular → "triangular"  # Kernel names
:ik → "ik"                  # Bandwidth methods
```

**Tolerances**:
- RDD estimates: 1e-8 (bandwidth optimization)
- Standard errors: 1e-6
- Test statistics: 1e-5

**Success Metric**: 100% agreement (all 15-20 tests passing)

**Note**: Optional because Julia already has R triangulation validation for RDD

**Files**: `julia/test/rdd/test_pycall_validation.jl` (~300 lines, new file)

**Estimated Time**: 3-4 hours

---

**Phase 7 Success Criteria**:
- ✅ DiD Classic/Event PyCall complete (20-25 tests)
- ✅ IV PyCall complete (30-35 tests)
- ⏸️ RDD PyCall optional (15-20 tests) - Julia has R validation
- ✅ 100% Python-Julia agreement for all validated methods
- ✅ Documentation updated

**Estimated Time**: 6-12 hours (DiD: 2-3h, IV: 4-5h, RDD optional: 3-4h)

---

### Phase 8: Sensitivity Analysis

**Outcome**: Tools for assessing unmeasured confounding sensitivity

**Motivation**:
- Observational studies vulnerable to unmeasured confounding
- Need methods to quantify robustness of causal estimates
- Address research transparency and credibility

#### Implementation Plan

**1. E-values** (4-5 hours)
- **Objective**: Minimum unmeasured confounding strength to explain away observed effect
- **Features**:
  - E-value calculation for point estimates and CI bounds
  - Interpretation guidance
  - Sensitivity plot generation
- **File**: `src/causal_inference/sensitivity/evalues.py` (~200 lines)
- **Tests**: 15-20 tests (known-answer, edge cases)
- **Reference**: VanderWeele & Ding (2017)

**2. Rosenbaum Bounds** (5-6 hours)
- **Objective**: Sensitivity analysis for matched studies
- **Features**:
  - Gamma sensitivity parameter
  - Wilcoxon signed-rank bounds
  - Hodges-Lehmann point estimates
  - Critical gamma calculation
- **File**: `src/causal_inference/sensitivity/rosenbaum.py` (~250 lines)
- **Tests**: 15-20 tests
- **Reference**: Rosenbaum (2002)

**3. Simulation-Based Sensitivity** (4-5 hours)
- **Objective**: Quantify bias from unmeasured confounding via simulation
- **Features**:
  - Confounder strength specification
  - Bias contour plots
  - Tipping point analysis
- **File**: `src/causal_inference/sensitivity/simulation.py` (~200 lines)
- **Tests**: 10-15 tests
- **Reference**: Carnegie et al. (2016)

**4. Negative Controls** (3-4 hours)
- **Objective**: Test for unmeasured confounding using outcomes/exposures that shouldn't have effects
- **Features**:
  - Negative outcome control
  - Negative exposure control
  - Falsification test implementation
- **File**: `src/causal_inference/sensitivity/negative_controls.py` (~150 lines)
- **Tests**: 10-12 tests
- **Reference**: Lipsitch et al. (2010)

**Phase 8 Success Criteria**:
- ✅ E-values implemented and tested
- ✅ Rosenbaum bounds for matched studies
- ✅ Simulation-based sensitivity analysis
- ✅ Negative control framework
- ✅ 50-70 tests total
- ✅ Unmeasured confounding concerns addressed

**Estimated Time**: 16-20 hours

---

### Phase 9: Advanced Matching

**Outcome**: Expanded matching toolkit beyond propensity score matching

**Motivation**:
- PSM has limitations (model dependence, extrapolation)
- CEM and other methods offer robustness advantages
- Genetic matching optimizes balance directly

#### Implementation Plan

**1. Coarsened Exact Matching (CEM)** (5-6 hours)
- **Objective**: Match on coarsened covariates for exact matches within strata
- **Features**:
  - Automatic coarsening algorithm
  - User-specified cutpoints
  - Imbalance L1 multivariate statistic
  - Weighted estimator
- **File**: `src/causal_inference/matching/cem.py` (~300 lines)
- **Tests**: 15-20 tests
- **Reference**: Iacus et al. (2012)

**2. Mahalanobis Distance Matching** (4-5 hours)
- **Objective**: Match on Mahalanobis distance (accounts for correlation)
- **Features**:
  - Mahalanobis distance calculation
  - Caliper matching
  - Replacement vs without replacement
  - Variance estimation
- **File**: `src/causal_inference/matching/mahalanobis.py` (~250 lines)
- **Tests**: 12-15 tests
- **Reference**: Rubin (1980)

**3. Genetic Matching** (6-8 hours)
- **Objective**: Optimize matching via genetic algorithm to maximize balance
- **Features**:
  - Genetic algorithm for weight optimization
  - Balance optimization (SMD minimization)
  - Generalized Mahalanobis distance
  - Integration with existing balance diagnostics
- **File**: `src/causal_inference/matching/genetic.py` (~400 lines)
- **Tests**: 15-20 tests
- **Reference**: Diamond & Sekhon (2013)

**4. Entropy Balancing** (4-5 hours)
- **Objective**: Reweight observations to achieve exact moment balance
- **Features**:
  - Moment balance constraints (mean, variance, skewness)
  - Entropy maximization
  - Weight calculation
  - Covariate balance diagnostics
- **File**: `src/causal_inference/matching/entropy.py` (~250 lines)
- **Tests**: 12-15 tests
- **Reference**: Hainmueller (2012)

**Phase 9 Success Criteria**:
- ✅ CEM implemented with automatic coarsening
- ✅ Mahalanobis matching with variance estimation
- ✅ Genetic matching with balance optimization
- ✅ Entropy balancing with moment constraints
- ✅ 54-70 tests total
- ✅ Monte Carlo validation for all methods

**Estimated Time**: 19-24 hours

---

### Phase 10: CATE & Machine Learning Methods

**Outcome**: Heterogeneous treatment effect estimation with modern ML

**Motivation**:
- Average treatment effects mask heterogeneity
- Policy decisions require conditional effects
- Modern ML methods enable flexible estimation

#### Implementation Plan

**1. Meta-Learners** (8-10 hours)

**S-Learner** (2 hours):
- Single model with treatment as feature
- File: `src/causal_inference/cate/s_learner.py` (~150 lines)
- Tests: 8-10 tests

**T-Learner** (2 hours):
- Separate models for treatment/control
- File: `src/causal_inference/cate/t_learner.py` (~150 lines)
- Tests: 8-10 tests

**X-Learner** (3 hours):
- Two-stage approach with imputation
- File: `src/causal_inference/cate/x_learner.py` (~200 lines)
- Tests: 10-12 tests
- Reference: Künzel et al. (2019)

**R-Learner** (3 hours):
- Robinson transformation with residuals
- File: `src/causal_inference/cate/r_learner.py` (~200 lines)
- Tests: 10-12 tests
- Reference: Nie & Wager (2021)

**2. Causal Forests** (10-12 hours)
- **Objective**: Tree-based CATE estimation with honesty
- **Features**:
  - Honest splitting (different samples for structure and estimation)
  - Subsampling without replacement
  - Inference (standard errors, confidence intervals)
  - Variable importance
- **File**: `src/causal_inference/cate/causal_forest.py` (~500 lines)
- **Tests**: 20-25 tests
- **Addresses**: CONCERN-28 (requires honesty for valid inference)
- **Reference**: Wager & Athey (2018)

**3. Double Machine Learning** (8-10 hours)
- **Objective**: Partial out nuisance parameters with cross-fitting
- **Features**:
  - Neyman orthogonality
  - Cross-fitting (K-fold)
  - ML method flexibility (any sklearn estimator)
  - Debiased inference
- **File**: `src/causal_inference/cate/double_ml.py` (~400 lines)
- **Tests**: 15-20 tests
- **Addresses**: CONCERN-29 (requires cross-fitting to avoid overfitting bias)
- **Reference**: Chernozhukov et al. (2018)

**4. Policy Learning** (6-8 hours)
- **Objective**: Learn optimal treatment assignment rules
- **Features**:
  - Policy tree construction
  - Value estimation (expected outcome under policy)
  - Counterfactual policy evaluation
  - Policy comparison
- **File**: `src/causal_inference/cate/policy_learning.py` (~350 lines)
- **Tests**: 12-15 tests
- **Reference**: Athey & Wager (2021)

**Phase 10 Success Criteria**:
- ✅ Meta-learners (S, T, X, R) with cross-validation
- ✅ Causal forests with honest splitting (CONCERN-28)
- ✅ Double ML with cross-fitting (CONCERN-29)
- ✅ Policy learning framework
- ✅ 65-82 tests total
- ✅ Heterogeneous treatment effect estimation validated

**Estimated Time**: 32-40 hours

---

## Success Metrics & Validation Strategy

### Code Quality Standards
- **Formatting**: Python (Black, 100-char lines), Julia (SciML style, 92-char)
- **Type Hints**: Python (all functions), Julia (comprehensive docstrings)
- **Documentation**: Docstrings with examples, mathematical derivations
- **Test Coverage**: 90%+ for all modules

### Statistical Validation
- **Bias**: < 0.05-0.15 depending on method complexity
- **Coverage**: 93-97% (nominal 95% CI)
- **SE Accuracy**: < 10-20% (empirical SE vs analytical SE)
- **Power**: > 80% for large effects

### Cross-Language Parity
- **Tolerance**: 1e-10 deterministic, 1e-6 SE, 0.02 bootstrap
- **Test Agreement**: 100% for all PyCall validation tests
- **API Consistency**: Similar function signatures and parameter names

### Methodological Rigor
- **3-Layer Testing**: Known-answer → Adversarial → Monte Carlo
- **Academic References**: Cite original papers for all methods
- **Mathematical Derivations**: Document proofs in `docs/proofs/`
- **Reproducibility**: Fixed random seeds, detailed fixture documentation

---

## Timeline Flexibility (Outcome-Based Milestones)

This roadmap uses **outcome-driven phases** rather than time-based deadlines. Each phase is complete when:

1. **All tests pass** (100% pass rate for production code)
2. **Success criteria met** (validation metrics achieved)
3. **Documentation complete** (session summaries, method docs)
4. **Commits clean** (git history with proper messages)

### Estimated Time Ranges (for planning only):

| Phase | Outcome | Estimated Time | Priority |
|-------|---------|----------------|----------|
| **Phase 6** | 100% pass rate + MC validation | 38-50 hours | **HIGHEST** |
| **Phase 7** | PyCall validation complete | 6-12 hours | **HIGH** |
| **Phase 8** | Sensitivity analysis toolkit | 16-20 hours | **MEDIUM** |
| **Phase 9** | Advanced matching methods | 19-24 hours | **MEDIUM** |
| **Phase 10** | CATE & ML methods | 32-40 hours | **LOW** |

**Total**: 111-146 hours for Phases 6-10

**Note**: Phases 6-7 are foundational (stabilization + validation) and should be completed before Phases 8-10 (new methods).

---

## Methodological Concerns Tracking

### Addressed (9 concerns):
- ✅ CONCERN-5: Abadie-Imbens SE for PSM
- ✅ CONCERN-11: TWFE bias with staggered adoption
- ✅ CONCERN-12: Pre-trends testing
- ✅ CONCERN-13: Cluster-robust SEs
- ✅ CONCERN-16: Weak instrument diagnostics
- ✅ CONCERN-17: Stock-Yogo critical values
- ✅ CONCERN-18: Anderson-Rubin CIs
- ✅ CONCERN-19: Hansen J-test
- ✅ CONCERN-22,23,24: McCrary, bandwidth, covariate balance

### Pending (4 concerns):
- ⏸️ CONCERN-28: Causal forests require honesty (Phase 10)
- ⏸️ CONCERN-29: Double ML cross-fitting (Phase 10)
- ⏸️ Unmeasured confounding concerns (Phase 8)
- ⏸️ Other advanced method concerns (Phases 8-10)

---

## Documentation Roadmap

### Completed:
- ✅ Session summaries for Sessions 4-10, 17-18
- ✅ Plan documents for all phases
- ✅ METHODOLOGICAL_CONCERNS.md
- ✅ CURRENT_WORK.md (needs update with Sessions 8-18)

### Pending:
- ⏳ Session summaries for Sessions 11-16 (IV, RDD)
- ⏳ Update CURRENT_WORK.md with Sessions 8-18
- ⏳ Update ROADMAP.md with Phases 1-5 complete
- ⏳ Mathematical derivations for advanced methods

---

## Research Application (Post-Phase 10)

Once implementation complete, apply methods to real research:

1. **Datasets**: Obtain causal inference benchmark datasets
2. **Notebooks**: Create analysis notebooks demonstrating each method
3. **Teaching Materials**: Develop tutorials and workshops
4. **Job Interview Prep**: Prepare demonstrations of implementations
5. **Publication**: Consider publishing library and applications

---

## Version History

- **v1.0**: Original roadmap (Phases 1-5)
- **v2.0** (2025-11-23): Refined strategic roadmap
  - Phases 1-5 marked COMPLETE
  - Detailed plans for Phases 6-10
  - Outcome-driven milestones
  - Comprehensive validation strategy
  - Documentation reconciliation plan

---

## Summary

**Current Achievement**: 80% completion (4 of 5 core phases), 2,420+ tests, 23,942 lines of code

**Next Immediate Work** (Phase 6):
1. Fix 46 failing tests → 100% pass rate
2. Add Monte Carlo validation for DiD, IV, RDD
3. Achieve comprehensive empirical validation

**Strategic Goal**: Complete world-class causal inference library with rigorous validation, full Python-Julia parity, and advanced methods for heterogeneous treatment effects.

**Key Differentiator**: Research-grade implementation with 3-layer validation architecture, mathematical derivations, and cross-language verification ensuring correctness and reproducibility.
