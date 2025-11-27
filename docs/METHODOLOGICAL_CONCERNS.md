# Methodological Concerns Tracking

**Created**: 2025-11-21
**Last Updated**: 2025-11-21
**Purpose**: Track all methodological concerns across causal inference implementations

---

## Overview

This document tracks methodological concerns that must be addressed to ensure rigorous causal inference implementations. Each concern is linked to specific phases, implementation status, and test validation.

**Status Legend**:
- ✅ **Addressed**: Implemented and tested
- 🟡 **Partial**: Implementation exists but needs validation
- ⏳ **Planned**: Documented in roadmap, not yet implemented
- ❌ **Unaddressed**: Needs attention

---

## Phase 1: RCT Foundation

### ✅ Addressed
*No major methodological concerns for simple RCT with proper randomization*

**Validation**:
- Session 4 (2025-11-21): 73 tests, 95.22% coverage
- Bias < 0.05, Coverage 93-97%
- All estimators validated via Monte Carlo

---

## Observational Extensions

### ✅ Addressed
*IPW and DR estimators validated with confounded DGPs*

**Sessions 5-6 (2025-11-21)**:
- IPW: 55 tests, propensity clipping (ε=1e-6), trimming
- DR: 49 tests, double robustness validated via 25k sims
- Bias < 0.10, Coverage 93-97.5%

---

## Phase 2: Propensity Score Matching

### CONCERN-5: Bootstrap SE Invalid for Matching With Replacement
**Phase**: Phase 2 (PSM)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: Standard bootstrap SE underestimates variance when matching with replacement (same control unit used multiple times).

**Solution**: Abadie-Imbens (2008) conditional variance estimator

**Implementation**:
- File: `src/causal_inference/psm/matching.py`
- Function: `calculate_abadie_imbens_variance()`
- Must account for: Number of times each control unit matched

**Tests Required**:
- Monte Carlo validation: With replacement variance > without replacement
- Compare Abadie-Imbens SE to naive bootstrap SE
- Coverage rate with Abadie-Imbens SE should be 93-97%

**References**:
- Abadie & Imbens (2006): "Large sample properties of matching estimators"
- Abadie & Imbens (2008): "On the failure of the bootstrap for matching estimators"

**Session**: Session 7 (PSM Monte Carlo)

---

## Phase 3: Difference-in-Differences

### CONCERN-11: TWFE Bias with Staggered Adoption
**Phase**: Phase 3 (DiD)
**Status**: ⏳ PLANNED
**Priority**: CRITICAL

**Issue**: Two-way fixed effects (TWFE) produces biased estimates with:
- Staggered treatment timing (units treated at different times)
- Heterogeneous treatment effects
- Uses already-treated units as "controls" (negative weights)

**Solution**: Modern DiD methods (Callaway-Sant'Anna 2021, Sun-Abraham 2021)

**Implementation**:
- File: `src/causal_inference/did/staggered_did.py`
- Callaway-Sant'Anna estimator (~200 lines)
- Sun-Abraham interaction-weighted estimator (~200 lines)
- Compare to naive TWFE (document bias)

**Tests Required**:
- Monte Carlo DGP with staggered adoption + heterogeneous effects
- Show TWFE bias > 0.50 (severe)
- Show CS/SA bias < 0.10 (corrected)
- Validate with Julia Phase 3 cross-validation

**References**:
- Callaway & Sant'Anna (2021): "Difference-in-Differences with multiple time periods"
- Sun & Abraham (2021): "Estimating dynamic treatment effects in event studies"
- Goodman-Bacon (2021): "Difference-in-differences with variation in treatment timing"

**Session**: Session 10 (Modern DiD)

---

### CONCERN-12: Pre-Trends Testing
**Phase**: Phase 3 (DiD)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: Parallel trends assumption is untestable but can check pre-treatment balance

**Solution**: Event study design with leads (pre-treatment periods)

**Implementation**:
- File: `src/causal_inference/did/event_study.py`
- Test coefficients on lead periods = 0
- Plot event study with confidence bands
- Report joint F-test for all leads

**Tests Required**:
- Monte Carlo with parallel trends violations
- Show event study detects violations
- Validate joint F-test has correct size (5%)

**References**:
- Roth (2022): "Pretest with caution: Event-study estimates after testing for parallel trends"

**Session**: Session 9 (Event Study Design)

---

### CONCERN-13: Cluster-Robust SEs
**Phase**: Phase 3 (DiD)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: DiD regression residuals correlated within units (serial correlation)

**Solution**: Cluster standard errors at unit level

**Implementation**:
- Use statsmodels `cov_type="cluster"` with `cov_kwds={"groups": unit_id}`
- Default clustering in all DiD estimators
- Report both clustered and robust SEs for comparison

**Tests Required**:
- Monte Carlo with serial correlation
- Show naive SE too small (undercoverage < 90%)
- Show cluster SE correct (coverage 93-97%)

**References**:
- Bertrand, Duflo, Mullainathan (2004): "How much should we trust DD estimates?"

**Session**: Session 8 (DiD Foundation)

---

## Phase 4: Instrumental Variables

### CONCERN-16: Weak Instrument Diagnostics (F > 10)
**Phase**: Phase 4 (IV)
**Status**: ⏳ PLANNED
**Priority**: CRITICAL

**Issue**: Weak instruments (F < 10) produce biased 2SLS estimates even in large samples

**Solution**: First-stage F-statistic, Stock-Yogo critical values

**Implementation**:
- File: `src/causal_inference/iv/diagnostics.py`
- Function: `weak_instrument_test()`
- Compute first-stage F-statistic
- Compare to Stock-Yogo critical values (size=10%, 15%, 20%, 25%)
- Raise warning if F < 10

**Tests Required**:
- Monte Carlo with weak instruments (F ∈ [2, 5, 10, 20])
- Show 2SLS bias increases as F decreases
- Validate first-stage F computation

**References**:
- Stock & Yogo (2005): "Testing for weak instruments in linear IV regression"
- Staiger & Stock (1997): "Instrumental variables regression with weak instruments"

**Session**: Session 12 (Weak Instruments)

---

### CONCERN-17: Stock-Yogo Critical Values
**Phase**: Phase 4 (IV)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: F > 10 rule-of-thumb is conservative, Stock-Yogo provides exact critical values

**Solution**: Table lookup based on number of instruments and endogenous regressors

**Implementation**:
- File: `src/causal_inference/iv/diagnostics.py`
- Function: `stock_yogo_critical_values(K1, K2, size)`
- Hard-coded table from Stock & Yogo (2005) Table 1
- Report: F-stat, critical value, max IV bias, max size distortion

**Tests Required**:
- Verify table values match Stock & Yogo (2005)
- Test with 1, 2, 3 instruments
- Test with 1, 2 endogenous regressors

**Session**: Session 12 (Weak Instruments)

---

### CONCERN-18: Anderson-Rubin CIs
**Phase**: Phase 4 (IV)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: Standard 2SLS CIs invalid with weak instruments (undercoverage)

**Solution**: Anderson-Rubin confidence intervals (valid regardless of instrument strength)

**Implementation**:
- File: `src/causal_inference/iv/inference.py`
- Function: `anderson_rubin_ci()`
- Invert Anderson-Rubin test statistic
- Return robust CI (may be infinite with very weak instruments)

**Tests Required**:
- Monte Carlo with weak instruments
- Show standard CIs have coverage < 90%
- Show AR CIs have coverage 93-97% (even with F < 10)

**References**:
- Anderson & Rubin (1949): "Estimation of the parameters of a single equation"
- Chernozhukov & Hansen (2008): "Instrumental variable quantile regression"

**Session**: Session 12 (Weak Instruments)

---

### CONCERN-19: Overidentification Testing
**Phase**: Phase 4 (IV)
**Status**: ⏳ PLANNED
**Priority**: MEDIUM

**Issue**: With multiple instruments, can test (but not prove) exogeneity

**Solution**: Hansen J-test (Sargan test robust to heteroskedasticity)

**Implementation**:
- File: `src/causal_inference/iv/diagnostics.py`
- Function: `hansen_j_test()`
- Test: J ~ χ²(K - L) under exogeneity
- Report: J-statistic, p-value, df

**Tests Required**:
- Monte Carlo with exogenous instruments (p-value uniform)
- Monte Carlo with invalid instruments (p-value < 0.05)

**References**:
- Hansen (1982): "Large sample properties of generalized method of moments estimators"

**Session**: Session 13 (LIML & GMM)

---

## Phase 5: Regression Discontinuity

### CONCERN-22: McCrary Density Test for Manipulation
**Phase**: Phase 5 (RDD)
**Status**: ⏳ PLANNED
**Priority**: CRITICAL

**Issue**: If units manipulate running variable to cross threshold, RDD invalid

**Solution**: McCrary (2008) density test for discontinuity at cutoff

**Implementation**:
- File: `src/causal_inference/rdd/diagnostics.py`
- Function: `mccrary_test()`
- Estimate density on either side of cutoff
- Test for discontinuity in density
- Plot density with confidence bands

**Tests Required**:
- Monte Carlo with no manipulation (p-value uniform)
- Monte Carlo with manipulation (p-value < 0.05)
- Validate against R package `rdd`

**References**:
- McCrary (2008): "Manipulation of the running variable in the RDD"
- Cattaneo, Jansson, Ma (2020): "Simple local polynomial density estimators"

**Session**: Session 15 (RDD Diagnostics)

---

### CONCERN-23: Bandwidth Sensitivity Analysis
**Phase**: Phase 5 (RDD)
**Status**: ⏳ PLANNED
**Priority**: HIGH

**Issue**: RDD estimates sensitive to bandwidth choice

**Solution**: Report estimates for multiple bandwidths, plot sensitivity

**Implementation**:
- File: `src/causal_inference/rdd/sensitivity.py`
- Function: `bandwidth_sensitivity()`
- Compute estimates for h ∈ [0.5×h_opt, 2.0×h_opt]
- Plot estimates vs bandwidth with CIs
- Report IK, CCT bandwidth selectors

**Tests Required**:
- Monte Carlo: Estimates should be stable around h_opt
- Validate bandwidth selectors match rdrobust package

**References**:
- Imbens & Kalyanaraman (2012): "Optimal bandwidth choice for RDD"
- Calonico, Cattaneo, Titiunik (2014): "Robust data-driven inference in RDD"

**Session**: Session 16 (RDD Robustness)

---

### CONCERN-24: Covariate Balance Checks
**Phase**: Phase 5 (RDD)
**Status**: ⏳ PLANNED
**Priority**: MEDIUM

**Issue**: RDD assumes no discontinuity in covariates at cutoff (placebo test)

**Solution**: Run RDD on pre-treatment covariates (should find no effect)

**Implementation**:
- File: `src/causal_inference/rdd/diagnostics.py`
- Function: `covariate_balance_test()`
- Run RDD on each covariate
- Report: estimates, p-values, Bonferroni correction
- Flag if any covariate shows discontinuity

**Tests Required**:
- Monte Carlo with balanced covariates (no discontinuities)
- Monte Carlo with unbalanced covariates (detect discontinuities)

**References**:
- Lee & Lemieux (2010): "Regression discontinuity designs in economics"

**Session**: Session 16 (RDD Robustness)

---

## Phase 8: CATE & Advanced Methods

### CONCERN-28: Causal Forests Require Honesty for Valid Inference
**Phase**: Phase 8 (CATE)
**Status**: ⏳ PLANNED
**Priority**: CRITICAL

**Issue**: Standard random forests overfit → invalid CIs for treatment effects

**Solution**: Honest forests (split sample for tree building vs estimation)

**Implementation**:
- Use `grf` package (R) or `econml.CausalForestDML` (Python)
- Verify honesty parameter enabled
- Report: estimates, standard errors, confidence intervals

**Tests Required**:
- Monte Carlo with heterogeneous effects
- Show honest forests have correct coverage (93-97%)
- Show non-honest forests undercover (<90%)

**References**:
- Wager & Athey (2018): "Estimation and inference of heterogeneous treatment effects using random forests"
- Athey, Tibshirani, Wager (2019): "Generalized random forests"

**Session**: Session 22 (Causal Forests)

---

### CONCERN-29: Double ML Requires Cross-Fitting to Remove Regularization Bias
**Phase**: Phase 8 (CATE)
**Status**: ⏳ PLANNED
**Priority**: CRITICAL

**Issue**: Using same sample for ML model fitting and treatment effect estimation introduces bias

**Solution**: K-fold cross-fitting (Chernozhukov et al. 2018)

**Implementation**:
- File: `src/causal_inference/dml/cross_fit.py`
- Function: `double_ml_cross_fit()`
- 5-fold or 10-fold cross-fitting
- Average estimates across folds
- Variance estimation accounts for cross-fitting

**Tests Required**:
- Monte Carlo: Show cross-fit DML unbiased
- Show non-cross-fit DML has bias > 0.20
- Validate coverage 93-97%

**References**:
- Chernozhukov et al. (2018): "Double/debiased machine learning for treatment and structural parameters"

**Session**: Session 23 (Double Machine Learning)

---

## Summary by Priority

### CRITICAL (Must Address)
- CONCERN-11: TWFE bias with staggered adoption (DiD)
- CONCERN-16: Weak instrument diagnostics (IV)
- CONCERN-22: McCrary density test (RDD)
- CONCERN-28: Causal forests honesty (CATE)
- CONCERN-29: Double ML cross-fitting (CATE)

### HIGH (Important for Rigor)
- CONCERN-5: Bootstrap SE for PSM
- CONCERN-12: Pre-trends testing (DiD)
- CONCERN-13: Cluster-robust SEs (DiD)
- CONCERN-17: Stock-Yogo critical values (IV)
- CONCERN-18: Anderson-Rubin CIs (IV)
- CONCERN-23: Bandwidth sensitivity (RDD)

### MEDIUM (Nice to Have)
- CONCERN-19: Overidentification testing (IV)
- CONCERN-24: Covariate balance checks (RDD)

---

## Validation Status

**Total Concerns**: 13 identified
**Addressed**: 0 (Phases 1-2 completed, no major concerns flagged)
**Planned**: 13 (Phases 3-8)

**Next Session (Session 7)**: Address CONCERN-5 (Abadie-Imbens SE for PSM)

---

**References**: See `docs/ROADMAP.md` for complete methodological references
**Last Reviewed**: 2025-11-21
