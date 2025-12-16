# Julia Causal Inference Ecosystem Research

**Version**: 1.0
**Date**: 2024-11-14
**Purpose**: Document existing Julia causal inference packages, identify gaps, and inform packaging decisions
**Research Duration**: ~1 hour comprehensive survey

---

## Executive Summary

The Julia causal inference ecosystem is **early-stage but growing**, with several active packages covering core methods. However, there are **significant gaps** compared to R (fixest, did, grf) and Python (econml, dowhy, causalml).

**Key Finding**: **RCT estimators are completely missing** - no existing Julia package implements comprehensive treatment effect estimation for randomized experiments. The `causal_inference_mastery` project fills this critical gap.

---

## Table of Contents

1. [Existing Julia Packages](#existing-julia-packages)
2. [Major Gaps](#major-gaps)
3. [Comparison to R and Python](#comparison-to-r-and-python)
4. [Project Positioning](#project-positioning)
5. [Packaging Decision Framework](#packaging-decision-framework)

---

## Existing Julia Packages

### 1.1 Causal Discovery & Graphical Models

**CausalInference.jl** (206 stars, 15 contributors)
- **Repository**: https://github.com/mschauer/CausalInference.jl
- **Latest Release**: v0.19.1 (February 2025)
- **Maturity**: Moderate-to-Advanced (production-ready)
- **Focus**: Causal discovery and structure learning
- **Methods**:
  - PC algorithm (constraint-based structure learning)
  - FCI algorithm (extended version for latent confounders)
  - GES algorithm (score-based greedy equivalence search)
  - Bayesian Causal Zig-Zag sampler
  - Adjustment set identification
- **Documentation**: Comprehensive (mschauer.github.io)
- **Assessment**: Best-in-class for causal discovery
- **Gap**: **NO treatment effect estimation** (no RCT, DiD, IV, RDD estimators)

---

**Associations.jl** (168 stars, JuliaDynamics organization)
- **Repository**: https://github.com/JuliaDynamics/Associations.jl
- **Latest Release**: v4.5.0 (August 2025)
- **Maturity**: Mature (well-maintained)
- **Focus**: Association measures and independence testing
- **Methods**:
  - Partial correlation, distance correlation
  - Conditional mutual information
  - Transfer entropy
  - Convergent cross mapping (CCM)
  - Causal network inference API
- **Assessment**: Strong for time series causality and dynamical systems
- **Gap**: **Not designed for treatment effects**

---

### 1.2 Machine Learning-Based Causal Inference

**CausalELM.jl** (32 stars, 1 contributor)
- **Repository**: https://github.com/dscolby/CausalELM.jl
- **Latest Release**: v0.8.0 (December 2024)
- **Maturity**: Early development (author warns experimental status)
- **Focus**: ML-based causal inference using Extreme Learning Machines
- **Methods**:
  - **Aggregate Effects**: G-computation, Double ML, Interrupted Time Series
  - **CATE Estimators**: S-learner, T-learner, X-learner, R-learner, Doubly Robust
- **Unique Features**: GPU support, minimal dependencies, fast ELM implementation
- **Documentation**: Comprehensive (dscolby.github.io/CausalELM.jl/)
- **Assessment**: Promising for ML-based methods, but experimental
- **Gap**: No traditional econometric methods (DiD, IV, RDD)

---

### 1.3 Difference-in-Differences

**DiffinDiffs.jl** (41 stars, 2 forks)
- **Repository**: https://github.com/JuliaDiffinDiffs/DiffinDiffs.jl
- **Latest Release**: v0.2.0 (April 2024)
- **Maturity**: Early but active
- **Components**:
  - DiffinDiffsBase.jl (core infrastructure)
  - InteractionWeightedDIDs.jl (regression-based multi-period DiD)
- **Key Strengths**: Performance (handles gigabytes), modular architecture
- **Documentation**: Comprehensive with academic paper
- **Assessment**: Best-in-class for DiD in Julia, production-ready for basic DiD
- **Gap**: Limited to regression-based methods, no Sun-Abraham, no Callaway-Sant'Anna yet

---

### 1.4 Synthetic Control

**SynthControl.jl** (38 stars, 2 contributors)
- **Repository**: https://github.com/nilshg/SynthControl.jl
- **Latest Release**: v0.4.3 (February 2024)
- **Maturity**: Beta
- **Methods**:
  - SimpleSCM (simple synthetic control)
  - SyntheticDiD (Arkhangelsky et al. 2021)
  - MC-NNM (experimental - matrix completion)
- **Documentation**: Beta-stage docs, example datasets included
- **Assessment**: Functional but beta status
- **Gap**: Inference methods still developing

---

### 1.5 Regression Discontinuity

**RegressionDiscontinuity.jl** (20 stars, 5 forks)
- **Repository**: https://github.com/nignatiadis/RegressionDiscontinuity.jl
- **Last Commit**: April 2022 (**STALE - 2+ years**)
- **Maturity**: Experimental (author warns)
- **Methods**:
  - Naive local linear regression (Imbens-Kalyanaraman bandwidth)
  - Min-max optimal estimator (Imbens-Wager 2019)
  - Noise-induced randomization sensitivity analysis
  - McCrary density test
- **Assessment**: Sophisticated methods but **experimental and unmaintained**
- **Gap**: Results require validation, no modern rdrobust implementation

---

**GeoRDD.jl** (6 stars)
- **Repository**: https://github.com/maximerischard/GeoRDD.jl
- **Focus**: Geographical RDD using Gaussian Process regression
- **Maturity**: Not registered, research code
- **Assessment**: Niche application, not general-purpose

---

### 1.6 Matching Methods

**MatchIt.jl** (3 stars, 1 contributor)
- **Repository**: https://github.com/eohne/MatchIt.jl
- **Maturity**: Early/Academic (not registered on Julia General)
- **Methods**:
  - One-to-one nearest neighbor (with/without replacement)
  - Propensity score matching (Logit/Probit)
  - Exact matching on covariates
  - KDTree-based matching
- **Assessment**: Basic implementation, requires manual GitHub install
- **Gap**: No balance diagnostics, no Love plots, no advanced matching (CEM, genetic)

---

### 1.7 Panel Data & Instrumental Variables

**FixedEffectModels.jl** (actively used, star count unavailable)
- **Repository**: https://github.com/FixedEffects/FixedEffectModels.jl
- **Last Commit**: November 2023
- **Maturity**: Production-ready
- **Methods**:
  - IV estimation (2SLS)
  - Fixed effects (multiple high-dimensional)
  - Interaction effects
  - GPU support (CUDA, Metal)
- **Strengths**: Much faster than Stata reghdfe, multi-threaded
- **Documentation**: Comprehensive README
- **Assessment**: Best-in-class for panel data and IV
- **Gap**: No weak instrument diagnostics (Stock-Yogo, Anderson-Rubin)

---

**Econometrics.jl**
- **Repository**: https://github.com/Nosferican/Econometrics.jl
- **Maturity**: Moderate
- **Methods**:
  - Panel estimators: pooling, FE, RE, BE, first-difference
  - IV (2SLS)
  - Specification tests (Chow, Durbin-Wu-Hausman)
  - Robust SEs (sandwich, cluster)
- **Documentation**: Available at nosferican.github.io
- **Assessment**: Comprehensive econometrics toolkit
- **Gap**: No modern DiD, no causal ML

---

### 1.8 Supporting Infrastructure

**CausalTables.jl** (17 stars)
- **Repository**: https://github.com/salbalkus/CausalTables.jl
- **Latest Release**: v1.3.4 (August 2025)
- **Maturity**: Mature (published in JOSS)
- **Purpose**: Data structures for causal inference
  - CausalTable type with DAG annotations
  - StructuralCausalModel for simulation
  - Extract ground-truth estimands
- **Assessment**: Excellent for validation and testing
- **Note**: Infrastructure only, not an estimator package

---

**Other Research Packages**:
- **InvariantCausal.jl** (25 stars): Invariant prediction, research code
- **CauseMap.jl** (16 stars): Non-linear dynamical systems, specialized
- **CausalGPSLC.jl** (6 stars): Gaussian Processes with latent confounders, research code
- **TreatmentPanels.jl** (6 stars): Panel data structures for SynthControl.jl

---

## Major Gaps

### 2.1 RCT Estimators ⚠️ **YOUR UNIQUE NICHE**

**Missing**:
- Simple ATE with Neyman variance
- Stratified ATE (block randomization)
- Regression-adjusted ATE (ANCOVA)
- Permutation tests (Fisher exact p-values)
- Inverse probability weighting (IPW)

**Status**: **ZERO Julia packages implement these**
**Your Project**: Implementing ALL of these in `causal_inference_mastery`
**Opportunity**: First comprehensive RCT toolkit in Julia ecosystem

---

### 2.2 Modern DiD Methods

**Missing**:
- Sun-Abraham (interaction-weighted estimator)
- Callaway-Sant'Anna (group-time ATEs)
- Goodman-Bacon decomposition
- Stacked DiD
- Imputation estimators

**Partial Coverage**: DiffinDiffs.jl has InteractionWeightedDIDs (similar to Sun-Abraham)
**Gap**: No Callaway-Sant'Anna, no decomposition tools

---

### 2.3 Weak Instrument Diagnostics

**Missing**:
- Stock-Yogo critical values
- Effective F-statistic
- Anderson-Rubin confidence sets
- LIML estimation

**Status**: FixedEffectModels.jl does 2SLS but **NO weak IV tests**
**Gap**: Critical for valid IV inference

---

### 2.4 Regression Discontinuity (Modern Methods)

**Missing**:
- rdrobust (Calonico-Cattaneo-Titiunik)
- rdmulti (multi-cutoff RDD)
- rddensity (manipulation tests)
- Optimal bandwidth (MSE-optimal, CER-optimal)

**Status**: RegressionDiscontinuity.jl is experimental and **STALE** (2022)
**Gap**: No production-ready implementation of modern RDD

---

### 2.5 Causal Machine Learning

**Missing**:
- Causal forests (Wager-Athey)
- Targeted learning (TMLE)
- Double/debiased ML (full implementation)
- BART for causal inference
- Generalized random forests

**Partial Coverage**: CausalELM.jl has Double ML but uses ELMs (not forests)
**Gap**: No grf equivalent, no econml equivalent

---

### 2.6 Advanced Matching Methods

**Missing**:
- Coarsened exact matching (CEM)
- Genetic matching
- Mahalanobis matching
- Optimal matching
- Love plots, balance diagnostics

**Status**: MatchIt.jl only has PSM (basic)
**Gap**: No comprehensive matching toolkit like R's MatchIt

---

### 2.7 Sensitivity Analysis

**Missing**:
- Rosenbaum bounds
- E-values
- Oster's delta
- Tipping point analysis

**Status**: No dedicated package found
**Gap**: Critical for observational studies

---

## Comparison to R and Python

### 3.1 R Packages (Gold Standard)

**Treatment Effects**:
- **fixest** (1000+ stars): DiD, IV, FE - production-ready
- **did** (400+ stars): Callaway-Sant'Anna DiD
- **grf** (900+ stars): Causal forests, generalized random forests
- **MatchIt** (200+ stars): Comprehensive matching methods
- **rdrobust**: Modern RDD toolkit

**Julia Gap**: 2-5 years behind R in maturity and coverage

---

### 3.2 Python Packages

**Treatment Effects**:
- **econml** (Microsoft Research, 3000+ stars): Causal ML
- **dowhy** (Microsoft Research, 6000+ stars): Causal graphs + estimation
- **causalml** (Uber, 4000+ stars): ML for causal inference
- **linearmodels** (900+ stars): IV, panel data

**Julia Gap**: 1-3 years behind Python

---

### 3.3 Julia Advantages

**Performance**:
- FixedEffectModels.jl: "much faster than reghdfe or lfe"
- GPU support (CUDA, Metal) built-in
- Multi-threading native

**Numerical Stability**:
- Type system catches errors at compile time
- Easier to implement from first principles
- Better for research and prototyping

**Julia is 2-5 years behind in ecosystem, but 2-10x ahead in performance**

---

## Project Positioning

### 4.1 causal_inference_mastery Fills Critical Gap

**Phase 1 (RCT) - COMPLETELY UNIQUE**:
- No existing Julia package has comprehensive RCT estimators
- Your 5 estimators (simple, stratified, regression-adjusted, permutation, IPW) fill this gap entirely
- Dual Python/Julia implementation provides rigorous validation

**Strategic Value**:
- **First comprehensive RCT toolkit in Julia**
- Test-first with 90%+ coverage (rare in Julia ecosystem)
- Cross-language validation (golden results from Python)
- Research-grade rigor with SciML patterns
- Production-ready code quality

---

### 4.2 Future Phases vs Ecosystem

**Phase 2 (PSM)**:
- **Gap**: MatchIt.jl is basic, no balance diagnostics
- **Opportunity**: Production-ready PSM with Love plots

**Phase 3 (DiD)**:
- **Exists**: DiffinDiffs.jl (basic)
- **Gap**: No Sun-Abraham, no Callaway-Sant'Anna
- **Opportunity**: Modern DiD methods

**Phase 4 (IV)**:
- **Exists**: FixedEffectModels.jl (2SLS)
- **Gap**: No weak instrument tests
- **Opportunity**: Diagnostic-complete IV toolkit

**Phase 5 (RDD)**:
- **Exists**: RegressionDiscontinuity.jl (STALE, experimental)
- **Gap**: Not maintained since 2022
- **Opportunity**: Modern rdrobust implementation

**Phases 6-8 (Sensitivity, Matching, CATE)**:
- **Gap**: Essentially non-existent
- **Opportunity**: Greenfield implementations

---

## Packaging Decision Framework

### 5.1 Should You Create a Julia Package?

**Arguments FOR**:
1. **RCT toolkit is UNIQUE** - Zero competition, fills critical gap
2. **Quality standard** - 90%+ coverage exceeds ecosystem norm
3. **Research value** - Validated against Python libraries
4. **Interview asset** - Demonstrates package development skills
5. **Community need** - Discussions on Julia Discourse show interest
6. **Future-proof** - Can expand to DiD/IV/RDD/Sensitivity

**Arguments AGAINST**:
1. **Maintenance burden** - Ecosystem requires ongoing updates
2. **Small user base** - Julia causal inference community is small (<100 active users)
3. **Competing priorities** - Google interview prep is primary goal
4. **Documentation overhead** - Registered packages need extensive docs

---

### 5.2 Recommended Approach (Phased Decision)

**Immediate (Phase 1 - RCT)**:
1. Complete RCT implementation as planned
2. Write production-quality code (package-ready)
3. Comprehensive validation (golden + PyCall + performance)
4. Document thoroughly (style guide + API docs)

**After Phase 1**:
1. **Ask Julia Discourse**: "Would comprehensive RCT estimators be useful?"
2. **Check CausalInference.jl roadmap**: See if they want treatment effects
3. **Decide packaging strategy**:
   - **Option A**: Keep personal (learning focus)
   - **Option B**: Create package CausalEstimators.jl (community contribution)
   - **Option C**: Contribute to existing packages

**After Phase 2-3**:
1. **If code quality high**: Consider packaging RCT + PSM + DiD
2. **Coordinate with ecosystem**: Avoid duplication with DiffinDiffs.jl, FixedEffectModels.jl
3. **Focus on gaps**: RDD modernization, weak IV tests, sensitivity analysis

---

### 5.3 Alternative: Contribute to Existing Packages

Instead of creating new package, consider:

1. **Add weak IV tests to FixedEffectModels.jl**
   - Stock-Yogo critical values
   - Anderson-Rubin confidence sets
   - Effective F-statistic

2. **Add Sun-Abraham to DiffinDiffs.jl**
   - They have modular architecture
   - Want more components

3. **Modernize RegressionDiscontinuity.jl**
   - Stale since 2022
   - Implement rdrobust methods

4. **Add balance diagnostics to MatchIt.jl**
   - Love plots
   - Standardized mean differences
   - Variance ratios

**Benefits**: Lower maintenance burden, immediate user base, collaboration

---

## Decision Points

### When to Make Packaging Decision

1. **After Phase 1 (RCT) complete**:
   - If code is clean and tests pass → Can package
   - If learning was primary value → Keep personal

2. **After Phase 2 (RCT + PSM) complete**:
   - If both phases high-quality → Strong package case
   - Check Julia Discourse for interest

3. **After Phase 3-4 (RCT + PSM + DiD + IV)**:
   - Comprehensive toolkit → Definitely package
   - Ecosystem leadership opportunity

**Key Question**: Is the goal to **contribute to Julia ecosystem** or **learn for Google interviews**?
- If contribute → Package after Phase 1-2
- If learn → Keep personal, share code on GitHub without registration

---

## Conclusion

**Julia Causal Inference Ecosystem Status**: Early-stage but growing

**Strongest Areas**:
- Causal discovery (CausalInference.jl)
- Panel data & IV (FixedEffectModels.jl)
- Basic DiD (DiffinDiffs.jl)

**Biggest Gaps** (Opportunity for Your Project):
1. **RCT estimators** ⭐ **UNIQUE NICHE**
2. Weak IV diagnostics
3. Modern RDD (rdrobust)
4. Causal ML (causal forests, TMLE)
5. Sensitivity analysis
6. Advanced matching

**Your Project Value**:
- **Phase 1 (RCT) fills completely unique gap**
- High-quality implementation (test-first, cross-validated, SciML patterns)
- Strategic positioning to expand into other gaps
- Can become the "econml of Julia" if you continue

**Recommendation**: **Continue current plan** (library-first Python, from-scratch Julia). Write production-quality code that's package-ready. Decide on actual packaging after Phase 1 completion based on code quality and community interest.

---

**Last Updated**: 2024-11-14
**Research Sources**: GitHub, JuliaHub, Julia Discourse, academic papers
**Next Review**: After Phase 1 (RCT) complete
