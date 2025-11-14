# Causal Inference Mastery - Project Roadmap

**Created**: 2024-11-14
**Last Updated**: 2024-11-14
**Project Status**: Phase 1 - IN PROGRESS

## Project Goal

Build deep, rigorous understanding of causal inference methods through dual-language implementation (Python + Julia) with research-grade validation. Focus on correctness, mathematical rigor, and practical application for both personal learning and professional interviews.

## Completed Phases

### Phase 0: Project Setup ✅ COMPLETE
**Duration**: 1 hour
**Completed**: 2024-11-14

**Deliverables**:
- Directory structure created
- Planning documents initialized
- Quality infrastructure configured

**Evidence**:
- Directories: `docs/`, `src/`, `tests/`, `julia/`, `notebooks/`, `validation/`
- Files: `ROADMAP.md`, `CURRENT_WORK.md`, `pyproject.toml`, `.pre-commit-config.yaml`

---

## Current Phase

### Phase 1: RCT Foundation 🔄 IN PROGRESS
**Started**: 2024-11-14
**Estimated Duration**: 20-25 hours
**Target Completion**: Week 1

**Objective**: Implement Randomized Controlled Trial estimators with full mathematical rigor, test-first development, and cross-language validation. Establish patterns for all subsequent methods.

**Deliverables**:
1. Python Implementation
   - `src/causal_inference/rct/estimators.py` - Core ATE estimation
   - `src/causal_inference/data/dgp.py` - Data Generating Process
   - Tests with 90%+ coverage
   - Known-answer validation
   - Monte Carlo validation (1000 runs)

2. Julia Implementation
   - `julia/src/RCT.jl` - From-scratch implementation
   - `julia/test/test_rct.jl` - Test suite
   - Mathematical clarity

3. Cross-Language Validation
   - Agreement to 10 decimal places
   - Validation scripts in `validation/cross_language/`

4. Documentation
   - `proofs/01_RCT_Mathematics.md` - Full derivations
   - `notebooks/01_data_quality_validation.ipynb`
   - `notebooks/02_rct_implementation.ipynb`

**Success Criteria**:
- [ ] All Python tests pass (coverage > 90%)
- [ ] All Julia tests pass
- [ ] Known-answer tests pass (hand-calculated examples)
- [ ] Monte Carlo: bias < 0.05, coverage 94-96%
- [ ] Cross-language: Julia == Python (rtol=1e-10)
- [ ] Notebooks execute end-to-end
- [ ] Mathematical proofs complete

**Status**: Setting up infrastructure

---

## Planned Phases

### Phase 2: Propensity Score Methods 📅 PLANNED
**Estimated Duration**: 20-25 hours
**Dependencies**: Phase 1 complete

**Objective**: Implement PSM with correct bootstrap standard errors, avoiding Abadie-Imbens issues for matching with replacement.

**Key Deliverables**:
- Bootstrap SE implementation
- Balance diagnostics (SMD < 0.1)
- Love plots
- LaLonde dataset application

### Phase 3: Difference-in-Differences 📅 PLANNED
**Estimated Duration**: 25-30 hours
**Dependencies**: Phase 1, 2 complete

**Objective**: Deep dive into TWFE bias, implement modern estimators (Sun-Abraham, Callaway-Sant'Anna).

**Key Deliverables**:
- Goodman-Bacon decomposition
- TWFE bias visualization
- Sun-Abraham estimator
- Card-Krueger replication

### Phase 4: Instrumental Variables 📅 PLANNED
**Estimated Duration**: 20-25 hours
**Dependencies**: Phase 1-3 complete

**Objective**: IV with proper weak instrument diagnostics and LATE vs ATE distinction.

**Key Deliverables**:
- 2SLS implementation
- Stock-Yogo weak instrument tests
- LATE interpretation
- Angrist-Krueger replication

### Phase 5: Regression Discontinuity 📅 PLANNED
**Estimated Duration**: 15-20 hours

**Objective**: RDD with optimal bandwidth selection.

### Phase 6: Sensitivity Analysis 📅 PLANNED
**Estimated Duration**: 15-20 hours

**Objective**: Robustness to unmeasured confounding.

### Phase 7: Matching Methods 📅 PLANNED
**Estimated Duration**: 15-20 hours

**Objective**: Beyond PSM - CEM, Mahalanobis, Genetic matching.

### Phase 8: CATE & Advanced Methods 📅 PLANNED
**Estimated Duration**: 20-25 hours

**Objective**: Heterogeneous treatment effects, meta-learners, causal forests.

---

## Decision Log

### 2024-11-14: Project Inception
**Decision**: Created causal_inference_mastery as standalone research project

**Context**: Need rigorous causal inference understanding for:
- Google L5 interview preparation
- Personal research projects
- Professional work at Prudential
- Deep mathematical understanding

**Rationale**:
1. Existing guide (causal_inference_guide_2025) has embedded code but no validation
2. Notebooks from Python Causality Handbook have known methodological issues (PSM SEs, TWFE bias)
3. Dual Julia/Python implementation provides cross-validation confidence
4. Research mindset (no deadline pressure) allows proper depth

**Impact**:
- New top-level project at `~/Claude/causal_inference_mastery/`
- Will be tracked by ProjectRegistry
- Separate from job_applications guide (which will be rewritten later using this research)

**Files Created**: Directory structure, planning documents

---

### 2024-11-14: Start with RCT (Not DiD)
**Decision**: Implement RCT first, not DiD

**Context**: Initial plan was to start with DiD (most critical for L5 interviews)

**Rationale**:
1. RCT is the gold standard - always works, builds confidence
2. DiD has complex TWFE bias issue - better to tackle after solid foundation
3. Progressive complexity: RCT → PSM → DiD → IV
4. User's ADHD workflow benefits from early wins

**Alternative Considered**: Start with DiD
- Pro: Most interview-critical, shows advanced knowledge
- Con: Risk of getting stuck on complexity early, losing momentum

**Impact**: Week 1 focuses on RCT, DiD moved to Week 3

**Files Affected**: Phase 1 plan, timeline

---

### 2024-11-14: Python First, Then Julia
**Decision**: Implement Python version first, then Julia

**Context**: Debated whether to do Julia first (deeper understanding) or Python first (working code)

**Rationale**:
1. Python with libraries provides quick validation of approach
2. Interviews are Python-based, need this solid first
3. Julia deepens understanding after confirming correctness
4. Cross-validation happens after both complete (per major component)

**Alternative Considered**: Julia first
- Pro: Forces understanding from first principles
- Con: Slower initial progress, risk of implementing wrong approach

**Impact**: Each method follows Python → Julia → Cross-validate workflow

---

### 2024-11-14: Modules Before Notebooks
**Decision**: Develop modules in `src/` before creating notebooks

**Context**: Debated exploratory notebook-first vs structured module-first

**Rationale**:
1. Test-first development requires clean module structure
2. Notebooks for demonstration after correctness established
3. Lessons from annuity_forecasting: notebooks 01-04 sequence after modules solid
4. Prevents "notebook mess" anti-pattern

**Alternative Considered**: Notebooks first
- Pro: More exploratory, iterate faster
- Con: Risk of not modularizing, harder to test

**Impact**: Notebooks created on Day 6-7 of each phase (after modules validated)

---

### 2024-11-14: Test-First Development (MANDATORY)
**Decision**: Write tests BEFORE implementation, enforce 90%+ coverage

**Context**: Inspired by annuity_forecasting Phase 0-9 wrong specification lesson

**Rationale**:
1. Known-answer tests catch subtle bugs (proven in past projects)
2. 90%+ coverage mandatory via pytest configuration
3. Pre-commit hooks enforce quality
4. Monte Carlo validation (1000 runs) ensures statistical properties

**Impact**:
- Every function gets known-answer test first
- Coverage enforced by pytest (fail build if <90%)
- Cross-language validation provides additional confidence

**Files Affected**: `pyproject.toml`, `.pre-commit-config.yaml`, all test files

---

### 2024-11-14: Library-First, Julia-Deep Validation Strategy
**Decision**: Use Python libraries (linearmodels, pyfixest, econml) for initial implementations, then implement from scratch in Julia using library outputs as validation benchmarks.

**Context**: After implementing simple_ate from scratch in Python, reconsidered approach for remaining estimators based on:
- User preference for research depth + essential Julia
- Need for both practical skills (library usage) and theoretical understanding (from-scratch)
- Efficiency of using established libraries as "known good" references

**Rationale**:
1. **Library-First Python**: Leverage battle-tested implementations (linearmodels, pyfixest, econml)
   - Faster initial progress
   - Learn best practices from established code
   - Provides "golden results" for validation
   - Interview-relevant (employers use these libraries)

2. **From-Scratch Julia**: Implement all methods from mathematical first principles
   - Deep understanding of algorithms
   - Numerical intuition from debugging
   - Cross-validation ensures correctness
   - Research-quality implementation

3. **Cross-Language Validation**: Library outputs benchmark Julia implementations
   - Julia must match Python libraries to rtol < 1e-10
   - If they disagree, investigate until understood
   - Builds confidence in both implementations

**Alternative Considered**: From-scratch Python first, then Julia
- Pro: Deeper Python understanding
- Con: Slower progress, risk of implementing bugs in both languages
- Con: Less exposure to production-quality code

**Impact on Phase 1**:
- Task 5-8: Implement stratified_ate, regression_adjusted_ate, permutation_test, ipw_ate using Python libraries
- Task 9: Capture "golden results" for Julia benchmarking
- Task 10-14: Julia from-scratch implementations of all 5 RCT estimators
- Task 15: Cross-language validation (Julia vs Python library outputs)
- Task 16-17: Monte Carlo validation on BOTH implementations
- Task 18: Comparative documentation

**Impact on Phases 2-8**:
- All future methods follow same pattern: Python libraries → Julia from-scratch → Cross-validate
- Examples:
  - Phase 3 DiD: pyfixest Sun-Abraham → Julia from-scratch → Validate
  - Phase 4 IV: linearmodels 2SLS → Julia from-scratch → Validate

**Benefits**:
- Best of both worlds: practical + theoretical
- Faster progress while maintaining depth
- Production code quality from libraries
- Research understanding from Julia
- Interview-ready on both fronts

**Files Affected**: Phase 1 plan, all future phase plans

---

## Project Metrics

### Current Status
- **Phases Complete**: 0 / 8
- **Methods Implemented**: 0 / 8
- **Test Coverage**: 0% (target: 90%+)
- **Cross-Language Validations**: 0 / 8
- **Notebooks Created**: 0
- **Git Commits**: 1

### Quality Metrics (Targets)
- Test Coverage: >90%
- Monte Carlo Bias: <0.05
- Monte Carlo Coverage: 94-96%
- Cross-Language Agreement: rtol < 1e-10
- Commits per day: 3-5
- Documentation: Every function

---

## Timeline

**Total Estimated Duration**: 12-16 weeks (research mode, no rush)

- **Week 1**: Phase 1 (RCT)
- **Week 2**: Phase 2 (PSM)
- **Week 3**: Phase 3 (DiD)
- **Week 4**: Phase 4 (IV)
- **Week 5-6**: Phase 5-6 (RDD, Sensitivity)
- **Week 7-8**: Phase 7-8 (Matching, CATE)
- **Week 9-16**: Refinement, additional methods, LaTeX guide creation

---

## References

**Key Papers** (to be added as we implement each method):
- Neyman (1923) - Potential outcomes framework
- Fisher (1935) - Randomization inference
- Rubin (1974) - Causal model
- Rosenbaum & Rubin (1983) - Propensity score matching
- Angrist & Imbens (1995) - LATE theorem
- Chernozhukov et al. (2018) - Double machine learning

**Methodological Guidance**:
- Imbens & Rubin (2015) - Causal Inference for Statistics
- Angrist & Pischke (2009) - Mostly Harmless Econometrics
- Cunningham (2021) - Causal Inference: The Mixtape
- Facure (2022) - Causal Inference for the Brave and True

---

**Last Updated**: 2024-11-14 12:31
**Next Update**: After Phase 1 completion
