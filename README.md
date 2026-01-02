# Causal Inference Mastery

[![CI](https://github.com/brandondocusen/causal_inference_mastery/actions/workflows/ci.yml/badge.svg)](https://github.com/brandondocusen/causal_inference_mastery/actions/workflows/ci.yml)
[![Full Tests](https://github.com/brandondocusen/causal_inference_mastery/actions/workflows/full-test.yml/badge.svg)](https://github.com/brandondocusen/causal_inference_mastery/actions/workflows/full-test.yml)

**Status**: Python Phases 1-15+ COMPLETE | Julia Phases 1-15+ COMPLETE
**Created**: 2024-11-14
**Last Updated**: 2026-01-01 (Session 167 - Infrastructure Activation)
**Goal**: Deep, rigorous understanding of causal inference through dual-language implementation

---

## Project Overview

This project implements causal inference methods from first principles using both Python (with modern libraries) and Julia (from scratch). The dual-language approach provides cross-validation confidence while building deep mathematical understanding.

### Current Metrics (Verified Session 167 - Independent Audit)

| Metric | Value |
|--------|-------|
| Python source lines | 54,727 |
| Julia source lines | 43,699 |
| **Total source lines** | **98,426** |
| Python test functions | 3,854 |
| Julia test assertions | 5,121 |
| **Total test assertions** | **8,975** |
| Method families | 25 |
| Known bugs outstanding | 0 |

*Regenerate metrics: `python scripts/update_metrics.py --output`*

### Design Principles

1. **Test-First Development** - All tests written before implementation
2. **Known-Answer Validation** - Hand-calculated expected values
3. **Monte Carlo Validation** - 500-5000 run simulations confirm statistical properties
4. **Cross-Language Validation** - Python and Julia must agree to 10 decimal places
5. **Research-Grade Quality** - 90%+ test coverage, rigorous documentation

---

## Methods Implemented

### Python Implementation (25 Method Families)

| Phase | Method Family | Status | Key Features |
|-------|--------------|--------|--------------|
| 1 | **RCT** | ✅ Complete | ATE, IPW, stratified, permutation, regression-adjusted |
| 2 | **Observational** | ✅ Complete | IPW, DR, outcome regression |
| 3 | **PSM** | ✅ Complete | Nearest neighbor, Abadie-Imbens variance, balance diagnostics |
| 4 | **DiD** | ✅ Complete | Classic 2x2, TWFE, Callaway-Sant'Anna, Sun-Abraham, event study |
| 5 | **IV** | ✅ Complete | 2SLS, LIML, GMM, AR/CLR weak instrument tests |
| 6 | **RDD** | ✅ Complete | Sharp, fuzzy, bandwidth (IK, CCT approx), McCrary density test |
| 7 | **SCM** | ✅ Complete | Basic, augmented, placebo inference, donor diagnostics |
| 8 | **CATE** | ✅ Complete | Causal forest, DML, meta-learners, neural CATE |
| 9 | **Sensitivity** | ✅ Complete | E-value, Rosenbaum bounds |
| 10 | **RKD** | ✅ Complete | Sharp kink, fuzzy kink, bandwidth selection |
| 11 | **Bunching** | ✅ Complete | Excess mass, counterfactual estimation |
| 12 | **Selection** | ✅ Complete | Heckman two-step, MLE |
| 13 | **Bounds** | ✅ Complete | Manski bounds, Lee bounds |
| 14 | **Mediation** | ✅ Complete | Baron-Kenny, causal mediation |
| 15 | **Time Series** | ✅ Complete | VAR, SVAR, IRF, Granger, VECM, cointegration |
| + | **Additional** | ✅ Complete | QTE, MTE, principal stratification, shift-share, panel |

### Julia Implementation (Cross-Language Parity)

All major method families implemented with cross-language validation tests.
See `julia/src/` for implementations and `julia/test/` for validation.

---

## Quick Start

### Installation

```bash
# Clone repository
cd ~/Claude/causal_inference_mastery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Julia Setup

```bash
cd julia/
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project test/runtests.jl
```

---

## Project Structure

```
causal_inference_mastery/
├── src/causal_inference/          # Python modules (54,727 lines)
│   ├── rct/                       # RCT estimators
│   ├── observational/             # IPW, DR, outcome regression
│   ├── psm/                       # Propensity score matching
│   ├── did/                       # Difference-in-differences
│   ├── iv/                        # Instrumental variables
│   ├── rdd/                       # Regression discontinuity
│   ├── scm/                       # Synthetic control
│   ├── cate/                      # Treatment effect heterogeneity
│   ├── timeseries/                # VAR, SVAR, IRF, Granger
│   └── [15 more families...]      # See docs/METRICS_CURRENT.md
├── julia/                         # Julia implementation (43,699 lines)
│   ├── src/                       # Julia modules
│   └── test/                      # Julia tests (5,121 assertions)
├── tests/                         # Python test suite (3,854 tests)
│   ├── test_*/                    # Method-specific tests
│   └── validation/                # 4-layer validation
│       ├── monte_carlo/           # Statistical simulations
│       ├── adversarial/           # Edge case tests
│       ├── cross_language/        # Python ↔ Julia parity
│       └── audit/                 # Bug exposure tests
├── docs/                          # Documentation (70+ files)
│   ├── KNOWN_BUGS.md              # Bug tracking (0 outstanding)
│   ├── METRICS_CURRENT.md         # Verified metrics
│   ├── METHODOLOGICAL_CONCERNS.md # Implementation notes
│   └── plans/                     # Session plans
├── book/                          # LaTeX textbook (1MB PDF)
│   ├── main.tex                   # Book source
│   ├── chapters/                  # 7 parts (LaTeX)
│   └── main.pdf                   # Compiled book
├── scripts/                       # Automation
│   └── update_metrics.py          # Metrics generator
└── benchmarks/                    # Performance testing
```

---

## Validation Architecture (6 Layers)

| Layer | Purpose | Status |
|-------|---------|--------|
| 1 | Known-Answer | Hand-calculated expected values | ✅ VERIFIED |
| 2 | Adversarial | Edge cases, boundary conditions | ✅ VERIFIED |
| 3 | Monte Carlo | 500-5000 run simulations | ✅ VERIFIED |
| 4 | Cross-Language | Python ↔ Julia parity | ⏸️ Manual validation |
| 5 | R Triangulation | External reference | ⚠️ PARTIAL (8/25 families) |
| 6 | Golden Reference | Frozen JSON results | ✅ ACTIVE (11 tests) |

---

## Quality Standards

### Test Coverage
- Modules: **90%+** (enforced by pytest)
- Collection errors: **0** (verified Session 158)

### Validation Requirements

Before any method is considered complete:
- [x] All Python tests pass
- [x] All Julia tests pass
- [x] Known-answer validation passes
- [x] Monte Carlo: bias within method-specific thresholds
- [x] Monte Carlo: coverage 93-97%
- [x] Cross-language agreement (rtol < 1e-10)

---

## Development Workflow

### Test-First Cycle

1. **Write tests** with known answers (tests should FAIL)
2. **Implement** function to pass tests
3. **Run tests** until passing
4. **Validate** with Monte Carlo
5. **Cross-validate** Julia and Python
6. **Document** with proofs and notebooks
7. **Commit** with meaningful message

### Git Workflow

```bash
# Run quality checks
black src/ tests/
ruff src/ tests/
mypy src/
pytest tests/

# Commit
git add .
git commit -m "feat(method): Description"
```

---

## Book

A comprehensive LaTeX book accompanies this codebase:

**Causal Inference Mastery: A Computational Approach**

| Part | Topics |
|------|--------|
| 1. Foundations | Potential outcomes, identification, estimation |
| 2. Selection on Observables | IPW, matching, doubly robust |
| 3. Selection on Unobservables | IV, control function |
| 4. Natural Experiments | DiD, RDD, SCM |
| 5. Advanced Methods | CATE, heterogeneous effects |
| 6. Implementation | Computational approaches |
| 7. Time Series | VAR, SVAR, causal discovery |

**Location**: `book/main.pdf` (compiled), `book/chapters/` (LaTeX source)

---

## Key References

**Planning**:
- `CLAUDE.md` - Claude Code instructions
- `CURRENT_WORK.md` - Current session context
- `docs/ROADMAP.md` - Master plan
- `docs/METRICS_CURRENT.md` - Verified metrics
- `book/main.pdf` - Comprehensive textbook

**Methodological**:
- Imbens & Rubin (2015) - Causal Inference for Statistics
- Angrist & Pischke (2009) - Mostly Harmless Econometrics
- Cunningham (2021) - Causal Inference: The Mixtape

---

## Session History (Recent)

| Session | Focus | Status |
|---------|-------|--------|
| 167 | **Infrastructure Activation** - CI/CD commit, book commit, metrics update | ✅ Complete |
| 166 | Independent audit, golden reference tests | ✅ Complete |
| 159-165 | Time series (LP, Sign Restrictions, Proxy SVAR, TVP-VAR), Audit | ✅ Complete |
| 150-158 | Time series (VAR, SVAR, IRF), Causal Forest, Remediation | ✅ Complete |
| 140-149 | CATE parity, Neural CATE, OML | ✅ Complete |

---

## License

MIT License - Brandon Behring (2024-2025)

See `LICENSE` file for details.
