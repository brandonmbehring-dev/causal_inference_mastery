# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

---

## Shared Foundation (Hub Reference)

**This project uses shared patterns from lever_of_archimedes**:

See: `~/Claude/lever_of_archimedes/patterns/` for:
- `testing.md` - 6-layer validation architecture
- `sessions.md` - Session workflow (CURRENT_WORK.md, ROADMAP.md)
- `git.md` - Commit format and workflow
- `style/python_style.yaml` - Black 100-char, strict mypy
- `style/julia_style.yaml` - SciML formatting, 92-char lines

**Core principles** (from hub):
1. NEVER fail silently - explicit errors always
2. Simplicity over complexity - 20-50 line functions
3. Immutability by default
4. Fail fast with diagnostics

---

## Project Overview

Dual-language causal inference implementation for deep methodological understanding through rigorous, research-grade implementations.

- **Python**: Modern libraries (pyfixest, linearmodels, econml, dowhy)
- **Julia**: From-scratch implementations for mathematical rigor
- **Goal**: Cross-language validation to 10 decimal places

### Current Status
- **Python**: Phases 1-5 COMPLETE (RCT, IPW, DR, PSM, DiD, IV, RDD)
- **Julia**: Phases 1-5 COMPLETE (RCT, PSM, DiD, IV, RDD)
- **Tests**: 2,420+ across both languages
- **Coverage**: 90%+ (Python), 99.6% pass rate (Julia)

---

## Quick Reference Commands

```bash
# Python tests (full suite with coverage)
pytest tests/ --cov=src/causal_inference --cov-report=term-missing --cov-fail-under=90

# Python tests (fast, skip slow)
pytest tests/ -m "not slow"

# Python tests (single module)
pytest tests/test_rct/ -v

# Julia tests
cd julia && julia --project test/runtests.jl

# Julia tests (specific file)
julia --project -e "using Pkg; Pkg.test()" -- test/did/test_classic_did.jl

# Code quality
black src/ tests/
ruff src/ tests/
mypy src/

# Pre-commit hooks
pre-commit run --all-files
```

---

## Project Structure

```
causal_inference_mastery/
├── src/causal_inference/           # Python modules (11,857 lines)
│   ├── rct/                        # 5 RCT estimators
│   ├── observational/              # IPW, DR, outcome regression
│   ├── psm/                        # Propensity score matching
│   ├── did/                        # Difference-in-differences
│   ├── iv/                         # Instrumental variables
│   └── rdd/                        # Regression discontinuity
├── julia/src/                      # Julia modules (12,084 lines)
│   ├── did/, iv/, rdd/             # Method implementations
│   └── CausalEstimators.jl         # Main module
├── tests/                          # Python test suite
│   ├── test_rct/, test_psm/        # Method-specific tests
│   └── validation/                 # 3-layer validation
│       ├── monte_carlo/            # Statistical simulations
│       ├── adversarial/            # Edge case tests
│       └── cross_language/         # Python ↔ Julia parity
└── docs/                           # Documentation (53 files)
    ├── plans/active/               # Current session plans
    ├── plans/implemented/          # Completed sessions
    └── SESSION_*.md                # Session documentation
```

---

## Validation Architecture (6 Layers)

| Layer | Purpose | Implementation | Target |
|-------|---------|----------------|--------|
| 1 | Known-Answer | Hand-calculated expected values | 100% pass |
| 2 | Adversarial | Edge cases, boundary conditions | 100% pass |
| 3 | Monte Carlo | 5,000-25,000 run simulations | Bias < 0.05-0.10 |
| 4 | Cross-Language | Python ↔ Julia parity | rtol < 1e-10 |
| 5 | R Triangulation | External reference (deferred) | - |
| 6 | Golden Reference | 111KB JSON frozen results | Exact match |

### Monte Carlo Validation Standards

| Method Type | Bias Target | Coverage Target | SE Accuracy |
|-------------|-------------|-----------------|-------------|
| RCT (unconfounded) | < 0.05 | 93-97% | < 10% |
| Observational (confounded) | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 (expected) | ≥ 95% | < 150% (conservative) |

---

## Quality Standards

- **Test Coverage**: 90%+ (enforced by pytest)
- **Test-First Development**: MANDATORY - tests before implementation
- **Pre-commit Hooks**: Black, Ruff, Mypy, large commit warnings
- **Function Length**: 20-50 lines target

---

## Documentation Hierarchy

| Document | Purpose |
|----------|---------|
| `CURRENT_WORK.md` | 30-second context resume (session tracking) |
| `docs/ROADMAP.md` | Master plan, phase tracking |
| `docs/METHODOLOGICAL_CONCERNS.md` | 13 tracked concerns (CRITICAL → MEDIUM) |
| `docs/SESSION_*.md` | Per-session documentation |
| `docs/plans/active/` | In-progress phase plans |
| `docs/plans/implemented/` | Completed phase plans |

---

## Methodological Concerns (Critical)

**ALWAYS check `docs/METHODOLOGICAL_CONCERNS.md` before implementing**:

| ID | Phase | Issue | Priority |
|----|-------|-------|----------|
| CONCERN-5 | PSM | Bootstrap SE invalid for matching with replacement | HIGH |
| CONCERN-11 | DiD | TWFE bias with staggered adoption | CRITICAL |
| CONCERN-12 | DiD | Pre-trends testing limitations | CRITICAL |
| CONCERN-13 | DiD | Cluster SE with few clusters | HIGH |
| CONCERN-16 | IV | Weak instrument detection | CRITICAL |
| CONCERN-22 | RDD | McCrary density test validity | HIGH |

---

## Code Style

### Python
- **Formatter**: Black with 100-character lines
- **Type hints**: Required everywhere (mypy strict mode)
- **Docstrings**: NumPy style with Parameters, Returns, Raises, Examples

### Julia
- **Formatter**: SciML style, 92-character lines
- **Documentation**: Full docstrings with examples
- **Immutability**: Functions ending in `!` mutate, otherwise return new

---

## Session Workflow

This project follows the hub session pattern (see `patterns/sessions.md`):

1. **Start**: Check `CURRENT_WORK.md` for context
2. **Plan**: For tasks >1hr, create `docs/plans/active/SESSION_N_*.md`
3. **Implement**: Test-first, commit frequently
4. **Document**: Update session file with results
5. **Complete**: Move plan to `implemented/`, update `CURRENT_WORK.md`

### Git Commit Format
```
type(scope): Short description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `validate`

---

## Key Implementation Patterns

### Error Handling
```python
# ALWAYS fail explicitly, NEVER fail silently
if propensity_scores.min() < 1e-6:
    raise ValueError(
        f"Propensity scores too close to 0 (min={propensity_scores.min():.2e}). "
        f"Positivity violation detected. Consider trimming or checking covariates."
    )
```

### Input Validation
```python
def simple_ate(outcome: np.ndarray, treatment: np.ndarray) -> ATEResult:
    """Validate inputs immediately, fail fast with diagnostics."""
    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")
```

### Monte Carlo Testing Pattern
```python
@pytest.mark.monte_carlo
def test_simple_ate_unbiased():
    """Monte Carlo validation: 5000 runs, bias < 0.05, coverage 93-97%."""
    results = []
    for _ in range(5000):
        y, t = generate_rct_dgp(n=200, true_ate=2.0)
        result = simple_ate(y, t)
        results.append(result)

    bias = np.mean([r.ate for r in results]) - 2.0
    coverage = np.mean([r.ci_lower < 2.0 < r.ci_upper for r in results])

    assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold"
    assert 0.93 < coverage < 0.97, f"Coverage {coverage:.2%} outside range"
```

---

## Cross-Language Validation (Python ↔ Julia)

```python
# tests/validation/cross_language/test_rct_parity.py
def test_simple_ate_matches_julia():
    """Python and Julia must agree to 10 decimal places."""
    from julia import Julia
    jl = Julia(compiled_modules=False)

    # Same data in both languages
    y, t = load_test_fixture("rct_simple")

    py_result = simple_ate(y, t)
    jl_result = jl.eval("simple_ate(y, t)")

    assert np.isclose(py_result.ate, jl_result.ate, rtol=1e-10)
    assert np.isclose(py_result.se, jl_result.se, rtol=1e-10)
```

---

## Environment Setup

```bash
# Python
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Julia
cd julia
julia --project -e "using Pkg; Pkg.instantiate()"
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `CURRENT_WORK.md` | 30-second context resume |
| `docs/ROADMAP.md` | Master plan (761 lines) |
| `docs/METHODOLOGICAL_CONCERNS.md` | 13 concerns tracked |
| `pyproject.toml` | Python config (Black, Ruff, Mypy, pytest) |
| `julia/Project.toml` | Julia dependencies |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/golden_results/` | Reference JSON (111KB) |

---

## Known Issues & Limitations

1. **PSM Limited Overlap**: xfail test documents coverage drops to 31% with extreme propensity separation
2. **TWFE Bias**: Documented in CONCERN-11, use Callaway-Sant'Anna or Sun-Abraham instead
3. **Cross-Language Tests**: Infrastructure exists but deferred pending full Python parity

---

## Session History (Recent)

| Session | Focus | Status |
|---------|-------|--------|
| 37 | Test suite stabilization (IPW adversarial fixes) | ✅ Complete |
| 36 | SimpleATE cross-language CI parity | ✅ Complete |
| 35 | DiD Event Study & TWFE cross-language validation | ✅ Complete |
| 34 | Observational cross-language validation (IPW, DR) | ✅ Complete |
| 22 | Project audit & documentation cleanup | ✅ Complete |
| 21 | Phase 2 Monte Carlo validation (IV, RDD) | ✅ Complete |
| 20 | Phase 0/0.5/1 Statistical correctness | ✅ Complete |

**Current**: Session 37.5 - Context engineering & documentation overhaul
