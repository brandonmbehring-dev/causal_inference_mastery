# Quick Reference

Copy/paste-ready commands and quick lookups for causal_inference_mastery.

---

## Method Selection

| Your Situation | Method |
|----------------|--------|
| Randomized experiment | RCT (simple, stratified, regression-adjusted) |
| Observational with confounders | IPW, Doubly Robust, PSM |
| Before/after comparison | Difference-in-Differences (DiD) |
| Endogenous variable | Instrumental Variables (IV) |
| Threshold/cutoff | Regression Discontinuity (RDD) |

---

## Test Commands

### Python (run from repo root)

```bash
# Full test suite with coverage
pytest tests/ --cov=src/causal_inference --cov-report=term-missing --cov-fail-under=90

# Fast tests (skip slow/Monte Carlo)
pytest tests/ -m "not slow"

# Single module
pytest tests/test_rct/ -v
pytest tests/test_did/ -v
pytest tests/observational/ -v

# Specific test file
pytest tests/test_rct/test_simple_ate.py -v

# Cross-language validation
pytest tests/validation/cross_language/ -v
```

### Julia (run from julia/ directory)

```bash
# Full test suite
cd julia && julia --project test/runtests.jl

# Specific test file
cd julia && julia --project -e 'include("test/did/test_classic_did.jl")'

# With verbose output
cd julia && julia --project test/runtests.jl --verbose
```

### Code Quality (run from repo root)

```bash
# Format Python
black src/ tests/

# Lint Python
ruff src/ tests/

# Type check
mypy src/

# All pre-commit hooks
pre-commit run --all-files
```

---

## Validation Checklist (Core)

- [ ] Layer 1: Known-answer tests pass
- [ ] Layer 2: Adversarial tests pass
- [ ] Layer 3: Monte Carlo validation
- [ ] Layer 4: Cross-language parity

See [patterns/validation.md](patterns/validation.md) for Layers 5-6 (R Triangulation, Golden Reference).

---

## Monte Carlo Targets

| Method | Bias | Coverage | SE Accuracy |
|--------|------|----------|-------------|
| RCT | < 0.05 | 93-97% | < 10% |
| IPW/DR | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 | >= 95% | < 150% |
| DiD | < 0.10 | 93-97% | < 15% |
| IV | < 0.15 | 90-97% | < 20% |
| RDD | < 0.15 | 90-97% | < 20% |

---

## Key Files

| File | Purpose |
|------|---------|
| `CURRENT_WORK.md` | Session context (start here) |
| `CLAUDE.md` | AI assistant guidance |
| `pyproject.toml` | Python config |
| `julia/Project.toml` | Julia config |
| `tests/conftest.py` | Shared fixtures |
| `tests/golden_results/` | Reference JSON |

---

## Git Commit Format

```
type(scope): Short description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `validate`

---

*Last updated: 2025-12-16 (Session 37.5)*
