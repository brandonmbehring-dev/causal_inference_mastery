# Validation Architecture

**Purpose**: Comprehensive correctness verification for causal inference implementations.

---

## 6-Layer Validation System

| Layer | Directory | Purpose | Status |
|-------|-----------|---------|--------|
| 1 | `tests/test_*/` | **Known-Answer**: Hand-calculated expected values | ✅ Active |
| 2 | `adversarial/` | **Edge Cases**: Boundary conditions, extreme values | ✅ Active |
| 3 | `monte_carlo/` | **Statistical**: 5K-25K run simulations | ✅ Active |
| 4 | `cross_language/` | **Parity**: Python ↔ Julia agreement | ⚠️ Conditional |
| 5 | `r_triangulation/` | **External**: R package validation | ⚠️ Manual |
| 6 | `test_golden_reference.py` | **Regression**: Frozen result comparison | ✅ Active |

---

## Cross-Language Tests (Layer 4)

### Execution Conditions

Cross-language tests **skip automatically** if Julia is unavailable:

```python
@pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-validation"
)
```

**Requirements for cross-language tests to run:**

1. **Julia installed** and accessible via PATH
2. **juliacall** Python package installed (`pip install juliacall`)
3. **Julia project** activated (`julia/Project.toml` dependencies installed)

### Running Cross-Language Tests

```bash
# Check if Julia is available
python -c "from tests.validation.cross_language.julia_interface import is_julia_available; print(is_julia_available())"

# Run cross-language tests (skips if Julia unavailable)
pytest tests/validation/cross_language/ -v

# Run only if Julia is available
pytest tests/validation/cross_language/ -v --strict-markers
```

### Validation Tolerance

All cross-language tests use `rtol=1e-10` for numerical comparisons:

```python
assert np.isclose(python_result, julia_result, rtol=1e-10)
```

This ensures 10 decimal place agreement between implementations.

---

## Monte Carlo Standards (Layer 3)

| Method Type | Bias Target | Coverage Target | SE Accuracy |
|-------------|-------------|-----------------|-------------|
| RCT (unconfounded) | < 0.05 | 93-97% | < 10% |
| Observational (confounded) | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 (expected) | ≥ 95% | < 150% (conservative) |

### Running Monte Carlo Tests

```bash
# Fast tests (exclude Monte Carlo)
pytest tests/ -m "not slow and not monte_carlo"

# Monte Carlo only (slow, ~30 min)
pytest tests/validation/monte_carlo/ -v

# Single method family
pytest tests/validation/monte_carlo/test_monte_carlo_did.py -v
```

---

## Golden Reference Tests (Layer 6)

### Purpose

Catch regressions by comparing current outputs to frozen reference results.

**File**: `tests/golden_results/python_golden_results.json` (111KB)

### When Tests Fail

If a golden reference test fails, either:

1. **Bug introduced** → Fix the code
2. **Intentional change** → Update golden file with new correct results

### Regenerating Golden Results

```bash
# Generate new golden results (after verifying correctness)
python scripts/generate_golden_results.py

# Compare old vs new
diff tests/golden_results/python_golden_results.json tests/golden_results/python_golden_results.json.new
```

---

## CI/CD Integration

### PR Workflow (`ci.yml`)

1. **Lint** → Black, Ruff formatting check
2. **Collect** → `pytest --collect-only` catches import errors
3. **Fast Tests** → Skip slow/monte_carlo markers

### Merge Workflow (`full-test.yml`)

1. **Full Tests** → Complete test suite
2. **Collection Check** → Verify no collection errors

### Scheduled Workflow (`scheduled.yml`)

- Weekly comprehensive validation
- Monte Carlo tests included

---

## Troubleshooting

### "Julia not available" Skips

```bash
# Install Julia
curl -fsSL https://install.julialang.org | sh

# Install juliacall
pip install juliacall

# Activate Julia project
cd julia && julia --project -e "using Pkg; Pkg.instantiate()"
```

### Collection Errors

```bash
# Identify broken imports
pytest tests/ --collect-only 2>&1 | grep -i error

# Fix: Usually missing imports or circular dependencies
```

### Monte Carlo Failures

Monte Carlo tests have inherent randomness. A single failure may be:

- **Flaky** → Re-run 2-3 times before investigating
- **Real issue** → Check bias/coverage systematically

---

**Created**: Session 168 (2026-01-01)
**Maintainer**: Brandon Behring
