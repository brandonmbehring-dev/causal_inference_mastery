# Validation Architecture

6-layer validation framework for causal inference implementations.

---

## The 6 Layers

| Layer | Purpose | Implementation | Target |
|-------|---------|----------------|--------|
| 1 | **Known-Answer** | Hand-calculated expected values | 100% pass |
| 2 | **Adversarial** | Edge cases, boundary conditions | 100% pass |
| 3 | **Monte Carlo** | 5,000-25,000 run simulations | Bias/coverage targets |
| 4 | **Cross-Language** | Python ↔ Julia parity | rtol < 1e-10 |
| 5 | **R Triangulation** | External reference validation | (deferred) |
| 6 | **Golden Reference** | 111KB JSON frozen results | Exact match |

---

## Layer Details

### Layer 1: Known-Answer Tests
- Hand-calculated expected values
- Simple DGPs with known analytical solutions
- Example: 2-group difference with known mean difference

### Layer 2: Adversarial Tests
- Edge cases: empty groups, single observation, extreme values
- Boundary conditions: perfect separation, collinearity
- Error handling: appropriate exceptions with diagnostics

### Layer 3: Monte Carlo Validation
Statistical property verification across repeated samples:

| Method | Bias Target | Coverage Target | SE Accuracy |
|--------|-------------|-----------------|-------------|
| RCT (unconfounded) | < 0.05 | 93-97% | < 10% |
| Observational (IPW, DR) | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 (expected) | >= 95% | < 150% |
| DiD | < 0.10 | 93-97% | < 15% |
| IV | < 0.15 | 90-97% | < 20% |
| RDD | < 0.15 | 90-97% | < 20% |

**Why PSM has looser targets**: Matching introduces additional variance and can have inherent bias from imperfect matches.

### Layer 4: Cross-Language Parity
- Python and Julia must agree to 10+ decimal places
- Same input data, same algorithm, same results
- Uses PyCall for Julia→Python validation

### Layer 5: R Triangulation (Deferred)
- Compare against established R packages (MatchIt, did, rdrobust)
- External validation of our implementations

### Layer 6: Golden Reference
- 111KB JSON file with frozen test results
- Prevents regressions
- Updated only with explicit approval

---

## Monte Carlo Test Pattern

```python
@pytest.mark.monte_carlo
def test_estimator_unbiased():
    """Monte Carlo validation: 5000 runs, bias < 0.05, coverage 93-97%."""
    TRUE_ATE = 2.0
    results = []

    for _ in range(5000):
        y, t = generate_dgp(n=200, true_ate=TRUE_ATE)
        result = estimator(y, t)
        results.append(result)

    # Bias check
    estimates = [r.ate for r in results]
    bias = np.mean(estimates) - TRUE_ATE
    assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold"

    # Coverage check
    coverage = np.mean([
        r.ci_lower < TRUE_ATE < r.ci_upper
        for r in results
    ])
    assert 0.93 < coverage < 0.97, f"Coverage {coverage:.2%} outside range"

    # SE accuracy check
    empirical_se = np.std(estimates)
    mean_reported_se = np.mean([r.se for r in results])
    se_ratio = mean_reported_se / empirical_se
    assert 0.9 < se_ratio < 1.1, f"SE ratio {se_ratio:.2f} outside range"
```

---

## Cross-Language Parity Pattern

```python
def test_method_matches_julia():
    """Python and Julia must agree to 10 decimal places."""
    from tests.validation.cross_language.julia_interface import julia_method

    # Same data
    y, t, X = load_fixture("test_data")

    # Python result
    py_result = python_method(y, t, X)

    # Julia result (via PyCall wrapper)
    jl_result = julia_method(y, t, X)

    # Verify parity
    assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=1e-10)
    assert np.isclose(py_result.se, jl_result["se"], rtol=1e-10)
```

---

## Test Organization

```
tests/
├── test_rct/                    # Layer 1 + 2 for RCT
├── test_did/                    # Layer 1 + 2 for DiD
├── observational/               # Layer 1 + 2 for IPW/DR/PSM
├── validation/
│   ├── monte_carlo/             # Layer 3
│   │   ├── test_rct_monte_carlo.py
│   │   ├── test_did_monte_carlo.py
│   │   └── ...
│   ├── adversarial/             # Layer 2 (stress tests)
│   │   ├── test_ipw_adversarial.py
│   │   └── ...
│   └── cross_language/          # Layer 4
│       ├── julia_interface.py   # PyCall wrappers
│       ├── test_python_julia_rct.py
│       └── ...
└── golden_results/              # Layer 6
    └── python_golden_results.json
```

---

*Last updated: 2025-12-16 (Session 37.5)*
