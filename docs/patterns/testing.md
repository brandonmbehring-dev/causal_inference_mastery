# Testing Patterns

Test-first development workflow and standards for causal inference implementations.

---

## Core Principle: Test-First Development

**MANDATORY**: Write tests before implementation.

1. Define the expected behavior
2. Write tests that verify it
3. Implement until tests pass
4. Refactor while keeping tests green

---

## Test Categories

### Unit Tests (Layer 1)
- Test individual functions in isolation
- Use known-answer inputs with hand-calculated outputs
- Fast execution (< 1 second each)

### Adversarial Tests (Layer 2)
- Edge cases and boundary conditions
- Error handling and input validation
- Examples: empty groups, perfect separation, extreme values

### Monte Carlo Tests (Layer 3)
- Statistical property validation
- 5,000-25,000 simulation runs
- Mark with `@pytest.mark.slow` or `@pytest.mark.monte_carlo`

### Cross-Language Tests (Layer 4)
- Python ↔ Julia parity
- Located in `tests/validation/cross_language/`

---

## Error Handling Pattern

**NEVER fail silently. Always raise explicit errors.**

```python
def estimate_ate(outcome: np.ndarray, treatment: np.ndarray) -> ATEResult:
    """Validate inputs immediately, fail fast with diagnostics."""

    # Input validation
    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated units found")
    if n_control == 0:
        raise ValueError("No control units found")

    # Proceed with estimation...
```

---

## Input Validation Pattern

```python
def validate_propensity_scores(
    propensity: np.ndarray,
    trim_at: tuple[float, float] = (0.01, 0.99)
) -> np.ndarray:
    """Validate and optionally trim propensity scores."""

    # Check bounds
    if np.any(propensity <= 0) or np.any(propensity >= 1):
        raise ValueError(
            f"Propensity scores must be in (0, 1). "
            f"Found min={propensity.min():.4f}, max={propensity.max():.4f}"
        )

    # Check for perfect separation
    if np.any(propensity < 1e-10) or np.any(propensity > 1 - 1e-10):
        raise ValueError(
            "Perfect separation detected. "
            "Propensity scores are degenerate (0 or 1)."
        )

    # Warn about extreme values
    if np.any(propensity < trim_at[0]) or np.any(propensity > trim_at[1]):
        warnings.warn(
            f"Extreme propensity scores detected. "
            f"Consider trimming at ({trim_at[0]}, {trim_at[1]})"
        )

    return propensity
```

---

## pytest Markers

```python
# In conftest.py or pyproject.toml
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "monte_carlo: marks Monte Carlo simulation tests",
    "cross_language: marks Python-Julia parity tests",
]
```

Usage:
```bash
# Run fast tests only
pytest -m "not slow"

# Run Monte Carlo tests only
pytest -m "monte_carlo"

# Skip cross-language tests
pytest -m "not cross_language"
```

---

## Fixture Patterns

### Deterministic Test Data
```python
@pytest.fixture
def simple_rct_data():
    """Deterministic RCT data for reproducible tests."""
    np.random.seed(42)
    n = 100
    treatment = np.array([1] * 50 + [0] * 50)
    outcome = 2.0 * treatment + np.random.normal(0, 1, n)
    return outcome, treatment
```

### DGP Generators
```python
def generate_rct_dgp(
    n: int = 200,
    true_ate: float = 2.0,
    noise_sd: float = 1.0,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate RCT data with known true ATE."""
    if seed is not None:
        np.random.seed(seed)

    treatment = np.random.binomial(1, 0.5, n)
    noise = np.random.normal(0, noise_sd, n)
    outcome = true_ate * treatment + noise

    return outcome, treatment
```

---

## xfail and skip Patterns

```python
@pytest.mark.xfail(
    reason="Known limitation: PSM coverage drops with extreme propensity separation",
    strict=False  # Allow passing (not a regression if it passes)
)
def test_psm_extreme_separation():
    """Document known limitation."""
    ...

@pytest.mark.skip(reason="R triangulation deferred to future phase")
def test_compare_to_matchit():
    """Placeholder for R comparison."""
    ...
```

---

## Test File Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_rct/
│   ├── test_simple_ate.py   # Unit tests
│   ├── test_regression_ate.py
│   └── test_stratified_ate.py
├── observational/
│   ├── test_ipw.py
│   └── test_doubly_robust.py
└── validation/
    ├── monte_carlo/
    │   └── test_rct_monte_carlo.py
    ├── adversarial/
    │   └── test_ipw_adversarial.py
    └── cross_language/
        └── test_python_julia_rct.py
```

---

*Last updated: 2025-12-16 (Session 37.5)*
