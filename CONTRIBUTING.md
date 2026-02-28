# Contributing to Causal Inference Mastery

Thank you for your interest in contributing! This project implements causal inference methods from first principles in both Python and Julia with rigorous validation.

## Development Setup

```bash
# Clone and install
git clone https://github.com/brandon-behring/causal_inference_mastery.git
cd causal_inference_mastery
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/
pytest tests/ -m "not slow"  # Skip Monte Carlo simulations
```

### Julia Setup

```bash
cd julia/
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project test/runtests.jl
```

## Code Style

### Python
- **Formatter**: Black with 100-character line length
- **Linter**: Ruff
- **Type checking**: mypy (strict mode)
- **Docstrings**: NumPy style for all public functions

### Julia
- SciML formatting conventions
- 92-character line length

## Testing Requirements

Every method implementation requires:

1. **Known-answer tests** with hand-calculated expected values
2. **Monte Carlo validation** (500-5000 runs) confirming:
   - Bias within method-specific thresholds
   - Coverage between 93-97%
3. **Cross-language parity** — Python and Julia must agree to `rtol < 1e-10`
4. **Edge case tests** for boundary conditions

### Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific method family
pytest tests/test_did/ -v

# Monte Carlo only (slow)
pytest tests/validation/monte_carlo/ -v

# Cross-language validation
pytest tests/validation/cross_language/ -v
```

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests first** — tests should fail before implementation
3. **Implement** the method to pass tests
4. **Run the full test suite** — all tests must pass
5. **Run pre-commit hooks**: `pre-commit run --all-files`
6. **Submit PR** with a clear description of the method and validation results

## Adding a New Method

1. Create Python module in `src/causal_inference/<family>/`
2. Create Julia module in `julia/src/<Family>/`
3. Add known-answer tests in `tests/test_<family>/`
4. Add Monte Carlo validation in `tests/validation/monte_carlo/`
5. Add cross-language tests in `tests/validation/cross_language/`
6. Update golden reference if applicable

## Questions?

Open an issue for questions about methodology, implementation approach, or validation strategy.
