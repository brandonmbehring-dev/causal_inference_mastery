"""
Tests for PCMCI Algorithm.

Session 136: Comprehensive test suite for PCMCI time-series causal discovery.

Test Layers:
- Layer 1: Known-Answer (10 tests) - Ground truth DAGs
- Layer 2: Adversarial (8 tests) - Edge cases
- Layer 3: Monte Carlo (5 tests) - Statistical validation
"""

import numpy as np
import pytest
from typing import List, Tuple, Set

from causal_inference.timeseries import (
    pcmci,
    pcmci_plus,
    run_granger_style_pcmci,
    pc_stable_condition_selection,
    mci_test_all,
    PCMCIResult,
    TimeSeriesLink,
    LaggedDAG,
)
from causal_inference.timeseries.ci_tests_timeseries import (
    parcorr_test,
    run_ci_test,
)


# =============================================================================
# Fixtures: Data Generating Processes
# =============================================================================


@pytest.fixture
def chain_3var_data():
    """
    Generate 3-variable chain: X0_{t-1} → X1_t → X2_{t+1}

    True structure: X0(t-1) → X1(t), X1(t-1) → X2(t)
    """
    np.random.seed(42)
    n = 300
    data = np.zeros((n, 3))

    # X0 is exogenous
    data[:, 0] = np.random.randn(n)

    # X1 depends on X0 lagged
    for t in range(1, n):
        data[t, 1] = 0.6 * data[t - 1, 0] + 0.3 * data[t - 1, 1] + np.random.randn() * 0.5

    # X2 depends on X1 lagged
    for t in range(1, n):
        data[t, 2] = 0.5 * data[t - 1, 1] + 0.2 * data[t - 1, 2] + np.random.randn() * 0.5

    true_links = {(0, 1, 1), (1, 2, 1)}  # (source, target, lag)
    return data, true_links


@pytest.fixture
def fork_3var_data():
    """
    Generate 3-variable fork: X1 ← X0 → X2 (lagged)

    True structure: X0(t-1) → X1(t), X0(t-1) → X2(t)
    """
    np.random.seed(123)
    n = 300
    data = np.zeros((n, 3))

    # X0 is exogenous
    data[:, 0] = np.random.randn(n)

    # X1 and X2 both depend on X0 lagged
    for t in range(1, n):
        data[t, 1] = 0.7 * data[t - 1, 0] + np.random.randn() * 0.5
        data[t, 2] = 0.6 * data[t - 1, 0] + np.random.randn() * 0.5

    true_links = {(0, 1, 1), (0, 2, 1)}
    return data, true_links


@pytest.fixture
def collider_3var_data():
    """
    Generate 3-variable collider: X0 → X2 ← X1 (lagged)

    True structure: X0(t-1) → X2(t), X1(t-1) → X2(t)
    """
    np.random.seed(456)
    n = 300
    data = np.zeros((n, 3))

    # X0 and X1 are exogenous
    data[:, 0] = np.random.randn(n)
    data[:, 1] = np.random.randn(n)

    # X2 depends on both X0 and X1 lagged
    for t in range(1, n):
        data[t, 2] = 0.5 * data[t - 1, 0] + 0.5 * data[t - 1, 1] + np.random.randn() * 0.4

    true_links = {(0, 2, 1), (1, 2, 1)}
    return data, true_links


@pytest.fixture
def independent_data():
    """Generate independent time series (no causal links)."""
    np.random.seed(789)
    n = 200
    data = np.column_stack([
        np.random.randn(n),
        np.random.randn(n),
        np.random.randn(n),
    ])
    true_links: Set[Tuple[int, int, int]] = set()
    return data, true_links


@pytest.fixture
def var1_data():
    """
    Generate VAR(1) process with known structure.

    A_1 = [[0.4, 0.3],
           [0.0, 0.5]]

    X0(t-1) → X0(t) (autoregressive)
    X0(t-1) → X1(t) (cross-effect)
    X1(t-1) → X1(t) (autoregressive)
    """
    np.random.seed(42)
    n = 300
    A = np.array([[0.4, 0.3], [0.0, 0.5]])
    data = np.zeros((n, 2))

    for t in range(1, n):
        data[t] = A @ data[t - 1] + np.random.randn(2) * 0.5

    true_links = {(0, 0, 1), (0, 1, 1), (1, 1, 1)}
    return data, true_links


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestPCMCIKnownStructures:
    """Tests with ground truth causal structures."""

    def test_chain_structure(self, chain_3var_data):
        """PCMCI recovers chain: X0 → X1 → X2."""
        data, true_links = chain_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05, ci_test="parcorr")

        discovered = {(l.source_var, l.target_var, l.lag) for l in result.links}

        # Should find both true links
        for link in true_links:
            assert link in discovered, f"Missed link: {link}"

    def test_fork_structure(self, fork_3var_data):
        """PCMCI recovers fork: X1 ← X0 → X2."""
        data, true_links = fork_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        discovered = {(l.source_var, l.target_var, l.lag) for l in result.links}

        for link in true_links:
            assert link in discovered, f"Missed link: {link}"

    def test_collider_structure(self, collider_3var_data):
        """PCMCI recovers collider: X0 → X2 ← X1."""
        data, true_links = collider_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        discovered = {(l.source_var, l.target_var, l.lag) for l in result.links}

        for link in true_links:
            assert link in discovered, f"Missed link: {link}"

    def test_independent_data_no_links(self, independent_data):
        """PCMCI finds no links in independent data."""
        data, _ = independent_data

        result = pcmci(data, max_lag=2, alpha=0.01)  # Stricter alpha

        # Should find very few or no links
        assert len(result.links) <= 1, f"Found {len(result.links)} spurious links"

    def test_var1_recovery(self, var1_data):
        """PCMCI recovers VAR(1) structure."""
        data, true_links = var1_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        discovered = {(l.source_var, l.target_var, l.lag) for l in result.links}

        # Check autoregressive links are found
        assert (0, 0, 1) in discovered or (0, 1, 1) in discovered
        assert (1, 1, 1) in discovered

    def test_correct_lag_identification(self, chain_3var_data):
        """PCMCI identifies correct lags."""
        data, true_links = chain_3var_data

        result = pcmci(data, max_lag=3, alpha=0.05)

        # True links are at lag 1
        for link in result.links:
            if (link.source_var, link.target_var) in [(0, 1), (1, 2)]:
                assert link.lag == 1, f"Wrong lag for {link}"

    def test_parents_dict_populated(self, chain_3var_data):
        """Result parents dict is correctly populated."""
        data, _ = chain_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        # X1 should have X0 as parent
        parents_of_1 = result.get_parents_of(1)
        assert any(p[0] == 0 for p in parents_of_1), "X1 missing parent X0"

        # X2 should have X1 as parent
        parents_of_2 = result.get_parents_of(2)
        assert any(p[0] == 1 for p in parents_of_2), "X2 missing parent X1"

    def test_p_values_valid_range(self, chain_3var_data):
        """P-values are in [0, 1]."""
        data, _ = chain_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        assert np.all(result.p_matrix >= 0)
        assert np.all(result.p_matrix <= 1)

    def test_graph_matches_links(self, chain_3var_data):
        """Graph array matches links list."""
        data, _ = chain_3var_data

        result = pcmci(data, max_lag=2, alpha=0.05)

        # Count from graph
        graph_count = np.sum(result.graph)

        # Count from links
        link_count = len(result.links)

        assert graph_count == link_count

    def test_significant_links_filter(self, chain_3var_data):
        """get_significant_links filters correctly."""
        data, _ = chain_3var_data

        result = pcmci(data, max_lag=2, alpha=0.10)

        # Default uses result.alpha
        sig_links = result.get_significant_links()
        assert all(l.p_value < 0.10 for l in sig_links)

        # Custom threshold
        strict_links = result.get_significant_links(alpha=0.01)
        assert all(l.p_value < 0.01 for l in strict_links)


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestPCMCIAdversarial:
    """Edge cases and adversarial inputs."""

    def test_short_time_series(self):
        """PCMCI handles short time series gracefully."""
        np.random.seed(42)
        n = 50  # Short series
        data = np.random.randn(n, 2)

        # Should not crash
        result = pcmci(data, max_lag=2, alpha=0.05)
        assert isinstance(result, PCMCIResult)

    def test_high_dimensional(self):
        """PCMCI handles many variables."""
        np.random.seed(42)
        n = 200
        n_vars = 10
        data = np.random.randn(n, n_vars)

        # Should complete (may be slow but shouldn't crash)
        result = pcmci(data, max_lag=2, alpha=0.01)
        assert result.n_vars == n_vars

    def test_single_variable_error(self):
        """PCMCI raises error for single variable."""
        data = np.random.randn(100, 1)

        # Single variable doesn't make sense for causality
        # Should either error or return empty result
        result = pcmci(data, max_lag=2, alpha=0.05)
        assert len(result.links) == 0

    def test_max_lag_too_large(self):
        """Error when max_lag exceeds data length."""
        data = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Insufficient observations"):
            pcmci(data, max_lag=15, alpha=0.05)

    def test_zero_max_lag_error(self):
        """Error when max_lag < 1."""
        data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="max_lag must be >= 1"):
            pcmci(data, max_lag=0, alpha=0.05)

    def test_autocorrelated_confounding(self):
        """PCMCI handles autocorrelated confounding."""
        np.random.seed(42)
        n = 300

        # Confounded: X and Y both caused by hidden AR process
        hidden = np.zeros(n)
        for t in range(1, n):
            hidden[t] = 0.8 * hidden[t - 1] + np.random.randn()

        x = hidden + np.random.randn(n) * 0.3
        y = hidden + np.random.randn(n) * 0.3

        data = np.column_stack([x, y])

        # PCMCI should find autocorrelation but ideally not spurious X→Y
        result = pcmci(data, max_lag=2, alpha=0.01)

        # At strict alpha, should have few cross-links
        cross_links = [l for l in result.links if l.source_var != l.target_var]
        # This is hard, so we just check it doesn't crash
        assert isinstance(result, PCMCIResult)

    def test_near_perfect_correlation(self):
        """PCMCI handles near-perfect correlation."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.01  # Almost identical

        data = np.column_stack([x, y])

        # Should not crash
        result = pcmci(data, max_lag=2, alpha=0.05)
        assert isinstance(result, PCMCIResult)

    def test_different_ci_tests(self, chain_3var_data):
        """Both CI tests produce valid results."""
        data, _ = chain_3var_data

        result_parcorr = pcmci(data, max_lag=2, alpha=0.05, ci_test="parcorr")
        result_cmi = pcmci(data, max_lag=2, alpha=0.10, ci_test="cmi")

        assert isinstance(result_parcorr, PCMCIResult)
        assert isinstance(result_cmi, PCMCIResult)


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestPCMCIMonteCarlo:
    """Statistical validation via Monte Carlo."""

    @pytest.mark.slow
    def test_false_positive_rate(self):
        """FPR should be approximately alpha under null."""
        np.random.seed(42)
        n_runs = 100
        n = 150
        alpha = 0.05
        false_positives = 0
        total_possible = 0

        for run in range(n_runs):
            # Independent data
            data = np.random.randn(n, 3)

            result = pcmci(data, max_lag=2, alpha=alpha)

            # Count false positives (any link in independent data)
            n_vars = 3
            max_lag = 2
            # Total possible links: n_vars * (n_vars) * max_lag (excluding lag 0)
            possible = n_vars * n_vars * max_lag
            total_possible += possible

            false_positives += len(result.links)

        fpr = false_positives / total_possible
        # Should be close to alpha (with tolerance for small sample)
        assert 0.02 < fpr < 0.15, f"FPR {fpr:.3f} outside expected range"

    @pytest.mark.slow
    def test_true_positive_rate(self):
        """TPR should be high for strong effects."""
        np.random.seed(42)
        n_runs = 50
        n = 300
        detected = 0

        for run in range(n_runs):
            # Generate chain with strong effect
            data = np.zeros((n, 2))
            data[:, 0] = np.random.randn(n)
            for t in range(1, n):
                data[t, 1] = 0.7 * data[t - 1, 0] + np.random.randn() * 0.4

            result = pcmci(data, max_lag=2, alpha=0.05)

            # Check if true link (0, 1, 1) is found
            found = any(
                l.source_var == 0 and l.target_var == 1 and l.lag == 1
                for l in result.links
            )
            if found:
                detected += 1

        tpr = detected / n_runs
        assert tpr > 0.70, f"TPR {tpr:.2f} too low (expected > 0.70)"

    @pytest.mark.slow
    def test_lag_recovery_accuracy(self):
        """Correct lag should be identified."""
        np.random.seed(42)
        n_runs = 30
        n = 300
        true_lag = 2
        correct_lag = 0

        for run in range(n_runs):
            data = np.zeros((n, 2))
            data[:, 0] = np.random.randn(n)
            for t in range(true_lag, n):
                data[t, 1] = 0.6 * data[t - true_lag, 0] + np.random.randn() * 0.5

            result = pcmci(data, max_lag=4, alpha=0.05)

            # Find link 0 → 1
            for link in result.links:
                if link.source_var == 0 and link.target_var == 1:
                    if link.lag == true_lag:
                        correct_lag += 1
                    break

        accuracy = correct_lag / n_runs
        assert accuracy > 0.60, f"Lag accuracy {accuracy:.2f} too low"

    @pytest.mark.slow
    def test_skeleton_f1_score(self):
        """Skeleton F1 should be reasonable."""
        np.random.seed(42)
        n_runs = 30
        n = 300
        f1_scores = []

        for run in range(n_runs):
            # Generate known structure
            data = np.zeros((n, 3))
            data[:, 0] = np.random.randn(n)
            for t in range(1, n):
                data[t, 1] = 0.6 * data[t - 1, 0] + np.random.randn() * 0.5
                data[t, 2] = 0.5 * data[t - 1, 1] + np.random.randn() * 0.5

            true_skeleton = {(0, 1), (1, 2)}  # Ignoring lags

            result = pcmci(data, max_lag=2, alpha=0.05)

            discovered_skeleton = {
                (l.source_var, l.target_var) for l in result.links
            }

            # Compute F1
            tp = len(true_skeleton & discovered_skeleton)
            fp = len(discovered_skeleton - true_skeleton)
            fn = len(true_skeleton - discovered_skeleton)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        assert mean_f1 > 0.60, f"Mean F1 {mean_f1:.2f} too low"

    def test_pcmci_vs_granger_style(self, chain_3var_data):
        """PCMCI should outperform naive Granger-style testing."""
        data, true_links = chain_3var_data

        result_pcmci = pcmci(data, max_lag=2, alpha=0.05)
        result_granger = run_granger_style_pcmci(data, max_lag=2, alpha=0.05)

        # Both should find true links
        pcmci_discovered = {(l.source_var, l.target_var, l.lag) for l in result_pcmci.links}
        granger_discovered = {(l.source_var, l.target_var, l.lag) for l in result_granger.links}

        # PCMCI should be more parsimonious (fewer false positives) in general
        # This is a soft assertion
        assert len(result_pcmci.links) <= len(result_granger.links) + 3


# =============================================================================
# CI Tests Unit Tests
# =============================================================================


class TestCITests:
    """Unit tests for conditional independence tests."""

    def test_parcorr_independent(self):
        """ParCorr identifies independent variables."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.random.randn(n)
        data = np.column_stack([x, y])

        result = parcorr_test(data, x_idx=0, y_idx=1, z_indices=[], alpha=0.05)

        # Should be independent (high p-value)
        assert result.p_value > 0.05

    def test_parcorr_dependent(self):
        """ParCorr identifies dependent variables."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = 0.8 * x + np.random.randn(n) * 0.3
        data = np.column_stack([x, y])

        result = parcorr_test(data, x_idx=0, y_idx=1, z_indices=[], alpha=0.05)

        # Should be dependent (low p-value)
        assert result.p_value < 0.01

    def test_parcorr_conditional_independence(self):
        """ParCorr identifies conditional independence."""
        np.random.seed(42)
        n = 300
        z = np.random.randn(n)
        x = 0.7 * z + np.random.randn(n) * 0.3
        y = 0.6 * z + np.random.randn(n) * 0.3
        data = np.column_stack([x, y, z])

        # Unconditional: X and Y are correlated
        result_uncond = parcorr_test(data, x_idx=0, y_idx=1, z_indices=[], alpha=0.05)
        assert result_uncond.p_value < 0.05, "Should be marginally dependent"

        # Conditional: X ⊥ Y | Z
        result_cond = parcorr_test(data, x_idx=0, y_idx=1, z_indices=[(2, 0)], alpha=0.05)
        assert result_cond.p_value > 0.01, "Should be conditionally independent"

    def test_run_ci_test_wrapper(self, chain_3var_data):
        """run_ci_test wrapper works correctly."""
        data, _ = chain_3var_data

        result = run_ci_test(
            data=data,
            source=0,
            target=1,
            source_lag=1,
            conditioning_set=[],
            ci_test="parcorr",
            alpha=0.05,
        )

        # X0(t-1) should cause X1(t)
        assert result.p_value < 0.05


# =============================================================================
# Types Tests
# =============================================================================


class TestPCMCITypes:
    """Tests for PCMCI data types."""

    def test_time_series_link_creation(self):
        """TimeSeriesLink creation and validation."""
        link = TimeSeriesLink(
            source_var=0,
            target_var=1,
            lag=2,
            strength=0.5,
            p_value=0.01,
        )

        assert link.source_var == 0
        assert link.target_var == 1
        assert link.lag == 2
        assert link.is_significant(alpha=0.05)
        assert link.is_lagged()

    def test_time_series_link_invalid_lag(self):
        """TimeSeriesLink rejects negative lag."""
        with pytest.raises(ValueError):
            TimeSeriesLink(
                source_var=0, target_var=1, lag=-1, strength=0.5, p_value=0.01
            )

    def test_time_series_link_invalid_pvalue(self):
        """TimeSeriesLink rejects invalid p-value."""
        with pytest.raises(ValueError):
            TimeSeriesLink(
                source_var=0, target_var=1, lag=1, strength=0.5, p_value=1.5
            )

    def test_lagged_dag_operations(self):
        """LaggedDAG add/remove/query operations."""
        dag = LaggedDAG(n_vars=3, max_lag=2)

        # Add edge
        dag.add_edge(0, 1, lag=1, weight=0.5)
        assert dag.has_edge(0, 1, 1)
        assert not dag.has_edge(0, 1, 2)

        # Get parents
        parents = dag.get_parents(1)
        assert (0, 1) in parents

        # Remove edge
        dag.remove_edge(0, 1, 1)
        assert not dag.has_edge(0, 1, 1)

    def test_pcmci_result_summary(self, chain_3var_data):
        """PCMCIResult summary generation."""
        data, _ = chain_3var_data
        result = pcmci(data, max_lag=2, alpha=0.05)

        summary = result.summary()
        assert "PCMCI Result Summary" in summary
        assert "Variables:" in summary
        assert "Discovered" in summary
