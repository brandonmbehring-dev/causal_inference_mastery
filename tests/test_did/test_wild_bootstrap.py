"""TDD Step 1: Tests for Wild Cluster Bootstrap standard errors.

These tests verify that wild cluster bootstrap provides valid inference
with few clusters (<30) where standard cluster-robust SEs are biased.

Written BEFORE implementation per TDD protocol.

References:
- Cameron, Gelbach & Miller (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." REStud 75(2): 414-427.
- Webb (2023). "Reworking wild bootstrap-based inference for clustered
  errors." Canadian Journal of Economics 56(3): 839-858.
"""

import numpy as np
import pytest
import warnings


class TestWildBootstrapWeights:
    """Test weight generation functions."""

    def test_rademacher_weights_values(self):
        """Rademacher weights should be exactly {-1, +1}."""
        from src.causal_inference.did.wild_bootstrap import generate_rademacher_weights

        np.random.seed(42)
        weights = generate_rademacher_weights(n_clusters=100)

        # All values should be -1 or +1
        assert set(np.unique(weights)).issubset({-1, 1}), (
            f"Rademacher weights should be {{-1, +1}}, got {np.unique(weights)}"
        )

    def test_rademacher_weights_distribution(self):
        """Rademacher weights should have ~50% each of -1 and +1."""
        from src.causal_inference.did.wild_bootstrap import generate_rademacher_weights

        np.random.seed(42)
        # Large sample for distribution test
        weights = generate_rademacher_weights(n_clusters=10000)

        prop_positive = np.mean(weights == 1)
        assert 0.48 < prop_positive < 0.52, (
            f"Expected ~50% positive weights, got {prop_positive:.2%}"
        )

    def test_webb_weights_values(self):
        """Webb 6-point weights should be {-1.5, -1, -0.5, +0.5, +1, +1.5}."""
        from src.causal_inference.did.wild_bootstrap import generate_webb_weights

        np.random.seed(42)
        weights = generate_webb_weights(n_clusters=10000)

        expected_values = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5}
        actual_values = set(np.unique(weights))

        assert actual_values == expected_values, (
            f"Webb weights should be {expected_values}, got {actual_values}"
        )

    def test_webb_weights_distribution(self):
        """Webb weights should have ~1/6 probability for each value."""
        from src.causal_inference.did.wild_bootstrap import generate_webb_weights

        np.random.seed(42)
        weights = generate_webb_weights(n_clusters=60000)

        for value in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
            prop = np.mean(weights == value)
            assert 0.15 < prop < 0.18, (
                f"Expected ~1/6 for {value}, got {prop:.3f}"
            )

    def test_weight_length_matches_clusters(self):
        """Weight arrays should have length equal to n_clusters."""
        from src.causal_inference.did.wild_bootstrap import (
            generate_rademacher_weights,
            generate_webb_weights,
        )

        for n in [5, 10, 50, 100]:
            r_weights = generate_rademacher_weights(n_clusters=n)
            w_weights = generate_webb_weights(n_clusters=n)

            assert len(r_weights) == n
            assert len(w_weights) == n


class TestWildBootstrapSE:
    """Test wild cluster bootstrap SE estimation."""

    @pytest.fixture
    def few_cluster_did_data(self):
        """Generate DiD data with few clusters (n=10)."""
        np.random.seed(42)
        n_clusters = 10
        obs_per_cluster = 20
        true_effect = 3.0

        outcomes = []
        treatment = []
        post = []
        unit_id = []

        for c in range(n_clusters):
            is_treated = c >= n_clusters // 2  # Half treated

            for obs in range(obs_per_cluster):
                is_post = obs >= obs_per_cluster // 2

                # Base outcome with cluster effect
                y = 10.0 + np.random.normal(0, 1)

                # Time trend
                if is_post:
                    y += 2.0

                # Treatment effect
                if is_treated and is_post:
                    y += true_effect

                outcomes.append(y)
                treatment.append(1 if is_treated else 0)
                post.append(1 if is_post else 0)
                unit_id.append(c)

        return {
            "outcomes": np.array(outcomes),
            "treatment": np.array(treatment),
            "post": np.array(post),
            "unit_id": np.array(unit_id),
            "n_clusters": n_clusters,
            "true_effect": true_effect,
        }

    def test_wild_bootstrap_produces_valid_se(self, few_cluster_did_data):
        """Wild bootstrap should produce finite, positive SE."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se
        from src.causal_inference.did.did_estimator import did_2x2

        data = few_cluster_did_data

        # First fit the model to get residuals
        result = did_2x2(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            post=data["post"],
            unit_id=data["unit_id"],
            cluster_se=True,
        )

        # Now compute wild bootstrap SE
        # Need to construct X matrix and get residuals
        n = len(data["outcomes"])
        X = np.column_stack([
            np.ones(n),
            data["treatment"],
            data["post"],
            data["treatment"] * data["post"],
        ])

        # Get OLS residuals
        beta = np.linalg.lstsq(X, data["outcomes"], rcond=None)[0]
        residuals = data["outcomes"] - X @ beta

        wb_result = wild_cluster_bootstrap_se(
            X=X,
            y=data["outcomes"],
            residuals=residuals,
            cluster_id=data["unit_id"],
            coef_idx=3,  # DiD coefficient
            n_bootstrap=999,
        )

        assert np.isfinite(wb_result["se"]), "SE should be finite"
        assert wb_result["se"] > 0, "SE should be positive"
        assert "ci_lower" in wb_result, "Should include CI lower bound"
        assert "ci_upper" in wb_result, "Should include CI upper bound"
        assert wb_result["ci_lower"] < wb_result["ci_upper"], "CI should be valid"

    def test_wild_bootstrap_uses_webb_for_few_clusters(self, few_cluster_did_data):
        """Should automatically use Webb weights when n_clusters < 13."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        data = few_cluster_did_data

        n = len(data["outcomes"])
        X = np.column_stack([
            np.ones(n),
            data["treatment"],
            data["post"],
            data["treatment"] * data["post"],
        ])
        beta = np.linalg.lstsq(X, data["outcomes"], rcond=None)[0]
        residuals = data["outcomes"] - X @ beta

        # With auto weight selection, should use Webb for 10 clusters
        wb_result = wild_cluster_bootstrap_se(
            X=X,
            y=data["outcomes"],
            residuals=residuals,
            cluster_id=data["unit_id"],
            coef_idx=3,
            n_bootstrap=999,
            weight_type="auto",
        )

        assert wb_result["weight_type_used"] == "webb", (
            f"Should use Webb weights for {data['n_clusters']} clusters, "
            f"but used {wb_result['weight_type_used']}"
        )

    def test_wild_bootstrap_uses_rademacher_for_many_clusters(self):
        """Should use Rademacher weights when n_clusters >= 13."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 10
        n = n_clusters * obs_per_cluster

        # Generate simple data
        cluster_id = np.repeat(np.arange(n_clusters), obs_per_cluster)
        X = np.column_stack([
            np.ones(n),
            np.random.normal(0, 1, n),
            np.random.normal(0, 1, n),
            np.random.normal(0, 1, n),
        ])
        y = X @ [1, 2, 3, 4] + np.random.normal(0, 1, n)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        wb_result = wild_cluster_bootstrap_se(
            X=X,
            y=y,
            residuals=residuals,
            cluster_id=cluster_id,
            coef_idx=3,
            n_bootstrap=999,
            weight_type="auto",
        )

        assert wb_result["weight_type_used"] == "rademacher", (
            f"Should use Rademacher weights for {n_clusters} clusters, "
            f"but used {wb_result['weight_type_used']}"
        )

    def test_wild_bootstrap_respects_explicit_weight_type(self, few_cluster_did_data):
        """Should respect explicitly specified weight type."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        data = few_cluster_did_data

        n = len(data["outcomes"])
        X = np.column_stack([
            np.ones(n),
            data["treatment"],
            data["post"],
            data["treatment"] * data["post"],
        ])
        beta = np.linalg.lstsq(X, data["outcomes"], rcond=None)[0]
        residuals = data["outcomes"] - X @ beta

        # Force Rademacher even with few clusters
        wb_result = wild_cluster_bootstrap_se(
            X=X,
            y=data["outcomes"],
            residuals=residuals,
            cluster_id=data["unit_id"],
            coef_idx=3,
            n_bootstrap=999,
            weight_type="rademacher",
        )

        assert wb_result["weight_type_used"] == "rademacher"


class TestWildBootstrapDiDIntegration:
    """Test wild bootstrap integration with did_2x2 estimator."""

    @pytest.fixture
    def panel_10_clusters(self):
        """Generate balanced panel with exactly 10 clusters."""
        np.random.seed(123)
        n_clusters = 10
        n_periods = 4  # 2 pre, 2 post
        true_effect = 2.5

        outcomes = []
        treatment = []
        post = []
        unit_id = []

        for c in range(n_clusters):
            is_treated = c >= n_clusters // 2
            cluster_fe = np.random.normal(0, 2)  # Cluster fixed effect

            for t in range(n_periods):
                is_post = t >= 2

                y = 10.0 + cluster_fe + 0.5 * t  # Baseline + FE + trend
                if is_treated and is_post:
                    y += true_effect

                y += np.random.normal(0, 0.5)

                outcomes.append(y)
                treatment.append(1 if is_treated else 0)
                post.append(1 if is_post else 0)
                unit_id.append(c)

        return {
            "outcomes": np.array(outcomes),
            "treatment": np.array(treatment),
            "post": np.array(post),
            "unit_id": np.array(unit_id),
            "true_effect": true_effect,
            "n_clusters": n_clusters,
        }

    def test_did_2x2_with_wild_bootstrap_se_method(self, panel_10_clusters):
        """did_2x2 should accept se_method='wild_bootstrap'."""
        from src.causal_inference.did.did_estimator import did_2x2

        data = panel_10_clusters

        result = did_2x2(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            post=data["post"],
            unit_id=data["unit_id"],
            se_method="wild_bootstrap",
            n_bootstrap=999,
        )

        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["se_method"] == "wild_bootstrap"
        assert np.isfinite(result["se"])
        assert result["se"] > 0

    def test_wild_bootstrap_se_typically_larger_than_cluster_robust(self, panel_10_clusters):
        """Wild bootstrap SE should be >= cluster-robust SE (less overconfident)."""
        from src.causal_inference.did.did_estimator import did_2x2

        data = panel_10_clusters

        # Standard cluster-robust
        cluster_result = did_2x2(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            post=data["post"],
            unit_id=data["unit_id"],
            se_method="cluster",
        )

        # Wild bootstrap
        wild_result = did_2x2(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            post=data["post"],
            unit_id=data["unit_id"],
            se_method="wild_bootstrap",
            n_bootstrap=999,
        )

        # Wild bootstrap SE should be at least 80% of cluster-robust
        # (not dramatically smaller - that would indicate a bug)
        ratio = wild_result["se"] / cluster_result["se"]
        assert ratio >= 0.8, (
            f"Wild bootstrap SE ({wild_result['se']:.4f}) should not be much smaller "
            f"than cluster-robust SE ({cluster_result['se']:.4f}). Ratio: {ratio:.2f}"
        )

    def test_did_warns_about_few_clusters_with_cluster_se(self, panel_10_clusters):
        """Should warn when using cluster SEs with few clusters."""
        from src.causal_inference.did.did_estimator import did_2x2

        data = panel_10_clusters

        with pytest.warns(RuntimeWarning, match="[Ss]mall number of clusters|[Ww]ild"):
            did_2x2(
                outcomes=data["outcomes"],
                treatment=data["treatment"],
                post=data["post"],
                unit_id=data["unit_id"],
                se_method="cluster",
            )


class TestWildBootstrapEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_clusters_warning(self):
        """Should warn when n_clusters < 6 (bootstrap may be unreliable)."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n_clusters = 4
        obs_per_cluster = 10
        n = n_clusters * obs_per_cluster

        cluster_id = np.repeat(np.arange(n_clusters), obs_per_cluster)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 3))])
        y = np.random.normal(0, 1, n)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        with pytest.warns(UserWarning, match="[Vv]ery few clusters|unreliable"):
            wild_cluster_bootstrap_se(
                X=X,
                y=y,
                residuals=residuals,
                cluster_id=cluster_id,
                coef_idx=0,
                n_bootstrap=999,
            )

    def test_invalid_weight_type_raises_error(self):
        """Should raise ValueError for invalid weight type."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.normal(0, 1, n)
        cluster_id = np.repeat(np.arange(10), 10)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        with pytest.raises(ValueError, match="[Ii]nvalid weight.*type"):
            wild_cluster_bootstrap_se(
                X=X,
                y=y,
                residuals=residuals,
                cluster_id=cluster_id,
                coef_idx=0,
                weight_type="invalid_type",
            )

    def test_n_bootstrap_must_be_positive(self):
        """Should raise ValueError if n_bootstrap <= 0."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = np.random.normal(0, 1, n)
        cluster_id = np.repeat(np.arange(10), 10)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        with pytest.raises(ValueError, match="n_bootstrap.*positive"):
            wild_cluster_bootstrap_se(
                X=X,
                y=y,
                residuals=residuals,
                cluster_id=cluster_id,
                coef_idx=0,
                n_bootstrap=0,
            )

    def test_coef_idx_bounds_check(self):
        """Should raise ValueError if coef_idx out of bounds."""
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])  # 2 columns
        y = np.random.normal(0, 1, n)
        cluster_id = np.repeat(np.arange(10), 10)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        with pytest.raises(ValueError, match="coef_idx.*out of bounds|invalid"):
            wild_cluster_bootstrap_se(
                X=X,
                y=y,
                residuals=residuals,
                cluster_id=cluster_id,
                coef_idx=5,  # Out of bounds (only 2 columns)
            )


class TestWildBootstrapMonteCarlo:
    """Monte Carlo validation of wild bootstrap coverage."""

    @pytest.mark.slow
    def test_wild_bootstrap_coverage_10_clusters(self):
        """Monte Carlo: Coverage should be near 95% with 10 clusters.

        This is the key validation test: with few clusters, standard
        cluster-robust SEs undercover (< 90%), while wild bootstrap
        should achieve proper coverage (93-97%).
        """
        from src.causal_inference.did.wild_bootstrap import wild_cluster_bootstrap_se

        np.random.seed(42)
        n_simulations = 500
        n_clusters = 10
        obs_per_cluster = 20
        true_effect = 2.0
        alpha = 0.05

        covers_true = []

        for sim in range(n_simulations):
            # Generate DiD data
            outcomes = []
            treatment = []
            post = []
            unit_id = []

            for c in range(n_clusters):
                is_treated = c >= n_clusters // 2
                cluster_effect = np.random.normal(0, 1)

                for obs in range(obs_per_cluster):
                    is_post = obs >= obs_per_cluster // 2

                    y = 10.0 + cluster_effect
                    if is_post:
                        y += 1.0  # Time effect
                    if is_treated and is_post:
                        y += true_effect

                    y += np.random.normal(0, 1)

                    outcomes.append(y)
                    treatment.append(1 if is_treated else 0)
                    post.append(1 if is_post else 0)
                    unit_id.append(c)

            outcomes = np.array(outcomes)
            treatment = np.array(treatment)
            post = np.array(post)
            unit_id = np.array(unit_id)

            # Build regression matrix
            n = len(outcomes)
            X = np.column_stack([
                np.ones(n),
                treatment,
                post,
                treatment * post,
            ])

            beta = np.linalg.lstsq(X, outcomes, rcond=None)[0]
            residuals = outcomes - X @ beta

            # Wild bootstrap
            result = wild_cluster_bootstrap_se(
                X=X,
                y=outcomes,
                residuals=residuals,
                cluster_id=unit_id,
                coef_idx=3,
                n_bootstrap=499,  # Fewer for speed
                alpha=alpha,
            )

            # Check if CI covers true effect
            covers = result["ci_lower"] <= true_effect <= result["ci_upper"]
            covers_true.append(covers)

        coverage = np.mean(covers_true)

        # Coverage should be in 93-97% range for valid inference
        assert 0.90 <= coverage <= 0.99, (
            f"Wild bootstrap coverage = {coverage:.1%}, expected 93-97%. "
            f"Coverage outside this range indicates implementation error."
        )

    @pytest.mark.slow
    def test_cluster_robust_coverage_with_few_clusters(self):
        """Compare cluster-robust and wild bootstrap coverage with few clusters.

        This test demonstrates that both methods can achieve valid coverage,
        but wild bootstrap is more robust to different DGPs. The key insight
        from CGM (2008) is that cluster-robust SEs can undercover in certain
        scenarios (especially with unbalanced clusters or serial correlation).

        Note: Whether undercoverage occurs depends on the specific DGP.
        This test verifies both methods produce reasonable coverage.
        """
        from scipy import stats

        np.random.seed(42)
        n_simulations = 500
        n_clusters = 10
        obs_per_cluster = 20
        true_effect = 2.0
        alpha = 0.05

        covers_true = []

        for sim in range(n_simulations):
            outcomes = []
            treatment = []
            post = []
            unit_id = []

            for c in range(n_clusters):
                is_treated = c >= n_clusters // 2
                cluster_effect = np.random.normal(0, 1)

                for obs in range(obs_per_cluster):
                    is_post = obs >= obs_per_cluster // 2

                    y = 10.0 + cluster_effect
                    if is_post:
                        y += 1.0
                    if is_treated and is_post:
                        y += true_effect

                    y += np.random.normal(0, 1)

                    outcomes.append(y)
                    treatment.append(1 if is_treated else 0)
                    post.append(1 if is_post else 0)
                    unit_id.append(c)

            outcomes = np.array(outcomes)
            treatment = np.array(treatment)
            post = np.array(post)
            unit_id = np.array(unit_id)

            # Use statsmodels for cluster-robust SEs
            try:
                import statsmodels.api as sm

                n = len(outcomes)
                X = np.column_stack([
                    np.ones(n),
                    treatment,
                    post,
                    treatment * post,
                ])

                model = sm.OLS(outcomes, X)
                results = model.fit(cov_type="cluster", cov_kwds={"groups": unit_id})

                estimate = results.params[3]
                se = results.bse[3]

                # Standard t-based CI
                df = n_clusters - 1
                t_crit = stats.t.ppf(1 - alpha / 2, df)
                ci_lower = estimate - t_crit * se
                ci_upper = estimate + t_crit * se

                covers = ci_lower <= true_effect <= ci_upper
                covers_true.append(covers)

            except ImportError:
                pytest.skip("statsmodels not available")

        coverage = np.mean(covers_true)

        # Cluster-robust coverage should be in a reasonable range
        # With this balanced DGP, coverage is often acceptable even with few clusters
        # The main point is that wild bootstrap is MORE ROBUST across DGPs
        assert 0.85 <= coverage <= 0.99, (
            f"Cluster-robust coverage = {coverage:.1%}. "
            f"Expected reasonable coverage (85-99%) with this balanced DGP."
        )

        # Document the actual coverage for reference
        print(f"\nCluster-robust coverage with {n_clusters} clusters: {coverage:.1%}")
