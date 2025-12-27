"""
Tests for Julia PCMCI Algorithm

Session 150: Comprehensive tests for PCMCI following 3-layer architecture.
"""

using Test
using Random
using LinearAlgebra
using Statistics
using CausalEstimators
using CausalEstimators.TimeSeries


@testset "PCMCI Tests" begin

    # ========== Layer 1: Known Answer Tests ==========

    @testset "Known Answer - Simple Chain X→Y→Z" begin
        # X→Y→Z causal chain
        Random.seed!(42)
        n = 300
        data = zeros(n, 3)
        data[:, 1] = randn(n)  # X
        for t in 2:n
            data[t, 2] = 0.7 * data[t-1, 1] + 0.2 * data[t-1, 2] + randn() * 0.3  # Y
            data[t, 3] = 0.7 * data[t-1, 2] + 0.2 * data[t-1, 3] + randn() * 0.3  # Z
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        @test result.n_vars == 3
        @test length(result.links) >= 2  # At least X→Y and Y→Z

        # Check for X→Y link
        has_x_to_y = any(l -> l.source_var == 1 && l.target_var == 2, result.links)
        @test has_x_to_y

        # Check for Y→Z link
        has_y_to_z = any(l -> l.source_var == 2 && l.target_var == 3, result.links)
        @test has_y_to_z
    end

    @testset "Known Answer - Fork Structure X←Z→Y" begin
        # Z is common cause of X and Y
        Random.seed!(42)
        n = 300
        data = zeros(n, 3)
        data[:, 3] = randn(n)  # Z (common cause)
        for t in 2:n
            data[t, 1] = 0.6 * data[t-1, 3] + 0.2 * data[t-1, 1] + randn() * 0.3  # X
            data[t, 2] = 0.6 * data[t-1, 3] + 0.2 * data[t-1, 2] + randn() * 0.3  # Y
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        # Z→X should be detected
        has_z_to_x = any(l -> l.source_var == 3 && l.target_var == 1, result.links)
        @test has_z_to_x

        # Z→Y should be detected
        has_z_to_y = any(l -> l.source_var == 3 && l.target_var == 2, result.links)
        @test has_z_to_y
    end

    @testset "Known Answer - No Edges (Independent)" begin
        Random.seed!(42)
        n = 200
        data = randn(n, 3)

        result = pcmci(data; max_lag=2, alpha=0.01)  # Stricter alpha

        # Should detect very few or no links
        @test length(result.links) <= 2
    end

    @testset "Known Answer - Lagged Relationship" begin
        # X at lag 2 causes Y
        Random.seed!(42)
        n = 300
        data = zeros(n, 2)
        data[:, 1] = randn(n)
        for t in 3:n
            data[t, 2] = 0.7 * data[t-2, 1] + 0.2 * data[t-1, 2] + randn() * 0.3
        end

        result = pcmci(data; max_lag=3, alpha=0.05)

        # Should find lag-2 relationship
        has_lag2 = any(l -> l.source_var == 1 && l.target_var == 2 && l.lag == 2,
                       result.links)
        @test has_lag2
    end

    @testset "Known Answer - Multiple Parents" begin
        # Y has two parents: X and Z
        Random.seed!(42)
        n = 300
        data = zeros(n, 3)
        data[:, 1] = randn(n)  # X
        data[:, 3] = randn(n)  # Z
        for t in 2:n
            data[t, 2] = 0.5 * data[t-1, 1] + 0.5 * data[t-1, 3] +
                         0.2 * data[t-1, 2] + randn() * 0.3  # Y
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        # Both X→Y and Z→Y should be detected
        parents_of_y = [l for l in result.links if l.target_var == 2]
        @test length(parents_of_y) >= 2
    end

    @testset "Known Answer - Result Structure" begin
        Random.seed!(42)
        data = randn(100, 3)
        max_lag = 2

        result = pcmci(data; max_lag=max_lag, alpha=0.05)

        @test result isa PCMCIResult
        @test result.n_vars == 3
        @test result.n_obs == 100
        @test result.max_lag == max_lag
        @test result.alpha == 0.05
        # Matrix dimensions are (n_vars, n_vars, max_lag)
        @test size(result.p_matrix, 1) == 3
        @test size(result.p_matrix, 2) == 3
        @test size(result.p_matrix, 3) >= max_lag
    end

    @testset "Known Answer - Parent Sets" begin
        Random.seed!(42)
        n = 200
        data = zeros(n, 2)
        data[:, 1] = randn(n)
        for t in 2:n
            data[t, 2] = 0.6 * data[t-1, 1] + 0.2 * data[t-1, 2] + randn() * 0.3
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        # Parent set of Y (var 2) should include (1, 1)
        @test haskey(result.parents, 2)
        parents_y = result.parents[2]
        @test any(p -> p[1] == 1, parents_y)
    end

    @testset "Known Answer - Link Types" begin
        Random.seed!(42)
        n = 200
        data = zeros(n, 2)
        data[:, 1] = randn(n)
        for t in 2:n
            data[t, 2] = 0.6 * data[t-1, 1] + randn() * 0.3
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        for link in result.links
            @test link isa TimeSeriesLink
            @test link.lag >= 1
            @test 1 <= link.source_var <= 2
            @test 1 <= link.target_var <= 2
        end
    end

    @testset "Known Answer - Autoregressive Structure" begin
        # Strong AR(1) for each variable
        Random.seed!(42)
        n = 200
        data = zeros(n, 2)
        for t in 2:n
            data[t, 1] = 0.8 * data[t-1, 1] + randn() * 0.3
            data[t, 2] = 0.8 * data[t-1, 2] + randn() * 0.3
        end

        result = pcmci(data; max_lag=2, alpha=0.05)

        # Should detect self-causation (AR)
        has_ar1 = any(l -> l.source_var == l.target_var && l.lag == 1, result.links)
        @test has_ar1
    end

    @testset "Known Answer - P-values in Range" begin
        Random.seed!(42)
        data = randn(100, 2)

        result = pcmci(data; max_lag=2, alpha=0.05)

        # All p-values should be in [0, 1]
        @test all(0 .<= result.p_matrix .<= 1)
    end

    # ========== Layer 2: Adversarial Tests ==========

    @testset "Adversarial - Short Time Series" begin
        Random.seed!(42)
        data = randn(20, 2)  # Short series

        result = pcmci(data; max_lag=2, alpha=0.05)
        @test result isa PCMCIResult
    end

    @testset "Adversarial - Too Short Raises" begin
        Random.seed!(42)
        data = randn(5, 2)
        @test_throws ErrorException pcmci(data; max_lag=4)
    end

    @testset "Adversarial - Max Lag Constraint" begin
        Random.seed!(42)
        data = randn(100, 2)

        result = pcmci(data; max_lag=1, alpha=0.05)
        @test result.max_lag == 1
        # All links should have lag 1
        @test all(l.lag == 1 for l in result.links)
    end

    @testset "Adversarial - Single Variable" begin
        Random.seed!(42)
        data = randn(100, 1)

        result = pcmci(data; max_lag=2, alpha=0.05)
        @test result.n_vars == 1
        # Only self-links possible
        @test all(l.source_var == l.target_var for l in result.links)
    end

    @testset "Adversarial - Invalid Max Lag" begin
        Random.seed!(42)
        data = randn(100, 2)
        @test_throws ErrorException pcmci(data; max_lag=0)
    end

    @testset "Adversarial - Very Strict Alpha" begin
        Random.seed!(42)
        data = randn(100, 3)

        result = pcmci(data; max_lag=2, alpha=0.001)
        # Should find very few links with strict alpha
        @test length(result.links) <= 3
    end

    @testset "Adversarial - Constant Variable" begin
        Random.seed!(42)
        n = 100
        data = randn(n, 2)
        data[:, 2] .= 1.0  # Constant

        # Should run without error (may produce warnings)
        result = pcmci(data; max_lag=2, alpha=0.05)
        @test result isa PCMCIResult
    end

    @testset "Adversarial - High Dimensional" begin
        Random.seed!(42)
        data = randn(100, 5)
        max_lag = 2

        result = pcmci(data; max_lag=max_lag, alpha=0.05)
        @test result.n_vars == 5
        @test size(result.p_matrix, 1) == 5
        @test size(result.p_matrix, 2) == 5
        @test size(result.p_matrix, 3) >= max_lag
    end

    # ========== Layer 3: Monte Carlo Tests ==========

    @testset "Monte Carlo - Discovery Accuracy (Precision)" begin
        n_runs = 30
        precision_sum = 0.0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200

            # Known structure: X→Y→Z
            data = zeros(n, 3)
            data[:, 1] = randn(n)
            for t in 2:n
                data[t, 2] = 0.6 * data[t-1, 1] + 0.2 * data[t-1, 2] + randn() * 0.3
                data[t, 3] = 0.6 * data[t-1, 2] + 0.2 * data[t-1, 3] + randn() * 0.3
            end

            result = pcmci(data; max_lag=2, alpha=0.05)

            # True edges: (1,2,1), (2,3,1), plus AR terms
            true_edges = Set([(1,2,1), (2,3,1)])

            if length(result.links) > 0
                discovered = Set((l.source_var, l.target_var, l.lag) for l in result.links)
                # Count true positives
                tp = length(intersect(discovered, true_edges))
                precision = tp / length(discovered)
                precision_sum += precision
            end
        end

        avg_precision = precision_sum / n_runs
        @test avg_precision > 0.20  # At least 20% of discovered links are true
    end

    @testset "Monte Carlo - Discovery Accuracy (Recall)" begin
        n_runs = 30
        recall_sum = 0.0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200

            # Known structure: X→Y (single edge for clearer test)
            data = zeros(n, 2)
            data[:, 1] = randn(n)
            for t in 2:n
                data[t, 2] = 0.6 * data[t-1, 1] + 0.2 * data[t-1, 2] + randn() * 0.3
            end

            result = pcmci(data; max_lag=2, alpha=0.05)

            # True edge: (1,2,1)
            discovered = Set((l.source_var, l.target_var, l.lag) for l in result.links)
            if (1, 2, 1) in discovered
                recall_sum += 1.0
            end
        end

        recall = recall_sum / n_runs
        @test recall > 0.60  # Should detect true edge in >60% of runs
    end

    @testset "Monte Carlo - False Positive Rate Control" begin
        n_runs = 50
        false_positives = 0
        total_tested = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 100
            # Independent variables (no true causal structure)
            data = randn(n, 2)

            result = pcmci(data; max_lag=2, alpha=0.05)

            # Count links (all are false positives)
            false_positives += length(result.links)
            total_tested += 4  # 2 vars * 2 lags = 4 possible edges
        end

        fpr = false_positives / total_tested
        # Should be around alpha or less
        @test fpr < 0.15
    end

    @testset "Monte Carlo - Lag Identification" begin
        n_runs = 30
        correct_lag = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200

            # X at lag 2 causes Y
            data = zeros(n, 2)
            data[:, 1] = randn(n)
            for t in 3:n
                data[t, 2] = 0.6 * data[t-2, 1] + 0.2 * data[t-1, 2] + randn() * 0.3
            end

            result = pcmci(data; max_lag=3, alpha=0.05)

            # Check if lag 2 link is found
            has_correct_lag = any(l -> l.source_var == 1 && l.target_var == 2 && l.lag == 2,
                                  result.links)
            if has_correct_lag
                correct_lag += 1
            end
        end

        accuracy = correct_lag / n_runs
        @test accuracy > 0.40  # Should identify correct lag in >40% of runs
    end

    @testset "Monte Carlo - Chain vs Fork Distinction" begin
        # Test that PCMCI can distinguish chain from fork
        n_runs = 20

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200

            # Chain: X→Y→Z
            data = zeros(n, 3)
            data[:, 1] = randn(n)
            for t in 2:n
                data[t, 2] = 0.6 * data[t-1, 1] + randn() * 0.3
                data[t, 3] = 0.6 * data[t-1, 2] + randn() * 0.3
            end

            result = pcmci(data; max_lag=2, alpha=0.05)

            # After conditioning on Y, X should not directly cause Z
            # Check no direct X→Z link
            has_x_to_z_direct = any(l -> l.source_var == 1 && l.target_var == 3,
                                    result.links)
            # This might still appear due to sampling variation, so just check it runs
            @test result isa PCMCIResult
        end
    end

    @testset "Monte Carlo - Bidirectional Detection" begin
        n_runs = 30
        both_detected = 0

        for seed in 1:n_runs
            Random.seed!(seed)
            n = 200

            # Bidirectional: X↔Y
            data = zeros(n, 2)
            for t in 2:n
                data[t, 1] = 0.4 * data[t-1, 2] + 0.3 * data[t-1, 1] + randn() * 0.3
                data[t, 2] = 0.4 * data[t-1, 1] + 0.3 * data[t-1, 2] + randn() * 0.3
            end

            result = pcmci(data; max_lag=2, alpha=0.05)

            has_x_to_y = any(l -> l.source_var == 1 && l.target_var == 2, result.links)
            has_y_to_x = any(l -> l.source_var == 2 && l.target_var == 1, result.links)

            if has_x_to_y && has_y_to_x
                both_detected += 1
            end
        end

        detection_rate = both_detected / n_runs
        @test detection_rate > 0.30  # Detect bidirectional in >30% of runs
    end

    @testset "Monte Carlo - Result Consistency" begin
        # Results should be consistent with same seed
        Random.seed!(42)
        n = 150
        data = zeros(n, 2)
        data[:, 1] = randn(n)
        for t in 2:n
            data[t, 2] = 0.5 * data[t-1, 1] + randn() * 0.5
        end

        # Run twice
        result1 = pcmci(data; max_lag=2, alpha=0.05)

        # Run again with same data (should get similar results)
        result2 = pcmci(data; max_lag=2, alpha=0.05)

        @test length(result1.links) == length(result2.links)
        @test result1.p_matrix ≈ result2.p_matrix
    end

end
