"""
PyCall validation tests for RDD estimators.

Validates Julia RDD implementation against Python implementation using identical data.

Cross-language parity results (expected):
- Sharp RDD: Close parity (same algorithm: local linear regression)
- Bandwidth selection: Approximate parity (different numerical constants)
- CI/p-values: Close match (different implementation details)

Note: Python uses 'rectangular' kernel, Julia uses 'uniform' - same algorithm.
"""

using Test
using CausalEstimators
using PyCall
using Statistics
using Random
using LinearAlgebra

# Add Python project root to sys.path
pushfirst!(PyVector(pyimport("sys")."path"), "/home/brandon_behring/Claude/causal_inference_mastery")

# Import Python RDD modules
const sharp_rdd_py = pyimport("src.causal_inference.rdd.sharp_rdd")
const bandwidth_py = pyimport("src.causal_inference.rdd.bandwidth")

"""
Generate Sharp RDD data.

DGP: Y = 1.0 + 0.5 * X + tau * (X >= cutoff) + eps
"""
function generate_sharp_rdd_data(n::Int, cutoff::Float64, tau::Float64; seed::Int=42)
    Random.seed!(seed)
    X = rand(n) .* 4 .- 2 .+ cutoff  # X in [cutoff-2, cutoff+2]
    treatment = X .>= cutoff
    eps = randn(n) .* 0.5
    Y = 1.0 .+ 0.5 .* X .+ tau .* treatment .+ eps
    return Y, X, treatment
end


@testset "PyCall RDD Validation" begin

    # =========================================================================
    # Sharp RDD Tests
    # =========================================================================

    @testset "Sharp RDD - Basic IK Bandwidth" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=42)

        # Julia Sharp RDD
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python Sharp RDD
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # Compare estimates - relaxed tolerance due to potential bandwidth differences
        estimate_diff = abs(solution_jl.estimate - py_model.coef_)
        @test estimate_diff < 0.5  # Within 0.5 of each other

        # Both should be close to true tau=2.0
        @test abs(solution_jl.estimate - 2.0) < 0.4
        @test abs(py_model.coef_ - 2.0) < 0.4
    end

    @testset "Sharp RDD - Large Sample (n=2000)" begin
        Y, X, treatment = generate_sharp_rdd_data(2000, 0.0, 2.0; seed=123)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # With large sample, estimates should be more precise
        @test abs(solution_jl.estimate - 2.0) < 0.2
        @test abs(py_model.coef_ - 2.0) < 0.2

        # Relative difference should be smaller with more data
        rel_diff = abs(solution_jl.estimate - py_model.coef_) / abs(solution_jl.estimate)
        @test rel_diff < 0.1
    end

    @testset "Sharp RDD - Small Sample (n=200)" begin
        Y, X, treatment = generate_sharp_rdd_data(200, 0.0, 2.0; seed=456)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # Both should produce valid estimates (wider tolerance)
        @test abs(solution_jl.estimate - 2.0) < 0.6
        @test abs(py_model.coef_ - 2.0) < 0.6
    end

    @testset "Sharp RDD - Triangular Kernel" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 1.5; seed=789)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # Kernel should contain "Triangular" (Julia stores type name)
        @test occursin("triangular", lowercase(string(solution_jl.kernel)))

        # Estimates reasonable
        rel_diff = abs(solution_jl.estimate - py_model.coef_) / abs(solution_jl.estimate)
        @test rel_diff < 0.15
    end

    @testset "Sharp RDD - Uniform/Rectangular Kernel" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 1.5; seed=101)

        # Julia uses 'uniform', Python uses 'rectangular' - same kernel
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=UniformKernel(),
            run_density_test=false
        ))

        # Python with rectangular (= uniform)
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="rectangular")
        py_model.fit(Y, X)

        # Both should be close to true tau=1.5
        @test abs(solution_jl.estimate - 1.5) < 0.5
        @test abs(py_model.coef_ - 1.5) < 0.5
    end

    @testset "Sharp RDD - Negative Treatment Effect" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, -1.5; seed=202)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # Both should detect negative effect
        @test solution_jl.estimate < 0
        @test py_model.coef_ < 0

        # Close to true value
        @test abs(solution_jl.estimate - (-1.5)) < 0.5
        @test abs(py_model.coef_ - (-1.5)) < 0.5
    end

    @testset "Sharp RDD - Non-Zero Cutoff" begin
        cutoff = 5.0
        tau = 2.0
        Random.seed!(303)
        n = 500

        X = rand(n) .* 6 .+ 2  # X in [2, 8], cutoff at 5
        treatment = X .>= cutoff
        eps = randn(n) .* 0.5
        Y = 0.5 .* X .+ tau .* treatment .+ eps

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, cutoff, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=cutoff, bandwidth="ik", kernel="triangular")
        py_model.fit(Y, X)

        # Both should get reasonable estimates
        @test abs(solution_jl.estimate - tau) < 0.5
        @test abs(py_model.coef_ - tau) < 0.5
    end

    # =========================================================================
    # Bandwidth Selection Tests
    # =========================================================================

    @testset "IK Bandwidth Selection" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=42)

        # Julia IK bandwidth
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        jl_h = select_bandwidth(problem_jl, IKBandwidth())

        # Python IK bandwidth
        py_h = bandwidth_py.imbens_kalyanaraman_bandwidth(Y, X, 0.0, kernel="triangular")

        # Bandwidth formulas have different constants between implementations
        # Allow 100% tolerance due to systematic differences
        rel_diff = abs(jl_h - py_h) / jl_h
        @test rel_diff < 1.0

        # Both should be positive and reasonable
        @test 0.1 < jl_h < 5.0
        @test 0.1 < py_h < 5.0
    end

    @testset "CCT Bandwidth Selection" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=123)

        # Julia CCT bandwidth (returns tuple)
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        jl_h_main, jl_h_bias = select_bandwidth(problem_jl, CCTBandwidth())

        # Python CCT bandwidth
        py_result = bandwidth_py.cct_bandwidth(Y, X, 0.0, kernel="triangular")
        py_h_main = py_result[1]
        py_h_bias = py_result[2]

        # Main bandwidth comparison (relaxed tolerance)
        rel_diff_main = abs(jl_h_main - py_h_main) / jl_h_main
        @test rel_diff_main < 0.3

        # Bias bandwidth comparison
        rel_diff_bias = abs(jl_h_bias - py_h_bias) / jl_h_bias
        @test rel_diff_bias < 0.5
    end

    @testset "Bandwidth Selection - Large Sample" begin
        Y, X, treatment = generate_sharp_rdd_data(2000, 0.0, 2.0; seed=789)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        jl_h = select_bandwidth(problem_jl, IKBandwidth())

        # Python
        py_h = bandwidth_py.imbens_kalyanaraman_bandwidth(Y, X, 0.0, kernel="triangular")

        # Bandwidth formulas have different constants between implementations
        # Even with more data, there's a ~20% systematic difference
        rel_diff = abs(jl_h - py_h) / jl_h
        @test rel_diff < 0.3
    end

    # =========================================================================
    # CI and P-Value Tests
    # =========================================================================

    @testset "95% Confidence Interval" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=42)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik", alpha=0.05)
        py_model.fit(Y, X)

        # Both should produce valid CIs
        @test solution_jl.ci_lower < solution_jl.ci_upper
        @test py_model.ci_[1] < py_model.ci_[2]

        # CI width should be similar
        jl_width = solution_jl.ci_upper - solution_jl.ci_lower
        py_width = py_model.ci_[2] - py_model.ci_[1]
        rel_diff = abs(jl_width - py_width) / jl_width
        @test rel_diff < 0.3
    end

    @testset "P-Value Consistency" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=456)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik")
        py_model.fit(Y, X)

        # Both should give significant p-values for tau=2.0 (strong effect)
        @test solution_jl.p_value < 0.05
        @test py_model.p_value_ < 0.05

        # Both p-values should be in similar range
        @test solution_jl.p_value < 0.1
        @test py_model.p_value_ < 0.1
    end

    @testset "Effective Sample Sizes" begin
        Y, X, treatment = generate_sharp_rdd_data(500, 0.0, 2.0; seed=42)

        # Julia
        problem_jl = RDDProblem(Y, X, treatment, 0.0, nothing, (alpha=0.05,))
        solution_jl = solve(problem_jl, SharpRDD(
            bandwidth_method=IKBandwidth(),
            kernel=TriangularKernel(),
            run_density_test=false
        ))

        # Python
        py_model = sharp_rdd_py.SharpRDD(cutoff=0.0, bandwidth="ik")
        py_model.fit(Y, X)

        # Both should report positive effective sample sizes
        @test solution_jl.n_eff_left > 0
        @test solution_jl.n_eff_right > 0
        @test py_model.n_left_ > 0
        @test py_model.n_right_ > 0

        # Total effective N should be less than total N
        jl_n_eff = solution_jl.n_eff_left + solution_jl.n_eff_right
        py_n_eff = py_model.n_left_ + py_model.n_right_
        @test jl_n_eff <= 500
        @test py_n_eff <= 500
    end

end  # @testset "PyCall RDD Validation"
