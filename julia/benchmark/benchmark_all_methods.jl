"""
Comprehensive cross-language benchmark suite for CausalEstimators.jl

Session 132: Extended benchmark coverage for Python vs Julia comparison.

Benchmarks all method families across multiple sample sizes:
- RCT (5 estimators)
- IV (TSLS, LIML, GMM)
- DiD (Classic, Event Study)
- RDD (Sharp, Fuzzy)
- Observational (IPW, DR)
- CATE (S/T/X/R-learner, DML)
- SCM (Synthetic Control)
- Sensitivity (E-Value, Rosenbaum)
- Bounds (Manski, Lee)
- QTE (Unconditional)
- Principal Stratification (CACE)

Usage:
    cd julia
    julia --project=. benchmark/benchmark_all_methods.jl
"""

using BenchmarkTools
using CausalEstimators
using Random
using Statistics
using LinearAlgebra
using Printf
using Dates

# ============================================================================
# Data Generation Utilities
# ============================================================================

"""Generate RCT data for simple_ate benchmark."""
function generate_rct_data(n::Int; effect_size::Float64=0.5, seed::Int=42)
    Random.seed!(seed)
    n_treated = div(n, 2)
    n_control = n - n_treated

    outcomes = vcat(
        randn(n_treated) .+ effect_size,
        randn(n_control)
    )
    treatment = vcat(fill(true, n_treated), fill(false, n_control))

    return outcomes, treatment
end

"""Generate stratified RCT data."""
function generate_stratified_data(n::Int; n_strata::Int=5, seed::Int=42)
    Random.seed!(seed)
    strata = repeat(1:n_strata, outer=div(n, n_strata))
    strata = vcat(strata, fill(n_strata, n - length(strata)))

    outcomes = randn(n)
    treatment = rand(Bool, n)

    return outcomes, treatment, strata
end

"""Generate regression RCT data with covariates."""
function generate_regression_data(n::Int; seed::Int=42)
    Random.seed!(seed)
    covariate = randn(n)
    treatment = rand(Bool, n)
    outcomes = 0.5 .* treatment .+ 0.3 .* covariate .+ randn(n) .* 0.5

    return outcomes, treatment, reshape(covariate, n, 1)
end

"""Generate IV data with instrument."""
function generate_iv_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    # Instrument (binary for simplicity)
    Z = rand(n) .> 0.5

    # Endogenous treatment with correlation to Z
    D = 0.5 .* Z .+ randn(n) .* 0.3 .> 0.3

    # Outcome with endogeneity
    U = randn(n)
    Y = 2.0 .* D .+ 0.5 .* U .+ randn(n)

    # Controls (optional)
    X = reshape(randn(n), n, 1)

    return Y, Float64.(D), Float64.(Z), X
end

"""Generate DiD data (2x2 panel)."""
function generate_did_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    n_units = div(n, 2)

    # Unit IDs
    unit_id = vcat(1:n_units, 1:n_units)

    # Time periods (0 = pre, 1 = post)
    time = vcat(fill(0, n_units), fill(1, n_units))

    # Treatment (first half of units treated)
    n_treated_units = div(n_units, 2)
    treatment = vcat(
        fill(false, n_treated_units), fill(false, n_units - n_treated_units),
        fill(true, n_treated_units), fill(false, n_units - n_treated_units)
    )

    # Post indicator
    post = time .== 1

    # Outcomes with treatment effect = 2.0
    outcomes = randn(n) .+ 2.0 .* treatment .* post

    return outcomes, treatment, post, unit_id
end

"""Generate RDD data near cutoff."""
function generate_rdd_data(n::Int; cutoff::Float64=0.0, bandwidth::Float64=1.0, seed::Int=42)
    Random.seed!(seed)

    # Running variable centered around cutoff
    running_var = cutoff .+ bandwidth .* (rand(n) .- 0.5) .* 2

    # Sharp RDD: treatment = running_var >= cutoff
    treatment = running_var .>= cutoff

    # Outcome with jump at cutoff
    outcomes = 0.5 .* running_var .+ 2.0 .* treatment .+ randn(n) .* 0.5

    return outcomes, treatment, running_var
end

"""Generate observational data with confounding."""
function generate_observational_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    # Covariates
    X = randn(n, 2)

    # Treatment depends on X (confounding)
    propensity = 1.0 ./ (1.0 .+ exp.(-(0.5 .* X[:, 1] .+ 0.3 .* X[:, 2])))
    treatment = rand(n) .< propensity

    # Outcome depends on treatment and X
    outcomes = 2.0 .* treatment .+ 0.5 .* X[:, 1] .+ 0.3 .* X[:, 2] .+ randn(n)

    return outcomes, treatment, X, propensity
end

"""Generate CATE data with heterogeneous effects."""
function generate_cate_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    X = randn(n, 3)
    treatment = rand(Bool, n)

    # Heterogeneous effect: tau(x) = 1.0 + 0.5*x1
    tau = 1.0 .+ 0.5 .* X[:, 1]
    outcomes = tau .* treatment .+ 0.5 .* X[:, 2] .+ randn(n)

    return outcomes, treatment, X
end

"""Generate SCM data (panel with one treated unit)."""
function generate_scm_data(n_units::Int, n_periods::Int; treatment_period::Int=div(n_periods, 2), seed::Int=42)
    Random.seed!(seed)

    # Panel outcomes: units × periods
    outcomes = randn(n_units, n_periods)

    # Add common trend
    trend = cumsum(randn(n_periods) .* 0.1)
    for t in 1:n_periods
        outcomes[:, t] .+= trend[t]
    end

    # Add treatment effect to unit 1 post-treatment
    for t in treatment_period:n_periods
        outcomes[1, t] += 2.0
    end

    return outcomes, treatment_period
end

"""Generate sensitivity analysis data."""
function generate_sensitivity_data(n::Int; seed::Int=42)
    outcomes, treatment = generate_rct_data(n; seed=seed)

    # Match pairs (simple: sort by outcome, pair adjacent)
    sorted_idx = sortperm(outcomes)
    n_pairs = div(n, 2)
    paired_diff = [outcomes[sorted_idx[2i]] - outcomes[sorted_idx[2i-1]] for i in 1:n_pairs]

    return paired_diff, 2.0, 0.5  # effect estimate, SE
end

"""Generate QTE data."""
function generate_qte_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    treatment = rand(Bool, n)
    X = randn(n, 2)

    # Heterogeneous effects in quantiles
    outcomes = randn(n) .+ 2.0 .* treatment .+ 0.5 .* X[:, 1]

    return outcomes, Float64.(treatment), X
end

"""Generate bounds data with missing outcomes."""
function generate_bounds_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    treatment = rand(Bool, n)
    outcomes = randn(n) .+ 2.0 .* treatment

    # Selection (some outcomes missing)
    selection = rand(n) .< (0.8 .+ 0.1 .* treatment)

    return outcomes, treatment, selection
end

"""Generate principal stratification data (encouragement design)."""
function generate_ps_data(n::Int; seed::Int=42)
    Random.seed!(seed)

    # Assignment (encouragement)
    Z = rand(Bool, n)

    # Compliance depends on assignment
    compliance_prob = 0.3 .+ 0.5 .* Z
    D = rand(n) .< compliance_prob

    # Outcome
    Y = 2.0 .* D .+ randn(n)

    return Y, Float64.(Z), Float64.(D)
end

# ============================================================================
# Benchmark Functions by Method Family
# ============================================================================

"""Benchmark RCT estimators."""
function benchmark_rct(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("RCT Estimators")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment = generate_rct_data(n)
        problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

        # SimpleATE
        b = @benchmark solve($problem, SimpleATE()) samples=20 evals=1
        push!(results, (family="rct", method="simple_ate", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Regression ATE
        outcomes_reg, treatment_reg, covs = generate_regression_data(n)
        problem_reg = RCTProblem(outcomes_reg, treatment_reg, covs, nothing, (alpha=0.05,))
        b = @benchmark solve($problem_reg, RegressionATE()) samples=20 evals=1
        push!(results, (family="rct", method="regression_ate", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark IV estimators."""
function benchmark_iv(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("IV Estimators (TSLS, LIML, GMM)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        Y, D, Z, X = generate_iv_data(n)
        problem = IVProblem(Y, reshape(D, :, 1), reshape(Z, :, 1), X)

        # TSLS
        b = @benchmark solve($problem, TSLS()) samples=20 evals=1
        push!(results, (family="iv", method="tsls", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # LIML
        b = @benchmark solve($problem, LIML()) samples=20 evals=1
        push!(results, (family="iv", method="liml", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # GMM
        b = @benchmark solve($problem, GMM()) samples=20 evals=1
        push!(results, (family="iv", method="gmm", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark DiD estimators."""
function benchmark_did(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("DiD Estimators (Classic, Event Study)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, post, unit_id = generate_did_data(n)
        problem = DiDProblem(outcomes, treatment, post, unit_id, nothing)

        # Classic DiD
        b = @benchmark solve($problem, ClassicDiD()) samples=20 evals=1
        push!(results, (family="did", method="classic_did", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark RDD estimators."""
function benchmark_rdd(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("RDD Estimators (Sharp, Fuzzy)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, running_var = generate_rdd_data(n)
        problem = RDDProblem(outcomes, running_var, 0.0, nothing,
                            (bandwidth=1.0, kernel=:triangular, alpha=0.05))

        # Sharp RDD
        b = @benchmark solve($problem, SharpRDD()) samples=20 evals=1
        push!(results, (family="rdd", method="sharp_rdd", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Fuzzy RDD
        problem_fuzzy = RDDProblem(outcomes, running_var, 0.0, Float64.(treatment),
                                   (bandwidth=1.0, kernel=:triangular, alpha=0.05))
        b = @benchmark solve($problem_fuzzy, FuzzyRDD()) samples=20 evals=1
        push!(results, (family="rdd", method="fuzzy_rdd", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark Observational estimators."""
function benchmark_observational(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("Observational Estimators (IPW, DR)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, X, _ = generate_observational_data(n)
        problem = ObservationalProblem(outcomes, Float64.(treatment), X,
                                       (alpha=0.05, trim_threshold=0.01))

        # IPW
        b = @benchmark solve($problem, ObservationalIPW()) samples=20 evals=1
        push!(results, (family="observational", method="ipw_ate", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Doubly Robust
        b = @benchmark solve($problem, DoublyRobust()) samples=20 evals=1
        push!(results, (family="observational", method="dr_ate", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark CATE estimators."""
function benchmark_cate(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("CATE Estimators (S/T/X/R-learner, DML)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, X = generate_cate_data(n)
        problem = CATEProblem(outcomes, Float64.(treatment), X,
                              (alpha=0.05, n_folds=3))

        # S-Learner
        b = @benchmark solve($problem, SLearner()) samples=20 evals=1
        push!(results, (family="cate", method="s_learner", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # T-Learner
        b = @benchmark solve($problem, TLearner()) samples=20 evals=1
        push!(results, (family="cate", method="t_learner", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # X-Learner
        b = @benchmark solve($problem, XLearner()) samples=20 evals=1
        push!(results, (family="cate", method="x_learner", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # R-Learner
        b = @benchmark solve($problem, RLearner()) samples=20 evals=1
        push!(results, (family="cate", method="r_learner", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Double ML
        b = @benchmark solve($problem, DoubleMachineLearning()) samples=20 evals=1
        push!(results, (family="cate", method="double_ml", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark SCM estimators."""
function benchmark_scm(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("SCM Estimators (Synthetic Control)")
    println("─" ^ 80)

    results = []

    # SCM uses different sample size interpretation (units × periods)
    scm_sizes = [10, 20, 50]

    for n_units in scm_sizes
        n_periods = 20
        outcomes, treatment_period = generate_scm_data(n_units, n_periods)
        problem = SCMProblem(outcomes, 1, treatment_period, nothing)

        # Synthetic Control
        b = @benchmark solve($problem, SyntheticControl()) samples=20 evals=1
        push!(results, (family="scm", method="synthetic_control", n=n_units*n_periods,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark Sensitivity estimators."""
function benchmark_sensitivity(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("Sensitivity Estimators (E-Value, Rosenbaum)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        paired_diff, estimate, se = generate_sensitivity_data(n)

        # E-Value
        problem_e = EValueProblem(estimate, se, ATE())
        b = @benchmark solve($problem_e, EValue()) samples=20 evals=1
        push!(results, (family="sensitivity", method="e_value", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Rosenbaum Bounds
        problem_r = RosenbaumProblem(paired_diff, [1.0, 1.5, 2.0])
        b = @benchmark solve($problem_r, RosenbaumBounds()) samples=20 evals=1
        push!(results, (family="sensitivity", method="rosenbaum_bounds", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark Bounds estimators."""
function benchmark_bounds(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("Bounds Estimators (Manski, Lee)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, selection = generate_bounds_data(n)

        # Manski worst-case bounds
        b = @benchmark manski_worst_case($outcomes, Float64.($treatment), -5.0, 5.0) samples=20 evals=1
        push!(results, (family="bounds", method="manski_worst_case", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))

        # Lee bounds
        b = @benchmark lee_bounds($outcomes, Float64.($treatment), Float64.($selection)) samples=20 evals=1
        push!(results, (family="bounds", method="lee_bounds", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark QTE estimators."""
function benchmark_qte(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("QTE Estimators (Unconditional)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        outcomes, treatment, X = generate_qte_data(n)

        # Unconditional QTE
        b = @benchmark unconditional_qte($outcomes, $treatment, 0.5) samples=20 evals=1
        push!(results, (family="qte", method="unconditional_qte", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

"""Benchmark Principal Stratification estimators."""
function benchmark_principal_strat(sample_sizes::Vector{Int})
    println("─" ^ 80)
    println("Principal Stratification (CACE)")
    println("─" ^ 80)

    results = []

    for n in sample_sizes
        Y, Z, D = generate_ps_data(n)

        # CACE 2SLS
        b = @benchmark cace_2sls($Y, $Z, $D) samples=20 evals=1
        push!(results, (family="principal_strat", method="cace_2sls", n=n,
                        median_ms=median(b).time/1e6, allocs=b.allocs, memory_kb=b.memory/1024))
    end

    for r in results
        @printf("  %-20s n=%5d: %.3f ms | %d allocs | %.1f KB\n",
                r.method, r.n, r.median_ms, r.allocs, r.memory_kb)
    end
    println()

    return results
end

# ============================================================================
# Main Benchmark Suite
# ============================================================================

function run_all_benchmarks(; sample_sizes::Vector{Int}=[100, 500, 1000, 5000])
    println("=" ^ 80)
    println("CausalEstimators.jl Comprehensive Benchmark Suite")
    println("=" ^ 80)
    println("Julia Version: ", VERSION)
    println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Sample sizes: ", sample_sizes)
    println("=" ^ 80)
    println()

    all_results = []

    # Run all benchmarks
    append!(all_results, benchmark_rct(sample_sizes))
    append!(all_results, benchmark_iv(sample_sizes))
    append!(all_results, benchmark_did(sample_sizes))
    append!(all_results, benchmark_rdd(sample_sizes))
    append!(all_results, benchmark_observational(sample_sizes))
    append!(all_results, benchmark_cate(sample_sizes))
    append!(all_results, benchmark_scm(sample_sizes))
    append!(all_results, benchmark_sensitivity(sample_sizes))
    append!(all_results, benchmark_bounds(sample_sizes))
    append!(all_results, benchmark_qte(sample_sizes))
    append!(all_results, benchmark_principal_strat(sample_sizes))

    # Summary
    println("=" ^ 80)
    println("SUMMARY (at n=1000)")
    println("=" ^ 80)

    families = unique([r.family for r in all_results])
    for family in families
        family_results = filter(r -> r.family == family && r.n == 1000, all_results)
        if !isempty(family_results)
            println("\n$family:")
            for r in family_results
                @printf("  %-20s: %8.3f ms\n", r.method, r.median_ms)
            end
        end
    end

    println("\n" * "=" ^ 80)
    println("Benchmark Complete!")
    println("=" ^ 80)

    return all_results
end

# ============================================================================
# Export Results to JSON
# ============================================================================

"""Export benchmark results to JSON format for Python comparison."""
function export_results_json(results, filepath::String)
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"metadata\": {")
        println(io, "    \"julia_version\": \"$(VERSION)\",")
        println(io, "    \"timestamp\": \"$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))\",")
        println(io, "    \"n_benchmarks\": $(length(results))")
        println(io, "  },")
        println(io, "  \"results\": [")

        for (i, r) in enumerate(results)
            comma = i < length(results) ? "," : ""
            println(io, "    {\"family\": \"$(r.family)\", \"method\": \"$(r.method)\", " *
                       "\"n\": $(r.n), \"median_ms\": $(round(r.median_ms, digits=4)), " *
                       "\"allocs\": $(r.allocs), \"memory_kb\": $(round(r.memory_kb, digits=2))}$comma")
        end

        println(io, "  ]")
        println(io, "}")
    end
    println("Results exported to: $filepath")
end

# ============================================================================
# Run if called directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_all_benchmarks()

    # Export to JSON
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    mkpath("benchmark/results")
    export_results_json(results, "benchmark/results/julia_benchmarks_$timestamp.json")
end
