"""
    Independence Tests

Session 133: Conditional independence tests for causal discovery.

# Functions
- `partial_correlation`: Compute partial correlation coefficient
- `fisher_z_test`: Fisher's Z-transform CI test
- `ci_test`: Unified CI test interface
"""
module IndependenceTests

using LinearAlgebra
using Statistics
using Distributions

# Import CITestResult from sibling module
using ..DiscoveryTypes: CITestResult

export partial_correlation, fisher_z_test, ci_test, CITestResult


"""
    partial_correlation(data, x, y, z=Int[])

Compute partial correlation between X and Y given Z.

Uses precision matrix approach for numerical stability.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `x::Int`: Index of first variable
- `y::Int`: Index of second variable
- `z::Vector{Int}`: Indices of conditioning variables

# Returns
- `Float64`: Partial correlation coefficient ρ(X, Y | Z)
"""
function partial_correlation(data::Matrix{Float64}, x::Int, y::Int, z::Vector{Int}=Int[])
    if isempty(z)
        # Simple correlation
        return cor(data[:, x], data[:, y])
    end

    # Precision matrix approach
    indices = [x, y, z...]
    sub_data = data[:, indices]

    # Compute correlation matrix
    C = cor(sub_data)

    # Handle numerical issues
    if any(isnan, C) || any(isinf, C)
        return 0.0
    end

    # Partial correlation via precision matrix
    try
        P = pinv(C)
        denom = sqrt(abs(P[1, 1] * P[2, 2]))
        denom < 1e-10 && return 0.0
        ρ = -P[1, 2] / denom
        return clamp(ρ, -1.0, 1.0)
    catch
        return 0.0
    end
end


"""
    fisher_z_test(data, x, y, z=Int[]; alpha=0.01)

Fisher's Z-transform test for conditional independence.

Tests H₀: X ⊥ Y | Z assuming linear Gaussian relationships.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_variables)
- `x::Int`: Index of first variable
- `y::Int`: Index of second variable
- `z::Vector{Int}`: Indices of conditioning variables
- `alpha::Float64`: Significance level (default 0.01)

# Returns
- `CITestResult`: Test result with independence decision and p-value
"""
function fisher_z_test(data::Matrix{Float64}, x::Int, y::Int, z::Vector{Int}=Int[];
                       alpha::Float64=0.01)
    n = size(data, 1)
    k = length(z)

    # Compute partial correlation
    ρ = partial_correlation(data, x, y, z)

    # Degrees of freedom
    dof = n - k - 3

    if dof < 1
        return CITestResult(true, 1.0, 0.0, alpha, z)
    end

    # Fisher's Z-transform
    ρ_clipped = clamp(ρ, -0.9999, 0.9999)
    z_stat = 0.5 * sqrt(dof) * log((1 + ρ_clipped) / (1 - ρ_clipped))

    # Two-tailed p-value
    pvalue = 2 * (1 - cdf(Normal(), abs(z_stat)))

    CITestResult(pvalue > alpha, pvalue, z_stat, alpha, z)
end


"""
    ci_test(data, x, y, z=Int[]; alpha=0.01, method=:fisher_z)

Unified conditional independence test interface.

# Arguments
- `data::Matrix{Float64}`: Data matrix
- `x::Int`: First variable index
- `y::Int`: Second variable index
- `z::Vector{Int}`: Conditioning set indices
- `alpha::Float64`: Significance level
- `method::Symbol`: Test method (:fisher_z only currently)

# Returns
- `CITestResult`: Test result
"""
function ci_test(data::Matrix{Float64}, x::Int, y::Int, z::Vector{Int}=Int[];
                 alpha::Float64=0.01, method::Symbol=:fisher_z)
    if method == :fisher_z
        return fisher_z_test(data, x, y, z; alpha=alpha)
    else
        error("Unknown method: $method")
    end
end

end # module
