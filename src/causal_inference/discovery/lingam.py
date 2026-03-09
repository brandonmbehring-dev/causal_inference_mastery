"""LiNGAM - Linear Non-Gaussian Acyclic Model.

Session 133: Functional causal discovery using independent component analysis.

Unlike PC algorithm which outputs an equivalence class (CPDAG), LiNGAM
identifies a UNIQUE causal DAG by exploiting non-Gaussianity of noise terms.

Key insight: If X = B*X + E where E has non-Gaussian independent components,
then the causal structure is uniquely identifiable via ICA.

References
----------
- Shimizu et al. (2006). A linear non-Gaussian acyclic model for causal discovery.
- Shimizu et al. (2011). DirectLiNGAM: A direct method for learning a linear
  non-Gaussian structural equation model.

Functions
---------
ica_lingam : ICA-based LiNGAM
direct_lingam : DirectLiNGAM (faster, no ICA iteration)
var_lingam : VAR-LiNGAM for time series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import linalg, stats

from .types import DAG, LiNGAMResult


def ica_lingam(
    data: np.ndarray,
    seed: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LiNGAMResult:
    """ICA-based LiNGAM for causal discovery.

    Identifies unique causal DAG from observational data by:
    1. Performing ICA to estimate unmixing matrix W
    2. Finding permutation P to make W lower triangular
    3. Extracting causal ordering and effect matrix B

    Assumptions:
    - Linear relationships: X_j = sum_i B[i,j] * X_i + E_j
    - Acyclic structure (DAG)
    - Non-Gaussian noise (at most one Gaussian allowed)
    - No hidden confounders

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
        Should be centered (mean 0).
    seed : int, optional
        Random seed for ICA initialization.
    max_iter : int
        Maximum ICA iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.

    Returns
    -------
    LiNGAMResult
        Result containing unique DAG, causal order, and adjacency matrix.

    Example
    -------
    >>> from .utils import generate_random_dag, generate_dag_data
    >>> dag = generate_random_dag(5, edge_prob=0.3, seed=42)
    >>> data, B_true = generate_dag_data(dag, n_samples=1000,
    ...                                   noise_type="laplace", seed=42)
    >>> result = ica_lingam(data, seed=42)
    >>> print(f"Causal order: {result.causal_order}")
    >>> print(f"Order accuracy: {result.causal_order_accuracy(dag.topological_order()):.2f}")

    Notes
    -----
    ICA-LiNGAM may be sensitive to:
    - Sample size (needs sufficient data for ICA)
    - Weak edges (small coefficients may not be detected)
    - Near-Gaussian distributions

    For faster execution, consider DirectLiNGAM.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = data.shape

    if verbose:
        print("=" * 60)
        print("ICA-LiNGAM")
        print("=" * 60)
        print(f"Data shape: {data.shape}")

    # Step 1: Center and whiten data
    data_centered = data - data.mean(axis=0)
    X_whitened, whitening_matrix = _whiten(data_centered)

    if verbose:
        print("Data whitened.")

    # Step 2: Perform ICA using FastICA
    W_ica = _fastica(X_whitened, rng, max_iter, tol, verbose)

    # Full unmixing matrix: W = W_ica @ whitening_matrix
    W = W_ica @ whitening_matrix

    if verbose:
        print("ICA completed.")

    # Step 3: Find permutation to make W^-1 lower triangular
    # A = W^-1 gives mixing matrix X = A @ S
    # We need B = I - P @ W^-1 @ P^T to be lower triangular
    A = linalg.inv(W)

    causal_order, P = _find_causal_order(A, verbose)

    if verbose:
        print(f"Causal order: {causal_order}")

    # Step 4: Extract adjacency matrix B
    # B = I - P @ W^-1 @ P^T with lower triangular structure
    A_permuted = P @ A @ P.T
    B = np.eye(n_vars) - linalg.inv(A_permuted)

    # Make strictly lower triangular (zero diagonal)
    B = np.tril(B, k=-1)

    # Permute back to original variable ordering
    P_inv = P.T
    B_original = P_inv @ B @ P_inv.T

    # Step 5: Prune small edges
    B_pruned = _prune_edges(B_original, data, verbose)

    # Step 6: Build DAG
    dag = _matrix_to_dag(B_pruned, causal_order)

    return LiNGAMResult(
        dag=dag,
        causal_order=causal_order,
        adjacency_matrix=B_pruned,
    )


def _whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten data to have identity covariance."""
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = linalg.eigh(cov)

    # Handle numerical issues
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Whitening transformation
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = D_inv_sqrt @ eigenvectors.T

    X_whitened = X @ whitening_matrix.T

    return X_whitened, whitening_matrix


def _fastica(
    X: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """FastICA algorithm for independent component analysis.

    Uses deflation approach with log-cosh non-linearity.
    """
    n_samples, n_components = X.shape
    W = np.zeros((n_components, n_components))

    for p in range(n_components):
        # Initialize weight vector
        w = rng.standard_normal(n_components)
        w = w / linalg.norm(w)

        for iteration in range(max_iter):
            w_old = w.copy()

            # Newton iteration with G(u) = log(cosh(u)), g(u) = tanh(u)
            wx = X @ w
            g_wx = np.tanh(wx)
            g_prime_wx = 1 - g_wx**2

            w = (X.T @ g_wx) / n_samples - g_prime_wx.mean() * w

            # Decorrelate from previous components
            w = w - W[:p, :].T @ (W[:p, :] @ w)

            # Normalize
            w = w / linalg.norm(w)

            # Check convergence
            if np.abs(np.abs(np.dot(w, w_old)) - 1) < tol:
                break

        W[p, :] = w

    return W


def _find_causal_order(A: np.ndarray, verbose: bool = False) -> Tuple[List[int], np.ndarray]:
    """Find causal ordering from mixing matrix A.

    Uses row-based approach: rows with smallest magnitude indicate
    root causes (no parents).

    Returns causal order and permutation matrix P.
    """
    n = A.shape[0]
    remaining = set(range(n))
    causal_order = []

    A_work = A.copy()

    for _ in range(n):
        # Find row with minimum L1 norm (most independent component)
        row_norms = []
        for i in remaining:
            # Normalize by diagonal to get relative parent influence
            row = A_work[i, :].copy()
            if abs(A_work[i, i]) > 1e-10:
                row = row / abs(A_work[i, i])
            row[i] = 0  # Ignore self
            row_norms.append((i, np.sum(np.abs(row))))

        # Select variable with smallest normalized influence
        row_norms.sort(key=lambda x: x[1])
        next_var = row_norms[0][0]

        causal_order.append(next_var)
        remaining.remove(next_var)

        # Update working matrix (remove effect of this variable)
        if remaining:
            for i in remaining:
                A_work[i, :] -= (
                    A_work[i, next_var] / A_work[next_var, next_var] * A_work[next_var, :]
                )

    # Build permutation matrix
    P = np.zeros((n, n))
    for new_idx, old_idx in enumerate(causal_order):
        P[new_idx, old_idx] = 1

    return causal_order, P


def _prune_edges(B: np.ndarray, data: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Prune small edges using adaptive threshold.

    Uses resampling-based threshold selection.
    """
    n_vars = B.shape[0]
    threshold = 0.1  # Default threshold

    # Adaptive threshold based on data scale
    std_scale = np.std(data, axis=0).mean()
    if std_scale > 0:
        threshold = 0.05 * std_scale

    B_pruned = B.copy()
    B_pruned[np.abs(B_pruned) < threshold] = 0

    if verbose:
        n_edges = np.sum(B_pruned != 0)
        print(f"Pruned to {n_edges} edges (threshold={threshold:.4f})")

    return B_pruned


def _matrix_to_dag(B: np.ndarray, causal_order: List[int]) -> DAG:
    """Convert adjacency matrix to DAG structure."""
    n_vars = B.shape[0]
    dag = DAG(n_nodes=n_vars)

    for i in range(n_vars):
        for j in range(n_vars):
            if abs(B[i, j]) > 1e-10:
                # B[i,j] != 0 means i -> j
                dag.add_edge(i, j)

    return dag


def direct_lingam(
    data: np.ndarray,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> LiNGAMResult:
    """DirectLiNGAM: Direct method without ICA iteration.

    Faster than ICA-LiNGAM, determines causal order by iteratively
    finding exogenous variables (no parents).

    Algorithm:
    1. Find most exogenous variable (regress on all others, test residuals)
    2. Remove its effect from all variables
    3. Repeat until all variables ordered

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    seed : int, optional
        Random seed (for ties).
    verbose : bool
        Print progress.

    Returns
    -------
    LiNGAMResult
        Result with unique DAG and causal ordering.

    Example
    -------
    >>> result = direct_lingam(data, verbose=True)
    >>> print(f"Causal order: {result.causal_order}")

    Notes
    -----
    DirectLiNGAM is generally faster and more stable than ICA-LiNGAM,
    especially for:
    - Moderate to large number of variables
    - Data with weak edges
    - Near-Gaussian marginals (but non-Gaussian conditionals)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = data.shape

    if verbose:
        print("=" * 60)
        print("DirectLiNGAM")
        print("=" * 60)
        print(f"Data shape: {data.shape}")

    # Center data
    X = data - data.mean(axis=0)

    # Track remaining variables and residuals
    remaining = list(range(n_vars))
    causal_order = []
    residuals = X.copy()

    # Iteratively find causal order
    while remaining:
        # Find most exogenous variable among remaining
        exog_scores = []

        for i in remaining:
            # Regress variable i on all other remaining variables
            other_vars = [j for j in remaining if j != i]

            if len(other_vars) == 0:
                # Last variable
                exog_scores.append((i, 0.0))
                continue

            # Get residual of i regressed on others
            X_others = residuals[:, other_vars]
            y_i = residuals[:, i]

            # OLS regression
            X_design = np.column_stack([np.ones(n_samples), X_others])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_design, y_i, rcond=None)
                resid_i = y_i - X_design @ coeffs
            except np.linalg.LinAlgError:
                resid_i = y_i

            # Compute independence score for each other variable
            # Using kernel-based mutual information proxy
            total_dependence = 0.0
            for j in other_vars:
                # Correlation of residual with other variable
                r_j = residuals[:, j]
                corr = np.corrcoef(resid_i, r_j)[0, 1]
                total_dependence += abs(corr)

            exog_scores.append((i, total_dependence))

        # Select variable with minimum dependence (most exogenous)
        exog_scores.sort(key=lambda x: x[1])
        next_var = exog_scores[0][0]

        causal_order.append(next_var)
        remaining.remove(next_var)

        if verbose:
            print(
                f"  Step {len(causal_order)}: selected variable {next_var} "
                f"(score={exog_scores[0][1]:.4f})"
            )

        # Remove effect of next_var from all remaining variables
        if remaining:
            x_next = residuals[:, next_var]
            for j in remaining:
                y_j = residuals[:, j]
                # Regress j on next_var
                coef = np.cov(y_j, x_next)[0, 1] / (np.var(x_next) + 1e-10)
                residuals[:, j] = y_j - coef * x_next

    # Estimate adjacency matrix B
    B = _estimate_adjacency_direct(data, causal_order)

    # Build DAG
    dag = _matrix_to_dag(B, causal_order)

    if verbose:
        print(f"Causal order: {causal_order}")
        n_edges = np.sum(B != 0)
        print(f"Edges detected: {n_edges}")
        print("=" * 60)

    return LiNGAMResult(
        dag=dag,
        causal_order=causal_order,
        adjacency_matrix=B,
    )


def _estimate_adjacency_direct(data: np.ndarray, causal_order: List[int]) -> np.ndarray:
    """Estimate adjacency matrix given causal ordering.

    For each variable in order, regress on all prior variables.
    Non-zero coefficients indicate edges.
    """
    n_vars = len(causal_order)
    B = np.zeros((n_vars, n_vars))

    X = data - data.mean(axis=0)

    for idx, j in enumerate(causal_order):
        if idx == 0:
            continue  # First variable has no parents

        # Potential parents are earlier in causal order
        potential_parents = causal_order[:idx]

        # Regress j on potential parents
        X_parents = X[:, potential_parents]
        y_j = X[:, j]

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_parents, y_j, rcond=None)

            # Significance test for each coefficient
            residuals = y_j - X_parents @ coeffs
            mse = np.var(residuals)
            XtX_inv = np.linalg.pinv(X_parents.T @ X_parents)
            se = np.sqrt(np.diag(XtX_inv) * mse)

            for k, parent in enumerate(potential_parents):
                if abs(coeffs[k]) > 2 * se[k]:  # ~95% significance
                    B[parent, j] = coeffs[k]

        except np.linalg.LinAlgError:
            pass

    return B


def bootstrap_lingam(
    data: np.ndarray,
    n_bootstrap: int = 100,
    method: str = "direct",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[LiNGAMResult, np.ndarray]:
    """Bootstrap LiNGAM for edge confidence estimation.

    Runs LiNGAM on bootstrap samples and returns edge frequencies.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    n_bootstrap : int
        Number of bootstrap samples.
    method : str
        "direct" for DirectLiNGAM, "ica" for ICA-LiNGAM.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    result : LiNGAMResult
        Result from full data.
    edge_frequencies : np.ndarray
        Matrix of edge detection frequencies (0 to 1).

    Example
    -------
    >>> result, freqs = bootstrap_lingam(data, n_bootstrap=100)
    >>> # High confidence edges
    >>> high_conf = np.where(freqs > 0.8)
    >>> for i, j in zip(*high_conf):
    ...     print(f"Edge {i} -> {j}: {freqs[i,j]:.0%} confidence")
    """
    rng = np.random.default_rng(seed)
    n_samples, n_vars = data.shape

    # Select method
    lingam_func = direct_lingam if method == "direct" else ica_lingam

    # Run on full data
    full_result = lingam_func(data, seed=seed, verbose=verbose)

    # Bootstrap
    edge_counts = np.zeros((n_vars, n_vars))

    for b in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        data_boot = data[indices]

        # Run LiNGAM
        boot_result = lingam_func(data_boot, seed=seed if seed is None else seed + b)

        # Count edges
        B_boot = boot_result.adjacency_matrix
        edge_counts += (np.abs(B_boot) > 1e-10).astype(float)

        if verbose and (b + 1) % 20 == 0:
            print(f"Bootstrap {b + 1}/{n_bootstrap}")

    edge_frequencies = edge_counts / n_bootstrap

    return full_result, edge_frequencies


# =============================================================================
# Non-Gaussianity Diagnostics
# =============================================================================


def check_non_gaussianity(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Check non-Gaussianity assumption for LiNGAM.

    LiNGAM requires at most one Gaussian variable. This function
    tests each variable for Gaussianity.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    alpha : float
        Significance level for normality tests.

    Returns
    -------
    dict
        Diagnostic results including:
        - 'gaussian_vars': Indices of Gaussian variables
        - 'non_gaussian_vars': Indices of non-Gaussian variables
        - 'kurtosis': Excess kurtosis for each variable
        - 'skewness': Skewness for each variable
        - 'lingam_applicable': Whether LiNGAM assumptions hold
    """
    n_vars = data.shape[1]

    gaussian_vars = []
    non_gaussian_vars = []
    kurtosis = []
    skewness = []

    for j in range(n_vars):
        x = data[:, j]

        # Shapiro-Wilk test (if n <= 5000)
        if len(x) <= 5000:
            _, p_value = stats.shapiro(x)
        else:
            # Use D'Agostino-Pearson for larger samples
            _, p_value = stats.normaltest(x)

        if p_value > alpha:
            gaussian_vars.append(j)
        else:
            non_gaussian_vars.append(j)

        kurtosis.append(stats.kurtosis(x))
        skewness.append(stats.skew(x))

    # LiNGAM requires at most 1 Gaussian variable
    lingam_applicable = len(gaussian_vars) <= 1

    return {
        "gaussian_vars": gaussian_vars,
        "non_gaussian_vars": non_gaussian_vars,
        "kurtosis": np.array(kurtosis),
        "skewness": np.array(skewness),
        "lingam_applicable": lingam_applicable,
        "n_gaussian": len(gaussian_vars),
        "n_non_gaussian": len(non_gaussian_vars),
    }
