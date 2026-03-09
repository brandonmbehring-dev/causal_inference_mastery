"""Score Functions for Score-Based Causal Discovery.

Session 138: BIC, AIC, and local scores for GES algorithm.

Score-based methods search for the DAG that maximizes a penalized
likelihood score. Common choices:

- **BIC (Bayesian Information Criterion)**: log L - (k/2) log n
- **AIC (Akaike Information Criterion)**: log L - k
- **BGe (Bayesian Gaussian equivalent)**: Marginal likelihood

For Gaussian linear models:
    log L = -n/2 * [log(2π) + log(σ²) + 1]

where σ² is the residual variance from regressing X_i on its parents.

References
----------
- Chickering (2002). Optimal structure identification with greedy search.
- Schwarz (1978). Estimating the dimension of a model.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.linalg import lstsq


class ScoreType(Enum):
    """Score function types."""

    BIC = "bic"
    AIC = "aic"
    BIC_G = "bic_g"  # BIC with Gaussian assumption


@dataclass
class LocalScore:
    """Score for a single node given its parents.

    Attributes
    ----------
    node : int
        Node index
    parents : Set[int]
        Parent node indices
    score : float
        Score value (higher is better)
    n_params : int
        Number of parameters
    rss : float
        Residual sum of squares
    """

    node: int
    parents: Set[int]
    score: float
    n_params: int
    rss: float


def _compute_rss(data: np.ndarray, node: int, parents: Set[int]) -> float:
    """Compute residual sum of squares for node given parents.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    node : int
        Target node index
    parents : Set[int]
        Parent node indices

    Returns
    -------
    float
        Residual sum of squares
    """
    n = data.shape[0]
    y = data[:, node]

    if len(parents) == 0:
        # No parents: RSS = sum of squared deviations from mean
        return np.sum((y - np.mean(y)) ** 2)

    # Regress node on parents
    X = data[:, list(parents)]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # OLS: y = X @ beta
    beta, residuals, _, _ = lstsq(X_with_intercept, y, rcond=None)

    # Compute residuals manually if not returned
    y_pred = X_with_intercept @ beta
    rss = np.sum((y - y_pred) ** 2)

    return rss


def local_score_bic(
    data: np.ndarray,
    node: int,
    parents: Set[int],
    cache: Optional[Dict[Tuple[int, frozenset], float]] = None,
) -> LocalScore:
    """Compute local BIC score for a node given its parents.

    BIC = log L - (k/2) * log(n)

    For Gaussian linear model:
    log L = -n/2 * [log(2π) + log(σ²) + 1]

    So: BIC = -n/2 * log(RSS/n) - (k/2) * log(n) + const

    We drop constants and use: -n * log(RSS/n) - k * log(n)
    (Higher is better)

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    node : int
        Target node index
    parents : Set[int]
        Parent node indices
    cache : Optional[Dict]
        Cache for computed scores

    Returns
    -------
    LocalScore
        Local score result
    """
    n_samples = data.shape[0]

    # Check cache
    cache_key = (node, frozenset(parents))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Compute RSS
    rss = _compute_rss(data, node, parents)

    # Number of parameters: intercept + parents
    n_params = 1 + len(parents)

    # BIC score (higher is better)
    # -n/2 * log(RSS/n) - k/2 * log(n)
    if rss <= 0:
        rss = 1e-10  # Numerical stability

    log_likelihood = -n_samples / 2 * np.log(rss / n_samples)
    penalty = n_params / 2 * np.log(n_samples)
    score = log_likelihood - penalty

    result = LocalScore(node=node, parents=parents, score=score, n_params=n_params, rss=rss)

    # Cache result
    if cache is not None:
        cache[cache_key] = result

    return result


def local_score_aic(
    data: np.ndarray,
    node: int,
    parents: Set[int],
    cache: Optional[Dict[Tuple[int, frozenset], float]] = None,
) -> LocalScore:
    """Compute local AIC score for a node given its parents.

    AIC = log L - k

    (Higher is better)
    """
    n_samples = data.shape[0]

    cache_key = (node, frozenset(parents))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    rss = _compute_rss(data, node, parents)
    n_params = 1 + len(parents)

    if rss <= 0:
        rss = 1e-10

    log_likelihood = -n_samples / 2 * np.log(rss / n_samples)
    penalty = n_params
    score = log_likelihood - penalty

    result = LocalScore(node=node, parents=parents, score=score, n_params=n_params, rss=rss)

    if cache is not None:
        cache[cache_key] = result

    return result


def local_score(
    data: np.ndarray,
    node: int,
    parents: Set[int],
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
) -> LocalScore:
    """Compute local score for a node given its parents.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    node : int
        Target node index
    parents : Set[int]
        Parent node indices
    score_type : ScoreType
        Score function to use
    cache : Optional[Dict]
        Cache for computed scores

    Returns
    -------
    LocalScore
        Local score result
    """
    if score_type in (ScoreType.BIC, ScoreType.BIC_G):
        return local_score_bic(data, node, parents, cache)
    elif score_type == ScoreType.AIC:
        return local_score_aic(data, node, parents, cache)
    else:
        raise ValueError(f"Unknown score type: {score_type}")


def total_score(
    data: np.ndarray,
    adjacency: np.ndarray,
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
) -> float:
    """Compute total score for a DAG.

    Total score = sum of local scores.

    Parameters
    ----------
    data : np.ndarray
        (n_samples, n_vars) data matrix
    adjacency : np.ndarray
        (n_vars, n_vars) adjacency matrix where adj[i,j]=1 means i→j
    score_type : ScoreType
        Score function to use
    cache : Optional[Dict]
        Cache for computed scores

    Returns
    -------
    float
        Total score (higher is better)
    """
    n_vars = data.shape[1]
    total = 0.0

    for node in range(n_vars):
        # Find parents of this node
        parents = set(np.where(adjacency[:, node] == 1)[0])
        ls = local_score(data, node, parents, score_type, cache)
        total += ls.score

    return total


def score_delta_add(
    data: np.ndarray,
    node: int,
    current_parents: Set[int],
    new_parent: int,
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
) -> float:
    """Compute score change from adding an edge.

    Returns score(new) - score(old), positive if improvement.
    """
    old_score = local_score(data, node, current_parents, score_type, cache)
    new_parents = current_parents | {new_parent}
    new_score = local_score(data, node, new_parents, score_type, cache)
    return new_score.score - old_score.score


def score_delta_remove(
    data: np.ndarray,
    node: int,
    current_parents: Set[int],
    parent_to_remove: int,
    score_type: ScoreType = ScoreType.BIC,
    cache: Optional[Dict] = None,
) -> float:
    """Compute score change from removing an edge.

    Returns score(new) - score(old), positive if improvement.
    """
    old_score = local_score(data, node, current_parents, score_type, cache)
    new_parents = current_parents - {parent_to_remove}
    new_score = local_score(data, node, new_parents, score_type, cache)
    return new_score.score - old_score.score
