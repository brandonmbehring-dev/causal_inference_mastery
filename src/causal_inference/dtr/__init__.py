"""Dynamic Treatment Regimes (DTR) module.

Provides methods for estimating optimal dynamic treatment regimes
in multi-stage treatment settings.

Methods
-------
q_learning : Q-learning with backward induction
q_learning_single_stage : Q-learning convenience wrapper for single-stage
a_learning : A-learning (doubly robust) with backward induction
a_learning_single_stage : A-learning convenience wrapper for single-stage

Data Structures
---------------
DTRData : Multi-stage treatment data container
QLearningResult : Q-learning estimation results
ALearningResult : A-learning estimation results

References
----------
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B.
Robins, J. M. (2004). Optimal structural nested models for optimal sequential
    decisions. In Proceedings of the Second Seattle Symposium on Biostatistics.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating
    optimal dynamic treatment regimes. Statistical Science.

Examples
--------
>>> import numpy as np
>>> from causal_inference.dtr import DTRData, q_learning, a_learning
>>>
>>> # Single-stage example with Q-learning
>>> n = 500
>>> X = np.random.randn(n, 3)
>>> A = np.random.binomial(1, 0.5, n)
>>> Y = X[:, 0] + 2.0 * A + np.random.randn(n)
>>> result = q_learning_single_stage(Y, A, X)
>>> print(f"Q-learning optimal value: {result.value_estimate:.3f}")
>>>
>>> # A-learning (doubly robust)
>>> result = a_learning_single_stage(Y, A, X)
>>> print(f"A-learning optimal value: {result.value_estimate:.3f}")
>>>
>>> # Multi-stage example
>>> data = DTRData(outcomes=[Y1, Y2], treatments=[A1, A2], covariates=[X1, X2])
>>> result = q_learning(data)
>>> print(result.summary())
"""

from .types import DTRData, QLearningResult, ALearningResult
from .q_learning import q_learning, q_learning_single_stage
from .a_learning import a_learning, a_learning_single_stage

__all__ = [
    # Data structures
    "DTRData",
    "QLearningResult",
    "ALearningResult",
    # Q-learning estimators
    "q_learning",
    "q_learning_single_stage",
    # A-learning estimators
    "a_learning",
    "a_learning_single_stage",
]
