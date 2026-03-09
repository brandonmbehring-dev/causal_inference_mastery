"""Time-series aware cross-fitting strategies for Dynamic DML.

Implements cross-validation splits that respect temporal ordering to prevent
information leakage from future to past observations.

Three strategies:
1. BlockedTimeSeriesSplit: Divide into K contiguous blocks
2. RollingOriginSplit: Expanding window (walk-forward validation)
3. PanelStratifiedSplit: Split by unit for panel data

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for
time series predictor evaluation. Information Sciences, 191, 192-213.
"""

from __future__ import annotations

from typing import Generator, Literal, Optional

import numpy as np


class BlockedTimeSeriesSplit:
    """Time-blocked cross-validation for single time series.

    Divides time series into K contiguous blocks. For each fold, one block
    is held out for testing and the remaining K-1 blocks are used for training.

    Unlike standard K-fold, this preserves temporal contiguity within blocks,
    though it does allow training on "future" data when testing on middle blocks.
    For strict forward-only validation, use RollingOriginSplit.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds/blocks.
    gap : int, default=0
        Number of observations to exclude between train and test to avoid
        leakage from autocorrelation.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    gap : int
        Gap between train and test sets.

    Examples
    --------
    >>> cv = BlockedTimeSeriesSplit(n_splits=5)
    >>> for train_idx, test_idx in cv.split(np.arange(100)):
    ...     print(f"Train: {train_idx[:3]}...{train_idx[-3:]}, Test: {test_idx}")
    Train: [20 21 22]...[97 98 99], Test: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    ...
    """

    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")

        self.n_splits = n_splits
        self.gap = gap

    def split(self, X: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples,) or (n_samples, n_features).
            Only the length is used.

        Yields
        ------
        train_idx : np.ndarray
            Training set indices.
        test_idx : np.ndarray
            Test set indices.
        """
        n_samples = len(X)
        block_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # Test block boundaries
            test_start = i * block_size
            test_end = (i + 1) * block_size if i < self.n_splits - 1 else n_samples

            test_idx = indices[test_start:test_end]

            # Training: all other blocks with gap
            train_before_end = max(0, test_start - self.gap)
            train_after_start = min(n_samples, test_end + self.gap)

            train_idx = np.concatenate([indices[:train_before_end], indices[train_after_start:]])

            if len(train_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class RollingOriginSplit:
    """Expanding window cross-validation (walk-forward validation).

    Implements the time series cross-validation approach where training
    window expands forward, always testing on future data.

    For each split:
    - Train on observations 0 to (initial_window + step * i)
    - Test on observations (train_end + gap) to (train_end + gap + horizon)

    This is the most conservative approach for time series as it never
    trains on future data.

    Parameters
    ----------
    initial_window : int
        Minimum number of observations in initial training set.
    step : int, default=1
        Number of observations to add to training set for each split.
    horizon : int, default=1
        Number of observations in each test set.
    gap : int, default=0
        Number of observations between train and test sets.
    max_train_size : int, optional
        Maximum training set size. If None, training window grows unbounded.

    Examples
    --------
    >>> cv = RollingOriginSplit(initial_window=50, step=10, horizon=5)
    >>> n_obs = 100
    >>> for train_idx, test_idx in cv.split(np.arange(n_obs)):
    ...     print(f"Train: 0-{train_idx[-1]}, Test: {test_idx[0]}-{test_idx[-1]}")
    Train: 0-49, Test: 50-54
    Train: 0-59, Test: 60-64
    ...
    """

    def __init__(
        self,
        initial_window: int,
        step: int = 1,
        horizon: int = 1,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ) -> None:
        if initial_window < 1:
            raise ValueError(f"initial_window must be positive, got {initial_window}")
        if step < 1:
            raise ValueError(f"step must be positive, got {step}")
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")

        self.initial_window = initial_window
        self.step = step
        self.horizon = horizon
        self.gap = gap
        self.max_train_size = max_train_size

    def split(self, X: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples,) or (n_samples, n_features).

        Yields
        ------
        train_idx : np.ndarray
            Training set indices.
        test_idx : np.ndarray
            Test set indices.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        train_end = self.initial_window

        while train_end + self.gap + self.horizon <= n_samples:
            # Training indices
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            train_idx = indices[train_start:train_end]

            # Test indices
            test_start = train_end + self.gap
            test_end = min(test_start + self.horizon, n_samples)
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

            train_end += self.step

    def get_n_splits(self, n_samples: int) -> int:
        """Return number of splits for given sample size."""
        n_splits = 0
        train_end = self.initial_window
        while train_end + self.gap + self.horizon <= n_samples:
            n_splits += 1
            train_end += self.step
        return n_splits


class PanelStratifiedSplit:
    """Cross-fitting for panel data by stratifying on units.

    Following Lewis & Syrgkanis (2021) Algorithm 1, splits panel data
    by unit rather than by time. This preserves the full time series
    structure within each unit while allowing cross-fitting across units.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle units before splitting.
    random_state : int, optional
        Random seed for shuffling.

    Notes
    -----
    This is the preferred cross-fitting strategy for panel data as it
    maintains temporal structure within each unit, which is critical
    for dynamic treatment effect estimation.

    Examples
    --------
    >>> cv = PanelStratifiedSplit(n_splits=5)
    >>> unit_id = np.repeat(np.arange(50), 10)  # 50 units, 10 periods each
    >>> for train_idx, test_idx in cv.split(np.arange(500), unit_id=unit_id):
    ...     train_units = np.unique(unit_id[train_idx])
    ...     test_units = np.unique(unit_id[test_idx])
    ...     print(f"Train units: {len(train_units)}, Test units: {len(test_units)}")
    Train units: 40, Test units: 10
    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        unit_id: np.ndarray,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples,) or (n_samples, n_features).
        unit_id : np.ndarray
            Unit identifiers, shape (n_samples,).

        Yields
        ------
        train_idx : np.ndarray
            Training set indices (all observations from training units).
        test_idx : np.ndarray
            Test set indices (all observations from test units).
        """
        unique_units = np.unique(unit_id)
        n_units = len(unique_units)

        if n_units < self.n_splits:
            raise ValueError(
                f"Number of units ({n_units}) must be at least n_splits ({self.n_splits})"
            )

        # Optionally shuffle units
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            unit_order = rng.permutation(unique_units)
        else:
            unit_order = unique_units

        # Split units into folds
        fold_size = n_units // self.n_splits
        for i in range(self.n_splits):
            # Test units for this fold
            test_start = i * fold_size
            if i == self.n_splits - 1:
                test_units = unit_order[test_start:]
            else:
                test_units = unit_order[test_start : test_start + fold_size]

            # Training units: all other units
            train_units = np.setdiff1d(unique_units, test_units)

            # Convert unit membership to observation indices
            test_mask = np.isin(unit_id, test_units)
            train_mask = np.isin(unit_id, train_units)

            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class ProgressiveBlockSplit:
    """Progressive block cross-fitting for single long time series.

    Implements Lewis & Syrgkanis (2021) Algorithm 2. For block b,
    trains nuisance models on blocks 1 through b-1, predicts for block b.

    This is more conservative than BlockedTimeSeriesSplit as it only
    uses past data for training, never future data.

    Parameters
    ----------
    n_blocks : int, default=10
        Number of blocks to divide the time series into.
    min_train_blocks : int, default=2
        Minimum number of training blocks before starting predictions.
        First min_train_blocks are used only for training.

    Notes
    -----
    This results in fewer test observations than BlockedTimeSeriesSplit
    since the first min_train_blocks are never tested, but provides
    valid forward-looking inference.

    Examples
    --------
    >>> cv = ProgressiveBlockSplit(n_blocks=10, min_train_blocks=3)
    >>> for train_idx, test_idx in cv.split(np.arange(1000)):
    ...     print(f"Train blocks: {len(train_idx) / 100:.0f}, Test block size: {len(test_idx)}")
    Train blocks: 3, Test block size: 100
    Train blocks: 4, Test block size: 100
    ...
    """

    def __init__(self, n_blocks: int = 10, min_train_blocks: int = 2) -> None:
        if n_blocks < 3:
            raise ValueError(f"n_blocks must be at least 3, got {n_blocks}")
        if min_train_blocks < 1:
            raise ValueError(f"min_train_blocks must be positive, got {min_train_blocks}")
        if min_train_blocks >= n_blocks:
            raise ValueError(
                f"min_train_blocks ({min_train_blocks}) must be less than n_blocks ({n_blocks})"
            )

        self.n_blocks = n_blocks
        self.min_train_blocks = min_train_blocks

    def split(self, X: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for train/test splits.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples,) or (n_samples, n_features).

        Yields
        ------
        train_idx : np.ndarray
            Training set indices (all previous blocks).
        test_idx : np.ndarray
            Test set indices (current block only).
        """
        n_samples = len(X)
        block_size = n_samples // self.n_blocks
        indices = np.arange(n_samples)

        # Start predicting from block min_train_blocks
        for b in range(self.min_train_blocks, self.n_blocks):
            # Train on all blocks before b
            train_end = b * block_size
            train_idx = indices[:train_end]

            # Test on block b
            test_start = b * block_size
            test_end = (b + 1) * block_size if b < self.n_blocks - 1 else n_samples
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_blocks - self.min_train_blocks


def get_cross_validator(
    strategy: Literal["blocked", "rolling", "panel", "progressive"],
    n_samples: Optional[int] = None,
    n_folds: int = 5,
    **kwargs,
) -> BlockedTimeSeriesSplit | RollingOriginSplit | PanelStratifiedSplit | ProgressiveBlockSplit:
    """Factory function to get appropriate cross-validator.

    Parameters
    ----------
    strategy : {"blocked", "rolling", "panel", "progressive"}
        Cross-validation strategy.
    n_samples : int, optional
        Number of samples. Required for "rolling" strategy to set initial_window.
    n_folds : int, default=5
        Number of folds/splits.
    **kwargs
        Additional arguments passed to the cross-validator constructor.

    Returns
    -------
    Cross-validator instance.

    Examples
    --------
    >>> cv = get_cross_validator("blocked", n_folds=5)
    >>> cv = get_cross_validator("rolling", n_samples=1000, n_folds=10)
    >>> cv = get_cross_validator("panel", n_folds=5, shuffle=True)
    """
    if strategy == "blocked":
        return BlockedTimeSeriesSplit(n_splits=n_folds, **kwargs)

    elif strategy == "rolling":
        if n_samples is None:
            raise ValueError("n_samples required for rolling strategy")
        # Set initial_window to use first 50% of data, create n_folds splits
        initial_window = n_samples // 2
        horizon = (n_samples - initial_window) // n_folds
        step = horizon
        return RollingOriginSplit(
            initial_window=initial_window,
            step=step,
            horizon=horizon,
            **kwargs,
        )

    elif strategy == "panel":
        return PanelStratifiedSplit(n_splits=n_folds, **kwargs)

    elif strategy == "progressive":
        return ProgressiveBlockSplit(n_blocks=n_folds * 2, min_train_blocks=n_folds, **kwargs)

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: 'blocked', 'rolling', 'panel', 'progressive'"
        )
