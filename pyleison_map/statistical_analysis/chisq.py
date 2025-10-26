"""
Chi-square lesion-symptom analysis adapted from LESYMAP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import chi2  # type: ignore[import]
except Exception:  # pragma: no cover - handled at runtime
    chi2 = None  # type: ignore

__all__ = ["ChiSquareResult", "run_chi_square"]


@dataclass
class ChiSquareResult:
    """
    Output from chi-square lesion-symptom mapping.
    """

    statistic: np.ndarray
    pvalue: np.ndarray
    yates_correction: bool
    n_permutations: int


def run_chi_square(
    lesmat: np.ndarray,
    behavior: Sequence,
    *,
    yates_correction: bool = True,
    n_permutations: int = 0,
    random_state: Optional[int] = None,
) -> ChiSquareResult:
    """
    Perform voxel-wise chi-square tests between binary lesion matrix and binary behavior.

    Parameters
    ----------
    lesmat:
        2D array with shape (subjects, voxels). Values must be binary (0/1).
    behavior:
        Binary target vector aligned with rows of `lesmat`.
    yates_correction:
        Apply Yates continuity correction.
    n_permutations:
        If > 0, estimate p-values using permutation testing with the specified number of shuffles.
    random_state:
        Optional seed for the permutation RNG.
    """

    lesmat = np.asarray(lesmat, dtype=np.float64)
    if lesmat.ndim != 2:
        raise ValueError("`lesmat` must be a 2D array (subjects x voxels).")

    behavior_arr, behavior_values = _prepare_behavior(behavior)
    if lesmat.shape[0] != behavior_arr.size:
        raise ValueError("`behavior` length must match number of rows in `lesmat`.")

    if not np.array_equal(lesmat, lesmat.astype(bool)):
        raise ValueError("`lesmat` must be binary (0/1).")

    stats = _chi_square_statistics(lesmat, behavior_arr, yates_correction)

    if n_permutations > 0:
        pvalues = _permutation_pvalues(
            lesmat, behavior_arr, stats, yates_correction, n_permutations, random_state
        )
    else:
        if chi2 is None:
            raise ImportError("scipy is required to compute chi-square p-values.")
        pvalues = chi2.sf(stats, df=1)  # type: ignore[arg-type]

    return ChiSquareResult(
        statistic=stats.astype(np.float32, copy=False),
        pvalue=pvalues.astype(np.float32, copy=False),
        yates_correction=yates_correction,
        n_permutations=int(n_permutations),
    )


def _prepare_behavior(behavior: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    behavior_arr = np.asarray(behavior)
    unique = np.unique(behavior_arr)
    if unique.size != 2:
        raise ValueError("`behavior` must have exactly two unique values for chi-square analysis.")
    # Map to {0,1}
    mapping = {unique[0]: 0.0, unique[1]: 1.0}
    mapped = np.vectorize(mapping.get, otypes=[float])(behavior_arr)
    return mapped.astype(np.float64), unique


def _chi_square_statistics(
    lesmat: np.ndarray,
    behavior: np.ndarray,
    yates_correction: bool,
) -> np.ndarray:
    behav_on = behavior.sum()
    behav_off = behavior.size - behav_on

    les_on = lesmat.sum(axis=0)
    les_on_behav_on = lesmat.T @ behavior
    les_on_behav_off = les_on - les_on_behav_on
    les_off_behav_on = behav_on - les_on_behav_on
    les_off_behav_off = behav_off - les_on_behav_off

    a = les_on_behav_on
    b = les_on_behav_off
    c = les_off_behav_on
    d = les_off_behav_off

    total = a + b + c + d  # constant across voxels
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    valid = denom > 0

    stats = np.zeros_like(a, dtype=np.float64)
    if not np.any(valid):
        return stats

    if yates_correction:
        numerator = np.abs(a * d - b * c)
        correction = total / 2.0
        numerator = np.maximum(0.0, numerator - correction)
        stats[valid] = total * (numerator[valid] ** 2) / denom[valid]
    else:
        numerator = (a * d - b * c) ** 2
        stats[valid] = total * numerator[valid] / denom[valid]

    return stats


def _permutation_pvalues(
    lesmat: np.ndarray,
    behavior: np.ndarray,
    observed_stats: np.ndarray,
    yates_correction: bool,
    n_permutations: int,
    random_state: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    exceed = np.ones_like(observed_stats, dtype=np.float64)  # add-one smoothing
    for _ in range(n_permutations):
        perm_behavior = rng.permutation(behavior)
        perm_stats = _chi_square_statistics(lesmat, perm_behavior, yates_correction)
        exceed += perm_stats >= observed_stats
    return exceed / (n_permutations + 1.0)

