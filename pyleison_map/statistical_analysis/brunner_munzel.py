"""
Brunner-Munzel lesion-symptom analysis (fast variant) adapted from LESYMAP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.stats import brunnermunzel, norm  # type: ignore[import]
except Exception:  # pragma: no cover - runtime check
    brunnermunzel = None  # type: ignore
    norm = None  # type: ignore

__all__ = ["BrunnerMunzelResult", "run_brunner_munzel"]


@dataclass
class BrunnerMunzelResult:
    """
    Output of voxel-wise Brunner-Munzel testing.
    """

    statistic: np.ndarray
    pvalue: np.ndarray
    zscore: np.ndarray
    degrees_of_freedom: Optional[np.ndarray]
    alternative: str
    permuted_voxels_mask: np.ndarray
    voxel_pvalue_permutations: Optional[np.ndarray]
    fwer_threshold: Optional[np.ndarray]
    fwer_quantiles: Optional[np.ndarray]
    fwer_distribution: Optional[np.ndarray]


def run_brunner_munzel(
    lesmat: np.ndarray,
    behavior: Sequence[float],
    *,
    permute_n_threshold: int = 9,
    alternative: str = "greater",
    permute_all_voxels: bool = False,
    nperm_voxel: int = 20000,
    nperm_fwer: int = 0,
    fwer_rank: int = 1,
    p_threshold: float = 0.05,
    random_state: Optional[int] = None,
) -> BrunnerMunzelResult:
    """
    Perform voxel-wise Brunner-Munzel tests with optional permutation-based p-values
    and family-wise error rate (FWER) thresholding.

    Parameters mirror LESYMAP::lsm_BMfast where practical.
    """

    if brunnermunzel is None or norm is None:
        raise ImportError("scipy is required for Brunner-Munzel analysis (missing scipy.stats).")

    lesmat = np.asarray(lesmat, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64).reshape(-1)

    if lesmat.ndim != 2:
        raise ValueError("`lesmat` must be a 2D array (subjects x voxels).")
    if lesmat.shape[0] != behavior.size:
        raise ValueError("`behavior` length must match number of rows in `lesmat`.")
    if not np.array_equal(lesmat, lesmat.astype(bool)):
        raise ValueError("`lesmat` must be binary (0/1).")

    alt = alternative.lower()
    if alt not in {"greater", "less", "two.sided"}:
        raise ValueError("`alternative` must be one of {'greater', 'less', 'two.sided'}.")

    n_subjects, n_voxels = lesmat.shape
    lesion_counts = lesmat.sum(axis=0)
    nonlesion_counts = n_subjects - lesion_counts
    perm_mask = (lesion_counts <= permute_n_threshold) | (nonlesion_counts <= permute_n_threshold)
    if permute_all_voxels:
        perm_mask[:] = True

    stats = np.zeros(n_voxels, dtype=np.float64)
    pvalues = np.ones(n_voxels, dtype=np.float64)
    dof = np.full(n_voxels, np.nan, dtype=np.float64)

    for idx in range(n_voxels):
        mask = lesmat[:, idx].astype(bool)
        grp0 = behavior[~mask]
        grp1 = behavior[mask]
        if grp0.size < 2 or grp1.size < 2:
            continue
        res = brunnermunzel(grp0, grp1, alternative="two-sided")
        stats[idx] = res.statistic
        pvalues[idx] = res.pvalue
        dof[idx] = _estimate_bm_df(grp0.size, grp1.size)

    # Adjust p-values for alternative hypothesis
    if alt == "greater":
        pvalues = _bm_adjust_pvalue(stats, pvalues, direction="greater")
        zscore = norm.isf(pvalues)
    elif alt == "less":
        pvalues = _bm_adjust_pvalue(stats, pvalues, direction="less")
        zscore = norm.ppf(pvalues)
    else:
        zscore = np.zeros_like(pvalues)
        pos = stats > 0
        neg = stats < 0
        two_sided_p = _bm_adjust_pvalue(stats, pvalues, direction="two.sided")
        zscore[pos] = norm.isf(two_sided_p[pos] / 2.0)
        zscore[neg] = norm.ppf(two_sided_p[neg] / 2.0)
        pvalues = two_sided_p

    zscore[~np.isfinite(zscore)] = np.sign(zscore[~np.isfinite(zscore)]) * np.finfo(np.float64).max

    voxel_perm_pvalues = None
    rng = np.random.default_rng(random_state)
    if np.any(perm_mask) and nperm_voxel > 0:
        voxel_perm_pvalues = np.ones(n_voxels, dtype=np.float64)
        for idx in np.where(perm_mask)[0]:
            mask = lesmat[:, idx].astype(bool)
            grp0 = behavior[~mask]
            grp1 = behavior[mask]
            if grp0.size < 2 or grp1.size < 2:
                continue
            observed = stats[idx]
            exceed = 1
            for _ in range(nperm_voxel):
                perm = rng.permutation(behavior)
                perm_grp0 = perm[~mask]
                perm_grp1 = perm[mask]
                perm_stat = brunnermunzel(
                    perm_grp0, perm_grp1, alternative="two-sided"
                ).statistic
                if _perm_exceed(perm_stat, observed, alt):
                    exceed += 1
            voxel_perm_pvalues[idx] = exceed / (nperm_voxel + 1.0)
        pvalues[perm_mask] = voxel_perm_pvalues[perm_mask]
        if alt == "greater":
            zscore = norm.isf(pvalues)
        elif alt == "less":
            zscore = norm.ppf(pvalues)
        else:
            pos = stats > 0
            neg = stats < 0
            zscore = np.zeros_like(pvalues)
            zscore[pos] = norm.isf(pvalues[pos] / 2.0)
            zscore[neg] = norm.ppf(pvalues[neg] / 2.0)

    fwer_threshold = None
    fwer_quantiles = None
    fwer_distribution = None
    if nperm_fwer > 0:
        maxvec = np.empty(nperm_fwer, dtype=np.float64)
        for perm_idx in range(nperm_fwer):
            perm_behavior = rng.permutation(behavior)
            perm_stats = np.empty(n_voxels, dtype=np.float64)
            for idx in range(n_voxels):
                mask = lesmat[:, idx].astype(bool)
                grp0 = perm_behavior[~mask]
                grp1 = perm_behavior[mask]
                if grp0.size < 2 or grp1.size < 2:
                    perm_stats[idx] = 0.0
                    continue
                perm_stats[idx] = brunnermunzel(
                    grp0, grp1, alternative="two-sided"
                ).statistic
            sorted_stats = np.sort(perm_stats)
            if alt == "greater":
                maxvec[perm_idx] = sorted_stats[-fwer_rank]
            elif alt == "less":
                maxvec[perm_idx] = sorted_stats[fwer_rank - 1]
            else:
                top = sorted_stats[-fwer_rank]
                bottom = sorted_stats[fwer_rank - 1]
                maxvec[perm_idx] = top if abs(top) >= abs(bottom) else bottom

        if alt == "greater":
            quantile = 1.0 - p_threshold
            thresh = np.quantile(maxvec, quantile)
            stats[stats < thresh] = 0.0
            fwer_quantiles = np.array([quantile], dtype=np.float64)
            fwer_threshold = np.array([thresh], dtype=np.float64)
        elif alt == "less":
            quantile = p_threshold
            thresh = np.quantile(maxvec, quantile)
            stats[stats > thresh] = 0.0
            fwer_quantiles = np.array([quantile], dtype=np.float64)
            fwer_threshold = np.array([thresh], dtype=np.float64)
        else:
            lower_q = p_threshold / 2.0
            upper_q = 1.0 - p_threshold / 2.0
            thresh = np.quantile(maxvec, [lower_q, upper_q])
            mask_keep = (stats <= thresh[0]) | (stats >= thresh[1])
            stats[~mask_keep] = 0.0
            fwer_quantiles = np.array([lower_q, upper_q], dtype=np.float64)
            fwer_threshold = thresh.astype(np.float64)

        fwer_distribution = maxvec.astype(np.float32, copy=False)

    return BrunnerMunzelResult(
        statistic=stats.astype(np.float32, copy=False),
        pvalue=pvalues.astype(np.float32, copy=False),
        zscore=zscore.astype(np.float32, copy=False),
        degrees_of_freedom=None if np.isnan(dof).all() else dof.astype(np.float32, copy=False),
        alternative=alt,
        permuted_voxels_mask=perm_mask.astype(bool, copy=False),
        voxel_pvalue_permutations=None if voxel_perm_pvalues is None else voxel_perm_pvalues.astype(np.float32, copy=False),
        fwer_threshold=None if fwer_threshold is None else fwer_threshold.astype(np.float32, copy=False),
        fwer_quantiles=None if fwer_quantiles is None else fwer_quantiles.astype(np.float32, copy=False),
        fwer_distribution=fwer_distribution,
    )


def _bm_adjust_pvalue(statistic: np.ndarray, pvalues: np.ndarray, direction: str) -> np.ndarray:
    if direction == "greater":
        adjusted = np.where(statistic >= 0, pvalues / 2.0, 1.0 - pvalues / 2.0)
    elif direction == "less":
        adjusted = np.where(statistic <= 0, pvalues / 2.0, 1.0 - pvalues / 2.0)
    else:
        adjusted = pvalues
    return np.clip(adjusted, 0.0, 1.0)


def _perm_exceed(perm_stat: float, observed: float, alternative: str) -> bool:
    if alternative == "greater":
        return perm_stat >= observed
    if alternative == "less":
        return perm_stat <= observed
    return abs(perm_stat) >= abs(observed)


def _estimate_bm_df(n0: int, n1: int) -> float:
    # Approximate df similar to Welch-Satterthwaite
    return float(n0 + n1 - 2)

