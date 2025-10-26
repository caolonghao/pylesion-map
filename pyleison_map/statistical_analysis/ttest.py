"""
Voxel-wise two-sample t-tests (equal variance or Welch) for lesion-symptom analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.stats import t as t_dist, norm as norm_dist, f as f_dist, shapiro  # type: ignore[import]
except Exception:  # pragma: no cover - handled via runtime checks
    t_dist = norm_dist = f_dist = shapiro = None  # type: ignore

__all__ = ["TTestResult", "TTestAssumptionResult", "run_ttest"]


@dataclass
class TTestAssumptionResult:
    """
    Stores results of optional t-test assumption checks.

    Attributes
    ----------
    variance_fail:
        Boolean mask marking voxels that failed the variance homogeneity test.
    normality_fail:
        Boolean mask marking voxels that failed the normality test
        (either group deviated from normality).
    alpha:
        Significance threshold used for the assumption checks.
    """

    variance_fail: np.ndarray
    normality_fail: np.ndarray
    alpha: float


@dataclass
class TTestResult:
    """
    Output of voxel-wise t-test (or Welch test).
    """

    statistic: np.ndarray
    pvalue: np.ndarray
    zscore: np.ndarray
    df: np.ndarray
    equal_variance: bool
    alternative: str
    assumptions: Optional[TTestAssumptionResult] = None


def run_ttest(
    lesmat: np.ndarray,
    behavior: Sequence[float],
    *,
    equal_variance: bool = True,
    alternative: str = "greater",
    check_assumptions: bool = False,
    assumption_alpha: float = 0.05,
) -> TTestResult:
    """
    Run voxel-wise two-sample t-tests using lesion presence/absence to split subjects.

    Parameters
    ----------
    lesmat:
        2D array (subjects x voxels) containing binary lesion indicators (0/1).
    behavior:
        1D sequence of behavioral scores aligned with `lesmat` rows.
    equal_variance:
        If True, assume equal variances (standard two-sample t-test). If False, perform Welch's test.
    alternative:
        One of {"greater", "less", "two-sided"} indicating the alternative hypothesis:
            - "greater": mean(non-lesioned) > mean(lesioned)
            - "less": mean(non-lesioned) < mean(lesioned)
            - "two-sided": means are different
    check_assumptions:
        When True and `equal_variance` is True, perform variance homogeneity and normality checks.
    assumption_alpha:
        Significance threshold for assumption tests.
    """

    if t_dist is None or norm_dist is None:
        raise ImportError("scipy is required to run t-tests (missing scipy.stats).")

    lesmat = np.asarray(lesmat, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64).reshape(-1)

    if lesmat.ndim != 2:
        raise ValueError("`lesmat` must be a 2D array (subjects x voxels).")
    if lesmat.shape[0] != behavior.size:
        raise ValueError("`behavior` length must match number of rows in `lesmat`.")
    if not np.array_equal(lesmat, lesmat.astype(bool)):
        raise ValueError("`lesmat` must be binary (0/1).")

    n_subjects, n_voxels = lesmat.shape
    lesion_mask = lesmat.astype(bool)
    nonlesion_mask = ~lesion_mask

    n0 = nonlesion_mask.sum(axis=0)
    n1 = lesion_mask.sum(axis=0)
    valid = (n0 > 1) & (n1 > 1)
    if not np.any(valid):
        raise ValueError("No voxels have enough subjects in both groups for a t-test.")

    means0 = np.zeros(n_voxels, dtype=np.float64)
    means1 = np.zeros(n_voxels, dtype=np.float64)
    vars0 = np.zeros(n_voxels, dtype=np.float64)
    vars1 = np.zeros(n_voxels, dtype=np.float64)

    # Compute means and variances per voxel
    with np.errstate(invalid="ignore"):
        for idx in range(n_voxels):
            grp0 = behavior[nonlesion_mask[:, idx]]
            grp1 = behavior[lesion_mask[:, idx]]
            if grp0.size > 0:
                means0[idx] = grp0.mean()
                vars0[idx] = grp0.var(ddof=1) if grp0.size > 1 else 0.0
            if grp1.size > 0:
                means1[idx] = grp1.mean()
                vars1[idx] = grp1.var(ddof=1) if grp1.size > 1 else 0.0

    diff = means0 - means1

    if equal_variance:
        df = n0 + n1 - 2
        pooled_var = ((n0 - 1) * vars0 + (n1 - 1) * vars1) / np.maximum(df, 1)
        se = np.sqrt(pooled_var * (1.0 / n0 + 1.0 / n1))
    else:
        se = np.sqrt(vars0 / n0 + vars1 / n1)
        with np.errstate(divide="ignore", invalid="ignore"):
            df = (vars0 / n0 + vars1 / n1) ** 2 / (
                (vars0 ** 2) / (n0 ** 2 * (n0 - 1)) + (vars1 ** 2) / (n1 ** 2 * (n1 - 1))
            )

    t_stat = np.zeros(n_voxels, dtype=np.float64)
    valid_se = se > 0
    t_stat[valid_se] = diff[valid_se] / se[valid_se]

    # Compute p-values based on alternative hypothesis
    alt = alternative.lower()
    if alt not in {"greater", "less", "two-sided"}:
        raise ValueError("`alternative` must be one of {'greater', 'less', 'two-sided'}.")

    pvalues = np.ones(n_voxels, dtype=np.float64)
    abs_t = np.abs(t_stat)
    if alt == "greater":
        pvalues = t_dist.sf(t_stat, df)
    elif alt == "less":
        pvalues = t_dist.cdf(t_stat, df)
    else:
        pvalues = t_dist.sf(abs_t, df) * 2.0

    # Convert to z-scores (consistent with LESYMAP)
    if alt == "less":
        zscore = norm_dist.ppf(pvalues)
    elif alt == "greater":
        zscore = norm_dist.isf(pvalues)
    else:
        zscore = norm_dist.isf(pvalues / 2.0)

    zscore[~np.isfinite(zscore)] = np.sign(zscore[~np.isfinite(zscore)]) * np.finfo(np.float64).max

    assumptions = None
    if check_assumptions and equal_variance:
        if f_dist is None or shapiro is None:
            raise ImportError("scipy is required for assumption checks (F-test and Shapiro).")
        variance_fail = np.zeros(n_voxels, dtype=bool)
        normality_fail = np.zeros(n_voxels, dtype=bool)
        for idx in range(n_voxels):
            grp0 = behavior[nonlesion_mask[:, idx]]
            grp1 = behavior[lesion_mask[:, idx]]
            if grp0.size > 1 and grp1.size > 1 and vars0[idx] >= 0 and vars1[idx] >= 0:
                if vars0[idx] > 0 and vars1[idx] > 0:
                    if vars0[idx] >= vars1[idx]:
                        f_stat = vars0[idx] / vars1[idx]
                        df1 = grp0.size - 1
                        df2 = grp1.size - 1
                    else:
                        f_stat = vars1[idx] / vars0[idx]
                        df1 = grp1.size - 1
                        df2 = grp0.size - 1
                    p_lower = f_dist.cdf(f_stat, df1, df2)
                    p_upper = f_dist.sf(f_stat, df1, df2)
                    p_var = 2 * min(p_lower, p_upper)
                    variance_fail[idx] = p_var <= assumption_alpha
                else:
                    variance_fail[idx] = True
            else:
                variance_fail[idx] = True

            normality_fail[idx] = _shapiro_fail(grp0, assumption_alpha) or _shapiro_fail(grp1, assumption_alpha)

        assumptions = TTestAssumptionResult(
            variance_fail=variance_fail,
            normality_fail=normality_fail,
            alpha=float(assumption_alpha),
        )

    return TTestResult(
        statistic=t_stat.astype(np.float32, copy=False),
        pvalue=pvalues.astype(np.float32, copy=False),
        zscore=zscore.astype(np.float32, copy=False),
        df=df.astype(np.float32, copy=False),
        equal_variance=equal_variance,
        alternative=alt,
        assumptions=assumptions,
    )


def _shapiro_fail(samples: np.ndarray, alpha: float) -> bool:
    if shapiro is None:
        raise ImportError("scipy is required for Shapiro-Wilk test.")
    if samples.size < 3:
        return True
    if samples.size > 5000:
        # Shapiro test is not recommended for n>5000; treat as pass
        return False
    try:
        stat, p = shapiro(samples)
    except Exception:
        return True
    return bool(p <= alpha)
