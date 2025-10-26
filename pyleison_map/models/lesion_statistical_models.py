"""
Statistical lesion-symptom mapping utilities (e.g., SCCAN) adapted from LESYMAP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import ants  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime
    ants = None  # type: ignore[assignment]

from sklearn.preprocessing import StandardScaler

ANTsImageLike = Union[str, Path, "ants.ANTsImage"]

__all__ = [
    "SCCANResult",
    "run_sccan",
    "optimize_sccan_sparseness",
]


@dataclass
class SCCANResult:
    """
    Container for SCCAN outputs.
    """

    statistic: np.ndarray
    raw_weights_image: Optional["ants.ANTsImage"]
    eig1: np.ndarray
    eig2: np.ndarray
    cca_correlation: float
    optimal_sparseness: Optional[float]
    cv_correlation: Optional[float]
    cv_pvalue: Optional[float]
    behavior_scale: Optional[np.ndarray]
    behavior_center: Optional[np.ndarray]
    lesmat_scale: Optional[np.ndarray]
    lesmat_center: Optional[np.ndarray]


def run_sccan(
    lesmat: np.ndarray,
    behavior: np.ndarray,
    mask: ANTsImageLike,
    *,
    optimize_sparseness: bool = True,
    validate_sparseness: bool = False,
    p_threshold: float = 0.05,
    sparseness: float = 0.045,
    sparseness_behavior: float = -0.99,
    mycoption: int = 1,
    robust: int = 1,
    nvecs: int = 1,
    cthresh: int = 150,
    its: int = 20,
    nperms: int = 0,
    smooth: float = 0.4,
    max_based: bool = False,
    directional: bool = True,
    sparseness_penalty: float = 0.03,
    lower_sparseness: float = -0.9,
    upper_sparseness: float = 0.9,
    tol: float = 0.03,
    n_folds: int = 4,
    cv_repeats: Optional[int] = None,
    random_state: Optional[int] = None,
) -> SCCANResult:
    """
    Run SCCAN on an already vectorised lesion matrix and behavior scores.

    Parameters mirror LESYMAP::lsm_sccan where possible.
    """

    if ants is None:
        raise ImportError("run_sccan requires antspy (install with `pip install antspyx`).")

    lesmat = np.asarray(lesmat, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64).reshape(-1)

    if lesmat.ndim != 2:
        raise ValueError("`lesmat` must be a 2D array (subjects x voxels).")
    if behavior.shape[0] != lesmat.shape[0]:
        raise ValueError("`behavior` length must match number of rows in `lesmat`.")

    mask_img = _ensure_ants_image(mask)
    voxel_count = int((mask_img.numpy() > 0).sum())
    if lesmat.shape[1] != voxel_count:
        raise ValueError(
            f"`lesmat` column count ({lesmat.shape[1]}) does not match mask voxel count ({voxel_count})."
        )

    scaler_behavior = StandardScaler(with_mean=True, with_std=True)
    behavior_scaled = scaler_behavior.fit_transform(behavior.reshape(-1, 1)).reshape(-1)

    scaler_lesmat = StandardScaler(with_mean=True, with_std=True)
    lesmat_scaled = scaler_lesmat.fit_transform(lesmat)

    chosen_sparseness = sparseness
    cv_corr = None
    cv_pval = None

    if optimize_sparseness or validate_sparseness:
        opt = optimize_sccan_sparseness(
            lesmat_scaled,
            behavior_scaled,
            mask_img,
            sparseness_penalty=sparseness_penalty,
            lower_sparseness=lower_sparseness,
            upper_sparseness=upper_sparseness,
            tol=tol,
            just_validate=validate_sparseness and not optimize_sparseness,
            n_folds=n_folds,
            cv_repeats=cv_repeats,
            sparseness=chosen_sparseness,
            sparseness_behavior=sparseness_behavior,
            mycoption=mycoption,
            robust=robust,
            nvecs=nvecs,
            cthresh=cthresh,
            its=its,
            nperms=nperms,
            smooth=smooth,
            max_based=max_based,
            directional=directional,
            random_state=random_state,
        )
        chosen_sparseness = opt["minimum"]
        cv_corr = opt.get("cv_correlation")
        cv_pval = opt.get("cv_pvalue")
        if not validate_sparseness and optimize_sparseness and cv_pval is not None and cv_pval > p_threshold:
            zeros = np.zeros(lesmat.shape[1], dtype=np.float32)
            return SCCANResult(
                statistic=zeros,
                raw_weights_image=None,
                eig1=zeros,
                eig2=np.zeros((nvecs, 1), dtype=np.float32),
                cca_correlation=float("nan"),
                optimal_sparseness=chosen_sparseness,
                cv_correlation=cv_corr,
                cv_pvalue=cv_pval,
                behavior_scale=scaler_behavior.scale_,
                behavior_center=scaler_behavior.mean_,
                lesmat_scale=scaler_lesmat.scale_,
                lesmat_center=scaler_lesmat.mean_,
            )

    sccan_out = _run_sparse_decom2(
        lesmat_scaled,
        behavior_scaled,
        mask_img,
        sparseness=chosen_sparseness,
        sparseness_behavior=sparseness_behavior,
        mycoption=mycoption,
        robust=robust,
        nvecs=nvecs,
        cthresh=cthresh,
        its=its,
        nperms=nperms,
        smooth=smooth,
        max_based=max_based,
    )

    eig1 = _format_eig1(sccan_out["eig1"], nvecs=nvecs, n_features=lesmat.shape[1])
    eig2 = _format_eig2(sccan_out["eig2"], nvecs=nvecs, n_behaviors=1)

    primary_weights = eig1[0].copy()
    if np.allclose(primary_weights, 0):
        statistic = np.zeros_like(primary_weights, dtype=np.float32)
    else:
        statistic = (primary_weights / np.max(np.abs(primary_weights))).astype(np.float32)

    projections = lesmat_scaled @ eig1.T
    predicted = projections @ eig2.T

    if directional:
        behavior_sign = np.sign(eig2[0, 0]) if eig2.size > 0 else 1.0
        corr = _safe_corrcoef(predicted.reshape(-1), behavior_scaled)
        corr_sign = np.sign(corr) if not np.isnan(corr) else 1.0
        statistic = statistic * float(behavior_sign * corr_sign)
    else:
        statistic = np.abs(statistic)

    if not max_based:
        statistic[np.abs(statistic) < 0.1] = 0.0

    weights_img = ants.make_image(mask_img, primary_weights.astype(np.float32))
    temp = ants.make_image(mask_img, statistic.astype(np.float32))
    labeled = ants.label_clusters(
        temp.abs(),
        min_cluster_size=cthresh,
        min_thresh=np.finfo(np.float32).eps,
        max_thresh=float("inf"),
    )
    filtered = temp * ants.threshold_image(labeled, np.finfo(np.float32).eps, float("inf"))
    statistic_filtered = ants.image_list_to_matrix([filtered], mask_img)[0].astype(np.float32)
    if not statistic_filtered.any():
        statistic_filtered = statistic

    cca_corr = float(_safe_corrcoef(predicted.reshape(-1), behavior_scaled))

    return SCCANResult(
        statistic=statistic_filtered,
        raw_weights_image=weights_img,
        eig1=eig1.astype(np.float32),
        eig2=eig2.astype(np.float32),
        cca_correlation=cca_corr,
        optimal_sparseness=chosen_sparseness if optimize_sparseness else None,
        cv_correlation=cv_corr,
        cv_pvalue=cv_pval,
        behavior_scale=scaler_behavior.scale_,
        behavior_center=scaler_behavior.mean_,
        lesmat_scale=scaler_lesmat.scale_,
        lesmat_center=scaler_lesmat.mean_,
    )


def optimize_sccan_sparseness(
    lesmat_scaled: np.ndarray,
    behavior_scaled: np.ndarray,
    mask: "ants.ANTsImage",
    *,
    sparseness_penalty: float = 0.03,
    lower_sparseness: float = -0.9,
    upper_sparseness: float = 0.9,
    tol: float = 0.03,
    just_validate: bool = False,
    n_folds: int = 4,
    cv_repeats: Optional[int] = None,
    sparseness: Optional[float] = None,
    sparseness_behavior: float = -0.99,
    mycoption: int = 1,
    robust: int = 1,
    nvecs: int = 1,
    cthresh: int = 150,
    its: int = 30,
    nperms: int = 0,
    smooth: float = 0.4,
    max_based: bool = False,
    directional: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Optimize SCCAN sparseness using repeated k-fold cross validation.
    """

    if ants is None:
        raise ImportError("optimize_sccan_sparseness requires antspy.")

    rng = np.random.default_rng(random_state)
    n = behavior_scaled.shape[0]
    if cv_repeats is None:
        if n <= 30:
            cv_repeats = 6
        elif n <= 40:
            cv_repeats = 5
        elif n <= 50:
            cv_repeats = 4
        else:
            cv_repeats = 3

    folds = [_create_folds(behavior_scaled, n_folds, rng) for _ in range(cv_repeats)]

    def objective(sparse_value: float) -> Tuple[float, float]:
        sccan_sparsity = (sparse_value, sparseness_behavior)
        cv_corrs: List[float] = []
        for repetition in folds:
            behavior_pred = np.zeros_like(behavior_scaled)
            for test_idx in repetition:
                train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=True)
                sccan_out = _run_sparse_decom2(
                    lesmat_scaled[train_idx],
                    behavior_scaled[train_idx],
                    mask,
                    sparseness=sparse_value,
                    sparseness_behavior=sparseness_behavior,
                    mycoption=mycoption,
                    robust=robust,
                    nvecs=nvecs,
                    cthresh=cthresh,
                    its=its,
                    nperms=nperms,
                    smooth=smooth,
                    max_based=max_based,
                )
                eig1 = _format_eig1(sccan_out["eig1"], nvecs=nvecs, n_features=lesmat_scaled.shape[1])
                eig2 = _format_eig2(sccan_out["eig2"], nvecs=nvecs, n_behaviors=1)
                proj = lesmat_scaled[test_idx] @ eig1.T
                pred = proj @ eig2.T
                behavior_pred[test_idx] = pred.reshape(-1)
            corr = abs(_safe_corrcoef(behavior_scaled, behavior_pred))
            cv_corrs.append(corr)
        mean_corr = float(np.nanmean(cv_corrs))
        cost = 1.0 - (mean_corr - abs(sparse_value) * sparseness_penalty)
        return cost, mean_corr

    if just_validate:
        if sparseness is None:
            raise ValueError("`sparseness` must be provided when `just_validate` is True.")
        cost, mean_corr = objective(sparseness)
        pval = _corr_to_pvalue(mean_corr, n)
        return {"minimum": sparseness, "objective": cost, "cv_correlation": mean_corr, "cv_pvalue": pval}

    best_val = None
    best_cost = float("inf")
    a, b = lower_sparseness, upper_sparseness
    phi = (1 + math.sqrt(5)) / 2
    invphi = 1 / phi
    invphi2 = invphi ** 2
    h = b - a
    if h <= tol:
        sparse_candidates = [(a + b) / 2]
    else:
        n_iter = int(math.ceil(math.log(tol / h) / math.log(invphi)))
        c = a + invphi2 * h
        d = a + invphi * h
        f_c, corr_c = objective(c)
        f_d, corr_d = objective(d)
        for _ in range(n_iter):
            if f_c < f_d:
                b, f_d, corr_d = d, f_c, corr_c
                d = c
                c = a + invphi2 * (b - a)
                f_c, corr_c = objective(c)
            else:
                a, f_c, corr_c = c, f_d, corr_d
                c = d
                d = a + invphi * (b - a)
                f_d, corr_d = objective(d)
        sparse_candidates = [(a + b) / 2]

    for candidate in sparse_candidates:
        cost, corr = objective(candidate)
        if cost < best_cost:
            best_cost = cost
            best_val = (candidate, corr)

    if best_val is None:
        raise RuntimeError("Sparseness optimization failed.")

    best_sparse, best_corr = best_val
    pval = _corr_to_pvalue(best_corr, n)
    return {"minimum": best_sparse, "objective": best_cost, "cv_correlation": best_corr, "cv_pvalue": pval}


def _run_sparse_decom2(
    lesmat_scaled: np.ndarray,
    behavior_scaled: np.ndarray,
    mask: "ants.ANTsImage",
    *,
    sparseness: float,
    sparseness_behavior: float,
    mycoption: int,
    robust: int,
    nvecs: int,
    cthresh: int,
    its: int,
    nperms: int,
    smooth: float,
    max_based: bool,
) -> Dict[str, Any]:
    sparseness_params = (sparseness, sparseness_behavior)
    cthresh_params = (cthresh, 0)
    inmatrix = [lesmat_scaled, behavior_scaled.reshape(-1, 1)]
    inmask: List[Optional["ants.ANTsImage"]] = [mask, None]
    result = ants.sparse_decom2(
        inmatrix=inmatrix,
        inmask=inmask,
        mycoption=mycoption,
        robust=robust,
        sparseness=sparseness_params,
        nvecs=nvecs,
        cthresh=cthresh_params,
        its=its,
        perms=nperms,
        smooth=smooth,
        maxBased=max_based,
    )
    return result


def _format_eig1(arr: Any, *, nvecs: int, n_features: int) -> np.ndarray:
    mat = np.asarray(arr)
    if mat.ndim == 1:
        return mat.reshape(1, -1)
    if mat.shape == (nvecs, n_features):
        return mat
    if mat.shape == (n_features, nvecs):
        return mat.T
    raise ValueError(f"Unexpected eig1 shape {mat.shape}, expected ({nvecs}, {n_features}) or transpose.")


def _format_eig2(arr: Any, *, nvecs: int, n_behaviors: int) -> np.ndarray:
    mat = np.asarray(arr)
    if mat.ndim == 1:
        if mat.size == nvecs:
            return mat.reshape(nvecs, 1)
        return mat.reshape(1, -1)
    if mat.shape == (nvecs, n_behaviors):
        return mat
    if mat.shape == (n_behaviors, nvecs):
        return mat.T
    raise ValueError(f"Unexpected eig2 shape {mat.shape}.")


def _ensure_ants_image(mask: ANTsImageLike) -> "ants.ANTsImage":
    if isinstance(mask, (str, Path)):
        mask_img = ants.image_read(str(mask))
    elif hasattr(mask, "clone"):
        mask_img = mask.clone()
    else:
        raise TypeError("mask must be an ANTsImage or path to one.")
    if mask_img.dimension != 3:
        raise ValueError("Mask must be 3D.")
    return mask_img


def _create_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    if np.issubdtype(y.dtype, np.number):
        quantiles = np.linspace(0, 1, min(max(2, len(y) // k), 5) + 1)
        groups = np.digitize(y, np.quantile(y, quantiles), right=True)
    else:
        groups = y
    unique_groups = np.unique(groups)
    folds = [[] for _ in range(k)]
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        rng.shuffle(idx)
        for i, index in enumerate(idx):
            folds[i % k].append(index)
    return [np.array(sorted(fold), dtype=int) for fold in folds if fold]


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _corr_to_pvalue(r: float, n: int) -> Optional[float]:
    if n <= 2:
        return None
    r = max(min(r, 0.999999), -0.999999)
    t = abs(r) * math.sqrt((n - 2) / (1 - r ** 2))
    try:
        from scipy.stats import t as t_dist  # type: ignore[import]
    except Exception:
        return None
    return float(2 * t_dist.sf(t, df=n - 2))
