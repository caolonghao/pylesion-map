"""
Orchestration helpers for lesion-symptom workflows.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from pyleison_map.preprocess import (
    ImageLike,
    PreprocessOptions,
    LesionMatrixResult,
    build_lesion_matrix,
)
from pyleison_map.models import lesion_ml_models, lesion_statistical_models
from pyleison_map.statistical_analysis import (
    run_chi_square,
    run_ttest,
    run_brunner_munzel,
    ChiSquareResult,
    TTestResult,
    BrunnerMunzelResult,
)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base result containers
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    method: str
    preprocess: PreprocessOptions
    kept_indices: np.ndarray


# ---------------------------------------------------------------------------
# Machine-learning models
# ---------------------------------------------------------------------------


@dataclass
class MLModelResult(WorkflowResult):
    model: lesion_ml_models.LesionModelBase


def run_ml_model(
    images: Sequence[ImageLike],
    behavior: Sequence[float],
    *,
    method: str,
    task: str = "regression",
    preprocess: Optional[PreprocessOptions] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> MLModelResult:
    """
    Train an ML-based lesion model (Lasso, SVR, RF, etc.).

    Parameters
    ----------
    images:
        Sequence of lesion images (numpy arrays, paths, SimpleITK/ANTs objects).
    behavior:
        Target values aligned with `images`.
    method:
        Identifier matching `lesion_ml_models.make_model` kinds, e.g.
        {"lasso", "linear_svr", "linear_svm", "logistic_l1", "rf", "ada", "xgb", "svr_rbf"}.
    task:
        "regression" or "classification".
    model_kwargs:
        Additional keyword arguments forwarded to `make_model`.
    """

    prep_opts = deepcopy(preprocess) if preprocess is not None else PreprocessOptions(apply_subjectwise_l2=True)
    kwargs = lesion_ml_models.preprocessing_kwargs_from_options(
        prep_opts,
        overrides=deepcopy(model_kwargs) if model_kwargs else None,
    )
    model = lesion_ml_models.make_model(kind=method, task=task, **kwargs)
    model.fit(images, np.asarray(behavior), binarize=prep_opts.binarize)
    kept = getattr(model.prep, "kept_idx_", None)
    if kept is None:
        kept_indices = np.arange(len(images), dtype=np.int64)
    else:
        kept_indices = kept.astype(np.int64, copy=False)
    return MLModelResult(
        method=method,
        preprocess=prep_opts,
        kept_indices=kept_indices,
        model=model,
    )


# ---------------------------------------------------------------------------
# Statistical models with predictive capability (currently SCCAN)
# ---------------------------------------------------------------------------


@dataclass
class StatisticalModelResult(WorkflowResult):
    lesion_matrix: LesionMatrixResult
    result: lesion_statistical_models.SCCANResult


def run_statistical_model(
    images: Sequence[ImageLike],
    behavior: Sequence[float],
    *,
    preprocess: Optional[PreprocessOptions] = None,
    sccan_kwargs: Optional[Dict[str, Any]] = None,
) -> StatisticalModelResult:
    """
    Run statistical models that yield predictive outputs (currently SCCAN only).
    """
    prep_opts = deepcopy(preprocess) if preprocess is not None else PreprocessOptions()

    lesmat = _build_matrix(images, prep_opts)
    kwargs = deepcopy(sccan_kwargs) if sccan_kwargs else {}
    sccan_result = lesion_statistical_models.run_sccan(
        lesmat.matrix,
        np.asarray(behavior)[lesmat.kept_indices],
        mask=_mask_from_result(lesmat),
        **kwargs,
    )
    return StatisticalModelResult(
        method="sccan",
        preprocess=prep_opts,
        kept_indices=lesmat.kept_indices,
        lesion_matrix=lesmat,
        result=sccan_result,
    )


# ---------------------------------------------------------------------------
# Statistical analyses (non-predictive inference)
# ---------------------------------------------------------------------------


@dataclass
class StatisticalAnalysisResult(WorkflowResult):
    lesion_matrix: LesionMatrixResult
    result: Union[ChiSquareResult, TTestResult, BrunnerMunzelResult]


def run_statistical_analysis(
    images: Sequence[ImageLike],
    behavior: Sequence[float],
    *,
    method: str,
    preprocess: Optional[PreprocessOptions] = None,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> StatisticalAnalysisResult:
    """
    Execute a pure statistical analysis (chi-square, t-test, Brunner-Munzel).
    """
    prep_opts = deepcopy(preprocess) if preprocess is not None else PreprocessOptions()

    lesmat = _build_matrix(images, prep_opts)
    kwargs = deepcopy(method_kwargs) if method_kwargs else {}
    method_lower = method.lower()

    if method_lower in {"chi_square", "chisq"}:
        result = run_chi_square(
            lesmat.matrix,
            np.asarray(behavior)[lesmat.kept_indices],
            **kwargs,
        )
        label = "chi_square"
    elif method_lower in {"ttest", "welch"}:
        result = run_ttest(
            lesmat.matrix,
            np.asarray(behavior)[lesmat.kept_indices],
            equal_variance=(method_lower == "ttest"),
            **kwargs,
        )
        label = method_lower
    elif method_lower in {"brunner_munzel", "bm"}:
        result = run_brunner_munzel(
            lesmat.matrix,
            np.asarray(behavior)[lesmat.kept_indices],
            **kwargs,
        )
        label = "brunner_munzel"
    else:
        raise ValueError(f"Unknown statistical analysis method '{method}'.")

    return StatisticalAnalysisResult(
        method=label,
        preprocess=prep_opts,
        kept_indices=lesmat.kept_indices,
        lesion_matrix=lesmat,
        result=result,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_matrix(images: Sequence[ImageLike], options: PreprocessOptions) -> LesionMatrixResult:
    return build_lesion_matrix(
        images,
        mask=options.mask,
        binarize=options.binarize,
        min_voxel_lesion_count=options.min_voxel_lesion_count,
        drop_empty_rows=options.drop_empty_rows,
        apply_voxelwise_zscore=options.apply_voxelwise_zscore,
        apply_subjectwise_l2=options.apply_subjectwise_l2,
    )


def _mask_from_result(result: LesionMatrixResult) -> np.ndarray:
    mask = np.zeros(np.prod(result.volume_shape), dtype=bool)
    mask[result.feature_mask] = True
    return mask.reshape(result.volume_shape)
