"""
Persistence helpers for lesion ML models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union, TYPE_CHECKING

import joblib
import numpy as np

from .models import MODEL_REGISTRY
from .preprocessing import LesionPreprocessor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import LesionModelBase

__all__ = ["save_model", "load_model"]


def save_model(model: "LesionModelBase", directory: Union[str, Path]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": 1,
        "model_type": "lesion_ml",
        "class_name": model.__class__.__name__,
        "task": model.task,
        "signed_importance_for_trees": bool(getattr(model, "signed_importance_for_trees", False)),
        "is_fitted": bool(getattr(model, "_is_fitted", False)),
        "preprocessor": {
            "min_lesion_count": int(model.prep.min_lesion_count),
            "keep_empty_subjects": bool(model.prep.keep_empty_subjects),
            "voxelwise_zscore": bool(model.prep.voxelwise_zscore),
            "subjectwise_l2": bool(model.prep.subjectwise_l2),
        },
        "label_encoder": {
            "has_classes": bool(
                model.label_encoder is not None and hasattr(model.label_encoder, "classes_")
            ),
        },
    }

    metadata_path = path / "model.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    estimator_path = path / "estimator.joblib"
    joblib.dump(model.estimator, estimator_path)

    state_arrays: Dict[str, np.ndarray] = {}
    if model.prep.vol_shape_ is not None:
        state_arrays["vol_shape"] = np.asarray(model.prep.vol_shape_, dtype=np.int64)
    if model.prep.feat_mask_ is not None:
        state_arrays["feat_mask"] = model.prep.feat_mask_.astype(np.uint8)
    if model.prep.kept_idx_ is not None:
        state_arrays["kept_idx"] = model.prep.kept_idx_.astype(np.int64)
    if model.prep.brain_mask is not None:
        state_arrays["brain_mask"] = model.prep.brain_mask.astype(np.uint8)
    if metadata["label_encoder"]["has_classes"]:
        state_arrays["label_classes"] = np.asarray(model.label_encoder.classes_)
    if model.prep.voxel_means_ is not None:
        state_arrays["voxel_means"] = model.prep.voxel_means_.astype(np.float32)
    if model.prep.voxel_stds_ is not None:
        state_arrays["voxel_stds"] = model.prep.voxel_stds_.astype(np.float32)

    state_path = path / "state.npz"
    if state_arrays:
        np.savez_compressed(state_path, **state_arrays)
    elif state_path.exists():
        state_path.unlink()


def load_model(directory: Union[str, Path]) -> "LesionModelBase":
    path = Path(directory)
    metadata_path = path / "model.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file at {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    if metadata.get("model_type") != "lesion_ml":
        raise ValueError(f"Unexpected model_type {metadata.get('model_type')}")

    class_name = metadata["class_name"]
    if class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{class_name}' in metadata.")

    cls = MODEL_REGISTRY[class_name]
    model = cls.__new__(cls)  # type: ignore[misc]

    estimator_path = path / "estimator.joblib"
    if not estimator_path.exists():
        raise FileNotFoundError(f"Missing estimator file at {estimator_path}")
    estimator = joblib.load(estimator_path)

    state_path = path / "state.npz"
    state_arrays: Dict[str, np.ndarray] = {}
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state_data:
            for key in state_data.files:
                state_arrays[key] = state_data[key]

    brain_mask: Optional[np.ndarray] = None
    if "brain_mask" in state_arrays and state_arrays["brain_mask"].size > 0:
        brain_mask = state_arrays["brain_mask"].astype(bool)

    preproc_meta = metadata["preprocessor"]
    preproc = LesionPreprocessor(
        min_lesion_count=int(preproc_meta["min_lesion_count"]),
        brain_mask=brain_mask,
        keep_empty_subjects=bool(preproc_meta["keep_empty_subjects"]),
        voxelwise_zscore=bool(preproc_meta.get("voxelwise_zscore", False)),
        subjectwise_l2=bool(preproc_meta.get("subjectwise_l2", True)),
    )

    if "vol_shape" in state_arrays and state_arrays["vol_shape"].size > 0:
        preproc.vol_shape_ = tuple(int(x) for x in state_arrays["vol_shape"].tolist())  # type: ignore[assignment]
    if "feat_mask" in state_arrays and state_arrays["feat_mask"].size > 0:
        preproc.feat_mask_ = state_arrays["feat_mask"].astype(bool)
    if "kept_idx" in state_arrays and state_arrays["kept_idx"].size > 0:
        preproc.kept_idx_ = state_arrays["kept_idx"].astype(np.int64)
    if "voxel_means" in state_arrays:
        preproc.voxel_means_ = state_arrays["voxel_means"]
    if "voxel_stds" in state_arrays:
        preproc.voxel_stds_ = state_arrays["voxel_stds"]

    label_info = metadata.get("label_encoder", {})
    if label_info.get("has_classes") and "label_classes" in state_arrays:
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        label_encoder.classes_ = state_arrays["label_classes"]
    else:
        label_encoder = None

    model.estimator = estimator  # type: ignore[attr-defined]
    model.task = metadata["task"]  # type: ignore[attr-defined]
    model.prep = preproc  # type: ignore[attr-defined]
    model.signed_importance_for_trees = metadata.get("signed_importance_for_trees", False)  # type: ignore[attr-defined]
    model.label_encoder = label_encoder  # type: ignore[attr-defined]
    model._is_fitted = metadata.get("is_fitted", False)  # type: ignore[attr-defined]

    return model
