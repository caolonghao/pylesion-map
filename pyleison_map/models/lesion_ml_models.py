
# -*- coding: utf-8 -*-
"""
lesion_models.py

A lightweight framework for lesion–symptom modeling with dTLVC (direct Total Lesion Volume Control)
and beta-map export, supporting both regression and classification models.

- Inputs: list of numpy/SimpleITK/ANTs images or paths (already normalized to a standard space; typically binary lesion masks)
         and targets y (continuous for regression; labels for classification)
- dTLVC: per-subject unit L2-norm normalization of lesion vectors (acts as total-lesion-volume control)
- Beta maps:
    * Linear models (Lasso, LinearSVR/SVC/Logistic): coefficient vector reshaped back to 3D volume
    * Tree/Boosting (RF, AdaBoost, XGBoost): feature_importances_ reshaped (optionally signed)
    * SVR-RBF (optional): preimage/sensitivity-style approximation using dual_coef_ ⊤ support_vectors_
      to obtain a voxelwise weight vector (approximate "beta map")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import joblib

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVR, LinearSVC, SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.utils.validation import check_is_fitted

from sklearn.multiclass import OneVsRestClassifier

# XGBoost is optional
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

from pyleison_map.preprocess.lesion_matrix import (
    ImageLike,
    build_lesion_matrix,
    vectorize_image_to_mask,
)


def _signed_from_groups(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute a light-weight sign proxy per feature:
    - For regression: sign(mean(feature | y > median) - mean(feature | y <= median))
    - For binary classification: sign(mean(feature | y==1) - mean(feature | y == 0))
    """
    if y.dtype.kind in ("f", "i") and np.unique(y).size > 2:
        med = np.median(y)
        grp_hi = X[y > med].mean(axis=0) if np.any(y > med) else np.zeros(X.shape[1], dtype=X.dtype)
        grp_lo = X[y <= med].mean(axis=0) if np.any(y <= med) else np.zeros(X.shape[1], dtype=X.dtype)
        diff = grp_hi - grp_lo
    else:
        # assume binary labels {0,1} after encoding
        grp1 = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(X.shape[1], dtype=X.dtype)
        grp0 = X[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(X.shape[1], dtype=X.dtype)
        diff = grp1 - grp0
    signs = np.sign(diff, dtype=X.dtype)
    return signs.astype(np.float32, copy=False)


# ---------------------------
# Preprocessor with dTLVC
# ---------------------------

@dataclass
class LesionPreprocessor:
    """
    Handles masking, vectorization, and dTLVC normalization.

    Attributes
    ----------
    min_lesion_count : int
        Keep voxels that are lesioned in at least this many subjects.
    brain_mask : Optional[np.ndarray]
        3D boolean mask to restrict analysis (e.g., atlas-derived). If None, use union of lesions.
    keep_empty_subjects : bool
        Whether to keep samples with zero lesion after masking. If False, such samples are dropped.
    voxelwise_zscore : bool
        Apply per-voxel z-score normalization across subjects.
    subjectwise_l2 : bool
        Apply per-subject L2 normalization (dTLVC) to the flattened lesion vectors.
    """
    min_lesion_count: int = 1
    brain_mask: Optional[np.ndarray] = None
    keep_empty_subjects: bool = True
    voxelwise_zscore: bool = False
    subjectwise_l2: bool = True

    # Populated after `fit_transform`
    vol_shape_: Optional[Tuple[int, int, int]] = None
    feat_mask_: Optional[np.ndarray] = None  # 1D boolean mask after flatten
    kept_idx_: Optional[np.ndarray] = None   # indices of samples retained (if some were dropped)
    voxel_means_: Optional[np.ndarray] = None
    voxel_stds_: Optional[np.ndarray] = None

    def fit_transform(
        self,
        imgs: Sequence[ImageLike],
        y: np.ndarray,
        binarize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize lesion volumes, build feature mask, and apply dTLVC (unit L2 norm per row).
        Returns X_dtlvc (n, p_kept), y_kept.

        Parameters
        ----------
        imgs:
            Sequence of lesion volumes provided as numpy arrays, SimpleITK images,
            ANTs images, or filesystem paths to disk-backed images.
        """
        # Convert all to numpy (n, z, y, x)
        result = build_lesion_matrix(
            imgs,
            mask=self.brain_mask,
            binarize=binarize,
            min_voxel_lesion_count=self.min_lesion_count,
            drop_empty_rows=not self.keep_empty_subjects,
            apply_voxelwise_zscore=self.voxelwise_zscore,
            apply_subjectwise_l2=self.subjectwise_l2,
            dtype=np.float32,
        )

        self.vol_shape_ = result.volume_shape
        self.feat_mask_ = result.feature_mask
        self.kept_idx_ = result.kept_indices
        self.voxel_means_ = result.voxel_means
        self.voxel_stds_ = result.voxel_stds

        y_kept = y[result.kept_indices]

        return result.matrix.astype(np.float32, copy=False), y_kept

    def transform_single(self, img: ImageLike, binarize: bool = True) -> np.ndarray:
        """Vectorize one image (numpy/SimpleITK/ANTs/path) and dTLVC-normalize it."""
        if self.vol_shape_ is None or self.feat_mask_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")
        return vectorize_image_to_mask(
            img,
            feature_mask=self.feat_mask_,
            volume_shape=self.vol_shape_,
            binarize=binarize,
            apply_voxelwise_zscore=self.voxelwise_zscore,
            voxel_means=self.voxel_means_,
            voxel_stds=self.voxel_stds_,
            apply_subjectwise_l2=self.subjectwise_l2,
        )

    def vector_to_volume(self, vec: np.ndarray) -> np.ndarray:
        """Scatter a feature vector back to 3D using the feature mask."""
        if self.vol_shape_ is None or self.feat_mask_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        vol = np.zeros(int(np.prod(self.vol_shape_)), dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("vec must be 1D")
        vol[self.feat_mask_] = vec.astype(np.float32, copy=False)
        return vol.reshape(self.vol_shape_)


# ---------------------------
# Base model + subclasses
# ---------------------------

class LesionModelBase:
    """
    Base class wrapping a sklearn/xgboost model with:
    - preprocessing (masking + dTLVC)
    - fit/predict APIs for (regression | classification)
    - beta_map() export as a 3D numpy volume (or dict for multiclass)

    Subclasses must implement `_beta_vector()` to return a 1D vector of length n_features.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        task: str,  # "regression" | "classification"
        min_lesion_count: int = 1,
        brain_mask: Optional[np.ndarray] = None,
        keep_empty_subjects: bool = True,
        signed_importance_for_trees: bool = False,
    ):
        self.estimator = estimator
        self.task = task
        self.prep = LesionPreprocessor(
            min_lesion_count=min_lesion_count,
            brain_mask=brain_mask,
            keep_empty_subjects=keep_empty_subjects,
        )
        self.signed_importance_for_trees = signed_importance_for_trees
        self.label_encoder: Optional[LabelEncoder] = None
        self._is_fitted = False

    # ------------- core API -------------

    def fit(
        self,
        imgs: Sequence[ImageLike],
        y: Union[np.ndarray, List[Union[float, int, str]]],
        binarize: bool = True,
    ) -> "LesionModelBase":
        y_arr = np.asarray(y)
        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            y_arr = self.label_encoder.fit_transform(y_arr)
        Xn, y_kept = self.prep.fit_transform(imgs, y_arr, binarize=binarize)
        self.estimator.fit(Xn, y_kept)
        self._is_fitted = True
        return self

    def predict(self, img: ImageLike, return_proba: bool = False) -> Any:
        check_is_fitted(self.estimator)
        x = self.prep.transform_single(img)
        x = x.reshape(1, -1)
        if self.task == "regression":
            pred = self.estimator.predict(x)[0]
            return float(pred)
        else:
            if return_proba and hasattr(self.estimator, "predict_proba"):
                proba = self.estimator.predict_proba(x)[0]
                if self.label_encoder is not None:
                    labels = self.label_encoder.inverse_transform(np.arange(proba.shape[0]))
                    return dict(zip(map(str, labels), map(float, proba)))
                return proba
            pred = self.estimator.predict(x)[0]
            if self.label_encoder is not None:
                pred = self.label_encoder.inverse_transform([int(pred)])[0]
            return pred

    def beta_map(self) -> Union[np.ndarray, Dict[Any, np.ndarray]]:
        """
        Return a 3D beta/importance map (or class->map dict for multiclass linear classifiers).
        For tree/boosting models: returns importance volume (unsigned unless `signed_importance_for_trees=True`).
        """
        check_is_fitted(self.estimator)
        beta = self._beta_vector()
        # Linear multiclass may return 2D (n_classes, n_features)
        if beta.ndim == 2:
            if self.task == "classification" and getattr(self.estimator, "classes_", None) is not None:
                classes = getattr(self.estimator, "classes_")
                out = {}
                for i, cls in enumerate(classes):
                    out[cls] = self.prep.vector_to_volume(beta[i])
                return out
            else:
                vec = np.mean(beta, axis=0)
                return self.prep.vector_to_volume(vec)
        else:
            return self.prep.vector_to_volume(beta)

    # ------------- hooks -------------

    def _beta_vector(self) -> np.ndarray:
        raise NotImplementedError

    # ---------------------------
    # Persistence
    # ---------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """
        Persist the trained model, estimator, and preprocessing state to disk.
        """
        _save_ml_model(self, directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "LesionModelBase":
        """
        Restore a saved model from disk.
        """
        return load_model(directory)


# ---------------------------
# Concrete model wrappers
# ---------------------------

class LassoRegression(LesionModelBase):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(Lasso(alpha=alpha, max_iter=5000), task="regression", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            raise RuntimeError("Estimator not fitted.")
        return coef.astype(np.float32, copy=False)


class LinearSVRModel(LesionModelBase):
    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(LinearSVR(C=C, loss="epsilon_insensitive"), task="regression", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            raise RuntimeError("Estimator not fitted.")
        return coef.astype(np.float32, copy=False).ravel()


class LinearSVMClassifier(LesionModelBase):
    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(LinearSVC(C=C), task="classification", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            raise RuntimeError("Estimator not fitted.")
        return coef.astype(np.float32, copy=False)


class L1LogisticClassifier(LesionModelBase):
    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(LogisticRegression(
            penalty="l1", solver="saga", C=C, max_iter=3000
        ), task="classification", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            raise RuntimeError("Estimator not fitted.")
        return coef.astype(np.float32, copy=False)


class RandomForestModel(LesionModelBase):
    def __init__(self, n_estimators: int = 200, task: str = "regression", **kwargs):
        if task == "regression":
            est = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        else:
            est = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        super().__init__(estimator=est, task=task, **kwargs)

    def _beta_vector(self) -> np.ndarray:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            raise RuntimeError("Estimator not fitted or lacks feature_importances_.")
        vec = imp.astype(np.float32, copy=False)
        return vec


class AdaBoostModel(LesionModelBase):
    def __init__(self, n_estimators: int = 200, task: str = "regression", **kwargs):
        if task == "regression":
            est = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
        else:
            est = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        super().__init__(estimator=est, task=task, **kwargs)

    def _beta_vector(self) -> np.ndarray:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            raise RuntimeError("Estimator not fitted or lacks feature_importances_.")
        return imp.astype(np.float32, copy=False)


class XGBoostModel(LesionModelBase):
    def __init__(self, task: str = "regression", n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        if xgb is None:
            raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
        if task == "regression":
            est = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
            )
        else:
            est = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                objective="binary:logistic"
            )
        super().__init__(estimator=est, task=task, **kwargs)

    def _beta_vector(self) -> np.ndarray:
        try:
            imp = self.estimator.feature_importances_
        except Exception as e:
            raise RuntimeError(f"Failed to get XGBoost feature_importances_: {e}")
        return np.asarray(imp, dtype=np.float32)


class SVR_RBF_Model(LesionModelBase):
    """
    SVR with RBF kernel and approximate beta-map via preimage/sensitivity:
    beta ≈ sum_i k_i * x_i over support vectors (up to a scalar factor),
    where k_i are dual coefficients and x_i are support vectors in input space.

    This follows the preimage approximation discussed in SVR-LSM literature.
    """
    def __init__(self, C: float = 30.0, gamma: float = 5.0, epsilon: float = 0.1, **kwargs):
        super().__init__(SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf"), task="regression", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        sv = getattr(self.estimator, "support_vectors_", None)
        dual = getattr(self.estimator, "dual_coef_", None)
        
        # 检查SVR模型是否已经fit，并获取gamma值
        if sv is None or dual is None or not hasattr(self.estimator, "gamma"):
            raise RuntimeError("SVR not fitted or missing attributes (sv, dual, or gamma).")
        
        # 获取gamma的数值
        gamma_val = float(self.estimator.gamma) 
            
        k = dual.ravel().astype(np.float64)
        
        # beta_base = sum(lambda_i * x_i)
        beta_base = k @ sv  # (n_features,)
        
        # 应用论文附录中的 2*gamma 因子 
        beta = beta_base * 2.0 * gamma_val
        
        return beta.astype(np.float32, copy=False)



class RBF_SVM_OVR_Classifier(LesionModelBase):
    """
    Nonlinear SVM classifier with RBF kernel using One-vs-Rest (OVR).
    Provides an approximate per-class beta-map by summing dual coefficients times support vectors
    for each one-vs-rest binary SVC:
        beta_c ≈ dual_coef_c @ support_vectors_c
    Note: This is an interpretability approximation for visualization.
    """
    def __init__(self, C: float = 1.0, gamma: float = "scale", **kwargs):
        base = OneVsRestClassifier(
            SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
        )
        super().__init__(base, task="classification", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        # For each binary classifier in OVR: beta ≈ dual @ SV
        est = self.estimator
        if not hasattr(est, "estimators_"):
            raise RuntimeError("Estimator not fitted.")
        rows = []
        for bin_est in est.estimators_:
            sv = getattr(bin_est, "support_vectors_", None)
            dual = getattr(bin_est, "dual_coef_", None)
            if sv is None or dual is None:
                # If something is missing (e.g., no SV for a rare class), append zeros
                if sv is None and hasattr(self.prep, "feat_mask_") and self.prep.feat_mask_ is not None:
                    rows.append(np.zeros(int(self.prep.feat_mask_.sum()), dtype=np.float32))
                else:
                    # Fallback zero vector length unknown: infer from any previous row
                    if rows:
                        rows.append(np.zeros_like(rows[0]))
                    else:
                        raise RuntimeError("Cannot infer feature length for beta map.")
                continue
            beta = (dual @ sv).ravel().astype(np.float32, copy=False)
            rows.append(beta)
        return np.vstack(rows)


# ---------------------------
# Factory / convenience
# ---------------------------

# ---------------------------
# Factory / convenience
# ---------------------------

def make_model(
    kind: str,
    task: str,
    **kwargs
) -> LesionModelBase:
    """
    Factory to build a lesion model:
        kind ∈ {"lasso", "linear_svr", "linear_svm", "logistic_l1", "rf", "ada", "xgb", "svr_rbf"}
        task ∈ {"regression", "classification"}
    """
    kind = kind.lower()
    if kind == "lasso":
        if task != "regression":
            raise ValueError("Lasso is regression-only.")
        return LassoRegression(**kwargs)
    if kind == "linear_svr":
        if task != "regression":
            raise ValueError("linear_svr is regression-only.")
        return LinearSVRModel(**kwargs)
    if kind == "linear_svm":
        if task != "classification":
            raise ValueError("linear_svm is classification-only.")
        return LinearSVMClassifier(**kwargs)
    if kind == "logistic_l1":
        if task != "classification":
            raise ValueError("logistic_l1 is classification-only.")
        return L1LogisticClassifier(**kwargs)
    if kind == "rf":
        return RandomForestModel(task=task, **kwargs)
    if kind == "ada":
        return AdaBoostModel(task=task, **kwargs)
    if kind == "xgb":
        return XGBoostModel(task=task, **kwargs)
    if kind == "svr_rbf":
        if task != "regression":
            raise ValueError("svr_rbf is regression-only.")
        return SVR_RBF_Model(**kwargs)
    raise ValueError(f"Unknown kind={kind}")


_MODEL_REGISTRY: Dict[str, Any] = {
    cls.__name__: cls
    for cls in [
        LassoRegression,
        LinearSVRModel,
        LinearSVMClassifier,
        L1LogisticClassifier,
        RandomForestModel,
        AdaBoostModel,
        XGBoostModel,
        SVR_RBF_Model,
        RBF_SVM_OVR_Classifier,
    ]
}


def _save_ml_model(model: LesionModelBase, directory: Union[str, Path]) -> None:
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
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

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


def load_model(directory: Union[str, Path]) -> LesionModelBase:
    path = Path(directory)
    metadata_path = path / "model.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file at {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("model_type") != "lesion_ml":
        raise ValueError(f"Unexpected model_type {metadata.get('model_type')}")

    class_name = metadata["class_name"]
    if class_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{class_name}' in metadata.")

    cls = _MODEL_REGISTRY[class_name]
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

    brain_mask = None
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
