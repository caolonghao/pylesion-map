
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

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

# Optional imports guarded
try:
    import SimpleITK as sitk
except Exception:
    sitk = None  # type: ignore

try:
    from ants.core.ants_image import ANTsImage  # type: ignore[attr-defined]
except Exception:
    ANTsImage = None  # type: ignore

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

if TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage as _ANTsImageType  # pragma: no cover
else:
    _ANTsImageType = Any  # pragma: no cover


ImageLike = Union[np.ndarray, "sitk.Image", "_ANTsImageType", str, Path]


# ---------------------------
# Utility helpers
# ---------------------------

def _to_numpy3d(img: ImageLike) -> np.ndarray:
    """Convert numpy, file path, SimpleITK, or ANTs image to a 3D numpy array (z, y, x)."""
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, (str, Path)):
        path = Path(img)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")
        if ANTsImage is not None:
            try:
                import ants  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "antspy is required to read images from disk when using ANTs backend."
                ) from exc
            arr = ants.image_read(str(path)).numpy()
        else:
            if sitk is None:
                raise RuntimeError(
                    "Neither antspy nor SimpleITK is available to load image paths."
                )
            img_obj = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img_obj)
    elif ANTsImage is not None and isinstance(img, ANTsImage):
        arr = img.numpy()
    else:
        if sitk is None or not isinstance(img, sitk.Image):
            raise TypeError(
                "Input must be numpy.ndarray, str/Path to an image, SimpleITK.Image, or ants.core.ANTsImage."
            )
        arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _flatten_with_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten a volume using a boolean mask -> 1D feature vector."""
    return vol[mask].ravel()


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-wise L2 normalization with epsilon (dTLVC).
    Returns normalized X and original row norms (TLV proxies).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, eps)
    Xn = X / norms_safe
    return Xn, norms.squeeze(1)


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
    """
    min_lesion_count: int = 1
    brain_mask: Optional[np.ndarray] = None
    keep_empty_subjects: bool = True

    # Populated after `fit_transform`
    vol_shape_: Optional[Tuple[int, int, int]] = None
    feat_mask_: Optional[np.ndarray] = None  # 1D boolean mask after flatten
    kept_idx_: Optional[np.ndarray] = None   # indices of samples retained (if some were dropped)

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
        arrs = [ _to_numpy3d(im) for im in imgs ]
        self.vol_shape_ = arrs[0].shape

        # Optional binarization (typical for lesion masks)
        if binarize:
            arrs = [(a > 0).astype(np.float32) for a in arrs]

        # Build base mask
        if self.brain_mask is None:
            union = np.zeros(self.vol_shape_, dtype=bool)
            for a in arrs:
                union |= (a > 0)
            base_mask = union
        else:
            if self.brain_mask.shape != self.vol_shape_:
                raise ValueError("brain_mask shape mismatch")
            base_mask = self.brain_mask.astype(bool)

        # Apply min_lesion_count (valid-voxel mask)
        stack = np.stack([(a > 0) for a in arrs], axis=0)  # (n, z, y, x)
        lesion_counts = stack.sum(axis=0)
        valid_mask = base_mask & (lesion_counts >= self.min_lesion_count)

        # Flatten features
        X = np.stack([_flatten_with_mask(a, valid_mask) for a in arrs], axis=0)  # (n, p)

        # Optionally drop empty rows (all zeros after mask)
        row_nonzero = (X.sum(axis=1) > 0)
        if not self.keep_empty_subjects:
            kept_idx = np.where(row_nonzero)[0]
        else:
            kept_idx = np.arange(X.shape[0])
        X = X[kept_idx]
        y_kept = y[kept_idx]

        # dTLVC (unit-norm per subject vector)
        Xn, norms = _unit_norm_rows(X)

        # Save masks
        self.feat_mask_ = valid_mask.ravel()
        self.kept_idx_ = kept_idx

        return Xn, y_kept

    def transform_single(self, img: ImageLike, binarize: bool = True) -> np.ndarray:
        """Vectorize one image (numpy/SimpleITK/ANTs/path) and dTLVC-normalize it."""
        if self.vol_shape_ is None or self.feat_mask_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")
        a = _to_numpy3d(img)
        if a.shape != self.vol_shape_:
            raise ValueError(f"Image shape {a.shape} != training shape {self.vol_shape_}")
        if binarize:
            a = (a > 0).astype(np.float32)
        x = a.ravel()[self.feat_mask_].astype(np.float32, copy=False)
        # dTLVC on the single vector
        n = np.linalg.norm(x)
        if n > 1e-12:
            x = x / n
        return x

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
