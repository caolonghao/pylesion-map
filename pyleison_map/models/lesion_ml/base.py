"""
Shared base class infrastructure for lesion ML models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from pyleison_map.preprocess.lesion_matrix import ImageLike

from .preprocessing import LesionPreprocessor

__all__ = [
    "LesionModelBase",
]


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
        grp1 = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(X.shape[1], dtype=X.dtype)
        grp0 = X[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(X.shape[1], dtype=X.dtype)
        diff = grp1 - grp0
    signs = np.sign(diff, dtype=X.dtype)
    return signs.astype(np.float32, copy=False)


class LesionModelBase:
    """
    Base class wrapping a sklearn/xgboost model with shared preprocessing,
    fit/predict APIs, and beta-map export utilities.

    Subclasses must implement `_beta_vector()` to return a 1D vector of length n_features.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        task: str,
        min_lesion_count: int = 1,
        brain_mask: Optional[np.ndarray] = None,
        keep_empty_subjects: bool = True,
        voxelwise_zscore: bool = False,
        subjectwise_l2: bool = True,
        signed_importance_for_trees: bool = False,
    ):
        self.estimator = estimator
        self.task = task
        self.prep = LesionPreprocessor(
            min_lesion_count=min_lesion_count,
            brain_mask=brain_mask,
            keep_empty_subjects=keep_empty_subjects,
            voxelwise_zscore=voxelwise_zscore,
            subjectwise_l2=subjectwise_l2,
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
        x = self.prep.transform_single(img).reshape(1, -1)
        if self.task == "regression":
            pred = self.estimator.predict(x)[0]
            return float(pred)

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
        if beta.ndim == 2:
            if self.task == "classification" and getattr(self.estimator, "classes_", None) is not None:
                classes = getattr(self.estimator, "classes_")
                return {
                    cls: self.prep.vector_to_volume(beta[i])
                    for i, cls in enumerate(classes)
                }
            vec = np.mean(beta, axis=0)
            return self.prep.vector_to_volume(vec)
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
        from .persistence import save_model

        save_model(self, directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "LesionModelBase":
        """
        Restore a saved model from disk.
        """
        from .persistence import load_model

        return load_model(directory)
