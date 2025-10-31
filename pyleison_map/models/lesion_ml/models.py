"""
Concrete sklearn/xgboost wrappers for lesion ML workflows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR

# Optional dependency
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - runtime optional
    xgb = None  # type: ignore

from .base import LesionModelBase

__all__ = [
    "LassoRegression",
    "LinearSVRModel",
    "LinearSVMClassifier",
    "L1LogisticClassifier",
    "RandomForestModel",
    "AdaBoostModel",
    "XGBoostModel",
    "SVR_RBF_Model",
    "RBF_SVM_OVR_Classifier",
    "make_model",
    "MODEL_REGISTRY",
]


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
        super().__init__(
            LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000),
            task="classification",
            **kwargs,
        )

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
        return imp.astype(np.float32, copy=False)


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
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        if xgb is None:
            raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
        if task == "regression":
            est = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
            )
        else:
            est = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                objective="binary:logistic",
            )
        super().__init__(estimator=est, task=task, **kwargs)

    def _beta_vector(self) -> np.ndarray:
        try:
            imp = self.estimator.feature_importances_
        except Exception as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(f"Failed to get XGBoost feature_importances_: {exc}") from exc
        return np.asarray(imp, dtype=np.float32)


class SVR_RBF_Model(LesionModelBase):
    """
    SVR with RBF kernel and approximate beta-map via preimage/sensitivity.
    """

    def __init__(self, C: float = 30.0, gamma: float = 5.0, epsilon: float = 0.1, **kwargs):
        super().__init__(SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf"), task="regression", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        sv = getattr(self.estimator, "support_vectors_", None)
        dual = getattr(self.estimator, "dual_coef_", None)
        if sv is None or dual is None or not hasattr(self.estimator, "gamma"):
            raise RuntimeError("SVR not fitted or missing attributes (sv, dual, or gamma).")

        gamma_val = float(self.estimator.gamma)
        coeff = dual.ravel().astype(np.float64)
        beta_base = coeff @ sv
        beta = beta_base * 2.0 * gamma_val
        return beta.astype(np.float32, copy=False)


class RBF_SVM_OVR_Classifier(LesionModelBase):
    """
    Nonlinear SVM classifier with RBF kernel using One-vs-Rest (OVR).
    Provides an approximate per-class beta-map by summing dual coefficients times support vectors.
    """

    def __init__(self, C: float = 1.0, gamma: float = "scale", **kwargs):
        base = OneVsRestClassifier(
            SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
        )
        super().__init__(base, task="classification", **kwargs)

    def _beta_vector(self) -> np.ndarray:
        est = self.estimator
        if not hasattr(est, "estimators_"):
            raise RuntimeError("Estimator not fitted.")

        rows = []
        for bin_est in est.estimators_:
            sv = getattr(bin_est, "support_vectors_", None)
            dual = getattr(bin_est, "dual_coef_", None)
            if sv is None or dual is None:
                if sv is None and self.prep.feat_mask_ is not None:
                    rows.append(np.zeros(int(self.prep.feat_mask_.sum()), dtype=np.float32))
                elif rows:
                    rows.append(np.zeros_like(rows[0]))
                else:
                    raise RuntimeError("Cannot infer feature length for beta map.")
                continue
            beta = (dual @ sv).ravel().astype(np.float32, copy=False)
            rows.append(beta)
        return np.vstack(rows)


def make_model(
    kind: str,
    task: str,
    **kwargs,
) -> LesionModelBase:
    """
    Factory to build a lesion model:
        kind ∈ {"lasso", "linear_svr", "linear_svm", "logistic_l1", "rf", "ada", "xgb", "svr_rbf"}
        task ∈ {"regression", "classification"}
    """
    kind_lower = kind.lower()
    if kind_lower == "lasso":
        if task != "regression":
            raise ValueError("Lasso is regression-only.")
        return LassoRegression(**kwargs)
    if kind_lower == "linear_svr":
        if task != "regression":
            raise ValueError("linear_svr is regression-only.")
        return LinearSVRModel(**kwargs)
    if kind_lower == "linear_svm":
        if task != "classification":
            raise ValueError("linear_svm is classification-only.")
        return LinearSVMClassifier(**kwargs)
    if kind_lower == "logistic_l1":
        if task != "classification":
            raise ValueError("logistic_l1 is classification-only.")
        return L1LogisticClassifier(**kwargs)
    if kind_lower == "rf":
        return RandomForestModel(task=task, **kwargs)
    if kind_lower == "ada":
        return AdaBoostModel(task=task, **kwargs)
    if kind_lower == "xgb":
        return XGBoostModel(task=task, **kwargs)
    if kind_lower == "svr_rbf":
        if task != "regression":
            raise ValueError("svr_rbf is regression-only.")
        return SVR_RBF_Model(**kwargs)
    raise ValueError(f"Unknown kind={kind}")


MODEL_REGISTRY: Dict[str, Any] = {
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
