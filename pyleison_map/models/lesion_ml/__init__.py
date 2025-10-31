"""
Modular lesion ML model components (preprocessing, wrappers, persistence).
"""
from __future__ import annotations

from .preprocessing import LesionPreprocessor, preprocessing_kwargs_from_options
from .base import LesionModelBase
from .models import (
    AdaBoostModel,
    L1LogisticClassifier,
    LassoRegression,
    LinearSVMClassifier,
    LinearSVRModel,
    MODEL_REGISTRY,
    RBF_SVM_OVR_Classifier,
    RandomForestModel,
    SVR_RBF_Model,
    XGBoostModel,
    make_model,
)
from .persistence import load_model, save_model

__all__ = [
    "LesionPreprocessor",
    "preprocessing_kwargs_from_options",
    "LesionModelBase",
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
    "save_model",
    "load_model",
]
