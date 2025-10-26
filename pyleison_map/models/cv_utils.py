
# -*- coding: utf-8 -*-
"""
cv_utils.py

Cross-validation & grid-search helpers for lesion_models.py (dTLVC-based LSM framework).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)

# -------------------------------
# Metric utilities
# -------------------------------

def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Spearman/Pearson
    try:
        from scipy.stats import spearmanr, pearsonr
        sp = float(spearmanr(y_true, y_pred).correlation)
        pr = float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        sp, pr = np.nan, np.nan
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "pearson_r": float(pr),
        "spearman_r": float(sp),
    }


def _classification_metrics(y_true, y_pred, y_score=None, labels=None) -> Dict[str, float]:
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["bacc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    # Prob/score-based metrics (if available)
    if y_score is not None:
        try:
            if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
                out["ap"] = float(average_precision_score(y_true, y_score))
            else:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels))
                # log loss needs probs that sum to 1
                ssum = np.sum(y_score, axis=1, keepdims=True)
                probs = y_score / np.clip(ssum, 1e-12, None)
                out["log_loss"] = float(log_loss(y_true, probs, labels=labels))
        except Exception:
            pass
    return out


def _stack_results(per_fold: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*[d.keys() for d in per_fold]))
    avg = {k: float(np.nanmean([d.get(k, np.nan) for d in per_fold])) for k in keys}
    return avg


# -------------------------------
# Helpers for model building/param setting
# -------------------------------

def _clone_model(base_kind: str, task: str, base_kwargs: Dict[str, Any]) -> Any:
    return lesion_models.make_model(kind=base_kind, task=task, **base_kwargs)


def _apply_params(model, params: Dict[str, Any]) -> None:
    """
    Accepts keys like:
      - 'est__C': 1.0                    -> model.estimator.set_params(C=1.0)
      - 'prep__min_lesion_count': 10     -> model.prep.min_lesion_count = 10
      - 'signed_importance_for_trees': True -> setattr(model, 'signed_importance_for_trees', True)
    """
    if not params:
        return
    est_updates = {}
    for k, v in params.items():
        if k.startswith("est__"):
            est_updates[k[len("est__"):]] = v
        elif k.startswith("prep__"):
            attr = k[len("prep__"):]
            setattr(model.prep, attr, v)
        else:
            if hasattr(model, k):
                setattr(model, k, v)
            else:
                est_updates[k] = v
    if est_updates:
        try:
            model.estimator.set_params(**est_updates)
        except Exception:
            pass


# -------------------------------
# Cross-validation
# -------------------------------

def cross_validate_lesion_model(
    imgs: List[Any],
    y: np.ndarray,
    kind: str,
    task: str,
    base_kwargs: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """
    Run K-fold CV for a single (kind, task) model configuration.
    Returns:
      - 'per_fold': list of metric dicts
      - 'mean': averaged metric dict
    """
    base_kwargs = base_kwargs or {}
    y_arr = np.asarray(y)
    n = len(imgs)
    assert n == len(y_arr), "imgs/y length mismatch"

    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_iter = cv.split(np.arange(n), y_arr)
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_iter = cv.split(np.arange(n))

    fold_metrics = []
    for tr_idx, va_idx in split_iter:
        model = _clone_model(kind, task, base_kwargs)

        X_tr = [imgs[i] for i in tr_idx]
        y_tr = y_arr[tr_idx]
        X_va = [imgs[i] for i in va_idx]
        y_va = y_arr[va_idx]

        model.fit(X_tr, y_tr)

        if task == "regression":
            preds = np.array([model.predict(im) for im in X_va], dtype=float)
            fold_metrics.append(_regression_metrics(y_va, preds))
        else:
            y_pred = np.array([model.predict(im) for im in X_va])
            y_score = None
            # Try probas
            try:
                probs = []
                classes = getattr(model.estimator, "classes_", None)
                for im in X_va:
                    p = model.predict(im, return_proba=True)
                    if isinstance(p, dict) and classes is not None:
                        row = np.array([p.get(str(c), p.get(c, 0.0)) for c in classes], dtype=float)
                    else:
                        row = np.asarray(p, dtype=float)
                        if row.ndim == 0:
                            row = np.array([row], dtype=float)
                    probs.append(row)
                y_score = np.vstack(probs)
                if y_score.ndim == 2 and y_score.shape[1] == 2:
                    y_score = y_score[:, 1]
            except Exception:
                # Try decision_function (e.g., LinearSVC)
                try:
                    scores = []
                    for im in X_va:
                        x = model.prep.transform_single(im).reshape(1, -1)
                        s = model.estimator.decision_function(x)
                        scores.append(s)
                    y_score = np.vstack(scores)
                    if y_score.ndim == 2 and y_score.shape[1] == 1:
                        y_score = y_score.ravel()
                except Exception:
                    y_score = None

            fold_metrics.append(_classification_metrics(y_va, y_pred, y_score=y_score, labels=getattr(model.estimator, "classes_", None)))

    return {"per_fold": fold_metrics, "mean": _stack_results(fold_metrics)}


# -------------------------------
# Grid search (serial, simple)
# -------------------------------

def _param_grid_iter(param_grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(param_grid.keys())
    vals = [list(param_grid[k]) for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def grid_search_lesion_model(
    imgs: List[Any],
    y: np.ndarray,
    kind: str,
    task: str,
    base_kwargs: Optional[Dict[str, Any]],
    param_grid: Dict[str, Iterable[Any]],
    primary_metric: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    refit: bool = True,
) -> Dict[str, Any]:
    """
    Simple serial grid search. Param keys support:
      - 'est__*' for estimator params (e.g., 'est__C', 'est__alpha')
      - 'prep__*' for preprocessor params (e.g., 'prep__min_lesion_count')
      - top-level attributes (e.g., 'signed_importance_for_trees')
    Returns:
      - 'cv_results': pandas.DataFrame (per setting mean metrics)
      - 'best_params': dict
      - 'best_score': float
      - 'best_model': fitted model on ALL data with best params (if refit=True)
    """
    base_kwargs = base_kwargs or {}

    if primary_metric is None:
        primary_metric = "r2" if task == "regression" else "acc"
    greater_is_better = primary_metric not in {"mse", "rmse", "mae", "log_loss"}

    records = []
    best_score = -np.inf if greater_is_better else np.inf
    best_params = None
    best_model = None

    for params in _param_grid_iter(param_grid):
        model = _clone_model(kind, task, dict(base_kwargs))
        _apply_params(model, params)

        # NOTE: pass current preprocessor toggles via base_kwargs for CV clones
        cv_out = cross_validate_lesion_model(
            imgs=imgs, y=y, kind=kind, task=task,
            base_kwargs=dict(
                min_lesion_count=model.prep.min_lesion_count,
                brain_mask=model.prep.brain_mask,
                keep_empty_subjects=model.prep.keep_empty_subjects,
                signed_importance_for_trees=getattr(model, "signed_importance_for_trees", False),
                # estimator params are set inside fold via _apply_params on a fresh model if needed
            ),
            n_splits=n_splits, random_state=random_state, shuffle=shuffle
        )

        mean_metrics = cv_out["mean"]
        row = {"params": params}
        row.update(mean_metrics)
        records.append(row)

        score = mean_metrics.get(primary_metric, np.nan)
        if np.isnan(score):
            continue
        is_better = (score > best_score) if greater_is_better else (score < best_score)
        if is_better:
            best_score = score
            best_params = params
            if refit:
                best_model = _clone_model(kind, task, dict(base_kwargs))
                _apply_params(best_model, best_params)
                best_model.fit(imgs, y)

    df = pd.DataFrame.from_records(records)
    return {
        "cv_results": df,
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "primary_metric": primary_metric,
    }
