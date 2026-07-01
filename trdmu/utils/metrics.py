from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def best_f1_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    y_true = y_true.astype(np.int8)
    scores = scores.astype(float)
    if len(scores) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    best_key = None
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int8)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        key = (float(f1), float(recall), float(precision), -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best_threshold, best_metrics


def binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, object]:
    y_true = y_true.astype(np.int8)
    scores = scores.astype(float)
    pred = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out: Dict[str, object] = {
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "predicted_positives": int(pred.sum()),
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if len(y_true) and np.unique(y_true).size == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out
