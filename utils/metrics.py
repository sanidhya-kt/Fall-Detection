"""
utils/metrics.py
=================
Accuracy / Precision / Recall / F1 exactly as defined in Section 5.2.2.
Binary classification: class 1 = Fall, class 0 = Non-fall (ADL).
"""
import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred) -> dict:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    eps = 1e-12
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "specificity": specificity * 100,
    }


def fmt_metrics(m: dict) -> str:
    return (f"Acc={m['accuracy']:.2f}% Prec={m['precision']:.2f}% "
            f"Recall={m['recall']:.2f}% F1={m['f1']:.2f}%")
