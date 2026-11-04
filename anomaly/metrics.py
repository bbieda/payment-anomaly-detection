"""Metryki dla danych niezbalansowanych. Accuracy celowo pominięte."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def best_f1_threshold(y_true, score) -> float:
    """Próg maksymalizujący F1 -- wyznaczany na zbiorze WALIDACYJNYM, nie testowym."""
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def evaluate(y_true, score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


def _thin(a, b, max_points):
    idx = np.unique(np.linspace(0, len(a) - 1, min(len(a), max_points)).astype(int))
    return np.asarray(a)[idx].tolist(), np.asarray(b)[idx].tolist()


def curves(y_true, score, max_points: int = 200) -> dict:
    """Krzywe ROC i PR (przerzedzone) -- gotowe do wysłania jako JSON na frontend."""
    fpr, tpr, _ = roc_curve(y_true, score)
    p, r, _ = precision_recall_curve(y_true, score)
    fpr, tpr = _thin(fpr, tpr, max_points)
    r, p = _thin(r, p, max_points)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"recall": r, "precision": p}}
