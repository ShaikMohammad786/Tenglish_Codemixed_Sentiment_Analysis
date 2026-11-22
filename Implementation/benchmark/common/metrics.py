"""Metrics: macro-F1, per-class, MCC, ECE, Brier, risk-coverage."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .paths import LABEL_NAMES


def ece(y_true, probs, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def brier(y_true, probs) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def risk_coverage(y_true, probs, coverages=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> dict:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    order = np.argsort(-conf)
    n = len(y)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        out[str(c)] = {
            "coverage": k / n,
            "accuracy": float((pred[idx] == y[idx]).mean()),
            "macro_f1": float(f1_score(y[idx], pred[idx], average="macro", zero_division=0)),
        }
    return out


def compute_metrics(y_true, y_pred, probs=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
            }
            for i in range(3)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "report": classification_report(
            y_true, y_pred, digits=4, zero_division=0, target_names=LABEL_NAMES
        ),
    }
    if probs is not None:
        m["ece"] = ece(y_true, probs)
        m["brier"] = brier(y_true, probs)
        m["risk_coverage"] = risk_coverage(y_true, probs)
    return m
