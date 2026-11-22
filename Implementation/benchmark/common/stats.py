"""Statistical tests matched to the resampling design."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import f1_score


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_ci(y_true, y_pred, metric=_macro_f1, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(metric(y_true[idx], y_pred[idx]))
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
    }


def bootstrap_diff_ci(y_true, pred_a, pred_b, metric=_macro_f1, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    N = len(y)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    return {
        "mean": float(diffs.mean()),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
    }


def mcnemar(y_true, pred_a, pred_b, exact: bool = False) -> dict:
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    total = n01 + n10
    if total == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p": 1.0}
    if exact or total < 25:
        p = float(sps.binomtest(n01, total, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        stat = float((abs(n01 - n10) - 1) ** 2 / total)
        p = float(sps.chi2.sf(stat, 1))
    return {"n01": n01, "n10": n10, "stat": stat, "p": p}


def nadeau_bengio(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test for repeated CV/resampling (Nadeau & Bengio)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    J = len(a)
    if J < 2:
        return {"stat": float("nan"), "p": float("nan")}
    diff = a - b
    dbar = float(diff.mean())
    var = float(diff.var(ddof=1))
    denom = np.sqrt(var * (1.0 / J + n_test / n_train))
    if denom == 0:
        return {"stat": float("nan"), "p": float("nan")}
    stat = dbar / denom
    p = float(2 * sps.t.sf(abs(stat), df=J - 1))
    return {"stat": float(stat), "p": p}


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * p[i])
        v = max(v, prev)
        adj[i] = v
        prev = v
    return adj.tolist()
