#!/usr/bin/env python3
"""Simplex-constrained probability blend, fitted on validation only.

Claude's review warning: an unconstrained stacker on 3,459 val rows overfits.
This script uses non-negative weights that sum to one (a convex blend), fits
them by validation log-loss, estimates generalisation with 5-fold CV on val,
and evaluates the frozen blend on test exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "ensemble" / "results"
OUT.mkdir(parents=True, exist_ok=True)

TRANSFORMERS = {"muril_fullft", "telugu_bert_lora", "muril_lora", "indicbert_lora",
                "xlmr_lora", "distilmbert_lora", "mbert_lora"}
CLASSICAL_FINAL = {"logistic_regression", "linear_svc", "soft_voting", "multinomial_nb",
                   "extra_trees", "random_forest", "lightgbm", "hist_gradient_boosting",
                   "xgboost", "knn", "dummy"}

SHORTLIST = [
    "muril_fullft",
    "telugu_bert_lora",
    "muril_lora",
    "logreg_union",
    "linearsvc_union",
    "soft_voting",
    "linear_svc",
    "logreg_char25",
]

PRIOR6 = ["muril_fullft", "linearsvc_union", "logreg_union", "muril_lora",
          "soft_voting", "linear_svc"]


def load(tag: str) -> dict:
    cands = [
        BENCH / "ensemble" / "preds_val" / tag / "preds.npz",
        BENCH / "ensemble" / "preds" / tag / "preds.npz",
        BENCH / "classical" / tag / "results" / "CMTE-NITANP" / "final" / "seed42" / "preds.npz",
    ]
    for p in cands:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            return {k: d[k] for k in d.files}
    raise FileNotFoundError(tag)


def stack(tags: list[str]):
    data = {t: load(t) for t in tags}
    ref = data[tags[0]]
    v_ids = [str(x) for x in ref["val_ids"]]
    t_ids = [str(x) for x in ref["test_ids"]]
    v_idx = {v: i for i, v in enumerate(v_ids)}
    t_idx = {v: i for i, v in enumerate(t_ids)}
    yv = np.asarray(ref["val_gold"])
    yt = np.asarray(ref["test_gold"])
    Pv, Pt = [], []
    for t in tags:
        d = data[t]
        # Older classical preds.npz files omit val_ids; the split order is fixed.
        vo = ([v_idx[str(x)] for x in d["val_ids"]] if "val_ids" in d
              else list(range(len(d["val_gold"]))))
        to = ([t_idx[str(x)] for x in d["test_ids"]] if "test_ids" in d
              else list(range(len(d["test_gold"]))))
        Pv.append(np.asarray(d["val_proba"], float)[vo])
        Pt.append(np.asarray(d["test_proba"], float)[to])
        assert np.array_equal(np.asarray(d["val_gold"])[vo], yv), f"val order {t}"
        assert np.array_equal(np.asarray(d["test_gold"])[to], yt), f"test order {t}"
    return np.stack(Pv, 1), np.stack(Pt, 1), yv, yt


def f1_of(w, P, y):
    return f1_score(y, np.einsum("i,nij->nj", w, P).argmax(1), average="macro")


def acc_of(w, P, y):
    return accuracy_score(y, np.einsum("i,nij->nj", w, P).argmax(1))


def neg_logloss(w, P, y):
    p = np.clip(np.einsum("i,nij->nj", w, P), 1e-9, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def fit_weights(P, y, starts: int = 6):
    n = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n
    best = None
    rng = np.random.default_rng(42)
    inits = [np.ones(n) / n] + [rng.dirichlet(np.ones(n) * 0.7) for _ in range(starts)]
    for w0 in inits:
        r = minimize(neg_logloss, w0, args=(P, y), method="SLSQP",
                     bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return best.x if best is not None else np.ones(n) / n


def evaluate(tags, label):
    Pv, Pt, yv, yt = stack(tags)
    n = len(tags)
    w_u = np.ones(n) / n
    w = fit_weights(Pv, yv)
    # 5-fold CV estimate of the weight-fitting procedure
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv = []
    for tr, va in skf.split(Pv[:, 0, 0], yv):
        wf = fit_weights(Pv[tr], yv[tr])
        cv.append(f1_of(wf, Pv[va], yv[va]))
    rows = [
        (label, "uniform", w_u, f1_of(w_u, Pv, yv), f1_of(w_u, Pt, yt), acc_of(w_u, Pt, yt)),
        (label, "simplex_logloss", w, f1_of(w, Pv, yv), f1_of(w, Pt, yt), acc_of(w, Pt, yt)),
    ]
    print(f"\n{label}: {tags}")
    for _, name, ww, vf, tf, ta in rows:
        print(f"  {name:16s} val_f1={vf:.4f} test_f1={tf:.4f} test_acc={ta:.4f} "
              f"w={np.round(ww, 3)}")
    print(f"  simplex 5-fold val macro-F1 = {np.mean(cv):.4f} +/- {np.std(cv):.4f}")
    return rows, w, np.mean(cv), np.std(cv), Pv, Pt, yv, yt


if __name__ == "__main__":
    all_rows = []
    for label, tags in [("prior6", PRIOR6), ("shortlist8", SHORTLIST)]:
        rows, w, cv, sd, Pv, Pt, yv, yt = evaluate(tags, label)
        all_rows += [dict(ensemble=r[0], method=r[1], val_f1=round(r[3], 4),
                          test_f1=round(r[4], 4), test_acc=round(r[5], 4),
                          cv_val_f1=round(cv, 4), cv_val_std=round(sd, 4),
                          weights={t: round(float(x), 4) for t, x in zip(tags, w)})
                     for r in rows]
    (OUT / "blend_results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {OUT/'blend_results.json'}")
