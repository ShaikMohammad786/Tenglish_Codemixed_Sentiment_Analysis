#!/usr/bin/env python3
"""Label-noise audit + cleaning study for CMTE-NITANP Split B.

What this does
--------------
1. Builds TF-IDF word+char features on the *training* split.
2. Gets 5-fold out-of-fold (OOF) LinearSVC decisions on the training rows.
3. Flags training rows where the OOF model confidently disagrees with the gold label.
4. Compares four training policies on the untouched test set:
   - baseline: all training rows
   - remove: drop flagged rows
   - relabel: replace flagged labels with the OOF prediction
   - downweight: keep flagged rows with weight 0.2
5. Adds a clean+PLD multi-view model, and a small C search on validation.
6. Reports a separate *model-consensus audited test* number. This is NOT human
   ground truth and is never used to select the model.

The test set is never relabeled for the main results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.data import load_records, to_arrays  # noqa: E402
from common.paths import SPLITS  # noqa: E402

OUT = Path(__file__).resolve().parent / "label_audit_results.json"
DATASET = "CMTE-NITANP"
SPLIT = "B"
SEED = 42


def union_vectorizer():
    return FeatureUnion([
        ("w", TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True, max_features=400000)),
    ])


def matrix(view: dict, fit: bool = False):
    vec = union_vectorizer()
    if fit:
        return vec, vec.fit_transform(view["train"]), vec.transform(view["val"]), vec.transform(view["test"])
    return vec.transform(view["train"]), vec.transform(view["val"]), vec.transform(view["test"])


def score(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def fit_eval(Xtr, ytr, Xte, yte, C=0.1, weights=None):
    clf = CalibratedClassifierCV(LinearSVC(C=C, max_iter=5000, random_state=SEED),
                                 cv=3, method="sigmoid")
    if weights is None:
        clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr, sample_weight=weights)
    pred = clf.predict(Xte)
    return clf, score(yte, pred), pred


def main():
    recs = load_records(DATASET, SPLIT)
    arr = {s: to_arrays(recs[s], "text_clean") for s in SPLITS}
    norm = {s: to_arrays(recs[s], "text_norm") for s in SPLITS}
    texts = {s: arr[s][0] for s in SPLITS}
    texts_norm = {s: norm[s][0] for s in SPLITS}
    y = {s: arr[s][1] for s in SPLITS}
    ids = {s: arr[s][3] for s in SPLITS}

    # --- clean view ---------------------------------------------------------
    vec, Xtr, Xva, Xte = matrix(texts, fit=True)
    print("clean features:", Xtr.shape, Xva.shape, Xte.shape)

    # --- OOF audit on training rows only -----------------------------------
    pipe = Pipeline([("vec", union_vectorizer()), ("clf", LinearSVC(C=0.1, max_iter=5000, random_state=SEED))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    dec = cross_val_predict(pipe, texts["train"], y["train"], cv=cv,
                            method="decision_function", n_jobs=1)
    prob = softmax(dec, axis=1)
    oof_pred = prob.argmax(1)
    oof_conf = prob.max(1)
    wrong = oof_pred != y["train"]
    flag = wrong & (oof_conf >= 0.70)
    print(f"train={len(y['train'])} wrong={wrong.sum()} flagged={flag.sum()} ({100*flag.mean():.2f}%)")

    # --- test policies ------------------------------------------------------
    results = {}
    _, results["baseline"], base_pred = fit_eval(Xtr, y["train"], Xte, y["test"], C=0.1)

    keep = ~flag
    _, results["remove"], _ = fit_eval(Xtr[keep], y["train"][keep], Xte, y["test"], C=0.1)

    y_relab = y["train"].copy()
    y_relab[flag] = oof_pred[flag]
    _, results["relabel"], _ = fit_eval(Xtr, y_relab, Xte, y["test"], C=0.1)

    w = np.ones(len(y["train"]))
    w[flag] = 0.2
    _, results["downweight"], _ = fit_eval(Xtr, y["train"], Xte, y["test"], C=0.1, weights=w)

    # --- validation C search on the cleaned/relabelled view -----------------
    best = None
    for C in (0.03, 0.1, 0.3):
        _, val_score, _ = fit_eval(Xtr[keep], y["train"][keep], Xva, y["val"], C=C)
        if best is None or val_score["macro_f1"] > best[1]["macro_f1"]:
            best = (C, val_score)
    C_star = best[0]
    _, results["remove_valC"], _ = fit_eval(Xtr[keep], y["train"][keep], Xte, y["test"], C=C_star)

    # --- multi-view: clean + PLD-normalised view ----------------------------
    vecn, Ntr, Nva, Nte = matrix(texts_norm, fit=True)
    Mtr = sparse.hstack([Xtr, Ntr]).tocsr()
    Mva = sparse.hstack([Xva, Nva]).tocsr()
    Mte = sparse.hstack([Xte, Nte]).tocsr()
    _, results["multiview"], _ = fit_eval(Mtr, y["train"], Mte, y["test"], C=0.1)
    _, results["multiview_remove"], _ = fit_eval(Mtr[keep], y["train"][keep], Mte, y["test"], C=0.1)

    # --- model-consensus test audit (NOT ground truth) ----------------------
    # Use existing final seed42 predictions, one per model family/config.
    pred_files = sorted(BENCH.glob("classical/*/results/CMTE-NITANP/final/seed42/preds.npz"))
    pred_files += sorted(BENCH.glob("deep/*/results/CMTE-NITANP/final/seed42/preds.npz"))
    pred_files += sorted(BENCH.glob("hf_transformers/*/results/CMTE-NITANP/final/seed42/preds.npz"))
    votes = []
    ref_ids = None
    ref_gold = None
    for p in pred_files:
        d = np.load(p, allow_pickle=True)
        if "test_pred" not in d or "test_ids" not in d:
            continue
        ids_p = d["test_ids"].astype(str)
        if ref_ids is None:
            ref_ids = ids_p
            ref_gold = d["test_gold"].astype(int)
        if not np.array_equal(ids_p, ref_ids):
            continue
        votes.append(d["test_pred"].astype(int))
    audit = None
    if votes:
        V = np.stack(votes, axis=1)
        maj = np.apply_along_axis(lambda r: np.bincount(r, minlength=3).argmax(), 1, V)
        frac = np.array([np.bincount(r, minlength=3).max() / len(r) for r in V])
        audit_flag = (frac >= 0.80) & (maj != ref_gold)
        audit = {
            "n_models": int(V.shape[1]),
            "n_consensus_flags": int(audit_flag.sum()),
            "fraction": float(audit_flag.mean()),
            "original_test": score(ref_gold, base_pred),
            "audited_test": score(maj, base_pred),
            "flag_overlap_with_base_errors": int(((base_pred != ref_gold) & audit_flag).sum()),
            "note": "Audited labels are model-consensus, not human ground truth; never used for model selection.",
        }

    payload = {
        "dataset": DATASET,
        "split": SPLIT,
        "train_rows": int(len(y["train"])),
        "flagged_train_rows": int(flag.sum()),
        "flagged_fraction": float(flag.mean()),
        "oof_wrong_rows": int(wrong.sum()),
        "C_star_on_clean": C_star,
        "test_original_labels": results,
        "model_consensus_test_audit": audit,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
