#!/usr/bin/env python3
"""Train the strong word+char linear variants that were measured in the
feature ablation but never had their predictions saved.

These are cheap CPU models (seconds to a couple of minutes) and add genuine
diversity to the full-FT / LoRA ensemble: the union and char views are the
strongest non-neural configurations (test 0.8324 / 0.8276).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "classical"))

from common.data import load_records, to_arrays  # noqa: E402
from common.paths import SPLITS  # noqa: E402
import run_classical as rc  # noqa: E402

DATASET = "CMTE-NITANP"
SPLIT = "B"
TEXT_FIELD = "text_clean"
OUT = BENCH / "ensemble" / "preds"


def train(tag: str, model: str, feature: str, params: dict, text_field: str = TEXT_FIELD) -> None:
    recs = load_records(DATASET, SPLIT)
    arr = {s: to_arrays(recs[s], text_field) for s in SPLITS}
    texts = {s: arr[s][0] for s in SPLITS}
    y = {s: arr[s][1] for s in SPLITS}
    ids = {s: arr[s][3] for s in SPLITS}

    X, meta, cached = rc.build_features(texts, feature, f"{DATASET}:{SPLIT}:{text_field}")
    clf = rc.build_model(model, params, 42).fit(X["train"], y["train"])

    val_proba = clf.predict_proba(X["val"])
    test_proba = clf.predict_proba(X["test"])
    val_pred = val_proba.argmax(1)
    test_pred = test_proba.argmax(1)

    rd = OUT / tag
    rd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        rd / "preds.npz",
        val_pred=val_pred,
        val_proba=val_proba,
        val_gold=y["val"],
        val_ids=np.array([str(i) for i in ids["val"]], dtype=object),
        test_pred=test_pred,
        test_proba=test_proba,
        test_gold=y["test"],
        test_ids=np.array([str(i) for i in ids["test"]], dtype=object),
    )
    vf = f1_score(y["val"], val_pred, average="macro")
    tf = f1_score(y["test"], test_pred, average="macro")
    ta = accuracy_score(y["test"], test_pred)
    print(f"{tag:28s} features={meta['n_features']:>7,} cached={cached} "
          f"val_f1={vf:.4f} test_f1={tf:.4f} test_acc={ta:.4f}")


if __name__ == "__main__":
    train("logreg_union", "logistic_regression", "union", {"C": 1.0})
    train("logreg_char25", "logistic_regression", "char25", {"C": 1.0})
    train("linearsvc_union", "linear_svc", "union", {"C": 0.1})
    train("softvote_union", "soft_voting", "union", {})
    # Fair preprocessing comparison: same linear models on PLD-normalized text.
    train("logreg_union_norm", "logistic_regression", "union", {"C": 1.0}, text_field="text_norm")
    train("linearsvc_union_norm", "linear_svc", "union", {"C": 0.1}, text_field="text_norm")
    train("logreg_word11_norm", "logistic_regression", "word11", {"C": 1.0}, text_field="text_norm")
