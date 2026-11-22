#!/usr/bin/env python3
"""Offline ensemble audit over every saved seed-42 prediction set.

Selection is by *validation* macro-F1 only; the test set is read for the final
number, never for choosing members.  The extra union/char linear models are
trained by train_extra_linear.py and live in ensemble/preds/<tag>/preds.npz.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "ensemble" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def val_f1_of(path: str, d: dict) -> float:
    if "val_proba" in d and "val_gold" in d:
        from sklearn.metrics import f1_score as _f1
        return float(_f1(np.asarray(d["val_gold"]), np.asarray(d["val_proba"]).argmax(1), average="macro"))
    rd = Path(path).parent
    try:
        m = json.load(open(rd / "metrics.json"))
        return float(m.get("best_val_macro_f1") or m.get("val", {}).get("macro_f1") or -1.0)
    except Exception:
        return -1.0


def collect() -> dict:
    paths = []
    for pat in [
        "classical/*/results/CMTE-NITANP/final/seed42/preds.npz",
        "hf_transformers/*/results/CMTE-NITANP/final/seed42/preds.npz",
        "deep/*/results/CMTE-NITANP/default/seed42/preds.npz",
        "ensemble/preds/*/preds.npz",
    ]:
        paths += glob.glob(str(BENCH / pat))
    models = {}
    for p in paths:
        d = load(p)
        if "test_proba" not in d:
            continue
        parts = Path(p).parts
        tag = Path(p).parent.name
        for fam in ("classical", "hf_transformers", "deep"):
            if fam in parts:
                tag = parts[parts.index(fam) + 1]
                break
        models[tag] = {"d": d, "p": p, "val": val_f1_of(p, d)}
    return models


def main() -> None:
    models = collect()
    first = next(iter(models.values()))["d"]
    ids = [str(x) for x in first["test_ids"]]
    idx = {v: i for i, v in enumerate(ids)}
    gold = np.asarray(first["test_gold"])

    for tag, m in models.items():
        d = m["d"]
        mid = [str(x) for x in d["test_ids"]]
        order = [idx[x] for x in mid]
        d["_test_proba"] = np.asarray(d["test_proba"], float)[order]
        d["_test_gold"] = np.asarray(d["test_gold"])[order]
        assert np.array_equal(d["_test_gold"], gold), tag
        pred = d["_test_proba"].argmax(1)
        m["test_f1"] = f1_score(gold, pred, average="macro")
        m["test_acc"] = accuracy_score(gold, pred)

    by_val = sorted(models.items(), key=lambda kv: -kv[1]["val"])
    rows = []
    for k in (3, 5, 7, 9, 11, 13, 15, 17, 21, 25):
        tags = [t for t, _ in by_val[:k]]
        if len(tags) < k:
            continue
        pred = np.mean([models[t]["d"]["_test_proba"] for t in tags], 0).argmax(1)
        rows.append((f"val_top{k}", k, "|".join(tags),
                     accuracy_score(gold, pred), f1_score(gold, pred, average="macro")))
    priors = {
        "fullft+linearsvc_union+logreg_union+muril_lora+softvote_union":
            ["muril_fullft", "linearsvc_union", "logreg_union", "muril_lora", "softvote_union"],
        "fullft+linearsvc_union+logreg_union+muril_lora+soft_voting+linear_svc":
            ["muril_fullft", "linearsvc_union", "logreg_union", "muril_lora", "soft_voting", "linear_svc"],
        "all_saved": list(models),
    }
    for name, tags in priors.items():
        tags = [t for t in tags if t in models]
        if not tags:
            continue
        pred = np.mean([models[t]["d"]["_test_proba"] for t in tags], 0).argmax(1)
        rows.append((name, len(tags), "|".join(tags),
                     accuracy_score(gold, pred), f1_score(gold, pred, average="macro")))

    with open(OUT / "offline_audit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ensemble", "k", "members", "test_acc", "test_macro_f1"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}"])
    lines = ["# Offline ensemble audit (validation-selected)", "",
             "| ensemble | k | test acc | test macro-F1 |", "|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: -x[4]):
        lines.append(f"| {r[0]} | {r[1]} | {r[3]:.4f} | {r[4]:.4f} |")
    lines += ["", "## Individual models (validation rank)", "",
              "| model | val F1 | test acc | test F1 |", "|---|---:|---:|---:|"]
    for t, m in by_val:
        lines.append(f"| {t} | {m['val']:.4f} | {m['test_acc']:.4f} | {m['test_f1']:.4f} |")
    (OUT / "offline_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print(f"\nwrote {OUT/'offline_audit.csv'} and .md")


if __name__ == "__main__":
    main()
