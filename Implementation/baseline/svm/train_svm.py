#!/usr/bin/env python3
"""Baseline 1: TF-IDF + linear SVM, evaluated on Split A and Split B.

For each split it fits TF-IDF on train only, picks the SVM ``C`` on validation
(never on test), then reports test accuracy and macro-F1.  The leakage gap is

    gap = macroF1(Split A) - macroF1(Split B)

Split A ignores near-duplicate clusters (leakage present); Split B respects them
(leakage removed).  A large positive gap means duplicated rows inflated Split A.

Input  : datasets/splits/split{A,B}_{train,val,test}.jsonl
Outputs: Implementation/baseline/svm/results/{svm_results.md, svm_results.json}

Run with the project venv:
    .venv/bin/python Implementation/baseline/svm/train_svm.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

SPLITS = ("train", "val", "test")
C_GRID = (0.1, 1.0, 10.0)


def load_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def evaluate_split(splits_dir: Path, name: str, text_field: str, seed: int) -> dict:
    """Fit TF-IDF + LinearSVC for split ``name`` ('A' or 'B') and score the test set."""
    parts = {s: load_split(splits_dir / f"split{name}_{s}.jsonl") for s in SPLITS}
    text = {s: [r[text_field] for r in parts[s]] for s in SPLITS}
    y = {s: np.array([r["label"] for r in parts[s]]) for s in SPLITS}

    vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(text["train"])  # vocabulary learned on train only
    Xva, Xte = vec.transform(text["val"]), vec.transform(text["test"])

    best_c, best_val = None, -1.0
    for c in C_GRID:
        clf = LinearSVC(C=c, max_iter=5000, random_state=seed).fit(Xtr, y["train"])
        val_f1 = f1_score(y["val"], clf.predict(Xva), average="macro")
        if val_f1 > best_val:
            best_c, best_val = c, val_f1

    clf = LinearSVC(C=best_c, max_iter=5000, random_state=seed).fit(Xtr, y["train"])
    pred = clf.predict(Xte)
    return {
        "split": name,
        "rows": {s: len(parts[s]) for s in SPLITS},
        "vocab": int(Xtr.shape[1]),
        "C": best_c,
        "val_macro_f1": round(best_val, 4),
        "test_accuracy": round(accuracy_score(y["test"], pred), 4),
        "test_macro_f1": round(f1_score(y["test"], pred, average="macro"), 4),
        "per_class": classification_report(y["test"], pred, digits=4, zero_division=0),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=root / "datasets/splits")
    ap.add_argument("--text-field", choices=("text_clean", "text_norm"), default="text_clean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    results = {name: evaluate_split(args.splits_dir, name, args.text_field, args.seed)
               for name in ("A", "B")}
    a, b = results["A"], results["B"]
    gap_f1 = a["test_macro_f1"] - b["test_macro_f1"]
    gap_acc = a["test_accuracy"] - b["test_accuracy"]

    print(f"TF-IDF + LinearSVC  (text={args.text_field}, splits={args.splits_dir})")
    print(f"{'split':6} {'rows(tr/va/te)':>18} {'C':>5} {'val F1':>8} {'test acc':>9} {'test macro-F1':>14}")
    for r in (a, b):
        rows = "/".join(f"{r['rows'][s]:,}" for s in SPLITS)
        print(f"{r['split']:6} {rows:>18} {r['C']:>5} {r['val_macro_f1']:>8.4f} "
              f"{r['test_accuracy']:>9.4f} {r['test_macro_f1']:>14.4f}")
    print(f"\nLeakage gap (A - B): macro-F1 {gap_f1:+.4f} | accuracy {gap_acc:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "svm_results.json").write_text(
        json.dumps({"config": vars(args) | {"splits_dir": str(args.splits_dir), "out_dir": str(args.out_dir)},
                    "results": results, "gap": {"macro_f1": round(gap_f1, 4), "accuracy": round(gap_acc, 4)}},
                   indent=2, default=str), encoding="utf-8")
    (args.out_dir / "svm_results.md").write_text(build_md(args, a, b, gap_f1, gap_acc), encoding="utf-8")
    print(f"\nWrote:\n  {(args.out_dir / 'svm_results.md').resolve()}\n  {(args.out_dir / 'svm_results.json').resolve()}")


def build_md(args, a, b, gap_f1, gap_acc) -> str:
    out = ["# Baseline: TF-IDF + LinearSVC", "",
           f"text field: `{args.text_field}` | splits: `{args.splits_dir}` | seed: {args.seed}", "",
           "| split | rows (train/val/test) | vocab | C | val macro-F1 | test accuracy | test macro-F1 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in (a, b):
        rows = " / ".join(f"{r['rows'][s]:,}" for s in SPLITS)
        out.append(f"| {r['split']} | {rows} | {r['vocab']:,} | {r['C']} | {r['val_macro_f1']:.4f} | "
                   f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |")
    out += ["", f"**Leakage gap (A - B): macro-F1 {gap_f1:+.4f} | accuracy {gap_acc:+.4f}**", "",
            f"C selected on val only; test never used for tuning.", ""]
    out += ["## Per-class report", "", "### Split A", "", "```", a["per_class"].rstrip(), "```", "",
            "### Split B", "", "```", b["per_class"].rstrip(), "```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
