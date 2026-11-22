#!/usr/bin/env python3
"""Feature-space ablation: does lexical/orthographic representation matter?

For a set of fast models, fit the frozen-style grid on train and evaluate test for
each TF-IDF space: word11 (existing default), word12, char25, union.
Writes per (model, feature) runs under classical/<model>/results/<dataset>/ablation_<feature>/seed42.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "classical"))

from common.data import load_records, to_arrays  # noqa: E402
from common.io import atomic_write_json, atomic_write_text, write_status  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from common.paths import SPLITS, model_dir  # noqa: E402
from common.runner import set_seed  # noqa: E402
from classical.run_classical import FEATURES, GRIDS, build_features, build_model  # noqa: E402

MODELS = ("logistic_regression", "linear_svc", "multinomial_nb", "soft_voting", "xgboost", "lightgbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", nargs="+", default=list(FEATURES))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()

    set_seed(args.seed)
    recs = load_records(args.dataset, "B")
    arr = {s: to_arrays(recs[s]) for s in SPLITS}
    texts = {s: arr[s][0] for s in SPLITS}
    y = {s: arr[s][1] for s in SPLITS}

    rows = []
    for feature in args.features:
        X, meta, cached = build_features(texts, feature, f"{args.dataset}:B:text_clean")
        print(f"[ablation] feature={feature} n_features={meta['n_features']:,} cached={cached}")
        for model in args.models:
            history, best = [], None
            for params in GRIDS[model]:
                clf = build_model(model, params, args.seed).fit(X["train"], y["train"])
                vf = float(f1_score(y["val"], clf.predict(X["val"]), average="macro", zero_division=0))
                history.append({"params": params, "val_macro_f1": round(vf, 4)})
                if best is None or vf > best[0]:
                    best = (vf, params)
            vf, params = best
            clf = build_model(model, params, args.seed).fit(X["train"], y["train"])
            proba = clf.predict_proba(X["test"])
            m = compute_metrics(y["test"], proba.argmax(1), probs=proba)
            rd = model_dir("classical", model) / "results" / args.dataset / f"ablation_{feature}" / f"seed{args.seed}"
            rd.mkdir(parents=True, exist_ok=True)
            result = {"run_id": f"classical/{model}/{args.dataset}/ablation_{feature}/seed{args.seed}",
                      "family": "classical", "model": model, "phase": "ablation",
                      "config": f"ablation_{feature}", "seed": args.seed, "dataset": args.dataset,
                      "split_name": "B", "text_field": "text_clean", "feature": feature,
                      "n_features": meta["n_features"], "params": params,
                      "rows": {s: len(y[s]) for s in SPLITS}, "best_val_macro_f1": round(vf, 4),
                      "test": m, "history": history}
            atomic_write_json(rd / "metrics.json", result)
            atomic_write_json(rd / "history.json", history)
            atomic_write_text(rd / "per_class.txt", m["report"])
            write_status(rd, "done", test_macro_f1=m["macro_f1"])
            rows.append({"model": model, "feature": feature, "n_features": meta["n_features"],
                         "val_macro_f1": round(vf, 4), "test_macro_f1": round(m["macro_f1"], 4),
                         "delta_vs_word11": None})
            print(f"  {model:22s} {feature:8s} nfeat={meta['n_features']:>8,} "
                  f"val={vf:.4f} test={m['macro_f1']:.4f}")

    # delta vs word11 per model
    base = {r["model"]: r["test_macro_f1"] for r in rows if r["feature"] == "word11"}
    for r in rows:
        if r["model"] in base:
            r["delta_vs_word11"] = round(r["test_macro_f1"] - base[r["model"]], 4)
    out = BENCH / "results" / "feature_ablation.csv"
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
