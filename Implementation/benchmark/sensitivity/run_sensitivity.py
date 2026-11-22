#!/usr/bin/env python3
"""Hyperparameter sensitivity for transformers (val only, fixed epoch budget).

Grid per model: lr {2e-4, 4e-4} x max_len {64, 96} + one batch=8 variant,
with effective batch held at 32 (batch 16 x accum 2, or batch 8 x accum 4).

Runs the transformer runner with --phase sensitivity (val-only, no test), then
writes sensitivity/results/<dataset>/sensitivity.csv and .md.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(BENCH))

from common.io import atomic_write_text  # noqa: E402
from common.paths import DATASETS  # noqa: E402

# One-factor-at-a-time around the frozen baseline (lr 4e-4, len 64, bs 16x2).
# The baseline itself is the main run and is reused, so only 3 new runs per model.
GRID = [
    {"tag": "lr2e4_len64_bs16", "lr": 2e-4, "max_len": 64, "batch": 16, "accum": 2},
    {"tag": "lr4e4_len96_bs16", "lr": 4e-4, "max_len": 96, "batch": 16, "accum": 2},
    {"tag": "lr4e4_len64_bs8", "lr": 4e-4, "max_len": 64, "batch": 8, "accum": 4},
]
BASELINE = {"lr": 4e-4, "max_len": 64, "batch": 16, "accum": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["muril_lora", "mbert_lora", "distilmbert_lora"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for model in args.models:
        for cfg in GRID:
            rd = (BENCH / "hf_transformers" / model / "results" / args.dataset
                  / f"sens_{cfg['tag']}" / f"seed{args.seed}")
            if (rd / "run_status.json").exists() and not args.force:
                if json.loads((rd / "run_status.json").read_text()).get("status") == "done":
                    print(f"[skip] {model} {cfg['tag']}")
                    continue
            cmd = [PY, str(BENCH / "hf_transformers" / model / "run.py"),
                   "--phase", "sensitivity", "--dataset", args.dataset,
                   "--config", f"sens_{cfg['tag']}", "--seed", str(args.seed),
                   "--epochs", str(args.epochs), "--min-epochs", str(args.epochs),
                   "--patience", str(args.epochs),
                   "--lr", str(cfg["lr"]), "--max-len", str(cfg["max_len"]),
                   "--batch-size", str(cfg["batch"]), "--grad-accum", str(cfg["accum"])]
            print(f"[run] {model} {cfg['tag']}")
            subprocess.run(cmd, cwd=BENCH, check=False)

    rows = []
    for model in args.models:
        # baseline = the main final run (no extra compute)
        main_mp = (BENCH / "hf_transformers" / model / "results" / args.dataset
                   / "final" / f"seed{args.seed}" / "metrics.json")
        if main_mp.exists():
            mm = json.loads(main_mp.read_text())
            rows.append({
                "model": model, "config": "baseline_main", "lr": BASELINE["lr"],
                "max_len": BASELINE["max_len"], "batch": BASELINE["batch"],
                "effective_batch": BASELINE["batch"] * BASELINE["accum"],
                "epochs": mm.get("hyper", {}).get("epochs"), "best_epoch": mm.get("best_epoch"),
                "val_macro_f1": round(mm.get("best_val_macro_f1", float("nan")), 4),
                "val_accuracy": None, "elapsed_s": mm.get("elapsed_s"),
            })
        for cfg in GRID:
            mp = (BENCH / "hf_transformers" / model / "results" / args.dataset
                  / f"sens_{cfg['tag']}" / f"seed{args.seed}" / "metrics.json")
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            rows.append({
                "model": model, "config": cfg["tag"], "lr": cfg["lr"],
                "max_len": cfg["max_len"], "batch": cfg["batch"],
                "effective_batch": cfg["batch"] * cfg["accum"],
                "epochs": args.epochs, "best_epoch": m.get("best_epoch"),
                "val_macro_f1": round(m.get("val", {}).get("macro_f1", float("nan")), 4),
                "val_accuracy": round(m.get("val", {}).get("accuracy", float("nan")), 4),
                "elapsed_s": m.get("elapsed_s"),
            })

    out_dir = BENCH / "sensitivity" / "results" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "sensitivity.csv"
    if rows:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    lines = ["# Transformer hyperparameter sensitivity (val only, fixed epoch budget)", "",
             f"Seed {args.seed} | epochs {args.epochs} | dataset {args.dataset}", "",
             "| model | config | lr | max_len | batch | eff. batch | best epoch | val macro-F1 | val acc |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        acc = "--" if r["val_accuracy"] is None else f"{r['val_accuracy']:.4f}"
        lines.append(f"| {r['model']} | {r['config']} | {r['lr']} | {r['max_len']} | {r['batch']} | "
                     f"{r['effective_batch']} | {r['best_epoch']} | {r['val_macro_f1']:.4f} | "
                     f"{acc} |")
    atomic_write_text(out_dir / "sensitivity.md", "\n".join(lines) + "\n")
    print(f"wrote {out_dir/'sensitivity.md'} and {p}")


if __name__ == "__main__":
    main()
