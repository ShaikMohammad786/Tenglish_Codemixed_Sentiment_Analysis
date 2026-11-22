#!/usr/bin/env python3
"""Validate run artifacts before aggregation.

Checks, for every final run:
  - metrics.json, history.json, preds.npz exist;
  - test macro-F1 recomputed from preds matches metrics.json;
  - best_epoch is within the recorded history length;
  - frozen hash is present for final runs that used a frozen config.

Exit code 1 if any run is invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_json  # noqa: E402
from common.paths import BENCH  # noqa: E402


def main() -> int:
    bad, checked = [], 0
    for mpath in sorted(BENCH.rglob("results/**/metrics.json")):
        if "smoke" in str(mpath):
            continue
        m = load_json(mpath)
        if m.get("phase") not in (None, "final"):
            continue
        run = mpath.parent
        checked += 1
        problems = []
        if not (run / "history.json").exists() and m.get("history"):
            problems.append("missing history.json")
        preds_path = run / "preds.npz"
        if not preds_path.exists():
            problems.append("missing preds.npz")
        else:
            p = np.load(preds_path, allow_pickle=True)
            gold = p["test_gold"]
            pred = p["test_pred"]
            recomputed = float(f1_score(gold, pred, average="macro", zero_division=0))
            reported = m.get("test", {}).get("macro_f1")
            if reported is None or abs(recomputed - reported) > 1e-6:
                problems.append(f"macro-F1 mismatch reported={reported} recomputed={recomputed:.6f}")
        hist = m.get("history") or []
        be = m.get("best_epoch")
        if be is not None and hist and be > len(hist):
            problems.append(f"best_epoch {be} > history {len(hist)}")
        if m.get("family") in ("classical", "transformers") and m.get("phase") == "final" \
                and m.get("frozen_hash") is None:
            problems.append("missing frozen_hash")
        if problems:
            bad.append((str(run.relative_to(BENCH)), problems))

    print(f"checked {checked} final runs; invalid: {len(bad)}")
    for run, problems in bad:
        print(f"  INVALID {run}: {problems}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
