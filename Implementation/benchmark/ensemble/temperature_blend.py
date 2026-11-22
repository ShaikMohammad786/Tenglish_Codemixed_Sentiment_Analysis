#!/usr/bin/env python3
"""Temperature-calibrate each base model on validation, then uniform-blend.

Claude's 90%-ceiling review recommended a temperature-calibrated soft vote
(T ~ 1.25 for transformers, T ~ 1.0 for linear) before blending.  This script
fits one temperature per model on validation log-loss, freezes it, and
evaluates the uniform blend on test once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

import blend

OUT = blend.BENCH / "ensemble" / "results"
PRIOR6 = blend.PRIOR6


def calibrate(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(P, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_T(P: np.ndarray, y: np.ndarray) -> float:
    def nll(T):
        q = calibrate(P, float(T))
        return float(-np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0))))
    r = minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded")
    return float(r.x)


def main() -> None:
    tags = PRIOR6
    Pv, Pt, yv, yt = blend.stack(tags)
    Ts = [fit_T(Pv[:, i, :], yv) for i in range(len(tags))]
    Pv_c = np.stack([calibrate(Pv[:, i, :], Ts[i]) for i in range(len(tags))], 1)
    Pt_c = np.stack([calibrate(Pt[:, i, :], Ts[i]) for i in range(len(tags))], 1)

    def report(name, Pval, Ptest):
        pred = Ptest.mean(1).argmax(1)
        print(f"{name:34s} val_f1={f1_score(yv, Pval.mean(1).argmax(1), average='macro'):.4f} "
              f"test_f1={f1_score(yt, pred, average='macro'):.4f} "
              f"test_acc={accuracy_score(yt, pred):.4f}")

    report("uncalibrated uniform", Pv, Pt)
    report("temperature-calibrated uniform", Pv_c, Pt_c)
    print("temperatures:", {t: round(T, 4) for t, T in zip(tags, Ts)})
    rows = [{"ensemble": "prior6", "method": "uniform_uncalibrated",
             "test_acc": round(accuracy_score(yt, Pt.mean(1).argmax(1)), 4),
             "test_f1": round(f1_score(yt, Pt.mean(1).argmax(1), average="macro"), 4),
             "temperatures": {t: round(T, 4) for t, T in zip(tags, Ts)}},
            {"ensemble": "prior6", "method": "uniform_calibrated",
             "test_acc": round(accuracy_score(yt, Pt_c.mean(1).argmax(1)), 4),
             "test_f1": round(f1_score(yt, Pt_c.mean(1).argmax(1), average="macro"), 4),
             "temperatures": {t: round(T, 4) for t, T in zip(tags, Ts)}}]
    (OUT / "temperature_results.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT/'temperature_results.json'}")


if __name__ == "__main__":
    main()
