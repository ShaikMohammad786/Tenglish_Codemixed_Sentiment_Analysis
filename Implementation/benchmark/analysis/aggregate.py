#!/usr/bin/env python3
"""Aggregate benchmark runs into master CSVs, seed statistics, paired tests, figures, REPORT.md."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import atomic_write_text, load_json  # noqa: E402
from common.metrics import ece, risk_coverage  # noqa: E402
from common.paths import BENCH, LABEL_NAMES, REPORTS, RESULTS  # noqa: E402
from common.stats import bootstrap_diff_ci, mcnemar  # noqa: E402

PLOTS = REPORTS / "plots"
TABLES = REPORTS / "tables"
STATS = RESULTS / "stats"


def collect_runs() -> list[dict]:
    rows = []
    for p in sorted(BENCH.rglob("results/**/metrics.json")):
        if "smoke" in str(p):
            continue
        m = load_json(p)
        t = m.get("test", {})
        v = m.get("val", {})
        hyper = m.get("hyper") or m.get("params") or {}
        phase = m.get("phase")
        if phase is None and m.get("family") == "deep":
            phase = "final"  # deep runs predate the explicit phase field
        rows.append({
            "run_id": m.get("run_id"), "family": m.get("family"), "model": m.get("model"),
            "phase": phase, "config": m.get("config"), "seed": m.get("seed"),
            "dataset": m.get("dataset"), "split": m.get("split_name"),
            "feature": m.get("feature"), "hyper": str(hyper),
            "val_macro_f1": m.get("best_val_macro_f1", v.get("macro_f1")),
            "test_macro_f1": t.get("macro_f1"), "test_accuracy": t.get("accuracy"),
            "test_weighted_f1": t.get("weighted_f1"), "test_mcc": t.get("mcc"),
            "test_ece": t.get("ece"), "test_brier": t.get("brier"),
            "best_epoch": m.get("best_epoch"), "elapsed_s": m.get("elapsed_s"),
            "trainable_params": m.get("trainable_params"), "total_params": m.get("total_params"),
            "path": str(p.parent),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def seed_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if r["phase"] == "final":
            groups[(r["family"], r["model"])].append(r)
    out = []
    for (family, model), rs in groups.items():
        f1 = np.array([r["test_macro_f1"] for r in rs if r["test_macro_f1"] is not None])
        acc = np.array([r["test_accuracy"] for r in rs if r["test_accuracy"] is not None])
        eces = np.array([r["test_ece"] for r in rs if r["test_ece"] is not None])
        if len(f1) == 0:
            continue
        out.append({
            "family": family, "model": model, "n_seeds": len(f1),
            "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std(ddof=1)) if len(f1) > 1 else 0.0, 4),
            "acc_mean": round(float(acc.mean()), 4) if len(acc) else None,
            "ece_mean": round(float(eces.mean()), 4) if len(eces) else None,
            "best_seed_f1": round(float(f1.max()), 4),
        })
    out.sort(key=lambda r: -r["macro_f1_mean"])
    for r in out:
        pred = find_preds(rows, r["family"], r["model"], 42)
        if pred is not None:
            from common.stats import bootstrap_ci
            ci = bootstrap_ci(pred["test_gold"], pred["test_pred"], n=500, seed=42)
            r["f1_ci_lo"], r["f1_ci_hi"] = round(ci["lo"], 4), round(ci["hi"], 4)
        else:
            r["f1_ci_lo"], r["f1_ci_hi"] = None, None
    return out


def find_preds(rows, family, model, seed=42):
    for r in rows:
        if (r["family"], r["model"], r["seed"], r["phase"]) == (family, model, seed, "final"):
            p = Path(r["path"]) / "preds.npz"
            if p.exists():
                return np.load(p, allow_pickle=True)
    return None


def pairwise_stats(rows):
    base = find_preds(rows, "classical", "logistic_regression", 42)
    if base is None:
        return []
    y = base["test_gold"]
    out = []
    for r in rows:
        if r["phase"] != "final" or r["seed"] != 42:
            continue
        if (r["family"], r["model"]) == ("classical", "logistic_regression"):
            continue
        pred = find_preds(rows, r["family"], r["model"], 42)
        if pred is None:
            continue
        diff = bootstrap_diff_ci(y, pred["test_pred"], base["test_pred"], n=500, seed=42)
        mc = mcnemar(y, pred["test_pred"], base["test_pred"])
        out.append({"model": f"{r['family']}/{r['model']}", "diff_mean": round(diff["mean"], 4),
                    "diff_lo": round(diff["lo"], 4), "diff_hi": round(diff["hi"], 4),
                    "mcnemar_p": round(mc["p"], 4)})
    if out:
        from common.stats import holm
        adj = holm([x["mcnemar_p"] for x in out])
        for x, a in zip(out, adj):
            x["mcnemar_p_holm"] = round(a, 4)
    return out


def plot_seed_summary(summary):
    PLOTS.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['model']}" for r in summary]
    means = [r["macro_f1_mean"] for r in summary]
    stds = [r["macro_f1_std"] for r in summary]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    ax.barh(labels[::-1], means[::-1], xerr=stds[::-1], color="#4C78A8")
    ax.set_xlabel("test macro-F1 (mean ± std over seeds)")
    ax.axvline(0.8293, color="red", linestyle="--", linewidth=1, label="LogReg 0.8293")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "01_model_comparison.png", dpi=160)
    plt.close(fig)


def plot_per_class(rows):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42]
    models, data = [], []
    for r in reps:
        m = load_json(Path(r["path"]) / "metrics.json")
        pc = m.get("test", {}).get("per_class")
        if not pc:
            continue
        models.append(r["model"])
        data.append([pc[k]["f1"] for k in LABEL_NAMES])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(models))))
    im = ax.imshow(np.array(data), aspect="auto", cmap="viridis", vmin=0.4, vmax=0.9)
    ax.set_xticks(range(3), LABEL_NAMES)
    ax.set_yticks(range(len(models)), models)
    fig.colorbar(im, ax=ax, label="F1")
    ax.set_title("Per-class test F1 (seed 42)")
    fig.tight_layout()
    fig.savefig(PLOTS / "02_per_class_f1.png", dpi=160)
    plt.close(fig)


def plot_confusions(rows, top=3):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42]
    reps.sort(key=lambda r: -(r["test_macro_f1"] or 0))
    reps = [r for r in reps if r["family"] in ("classical", "transformers")][:top]
    if not reps:
        return
    fig, axes = plt.subplots(1, len(reps), figsize=(4 * len(reps), 3.6))
    if len(reps) == 1:
        axes = [axes]
    for ax, r in zip(axes, reps):
        m = load_json(Path(r["path"]) / "metrics.json")
        cm = np.array(m["test"]["confusion"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['model']}\nF1={r['test_macro_f1']:.3f}", fontsize=9)
        ax.set_xticks(range(3), LABEL_NAMES, rotation=30, fontsize=8)
        ax.set_yticks(range(3), LABEL_NAMES, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_confusions.png", dpi=160)
    plt.close(fig)


def plot_reliability(rows):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42][:8]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for r in reps:
        pred = find_preds(rows, r["family"], r["model"], 42)
        if pred is None:
            continue
        p = pred["test_proba"]
        conf = p.max(1)
        acc = (p.argmax(1) == pred["test_gold"]).astype(float)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi) if hi < 1 else (conf >= lo) & (conf <= hi)
            if m.sum() > 0:
                xs.append(conf[m].mean())
                ys.append(acc[m].mean())
        ax.plot(xs, ys, marker="o", markersize=3, label=f"{r['model']} (ECE={r['test_ece']:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_reliability.png", dpi=160)
    plt.close(fig)


def plot_risk_coverage(rows):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42 and r["family"] == "transformers"][:4]
    reps += [r for r in rows if r["phase"] == "final" and r["seed"] == 42 and r["family"] == "classical"][:2]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r in reps:
        pred = find_preds(rows, r["family"], r["model"], 42)
        if pred is None:
            continue
        rc = risk_coverage(pred["test_gold"], pred["test_proba"], coverages=tuple(np.linspace(0.3, 1.0, 8)))
        xs = [v["coverage"] for v in rc.values()]
        ys = [v["macro_f1"] for v in rc.values()]
        ax.plot(xs, ys, marker="o", markersize=3, label=r["model"])
    ax.set_xlabel("coverage")
    ax.set_ylabel("macro-F1")
    ax.set_title("Risk–coverage (abstention)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PLOTS / "05_risk_coverage.png", dpi=160)
    plt.close(fig)


def plot_cost(rows):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42 and r["elapsed_s"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in reps:
        ax.scatter(max(r["elapsed_s"], 1) / 60, r["test_macro_f1"], s=30)
        ax.annotate(r["model"], (max(r["elapsed_s"], 1) / 60, r["test_macro_f1"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("train time (minutes, log)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Cost–benefit")
    fig.tight_layout()
    fig.savefig(PLOTS / "06_cost_benefit.png", dpi=160)
    plt.close(fig)


def plot_learning_curves(rows):
    reps = [r for r in rows if r["phase"] == "final" and r["seed"] == 42]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in reps:
        h = Path(r["path"]) / "history.json"
        if not h.exists():
            continue
        hist = load_json(h)
        if not isinstance(hist, list) or not hist:
            continue
        xs = [x["epoch"] for x in hist if "val_macro_f1" in x]
        ys = [x["val_macro_f1"] for x in hist if "val_macro_f1" in x]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=r["model"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Learning curves")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PLOTS / "07_learning_curves.png", dpi=160)
    plt.close(fig)


def rq_section(summary, pairwise, rows):
    lines = ["## Research-question answers (auto-derived)", ""]
    if summary:
        best = summary[0]
        lines.append(f"- **RQ1 (ranking):** best = `{best['family']}/{best['model']}` at "
                     f"{best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}.")
        cls = [r for r in summary if r["family"] == "classical"]
        tr = [r for r in summary if r["family"] == "transformers"]
        if cls and tr:
            bc = max(cls, key=lambda r: r["macro_f1_mean"])
            bt = max(tr, key=lambda r: r["macro_f1_mean"])
            lines.append(f"- **RQ2 (lexical vs transformer):** best classical `{bc['model']}` "
                         f"{bc['macro_f1_mean']:.4f} vs best transformer `{bt['model']}` "
                         f"{bt['macro_f1_mean']:.4f} (Δ={bc['macro_f1_mean'] - bt['macro_f1_mean']:+.4f}).")
    if pairwise:
        sig = [p for p in pairwise if p["mcnemar_p"] < 0.05]
        lines.append(f"- **Significance:** {len(sig)}/{len(pairwise)} models differ from LogReg "
                     f"at p<0.05 (paired McNemar on the test set).")
    params_by_model = {r["model"]: r["total_params"] for r in rows
                       if r["phase"] == "final" and r["seed"] == 42 and r.get("total_params")}
    pairs = [(params_by_model[r["model"]], r["macro_f1_mean"]) for r in summary if r["model"] in params_by_model]
    if len(pairs) >= 3:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
        verdict = "does predict" if (pval < 0.05 and rho > 0.5) else "does not predict"
        lines.append(f"- **RQ3 (capacity):** Spearman ρ={rho:.3f} (p={pval:.3f}) between parameter "
                     f"count and macro-F1 → capacity {verdict} accuracy.")
    cal = [r for r in summary if r["ece_mean"] is not None]
    if cal:
        b = min(cal, key=lambda r: r["ece_mean"])
        lines.append(f"- **RQ6 (calibration):** best-calibrated `{b['family']}/{b['model']}` "
                     f"ECE={b['ece_mean']:.4f}.")
    rr = [r for r in rows if r["phase"] == "final" and r["seed"] == 42
          and r["elapsed_s"] and r["test_macro_f1"]]
    if rr:
        bc = max(rr, key=lambda r: r["test_macro_f1"] / max(r["elapsed_s"], 1))
        lines.append(f"- **RQ5 (cost–benefit):** best F1-per-second `{bc['model']}` "
                     f"({bc['test_macro_f1']:.4f} in {bc['elapsed_s']:.0f}s).")
    rob_path = Path(__file__).resolve().parents[1] / "robustness" / "results" / "CMTE-NITANP" / "robustness.json"
    if rob_path.exists():
        rob = load_json(rob_path)
        deltas = []
        for fam, models in rob.items():
            for m, r in models.items():
                ds = [v["delta"] for v in r.get("variants", {}).values()]
                if ds:
                    deltas.append((f"{fam}/{m}", max(ds)))
        if deltas:
            best = min(deltas, key=lambda x: x[1])
            lines.append(f"- **RQ4 (robustness):** smallest worst-case degradation `{best[0]}` "
                         f"(max Δ={best[1]:+.4f} across variants).")
    xc_path = Path(__file__).resolve().parents[1] / "cross_corpus" / "results" / "CMTET" / "cross_corpus.json"
    if xc_path.exists():
        xc = load_json(xc_path)
        if xc:
            best = max(xc.items(), key=lambda kv: kv[1]["metrics"]["all"]["macro_f1"])
            lines.append(f"- **RQ7 (CMTET transfer):** best zero-shot `{best[0]}` "
                         f"F1={best[1]['metrics']['all']['macro_f1']:.4f} "
                         f"(drop {best[1].get('transfer_drop', float('nan')):+.4f}).")
    else:
        lines.append("- **RQ7 (CMTET transfer):** pending (cross-corpus run not yet generated).")
    lines.append("- **RQ8 (abstention):** see `05_risk_coverage.png` and the per-run `risk_coverage` blocks.")
    lines.append("")
    return lines


def plot_per_model_curves(rows):
    """One training-curve figure per model run (train loss + val macro-F1, best epoch marked)."""
    out = PLOTS / "per_model"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["phase"] != "final":
            continue
        hp = Path(r["path"]) / "history.json"
        if not hp.exists():
            continue
        hist = load_json(hp)
        if not isinstance(hist, list) or not hist:
            continue
        epochs = [h.get("epoch") for h in hist if h.get("epoch") is not None]
        if not epochs:
            continue
        losses = [h.get("train_loss") for h in hist]
        vals = [h.get("val_macro_f1") for h in hist]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(epochs, losses, "o-", color="#C44E52", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color="#C44E52")
        ax2 = ax1.twinx()
        ax2.plot(epochs, vals, "s-", color="#4C78A8", label="val macro-F1")
        ax2.set_ylabel("val macro-F1", color="#4C78A8")
        be = r.get("best_epoch")
        if be:
            ax1.axvline(be, color="gray", ls="--", lw=1)
        ax1.set_title(f"{r['family']}/{r['model']} seed {r['seed']} (best epoch {be})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{r['family']}__{r['model']}__seed{r['seed']}.png", dpi=150)
        plt.close(fig)


def plot_per_model_confusions(rows):
    """One confusion-matrix figure for every final run (classical, deep, transformers)."""
    out = PLOTS / "per_model"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["phase"] != "final":
            continue
        mp = Path(r["path"]) / "metrics.json"
        if not mp.exists():
            continue
        m = load_json(mp)
        cm = np.array(m.get("test", {}).get("confusion", []))
        if cm.size != 9:
            continue
        fig, ax = plt.subplots(figsize=(3.8, 3.3))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), LABEL_NAMES, rotation=30, fontsize=7)
        ax.set_yticks(range(3), LABEL_NAMES, fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set_title(f"{r['family']}/{r['model']} seed {r['seed']}\n"
                     f"F1={r['test_macro_f1']:.4f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{r['family']}__{r['model']}__seed{r['seed']}_confusion.png", dpi=150)
        plt.close(fig)


def write_latex(summary, pairwise):
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{llrrrr}", "\\hline",
             "Family & Model & Seeds & Macro-F1 & 95\\% CI & ECE \\\\", "\\hline"]
    for r in summary:
        ci = f"[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}]" if r.get("f1_ci_lo") is not None else "--"
        ece = f"{r['ece_mean']:.4f}" if r["ece_mean"] is not None else "--"
        name = r["model"].replace("_", "\\_")
        lines.append(f"{r['family']} & {name} & {r['n_seeds']} & "
                     f"{r['macro_f1_mean']:.4f} $\\pm$ {r['macro_f1_std']:.4f} & {ci} & {ece} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    atomic_write_text(TABLES / "seed_summary.tex", "\n".join(lines) + "\n")
    if pairwise:
        lines = ["\\begin{tabular}{lrrr}", "\\hline",
                 "Model & $\\Delta$ Macro-F1 & 95\\% CI & McNemar $p$ (Holm) \\\\", "\\hline"]
        for r in pairwise:
            name = r["model"].replace("_", "\\_")
            lines.append(f"{name} & {r['diff_mean']:+.4f} & "
                         f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] & "
                         f"{r.get('mcnemar_p_holm', r['mcnemar_p']):.4f} \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        atomic_write_text(TABLES / "pairwise_vs_logreg.tex", "\n".join(lines) + "\n")


def load_sensitivity():
    p = BENCH / "sensitivity" / "results" / "CMTE-NITANP" / "sensitivity.csv"
    if not p.exists():
        return []
    import csv as _csv
    with open(p, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def sensitivity_section():
    rows = load_sensitivity()
    if not rows:
        return ["## Hyperparameter sensitivity (val only)", "", "_not generated yet_", ""]
    lines = ["## Hyperparameter sensitivity (val only, fixed epoch budget)", "",
             "| model | config | lr | max_len | batch | eff. batch | best epoch | val macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['config']} | {r['lr']} | {r['max_len']} | {r['batch']} | "
                     f"{r['effective_batch']} | {r['best_epoch']} | {float(r['val_macro_f1']):.4f} |")
    lines.append("")
    return lines


def plot_sensitivity():
    rows = load_sensitivity()
    if not rows:
        return
    labels = [f"{r['model'].replace('_lora','')} | lr{r['lr']} len{r['max_len']} bs{r['batch']}" for r in rows]
    vals = [float(r["val_macro_f1"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(rows))))
    ax.barh(labels[::-1], vals[::-1], color="#72B7B2")
    ax.set_xlabel("val macro-F1")
    ax.set_title("Transformer hyperparameter sensitivity (val only)")
    fig.tight_layout()
    fig.savefig(PLOTS / "08_sensitivity.png", dpi=160)
    plt.close(fig)


def write_report(summary, pairwise, rows):
    lines = ["# Benchmark report — Romanized Telugu–English sentiment (CMTE-NITANP Split B)", "",
             "Auto-generated from `Implementation/benchmark/**/results/**/metrics.json`.", ""]
    lines += rq_section(summary, pairwise, rows)
    lines += ["## Seed summary (final runs, mean ± std, 95% bootstrap CI)", "",
              "| family | model | seeds | macro-F1 | 95% CI | accuracy | ECE |",
              "|---|---|---:|---:|---|---:|---:|"]
    for r in summary:
        ci = f"[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}]" if r.get("f1_ci_lo") is not None else "--"
        acc = f"{r['acc_mean']:.4f}" if r["acc_mean"] is not None else "--"
        ece = f"{r['ece_mean']:.4f}" if r["ece_mean"] is not None else "--"
        lines.append(f"| {r['family']} | {r['model']} | {r['n_seeds']} | "
                     f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {ci} | {acc} | {ece} |")
    if pairwise:
        lines += ["", "## Paired tests vs LogReg (seed 42; paired bootstrap + McNemar)", "",
                  "| model | Δ macro-F1 | 95% CI | McNemar p | Holm p |", "|---|---:|---|---:|---:|"]
        for r in pairwise:
            lines.append(f"| {r['model']} | {r['diff_mean']:+.4f} | [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] | "
                         f"{r['mcnemar_p']:.4f} | {r.get('mcnemar_p_holm', r['mcnemar_p']):.4f} |")
    lines += sensitivity_section()
    lines += ["", "## External reference points (published; NOT reproduced here)", "",
              "| source | setting | reported |", "|---|---|---|",
              "| Rayala & Seshadri (Springer 2025) | Telugu–English transformers on CMTE-NITANP | +5% acc / +6% F1 vs baselines (as reported) |",
              "| IEEE 11346359 | Telugu–English hybrid/ensemble | 83.4% accuracy (best ensemble) |",
              "| DravidianCodeMix (LRE 2022) | Tamil/Malayalam/Kannada, 5-class, random split | multi-model benchmark (not Telugu) |",
              "| Rahul & Adebayo (DravidianLangTech 2026) | Hinglish robustness | XLM-R 99.7% in-dist → 42.5% adversarial |",
              "", "These are context only; **our numbers use a different, leakage-audited protocol** and are not directly comparable.", ""]
    lines += ["", "## Figures", ""]
    for p in sorted(PLOTS.glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines.append("")
    TABLES.mkdir(parents=True, exist_ok=True)
    atomic_write_text(REPORTS / "REPORT.md", "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    rows = collect_runs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "master_results.csv", rows)
    summary = seed_summary(rows)
    write_csv(RESULTS / "seed_summary.csv", summary)
    pairwise = pairwise_stats(rows)
    STATS.mkdir(parents=True, exist_ok=True)
    write_csv(STATS / "pairwise_vs_logreg.csv", pairwise)
    write_report(summary, pairwise, rows)
    write_latex(summary, pairwise)
    if args.plots:
        plot_seed_summary(summary)
        plot_per_class(rows)
        plot_confusions(rows)
        plot_reliability(rows)
        plot_risk_coverage(rows)
        plot_cost(rows)
        plot_learning_curves(rows)
        plot_per_model_curves(rows)
        plot_per_model_confusions(rows)
        plot_sensitivity()
    print(f"runs={len(rows)} models={len(summary)} pairwise={len(pairwise)}")
    print(f"wrote {RESULTS/'master_results.csv'} and {REPORTS/'REPORT.md'}")


if __name__ == "__main__":
    main()
