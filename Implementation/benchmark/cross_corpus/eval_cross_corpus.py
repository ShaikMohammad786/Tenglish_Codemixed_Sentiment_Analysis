#!/usr/bin/env python3
"""Zero-shot NITANP -> CMTET transfer (RQ7).

Trains/fits each finished model on CMTE-NITANP Split B only (no CMTET data),
predicts CMTET, and reports overall + cluster/label-conflict-stratified metrics.

Outputs: Implementation/benchmark/cross_corpus/results/CMTET/{cross_corpus.json, .md}
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "classical"))
sys.path.insert(0, str(BENCH / "hf_transformers"))

from common.io import atomic_write_json, atomic_write_text, load_json  # noqa: E402
from common.paths import DATASETS, REPO, model_dir  # noqa: E402
from common.runner import set_seed  # noqa: E402

LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cmtet():
    rows = list(csv.DictReader((REPO / "datasets/processed/CMTET/labeled.csv").open(encoding="utf-8")))
    clusters = {r["id"]: int(r["group_id"])
                for r in csv.DictReader((DATASETS["CMTET"] / "clustered.csv").open(encoding="utf-8"))}
    gsizes = Counter(clusters.values())
    by_group = {}
    for r in rows:
        g = clusters.get(r["id"], -1)
        by_group.setdefault(g, set()).add(r["label"])
    multi = np.array([gsizes.get(clusters.get(r["id"], -1), 1) > 1 for r in rows])
    conflict = np.array([len(by_group.get(clusters.get(r["id"], -1), set())) > 1 for r in rows])
    texts = [r["text_clean"] for r in rows]
    y = np.array([LABELS[r["label"]] for r in rows if r["label"] in LABELS])
    # keep alignment if any unknown labels were dropped
    keep = np.array([r["label"] in LABELS for r in rows])
    texts = [t for t, k in zip(texts, keep) if k]
    multi, conflict = multi[keep], conflict[keep]
    return texts, y, multi, conflict, rows


def metrics(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
            "n": int(len(y))}


def stratify(y, p, multi, conflict):
    out = {"all": metrics(y, p)}
    for name, mask in (("singleton", ~multi), ("multi_cluster", multi),
                       ("conflict_cluster", conflict), ("non_conflict", ~conflict)):
        if mask.sum() > 0:
            out[name] = metrics(y[mask], p[mask])
    return out


def eval_classical(cmtet_texts):
    from classical.run_classical import build_model, make_vectorizer

    train = load_jsonl(DATASETS["CMTE-NITANP"] / "splitB_train.jsonl")
    tr_texts = [r["text_clean"] for r in train]
    ytr = np.array([LABELS[r["label"]] for r in train])
    out = {}
    for mdir in sorted(model_dir("classical", "").parent.iterdir()):
        if not mdir.is_dir():
            continue
        cfgp = mdir / "results" / "CMTE-NITANP" / "best_config.json"
        if not cfgp.exists():
            continue
        cfg = load_json(cfgp)
        vec = make_vectorizer(cfg["feature"]).fit(tr_texts)
        clf = build_model(mdir.name, cfg["params"], 42).fit(vec.transform(tr_texts), ytr)
        pred = clf.predict(vec.transform(cmtet_texts))
        out[mdir.name] = (pred, cfg.get("feature"), cfg.get("params"))
        print(f"[xcorpus] classical {mdir.name} done")
    return out


def eval_transformers(cmtet_texts):
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging

    from common.ckpt import load_checkpoint, load_trainable_state
    from common.data import IndexDataset
    from hf_transformers.run_transformer import SPECS, predict

    hf_logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = {}
    for model, spec in SPECS.items():
        rd = model_dir("transformers", model) / "results" / "CMTE-NITANP" / "final" / "seed42"
        mp, ckpt = rd / "metrics.json", rd / "checkpoint_best.pt"
        if not ckpt.exists() and (rd / "checkpoint_best.path").exists():
            ckpt = Path((rd / "checkpoint_best.path").read_text().strip())
        if not mp.exists() or not ckpt.exists():
            continue
        meta = load_json(mp)
        max_len = meta.get("hyper", {}).get("max_len", 64)
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        enc = tok(cmtet_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        ds = IndexDataset(dict(enc), np.zeros(len(cmtet_texts), dtype=int))
        loader = DataLoader(ds, batch_size=32)
        set_seed(42)  # TeluguBERT has no pooler weights in its checkpoint; reproduce the main run's seeded init
        base = AutoModelForSequenceClassification.from_pretrained(spec["hf"], num_labels=3)
        if spec["full_ft"]:
            obj = base
        else:
            obj = get_peft_model(base, LoraConfig(
                r=meta.get("hyper", {}).get("lora_r", 16),
                lora_alpha=meta.get("hyper", {}).get("lora_alpha", 32), lora_dropout=0.1,
                target_modules=spec["targets"], bias="none", task_type="SEQ_CLS"))
        ck = load_checkpoint(ckpt, device)
        if spec["full_ft"]:
            obj.load_state_dict(ck["model"])
        else:
            load_trainable_state(obj, ck["model"])
        obj.to(device)
        pred, _, _ = predict(obj, loader, device, amp)
        out[model] = (pred, spec["hf"], meta.get("hyper"))
        print(f"[xcorpus] transformer {model} done")
        del obj
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (BENCH / "cross_corpus" / "results" / "CMTET")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts, y, multi, conflict, _ = load_cmtet()
    print(f"[xcorpus] CMTET rows={len(y):,} multi={multi.sum():,} conflict={conflict.sum():,}")

    results = {}
    for fam, preds in (("classical", eval_classical(texts)),
                       ("transformers", eval_transformers(texts))):
        for model, (pred, feat, params) in preds.items():
            results[f"{fam}/{model}"] = {"feature": feat, "params": params,
                                         "metrics": stratify(y, pred, multi, conflict)}

    # transfer drop vs the model's NITANP test macro-F1
    for key, r in results.items():
        fam, model = key.split("/", 1)
        mp = model_dir(fam, model) / "results" / "CMTE-NITANP" / "final" / "seed42" / "metrics.json"
        if mp.exists():
            indomain = load_json(mp).get("test", {}).get("macro_f1")
            if indomain is not None:
                r["in_domain_macro_f1"] = indomain
                r["transfer_drop"] = round(indomain - r["metrics"]["all"]["macro_f1"], 4)

    atomic_write_json(out_dir / "cross_corpus.json", results)
    lines = ["# Zero-shot transfer: CMTE-NITANP (Split B) → CMTET", "",
             f"CMTET rows: {len(y):,} | multi-cluster: {int(multi.sum()):,} | "
             f"conflicting-label clusters: {int(conflict.sum()):,}", "",
             "| model | in-domain F1 | CMTET F1 | transfer drop | CMTET acc | singleton F1 | conflict F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, r in sorted(results.items(), key=lambda kv: -kv[1]["metrics"]["all"]["macro_f1"]):
        m = r["metrics"]
        lines.append(f"| {key} | {r.get('in_domain_macro_f1', float('nan')):.4f} | "
                     f"{m['all']['macro_f1']:.4f} | {r.get('transfer_drop', float('nan')):+.4f} | "
                     f"{m['all']['accuracy']:.4f} | {m.get('singleton', {}).get('macro_f1', float('nan')):.4f} | "
                     f"{m.get('conflict_cluster', {}).get('macro_f1', float('nan')):.4f} |")
    atomic_write_text(out_dir / "cross_corpus.md", "\n".join(lines) + "\n")
    print(f"wrote {out_dir/'cross_corpus.md'}")


if __name__ == "__main__":
    main()
