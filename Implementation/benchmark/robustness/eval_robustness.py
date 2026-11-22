#!/usr/bin/env python3
"""Orthographic robustness: clean TEST vs perturbed TEST, per model.

Correct comparison: clean baseline is the original splitB_test.jsonl, and every
variant is the same test set with one perturbation applied.

Classical models are refit from their frozen config; transformers from
checkpoint_best.pt; deep models from checkpoint_best.pt.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "classical"))
sys.path.insert(0, str(BENCH / "deep"))
sys.path.insert(0, str(BENCH / "hf_transformers"))

from common.io import atomic_write_json, atomic_write_text, load_json  # noqa: E402
from common.paths import DATASETS, model_dir  # noqa: E402
from common.runner import set_seed  # noqa: E402

LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}
VARIANTS = ("vowel", "voicing", "aspiration", "combined")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def y_of(rows):
    return np.array([LABELS[r["label"]] for r in rows])


def eval_classical(dataset, data_dir):
    from classical.run_classical import build_model, make_vectorizer

    train = load_jsonl(DATASETS[dataset] / "splitB_train.jsonl")
    test = load_jsonl(DATASETS[dataset] / "splitB_test.jsonl")
    tr_texts, te_texts = [r["text_clean"] for r in train], [r["text_clean"] for r in test]
    ytr, yte = y_of(train), y_of(test)
    results = {}
    for mdir in sorted(model_dir("classical", "").parent.iterdir()):
        if not mdir.is_dir():
            continue
        cfgp = mdir / "results" / dataset / "best_config.json"
        if not cfgp.exists():
            continue
        cfg = load_json(cfgp)
        vec = make_vectorizer(cfg["feature"]).fit(tr_texts)
        clf = build_model(mdir.name, cfg["params"], 42).fit(vec.transform(tr_texts), ytr)
        clean = macro_f1(yte, clf.predict(vec.transform(te_texts)))
        row = {"model": mdir.name, "feature": cfg["feature"], "params": cfg["params"],
               "clean": round(clean, 4), "variants": {}}
        for v in VARIANTS:
            p = data_dir / f"{v}.jsonl"
            if not p.exists():
                continue
            rows = load_jsonl(p)
            f1 = macro_f1(y_of(rows), clf.predict(vec.transform([r["text_clean"] for r in rows])))
            row["variants"][v] = {"macro_f1": round(f1, 4), "delta": round(clean - f1, 4)}
        results[mdir.name] = row
        print(f"[robustness] classical {mdir.name} clean={clean:.4f}")
    return results


def _deep_args():
    return SimpleNamespace(emb_dim=100, filters=100, hidden=64, dropout=0.3)


def eval_deep(dataset, data_dir):
    from common.ckpt import load_checkpoint
    from common.paths import CACHE
    from deep.run_deep import PAD, build_model, build_vocab, encode_word_cached, predict as dpredict
    from torch.utils.data import DataLoader
    from common.data import IndexDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    train = load_jsonl(DATASETS[dataset] / "splitB_train.jsonl")
    test = load_jsonl(DATASETS[dataset] / "splitB_test.jsonl")
    vocab = build_vocab([r["text_clean"] for r in train])
    results = {}
    for model in ("textcnn", "bilstm", "cnn_bilstm"):
        rd = model_dir("deep", model) / "results" / dataset / "final" / "seed42"
        ck = rd / "checkpoint_best.pt"
        if not ck.exists():
            continue
        meta = load_json(rd / "metrics.json")
        max_len = meta.get("max_len", 64)
        obj = build_model(model, len(vocab), PAD, _deep_args()).to(device)
        obj.load_state_dict(load_checkpoint(ck, device)["model"])

        def run(rows):
            enc = encode_word_cached([r["text_clean"] for r in rows], vocab, max_len,
                                     f"{dataset}:{meta.get('split_name','B')}:text_clean:word")
            ds = IndexDataset(enc, y_of(rows))
            pred, _, gold = dpredict(obj, DataLoader(ds, batch_size=128), device, amp)
            return macro_f1(gold, pred)

        clean = run(test)
        row = {"model": model, "clean": round(clean, 4), "variants": {}}
        for v in VARIANTS:
            p = data_dir / f"{v}.jsonl"
            if p.exists():
                f1 = run(load_jsonl(p))
                row["variants"][v] = {"macro_f1": round(f1, 4), "delta": round(clean - f1, 4)}
        results[model] = row
        print(f"[robustness] deep {model} clean={clean:.4f}")
    return results


def eval_transformers(dataset, data_dir):
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging

    from common.ckpt import load_checkpoint, load_trainable_state
    from common.data import IndexDataset
    from hf_transformers.run_transformer import SPECS, predict as tpredict

    hf_logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    test = load_jsonl(DATASETS[dataset] / "splitB_test.jsonl")
    results = {}
    for model, spec in SPECS.items():
        rd = model_dir("transformers", model) / "results" / dataset / "final" / "seed42"
        mp, ckpt = rd / "metrics.json", rd / "checkpoint_best.pt"
        if not ckpt.exists() and (rd / "checkpoint_best.path").exists():
            ckpt = Path((rd / "checkpoint_best.path").read_text().strip())
        if not mp.exists() or not ckpt.exists():
            continue
        meta = load_json(mp)
        max_len = meta.get("hyper", {}).get("max_len", 64)
        tok = AutoTokenizer.from_pretrained(spec["hf"])
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

        def run(rows):
            enc = tok([r["text_clean"] for r in rows], truncation=True, padding="max_length",
                      max_length=max_len, return_tensors="pt")
            ds = IndexDataset(dict(enc), y_of(rows))
            pred, _, gold = tpredict(obj, DataLoader(ds, batch_size=32), device, amp)
            return macro_f1(gold, pred)

        clean = run(test)
        row = {"model": model, "clean": round(clean, 4), "variants": {}}
        for v in VARIANTS:
            p = data_dir / f"{v}.jsonl"
            if p.exists():
                f1 = run(load_jsonl(p))
                row["variants"][v] = {"macro_f1": round(f1, 4), "delta": round(clean - f1, 4)}
        results[model] = row
        print(f"[robustness] transformer {model} clean={clean:.4f}")
        del obj
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--families", nargs="+", default=["classical", "deep", "transformers"])
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "data" / args.dataset
    out_dir = Path(__file__).resolve().parent / "results" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    if "classical" in args.families:
        all_results["classical"] = eval_classical(args.dataset, data_dir)
    if "deep" in args.families:
        all_results["deep"] = eval_deep(args.dataset, data_dir)
    if "transformers" in args.families:
        all_results["transformers"] = eval_transformers(args.dataset, data_dir)

    atomic_write_json(out_dir / "robustness.json", all_results)
    lines = ["# Orthographic robustness (clean test vs perturbed test)", "",
             "| model | clean | " + " | ".join(VARIANTS) + " |",
             "|---|---:|" + "---:|" * len(VARIANTS)]
    for fam, rows in all_results.items():
        for model, r in rows.items():
            cells = []
            for v in VARIANTS:
                d = r["variants"].get(v)
                cells.append(f"{d['macro_f1']:.4f} (Δ{d['delta']:+.4f})" if d else "-")
            lines.append(f"| {fam}/{model} | {r['clean']:.4f} | " + " | ".join(cells) + " |")
    atomic_write_text(out_dir / "robustness.md", "\n".join(lines) + "\n")
    print(f"wrote {out_dir/'robustness.md'}")


if __name__ == "__main__":
    main()
