#!/usr/bin/env python3
"""Export validation + test probabilities for trained transformer checkpoints.

The final-phase transformer runs saved test-only `preds.npz`; a val-fitted
blend needs the validation probabilities too.  This script loads the frozen
`checkpoint_best.pt` for each model and runs inference once on val and test.

Output: ensemble/preds_val/<model>/preds.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "hf_transformers"))

from common.ckpt import load_checkpoint, load_trainable_state  # noqa: E402
from common.data import IndexDataset, load_records, to_arrays, tokenize_cached  # noqa: E402
from common.paths import CACHE, SPLITS  # noqa: E402
from common.runner import set_seed  # noqa: E402
import run_transformer as rt  # noqa: E402

DATASET = "CMTE-NITANP"
SPLIT = "B"
TEXT_FIELD = "text_clean"
MAX_LEN = 64
OUT = BENCH / "ensemble" / "preds_val"


def export(model: str) -> None:
    spec = rt.SPECS[model]
    ckpt = BENCH / "hf_transformers" / model / "results" / DATASET / "final" / "seed42" / "checkpoint_best.pt"
    if not ckpt.exists():
        print(f"[skip] {model}: no checkpoint at {ckpt}")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["hf"])
    recs = load_records(DATASET, SPLIT)
    arr = {s: to_arrays(recs[s], TEXT_FIELD) for s in SPLITS}
    texts = {s: arr[s][0] for s in SPLITS}
    y = {s: arr[s][1] for s in SPLITS}
    ids = {s: arr[s][3] for s in SPLITS}
    enc = {s: tokenize_cached(tok, texts[s], MAX_LEN, CACHE, f"export:{model}:{s}:{TEXT_FIELD}")
           for s in SPLITS}
    loaders = {s: DataLoader(IndexDataset(enc[s], y[s]), batch_size=32) for s in SPLITS}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)  # TeluguBERT ships no pooler weights; reproduce the seeded pooler
    args = SimpleNamespace(grad_checkpointing=False, lora_r=16, lora_alpha=32, lora_dropout=0.1)
    model_obj = rt.build_model(spec, 42, args).to(device)
    ck = load_checkpoint(ckpt, device)
    if spec["full_ft"]:
        model_obj.load_state_dict(ck["model"])
    else:
        load_trainable_state(model_obj, ck["model"])

    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    out = {}
    for s in ("val", "test"):
        pred, proba, gold = rt.predict(model_obj, loaders[s], device, amp)
        out[f"{s}_pred"] = pred
        out[f"{s}_proba"] = proba
        out[f"{s}_gold"] = gold
        out[f"{s}_ids"] = np.array([str(i) for i in ids[s]], dtype=object)

    rd = OUT / model
    rd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "preds.npz", **out)
    from sklearn.metrics import accuracy_score, f1_score
    print(f"{model:20s} val_f1={f1_score(out['val_gold'], out['val_proba'].argmax(1), average='macro'):.4f} "
          f"test_f1={f1_score(out['test_gold'], out['test_proba'].argmax(1), average='macro'):.4f} "
          f"test_acc={accuracy_score(out['test_gold'], out['test_proba'].argmax(1)):.4f} -> {rd}")
    del model_obj
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["muril_fullft", "telugu_bert_lora", "muril_lora", "indicbert_lora"])
    a = ap.parse_args()
    for m in a.models:
        export(m)
