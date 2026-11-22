#!/usr/bin/env python3
"""Feature-level hybrid: MuRIL CLS embedding + linear-model probabilities.

Honest protocol: the transformer was trained on train, so its *train* CLS
embeddings are in-sample and would leak labels into a hybrid trained on train.
We therefore train the hybrid on validation (out-of-sample for both the
transformer and the linear models) and evaluate test once.  This is an
exploratory 3,459-row hybrid, not a train-fitted model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "hf_transformers"))

from common.ckpt import load_checkpoint  # noqa: E402
from common.data import IndexDataset, load_records, to_arrays, tokenize_cached  # noqa: E402
from common.paths import CACHE, SPLITS  # noqa: E402
from common.runner import set_seed  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import run_transformer as rt  # noqa: E402

DATASET = "CMTE-NITANP"
SPLIT = "B"
MAX_LEN = 64
OUT = BENCH / "ensemble" / "hybrid"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_cls(model, loader, device, amp):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels")
        with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
            h = model.bert(**batch).last_hidden_state[:, 0, :]
        out.append(h.float().cpu().numpy())
    return np.concatenate(out, 0)


def linear_probs(tag: str):
    p = BENCH / "ensemble" / "preds" / tag / "preds.npz"
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    recs = load_records(DATASET, SPLIT)
    arr = {s: to_arrays(recs[s], "text_clean") for s in SPLITS}
    texts = {s: arr[s][0] for s in SPLITS}
    y = {s: arr[s][1] for s in SPLITS}
    ids = {s: arr[s][3] for s in SPLITS}

    tok = AutoTokenizer.from_pretrained(rt.SPECS["muril_fullft"]["hf"])
    enc = {s: tokenize_cached(tok, texts[s], MAX_LEN, CACHE, f"hybrid:{s}:64")
           for s in ("val", "test")}
    loaders = {s: DataLoader(IndexDataset(enc[s], y[s]), batch_size=32) for s in ("val", "test")}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    ckpt = BENCH / "hf_transformers" / "muril_fullft" / "results" / DATASET / "final" / "seed42" / "checkpoint_best.pt"
    args = SimpleNamespace(grad_checkpointing=False, lora_r=16, lora_alpha=32, lora_dropout=0.1)
    model = rt.build_model(rt.SPECS["muril_fullft"], 42, args).to(device)
    model.load_state_dict(load_checkpoint(ckpt, device)["model"])
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    cls = {s: extract_cls(model, loaders[s], device, amp) for s in ("val", "test")}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Linear probabilities (out-of-sample on val/test)
    lp = {s: {} for s in ("val", "test")}
    ref_ids = {s: [str(x) for x in ids[s]] for s in ("val", "test")}
    for tag in ("logreg_union", "linearsvc_union"):
        d = linear_probs(tag)
        for s in ("val", "test"):
            mid = [str(x) for x in d[f"{s}_ids"]]
            order = [mid.index(x) for x in ref_ids[s]]
            lp[s][tag] = np.asarray(d[f"{s}_proba"], float)[order]

    def build(split):
        return np.hstack([cls[split]] + [lp[split][t] for t in ("logreg_union", "linearsvc_union")])

    Xv, Xt = build("val"), build("test")
    yv, yt = y["val"], y["test"]
    print(f"val {Xv.shape} test {Xt.shape}")

    # Standardise + PCA on the 768-d CLS block only; keep the 6 linear probs.
    sc = StandardScaler().fit(cls["val"])
    Zv, Zt = sc.transform(cls["val"]), sc.transform(cls["test"])
    Lv, Lt = lp["val"]["logreg_union"], lp["val"]["linearsvc_union"]
    Ltest = np.hstack([lp["test"]["logreg_union"], lp["test"]["linearsvc_union"]])
    Lval = np.hstack([Lv, Lt])
    results = []
    for ncomp in (64, 128, 256):
        pca = PCA(n_components=ncomp, random_state=42).fit(Zv)
        Av = np.hstack([pca.transform(Zv), Lval])
        At = np.hstack([pca.transform(Zt), Ltest])
        best = None
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for C in (0.001, 0.01, 0.1, 1.0):
            cv = []
            for tr, va in skf.split(Av, yv):
                clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Av[tr], yv[tr])
                cv.append(f1_score(yv[va], clf.predict(Av[va]), average="macro"))
            m = float(np.mean(cv))
            if best is None or m > best[0]:
                best = (m, C)
        C = best[1]
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Av, yv)
        pred = clf.predict(At)
        res = {"n_components": ncomp, "C": C, "cv_val_f1": round(best[0], 4),
               "test_acc": round(accuracy_score(yt, pred), 4),
               "test_f1": round(f1_score(yt, pred, average="macro"), 4)}
        print(res)
        results.append(res)
        np.savez_compressed(OUT / f"preds_pca{ncomp}.npz", val_pred=clf.predict(Av),
                            val_proba=clf.predict_proba(Av), val_gold=yv,
                            test_pred=pred, test_proba=clf.predict_proba(At), test_gold=yt,
                            test_ids=np.array([str(x) for x in ids["test"]], dtype=object))
    (OUT / "hybrid_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'hybrid_results.json'}")


if __name__ == "__main__":
    main()
