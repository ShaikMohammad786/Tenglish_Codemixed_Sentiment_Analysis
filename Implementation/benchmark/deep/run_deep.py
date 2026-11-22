#!/usr/bin/env python3
"""Generic deep-learning runner: TextCNN, BiLSTM, CNN+BiLSTM.

Per-epoch checkpoints (last + best), resume, early stopping, best-epoch restore,
per-run logs/metrics/preds. Each model has its own folder + results folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ckpt import load_checkpoint, save_checkpoint  # noqa: E402
from common.data import IndexDataset, load_records, to_arrays, tokenize_cached  # noqa: E402
from common.io import (  # noqa: E402
    RunLogger,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_text,
    write_status,
)
from common.metrics import compute_metrics  # noqa: E402
from common.paths import CACHE, LABEL_NAMES, SPLITS, model_dir  # noqa: E402
from common.runner import EarlyStopping, Timer, set_seed, status_line  # noqa: E402

PAD, UNK = 0, 1
MODELS = ("textcnn", "bilstm", "cnn_bilstm")


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in texts:
        for tok in t.split():
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in sorted(counts.items()):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_word(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, tok in enumerate(t.split()[:max_len]):
            out[i, j] = vocab.get(tok, UNK)
    return torch.from_numpy(out)


def encode_word_cached(texts, vocab, max_len, tag):
    key = sha256_text(f"{tag}|{max_len}|{len(texts)}|{sha256_text(chr(10).join(texts))[:16]}")
    d = CACHE / "tokenized" / key
    f = d / "enc.pt"
    if f.exists():
        return torch.load(f, map_location="cpu")
    enc = {"input_ids": encode_word(texts, vocab, max_len)}
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "enc.pt.tmp"
    torch.save(enc, tmp)
    tmp.replace(f)
    return enc


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=100, filters=100, n_class=3, dropout=0.3, pad_id=PAD):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k, padding=k // 2) for k in (3, 4, 5)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * 3, n_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x = self.emb(input_ids).transpose(1, 2)
        hs = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        logits = self.fc(self.drop(torch.cat(hs, 1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb=100, hidden=128, n_class=3, dropout=0.3, pad_id=PAD):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        mask = (input_ids != self.pad_id).unsqueeze(-1)
        out, _ = self.lstm(self.emb(input_ids))
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.fc(self.drop(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=100, filters=100, hidden=64, n_class=3, dropout=0.3, pad_id=PAD):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.conv = nn.Conv1d(emb, filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        h = torch.relu(self.conv(self.emb(input_ids).transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(h)
        mask = (input_ids != self.pad_id).unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.fc(self.drop(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def build_model(name, vocab_size, pad_id, args):
    kw = dict(vocab_size=vocab_size, emb=args.emb_dim, n_class=3, dropout=args.dropout, pad_id=pad_id)
    if name == "textcnn":
        return TextCNN(filters=args.filters, **kw)
    if name == "bilstm":
        return BiLSTMModel(hidden=args.hidden, **kw)
    if name == "cnn_bilstm":
        return CNNBiLSTM(filters=args.filters, hidden=args.hidden, **kw)
    raise ValueError(name)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    preds, gold, probs = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
            out = model(**batch)
        p = torch.softmax(out["logits"].float(), dim=-1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
        gold.append(labels.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(gold)


def main(default_model: str | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model, required=default_model is None, choices=MODELS)
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--split-name", default="B", choices=("A", "B"))
    ap.add_argument("--text-field", default="text_clean")
    ap.add_argument("--tokenizer", choices=("word", "wordpiece"), default="word")
    ap.add_argument("--tokenizer-name", default="google/muril-base-cased")
    ap.add_argument("--config", default="default")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--filters", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rd = model_dir("deep", args.model) / "results" / args.dataset / args.config / f"seed{args.seed}"
    if (rd / "run_status.json").exists() and not args.force:
        if load_json(rd / "run_status.json").get("status") == "done":
            print(f"[skip] {args.model} already done")
            return

    set_seed(args.seed)
    rd.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with RunLogger(rd) as log:
        write_status(rd, "running", model=args.model, seed=args.seed)
        recs = load_records(args.dataset, args.split_name)
        arrays = {s: to_arrays(recs[s], args.text_field) for s in SPLITS}
        texts = {s: arrays[s][0] for s in SPLITS}
        y = {s: arrays[s][1] for s in SPLITS}
        ids = {s: arrays[s][3] for s in SPLITS}

        if args.tokenizer == "word":
            vocab = build_vocab(texts["train"])
            enc = {s: encode_word_cached(texts[s], vocab, args.max_len,
                                         f"{args.dataset}:{args.split_name}:{args.text_field}:word")
                   for s in SPLITS}
            vocab_size, pad_id = len(vocab), PAD
        else:
            from transformers import AutoTokenizer, logging as hf_logging
            hf_logging.set_verbosity_error()
            tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
            enc = {s: tokenize_cached(tok, texts[s], args.max_len, CACHE,
                                      f"{args.tokenizer_name}:{args.dataset}:{args.split_name}:{args.text_field}")
                   for s in SPLITS}
            vocab_size, pad_id = tok.vocab_size, (tok.pad_token_id or PAD)

        ds = {s: IndexDataset(enc[s], y[s]) for s in SPLITS}
        loaders = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=(s == "train")) for s in SPLITS}
        model = build_model(args.model, vocab_size, pad_id, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        est = EarlyStopping(patience=args.patience, min_epochs=args.min_epochs)
        log.log(f"model={args.model} device={device} amp={amp} vocab={vocab_size:,} rows={ {s: len(y[s]) for s in SPLITS} }")

        start_epoch = 0
        last_path = rd / "checkpoint_last.pt"
        best_path = rd / "checkpoint_best.pt"
        if last_path.exists() and not args.force:
            ck = load_checkpoint(last_path, device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"]
            est.best, est.best_epoch, est.bad = ck.get("best"), ck.get("best_epoch", 0), ck.get("bad", 0)
            log.log(f"resumed from epoch {start_epoch}")

        history = []
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running, seen = 0.0, 0
            for batch in loaders["train"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                    loss = model(**batch)["loss"]
                loss.backward()
                opt.step()
                running += loss.item() * len(batch["labels"])
                seen += len(batch["labels"])
            val_pred, _, val_gold = predict(model, loaders["val"], device, amp)
            val_f1 = float(f1_score(val_gold, val_pred, average="macro", zero_division=0))
            improved, stop = est.step(val_f1, epoch + 1)
            history.append({"epoch": epoch + 1, "train_loss": round(running / seen, 4),
                            "val_macro_f1": round(val_f1, 4), "best": bool(improved)})
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                     "best": est.best, "best_epoch": est.best_epoch, "bad": est.bad}
            save_checkpoint(last_path, state)
            if improved:
                save_checkpoint(best_path, {"model": model.state_dict(), "epoch": epoch + 1,
                                            "val_macro_f1": val_f1})
            vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
            log.log(status_line(args.model, args.config, args.seed, epoch + 1, args.epochs,
                                val_f1, est.best, timer.fmt(), vram))
            if stop:
                log.log(f"early stop at epoch {epoch + 1} (best epoch {est.best_epoch})")
                break

        if best_path.exists():
            model.load_state_dict(load_checkpoint(best_path, device)["model"])
        test_pred, test_proba, test_gold = predict(model, loaders["test"], device, amp)
        metrics = compute_metrics(test_gold, test_pred, probs=test_proba)
        np.savez_compressed(rd / "preds.npz", test_pred=test_pred, test_proba=test_proba,
                            test_gold=test_gold, test_ids=np.array([str(i) for i in ids["test"]], dtype=object))
        result = {
            "run_id": f"deep/{args.model}/{args.dataset}/{args.config}/seed{args.seed}",
            "family": "deep", "model": args.model, "phase": "final",
            "config": args.config, "seed": args.seed,
            "dataset": args.dataset, "split_name": args.split_name, "text_field": args.text_field,
            "tokenizer": args.tokenizer, "vocab": vocab_size, "rows": {s: len(y[s]) for s in SPLITS},
            "best_epoch": est.best_epoch, "best_val_macro_f1": round(float(est.best), 4),
            "test": metrics, "history": history, "elapsed_s": round(timer.elapsed(), 1),
        }
        atomic_write_json(rd / "metrics.json", result)
        atomic_write_json(rd / "history.json", history)
        atomic_write_text(rd / "per_class.txt", metrics["report"])
        atomic_write_text(rd / "report.md", build_report(result))
        write_status(rd, "done", elapsed_s=round(timer.elapsed(), 1),
                     test_macro_f1=metrics["macro_f1"], test_accuracy=metrics["accuracy"])
        log.log(f"DONE test_macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} elapsed={timer.fmt()}")


def build_report(r: dict) -> str:
    t = r["test"]
    lines = [f"# {r['model']} ({r['tokenizer']})", "",
             f"config: `{r['config']}` | seed: {r['seed']} | dataset: {r['dataset']} | split: {r['split_name']} | "
             f"vocab: {r['vocab']:,} | best epoch: {r['best_epoch']}", "",
             "| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |",
             "|---|---:|---:|---:|---:|---:|",
             f"| test | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['weighted_f1']:.4f} | "
             f"{t['mcc']:.4f} | {t.get('ece', float('nan')):.4f} |", "",
             "## Per-class (test)", "", "```", t["report"].rstrip(), "```", "",
             "## Epoch history", "", "| epoch | train loss | val macro-F1 | best |", "|---:|---:|---:|:---:|"]
    for h in r["history"]:
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_macro_f1']:.4f} | {h['best']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
