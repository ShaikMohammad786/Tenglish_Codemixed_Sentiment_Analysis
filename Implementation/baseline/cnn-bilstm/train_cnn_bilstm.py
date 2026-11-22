#!/usr/bin/env python3
"""Baseline 2: CNN + BiLSTM, evaluated on Split A and Split B.

Architecture (small, CPU-friendly): Embedding -> Conv1d -> ReLU -> BiLSTM ->
masked mean-pool -> Dropout -> Linear(3).  The embedding table is either a
from-scratch word vocabulary (built from train only) or a frozen WordPiece
subword vocabulary taken from a HuggingFace tokenizer.  The best epoch is
selected on validation macro-F1 (never on test).

    leakage gap = macroF1(Split A) - macroF1(Split B)

Everything except the tokenizer kind is held fixed, so the two runs are a
controlled comparison of word-level vs subword segmentation for this model.

Input  : datasets/splits/<dataset>/split{A,B}_{train,val,test}.jsonl
Outputs: Implementation/baseline/cnn-bilstm/results/<dataset>/{cnn_bilstm_results[_<tokenizer>].md, .json}

Run with the project venv:
    .venv/bin/python Implementation/baseline/cnn-bilstm/train_cnn_bilstm.py
    .venv/bin/python Implementation/baseline/cnn-bilstm/train_cnn_bilstm.py \
        --tokenizer wordpiece --tokenizer-name google/muril-base-cased
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "val", "test")
LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}
PAD, UNK = 0, 1


def load_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in texts:
        for tok in t.split():
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in sorted(counts.items()):  # sorted => deterministic ids
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_word(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, tok in enumerate(t.split()[:max_len]):
            out[i, j] = vocab.get(tok, UNK)
    return torch.from_numpy(out)


def encode_wordpiece(texts: list[str], tok, max_len: int) -> torch.Tensor:
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len,
              return_tensors="np")
    return torch.from_numpy(enc["input_ids"].astype("int64"))


def raw_token_counts(texts: list[str], tokenizer: str, tok=None) -> np.ndarray:
    """Token count per comment before truncation (word split or WordPiece)."""
    if tokenizer == "word":
        return np.array([len(t.split()) for t in texts])
    return np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb: int, filters: int, hidden: int, n_class: int,
                 dropout: float, pad_id: int = PAD):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.conv = nn.Conv1d(emb, filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(self.emb(x).transpose(1, 2))).transpose(1, 2)  # B,L,filters
        out, _ = self.lstm(h)  # B,L,2*hidden
        mask = (x != self.pad_id).unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean pool
        return self.fc(self.drop(pooled))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def evaluate_split(splits_dir: Path, name: str, text_field: str, args) -> dict:
    parts = {s: load_split(splits_dir / f"split{name}_{s}.jsonl") for s in SPLITS}
    text = {s: [r[text_field] for r in parts[s]] for s in SPLITS}
    y = {s: np.array([LABELS[r["label"]] for r in parts[s]]) for s in SPLITS}

    if args.tokenizer == "word":
        vocab = build_vocab(text["train"])
        vocab_size, pad_id, tok = len(vocab), PAD, None
        encoder = lambda ts: encode_word(ts, vocab, args.max_len)  # noqa: E731
    else:
        from transformers import AutoTokenizer, logging as hf_logging  # lazy: only needed here
        hf_logging.set_verbosity_error()  # raw counts exceed model_max_length; truncation happens later
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        vocab_size, pad_id = tok.vocab_size, (tok.pad_token_id or PAD)
        encoder = lambda ts: encode_wordpiece(ts, tok, args.max_len)  # noqa: E731

    X = {s: encoder(text[s]) for s in SPLITS}
    raw = {s: raw_token_counts(text[s], args.tokenizer, tok) for s in SPLITS}
    yt = {s: torch.from_numpy(y[s]) for s in SPLITS}
    loaders = {s: DataLoader(TensorDataset(X[s], yt[s]), batch_size=args.batch_size,
                             shuffle=(s == "train")) for s in SPLITS}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNBiLSTM(vocab_size, args.emb_dim, args.filters, args.hidden, len(LABELS),
                      args.dropout, pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        val_f1 = f1_score(y["val"], predict(model, loaders["val"], device), average="macro")
        history.append({"epoch": epoch, "train_loss": round(running / len(X["train"]), 4),
                        "val_macro_f1": round(val_f1, 4)})
        print(f"  [{name}] epoch {epoch}/{args.epochs}  loss={running / len(X['train']):.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pred = predict(model, loaders["test"], device)
    return {
        "split": name,
        "rows": {s: len(parts[s]) for s in SPLITS},
        "vocab": vocab_size,
        "pad_id": pad_id,
        "tokens_per_item_mean": round(float(raw["test"].mean()), 2),
        "truncation_pct": round(float((raw["test"] > args.max_len).mean() * 100), 2),
        "device": device,
        "best_val_macro_f1": round(best_val, 4),
        "test_accuracy": round(accuracy_score(y["test"], pred), 4),
        "test_macro_f1": round(f1_score(y["test"], pred, average="macro"), 4),
        "per_class": classification_report(y["test"], pred, digits=4, zero_division=0,
                                           target_names=list(LABELS)),
        "history": history,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=root / "datasets/splits/CMTE-NITANP")
    ap.add_argument("--text-field", choices=("text_clean", "text_norm"), default="text_clean")
    ap.add_argument("--tokenizer", choices=("word", "wordpiece"), default="word",
                    help="word = from-scratch whitespace vocab; wordpiece = HuggingFace subwords")
    ap.add_argument("--tokenizer-name", default="google/muril-base-cased",
                    help="HF tokenizer id used when --tokenizer wordpiece")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--filters", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to results/<dataset>, next to the other per-dataset outputs")
    args = ap.parse_args()
    if args.out_dir is None:  # one results folder per dataset, mirroring datasets/splits/<dataset>
        args.out_dir = Path(__file__).resolve().parent / "results" / args.splits_dir.name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {name: evaluate_split(args.splits_dir, name, args.text_field, args)
               for name in ("A", "B")}
    a, b = results["A"], results["B"]
    gap_f1 = a["test_macro_f1"] - b["test_macro_f1"]
    gap_acc = a["test_accuracy"] - b["test_accuracy"]

    print(f"\nCNN + BiLSTM  (tokenizer={args.tokenizer}, text={args.text_field}, device={a['device']})")
    print(f"{'split':6} {'test acc':>9} {'test macro-F1':>14}")
    for r in (a, b):
        print(f"{r['split']:6} {r['test_accuracy']:>9.4f} {r['test_macro_f1']:>14.4f}")
    print(f"\nLeakage gap (A - B): macro-F1 {gap_f1:+.4f} | accuracy {gap_acc:+.4f}")

    suffix = "" if args.tokenizer == "word" else f"_{args.tokenizer}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "results": results, "gap": {"macro_f1": round(gap_f1, 4), "accuracy": round(gap_acc, 4)}}
    (args.out_dir / f"cnn_bilstm_results{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / f"cnn_bilstm_results{suffix}.md").write_text(build_md(args, a, b, gap_f1, gap_acc), encoding="utf-8")
    print(f"\nWrote:\n  {(args.out_dir / f'cnn_bilstm_results{suffix}.md').resolve()}\n  {(args.out_dir / f'cnn_bilstm_results{suffix}.json').resolve()}")


def build_md(args, a, b, gap_f1, gap_acc) -> str:
    tok = args.tokenizer if args.tokenizer == "word" else f"wordpiece:{args.tokenizer_name}"
    out = ["# Baseline: CNN + BiLSTM", "",
           f"text field: `{args.text_field}` | tokenizer: `{tok}` | splits: `{args.splits_dir}` | "
           f"device: {a['device']} | epochs: {args.epochs} | emb: {args.emb_dim} | filters: {args.filters} | "
           f"hidden: {args.hidden} | max_len: {args.max_len} | seed: {args.seed}", "",
           "| split | rows (train/val/test) | vocab | tokens/item | trunc % | val macro-F1 | test accuracy | test macro-F1 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in (a, b):
        rows = " / ".join(f"{r['rows'][s]:,}" for s in SPLITS)
        out.append(f"| {r['split']} | {rows} | {r['vocab']:,} | {r['tokens_per_item_mean']:.2f} | "
                   f"{r['truncation_pct']:.2f} | {r['best_val_macro_f1']:.4f} | "
                   f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |")
    out += ["", f"**Leakage gap (A - B): macro-F1 {gap_f1:+.4f} | accuracy {gap_acc:+.4f}**", "",
            "Best epoch chosen on val macro-F1; test never used for tuning.", "",
            "## Per-class report", "", "### Split A", "", "```", a["per_class"].rstrip(), "```", "",
            "### Split B", "", "```", b["per_class"].rstrip(), "```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
