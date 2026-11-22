#!/usr/bin/env python3
"""Baseline 3: MuRIL-base fine-tuned with LoRA, evaluated on Split A and Split B.

Parameter-efficient fine-tuning of google/muril-base-cased for 3-class sentiment.
LoRA adapters (r=16, alpha=32) are injected into the attention query/value
projections; the base model stays frozen.  The best epoch is selected on
validation macro-F1 (never on test).

    leakage gap = macroF1(Split A) - macroF1(Split B)

Input  : datasets/splits/<dataset>/split{A,B}_{train,val,test}.jsonl
Outputs: Implementation/baseline/muril/results/<dataset>/{muril_results.md, .json}

Run with the project venv (needs a CUDA GPU for reasonable speed):
    .venv/bin/python Implementation/baseline/muril/train_muril_lora.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SPLITS = ("train", "val", "test")
LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}


def load_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class CommentDataset(Dataset):
    """Tokenized comments padded to a fixed length, with integer labels."""

    def __init__(self, rows: list[dict], tok, text_field: str, max_len: int):
        enc = tok([r[text_field] for r in rows], truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        self.enc = dict(enc)
        self.labels = torch.tensor([LABELS[r["label"]] for r in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}


@torch.no_grad()
def predict(model, loader, device, amp) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gold = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels").cpu().numpy()
        with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
            logits = model(**batch).logits
        preds.append(logits.float().argmax(-1).cpu().numpy())
        gold.append(y)
    return np.concatenate(preds), np.concatenate(gold)


def train_split(parts: dict[str, list[dict]], name: str, args, tok) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    ds = {s: CommentDataset(parts[s], tok, args.text_field, args.max_len) for s in SPLITS}
    loaders = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=(s == "train"))
               for s in SPLITS}

    from peft import LoraConfig, get_peft_model

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS))
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.lora_targets, bias="none", task_type="SEQ_CLS")
    model = get_peft_model(model, lora).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{name}] trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loaders["train"]) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:  # linear warmup, then linear decay
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        opt.zero_grad()
        for step, batch in enumerate(loaders["train"], 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum * len(batch["labels"])
            seen += len(batch["labels"])
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

        val_pred, val_gold = predict(model, loaders["val"], device, amp)
        val_f1 = f1_score(val_gold, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(running / seen, 4),
                        "val_macro_f1": round(float(val_f1), 4)})
        print(f"  [{name}] epoch {epoch}/{args.epochs}  loss={running / seen:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in best_state:
                v.copy_(best_state[k].to(v.device))

    test_pred, test_gold = predict(model, loaders["test"], device, amp)
    return {
        "split": name,
        "rows": {s: len(parts[s]) for s in SPLITS},
        "trainable_params": trainable,
        "total_params": total,
        "amp_dtype": str(amp).replace("torch.", ""),
        "device": device,
        "best_val_macro_f1": round(float(best_val), 4),
        "test_accuracy": round(float(accuracy_score(test_gold, test_pred)), 4),
        "test_macro_f1": round(float(f1_score(test_gold, test_pred, average="macro")), 4),
        "per_class": classification_report(test_gold, test_pred, digits=4, zero_division=0,
                                           target_names=list(LABELS)),
        "history": history,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=root / "datasets/splits/CMTE-NITANP")
    ap.add_argument("--text-field", choices=("text_clean", "text_norm"), default="text_clean")
    ap.add_argument("--model-name", default="google/muril-base-cased")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-targets", nargs="+", default=["query", "value"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", choices=("A", "B"), default=["A", "B"],
                    help="which split families to train and evaluate")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to results/<dataset>, next to the other per-dataset outputs")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = Path(__file__).resolve().parent / "results" / args.splits_dir.name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    results = {}
    for name in args.splits:
        parts = {s: load_split(args.splits_dir / f"split{name}_{s}.jsonl") for s in SPLITS}
        results[name] = train_split(parts, name, args, tok)

    a, b = results.get("A"), results.get("B")
    gap = None
    if a and b:
        gap = {"macro_f1": round(a["test_macro_f1"] - b["test_macro_f1"], 4),
               "accuracy": round(a["test_accuracy"] - b["test_accuracy"], 4)}

    print(f"\nMuRIL-base + LoRA  (text={args.text_field}, device={next(iter(results.values()))['device']}, "
          f"amp={next(iter(results.values()))['amp_dtype']})")
    print(f"{'split':6} {'test acc':>9} {'test macro-F1':>14}")
    for r in results.values():
        print(f"{r['split']:6} {r['test_accuracy']:>9.4f} {r['test_macro_f1']:>14.4f}")
    if gap:
        print(f"\nLeakage gap (A - B): macro-F1 {gap['macro_f1']:+.4f} | accuracy {gap['accuracy']:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "results": results, "gap": gap}
    (args.out_dir / "muril_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / "muril_results.md").write_text(build_md(args, results, gap), encoding="utf-8")
    print(f"\nWrote:\n  {(args.out_dir / 'muril_results.md').resolve()}\n  {(args.out_dir / 'muril_results.json').resolve()}")


def build_md(args, results, gap) -> str:
    first = next(iter(results.values()))
    out = ["# Baseline: MuRIL-base + LoRA", "",
           f"model: `{args.model_name}` | adapter: LoRA r={args.lora_r} alpha={args.lora_alpha} "
           f"({', '.join(args.lora_targets)}) | text field: `{args.text_field}` | splits: `{args.splits_dir}` | "
           f"device: {first['device']} ({first['amp_dtype']}) | epochs: {args.epochs} | max_len: {args.max_len} | "
           f"batch: {args.batch_size}x{args.grad_accum} | lr: {args.lr} | seed: {args.seed}", "",
           f"trainable params: {first['trainable_params']:,} / {first['total_params']:,} "
           f"({100 * first['trainable_params'] / first['total_params']:.2f}%)", "",
           "| split | rows (train/val/test) | val macro-F1 | test accuracy | test macro-F1 |",
           "|---|---|---:|---:|---:|"]
    for r in results.values():
        rows = " / ".join(f"{r['rows'][s]:,}" for s in SPLITS)
        out.append(f"| {r['split']} | {rows} | {r['best_val_macro_f1']:.4f} | "
                   f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |")
    if gap:
        out += ["", f"**Leakage gap (A - B): macro-F1 {gap['macro_f1']:+.4f} | accuracy {gap['accuracy']:+.4f}**"]
    out += ["", "Best checkpoint chosen on val macro-F1; test never used for tuning.", "",
            "## Per-class report", ""]
    for r in results.values():
        out += [f"### Split {r['split']}", "", "```", r["per_class"].rstrip(), "```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
