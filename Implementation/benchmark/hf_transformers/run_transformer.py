#!/usr/bin/env python3
"""Transformer runner (LoRA + optional full fine-tuning).

- 10 epochs max, early stopping (patience 2), best-on-val restore.
- Per-epoch atomic checkpoints (last + best) with resume.
- Tokenization cache.
- SEARCH freezes hyperparameters; FINAL uses the frozen config for every seed
  (seeds change only init/shuffle, never lr/epochs/rank).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# All benchmark models are cached locally; never block on a Hub network call.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ckpt import load_checkpoint, load_trainable_state, save_checkpoint, trainable_state  # noqa: E402
from common.data import IndexDataset, load_records, to_arrays, tokenize_cached  # noqa: E402
from common.io import (  # noqa: E402
    RunLogger,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_json,
    utcnow,
    write_status,
)
from common.metrics import compute_metrics  # noqa: E402
from common.paths import CACHE, SPLITS, model_dir  # noqa: E402
from common.runner import EarlyStopping, Timer, set_seed, status_line  # noqa: E402

SPECS = {
    "muril_lora": {"hf": "google/muril-base-cased", "targets": ["query", "value"], "full_ft": False},
    "xlmr_lora": {"hf": "xlm-roberta-base", "targets": ["query", "value"], "full_ft": False},
    "indicbert_lora": {"hf": "ai4bharat/IndicBERTv2-MLM-only", "targets": ["query", "value"], "full_ft": False},
    "telugu_bert_lora": {"hf": "l3cube-pune/telugu-bert", "targets": ["query", "value"], "full_ft": False},
    "mbert_lora": {"hf": "bert-base-multilingual-cased", "targets": ["query", "value"], "full_ft": False},
    "distilmbert_lora": {"hf": "distilbert-base-multilingual-cased", "targets": ["q_lin", "v_lin"], "full_ft": False},
    "muril_fullft": {"hf": "google/muril-base-cased", "targets": [], "full_ft": True},
    "bert_base": {"hf": "/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/models/bert-base-uncased", "targets": [], "full_ft": True},
    "electra_base": {"hf": "/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/models/electra-base-discriminator", "targets": [], "full_ft": True},
}


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    preds, probs, gold = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels").cpu().numpy()
        with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
            logits = model(**batch).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(1))
        gold.append(y)
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(gold)


def build_model(spec, seed, args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(spec["hf"], num_labels=3)
    model.config.use_cache = False
    if spec["full_ft"]:
        for p in model.parameters():
            p.requires_grad_(True)
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable()
        return model
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=spec["targets"], bias="none", task_type="SEQ_CLS")
    return get_peft_model(model, lora)


def main(default_model: str | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model, required=default_model is None, choices=list(SPECS))
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--split-name", default="B", choices=("A", "B"))
    ap.add_argument("--text-field", default="text_clean")
    ap.add_argument("--phase", choices=("search", "final", "sensitivity"), default="final")
    ap.add_argument("--config", default=None)
    ap.add_argument("--best-config", type=Path, default=None)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="load model weights only (no optimizer/epoch) before training")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="freeze the base embedding matrix (saves ~600 MB of gradient on 4 GB GPUs)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--neutral-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = SPECS[args.model]
    if spec["full_ft"]:
        # 4 GB budget: MuRIL needs batch 8; BERT/Electra (110M) fit the paper's batch 16.
        max_bs = 16 if any(x in spec["hf"] for x in ("bert-base-uncased", "electra-base-discriminator")) else 8
        args.batch_size = min(args.batch_size, max_bs)
        args.grad_accum = max(args.grad_accum, 32 // args.batch_size)
        args.grad_checkpointing = True
    default_lr = 2e-5 if spec["full_ft"] else args.lr or 4e-4
    config = args.config or ("search" if args.phase == "search"
                             else "sensitivity" if args.phase == "sensitivity" else "final")
    root = model_dir("transformers", args.model)
    best_path = args.best_config or (root / "results" / args.dataset / "best_config.json")
    rd = root / "results" / args.dataset / config / f"seed{args.seed}"

    attempt = 1
    if (rd / "run_status.json").exists() and not args.force:
        st = load_json(rd / "run_status.json")
        if st.get("status") == "done":
            print(f"[skip] {args.model} phase={args.phase} seed={args.seed} already done")
            return
        if st.get("status") in ("failed", "skipped"):
            attempt = int(st.get("attempts", 1)) + 1
            if attempt > 3:
                print(f"[skip] {args.model} failed {attempt - 1} times; giving up")
                write_status(rd, "skipped", attempts=attempt - 1, error=st.get("error"))
                return

    set_seed(args.seed)
    rd.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with RunLogger(rd) as log:
        write_status(rd, "running", model=args.model, phase=args.phase, seed=args.seed, attempts=attempt)
        frozen = None
        if args.phase == "final":
            if best_path.exists():
                frozen = load_json(best_path)
                lr = args.lr or frozen["hyper"]["lr"]
                log.log(f"FINAL: frozen hyper={frozen['hyper']} hash={frozen['hash'][:12]}")
            else:
                lr = args.lr or default_lr
                frozen = {"model": args.model, "dataset": args.dataset, "split_name": args.split_name,
                          "hyper": {"lr": lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                                    "max_len": args.max_len, "epochs": args.epochs,
                                    "batch_size": args.batch_size, "grad_accum": args.grad_accum},
                          "frozen_from": "args", "frozen_at": utcnow()}
                frozen["hash"] = sha256_json(frozen["hyper"])
                atomic_write_json(best_path, frozen)
                log.log(f"FROZE (from args) {best_path.name}: {frozen['hyper']}")
        else:
            lr = args.lr or default_lr

        recs = load_records(args.dataset, args.split_name)
        arr = {s: to_arrays(recs[s], args.text_field) for s in SPLITS}
        texts = {s: arr[s][0] for s in SPLITS}
        y = {s: arr[s][1] for s in SPLITS}
        ids = {s: arr[s][3] for s in SPLITS}

        from transformers import AutoTokenizer, logging as hf_logging
        hf_logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        enc = {s: tokenize_cached(tok, texts[s], args.max_len, CACHE,
                                  f"{spec['hf']}:{args.dataset}:{args.split_name}:{args.text_field}")
               for s in SPLITS}
        ds = {s: IndexDataset(enc[s], y[s]) for s in SPLITS}
        loaders = {s: DataLoader(ds[s], batch_size=args.batch_size, shuffle=(s == "train")) for s in SPLITS}

        model = build_model(spec, args.seed, args).to(device)
        if args.init_from:
            ck = load_checkpoint(args.init_from, device)
            if spec["full_ft"]:
                model.load_state_dict(ck["model"])
            else:
                load_trainable_state(model, ck["model"])
            log.log(f"initialized model weights from {args.init_from} (fresh optimizer/epoch)")
        if args.freeze_embeddings and spec["full_ft"]:
            base = getattr(model, model.base_model_prefix, None)
            if base is not None and hasattr(base, "embeddings"):
                for p in base.embeddings.parameters():
                    p.requires_grad_(False)
                log.log("frozen base embeddings (saves the ~600 MB embedding gradient)")
            else:
                log.log("freeze-embeddings requested but base embeddings not found")
        params = [p for p in model.parameters() if p.requires_grad]
        if spec["full_ft"]:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        total_steps = max(1, len(loaders["train"]) // args.grad_accum) * args.epochs
        warmup = int(args.warmup_ratio * total_steps)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        est = EarlyStopping(patience=args.patience, min_epochs=args.min_epochs)
        n_trainable = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in model.parameters())
        log.log(f"model={args.model} hf={spec['hf']} full_ft={spec['full_ft']} device={device} amp={amp}")
        log.log(f"trainable={n_trainable:,}/{n_total:,} lr={lr} batch={args.batch_size}x{args.grad_accum}")

        ckdir = Path(os.environ["BENCH_CKPT_DIR"]) if os.environ.get("BENCH_CKPT_DIR") else rd
        ckdir.mkdir(parents=True, exist_ok=True)
        last_path, best_path_ckpt = ckdir / "checkpoint_last.pt", ckdir / "checkpoint_best.pt"
        start_epoch = 0
        if last_path.exists() and not args.force:
            ck = load_checkpoint(last_path, device)
            if spec["full_ft"]:
                model.load_state_dict(ck["model"])
            else:
                load_trainable_state(model, ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"]
            est.best, est.best_epoch, est.bad = ck.get("best"), ck.get("best_epoch", 0), ck.get("bad", 0)
            log.log(f"resumed from epoch {start_epoch}")

        history = []
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running, seen, step = 0.0, 0, 0
            opt.zero_grad()
            class_weight = None
            if args.neutral_weight != 1.0:
                class_weight = torch.tensor([1.0, args.neutral_weight, 1.0], device=device)
            for batch in loaders["train"]:
                step += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                    logits = model(**batch).logits
                    loss = F.cross_entropy(logits, labels, weight=class_weight,
                                           label_smoothing=args.label_smoothing) / args.grad_accum
                loss.backward()
                running += loss.item() * args.grad_accum * len(labels)
                seen += len(labels)
                if step % args.grad_accum == 0 or step == len(loaders["train"]):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
            val_pred, _, val_gold = predict(model, loaders["val"], device, amp)
            val_f1 = float(f1_score(val_gold, val_pred, average="macro", zero_division=0))
            improved, stop = est.step(val_f1, epoch + 1)
            history.append({"epoch": epoch + 1, "train_loss": round(running / seen, 4),
                            "val_macro_f1": round(val_f1, 4), "best": bool(improved)})
            state = {"model": model.state_dict() if spec["full_ft"] else trainable_state(model),
                     "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "epoch": epoch + 1, "best": est.best, "best_epoch": est.best_epoch, "bad": est.bad}
            save_checkpoint(last_path, state)
            if improved:
                save_checkpoint(best_path_ckpt, {"model": state["model"], "epoch": epoch + 1,
                                                 "val_macro_f1": val_f1})
                if spec["full_ft"]:
                    (rd / "checkpoint_best.path").write_text(str(best_path_ckpt), encoding="utf-8")
            vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
            log.log(status_line(args.model, config, args.seed, epoch + 1, args.epochs,
                                val_f1, est.best, timer.fmt(), vram))
            if stop:
                log.log(f"early stop at epoch {epoch + 1} (best {est.best_epoch})")
                break

        if best_path_ckpt.exists():
            ck = load_checkpoint(best_path_ckpt, device)
            if spec["full_ft"]:
                model.load_state_dict(ck["model"])
            else:
                load_trainable_state(model, ck["model"])
        if args.phase == "sensitivity":
            val_pred, val_proba, val_gold = predict(model, loaders["val"], device, amp)
            metrics = compute_metrics(val_gold, val_pred, probs=val_proba)
            np.savez_compressed(rd / "preds.npz", val_pred=val_pred,
                                val_proba=val_proba, val_gold=val_gold)
            hyper = {"lr": lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                     "max_len": args.max_len, "epochs": args.epochs,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum}
            result = {
                "run_id": f"transformers/{args.model}/{args.dataset}/{config}/seed{args.seed}",
                "family": "transformers", "model": args.model, "phase": "sensitivity",
                "config": config, "seed": args.seed, "dataset": args.dataset,
                "split_name": args.split_name, "text_field": args.text_field,
                "hf": spec["hf"], "full_ft": spec["full_ft"], "hyper": hyper,
                "best_epoch": est.best_epoch, "best_val_macro_f1": round(float(est.best), 4),
                "val": metrics, "history": history, "elapsed_s": round(timer.elapsed(), 1),
            }
            atomic_write_json(rd / "metrics.json", result)
            atomic_write_json(rd / "history.json", history)
            atomic_write_text(rd / "per_class.txt", metrics["report"])
            write_status(rd, "done", elapsed_s=round(timer.elapsed(), 1),
                         val_macro_f1=metrics["macro_f1"])
            log.log(f"DONE sensitivity val_macro_f1={metrics['macro_f1']:.4f} elapsed={timer.fmt()}")
            return

        test_pred, test_proba, test_gold = predict(model, loaders["test"], device, amp)
        metrics = compute_metrics(test_gold, test_pred, probs=test_proba)
        np.savez_compressed(rd / "preds.npz", test_pred=test_pred, test_proba=test_proba,
                            test_gold=test_gold, test_ids=np.array([str(i) for i in ids["test"]], dtype=object))
        result = {
            "run_id": f"transformers/{args.model}/{args.dataset}/{config}/seed{args.seed}",
            "family": "transformers", "model": args.model, "phase": args.phase, "config": config,
            "seed": args.seed, "dataset": args.dataset, "split_name": args.split_name,
            "text_field": args.text_field, "hf": spec["hf"], "full_ft": spec["full_ft"],
            "hyper": frozen["hyper"] if frozen else {}, "frozen_hash": frozen.get("hash") if frozen else None,
            "trainable_params": n_trainable, "total_params": n_total,
            "rows": {s: len(y[s]) for s in SPLITS},
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
    lines = [f"# {r['model']} ({'full FT' if r['full_ft'] else 'LoRA'}) — {r['phase']}", "",
             f"seed: {r['seed']} | dataset: {r['dataset']} | split: {r['split_name']} | hf: `{r['hf']}`", "",
             f"hyper: `{r['hyper']}` | frozen hash: `{r.get('frozen_hash')}` | "
             f"trainable: {r['trainable_params']:,}/{r['total_params']:,}", "",
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
