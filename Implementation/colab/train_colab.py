#!/usr/bin/env python3
"""
SOTA Training Script for Code-Mixed Telugu-English Sentiment Analysis (CMTE-NITANP)
Designed for Google Colab (T4 / L4 / A100) or local GPU environments.

Features:
- Dual-Stream Orthographic Tokenization: combines raw transliteration with PLD normalization.
- Symmetric Cross-Entropy Loss (SCE) + Label Smoothing: handles annotator label noise.
- Automatic Persistent Checkpoints: resumes seamlessly from Colab disconnections.
- Supports DeBERTa-v3-large, ELECTRA-large, and base models.
- Generates test predictions, per-class metrics, and confusion matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

# --------------------------------------------------------------------------- #
# Constants & Defaults
# --------------------------------------------------------------------------- #
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
RAW_DATA_URL = "https://raw.githubusercontent.com/rayalaupendar/Code-Mixed-Tel-Eng-34k/main/34k_Code_Mixed_Tel-Eng.csv"
RAW_PLD_URL = "https://raw.githubusercontent.com/rayalaupendar/Code-Mixed-Tel-Eng-34k/main/dict_1pld.csv"

# --------------------------------------------------------------------------- #
# Seed & Environment
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data Cleaning & Normalization
# --------------------------------------------------------------------------- #
LINE_BREAK_RE = re.compile(r"[\t\n\r\f\v]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b\u200c\u200d\ufeff]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Canonical text cleaning preserving emojis and expressive repetitions."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = LINE_BREAK_RE.sub(" ", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = URL_RE.sub("<url>", text)
    text = MENTION_RE.sub("<user>", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def load_pld_dict(pld_path_or_url: str) -> Dict[str, str]:
    """Load phonetic Levenshtein normalization dictionary (noisy -> canonical)."""
    mapping = {}
    if os.path.exists(pld_path_or_url):
        with open(pld_path_or_url, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    mapping[row[0].strip().lower()] = row[1].strip().lower()
    else:
        import urllib.request
        try:
            print(f"[Data] Downloading PLD dictionary from {pld_path_or_url}...")
            with urllib.request.urlopen(pld_path_or_url) as resp:
                lines = [line.decode("utf-8-sig") for line in resp.readlines()]
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[0].strip():
                        mapping[parts[0].strip().lower()] = parts[1].strip().lower()
        except Exception as e:
            print(f"[Warning] Failed to fetch PLD dict from remote: {e}. Normalization will be identity.")
    print(f"[Data] Loaded {len(mapping)} PLD normalization pairs.")
    return mapping


def normalize_pld(text: str, pld_dict: Dict[str, str]) -> str:
    """Normalize text using PLD dictionary."""
    if not pld_dict:
        return text
    tokens = text.split()
    return " ".join(pld_dict.get(t, t) for t in tokens)


# --------------------------------------------------------------------------- #
# Dataset Loader
# --------------------------------------------------------------------------- #
def load_and_preprocess_dataset(
    data_path_or_url: str,
    pld_path_or_url: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Load 34k dataset, remove exact duplicates as in Rayala & Seshadri (2026),
    and split into train, val, and test.
    
    Total clean rows: 33,352 (matching the paper's Table 1: 26,682 train, 6,670 test).
    """
    raw_rows = []
    if os.path.exists(data_path_or_url):
        with open(data_path_or_url, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 and row and "comment" in row[0].lower():
                    continue
                if len(row) >= 2:
                    raw_rows.append((row[0], row[1].strip()))
    else:
        import urllib.request
        print(f"[Data] Downloading dataset from {data_path_or_url}...")
        with urllib.request.urlopen(data_path_or_url) as resp:
            lines = [line.decode("utf-8-sig") for line in resp.readlines()]
            reader = csv.reader(lines)
            for i, row in enumerate(reader):
                if i == 0 and row and "comment" in row[0].lower():
                    continue
                if len(row) >= 2:
                    raw_rows.append((row[0], row[1].strip()))

    label_map = {"-1": "Negative", "0": "Neutral", "1": "Positive",
                 "neg": "Negative", "ntl": "Neutral", "pos": "Positive",
                 "negative": "Negative", "neutral": "Neutral", "positive": "Positive"}

    pld_dict = load_pld_dict(pld_path_or_url)

    # Clean & deduplicate by text_clean as done in Rayala & Seshadri (2026)
    seen_texts = set()
    cleaned_records = []
    dropped_dupes = 0

    for comment, raw_lbl in raw_rows:
        std_lbl = label_map.get(raw_lbl.lower())
        if not std_lbl:
            continue
        cleaned = clean_text(comment)
        if not cleaned:
            continue
        if cleaned in seen_texts:
            dropped_dupes += 1
            continue
        seen_texts.add(cleaned)

        norm_text = normalize_pld(cleaned, pld_dict)
        cleaned_records.append({
            "text_clean": cleaned,
            "text_norm": norm_text,
            "label_str": std_lbl,
            "label": LABEL2ID[std_lbl],
        })

    print(f"[Data] Total valid deduplicated rows: {len(cleaned_records)} (removed {dropped_dupes} duplicate entries).")
    label_counts = Counter(r["label_str"] for r in cleaned_records)
    print(f"[Data] Class distribution: {dict(label_counts)}")

    # 80/20 train/test split matching paper (26,682 train, 6,670 test)
    labels = [r["label"] for r in cleaned_records]
    train_val_records, test_records = train_test_split(
        cleaned_records,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Sub-split train_val into train and internal validation
    train_val_labels = [r["label"] for r in train_val_records]
    rel_val_ratio = val_ratio / (1.0 - test_ratio)
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=rel_val_ratio,
        random_state=seed,
        stratify=train_val_labels,
    )

    print(f"[Data] Final splits: Train = {len(train_records)}, Val = {len(val_records)}, Test = {len(test_records)}")
    return train_records, val_records, test_records


# --------------------------------------------------------------------------- #
# PyTorch Dataset for Dual-View Tokenization
# --------------------------------------------------------------------------- #
class DualStreamDataset(Dataset):
    """
    Feeds both raw/clean transliteration and canonical PLD-normalized text into
    the transformer sequence classification architecture.
    """
    def __init__(self, records: List[dict], tokenizer, max_len: int = 64, mode: str = "dual"):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        t_clean = item["text_clean"]
        t_norm = item["text_norm"]

        if self.mode == "dual":
            # Pass pair (text_a, text_b) to create cross-attention between clean & norm
            encoded = self.tokenizer(
                t_clean,
                t_norm,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        elif self.mode == "norm":
            encoded = self.tokenizer(
                t_norm,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:  # "clean"
            encoded = self.tokenizer(
                t_clean,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        batch_item = {k: v.squeeze(0) for k, v in encoded.items()}
        batch_item["label"] = torch.tensor(item["label"], dtype=torch.long)
        return batch_item


# --------------------------------------------------------------------------- #
# Loss Function: Symmetric Cross-Entropy with Label Smoothing
# --------------------------------------------------------------------------- #
class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross-Entropy (SCE) combines Cross-Entropy with Reverse Cross-Entropy:
        L_SCE = alpha * L_CE + beta * L_RCE
    
    Robust against label noise (~10% boundary ambiguity in social media comments).
    """
    def __init__(self, num_classes: int = 3, alpha: float = 1.0, beta: float = 0.5, label_smoothing: float = 0.08):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        probs_clamped = torch.clamp(probs, min=1e-7, max=1.0)

        # One-hot encoded labels with label smoothing
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        if self.label_smoothing > 0.0:
            smooth_val = self.label_smoothing / self.num_classes
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth_val

        # Cross Entropy
        ce_loss = -torch.sum(one_hot * torch.log(probs_clamped), dim=-1).mean()

        # Reverse Cross Entropy: - sum( probs * log(one_hot) )
        one_hot_clamped = torch.clamp(one_hot, min=1e-4, max=1.0)
        rce_loss = -torch.sum(probs_clamped * torch.log(one_hot_clamped), dim=-1).mean()

        return self.alpha * ce_loss + self.beta * rce_loss


# --------------------------------------------------------------------------- #
# Model Checkpointing Utilities
# --------------------------------------------------------------------------- #
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.checkpoint_dir / "checkpoint_best.pt"
        self.last_ckpt_path = self.checkpoint_dir / "checkpoint_last.pt"
        self.meta_path = self.checkpoint_dir / "checkpoint_meta.json"

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        best_metric: float,
        is_best: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "step": step,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
        }
        # Atomic write
        temp_last = self.checkpoint_dir / "temp_last.pt"
        torch.save(state, temp_last)
        if temp_last.exists():
            temp_last.replace(self.last_ckpt_path)

        if is_best:
            temp_best = self.checkpoint_dir / "temp_best.pt"
            torch.save(state, temp_best)
            if temp_best.exists():
                temp_best.replace(self.best_ckpt_path)
            print(f"  [Checkpoint] Saved NEW BEST model with Val Macro-F1: {best_metric:.4f}")

        meta = {"last_epoch": epoch, "last_step": step, "best_metric": best_metric, "timestamp": time.time()}
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        load_best: bool = False,
    ) -> Tuple[int, int, float]:
        target_path = self.best_ckpt_path if load_best else self.last_ckpt_path
        if not target_path.exists():
            print(f"[Checkpoint] No checkpoint found at {target_path}. Starting fresh.")
            return 0, 0, 0.0

        print(f"[Checkpoint] Loading checkpoint from {target_path}...")
        checkpoint = torch.load(target_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if optimizer and "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        best_metric = checkpoint.get("best_metric", 0.0)
        print(f"[Checkpoint] Restored epoch {epoch}, step {step}, best_metric: {best_metric:.4f}")
        return epoch, step, best_metric


# --------------------------------------------------------------------------- #
# Evaluation Function
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, amp_dtype=torch.float16) -> Dict:
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    use_amp = device.type == "cuda"
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["label"].cpu().numpy()

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = model(**kwargs)
            logits = outputs.logits

        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(labels)

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, name in ID2LABEL.items():
        per_class_metrics[name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1_per_class[idx]),
        }

    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_mat,
        "predictions": y_pred,
        "probabilities": y_prob,
        "ground_truth": y_true,
    }


# --------------------------------------------------------------------------- #
# Training Pipeline
# --------------------------------------------------------------------------- #
def train_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[Device] Using {device} (amp: {amp_dtype})")

    # 1. Load Data
    train_recs, val_recs, test_recs = load_and_preprocess_dataset(
        args.data_path,
        args.pld_path,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 2. Tokenizer & Model
    print(f"[Model] Loading tokenizer and model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=3)
    config.classifier_dropout = args.dropout
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Model] Gradient checkpointing enabled.")

    model.to(device)

    # 3. DataLoaders
    train_ds = DualStreamDataset(train_recs, tokenizer, max_len=args.max_len, mode=args.input_mode)
    val_ds = DualStreamDataset(val_recs, tokenizer, max_len=args.max_len, mode=args.input_mode)
    test_ds = DualStreamDataset(test_recs, tokenizer, max_len=args.max_len, mode=args.input_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    # 4. Optimizer, Scheduler, Loss
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    total_training_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    if args.use_sce:
        criterion = SymmetricCrossEntropyLoss(
            num_classes=3,
            alpha=args.sce_alpha,
            beta=args.sce_beta,
            label_smoothing=args.label_smoothing,
        )
        print(f"[Loss] Using Symmetric Cross-Entropy (alpha={args.sce_alpha}, beta={args.sce_beta}, smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[Loss] Using Standard Cross-Entropy with label smoothing={args.label_smoothing}")

    # 5. Checkpoint Manager & Resume
    ckpt_mgr = CheckpointManager(args.checkpoint_dir)
    start_epoch = 0
    best_val_macro_f1 = 0.0

    if args.resume:
        start_epoch, _, best_val_macro_f1 = ckpt_mgr.load_checkpoint(model, optimizer, scheduler, load_best=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 6. Training Loop
    print("\n" + "=" * 70)
    print(f"Starting Training: {args.epochs} Epochs | Batch Size: {args.batch_size} * {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            targets = batch["label"].to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                outputs = model(**kwargs)
                logits = outputs.logits
                loss = criterion(logits, targets) / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 200 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] - Loss: {loss.item() * args.grad_accum:.4f} - LR: {cur_lr:.2e}")

        train_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time

        # Validation
        val_res = evaluate_model(model, val_loader, device, amp_dtype)
        val_acc = val_res["accuracy"] * 100.0
        val_f1 = val_res["macro_f1"] * 100.0

        is_best = val_res["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_res["macro_f1"]

        print(f"\n>> Epoch {epoch+1} Complete ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%")

        ckpt_mgr.save_checkpoint(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_macro_f1,
            is_best=is_best,
        )

    # 7. Final Test Evaluation using BEST Checkpoint
    print("\n" + "=" * 70)
    print("Evaluating Test Set using BEST Validation Checkpoint...")
    print("=" * 70)
    ckpt_mgr.load_checkpoint(model, load_best=True)

    test_res = evaluate_model(model, test_loader, device, amp_dtype)
    test_acc = test_res["accuracy"] * 100.0
    test_f1 = test_res["macro_f1"] * 100.0

    print(f"\n⭐⭐ FINAL TEST RESULTS on CMTE-NITANP ({len(test_recs)} test rows) ⭐⭐")
    print(f" Accuracy : {test_acc:.2f}% (Rayala & Seshadri Electra: 86.22%)")
    print(f" Macro-F1 : {test_f1:.2f}% (Rayala & Seshadri Electra: 88.69%)")
    print("\nPer-Class Breakdown:")
    for cls_name, m in test_res["per_class"].items():
        print(f"  {cls_name:<10}: Precision={m['precision']*100:.2f}% | Recall={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}%")

    print("\nConfusion Matrix [Neg, Neu, Pos]:")
    for row in test_res["confusion_matrix"]:
        print(" ", row)

    # Save artifacts
    results_payload = {
        "model_name": args.model_name,
        "input_mode": args.input_mode,
        "test_accuracy": test_res["accuracy"],
        "test_macro_f1": test_res["macro_f1"],
        "test_weighted_f1": test_res["weighted_f1"],
        "per_class": test_res["per_class"],
        "confusion_matrix": test_res["confusion_matrix"],
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        preds=test_res["predictions"],
        probs=test_res["probabilities"],
        labels=test_res["ground_truth"],
    )
    print(f"\n[Saved] Test metrics and prediction arrays saved to {out_dir}/")


# --------------------------------------------------------------------------- #
# CLI Parser
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train SOTA Transformer on Code-Mixed Telugu-English Sentiment")
    parser.add_argument("--model-name", type=str, default="google/electra-large-discriminator",
                        choices=["google/electra-large-discriminator", "microsoft/deberta-v3-large",
                                 "google/electra-base-discriminator", "microsoft/deberta-v3-base"])
    parser.add_argument("--data-path", type=str, default=RAW_DATA_URL)
    parser.add_argument("--pld-path", type=str, default=RAW_PLD_URL)
    parser.add_argument("--input-mode", type=str, default="dual", choices=["dual", "clean", "norm"],
                        help="dual: (clean, norm) cross-attention pair; clean: raw text; norm: PLD normalized")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-sce", action="store_true", default=True, help="Enable Symmetric Cross-Entropy")
    parser.add_argument("--sce-alpha", type=float, default=1.0)
    parser.add_argument("--sce-beta", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
