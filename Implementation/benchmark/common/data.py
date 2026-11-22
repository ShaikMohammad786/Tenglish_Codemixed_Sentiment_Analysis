"""Data loading and tokenization cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import sha256_text
from .paths import LABELS, SPLITS, split_path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(dataset: str, name: str = "B") -> dict[str, list[dict]]:
    return {s: load_jsonl(split_path(dataset, s, name)) for s in SPLITS}


def to_arrays(records: list[dict], text_field: str = "text_clean"):
    texts = [r[text_field] for r in records]
    y = np.array([LABELS[r["label"]] for r in records], dtype=np.int64)
    groups = np.array([int(r.get("group_id", i)) for i, r in enumerate(records)], dtype=np.int64)
    ids = [r.get("id") for r in records]
    return texts, y, groups, ids


class IndexDataset(Dataset):
    """Wrap pre-tokenized tensors + labels; default collate stacks them."""

    def __init__(self, enc: dict, labels: np.ndarray):
        self.enc = enc
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def tokenize_cached(tok, texts: list[str], max_len: int, cache_root: Path, tag: str) -> dict:
    """Tokenize once and cache the tensors under cache/tokenized/<key>/enc.pt."""
    key = sha256_text(f"{tag}|{max_len}|{len(texts)}|{sha256_text(chr(10).join(texts))[:16]}")
    d = Path(cache_root) / "tokenized" / key
    f = d / "enc.pt"
    if f.exists():
        return torch.load(f, map_location="cpu")
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    enc = dict(enc)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "enc.pt.tmp"
    torch.save(enc, tmp)
    tmp.replace(f)
    return enc
