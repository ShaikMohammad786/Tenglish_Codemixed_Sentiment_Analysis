"""Seeding, timing, early stopping, and status-board helpers."""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def limit_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def fmt(self) -> str:
        s = int(self.elapsed())
        return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


class EarlyStopping:
    """Track best-on-val epoch and request stopping after `patience` bad epochs."""

    def __init__(self, patience: int = 2, min_epochs: int = 3, mode: str = "max"):
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.best = None
        self.best_epoch = 0
        self.bad = 0

    def step(self, score: float, epoch: int) -> tuple[bool, bool]:
        improved = self.best is None or (
            score > self.best if self.mode == "max" else score < self.best
        )
        if improved:
            self.best = score
            self.best_epoch = epoch
            self.bad = 0
        else:
            self.bad += 1
        stop = epoch >= self.min_epochs and self.bad >= self.patience
        return improved, stop


def status_line(model: str, config: str, seed: int, epoch: int, max_epochs: int,
                val_f1: float, best_f1: float, elapsed: str, vram_mb: float = 0.0) -> str:
    return (f"[{model}/{config}/seed{seed}] epoch {epoch}/{max_epochs} | "
            f"val_f1={val_f1:.4f} | best={best_f1:.4f} | elapsed={elapsed} | VRAM={vram_mb:.0f}MB")
