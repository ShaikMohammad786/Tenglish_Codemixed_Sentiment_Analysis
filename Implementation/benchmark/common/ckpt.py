"""Atomic checkpoints and trainable-parameter helpers."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def trainable_state(model) -> dict:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.named_parameters()
        if v.requires_grad
    }


def load_trainable_state(model, state: dict) -> None:
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in state:
                v.copy_(state[k].to(v.device))
