"""Shared paths and constants for the benchmark suite."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parents[1]

CACHE = BENCH / "cache"
RESULTS = BENCH / "results"
REPORTS = BENCH / "reports"
LOGS = BENCH / "logs"
CONFIGS = BENCH / "configs"

DATASETS = {
    "CMTE-NITANP": REPO / "datasets/splits/CMTE-NITANP",
    "CMTET": REPO / "datasets/splits/CMTET",
    "CMTE-NITANP-paper2": REPO / "datasets/splits/CMTE-NITANP-paper2",
}
SPLITS = ("train", "val", "test")
LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def split_path(dataset: str, split: str, name: str = "B") -> Path:
    return DATASETS[dataset] / f"split{name}_{split}.jsonl"


def model_dir(family: str, model: str) -> Path:
    sub = {"classical": "classical", "deep": "deep", "transformers": "hf_transformers"}[family]
    return BENCH / sub / model


def run_dir(family: str, model: str, dataset: str, config: str, seed: int) -> Path:
    return model_dir(family, model) / "results" / dataset / config / f"seed{seed}"
