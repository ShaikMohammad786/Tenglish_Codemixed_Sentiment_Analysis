#!/usr/bin/env python3
"""Thin per-model entry point. Folder name == model name."""
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH))

from classical.run_classical import main  # noqa: E402

MODEL = Path(__file__).resolve().parent.name

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--model", MODEL, *sys.argv[1:]]
    main()
