#!/usr/bin/env python3
"""Generate meaning-preserving orthographic perturbations of the Split-B test set.

Variants: vowel elongation, consonant voicing, aspiration softening, combined.
Outputs: robustness/data/<dataset>/{vowel,voicing,aspiration,combined}.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import atomic_write_text  # noqa: E402
from common.paths import DATASETS, SPLITS  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def elongate(tok: str, rng: random.Random) -> str:
    return re.sub(r"([aeiou])\1*", lambda m: m.group(1) * rng.choice([1, 1, 2, 3]), tok)


def voice(tok: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        tok = tok.replace("g", "v") if "g" in tok else tok.replace("v", "g")
    if "ch" in tok and rng.random() < 0.3:
        tok = tok.replace("ch", "ts")
    return tok


def aspirate(tok: str, rng: random.Random) -> str:
    if rng.random() < 0.2:
        tok = tok.replace("dh", "d").replace("th", "t").replace("bh", "b")
    return tok


def perturb(text: str, kind: str, rng: random.Random) -> str:
    out = []
    for tok in text.split():
        t = tok
        if kind in ("vowel", "combined"):
            t = elongate(t, rng)
        if kind in ("voicing", "combined"):
            t = voice(t, rng)
        if kind in ("aspiration", "combined"):
            t = aspirate(t, rng)
        out.append(t)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (Path(__file__).resolve().parent / "data" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(DATASETS[args.dataset] / f"split{args.name}_{args.split}.jsonl")
    for kind in ("vowel", "voicing", "aspiration", "combined"):
        rng = random.Random(args.seed)
        recs = []
        for r in rows:
            rec = dict(r)
            rec["text_clean_orig"] = r["text_clean"]
            rec["text_clean"] = perturb(r["text_clean"], kind, rng)
            recs.append(rec)
        p = out_dir / f"{kind}.jsonl"
        atomic_write_text(p, "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n")
        print(f"wrote {p} ({len(recs)} rows)")


if __name__ == "__main__":
    main()
