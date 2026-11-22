#!/usr/bin/env python3
"""Exact-Jaccard audit of the MinHash-LSH split.

For each split (A and B):
  1. Build MinHash-LSH over the TRAIN rows (same shingles as dedup_split.py).
  2. For every TEST row, query LSH candidates in train.
  3. Compute EXACT Jaccard (word 3-grams) for each candidate pair.
  4. Report: max exact Jaccard, pairs >= 0.8 (leakage), borderline 0.7-0.8,
     and LSH candidates whose exact Jaccard is below 0.8 (approximation false positives).

Output: Implementation/benchmark/audit/results/CMTE-NITANP/split_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import atomic_write_json, atomic_write_text, utcnow  # noqa: E402
from common.paths import DATASETS, REPO, SPLITS  # noqa: E402

from datasketch import MinHash, MinHashLSH  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def shingles(text: str) -> set[str]:
    toks = text.split()
    out = {" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)}
    if not out and toks:
        out = {" ".join(toks)}
    return out


def exact_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def minhash_of(sh: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in sh:
        mh.update(s.encode("utf-8"))
    return mh


def audit_split(train_records, test_records, threshold: float, num_perm: int) -> dict:
    train_sh = [shingles(r["text_clean"]) for r in train_records]
    test_sh = [shingles(r["text_clean"]) for r in test_records]
    train_mh = [minhash_of(s, num_perm) for s in train_sh]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, mh in enumerate(train_mh):
        lsh.insert(i, mh)

    max_j = 0.0
    leaking_pairs = 0
    borderline_pairs = 0
    candidates = 0
    approx_false = 0
    leaking_ids = []

    for ti, (rec, sh) in enumerate(zip(test_records, test_sh)):
        mh = minhash_of(sh, num_perm)
        cands = lsh.query(mh)
        candidates += len(cands)
        for ci in cands:
            j = exact_jaccard(sh, train_sh[ci])
            max_j = max(max_j, j)
            if j >= threshold:
                leaking_pairs += 1
                leaking_ids.append({"test_id": rec.get("id"), "train_id": train_records[ci].get("id"), "jaccard": round(j, 4)})
            elif j >= 0.7:
                borderline_pairs += 1
            else:
                approx_false += 1

    return {
        "n_train": len(train_records),
        "n_test": len(test_records),
        "threshold": threshold,
        "num_perm": num_perm,
        "lsh_candidate_pairs": candidates,
        "exact_pairs_ge_threshold": leaking_pairs,
        "exact_pairs_0.7_to_threshold": borderline_pairs,
        "lsh_false_positives_below_threshold": approx_false,
        "max_exact_jaccard_train_test": round(max_j, 6),
        "leaking_pairs_sample": leaking_ids[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--split-name", default="B")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    splits_dir = DATASETS[args.dataset]
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "results" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"dataset": args.dataset, "audited_at": utcnow(), "splits": {}}
    for name in ("A", "B"):
        recs = {s: load_jsonl(splits_dir / f"split{name}_{s}.jsonl") for s in SPLITS}
        print(f"[audit] split {name}: train={len(recs['train']):,} test={len(recs['test']):,}")
        results["splits"][name] = audit_split(
            recs["train"], recs["test"], args.threshold, args.num_perm
        )
        r = results["splits"][name]
        print(f"  max exact Jaccard(train,test) = {r['max_exact_jaccard_train_test']}")
        print(f"  pairs >= {args.threshold}: {r['exact_pairs_ge_threshold']} | "
              f"borderline 0.7-{args.threshold}: {r['exact_pairs_0.7_to_threshold']} | "
              f"LSH candidates: {r['lsh_candidate_pairs']}")

    atomic_write_json(out_dir / "split_audit.json", results)

    md = ["# Split audit — exact Jaccard verification", "",
          f"Dataset: **{args.dataset}** | threshold={args.threshold} | num_perm={args.num_perm} | {results['audited_at']}", "",
          "> `exact_pairs_ge_threshold` is the number of (test, train) pairs with exact word-3gram",
          "> Jaccard >= threshold. Non-zero means the MinHash-LSH grouping missed real leakage.", "",
          "| split | n_train | n_test | LSH candidates | exact >= θ | borderline 0.7–θ | max exact Jaccard |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results["splits"].items():
        md.append(f"| {name} | {r['n_train']:,} | {r['n_test']:,} | {r['lsh_candidate_pairs']:,} | "
                  f"{r['exact_pairs_ge_threshold']:,} | {r['exact_pairs_0.7_to_threshold']:,} | "
                  f"{r['max_exact_jaccard_train_test']} |")
    md += ["", "**Interpretation.** Split B is considered leakage-audited only if "
           "`exact_pairs_ge_threshold == 0`. Otherwise rebuild the split with an exact-Jaccard pass.", ""]
    atomic_write_text(out_dir / "split_audit.md", "\n".join(md))
    print(f"\nWrote:\n  {(out_dir / 'split_audit.md').resolve()}\n  {(out_dir / 'split_audit.json').resolve()}")


if __name__ == "__main__":
    main()
