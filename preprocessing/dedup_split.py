#!/usr/bin/env python3
"""De-duplicate + cluster, then build the two splits (Split A and Split B).

This script does DATA PREPARATION ONLY - no model is trained here.  The one-line
logic of the two splits:

    group_id (near-duplicate clusters) is the grouping key
      -> Split A ignores it  (duplicates may span train/val/test -> leakage present)
      -> Split B respects it (a whole cluster lands in one split -> leakage removed)

Phases
------
1. Exact duplicates (identical ``text_clean``) + MinHash/LSH near-duplicate
   clusters, union-found into one ``group_id`` per row (singletons get their own).
2. Split A  - stratified random split, IGNORES ``group_id``.
3. Split B  - ``StratifiedGroupKFold`` so a whole cluster lands in one split,
   with the split sizes matched to Split A for an apples-to-apples comparison.
4. Save JSONL + a report of the clustering and split statistics.

Model evaluation (TF-IDF+SVM, CNN+BiLSTM) and the leakage gap are measured
separately, in ``Implementation/baseline/``.

Input  : datasets/processed/labeled.csv
Outputs: datasets/splits/{clustered.csv, splitA_*.jsonl, splitB_*.jsonl, split_report.md}

By default this runs on **CMTE-NITANP** alone and keeps **CMTET** 100% held out
(the cross-dataset test); pass ``--dataset all`` to use the full corpus.  Each
JSONL record carries id, dataset, text_raw, text_clean, label, group_id, split
(plus text_norm when available) so the baselines need no manual join.

Requires: datasketch, scikit-learn, numpy  (see preprocessing/requirements.txt).
Run with the project venv:  .venv/bin/python preprocessing/dedup_split.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Phase 1: near-duplicate clustering
# --------------------------------------------------------------------------- #
def make_shingles(text: str, mode: str) -> list[str]:
    """Turn one text into shingles: word 3-grams by default, char 5-grams if very short."""
    if mode == "char5":
        s = re.sub(r"\s+", " ", text).strip()
        return [s[i : i + 5] for i in range(len(s) - 4)] or ([s] if s else [])
    tokens = text.split()
    return [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)] or (
        [" ".join(tokens)] if tokens else []
    )


def build_minhash(text: str, num_perm: int, mode: str) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in make_shingles(text, mode):
        mh.update(shingle.encode("utf-8"))
    return mh


def cluster(minhashes: list[MinHash], threshold: float, num_perm: int) -> np.ndarray:
    """Union-find the LSH matches and return a ``group_id`` per row."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, mh in enumerate(minhashes):
        lsh.insert(i, mh)
    parent = list(range(len(minhashes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, mh in enumerate(minhashes):
        for j in lsh.query(mh):
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[rb] = ra

    # Relabel roots to dense 0..k-1 ids (a root of a singleton is its own group).
    root_to_group: dict[int, int] = {}
    group_ids = np.empty(len(minhashes), dtype=np.int64)
    for i in range(len(minhashes)):
        root = find(i)
        group_ids[i] = root_to_group.setdefault(root, len(root_to_group))
    return group_ids


def cluster_stats(group_ids: np.ndarray) -> dict:
    sizes = Counter(group_ids.tolist())  # group_id -> number of rows in that group
    dist = Counter(sizes.values())  # group size -> number of groups
    multi = sum(size for size in sizes.values() if size > 1)
    return {
        "clusters": len(sizes),
        "size_dist": dict(sorted(dist.items())),
        "rows_in_multi": multi,
        "pct_in_multi": 100.0 * multi / len(group_ids),
    }


def cluster_label_conflicts(group_ids: np.ndarray, labels: list[str]) -> dict:
    """Quantify annotator noise: multi-row clusters whose members disagree on the label."""
    by_group: dict[int, Counter] = {}
    for g, lab in zip(group_ids.tolist(), labels):
        by_group.setdefault(g, Counter())[lab] += 1
    multi = {g: c for g, c in by_group.items() if sum(c.values()) > 1}
    conflicting = {g: c for g, c in multi.items() if len(c) > 1}

    entropies = []
    for c in multi.values():
        total = sum(c.values())
        entropies.append(-sum((k / total) * math.log2(k / total) for k in c.values()))
    return {
        "multi_clusters": len(multi),
        "conflicting_clusters": len(conflicting),
        "conflicting_rows": sum(sum(c.values()) for c in conflicting.values()),
        "pct_conflicting": 100.0 * len(conflicting) / len(multi) if multi else 0.0,
        "mean_entropy": statistics.fmean(entropies) if entropies else 0.0,
    }


# --------------------------------------------------------------------------- #
# Phase 2: Split A (random / control)
# --------------------------------------------------------------------------- #
def split_a(labels: list[str], ratios: tuple[float, float, float], seed: int) -> np.ndarray:
    """Stratified random split that ignores group_id."""
    idx = np.arange(len(labels))
    y = np.asarray(labels)
    tr_va, te = train_test_split(idx, test_size=ratios[2], random_state=seed, stratify=y)
    val_frac = ratios[1] / (ratios[0] + ratios[1])
    tr, va = train_test_split(tr_va, test_size=val_frac, random_state=seed, stratify=y[tr_va])

    assign = np.empty(len(labels), dtype=object)
    assign[tr], assign[va], assign[te] = "train", "val", "test"
    return assign


# --------------------------------------------------------------------------- #
# Phase 3: Split B (leakage-controlled / treatment)
# --------------------------------------------------------------------------- #
def split_b(labels: list[str], group_ids: np.ndarray, targets: dict[str, int], seed: int) -> np.ndarray:
    """StratifiedGroupKFold split, with split sizes matched to Split A's targets."""
    y = np.asarray(labels)
    groups = group_ids
    gsize = Counter(groups.tolist())

    # 10 label-balanced, group-respecting folds.
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    group_fold: dict[int, int] = {}
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(labels)), y, groups=groups)):
        for g in np.unique(groups[test_idx]):
            group_fold[int(g)] = fold
    fold_of_group = {int(g): group_fold[int(g)] for g in gsize}
    fold_size = Counter()
    for g, f in fold_of_group.items():
        fold_size[f] += gsize[g]

    # Pick the (test fold, val fold) pair whose sizes best match Split A.
    folds = sorted(fold_size)
    best = None
    for tf in folds:
        for vf in folds:
            if tf == vf:
                continue
            score = abs(fold_size[tf] - targets["test"]) + abs(fold_size[vf] - targets["val"])
            if best is None or score < best[0]:
                best = (score, tf, vf)
    _, test_fold, val_fold = best

    group_split = {
        g: ("test" if f == test_fold else "val" if f == val_fold else "train")
        for g, f in fold_of_group.items()
    }

    # Nudge whole clusters between train<->test and train<->val to hit exact sizes.
    _adjust(group_split, gsize, "train", "test", targets["test"])
    _adjust(group_split, gsize, "train", "val", targets["val"])

    return np.array([group_split[int(g)] for g in groups], dtype=object)


def _adjust(split_of: dict[int, str], gsize: Counter, src: str, dst: str, target: int) -> None:
    """Move whole clusters between two splits, best-fit, until ``dst`` hits ``target``."""
    current = sum(gsize[g] for g, s in split_of.items() if s == dst)
    for _ in range(100_000):
        if current == target:
            break
        need = target - current
        pool = [g for g, s in split_of.items() if s == (src if need > 0 else dst)]
        if not pool:
            break
        want = abs(need)
        fits = [g for g in pool if gsize[g] <= want]
        pick = max(fits, key=lambda g: gsize[g]) if fits else min(pool, key=lambda g: gsize[g])
        split_of[pick] = dst if need > 0 else src
        current += gsize[pick] if need > 0 else -gsize[pick]


# --------------------------------------------------------------------------- #
# Split sanity checks
# --------------------------------------------------------------------------- #
def overlap_count(assign: np.ndarray, group_ids: np.ndarray) -> tuple[int, float]:
    """Test rows whose cluster also appears in train (i.e. a twin leaked into train)."""
    train_groups = set(group_ids[assign == "train"].tolist())
    test_groups = group_ids[assign == "test"]
    leaked = sum(1 for g in test_groups.tolist() if g in train_groups)
    return leaked, 100.0 * leaked / max(len(test_groups), 1)


def symmetric_diff(assign_a: np.ndarray, assign_b: np.ndarray, split: str) -> int:
    """Rows whose membership differs between Split A and Split B (0 = identical split)."""
    a = set(np.where(assign_a == split)[0].tolist())
    b = set(np.where(assign_b == split)[0].tolist())
    return len(a ^ b)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labeled", type=Path,
                    default=root / "datasets/processed/CMTE-NITANP/labeled.csv",
                    help="one dataset's labeled.csv (datasets are kept separate)")
    ap.add_argument("--normalized", type=Path, default=None,
                    help="text_norm source; defaults to normalized.csv next to --labeled")
    ap.add_argument("--out-dir", type=Path, default=root / "datasets/splits/CMTE-NITANP")
    ap.add_argument("--shingle", choices=("word3", "char5"), default="word3")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test ratios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (smoke test)")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    normalized_path = args.normalized or (args.labeled.parent / "normalized.csv")

    with args.labeled.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    # text_norm is optional; merge it in by id so JSONL records can carry it too.
    if normalized_path.exists():
        norm = {r["id"]: r.get("text_norm")
                for r in csv.DictReader(normalized_path.open(encoding="utf-8"))}
        for r in rows:
            r["text_norm"] = norm.get(r["id"])

    dataset_name = rows[0]["dataset"] if rows else "(empty)"
    texts = [r["text_clean"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"Splitting '{dataset_name}': {len(rows):,} rows | shingle={args.shingle} "
          f"threshold={args.threshold} seed={args.seed}")

    # -- Phase 1: exact dups + near-dup clusters -------------------------- #
    text_to_exact: dict[str, int] = {}
    exact_ids = np.empty(len(rows), dtype=np.int64)
    for i, t in enumerate(texts):
        exact_ids[i] = text_to_exact.setdefault(t, len(text_to_exact))
    exact_counts = Counter(texts)
    exact_dup_groups = sum(1 for c in exact_counts.values() if c > 1)
    exact_dup_rows = sum(c - 1 for c in exact_counts.values())

    minhashes = [build_minhash(t, args.num_perm, args.shingle) for t in texts]
    group_ids = cluster(minhashes, args.threshold, args.num_perm)
    stats = cluster_stats(group_ids)
    conflicts = cluster_label_conflicts(group_ids, labels)
    print(f"Phase 1: distinct texts={len(text_to_exact):,} exact-dup groups={exact_dup_groups:,} "
          f"exact-dup rows={exact_dup_rows:,} clusters={stats['clusters']:,} "
          f"rows in cluster>1={stats['pct_in_multi']:.2f}%")
    print(f"         conflicting-label clusters={conflicts['conflicting_clusters']:,} "
          f"({conflicts['pct_conflicting']:.2f}% of multi-row clusters)")

    # -- Phases 2/3: Split A (control) then Split B (treatment) ----------- #
    assign_a = split_a(labels, ratios, args.seed)
    targets = {s: int(np.sum(assign_a == s)) for s in SPLITS}

    def run_split_b(gids: np.ndarray):
        """Build Split B for a given grouping; also return the A/B test leakage."""
        ab = split_b(labels, gids, targets, args.seed)
        return ab, overlap_count(assign_a, gids), overlap_count(ab, gids)

    assign_b, leak_a, leak_b = run_split_b(group_ids)
    print(f"Leakage: A test-rows-with-twin-in-train={leak_a[0]:,} ({leak_a[1]:.2f}%) | "
          f"B={leak_b[0]:,} ({leak_b[1]:.2f}%)")

    # How different are A and B really? (0 = identical membership)
    symdiff = {s: symmetric_diff(assign_a, assign_b, s) for s in SPLITS}
    print(f"|A ^ B| rows: train={symdiff['train']:,} val={symdiff['val']:,} test={symdiff['test']:,}")

    # Threshold sensitivity of the clustering/leakage (no model involved).
    threshold_rows = [(args.threshold, stats, leak_a, leak_b)]
    for thr in (0.7, 0.9):
        g = cluster(minhashes, thr, args.num_perm)
        _, la, lb = run_split_b(g)
        threshold_rows.append((thr, cluster_stats(g), la, lb))
    threshold_rows.sort(key=lambda r: r[0])

    # -- Phase 4: save ---------------------------------------------------- #
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "clustered.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dataset", "text_clean", "label", "exact_id", "group_id"])
        for i, r in enumerate(rows):
            w.writerow([r["id"], r["dataset"], r["text_clean"], r["label"], int(exact_ids[i]), int(group_ids[i])])

    for name, assign in (("A", assign_a), ("B", assign_b)):
        for s in SPLITS:
            path = args.out_dir / f"split{name}_{s}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for i, r in enumerate(rows):
                    if assign[i] != s:
                        continue
                    rec = {
                        "id": r["id"],
                        "dataset": r["dataset"],
                        "text_raw": r["text_raw"],
                        "text_clean": r["text_clean"],
                        "label": r["label"],
                        "group_id": int(group_ids[i]),
                        "split": s,
                    }
                    if r.get("text_norm") is not None:
                        rec["text_norm"] = r["text_norm"]
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = build_report(len(rows), args, dataset_name, exact_dup_groups, exact_dup_rows,
                          len(text_to_exact), stats, conflicts, assign_a, assign_b,
                          leak_a, leak_b, symdiff, labels, threshold_rows)
    (args.out_dir / "split_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Wrote:")
    for f in sorted(args.out_dir.glob("*")):
        print(f"  {f.resolve()}")


def build_report(n, args, dataset_name, exact_dup_groups, exact_dup_rows, distinct,
                 stats, conflicts, assign_a, assign_b, leak_a, leak_b, symdiff,
                 labels, threshold_rows) -> str:
    out = ["# De-duplication & split report", "",
           f"Corpus: **{dataset_name}**, {n:,} rows | shingle={args.shingle} | "
           f"threshold={args.threshold} | num_perm={args.num_perm} | seed={args.seed} | "
           f"ratios={args.ratios}", ""]

    out += ["## Phase 1 - clusters", "",
            "| metric | value |", "|---|---:|",
            f"| distinct `text_clean` | {distinct:,} |",
            f"| exact-duplicate groups | {exact_dup_groups:,} |",
            f"| exact-duplicate rows (beyond first) | {exact_dup_rows:,} |",
            f"| near-dup clusters (group_id) | {stats['clusters']:,} |",
            f"| rows in a cluster of size >1 | {stats['rows_in_multi']:,} ({stats['pct_in_multi']:.2f}%) |",
            f"| multi-row clusters with conflicting labels | "
            f"{conflicts['conflicting_clusters']:,} ({conflicts['pct_conflicting']:.2f}% of multi-row) |",
            f"| rows inside conflicting-label clusters | {conflicts['conflicting_rows']:,} |",
            f"| mean label entropy (multi-row clusters) | {conflicts['mean_entropy']:.3f} bits |",
            "", "Cluster-size distribution (size: clusters): " +
            ", ".join(f"{k}:{v:,}" for k, v in stats["size_dist"].items()), ""]

    out += ["## Phases 2/3 - splits", "",
            "| split | rows | Negative | Neutral | Positive | test rows w/ twin in train |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, assign, leak in (("A (random)", assign_a, leak_a), ("B (grouped)", assign_b, leak_b)):
        dist = {s: Counter(np.asarray(labels)[assign == s].tolist()) for s in SPLITS}
        for s in SPLITS:
            d = dist[s]
            twin = f"{leak[0]:,} ({leak[1]:.2f}%)" if s == "test" else "-"
            out.append(f"| {name} / {s} | {int((assign==s).sum()):,} | {d.get('Negative',0):,} | "
                       f"{d.get('Neutral',0):,} | {d.get('Positive',0):,} | {twin} |")
    out += [""]

    out += ["## Is Split B a genuine regroup or just Split A with twins moved?", "",
            "Symmetric difference of row membership between A and B. `|A ^ B|` counts rows that",
            "leave *and* enter a split, so it is even and reaches at most 2x the split size; the",
            "middle column is the number of rows whose membership actually changed.",
            "A small train value (~hundreds) would mean B is just A with the duplicate rows",
            "relocated; a large value means B is an independent regroup.",
            "",
            "| split | \\|A ^ B\\| rows | rows changed (\\|A^B\\|/2) | share of split |",
            "|---|---:|---:|---:|"]
    for s in SPLITS:
        size = int((assign_a == s).sum())
        changed = symdiff[s] // 2
        out.append(f"| {s} | {symdiff[s]:,} | {changed:,} | {100 * changed / max(size, 1):.2f}% |")
    out += [""]

    out += [f"## Threshold sensitivity (shingle={args.shingle})", "",
            "| threshold | clusters | rows in cluster>1 | A test leak | B test leak |",
            "|---|---:|---:|---:|---:|"]
    for thr, st, la, lb in threshold_rows:
        out.append(f"| {thr:.1f} | {st['clusters']:,} | {st['rows_in_multi']:,} "
                   f"({st['pct_in_multi']:.2f}%) | {la[1]:.2f}% | {lb[1]:.2f}% |")
    out += ["", "Model evaluation (TF-IDF+SVM, CNN+BiLSTM) lives in `Implementation/baseline/`.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
