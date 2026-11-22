#!/usr/bin/env python3
"""Preprocess the code-mixed sentiment corpora - ONE dataset at a time, no pooling.

Each dataset is parsed and written to its own folder; nothing is merged into a
single file:

    datasets/raw/CMTE-NITANP/34k_Code_Mixed_Tel-Eng.csv  ─┐
    datasets/raw/CMTET/codemix_sentiment_data.txt        ─┤
    datasets/raw/DravidianCodeMix-Tamil/…                ─┴─▶ parse ▶ clean
                                                             ▶ label-driven drops
                                                             ▶ [PLD norm, Telugu only]
                                                             ▶ write per dataset

Outputs (under --out-dir, default ``datasets/processed``):
    <dataset>/labeled.csv        id, dataset, text_raw, text_clean, label
    <dataset>/normalized.csv     id, dataset, text_raw, text_clean, label, text_norm
    <dataset>/preprocess_log.md  per-dataset funnel + signal audit + length stats

Deliberate choices
------------------
* ``text_raw`` is always kept next to ``text_clean`` (needed for error analysis).
* One ``clean_text`` function is applied to every dataset.
* Emojis, punctuation, numbers and character elongations are NOT removed.
* The PLD dictionary is Telugu-specific and is applied to CMTE-NITANP only.
* No de-duplication and no splitting happen here.

Usage
-----
    python preprocessing/preprocess.py
    python preprocessing/preprocess.py --out-dir datasets/processed --drop-monolingual
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Label schemes. Telugu corpora are 3-class; Tamil (DravidianCodeMix) is native
# 5-class and is left as-is (harmonisation, if wanted, happens downstream).
# --------------------------------------------------------------------------- #
NITANP_LABELS = {"-1": "Negative", "0": "Neutral", "1": "Positive"}
CMTET_LABELS = {"POS": "Positive", "NEG": "Negative", "NTL": "Neutral"}
TAMIL_LABELS = {"Positive", "Negative", "Mixed_feelings", "unknown_state", "not-Tamil"}

DATASETS = ("CMTE-NITANP", "CMTET", "DravidianCodeMix-Tamil")

# --------------------------------------------------------------------------- #
# Cleaning regexes
# --------------------------------------------------------------------------- #
# Line breaks become spaces first; the remaining control + zero-width characters
# are then *deleted* (never merged into a neighbouring word).
LINE_BREAK_RE = re.compile(r"[\t\n\r\f\v]")
CONTROL_OR_ZERO_WIDTH_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b\u200c\u200d\ufeff]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
# Used only to *audit* that emojis survive cleaning - never used to delete.
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
)
DIGIT_RE = re.compile(r"\d")
EMOTIVE_PUNCT_RE = re.compile(r"[!?]")


# --------------------------------------------------------------------------- #
# Parsing (Phase A)
# --------------------------------------------------------------------------- #
def _read_csv_rows(path: Path):
    """Yield rows from a CSV file, tolerating a UTF-8 BOM."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        yield from csv.reader(fh)


def load_cmte_nitanp(path: Path) -> list[dict]:
    """Parse ``34k_Code_Mixed_Tel-Eng.csv`` (columns: comment, label, <empty>)."""
    records = []
    for i, row in enumerate(_read_csv_rows(path)):
        if i == 0 and row and row[0].strip().lower() == "comment":
            continue  # header
        if len(row) < 2:
            continue
        records.append(
            {
                "id": f"nitanp_{len(records)}",
                "dataset": "CMTE-NITANP",
                "text_raw": row[0],
                "label": NITANP_LABELS.get(row[1].strip()),
                "lid_tags": None,
            }
        )
    return records


def load_cmtet(path: Path) -> list[dict]:
    """Parse ``codemix_sentiment_data.txt``.

    Each record is two non-blank lines (``<TAG>: sentence`` then the per-word LID
    tags), separated by a blank-line pair. Pairing the non-blank lines keeps us
    robust to the exact number of blank lines.
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) % 2:
        raise ValueError(f"{path}: expected an even number of content lines, got {len(lines)}")

    records = []
    for i in range(0, len(lines), 2):
        head, tags = lines[i], lines[i + 1]
        tag, _, sentence = head.partition(":")
        records.append(
            {
                "id": f"cmtet_{len(records)}",
                "dataset": "CMTET",
                "text_raw": sentence.strip(),
                "label": CMTET_LABELS.get(tag.strip().upper()),
                "lid_tags": tags.split(),
            }
        )
    return records


def load_dravidian_tamil(directory: Path) -> list[dict]:
    """Parse the DravidianCodeMix Tamil sentiment splits.

    This corpus differs from the Telugu ones: three files with **two different
    separators**, and the label is the **last** field (a few rows embed the
    separator inside the text, so we ``rpartition`` rather than index a column):
        train : ``text<TAB>label``
        dev   : ``text;label;``
        test  : ``text;label;``
    No LID tags; labels use the native 5-class scheme (kept as-is).
    """
    files = [("tamil_sentiment_full_train.csv", "\t"),
             ("tamil_sentiment_full_dev.csv", ";"),
             ("tamil_sentiment_full_test.csv", ";")]
    records = []
    for fname, sep in files:
        path = directory / "DravidianCodeMix" / fname
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.rstrip()
            if not line:
                continue
            line = line.rstrip(sep)  # drop the trailing ';' in dev/test
            text, found, label = line.rpartition(sep)
            if not found:
                continue  # no separator at all -> unparseable, skip
            label = label.strip()
            records.append(
                {
                    "id": f"tamil_{len(records)}",
                    "dataset": "DravidianCodeMix-Tamil",
                    "text_raw": text,
                    "label": label if label in TAMIL_LABELS else None,  # None -> dropped in Phase D
                    "lid_tags": None,
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Cleaning (Phase B/C: minimal, deterministic, signal-preserving)
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Clean one string; identical treatment for every dataset.

    Keeps emojis, punctuation, numbers and elongations on purpose. Only removes
    encoding noise and normalises social-media artefacts.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)  # 1. canonical Unicode (Telugu/Tamil combining marks)
    text = html.unescape(text)  # 2. &amp; -> &, &#39; -> '  (before stripping so that
    #                               entity-encoded zero-width/control chars also go)
    text = LINE_BREAK_RE.sub(" ", text)  # 3. newlines/tabs -> space (don't glue words)
    text = CONTROL_OR_ZERO_WIDTH_RE.sub("", text)  #    drop control + zero-width chars
    text = URL_RE.sub("<url>", text)  # 4. URLs
    text = MENTION_RE.sub("<user>", text)  #    @mentions
    text = HASHTAG_RE.sub(r"\1", text)  # 5. keep the word, drop the '#'
    text = WHITESPACE_RE.sub(" ", text).strip()  # 6. collapse whitespace
    text = text.lower()  # 7. lowercase (Latin only; Telugu/Tamil are caseless)
    return text


def load_pld_dict(path: Path) -> dict:
    """Load the PLD normalization pairs (variant -> canonical), e.g. chaala -> chala."""
    pairs = {}
    for row in _read_csv_rows(path):
        if len(row) >= 2 and row[0].strip():
            pairs[row[0].strip()] = row[1].strip()
    return pairs


def normalize_text(text: str, mapping: dict) -> str:
    """Word-wise PLD normalization applied on top of ``text_clean``."""
    return " ".join(mapping.get(tok, tok) for tok in text.split())


def content_languages(lid_tags) -> set | None:
    """Content languages (``en``/``te``) from a record's word-level LID tags.

    Returns ``None`` when a dataset has no LID tags (row cannot be judged, kept).
    """
    if not lid_tags:
        return None
    return {t for t in lid_tags if t in {"en", "te"}}


def is_monolingual(row: dict) -> bool:
    """True when the LID tags name exactly one content language (all-English or all-Telugu)."""
    langs = content_languages(row["lid_tags"])
    return langs is not None and len(langs) == 1


# --------------------------------------------------------------------------- #
# Reporting helpers (Phase F)
# --------------------------------------------------------------------------- #
def length_stats(rows: list[dict], key: str) -> dict:
    toks = [len(r[key].split()) for r in rows]
    chars = [len(r[key]) for r in rows]
    return {
        "n": len(rows),
        "tokens": (min(toks), statistics.median(toks), max(toks)),
        "chars": (min(chars), statistics.median(chars), max(chars)),
    }


def signal_audit(rows: list[dict], key: str = "text_raw") -> Counter:
    """Count how many rows carry each signal (proof they are handled/preserved)."""
    counts = Counter(
        {k: 0 for k in ("has_url", "has_mention", "has_hashtag", "has_emoji", "has_!?_punct", "has_digit")}
    )
    for r in rows:
        t = r[key]
        if URL_RE.search(t):
            counts["has_url"] += 1
        if MENTION_RE.search(t):
            counts["has_mention"] += 1
        if HASHTAG_RE.search(t):
            counts["has_hashtag"] += 1
        if EMOJI_RE.search(t):
            counts["has_emoji"] += 1
        if EMOTIVE_PUNCT_RE.search(t):
            counts["has_!?_punct"] += 1
        if DIGIT_RE.search(t):
            counts["has_digit"] += 1
    return counts


def format_funnel(funnel: list[tuple[str, int]]) -> str:
    lines = ["| step | rows | delta |", "|---|---:|---:|"]
    prev = None
    for label, count in funnel:
        delta = "" if prev is None else f"{count - prev:+d}"
        lines.append(f"| {label} | {count:,} | {delta} |")
        prev = count
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, default=root / "datasets" / "raw")
    ap.add_argument("--out-dir", type=Path, default=root / "datasets" / "processed")
    ap.add_argument(
        "--drop-monolingual",
        action="store_true",
        help="optional: drop rows with a single content language (default: keep them)",
    )
    args = ap.parse_args()

    # -- Phase A: load & parse (everything stays in ONE list, split by `dataset`) -- #
    nitanp = load_cmte_nitanp(args.raw_dir / "CMTE-NITANP" / "34k_Code_Mixed_Tel-Eng.csv")
    cmtet = load_cmtet(args.raw_dir / "CMTET" / "codemix_sentiment_data.txt")
    tamil = load_dravidian_tamil(args.raw_dir / "DravidianCodeMix-Tamil")
    rows = nitanp + cmtet + tamil

    # Per-dataset snapshots after each stage -> per-dataset funnels, no mixing.
    stages: list[tuple[str, list[dict]]] = [("parsed", rows)]

    # -- Phase B/C: clean every row (row count unchanged) ----------------- #
    for r in rows:
        r["text_clean"] = clean_text(r["text_raw"])
    stages.append(("after cleaning", rows))

    # -- Phase D: label-driven drops (each logged per dataset) ------------ #
    rows = [r for r in rows if r["text_clean"]]
    stages.append(("drop empty text_clean", rows))

    rows = [r for r in rows if r["label"]]
    stages.append(("drop invalid/missing label", rows))

    if args.drop_monolingual:
        rows = [r for r in rows if not is_monolingual(r)]
        stages.append(("drop monolingual (--drop-monolingual)", rows))

    # -- Phase E: PLD normalization - Telugu rows only -------------------- #
    pld = load_pld_dict(args.raw_dir / "CMTE-NITANP" / "dict_1pld.csv")
    changed_by_ds: Counter = Counter()
    for r in rows:
        if r["dataset"] == "CMTE-NITANP":
            r["text_norm"] = normalize_text(r["text_clean"], pld)
        else:
            r["text_norm"] = r["text_clean"]  # PLD dict is Telugu-specific; leave others untouched
        if r["text_norm"] != r["text_clean"]:
            changed_by_ds[r["dataset"]] += 1

    # -- Phase F: write each dataset to its own folder -------------------- #
    funnel_by_ds = {
        ds: [(step, sum(1 for r in snap if r["dataset"] == ds)) for step, snap in stages]
        for ds in DATASETS
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'dataset':24} {'rows':>8}  output")
    for ds in DATASETS:
        sub = args.out_dir / ds
        sub.mkdir(parents=True, exist_ok=True)
        ds_rows = [r for r in rows if r["dataset"] == ds]

        _write_csv(sub / "labeled.csv",
                   ["id", "dataset", "text_raw", "text_clean", "label"], ds_rows)
        _write_csv(sub / "normalized.csv",
                   ["id", "dataset", "text_raw", "text_clean", "label", "text_norm"], ds_rows)

        log = _build_dataset_log(
            ds, ds_rows, funnel_by_ds[ds], len(pld), changed_by_ds.get(ds, 0),
            sum(1 for r in ds_rows if is_monolingual(r)), args.drop_monolingual,
        )
        (sub / "preprocess_log.md").write_text(log, encoding="utf-8")
        print(f"{ds:24} {len(ds_rows):>8,}  {sub}")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])


def _build_dataset_log(ds, rows, funnel, dict_size, changed, monolingual_count,
                       dropped_monolingual) -> str:
    audit = signal_audit(rows, "text_raw")
    stats = length_stats(rows, "text_clean")
    label_counts = Counter(r["label"] for r in rows)

    out = [f"# Preprocessing log - {ds}", "", format_funnel(funnel), ""]

    out += ["## Signal audit (rows containing, before cleaning)", ""]
    out += ["| signal | rows |", "|---|---:|"]
    out += [f"| {k} | {v:,} |" for k, v in sorted(audit.items())]
    out += ["", "Emojis, punctuation, numbers and elongations are kept in `text_clean`.", ""]

    out += ["## Length stats (text_clean: min / median / max)", "",
            "| rows | tokens | chars |", "|---:|---|---|",
            f"| {stats['n']:,} | {stats['tokens'][0]} / {stats['tokens'][1]:g} / {stats['tokens'][2]} "
            f"| {stats['chars'][0]} / {stats['chars'][1]:g} / {stats['chars'][2]} |", ""]

    out += ["## Labels", "", "| label | rows |", "|---|---:|"]
    out += [f"| {k} | {v:,} |" for k, v in sorted(label_counts.items())]
    out += [""]

    out += ["## Normalization (Phase E)", "",
            f"- PLD dictionary entries: {dict_size:,}",
            f"- PLD applied to this dataset: "
            f"{'yes' if ds == 'CMTE-NITANP' else 'no (dictionary is Telugu-specific)'}",
            f"- Rows changed by normalization: {changed:,} / {len(rows):,}",
            f"- Monolingual rows detected: {monolingual_count:,}",
            f"- Monolingual rows dropped: {'yes' if dropped_monolingual else 'no (kept by default)'}",
            "",
            "De-duplication and splitting are intentionally NOT done here.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
