"""Atomic IO, hashing, run status, and tee/event logging."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


class RunLogger:
    """Tee stdout/stderr into run.log and append structured events to events.jsonl."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run.log"
        self.events_path = self.run_dir / "events.jsonl"
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_out, self._fh)
        sys.stderr = _Tee(self._old_err, self._fh)

    def log(self, msg: str) -> None:
        print(f"[{utcnow()}] {msg}", flush=True)

    def event(self, kind: str, **fields) -> None:
        rec = {"ts": utcnow(), "event": kind, **fields}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        sys.stdout, sys.stderr = self._old_out, self._old_err
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_status(run_dir: Path, status: str, **fields) -> None:
    atomic_write_json(Path(run_dir) / "run_status.json", {"status": status, "ts": utcnow(), **fields})


def read_status(run_dir: Path):
    p = Path(run_dir) / "run_status.json"
    return load_json(p) if p.exists() else None
