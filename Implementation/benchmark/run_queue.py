#!/usr/bin/env python3
"""Queue runner: concurrent CPU (classical) + GPU (deep, transformers) lanes.

Every job inherits checkpoint/resume + skip-if-done from its runner, so this
script is safe to re-invoke. A per-job timeout prevents a hung download from
blocking the lane; the time guard stops optional work at --hours.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
PY = sys.executable

CLASSICAL = ["dummy", "multinomial_nb", "logistic_regression", "linear_svc",
             "random_forest", "extra_trees", "hist_gradient_boosting", "knn", "soft_voting",
             "xgboost", "lightgbm"]
# deterministic fits: one seed is enough (multiple seeds would be identical)
DETERMINISTIC = {"dummy", "multinomial_nb", "logistic_regression", "linear_svc", "knn"}
DEEP = ["textcnn", "bilstm", "cnn_bilstm"]
# slow / download-heavy models last
TRANSFORMERS = ["muril_lora", "mbert_lora", "distilmbert_lora", "xlmr_lora",
                "indicbert_lora", "muril_fullft", "telugu_bert_lora"]


def classical_jobs(seeds):
    jobs = []
    for m in CLASSICAL:
        jobs.append([PY, f"classical/{m}/run.py", "--phase", "search"])
        s_list = [42] if m in DETERMINISTIC else seeds
        for s in s_list:
            jobs.append([PY, f"classical/{m}/run.py", "--phase", "final", "--seed", str(s)])
    return jobs


def deep_jobs(seeds):
    return [[PY, f"deep/{m}/run.py", "--seed", str(s)] for m in DEEP for s in seeds]


def transformer_jobs(seeds, fullft_seeds=(42,)):
    jobs = []
    for m in TRANSFORMERS:
        s_list = fullft_seeds if m == "muril_fullft" else seeds
        for s in s_list:
            jobs.append([PY, f"hf_transformers/{m}/run.py", "--phase", "final", "--seed", str(s)])
    return jobs


def mark_failed(job) -> None:
    """Best-effort: mark a failed transformer run so the bounded-retry guard applies."""
    import datetime
    import json
    from pathlib import Path as _P
    if len(job) < 2 or "hf_transformers" not in job[1]:
        return
    parts = _P(job[1]).parts
    if len(parts) < 2:
        return
    model = parts[1]
    a = job[2:]

    def get(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    phase = get("--phase", "final")
    seed = get("--seed", "42")
    cfg = get("--config") or ("search" if phase == "search"
                              else "sensitivity" if phase == "sensitivity" else "final")
    rd = BENCH / "hf_transformers" / model / "results" / "CMTE-NITANP" / cfg / f"seed{seed}"
    rd.mkdir(parents=True, exist_ok=True)
    sp = rd / "run_status.json"
    prior = 0
    if sp.exists():
        try:
            prior = int(json.loads(sp.read_text()).get("attempts", 0))
        except Exception:
            prior = 0
    sp.write_text(json.dumps({"status": "failed", "attempts": prior + 1,
                              "error": "nonzero exit (see run.log)",
                              "ts": datetime.datetime.now().isoformat()}, indent=2), encoding="utf-8")


def run_lane(name, jobs, log_path, deadline, job_timeout, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== lane {name} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for job in jobs:
            if time.time() > deadline:
                f.write(f"[time guard] stopping lane {name}\n")
                break
            rel = " ".join(job[1:])
            f.write(f"\n>>> {time.strftime('%H:%M:%S')} {rel}\n")
            f.flush()
            print(f"[{name}] {rel}", flush=True)
            if dry_run:
                continue
            try:
                proc = subprocess.run(job, cwd=BENCH, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=job_timeout)
                f.write(f"<<< exit={proc.returncode} {rel}\n")
                if proc.returncode != 0:
                    try:
                        mark_failed(job)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                f.write(f"<<< TIMEOUT after {job_timeout}s {rel}\n")
            except Exception as exc:  # noqa: BLE001
                f.write(f"<<< ERROR {exc} {rel}\n")
            f.flush()
        f.write(f"===== lane {name} done {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=("cpu", "gpu", "all"), default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--transformer-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--job-timeout", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    logs = BENCH / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    complete_flag = logs / "queue_complete.flag"
    complete_flag.unlink(missing_ok=True)
    lanes = []
    if args.lane in ("cpu", "all"):
        lanes.append(("cpu", classical_jobs(args.seeds), logs / "lane_cpu.log"))
    if args.lane in ("gpu", "all"):
        # deep models are fast and GPU-backed: run them first, then transformers.
        gpu_jobs = deep_jobs(args.seeds) + transformer_jobs(args.transformer_seeds)
        lanes.append(("gpu", gpu_jobs, logs / "lane_gpu.log"))

    if args.dry_run:
        for name, jobs, _ in lanes:
            print(f"--- {name}: {len(jobs)} jobs")
            for j in jobs:
                print("   ", " ".join(j[1:]))
        return

    threads = [threading.Thread(target=run_lane,
                                args=(n, j, p, deadline, args.job_timeout), daemon=False)
               for n, j, p in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    complete_flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    print("queue complete", flush=True)


if __name__ == "__main__":
    main()
