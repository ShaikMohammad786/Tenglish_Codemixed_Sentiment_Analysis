#!/usr/bin/env python3
"""Unattended watchdog for the benchmark run.

Every --interval seconds it:
  - checks queue/finalizer PIDs and logs progress;
  - restarts the queue (resume-safe) if it died before all final runs are done;
  - kills a prematurely-finalizing finalizer and restarts both;
  - relaunches the finalizer if only it died;
  - stops when all expected runs + artifacts exist, or at the deadline.

PID files: logs/queue.pid, logs/finalizer.pid, logs/monitor.pid
Status:    logs/monitor.log, logs/monitor_status.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LOGS = BENCH / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
PY = sys.executable

# final runs expected: classical 41 (5 det with legacy 5 seeds + knn 1 + 4*5 stochastic);
# deep 15; transformers 6*3+1 = 19
EXPECTED_FINAL = 41 + 15 + 19 + 10  # + xgboost(5) + lightgbm(5)
NRESTART = 0
MAX_RESTARTS = 10


def log(msg: str) -> None:
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "monitor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(name: str) -> int:
    p = LOGS / name
    try:
        return int(p.read_text().strip())
    except Exception:
        return 0


def write_pid(name: str, pid: int) -> None:
    (LOGS / name).write_text(str(pid), encoding="utf-8")


def kill_pid(pid: int) -> None:
    if pid and alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def launch_queue() -> int:
    f = open(LOGS / "queue_nohup.log", "a", encoding="utf-8")
    p = subprocess.Popen(
        [PY, str(BENCH / "run_queue.py"), "--lane", "all", "--hours", "13", "--job-timeout", "7200"],
        cwd=BENCH, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    write_pid("queue.pid", p.pid)
    threading.Thread(target=p.wait, daemon=True).start()  # reap to avoid zombies
    log(f"launched queue pid={p.pid}")
    return p.pid


def launch_finalizer(qpid: int) -> int:
    f = open(LOGS / "finalize_nohup.log", "a", encoding="utf-8")
    cmd = ["bash", str(BENCH / "finalize_after_queue.sh")]
    if qpid:
        cmd.append(str(qpid))
    p = subprocess.Popen(cmd, cwd=BENCH, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write_pid("finalizer.pid", p.pid)
    threading.Thread(target=p.wait, daemon=True).start()  # reap to avoid zombies
    log(f"launched finalizer pid={p.pid} waiting_for={qpid or 'none'}")
    return p.pid


def count_final_runs() -> int:
    import json as _json
    n = 0
    for p in BENCH.rglob("results/**/metrics.json"):
        if "smoke" in str(p):
            continue
        try:
            m = _json.loads(p.read_text())
        except Exception:
            continue
        ph = m.get("phase")
        if ph is None and m.get("family") == "deep":
            ph = "final"
        if ph == "final":
            n += 1
    return n


def failed_runs() -> list[str]:
    out = []
    for p in BENCH.rglob("results/**/run_status.json"):
        if "smoke" in str(p):
            continue
        try:
            if json.loads(p.read_text()).get("status") == "failed":
                out.append(str(p.parent.relative_to(BENCH)))
        except Exception:
            pass
    return out


def scan_errors() -> list[str]:
    hits = []
    for name in ("lane_cpu.log", "lane_gpu.log"):
        p = LOGS / name
        if not p.exists():
            continue
        try:
            tail = p.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
        except Exception:
            continue
        # only inspect the most recent job section (stale tracebacks from earlier fixed runs are noise)
        start = 0
        for i, line in enumerate(tail):
            if line.startswith(">>>") or line.startswith("===== lane"):
                start = i
        tail = tail[start:]
        for line in tail:
            if any(k in line for k in ("Traceback", "CUDA out of memory", "TIMEOUT", "ERROR")):
                hits.append(f"{name}: {line[:160]}")
    return hits[-5:]


def artifacts_done() -> bool:
    return (
        (BENCH / "reports" / "REPORT.md").exists()
        and (BENCH / "robustness" / "results" / "CMTE-NITANP" / "robustness.json").exists()
        and (BENCH / "cross_corpus" / "results" / "CMTET" / "cross_corpus.json").exists()
    )


def main() -> None:
    global NRESTART, MAX_RESTARTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--deadline", default="", help="optional ISO deadline; empty = run until complete")
    ap.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited")
    args = ap.parse_args()
    MAX_RESTARTS = args.max_restarts
    deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None
    tzinfo = deadline.tzinfo if deadline else dt.datetime.now().astimezone().tzinfo
    write_pid("monitor.pid", os.getpid())
    log(f"watchdog start pid={os.getpid()} interval={args.interval}s "
        f"deadline={deadline.isoformat() if deadline else 'none'} expected_final={EXPECTED_FINAL}")

    while True:
        now = dt.datetime.now(tzinfo)
        if deadline and now >= deadline:
            log("DEADLINE reached; stopping watchdog (run may be incomplete)")
            break

        qpid = read_pid("queue.pid")
        fpid = read_pid("finalizer.pid")
        qa, fa = alive(qpid), alive(fpid)
        n = count_final_runs()
        errs = scan_errors()
        failed = failed_runs()
        status = {"ts": now.isoformat(), "queue_pid": qpid, "queue_alive": qa,
                  "finalizer_pid": fpid, "finalizer_alive": fa,
                  "final_runs": n, "expected_final": EXPECTED_FINAL,
                  "artifacts_done": artifacts_done(), "failed_runs": failed,
                  "recent_errors": errs, "restarts": NRESTART}
        (LOGS / "monitor_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        log(f"progress {n}/{EXPECTED_FINAL} | queue={'up' if qa else 'down'} "
            f"finalizer={'up' if fa else 'down'} | artifacts={artifacts_done()} "
            f"| failed={len(failed)} | errors={len(errs)}")

        if n >= EXPECTED_FINAL and artifacts_done() and not qa and not fa:
            log("COMPLETE: all final runs recorded and artifacts generated")
            break

        if qa and not fa and not artifacts_done():
            launch_finalizer(qpid)
        elif not qa:
            queue_complete = (LOGS / "queue_complete.flag").exists()
            if fa:
                # queue finished or died. If it finished normally, let the finalizer finish.
                if queue_complete or n >= EXPECTED_FINAL:
                    pass
                else:
                    # premature finalizer on incomplete runs -> restart both
                    if MAX_RESTARTS and NRESTART >= MAX_RESTARTS:
                        log("MAX_RESTARTS reached; stopping watchdog (ask the user)")
                        break
                    NRESTART += 1
                    log(f"queue down with incomplete runs -> restart #{NRESTART}")
                    kill_pid(fpid)
                    newq = launch_queue()
                    launch_finalizer(newq)
            else:
                if artifacts_done() and n >= EXPECTED_FINAL:
                    pass  # completion recorded; next loop exits
                elif n < EXPECTED_FINAL:
                    if MAX_RESTARTS and NRESTART >= MAX_RESTARTS:
                        log("MAX_RESTARTS reached; stopping watchdog (ask the user)")
                        break
                    NRESTART += 1
                    log(f"runs missing ({n}/{EXPECTED_FINAL}) -> restart queue #{NRESTART}")
                    newq = launch_queue()
                    launch_finalizer(newq)
                else:
                    # all runs done, finalizer dead, artifacts missing -> finalize now
                    log("all runs done; relaunching finalizer without queue")
                    launch_finalizer(0)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
