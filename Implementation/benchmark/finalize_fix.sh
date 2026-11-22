#!/usr/bin/env bash
# Wait for the current finalizer to finish, then re-run the evaluators with the
# seeded-pooler fix (TeluguBERT) and regenerate the report.
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(cd ../.. && pwd)"
PY="$ROOT/.venv/bin/python"
PID="${1:-}"
mkdir -p logs
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi
echo "=== robustness fix $(date) ===" >> logs/finalize_fix.log
"$PY" robustness/eval_robustness.py >> logs/robustness_fix.log 2>&1 || true
echo "=== cross-corpus fix $(date) ===" >> logs/finalize_fix.log
"$PY" cross_corpus/eval_cross_corpus.py >> logs/cross_corpus_fix.log 2>&1 || true
echo "=== aggregate fix $(date) ===" >> logs/finalize_fix.log
"$PY" analysis/aggregate.py --plots >> logs/aggregate_fix.log 2>&1 || true
echo "DONE $(date)" >> logs/finalize_fix.log
