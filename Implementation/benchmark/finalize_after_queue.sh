#!/usr/bin/env bash
# Wait for the queue PID, then validate -> robustness -> aggregate/report/plots.
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(cd ../.. && pwd)"
PY="$ROOT/.venv/bin/python"
PID="${1:-}"
mkdir -p logs

if [ -n "$PID" ]; then
  echo "waiting for queue pid $PID" >> logs/finalize.log
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi

echo "=== validate $(date) ===" >> logs/finalize.log
"$PY" validate_results.py > logs/validate.log 2>&1 || true
echo "=== perturbations $(date) ===" >> logs/finalize.log
"$PY" robustness/make_perturbations.py >> logs/robustness.log 2>&1 || true
echo "=== robustness $(date) ===" >> logs/finalize.log
"$PY" robustness/eval_robustness.py >> logs/robustness.log 2>&1 || true
echo "=== cross-corpus $(date) ===" >> logs/finalize.log
"$PY" cross_corpus/eval_cross_corpus.py >> logs/cross_corpus.log 2>&1 || true
echo "=== feature ablation $(date) ===" >> logs/finalize.log
"$PY" classical/ablation_features.py >> logs/ablation.log 2>&1 || true
echo "=== sensitivity $(date) ===" >> logs/finalize.log
"$PY" sensitivity/run_sensitivity.py >> logs/sensitivity.log 2>&1 || true
echo "=== aggregate $(date) ===" >> logs/finalize.log
"$PY" analysis/aggregate.py --plots >> logs/aggregate.log 2>&1 || true
echo "DONE $(date)" >> logs/finalize.log
