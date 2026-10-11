#!/usr/bin/env bash
# Train, evaluate and report all four models one after another (one GPU job at a time,
# so every model sees the same machine). Usage, from the repo root inside WSL:
#   bash scripts/run_all.sh [model ...]        default: all four
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=("${@:-maskrcnn yolo11n yolo26n rfdetr}")
read -r -a MODELS <<< "${MODELS[*]}"
mkdir -p runs
for m in "${MODELS[@]}"; do
  echo "[$(date +%T)] train $m"
  .venv/bin/python scripts/train.py --model "$m" > "runs/train_$m.log" 2>&1 || echo "[$(date +%T)] TRAIN FAILED $m (see runs/train_$m.log)"
done
for m in "${MODELS[@]}"; do
  [ -f "runs/train/$m/train_summary.json" ] || { echo "[$(date +%T)] skip eval $m (no weights)"; continue; }
  echo "[$(date +%T)] evaluate $m"
  .venv/bin/python scripts/evaluate.py --model "$m" > "runs/eval_$m.log" 2>&1 || echo "[$(date +%T)] EVAL FAILED $m (see runs/eval_$m.log)"
done
.venv/bin/python scripts/report.py --out docs/results_synthetic.md
echo "[$(date +%T)] done"
