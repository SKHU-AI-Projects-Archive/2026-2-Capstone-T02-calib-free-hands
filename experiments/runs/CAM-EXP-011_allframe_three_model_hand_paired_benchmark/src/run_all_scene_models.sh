#!/usr/bin/env bash
# All three scene models over every usable RGB frame, sequentially.
set -u
cd "$(dirname "$0")/.."
PY=/d/hand-demo/experiments/.venv-calib/Scripts/python
export PYTHONPATH=/d/hand-demo
for M in PERSPECTIVE_FIELDS ANYCALIB GEOCALIB; do
  echo "=== START $M $(date)"
  $PY src/run_scene_allframes.py --model "$M" 2>&1 | grep -vE "Warning|warn"
  echo "=== DONE $M $(date)"
done
echo "=== ALL_SCENE_MODELS_COMPLETE"
