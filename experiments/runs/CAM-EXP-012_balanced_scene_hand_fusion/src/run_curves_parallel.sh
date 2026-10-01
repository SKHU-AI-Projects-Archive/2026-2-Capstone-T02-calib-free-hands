#!/usr/bin/env bash
# Shard the cross-fitted hand-curve computation across CPU workers.
# Pure parallelism: identical code, identical frames, identical grid.
set -u
cd "$(dirname "$0")/.."
PY=/d/hand-demo/experiments/.venv/Scripts/python
VARIANT=${1:-correct}
N=${2:-8}
for ((k=0;k<N;k++)); do
  $PY src/compute_hand_curves.py --variant "$VARIANT" --shard $k --nshards $N \
      > "results/raw/_curves_${VARIANT}_shard${k}.log" 2>&1 &
done
wait
echo "=== ALL_${VARIANT}_CURVE_SHARDS_COMPLETE"
