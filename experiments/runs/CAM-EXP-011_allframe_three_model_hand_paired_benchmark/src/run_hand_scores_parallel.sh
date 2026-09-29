#!/usr/bin/env bash
# Shard the hand-score computation across CPU workers. Pure parallelism:
# each worker solves a disjoint set of videos with identical code and inputs.
set -u
cd "$(dirname "$0")/.."
PY=/d/hand-demo/experiments/.venv/Scripts/python
N=${1:-8}
for ((k=0;k<N;k++)); do
  $PY src/compute_generic_hand_score.py --shard $k --nshards $N \
      > "results/raw/_handscore_shard${k}.log" 2>&1 &
done
wait
echo "=== ALL_HAND_SCORE_SHARDS_COMPLETE"
