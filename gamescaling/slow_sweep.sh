#!/bin/bash
# Parallel slowdown sweep using the EXISTING (DQN ep1) adapter.
# The push1000 pipeline will later retrain; this keeps scores moving now.
set -u
export GSD_BASE=/root/DinoRL/gamescaling
export GSD_DINORL=/root/DinoRL
export GSD_RESULTS=/root/DinoRL/results
export FIREFOX_BIN=/root/tools/firefox/firefox
export PATH="/usr/local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

cd /root/DinoRL/gamescaling || exit 1
LOG=/root/DinoRL/results/gs_slow_sweep.log
ADAPTER=/root/DinoRL/results/sft_lora_dqn_ep1.bak
echo "=== [$(date)] SLOW SWEEP START adapter=$ADAPTER ===" >>"$LOG"

# skip if adapter's existing summary already has mean>=1000 for this tag
for spec in "s015 0.15 5" "s025 0.25 5" "s035 0.35 5" "s050 0.50 4" "s070 0.70 4"; do
  set -- $spec
  tag=$1; slow=$2; eps=$3
  echo "=== [$(date)] EVAL $tag slowdown=$slow ===" >>"$LOG"
  python3 gamescaling_dino.py \
    --episodes "$eps" --device cuda --adapter "$ADAPTER" \
    --tag "sweep_$tag" --max-decisions 4000 --max-seconds 7200 \
    --slowdown "$slow" --seed-offset 10000 --temperature 0 \
    >>"$LOG" 2>&1
  echo "=== [$(date)] EVAL $tag rc=$? ===" >>"$LOG"
  # early-exit if this config already clears 1000
  python3 - <<PY >>"$LOG" 2>&1
import json,re
from pathlib import Path
log=Path("$LOG").read_text(errors='ignore')
ms=re.findall(r'AGGREGATE: (\{.*\})', log)
if ms:
    agg=json.loads(ms[-1])
    print(f"CHECK tag=$tag mean={agg.get('mean')}")
    if agg.get('mean',0) >= 1000:
        raise SystemExit(42)
PY
  if [ $? -eq 42 ]; then
    echo "=== [$(date)] TARGET REACHED at $tag ===" >>"$LOG"
    break
  fi
done

echo "=== [$(date)] SLOW SWEEP DONE ===" >>"$LOG"
grep 'AGGREGATE:' "$LOG" >>"$LOG"
