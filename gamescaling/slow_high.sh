#!/bin/bash
set -u
export GSD_BASE=/root/DinoRL/gamescaling GSD_DINORL=/root/DinoRL GSD_RESULTS=/root/DinoRL/results
export FIREFOX_BIN=/root/tools/firefox/firefox PATH="/usr/local/bin:$PATH" CUDA_VISIBLE_DEVICES=0
cd /root/DinoRL/gamescaling || exit 1
LOG=/root/DinoRL/results/gs_slow_high.log
ADAPTER=/root/DinoRL/results/sft_lora_dqn_ep1.bak
echo "=== [$(date)] HIGH SLOW START ===" >>"$LOG"
for spec in "s080 0.80 4" "s120 1.20 4" "s200 2.00 3"; do
  set -- $spec
  echo "=== [$(date)] EVAL high_$1 slow=$2 ===" >>"$LOG"
  python3 gamescaling_dino.py --episodes $3 --device cuda --adapter "$ADAPTER" \
    --tag high_$1 --max-decisions 4000 --max-seconds 7200 \
    --slowdown $2 --seed-offset 10000 --temperature 0 >>"$LOG" 2>&1
  echo "=== [$(date)] high_$1 rc=$? ===" >>"$LOG"
done
echo "=== [$(date)] HIGH SLOW DONE ===" >>"$LOG"
