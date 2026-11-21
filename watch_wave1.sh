#!/usr/bin/env bash
# Long-poll watcher: exits 0 when wave1 finished, 1 if orchestrator died, 2 on chunk timeout.
cd /root/DinoRL
LOG=results/exp/orchestrator.log
for i in $(seq 1 22); do   # 22 * 5min ~= 110 min per invocation
  if grep -q "parallel orchestrator done" "$LOG" 2>/dev/null; then
    echo "WAVE1_DONE $(date)"
    tail -25 "$LOG"
    exit 0
  fi
  if ! pgrep -f "run_all_paralle[l]" >/dev/null; then
    echo "ORCHESTRATOR_DEAD $(date)"
    tail -40 "$LOG"
    exit 1
  fi
  if [ $((i % 5)) -eq 0 ]; then
    echo "=== heartbeat $(date) ==="
    for f in results/exp/*/status.txt; do
      [ -f "$f" ] && echo "-- $f: $(tail -1 "$f")"
    done
    grep -l ERROR results/exp/*/status.txt 2>/dev/null && echo "!!! ERROR DETECTED ABOVE !!!"
  fi
  sleep 300
done
echo "WATCH_CHUNK_END $(date) (wave1 still running)"
for f in results/exp/*/status.txt; do
  [ -f "$f" ] && echo "-- $f: $(tail -1 "$f")"
done
exit 2
