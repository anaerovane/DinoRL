#!/bin/bash
# Collect rule-expert data → LoRA SFT (many epochs) → sequential slowdown evals.
set -u
export GSD_BASE=/root/DinoRL/gamescaling
export GSD_DINORL=/root/DinoRL
export GSD_RESULTS=/root/DinoRL/results
export FIREFOX_BIN=/root/tools/firefox/firefox
export PATH="/usr/local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

cd /root/DinoRL/gamescaling || exit 1
LOG=/root/DinoRL/results/gs_push1000.log
echo "=== [$(date)] PUSH1000 START ===" >>"$LOG"

echo "=== [$(date)] STEP 1: collect rule expert ===" >>"$LOG"
python3 collect_expert.py --expert rule --episodes 10 --seed-offset 40000 --max-frames 20000 >>"$LOG" 2>&1
rc=$?
echo "=== [$(date)] collect rc=$rc ===" >>"$LOG"
if [ $rc -ne 0 ]; then echo "COLLECT FAILED" >>"$LOG"; exit 1; fi

python3 - <<'PY' >>"$LOG" 2>&1
import json
n=0; jumps=0; seeds=set()
for line in open('/root/DinoRL/results/expert_data.jsonl'):
    d=json.loads(line)
    n+=1
    seeds.add(d.get('seed'))
    if 'Up' in d.get('groups',''): jumps+=1
print(f'total={n} jumps={jumps} ({100*jumps/max(n,1):.1f}%) seeds={sorted(seeds)}')
assert jumps > 0 and n > 1000, 'expert data too small / no jumps'
print('expert data OK')
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "VERIFY FAILED" >>"$LOG"; exit 1; fi

echo "=== [$(date)] STEP 2: LoRA SFT epochs=10 per-seed=400 ===" >>"$LOG"
# fresh adapter dir
rm -rf /root/DinoRL/results/sft_lora_long
python3 sft_lora.py --epochs 10 --per-seed 400 --lr 2e-4 --out /root/DinoRL/results/sft_lora_long >>"$LOG" 2>&1
rc=$?
echo "=== [$(date)] sft rc=$rc ===" >>"$LOG"
if [ $rc -ne 0 ]; then echo "SFT FAILED" >>"$LOG"; exit 1; fi

ADAPTER=/root/DinoRL/results/sft_lora_long
# also refresh the default adapter path used by prior evals
rm -rf /root/DinoRL/results/sft_lora
mkdir -p /root/DinoRL/results/sft_lora
cp -a "$ADAPTER"/. /root/DinoRL/results/sft_lora/

run_eval() {
  local tag="$1" slow="$2" eps="$3"
  echo "=== [$(date)] EVAL tag=$tag slowdown=$slow episodes=$eps ===" >>"$LOG"
  python3 gamescaling_dino.py \
    --episodes "$eps" --device cuda --adapter "$ADAPTER" \
    --tag "$tag" --max-decisions 4000 --max-seconds 7200 \
    --slowdown "$slow" --seed-offset 10000 --temperature 0 \
    >>"$LOG" 2>&1
  echo "=== [$(date)] EVAL $tag rc=$? ===" >>"$LOG"
  python3 - <<PY >>"$LOG" 2>&1
import json,re,sys
from pathlib import Path
log=Path("$LOG").read_text(errors='ignore')
ms=re.findall(r'AGGREGATE: (\{.*\})', log)
print("TAG=$tag SLOW=$slow LAST_AGG=", ms[-1] if ms else "none")
if ms:
    agg=json.loads(ms[-1])
    if float(agg.get('mean',0)) >= 1000:
        print("TARGET_REACHED mean=", agg.get('mean'))
        sys.exit(42)
PY
  return $?
}

# Primary: slowdown 0.15 (historically best), escalate if needed
for spec in "lora_long_s10 0.10 5" "lora_long_s15 0.15 5" "lora_long_s25 0.25 5" "lora_long_s35 0.35 5" "lora_long_s50 0.50 4"; do
  set -- $spec
  run_eval "$1" "$2" "$3"
  rc=$?
  if [ $rc -eq 42 ]; then
    echo "=== [$(date)] TARGET ≥1000 REACHED at $1 ===" >>"$LOG"
    break
  fi
done

echo "=== [$(date)] PUSH1000 PIPELINE DONE ===" >>"$LOG"
grep -E 'AGGREGATE:|TAG=|TARGET' "$LOG" | tail -40 >>"$LOG"
