#!/bin/bash
set -e
export GSD_BASE=/root/DinoRL/gamescaling
export GSD_DINORL=/root/DinoRL
export GSD_RESULTS=/root/DinoRL/results
export FIREFOX_BIN=/root/tools/firefox/firefox

echo "=== [$(date)] STEP 1: collect_expert (fixed group recording) ==="
cd /root/DinoRL/gamescaling
python3 collect_expert.py --episodes 5 --seed-offset 30000 --max-frames 20000

echo "=== [$(date)] STEP 2: verify jump labels ==="
python3 -c "
import json
n=0; jumps=0
for line in open('/root/DinoRL/results/expert_data.jsonl'):
    d=json.loads(line)
    n+=1
    if 'Up' in d['groups']: jumps+=1
print(f'total={n} rows_with_jump={jumps} ({100*jumps/max(n,1):.1f}%)')
assert jumps > 0, 'No jump labels recorded — recording still broken!'
print('Jump labels OK')
"

echo "=== [$(date)] STEP 3: LoRA SFT training ==="
python3 sft_lora.py --out /root/DinoRL/results/sft_lora

echo "=== [$(date)] PIPELINE COMPLETE ==="
