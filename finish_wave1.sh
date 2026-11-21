#!/usr/bin/env bash
# Wait for the two wave-1 trainings to finish, then greedily evaluate just the
# ones still missing (ppo_vision / state_dqn are already evaluated).
set -u
cd /root/DinoRL
LOG=results/exp/finish_wave1.log
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "waiting for dqn_vision / c51_vision trainings..."
while pgrep -f -- '--name dqn_visio[n]' >/dev/null || pgrep -f -- '--name c51_visio[n]' >/dev/null; do
  sleep 60
done
say "both trainings done"

for pair in "dqn_vision:dqn" "c51_vision:c51"; do
  dir="${pair%%:*}"; nm="${pair##*:}"
  if [ -f "results/exp/$dir/eval_summary.csv" ]; then
    say "$nm already evaluated - skip"
    continue
  fi
  if [ ! -f "results/exp/$dir/best.pt" ]; then
    say "$nm missing best.pt - skip"
    continue
  fi
  algo=$(python3 -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('algo'))")
  obs=$(python3 -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('obs'))")
  say "eval $nm (algo=$algo obs=$obs)"
  python3 eval_policy.py --policy model --name "$nm" \
    --ckpt "results/exp/$dir/best.pt" --algo "$algo" --obs "$obs" --n-envs 4
  say "eval $nm rc=$?"
done

say "building compare artifacts"
python3 make_compare.py && say "make_compare OK" || say "make_compare FAILED"
say "wave-1 tail finished"
