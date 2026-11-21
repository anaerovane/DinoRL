#!/usr/bin/env bash
# Parallel DinoRL comparison experiments. 4 trainings run simultaneously.
set -u
cd /root/DinoRL
mkdir -p results/exp
LOG=results/exp/orchestrator.log
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "===== parallel orchestrator start (pid $$) ====="

# ---- Phase 1: fast evals in parallel (random / rule / reused ddqn) ----
say "PHASE1 evals in parallel"
python eval_policy.py --policy random --name random --n-envs 4 &
say "  random eval pid $!"
python eval_policy.py --policy rule --name rule --n-envs 4 &
say "  rule eval pid $!"
python eval_policy.py --policy model --name ddqn --ckpt results/dino_trained_best.pt --algo ddqn --obs vision --n-envs 4 &
say "  ddqn eval pid $!"
wait
say "PHASE1 done"

# ---- Phase 2: 4 trainings in parallel, 4 envs each (16 Firefox on 15 cores) ----
# Equal frame budget 2.23M for every method; hours cap only as a safety net.
say "PHASE2 trainings x4 in parallel"
python train_offpolicy.py --algo dqn --obs vision --name dqn_vision --n-envs 4 --hours 24 &
say "  dqn pid $!"
python train_offpolicy.py --algo c51 --obs vision --name c51_vision --n-envs 4 --hours 24 &
say "  c51 pid $!"
python train_ppo.py --name ppo_vision --n-envs 4 --hours 24 &
say "  ppo pid $!"
python train_offpolicy.py --algo ddqn --obs state --name state_dqn --n-envs 4 --hours 24 &
say "  state_dqn pid $!"
wait
say "PHASE2 done"

# ---- Phase 3: greedy eval of trained models in parallel ----
say "PHASE3 model evals in parallel"
for pair in "dqn_vision:dqn" "c51_vision:c51" "ppo_vision:ppo" "state_dqn:state_dqn"; do
  dir="${pair%%:*}"; nm="${pair##*:}"
  if [ -f "results/exp/$dir/best.pt" ]; then
    algo=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('algo'))")
    obs=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('obs'))")
    python eval_policy.py --policy model --name "$nm" --ckpt "results/exp/$dir/best.pt" --algo "$algo" --obs "$obs" --n-envs 4 &
    say "  eval $nm pid $!"
  else
    say "  missing best.pt for $dir - skip"
  fi
done
wait
say "PHASE3 done"

# ---- Phase 4: aggregate ----
python make_compare.py && say "compare artifacts built" || say "make_compare FAILED"
say "===== parallel orchestrator done ====="
