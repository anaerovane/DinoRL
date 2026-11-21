#!/usr/bin/env bash
# DinoRL comparison experiment orchestrator. Sequential, ~15h total.
set -u
cd /root/DinoRL
mkdir -p results/exp
LOG=results/exp/orchestrator.log
exec >> "$LOG" 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

run_eval() {
  say "EVAL $*"
  python eval_policy.py "$@" || say "EVAL FAILED: $*"
}

run_train() {
  say "TRAIN $*"
  python "$@" || say "TRAIN FAILED: $*"
}

say "===== orchestrator start ====="

# 1. non-learning baselines (fast)
run_eval --policy random --name random
run_eval --policy rule --name rule

# 2. vanilla DQN (vision)
run_train train_offpolicy.py --algo dqn --obs vision --name dqn_vision

# 3. eval the reused Double-DQN baseline (train_fast.py best)
run_eval --policy model --name ddqn --ckpt results/dino_trained_best.pt --algo ddqn --obs vision

# 4. C51 (vision)
run_train train_offpolicy.py --algo c51 --obs vision --name c51_vision

# 5. PPO (vision)
run_train train_ppo.py --name ppo_vision

# 6. State-DQN (non-vision, direct engine interface) - double dqn target
run_train train_offpolicy.py --algo ddqn --obs state --name state_dqn

# 7. eval newly trained models (greedy, 50 fixed seeds)
for pair in "dqn_vision:dqn" "c51_vision:c51" "ppo_vision:ppo" "state_dqn:state_dqn"; do
  dir="${pair%%:*}"; nm="${pair##*:}"
  if [ -f "results/exp/$dir/best.pt" ]; then
    algo=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('algo'))")
    obs=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('obs'))")
    run_eval --policy model --name "$nm" --ckpt "results/exp/$dir/best.pt" --algo "$algo" --obs "$obs"
  else
    say "missing best.pt for $dir - skip eval"
  fi
done

# 8. aggregate + plot
python make_compare.py && say "compare artifacts built" || say "make_compare FAILED"

say "===== orchestrator done ====="
