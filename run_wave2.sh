#!/usr/bin/env bash
# DinoRL wave 2: new experiments after the comparison suite.
#   A) behavioral cloning (rule teacher -> vision student), then eval
#   B) Rainbow (if train_rainbow.py exists) + seed-2 runs (dqn/c51/ppo)
#   C) seed-2 state_dqn + seed-3 runs (dqn/c51/ppo)
#   D) seed-3 state_dqn + remaining evals
set -u
cd /root/DinoRL
mkdir -p results/exp
LOG=results/exp/wave2.log
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "===== wave2 orchestrator start (pid $$) ====="

# ---- A: behavioral cloning ----
say "PHASE-A behavioral cloning"
python train_bc.py --name bc_vision --episodes 48 --n-envs 4 --epochs 8 --max-frames-ep 10000 \
  && say "bc training ok" || say "bc FAILED"
if [ -f results/exp/bc_vision/best.pt ]; then
  python eval_policy.py --policy model --name bc --ckpt results/exp/bc_vision/best.pt --n-envs 4 \
    || say "bc eval FAILED"
fi
say "PHASE-A done"

# ---- B: rainbow + seed2 (20261016) ----
say "PHASE-B rainbow + s2 {dqn,c51,ppo}"
if [ -f train_rainbow.py ]; then
  python train_rainbow.py --name rainbow_vision --n-envs 4 --hours 24 &
  say "  rainbow pid $!"
else
  say "  train_rainbow.py missing - skip"
fi
python train_offpolicy.py --algo dqn --obs vision --name dqn_s2 --seed 20261016 --n-envs 4 --hours 24 &
say "  dqn_s2 pid $!"
python train_offpolicy.py --algo c51 --obs vision --name c51_s2 --seed 20261016 --n-envs 4 --hours 24 &
say "  c51_s2 pid $!"
python train_ppo.py --name ppo_s2 --seed 20261016 --n-envs 4 --hours 24 &
say "  ppo_s2 pid $!"
wait
say "PHASE-B done"

# ---- C: s2 state + s3 {dqn,c51,ppo} ----
say "PHASE-C state_s2 + s3 {dqn,c51,ppo}"
python train_offpolicy.py --algo ddqn --obs state --name state_dqn_s2 --seed 20261016 --n-envs 4 --hours 24 &
say "  state_s2 pid $!"
python train_offpolicy.py --algo dqn --obs vision --name dqn_s3 --seed 20261017 --n-envs 4 --hours 24 &
say "  dqn_s3 pid $!"
python train_offpolicy.py --algo c51 --obs vision --name c51_s3 --seed 20261017 --n-envs 4 --hours 24 &
say "  c51_s3 pid $!"
python train_ppo.py --name ppo_s3 --seed 20261017 --n-envs 4 --hours 24 &
say "  ppo_s3 pid $!"
wait
say "PHASE-C done"

# ---- D: s3 state ----
say "PHASE-D state_s3"
python train_offpolicy.py --algo ddqn --obs state --name state_dqn_s3 --seed 20261017 --n-envs 4 --hours 24 &
say "  state_s3 pid $!"
wait
say "PHASE-D done"

# ---- E: evals of every wave2 model ----
say "PHASE-E evals"
for dir in rainbow_vision dqn_s2 c51_s2 ppo_s2 state_dqn_s2 dqn_s3 c51_s3 ppo_s3 state_dqn_s3; do
  if [ -f "results/exp/$dir/best.pt" ]; then
    algo=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('algo'))")
    obs=$(python -c "import torch;print(torch.load('results/exp/$dir/best.pt',map_location='cpu').get('obs'))")
    python eval_policy.py --policy model --name "$dir" --ckpt "results/exp/$dir/best.pt" --algo "$algo" --obs "$obs" --n-envs 4 &
    say "  eval $dir pid $!"
  else
    say "  missing best.pt for $dir - skip"
  fi
done
wait
say "PHASE-E done"

python make_wave2.py && say "wave2 report built" || say "make_wave2 FAILED"
say "===== wave2 orchestrator done ====="
