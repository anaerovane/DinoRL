#!/bin/bash
# Machine 2 experiments - run this on the second machine
# Requires: git clone https://github.com/anaerovane/DinoRL.git && cd DinoRL && pip install -r requirements.txt

cd "$(dirname "$0")"

echo "=== Machine 2 Experiments ==="
echo "Starting at $(date)"

# 1. PPO with high-res input (96x96)
echo "[1/3] PPO high-res 96x96..."
python3 train_ppo.py --obs-res 96 --name ppo_res96 --seed 20261016 &
PID1=$!

# 2. PPO + LSTM (temporal instead of frame stacking)
echo "[2/3] PPO LSTM..."
python3 train_ppo.py --use-lstm --name ppo_lstm --seed 20261016 &
PID2=$!

# 3. PPO + curiosity reward
echo "[3/3] PPO curiosity..."
python3 train_ppo.py --curiosity --name ppo_curiosity --seed 20261016 &
PID3=$!

echo "PIDs: $PID1 $PID2 $PID3"
echo "Waiting for completion..."
wait $PID1 $PID2 $PID3
echo "All machine 2 experiments done at $(date)"
