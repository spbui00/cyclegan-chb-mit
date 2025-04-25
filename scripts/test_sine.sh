#!/usr/bin/env bash
# Helper script to launch the 1D sine↔noisy CycleGAN dummy training
# and fire up Visdom on port 8098 so display_current_results() shows up.
# Usage: ./scripts/run_dummy_train.sh [additional options]

set -e

# 1) launch Visdom in the background on port 8098 (only if it isn't already running)
if ! lsof -i:8098 >/dev/null; then
  echo "Starting Visdom server on port 8098..."
  python3 -m visdom.server -port 8098 &>/dev/null &
  # give it a sec to spin up
  sleep 2
else
  echo "Visdom already running on port 8098"
fi

python test.py \
  --dataroot ./data \
  --results_dir ./results_dummy \
  --checkpoints_dir ./checkpoints \
  --name sine_with_noise \
  --model sine_cycle_gan \
  --dataset_mode dummy \
  --window_size 256 \
  --n_samples 100 \
  --phase test \
  --epoch latest \
  --num_test 100 \
  --n_blocks 6 \
  --eval
