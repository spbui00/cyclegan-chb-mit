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

# 2) run training, pointing it at our Visdom server
python train.py \
  --dataroot ./data \
  --name sine_with_noise \
  --model sine_cycle_gan \
  --dataset_mode dummy \
  --n_samples 100 \
  --window_size 256 \
  --noise_std 0.5 \
  --batch_size 8 \
  --epoch_count 1 \
  --n_epochs 2 \
  --display_id 1 \
  --display_freq 20 \
  --print_freq 20 \
  --save_latest_freq 50 \
  --save_epoch_freq 1 \
  --display_server http://localhost \
  --display_port 8098 \
  --display_env main \
  "$@"

