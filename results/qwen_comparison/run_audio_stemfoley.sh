#!/usr/bin/env bash
set -euo pipefail
cd /workspace/audio_presence_eval
for tau in 0.5 1.0; do
  echo "Starting Audio StemFoley tau=$tau"
  bash scripts/run_qwen25_comparison.sh --models qwen2_audio_7b_instruct --datasets stemfoley --tau "$tau" --output-tag "tau_$tau" --batch-size 16
done
echo 'Audio StemFoley: both tau runs completed'
