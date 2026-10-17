#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
KIMI_PYTHON="${KIMI_PYTHON:-.venv-kimi/bin/python}"
for model in qwen3_omni_instruct audio_flamingo_3 kimi_audio_7b; do
  runner="$PYTHON"
  if [ "$model" = kimi_audio_7b ]; then
    runner="$KIMI_PYTHON"
  fi
  "$runner" scripts/run_experiment.py --config config/stemfoley.yaml --task llm --model "$model" "$@"
done
for task in clap cnn14 report; do
  "$PYTHON" scripts/run_experiment.py --config config/stemfoley.yaml --task "$task" "$@"
done
