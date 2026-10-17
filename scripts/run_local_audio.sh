#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
export PATH="$ROOT_DIR/.venv/bin:$PATH"
case "${1:-all}" in
  gemma) MODELS=(gemma4_e4b_it) ;;
  qwen2) MODELS=(qwen2_audio_7b_instruct) ;;
  all) MODELS=(gemma4_e4b_it qwen2_audio_7b_instruct) ;;
  *) echo "Usage: bash $0 [gemma|qwen2|all] [--limit N]" >&2; exit 2 ;;
esac
if (( $# )); then shift; fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Create $ROOT_DIR/.venv and install requirements-local-audio.txt first." >&2
  exit 1
fi
# Use the ffmpeg binary bundled with the environment if no system binary exists.
if ! command -v ffmpeg >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
import imageio_ffmpeg
link = Path(sys.executable).parent / "ffmpeg"
if not link.exists():
    link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
fi
# Sequential models avoid loading both sets of weights on one GPU.
for model in "${MODELS[@]}"; do
  for dataset in stemfoley random1000_omni_intersection; do
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_experiment.py" \
      --config "$ROOT_DIR/config/${dataset}_local_audio.yaml" \
      --task llm --model "$model" "$@"
  done
done
