#!/usr/bin/env bash
# Keep one model loaded across datasets/settings; share preprocessing and logits.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Project environment missing. Run: bash scripts/setup_qwen_audio_env.sh' >&2
  exit 1
fi
exec "$PYTHON_BIN" -u scripts/run_qwen_shared.py "$@"
