#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-openrouter/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
case "${1:-}" in
  argon) model=gemini_argon_medium ;;
  pro) model=gemini_3_pro_medium ;;
  *) echo "Usage: bash $0 [argon|pro]" >&2; exit 2 ;;
esac
for dataset in random1000_omni_intersection stemfoley; do
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_experiment.py" \
    --config "$ROOT_DIR/config/${dataset}_gemini_medium.yaml" \
    --task llm --model "$model"
done
