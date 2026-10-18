#!/usr/bin/env bash
# Qwen2.5-Omni-7B: baseline + AAD/TAD at alpha 0.5 and 1.0.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
DATASETS=(stemfoley random1000_omni_intersection)
MODE="${1:-all}"
case "$MODE" in
  all|smoke|full) ;;
  *) echo "Usage: bash $0 [all|smoke|full]" >&2; exit 2 ;;
esac
"$PYTHON_BIN" - <<'PY'
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU required. Run this launcher on the GPU server.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

run_phase() {
  local phase="$1"
  local limit_args=()
  if [[ "$phase" == smoke ]]; then limit_args=(--limit 2); fi
  for dataset in "${DATASETS[@]}"; do
    for setting in baseline aad05 aad10 tad05 tad10; do
      local method alpha
      case "$setting" in
        baseline) method=default; alpha=1.0 ;;
        aad05) method=aad; alpha=0.5 ;;
        aad10) method=aad; alpha=1.0 ;;
        tad05) method=tad; alpha=0.5 ;;
        tad10) method=tad; alpha=1.0 ;;
      esac
      local output_dir="results/$dataset/qwen25_comparison/$setting"
      mkdir -p "$output_dir"
      "$PYTHON_BIN" -u scripts/run_qwen_tad.py \
        --config "config/${dataset}_local_audio.yaml" \
        --model qwen2_5_omni_7b --method "$method" \
        --alpha "$alpha" --tau 0.2 --gamma 2.5 \
        --batch-size 1 --max-new-tokens 16 --prompt config \
        --results-dir "$output_dir" "${limit_args[@]}" \
        2>&1 | tee "$output_dir/$phase.log"
    done
  done
}

# Fail on any invalid/failed smoke prediction before starting full evaluation.
if [[ "$MODE" != full ]]; then run_phase smoke; fi
if [[ "$MODE" != smoke ]]; then run_phase full; fi
