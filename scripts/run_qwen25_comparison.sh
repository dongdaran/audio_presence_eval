#!/usr/bin/env bash
# Qwen2.5-Omni baseline, plus Qwen2.5-Omni / Qwen2-Audio TAD (10 runs).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
DATASETS=(stemfoley random1000_omni_intersection)
MODE="${1:-all}"
case "$MODE" in
  all|smoke|full|list) ;;
  *) echo "Usage: bash $0 [all|smoke|full|list]" >&2; exit 2 ;;
esac
if [[ "$MODE" != list ]]; then
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Project environment missing. Run: bash scripts/setup_qwen_audio_env.sh' >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
try:
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
except ImportError as exc:
    raise SystemExit(f'{exc}\nRun: bash scripts/setup_qwen_audio_env.sh') from exc
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU required. Run this launcher on the GPU server.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
fi

run_phase() {
  local phase="$1"
  local limit_args=()
  if [[ "$phase" == smoke ]]; then limit_args=(--limit 2); fi
  for dataset in "${DATASETS[@]}"; do
    for model in qwen2_5_omni_7b qwen2_audio_7b_instruct; do
    for setting in baseline tad05 tad10; do
      if [[ "$model" == qwen2_audio_7b_instruct && "$setting" == baseline ]]; then
        continue
      fi
      local method alpha
      case "$setting" in
        baseline) method=default; alpha=1.0 ;;
        tad05) method=tad; alpha=0.5 ;;
        tad10) method=tad; alpha=1.0 ;;
      esac
      local comparison_dir=qwen25_comparison
      if [[ "$model" == qwen2_audio_7b_instruct ]]; then comparison_dir=qwen2_audio_comparison; fi
      local output_dir="results/$dataset/$comparison_dir/$setting"
      if [[ "$phase" == list ]]; then
        printf '%s\t%s\t%s\talpha=%s\ttau=0.2\tgamma=2.5\t%s\n' \
          "$dataset" "$model" "$method" "$alpha" "$output_dir"
        continue
      fi
      mkdir -p "$output_dir"
      "$PYTHON_BIN" -u scripts/run_qwen_tad.py \
        --config "config/${dataset}_local_audio.yaml" \
        --model "$model" --method "$method" \
        --alpha "$alpha" --tau 0.2 --gamma 2.5 \
        --batch-size 1 --max-new-tokens 16 --prompt config \
        --results-dir "$output_dir" "${limit_args[@]}" \
        2>&1 | tee "$output_dir/$phase.log"
    done
    done
  done
}

if [[ "$MODE" == list ]]; then run_phase list; exit 0; fi
# Fail on any invalid/failed smoke prediction before starting full evaluation.
if [[ "$MODE" != full ]]; then run_phase smoke; fi
if [[ "$MODE" != smoke ]]; then run_phase full; fi
