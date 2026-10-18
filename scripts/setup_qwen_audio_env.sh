#!/usr/bin/env bash
# Populate the existing project .venv; never recreate it or use system Python.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required to install into the project .venv.' >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  uv venv "$ROOT_DIR/.venv" --python 3.12
fi
uv pip install --python "$PYTHON_BIN" -c constraints-gpu.txt \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$PYTHON_BIN" -r requirements-local-audio.txt
"$PYTHON_BIN" - <<'PY'
import sys
import shutil
from pathlib import Path
import torch
import torchvision
import torchaudio
import transformers
import librosa
import soundfile
import accelerate
import yaml
import dotenv
from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
if shutil.which('ffmpeg') is None:
    import imageio_ffmpeg
    target = Path(sys.executable).parent / 'ffmpeg'
    if not target.exists():
        target.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    print('Bundled ffmpeg:', target)
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('Transformers:', transformers.__version__)
print('Qwen2-Audio / Qwen2.5-Omni imports: OK')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Packages installed; CUDA GPU is not visible on this machine. Evaluation requires a GPU.')
PY
