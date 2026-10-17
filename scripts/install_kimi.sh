#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python "${KIMI_PYTHON:-/venv/main/bin/python}" --allow-existing .venv-kimi
uv pip install --python .venv-kimi/bin/python -c constraints-gpu.txt \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-kimi/bin/python -c constraints-gpu.txt -r requirements-kimi.txt
kimi_source_dir="$PWD/model_cache/kimi-audio-source"
if [ ! -d "$kimi_source_dir/.git" ]; then
  git clone https://github.com/MoonshotAI/Kimi-Audio.git "$kimi_source_dir"
fi
git -C "$kimi_source_dir" checkout 349251e1d8f4f98d58fda59246381faecd7392e0
git -C "$kimi_source_dir" submodule update --init --recursive
uv pip install --python .venv-kimi/bin/python --no-build-isolation --no-deps "$kimi_source_dir"
.venv-kimi/bin/python - <<'CHECK'
import torch
from kimia_infer.api.kimia import KimiAudio
from flash_attn import flash_attn_func
assert torch.cuda.is_available()
q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(flash_attn_func(q, q, q)).all()
print("Kimi-Audio + FlashAttention OK:", torch.cuda.get_device_name(0))
CHECK

.venv-kimi/bin/python -m unittest discover -s tests -p test_kimi_compat.py
