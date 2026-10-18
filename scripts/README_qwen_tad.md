# Qwen2-Audio TAD

Standalone runner: `scripts/run_qwen_tad.py`. References:

- AAD implementation structure: https://github.com/GillbertHsu/Audio-Aware-Decoding/blob/main/inference_and_eval_qwen.py
- TAD equations 1–9: https://arxiv.org/html/2609.07286v1

Uses the existing Qwen2-Audio model loader and dataset/audio adapters. Unlike the
AAD repository's default `audio=None` embedding branch, this runner processes an
all-zero waveform of the same length through the audio encoder, as TAD specifies.
At every token it combines `(1+alpha)*real_logits - alpha*silent_logits`.
Only at the first token, it pools Yes/No variants with log-sum-exp and subtracts
`gamma` from Yes logits when the real-minus-silent Yes/No margin gain is below
`tau`. Defaults: alpha=1, tau=0.2, gamma=2.5. Subsequent steps use AAD alone.
Greedy decoding operates on the full vocabulary; it does not force a binary answer.

Run in a CUDA environment with this project's local audio dependencies installed:

```bash
python scripts/run_qwen_tad.py --config config/stemfoley_local_audio.yaml --limit 2
python scripts/run_qwen_tad.py --config config/stemfoley_local_audio.yaml
python scripts/run_qwen_tad.py --config config/random1000_omni_intersection_local_audio.yaml
```

`--method default|aad|tad` allows comparisons with the same prompt. The default
`--prompt config` preserves the existing experiment prompt; `--prompt paper`
uses the paper's focus/Yes-or-No prefix. Keep prompt and max-new-tokens identical
across comparisons. Answer parsing follows the existing evaluator: conflicting
Yes/No mentions are invalid, including explanations mentioning both answers.

Output defaults to `<original results>/tad/`, with method-specific TSV, run
metadata and metrics JSON (accuracy, Yes/No precision/recall/F1, balanced accuracy,
invalid/failed count). `--limit` writes separate smoke files. Successful rows
resume automatically; incompatible metadata requires another `--results-dir`.
Parameters can be changed with `--alpha`, `--tau`, `--gamma`, `--batch-size`, and
`--max-new-tokens`. Use a new results directory for each parameter configuration.

Default batch size is 1. The silent branch recomputes the complete prompt plus
generated history each step instead of relying on private Transformers cache
APIs; this is slower and uses additional memory. No Transformers patch is needed.
The gate and margins are computed in float32. Only Qwen2-Audio is supported.

Tests (require PyTorch, but no GPU or model download):

```bash
python -m unittest discover -s tests -p test_qwen_tad.py
```
