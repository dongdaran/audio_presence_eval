# Qwen2-Audio / Qwen2.5-Omni TAD

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
The gate and margins are computed in float32.

## Qwen2.5-Omni baseline + Qwen2-Audio / Qwen2.5-Omni TAD

Use `--model qwen2_5_omni_7b` to load `Qwen/Qwen2.5-Omni-7B` with the
text-only Thinker class (no speech generation), its official processor, and its
recommended system message. Every method uses the same user/system prompts.
The silent branch's mutable RoPE offsets are isolated from real generation.
TAD on Omni is an adaptation of the paper's algorithm, not a claim that the
paper evaluated this model.

```bash
# GPU server, dependencies: requirements-local-audio.txt (includes Transformers).
bash scripts/setup_qwen_audio_env.sh
bash scripts/run_qwen25_comparison.sh
# Print the ten full-evaluation combinations without loading models:
bash scripts/run_qwen25_comparison.sh list
# Only validate 2 examples (1 yes, 1 no) per dataset and setting:
bash scripts/run_qwen25_comparison.sh smoke
# Or select another environment:
PYTHON_BIN=/path/to/gpu/venv/bin/python bash scripts/run_qwen25_comparison.sh
```

To run smoke checks explicitly before launching a background full evaluation:

```bash
bash scripts/setup_qwen_audio_env.sh
bash scripts/run_qwen25_comparison.sh smoke
# Proceed only if smoke exits successfully.
mkdir -p results/qwen25_comparison
nohup bash scripts/run_qwen25_comparison.sh full \
  > results/qwen25_comparison/run.log 2>&1 &
tail -f results/qwen25_comparison/run.log
```

Check per-setting `*.metrics.json` for `yes_recall`, `no_recall`,
`balanced_accuracy` and `invalid_failed`. Compare methods within the same dataset.
The `full` mode skips smoke checks; the default `all` mode runs both phases
for the current model, then unloads it before loading the next model.

The launcher delegates to `scripts/run_qwen_shared.py`, sequentially on one GPU:
StemFoley (476) and random1000_omni_intersection (1000), each with:

| Model | Settings | Runs across both datasets |
|---|---|---:|
| Qwen2.5-Omni-7B | baseline | 2 |
| Qwen2.5-Omni-7B | TAD alpha=0.5/1.0 | 4 |
| Qwen2-Audio-7B-Instruct | TAD alpha=0.5/1.0 | 4 |

Ten full evaluations, 7,380 predictions. TAD uses tau=0.2 and gamma=2.5.
Smoke validation adds two examples per combination (20 predictions).
AAD is excluded from this launcher. The standalone runner retains `--method aad`
for a later experiment. Qwen2-Audio baseline is not rerun.
Outputs are isolated under `results/<dataset>/qwen25_comparison/<setting>/`
for Omni and `results/<dataset>/qwen2_audio_comparison/<setting>/` for Qwen2-Audio.
Invalid/error predictions stop the launcher after their results are saved.
Restarting the same command resumes successful rows. Running on a CUDA machine
is required; the CPU preflight stops before evaluation or weight downloads.

### Shared inference

- Each model loads once across both datasets, settings, and smoke/full phases.
- Each batch loads/resamples waveforms once and prepares real/silent features once
  for all active settings. Model-specific features are not shared between models.
- Real-audio prefill logits and KV state are computed once per batch and reused
  across settings. Each generation gets an independent copy of the real KV state
  to avoid mutations leaking between settings. Omni position offsets are restored.
- Silent-branch logits are memoized by the entire batch's exact generated prefix.
  Diverging histories are computed separately; no logits are reused across batches.
- Generation remains greedy, full vocabulary, max 16 tokens. This is not a
  first-token-only evaluation. The silent branch remains uncached in KV terms.
- Shared prefill retains a pristine snapshot plus a live generation cache, so
  VRAM can increase. It does not retain model weights for both models at once.
- Resume metadata and output locations remain compatible with independent runs.
  Success rows are preserved; only pending variants are computed. Results are
  checkpointed every 32 examples and at completion or an ordinary exception.
  A hard kill may require replaying the last incomplete checkpoint.
- Use the combined console/nohup log for progress; per-setting metrics/TSVs stay
  separate. `--batch-size N` can be passed to the shell launcher.

Actual GPU throughput and peak VRAM must be measured on the evaluation machine.

Tests (require PyTorch, but no GPU or model download):

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_qwen*.py'
```
