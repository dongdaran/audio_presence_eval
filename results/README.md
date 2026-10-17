# AudioSet caption presence evaluation results

Input examples: **25,007** (positive and complete-frame-negative evaluation clip/class pairs).

## LLM yes/no predictions

Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.

| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi_audio_7b | 20889 | 4118 | 0.6263 | 0.5603 | 0.9012 | 0.6910 | 8728 | 957 | 6849 | 4355 |

## Configured model IDs

- `gemini_3_pro`: `google/gemini-3.1-pro-preview` (max new tokens: `4096`)
- `gemini_argon`: `google/gemini-4-argon` (max new tokens: `4096`)
- `kimi_audio_7b`: `moonshotai/Kimi-Audio-7B-Instruct` (max new tokens: `1`)
- `audio_flamingo_3`: `nvidia/audio-flamingo-3-hf` (max new tokens: `1`)
- `qwen3_omni_instruct`: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (max new tokens: `1`)

- Maximum new tokens: `1`
- Score batch size: `16`
- LLM Transformers micro-batch size: `16`
- Requested GPU: `NVIDIA A100 80GB`

## CLAPScore / CNN14 mean scores by reference label

CLAPScore is audio/text cosine similarity. CNN14 is the AudioSet class score for the mapped MID. Neither score is thresholded into a yes/no prediction.

Pooled mean is the mean over all scored samples; pooled SD is their standard deviation. Per-MID macro mean averages each MID's mean equally. Class-level tables below show yes/no means and their difference for each MID.

| Score | Reference label | N | MIDs scored | Pooled mean | Pooled SD | Per-MID macro mean |
|---|---|---:|---:|---:|---:|---:|
| clap | yes | 0 | 0 | NA | NA | NA |
| clap | no | 0 | 0 | NA | NA | NA |
| cnn14 | yes | 0 | 0 | NA | NA | NA |
| cnn14 | no | 0 | 0 | NA | NA | NA |

## CLAP scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|

## CNN14 scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|

## Notes

- `yes`/`no` are the reference labels from the existing evaluation TSVs; each row is one clip/class pair.
- Sample duration follows the source TSV. Some positive clips are shorter than 10 seconds; no padding is applied to model input audio.
- Failed downloads/inferences and unmapped CNN14 classes remain visible in result TSVs and are excluded from score means where no score exists.
- CLAPScore values are similarity scores, not calibrated probabilities.
