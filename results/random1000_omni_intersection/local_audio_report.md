# AudioSet caption presence evaluation results

Input examples: **1,000** (positive and complete-frame-negative evaluation clip/class pairs).

## LLM yes/no predictions

Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.

| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini_3_pro_medium | 880 | 0 | 0.8011 | 0.7893 | 0.8519 | 0.8194 | 397 | 69 | 106 | 308 |
| gemini_argon_medium | 867 | 13 | 0.7624 | 0.7466 | 0.8344 | 0.7881 | 383 | 76 | 130 | 278 |
| qwen2_audio_7b_instruct | 1000 | 0 | 0.5200 | 0.4922 | 0.9528 | 0.6491 | 444 | 22 | 458 | 76 |

## Configured model IDs

- `gemma4_e4b_it`: `google/gemma-4-E4B-it` (max new tokens: `16`)
- `qwen2_audio_7b_instruct`: `Qwen/Qwen2-Audio-7B-Instruct` (max new tokens: `16`)

- Maximum new tokens: `1`
- Score batch size: `128`
- LLM Transformers micro-batch size: `128`
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
