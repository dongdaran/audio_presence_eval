# StemFoley source presence evaluation

Eligible test examples: **476**. Ground truth: target_audible; undetermined rows excluded.

## LLM yes/no predictions

Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.

| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_flamingo_3 | 476 | 0 | 0.3046 | 0.6154 | 0.1375 | 0.2248 | 48 | 301 | 30 | 97 |
| gemini_3_pro_medium | 476 | 0 | 0.8088 | 0.9161 | 0.8138 | 0.8619 | 284 | 65 | 26 | 101 |
| gemini_argon_medium | 432 | 44 | 0.7292 | 0.8796 | 0.7415 | 0.8047 | 241 | 84 | 33 | 74 |
| gemma4_e4b_it | 476 | 0 | 0.4160 | 0.9863 | 0.2063 | 0.3412 | 72 | 277 | 1 | 126 |
| kimi_audio_7b | 476 | 0 | 0.7752 | 0.8980 | 0.7822 | 0.8361 | 273 | 76 | 31 | 96 |
| qwen2_audio_7b_instruct | 476 | 0 | 0.8025 | 0.8000 | 0.9742 | 0.8786 | 340 | 9 | 85 | 42 |
| qwen3_omni_instruct | 476 | 0 | 0.8004 | 0.9045 | 0.8138 | 0.8567 | 284 | 65 | 30 | 97 |

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
| clap | yes | 349 | 19 | 0.305593 | 0.157569 | 0.383396 |
| clap | no | 127 | 7 | 0.112434 | 0.123745 | 0.109440 |
| cnn14 | yes | 60 | 17 | 0.458060 | 0.289802 | 0.438357 |
| cnn14 | no | 11 | 6 | 0.001844 | 0.002323 | 0.002240 |

## CLAP scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/028ght | applause | 0 | NA | 3 | 0.121354 | NA |
| /m/018w8 | Basketball bounce | 3 | 0.448282 | 0 | NA | NA |
| /m/018vs | bass guitar | 3 | 0.238847 | 0 | NA | NA |
| /m/01c3q | Bassoon | 3 | 0.214728 | 0 | NA | NA |
| /m/01xqw | cello | 2 | 0.237402 | 0 | NA | NA |
| /m/02hnl | Drum kit | 7 | 0.383362 | 2 | 0.127980 | 0.255381 |
| /m/02sgy | Electric guitar | 6 | 0.259163 | 1 | 0.144315 | 0.114847 |
| /t/dd00004 | Female singing | 1 | 0.468460 | 2 | -0.014773 | 0.483233 |
| /m/0l14j_ | Flute | 2 | 0.400569 | 0 | NA | NA |
|  | Infant vocalization | 286 | 0.287081 | 116 | 0.112936 | 0.174145 |
| /m/05pd6 | orchestra | 2 | 0.485894 | 0 | NA | NA |
| /m/05r5c | piano | 3 | 0.235106 | 0 | NA | NA |
| /m/06rvn | Snare drum | 3 | 0.275788 | 0 | NA | NA |
| /m/09x0r | speech | 2 | 0.097830 | 0 | NA | NA |
| /m/01p970 | Tabla | 6 | 0.365539 | 0 | NA | NA |
| /m/0h9mv | tire squeal | 3 | 0.613031 | 0 | NA | NA |
| /m/0284vy3 | train horn | 2 | 0.695283 | 0 | NA | NA |
| /m/07c6l | trombone | 3 | 0.564996 | 1 | 0.104773 | 0.460223 |
| /m/07gql | trumpet | 9 | 0.500157 | 2 | 0.169491 | 0.330666 |
| /m/03m9d0z | wind | 3 | 0.513007 | 0 | NA | NA |

## CNN14 scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/028ght | applause | 0 | NA | 3 | 0.000954 | NA |
| /m/018w8 | Basketball bounce | 3 | 0.193360 | 0 | NA | NA |
| /m/018vs | bass guitar | 3 | 0.201485 | 0 | NA | NA |
| /m/01c3q | Bassoon | 0 | NA | 0 | NA | NA |
| /m/01xqw | cello | 2 | 0.244280 | 0 | NA | NA |
| /m/02hnl | Drum kit | 7 | 0.500677 | 2 | 0.003010 | 0.497667 |
| /m/02sgy | Electric guitar | 6 | 0.453438 | 1 | 0.007392 | 0.446046 |
| /t/dd00004 | Female singing | 1 | 0.612816 | 2 | 0.001255 | 0.611561 |
| /m/0l14j_ | Flute | 2 | 0.873668 | 0 | NA | NA |
| /m/05pd6 | orchestra | 2 | 0.275311 | 0 | NA | NA |
| /m/05r5c | piano | 3 | 0.230589 | 0 | NA | NA |
| /m/06rvn | Snare drum | 3 | 0.553595 | 0 | NA | NA |
| /m/09x0r | speech | 2 | 0.829265 | 0 | NA | NA |
| /m/01p970 | Tabla | 6 | 0.755913 | 0 | NA | NA |
| /m/0h9mv | tire squeal | 3 | 0.102853 | 0 | NA | NA |
| /m/0284vy3 | train horn | 2 | 0.422907 | 0 | NA | NA |
| /m/07c6l | trombone | 3 | 0.547034 | 1 | 0.000164 | 0.546870 |
| /m/07gql | trumpet | 9 | 0.561280 | 2 | 0.000667 | 0.560613 |
| /m/03m9d0z | wind | 3 | 0.093606 | 0 | NA | NA |

## Notes

- Failed downloads/inferences and unmapped CNN14 classes remain visible in result TSVs and are excluded from score means where no score exists.
- CLAPScore values are similarity scores, not calibrated probabilities.


## Results by difficulty

CLAP/CNN14 report mean scores without a classification threshold. Failed, invalid, and unmapped rows are excluded from metrics; coverage is shown explicitly.

| Model | Difficulty | Source present | Rows saved | Valid | Accuracy / mean score |
|---|---|---|---:|---:|---:|
| llm_audio_flamingo_3 | easy | yes | 220 | 220 | 0.140909 |
| llm_audio_flamingo_3 | easy | no | 14 | 14 | 0.714286 |
| llm_audio_flamingo_3 | hard | yes | 129 | 129 | 0.131783 |
| llm_audio_flamingo_3 | hard | no | 113 | 113 | 0.769912 |
| llm_gemini_3_pro_medium | easy | yes | 220 | 220 | 0.827273 |
| llm_gemini_3_pro_medium | easy | no | 14 | 14 | 0.857143 |
| llm_gemini_3_pro_medium | hard | yes | 129 | 129 | 0.790698 |
| llm_gemini_3_pro_medium | hard | no | 113 | 113 | 0.787611 |
| llm_gemini_argon_medium | easy | yes | 220 | 200 | 0.775000 |
| llm_gemini_argon_medium | easy | no | 14 | 6 | 0.500000 |
| llm_gemini_argon_medium | hard | yes | 129 | 125 | 0.688000 |
| llm_gemini_argon_medium | hard | no | 113 | 101 | 0.702970 |
| llm_gemma4_e4b_it | easy | yes | 220 | 220 | 0.236364 |
| llm_gemma4_e4b_it | easy | no | 14 | 14 | 1.000000 |
| llm_gemma4_e4b_it | hard | yes | 129 | 129 | 0.155039 |
| llm_gemma4_e4b_it | hard | no | 113 | 113 | 0.991150 |
| llm_kimi_audio_7b | easy | yes | 220 | 220 | 0.795455 |
| llm_kimi_audio_7b | easy | no | 14 | 14 | 0.642857 |
| llm_kimi_audio_7b | hard | yes | 129 | 129 | 0.759690 |
| llm_kimi_audio_7b | hard | no | 113 | 113 | 0.769912 |
| llm_qwen2_audio_7b_instruct | easy | yes | 220 | 220 | 0.986364 |
| llm_qwen2_audio_7b_instruct | easy | no | 14 | 14 | 0.285714 |
| llm_qwen2_audio_7b_instruct | hard | yes | 129 | 129 | 0.953488 |
| llm_qwen2_audio_7b_instruct | hard | no | 113 | 113 | 0.336283 |
| llm_qwen3_omni_instruct | easy | yes | 220 | 220 | 0.859091 |
| llm_qwen3_omni_instruct | easy | no | 14 | 14 | 0.714286 |
| llm_qwen3_omni_instruct | hard | yes | 129 | 129 | 0.736434 |
| llm_qwen3_omni_instruct | hard | no | 113 | 113 | 0.769912 |
| clap_scores | easy | yes | 220 | 220 | 0.339358 |
| clap_scores | easy | no | 14 | 14 | 0.167997 |
| clap_scores | hard | yes | 129 | 129 | 0.248010 |
| clap_scores | hard | no | 113 | 113 | 0.105550 |
| cnn14_scores | easy | yes | 48 | 46 | 0.508823 |
| cnn14_scores | easy | no | 0 | 0 | NA |
| cnn14_scores | hard | yes | 15 | 14 | 0.291268 |
| cnn14_scores | hard | no | 11 | 11 | 0.001844 |