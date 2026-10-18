# Qwen2.5-Omni: absolute RMS −65 dBFS

All 476 StemFoley test clips completed successfully (0 failures).

Each 16 kHz mono float waveform is scaled using one factor over the entire clip:

```text
original_rms = sqrt(mean(waveform ** 2))
scale = 10 ** (-65 / 20) / original_rms
input_waveform = waveform * scale
```

This sets whole-clip RMS to −65 dBFS; it does not apply a fixed −65 dB gain or zero low-energy frames. Float amplitudes are preserved without PCM requantization or clipping. Source labels are unchanged.

Model: Qwen/Qwen2.5-Omni-7B (Thinker). Decision: first answer-token yes/no token-set logsumexp, identical prompt/system prompt to the original gain sweep. Exact prompts and sample IDs are in target_rms_logits.run.json. Batch started at 64 and fell back to 32 on CUDA OOM. Hardware: RTX 3090 24 GB.

| Condition | Accuracy | Macro F1 |
|---|---:|---:|
| Original (0 dB gain) | 72.90% | 0.7103 |
| Absolute RMS −65 dBFS | 66.18% | 0.6501 |

| Transition vs original | Samples |
|---|---:|
| Correct → wrong | 46 |
| Wrong → correct | 14 |
| Correct → correct | 301 |
| Wrong → wrong | 115 |

43 of the 46 correct→wrong samples changed from yes to no (ground truth yes); 3 changed from no to yes (ground truth no).

Files: target_rms_logits.tsv contains per-sample logits, original/actual RMS and applied gain; target_rms_logits.summary.json contains aggregate metrics; comparison_with_original.tsv and the two transition-specific TSVs contain paired comparisons.
