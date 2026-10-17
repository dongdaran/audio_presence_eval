# Actual model smoke validation

Checked: 2026-10-16T11:21:02+00:00

2 real dataset rows per model (positive and negative). This validates execution, not model accuracy or full-dataset throughput.

| Model | Successful rows | Result |
|---|---:|---|
| Gemini (OpenRouter) | 0/2 | RuntimeError: OpenRouter HTTP 402: Insufficient credits. Add more using https://openrouter.ai/settings/credits |
| Gemini Argon (OpenRouter) | 1/2 | invalid_response |
| Kimi-Audio | 2/2 | yes, YES |
| Audio Flamingo 3 | 2/2 | Yes, no |
| Qwen3-Omni | 2/2 | yes, no |
| CLAP | 2/2 | 0.1736416518688202, 0.2517961263656616 |
| CNN14 | 2/2 | 0.03194751217961311, 0.004850539844483137 |

Results: `results/smoke_*.tsv`; execution logs: `results/validation/*.log`.
