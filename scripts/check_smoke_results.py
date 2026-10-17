#!/usr/bin/env python3
"""Check actual smoke TSV outputs; runner exit code alone does not prove success."""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    'Gemini (OpenRouter)': 'smoke_llm_gemini_3_pro.tsv',
    'Gemini Argon (OpenRouter)': 'smoke_llm_gemini_argon.tsv',
    'Kimi-Audio': 'smoke_llm_kimi_audio_7b.tsv',
    'Audio Flamingo 3': 'smoke_llm_audio_flamingo_3.tsv',
    'Qwen3-Omni': 'smoke_llm_qwen3_omni_instruct.tsv',
    'CLAP': 'smoke_clap_scores.tsv',
    'CNN14': 'smoke_cnn14_scores.tsv',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=2)
    args = parser.parse_args()
    lines = [
        '# Actual model smoke validation', '',
        f'Checked: {datetime.now(timezone.utc).isoformat(timespec="seconds")}', '',
        f'{args.limit} real dataset rows per model (positive and negative). This validates execution, not model accuracy or full-dataset throughput.', '',
        '| Model | Successful rows | Result |', '|---|---:|---|',
    ]
    passed = True
    for name, filename in OUTPUTS.items():
        path = ROOT / 'results' / filename
        if not path.exists():
            lines.append(f'| {name} | 0/{args.limit} | Missing result |')
            passed = False
            continue
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        valid = []
        for row in rows:
            ok = row.get('status') == 'ok' and not row.get('error')
            if 'prediction' in row:
                ok = ok and row['prediction'] in {'yes', 'no'}
            else:
                try:
                    ok = ok and math.isfinite(float(row['score']))
                except (ValueError, KeyError):
                    ok = False
            valid.append(ok)
        ok = len(rows) == args.limit and all(valid)
        passed = passed and ok
        if ok:
            detail = ', '.join(row.get('raw_response', row.get('score', '')).strip() for row in rows)
        else:
            detail = next((row.get('error') or row.get('status', 'invalid') for row, good in zip(rows, valid) if not good), 'Unexpected row count')
        detail = ' '.join(detail.split()).replace('|', '/')
        lines.append(f'| {name} | {sum(valid)}/{args.limit} | {detail} |')
    lines += ['', 'Results: `results/smoke_*.tsv`; execution logs: `results/validation/*.log`.', '']
    report = ROOT / 'results/validation/README.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
