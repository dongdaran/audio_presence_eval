#!/usr/bin/env python3
"""Evaluate Qwen2.5-Omni first-token Yes/No logits at seven waveform gains."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen_tad import OMNI_SYSTEM, QwenTAD
from src.audio import audio_path
from src.common import load_samples, read_config, write_tsv
from src.logit_decision import decide

GAINS_DB = (-30, -20, -10, -5, 0, 5, 10)
MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'


def gain_waveform(waveform, gain_db):
    """Keep float values outside PCM's [-1, 1] range without clipping."""
    import numpy as np
    gained = np.asarray(waveform, dtype=np.float32) * np.float32(10 ** (gain_db / 20))
    if not np.isfinite(gained).all():
        raise ValueError('Non-finite waveform after gain')
    return gained


def waveform_stats(waveform):
    import numpy as np
    if not len(waveform):
        raise ValueError('Empty audio waveform')
    magnitude = np.abs(waveform)
    return {'peak_abs': float(magnitude.max()),
            'over_full_scale_count': int(np.count_nonzero(magnitude > 1)),
            'over_full_scale_fraction': float(np.mean(magnitude > 1))}


def scores(rows, sample_ids):
    by_key = {(r['sample_id'], int(r['gain_db'])): r for r in rows}
    result = {}
    for gain in GAINS_DB:
        group = [by_key[(sample_id, gain)] for sample_id in sample_ids
                 if (sample_id, gain) in by_key]
        valid = [r for r in group if r.get('status') == 'ok'
                 and r.get('prediction') in ('yes', 'no')]
        count = len(valid)
        accuracy = sum(r['prediction'] == r['label'] for r in valid) / count if count else None
        f1s = []
        for label in ('yes', 'no'):
            tp = sum(r['label'] == label and r['prediction'] == label for r in valid)
            fp = sum(r['label'] != label and r['prediction'] == label for r in valid)
            fn = sum(r['label'] == label and r['prediction'] != label for r in valid)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        compared = [(by_key[(sample_id, 0)], by_key[(sample_id, gain)])
                    for sample_id in sample_ids
                    if (sample_id, 0) in by_key and (sample_id, gain) in by_key
                    and by_key[(sample_id, 0)].get('status') == 'ok'
                    and by_key[(sample_id, gain)].get('status') == 'ok']
        result[str(gain)] = {
            'saved': len(group), 'valid': count, 'failed': len(group) - count,
            'accuracy': accuracy, 'macro_f1': sum(f1s) / 2 if count else None,
            'paired_with_0db': len(compared),
            'prediction_changed': sum(a['prediction'] != b['prediction'] for a, b in compared),
            'wrong_to_correct': sum(a['prediction'] != a['label'] and b['prediction'] == b['label']
                                    for a, b in compared),
            'correct_to_wrong': sum(a['prediction'] == a['label'] and b['prediction'] != b['label']
                                    for a, b in compared),
            'samples_over_full_scale': sum(float(r['over_full_scale_fraction']) > 0 for r in valid),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/stemfoley.yaml')
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT / 'results/stemfoley/qwen25_gain_logits')
    parser.add_argument('--limit', type=int, help='Balanced yes/no smoke sample count')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required; no evaluation started')
    import librosa
    config = read_config(args.config)
    samples = load_samples(config)
    if args.limit:
        samples = ([s for s in samples if s['label'] == 'yes'][:(args.limit + 1) // 2]
                   + [s for s in samples if s['label'] == 'no'][:args.limit // 2])
    sample_ids = [s['sample_id'] for s in samples]
    if len(sample_ids) != len(set(sample_ids)):
        parser.error('Duplicate sample IDs')
    output_dir = args.results_dir.resolve()
    output = output_dir / ('smoke_qwen25_gain_logits.tsv' if args.limit else 'qwen25_gain_logits.tsv')
    manifest = output.with_suffix('.run.json')
    summary = output.with_suffix('.summary.json')
    metadata = {'model_id': MODEL_ID, 'decision': 'first_token_logsumexp',
                'gain_db': list(GAINS_DB), 'gain_domain': 'float_waveform_no_clipping',
                'prompt': config['prompt']['template'], 'system_prompt': OMNI_SYSTEM,
                'sample_ids': sample_ids}
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        parser.error(f'Run metadata differs: {manifest}')
    if output.exists() and not manifest.exists():
        parser.error(f'Output has no run metadata: {output}')
    rows = {}
    if output.exists():
        with output.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                key = (row['sample_id'], int(row['gain_db']))
                if key in rows:
                    parser.error(f'Duplicate saved row: {key}')
                if key[0] in sample_ids and key[1] in GAINS_DB:
                    rows[key] = row
    pending = [(sample, gain) for sample in samples for gain in GAINS_DB
               if rows.get((sample['sample_id'], gain), {}).get('status') != 'ok']
    print(f'Qwen2.5-Omni gain logits: {len(pending)}/{len(samples) * len(GAINS_DB)} pending', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    if pending:
        engine = QwenTAD({'backend': 'qwen2_5_omni', 'model_id': MODEL_ID},
                         config, SimpleNamespace())
        rate = engine.processor.feature_extractor.sampling_rate
        pending_ids = {sample['sample_id'] for sample, _ in pending}
        completed = 0
        for sample in samples:
            if sample['sample_id'] not in pending_ids:
                continue
            try:
                path = audio_path(sample, config)
                original = librosa.load(str(path), sr=rate, mono=True)[0]
                if not len(original):
                    raise ValueError('Empty audio waveform')
            except Exception as exc:
                original = None
                load_error = f'{type(exc).__name__}: {exc}'
            for gain in GAINS_DB:
                key = (sample['sample_id'], gain)
                if rows.get(key, {}).get('status') == 'ok':
                    continue
                row = {'sample_id': sample['sample_id'], 'split': sample['split'],
                       'clip_id': sample['clip_id'], 'caption': sample['caption'],
                       'label': sample['label'], 'gain_db': gain, 'model_id': MODEL_ID,
                       'decision_method': 'first_token_logsumexp', 'status': 'ok', 'error': ''}
                try:
                    if original is None:
                        raise RuntimeError(load_error)
                    gained = gain_waveform(original, gain)
                    row.update(waveform_stats(gained))
                    inputs, _ = engine.prepare_batch([sample], [path], need_silent=False,
                                                     waveforms=[gained])
                    with torch.inference_mode():
                        logits = engine.model(**inputs, use_cache=False).logits[0, -1]
                    row.update(decide(logits, engine.processor.tokenizer))
                except Exception as exc:
                    row.update(prediction='invalid', status='error',
                               error=f'{type(exc).__name__}: {exc}')
                rows[key] = row
                completed += 1
                if completed % 16 == 0:
                    save(output, summary, rows, sample_ids)
            print(f'{sample["sample_id"]}: {completed}/{len(pending)}', flush=True)
    save(output, summary, rows, sample_ids)
    if any(rows.get((sample_id, gain), {}).get('status') != 'ok'
           for sample_id in sample_ids for gain in GAINS_DB):
        raise RuntimeError(f'Some rows failed; inspect {output}')
    print(f'Completed: {output}\nSummary: {summary}', flush=True)


def save(output, summary, rows, sample_ids):
    ordered = [rows[(sample_id, gain)] for sample_id in sample_ids for gain in GAINS_DB
               if (sample_id, gain) in rows]
    write_tsv(output, ordered)
    summary.write_text(json.dumps(scores(ordered, sample_ids), indent=2) + '\n')


if __name__ == '__main__':
    main()
