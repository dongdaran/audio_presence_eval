#!/usr/bin/env python3
"""Ten Qwen evaluations sharing model lifetime, preprocessing and prefix logits."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_qwen_tad import OMNI_SYSTEM, QwenTAD
from src.common import load_samples, normalize_yes_no, read_config, write_tsv

MODELS = ('qwen2_5_omni_7b', 'qwen2_audio_7b_instruct')
DATASETS = ('stemfoley', 'random1000_omni_intersection')


def variants(model):
    settings = {'baseline': ('default', 1.)} if model == MODELS[0] else {}
    return {**settings, 'tad05': ('tad', .5), 'tad10': ('tad', 1.)}


def output_path(dataset, model, setting, method, smoke):
    directory = 'qwen25_comparison' if model == MODELS[0] else 'qwen2_audio_comparison'
    key = f"{model.removesuffix('_instruct')}_{method}"
    return ROOT / 'results' / dataset / directory / setting / (
        f'{"smoke_" if smoke else ""}llm_{key}.tsv')


def metrics(rows):
    valid = [r for r in rows if r['prediction'] in {'yes', 'no'}]
    ratio = lambda a, b: a/b if b else None
    result = dict(saved=len(rows), valid=len(valid), invalid_failed=len(rows)-len(valid),
                  accuracy=ratio(sum(r['label']==r['prediction'] for r in valid), len(valid)))
    for label in ['yes', 'no']:
        correct = sum(r['label']==r['prediction']==label for r in valid)
        actual = sum(r['label']==label for r in valid)
        predicted = sum(r['prediction']==label for r in valid)
        result.update({f'{label}_recall': ratio(correct, actual),
                       f'{label}_precision': ratio(correct, predicted),
                       f'{label}_f1': ratio(2*correct, actual+predicted)})
    recalls = [result[f'{label}_recall'] for label in ['yes', 'no']]
    result['balanced_accuracy'] = sum(recalls)/2 if None not in recalls else None
    return result


def prepare_run(config, dataset, model, spec, args, smoke):
    samples = load_samples(config)
    if smoke:
        samples = ([s for s in samples if s['label']=='yes'][:1]
                   + [s for s in samples if s['label']=='no'][:1])
    ids = [s['sample_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs')
    states = {}
    for name, (method, alpha) in variants(model).items():
        output = output_path(dataset, model, name, method, smoke)
        metadata = dict(model_id=spec['model_id'], method=method, alpha=alpha,
                        tau=args.tau, gamma=args.gamma, max_new_tokens=args.max_new_tokens,
                        prompt=config['prompt']['template'], sample_ids=ids)
        if model == MODELS[0]:
            metadata['system_prompt'] = OMNI_SYSTEM
        manifest = output.with_suffix('.run.json')
        if manifest.exists() and json.loads(manifest.read_text()) != metadata:
            raise ValueError(f'Incompatible run metadata: {manifest}')
        if output.exists() and not manifest.exists():
            raise ValueError(f'Existing output without metadata: {output}')
        rows = {}
        if output.exists():
            with output.open() as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    if row['sample_id'] in ids:
                        if row['sample_id'] in rows:
                            raise ValueError(f'Duplicate saved sample: {output}')
                        rows[row['sample_id']] = row
        states[name] = dict(output=output, manifest=manifest, metadata=metadata, rows=rows)
    return samples, states


def evaluate(engine, samples, states, config, model, args):
    from src.audio import audio_path
    for st in states.values():
        st['output'].parent.mkdir(parents=True, exist_ok=True)
        st['manifest'].write_text(json.dumps(st['metadata'], ensure_ascii=False, indent=2)+'\n')
    def save():
        for st in states.values():
            ordered = [st['rows'][s['sample_id']] for s in samples if s['sample_id'] in st['rows']]
            write_tsv(st['output'], ordered)
            st['output'].with_suffix('.metrics.json').write_text(json.dumps(metrics(ordered), indent=2)+'\n')
    pending = [s for s in samples if any(st['rows'].get(s['sample_id'], {}).get('status') != 'ok'
                                       for st in states.values())]
    print(f'{model}: {len(pending)}/{len(samples)} pending', flush=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start+args.batch_size]
        active = {name: variants(model)[name] for name, st in states.items()
                  if any(st['rows'].get(s['sample_id'], {}).get('status') != 'ok' for s in batch)}
        try:
            responses = engine.predict_variants(batch, [audio_path(s, config) for s in batch], active)
            if any(len(responses[name]) != len(batch) for name in active):
                raise RuntimeError('Unexpected prediction batch size')
        except Exception:
            save()
            raise
        for name, (method, alpha) in active.items():
            for sample, raw in zip(batch, responses[name]):
                if states[name]['rows'].get(sample['sample_id'], {}).get('status') == 'ok':
                    continue
                prediction = normalize_yes_no(raw)
                states[name]['rows'][sample['sample_id']] = {
                    **{k: sample[k] for k in (
                        'sample_id', 'split', 'clip_id', 'video_id', 'clip_start_seconds',
                        'clip_end_seconds', 'duration_seconds', 'mid', 'caption', 'label',
                        'source_present', 'difficulty', 'contamination_label', 'local_audio_path') if k in sample},
                    'model': f"{model.removesuffix('_instruct')}_{method}",
                    'model_id': engine.spec['model_id'], 'raw_response': raw,
                    'prediction': prediction, 'status': 'ok' if prediction != 'invalid' else 'invalid_response',
                    'error': '',
                }
        if (start+len(batch)) // 32 != start // 32:
            save()
        print(f'{model}: {min(start+len(batch), len(pending))}/{len(pending)} processed', flush=True)
    save()
    if any(metrics(list(st['rows'].values()))['invalid_failed'] for st in states.values()):
        raise RuntimeError('Invalid predictions saved; inspect outputs before continuing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['all', 'smoke', 'full', 'list'], nargs='?', default='all')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--tau', type=float, default=.2)
    parser.add_argument('--gamma', type=float, default=2.5)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_new_tokens < 1 or args.gamma < 0
            or not all(math.isfinite(v) for v in (args.tau, args.gamma))):
        parser.error('Invalid batch size, token limit or gate parameters')
    if args.mode == 'list':
        for dataset in DATASETS:
            for model in MODELS:
                for name, (method, alpha) in variants(model).items():
                    print(f'{dataset}\t{model}\t{method}\talpha={alpha}\ttau={args.tau}'
                          f'\tgamma={args.gamma}\t{output_path(dataset, model, name, method, False).parent.relative_to(ROOT)}')
        return
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required; no evaluation started')
    configs = {d: read_config(ROOT / 'config' / f'{d}_local_audio.yaml') for d in DATASETS}
    phases = [True, False] if args.mode == 'all' else [args.mode == 'smoke']
    for model in MODELS:
        spec = (dict(backend='qwen2_5_omni', model_id='Qwen/Qwen2.5-Omni-7B') if model == MODELS[0]
                else dict(configs[DATASETS[0]]['models'][model]))
        runs = [(d, smoke, configs[d], *prepare_run(configs[d], d, model, spec, args, smoke))
                for smoke in phases for d in DATASETS]
        engine = None
        for dataset, smoke, config, samples, states in runs:
            print(f'Run: model={model} dataset={dataset} phase={"smoke" if smoke else "full"} '
                  f'settings={list(states)}', flush=True)
            need_work = any(st['rows'].get(s['sample_id'], {}).get('status') != 'ok'
                            for s in samples for st in states.values())
            if need_work and engine is None:
                model_args = SimpleNamespace(tau=args.tau, gamma=args.gamma,
                                             max_new_tokens=args.max_new_tokens)
                engine = QwenTAD(spec, config, model_args)
            if engine is not None:
                engine.config = config
                evaluate(engine, samples, states, config, model, args)
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
