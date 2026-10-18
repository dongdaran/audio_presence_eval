#!/usr/bin/env python3
"""Re-evaluate local audio LALMs using first-answer-token yes/no logits."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import audio_path
from src.common import load_samples, read_config, write_tsv
from src.logit_decision import decide
from src.models import LLM

OMNI_SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, '
               'Alibaba Group, capable of perceiving auditory and visual inputs, '
               'as well as generating text and speech.')


def score_qwen25(engine, sample, path):
    import torch
    inputs, _ = engine.prepare_batch([sample], [path], need_silent=False)
    with torch.inference_mode():
        logits = engine.model(**inputs, use_cache=False).logits[0, -1]
    return decide(logits, engine.processor.tokenizer)


def score_qwen3(engine, sample, path, template):
    import torch
    conversation = [{'role': 'user', 'content': [
        {'type': 'audio', 'path': str(path)},
        {'type': 'text', 'text': template.format(caption=sample['caption'])}]}]
    inputs = engine.processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt', text_kwargs={'padding': True},
    ).to(device=engine.model.device, dtype=engine.model.dtype)
    with torch.inference_mode():
        logits = engine.model(**inputs, use_cache=False).logits[0, -1]
    return decide(logits, engine.processor.tokenizer)


def score_kimi(engine, sample, path, template):
    import torch
    kimi = engine.model
    chats = [
        {'role': 'user', 'message_type': 'text',
         'content': template.format(caption=sample['caption'])},
        {'role': 'user', 'message_type': 'audio', 'content': str(path)},
    ]
    history = kimi.prompt_manager.get_prompt(chats, output_type='text')
    audio_ids, text_ids, continuous_mask, _, _ = history.to_tensor()
    device = torch.cuda.current_device()
    audio_ids = audio_ids.to(device)
    text_ids = text_ids.to(device)
    continuous_mask = continuous_mask.to(device)
    features = [feature.to(device) for feature in history.continuous_feature]
    positions = torch.arange(audio_ids.shape[1], device=device).unsqueeze(0).long()
    with torch.inference_mode():
        _, text_logits, _ = kimi.alm.forward(
            input_ids=audio_ids, text_input_ids=text_ids,
            whisper_input_feature=features, is_continuous_mask=continuous_mask,
            position_ids=positions, past_key_values=None, return_dict=False)
    return decide(text_logits[0, -1] if text_logits.ndim == 3 else text_logits[0],
                  kimi.prompt_manager.text_tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/experiment.yaml')
    parser.add_argument('--model', required=True, choices=(
        'qwen2_5_omni_7b', 'qwen3_omni_instruct', 'kimi_audio_7b'))
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Dedicated directory for logit results')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required')
    config = read_config(args.config)
    samples = load_samples(config)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        samples = ([s for s in samples if s['label'] == 'yes'][:(args.limit + 1)//2]
                   + [s for s in samples if s['label'] == 'no'][:args.limit//2])
    output = args.results_dir.resolve() / (
        f"{'smoke_' if args.limit else ''}llm_{args.model}_logits.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix('.run.json')
    model_id = ('Qwen/Qwen2.5-Omni-7B' if args.model == 'qwen2_5_omni_7b'
                else config['models'][args.model]['model_id'])
    metadata = {'model_id': model_id, 'decision': 'first_token_logsumexp',
                'prompt': config['prompt']['template'],
                'sample_ids': [s['sample_id'] for s in samples]}
    if args.model == 'qwen2_5_omni_7b':
        metadata['system_prompt'] = OMNI_SYSTEM
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        parser.error(f'Run metadata differs: {manifest}')
    if output.exists() and not manifest.exists():
        parser.error(f'Output has no run metadata: {output}')
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    rows = {}
    if output.exists():
        with output.open(newline='', encoding='utf-8') as handle:
            rows = {row['sample_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    pending = [sample for sample in samples if rows.get(sample['sample_id'], {}).get('status') != 'ok']
    print(f'{args.model}: {len(pending)} pending / {len(samples)} total; {output}', flush=True)
    if not pending:
        return
    if args.model == 'qwen2_5_omni_7b':
        from types import SimpleNamespace
        from scripts.run_qwen_tad import QwenTAD
        engine = QwenTAD({'backend': 'qwen2_5_omni', 'model_id': model_id},
                         config, SimpleNamespace())
        score = lambda sample, path: score_qwen25(engine, sample, path)
    else:
        engine = LLM(args.model, config['models'][args.model], config)
        if args.model == 'qwen3_omni_instruct':
            score = lambda sample, path: score_qwen3(engine, sample, path, config['prompt']['template'])
        else:
            score = lambda sample, path: score_kimi(engine, sample, path, config['prompt']['template'])
    for index, sample in enumerate(pending, 1):
        row = {**sample, 'model': args.model, 'model_id': model_id,
               'decision_method': 'first_token_logsumexp', 'status': 'ok', 'error': ''}
        try:
            row.update(score(sample, audio_path(sample, config)))
        except Exception as exc:
            row.update(prediction='invalid', status='error', error=f'{type(exc).__name__}: {exc}')
        rows[sample['sample_id']] = row
        write_tsv(output, [rows[s['sample_id']] for s in samples if s['sample_id'] in rows])
        print(f'{index}/{len(pending)} {sample["sample_id"]} {row["prediction"]}', flush=True)
    if any(row.get('status') != 'ok' for row in rows.values()):
        raise RuntimeError(f'Some rows failed; inspect {output}')


if __name__ == '__main__':
    main()
