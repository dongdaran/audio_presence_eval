#!/usr/bin/env python3
"""Qwen2-Audio / Qwen2.5-Omni AAD/TAD evaluation, isolated from existing predictions.

References (implementation written against this project's Transformers API):
https://github.com/GillbertHsu/Audio-Aware-Decoding/blob/main/inference_and_eval_qwen.py
https://arxiv.org/html/2609.07286v1 (equations 1--9)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OMNI_SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, '
               'Alibaba Group, capable of perceiving auditory and visual inputs, '
               'as well as generating text and speech.')


def token_sets(tokenizer):
    """Union all subword IDs, as specified in TAD section 2.2."""
    groups = []
    for word in ("yes", "no"):
        forms = [prefix + variant for prefix in ("", " ")
                 for variant in (word, word.title(), word.upper())]
        groups.append(sorted({i for form in forms
                              for i in tokenizer.encode(form, add_special_tokens=False)}))
    if not all(groups) or set(groups[0]) & set(groups[1]):
        raise ValueError("Yes/No token sets must be nonempty and disjoint")
    return groups


def combine_logits(audio, silent, yes_ids, no_ids, *, alpha=1.0,
                   tau=0.2, gamma=2.5, first_step=True):
    """Full-vocabulary AAD; TAD gates only affirmative tokens at step one."""
    import torch
    audio, silent = audio.float(), silent.float()
    combined = (1 + alpha) * audio - alpha * silent
    diagnostics = None
    if first_step:
        pool = lambda logits, ids: torch.logsumexp(logits[:, ids], dim=-1)
        audio_margin = pool(audio, yes_ids) - pool(audio, no_ids)
        silent_margin = pool(silent, yes_ids) - pool(silent, no_ids)
        delta = audio_margin - silent_margin
        gate = delta < tau
        combined[:, yes_ids] -= gamma * gate[:, None].to(combined.dtype)
        diagnostics = dict(audio_margin=audio_margin.tolist(),
                           silent_margin=silent_margin.tolist(),
                           delta=delta.tolist(), gate=gate.tolist())
    return combined, diagnostics


class TADLogitsProcessor:
    """AAD-style callable processor with a same-length silent waveform branch.

    The silent branch sees the SAME generated tokens as the real branch.
    Recomputing its full prefix avoids version-specific Qwen cache handling;
    it costs an extra full forward pass per generated token.
    """
    def __init__(self, model, silent_inputs, prompt_length, yes_ids, no_ids,
                 *, alpha=1.0, tau=0.2, gamma=2.5):
        self.model = model
        self.silent_inputs = silent_inputs
        self.prompt_length = prompt_length
        self.yes_ids, self.no_ids = yes_ids, no_ids
        self.alpha, self.tau, self.gamma = alpha, tau, gamma
        self.diagnostics = None

    def __call__(self, input_ids, scores):
        import torch
        if input_ids.shape[0] != self.silent_inputs['input_ids'].shape[0]:
            raise ValueError("Beam expansion is unsupported; use num_beams=1")
        suffix = input_ids[:, self.prompt_length:]
        silent = dict(self.silent_inputs)
        silent['input_ids'] = torch.cat([silent['input_ids'], suffix], dim=1)
        silent['attention_mask'] = torch.cat([
            silent['attention_mask'], torch.ones_like(suffix)], dim=1)
        # Omni stores multimodal position offsets on the model. The auxiliary
        # forward must not replace the real generation branch's offsets.
        has_rope = hasattr(self.model, 'rope_deltas')
        real_rope = getattr(self.model, 'rope_deltas', None)
        try:
            if has_rope:
                self.model.rope_deltas = None
            with torch.inference_mode():
                reference = self.model(**silent, use_cache=False).logits[:, -1, :]
        finally:
            if has_rope:
                self.model.rope_deltas = real_rope
        combined, info = combine_logits(
            scores, reference, self.yes_ids, self.no_ids,
            alpha=self.alpha, tau=self.tau, gamma=self.gamma,
            first_step=suffix.shape[1] == 0)
        if info is not None:
            self.diagnostics = info
        return combined


class QwenTAD:
    def __init__(self, spec, config, args):
        if spec['backend'] == 'qwen2_5_omni':
            import torch
            from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.processor = Qwen2_5OmniProcessor.from_pretrained(spec['model_id'])
            self.processor.tokenizer.padding_side = 'left'
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                spec['model_id'], torch_dtype=dtype, device_map='auto').eval()
        else:
            from src.models import LLM
            loaded = LLM('qwen2_audio_7b_instruct', spec, config)
            self.model, self.processor = loaded.model, loaded.processor
        self.spec, self.config, self.args = spec, config, args
        self.yes_ids, self.no_ids = token_sets(self.processor.tokenizer)

    def predict_batch(self, samples, audio_files):
        import librosa
        import numpy as np
        import torch
        from transformers import LogitsProcessorList
        rate = self.processor.feature_extractor.sampling_rate
        template = self.config['prompt']['template']
        is_omni = self.spec['backend'] == 'qwen2_5_omni'
        conversations = [[{'role': 'user', 'content': [
            {'type': 'audio', ('audio' if is_omni else 'audio_url'): str(path)},
            {'type': 'text', 'text': template.format(caption=sample['caption'])},
        ]}] for sample, path in zip(samples, audio_files)]
        if is_omni:
            conversations = [[{'role': 'system', 'content': [
                {'type': 'text', 'text': OMNI_SYSTEM}]}] + c for c in conversations]
        texts = [self.processor.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True) for c in conversations]
        waves = [librosa.load(str(path), sr=rate, mono=True)[0] for path in audio_files]
        def prepare(audios):
            inputs = self.processor(text=texts, audio=audios, sampling_rate=rate,
                                    padding=True, return_tensors='pt')
            # Keep token IDs and masks integer; cast only floating tensors.
            return {k: v.to(device=self.model.device,
                            dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                    for k, v in inputs.items()}
        inputs = prepare(waves)
        processors = LogitsProcessorList()
        if self.args.method != 'default':
            silent = prepare([np.zeros_like(wave) for wave in waves])
            if not torch.equal(inputs['input_ids'], silent['input_ids']):
                raise ValueError('Real/silent audio token layouts differ')
            processors.append(TADLogitsProcessor(
                self.model, silent, inputs['input_ids'].shape[1],
                self.yes_ids, self.no_ids, alpha=self.args.alpha,
                tau=self.args.tau, gamma=self.args.gamma if self.args.method == 'tad' else 0.0))
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, logits_processor=processors, do_sample=False,
                num_beams=1, repetition_penalty=1.0,
                max_new_tokens=self.args.max_new_tokens)
        return self.processor.batch_decode(
            output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen2_audio_7b_instruct', 'qwen2_5_omni_7b'],
                        default='qwen2_audio_7b_instruct')
    parser.add_argument('--method', choices=['default', 'aad', 'tad'], default='tad')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=2.5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--prompt', choices=['config', 'paper'], default='config')
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in [args.alpha, args.tau, args.gamma])
            or args.alpha < 0 or args.gamma < 0 or args.batch_size < 1
            or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1)):
        parser.error('Require finite parameters, alpha/gamma >= 0, positive sizes/limit')
    from src.common import load_samples, read_config
    from scripts.run_experiment import run_llm
    config = read_config(args.config)
    if args.model == 'qwen2_5_omni_7b':
        spec = dict(backend='qwen2_5_omni', model_id='Qwen/Qwen2.5-Omni-7B')
    else:
        spec = dict(config['models'][args.model])
        if spec['backend'] != 'qwen2_audio':
            parser.error('Expected the qwen2_audio backend')
    import torch
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required; no evaluation has been started')
    if args.prompt == 'paper':
        config['prompt']['template'] = (
            'Focus on the given audio and answer the following question. '
            "Answer in the format: 'Yes, …' or 'No, …', and always start with 'Yes' or 'No'. "
            'Does this audio contain the sound of {caption}?')
    key = f"{args.model.removesuffix('_instruct')}_{args.method}"
    spec['llm_micro_batch_size'] = args.batch_size
    config['models'][key] = spec
    config['experiment']['batch_size'] = args.batch_size
    samples = load_samples(config)
    if args.limit:
        samples = ([s for s in samples if s['label'] == 'yes'][:(args.limit + 1)//2]
                   + [s for s in samples if s['label'] == 'no'][:args.limit//2])
    results = (args.results_dir or config['paths']['results'] / 'tad').resolve()
    results.mkdir(parents=True, exist_ok=True)
    output = results / (f'smoke_llm_{key}.tsv' if args.limit else f'llm_{key}.tsv')
    metadata = dict(model_id=spec['model_id'], method=args.method, alpha=args.alpha,
                    tau=args.tau, gamma=args.gamma, max_new_tokens=args.max_new_tokens,
                    prompt=config['prompt']['template'],
                    sample_ids=[s['sample_id'] for s in samples])
    if args.model == 'qwen2_5_omni_7b':
        metadata['system_prompt'] = OMNI_SYSTEM
    manifest = output.with_suffix('.run.json')
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        parser.error('Run parameters changed; choose another --results-dir')
    if output.exists() and not manifest.exists():
        parser.error('Output without run metadata exists; choose another --results-dir')
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    model = QwenTAD(spec, config, args)
    rows = run_llm(key, samples, config, results, output_path=output,
                   make_report=False, model=model)
    valid = [r for r in rows if r['prediction'] in {'yes', 'no'}]
    summary = dict(saved=len(rows), valid=len(valid), invalid_failed=len(rows)-len(valid))
    def divide(a, b):
        return a / b if b else None
    summary['accuracy'] = divide(sum(r['prediction']==r['label'] for r in valid), len(valid))
    for label in ['yes', 'no']:
        correct = sum(r['label']==label and r['prediction']==label for r in valid)
        actual = sum(r['label']==label for r in valid)
        predicted = sum(r['prediction']==label for r in valid)
        summary[f'{label}_recall'] = divide(correct, actual)
        summary[f'{label}_precision'] = divide(correct, predicted)
        summary[f'{label}_f1'] = divide(2*correct, actual+predicted)
    recalls = [summary[f'{label}_recall'] for label in ['yes', 'no']]
    summary['balanced_accuracy'] = sum(recalls)/2 if None not in recalls else None
    output.with_suffix('.metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if summary['invalid_failed']:
        raise SystemExit('Evaluation contains invalid/failed rows; inspect the TSV before continuing')


if __name__ == '__main__':
    main()
