"""Independent Kimi replicas with disjoint audio groups and parent-only result merging."""
from __future__ import annotations

import contextlib
import csv
import json
import multiprocessing
import os
import subprocess
import time
import traceback
from pathlib import Path

from src.common import load_samples, write_tsv
from src.report import generate_report


def partition_samples(samples, workers):
    groups = {}
    for sample in samples:
        key = (sample['video_id'], round(sample['clip_start_seconds'], 3), round(sample['clip_end_seconds'], 3))
        groups.setdefault(key, []).append(sample)
    shards = [[] for _ in range(min(workers, len(groups))) ]
    for group in sorted(groups.values(), key=len, reverse=True):
        min(shards, key=len).extend(group)
    return shards


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def merge_rows(samples, original, shards):
    by_id = {row['sample_id']: row for row in original}
    for rows in shards:
        for row in rows:
            previous = by_id.get(row['sample_id'])
            if previous is None or previous.get('status') != 'ok' or row.get('status') == 'ok':
                by_id[row['sample_id']] = row
    return [by_id[s['sample_id']] for s in samples if s['sample_id'] in by_id]


def _worker(index, model_key, samples, config, output, seed, barrier):
    output = Path(output)
    metrics = {'worker': index, 'assigned': len(samples)}
    with output.with_suffix('.log').open('w', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import torch
                from src.models import LLM
                from scripts.run_experiment import run_llm

                torch.set_num_threads(2)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                model = LLM(model_key, config['models'][model_key], config)
                torch.cuda.synchronize()
                metrics['load_seconds'] = time.perf_counter() - started
                # All replicas must fit simultaneously before inference starts.
                barrier.wait(timeout=180)
                started = time.perf_counter()
                rows = run_llm(model_key, samples, config, config['paths']['results'],
                               output_path=output, make_report=False, initial_rows=seed, model=model)
                torch.cuda.synchronize()
                metrics['inference_seconds'] = time.perf_counter() - started
                metrics['successful'] = sum(row.get('status') == 'ok' for row in rows)
                metrics['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                metrics['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
            except BaseException as exc:
                metrics['error'] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
                barrier.abort()
                raise
            finally:
                output.with_suffix('.json').write_text(json.dumps(metrics, indent=2))


def _gpu_memory_mib():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True)
        return max(int(value) for value in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def run_parallel_kimi(model_key, samples, config, results, workers, limit=None):
    import fcntl

    output = results / (f'smoke_llm_{model_key}.tsv' if limit else f'llm_{model_key}.tsv')
    workdir = results / '.workers' / model_key / (f'smoke-{limit}' if limit else 'full')
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another parallel run is using {output}') from None
        original = read_rows(output)
        done = {row['sample_id'] for row in original if row.get('status') == 'ok'}
        pending = [sample for sample in samples if sample['sample_id'] not in done]
        shards = partition_samples(pending, workers)
        if not shards:
            print(f'{model_key}: all {len(samples)} rows already complete', flush=True)
            return
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(len(shards))
        processes, paths = [], []
        started = time.perf_counter()
        peak_mib = _gpu_memory_mib()
        try:
            for index, shard in enumerate(shards):
                path = workdir / f'worker-{index}.tsv'
                paths.append(path)
                process = context.Process(target=_worker, args=(index, model_key, shard, config, path, original, barrier))
                process.start()
                processes.append(process)
                print(f'Kimi worker {index}: {len(shard)} rows; log: {path.with_suffix(".log")}', flush=True)
            while any(process.is_alive() for process in processes):
                peak_mib = max(peak_mib, _gpu_memory_mib())
                time.sleep(1)
            for process in processes:
                process.join()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            rows = merge_rows(samples, original, [read_rows(path) for path in paths])
            # Only the parent writes the canonical TSV and report.
            temporary = output.with_suffix('.tsv.tmp')
            write_tsv(temporary, rows)
            if temporary.exists():
                temporary.replace(output)
            generate_report(results, config['report']['output'], len(load_samples(config)), config)
        metrics = {
            'workers': len(shards), 'requested_workers': workers,
            'wall_seconds': time.perf_counter() - started,
            'peak_gpu_used_mib': peak_mib, 'rows': len(rows),
            'successful': sum(row.get('status') == 'ok' for row in rows),
            'worker_metrics': [json.loads(path.with_suffix('.json').read_text())
                               for path in paths if path.with_suffix('.json').exists()],
        }
        (workdir / 'summary.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
        if any(process.exitcode != 0 for process in processes) or len(rows) != len(samples):
            raise RuntimeError(f'Kimi worker failed; checkpoints preserved under {workdir}')
        if any(row.get('status') != 'ok' for row in rows):
            raise RuntimeError(f'Kimi completed with failed/invalid rows; inspect {output}')
