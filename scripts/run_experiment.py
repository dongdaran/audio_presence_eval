#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import audio_path
from src.common import load_samples, normalize_yes_no, read_config, write_tsv
from src.models import LLM
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run AudioSet clip-level caption presence evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.yaml")
    parser.add_argument("--task", choices=("llm", "clap", "cnn14", "report"), required=True)
    parser.add_argument("--model", help="LLM config key, e.g. qwen3_omni_instruct")
    parser.add_argument("--limit", type=int, help="Run only the first N rows (smoke test)")
    args = parser.parse_args()

    config = read_config(args.config)
    samples = load_samples(config)
    if args.limit:
        half = (args.limit + 1) // 2
        yes_samples = [sample for sample in samples if sample["label"] == "yes"][:half]
        no_samples = [sample for sample in samples if sample["label"] == "no"][:args.limit // 2]
        samples = yes_samples + no_samples
    results: Path = config["paths"]["results"]
    results.mkdir(parents=True, exist_ok=True)
    if args.task == "llm":
        if args.model not in config["models"]:
            parser.error("--model must be one of: " + ", ".join(config["models"]))
        run_llm(args.model, samples, config, results)
    elif args.task == "clap":
        from src.scoring import run_clap
        output = results / ("clap_scores.tsv" if not args.limit else "smoke_clap_scores.tsv")
        run_clap(samples, config, output)
        generate_report(results, config["report"]["output"], len(load_samples(config)), config)
    elif args.task == "cnn14":
        from src.scoring import run_cnn14
        output = results / ("cnn14_scores.tsv" if not args.limit else "smoke_cnn14_scores.tsv")
        run_cnn14(samples, config, config["paths"]["ontology_json"], output)
        generate_report(results, config["report"]["output"], len(load_samples(config)), config)
    else:
        generate_report(results, config["report"]["output"], len(load_samples(config)), config)


def run_llm(model_key, samples, config, results):
    model_spec = config["models"][model_key]
    all_samples_count = len(load_samples(config))
    smoke = len(samples) < all_samples_count
    output = results / (f"smoke_llm_{model_key}.tsv" if smoke else f"llm_{model_key}.tsv")
    existing = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            existing = {row["sample_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    model = LLM(model_key, model_spec, config)
    rows = list(existing.values())
    checkpoint = int(config["experiment"].get("batch_size", 16))
    micro_batch = max(1, int(config["experiment"].get("llm_micro_batch_size", 1)))
    if model_spec["backend"] in {"gemini", "kimi"}:
        micro_batch = 1
    done = {sample_id for sample_id, row in existing.items() if row.get("status") != "error"}
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    cursor = 0
    while cursor < len(pending):
        batch = pending[cursor:cursor + micro_batch]
        bases = [{
            "sample_id": sample["sample_id"], "split": sample["split"], "clip_id": sample["clip_id"],
            "video_id": sample["video_id"], "clip_start_seconds": sample["clip_start_seconds"],
            "clip_end_seconds": sample["clip_end_seconds"], "duration_seconds": sample["duration_seconds"],
            "mid": sample["mid"], "caption": sample["caption"], "label": sample["label"],
            "model": model_key, "model_id": model_spec["model_id"],
        } for sample in batch]
        prepared_samples, prepared_paths = [], []
        try:
            for sample, base in zip(batch, bases):
                try:
                    prepared_paths.append(audio_path(sample, config))
                    prepared_samples.append(sample)
                except Exception as exc:
                    base.update(raw_response="", prediction="invalid", status="error", error=f"{type(exc).__name__}: {exc}")
            raw_responses = model.predict_batch(prepared_samples, prepared_paths) if prepared_samples else []
            if len(raw_responses) != len(prepared_samples):
                raise RuntimeError(f"Model returned {len(raw_responses)} responses for {len(prepared_samples)} samples")
            base_by_id = {base["sample_id"]: base for base in bases}
            for sample, raw in zip(prepared_samples, raw_responses):
                base = base_by_id[sample["sample_id"]]
                prediction = normalize_yes_no(raw)
                base.update(raw_response=raw, prediction=prediction, status="ok" if prediction != "invalid" else "invalid_response", error="")
        except Exception as exc:
            if "out of memory" in str(exc).lower() and micro_batch > 1:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                micro_batch = max(1, micro_batch // 2)
                print(f"CUDA OOM; reducing LLM micro-batch to {micro_batch} and retrying", flush=True)
                continue
            traceback.print_exc()
            prepared_ids = {sample["sample_id"] for sample in prepared_samples}
            for base in bases:
                if base["sample_id"] in prepared_ids:
                    base.update(raw_response="", prediction="invalid", status="error", error=f"{type(exc).__name__}: {exc}")
        row_positions = {row["sample_id"]: index for index, row in enumerate(rows)}
        for base in bases:
            if base["sample_id"] in row_positions:
                rows[row_positions[base["sample_id"]]] = base
            else:
                rows.append(base)
        done.update(row["sample_id"] for row in bases)
        print(f"{model_key}: {len(done)}/{len(samples)}", flush=True)
        if len(rows) % checkpoint < len(bases) or len(rows) % checkpoint == 0:
            write_tsv(output, rows)
        cursor += len(batch)
    write_tsv(output, rows)
    generate_report(results, config["report"]["output"], len(load_samples(config)), config)


if __name__ == "__main__":
    main()
