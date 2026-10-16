from __future__ import annotations

import csv
import math
from pathlib import Path


def _read(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def generate_report(results: Path, out: Path, sample_count: int, config=None):
    llm_files = sorted(results.glob("llm_*.tsv"))
    score_files = [results / "clap_scores.tsv", results / "cnn14_scores.tsv"]
    lines = [
        "# AudioSet caption presence evaluation results", "",
        f"Input examples: **{sample_count:,}** (positive and complete-frame-negative evaluation clip/class pairs).", "",
        "## LLM yes/no predictions", "",
        "Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.", "",
        "| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in llm_files:
        rows = _read(path)
        scored = [r for r in rows if r.get("prediction") in {"yes", "no"}]
        tp = sum(r["label"] == "yes" and r["prediction"] == "yes" for r in scored)
        fn = sum(r["label"] == "yes" and r["prediction"] == "no" for r in scored)
        fp = sum(r["label"] == "no" and r["prediction"] == "yes" for r in scored)
        tn = sum(r["label"] == "no" and r["prediction"] == "no" for r in scored)
        n = len(scored)
        accuracy = (tp + tn) / n if n else math.nan
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall else math.nan
        fmt = lambda value: f"{value:.4f}" if math.isfinite(value) else "NA"
        lines.append(f"| {path.stem[4:]} | {n} | {len(rows)-n} | {fmt(accuracy)} | {fmt(precision)} | {fmt(recall)} | {fmt(f1)} | {tp} | {fn} | {fp} | {tn} |")
    if config:
        lines += ["", "## Configured model IDs", ""]
        for key, spec in config["models"].items():
            tokens = spec.get("max_new_tokens", config["experiment"].get("max_new_tokens", 1))
            lines.append(f"- `{key}`: `{spec['model_id']}` (max new tokens: `{tokens}`)")
        lines += [
            "", f"- Maximum new tokens: `{config['experiment'].get('max_new_tokens', 1)}`",
            f"- Score batch size: `{config['experiment'].get('batch_size', 16)}`",
            f"- LLM Transformers micro-batch size: `{config['experiment'].get('llm_micro_batch_size', 1)}`",
        ]
        if config.get("hardware"):
            lines.append(f"- Requested GPU: `{config['hardware'].get('gpu', 'unspecified')}`")
    lines += ["", "## CLAPScore / CNN14 mean scores by reference label", "", "CLAPScore is audio/text cosine similarity. CNN14 is the AudioSet class score for the mapped MID. Neither score is thresholded into a yes/no prediction.", "", "Pooled mean is the mean over all scored samples; pooled SD is their standard deviation. Per-MID macro mean averages each MID's mean equally. Class-level tables below show yes/no means and their difference for each MID.", "", "| Score | Reference label | N | MIDs scored | Pooled mean | Pooled SD | Per-MID macro mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for path in score_files:
        for label in ("yes", "no"):
            rows = [r for r in _read(path) if r.get("label") == label and _finite(r.get("score"))]
            values = [float(r["score"]) for r in rows]
            mean = sum(values) / len(values) if values else math.nan
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else math.nan
            by_caption = {}
            for row in rows:
                by_caption.setdefault(row["mid"], []).append(float(row["score"]))
            macro = sum(sum(vs) / len(vs) for vs in by_caption.values()) / len(by_caption) if by_caption else math.nan
            lines.append(f"| {path.stem.replace('_scores', '')} | {label} | {len(values)} | {len(by_caption)} | {mean:.6f} | {std:.6f} | {macro:.6f} |" if values else f"| {path.stem.replace('_scores', '')} | {label} | 0 | 0 | NA | NA | NA |")
    for path in score_files:
        score_name = path.stem.replace("_scores", "")
        rows = _read(path)
        grouped = {}
        for row in rows:
            mid = row.get("mid", "")
            item = grouped.setdefault(mid, {"caption": row.get("caption", ""), "yes": [], "no": []})
            if row.get("label") in {"yes", "no"} and _finite(row.get("score")):
                item[row["label"]].append(float(row["score"]))
        lines += ["", f"## {score_name.upper()} scores by class", "", "| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |", "|---|---|---:|---:|---:|---:|---:|"]
        for mid, item in sorted(grouped.items(), key=lambda pair: (pair[1]["caption"].casefold(), pair[0])):
            yes, no = item["yes"], item["no"]
            yes_mean = sum(yes) / len(yes) if yes else math.nan
            no_mean = sum(no) / len(no) if no else math.nan
            delta = yes_mean - no_mean if yes and no else math.nan
            fmt = lambda value: f"{value:.6f}" if math.isfinite(value) else "NA"
            lines.append(f"| {mid} | {item['caption']} | {len(yes)} | {fmt(yes_mean)} | {len(no)} | {fmt(no_mean)} | {fmt(delta)} |")
    lines += ["", "## Notes", "", "- `yes`/`no` are the reference labels from the existing evaluation TSVs; each row is one clip/class pair.", "- Sample duration follows the source TSV. Some positive clips are shorter than 10 seconds; no padding is applied to model input audio.", "- Failed downloads/inferences and unmapped CNN14 classes remain visible in result TSVs and are excluded from score means where no score exists.", "- CLAPScore values are similarity scores, not calibrated probabilities.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False
