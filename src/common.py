from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = path.resolve().parent.parent
    for key, value in config["paths"].items():
        config["paths"][key] = (root / value).resolve()
    config["report"]["output"] = (root / config["report"]["output"]).resolve()
    return config


def load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for label, key in (("yes", "positive_tsv"), ("no", "negative_tsv")):
        path: Path = config["paths"][key]
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["label"] = label
                row["clip_start_seconds"] = float(row["clip_start_seconds"])
                row["clip_end_seconds"] = float(row["clip_end_seconds"])
                row["duration_seconds"] = row["clip_end_seconds"] - row["clip_start_seconds"]
                if row["duration_seconds"] <= 0:
                    raise ValueError(f"Invalid clip duration for {row['clip_id']}")
                row["sample_id"] = hashlib.sha1(
                    f"{label}\t{row['clip_id']}\t{row['mid']}".encode()
                ).hexdigest()
                samples.append(row)
    if not samples:
        raise ValueError("No evaluation samples found")
    return samples


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_yes_no(text: str) -> str:
    cleaned = re.sub(r"[^a-z]+", " ", text.lower()).strip().split()
    if not cleaned:
        return "invalid"
    first = cleaned[0]
    if first in {"yes", "no"}:
        return first
    if first in {"yes.", "no."}:
        return first.rstrip(".")
    return "invalid"
