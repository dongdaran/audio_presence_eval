"""Local StemFoley test set adapter for the existing model runners."""
from __future__ import annotations

import csv
import json
import wave


def load_stemfoley(config):
    path = config["paths"]["stemfoley_csv"]
    ontology = json.loads(config["paths"]["ontology_json"].read_text())
    names = {item["name"].casefold(): item["id"] for item in ontology}
    samples, seen = [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            audible, contamination = row["target_audible"], row["label"]
            if audible not in {"yes", "no"} or contamination not in {"clean", "contaminated"}:
                continue
            difficulty = "hard" if contamination == "contaminated" else "easy"
            if row["source_present"] != audible or row["difficulty"] != difficulty:
                raise ValueError(f"Inconsistent labels: {row['sample_id']}")
            if row["sample_id"] in seen:
                raise ValueError(f"Duplicate sample: {row['sample_id']}")
            seen.add(row["sample_id"])
            audio = (path.parent / row["audio_path"]).resolve()
            with wave.open(str(audio), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if duration <= 0:
                raise ValueError(f"Empty audio: {audio}")
            samples.append({
                **row, "label": audible, "source_present": audible,
                "contamination_label": contamination, "difficulty": difficulty,
                "caption": row["target"],
                "mid": names.get(row["target"].casefold(), ""),
                "video_id": "", "clip_start_seconds": 0.0,
                "clip_end_seconds": duration, "duration_seconds": duration,
                "local_audio_path": str(audio),
            })
    if not samples:
        raise ValueError("No eligible StemFoley test samples")
    return samples
