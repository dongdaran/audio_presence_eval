"""Build results restricted to samples successfully scored by all three audio LLMs."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report import generate_report


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def sample_id(row):
    # The source Qwen TSV has a literal typo in its first column name.
    return row.get("sample_id", row.get("nsample_id"))


def main():
    source = ROOT / "results/random3000"
    output = ROOT / "results/random3000_omni_intersection"
    models = ["audio_flamingo_3", "kimi_audio_7b", "qwen3_omni_instruct"]
    successful = {}
    for model in models:
        _, rows = read(source / f"llm_{model}.tsv")
        ids = [sample_id(row) for row in rows]
        assert None not in ids and len(ids) == len(set(ids)), model
        successful[model] = {
            sample_id(row) for row in rows
            if row["status"] == "ok" and row["prediction"] in {"yes", "no"}
        }
    common = set.intersection(*successful.values())
    assert common, "Empty intersection"
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for path in sorted(source.glob("*.tsv")):
        fields, rows = read(path)
        selected = [row for row in rows if sample_id(row) in common]
        selected_ids = [sample_id(row) for row in selected]
        assert len(selected_ids) == len(set(selected_ids)), path.name
        if path.name not in {"yes_samples.tsv", "no_samples.tsv"}:
            assert set(selected_ids) == common, path.name
        # Normalize the Qwen column typo in the derived output only.
        if "nsample_id" in fields:
            fields = ["sample_id" if field == "nsample_id" else field for field in fields]
            for row in selected:
                row["sample_id"] = row.pop("nsample_id")
        with (output / path.name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
        counts[path.name] = {
            "rows": len(selected),
            "statuses": dict(Counter(row.get("status", "") for row in selected)),
        }
    metadata = {
        "source": "results/random3000",
        "criterion": "All three audio LLMs have status=ok and prediction in {yes,no}, joined by sample_id",
        "models": models,
        "successful_counts_before_intersection": {key: len(value) for key, value in successful.items()},
        "intersection_count": len(common),
        "excluded_count": 3000 - len(common),
        "files": counts,
    }
    (output / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    generate_report(output, output / "README.md", len(common))
    report = output / "README.md"
    note = (
        "# random3000 — Omni 모델 성공 샘플 교집합\n\n"
        f"원본 3,000개 중 **{len(common):,}개**를 선택했습니다. "
        f"제외: **{3000-len(common):,}개**.\n\n"
        "Audio Flamingo 3, Kimi Audio 7B, Qwen3 Omni 모두 `status=ok`이고 "
        "`prediction`이 `yes` 또는 `no`인 샘플의 `sample_id` 교집합입니다. "
        "정답 여부는 선택 조건에 포함하지 않습니다.\n\n"
        "원본은 `../random3000/`에 보존했습니다. CLAP/CNN14와 샘플 목록에도 "
        "동일한 교집합을 적용했으며, CLAP/CNN14의 실패 여부는 선택 조건이 아닙니다. "
        "Qwen TSV의 `nsample_id` 열 이름은 파생 파일에서 `sample_id`로 정정했습니다.\n\n"
        "재생성: `python3 scripts/build_random3000_intersection.py`\n\n"
    )
    report.write_text(note + report.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
