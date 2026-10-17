import sys
import csv
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_samples, normalize_yes_no, read_config
from src.report import generate_report


class ExperimentCommonTests(unittest.TestCase):
    def test_normalizes_short_binary_responses(self):
        self.assertEqual(normalize_yes_no("yes"), "yes")
        self.assertEqual(normalize_yes_no("No."), "no")
        self.assertEqual(normalize_yes_no("maybe"), "invalid")

    def test_extracts_binary_answer_from_explanation(self):
        for response, expected in [
            ('The answer is **YES**. A dog is barking.', 'yes'),
            ('After listening, my answer is: no.', 'no'),
            ('```json\n{"answer": "yes"}\n```', 'yes'),
            ('Yes. YES.', 'yes'),
            ('yes or no', 'invalid'),
            ('Yesterday was noisy.', 'invalid'),
            ('', 'invalid'),
        ]:
            with self.subTest(response=response):
                self.assertEqual(normalize_yes_no(response), expected)

    def test_loads_existing_eval_pairs_with_unique_ids(self):
        config = read_config(ROOT / "config/experiment.yaml")
        samples = load_samples(config)
        self.assertEqual(len(samples), 25007)
        self.assertEqual(sum(row["label"] == "yes" for row in samples), 11745)
        self.assertEqual(sum(row["label"] == "no" for row in samples), 13262)
        self.assertEqual(len({row["sample_id"] for row in samples}), len(samples))

    def test_report_writes_confusion_counts_and_yes_no_score_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            results.mkdir()
            with (results / "llm_demo.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "prediction", "status"], delimiter="\t")
                writer.writeheader()
                writer.writerows([
                    {"label": "yes", "prediction": "yes", "status": "ok"},
                    {"label": "no", "prediction": "yes", "status": "ok"},
                ])
            with (results / "clap_scores.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "score", "mid"], delimiter="\t")
                writer.writeheader()
                writer.writerows([
                    {"label": "yes", "score": "0.8", "mid": "/m/a"},
                    {"label": "no", "score": "0.2", "mid": "/m/a"},
                ])
            output = results / "README.md"
            generate_report(results, output, 2)
            report = output.read_text(encoding="utf-8")
            self.assertIn("| 1 | 0 | 1 | 0 |", report)
            self.assertIn("| clap | yes | 1 | 1 | 0.800000", report)
            self.assertIn("| clap | no | 1 | 1 | 0.200000", report)
            self.assertIn("## CLAP scores by class", report)
            self.assertIn("| /m/a |  | 1 | 0.800000 | 1 | 0.200000 | 0.600000 |", report)


if __name__ == "__main__":
    unittest.main()
