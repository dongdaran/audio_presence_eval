import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

from src.common import load_samples
from src.audio import audio_path


class StemFoleyTests(unittest.TestCase):
    def test_local_labels_and_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with wave.open(str(root / "clip.wav"), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 16000)
            (root / "ontology.json").write_text(json.dumps([{"id": "/m/speech", "name": "Speech"}]))
            rows = [dict(sample_id="a", split="test", clip_id="clip", target="Speech",
                         label="contaminated", target_audible="no", source_present="no",
                         difficulty="hard", audio_path="clip.wav")]
            rows.append({**rows[0], "sample_id": "b", "target_audible": "undetermined"})
            path = root / "test.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = {"paths": {"stemfoley_csv": path, "ontology_json": root / "ontology.json"}}
            samples = load_samples(config)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample["label"], "no")
            self.assertEqual(sample["contamination_label"], "contaminated")
            self.assertEqual(sample["difficulty"], "hard")
            self.assertEqual(sample["mid"], "/m/speech")
            self.assertEqual(sample["duration_seconds"], 1)
            self.assertEqual(audio_path(sample, config), root / "clip.wav")
