import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_samples, read_config
from src.scoring import run_clap, run_cnn14


class ScoreCheckpointTests(unittest.TestCase):
    def test_clap_interruption_keeps_completed_batch_and_resumes(self):
        config = read_config(ROOT / 'config/experiment.yaml')
        config['scoring']['clap']['batch_size'] = 1
        samples = load_samples(config)[:3]
        model = Mock()
        model.get_audio_embedding_from_filelist.side_effect = [
            np.array([[1., 0.]]), RuntimeError('interrupted')]
        model.get_text_embedding.return_value = np.array([[1., 0.]])
        modules = {
            'laion_clap': SimpleNamespace(CLAP_Module=Mock(return_value=model)),
            'huggingface_hub': SimpleNamespace(hf_hub_download=Mock(return_value='checkpoint')),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clap_scores.tsv'
            with patch.dict(sys.modules, modules), patch('src.audio.audio_path', return_value=Path('audio.wav')), patch('src.scoring._has_cuda', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_clap(samples, config, output)
                with output.open() as handle:
                    saved = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual([r['sample_id'] for r in saved], [samples[0]['sample_id']])
                model.get_audio_embedding_from_filelist.side_effect = None
                model.get_audio_embedding_from_filelist.return_value = np.array([[1., 0.]])
                model.get_audio_embedding_from_filelist.reset_mock()
                run_clap(samples, config, output)
                self.assertEqual(model.get_audio_embedding_from_filelist.call_count, 2)
                with output.open() as handle:
                    saved = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual(len(saved), 3)
                self.assertEqual(len({r['sample_id'] for r in saved}), 3)
                # Completed score runs return before loading dependencies/models.
                with patch.dict(sys.modules, {'laion_clap': None, 'panns_inference': None}):
                    run_clap(samples, config, output)
                    run_cnn14(samples, config, ROOT / 'data/ontology.json', output)


if __name__ == '__main__':
    unittest.main()
