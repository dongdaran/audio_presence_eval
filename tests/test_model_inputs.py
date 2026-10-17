import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.models import LLM


class ModelInputTests(unittest.TestCase):
    def test_audio_features_match_model_dtype_and_token_ids_stay_integer(self):
        import torch
        from transformers.feature_extraction_utils import BatchFeature

        model = LLM.__new__(LLM)
        model.backend = 'audio_flamingo'
        model.spec = {}
        model.config = {'experiment': {'max_new_tokens': 1}}
        model.processor = Mock()
        model.processor.apply_chat_template.return_value = BatchFeature({
            'input_ids': torch.tensor([[1, 2]], dtype=torch.long),
            'input_features': torch.ones(1, 4, dtype=torch.float32),
        })
        model.processor.batch_decode.return_value = ['yes']

        def generate(**inputs):
            self.assertEqual(inputs['input_ids'].dtype, torch.long)
            self.assertEqual(inputs['input_features'].dtype, torch.bfloat16)
            return torch.tensor([[1, 2, 3]])

        model.model = SimpleNamespace(device=torch.device('cpu'), dtype=torch.bfloat16, generate=generate)
        self.assertEqual(model.predict_batch([{'caption': 'speech'}], [Path('test.wav')]), ['yes'])
        kwargs = model.processor.apply_chat_template.call_args.kwargs
        self.assertNotIn('padding', kwargs)
        self.assertEqual(kwargs['text_kwargs']['padding'], True)
