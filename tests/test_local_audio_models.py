import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models import LLM


class LocalAudioTests(unittest.TestCase):
    def make_model(self, backend):
        model = LLM.__new__(LLM)
        model.backend = backend
        model.spec = {'max_new_tokens': 16}
        model.config = {'experiment': {}}
        model.processor = Mock()
        model.processor.feature_extractor.sampling_rate = 16000
        model.processor.batch_decode.return_value = ['yes', 'no']
        inputs = Mock()
        inputs.to.return_value = inputs
        inputs.__getitem__ = Mock(return_value=SimpleNamespace(shape=(2, 42)))
        inputs.keys.return_value = ['input_ids']
        model.processor.return_value = inputs
        model.processor.apply_chat_template.return_value = inputs
        outputs = Mock()
        outputs.__getitem__ = Mock(return_value='generated tokens')
        model.model = SimpleNamespace(device='cuda:0', dtype='bfloat16', generate=Mock(return_value=outputs))
        return model, inputs, outputs

    def test_gemma_audio_follows_text_and_thinking_is_disabled(self):
        model, inputs, outputs = self.make_model('gemma4')
        torch = SimpleNamespace(inference_mode=nullcontext)
        with patch.dict(sys.modules, {'torch': torch}):
            result = model.predict_batch([{'caption': 'dog'}, {'caption': 'cat'}], [Path('a.wav'), Path('b.wav')])
        self.assertEqual(result, ['yes', 'no'])
        args, kwargs = model.processor.apply_chat_template.call_args
        self.assertEqual(args[0][0][0]['content'][1], {'type': 'audio', 'audio': 'a.wav'})
        self.assertEqual(args[0][0][0]['content'][0]['type'], 'text')
        self.assertFalse(kwargs['enable_thinking'])
        self.assertTrue(kwargs['text_kwargs']['padding'])
        inputs.to.assert_called_once_with(device='cuda:0', dtype='bfloat16')
        self.assertEqual(outputs.__getitem__.call_args.args[0], (slice(None), slice(42, None)))
        self.assertEqual(model.model.generate.call_args.kwargs['max_new_tokens'], 16)

    def test_qwen_resamples_to_processor_rate_and_batches_audio(self):
        model, inputs, outputs = self.make_model('qwen2_audio')
        model.processor.apply_chat_template.return_value = 'rendered prompt'
        librosa = SimpleNamespace(load=Mock(side_effect=[('wave-a', 16000), ('wave-b', 16000)]))
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=nullcontext), 'librosa': librosa}):
            result = model.predict_batch([{'caption': 'dog'}, {'caption': 'cat'}], [Path('a.wav'), Path('b.wav')])
        self.assertEqual(result, ['yes', 'no'])
        self.assertEqual(librosa.load.call_args_list[0].args, ('a.wav',))
        self.assertEqual(librosa.load.call_args_list[0].kwargs, {'sr': 16000, 'mono': True})
        model.processor.assert_called_once_with(text=['rendered prompt'] * 2, audio=['wave-a', 'wave-b'], sampling_rate=16000, padding=True, return_tensors='pt')
        model.processor.batch_decode.assert_called_once_with('generated tokens', skip_special_tokens=True)

    def test_empty_batch_does_not_load_audio(self):
        model, _, _ = self.make_model('qwen2_audio')
        self.assertEqual(model.predict_batch([], []), [])
        model.processor.assert_not_called()
