import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts.run_qwen_tad import QwenTAD, TADLogitsProcessor, combine_logits, token_sets


class TADTests(unittest.TestCase):
    def test_gate_is_per_sample_and_only_penalizes_yes_at_first_step(self):
        audio = torch.tensor([[3., 2., 8.], [6., 1., 9.]], dtype=torch.float16)
        silent = torch.tensor([[4., 2., 5.], [2., 1., 5.]])
        expected = 2 * audio.float() - silent
        actual, info = combine_logits(audio, silent, [0], [1])
        expected[0, 0] -= 2.5
        torch.testing.assert_close(actual, expected)
        self.assertEqual(info['gate'], [True, False])
        later, info = combine_logits(audio, silent, [0], [1], first_step=False)
        torch.testing.assert_close(later, 2 * audio.float() - silent)
        self.assertIsNone(info)

    def test_pooling_is_stable_and_threshold_is_strict(self):
        audio = torch.tensor([[1000., 1000., 999.]])
        # Equal margins => delta == tau == 0: do not activate gate.
        actual, info = combine_logits(audio, audio, [0, 1], [2], tau=0.)
        torch.testing.assert_close(actual, audio)
        self.assertEqual(info['gate'], [False])
        self.assertAlmostEqual(info['audio_margin'][0], 1.693147, places=4)

    def test_silent_branch_follows_actual_generated_prefix(self):
        calls = []
        def model(**inputs):
            calls.append(inputs)
            return SimpleNamespace(logits=torch.zeros(1, inputs['input_ids'].shape[1], 3))
        silent = dict(input_ids=torch.tensor([[10, 11]]),
                      attention_mask=torch.ones(1, 2, dtype=torch.long),
                      input_features=torch.zeros(1, 2, 3))
        proc = TADLogitsProcessor(model, silent, 2, [0], [1])
        proc(torch.tensor([[20, 21]]), torch.zeros(1, 3))
        proc(torch.tensor([[20, 21, 7]]), torch.zeros(1, 3))
        proc(torch.tensor([[20, 21, 7, 8]]), torch.zeros(1, 3))
        self.assertEqual(calls[0]['input_ids'].tolist(), [[10, 11]])
        self.assertEqual(calls[2]['input_ids'].tolist(), [[10, 11, 7, 8]])
        self.assertEqual(calls[2]['attention_mask'].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(silent['input_ids'].tolist(), [[10, 11]])
        self.assertFalse(calls[2]['use_cache'])

    def test_token_sets_deduplicate_and_reject_shared_fragments(self):
        class Tokenizer:
            def encode(self, form, **kwargs):
                return [1, 2] if form.strip().lower() == 'yes' else [3]
        self.assertEqual(token_sets(Tokenizer()), [[1, 2], [3]])
        with self.assertRaises(ValueError):
            token_sets(SimpleNamespace(encode=lambda *a, **k: [0]))

    def test_omni_silent_forward_restores_position_offsets_even_on_failure(self):
        class Model:
            rope_deltas = torch.tensor([[12]])
            def __call__(self, **kwargs):
                self.rope_deltas = torch.tensor([[99]])
                raise RuntimeError('forward failed')
        model = Model()
        original = model.rope_deltas
        proc = TADLogitsProcessor(model, dict(input_ids=torch.tensor([[10]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long)), 1, [0], [1])
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            proc(torch.tensor([[10]]), torch.zeros(1, 3))
        self.assertIs(model.rope_deltas, original)

    def test_omni_real_and_silent_waveforms_have_matching_inputs(self):
        import numpy as np
        from pathlib import Path
        class Processor:
            feature_extractor = SimpleNamespace(sampling_rate=16000)
            def __init__(self):
                self.waves = []
                self.conversations = []
            def apply_chat_template(self, conversation, **kwargs):
                self.conversations.append(conversation)
                return 'prompt'
            def __call__(self, **kwargs):
                self.waves.append(kwargs['audio'])
                return dict(input_ids=torch.tensor([[10, 11]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    input_features=torch.zeros(1, 2, 3))
            def batch_decode(self, generated, **kwargs):
                self.generated = generated
                return ['no']
        class Model:
            device, dtype = 'cpu', torch.float32
            def generate(self, **kwargs):
                self.processors = kwargs['logits_processor']
                self.inputs = kwargs
                return torch.tensor([[10, 11, 1]])
        obj = QwenTAD.__new__(QwenTAD)
        obj.model, obj.processor = Model(), Processor()
        obj.spec = dict(backend='qwen2_5_omni')
        obj.config = dict(prompt=dict(template='Is {caption} audible?'))
        obj.yes_ids, obj.no_ids = [0], [1]
        obj.args = SimpleNamespace(method='tad', alpha=.5, tau=.2, gamma=2.5, max_new_tokens=16)
        librosa = SimpleNamespace(load=Mock(return_value=(np.ones(20, dtype=np.float32), 16000)))
        transformers = SimpleNamespace(LogitsProcessorList=list)
        with patch.dict(sys.modules, {'librosa': librosa, 'transformers': transformers}):
            self.assertEqual(obj.predict_batch([{'caption': 'dog'}], [Path('clip.wav')]), ['no'])
        self.assertEqual(len(obj.processor.waves), 2)
        np.testing.assert_array_equal(obj.processor.waves[1][0], np.zeros(20))
        c = obj.processor.conversations[0]
        self.assertEqual(c[0]['role'], 'system')
        self.assertEqual(c[1]['content'][0], {'type': 'audio', 'audio': 'clip.wav'})
        self.assertEqual(obj.model.inputs['input_ids'].dtype, torch.long)
        self.assertEqual(obj.processor.generated.tolist(), [[1]])
        self.assertEqual(obj.model.processors[0].alpha, .5)
        obj.args.method = 'default'
        with patch.dict(sys.modules, {'librosa': librosa, 'transformers': transformers}):
            obj.predict_batch([{'caption': 'dog'}], [Path('clip.wav')])
        self.assertEqual(obj.model.processors, [])


if __name__ == '__main__':
    unittest.main()
