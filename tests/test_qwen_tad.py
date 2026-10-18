import unittest
from types import SimpleNamespace

import torch

from scripts.run_qwen_tad import TADLogitsProcessor, combine_logits, token_sets


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


if __name__ == '__main__':
    unittest.main()
