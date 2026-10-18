import unittest

import torch

from src.logit_decision import answer_token_ids, decide


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        word = text.strip().lower()
        return [1 if word == 'yes' else 3] if not text.startswith(' ') else [2 if word == 'yes' else 4]


class LogitDecisionTest(unittest.TestCase):
    def test_pools_answer_variants_and_records_margin(self):
        result = decide(torch.tensor([0., 1., 2., 0., 0.]), Tokenizer())
        self.assertEqual(result['prediction'], 'yes')
        self.assertEqual(result['yes_token_ids'], [1, 2])
        self.assertEqual(result['no_token_ids'], [3, 4])
        self.assertGreater(result['logit_margin'], 0)

    def test_includes_subword_ids_like_tad(self):
        class SplitTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2] if 'yes' in text.lower() else [3, 4]
        self.assertEqual(answer_token_ids(SplitTokenizer()), [[1, 2], [3, 4]])

    def test_rejects_overlapping_answer_ids(self):
        class SplitTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]
        with self.assertRaises(ValueError):
            answer_token_ids(SplitTokenizer())


if __name__ == '__main__':
    unittest.main()
