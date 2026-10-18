import unittest

import numpy as np

from scripts.run_qwen25_gain_logits import gain_waveform, scores, waveform_stats


class GainLogitTests(unittest.TestCase):
    def test_gain_keeps_values_beyond_pcm_range(self):
        original = np.array([0.0, 0.8, -0.8, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(gain_waveform(original, 0), original)
        gained = gain_waveform(original, 5)
        self.assertGreater(gained[1], 1.0)
        self.assertLess(gained[2], -1.0)
        self.assertEqual(waveform_stats(gained)['over_full_scale_count'], 2)
        self.assertEqual(waveform_stats(gained)['over_full_scale_fraction'], 0.5)

    def test_summary_counts_paired_corrections(self):
        rows = []
        for gain, predictions in ((0, ('no', 'no')), (5, ('yes', 'yes'))):
            for sample_id, label, prediction in zip(('a', 'b'), ('yes', 'no'), predictions):
                rows.append(dict(sample_id=sample_id, gain_db=gain, label=label,
                                 prediction=prediction, status='ok',
                                 over_full_scale_fraction=0.0))
        summary = scores(rows, ['a', 'b'])
        self.assertEqual(summary['5']['prediction_changed'], 2)
        self.assertEqual(summary['5']['wrong_to_correct'], 1)
        self.assertEqual(summary['5']['correct_to_wrong'], 1)
        self.assertEqual(summary['5']['accuracy'], 0.5)
        self.assertEqual(summary['5']['macro_f1'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
