import functools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts.run_qwen_tad import QwenTAD
from scripts import run_qwen_shared as runner


class SharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_omni_position_state_matches_independent_generation(self):
        from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
        config = Qwen2_5OmniThinkerConfig(
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                rope_scaling=dict(rope_type='default', mrope_section=[1, 1, 2])),
            audio_config=dict(d_model=16, encoder_layers=1, encoder_attention_heads=2,
                encoder_ffn_dim=32, output_dim=16),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                num_heads=2, out_hidden_size=16),
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
            vision_start_token_id=28, vision_end_token_id=29,
            image_token_index=30, video_token_index=31,
            audio_start_token_id=25, audio_end_token_id=26, audio_token_index=27)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
        # Exercise the exact same multi-token equivalence assertions with an
        # actual Omni Thinker, including its mutable multimodal RoPE offsets.
        with patch('transformers.GPT2LMHeadModel', return_value=model):
            self.test_real_generation_matches_independent_runs_with_divergent_prefixes()

    def test_real_generation_matches_independent_runs_with_divergent_prefixes(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(7)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
            eos_token_id=None, pad_token_id=0, bos_token_id=1)).eval()
        inputs = dict(input_ids=torch.tensor([[3, 4, 5]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        silent = dict(input_ids=torch.tensor([[3, 8, 5]]), attention_mask=inputs['attention_mask'])
        with torch.inference_mode():
            top = model(**inputs).logits[0, -1].topk(2).indices.tolist()
        calls = []
        original = model.forward
        @functools.wraps(original)
        def counted(*a, **kw):
            calls.append((kw.get('use_cache'), kw['input_ids'].shape[1]))
            return original(*a, **kw)
        model.forward = counted
        engine = QwenTAD.__new__(QwenTAD)
        engine.model = model
        engine.yes_ids, engine.no_ids = [top[0]], [top[1]]
        engine.args = SimpleNamespace(tau=100., gamma=100., max_new_tokens=4)
        engine.processor = SimpleNamespace(batch_decode=lambda x, **k: [str(r.tolist()) for r in x])
        settings = runner.variants(runner.MODELS[0])
        expected = {name: engine.generate_prepared(inputs, silent, method=method, alpha=alpha)
                    for name, (method, alpha) in settings.items()}
        self.assertNotEqual(expected['baseline'], expected['tad05'])
        independent_count = len(calls)
        calls.clear()
        engine.prepare_batch = Mock(return_value=(inputs, silent))
        actual = engine.predict_variants([], [], settings)
        self.assertEqual(actual, expected)
        self.assertLess(len(calls), independent_count)
        self.assertEqual(sum(cache is True and length==3 for cache, length in calls), 1)
        engine.prepare_batch.assert_called_once()
        self.assertIs(model.forward, counted)

    def test_partial_resume_preserves_success_and_shares_remaining_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [dict(sample_id='yes1', label='yes', caption='dog'),
                       dict(sample_id='no1', label='no', caption='dog')]
            args = SimpleNamespace(tau=.2, gamma=2.5, max_new_tokens=16, batch_size=2)
            config = dict(prompt=dict(template='Is {caption} audible?'))
            spec = dict(model_id='fake')
            with patch.object(runner, 'ROOT', Path(tmp)), patch.object(runner, 'load_samples', return_value=samples):
                selected, states = runner.prepare_run(config, 'test', runner.MODELS[0], spec, args, False)
                saved = dict(sample_id='yes1', label='yes', prediction='yes', status='ok', raw_response='original')
                states['baseline']['rows']['yes1'] = saved.copy()
                engine = SimpleNamespace(spec=spec, predict_variants=Mock(return_value={
                    n: ['yes', 'no'] for n in states}))
                with patch('src.audio.audio_path', return_value=Path('fake.wav')):
                    runner.evaluate(engine, selected, states, config, runner.MODELS[0], args)
                self.assertEqual(engine.predict_variants.call_count, 1)
                self.assertEqual(states['baseline']['rows']['yes1'], saved)
                self.assertEqual(runner.metrics(list(states['tad05']['rows'].values()))['accuracy'], 1.)
                _, resumed = runner.prepare_run(config, 'test', runner.MODELS[0], spec, args, False)
                engine.predict_variants.reset_mock()
                runner.evaluate(engine, selected, resumed, config, runner.MODELS[0], args)
                engine.predict_variants.assert_not_called()
                args.gamma = 3.
                with self.assertRaisesRegex(ValueError, 'Incompatible'):
                    runner.prepare_run(config, 'test', runner.MODELS[0], spec, args, False)

    def test_main_loads_each_model_once_across_phases_and_datasets(self):
        config = dict(models={'qwen2_audio_7b_instruct': dict(backend='qwen2_audio', model_id='fake')})
        samples = [dict(sample_id='x')]
        states = dict(tad05=dict(rows={}))
        with patch('sys.argv', ['run_qwen_shared.py', 'all']), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch.object(runner, 'read_config', return_value=config), \
             patch.object(runner, 'prepare_run', return_value=(samples, states)), \
             patch.object(runner, 'QwenTAD') as loader, \
             patch.object(runner, 'evaluate') as evaluate:
            loader.side_effect = [SimpleNamespace(), SimpleNamespace()]
            runner.main()
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(evaluate.call_count, 8)
            engines = [call.args[0] for call in evaluate.call_args_list]
            self.assertTrue(all(e is engines[0] for e in engines[:4]))
            self.assertTrue(all(e is engines[4] for e in engines[4:]))
            self.assertIsNot(engines[0], engines[4])


if __name__ == '__main__':
    unittest.main()
