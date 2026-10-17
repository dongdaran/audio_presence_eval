"""Run in .venv-kimi to exercise the checkpoint metadata regression."""
import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('kimia_infer'), 'requires Kimi environment')
class KimiCompatibilityTests(unittest.TestCase):
    def test_whisper_loads_safetensors_without_metadata(self):
        from pathlib import Path
        from safetensors.torch import save_file
        from transformers import WhisperConfig
        from kimia_infer.models.tokenizer.whisper_Lv3.modeling_whisper import WhisperModel

        config = WhisperConfig(
            d_model=8, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=16, decoder_ffn_dim=16, vocab_size=8,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        model = WhisperModel(config)
        with tempfile.TemporaryDirectory() as folder:
            config.save_pretrained(folder)
            save_file({key: value.detach().clone() for key, value in model.state_dict().items()},
                      str(Path(folder) / 'model.safetensors'))
            loaded = WhisperModel.from_pretrained(folder)
            self.assertEqual(loaded.config.d_model, 8)


if __name__ == '__main__':
    unittest.main()
