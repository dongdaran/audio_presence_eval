import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.common import read_config
from src.models import LLM

ROOT = Path(__file__).resolve().parents[1]


class OpenRouterTests(unittest.TestCase):
    def test_sends_audio_and_records_actual_model(self):
        config = read_config(ROOT / 'config/experiment.yaml')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key', 'OPENROUTER_MODEL': 'google/test-audio'}):
            model = LLM('gemini_3_pro', config['models']['gemini_3_pro'], config)
        model.model.close()
        requests = []

        def respond(request):
            import json
            requests.append((request, json.loads(request.content)))
            return httpx.Response(200, json={'choices': [{'message': {'content': 'yes'}}]})

        with httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=httpx.MockTransport(respond)) as client:
            model.model = client
            with tempfile.TemporaryDirectory() as folder:
                audio = Path(folder) / 'clip.wav'
                audio.write_bytes(b'test-wav-bytes')
                self.assertEqual(model.predict_batch([{'caption': 'a dog'}], [audio]), ['yes'])
                model.spec.update(max_new_tokens=None, reasoning_effort='medium')
                self.assertEqual(model.predict({'caption': 'a dog'}, audio), 'yes')
        self.assertNotIn('max_tokens', requests[1][1])
        self.assertEqual(requests[1][1]['reasoning'], {'effort': 'medium', 'exclude': True})
        request, payload = requests[0]
        self.assertEqual(str(request.url), 'https://openrouter.ai/api/v1/chat/completions')
        self.assertEqual(payload['model'], 'google/test-audio')
        self.assertGreater(payload['max_tokens'], 1)
        part = payload['messages'][0]['content'][1]
        self.assertEqual(part['type'], 'input_audio')
        self.assertEqual(part['input_audio']['format'], 'wav')
        self.assertEqual(base64.b64decode(part['input_audio']['data']), b'test-wav-bytes')
        self.assertEqual(model.spec['model_id'], 'google/test-audio')
        self.assertEqual(config['models']['gemini_3_pro']['model_id'], 'google/gemini-3.1-pro-preview')

    def test_missing_key_fails_before_api_request(self):
        config = read_config(ROOT / 'config/experiment.yaml')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                LLM('gemini_3_pro', config['models']['gemini_3_pro'], config)

    def test_reports_openrouter_credit_error(self):
        config = read_config(ROOT / 'config/experiment.yaml')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
            model = LLM('gemini_3_pro', config['models']['gemini_3_pro'], config)
        model.model.close()
        transport = httpx.MockTransport(lambda request: httpx.Response(
            402, json={'error': {'message': 'Insufficient credits'}}))
        with httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=transport) as client:
            model.model = client
            with tempfile.TemporaryDirectory() as folder:
                audio = Path(folder) / 'clip.wav'
                audio.write_bytes(b'audio')
                with self.assertRaisesRegex(RuntimeError, 'HTTP 402: Insufficient credits'):
                    model.predict({'caption': 'speech'}, audio)

    def test_loads_project_dotenv_without_export(self):
        with patch('dotenv.load_dotenv') as load:
            read_config(ROOT / 'config/experiment.yaml')
        load.assert_called_once_with(ROOT / '.env', override=False)


if __name__ == '__main__':
    unittest.main()
