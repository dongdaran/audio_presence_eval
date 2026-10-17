import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.audio import audio_path


class AudioDownloadTests(unittest.TestCase):
    def test_ffmpeg_crash_falls_back_to_local_trim(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return SimpleNamespace(returncode=1, stderr='ERROR: ffmpeg exited with code -11', stdout='')
            if len(calls) == 2:
                template = command[command.index('--output') + 1]
                Path(template.replace('%(ext)s', 'm4a')).write_bytes(b'source')
            else:
                Path(command[-1]).write_bytes(b'WAV' * 30)
            return SimpleNamespace(returncode=0, stderr='', stdout='')

        with tempfile.TemporaryDirectory() as folder:
            config = {'paths': {'audio_cache': Path(folder)}, 'audio': {'download_retries': 1}}
            sample = {'video_id': 'video', 'clip_id': 'clip', 'clip_start_seconds': 10., 'clip_end_seconds': 20.}
            with patch('src.audio.subprocess.run', side_effect=run):
                output = audio_path(sample, config)
                self.assertTrue(output.is_file())
                self.assertEqual(audio_path(sample, config), output)
            self.assertEqual(len(calls), 3)
            self.assertNotIn('--download-sections', calls[1])
            self.assertNotIn('--extract-audio', calls[1])
            self.assertEqual(calls[2][calls[2].index('-ss') + 1], '10.000')
            self.assertEqual(calls[2][calls[2].index('-t') + 1], '10.000')
            self.assertEqual(list(Path(folder).glob('*.download')), [])
