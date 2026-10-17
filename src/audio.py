from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path


def audio_path(sample: dict, config: dict) -> Path:
    if sample.get("local_audio_path"):
        path = Path(sample["local_audio_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cache: Path = config["paths"]["audio_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{sample['video_id']}\t{sample['clip_start_seconds']:.3f}\t{sample['clip_end_seconds']:.3f}".encode()
    ).hexdigest()
    output = cache / f"{key}.wav"
    if output.exists() and output.stat().st_size > 44:
        return output

    start = sample["clip_start_seconds"]
    end = sample["clip_end_seconds"]
    temporary = cache / f"{key}.download"
    temporary.mkdir(exist_ok=True)
    template = str(temporary / "audio.%(ext)s")
    url = f"https://www.youtube.com/watch?v={sample['video_id']}"
    command = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet", "--no-progress",
        "--format", config["audio"].get("yt_dlp_format", "bestaudio/best"),
        "--extract-audio", "--audio-format", "wav",
        "--download-sections", f"*{start:.3f}-{end:.3f}",
        "--force-keyframes-at-cuts", "--output", template, url,
    ]
    attempts = int(config["audio"].get("download_retries", 3))
    last_error = ""
    for attempt in range(attempts):
        result = subprocess.run(command, capture_output=True, text=True)
        wav_files = list(temporary.glob("*.wav"))
        if result.returncode == 0 and wav_files:
            normalized = temporary / "normalized.wav"
            conversion = subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-i", str(wav_files[0]),
                "-ac", str(int(config["audio"].get("channels", 1))),
                "-ar", str(int(config["audio"].get("sample_rate", 32000))),
                "-c:a", "pcm_s16le", str(normalized),
            ], capture_output=True, text=True)
            if conversion.returncode == 0 and normalized.exists():
                normalized.replace(output)
                shutil.rmtree(temporary)
                return output
            last_error = (conversion.stderr or conversion.stdout)[-2000:]
        else:
            last_error = (result.stderr or result.stdout)[-2000:]
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(exist_ok=True)
        time.sleep(float(config["audio"].get("request_sleep_seconds", 0.5)) * (attempt + 1))
    raise RuntimeError(f"Audio download failed for {sample['clip_id']}: {last_error}")
