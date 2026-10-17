from __future__ import annotations

import base64
import os
from pathlib import Path
from types import SimpleNamespace


# q3 + r1, https://arxiv.org/html/2603.03855v1#S2.SS4 (Table 3).
PROMPT = "Does this audio contain the sound of {caption}? Respond using only one word: yes or no."


class LLM:
    def __init__(self, model_key: str, model_config: dict, config: dict):
        self.key = model_key
        self.spec = model_config
        self.config = config
        self.backend = model_config["backend"]
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        model_id = self.spec["model_id"]
        if self.backend == "openrouter":
            import httpx
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not api_key or api_key == "your_openrouter_api_key_here":
                raise ValueError("Set OPENROUTER_API_KEY in the project .env file")
            self.spec = dict(self.spec)
            self.spec["model_id"] = os.getenv(self.spec.get("model_id_env", "OPENROUTER_MODEL"), "").strip() or model_id
            self.model = httpx.Client(
                base_url="https://openrouter.ai/api/v1/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120.0,
            )
        elif self.backend == "kimi":
            from kimia_infer.api import kimia
            # Upstream hard-codes disable=False for a new token bar on every sample.
            # Override only Kimi's local tqdm reference, leaving loading/evaluation progress intact.
            kimia.tqdm = SimpleNamespace(tqdm=lambda iterable, *args, **kwargs: iterable)
            self.model = kimia.KimiAudio(model_path=model_id, load_detokenizer=False)
        elif self.backend == "audio_flamingo":
            import torch
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )
        elif self.backend == "qwen3_omni":
            import torch
            from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
            self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def predict(self, sample: dict, audio_file: Path) -> str:
        prompt = self.config.get("prompt", {}).get("template", PROMPT).format(caption=sample["caption"])
        max_tokens = int(self.spec.get("max_new_tokens", self.config["experiment"].get("max_new_tokens", 1)))
        if self.backend == "openrouter":
            response = self.model.post(
                "chat/completions",
                json={
                    "model": self.spec["model_id"],
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {
                            "data": base64.b64encode(audio_file.read_bytes()).decode("ascii"),
                            "format": "wav",
                        }},
                    ]}],
                    "temperature": float(self.config["experiment"].get("temperature", 0.0)),
                    "max_tokens": max_tokens,
                    "reasoning": {"effort": "low", "exclude": True},
                },
            )
            if response.is_error:
                try:
                    message = response.json().get("error", {}).get("message", response.reason_phrase)
                except (ValueError, AttributeError):
                    message = response.reason_phrase
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {message}")
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(f"OpenRouter error: {payload['error']}")
            text = payload["choices"][0]["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("OpenRouter returned no answer; check the model token budget")
            return text
        if self.backend == "kimi":
            _, text = self.model.generate(
                [
                    {"role": "user", "message_type": "text", "content": prompt},
                    {"role": "user", "message_type": "audio", "content": str(audio_file)},
                ],
                output_type="text", max_new_tokens=max_tokens,
                text_temperature=0.0, text_top_k=1,
            )
            return text

        return self.predict_batch([sample], [audio_file])[0]

    def predict_batch(self, samples: list[dict], audio_files: list[Path]) -> list[str]:
        """Batch Transformers backends where supported; Kimi and OpenRouter use one request per sample."""
        if len(samples) != len(audio_files):
            raise ValueError("samples/audio_files length mismatch")
        if self.backend in {"openrouter", "kimi"}:
            return [self.predict(sample, path) for sample, path in zip(samples, audio_files)]
        max_tokens = int(self.spec.get("max_new_tokens", self.config["experiment"].get("max_new_tokens", 1)))
        template = self.config.get("prompt", {}).get("template", PROMPT)
        conversations = []
        for sample, audio_file in zip(samples, audio_files):
            prompt = template.format(caption=sample["caption"])
            if self.backend == "audio_flamingo":
                conversations.append([{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio_file)},
                ]}])
            else:
                conversations.append([
                    {"role": "user", "content": [
                        {"type": "audio", "path": str(audio_file)},
                        {"type": "text", "text": prompt},
                    ]},
                ])
        inputs = self.processor.apply_chat_template(
            conversations if len(conversations) > 1 else conversations[0],
            tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", text_kwargs={"padding": True},
        ).to(device=self.model.device, dtype=self.model.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

