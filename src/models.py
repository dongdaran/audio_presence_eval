from __future__ import annotations

from pathlib import Path


PROMPT = """Listen to the entire audio clip. Is the sound described as \"{caption}\" audible at any point in this clip? Answer with exactly one word: yes or no."""


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
        if self.backend == "gemini":
            from google import genai
            self.model = genai.Client()
        elif self.backend == "kimi":
            from kimia_infer.api.kimia import KimiAudio
            self.model = KimiAudio(model_path=model_id, load_detokenizer=False)
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
        if self.backend == "gemini":
            from google.genai import types
            response = self.model.models.generate_content(
                model=self.spec["model_id"],
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_file.read_bytes(), mime_type="audio/wav"),
                ],
                config=types.GenerateContentConfig(
                    temperature=float(self.config["experiment"].get("temperature", 0.0)),
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
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
        """Batch Transformers backends where supported; Kimi and Gemini use one request per sample."""
        if len(samples) != len(audio_files):
            raise ValueError("samples/audio_files length mismatch")
        if self.backend in {"gemini", "kimi"}:
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
                    {"role": "system", "content": [
                        {"type": "text", "text": "You are an audio understanding assistant. Answer the user's question about the audio."}
                    ]},
                    {"role": "user", "content": [
                        {"type": "audio", "path": str(audio_file)},
                        {"type": "text", "text": prompt},
                    ]},
                ])
        inputs = self.processor.apply_chat_template(
            conversations if len(conversations) > 1 else conversations[0],
            tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True,
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

