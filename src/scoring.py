from __future__ import annotations

import json
import re

def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def run_clap(samples, config, output_path):
    import laion_clap
    from src.audio import audio_path
    from src.common import write_tsv

    model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda" if _has_cuda() else "cpu")
    model.load_ckpt()
    batch_size = int(config["scoring"]["clap"].get("batch_size", 16))
    rows = []
    by_id = {sample["sample_id"]: sample for sample in samples}
    ids = list(by_id)
    for start in range(0, len(ids), batch_size):
        batch = [by_id[sample_id] for sample_id in ids[start:start + batch_size]]
        valid, paths = [], []
        for sample in batch:
            try:
                paths.append(str(audio_path(sample, config)))
                valid.append(sample)
            except Exception as exc:
                rows.append(_score_row(sample, "clap", None, "", status="audio_error", error=str(exc)))
        if not valid:
            continue
        template = config["scoring"]["clap"].get("text_template", "The sound of {caption}.")
        texts = [template.format(caption=s["caption"]) for s in valid]
        audio_features = model.get_audio_embedding_from_filelist(x=paths, use_tensor=True)
        text_features = model.get_text_embedding(texts, use_tensor=True)
        audio_features = _normalize(audio_features)
        text_features = _normalize(text_features)
        scores = (audio_features * text_features).sum(axis=1)
        for sample, text, score in zip(valid, texts, scores):
            rows.append(_score_row(sample, "clap", float(score), text))
        print(f"CLAP: {min(start + len(batch), len(ids))}/{len(ids)}", flush=True)
    write_tsv(output_path, rows)


def run_cnn14(samples, config, ontology_path, output_path):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from panns_inference import AudioTagging, labels
    from src.audio import audio_path
    from src.common import write_tsv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tagger = AudioTagging(checkpoint_path=None, device=device)
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    names_by_mid = {item["id"]: item["name"] for item in ontology}
    indices = {_norm(name): i for i, name in enumerate(labels)}
    batch_size = int(config["scoring"]["cnn14"].get("batch_size", 16))
    rows = []
    sr = int(config["scoring"]["cnn14"].get("sample_rate", 32000))
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        valid, waveforms, mapped = [], [], []
        for sample in batch:
            try:
                path = audio_path(sample, config)
                waveform, source_sr = sf.read(path, dtype="float32", always_2d=True)
                waveform = waveform.mean(axis=1)
                if source_sr != sr:
                    waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=sr)
                waveform = waveform[:sr * 10]
                if len(waveform) == 0:
                    raise ValueError("empty audio waveform")
                valid.append(sample)
                waveforms.append(waveform)
                name = names_by_mid.get(sample["mid"], sample["caption"])
                mapped.append(indices.get(_norm(name)))
            except Exception as exc:
                class_name = names_by_mid.get(sample["mid"], sample["caption"])
                rows.append(_score_row(sample, "cnn14", None, class_name, map_status="audio_error", error=str(exc)))
        if not valid:
            continue
        max_length = max(map(len, waveforms))
        batch_audio = np.zeros((len(waveforms), max_length), dtype=np.float32)
        for i, waveform in enumerate(waveforms):
            batch_audio[i, :len(waveform)] = waveform
        try:
            clipwise, _ = tagger.inference(batch_audio)
        except Exception as exc:
            for sample, name_index in zip(valid, mapped):
                class_name = names_by_mid.get(sample["mid"], sample["caption"])
                rows.append(_score_row(sample, "cnn14", None, class_name, map_status="inference_error", error=str(exc)))
            continue
        for batch_index, (sample, name_index) in enumerate(zip(valid, mapped)):
            class_name = names_by_mid.get(sample["mid"], sample["caption"])
            score = float(clipwise[batch_index, name_index]) if name_index is not None else None
            status = "ok" if name_index is not None else "unmapped"
            rows.append(_score_row(sample, "cnn14", score, class_name, map_status=status))
        print(f"CNN14: {min(start + len(batch), len(samples))}/{len(samples)}", flush=True)
    write_tsv(output_path, rows)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _normalize(values):
    import numpy as np
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _score_row(sample, model, score, text, map_status="ok", error=""):
    return {
        "sample_id": sample["sample_id"], "split": sample["split"],
        "clip_id": sample["clip_id"], "video_id": sample["video_id"],
        "clip_start_seconds": sample["clip_start_seconds"],
        "clip_end_seconds": sample["clip_end_seconds"], "duration_seconds": sample["duration_seconds"],
        "mid": sample["mid"], "caption": sample["caption"], "label": sample["label"],
        "model": model, "score": score, "score_text": text,
        "map_status": map_status, "status": map_status, "error": error,
    }
