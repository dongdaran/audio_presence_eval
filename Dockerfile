FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/experiments/audio_presence_eval
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/MoonshotAI/Kimi-Audio.git

COPY . .
ENTRYPOINT ["python", "scripts/run_experiment.py"]
