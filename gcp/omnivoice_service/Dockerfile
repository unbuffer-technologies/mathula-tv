# OmniVoice (Zulu LoRA) on Cloud Run with an NVIDIA L4.  Cloud Run's driver is 580 (CUDA 13), so the CUDA 12.8 torch wheels run as they are.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The base model and its audio tokenizer are public: bake them into the image so a cold start reads local disk and needs no token.
RUN python -c "import torch; from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32); print('models cached')"

# The forced aligner (Meta's MMS wav2vec2, about 1.2 GB) is baked in too, for the same reason; torchaudio 2.8 still ships it (deprecated, removed in 2.9: keep torch pinned).
RUN python -c "import torchaudio; torchaudio.pipelines.MMS_FA.get_model(with_star=False); print('aligner cached')"

COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
