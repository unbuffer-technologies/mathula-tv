# Speech-to-text and speaker diarization on Cloud Run with an NVIDIA L4: Whisper large-v3 (faster-whisper) for the words, pyannote for who spoke when.
# Cloud Run's driver is 580 (CUDA 13); the CUDA 12.8 wheels of torch 2.8 bring cuDNN 9 and cuBLAS 12, which CTranslate2 (faster-whisper) finds through LD_LIBRARY_PATH.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Whisper large-v3 is public: bake it into the image so a cold start reads local disk.  The pyannote pipeline is gated (it needs an accepted licence and HF_TOKEN), so it is fetched on first use.
RUN python -c "from faster_whisper.utils import download_model; print(download_model('large-v3', cache_dir='/models/whisper'))"

COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
