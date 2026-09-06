"""ECAPA voice-family classifier for Azure TTS dubbing.

Based on JaesungHuh/voice-gender-classifier
Source: https://github.com/JaesungHuh/voice-gender-classifier
License: MIT
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


VOICE_GENDER_MODEL_ID = "JaesungHuh/voice-gender-classifier"
VOICE_GENDER_MODEL_REVISION = "db1222153bd60337e900be22add7af180452adc0"


# Minimal model definition for inference (vendor from upstream)
# Based on https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
# Modified by JaesungHuh for binary gender classification


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128) -> None:
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int | None = None,
        dilation: int | None = None,
        scale: int = 8,
    ) -> None:
        super(Bottle2neck, self).__init__()
        width = int(planes // scale)
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = kernel_size // 2 * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_gender(nn.Module, PyTorchModelHubMixin):
    def __init__(self, C: int = 1024):
        super(ECAPA_gender, self).__init__()
        self.C = C
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 2)

    def logtorchfbank(self, x: torch.Tensor) -> torch.Tensor:
        # Preemphasis
        flipped_filter = torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0).to(x.device)
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), "reflect")
        x = F.conv1d(x, flipped_filter).squeeze(1)

        # Melspectrogram (using torchaudio.transforms if available, otherwise fallback)
        try:
            from torchaudio.transforms import MelSpectrogram
            x = MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ).to(x.device)(x) + 1e-6
        except Exception:
            # Fallback: use librosa if available, otherwise raise error
            try:
                import librosa
                import librosa.display

                # Convert to numpy for librosa
                x_np = x.cpu().numpy()
                mel = librosa.feature.melspectrogram(
                    y=x_np,
                    sr=16000,
                    n_fft=512,
                    hop_length=160,
                    win_length=400,
                    n_mels=80,
                    fmin=20,
                    fmax=7600,
                )
                x = torch.from_numpy(mel).float().to(x.device) + 1e-6
            except ImportError:
                raise ImportError("Neither torchaudio nor librosa available for mel spectrogram")

        # Log and normalize
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.logtorchfbank(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc7(x)

        return x


@dataclass
class VoiceFamilyClassificationResult:
    """Result of voice-family classification."""

    speaker_id: str
    method: str
    model: str
    device: str
    voice_family: str  # "masculine" | "feminine" | "unknown"
    confidence: float
    probabilities: dict[str, float]  # {"masculine": float, "feminine": float}

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "method": self.method,
            "model": self.model,
            "device": self.device,
            "voice_family": self.voice_family,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


# Global model cache
_model_cache: ECAPA_gender | None = None


def _load_model() -> ECAPA_gender:
    """Load the ECAPA voice-family classifier from Hugging Face (cached)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    # PyTorchModelHubMixin loads this repository from model.safetensors.
    # huggingface_hub treats safetensors as an optional dependency and some
    # versions otherwise fail later with the misleading NameError
    # ``name 'safetensors' is not defined``.  Fail here with an actionable
    # Mathula error before touching the model cache.
    try:
        import safetensors  # noqa: F401
        from safetensors.torch import load_model as _load_safetensor_model  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing safetensors runtime required by the pinned voice-family model. "
            "Install Mathula voice dependencies with "
            "`python -m pip install -e \".[speaker-recognition]\" --no-build-isolation` "
            "or run `python -m pip install \"safetensors>=0.8,<1\"`."
        ) from exc

    try:
        model = ECAPA_gender.from_pretrained(
            VOICE_GENDER_MODEL_ID,
            revision=VOICE_GENDER_MODEL_REVISION,
            local_files_only=True,
        )
        model.eval()
        _model_cache = model
        return model
    except Exception as exc:
        raise RuntimeError(
            "The pinned voice-family classifier is not installed locally. "
            "Run `python scripts/install_voice_gender_classifier.py` once. "
            f"Runtime network access is disabled for {VOICE_GENDER_MODEL_ID}@"
            f"{VOICE_GENDER_MODEL_REVISION}: {exc}"
        ) from exc


def _load_wav_as_float32(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load WAV file as float32 numpy array using wave module (avoids torchaudio)."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    # Convert to numpy
    dtype = np.int16 if sample_width == 2 else np.int32
    samples = np.frombuffer(frames, dtype=dtype)

    # Convert to float32 and normalize
    if dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32) / 2147483648.0

    # Convert to mono if stereo
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    # Resample if needed using simple linear interpolation
    if sample_rate != target_sr:
        # Simple linear resampling without scipy
        ratio = target_sr / sample_rate
        num_samples = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, num_samples)
        samples = np.interp(indices, np.arange(len(samples)), samples)

    return samples, target_sr


def classify_voice_family(
    speaker_audio_path: Path,
    speaker_id: str,
    confidence_threshold: float = 0.70,
) -> VoiceFamilyClassificationResult:
    """Classify voice family using ECAPA model.

    Args:
        speaker_audio_path: Path to canonical speaker WAV file
        speaker_id: Speaker identifier
        confidence_threshold: Minimum confidence threshold (default 0.70)

    Returns:
        VoiceFamilyClassificationResult with classification

    Raises:
        FileNotFoundError: If speaker audio file doesn't exist
        RuntimeError: If model loading or inference fails
    """
    if not speaker_audio_path.exists():
        raise FileNotFoundError(f"Speaker audio not found: {speaker_audio_path}")

    # Load audio using wave module (avoid torchaudio)
    audio, sample_rate = _load_wav_as_float32(speaker_audio_path)

    # Ensure 16kHz
    if sample_rate != 16000:
        raise ValueError(f"Audio must be 16kHz, got {sample_rate}Hz")

    # Convert to torch tensor with shape [1, samples]
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

    # Load model (cached)
    model = _load_model()
    device = torch.device("cpu")
    model = model.to(device)
    audio_tensor = audio_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(audio_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    # Extract probabilities (index 0 = masculine, index 1 = feminine)
    prob_masculine = float(probabilities[0].item())
    prob_feminine = float(probabilities[1].item())

    # Validate probabilities
    if not (np.isfinite(prob_masculine) and np.isfinite(prob_feminine)):
        raise RuntimeError("Invalid probabilities from classifier")
    if not (0.0 <= prob_masculine <= 1.0 and 0.0 <= prob_feminine <= 1.0):
        raise RuntimeError("Probabilities out of range")
    if not np.isclose(prob_masculine + prob_feminine, 1.0, atol=0.01):
        raise RuntimeError("Probabilities don't sum to 1")

    # Determine voice family and confidence
    if prob_masculine > prob_feminine:
        voice_family = "masculine"
        confidence = prob_masculine
    else:
        voice_family = "feminine"
        confidence = prob_feminine

    # Apply confidence threshold
    if confidence < confidence_threshold:
        voice_family = "unknown"
        confidence = 0.0

    return VoiceFamilyClassificationResult(
        speaker_id=speaker_id,
        method="ecapa_voice_family_classifier",
        model="JaesungHuh/voice-gender-classifier",
        device=str(device),
        voice_family=voice_family,
        confidence=confidence,
        probabilities={"masculine": prob_masculine, "feminine": prob_feminine},
    )


__all__ = [
    "VoiceFamilyClassificationResult",
    "classify_voice_family",
]
