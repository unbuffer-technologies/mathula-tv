from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openvoice import OpenVoiceRuntimeInfo


_CONFIG_CHECKPOINT = "converter_config"
_MODEL_CHECKPOINT = "converter_checkpoint"
_REQUIRED_CHECKPOINTS = {
    _CONFIG_CHECKPOINT,
    _MODEL_CHECKPOINT,
}
_OUTPUT_SAMPLE_RATE = 24000


@dataclass
class _LoadedOpenVoiceModel:
    converter: Any
    device: str
    precision: str


def _openvoice_checkout() -> Path:
    value = os.environ.get("MATHULA_TV_OPENVOICE_CHECKOUT", "").strip()

    if not value:
        raise RuntimeError("OpenVoice checkout is not configured")

    checkout = Path(value).expanduser().resolve()

    if not checkout.is_dir():
        raise RuntimeError("Configured OpenVoice checkout is unavailable")

    return checkout


def _checkout_revision(checkout: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )

    revision = completed.stdout.strip()

    if completed.returncode != 0 or len(revision) != 40:
        raise RuntimeError("OpenVoice revision inspection failed")

    return revision


def _import_openvoice(checkout: Path) -> tuple[Any, Any]:
    checkout_text = str(checkout)

    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)

    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

    return OpenVoiceBaseClass, ToneColorConverter


def _normalise_device(device: str) -> str:
    value = device.strip().lower()

    if value == "cuda":
        return "cuda:0"

    if value.startswith("cuda:") or value == "cpu":
        return value

    raise ValueError("OpenVoice device must be cuda, cuda:N, or cpu")


class OpenVoiceV2Runtime:
    """Converter-only OpenVoice V2 runtime for Mathula TV."""

    def inspect(self) -> OpenVoiceRuntimeInfo:
        import torch

        checkout = _openvoice_checkout()

        gpu_name: str | None = None

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        return OpenVoiceRuntimeInfo(
            openvoice_revision=_checkout_revision(checkout),
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            gpu_name=gpu_name,
        )

    def load_model(
        self,
        *,
        revision: str,
        checkpoints: dict[str, Path],
        device: str,
        precision: str,
    ) -> _LoadedOpenVoiceModel:
        import torch

        checkout = _openvoice_checkout()

        if _checkout_revision(checkout) != revision:
            raise RuntimeError("OpenVoice checkout revision does not match the plan")

        if set(checkpoints) != _REQUIRED_CHECKPOINTS:
            raise ValueError(
                "OpenVoice requires converter_config and converter_checkpoint"
            )

        config_path = checkpoints[_CONFIG_CHECKPOINT]
        checkpoint_path = checkpoints[_MODEL_CHECKPOINT]

        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("A required OpenVoice checkpoint is missing")

        # The converter is small enough to run safely in float32 on a T4.
        # OpenVoice's unmodified extract_se implementation creates float32
        # spectrograms, so fp16 conversion would otherwise cause dtype mismatch.
        if precision != "fp32":
            raise ValueError(
                "This OpenVoice V2 runtime currently supports fp32 only"
            )

        resolved_device = _normalise_device(device)

        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        OpenVoiceBaseClass, ToneColorConverter = _import_openvoice(checkout)

        class NoWatermarkToneColorConverter(ToneColorConverter):
            def __init__(
                self,
                converter_config: str,
                runtime_device: str,
            ) -> None:
                # The pinned upstream constructor incorrectly forwards
                # enable_watermark to OpenVoiceBaseClass. Initialise the base
                # class directly and disable the optional wavmark model.
                OpenVoiceBaseClass.__init__(
                    self,
                    config_path=converter_config,
                    device=runtime_device,
                )

                self.watermark_model = None
                self.version = getattr(self.hps, "_version_", "v1")

        converter = NoWatermarkToneColorConverter(
            converter_config=str(config_path),
            runtime_device=resolved_device,
        )

        checkpoint_payload = torch.load(
            str(checkpoint_path),
            map_location=torch.device(resolved_device),
            weights_only=True,
        )

        model_state = checkpoint_payload.get("model")

        if not isinstance(model_state, dict):
            raise RuntimeError("OpenVoice checkpoint has no model state")

        missing_keys, unexpected_keys = converter.model.load_state_dict(
            model_state,
            strict=False,
        )

        if missing_keys or unexpected_keys:
            raise RuntimeError("OpenVoice checkpoint is incompatible")

        converter.model.eval()

        return _LoadedOpenVoiceModel(
            converter=converter,
            device=resolved_device,
            precision=precision,
        )

    def extract_embedding(
        self,
        model: _LoadedOpenVoiceModel,
        *,
        audio_path: Path,
        output_path: Path,
        embedding_kind: str,
    ) -> None:
        import torch

        if embedding_kind not in {"source", "target"}:
            raise ValueError("Embedding kind must be source or target")

        if not audio_path.is_file():
            raise FileNotFoundError("Embedding reference audio is missing")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        embedding = model.converter.extract_se(
            [str(audio_path)],
            se_save_path=None,
        )

        if not torch.is_tensor(embedding) or embedding.numel() == 0:
            raise RuntimeError("OpenVoice returned an invalid embedding")

        torch.save(
            embedding.detach().cpu().float(),
            str(output_path),
        )

    def convert(
        self,
        model: _LoadedOpenVoiceModel,
        *,
        source_audio_path: Path,
        source_embedding_path: Path,
        target_embedding_path: Path,
        output_path: Path,
        target_language: str,
        settings: dict[str, Any],
    ) -> None:
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly

        if not target_language.strip():
            raise ValueError("Target language is required")

        if not source_audio_path.is_file():
            raise FileNotFoundError("OpenVoice source audio is missing")

        if (
            not source_embedding_path.is_file()
            or not target_embedding_path.is_file()
        ):
            raise FileNotFoundError("OpenVoice embedding asset is missing")

        tau = float(settings.get("tau", 0.3))

        if not math.isfinite(tau) or tau < 0.0 or tau > 1.0:
            raise ValueError("OpenVoice tau must be between 0 and 1")

        source_embedding = torch.load(
            str(source_embedding_path),
            map_location=torch.device(model.device),
            weights_only=True,
        )
        target_embedding = torch.load(
            str(target_embedding_path),
            map_location=torch.device(model.device),
            weights_only=True,
        )

        if (
            not torch.is_tensor(source_embedding)
            or not torch.is_tensor(target_embedding)
        ):
            raise RuntimeError("OpenVoice embedding data is invalid")

        source_embedding = source_embedding.float().to(model.device)
        target_embedding = target_embedding.float().to(model.device)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".openvoice-converted-",
            suffix=".wav",
            dir=output_path.parent,
        )
        os.close(descriptor)

        temporary_path = Path(temporary_name)

        try:
            model.converter.convert(
                audio_src_path=str(source_audio_path),
                src_se=source_embedding,
                tgt_se=target_embedding,
                output_path=str(temporary_path),
                tau=tau,
                message="",
            )

            audio, actual_sample_rate = sf.read(
                str(temporary_path),
                dtype="float32",
                always_2d=False,
            )

            audio_array = np.asarray(audio, dtype=np.float32)

            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

            if audio_array.ndim != 1 or audio_array.size == 0:
                raise RuntimeError("OpenVoice returned invalid audio")

            if actual_sample_rate != _OUTPUT_SAMPLE_RATE:
                divisor = math.gcd(
                    int(actual_sample_rate),
                    _OUTPUT_SAMPLE_RATE,
                )
                up = _OUTPUT_SAMPLE_RATE // divisor
                down = int(actual_sample_rate) // divisor

                audio_array = resample_poly(
                    audio_array,
                    up,
                    down,
                ).astype(np.float32)

            audio_array = np.clip(audio_array, -1.0, 1.0)

            sf.write(
                str(output_path),
                audio_array,
                _OUTPUT_SAMPLE_RATE,
                subtype="PCM_16",
                format="WAV",
            )
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def create_runtime() -> OpenVoiceV2Runtime:
    """Zero-argument factory used by the Colab worker."""

    return OpenVoiceV2Runtime()
