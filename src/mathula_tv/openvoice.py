from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import PipelineError
from .quality import inspect_wav


OPENVOICE_BACKEND = "openvoice"
OPENVOICE_RESULT_SCHEMA = "openvoice-conversion-result-v1"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

__all__ = [
    "CheckpointAsset",
    "OpenVoiceAssetHashMismatch",
    "OpenVoiceAttemptLimitExceeded",
    "OpenVoiceBackend",
    "OpenVoiceCheckpointHashMismatch",
    "OpenVoiceCheckpointMissing",
    "OpenVoiceConversionRequest",
    "OpenVoiceConversionResult",
    "OpenVoiceCudaOutOfMemory",
    "OpenVoiceCudaUnavailable",
    "OpenVoiceDependencyUnavailable",
    "OpenVoiceError",
    "OpenVoiceInferenceError",
    "OpenVoiceInvalidOutput",
    "OpenVoiceLeaseLost",
    "OpenVoiceModelLoadError",
    "OpenVoicePins",
    "OpenVoiceRevisionMismatch",
    "OpenVoiceRuntime",
    "OpenVoiceRuntimeInfo",
    "OpenVoiceRuntimeMismatch",
    "sha256_file",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OpenVoiceError(PipelineError):
    """Secret-safe, actionable OpenVoice worker error."""

    code = "openvoice_error"
    retryable = False
    action = "Inspect the OpenVoice worker configuration and retry explicitly."
    stage = "voice_conversion"

    def as_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "error_type": type(self).__name__,
            "message": str(self),
            "retryable": self.retryable,
            "action": self.action,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_record()


class OpenVoiceDependencyUnavailable(OpenVoiceError):
    code = "openvoice_dependency_unavailable"
    action = "Install the pinned OpenVoice GPU worker environment."


class OpenVoiceRevisionMismatch(OpenVoiceError):
    code = "openvoice_revision_mismatch"
    action = "Check out the OpenVoice revision pinned by the conversion plan."


class OpenVoiceRuntimeMismatch(OpenVoiceError):
    code = "openvoice_runtime_mismatch"
    action = "Use the pinned Python and Torch worker environment."


class OpenVoiceCheckpointMissing(OpenVoiceError):
    code = "openvoice_checkpoint_missing"
    action = "Materialise every checkpoint listed in the conversion plan."


class OpenVoiceCheckpointHashMismatch(OpenVoiceError):
    code = "openvoice_checkpoint_hash_mismatch"
    action = "Delete the invalid model asset and materialise the pinned checkpoint again."


class OpenVoiceAssetHashMismatch(OpenVoiceError):
    code = "openvoice_asset_hash_mismatch"
    action = "Download the planned input again and verify its authoritative hash."


class OpenVoiceCudaUnavailable(OpenVoiceError):
    code = "cuda_unavailable"
    action = "Run this plan in a compatible CUDA-enabled Colab or Kaggle session."


class OpenVoiceCudaOutOfMemory(OpenVoiceError):
    code = "cuda_out_of_memory"
    retryable = True
    action = "Restart the GPU runtime or select a worker with sufficient GPU memory."


class OpenVoiceModelLoadError(OpenVoiceError):
    code = "openvoice_model_load_failed"
    retryable = True
    action = "Inspect the pinned GPU runtime and model assets, then retry."


class OpenVoiceInferenceError(OpenVoiceError):
    code = "openvoice_inference_failed"
    retryable = True
    action = "Retry the affected unit; inspect the GPU runtime if the failure repeats."


class OpenVoiceInvalidOutput(OpenVoiceError):
    code = "openvoice_invalid_output"
    retryable = True
    action = "Retry the affected unit and inspect the OpenVoice audio output."


class OpenVoiceAttemptLimitExceeded(OpenVoiceError):
    code = "openvoice_attempt_limit_exceeded"
    action = "Review the failed unit before authorising another conversion attempt."


class OpenVoiceLeaseLost(OpenVoiceError):
    code = "lease_lost"
    stage = "worker"
    action = "Stop this stale worker and let the current lease owner resume the plan."


@dataclass(frozen=True)
class CheckpointAsset:
    name: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _validate_identifier("checkpoint name", self.name)
        _validate_sha256("checkpoint", self.sha256)


@dataclass(frozen=True)
class OpenVoicePins:
    revision: str
    python_version: str
    torch_version: str
    checkpoints: tuple[CheckpointAsset, ...]

    def __post_init__(self) -> None:
        if not self.revision.strip():
            raise ValueError("A pinned OpenVoice revision is required")
        if not self.python_version.strip() or not self.torch_version.strip():
            raise ValueError("Pinned Python and Torch versions are required")
        if not self.checkpoints:
            raise ValueError("At least one pinned OpenVoice checkpoint is required")
        names = [item.name for item in self.checkpoints]
        if len(names) != len(set(names)):
            raise ValueError("OpenVoice checkpoint names must be unique")


@dataclass(frozen=True)
class OpenVoiceRuntimeInfo:
    openvoice_revision: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None = None
    gpu_name: str | None = None


@dataclass(frozen=True)
class OpenVoiceConversionRequest:
    job_id: str
    unit_id: str
    speaker_id: str
    source_audio_path: Path
    source_audio_sha256: str
    source_embedding_path: Path
    source_embedding_sha256: str
    target_embedding_path: Path
    target_embedding_sha256: str
    target_language: str
    azure_voice: str
    conversion_settings: dict[str, Any]
    output_path: Path
    lease_identity: str
    attempt: int = 1
    sample_rate: int = 24000
    channels: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("job ID", self.job_id),
            ("unit ID", self.unit_id),
            ("speaker ID", self.speaker_id),
            ("lease identity", self.lease_identity),
        ):
            _validate_identifier(label, value)
        for label, value in (
            ("source audio", self.source_audio_sha256),
            ("source embedding", self.source_embedding_sha256),
            ("target embedding", self.target_embedding_sha256),
        ):
            _validate_sha256(label, value)
        if not self.target_language.strip() or not self.azure_voice.strip():
            raise ValueError("Target language and selected Azure voice are required")
        if self.attempt < 1:
            raise ValueError("OpenVoice attempt must be positive")
        if self.sample_rate <= 0 or self.channels != 1:
            raise ValueError("OpenVoice output must be canonical mono PCM audio")


@dataclass(frozen=True)
class OpenVoiceConversionResult:
    backend_revision: str
    unit_id: str
    speaker_id: str
    azure_voice: str
    source_audio_sha256: str
    source_embedding_sha256: str
    target_embedding_sha256: str
    output_sha256: str
    source_duration_ms: int
    converted_duration_ms: int
    duration_ratio: float
    sample_rate: int
    channels: int
    device: str
    precision: str
    attempt: int
    warnings: list[str] = field(default_factory=list)
    backend: str = OPENVOICE_BACKEND
    schema_version: str = OPENVOICE_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenVoiceRuntime(Protocol):
    """GPU-only implementation boundary.

    Implementations may import Torch and OpenVoice internally.  This protocol's
    module deliberately does not import either package.
    """

    def inspect(self) -> OpenVoiceRuntimeInfo: ...

    def load_model(
        self,
        *,
        revision: str,
        checkpoints: dict[str, Path],
        device: str,
        precision: str,
    ) -> Any: ...

    def extract_embedding(
        self,
        model: Any,
        *,
        audio_path: Path,
        output_path: Path,
        embedding_kind: str,
    ) -> None: ...

    def convert(
        self,
        model: Any,
        *,
        source_audio_path: Path,
        source_embedding_path: Path,
        target_embedding_path: Path,
        output_path: Path,
        target_language: str,
        settings: dict[str, Any],
    ) -> None: ...


class OpenVoiceBackend:
    """Pinned, injected OpenVoice model session shared by a worker."""

    def __init__(
        self,
        runtime: OpenVoiceRuntime,
        pins: OpenVoicePins,
        *,
        device: str = "cuda",
        precision: str = "fp16",
    ) -> None:
        self.runtime = runtime
        self.pins = pins
        self.device = device
        self.precision = precision
        self._model: Any = None
        self._runtime_info: OpenVoiceRuntimeInfo | None = None
        self._checkpoint_hashes: dict[str, str] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def runtime_info(self) -> OpenVoiceRuntimeInfo | None:
        return self._runtime_info

    @property
    def checkpoint_hashes(self) -> dict[str, str]:
        return dict(self._checkpoint_hashes or {})

    def load(self) -> OpenVoiceRuntimeInfo:
        if self._model is not None and self._runtime_info is not None:
            return self._runtime_info

        checkpoint_hashes = self._verify_checkpoints()
        try:
            runtime_info = self.runtime.inspect()
        except ImportError as exc:
            raise OpenVoiceDependencyUnavailable(
                "The pinned OpenVoice runtime is not installed"
            ) from exc
        except Exception as exc:
            raise OpenVoiceDependencyUnavailable(
                f"OpenVoice runtime inspection failed ({type(exc).__name__})"
            ) from exc
        if not isinstance(runtime_info, OpenVoiceRuntimeInfo):
            raise OpenVoiceRuntimeMismatch(
                "OpenVoice runtime inspection returned incompatible metadata"
            )

        if runtime_info.openvoice_revision != self.pins.revision:
            raise OpenVoiceRevisionMismatch(
                "The installed OpenVoice revision does not match the conversion plan"
            )
        mismatches = []
        if runtime_info.python_version != self.pins.python_version:
            mismatches.append("Python")
        if runtime_info.torch_version != self.pins.torch_version:
            mismatches.append("Torch")
        if mismatches:
            raise OpenVoiceRuntimeMismatch(
                "Pinned runtime mismatch: " + ", ".join(mismatches)
            )
        if self.device.startswith("cuda") and not runtime_info.cuda_available:
            raise OpenVoiceCudaUnavailable("CUDA is unavailable in this worker session")

        try:
            model = self.runtime.load_model(
                revision=self.pins.revision,
                checkpoints={item.name: item.path for item in self.pins.checkpoints},
                device=self.device,
                precision=self.precision,
            )
        except ImportError as exc:
            raise OpenVoiceDependencyUnavailable(
                "The pinned OpenVoice runtime is not installed"
            ) from exc
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise OpenVoiceCudaOutOfMemory(
                    "CUDA ran out of memory while loading OpenVoice"
                ) from exc
            raise OpenVoiceModelLoadError(
                f"OpenVoice model loading failed ({type(exc).__name__})"
            ) from exc
        if model is None:
            raise OpenVoiceModelLoadError("OpenVoice model loading returned no model")

        self._model = model
        self._runtime_info = runtime_info
        self._checkpoint_hashes = checkpoint_hashes
        return runtime_info

    def extract_embedding(
        self,
        audio_path: Path,
        output_path: Path,
        *,
        embedding_kind: str,
        expected_audio_sha256: str | None = None,
    ) -> str:
        if embedding_kind not in {"source", "target"}:
            raise ValueError("Embedding kind must be source or target")
        if expected_audio_sha256 is not None:
            _verify_asset("embedding audio", audio_path, expected_audio_sha256)
        self.load()
        temporary = _temporary_output(output_path)
        try:
            self.runtime.extract_embedding(
                self._model,
                audio_path=audio_path,
                output_path=temporary,
                embedding_kind=embedding_kind,
            )
        except Exception as exc:
            _remove_if_present(temporary)
            if _is_cuda_oom(exc):
                raise OpenVoiceCudaOutOfMemory(
                    f"CUDA ran out of memory while extracting the {embedding_kind} embedding"
                ) from exc
            raise OpenVoiceInferenceError(
                f"OpenVoice {embedding_kind} embedding extraction failed "
                f"({type(exc).__name__})"
            ) from exc
        if not temporary.is_file() or temporary.stat().st_size == 0:
            _remove_if_present(temporary)
            raise OpenVoiceInvalidOutput(
                f"OpenVoice returned no {embedding_kind} embedding"
            )
        os.replace(temporary, output_path)
        return sha256_file(output_path)

    def convert(self, request: OpenVoiceConversionRequest) -> OpenVoiceConversionResult:
        _verify_asset(
            "Azure source audio", request.source_audio_path, request.source_audio_sha256
        )
        _verify_asset(
            "Azure source embedding",
            request.source_embedding_path,
            request.source_embedding_sha256,
        )
        _verify_asset(
            "target speaker embedding",
            request.target_embedding_path,
            request.target_embedding_sha256,
        )
        source_quality = _inspect_canonical_wav(
            request.source_audio_path,
            sample_rate=request.sample_rate,
            label="Azure source audio",
        )
        self.load()
        temporary = _temporary_output(request.output_path)
        try:
            self.runtime.convert(
                self._model,
                source_audio_path=request.source_audio_path,
                source_embedding_path=request.source_embedding_path,
                target_embedding_path=request.target_embedding_path,
                output_path=temporary,
                target_language=request.target_language,
                settings=dict(request.conversion_settings),
            )
        except Exception as exc:
            _remove_if_present(temporary)
            if _is_cuda_oom(exc):
                raise OpenVoiceCudaOutOfMemory(
                    "CUDA ran out of memory during OpenVoice conversion"
                ) from exc
            raise OpenVoiceInferenceError(
                f"OpenVoice conversion failed ({type(exc).__name__})"
            ) from exc

        try:
            output_quality = _inspect_canonical_wav(
                temporary,
                sample_rate=request.sample_rate,
                label="OpenVoice output",
            )
        except BaseException:
            _remove_if_present(temporary)
            raise
        source_duration_ms = round(float(source_quality["duration"]) * 1000)
        converted_duration_ms = round(float(output_quality["duration"]) * 1000)
        if source_duration_ms <= 0 or converted_duration_ms <= 0:
            _remove_if_present(temporary)
            raise OpenVoiceInvalidOutput(
                "OpenVoice source and converted audio must have measurable duration"
            )
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, request.output_path)

        ratio = converted_duration_ms / source_duration_ms
        warnings = []
        if float(output_quality.get("clipped_ratio", 0)) > 0:
            warnings.append("OpenVoice output contains clipped samples")
        runtime_info = self._runtime_info
        assert runtime_info is not None
        return OpenVoiceConversionResult(
            backend_revision=self.pins.revision,
            unit_id=request.unit_id,
            speaker_id=request.speaker_id,
            azure_voice=request.azure_voice,
            source_audio_sha256=request.source_audio_sha256,
            source_embedding_sha256=request.source_embedding_sha256,
            target_embedding_sha256=request.target_embedding_sha256,
            output_sha256=sha256_file(request.output_path),
            source_duration_ms=source_duration_ms,
            converted_duration_ms=converted_duration_ms,
            duration_ratio=round(ratio, 6),
            sample_rate=int(output_quality["sample_rate"]),
            channels=int(output_quality["channels"]),
            device=self.device,
            precision=self.precision,
            attempt=request.attempt,
            warnings=warnings,
        )

    def _verify_checkpoints(self) -> dict[str, str]:
        result = {}
        for checkpoint in self.pins.checkpoints:
            if not checkpoint.path.is_file():
                raise OpenVoiceCheckpointMissing(
                    f"Pinned OpenVoice checkpoint is missing: {checkpoint.name}"
                )
            actual = sha256_file(checkpoint.path)
            if actual != checkpoint.sha256:
                raise OpenVoiceCheckpointHashMismatch(
                    f"Pinned OpenVoice checkpoint hash mismatch: {checkpoint.name}"
                )
            result[checkpoint.name] = actual
        return result


def _validate_identifier(label: str, value: str) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {label}")


def _validate_sha256(label: str, value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {label} SHA-256")


def _verify_asset(label: str, path: Path, expected_sha256: str) -> None:
    _validate_sha256(label, expected_sha256)
    if not path.is_file():
        raise OpenVoiceAssetHashMismatch(f"Planned {label} is missing")
    if sha256_file(path) != expected_sha256:
        raise OpenVoiceAssetHashMismatch(f"Planned {label} hash does not match")


def _inspect_canonical_wav(path: Path, *, sample_rate: int, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OpenVoiceInvalidOutput(f"{label} was not created")
    try:
        quality = inspect_wav(path)
    except Exception as exc:
        raise OpenVoiceInvalidOutput(f"{label} is not a readable PCM WAV") from exc
    if (
        not quality.get("valid")
        or quality.get("channels") != 1
        or quality.get("sample_rate") != sample_rate
    ):
        raise OpenVoiceInvalidOutput(
            f"{label} must be valid mono {sample_rate} Hz PCM WAV audio"
        )
    return quality


def _temporary_output(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(fd)
    os.unlink(name)
    return Path(name)


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "oom" in message)
