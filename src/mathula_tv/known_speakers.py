"""Verified known-speaker enrolment and recognition with SpeechBrain ECAPA."""

from __future__ import annotations

import hashlib
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json


SPEECHBRAIN_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
KNOWN_SPEAKER_SCHEMA_VERSION = "mathula-known-speakers-v1"
DEFAULT_MATCH_THRESHOLD = 0.65
DEFAULT_MATCH_MARGIN = 0.10
MINIMUM_ENROLMENT_SAMPLES = 2
MINIMUM_RANGE_SECONDS = 3.0
_PERSON_ID_RE = re.compile(r"[a-z0-9][a-z0-9_]{1,79}")
_speechbrain_classifier_cache: dict[str, Any] = {}


class KnownSpeakerError(RuntimeError):
    """Raised when a verified speaker profile cannot be safely used."""


@dataclass(frozen=True)
class KnownSpeakerMatch:
    person_id: str
    display_name: str
    gender: str
    azure_voice: str
    score: float
    runner_up_score: float | None
    threshold: float
    margin: float
    model_id: str
    model_revision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "gender": self.gender,
            "azure_voice": self.azure_voice,
            "score": round(self.score, 6),
            "runner_up_score": (
                round(self.runner_up_score, 6)
                if self.runner_up_score is not None
                else None
            ),
            "threshold": self.threshold,
            "margin": self.margin,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }


class SpeechBrainECAPAEncoder:
    """Local-only wrapper around SpeechBrain's ECAPA speaker encoder."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self._classifier: Any | None = None

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        cache_key = str(self.cache_root.resolve())
        if cache_key in _speechbrain_classifier_cache:
            self._classifier = _speechbrain_classifier_cache[cache_key]
            return self._classifier
        try:
            from huggingface_hub import snapshot_download
            from speechbrain.inference.speaker import EncoderClassifier

            snapshot = snapshot_download(
                repo_id=SPEECHBRAIN_MODEL_ID,
                revision=SPEECHBRAIN_MODEL_REVISION,
                local_files_only=True,
            )
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self._classifier = EncoderClassifier.from_hparams(
                source=snapshot,
                savedir=str(self.cache_root),
                run_opts={"device": "cpu"},
            )
            _speechbrain_classifier_cache[cache_key] = self._classifier
            return self._classifier
        except Exception as exc:
            raise KnownSpeakerError(
                "The pinned SpeechBrain ECAPA model is not installed locally. "
                "Run `python scripts/install_speechbrain_ecapa.py` once. "
                f"Runtime network access is disabled for {SPEECHBRAIN_MODEL_ID}@"
                f"{SPEECHBRAIN_MODEL_REVISION}: {exc}"
            ) from exc

    def encode_file(self, path: Path) -> list[float]:
        audio, sample_rate = _load_pcm16_mono(path)
        if sample_rate != 16000:
            raise KnownSpeakerError(
                f"SpeechBrain speaker audio must be 16kHz, got {sample_rate}Hz"
            )
        try:
            import torch

            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding = self._load().encode_batch(waveform)
            values = embedding.detach().cpu().reshape(-1).tolist()
        except KnownSpeakerError:
            raise
        except Exception as exc:
            raise KnownSpeakerError(f"SpeechBrain ECAPA inference failed: {exc}") from exc
        return _normalise_embedding(values)


def known_speaker_root(work_dir: Path) -> Path:
    return work_dir / "known_speakers"


def enrol_known_speaker(
    *,
    work_dir: Path,
    job_id: str,
    analysis_audio_path: Path,
    name: str,
    gender: str,
    ranges: Sequence[str],
    azure_voice: str | None = None,
    person_id: str | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Create or replace one manually verified known-speaker profile."""
    display_name = " ".join(str(name).split())
    if not display_name:
        raise ValueError("Speaker name is required")
    selected_gender = str(gender).strip().lower()
    if selected_gender not in {"male", "female"}:
        raise ValueError("gender must be male or female")
    if len(ranges) < MINIMUM_ENROLMENT_SAMPLES:
        raise ValueError(
            f"Provide at least {MINIMUM_ENROLMENT_SAMPLES} clean --range values"
        )
    stable_id = person_id or _slugify(display_name)
    if not _PERSON_ID_RE.fullmatch(stable_id):
        raise ValueError("person_id must contain only lowercase letters, digits, and underscores")
    selected_voice = azure_voice or (
        "zu-ZA-ThembaNeural" if selected_gender == "male" else "zu-ZA-ThandoNeural"
    )

    root = known_speaker_root(work_dir)
    samples_root = root / "samples" / stable_id
    profile_path = root / "profiles" / f"{stable_id}.json"
    encoder = encoder or SpeechBrainECAPAEncoder(root / "model_runtime")
    embeddings: list[list[float]] = []
    sample_records: list[dict[str, Any]] = []

    for index, value in enumerate(ranges, start=1):
        start, end = parse_timeline_range(value)
        if end - start < MINIMUM_RANGE_SECONDS:
            raise ValueError(
                f"Range {value!r} is shorter than {MINIMUM_RANGE_SECONDS:.0f} seconds"
            )
        sample_path = samples_root / f"sample_{index:02d}.wav"
        extraction = extract_pcm16_range(
            analysis_audio_path,
            sample_path,
            start_seconds=start,
            end_seconds=end,
        )
        embedding = _normalise_embedding(encoder.encode_file(sample_path))
        embeddings.append(embedding)
        sample_records.append(
            {
                "range": value,
                "start_seconds": start,
                "end_seconds": end,
                "audio_path": str(sample_path),
                "audio_sha256": _sha256(sample_path),
                "duration_seconds": extraction["duration_seconds"],
            }
        )

    dimensions = {len(value) for value in embeddings}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) < 2:
        raise KnownSpeakerError("SpeechBrain returned inconsistent speaker embeddings")
    centroid = _normalise_embedding(
        [sum(items) / len(items) for items in zip(*embeddings, strict=True)]
    )
    consistency_scores = [_cosine_similarity(item, centroid) for item in embeddings]
    profile = {
        "schema_version": KNOWN_SPEAKER_SCHEMA_VERSION,
        "person_id": stable_id,
        "display_name": display_name,
        "gender": selected_gender,
        "azure_voice": selected_voice,
        "verified": True,
        "verification_method": "manual_timeline_enrolment",
        "source_job_id": job_id,
        "model": {
            "id": SPEECHBRAIN_MODEL_ID,
            "revision": SPEECHBRAIN_MODEL_REVISION,
            "embedding_dimensions": len(centroid),
        },
        "centroid_embedding": centroid,
        "samples": sample_records,
        "sample_consistency": {
            "minimum_cosine_similarity": round(min(consistency_scores), 6),
            "mean_cosine_similarity": round(
                sum(consistency_scores) / len(consistency_scores), 6
            ),
        },
    }
    atomic_write_json(profile_path, profile)
    registry = _load_registry(root / "registry.json")
    registry["speakers"][stable_id] = {
        "person_id": stable_id,
        "display_name": display_name,
        "gender": selected_gender,
        "azure_voice": selected_voice,
        "profile_path": str(profile_path),
        "verified": True,
        "model_id": SPEECHBRAIN_MODEL_ID,
        "model_revision": SPEECHBRAIN_MODEL_REVISION,
    }
    atomic_write_json(root / "registry.json", registry)
    return {
        "enrolled": True,
        "person_id": stable_id,
        "display_name": display_name,
        "gender": selected_gender,
        "azure_voice": selected_voice,
        "sample_count": len(sample_records),
        "profile_path": str(profile_path),
        "registry_path": str(root / "registry.json"),
        "model_id": SPEECHBRAIN_MODEL_ID,
        "model_revision": SPEECHBRAIN_MODEL_REVISION,
    }


def identify_known_speaker(
    speaker_audio_path: Path,
    *,
    work_dir: Path,
    encoder: Any | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = DEFAULT_MATCH_MARGIN,
) -> KnownSpeakerMatch | None:
    """Return one unambiguous verified registry match, otherwise ``None``."""
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= margin <= 1.0:
        raise ValueError("threshold and margin must be between 0 and 1")
    root = known_speaker_root(work_dir)
    registry = _load_registry(root / "registry.json")
    candidates = [
        value
        for value in registry["speakers"].values()
        if isinstance(value, Mapping)
        and value.get("verified") is True
        and value.get("model_id") == SPEECHBRAIN_MODEL_ID
        and value.get("model_revision") == SPEECHBRAIN_MODEL_REVISION
    ]
    if not candidates:
        return None
    encoder = encoder or SpeechBrainECAPAEncoder(root / "model_runtime")
    probe = _normalise_embedding(encoder.encode_file(speaker_audio_path))
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for item in candidates:
        profile_path = Path(str(item.get("profile_path") or ""))
        if not profile_path.is_file():
            continue
        profile = read_json(profile_path)
        centroid = profile.get("centroid_embedding")
        if not isinstance(centroid, list) or len(centroid) != len(probe):
            continue
        scored.append((_cosine_similarity(probe, centroid), item))
    if not scored:
        return None
    scored.sort(key=lambda value: value[0], reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None
    if best_score < threshold:
        return None
    if runner_up is not None and best_score - runner_up < margin:
        return None
    return KnownSpeakerMatch(
        person_id=str(best["person_id"]),
        display_name=str(best["display_name"]),
        gender=str(best["gender"]),
        azure_voice=str(best["azure_voice"]),
        score=best_score,
        runner_up_score=runner_up,
        threshold=threshold,
        margin=margin,
        model_id=SPEECHBRAIN_MODEL_ID,
        model_revision=SPEECHBRAIN_MODEL_REVISION,
    )


def parse_timeline_range(value: str) -> tuple[float, float]:
    try:
        raw_start, raw_end = str(value).split("-", 1)
        start = _parse_timestamp(raw_start)
        end = _parse_timestamp(raw_end)
    except Exception as exc:
        raise ValueError(
            f"Invalid timeline range {value!r}; use HH:MM:SS-HH:MM:SS"
        ) from exc
    if end <= start:
        raise ValueError(f"Timeline range must end after it starts: {value!r}")
    return start, end


def extract_pcm16_range(
    source_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with wave.open(str(source_path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise KnownSpeakerError("Enrolment audio must be mono PCM16 WAV")
        sample_rate = source.getframerate()
        if sample_rate != 16000:
            raise KnownSpeakerError("Enrolment audio must be 16kHz")
        total_frames = source.getnframes()
        start_frame = max(0, round(start_seconds * sample_rate))
        end_frame = min(total_frames, round(end_seconds * sample_rate))
        if end_frame <= start_frame:
            raise KnownSpeakerError("Timeline range is outside the source audio")
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.partial")
        with wave.open(str(temporary), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(sample_rate)
            destination.writeframes(frames)
        temporary.replace(output_path)
    return {
        "sample_rate": sample_rate,
        "duration_seconds": round((end_frame - start_frame) / sample_rate, 6),
    }


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": KNOWN_SPEAKER_SCHEMA_VERSION,
            "speakers": {},
        }
    value = read_json(path)
    if (
        value.get("schema_version") != KNOWN_SPEAKER_SCHEMA_VERSION
        or not isinstance(value.get("speakers"), dict)
    ):
        raise KnownSpeakerError(f"Invalid known-speaker registry: {path}")
    return value


def _parse_timestamp(value: str) -> float:
    parts = [float(item) for item in value.strip().split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0.0
        minutes, seconds = parts
    else:
        raise ValueError("timestamp must contain minutes and seconds")
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("invalid timestamp")
    return hours * 3600 + minutes * 60 + seconds


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Speaker name cannot form a stable person_id")
    return slug[:80]


def _normalise_embedding(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise KnownSpeakerError("Speaker embedding contains invalid values")
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1e-12:
        raise KnownSpeakerError("Speaker embedding has zero magnitude")
    return [value / norm for value in result]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise KnownSpeakerError("Speaker embedding dimensions do not match")
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _load_pcm16_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise KnownSpeakerError("Speaker audio must be mono PCM16 WAV")
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    return [value / 32768.0 for value in samples], sample_rate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_MATCH_MARGIN",
    "DEFAULT_MATCH_THRESHOLD",
    "KNOWN_SPEAKER_SCHEMA_VERSION",
    "KnownSpeakerError",
    "KnownSpeakerMatch",
    "SPEECHBRAIN_MODEL_ID",
    "SPEECHBRAIN_MODEL_REVISION",
    "SpeechBrainECAPAEncoder",
    "enrol_known_speaker",
    "identify_known_speaker",
    "parse_timeline_range",
]
