from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ``synthesis_*`` and ``translating`` are retained only so existing manifests
# remain readable and can be migrated. New production work uses the explicit
# Azure TTS, OpenVoice, compatibility, alignment, and background states.
PRODUCTION_STATES = [
    "created",
    "audio_prepared",
    "uploaded",
    "analysis_queued",
    "analysis_running",
    "analysis_ready",
    "translation_running",
    "translation_ready",
    "azure_tts_queued",
    "azure_tts_running",
    "azure_tts_ready",
    "voice_conversion_queued",
    "voice_conversion_running",
    "voice_conversion_ready",
    "compatibility_review",
    "alignment_running",
    "alignment_ready",
    "background_processing",
    "background_ready",
    "mix_ready",
    "rendering",
    "review_ready",
]
LEGACY_STATES = ["translating", "synthesis_queued", "synthesis_claimed", "synthesizing", "synthesis_ready"]
STATES = PRODUCTION_STATES + LEGACY_STATES
TERMINAL_OR_ERROR = {"failed", "failed_retryable", "failed_terminal", "cancelled"}

ALLOWED: dict[str, set[str]] = {
    "created": {"audio_prepared"},
    "audio_prepared": {"uploaded"},
    "uploaded": {"analysis_queued", "analysis_running"},
    "analysis_queued": {"analysis_running"},
    "analysis_running": {"analysis_ready"},
    "analysis_ready": {"translation_running", "translating"},
    "translation_running": {"translation_ready"},
    "translating": {"translation_ready"},
    "translation_ready": {"azure_tts_queued", "synthesis_queued"},
    "azure_tts_queued": {"azure_tts_running"},
    "azure_tts_running": {"azure_tts_ready"},
    "azure_tts_ready": {"azure_tts_queued", "voice_conversion_queued", "alignment_running"},
    "voice_conversion_queued": {"voice_conversion_running"},
    "voice_conversion_running": {"voice_conversion_ready"},
    "voice_conversion_ready": {"compatibility_review"},
    "compatibility_review": {"alignment_running"},
    "alignment_running": {"alignment_ready"},
    "alignment_ready": {"background_processing"},
    "background_processing": {"background_ready"},
    # The production pipeline verifies and writes the final mix before rendering.
    # Keep the direct rendering transition for legacy orchestrators that still
    # combine mixing and rendering in one stage.
    "background_ready": {"mix_ready", "rendering"},
    "mix_ready": {"rendering"},
    "rendering": {"review_ready"},
    "review_ready": set(),
    # Legacy rollback path. Production dispatch never selects a legacy backend.
    "synthesis_queued": {"synthesis_claimed", "azure_tts_queued"},
    "synthesis_claimed": {"synthesizing"},
    "synthesizing": {"synthesis_ready"},
    "synthesis_ready": {"rendering"},
}
for _state in STATES:
    ALLOWED[_state].update(TERMINAL_OR_ERROR)
ALLOWED["failed_retryable"] = set(STATES) | {"failed", "failed_terminal", "cancelled"}
ALLOWED["failed"] = set()
ALLOWED["failed_terminal"] = set()
ALLOWED["cancelled"] = set()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobManifest:
    job_id: str
    source_filename: str
    local_source_path: str
    source_checksum: str
    source_duration: float
    source_language: str = "en-ZA"
    target_language: str = "zu-ZA"
    schema_version: str = "job-manifest-v2"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    state: str = "created"
    completed_stages: list[str] = field(default_factory=list)
    attempt_counters: dict[str, int] = field(default_factory=dict)
    last_error: dict[str, Any] | None = None
    objects: dict[str, str] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)
    human_review_flags: list[str] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    compatibility_gate: dict[str, Any] = field(default_factory=lambda: {"state": "pending"})
    review_readiness: str = "compatibility_pending"
    consent_review: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        return cls(**data)

    def transition(self, target: str) -> None:
        if target == self.state:
            return
        if target not in ALLOWED.get(self.state, set()):
            raise ValueError(f"Invalid job transition: {self.state} -> {target}")
        self.state, self.updated_at = target, utcnow()
