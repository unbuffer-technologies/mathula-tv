"""Structured, secret-safe errors used by the production dubbing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging_utils import redact


@dataclass(frozen=True)
class ErrorDescriptor:
    code: str
    stage: str
    retryable: bool
    action: str


class PipelineError(RuntimeError):
    """An actionable error whose serialized form is safe for job artifacts."""

    descriptor = ErrorDescriptor("pipeline_error", "unknown", False, "Inspect the job report")

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        self.safe_message = str(redact(message))[:500]
        self.details = redact(details or {})
        super().__init__(self.safe_message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.descriptor.code,
            "stage": self.descriptor.stage,
            "retryable": self.descriptor.retryable,
            "message": self.safe_message,
            "action": self.descriptor.action,
            "details": self.details,
        }


def _error(name: str, code: str, stage: str, retryable: bool, action: str) -> type[PipelineError]:
    return type(name, (PipelineError,), {"descriptor": ErrorDescriptor(code, stage, retryable, action)})


MissingSourceMedia = _error("MissingSourceMedia", "missing_source_media", "source", False, "Restore the original source media")
InvalidSourceMedia = _error("InvalidSourceMedia", "invalid_source_media", "source", False, "Provide a valid audio/video source")
UnsupportedSourceDuration = _error("UnsupportedSourceDuration", "unsupported_source_duration", "source", False, "Use a source no longer than 240 minutes")
AzureSTTAuthenticationFailure = _error("AzureSTTAuthenticationFailure", "azure_stt_authentication", "analysis", False, "Check Azure Speech credentials and region")
AzureSTTRateLimit = _error("AzureSTTRateLimit", "azure_stt_rate_limit", "analysis", True, "Retry after the provider delay")
AzureSTTMalformedResponse = _error("AzureSTTMalformedResponse", "azure_stt_malformed_response", "analysis", True, "Retry and retain the raw provider response")
ClaudeAuthenticationFailure = _error("ClaudeAuthenticationFailure", "claude_authentication", "translation", False, "Check ANTHROPIC_API_KEY")
ClaudeRateLimit = _error("ClaudeRateLimit", "claude_rate_limit", "translation", True, "Retry after the provider delay")
ClaudeTimeout = _error("ClaudeTimeout", "claude_timeout", "translation", True, "Retry within the configured job limit")
ClaudeInvalidStructuredOutput = _error("ClaudeInvalidStructuredOutput", "claude_invalid_output", "translation", False, "Review the stored validation summary")
ClaudeRefusal = _error("ClaudeRefusal", "claude_refusal", "translation", False, "Send the unit for human translation review")
TranslationEntityMismatch = _error("TranslationEntityMismatch", "translation_entity_mismatch", "translation", False, "Repair protected entities or request human review")
TranslationTimingFailure = _error("TranslationTimingFailure", "translation_timing_failure", "translation", False, "Repair only the affected dubbing unit")
AzureTTSVoiceUnavailable = _error("AzureTTSVoiceUnavailable", "azure_tts_voice_unavailable", "azure_tts", False, "Run voice discovery and choose an available zu-ZA voice")
AzureTTSCancellation = _error("AzureTTSCancellation", "azure_tts_cancelled", "azure_tts", True, "Inspect the cancellation category and retry if transient")
AzureTTSMalformedWAV = _error("AzureTTSMalformedWAV", "azure_tts_malformed_wav", "azure_tts", True, "Regenerate the affected unit")
MissingSameSpeakerReference = _error("MissingSameSpeakerReference", "missing_same_speaker_reference", "references", False, "Provide clean audio from the same job-local speaker")
InsufficientReferenceQuality = _error("InsufficientReferenceQuality", "insufficient_reference_quality", "references", False, "Select a longer or cleaner same-speaker reference")
OpenVoiceDependencyUnavailable = _error("OpenVoiceDependencyUnavailable", "openvoice_dependency_unavailable", "voice_conversion", False, "Use the pinned Colab or Kaggle GPU runtime")
OpenVoiceCheckpointMissing = _error("OpenVoiceCheckpointMissing", "openvoice_checkpoint_missing", "voice_conversion", False, "Materialize the pinned OpenVoice checkpoints")
OpenVoiceCheckpointHashMismatch = _error("OpenVoiceCheckpointHashMismatch", "openvoice_checkpoint_hash_mismatch", "voice_conversion", False, "Discard and re-download the pinned checkpoint")
CUDAUnavailable = _error("CUDAUnavailable", "cuda_unavailable", "voice_conversion", False, "Select a CUDA GPU runtime")
CUDAOutOfMemory = _error("CUDAOutOfMemory", "cuda_out_of_memory", "voice_conversion", True, "Restart the worker or use a larger GPU")
LeaseLost = _error("LeaseLost", "lease_lost", "worker", True, "Stop the stale worker and reclaim after expiry")
GCSPreconditionFailure = _error("GCSPreconditionFailure", "gcs_precondition_failure", "storage", True, "Reload the current generation before retrying")
GCSHashMismatch = _error("GCSHashMismatch", "gcs_hash_mismatch", "storage", False, "Reject the object and repeat verified promotion")
PartialPromotion = _error("PartialPromotion", "partial_promotion", "storage", True, "Resume only missing verified artifacts")
InvalidWAV = _error("InvalidWAV", "invalid_wav", "audio", False, "Regenerate a mono PCM16 WAV with the required rate")
TimingBoundExceeded = _error("TimingBoundExceeded", "timing_bound_exceeded", "alignment", False, "Repair the translation for the affected unit")
CompatibilityGateFailure = _error("CompatibilityGateFailure", "compatibility_gate_failure", "compatibility", False, "Resolve the reported quality gate and repeat human review")
BackgroundStemUnavailable = _error("BackgroundStemUnavailable", "background_stem_unavailable", "background", False, "Provide a clean stem or configure a separation provider")
ResidualEnglishDetected = _error("ResidualEnglishDetected", "residual_english_detected", "background", False, "Improve speech removal before publication review")
LoudnessFailure = _error("LoudnessFailure", "loudness_failure", "mix", False, "Remix to the configured loudness and true-peak targets")
RenderFailure = _error("RenderFailure", "render_failure", "render", True, "Inspect ffmpeg validation and retry")
QCFailure = _error("QCFailure", "qc_failure", "review", False, "Resolve the failed QC checks")


__all__ = [name for name, value in list(globals().items()) if isinstance(value, type) and issubclass(value, PipelineError)]
