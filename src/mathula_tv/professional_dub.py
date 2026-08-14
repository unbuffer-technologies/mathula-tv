"""Editorially safe revision loop for professional target-language dubbing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


PROFESSIONAL_DUB_POLICY_VERSION = (
    "mathula-professional-dub-policy-v4-island-pauses-v13.18.42"
)
REVISION_REQUEST_SCHEMA_VERSION = "mathula-translation-revision-request-v1"
REVISION_DECISION_SCHEMA_VERSION = "mathula-translation-revision-decision-v1"
AUTO_REVISION_SCHEMA_VERSION = "mathula-auto-approved-cadence-revision-v1"


_LOW_RISK_CADENCE_REVISIONS: dict[str, dict[str, str]] = {
    "zu": {
        "siyabonga kakhulu ngesikhathi sakho": "Siyabonga.",
        "ngiyabonga kakhulu": "Ngiyabonga.",
    },
}


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", text))


_ATTRIBUTION_MARKERS = {
    "kusolwa",
    "okusolwa",
    "uthi",
    "uthe",
    "ngokusho",
    "kungenzeka",
    "kulindeleke",
    "alleged",
    "according to",
    "reportedly",
    "may",
    "might",
}


def _present_markers(text: str) -> tuple[str, ...]:
    folded = text.casefold()
    return tuple(sorted(marker for marker in _ATTRIBUTION_MARKERS if marker in folded))


@dataclass(frozen=True)
class ProfessionalDubPolicy:
    maximum_rate_delta_percent: int = 5
    maximum_whole_voice_change: float = 1.03
    maximum_cadence_repair_rate_delta_percent: int = 15
    maximum_cadence_repair_voice_change: float = 1.06
    preferred_transition_room_ms: int = 500
    maximum_added_pause_ms: int = 3000
    maximum_unexplained_trailing_room_ms: int = 750
    renderer_may_rewrite_text: bool = True
    wording_revision_allowed: bool = True
    automatic_semantic_rephrasing_allowed: bool = True
    pause_distribution_allowed: bool = True
    semantic_equivalence_required: bool = True
    preserve_facts: bool = True
    preserve_attribution: bool = True
    preserve_uncertainty: bool = True
    preserve_protected_terms: bool = True
    preserve_numbers: bool = True
    auto_approve_low_risk_acknowledgements: bool = True
    unrestricted_natural_code_switching: bool = True
    translation_purity_required: bool = False
    source_speed_and_tone_priority: bool = True
    version: str = PROFESSIONAL_DUB_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranslationRevision:
    revision_id: str
    block_id: str
    phrase_id: str | None
    before: str
    after: str
    reason: str
    semantic_equivalence_confirmed: bool
    protected_terms: tuple[str, ...] = ()
    reviewer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["protected_terms"] = list(self.protected_terms)
        return value


def validate_translation_revision(
    revision: TranslationRevision,
    *,
    policy: ProfessionalDubPolicy | None = None,
) -> dict[str, Any]:
    """Validate hard editorial invariants without demanding identical wording."""

    policy = policy or ProfessionalDubPolicy()
    issues: list[dict[str, Any]] = []
    before = revision.before.strip()
    after = revision.after.strip()
    if not before or not after:
        issues.append({"code": "empty_revision_text"})
    if policy.semantic_equivalence_required and not revision.semantic_equivalence_confirmed:
        issues.append({"code": "semantic_equivalence_unconfirmed"})
    if policy.preserve_protected_terms:
        missing = [
            term
            for term in revision.protected_terms
            if term.casefold() not in after.casefold()
        ]
        if missing:
            issues.append({"code": "protected_terms_removed", "terms": missing})
    if policy.preserve_numbers:
        before_numbers = _numbers(before)
        after_numbers = _numbers(after)
        if before_numbers != after_numbers:
            issues.append(
                {
                    "code": "numbers_changed",
                    "before": list(before_numbers),
                    "after": list(after_numbers),
                }
            )
    if policy.preserve_attribution or policy.preserve_uncertainty:
        before_markers = _present_markers(before)
        missing_markers = [
            marker for marker in before_markers if marker not in after.casefold()
        ]
        if missing_markers:
            issues.append(
                {
                    "code": "attribution_or_uncertainty_removed",
                    "markers": missing_markers,
                }
            )
    return {
        "schema_version": REVISION_DECISION_SCHEMA_VERSION,
        "revision_id": revision.revision_id,
        "accepted": not issues,
        "issues": issues,
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
        "wording_changed": before != after,
        "policy": policy.to_dict(),
    }


def auto_approve_low_risk_cadence_revision(
    *,
    block_id: str,
    text: str,
    target_language: str,
) -> dict[str, Any] | None:
    """Return an audited deterministic shortening for non-substantive dialogue.

    The allowlist is deliberately locale-specific and exact-match only. News
    claims, names, numbers, attribution and arbitrary model-generated rewrites
    can therefore never enter this path.
    """

    language = target_language.split("-", 1)[0].casefold()
    normalized = re.sub(r"[.!?]+$", "", text.strip()).casefold()
    replacement = _LOW_RISK_CADENCE_REVISIONS.get(language, {}).get(normalized)
    if replacement is None or replacement == text.strip():
        return None
    revision = TranslationRevision(
        revision_id=f"auto_{block_id}",
        block_id=block_id,
        phrase_id=None,
        before=text.strip(),
        after=replacement,
        reason="Low-risk acknowledgement shortened to fit natural cadence",
        semantic_equivalence_confirmed=True,
        reviewer="deterministic_low_risk_policy",
    )
    validation = validate_translation_revision(revision)
    if not validation["accepted"]:
        return None
    return {
        "schema_version": AUTO_REVISION_SCHEMA_VERSION,
        "decision": "auto_approved",
        "scope": "low_risk_conversational_acknowledgement",
        "revision": revision.to_dict(),
        "validation": validation,
    }


def build_translation_revision_request(
    *,
    job_id: str,
    translation_path: str,
    translation_sha256: str,
    blocks: Sequence[Mapping[str, Any]],
    policy: ProfessionalDubPolicy | None = None,
) -> dict[str, Any]:
    """Create precise repair targets from failed perceptual cadence evidence."""

    policy = policy or ProfessionalDubPolicy()
    requests: list[dict[str, Any]] = []
    for block in blocks:
        cadence_qc = block.get("cadence_qc") or {}
        if cadence_qc.get("passed") is True:
            continue
        phrases = list((block.get("cadence_plan") or {}).get("phrases") or [])
        requests.append(
            {
                "request_id": f"revision_{len(requests) + 1:04d}",
                "block_id": str(block.get("block_id")),
                "segment_ids": [str(value) for value in block.get("segment_ids") or []],
                "speaker_id": str(block.get("speaker_id")),
                "current_text": str(block.get("translated_text") or ""),
                "tts_text": str(block.get("tts_text") or ""),
                "cadence_issues": list(cadence_qc.get("issues") or []),
                "phrase_targets": [
                    {
                        "phrase_id": phrase.get("phrase_id"),
                        "text": phrase.get("text"),
                        "start_ms": phrase.get("start_ms"),
                        "end_ms": phrase.get("end_ms"),
                    }
                    for phrase in phrases
                ],
                "allowed_change": (
                    "Revise wording for natural delivery while preserving every fact, "
                    "number, protected entity, attribution, uncertainty and editorial intent."
                ),
                "forbidden_change": (
                    "Do not remove or invent claims, strengthen allegations, change numbers, "
                    "translate protected English spans, or alter speaker identity."
                ),
                "renderer_rewrite_allowed": False,
            }
        )
    artifact = {
        "schema_version": REVISION_REQUEST_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "job_id": job_id,
        "source_translation": {
            "path": translation_path,
            "sha256": translation_sha256,
        },
        "status": "revision_required" if requests else "not_required",
        "request_count": len(requests),
        "requests": requests,
    }
    artifact["artifact_sha256"] = _sha256(artifact)
    return artifact


def revision_history_entry(
    *,
    previous_translation_sha256: str,
    revised_translation_sha256: str,
    revisions: Iterable[TranslationRevision],
) -> dict[str, Any]:
    values = [revision.to_dict() for revision in revisions]
    return {
        "schema_version": "mathula-translation-revision-history-v1",
        "previous_translation_sha256": previous_translation_sha256,
        "revised_translation_sha256": revised_translation_sha256,
        "revision_count": len(values),
        "revisions": values,
    }


__all__ = [
    "PROFESSIONAL_DUB_POLICY_VERSION",
    "REVISION_DECISION_SCHEMA_VERSION",
    "REVISION_REQUEST_SCHEMA_VERSION",
    "ProfessionalDubPolicy",
    "TranslationRevision",
    "auto_approve_low_risk_cadence_revision",
    "build_translation_revision_request",
    "revision_history_entry",
    "validate_translation_revision",
]
