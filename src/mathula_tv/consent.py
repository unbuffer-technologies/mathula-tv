"""Versioned voice-rights review artifacts; consent is never inferred."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json


RIGHTS_STATUSES = {"approved", "restricted", "denied", "unknown", "pending_review"}


def build_voice_rights_review(
    speakers: list[dict[str, Any]],
    *,
    intended_use: str,
    proof_of_concept: bool = False,
) -> dict[str, Any]:
    records = []
    for speaker in speakers:
        status = str(speaker.get("consent_or_rights_status") or "unknown")
        if status not in RIGHTS_STATUSES:
            raise ValueError(f"Unknown rights status for {speaker.get('speaker_id')}")
        records.append(
            {
                "speaker_id": speaker["speaker_id"],
                "reference_source": speaker.get("reference_source"),
                "intended_use": intended_use,
                "consent_or_rights_status": status,
                "reviewer": speaker.get("reviewer"),
                "review_timestamp": speaker.get("review_timestamp"),
                "restrictions": list(speaker.get("restrictions", [])),
                "expiry_or_rereview_date": speaker.get("expiry_or_rereview_date"),
                "notes": list(speaker.get("notes", [])),
            }
        )
    denied = [item["speaker_id"] for item in records if item["consent_or_rights_status"] == "denied"]
    missing = [item["speaker_id"] for item in records if not _has_approval_evidence(item)]
    generation_allowed = not denied and (not missing or proof_of_concept)
    return {
        "schema_version": "voice-rights-review-v1",
        "intended_use": intended_use,
        "proof_of_concept": proof_of_concept,
        "generation_allowed": generation_allowed,
        "publication_approved": not missing and not denied,
        "speakers": records,
        "unapproved_speakers": missing,
        "human_review_required": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def require_generation_rights(review: dict[str, Any]) -> None:
    records = review.get("speakers")
    if not isinstance(records, list) or not records:
        raise PermissionError("Voice-rights review contains no speaker records")
    denied = [item.get("speaker_id") for item in records if item.get("consent_or_rights_status") == "denied"]
    missing = [item.get("speaker_id") for item in records if not _has_approval_evidence(item)]
    proof_of_concept = bool(review.get("proof_of_concept"))
    if denied or (missing and not proof_of_concept) or not review.get("generation_allowed"):
        raise PermissionError(
            "Voice-use metadata is absent or unapproved; rerun only with explicit proof-of-concept mode"
        )
    if review.get("publication_approved") and missing:
        raise PermissionError("Voice-rights artifact claims publication approval without review evidence")


def write_voice_rights_review(path: Path, review: dict[str, Any]) -> None:
    require_generation_rights(review)
    atomic_write_json(path, review)


def _has_approval_evidence(record: dict[str, Any]) -> bool:
    if record.get("consent_or_rights_status") != "approved":
        return False
    if not str(record.get("reviewer") or "").strip():
        return False
    try:
        reviewed_at = datetime.fromisoformat(str(record.get("review_timestamp") or ""))
    except ValueError:
        return False
    return reviewed_at.tzinfo is not None
