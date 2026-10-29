"""Speakers who share a Zulu voice are told apart by pitch, and that has to follow how they really sound.

Azure has one male Zulu voice (Themba) and one female (Thando), so two men in a clip are the same voice with a different pitch setting
(``direct_azure_persona``: the speaker whose source pitch is lower gets the lower setting, within +-10%).  That separation needs each speaker's
median pitch and pace.  A job whose voices were resolved from cache or by the classifier alone has no such figures in
``voice_resolution.json``, so the speakers were ordered by a hash of their ids: in clip 13eacc6f the anchor (158 Hz) was set +6% and the rally
speaker (215 Hz) -6%, backwards, and the two were heard as one voice.

The delivery sheet (``delivery``) measures both from the audio.  ``refresh`` copies them into the voice resolution where they are missing and
works the personas out again; the speech rate stays locked at 0%, as everywhere in the native dub.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_write_json, read_json
from .direct_azure_persona import apply_direct_azure_personas


def features_from_sheet(sheet: Mapping[str, Any], speaker_id: str) -> dict[str, float]:
    """The pitch and pace of one speaker as the persona code names them, from the delivery sheet; empty when the sheet has none."""
    row = (sheet.get("speakers") or {}).get(speaker_id) or {}
    out: dict[str, float] = {}
    pitch = row.get("pitch_hz_usual")
    if isinstance(pitch, (int, float)) and pitch > 0:
        out["median_f0_hz"] = float(pitch)
    pace = (row.get("pace_wps") or {}).get("usual")
    if isinstance(pace, (int, float)) and pace > 0:
        out["source_speech_rate_wps"] = float(pace)
    return out


def refresh(voice_resolution_path: Path, sheet: Mapping[str, Any]) -> dict[str, Any] | None:
    """Fill the missing pitch and pace of each speaker from ``sheet`` and recompute the personas.  Writes the file and returns
    {speaker: {"pitch_percent_before", "pitch_percent_after"}}, or None when nothing was missing (the file is left as it is)."""
    path = Path(voice_resolution_path)
    data = read_json(path)
    assignments = data.get("assignments")
    if not isinstance(assignments, dict) or len(assignments) < 2:
        return None                                   # one speaker needs no separation
    filled = False
    prepared: dict[str, dict[str, Any]] = {}
    for speaker, assignment in assignments.items():
        item = dict(assignment)
        features = dict(item.get("source_features") or {})
        for key, value in features_from_sheet(sheet, str(speaker)).items():
            if key not in features:
                features[key] = value
                filled = True
        item["source_features"] = features
        prepared[str(speaker)] = item
    if not filled:
        return None
    before = {s: int((a.get("base_prosody") or {}).get("pitch_percent") or 0) for s, a in assignments.items()}
    personas = apply_direct_azure_personas(prepared)
    for speaker, item in personas.items():
        base = dict(item.get("base_prosody") or {})
        base["legacy_persona_rate_percent_ignored"] = int(base.get("rate_percent") or 0)
        base["rate_percent"] = 0
        base["native_rate_policy"] = "hard_locked_zero_percent"
        base["source_features_from"] = "delivery_sheet"
        item["base_prosody"] = base
    data["assignments"] = personas
    if isinstance(data.get("speakers"), list):
        data["speakers"] = [({**dict(row), **personas[str(row.get("speaker_id"))]} if isinstance(row, Mapping) and str(row.get("speaker_id")) in personas else row)
                            for row in data["speakers"]]
    atomic_write_json(path, data)
    return {s: {"pitch_percent_before": before.get(s, 0), "pitch_percent_after": int((personas[s].get("base_prosody") or {}).get("pitch_percent") or 0)}
            for s in personas}
