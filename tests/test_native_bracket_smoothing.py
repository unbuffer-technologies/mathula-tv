"""Bracket smoothing: real user proposal, 2026-09-10, "I'm think[ing] of a
potential smoothing call for candidates that undershoot and overshoot, by
using the candidates to create [a] middle candidate."

Confirmed real, repeated shape this session (native_phrase_0024,
native_phrase_0003): the natural translation overshoots its window, and the
ONLY available safe cut (dropping one whole clause) overshoots the
CORRECTION instead, landing as an undershoot -- with nothing measuring
inside the window. _smooth_bracketing_candidates asks the model to
interpolate between the two real, already-measured, already-literal-safe
texts, using both their real durations AND a syllable-based target (a
second real user suggestion the same session, "feed the syllable
estimations to the model") as concrete anchors.
"""
from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.native_dub import (
    _request_middle_candidate_batch,
    _smooth_bracketing_candidates,
    _timing_candidate_rank,
    DEFAULT_MAX_NATURAL_SPEED_PERCENT,
)


def _write_real_wav(path: Path, *, ms: int, framerate: int = 16000) -> None:
    nframes = int(ms * framerate / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


def _group(group_id, *, source_text, shortened_candidates=(), speaker_id="S0", span_ms=10000):
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": 0,
        "source_end_ms": span_ms, "source_span_ms": span_ms, "source_text": source_text,
        "segment_ids": [f"{group_id}_seg"], "shortened_candidates": list(shortened_candidates),
    }


def _geometry(span_ms=10000):
    return {
        "preferred_start_ms": 0, "preferred_end_ms": span_ms,
        "hard_earliest_start_ms": 0, "hard_latest_end_ms": span_ms + 40,
        "source_window_ms": span_ms, "preferred_raw_ms": int(span_ms * 1.06),
        "minimum_usable_raw_ms": max(1, span_ms - 40),
        "maximum_usable_raw_ms": int((span_ms + 40) * 1.5),
        "mouth_close_early_tolerance_ms": 40, "mouth_close_late_tolerance_ms": 40, "free_gap_ms": 0,
    }


def _winner(spoken_text, measured_ms, required_speed_percent, path, *, fit=None, rank=None):
    if rank is None:
        geometry = _geometry()
        rank = _timing_candidate_rank(
            measured_ms, geometry["source_window_ms"], geometry["preferred_raw_ms"],
            max_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
            mouth_close_early_tolerance_ms=geometry["mouth_close_early_tolerance_ms"],
            mouth_close_late_tolerance_ms=geometry["mouth_close_late_tolerance_ms"],
        )
    return {
        "candidate_id": "grok_natural", "variant_id": "natural", "translator": "grok",
        "spoken_text": spoken_text, "measured_ms": measured_ms,
        "required_speed_percent": required_speed_percent, "qa_penalty": 0, "qa_record": None,
        "requires_review": False, "path": str(path),
        "fit": (required_speed_percent <= DEFAULT_MAX_NATURAL_SPEED_PERCENT) if fit is None else fit,
        "rank": list(rank),
    }


_VOICE_ASSIGNMENTS = {"S0": {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}}}


class _FakeSmoothingTts:
    default_voice = "zu-ZA-ThandoNeural"

    def __init__(self, duration_by_text: dict[str, int]):
        self._duration_by_text = duration_by_text
        self.calls: list[str] = []

    def synthesize(self, request, output_path, *, force=False):
        self.calls.append(request.text)
        duration_ms = self._duration_by_text.get(request.text, 10000)
        return SimpleNamespace(duration_ms=duration_ms)


class _FakeMiddleCandidateProvider:
    def __init__(self, *, responses=None, error=None):
        self._responses = responses or {}
        self._error = error
        self.calls: list[dict] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens):
        self.calls.append({"operation": operation, "payload": payload})
        if self._error is not None:
            raise self._error
        units = []
        for item in payload["units"]:
            result = self._responses.get(item["unit_id"])
            if result is None:
                continue
            units.append({"unit_id": item["unit_id"], "middle_text": result})
        return SimpleNamespace(data={"units": units}, input_tokens=80, output_tokens=40, attempts=1)


# --- _request_middle_candidate_batch --------------------------------------------------


def test_request_batch_returns_middle_text_per_unit():
    provider = _FakeMiddleCandidateProvider(responses={"g1": "Umusho ophakathi."})
    by_id, usage = _request_middle_candidate_batch(
        provider=provider,
        items=[{
            "unit_id": "g1", "source_text": "A dense sentence.", "fuller_text": "Umusho omude kakhulu.",
            "fuller_measured_ms": 12000, "fuller_syllables": 12, "shorter_text": "Umusho.",
            "shorter_measured_ms": 7000, "shorter_syllables": 4, "target_window_ms": 10000,
            "target_syllables": 9, "min_syllables": 7, "max_syllables": 11,
        }],
    )
    assert by_id == {"g1": "Umusho ophakathi."}
    assert usage == {"input_tokens": 80, "output_tokens": 40, "attempts": 1}
    assert provider.calls[0]["operation"] == "native_sentence_middle_candidate_batch"
    assert provider.calls[0]["payload"]["units"][0]["target_syllables"] == 9


def test_request_batch_raises_on_missing_unit():
    provider = _FakeMiddleCandidateProvider(responses={})
    with pytest.raises(ValueError, match="missing"):
        _request_middle_candidate_batch(
            provider=provider,
            items=[{
                "unit_id": "g1", "source_text": "X", "fuller_text": "Y", "fuller_measured_ms": 1,
                "fuller_syllables": 1, "shorter_text": "Z", "shorter_measured_ms": 1, "shorter_syllables": 1,
                "target_window_ms": 1, "target_syllables": 1, "min_syllables": 1, "max_syllables": 1,
            }],
        )


# --- _smooth_bracketing_candidates: orchestration ---------------------------------------


def test_smoothing_skips_a_block_that_already_fits(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=9000)
    group = _group("g1", source_text="A plain sentence.", shortened_candidates=[])
    winner = _winner("Umusho.", 9000, 0.0, wav, fit=True)
    provider = _FakeMiddleCandidateProvider()
    tts = _FakeSmoothingTts({})

    updated, usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert provider.calls == []
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_smoothing_skips_a_block_with_no_shortened_candidates(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    group = _group("g1", source_text="A dense sentence.", shortened_candidates=[])
    winner = _winner("Umusho omude.", 12000, 20.0, wav, fit=False)
    provider = _FakeMiddleCandidateProvider()
    tts = _FakeSmoothingTts({})

    updated, _usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert provider.calls == []


def test_smoothing_skips_when_the_lightest_cut_still_overshoots(tmp_path):
    # The lightest cut, once re-measured, is STILL "compress" (not a genuine
    # undershoot bracket) -- this is not this mechanism's job; it falls
    # through to whatever runs next (last-resort in the real pipeline).
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    shortened_text = "Umusho omfushane kancane."
    group = _group(
        "g1", source_text="A dense sentence that runs long.",
        shortened_candidates=[{"dropped_clause_ids": ["c2"], "isizulu_text": shortened_text}],
    )
    winner = _winner("Umusho omude kakhulu.", 12000, 20.0, wav, fit=False)
    tts = _FakeSmoothingTts({shortened_text: 11500})  # still over the 10000ms window
    provider = _FakeMiddleCandidateProvider()

    updated, _usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert provider.calls == []  # never even requested a middle candidate


def test_smoothing_promotes_a_genuine_middle_candidate(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    shortened_text = "Umusho omfushane."
    middle_text = "Umusho ophakathi kancane."
    source_text = "A dense sentence that runs long for its window."
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[{"dropped_clause_ids": ["c2"], "isizulu_text": shortened_text}],
    )
    winner = _winner("Umusho omude kakhulu ofanele iwindi.", 12000, 20.0, wav, fit=False)
    tts = _FakeSmoothingTts({shortened_text: 7000, middle_text: 10000})  # lands exactly in the window
    provider = _FakeMiddleCandidateProvider(responses={"g1": middle_text})

    updated, usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    promoted = updated["g1"]
    assert promoted["spoken_text"] == middle_text
    assert promoted["requires_review"] is True
    assert promoted["candidate_id"] == "g1__middle_candidate"
    assert usage["attempts"] == 1
    # Confirms the real bracket anchors reached the request payload.
    request_payload = provider.calls[0]["payload"]["units"][0]
    assert request_payload["shorter_text"] == shortened_text
    assert request_payload["fuller_text"] == winner["spoken_text"]
    assert "target_syllables" in request_payload


def test_smoothing_keeps_the_original_when_the_middle_candidate_does_not_improve(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    shortened_text = "Umusho omfushane."
    middle_text = "Umusho ophakathi."
    source_text = "A dense sentence that runs long for its window."
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[{"dropped_clause_ids": ["c2"], "isizulu_text": shortened_text}],
    )
    winner = _winner("Umusho omude kakhulu ofanele iwindi.", 12000, 20.0, wav, fit=False)
    # The "middle" candidate measures WORSE than the lightest cut -- still an
    # undershoot, and no closer to the window than what was already known.
    tts = _FakeSmoothingTts({shortened_text: 7000, middle_text: 6000})
    provider = _FakeMiddleCandidateProvider(responses={"g1": middle_text})

    updated, _usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"]["spoken_text"] == winner["spoken_text"]
    assert updated["g1"]["requires_review"] is False


def test_smoothing_discards_a_middle_candidate_that_drops_a_required_literal(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    source_text = "The report cites 24 documents and runs long for its window."
    shortened_text = "Umbiko ukhomba amadokhumenti angu-24 amaningi."
    middle_text = "Umbiko ukhomba amadokhumenti amaningi."  # silently drops "24"
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[{"dropped_clause_ids": ["c2"], "isizulu_text": shortened_text}],
    )
    winner = _winner("Umbiko ukhomba amadokhumenti angu-24 amaningi kakhulu ofanele.", 12000, 20.0, wav, fit=False)
    tts = _FakeSmoothingTts({shortened_text: 7000, middle_text: 9900})
    provider = _FakeMiddleCandidateProvider(responses={"g1": middle_text})

    updated, _usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"]["spoken_text"] == winner["spoken_text"]
    assert middle_text not in tts.calls  # never even measured -- discarded before spending on TTS


def test_smoothing_degrades_gracefully_when_the_provider_call_fails(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=12000)
    shortened_text = "Umusho omfushane."
    group = _group(
        "g1", source_text="A dense sentence that runs long.",
        shortened_candidates=[{"dropped_clause_ids": ["c2"], "isizulu_text": shortened_text}],
    )
    winner = _winner("Umusho omude kakhulu.", 12000, 20.0, wav, fit=False)
    tts = _FakeSmoothingTts({shortened_text: 7000})
    provider = _FakeMiddleCandidateProvider(error=RuntimeError("boom"))

    updated, usage = _smooth_bracketing_candidates(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_smoothing_batches_every_triggered_block_into_one_call(tmp_path):
    wav1 = tmp_path / "g1.wav"
    wav2 = tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=12000)
    _write_real_wav(wav2, ms=12000)
    shortened1, shortened2 = "Umusho 1 omfushane.", "Umusho 2 omfushane."
    middle1, middle2 = "Umusho 1 ophakathi.", "Umusho 2 ophakathi."
    groups = [
        _group("g1", source_text="First dense sentence.", shortened_candidates=[
            {"dropped_clause_ids": ["c2"], "isizulu_text": shortened1},
        ]),
        _group("g2", source_text="Second dense sentence.", speaker_id="S0", shortened_candidates=[
            {"dropped_clause_ids": ["c2"], "isizulu_text": shortened2},
        ]),
    ]
    winners = {
        "g1": _winner("Umusho 1 omude kakhulu.", 12000, 20.0, wav1, fit=False),
        "g2": _winner("Umusho 2 omude kakhulu.", 12000, 20.0, wav2, fit=False),
    }
    tts = _FakeSmoothingTts({
        shortened1: 7000, shortened2: 7000, middle1: 10000, middle2: 10000,
    })
    provider = _FakeMiddleCandidateProvider(responses={"g1": middle1, "g2": middle2})

    updated, _usage = _smooth_bracketing_candidates(
        provider=provider, groups=groups, winners=winners, tts=tts,
        geometries={"g1": _geometry(), "g2": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
    )
    assert len(provider.calls) == 1  # one batched call, not one per block
    assert updated["g1"]["spoken_text"] == middle1
    assert updated["g2"]["spoken_text"] == middle2
