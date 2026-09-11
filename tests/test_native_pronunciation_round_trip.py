"""Phase 18 of the --candidate-pool architecture (approved plan at
C:\\Users\\mokgethwa\\.claude\\plans\\calm-wiggling-parasol.md): automated
TTS+STT round-trip verification of Phase 17's web-researched pronunciation
corrections.

Motivation, confirmed by a real production failure this session: Phase 17's
web research reasons from text/web sources only -- it never listens to what
the actual zu-ZA Azure voice produces. A real accepted correction for "Brown
Mogotsi" explicitly decided the Setswana-origin surname "retains its local
pronunciation" (no respelling needed), and that textual judgment was
acoustically wrong for both configured voices -- only caught because a human
listened to the render. This phase closes that gap with a real TTS+STT round
trip on a short carrier phrase, scored via a best-matching-substring ratio
against the canonical name (a naive whole-string ratio was tried first and
rejected -- proven, on this session's own real Justice College STT data, to
be statistically unable to separate a garbled recovery from a clean one).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv import native_dub as nd
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    with_web_researched_organisation_pronunciations,
)


# --- _pronunciation_round_trip_score -------------------------------------------------


def test_score_separates_a_clean_recovery_from_a_garbled_one():
    # Real Justice College STT data from this session: unmodified spelling
    # recovered a garbled "Chastique Koleke"; the fixed respelling recovered
    # clean, literal "Justice College".
    garbled = nd._pronunciation_round_trip_score(
        "kukhulunywa ngo chastique koleke kulesi sigaba", "Justice College",
    )
    clean = nd._pronunciation_round_trip_score(
        "kukhulunywa ngo justice college kulesi sigaba", "Justice College",
    )
    assert clean > garbled
    assert clean >= nd._PRONUNCIATION_ROUND_TRIP_MIN_SCORE
    assert garbled < nd._PRONUNCIATION_ROUND_TRIP_MIN_SCORE


def test_score_is_zero_for_empty_input():
    assert nd._pronunciation_round_trip_score("", "Justice College") == 0.0
    assert nd._pronunciation_round_trip_score("something", "") == 0.0


# --- _measure_pronunciation_round_trip / _verify_pronunciation_corrections_round_trip ----


class _FakeRoundTripTts:
    default_voice = "zu-ZA-ThembaNeural"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, request, output_path, *, force=False):
        self.calls.append((request.text, request.voice))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-wav")
        return SimpleNamespace(duration_ms=1000)


class _FakeRoundTripStt:
    """Maps the exact synthesized text to a canned recovered transcript --
    real Azure STT is never called in these tests."""

    def __init__(self, transcript_by_text: dict[str, str]):
        self._transcript_by_text = transcript_by_text
        self._last_text: str | None = None

    def transcribe(self, path, *, locale):
        return {"__transcript__": self._transcript_by_text.get(self._last_text, "")}

    def synthesize_hook(self, text: str) -> None:
        self._last_text = text


@pytest.fixture(autouse=True)
def _fake_normalize(monkeypatch):
    """Bypass the real Azure response-shape validation entirely -- tests only
    care about the words a fake STT call "recovered", not Azure's wire format.
    """
    def fake_normalize(raw, **kwargs):
        text = raw.get("__transcript__", "")
        return {"words": [{"text": w} for w in text.split()]}

    monkeypatch.setattr(nd, "normalize_stt_response", fake_normalize)


def _wire_tts_stt_pair(tts: _FakeRoundTripTts, stt: _FakeRoundTripStt):
    """The fake STT needs to know which text was JUST synthesized (it has no
    real audio to inspect) -- wrap synthesize to record it for the next
    transcribe() call, keeping the fakes' interfaces exactly what real
    AzureTTSBackend/AzureFastTranscriptionBackend expose.
    """
    real_synthesize = tts.synthesize

    def synthesize(request, output_path, *, force=False):
        result = real_synthesize(request, output_path, force=force)
        stt.synthesize_hook(request.text)
        return result

    tts.synthesize = synthesize  # type: ignore[method-assign]


def test_measure_round_trip_verifies_when_candidate_recovers_better(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoJustice College kulesi sigaba.": "kukhulunywa ngo chastique koleke kulesi sigaba",
        "Kukhulunywa ngoJastis Koleji kulesi sigaba.": "kukhulunywa ngo justice college kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="jc1",
        canonical_text="Justice College", pronunciation_tts_text="Jastis Koleji",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["verified"] is True
    assert result["candidate_best_score"] > result["raw_best_score"]
    assert len(result["per_voice"]) == 1


def test_measure_round_trip_does_not_verify_when_candidate_is_no_better(tmp_path):
    tts = _FakeRoundTripTts()
    # Both variants recover equally poorly -- the candidate respelling didn't help.
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoBrown Mogotsi kulesi sigaba.": "kukhulunywa ngo brian mogotsi kulesi sigaba",
        "Kukhulunywa ngoBraun Mogotsi kulesi sigaba.": "kukhulunywa ngo brian mokotsi kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="bm1",
        canonical_text="Brown Mogotsi", pronunciation_tts_text="Braun Mogotsi",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["candidate_best_score"] < result["raw_best_score"]
    assert result["verified"] is False


def test_measure_round_trip_verifies_a_real_improvement_even_below_the_floor(tmp_path):
    # Real production case (job 33cd7b46...): "KZN" -> "Key Zed En" went from a
    # badly-wrong 0.333 to a clearly-improved-but-imperfect 0.667 -- a flat
    # "must clear 0.7" floor wrongly rejected this genuine improvement.
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoKZN kulesi sigaba.": "kukhulunywa nge zulu natal kulesi sigaba",
        "Kukhulunywa ngoKey Zed En kulesi sigaba.": "kukhulunywa ngo kzn kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="kzn1",
        canonical_text="KZN", pronunciation_tts_text="Key Zed En",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["candidate_best_score"] > result["raw_best_score"]
    assert result["verified"] is True


def test_measure_round_trip_accepts_a_tie_at_an_already_good_score(tmp_path):
    # Real production case: "modus operandi" tied raw and candidate at 0.929 --
    # a flat "+0.1 over raw" rule would reject every tie once raw is already
    # near the ceiling, since there's no room left to show a margin.
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngomodus operandi kulesi sigaba.": "kukhulunywa ngo modas operandi kulesi sigaba",
        "Kukhulunywa ngoMowdas opa-randii kulesi sigaba.": "kukhulunywa ngo modas operandi kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="mo1",
        canonical_text="modus operandi", pronunciation_tts_text="Mowdas opa-randii",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["candidate_best_score"] == result["raw_best_score"]
    assert result["verified"] is True


def test_measure_round_trip_rejects_a_tie_below_the_floor(tmp_path):
    # Real production case: "NPA" tied raw and candidate at 0.667 -- both
    # equally mediocre, and neither clears the "already good enough" floor,
    # so there's no evidence the correction is worth keeping.
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoNPA kulesi sigaba.": "kukhulunywa ngo en pee ay kulesi sigaba",
        "Kukhulunywa ngoEn Pii Ey kulesi sigaba.": "kukhulunywa ngo en pee ay kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="npa1",
        canonical_text="NPA", pronunciation_tts_text="En Pii Ey",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["candidate_best_score"] == result["raw_best_score"]
    assert result["verified"] is False


def test_measure_round_trip_tests_every_configured_voice(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoJustice College kulesi sigaba.": "garbled nonsense here",
        "Kukhulunywa ngoJastis Koleji kulesi sigaba.": "kukhulunywa ngo justice college kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="jc2",
        canonical_text="Justice College", pronunciation_tts_text="Jastis Koleji",
        voices=("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
    )
    assert len(result["per_voice"]) == 2
    assert {item["voice"] for item in result["per_voice"]} == {"zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"}


def test_measure_round_trip_skips_a_voice_that_raises_but_keeps_the_others(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoJastis Koleji kulesi sigaba.": "kukhulunywa ngo justice college kulesi sigaba",
        "Kukhulunywa ngoJustice College kulesi sigaba.": "kukhulunywa ngo justice college kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    real_transcribe = stt.transcribe
    calls = {"count": 0}

    def flaky_transcribe(path, *, locale):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Azure Speech returned an empty transcription")
        return real_transcribe(path, locale=locale)

    stt.transcribe = flaky_transcribe  # type: ignore[method-assign]

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="jc3",
        canonical_text="Justice College", pronunciation_tts_text="Jastis Koleji",
        voices=("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
    )
    # The first voice's raw call raised -- that whole voice is skipped, not fatal.
    assert len(result["per_voice"]) == 1
    assert result["verified"] is not None


def test_measure_round_trip_returns_unknown_when_every_voice_fails(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({})
    _wire_tts_stt_pair(tts, stt)

    def always_raises(path, *, locale):
        raise RuntimeError("Azure Speech returned an empty transcription")

    stt.transcribe = always_raises  # type: ignore[method-assign]

    result = nd._measure_pronunciation_round_trip(
        tts=tts, stt_backend=stt, job_root=tmp_path, entity_id="jc4",
        canonical_text="Justice College", pronunciation_tts_text="Jastis Koleji",
        voices=("zu-ZA-ThembaNeural",),
    )
    assert result["verified"] is None
    assert result["per_voice"] == []


def test_verify_pronunciation_corrections_annotates_every_accepted_correction(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoJustice College kulesi sigaba.": "kukhulunywa ngo chastique koleke kulesi sigaba",
        "Kukhulunywa ngoJastis Koleji kulesi sigaba.": "kukhulunywa ngo justice college kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    accepted = [{
        "entity_id": "jc5", "canonical_text": "Justice College",
        "pronunciation_tts_text": "Jastis Koleji", "confidence": 0.99,
    }]
    result = nd._verify_pronunciation_corrections_round_trip(
        accepted_corrections=accepted, tts=tts, stt_backend=stt,
        voices=("zu-ZA-ThembaNeural",), job_root=tmp_path,
    )
    assert len(result) == 1
    assert result[0]["round_trip_verified"] is True
    assert result[0]["round_trip_candidate_score"] > result[0]["round_trip_raw_score"]
    # The original list must never be mutated.
    assert "round_trip_verified" not in accepted[0]


def test_verify_pronunciation_corrections_passes_through_a_correction_with_no_tts_text(tmp_path):
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({})
    _wire_tts_stt_pair(tts, stt)

    accepted = [{"entity_id": "jc6", "canonical_text": "Justice College", "pronunciation_tts_text": ""}]
    result = nd._verify_pronunciation_corrections_round_trip(
        accepted_corrections=accepted, tts=tts, stt_backend=stt,
        voices=("zu-ZA-ThembaNeural",), job_root=tmp_path,
    )
    assert result == accepted
    assert tts.calls == []


# --- pronunciation.py's promotion gate ------------------------------------------------


def _correction(**overrides):
    base = {
        "canonical_text": "Brown Mogotsi", "entity_type": "person", "confidence": 0.99,
        "pronunciation_mode": "word_name", "pronunciation_confidence": 0.95,
        "pronunciation_tts_text": "Braun Mogotsi",
        "grounded_pronunciation_evidence_urls": ["https://example.com/a"],
    }
    base.update(overrides)
    return base


def test_round_trip_verified_false_blocks_promotion():
    base = PronunciationDictionary(dictionary_version="v1", language="zu-ZA")
    research = {"accepted_corrections": [_correction(round_trip_verified=False)]}
    dictionary = with_web_researched_organisation_pronunciations(base, research)
    result = dictionary.apply("uBrown Mogotsi esiza")
    assert "Braun" not in result.tts_text
    assert "Brown Mogotsi" in result.tts_text


def test_round_trip_verified_true_allows_promotion():
    base = PronunciationDictionary(dictionary_version="v1", language="zu-ZA")
    research = {"accepted_corrections": [_correction(round_trip_verified=True)]}
    dictionary = with_web_researched_organisation_pronunciations(base, research)
    result = dictionary.apply("uBrown Mogotsi esiza")
    assert "Braun Mogotsi" in result.tts_text


def test_missing_round_trip_field_defaults_to_allowed_for_backward_compatibility():
    base = PronunciationDictionary(dictionary_version="v1", language="zu-ZA")
    research = {"accepted_corrections": [_correction()]}  # no round_trip_verified key at all
    dictionary = with_web_researched_organisation_pronunciations(base, research)
    result = dictionary.apply("uBrown Mogotsi esiza")
    assert "Braun Mogotsi" in result.tts_text
