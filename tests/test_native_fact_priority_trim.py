"""Fact-priority trim: the sole wording-driven compaction mechanism in the
--candidate-pool architecture (Tier 1), plus the separate, later last-resort
protected-content compression mechanism (Tier 2).

Tier 1 (this file's ``_trim_by_fact_priority`` tests): shortened_candidates
are generated for free during translation as a MECHANICAL ladder built from
an explicit clause breakdown -- each segment's English content is split into
clauses (clause_id, english_text, rank), ranked by fact-priority and
story-centrality, and shortened_candidates are cumulative-prefix drops of
the lowest-ranked clauses (dropped_clause_ids -> a fresh isizulu_text for
the surviving clauses). This is a ZERO-LLM-CALL mechanism at trim time.
_filter_safe_shortened_candidates independently re-enforces the numeric/
acronym-literal half of the generation prompt's hard rule deterministically,
regardless of the model's own wording.

Design note (2026-09-03): an earlier version had the model tag raw
"droppable spans" (verbatim substrings) for mechanical removal via string
replacement. Two real production failures confirmed this is fundamentally
unsafe. A later version had the model freely author "shortened_candidates"
as complete alternative texts with no explicit ranking step, which could
(and did, on a real job) return an EMPTY ladder for a real, multi-clause
sentence with genuine droppable content -- a single-shot, no-redundancy
judgment call. This version makes clause enumeration and ranking a required
structural step, so "nothing droppable" is a conclusion reachable only
after ranking, never a way to skip it.

Tier 2 (2026-09-04, ``_compress_protected_content_last_resort`` tests): an
explicit, deliberate product decision that Mathula TV's fixed runtime is the
non-negotiable constraint, not fact-preservation -- an audibly rushed
sentence risks a viewer disengaging from the REST of the video, which is a
larger loss than an occasional, tightly-ordered compression of a normally-
protected fact. This is a SEPARATE, LATER mechanism: Tier 1's own hard rule
(never drop a name/number/date/quote/attribution/claim-denial) stays fully
intact; Tier 2 only ever runs for a block that already exhausted Tier 1's
entire ladder and still does not fit.

Refinement (2026-09-08, real user direction: "the only thing worthy of
protecting is the time frame and the story...no class should stop that"):
Tier 2's trigger is no longer one flat percentage. A block Tier 1 already
proved has nothing safer left (``ladder_exhausted_group_ids``, threaded
from ``_trim_by_fact_priority``'s own return) triggers Tier 2 at ANY
remaining overrun; a block Tier 1 never had a real shot at (nothing to
try, or turn-rebalancing residual alone) still needs the original 25% bar,
preserving the real bug that bar was calibrated against.
"""
from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.native_dub import (
    DEFAULT_MAX_NATURAL_SPEED_PERCENT,
    _MAX_SHORTENED_CANDIDATES_PER_SEGMENT,
    _canonicalize_segment_clauses,
    _canonicalize_shortened_candidates,
    _compress_protected_content_last_resort,
    _filter_safe_shortened_candidates,
    _timing_candidate_rank,
    _trim_by_fact_priority,
    _validate_turn_block_segments,
)


# --- _canonicalize_segment_clauses -------------------------------------------------------


def test_canonicalize_segment_clauses_keeps_a_real_ranked_breakdown():
    raw = [
        {"clause_id": "c1", "english_text": "Lincoln was in contact with Mogotsi", "rank": 1},
        {"clause_id": "c2", "english_text": "spoken to Mogotsi on the phone once", "rank": 3},
    ]
    result = _canonicalize_segment_clauses(raw)
    assert result == raw


def test_canonicalize_segment_clauses_rejects_empty_list():
    assert _canonicalize_segment_clauses([]) is None
    assert _canonicalize_segment_clauses(None) is None


def test_canonicalize_segment_clauses_rejects_duplicate_clause_ids():
    raw = [
        {"clause_id": "c1", "english_text": "First.", "rank": 1},
        {"clause_id": "c1", "english_text": "Second.", "rank": 2},
    ]
    assert _canonicalize_segment_clauses(raw) is None


def test_canonicalize_segment_clauses_rejects_malformed_entries():
    assert _canonicalize_segment_clauses([{"clause_id": "c1", "rank": 1}]) is None  # missing english_text
    assert _canonicalize_segment_clauses([{"clause_id": "", "english_text": "x", "rank": 1}]) is None
    assert _canonicalize_segment_clauses([{"clause_id": "c1", "english_text": "x", "rank": "not-an-int"}]) is None


# --- _canonicalize_shortened_candidates -------------------------------------------------


def test_canonicalize_shortened_candidates_keeps_only_real_alternatives_deduped():
    original = "Kwenzeka lokhu kaningi kakhulu, futhi kwaphinda kwaphindwa ngempela, base sekuqedile."
    raw = [
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Kwenzeka lokhu kaningi kakhulu, base sekuqedile."},
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Kwenzeka lokhu kaningi kakhulu, base sekuqedile."},  # dup
        {"dropped_clause_ids": ["c2"], "isizulu_text": "   "},  # blank after stripping -- dropped
        {"dropped_clause_ids": ["c2", "c3"], "isizulu_text": "Kwenzeka lokhu, base sekuqedile."},
    ]
    result = _canonicalize_shortened_candidates(raw, isizulu_text=original, clause_ids=["c1", "c2", "c3"])
    assert result == [
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Kwenzeka lokhu kaningi kakhulu, base sekuqedile."},
        {"dropped_clause_ids": ["c2", "c3"], "isizulu_text": "Kwenzeka lokhu, base sekuqedile."},
    ]


def test_canonicalize_shortened_candidates_caps_at_the_configured_maximum():
    original = "Original text here."
    candidates = [
        {"dropped_clause_ids": ["c1"], "isizulu_text": f"Alternative number {i}."}
        for i in range(_MAX_SHORTENED_CANDIDATES_PER_SEGMENT + 3)
    ]
    result = _canonicalize_shortened_candidates(candidates, isizulu_text=original, clause_ids=["c1"])
    assert len(result) == _MAX_SHORTENED_CANDIDATES_PER_SEGMENT
    assert result == candidates[:_MAX_SHORTENED_CANDIDATES_PER_SEGMENT]


def test_canonicalize_shortened_candidates_handles_none_gracefully():
    assert _canonicalize_shortened_candidates(None, isizulu_text="Abc def.") == []


def test_canonicalize_shortened_candidates_drops_a_candidate_identical_to_the_original():
    original = "Kwenzeka lokhu kaningi kakhulu."
    raw = [
        {"dropped_clause_ids": ["c1"], "isizulu_text": original},
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Kwenzeka lokhu."},
    ]
    result = _canonicalize_shortened_candidates(raw, isizulu_text=original, clause_ids=["c1"])
    assert result == [{"dropped_clause_ids": ["c1"], "isizulu_text": "Kwenzeka lokhu."}]


def test_canonicalize_shortened_candidates_drops_a_candidate_naming_an_unknown_clause_id():
    original = "Kwenzeka lokhu kaningi kakhulu."
    raw = [{"dropped_clause_ids": ["c99"], "isizulu_text": "Kwenzeka lokhu."}]
    result = _canonicalize_shortened_candidates(raw, isizulu_text=original, clause_ids=["c1", "c2"])
    assert result == []


def test_canonicalize_shortened_candidates_keeps_a_restructure_only_candidate_with_no_dropped_ids():
    # A restructure-only candidate (2026-09-05): every clause kept, dropped_clause_ids
    # deliberately empty, just a tighter/cleaner rewording of the SAME content -- a
    # real, distinct kind of candidate from a drop, not something to reject.
    original = "Kwenzeka lokhu kaningi kakhulu."
    raw = [{"dropped_clause_ids": [], "isizulu_text": "Kwenzeka lokhu njalo."}]
    result = _canonicalize_shortened_candidates(raw, isizulu_text=original, clause_ids=["c1"])
    assert result == [{"dropped_clause_ids": [], "isizulu_text": "Kwenzeka lokhu njalo."}]


def test_canonicalize_shortened_candidates_drops_a_restructure_only_candidate_identical_to_original():
    # An empty-drop candidate that doesn't actually change anything offers nothing,
    # exactly like a same-text drop candidate would.
    original = "Kwenzeka lokhu kaningi kakhulu."
    raw = [{"dropped_clause_ids": [], "isizulu_text": original}]
    result = _canonicalize_shortened_candidates(raw, isizulu_text=original, clause_ids=["c1"])
    assert result == []


# --- _validate_turn_block_segments: clauses required, shortened_candidates carried -------


def _clause(clause_id="c1", text="Kwenzeka lokhu manje.", rank=1):
    return {"clause_id": clause_id, "english_text": text, "rank": rank}


def test_validate_turn_block_segments_requires_a_real_clause_breakdown():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "clauses": [_clause()], "shortened_candidates": [],
    }]
    validated = _validate_turn_block_segments(member_count=1, segments=segments)
    assert validated is not None
    assert validated[0]["clauses"] == [_clause()]


def test_validate_turn_block_segments_rejects_a_segment_with_no_clauses():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "clauses": [], "shortened_candidates": [],
    }]
    assert _validate_turn_block_segments(member_count=1, segments=segments) is None


def test_validate_turn_block_segments_rejects_a_segment_with_missing_clauses_field():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "shortened_candidates": [],
    }]
    assert _validate_turn_block_segments(member_count=1, segments=segments) is None


def test_validate_turn_block_segments_carries_shortened_candidates_through():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje, ngempela.",
        "clauses": [_clause("c1", "Kwenzeka lokhu manje", 1), _clause("c2", "ngempela", 3)],
        "shortened_candidates": [{"dropped_clause_ids": ["c2"], "isizulu_text": "Kwenzeka lokhu manje."}],
    }]
    validated = _validate_turn_block_segments(member_count=1, segments=segments)
    assert validated is not None
    assert validated[0]["shortened_candidates"] == [
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Kwenzeka lokhu manje."}
    ]


def test_validate_turn_block_segments_carries_communicative_goal_through():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "communicative_goal": "answering when the event happened",
        "clauses": [_clause()],
    }]
    validated = _validate_turn_block_segments(member_count=1, segments=segments)
    assert validated is not None
    assert validated[0]["communicative_goal"] == "answering when the event happened"


def test_validate_turn_block_segments_defaults_missing_communicative_goal_to_empty_string():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "clauses": [_clause()],
    }]
    validated = _validate_turn_block_segments(member_count=1, segments=segments)
    assert validated is not None
    assert validated[0]["communicative_goal"] == ""


def test_validate_turn_block_segments_defaults_missing_shortened_candidates_to_empty():
    segments = [{
        "start_index": 1, "end_index": 1, "isizulu_text": "Kwenzeka lokhu manje.",
        "clauses": [_clause()],
    }]
    validated = _validate_turn_block_segments(member_count=1, segments=segments)
    assert validated is not None
    assert validated[0]["shortened_candidates"] == []


# --- _filter_safe_shortened_candidates: the deterministic literal-safety net ------------


def test_filter_safe_shortened_candidates_rejects_a_candidate_missing_a_required_literal():
    source_text = "The report cites 24 documents from the NPA archive."
    candidates = [
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Umbiko ukhomba amadokhumenti angu-24 avela ku-NPA."},
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Umbiko ukhomba amadokhumenti amaningi."},  # unsafe
    ]
    safe = _filter_safe_shortened_candidates(candidates=candidates, source_text=source_text)
    assert safe == [candidates[0]]


def test_filter_safe_shortened_candidates_keeps_order_of_survivors():
    source_text = "A plain sentence with nothing special in it."
    candidates = [
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Umusho ojwayelekile."},
        {"dropped_clause_ids": ["c1", "c2"], "isizulu_text": "Umusho."},
    ]
    safe = _filter_safe_shortened_candidates(candidates=candidates, source_text=source_text)
    assert safe == candidates


def test_filter_safe_shortened_candidates_permits_dropping_a_literal_inside_a_dropped_clause():
    # Real confirmed bug (job fb3d08b63fed4d90922b08f7e325b906, 2026-09-09):
    # a candidate that drops the reporter sign-off clause ("SABC News") was
    # rejected outright because the WHOLE segment's source_text still
    # "required" the acronym SABC -- even though dropping that exact clause
    # is the FIRST thing this file's own ranking guidance tells the model to
    # do. When clauses is supplied, a literal that lives only inside a
    # clause the candidate deliberately dropped is no longer required.
    source_text = "A report on the story. Ofentse Setimo, SABC News, eMalahleni."
    clauses = [
        {"clause_id": "c1", "english_text": "A report on the story.", "rank": 1},
        {"clause_id": "c2", "english_text": "Ofentse Setimo, SABC News, eMalahleni.", "rank": 5},
    ]
    candidates = [
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Umbiko ngendaba."},  # drops SABC deliberately
    ]
    safe = _filter_safe_shortened_candidates(candidates=candidates, source_text=source_text, clauses=clauses)
    assert safe == candidates


def test_filter_safe_shortened_candidates_still_rejects_a_missing_literal_from_a_surviving_clause():
    # The other half of the same fix: a literal that belongs to a clause the
    # candidate did NOT drop must still be required -- this isn't a blanket
    # relaxation, only a scoping fix.
    source_text = "The report cites 24 documents. Ofentse Setimo, SABC News, eMalahleni."
    clauses = [
        {"clause_id": "c1", "english_text": "The report cites 24 documents.", "rank": 1},
        {"clause_id": "c2", "english_text": "Ofentse Setimo, SABC News, eMalahleni.", "rank": 5},
    ]
    candidates = [
        # Drops the sign-off (fine) but ALSO silently lost the "24" that c1 (kept) still owns.
        {"dropped_clause_ids": ["c2"], "isizulu_text": "Umbiko ukhomba amadokhumenti amaningi."},
    ]
    safe = _filter_safe_shortened_candidates(candidates=candidates, source_text=source_text, clauses=clauses)
    assert safe == []


# --- _trim_by_fact_priority: orchestration ----------------------------------------------


def _write_real_wav(path: Path, *, ms: int, framerate: int = 16000) -> None:
    nframes = int(ms * framerate / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


def _shortened(text, dropped=("c2",)):
    return {"dropped_clause_ids": list(dropped), "isizulu_text": text}


def _group(group_id, *, source_text, shortened_candidates=(), speaker_id="S0", span_ms=3000):
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": 0,
        "source_end_ms": span_ms, "source_span_ms": span_ms, "source_text": source_text,
        "segment_ids": [f"{group_id}_seg"], "shortened_candidates": list(shortened_candidates),
    }


def _geometry(span_ms=3000):
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
        # Mirror _measure_temporal_candidate's real behavior (matched against this
        # file's default _geometry()) so selection-comparator tests actually exercise
        # rank-tuple comparison instead of trivially falling back to a shared default.
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


class _FakeTrimTts:
    default_voice = "zu-ZA-ThandoNeural"

    def __init__(self, duration_by_text: dict[str, int], *, default_ms: int = 3000):
        self._duration_by_text = duration_by_text
        self.calls: list[str] = []

    def synthesize(self, request, output_path, *, force=False):
        self.calls.append(request.text)
        duration_ms = self._duration_by_text.get(request.text, 3000)
        return SimpleNamespace(duration_ms=duration_ms)


_VOICE_ASSIGNMENTS = {"S0": {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}}}


def test_trim_skips_a_block_under_the_trigger_with_zero_tts_calls(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.", shortened_candidates=[_shortened("Umusho.")])
    winner = _winner("Umusho ojwayelekile, imininingwane engeziwe lapha.", 3000, 10.0, wav)  # below 12%
    tts = _FakeTrimTts({})

    updated, usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert tts.calls == []
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_trim_skips_a_block_with_no_shortened_candidates(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.", shortened_candidates=[])
    winner = _winner("Umusho ojwayelekile.", 6000, 80.0, wav)  # above 12%, but nothing to try
    tts = _FakeTrimTts({})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert tts.calls == []


def test_trim_skips_when_every_shortened_candidate_fails_literal_safety(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "The report cites 24 documents in total."
    spoken_text = "Umbiko ukhomba amadokhumenti angu-24 wonke."
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[_shortened("Umbiko ukhomba amadokhumenti amaningi.")],
    )
    winner = _winner(spoken_text, 6000, 80.0, wav)
    tts = _FakeTrimTts({})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert tts.calls == []


def test_trim_promotes_a_genuine_improvement_and_flags_it_for_review(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "A dense sentence that runs long for its window."
    spoken_text = "Umusho onzima kakhulu, futhi ophindaphindwe ngempela, oqhubeka isikhathi eside."
    shortened = "Umusho onzima kakhulu, oqhubeka isikhathi eside."
    group = _group("g1", source_text=source_text, shortened_candidates=[_shortened(shortened)])
    winner = _winner(spoken_text, 6000, 80.0, wav)  # above 12%, source_window_ms=3000 -> required speed high
    tts = _FakeTrimTts({shortened: 3200})  # a real, shorter measured duration once trimmed

    updated, usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    promoted = updated["g1"]
    assert promoted["spoken_text"] == shortened
    assert promoted["requires_review"] is True
    assert promoted["dropped_clause_ids"] == ["c2"]
    assert float(promoted["required_speed_percent"]) < 80.0
    assert usage["attempts"] == 0  # zero LLM calls -- everything here is real TTS + arithmetic


def test_trim_keeps_the_original_when_the_shortened_candidate_does_not_measure_better(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "A dense sentence that runs long for its window."
    spoken_text = "Umusho onzima kakhulu, futhi ophindaphindwe ngempela, oqhubeka isikhathi eside."
    shortened = "Umusho onzima kakhulu, oqhubeka isikhathi eside."
    group = _group("g1", source_text=source_text, shortened_candidates=[_shortened(shortened)])
    winner = _winner(spoken_text, 6000, 80.0, wav)
    # The shortened candidate measures LONGER than the original -- a real
    # Azure TTS call is not guaranteed to be shorter just because the text is.
    tts = _FakeTrimTts({shortened: 9000})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"]["spoken_text"] == spoken_text
    assert updated["g1"]["requires_review"] is False


def test_trim_measures_every_candidate_and_prefers_the_lightest_cut_that_fits(tmp_path):
    # Two safe candidates -- the lighter cut ALREADY fits comfortably, and the
    # heavier cut on top would also fit. The real-measurement-based selection
    # must prefer the lighter cut: never use more than what a real
    # measurement shows was actually necessary.
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "A dense sentence that runs long for its window and needs real trimming."
    spoken_text = (
        "Umusho onzima kakhulu, futhi ophindaphindwe ngempela, kanye nemininingwane "
        "engeziwe engabalulekile, oqhubeka isikhathi eside kakhulu."
    )
    light_cut = "Umusho onzima kakhulu, kanye nemininingwane engeziwe engabalulekile, oqhubeka isikhathi eside kakhulu."
    heavy_cut = "Umusho onzima kakhulu, oqhubeka isikhathi eside kakhulu."
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[_shortened(light_cut, ("c2",)), _shortened(heavy_cut, ("c2", "c3"))],
    )
    winner = _winner(spoken_text, 6000, 80.0, wav)
    # Both candidates measure comfortably within natural pace (span_ms=3000)
    # -- neither is starved for content, so the lighter cut should win.
    tts = _FakeTrimTts({light_cut: 3100, heavy_cut: 3050})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    # Both candidates were actually measured via real TTS -- no shortcut.
    assert sorted(tts.calls) == sorted([light_cut, heavy_cut])
    assert updated["g1"]["spoken_text"] == light_cut


def test_trim_falls_back_to_the_heavier_cut_when_the_lighter_one_still_does_not_fit(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "A dense sentence that runs long for its window and needs real trimming."
    spoken_text = "Umusho onzima kakhulu futhi ude kakhulu."
    light_cut = "Umusho onzima futhi ude kakhulu."
    heavy_cut = "Umusho onzima."
    group = _group(
        "g1", source_text=source_text,
        shortened_candidates=[_shortened(light_cut, ("c2",)), _shortened(heavy_cut, ("c2", "c3"))],
    )
    winner = _winner(spoken_text, 6000, 80.0, wav)
    # The light cut still doesn't fit (5500ms); only the heavy cut does (3050ms).
    tts = _FakeTrimTts({light_cut: 5500, heavy_cut: 3050})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"]["spoken_text"] == heavy_cut


def test_trim_attempts_a_block_that_is_audibly_rushed_but_far_under_the_old_50_percent_bar(tmp_path):
    # Reproduces the real regression that motivated tying the trigger to
    # DEFAULT_MAX_NATURAL_SPEED_PERCENT instead of a separate, higher bar:
    # a real job showed turn-1 blocks at 35-37% required speed -- audibly
    # rushed on real playback, but comfortably under the old 50% trigger --
    # sitting on real, unused shortened candidates. This asserts such a
    # block IS now attempted.
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=4000)
    source_text = "My colleague joins us now. Very good morning to you."
    spoken_text = "Umlingani wami usejoyina manje. Sawubona ekuseni."
    shortened = "Umlingani wami usejoyina manje."
    group = _group("g1", source_text=source_text, shortened_candidates=[_shortened(shortened)])
    winner = _winner(spoken_text, 4000, 35.0, wav)  # under the old 50% bar, over the 12% one
    tts = _FakeTrimTts({shortened: 3050})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert tts.calls == [shortened]  # it was actually attempted, not silently skipped
    assert updated["g1"]["spoken_text"] == shortened


def test_trim_still_leaves_a_block_that_already_fits_within_12_percent_untouched(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.", shortened_candidates=[_shortened("Umusho.")])
    winner = _winner("Umusho ojwayelekile, imininingwane engeziwe lapha.", 3000, 10.0, wav)  # under 12%
    tts = _FakeTrimTts({})

    updated, _usage, _exhausted = _trim_by_fact_priority(
        groups=[group], winners={"g1": winner}, tts=tts, geometries={"g1": _geometry()},
        voice_assignments=_VOICE_ASSIGNMENTS, measurement_audio_root=tmp_path,
        force=True, candidate_pool_tts_workers=1,
    )
    assert updated["g1"] is winner
    assert tts.calls == []


def test_trim_trigger_matches_the_shared_natural_speed_quality_bar():
    # Sanity guard: the trigger is DEFAULT_MAX_NATURAL_SPEED_PERCENT itself,
    # not an independent constant that could silently drift from it. Real
    # user direction, 2026-09-08 ("The rush should always be less than 10%"):
    # lowered from 12 to 10 -- this pinned value is a deliberate policy
    # choice, not an arbitrary constant, so update it only on the same kind
    # of explicit direction, never to silence this test.
    assert DEFAULT_MAX_NATURAL_SPEED_PERCENT == 10


# --- _compress_protected_content_last_resort: Tier 2, the final fallback ---------------


class _FakeLastResortProvider:
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
            units.append({"unit_id": item["unit_id"], **result})
        return SimpleNamespace(data={"units": units}, input_tokens=100, output_tokens=50, attempts=1)


def test_last_resort_skips_when_every_winner_already_fits(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.")
    winner = _winner("Umusho ojwayelekile.", 3000, 5.0, wav)  # fits comfortably
    tts = _FakeTrimTts({})
    provider = _FakeLastResortProvider()

    updated, usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert updated["g1"] is winner
    assert provider.calls == []
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_last_resort_skips_a_block_that_is_rushed_but_still_fits(tmp_path):
    # fit=True even at a real rush percentage above the natural bar (e.g. a
    # gap-aware absorption elsewhere already made it acceptable) must never
    # trigger this drastic, protected-content-touching fallback.
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.")
    winner = _winner("Umusho ojwayelekile.", 3000, 20.0, wav, fit=True)
    provider = _FakeLastResortProvider()

    updated, _usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=_FakeTrimTts({}),
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert updated["g1"] is winner
    assert provider.calls == []


def test_last_resort_skips_a_block_that_only_slightly_overshoots_the_natural_bar(tmp_path):
    # Real confirmed bug (job 8372120960474ef6b1d75af7de51a605, 2026-09-04):
    # gating this tier on the routine 12% "fits naturally" bar fired it on 8
    # individual members of one already-rebalanced turn, each sitting a mere
    # 0.2-0.6 points over 12% -- turn rebalancing spreading a turn's real
    # excess evenly leaves a small residual over 12% as NORMAL, not a sign
    # anything is genuinely stuck. This tier must only fire well above that,
    # at _PROTECTED_CONTENT_LAST_RESORT_TRIGGER_PERCENT (25%).
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.")
    winner = _winner("Umusho ojwayelekile.", 3000, 12.6, wav, fit=False)
    provider = _FakeLastResortProvider()

    updated, _usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=_FakeTrimTts({}),
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert updated["g1"] is winner
    assert provider.calls == []


def test_last_resort_fires_below_25_percent_for_a_block_that_already_exhausted_tier_1(tmp_path):
    # Real user direction, 2026-09-08: "the only thing worthy of protecting
    # is the time frame and the story...no class should stop that" -- once
    # a block has ALREADY had a real Tier 1 ladder and used the best safe
    # cut available (confirmed real case: a block landing at 24.4%, still
    # over budget, with nothing safer left to try), the 25% floor calibrated
    # for UNTRIED blocks must not also block this genuinely-stuck one.
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3732)
    group = _group("g1", source_text="A dense sentence with nothing safe left to cut.")
    # measured_ms genuinely overruns its 3000ms window (~24.4% required speed,
    # matching the real confirmed case) -- not just a fabricated percentage.
    winner = _winner("Umusho onzima.", 3732, 24.4, wav, fit=False)  # under 25%, but ladder-exhausted
    compressed_text = "Umusho."
    provider = _FakeLastResortProvider(responses={
        "g1": {"isizulu_text": compressed_text, "omitted_items": ["omitted a supporting detail"]},
    })
    # A duration that comfortably FITS the window (not merely a low
    # required_speed_percent) -- too far under would rank worse than the
    # rushed-but-atempo-rescuable baseline (real bug this file already
    # guards against: an undershoot means dead air, worse than a rush).
    tts = _FakeTrimTts({compressed_text: 3050})

    updated, usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={}, ladder_exhausted_group_ids=["g1"],
    )
    assert provider.calls  # it was actually attempted, not silently skipped
    assert updated["g1"]["spoken_text"] == compressed_text
    assert usage["attempts"] == 1


def test_last_resort_still_requires_25_percent_for_a_block_tier_1_never_tried(tmp_path):
    # The other half of the same fix: a block NOT in ladder_exhausted_group_ids
    # (never had a real ladder -- e.g. every clause ranked essential) must
    # keep the higher 25% bar exactly as before, guarding against the real,
    # confirmed bug this constant was originally calibrated for.
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=3000)
    group = _group("g1", source_text="A plain sentence.")
    winner = _winner("Umusho ojwayelekile.", 3000, 12.6, wav, fit=False)
    provider = _FakeLastResortProvider()

    updated, _usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=_FakeTrimTts({}),
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={}, ladder_exhausted_group_ids=[],
    )
    assert updated["g1"] is winner
    assert provider.calls == []


def test_last_resort_promotes_a_genuine_improvement_and_carries_omitted_items(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "I couldn't tell you if it was November of 2024 or December of 2024, but I spoke to him once."
    spoken_text = "Angikwazi ukusho ukuthi kwakunguNovemba ka-2024 noma uDisemba ka-2024, kodwa ngakhuluma naye kanye."
    group = _group("g1", source_text=source_text, span_ms=3000)
    winner = _winner(spoken_text, 6000, 80.0, wav, fit=False)
    compressed_text = "Angikwazi ukusho ukuthi kwakunguNovemba noma uDisemba 2024."
    provider = _FakeLastResortProvider(responses={
        "g1": {"isizulu_text": compressed_text, "omitted_items": ["stated the year once instead of twice"]},
    })
    tts = _FakeTrimTts({compressed_text: 3100})

    updated, usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    promoted = updated["g1"]
    assert promoted["spoken_text"] == compressed_text
    assert promoted["requires_review"] is True
    assert promoted["omitted_items"] == ["stated the year once instead of twice"]
    assert float(promoted["required_speed_percent"]) < 80.0
    assert usage["attempts"] == 1  # exactly one real LLM call, batched
    # Never runs the Tier 1 literal-safety filter -- omitting a literal is
    # exactly what this tier is allowed to do.
    assert tts.calls == [compressed_text]


def test_last_resort_keeps_the_original_when_the_compressed_candidate_does_not_improve(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    source_text = "A dense sentence that still runs long even after every safe cut."
    spoken_text = "Umusho onzima kakhulu ongakaze unciphe ngokwanele."
    group = _group("g1", source_text=source_text)
    winner = _winner(spoken_text, 6000, 80.0, wav, fit=False)
    compressed_text = "Umusho onzima."
    provider = _FakeLastResortProvider(responses={
        "g1": {"isizulu_text": compressed_text, "omitted_items": ["omitted a supporting detail"]},
    })
    # The compressed candidate measures LONGER than the original in practice.
    tts = _FakeTrimTts({compressed_text: 9000})

    updated, _usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=tts,
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert updated["g1"]["spoken_text"] == spoken_text
    assert updated["g1"]["requires_review"] is False


def test_last_resort_degrades_gracefully_when_the_provider_call_fails(tmp_path):
    wav = tmp_path / "g1.wav"
    _write_real_wav(wav, ms=6000)
    group = _group("g1", source_text="A dense sentence that still runs long.")
    winner = _winner("Umusho onzima kakhulu.", 6000, 80.0, wav, fit=False)
    provider = _FakeLastResortProvider(error=RuntimeError("boom"))

    updated, usage = _compress_protected_content_last_resort(
        provider=provider, groups=[group], winners={"g1": winner}, tts=_FakeTrimTts({}),
        geometries={"g1": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert updated["g1"] is winner
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_last_resort_batches_every_triggered_block_into_one_call(tmp_path):
    wav1 = tmp_path / "g1.wav"
    wav2 = tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=6000)
    _write_real_wav(wav2, ms=6000)
    groups = [
        _group("g1", source_text="First dense sentence that runs long.", span_ms=3000),
        _group("g2", source_text="Second dense sentence that runs long.", speaker_id="S0", span_ms=3000),
    ]
    winners = {
        "g1": _winner("Umusho wokuqala onzima.", 6000, 80.0, wav1, fit=False),
        "g2": _winner("Umusho wesibili onzima.", 6000, 80.0, wav2, fit=False),
    }
    compressed1, compressed2 = "Umusho 1.", "Umusho 2."
    provider = _FakeLastResortProvider(responses={
        "g1": {"isizulu_text": compressed1, "omitted_items": []},
        "g2": {"isizulu_text": compressed2, "omitted_items": []},
    })
    tts = _FakeTrimTts({compressed1: 3100, compressed2: 3100})

    updated, _usage = _compress_protected_content_last_resort(
        provider=provider, groups=groups, winners=winners, tts=tts,
        geometries={"g1": _geometry(), "g2": _geometry()}, voice_assignments=_VOICE_ASSIGNMENTS,
        measurement_audio_root=tmp_path, force=True, candidate_pool_tts_workers=1,
        glossary={},
    )
    assert len(provider.calls) == 1  # one batched call, not one per block
    assert updated["g1"]["spoken_text"] == compressed1
    assert updated["g2"]["spoken_text"] == compressed2
