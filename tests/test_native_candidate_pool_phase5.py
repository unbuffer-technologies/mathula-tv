"""Phase 9 of the --candidate-pool architecture (see the approved plan at
C:\\Users\\mokgethwa\\.claude\\plans\\calm-wiggling-parasol.md): translate each
sentence's ONE natural English text through Grok (no length ladder -- that was
removed in Phase 9) and measure it for real -- writing pass2_candidate_pool.json
only, never the real committed pass2 artifact. No automated QA/repair step runs
on the translation (removed 2026-09-03, per explicit user direction) -- the
base fake provider raises on any operation it doesn't explicitly register, so
an accidental QA/repair call would fail these tests immediately, not just go
unasserted. Discourse-segment grouping/rebalancing itself is covered in
test_native_timing_regions.py.

Fake providers here follow the same shape used throughout this session's
other provider-facing tests: plain objects exposing .config/.complete_json(...)
(FoundryGrokProvider) -- none of the functions under test type-check these,
they just call the protocol.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.native_dub import (
    ZULU_GRAMMAR_FOUNDATION,
    _candidate_translate_schema,
    _enrich_measured_candidates,
    _request_candidate_translation_batch,
    _translate_natural_via_grok,
    build_candidate_pool,
    compare_candidate_pool_to_committed_pass2,
    native_dub_paths,
)


def _group(group_id: str, source_text: str, *, speaker_id: str = "SPEAKER_00", start_ms: int = 0, span_ms: int = 3000) -> dict:
    return {
        "group_id": group_id,
        "speaker_id": speaker_id,
        "segment_ids": [f"{group_id}_seg"],
        "start_ms": start_ms,
        "source_end_ms": start_ms + span_ms,
        "source_span_ms": span_ms,
        "source_segments": [{"segment_id": f"{group_id}_seg", "source_text": source_text}],
        "source_text": source_text,
    }


# --- _candidate_translate_schema ----------------------------------------------------


def test_candidate_translate_schema_rejects_duplicate_unit_ids():
    with pytest.raises(ValueError):
        _candidate_translate_schema(["u1", "u1"])


# --- _request_candidate_translation_batch -------------------------------------------


class _FakeTranslateProvider:
    def __init__(self, translations: list[dict], *, delay: float = 0.0):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._translations = translations
        self._delay = delay

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        if self._delay:
            time.sleep(self._delay)
        assert operation == "native_candidate_translate_batch"
        return SimpleNamespace(
            data={"translations": self._translations}, model="grok-test",
            input_tokens=20, output_tokens=10, attempts=1,
        )


def test_request_candidate_translation_batch_returns_by_unit_id():
    # No "clauses" in the fake response -- a malformed/missing clauses shape
    # degrades ONLY this unit's shortened_candidates to empty, never fatal.
    provider = _FakeTranslateProvider([{"unit_id": "g1", "zulu_text": "Sawubona."}])
    by_id, usage = _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
    )
    assert by_id == {"g1": {"zulu_text": "Sawubona.", "clauses": None, "shortened_candidates": []}}
    assert usage == {"input_tokens": 20, "output_tokens": 10, "attempts": 1}


def test_request_candidate_translation_batch_canonicalizes_shortened_candidates():
    provider = _FakeTranslateProvider([{
        "unit_id": "g1", "zulu_text": "Naphezu kwesimo, izakhamuzi zaphuma.",
        "clauses": [
            {"clause_id": "c1", "english_text": "Despite the weather", "rank": 3},
            {"clause_id": "c2", "english_text": "residents turned out", "rank": 1},
        ],
        "shortened_candidates": [
            {"dropped_clause_ids": ["c1"], "isizulu_text": "Izakhamuzi zaphuma."},
        ],
    }])
    by_id, _ = _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Despite the weather, residents turned out."}],
        glossary={},
    )
    assert by_id["g1"]["shortened_candidates"] == [
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Izakhamuzi zaphuma."},
    ]


class _PayloadCapturingProvider:
    """Captures the exact payload sent, to prove context_ledger actually
    reaches the request rather than just being accepted and ignored.
    """

    def __init__(self, data):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._data = data
        self.last_payload = None

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.last_payload = payload
        return SimpleNamespace(data=self._data, model="grok-test", input_tokens=1, output_tokens=1, attempts=1)


def test_request_candidate_translation_batch_includes_context_ledger_in_payload():
    ledger = {"summary": "A single witness testifies.", "entities": [], "speaker_roles": []}
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Sawubona."}]})
    _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}],
        glossary={}, context_ledger=ledger,
    )
    assert provider.last_payload["context_ledger"] == ledger


def test_request_candidate_translation_batch_defaults_context_ledger_to_empty_when_omitted():
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Sawubona."}]})
    _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
    )
    assert provider.last_payload["context_ledger"] == {}


def test_request_candidate_translation_batch_includes_zulu_grammar_foundation_in_payload():
    # 2026-08-29: the mood/concord/verbal-extension rule system, not another incident
    # report -- must reach the actual call that generates isiZulu, not just exist as
    # an unused module constant.
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Sawubona."}]})
    _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
    )
    assert provider.last_payload["zulu_grammar_foundation"] == ZULU_GRAMMAR_FOUNDATION
    assert "SECONDARY subject concord" in ZULU_GRAMMAR_FOUNDATION


def test_request_candidate_translation_batch_includes_reference_vocabulary_when_present():
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Sawubona."}]})
    _request_candidate_translation_batch(
        provider=provider,
        units=[{"unit_id": "g1", "english_text": "Hello minister.", "reference_vocabulary": {"minister": ["ungqongqoshe"]}}],
        glossary={},
    )
    assert provider.last_payload["units"][0]["reference_vocabulary"] == {"minister": ["ungqongqoshe"]}


def test_request_candidate_translation_batch_omits_reference_vocabulary_when_absent():
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Sawubona."}]})
    _request_candidate_translation_batch(
        provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
    )
    assert "reference_vocabulary" not in provider.last_payload["units"][0]


def test_request_candidate_translation_batch_raises_on_missing_unit():
    provider = _FakeTranslateProvider([])
    with pytest.raises(ValueError, match="missing"):
        _request_candidate_translation_batch(
            provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
        )


def test_request_candidate_translation_batch_raises_on_unknown_unit():
    provider = _FakeTranslateProvider([{"unit_id": "ghost", "zulu_text": "x"}])
    with pytest.raises(ValueError, match="unknown"):
        _request_candidate_translation_batch(
            provider=provider, units=[{"unit_id": "g1", "english_text": "Hello."}], glossary={},
        )


# --- _translate_natural_via_grok -----------------------------------------------------


def test_translate_natural_via_grok_produces_one_candidate_per_sentence():
    groups = [_group("g1", "Hello there."), _group("g2", "Goodbye now.")]
    natural_english = {"g1": "Hello there.", "g2": "Goodbye now."}
    provider = _FakeTranslateProvider([
        {"unit_id": "g1", "zulu_text": "Sawubona."},
        {"unit_id": "g2", "zulu_text": "Sala kahle."},
    ])
    candidate_by_group, usage = _translate_natural_via_grok(
        provider=provider, groups=groups, natural_english_by_group=natural_english, glossary={}, workers=1,
    )
    assert set(candidate_by_group) == {"g1", "g2"}
    assert candidate_by_group["g1"] == {
        "candidate_id": "grok_natural", "variant_id": "natural", "translator": "grok", "spoken_text": "Sawubona.",
        "clauses": None, "shortened_candidates": [],
    }
    assert usage == {"input_tokens": 20, "output_tokens": 10, "attempts": 1}


def test_translate_natural_via_grok_passes_preceding_english_from_groups_order():
    # 2026-08-29: real grounding for elliptical/anaphoric sentences (e.g. "No, you
    # were not [discussing X]." needs the preceding question's content to resolve
    # what's actually being denied) -- computed from groups' own known order, no
    # sequential dependency reintroduced.
    groups = [_group("g1", "Were you discussing Crime Intelligence issues?"), _group("g2", "No, you were not.")]
    natural_english = {"g1": "Were you discussing Crime Intelligence issues?", "g2": "No, you were not."}
    provider = _PayloadCapturingProvider({
        "translations": [
            {"unit_id": "g1", "zulu_text": "Bengixoxa ngezindaba ze-Crime Intelligence na?"},
            {"unit_id": "g2", "zulu_text": "Cha, beningaxoxi ngalezondaba."},
        ],
    })
    _translate_natural_via_grok(
        provider=provider, groups=groups, natural_english_by_group=natural_english, glossary={}, workers=1,
    )
    units = provider.last_payload["units"]
    by_id = {u["unit_id"]: u for u in units}
    assert "preceding_english" not in by_id["g1"]  # first sentence has no predecessor
    assert by_id["g2"]["preceding_english"] == "Were you discussing Crime Intelligence issues?"


def test_translate_natural_via_grok_passes_real_lexicon_reference_vocabulary():
    # 2026-08-29: real, general-purpose vocabulary grounding (distinct from the
    # job-specific glossary) -- confirms the actual bundled Autshumato lexicon
    # lookup reaches the payload, not a placeholder.
    groups = [_group("g1", "The minister was told.")]
    natural_english = {"g1": "The minister was told."}
    provider = _PayloadCapturingProvider({"translations": [{"unit_id": "g1", "zulu_text": "Ungqongqoshe utsheliwe."}]})
    _translate_natural_via_grok(
        provider=provider, groups=groups, natural_english_by_group=natural_english, glossary={}, workers=1,
    )
    units = provider.last_payload["units"]
    reference_vocabulary = units[0]["reference_vocabulary"]
    assert "ungqongqoshe" in reference_vocabulary["minister"]


def test_translate_natural_via_grok_handles_no_groups_without_calling_provider():
    class _Exploding:
        config = SimpleNamespace(max_output_tokens=8192)

        def complete_json(self, **_kwargs):
            raise AssertionError("must not be called with no groups")

    candidate_by_group, usage = _translate_natural_via_grok(
        provider=_Exploding(), groups=[], natural_english_by_group={}, glossary={}, workers=1,
    )
    assert candidate_by_group == {}
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_translate_natural_via_grok_batches_run_concurrently():
    groups = [_group(f"g{i}", f"Sentence number {i}.") for i in range(8)]
    natural_english = {f"g{i}": f"Sentence number {i}." for i in range(8)}

    class _ConcurrencyTrackingTranslateProvider:
        def __init__(self):
            self.config = SimpleNamespace(max_output_tokens=8192)
            self._lock = threading.Lock()
            self._in_flight = 0
            self.max_concurrent_seen = 0

        def complete_json(self, *, operation, payload, **_kwargs):
            with self._lock:
                self._in_flight += 1
                self.max_concurrent_seen = max(self.max_concurrent_seen, self._in_flight)
            try:
                time.sleep(0.05)
                translations = [{"unit_id": u["unit_id"], "zulu_text": f"Z[{u['unit_id']}]"} for u in payload["units"]]
                return SimpleNamespace(data={"translations": translations}, model="grok-test",
                                        input_tokens=5, output_tokens=5, attempts=1)
            finally:
                with self._lock:
                    self._in_flight -= 1

    provider = _ConcurrencyTrackingTranslateProvider()
    candidate_by_group, _usage = _translate_natural_via_grok(
        provider=provider, groups=groups, natural_english_by_group=natural_english, glossary={},
        batch_size=1, workers=4,
    )
    assert provider.max_concurrent_seen > 1
    assert len(candidate_by_group) == 8


# --- _enrich_measured_candidates --------------------------------------------------------


def test_enrich_measured_candidates_merges_metadata_by_candidate_id():
    measured = [{"candidate_id": "grok_natural", "spoken_text": "x", "measured_ms": 1200, "fit": True, "rank": [0, 0, 0], "semantic_guard_pass": True}]
    original = [{"candidate_id": "grok_natural", "translator": "grok", "variant_id": "natural", "qa_penalty": 1}]
    enriched = _enrich_measured_candidates(measured=measured, original_candidates=original)
    assert enriched[0]["translator"] == "grok"
    assert enriched[0]["variant_id"] == "natural"
    assert enriched[0]["qa_penalty"] == 1
    assert enriched[0]["measured_ms"] == 1200  # measurement fields preserved


# --- build_candidate_pool (integration) -------------------------------------------------


class _FakeIntegrationProvider:
    """Dispatches every operation build_candidate_pool's own steps issue."""

    def __init__(self):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.calls: list[str] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append(operation)
        if operation == "native_zulu_glossary":
            return SimpleNamespace(
                data={"zulu_terminology_glossary": {
                    "terms": [], "do_not_translate": [],
                    "register": {"formality": "formal", "address_form": "neutral", "tense_default": "present"},
                }},
                model="grok-test", input_tokens=10, output_tokens=5, attempts=1,
            )
        if operation == "native_candidate_translate_batch":
            translations = [
                {"unit_id": u["unit_id"], "zulu_text": f"Zulu rendering of {u['unit_id']}."}
                for u in payload["units"]
            ]
            return SimpleNamespace(data={"translations": translations}, model="grok-test",
                                    input_tokens=10, output_tokens=5, attempts=1)
        if operation == "native_turn_block_translate_batch":
            # Phase 16: simulate "nothing merged" -- one segment per member,
            # using window_id (the first member's own group_id) for the
            # single-member case every test in this file exercises, so the
            # resulting spoken_text matches the pre-Phase-16 fixture strings
            # ("Zulu rendering of native_phrase_0001.") byte-for-byte.
            windows_response = []
            for window in payload["windows"]:
                members = window["members"]
                if len(members) == 1:
                    text_for = lambda member, window_id=window["window_id"]: f"Zulu rendering of {window_id}."
                else:
                    text_for = lambda member, window_id=window["window_id"]: (
                        f"Zulu rendering of {window_id}_m{member['index']}."
                    )
                windows_response.append({
                    "window_id": window["window_id"],
                    "segments": [
                        {"start_index": member["index"], "end_index": member["index"], "isizulu_text": text_for(member)}
                        for member in members
                    ],
                })
            return SimpleNamespace(data={"windows": windows_response}, model="grok-test",
                                    input_tokens=10, output_tokens=5, attempts=1)
        # No QA/repair branch registered on purpose: build_candidate_pool no
        # longer runs any back-translation audit or repair call, per explicit
        # user direction. An accidental call to either fails every test in
        # this file immediately via this fallback, proving the removal held.
        raise AssertionError(f"unexpected operation: {operation}")


class _FakePoolTts:
    """Minimal duck-typed AzureTTSBackend stand-in: build_candidate_pool's own
    measurement path only ever reads tts.default_voice and calls
    tts.synthesize(request, output_path, force=...).duration_ms -- see
    _synthesize_group/_measure_temporal_candidate. Duration is keyed by the
    synthesized text itself so ThreadPoolExecutor ordering cannot matter.
    """

    default_voice = "zu-ZA-ThandoNeural"

    def __init__(self, duration_by_text: dict[str, int], *, default_ms: int = 3000):
        self._duration_by_text = duration_by_text
        self._default_ms = default_ms

    def synthesize(self, request, output_path, *, force=False):
        duration_ms = self._duration_by_text.get(request.text, self._default_ms)
        return SimpleNamespace(duration_ms=duration_ms)


def _seed_job(tmp_path: Path, *, sentences: list[str]) -> Path:
    job_root = tmp_path / "job"
    paths = native_dub_paths(job_root)
    paths.pass1.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {"segment_id": f"seg{i:02d}", "speaker_id": "SPEAKER_00", "restored_text": text,
         "start_ms": i * 4000, "end_ms": i * 4000 + 3000}
        for i, text in enumerate(sentences)
    ]
    atomic_write_json(paths.pass1, {
        "corrections": [], "segments": segments,
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
    })
    atomic_write_json(paths.context_ledger, {
        "schema_version": "mathula-native-context-ledger-v1",
        "source_pass1_sha256": None,  # replaced below
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
        "usage": {"input_tokens": 0, "output_tokens": 0, "attempts": 0},
        "completed_at": "2026-01-01T00:00:00Z",
    })
    from mathula_tv.media import checksum
    ledger_doc = paths.context_ledger
    import json
    doc = json.loads(ledger_doc.read_text(encoding="utf-8"))
    doc["source_pass1_sha256"] = checksum(paths.pass1)
    atomic_write_json(ledger_doc, doc)
    return job_root


def test_build_candidate_pool_end_to_end_translates_and_measures_natural_text(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["He was late for the meeting."])
    provider = _FakeIntegrationProvider()
    durations = {"Zulu rendering of native_phrase_0001.": 2800}
    tts = _FakePoolTts(durations)

    artifact = build_candidate_pool(
        job_root=job_root, provider=provider, tts=tts,
        candidate_pool_workers=1, candidate_pool_tts_workers=1,
    )

    assert artifact["candidate_pool_mode"] is True
    record = artifact["candidate_pool"][0]
    assert record["group_id"] == "native_phrase_0001"
    assert record["discourse_unit_kind"] == "normal"
    winner = next(c for c in record["candidates"] if c["candidate_id"] == record["selected_candidate_id"])
    assert winner["candidate_id"] == "grok_natural"
    assert winner["spoken_text"] == "Zulu rendering of native_phrase_0001."
    assert winner["measured_ms"] == 2800
    assert winner["qa_penalty"] == 0
    assert winner["requires_review"] is False
    assert "native_qa_back_translation_batch" not in provider.calls  # no QA audit runs at all
    assert native_dub_paths(job_root).candidate_pool.is_file()


def test_build_candidate_pool_classifies_a_short_utterance_as_bridge(tmp_path):
    # _seed_job's fixed 3000ms-per-sentence spacing is too long to trigger the
    # bridge span threshold -- seed a genuinely short (900ms) source window directly.
    job_root = tmp_path / "job"
    paths = native_dub_paths(job_root)
    paths.pass1.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.pass1, {
        "corrections": [],
        "segments": [{"segment_id": "seg00", "speaker_id": "SPEAKER_00", "restored_text": "All right.",
                      "start_ms": 0, "end_ms": 900}],
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
    })
    atomic_write_json(paths.context_ledger, {
        "schema_version": "mathula-native-context-ledger-v1",
        "source_pass1_sha256": None,
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
        "usage": {"input_tokens": 0, "output_tokens": 0, "attempts": 0},
        "completed_at": "2026-01-01T00:00:00Z",
    })
    from mathula_tv.media import checksum
    import json
    doc = json.loads(paths.context_ledger.read_text(encoding="utf-8"))
    doc["source_pass1_sha256"] = checksum(paths.pass1)
    atomic_write_json(paths.context_ledger, doc)

    provider = _FakeIntegrationProvider()
    tts = _FakePoolTts({}, default_ms=900)

    artifact = build_candidate_pool(
        job_root=job_root, provider=provider, tts=tts,
        candidate_pool_workers=1, candidate_pool_tts_workers=1,
    )
    record = artifact["candidate_pool"][0]
    assert record["discourse_unit_kind"] == "bridge"


def test_build_candidate_pool_reuses_checkpoint_without_recalling_provider(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Short sentence here."])
    provider = _FakeIntegrationProvider()
    tts = _FakePoolTts({}, default_ms=3000)

    build_candidate_pool(
        job_root=job_root, provider=provider, tts=tts,
        candidate_pool_workers=1, candidate_pool_tts_workers=1,
    )
    first_call_count = len(provider.calls)
    assert first_call_count > 0

    build_candidate_pool(
        job_root=job_root, provider=provider, tts=tts,
        candidate_pool_workers=1, candidate_pool_tts_workers=1,
    )
    assert len(provider.calls) == first_call_count  # no new calls on the cached re-run


def test_build_candidate_pool_ships_the_natural_translation_unaudited(tmp_path):
    # No automated QA/repair step runs at all, per explicit user direction --
    # the natural translation is the winner regardless of how it reads. Using
    # the SAME base fake provider (which raises on any operation it doesn't
    # explicitly register) proves this: a real QA/repair call would fail the
    # test immediately, not just go unasserted.
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    provider = _FakeIntegrationProvider()
    tts = _FakePoolTts({}, default_ms=3000)

    artifact = build_candidate_pool(
        job_root=job_root, provider=provider, tts=tts,
        candidate_pool_workers=1, candidate_pool_tts_workers=1,
    )
    record = artifact["candidate_pool"][0]
    winner_id = record["selected_candidate_id"]
    winner = next(c for c in record["candidates"] if c["candidate_id"] == winner_id)
    assert winner["spoken_text"] == "Zulu rendering of native_phrase_0001."
    assert winner["qa_penalty"] == 0
    assert winner["requires_review"] is False
    assert record["requires_review"] is False


# --- compare_candidate_pool_to_committed_pass2 -------------------------------------------


def test_compare_candidate_pool_to_committed_pass2_flags_changed_sentences(tmp_path):
    job_root = tmp_path / "job"
    paths = native_dub_paths(job_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.pass2, {
        "temporal_mask_groups": [
            {"group_id": "native_phrase_0001", "spoken_text": "Old committed text."},
        ],
    })
    atomic_write_json(paths.candidate_pool, {
        "candidate_pool": [
            {
                "group_id": "native_phrase_0001",
                "selected_candidate_id": "grok_natural",
                "requires_review": False,
                "candidates": [
                    {"candidate_id": "grok_natural", "translator": "grok", "spoken_text": "New candidate-pool text.", "measured_ms": 3200},
                ],
            },
        ],
    })
    report = compare_candidate_pool_to_committed_pass2(job_root=job_root)
    assert report["sentence_count"] == 1
    assert report["changed_count"] == 1
    assert report["rows"][0]["same_text"] is False
    assert report["rows"][0]["candidate_pool_measured_ms"] == 3200
