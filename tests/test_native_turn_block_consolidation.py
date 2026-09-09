"""Regression coverage for Phase 23: whole-window clause ranking, replacing the
earlier per-window "does this merge show real consolidation" guard
(_turn_block_segment_shows_real_consolidation, deleted 2026-09-05).

Real motivating case, found by direct investigation of job
8372120960474ef6b1d75af7de51a605's turn native_phrase_0001..0006: sentences
0003-0006 are genuinely one train of thought (a question announced, restated
with more hedging, answered with evidence, then echoed as an unresolved
close) that Grok correctly recognizes and compacts as ONE unit when given the
chance -- but the old guard rejected exactly this kind of merge, because its
proxy metric (fewer output Zulu sentences than merged English inputs) doesn't
capture "did the clause-ranking ladder gain something real," only "did the
base translation compress into fewer sentences." The real, valuable ladder
here (drop the opener AND the closer together) actually produced 5 Zulu
sentences from 4 merged English ones -- MORE, not fewer -- so the old guard
would have wrongly rejected it.

The fix: a window is now ALWAYS translated and clause-ranked as one whole --
there is no longer a separate "should this merge" decision for a guard to
gate. The model's own hard rule (never drop a name/number/date/claim) is what
prevents real information loss, not a post-hoc sentence-count check.
_validate_turn_block_segments now requires EXACTLY ONE segment covering the
whole window; more than one, or a partial-coverage one, is a structural
failure that falls back to the existing, unmodified independent per-sentence
translation path -- the same two-tier discipline used everywhere else in
this file.
"""
from __future__ import annotations

from types import SimpleNamespace

from mathula_tv.native_dub import _build_speaker_turns, _translate_turn_blocks_via_grok


def _group(group_id: str, source_text: str, *, speaker_id: str = "SPEAKER_00", start_ms: int, span_ms: int = 3000) -> dict:
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": start_ms,
        "source_end_ms": start_ms + span_ms, "source_span_ms": span_ms, "source_text": source_text,
        "segment_ids": [f"{group_id}_seg"], "source_segments": [{"segment_id": f"{group_id}_seg"}],
    }


def _clause(clause_id: str, english_text: str, rank: int = 1) -> dict:
    return {"clause_id": clause_id, "english_text": english_text, "rank": rank}


class _FakeTurnBlockProvider:
    """Registers the two operations _translate_turn_blocks_via_grok can issue:
    the whole-window translate call, and, only for a genuine structural or
    literal-preservation failure, the existing, unmodified independent
    per-sentence fallback. Any other operation fails the test immediately.

    A window_id registered in ``responses`` gets that fixed raw response
    (whatever shape the test wants to exercise -- one segment, several, a
    partial-coverage one); any other window_id gets a default single-segment
    response covering the whole window with trivial per-member clauses.
    """

    def __init__(
        self, *, responses: dict[str, list[dict]] | None = None,
        fallback_extra: dict[str, dict] | None = None,
    ):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._responses = responses or {}
        self._fallback_extra = fallback_extra or {}
        self.calls: list[str] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append(operation)
        if operation == "native_turn_block_translate_batch":
            windows_response = []
            for w in payload["windows"]:
                window_id = w["window_id"]
                if window_id in self._responses:
                    segments = self._responses[window_id]
                else:
                    segments = [{
                        "start_index": 1, "end_index": len(w["members"]),
                        "isizulu_text": f"Default Zulu for {window_id}.",
                        "clauses": [
                            _clause(f"c{i + 1}", m["english_text"]) for i, m in enumerate(w["members"])
                        ],
                    }]
                windows_response.append({"window_id": window_id, "segments": segments})
            return SimpleNamespace(data={"windows": windows_response}, model="grok-test",
                                    input_tokens=10, output_tokens=5, attempts=1)
        if operation == "native_candidate_translate_batch":
            translations = [
                {
                    "unit_id": u["unit_id"], "zulu_text": f"Fallback Zulu for {u['unit_id']}.",
                    **self._fallback_extra.get(u["unit_id"], {}),
                }
                for u in payload["units"]
            ]
            return SimpleNamespace(data={"translations": translations}, model="grok-test",
                                    input_tokens=10, output_tokens=5, attempts=1)
        raise AssertionError(f"unexpected operation: {operation}")


def test_a_merge_with_no_sentence_count_reduction_now_stays_merged():
    # The real native_phrase_0024/0025 case that the OLD guard wrongly
    # rejected (5 Zulu sentences is not fewer than 2... well, 2 here): nothing
    # about this merge is unsafe, the guard's proxy metric was just measuring
    # the wrong thing. It must now be kept as ONE block, not retried apart.
    groups = [
        _group("native_phrase_0024", "Take it slow.", start_ms=0, span_ms=1000),
        _group("native_phrase_0025", "When did you speak to Mister Mogotsi?", start_ms=1000, span_ms=3000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        "native_phrase_0024": [{
            "start_index": 1, "end_index": 2,
            "isizulu_text": "Ungajahi. Wena wakhuluma nini noMnu. Mogotsi?",
            "clauses": [_clause("c1", "Take it slow."), _clause("c2", "When did you speak to Mister Mogotsi?")],
        }],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["native_phrase_0024"]
    assert block_groups[0]["member_group_ids"] == ["native_phrase_0024", "native_phrase_0025"]
    assert candidate_by_group["native_phrase_0024"]["spoken_text"] == (
        "Ungajahi. Wena wakhuluma nini noMnu. Mogotsi?"
    )
    assert "native_candidate_translate_batch" not in provider.calls
    assert provider.calls.count("native_turn_block_translate_batch") == 1  # no retry round exists anymore


def test_the_models_own_stated_reasoning_surfaces_on_the_block_group():
    # Real user direction, 2026-09-09: "isn't there a way to also log what
    # the model is thinking" -- communicative_goal and register_notes were
    # already required by the generation schema (steering the model's own
    # behavior) but silently discarded rather than carried anywhere
    # inspectable, same class of gap as the clauses-propagation fix above.
    groups = [_group("p1", "Good morning, everyone.", start_ms=0, span_ms=2000)]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        "p1": [{
            "start_index": 1, "end_index": 1,
            "isizulu_text": "Sanibonani nonke.",
            "communicative_goal": "Greet the audience warmly before the report begins.",
            "register_notes": "Used the plural greeting form since the speaker addresses a group.",
            "clauses": [_clause("c1", "Good morning, everyone.")],
        }],
    })
    block_groups, _candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert block_groups[0]["communicative_goal"] == "Greet the audience warmly before the report begins."
    assert block_groups[0]["register_notes"] == (
        "Used the plural greeting form since the speaker addresses a group."
    )


def test_the_real_train_of_thought_case_merges_with_a_droppable_ladder():
    # native_phrase_0003..0006: a question announced, restated with hedging,
    # answered with evidence, then echoed as an unresolved close -- one real
    # arc, with a genuinely valuable ladder dropping the opener AND the
    # closer together (5 Zulu sentences from 4 merged English ones -- the
    # OLD guard would have rejected this as "no consolidation").
    groups = [
        _group("native_phrase_0003", "So at this point now, the question around why the urgency.", start_ms=0, span_ms=3000),
        _group("native_phrase_0004", "Why was it so urgent to not allow the process to start?", start_ms=3000, span_ms=3000),
        _group("native_phrase_0005", "Here was somebody that laid charges within 24 hours.", start_ms=6000, span_ms=5000),
        _group("native_phrase_0006", "We're still stuck on why the urgency.", start_ms=11000, span_ms=2000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    full_text = (
        "Kulesi sigaba, umbuzo wawusengowokuthi kungani kwakuphuthuma. "
        "Kungani kwakuphuthuma kangaka kangangokuba inqubo ingavunyelwa ukuqala? "
        "Nangu umuntu owafaka amacala obugebengu kungakapheli amahora angu-24. "
        "Sisabambeke embuzweni wokuthi kungani kwakuphuthuma."
    )
    provider = _FakeTurnBlockProvider(responses={
        "native_phrase_0003": [{
            "start_index": 1, "end_index": 4,
            "isizulu_text": full_text,
            "clauses": [
                _clause("c1", "So at this point now,", rank=4),
                _clause("c2", "the question around why the urgency.", rank=3),
                _clause("c3", "Why was it so urgent to not allow the process to start?", rank=1),
                _clause("c4", "Here was somebody that laid charges within 24 hours.", rank=1),
                _clause("c5", "We're still stuck on why the urgency.", rank=2),
            ],
            "shortened_candidates": [
                {"dropped_clause_ids": ["c1"], "isizulu_text": "shorter without opener"},
                {"dropped_clause_ids": ["c1", "c2", "c5"], "isizulu_text": "core question and evidence only"},
            ],
        }],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["native_phrase_0003"]
    assert block_groups[0]["member_group_ids"] == [
        "native_phrase_0003", "native_phrase_0004", "native_phrase_0005", "native_phrase_0006",
    ]
    assert candidate_by_group["native_phrase_0003"]["spoken_text"] == full_text
    ladder = block_groups[0]["shortened_candidates"]
    assert [c["dropped_clause_ids"] for c in ladder] == [["c1"], ["c1", "c2", "c5"]]
    # The clause breakdown that produced this ladder must also survive onto
    # the block group itself (see the fallback-path regression test above
    # for the real bug this covers on the validated-window path too).
    assert [c["clause_id"] for c in block_groups[0]["clauses"]] == ["c1", "c2", "c3", "c4", "c5"]


def test_a_window_returning_more_than_one_segment_falls_back():
    groups = [
        _group("p1", "Good morning.", start_ms=0, span_ms=1000),
        _group("p2", "Thank you.", start_ms=1000, span_ms=1000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        "p1": [
            {"start_index": 1, "end_index": 1, "isizulu_text": "Sawubona.", "clauses": [_clause("c1", "Good morning.")]},
            {"start_index": 2, "end_index": 2, "isizulu_text": "Ngiyabonga.", "clauses": [_clause("c1", "Thank you.")]},
        ],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["p1", "p2"]
    assert candidate_by_group["p1"]["spoken_text"] == "Fallback Zulu for p1."
    assert candidate_by_group["p2"]["spoken_text"] == "Fallback Zulu for p2."
    assert "native_candidate_translate_batch" in provider.calls


# Real gap found on job fb3d08b63fed4d90922b08f7e325b906, 2026-09-07: a
# window falling back here used to permanently lose its droppable-content
# mechanism entirely, even for a sentence as obviously compactable as
# "Despite the cold and wet weather, ..." -- CANDIDATE_TRANSLATE_SYSTEM_PROMPT
# now mirrors the turn-block path's clauses/shortened_candidates, so a
# fallback-translated sentence still gets a real chance at fact-priority trim.
def test_a_fallback_translated_sentence_still_carries_shortened_candidates():
    groups = [
        _group("p1", "Good morning.", start_ms=0, span_ms=1000),
        _group("p2", "Thank you.", start_ms=1000, span_ms=1000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(
        responses={
            "p1": [
                {"start_index": 1, "end_index": 1, "isizulu_text": "Sawubona.", "clauses": [_clause("c1", "Good morning.")]},
                {"start_index": 2, "end_index": 2, "isizulu_text": "Ngiyabonga.", "clauses": [_clause("c1", "Thank you.")]},
            ],
        },
        fallback_extra={
            "p1": {
                "clauses": [_clause("c1", "Good morning.", rank=2)],
                "shortened_candidates": [{"dropped_clause_ids": ["c1"], "isizulu_text": "Sawubona."}],
            },
        },
    )
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    p1_group = next(b for b in block_groups if b["group_id"] == "p1")
    assert p1_group["shortened_candidates"] == [
        {"dropped_clause_ids": ["c1"], "isizulu_text": "Sawubona."},
    ]
    # Real confirmed bug (job fb3d08b63fed4d90922b08f7e325b906, 2026-09-09):
    # _build_turn_block_group carried shortened_candidates through but
    # silently dropped the clauses breakdown that produced them -- without
    # this, _trim_by_fact_priority's clause-scoped literal-safety check
    # always fell back to the whole segment's source_text, defeating the
    # reporter-sign-off ranking fix for every block reaching the fallback
    # path. clauses must now survive onto the committed block group too.
    assert p1_group["clauses"] == [{"clause_id": "c1", "english_text": "Good morning.", "rank": 2}]
    # p2's fallback response has no clauses at all -- degrades to an empty
    # shortened_candidates list and a None clauses, never fatal to the whole batch.
    p2_group = next(b for b in block_groups if b["group_id"] == "p2")
    assert p2_group["shortened_candidates"] == []
    assert p2_group["clauses"] is None


def test_a_window_returning_partial_coverage_falls_back():
    groups = [
        _group("p1", "Good morning.", start_ms=0, span_ms=1000),
        _group("p2", "Thank you.", start_ms=1000, span_ms=1000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        # Only covers position 1 -- end_index should be 2 (the window has 2 members).
        "p1": [{"start_index": 1, "end_index": 1, "isizulu_text": "Sawubona.", "clauses": [_clause("c1", "Good morning.")]}],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["p1", "p2"]
    assert candidate_by_group["p1"]["spoken_text"] == "Fallback Zulu for p1."
    assert candidate_by_group["p2"]["spoken_text"] == "Fallback Zulu for p2."


def test_falling_back_logs_the_specific_missing_literal_not_just_a_count():
    # Real user direction, 2026-09-09: "the model need to do better logging
    # [of the] reasons behind every decision" -- diagnosing why a real merge
    # fell back used to require a bespoke script; the reason is now
    # surfaced inline, per window, before the blanket count message.
    groups = [
        _group("p1", "The number is 24.", start_ms=0, span_ms=1000),
        _group("p2", "Confirmed by IDAC.", start_ms=1000, span_ms=1000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        "p1": [{
            "start_index": 1, "end_index": 2, "isizulu_text": "Kukhonjiwe.",
            "clauses": [_clause("c1", "The number is 24."), _clause("c2", "Confirmed by IDAC.")],
        }],
    })
    messages: list[str] = []
    _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
        progress=messages.append,
    )
    reason_lines = [m for m in messages if "dropped required literal" in m]
    assert len(reason_lines) == 1
    assert "'24'" in reason_lines[0]
    assert "'IDAC'" in reason_lines[0]
    assert "p1" in reason_lines[0]


def test_a_bare_ordinal_number_merge_no_longer_falls_back():
    # The other half of the same real fix: a merge that renders a bare
    # single-digit number ("1") as a natural isiZulu word instead of a
    # digit must now be ACCEPTED, not silently discarded -- this is exactly
    # the real case ("mhla wokuqala kuNovemba" for "November 1") that
    # motivated the exemption in _missing_required_literals.
    groups = [
        _group("p1", "Elections are set for November 1.", start_ms=0, span_ms=3000),
        _group("p2", "Residents are excited about it.", start_ms=3000, span_ms=2000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    merged_text = "Ukhetho luzoba mhla wokuqala kuNovemba, izakhamuzi zijabule ngalokho."
    provider = _FakeTurnBlockProvider(responses={
        "p1": [{
            "start_index": 1, "end_index": 2, "isizulu_text": merged_text,
            "clauses": [
                _clause("c1", "Elections are set for November 1."),
                _clause("c2", "Residents are excited about it."),
            ],
        }],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["p1"]
    assert candidate_by_group["p1"]["spoken_text"] == merged_text
    assert "native_candidate_translate_batch" not in provider.calls  # never fell back


def test_a_literal_dropping_merge_falls_back_to_independent_translation():
    groups = [
        _group("p1", "The number is 24.", start_ms=0, span_ms=1000),
        _group("p2", "Confirmed by IDAC.", start_ms=1000, span_ms=1000),
    ]
    turns = _build_speaker_turns(groups)
    natural_english = {g["group_id"]: g["source_text"] for g in groups}
    provider = _FakeTurnBlockProvider(responses={
        # Drops both required literals ("24", "IDAC") from the combined text.
        "p1": [{
            "start_index": 1, "end_index": 2, "isizulu_text": "Kukhonjiwe.",
            "clauses": [_clause("c1", "The number is 24."), _clause("c2", "Confirmed by IDAC.")],
        }],
    })
    block_groups, candidate_by_group, _usage = _translate_turn_blocks_via_grok(
        provider=provider, groups=groups, turns=turns, natural_english_by_group=natural_english, glossary={},
    )
    assert [b["group_id"] for b in block_groups] == ["p1", "p2"]
    assert candidate_by_group["p1"]["spoken_text"] == "Fallback Zulu for p1."
    assert candidate_by_group["p2"]["spoken_text"] == "Fallback Zulu for p2."
