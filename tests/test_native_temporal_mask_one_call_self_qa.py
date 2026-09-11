from mathula_tv.native_dub import (
    CONFIRMED_TRANSLATION_PITFALLS,
    TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT,
    TEMPORAL_MASK_PROMPT_VERSION,
    TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT,
    _estimate_zulu_syllables,
    _native_dub_tts_ready_text,
    _normalize_temporal_compact_response,
    _normalize_temporal_mask_response,
    _temporal_compact_schema,
    _temporal_compact_wire,
    _temporal_sentence_semantic_window,
    _temporal_mask_schema,
)


def test_temporal_mask_prompt_requires_internal_backtranslation_and_no_python_residual_loop():
    assert "ONE PROVIDER CALL = ONE COMPLETE WINDOW" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "Back-translate the COMPLETE Zulu window into English" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "Python will NOT send you a residual repair request for this" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "phase-a-no-self-check" in TEMPORAL_MASK_PROMPT_VERSION


# --- two-phase translate-then-compact split (retires the old single-call iterative
# pace-variant ladder) --------------------------------------------------------------
# Phase A (TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT) always returns exactly one natural,
# faithful translation per sentence -- no ladder, no numeric length target. Phase B
# (TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT) is called ONLY for sentences Azure measured as
# oversized after Phase A, handing Grok the REAL measured required_speed_percent as a
# concrete target instead of a blind linguistic guess, and may explicitly decline
# further compaction (declined_further_compaction) rather than force a defective
# rewrite. This replaced the old joint-window candidate ladder entirely.

def test_translate_prompt_returns_exactly_one_candidate_no_ladder():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "Return exactly ONE final isiZulu wording per requested mutable group_id" in normalized
    assert "there is no compaction ladder here" in normalized
    assert "ITERATIVE PACE-VARIANT CANDIDATES" not in normalized


# --- explicit worked JSON example in the output contract ----------------------------
# Regression test for a real, confirmed production crash: Grok reliably avoided the
# {"masks": [...]} top-level wrapper on its first attempt (two different windows, two
# different wrong shapes: a dict keyed directly by group_id, and a bare array with no
# wrapper at all). A schema alone did not teach it the exact required shape; an
# explicit worked example, naming both confirmed wrong shapes, does.

def test_translate_prompt_shows_the_exact_masks_wrapper_and_names_wrong_shapes():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert '"masks": [' in normalized
    assert "top-level object keyed directly by group_id" in normalized
    assert "bare top-level array with no" in normalized


def test_compact_prompt_shows_the_exact_results_wrapper_and_names_wrong_shapes():
    normalized = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())
    assert '"results": [' in normalized
    assert "never a top-level object keyed directly by group_id" in normalized
    assert "never a bare top-level array" in normalized


def test_compact_prompt_states_the_decline_condition_grounded_in_a_real_number():
    normalized = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())
    assert "required_speed_percent" in normalized
    assert "declined_further_compaction" in normalized
    assert "damaging fluidity, comprehension" in normalized


# --- concrete syllable target + isiZulu-native compaction lever --------------------
# Two research-backed additions to Phase B: (1) a computed target_syllable_count
# alongside required_speed_percent, so Grok doesn't have to do the percent-to-syllable
# arithmetic itself while also drafting content (the "budget allocation before writing"
# pattern length-controlled-generation research found beats leaving the conversion to
# the model mid-draft); (2) an isiZulu-native compaction lever (preferring a synthetic,
# concord-bearing verb form over an equally natural but more periphrastic one) alongside
# the previously English-transfer-only technique list (filler removal, self-inserted
# intensifier removal, honorific dropping).

def test_compact_prompt_gives_a_concrete_syllable_target_as_a_drafting_aid_not_a_contract():
    normalized = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())
    assert "target_syllable_count" in normalized
    assert "current_syllable_count" in normalized
    assert "DRAFTING AID, not a pass/fail contract" in normalized


def test_compact_prompt_teaches_the_synthetic_over_periphrastic_verb_lever():
    normalized = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())
    assert "SYNTHETIC" in normalized and "PERIPHRASTIC" in normalized
    assert "concords" in normalized
    assert "NOT a fidelity" in normalized


def test_temporal_mask_prompt_says_candidate_a_has_no_numeric_target():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "NOT benchmarked against a syllable count" in normalized


def test_translate_prompt_says_python_measures_afterward_not_a_guess():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "Python measures its real synthesized length afterward" in normalized
    assert "you never predict or aim for a length while drafting it" in normalized


def test_temporal_mask_schema_treats_candidate_count_as_a_ceiling_not_exact():
    schema = _temporal_mask_schema(3, expected_group_ids=["g1"])
    candidates_schema = schema["properties"]["masks"]["items"]["properties"]["candidates"]
    assert candidates_schema["minItems"] == 1
    assert candidates_schema["maxItems"] == 3
    assert candidates_schema["items"]["properties"]["candidate_id"]["enum"] == ["a", "b", "c"]
    assert "estimated_syllables" not in candidates_schema["items"]["properties"]


# --- Phase A no longer reports a self_check: production evidence showed the in-call
# self-report was unreliable (matches_source=true on translations that back-translated
# to a materially different claim), and it cost real output tokens on every one of
# ~150+ sentences per job. The internal revision-cycle discipline (draft, back-translate,
# fix) stays in the prompt -- that costs nothing extra -- but nothing requires Grok to
# show its work any more. The genuinely independent, Python-orchestrated
# run_native_qa_back_translation audit (with its own auto-repair loop) is the real
# backstop now, not a self-graded verdict produced mid-draft.

def test_translate_prompt_no_longer_requires_a_self_check_report():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "FINAL SELF-CHECK OUTPUT" not in normalized
    assert "self_check" not in normalized
    assert "A separate, independent verification pass runs later" in normalized


def test_translate_prompt_schema_omits_self_check_when_disabled():
    schema = _temporal_mask_schema(1, expected_group_ids=["g1"], require_self_check=False)
    candidate_schema = schema["properties"]["masks"]["items"]["properties"]["candidates"]["items"]
    assert "self_check" not in candidate_schema["required"]
    assert "self_check" not in candidate_schema["properties"]
    assert set(candidate_schema["required"]) == {"candidate_id", "spoken_text"}


def test_temporal_mask_schema_still_requires_self_check_by_default():
    # require_self_check defaults to True so the shared machinery's other callers
    # (and tests exercising the retired ladder shape) are unaffected.
    schema = _temporal_mask_schema(1, expected_group_ids=["g1"])
    candidate_schema = schema["properties"]["masks"]["items"]["properties"]["candidates"]["items"]
    assert "self_check" in candidate_schema["required"]


def test_compact_prompt_still_reuses_shared_drift_judgment_criteria():
    # Phase B's compaction subset is small, so its self_check/decline mechanism stays --
    # it still shares one calibrated criteria block with the native-qa back-translation
    # phase rather than maintaining a separate copy that can drift out of sync.
    assert "THE CENTRAL TEST" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT
    assert "isiZulu is a pro-drop language" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT
    assert "greeting or pleasantry" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT


def test_temporal_mask_prompt_mentions_confirmed_translation_pitfalls_payload_field():
    assert "confirmed_translation_pitfalls" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "confirmed_translation_pitfalls" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT


# --- negation-integrity / anti-echo strengthening -----------------------------------
# A live-run validation on the real job found the self_check upgrade above, while
# schema-compliant, was letting real drift through: for several groups the reported
# back_translation was word-for-word identical to the English source (never actually
# re-derived from the committed isiZulu), and the pipeline's own separate, independent
# native-qa back-translation pass caught 3 negation reversals (e.g. "I could have"
# committed as isiZulu that back-translates to "I could not have") that this in-call
# self_check had rated matches_source=True. These rules close that gap.

def test_temporal_mask_prompt_forbids_echoing_the_source_as_a_back_translation():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "not by recalling or restating the English source from memory" in normalized
    assert "which will always appear to match even when the Zulu itself is wrong" in normalized


def test_temporal_mask_prompt_requires_explicit_negation_polarity_verification():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "re-derive the polarity (positive/negative) and tense of every verb from its actual Zulu form" in normalized
    assert "confirm the polarity of every back-translated verb against its actual Zulu affix" in normalized


def test_temporal_mask_prompt_requires_finding_a_fix_not_just_reporting_the_defect():
    # Even without a reported self_check, the internal loop must still actually FIX a
    # found defect (not just notice it and move on) before returning.
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "is not acceptable to leave unresolved" in normalized
    assert "Find that wording and revise to it before returning your answer" in normalized


# --- embedded-attribution and sentence-slot strengthening ---------------------------
# Independent QA validation on the real job (after the negation fix above landed and
# was confirmed) found two further real defects the in-call self_check was not catching:
# (1) an embedded reporting-verb attribution swap ("General Lincoln...Mr. Brown Mogotsi,
# who HE told me was close to the minister" committed as isiZulu whose relative clause
# back-translates to "Mogotsi, who TOLD ME he was close to the minister" -- teller and
# described person swapped), and (2) a same-sentence-count content shuffle within a block
# (native_phrase_0039's sentence 9's translation held sentence 10's content and vice
# versa). Both are named, explicit checks now, tied back to the existing MEANING/
# COMPREHENSION HARD CONTRACT and SENTENCE-ORDER CORRESPONDENCE sections respectively.

def test_temporal_mask_prompt_requires_checking_embedded_reporting_verb_attribution():
    normalized = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
    assert "the teller and the person described swapped" in normalized
    assert "explicitly identify who the grammatical subject of the embedded reporting verb is" in normalized


def test_confirmed_translation_pitfalls_seeded_with_the_real_colleague_umlimi_case():
    assert any("umlimi" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)
    assert any("colleague" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


# The two entries below were added after the checklist strengthening above (embedded
# reporting-verb attribution, sentence-slot correspondence) turned out not to reliably
# prevent these two specific patterns from recurring across further live-run validation:
# native_phrase_0006's Lincoln/Mogotsi attribution swap reproduced verbatim even after
# the general checklist rule was added, and a new negation reversal appeared on a
# different "could have" sentence than the ones fixed earlier. Both are concrete,
# recurring, evidence-backed cases -- exactly what CONFIRMED_TRANSLATION_PITFALLS exists
# for, rather than trying to hand-author increasingly specific isiZulu grammar rules
# into the main prompt body.

def test_confirmed_translation_pitfalls_includes_the_mogotsi_attribution_case():
    assert any("Mogotsi" in entry and "told me" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


def test_confirmed_translation_pitfalls_includes_the_could_have_negation_case():
    assert any("-kwazi" in entry and "could have" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


# The three entries below were added 2026-08-29 after real-job QA flagged three more
# recurring patterns (native_phrase_0151/0153's counterfactual flattening, native_phrase_
# 0113/0123's multi-pronoun attribution swap, native_phrase_0056's denial-scope mismatch),
# researched against real isiZulu grammar sources (Halpert 2012 "Zulu Counterfactuals in
# and out of Conditionals" for the ngabe construction) rather than assumed.

def test_confirmed_translation_pitfalls_includes_the_ngabe_counterfactual_construction():
    assert any("ngabe" in entry and "counterfactual" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


def test_confirmed_translation_pitfalls_includes_the_multi_pronoun_attribution_case():
    assert any("Witness G" in entry and "concords" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


def test_confirmed_translation_pitfalls_multi_pronoun_case_gives_a_concrete_repeat_noun_fix():
    # 2026-08-29: a third real run reproduced yet another wrong concord chain on this
    # exact sentence despite the existing abstract advice -- strengthened with a
    # concrete, mechanical fix (repeat the proper noun instead of a pronoun/concord).
    assert any(
        "REPEAT THE" in entry and "PROPER NOUN" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_mogotsi_case_also_covers_the_split_sentence_recurrence():
    # 2026-08-29: a later real run reproduced the same teller/subject swap even after
    # the sentence was split into two, so the entry was extended with the same
    # repeat-the-proper-noun mechanical fix used for the Witness G case.
    assert any(
        "ULincoln wangitshela" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_past_tense_drift_case():
    assert any(
        "went about" in entry and "ya ku-[verb]" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_denial_scope_case():
    # 2026-09-02: trimmed of its worked wrong-form examples (audit-trail prose,
    # not needed by the model) -- the actionable rule (negate the ACTIVITY verb,
    # never a locative/existential one) survives; the specific example text now
    # asserted is the fix example that remains, not the original wrong forms.
    assert any(
        "denial" in entry and "anizange nixoxe" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_locative_house_case():
    # 2026-08-29: "endlini kaLincoln" -- reproduced with a different wrong form
    # (dropped augment or a split word boundary) on three separate real runs.
    assert any("endlini" in entry and "khona" in entry for entry in CONFIRMED_TRANSLATION_PITFALLS)


# The four entries below were added 2026-08-29 after a real-job run (with the newly
# universal, informational-only morphology check in place) flagged four more recurring
# patterns: native_phrase_0032's 2nd/3rd-person concord ambiguity, native_phrase_0113's
# dropped self-correction fact, native_phrase_0124's negation flip on a "was false" claim,
# and native_phrase_0149's weakened "exactly" comparison.


def test_confirmed_translation_pitfalls_includes_the_second_vs_third_person_concord_case():
    # 2026-09-02: trimmed of the specific "Mogotsi" worked example (audit-trail
    # prose) -- the actionable rule (2nd/3rd person concord collision) survives.
    assert any(
        "PARTICIPIAL" in entry and "e-" in entry and "wena" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_dropped_self_correction_fact_case():
    assert any(
        "self-correction" in entry and "SYNTAX" in entry and "FACT" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_was_false_negation_flip_case():
    assert any(
        "amanga" in entry and "iqiniso" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_weakened_exactly_case():
    assert any(
        "exactly what happened" in entry and "okufanayo" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_weakened_exactly_case_covers_bare_same_thing():
    # 2026-08-29: two more real runs reproduced the same "-fanayo" weakening on
    # "the same thing" (no "exactly"), so the entry was generalized to also cover
    # this bare form and recommend a demonstrative alternative.
    assert any(
        "into efanayo" in entry and "khona lokho" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


# The three entries below were added 2026-08-29 after re-running the pitfalls above and
# finding they resolved their four target sentences, but a fresh --force rebuild (natural
# translation reruns every sentence, reshuffling which ones trip QA) surfaced three
# different recurring patterns: native_phrase_0095's split-conditional-fragment misread
# as a question, native_phrase_0125's spurious "yini" insertion, and native_phrase_0137's
# "out of context" mistranslated via a document/book idiom.


def test_confirmed_translation_pitfalls_includes_the_split_conditional_fragment_case():
    assert any(
        "should you..." in entry and "CONDITIONAL CLAUSE" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_spurious_yini_case():
    assert any(
        "\"yini\"" in entry and "interrogative pronoun" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_out_of_context_case():
    assert any(
        "out of context" in entry and "umqulu" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_temporal_mask_schema_requires_self_check_on_every_candidate():
    schema = _temporal_mask_schema(1, expected_group_ids=["g1"])
    candidate_schema = schema["properties"]["masks"]["items"]["properties"]["candidates"]["items"]
    assert "self_check" in candidate_schema["required"]
    self_check_schema = candidate_schema["properties"]["self_check"]
    assert self_check_schema["required"] == ["back_translation", "matches_source", "discrepancy_summary"]
    assert self_check_schema["properties"]["discrepancy_summary"]["minLength"] == 1


def test_normalize_temporal_mask_response_carries_self_check_through():
    groups = [{"group_id": "g1"}]
    data = {
        "masks": [{
            "group_id": "g1",
            "candidates": [{
                "candidate_id": "a",
                "spoken_text": "Sawubona",
                "self_check": {
                    "back_translation": "Hello",
                    "matches_source": False,
                    "discrepancy_summary": "dropped the question",
                },
            }],
        }]
    }
    result = _normalize_temporal_mask_response(data, groups, candidate_count=1)
    self_check = result["g1"][0]["self_check"]
    assert self_check == {
        "back_translation": "Hello",
        "matches_source": False,
        "discrepancy_summary": "dropped the question",
    }


def test_normalize_temporal_mask_response_defaults_self_check_when_provider_omits_it():
    # Backward-compatible: a provider response missing self_check (e.g. an older fixture)
    # should not crash normalization -- it defaults to a passing, empty-explanation check.
    groups = [{"group_id": "g1"}]
    data = {"masks": [{"group_id": "g1", "candidates": [{"candidate_id": "a", "spoken_text": "Sawubona"}]}]}
    result = _normalize_temporal_mask_response(data, groups, candidate_count=1)
    assert result["g1"][0]["self_check"] == {
        "back_translation": "",
        "matches_source": True,
        "discrepancy_summary": "",
    }


def test_forward_window_does_not_left_pad_tail():
    groups = [
        {"group_id": f"g{i}", "speaker_id": "S", "source_text": f"E{i}"}
        for i in range(6)
    ]
    window = _temporal_sentence_semantic_window(
        groups, mutable_start=4, mutable_count=2, accepted=[{"group_id": "g0", "spoken_text": "Z0"}]
    )
    assert [item["group_id"] for item in window["blocks"]] == ["g4", "g5"]
    assert window["actual_block_count"] == 2
    assert all(item["locked"] is False for item in window["blocks"])


# --- honorific-abbreviation syllable awareness ---------------------------------------
# Live-run validation found native_phrase_0014 ("Ubusekhaya likaMnu. Lincoln.", a 1360ms
# window) rendering at +92.0% rush -- the deterministic syllable estimator counted the
# written abbreviation "Mnu." as roughly one syllable, when it is actually SYNTHESIZED as
# the full word "Mnumzane" (4 syllables) once the pronunciation dictionary expands it for
# TTS. That hid how tight the block really was from both Python's own warning and (per
# this prompt addition) Grok's internal self-check.

def test_temporal_mask_prompt_teaches_honorific_spoken_length_for_syllable_counting():
    # This last-resort technique only applies once Phase A's translation has already
    # measured oversized, so it lives in the compact prompt, not the translate prompt.
    normalized = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())
    assert "Mnumzane" in normalized
    assert "not the short written abbreviation" in normalized
    assert "dropping the honorific" in normalized
    assert "Try it only after the other techniques above" in normalized


def test_temporal_compact_wire_computes_a_concrete_syllable_target_from_the_real_gap():
    group = {"group_id": "g1", "speaker_id": "S", "source_text": "He said that."}
    a_record = {
        "spoken_text": "Wathi kanjalo ngempela impela okuphelele.",
        "measured_ms": 2000,
        "required_speed_percent": 25.0,
    }
    geometry = {"source_window_ms": 1600}

    wire = _temporal_compact_wire(group, a_record, geometry=geometry)

    expected_current = _estimate_zulu_syllables(_native_dub_tts_ready_text(a_record["spoken_text"]))
    assert wire["current_syllable_count"] == expected_current
    # 25% faster required -> target is current / 1.25, rounded.
    assert wire["target_syllable_count"] == round(expected_current / 1.25)
    assert wire["target_syllable_count"] < wire["current_syllable_count"]
    assert wire["required_speed_percent"] == 25.0


def test_temporal_compact_wire_counts_honorifics_at_their_spoken_form():
    # Same real production case as the honorific-dropping technique above: "Mnu." must
    # count as its full spoken "Mnumzane" (4 syllables), not the ~1-syllable written
    # abbreviation, so the target Grok drafts against isn't silently too generous.
    group = {"group_id": "g1", "speaker_id": "S", "source_text": "You were at Mr. Lincoln's house."}
    spoken_text = "Ubusekhaya likaMnu. Lincoln."
    a_record = {"spoken_text": spoken_text, "measured_ms": 2000, "required_speed_percent": 10.0}
    geometry = {"source_window_ms": 1800}

    wire = _temporal_compact_wire(group, a_record, geometry=geometry)

    written_only = _estimate_zulu_syllables(spoken_text)
    assert wire["current_syllable_count"] > written_only


def test_temporal_compact_schema_has_no_candidate_letter_concept():
    schema = _temporal_compact_schema(["g1", "g2"])
    results_schema = schema["properties"]["results"]
    assert results_schema["minItems"] == 2
    assert results_schema["maxItems"] == 2
    item_schema = results_schema["items"]
    assert set(item_schema["required"]) == {"group_id", "spoken_text", "self_check", "declined_further_compaction"}
    assert "candidate_id" not in item_schema["properties"]
    assert item_schema["properties"]["group_id"]["enum"] == ["g1", "g2"]


def test_normalize_temporal_compact_response_carries_decline_flag_through():
    data = {
        "results": [
            {
                "group_id": "g1",
                "spoken_text": "Sawubona kancane",
                "declined_further_compaction": True,
                "self_check": {
                    "back_translation": "Hello a bit",
                    "matches_source": True,
                    "discrepancy_summary": "faithful, no meaningful difference",
                },
            },
        ]
    }
    result = _normalize_temporal_compact_response(data, ["g1"])
    assert result["g1"]["spoken_text"] == "Sawubona kancane"
    assert result["g1"]["declined_further_compaction"] is True
    assert result["g1"]["self_check"]["matches_source"] is True


def test_normalize_temporal_compact_response_rejects_unrequested_group_id():
    data = {"results": [{"group_id": "unknown", "spoken_text": "x", "self_check": {}, "declined_further_compaction": False}]}
    try:
        _normalize_temporal_compact_response(data, ["g1"])
    except ValueError as exc:
        assert "missing=g1" in str(exc)
        assert "unknown=unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError for identity mismatch")


def test_estimated_syllables_in_the_main_loop_uses_tts_ready_text():
    import mathula_tv.native_dub as native_dub

    written_only = native_dub._estimate_zulu_syllables("Ubusekhaya likaMnu. Lincoln.")
    tts_ready = native_dub._estimate_zulu_syllables(
        native_dub._native_dub_tts_ready_text("Ubusekhaya likaMnu. Lincoln.")
    )
    # The TTS-ready (post-expansion) estimate must be meaningfully higher than the
    # raw-abbreviation estimate -- confirming the fix actually changes the number
    # fed into the rush/syllable-ceiling warnings, not just the prompt text.
    assert tts_ready > written_only + 1


# The two entries below were added 2026-08-30 after a real gpt-5.6-sol-1 job run
# flagged native_phrase_0081 with two distinct real defects in the same sentence:
# a broken English/isiZulu code-switch ("waz compare") and a passive-voice
# agent/patient inversion on "his house was raided".


def test_confirmed_translation_pitfalls_includes_the_general_code_switch_rule():
    # 2026-08-30: the original worked example ("waz compare" -> "-qhathanisa")
    # was removed once the real cause (a genuine zulu_lexicon.py vocabulary
    # gap for "compare"/"contrast") was fixed at the source -- see
    # test_zulu_lexicon.py's test_relevant_entries_includes_supplementary_entries.
    # This general rule stays: a hard, word-independent behavioral constraint
    # against a LAZY bare-English fallback from mere uncertainty, not a per-
    # word memorization aid. (2026-09-08: reworded to resolve a real, direct
    # contradiction with the newly-added deliberate register code-switch
    # guidance -- confirmed via a live diagnostic that the old unconditional
    # "never code-switched" wording was suppressing that guidance in every
    # real production call. The constraint against LAZY code-switching
    # survives; it now explicitly carves out the deliberate, confident case.)
    assert any(
        "UNCERTAIN" in entry and "lazy fallback" in entry and "register-driven" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_passive_voice_inversion_case():
    assert any(
        "-(i)wa" in entry and "waseshwa" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )


def test_confirmed_translation_pitfalls_includes_the_interesting_not_enjoyable_case():
    # 2026-08-30: real gpt-5.6-sol-1 job run, native_phrase_0018 -- "interesting"
    # (neutral) narrowed to "-jabulisa" (enjoyable/pleasing, positive).
    assert any(
        "-jabulisa" in entry and "kwakunakisa" in entry
        for entry in CONFIRMED_TRANSLATION_PITFALLS
    )
