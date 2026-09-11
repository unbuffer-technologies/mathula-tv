"""Tests for the back-translation QA phase: a genuine, separate, Python-orchestrated
verification pass that independently back-translates every committed sentence and
flags any whose back-translation drifts from the original English.

Confirmed real production gap this exists to close: a committed group with
internal_revision_cycles=null (the translation prompt's own self-reported QA loop
either didn't run or didn't report doing so) shipped "When did you speak to Mister
Mogotsi?" translated as isiZulu that independently back-translates to "When did you
say to speak with Mr. Mogotsi?" -- a different question, approved without anyone
(human or model) actually comparing the two.
"""
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.native_dub import (
    QA_BACK_TRANSLATION_SCHEMA_VERSION,
    QA_BACK_TRANSLATION_SYSTEM_PROMPT,
    QA_DIRECT_ADEQUACY_SYSTEM_PROMPT,
    QA_REPAIR_SEVERITY_FLOOR,
    ZULU_GRAMMAR_FOUNDATION,
    _qa_back_translation_schema,
    _qa_direct_adequacy_schema,
    _qa_sentence_is_negation_sensitive,
    _qa_sentence_pairs_for_group,
    _qa_severity_at_least,
    _qa_severity_max,
    _request_qa_back_translation_batch,
    _request_qa_direct_adequacy_batch,
    native_dub_paths,
    run_native_qa_back_translation,
)


class FakeQAProvider:
    def __init__(self, *, verdicts_by_batch=None, verdict_fn=None):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.provider = "fake-grok"
        self.deployment = "grok-4.3"
        self.config.deployment = "grok-4.3"
        self.calls: list[dict] = []
        self._verdicts_by_batch = list(verdicts_by_batch or [])
        self._verdict_fn = verdict_fn

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self._verdict_fn is not None:
            verdicts = self._verdict_fn(kwargs["payload"]["pairs"])
        else:
            verdicts = self._verdicts_by_batch.pop(0)
        return SimpleNamespace(data={"verdicts": verdicts}, input_tokens=50, output_tokens=20, attempts=1)


class DualCheckQAProvider:
    """Routes by operation name so both the back-translation batch AND the independent
    direct-adequacy batch (fired within the same _audit() call when direct_adequacy=True)
    each get their own scripted response instead of sharing one queue.
    """

    def __init__(self, *, bt_verdicts_by_batch=None, da_verdicts_by_batch=None):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.provider = "fake-grok"
        self.deployment = "grok-4.3"
        self.config.deployment = "grok-4.3"
        self.calls: list[dict] = []
        self._bt = list(bt_verdicts_by_batch or [])
        self._da = list(da_verdicts_by_batch or [])

    def complete_json(self, *, operation, **kwargs):
        self.calls.append({"operation": operation, **kwargs})
        if operation == "native_qa_back_translation_batch":
            verdicts = self._bt.pop(0)
        elif operation == "native_qa_direct_adequacy_batch":
            verdicts = self._da.pop(0)
        else:
            raise AssertionError(f"unexpected operation: {operation}")
        return SimpleNamespace(data={"verdicts": verdicts}, input_tokens=10, output_tokens=5, attempts=1)


def _group(group_id, en, zu):
    return {"group_id": group_id, "source_text": en, "spoken_text": zu, "speaker_id": "SPEAKER_00"}


# --- _qa_sentence_pairs_for_group -------------------------------------------------

def test_pairs_split_into_per_sentence_refs_when_counts_agree():
    group = _group("g1", "Hello there. When did you speak to him?", "Sawubona. Wathi nini ukukhuluma naye?")
    pairs = _qa_sentence_pairs_for_group(group)
    assert [ref for ref, _, _ in pairs] == ["g1__s00", "g1__s01"]
    assert pairs[0][1] == "Hello there."
    assert pairs[1][1] == "When did you speak to him?"


def test_pairs_fall_back_to_whole_group_when_sentence_counts_disagree():
    group = _group("g1", "One. Two. Three.", "Kunye kuphela.")
    pairs = _qa_sentence_pairs_for_group(group)
    assert pairs == [("g1", "One. Two. Three.", "Kunye kuphela.")]


def test_pairs_do_not_split_a_single_sentence_group():
    group = _group("g1", "Just one sentence.", "Umusho owodwa nje.")
    pairs = _qa_sentence_pairs_for_group(group)
    assert pairs == [("g1", "Just one sentence.", "Umusho owodwa nje.")]


def test_pairs_respect_title_abbreviations_on_both_sides():
    # Real production case: "Mr."/"uMnu." must not be treated as a sentence
    # boundary on either side -- the same fix speech_islands.py's splitter needed.
    group = _group(
        "g1",
        "Mr. Adams spoke. Then Mrs. Adams left.",
        "Mnu. Adams wakhuluma. Base uNkz. Adams wahamba.",
    )
    pairs = _qa_sentence_pairs_for_group(group)
    assert [en for _, en, _ in pairs] == ["Mr. Adams spoke.", "Then Mrs. Adams left."]


# --- v2 prompt refinements ---------------------------------------------------------
# Regression tests for a real calibration pass: a live run against job
# 33cd7b46d55647e39cc1e8d26ab962ed correctly caught two confirmed bugs (a
# mistranslation and a wrong-question drift) but also flagged ~23% of sentences,
# many of them false positives from two specific, fixable blind spots -- isiZulu's
# pro-drop grammar (no separate subject word) misread as a dropped pronoun, and
# legitimate discourse-filler dropping (this pipeline's own translation policy)
# misread as an omitted proposition. Flags with an empty discrepancy_summary
# correlated strongly with the false positives, so summaries are now mandatory.

def test_prompt_explains_isizulu_pro_drop_grammar():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "pro-drop language" in normalized
    assert "back-translation completeness gap, not a translation error" in normalized


def test_prompt_excludes_legitimate_filler_dropping_from_drift():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "this pipeline's own deliberate, correct translation policy, not an omission" in normalized


def test_prompt_requires_a_specific_non_vague_discrepancy_summary():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "A vague or generic summary is not acceptable" in normalized
    assert "Never leave it blank" in normalized


def test_schema_requires_a_non_empty_discrepancy_summary_for_every_verdict():
    schema = _qa_back_translation_schema(["g1"])
    item_schema = schema["properties"]["verdicts"]["items"]
    assert "discrepancy_summary" in item_schema["required"]
    assert item_schema["properties"]["discrepancy_summary"]["minLength"] == 1


# --- v3 prompt refinement: a principled test, not just a longer example list -----
# A second live-run pass on the same real job found the v2 example list still
# missed real filler/hedge cases it didn't happen to name ("he was somebody
# that", "for lack of a better term", "to be honest with you", "that's the word
# she used") and flagged legitimate lexical/register variance (synonym choice,
# greeting simplification) as if it were content loss. Rather than keep growing
# the example list forever, the prompt now teaches a general test to apply.

def test_prompt_teaches_a_general_test_not_just_a_fixed_example_list():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "THE CENTRAL TEST" in normalized
    assert "category to apply by reasoning, not a" in normalized
    assert "fixed list to pattern-match against" in normalized


def test_prompt_names_the_confirmed_framing_clause_and_hedge_gap_cases():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    for phrase in (
        "he was somebody that", "for lack of a better term", "to be honest with you",
        "that's the word she used",
    ):
        assert phrase in normalized


def test_prompt_exempts_greeting_simplification():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "greeting or pleasantry" in normalized
    assert "do not carry factual content requiring" in normalized


def test_prompt_distinguishes_lexical_variance_from_real_drift():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "Lexical/register variance" in normalized
    assert "not drift by itself" in normalized
    # The one real, confirmed case where a word-choice difference WAS drift must
    # still be named, so the exemption doesn't overcorrect into ignoring everything.
    assert "EXACTLY what" in normalized and "SIMILAR to what" in normalized


def test_prompt_requires_flagging_internal_repetition_independent_of_meaning():
    # Regression test for a real detection REGRESSION found while validating v3
    # against the live job: native_phrase_0049's confirmed sentence-duplication
    # bug ("Yes, and so what is in paragraph four is in fact false." committed as
    # the SAME isiZulu sentence twice back-to-back) was correctly flagged by v1
    # and v2, then stopped being flagged once the v3 lexical-variance exemption
    # was added -- the model started reading the duplicate as harmless restating
    # of the same true claim in different words. Duplication is a structural
    # defect independent of meaning-fidelity and needs its own explicit rule.
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "SEPARATELY from meaning-fidelity, check the COMMITTED ISIZULU ITSELF for internal repetition" in normalized
    assert "Do not let the lexical-variance guidance below excuse this" in normalized


def test_prompt_still_protects_a_genuine_impersonal_construction_shift():
    # Regression guard against over-correcting the pro-drop exemption: a real
    # isiZulu impersonal/passive construction (not just an omitted concord) must
    # still be flaggable -- confirmed real case: "you don't say X" rendered
    # impersonally as "it is not said X" via the aku- impersonal prefix.
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "genuinely shifted to an impersonal/" in normalized
    assert "a real isiZulu impersonal prefix, not just an omitted concord" in normalized


def test_prompt_explains_preceding_english_ellipsis_recovery():
    # 2026-08-30: native_phrase_0151's real false-positive "addition" flag --
    # see test_batch_request_includes_preceding_english_when_provided.
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "preceding_english" in normalized
    assert "I could have." in normalized
    assert 'category="addition"' in normalized


# --- _request_qa_back_translation_batch --------------------------------------------

def test_batch_request_maps_verdicts_back_to_refs():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
        {"ref": "g1__s01", "back_translation": "When did you say to speak?", "matches_original": False,
         "discrepancy_summary": "different question"},
    ]])
    items = [("g1__s00", "Hello.", "Sawubona."), ("g1__s01", "When did you speak?", "Wathi nini ukukhuluma?")]
    verdicts, usage = _request_qa_back_translation_batch(provider=provider, items=items)
    assert verdicts["g1__s00"]["matches_original"] is True
    assert verdicts["g1__s01"]["matches_original"] is False
    assert verdicts["g1__s01"]["discrepancy_summary"] == "different question"
    assert usage["input_tokens"] == 50


def test_batch_request_includes_context_ledger_in_payload():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
    ]])
    ledger = {"summary": "A single witness testifies.", "entities": [], "speaker_roles": []}
    _request_qa_back_translation_batch(
        provider=provider, items=[("g1__s00", "Hello.", "Sawubona.")], context_ledger=ledger,
    )
    assert provider.calls[0]["payload"]["context_ledger"] == ledger


def test_batch_request_defaults_context_ledger_to_empty_when_omitted():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
    ]])
    _request_qa_back_translation_batch(provider=provider, items=[("g1__s00", "Hello.", "Sawubona.")])
    assert provider.calls[0]["payload"]["context_ledger"] == {}


def test_batch_request_includes_preceding_english_when_provided():
    # 2026-08-30: real defect -- native_phrase_0151 ("I could have.") correctly
    # recovered its predicate from the immediately preceding sentence ("...I
    # could have worded it better") but was flagged as fabricated "addition"
    # by a QA check that had never seen that preceding sentence at all.
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1", "back_translation": "I could have.", "matches_original": True},
    ]])
    _request_qa_back_translation_batch(
        provider=provider, items=[("g1", "I could have.", "Ngabe ngikubeke kangcono.")],
        preceding_english_by_group_id={"g1": "I could have worded it better, Commissioner."},
    )
    assert provider.calls[0]["payload"]["pairs"][0]["preceding_english"] == (
        "I could have worded it better, Commissioner."
    )


def test_batch_request_omits_preceding_english_when_no_entry_for_that_group():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g2", "back_translation": "Hello.", "matches_original": True},
    ]])
    _request_qa_back_translation_batch(
        provider=provider, items=[("g2", "Hello.", "Sawubona.")],
        preceding_english_by_group_id={"g1": "Some other sentence."},
    )
    assert "preceding_english" not in provider.calls[0]["payload"]["pairs"][0]


def test_batch_request_resolves_preceding_english_via_base_group_id_for_split_refs():
    # A group split into multiple sentence-refs (g1__s00, g1__s01, ...) should
    # all resolve to the SAME preceding_english entry, keyed by the base
    # group_id, not the per-sentence ref.
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "A.", "matches_original": True},
        {"ref": "g1__s01", "back_translation": "B.", "matches_original": True},
    ]])
    _request_qa_back_translation_batch(
        provider=provider,
        items=[("g1__s00", "A.", "A."), ("g1__s01", "B.", "B.")],
        preceding_english_by_group_id={"g1": "Prior context."},
    )
    pairs = provider.calls[0]["payload"]["pairs"]
    assert pairs[0]["preceding_english"] == "Prior context."
    assert pairs[1]["preceding_english"] == "Prior context."


def test_batch_request_includes_zulu_grammar_foundation_in_payload():
    # 2026-08-29: the QA check itself needs the same mood/concord/verbal-extension
    # rule system as generation does, to judge a candidate on real grammatical
    # grounds rather than surface plausibility alone.
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
    ]])
    _request_qa_back_translation_batch(provider=provider, items=[("g1__s00", "Hello.", "Sawubona.")])
    assert provider.calls[0]["payload"]["zulu_grammar_foundation"] == ZULU_GRAMMAR_FOUNDATION


def test_batch_request_flags_a_missing_verdict_instead_of_crashing():
    # Same non-fatal-on-mismatch pattern as the timing-repair batch fix earlier
    # this session: a dropped ref must never crash the whole QA run.
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
    ]])
    items = [("g1__s00", "Hello.", "Sawubona."), ("g1__s01", "Bye.", "Sala kahle.")]
    verdicts, _ = _request_qa_back_translation_batch(provider=provider, items=items)
    assert verdicts["g1__s01"]["matches_original"] is False
    assert verdicts["g1__s01"]["verdict_missing"] is True


def test_batch_request_ignores_duplicate_and_unknown_refs():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1__s00", "back_translation": "Hello.", "matches_original": True},
        {"ref": "g1__s00", "back_translation": "Hi there.", "matches_original": False},
        {"ref": "gX", "back_translation": "extra", "matches_original": True},
    ]])
    items = [("g1__s00", "Hello.", "Sawubona.")]
    verdicts, _ = _request_qa_back_translation_batch(provider=provider, items=items)
    assert verdicts["g1__s00"]["matches_original"] is True  # first occurrence wins
    assert "gX" not in verdicts


# --- run_native_qa_back_translation --------------------------------------------

def _write_pass2(job_root, groups):
    atomic_write_json(job_root / "native_dub" / "pass2_natural_translation.json", {
        "temporal_mask_mode": True,
        "input_sha256": "abc123",
        "model": "grok-4.3",
        "temporal_mask_groups": groups,
    })


def test_run_flags_a_real_drifted_sentence_end_to_end(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [
        _group("native_phrase_0008", "When did you speak to Mister Mogotsi?", "Wathi nini ukukhuluma noMnu. Mogotsi?"),
        _group("native_phrase_0009", "I spoke to him once.", "Ngakhuluma naye kanye."),
    ])

    def verdict_fn(pairs):
        out = []
        for pair in pairs:
            if pair["ref"] == "native_phrase_0008":
                out.append({
                    "ref": pair["ref"], "back_translation": "When did you say to speak with Mr. Mogotsi?",
                    "matches_original": False, "discrepancy_summary": "different question: intent vs action",
                })
            else:
                out.append({"ref": pair["ref"], "back_translation": "I spoke to him once.", "matches_original": True})
        return out

    provider = FakeQAProvider(verdict_fn=verdict_fn)
    # repair=False: this test is about DETECTION in isolation -- the auto-repair loop
    # (which would otherwise fire here since a sentence is flagged) has its own
    # dedicated coverage in test_native_qa_repair.py.
    result = run_native_qa_back_translation(job_root=job_root, provider=provider, repair=False, progress=None)

    assert result["sentence_count"] == 2
    assert result["flagged_count"] == 1
    assert result["requires_review"] is True
    assert result["flagged"][0]["ref"] == "native_phrase_0008"
    assert result["flagged"][0]["discrepancy_summary"] == "different question: intent vs action"
    assert result["schema_version"] == QA_BACK_TRANSLATION_SCHEMA_VERSION

    paths = native_dub_paths(job_root)
    assert paths.qa_back_translation.is_file()
    on_disk = read_json(paths.qa_back_translation)
    assert on_disk["flagged_count"] == 1


def test_run_reports_zero_flags_when_everything_matches(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello.", "Sawubona.")])
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1", "back_translation": "Hello.", "matches_original": True},
    ]])
    result = run_native_qa_back_translation(job_root=job_root, provider=provider, progress=None)
    assert result["flagged_count"] == 0
    assert result["requires_review"] is False


def test_run_reuses_matching_checkpoint_without_a_new_provider_call(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello.", "Sawubona.")])
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1", "back_translation": "Hello.", "matches_original": True},
    ]])
    run_native_qa_back_translation(job_root=job_root, provider=provider, progress=None)
    assert len(provider.calls) == 1

    result = run_native_qa_back_translation(job_root=job_root, provider=provider, progress=None)
    assert len(provider.calls) == 1  # no second call
    assert result["flagged_count"] == 0


def test_run_requires_temporal_mask_translation(tmp_path):
    job_root = tmp_path / "job"
    atomic_write_json(job_root / "native_dub" / "pass2_natural_translation.json", {"temporal_mask_mode": False})
    with pytest.raises(ValueError, match="temporal-mask"):
        run_native_qa_back_translation(job_root=job_root, provider=FakeQAProvider(), progress=None)


def test_run_requires_pass2_to_exist(tmp_path):
    job_root = tmp_path / "job"
    with pytest.raises(ValueError, match="native-translate"):
        run_native_qa_back_translation(job_root=job_root, provider=FakeQAProvider(), progress=None)


# --- MQM-lite category/severity ----------------------------------------------------
# Research prompted this: back-translation alone gives a bare pass/fail, but the
# literature's own gold-standard human framework (MQM) and its LLM-automated form
# (GEMBA-MQM) both score drift with a category and a severity so a reviewer can triage
# instead of treating every flag as equally urgent. Confirmed real production case this
# also targets: native_phrase_0119's flagged "-nga-" ambiguity (isiZulu's negation/
# potential-mood infix) is exactly the kind of finding that should never be silently
# auto-accepted as low-stakes, hence the negation floor below.

def test_schema_requires_category_and_severity_on_every_verdict():
    schema = _qa_back_translation_schema(["g1"])
    item_schema = schema["properties"]["verdicts"]["items"]
    assert "category" in item_schema["required"]
    assert "severity" in item_schema["required"]
    assert "negation" in item_schema["properties"]["category"]["enum"]
    assert set(item_schema["properties"]["severity"]["enum"]) >= {"none", "minor", "major", "critical"}


def test_prompt_teaches_mqm_lite_category_and_severity():
    normalized = " ".join(QA_BACK_TRANSLATION_SYSTEM_PROMPT.split())
    assert "category" in normalized and "severity" in normalized
    assert '"negation"' in normalized
    assert "single most dangerous class of drift" in normalized


def test_batch_request_parses_explicit_category_and_severity():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {
            "ref": "g1", "back_translation": "He is here.", "matches_original": False,
            "discrepancy_summary": "dropped negation", "category": "negation", "severity": "critical",
        },
    ]])
    verdicts, _ = _request_qa_back_translation_batch(
        provider=provider, items=[("g1", "He is not here.", "Ukhona lapha.")],
    )
    assert verdicts["g1"]["category"] == "negation"
    assert verdicts["g1"]["severity"] == "critical"


def test_batch_request_defaults_unclassified_flag_to_major_not_minor():
    # An older/simulated caller (or a provider response missing the new fields) must
    # never have a real flag silently downgraded to "minor" and auto-accepted.
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1", "back_translation": "Bye.", "matches_original": False, "discrepancy_summary": "x"},
    ]])
    verdicts, _ = _request_qa_back_translation_batch(provider=provider, items=[("g1", "Hi.", "Sawubona.")])
    assert verdicts["g1"]["severity"] == "major"
    assert verdicts["g1"]["category"] == "other"


def test_missing_verdict_is_escalated_to_critical_not_downgraded():
    provider = FakeQAProvider(verdicts_by_batch=[[]])
    verdicts, _ = _request_qa_back_translation_batch(provider=provider, items=[("g1", "Hi.", "Sawubona.")])
    assert verdicts["g1"]["severity"] == "critical"
    assert verdicts["g1"]["verdict_missing"] is True


def test_severity_helpers_rank_correctly():
    assert _qa_severity_max("minor", "critical") == "critical"
    assert _qa_severity_max("major", "minor") == "major"
    assert _qa_severity_at_least("minor", "major") == "major"
    assert _qa_severity_at_least("critical", "major") == "critical"


# --- negation-sensitivity floor ------------------------------------------------------

def test_negation_sensitive_detects_common_cues():
    assert _qa_sentence_is_negation_sensitive("He didn't say that.")
    assert _qa_sentence_is_negation_sensitive("There is no evidence.")
    assert _qa_sentence_is_negation_sensitive("Nobody was there.")
    assert not _qa_sentence_is_negation_sensitive("He was there yesterday.")


def test_negation_bearing_flag_is_floored_to_major_even_if_model_calls_it_minor(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "He did not confirm the date.", "Waqinisekisa usuku.")])
    provider = FakeQAProvider(verdicts_by_batch=[[
        {
            "ref": "g1", "back_translation": "He confirmed the date.", "matches_original": False,
            "discrepancy_summary": "dropped negation", "category": "negation", "severity": "minor",
        },
    ]])
    result = run_native_qa_back_translation(job_root=job_root, provider=provider, repair=False, progress=None)
    assert result["flagged_count"] == 1
    assert result["flagged"][0]["severity"] == "major"
    assert result["flagged"][0]["negation_sensitive"] is True


# --- severity-floor triage: minor findings are accepted, not auto-repaired ---------

def test_minor_severity_finding_is_accepted_without_triggering_a_repair_call(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello there.", "Sawubona.")])
    provider = FakeQAProvider(verdicts_by_batch=[[
        {
            "ref": "g1", "back_translation": "Hi.", "matches_original": False,
            "discrepancy_summary": "slightly less formal register", "category": "fluency", "severity": "minor",
        },
    ]])
    result = run_native_qa_back_translation(job_root=job_root, provider=provider, progress=None)
    assert result["flagged_count"] == 0
    assert result["requires_review"] is False
    assert result["accepted_minor_count"] == 1
    assert result["accepted_minor"][0]["ref"] == "g1"
    assert len(provider.calls) == 1  # no repair batch fired for a minor-only finding


def test_group_with_mixed_severities_stays_repair_eligible_as_a_whole(tmp_path):
    # Repair rewrites a whole group's Zulu text at once, so a minor sentence sharing a
    # group with a critical one must not be peeled off into accepted_minor -- the repair
    # is about to overwrite that group's text anyway.
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello there. He did not arrive.", "Sawubona. Ufikile.")])

    def verdict_fn(pairs):
        return [
            {
                "ref": pair["ref"], "back_translation": pair["original_english"], "matches_original": False,
                "discrepancy_summary": "x",
                "category": "fluency" if pair["ref"].endswith("s00") else "negation",
                "severity": "minor" if pair["ref"].endswith("s00") else "critical",
            }
            for pair in pairs
        ]

    provider = FakeQAProvider(verdict_fn=verdict_fn)
    result = run_native_qa_back_translation(job_root=job_root, provider=provider, repair=False, progress=None)
    assert result["accepted_minor_count"] == 0
    assert result["flagged_count"] == 2


# --- independent direct-adequacy cross-check ----------------------------------------

def test_direct_adequacy_defaults_to_disabled_at_the_function_level(tmp_path):
    # CLI opts in by default; library callers (and every pre-existing test) keep the
    # original single-check cost unless they explicitly ask for the second opinion.
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello.", "Sawubona.")])
    provider = FakeQAProvider(verdicts_by_batch=[[
        {"ref": "g1", "back_translation": "Hello.", "matches_original": True},
    ]])
    run_native_qa_back_translation(job_root=job_root, provider=provider, progress=None)
    assert len(provider.calls) == 1
    assert provider.calls[0]["operation"] == "native_qa_back_translation_batch"


def test_direct_adequacy_schema_requires_no_back_translation_field():
    # The whole point of this check is that it must NOT go through a back-translation
    # step -- confirm the schema doesn't even offer that field.
    schema = _qa_direct_adequacy_schema(["g1"])
    item_schema = schema["properties"]["verdicts"]["items"]
    assert "back_translation" not in item_schema["properties"]
    assert "issue_summary" in item_schema["required"]


def test_direct_adequacy_prompt_explains_why_it_is_a_genuinely_separate_method():
    normalized = " ".join(QA_DIRECT_ADEQUACY_SYSTEM_PROMPT.split())
    assert "do NOT translate the isiZulu back into English" in normalized
    assert "self-consistent round-trip" in normalized


def test_direct_adequacy_flags_a_drift_back_translation_alone_missed(tmp_path):
    job_root = tmp_path / "job"
    _write_pass2(job_root, [_group("g1", "Hello there.", "Sawubona.")])
    provider = DualCheckQAProvider(
        bt_verdicts_by_batch=[[
            {
                "ref": "g1", "back_translation": "Hello there.", "matches_original": True,
                "discrepancy_summary": "faithful", "category": "none", "severity": "none",
            },
        ]],
        da_verdicts_by_batch=[[
            {
                "ref": "g1", "matches_original": False, "issue_summary": "tone shift, too casual",
                "category": "fluency", "severity": "major",
            },
        ]],
    )
    result = run_native_qa_back_translation(
        job_root=job_root, provider=provider, repair=False, direct_adequacy=True, progress=None,
    )
    assert result["flagged_count"] == 1
    assert result["flagged"][0]["checks_flagged"] == ["direct_adequacy"]
    assert result["flagged"][0]["direct_adequacy_issue"] == "tone shift, too casual"


def test_direct_adequacy_request_maps_verdicts_back_to_refs():
    provider = FakeQAProvider(verdicts_by_batch=[[
        {
            "ref": "g1", "matches_original": False, "issue_summary": "different agent",
            "category": "attribution", "severity": "critical",
        },
    ]])
    verdicts, usage = _request_qa_direct_adequacy_batch(
        provider=provider, items=[("g1", "She spoke.", "Wakhuluma yena.")],
    )
    assert verdicts["g1"]["matches_original"] is False
    assert verdicts["g1"]["category"] == "attribution"
    assert verdicts["g1"]["severity"] == "critical"
    assert usage["input_tokens"] == 50


def test_repair_floor_constant_excludes_minor():
    assert "minor" not in QA_REPAIR_SEVERITY_FLOOR
    assert {"major", "critical"} <= QA_REPAIR_SEVERITY_FLOOR
