"""Regression test for a real design correction.

The temporal-mask prompt used to tell the model to "prefer a natural realization that
is slightly longer rather than shorter" and that "if natural Zulu is substantially
longer, KEEP IT" -- an explicit one-sided bias toward length. Production evidence
showed this bias, combined with a numeric 0%-12% range, still let a candidate land at
+41.5% over its own budget with no attempt to tighten it: the model wasn't cutting
facts, but it also wasn't recognizing that unscripted English is full of hedges,
false starts, and discourse filler ("so at this point now", "if you will", "he was
somebody that") that carry no factual content and can be tightened for free.

The corrected principle: rushing is always preferred to slowing down, but distance
from the natural (source-equivalent) pace is what makes a rendering *less* natural in
EITHER direction -- a large rush is not something to shrug off just because rushing
is mechanically possible and slowing is not.
"""
from mathula_tv.native_dub import (
    TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT,
    TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT,
    _temporal_mask_wire,
)

_NORMALIZED_PROMPT = " ".join(TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT.split())
_NORMALIZED_COMPACT_PROMPT = " ".join(TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT.split())


def test_length_contract_grounds_translation_in_natural_fluency_not_a_number():
    # Superseded design correction: the translation is no longer benchmarked against a
    # syllable target with a symmetric deviation penalty -- it has no numeric target at
    # all. It is simply the most natural, complete, faithful translation; length is
    # something Python measures afterward, never something Grok aims for while drafting.
    assert "NOT benchmarked against a syllable count" in _NORMALIZED_PROMPT
    assert "single most natural, complete, and faithful isiZulu rendering" in _NORMALIZED_PROMPT


def test_length_contract_permits_tightening_discourse_filler():
    assert "carries no factual content" in _NORMALIZED_PROMPT
    assert "so at this point now" in _NORMALIZED_PROMPT


def test_length_contract_still_protects_deliberate_repetition_and_facts():
    assert "DELIBERATE repetition/emphasis" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "changes the meaning" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT


def test_length_contract_no_longer_defaults_to_preferring_longer():
    # The old one-sided bias must be gone, not just supplemented.
    assert "Prefer a natural realization that is slightly longer rather than shorter" not in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "If natural Zulu is substantially longer, KEEP IT" not in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT


def test_compact_prompt_may_decline_rather_than_force_a_defective_rewrite():
    # Superseded the old ladder's "stop producing further candidates" rule: Phase B
    # attempts exactly one targeted compaction per round and explicitly declines
    # further compaction (rather than silently returning a repeat or a damaged
    # rewrite) once no genuine technique remains that would not reintroduce a defect.
    assert "declined_further_compaction" in _NORMALIZED_COMPACT_PROMPT
    assert "forcing a defective rewrite" in _NORMALIZED_COMPACT_PROMPT


def test_length_contract_names_self_inserted_zulu_hedges_as_a_compression_lever():
    """Regression test for a real gap: the model is told to trim English-side filler, but
    nothing told it to audit its OWN Zulu draft for intensifiers/hedges it added itself --
    which do not exist in the English source, so the filler-spotting guidance above never
    catches them. Confirmed by hand: dropping "okuphelele", "mhlawumbe", and a "wayengumuntu
    owathi" framing clause from a real 253-syllable over-budget block (native_phrase_0001,
    224-syllable ceiling) saved 20 syllables without touching any fact or the deliberate
    "less than 24 hours" repetition.

    This technique only matters once a translation has already measured oversized, so it
    now lives in the Phase-B compaction prompt rather than the Phase-A translate prompt.
    """
    assert "self-inserted intensifiers and hedges" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT
    for hedge_word in ("kakhulu", "mhlawumbe", "okuphelele", "empeleni"):
        assert hedge_word in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT


def test_length_contract_frames_intensifiers_as_a_category_not_a_fixed_list():
    """Regression test for a real gap in the fix above.

    Confirmed in production: on the exact same block the hedge-word list was
    built from (native_phrase_0001), the model reached for "okuningiliziwe"
    (detailed/comprehensive) -- a synonym of "okuphelele" that was never named
    on the original 5-word list -- and it went untouched, since a fixed list
    only teaches recognition of specific words, not the underlying category.
    The guidance must name the CATEGORY (any word that intensifies/emphasizes
    completeness without adding a fact) and say explicitly that named examples
    are not exhaustive, so a different synonym doesn't slip through untouched.
    """
    assert "CATEGORY to hunt for, not a fixed list" in _NORMALIZED_COMPACT_PROMPT
    assert "a synonym is just as droppable" in _NORMALIZED_COMPACT_PROMPT
    assert "okuningiliziwe" in TEMPORAL_MASK_COMPACT_SYSTEM_PROMPT


def test_length_contract_worked_example_covers_framing_clauses_and_repeat_consistency():
    """Regression test for a real, measured shortfall in the previous fix.

    Adding the hedge-word list (see test above) cut native_phrase_0001 from 253 to 243
    syllables against a 224 ceiling -- real progress, but two things ate into the gain:
    the model left the "Wayengumuntu owathi" (he was somebody that) framing clause
    untouched even though "he was somebody that" is already named as filler earlier in
    this same prompt, and it re-derived a LONGER alternate wording for one occurrence of
    a deliberately repeated phrase ("emahoreni angaphansi kwama-24") that it had already
    rendered tightly the first time. A prose rule alone was not enough to make matching
    a named pattern actually happen; a worked before/after example targets that gap.
    """
    assert "WORKED EXAMPLE" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "matching a named pattern is not enough" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "reuse that exact wording for its repeat" in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT


def test_each_sentence_is_its_own_independently_timed_unit():
    # Superseded by the move to sentence-level atomic units (each group_id is now
    # exactly one English sentence): the old SENTENCE-ORDER CORRESPONDENCE WITHIN A
    # BLOCK section and the acknowledgment-collapsing guidance both existed only
    # because one group could carry several sentences that later needed to be split
    # back apart for timing -- that failure mode cannot occur when a group already
    # is one sentence. The prompt instead says each sentence is independently timed.
    assert "SENTENCE-ORDER CORRESPONDENCE WITHIN A BLOCK" not in TEMPORAL_MASK_TRANSLATE_SYSTEM_PROMPT
    assert "each sentence is now its own independently timed and measured unit" in _NORMALIZED_PROMPT.lower()


def test_wire_payload_no_longer_signals_prefer_longer_or_overlong_allowed():
    geometry = {
        "preferred_start_ms": 0,
        "preferred_end_ms": 38640,
    }
    wire = _temporal_mask_wire(
        [{"group_id": "g1", "speaker_id": "S0", "segment_ids": ["seg-1"], "source_text": "English text."}],
        0,
        accepted_previous=[],
        geometry=geometry,
        candidate_count=1,
    )
    mask = wire["temporal_mask"]
    assert "prefer_slightly_longer" not in mask
    assert "overlong_allowed" not in mask
    assert "minimum_extra_syllables" not in mask
    # Superseded: no numeric syllable target/ceiling is sent at all any more -- candidate
    # "a" is grounded purely in natural fluency, not a computed number.
    assert "target_syllables" not in mask
    assert "max_syllables" not in mask
    assert "min_syllables" not in mask
    assert mask["candidate_a_has_no_length_target"] is True
