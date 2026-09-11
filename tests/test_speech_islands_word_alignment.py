"""Regression test for a real optimization the user pushed for directly.

Before this fix, native_phrase_0001 (a 6-sentence block) fell back to a
character-length-proportional split for its ENTIRE group the moment ANY
sentence's word count disagreed with the raw ASR transcript -- caused here by
Pass 1 correcting "British Rabanda Justice College" to "the Bar and a Justice
College" inside sentence 1 only. That inflated required-rush warnings on
sentences that had nothing to do with the correction: island 5 ("We're still
stuck on why the urgency.") was allocated a proportional 2,121ms window
instead of its real 2,640ms window (hand-verified against the raw ASR word
timestamps: "We're" at 36.11s, "urgency." ending at 38.75s), inflating its
reported required speed from a real ~+31% to a shown +62.7%.

Word-level alignment (difflib-based, matching restored words against raw ASR
words) fixes this: only the sentence actually containing the correction loses
exact timing, anchored on either side by the words around it that DO match.
"""

from mathula_tv.speech_islands import build_island_phrase_groups, load_raw_word_index


def _word(word_id: str, text: str, start: float, end: float) -> dict:
    return {"word_id": word_id, "text": text, "start": start, "end": end}


# A compact stand-in for the real production case: sentence 1 has a corrected
# word ("British" -> "the Bar and a"), sentences 2 and 3 are untouched.
_WORDS = [
    _word("w1", "My", 0.0, 0.2),
    _word("w2", "colleague", 0.2, 0.6),
    _word("w3", "British", 0.6, 1.0),  # corrected to "the Bar and a" downstream
    _word("w4", "Nyati.", 1.0, 1.4),
    _word("w5", "Why", 2.0, 2.2),
    _word("w6", "was", 2.2, 2.4),
    _word("w7", "it", 2.4, 2.5),
    _word("w8", "urgent?", 2.5, 3.0),
    _word("w9", "We're", 4.0, 4.2),
    _word("w10", "still", 4.2, 4.5),
    _word("w11", "stuck.", 4.5, 5.0),
]
_WORDS_BY_ID = {w["word_id"]: w for w in _WORDS}
_SEGMENTS_BY_ID = {"seg-1": {"segment_id": "seg-1", "source_word_ids": [w["word_id"] for w in _WORDS]}}

_GROUP = {
    "group_id": "native_phrase_0001",
    "speaker_id": "SPEAKER_00",
    "segment_ids": ["seg-1"],
    "start_ms": 0,
    "source_end_ms": 5000,
    "source_span_ms": 5000,
    # Pass 1 restored "British" into a 4-word phrase -- a real correction like
    # "British Rabanda Justice College" -> "the Bar and a Justice College".
    "source_text": "My colleague the Bar and a Nyati. Why was it urgent? We're still stuck.",
    "parts": [{
        "type": "text", "segment_id": "seg-1",
        "text": "Umlingani wami e-Bar nase-Justice College. Kungani kwakuphuthuma? Sisabambekile.",
    }],
}


def test_only_the_corrected_sentence_loses_exact_alignment():
    pseudo_groups = build_island_phrase_groups(_GROUP, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert pseudo_groups is not None
    assert len(pseudo_groups) == 3
    assert [g["alignment_method"] for g in pseudo_groups] == [
        "word_aligned_partial", "word_exact", "word_exact",
    ]


def test_unaffected_sentences_get_their_real_word_boundaries_untouched():
    pseudo_groups = build_island_phrase_groups(_GROUP, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert pseudo_groups is not None
    # Sentence 2 ("Why was it urgent?") and sentence 3 ("We're still stuck.")
    # are entirely unaffected by sentence 1's correction and must land on their
    # exact real ASR timestamps, not a group-wide proportional guess.
    assert pseudo_groups[1]["start_ms"] == 2000 and pseudo_groups[1]["source_end_ms"] == 3000
    assert pseudo_groups[2]["start_ms"] == 4000 and pseudo_groups[2]["source_end_ms"] == 5000


def test_corrected_sentence_still_anchors_to_its_real_boundary():
    # Even though "the Bar and a" don't match any raw word, "My"/"colleague" and
    # "Nyati." on either side of the correction DO -- so the sentence's overall
    # span should still land close to its real boundary rather than drifting.
    pseudo_groups = build_island_phrase_groups(_GROUP, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert pseudo_groups is not None
    assert pseudo_groups[0]["start_ms"] == 0
    assert pseudo_groups[0]["source_end_ms"] == 1400


def test_real_production_case_recovers_the_correct_island_5_window():
    """The exact real job data from this session: native_phrase_0001's final sentence."""
    words = [
        _word("w0", "Stuck.", 34.00, 34.60),  # a preceding sentence, so there is something to split
        _word("w1", "We're", 36.11, 36.39),
        _word("w2", "still", 36.39, 36.91),
        _word("w3", "stuck", 36.91, 37.31),
        _word("w4", "on", 37.31, 37.55),
        _word("w5", "why", 37.55, 37.95),
        _word("w6", "the", 37.95, 38.07),
        _word("w7", "urgency.", 38.11, 38.75),
    ]
    words_by_id, segments_by_id = load_raw_word_index({
        "words": words,
        "segments": [{"segment_id": "seg-1", "source_word_ids": [w["word_id"] for w in words]}],
    })
    group = {
        "group_id": "native_phrase_0001",
        "speaker_id": "SPEAKER_00",
        "segment_ids": ["seg-1"],
        "start_ms": 34000,
        "source_end_ms": 38750,
        "source_span_ms": 4750,
        "source_text": "Stuck. We're still stuck on why the urgency.",
        "parts": [{
            "type": "text", "segment_id": "seg-1",
            "text": "Bambekile. Sisabambekile ukuthi kungani ukuphuthuma.",
        }],
    }
    from mathula_tv.speech_islands import derive_english_islands

    islands = derive_english_islands(group, words_by_id, segments_by_id)
    assert islands is not None
    assert len(islands) == 2
    final_island = islands[1]
    # 2,640ms real window (36.11s-38.75s) -- not the flawed 2,121ms proportional
    # estimate that inflated this island's reported required speed to +62.7%
    # when its real figure is closer to +31%.
    assert final_island["start_ms"] == 36110
    assert final_island["end_ms"] == 38750
    assert final_island["end_ms"] - final_island["start_ms"] == 2640
    assert final_island["alignment_method"] == "word_exact"
