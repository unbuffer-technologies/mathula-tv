import wave
from pathlib import Path

import pytest

from mathula_tv.speech_islands import (
    MIN_ISLAND_DURATION_MS,
    ENGLISH_TITLE_ABBREVIATIONS,
    _pairs_have_reliable_correspondence,
    split_into_sentences,
    build_island_phrase_groups,
    derive_english_islands,
    split_zulu_islands,
    stitch_islands_to_group_wav,
)


def _word(word_id: str, text: str, start: float, end: float) -> dict:
    return {"word_id": word_id, "text": text, "start": start, "end": end}


def _segment(segment_id: str, word_ids: list[str]) -> dict:
    return {"segment_id": segment_id, "source_word_ids": word_ids}


# Modeled on a real production block (job 33cd7b46..., native_phrase_0001): a single
# ASR segment spanning three short sentences, same shape as the real 124-word/6-sentence
# case but trimmed to a size that's readable in a test.
_WORDS = [
    _word("w1", "Good", 0.0, 0.4),
    _word("w2", "morning", 0.4, 0.9),
    _word("w3", "Ayanda.", 0.9, 1.3),
    _word("w4", "Why", 1.6, 1.8),
    _word("w5", "was", 1.8, 2.0),
    _word("w6", "it", 2.0, 2.1),
    _word("w7", "urgent?", 2.1, 2.6),
    _word("w8", "He", 3.0, 3.2),
    _word("w9", "laid", 3.2, 3.5),
    _word("w10", "charges.", 3.5, 4.0),
]
_WORDS_BY_ID = {w["word_id"]: w for w in _WORDS}
_SEGMENTS_BY_ID = {"seg-1": _segment("seg-1", [w["word_id"] for w in _WORDS])}

_GROUP = {
    "group_id": "native_phrase_0001",
    "speaker_id": "SPEAKER_00",
    "segment_ids": ["seg-1"],
    "start_ms": 0,
    "source_end_ms": 4000,
    "source_span_ms": 4000,
    "source_text": "Good morning Ayanda. Why was it urgent? He laid charges.",
    "parts": [{"type": "text", "segment_id": "seg-1", "text": "Sawubona Ayanda. Kungani kwakuphuthuma? Wafaka amacala."}],
}


def test_derive_english_islands_assigns_real_word_timestamps():
    islands = derive_english_islands(_GROUP, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert islands is not None
    assert [i["source_text"] for i in islands] == [
        "Good morning Ayanda.",
        "Why was it urgent?",
        "He laid charges.",
    ]
    assert islands[0]["start_ms"] == 0 and islands[0]["end_ms"] == 1300
    assert islands[1]["start_ms"] == 1600 and islands[1]["end_ms"] == 2600
    assert islands[2]["start_ms"] == 3000 and islands[2]["end_ms"] == 4000
    assert [i["island_id"] for i in islands] == [
        "native_phrase_0001__isl00",
        "native_phrase_0001__isl01",
        "native_phrase_0001__isl02",
    ]


def test_derive_english_islands_single_sentence_is_a_noop():
    group = {**_GROUP, "source_text": "Good morning Ayanda."}
    assert derive_english_islands(group, _WORDS_BY_ID, _SEGMENTS_BY_ID) is None


def test_derive_english_islands_word_count_mismatch_only_degrades_the_affected_sentence():
    # Pass 1 restoration inserted extra words into the FIRST sentence only (e.g. a
    # substantive STT correction like "British Rabanda" -> "the Bar and a" --
    # confirmed on a real production job). Only that sentence should lose exact
    # alignment; sentences 2 and 3, unaffected by the correction, must still get
    # real word-exact timestamps -- this is the whole point of aligning by word
    # content instead of falling back to a block-wide proportional guess the
    # moment any single word count disagrees anywhere in the group.
    group = {**_GROUP, "source_text": "Good morning to you Ayanda. Why was it urgent? He laid charges."}
    islands = derive_english_islands(group, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert islands is not None
    assert [i["alignment_method"] for i in islands] == [
        "word_aligned_partial", "word_exact", "word_exact",
    ]
    # The affected sentence still lands on its real boundary (anchored by the
    # unaffected "Good"/"morning"/"Ayanda." around the inserted "to you").
    assert islands[0]["start_ms"] == 0 and islands[0]["end_ms"] == 1300
    assert islands[1]["start_ms"] == 1600 and islands[1]["end_ms"] == 2600
    assert islands[2]["start_ms"] == 3000 and islands[2]["end_ms"] == 4000


def test_derive_english_islands_missing_segment_uses_proportional_fallback():
    group = {**_GROUP, "segment_ids": ["seg-does-not-exist"]}
    islands = derive_english_islands(group, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert islands is not None
    assert all(i["alignment_method"] == "proportional_fallback" for i in islands)


def test_derive_english_islands_does_not_itself_reject_short_individual_islands():
    # derive_english_islands no longer rejects a too-short individual island --
    # that decision moves to build_island_phrase_groups, which merges it into a
    # neighbor instead of discarding the whole group's split (see the merge tests
    # below). A degenerate group WINDOW (no real time at all) is still rejected,
    # since there is nothing to allocate to any sentence.
    words = [
        _word("w1", "Hi.", 0.0, 0.05),  # far below MIN_ISLAND_DURATION_MS
        _word("w2", "Why", 1.0, 1.2),
        _word("w3", "urgent?", 1.2, 1.6),
    ]
    words_by_id = {w["word_id"]: w for w in words}
    segments_by_id = {"seg-1": _segment("seg-1", [w["word_id"] for w in words])}
    group = {**_GROUP, "source_text": "Hi. Why urgent?"}
    islands = derive_english_islands(group, words_by_id, segments_by_id)
    assert islands is not None
    assert islands[0]["end_ms"] - islands[0]["start_ms"] < MIN_ISLAND_DURATION_MS

    degenerate_window_group = {**_GROUP, "segment_ids": ["seg-does-not-exist"], "start_ms": 0, "source_end_ms": 0}
    assert derive_english_islands(degenerate_window_group, _WORDS_BY_ID, _SEGMENTS_BY_ID) is None


def test_split_zulu_islands_matches_sentence_punctuation():
    assert split_zulu_islands("Sawubona Ayanda. Kungani kwakuphuthuma? Wafaka amacala.") == [
        "Sawubona Ayanda.",
        "Kungani kwakuphuthuma?",
        "Wafaka amacala.",
    ]


def test_split_into_sentences_does_not_split_after_a_title_abbreviation():
    # Real production case (job 33cd7b46..., seg-00007): a single, correctly
    # punctuated ASR segment got shredded into 5 garbage fragments (one a bare
    # "Mr.") because the naive punctuation splitter treated every period after
    # "Mr." as a sentence boundary. The translation model then padded those
    # fragments with invented content to make each feel complete -- a real
    # "do not invent information" violation whose actual root cause was this
    # splitter, not the translation step.
    text = (
        "Mr. Adams, there was a lot of activity in December 2024 and a lot of it "
        "involved Mr. Brown Mogotsi assisting people with opening cases in Soweto. "
        "Did you by any chance speak to Mr. Brown Mogotsi around November, December 2024."
    )
    assert split_into_sentences(text, ENGLISH_TITLE_ABBREVIATIONS) == [
        "Mr. Adams, there was a lot of activity in December 2024 and a lot of it "
        "involved Mr. Brown Mogotsi assisting people with opening cases in Soweto.",
        "Did you by any chance speak to Mr. Brown Mogotsi around November, December 2024.",
    ]


def test_split_into_sentences_handles_multiple_different_titles():
    # Real production case: "Adv." (advocate) shredded a segment the same way.
    text = "Dr. Nkosi met Adv. Zulu at the office. They discussed the case."
    assert split_into_sentences(text, ENGLISH_TITLE_ABBREVIATIONS) == [
        "Dr. Nkosi met Adv. Zulu at the office.",
        "They discussed the case.",
    ]


def test_split_into_sentences_still_splits_real_sentence_boundaries():
    text = "Mr. Adams left. Mrs. Adams stayed."
    assert split_into_sentences(text, ENGLISH_TITLE_ABBREVIATIONS) == [
        "Mr. Adams left.",
        "Mrs. Adams stayed.",
    ]


def test_split_zulu_islands_does_not_split_after_isizulu_title_prefix_forms():
    # isiZulu prefixes titles with its own noun-class agreement -- confirmed real
    # forms from production data: "uMnu." (direct u- prefix) and "u-Adv."
    # (hyphenated, for a borrowed English abbreviation). Other grammatical
    # concord forms (e.g. "no-") are not yet covered -- this only claims the
    # forms actually confirmed in real job data.
    assert split_zulu_islands("Wathi uMnu. Zulu ufikile. Wahamba kamuva.") == [
        "Wathi uMnu. Zulu ufikile.",
        "Wahamba kamuva.",
    ]
    assert split_zulu_islands(
        "Futhi umholi wobufakazi u-Adv. Liesl Ngube wafika. Wahamba kamuva."
    ) == [
        "Futhi umholi wobufakazi u-Adv. Liesl Ngube wafika.",
        "Wahamba kamuva.",
    ]


def test_build_island_phrase_groups_happy_path_carries_parent_identity():
    pseudo_groups = build_island_phrase_groups(_GROUP, _WORDS_BY_ID, _SEGMENTS_BY_ID)
    assert pseudo_groups is not None
    assert len(pseudo_groups) == 3
    first = pseudo_groups[0]
    assert first["group_id"] == "native_phrase_0001__isl00"
    assert first["segment_ids"] == ["native_phrase_0001__isl00"]
    assert first["parts"] == [{"type": "text", "segment_id": "native_phrase_0001__isl00", "text": "Sawubona Ayanda."}]
    assert first["source_span_ms"] == 1300
    assert first["parent_group_id"] == "native_phrase_0001"
    assert first["parent_segment_ids"] == ["seg-1"]
    assert first["parent_source_span_ms"] == 4000
    assert [g["island_index"] for g in pseudo_groups] == [0, 1, 2]
    assert all(g["island_count"] == 3 for g in pseudo_groups)


def test_build_island_phrase_groups_falls_back_on_en_zu_sentence_count_mismatch():
    # Translation merged two English sentences into one Zulu sentence -- 3 EN vs 2 ZU.
    group = {**_GROUP, "parts": [{
        "type": "text", "segment_id": "seg-1",
        "text": "Sawubona Ayanda futhi kungani kwakuphuthuma? Wafaka amacala.",
    }]}
    assert build_island_phrase_groups(group, _WORDS_BY_ID, _SEGMENTS_BY_ID) is None


# Regression fixture for a real production case: a phrase group whose source ends
# "...to the attention of IDAC. IDAC" -- the trailing bare "IDAC" becomes its own
# one-word "sentence" with almost no real duration, which used to force the whole
# 8-sentence group to give up on splitting entirely.
_TRAILING_FRAGMENT_WORDS = [
    _word("w1", "Good.", 0.0, 0.4),
    _word("w2", "Why", 1.0, 1.2),
    _word("w3", "urgent?", 1.2, 1.6),
    _word("w4", "IDAC", 1.6, 1.62),  # ~20ms: far below MIN_ISLAND_DURATION_MS
]
_TRAILING_FRAGMENT_WORDS_BY_ID = {w["word_id"]: w for w in _TRAILING_FRAGMENT_WORDS}
_TRAILING_FRAGMENT_SEGMENTS_BY_ID = {
    "seg-1": _segment("seg-1", [w["word_id"] for w in _TRAILING_FRAGMENT_WORDS]),
}
_TRAILING_FRAGMENT_GROUP = {
    "group_id": "native_phrase_0002",
    "speaker_id": "SPEAKER_00",
    "segment_ids": ["seg-1"],
    "start_ms": 0,
    "source_end_ms": 1620,
    "source_span_ms": 1620,
    "source_text": "Good. Why urgent? IDAC",
    "parts": [{"type": "text", "segment_id": "seg-1", "text": "Kulungile. Kungani kuphuthumile? IDAC"}],
}


def test_build_island_phrase_groups_merges_a_too_short_trailing_island_into_its_neighbor():
    pseudo_groups = build_island_phrase_groups(
        _TRAILING_FRAGMENT_GROUP, _TRAILING_FRAGMENT_WORDS_BY_ID, _TRAILING_FRAGMENT_SEGMENTS_BY_ID,
    )
    assert pseudo_groups is not None
    # 3 raw sentences ("Good.", "Why urgent?", "IDAC") collapse to 2 islands: the
    # bare "IDAC" fragment is absorbed into the sentence right before it rather
    # than discarding the split for the whole group.
    assert len(pseudo_groups) == 2
    assert pseudo_groups[0]["source_text"] == "Good."
    assert pseudo_groups[1]["source_text"] == "Why urgent? IDAC"
    assert pseudo_groups[1]["parts"][0]["text"] == "Kungani kuphuthumile? IDAC"
    assert pseudo_groups[1]["source_span_ms"] >= MIN_ISLAND_DURATION_MS
    # Merged span still covers the group's real end exactly.
    assert pseudo_groups[1]["source_end_ms"] == _TRAILING_FRAGMENT_GROUP["source_end_ms"]


def test_build_island_phrase_groups_merges_a_too_short_leading_island_into_its_neighbor():
    # A leading fragment has no previous island to fall back on, so it must merge
    # into the NEXT one -- with a third sentence present, the split still
    # survives (2 islands out of 3 sentences), rather than collapsing entirely.
    words = [
        _word("w1", "Hi.", 0.0, 0.02),  # ~20ms leading fragment
        _word("w2", "Why", 1.0, 1.2),
        _word("w3", "urgent?", 1.2, 1.6),
        _word("w4", "He", 2.0, 2.1),
        _word("w5", "laid", 2.1, 2.3),
        _word("w6", "charges.", 2.3, 2.8),
    ]
    words_by_id = {w["word_id"]: w for w in words}
    segments_by_id = {"seg-1": _segment("seg-1", [w["word_id"] for w in words])}
    group = {
        **_TRAILING_FRAGMENT_GROUP,
        "source_text": "Hi. Why urgent? He laid charges.",
        "source_end_ms": 2800,
        "parts": [{
            "type": "text", "segment_id": "seg-1",
            "text": "Sawubona. Kungani kuphuthumile? Wafaka amacala.",
        }],
    }
    pseudo_groups = build_island_phrase_groups(group, words_by_id, segments_by_id)
    assert pseudo_groups is not None
    assert len(pseudo_groups) == 2
    assert pseudo_groups[0]["source_text"] == "Hi. Why urgent?"
    assert pseudo_groups[0]["parts"][0]["text"] == "Sawubona. Kungani kuphuthumile?"
    assert pseudo_groups[0]["source_span_ms"] >= MIN_ISLAND_DURATION_MS
    assert pseudo_groups[1]["source_text"] == "He laid charges."
    assert pseudo_groups[1]["parts"][0]["text"] == "Wafaka amacala."


def test_build_island_phrase_groups_falls_back_when_merging_collapses_everything():
    # Every sentence is individually too short even after merging -- nothing left
    # worth splitting, so the whole group correctly stays intact.
    words = [
        _word("w1", "Hi.", 0.0, 0.02),
        _word("w2", "Bye.", 0.02, 0.04),
    ]
    words_by_id = {w["word_id"]: w for w in words}
    segments_by_id = {"seg-1": _segment("seg-1", [w["word_id"] for w in words])}
    group = {
        **_TRAILING_FRAGMENT_GROUP,
        "source_text": "Hi. Bye.",
        "source_end_ms": 40,
        "parts": [{"type": "text", "segment_id": "seg-1", "text": "Sawubona. Sala kahle."}],
    }
    assert build_island_phrase_groups(group, words_by_id, segments_by_id) is None


def test_pairs_have_reliable_correspondence_accepts_normal_length_variance():
    pairs = [
        {"en_text": "Good morning to everyone here today", "zu_text": "Sawubona kubantu bonke abakhona lapha namuhla"},
        {"en_text": "Why was this matter never properly investigated", "zu_text": "Kungani lolu daba lwalungaphenywanga ngendlela efanele neze"},
        {"en_text": "He laid several charges against the accused officer", "zu_text": "Wafaka amacala amaningana ngokumelene nesikhulu esisolwayo"},
    ]
    assert _pairs_have_reliable_correspondence(pairs) is True


def test_pairs_have_reliable_correspondence_rejects_a_shuffled_pair():
    # Modeled on a real production case (job 33cd7b46..., native_phrase_0039): an
    # 18-sentence group where Zulu sentence 12 was a near-duplicate of sentence
    # 11's much longer content instead of a translation of its own short English
    # sentence -- 64 Zulu words standing in for a 12-word English sentence (8.5x
    # this group's own ~0.6 median ratio), which then got squeezed via ~11x atempo
    # into an unintelligible slot since "compress" direction never goes to repair.
    pairs = [
        {"en_text": "Good morning to everyone here today", "zu_text": "Sawubona kubantu bonke abakhona lapha namuhla"},
        {"en_text": "Why was this matter never properly investigated", "zu_text": "Kungani lolu daba lwalungaphenywanga ngendlela efanele neze"},
        {
            "en_text": "He laid several charges against the accused officer",
            "zu_text": "Wafaka amacala amaningana ngokumelene nesikhulu esisolwayo kanjalo",
        },
        {
            # Only 4 English words, but the Zulu is a near-copy of the previous
            # pair's much longer content instead of its own short translation.
            "en_text": "Sinister things are happening",
            "zu_text": "Wafaka amacala amaningana ngokumelene nesikhulu esisolwayo futhi kwaba nzima kakhulu ukuqonda konke okwenzekile ngempela lapha",
        },
    ]
    assert _pairs_have_reliable_correspondence(pairs) is False


def test_pairs_have_reliable_correspondence_ignores_short_islands_below_the_word_floor():
    # Short islands (below MIN_WORDS_FOR_LENGTH_RATIO_CHECK) are naturally noisy --
    # Zulu morphology can pack one English word into several, or vice versa -- so
    # they must not trip the correspondence guard on their own.
    pairs = [
        {"en_text": "Yes", "zu_text": "Yebo impela ngokuqinisekile"},
        {"en_text": "No", "zu_text": "Cha"},
        {"en_text": "He laid several charges against the accused officer today", "zu_text": "Wafaka amacala amaningana ngokumelene nesikhulu esisolwayo namuhla"},
    ]
    assert _pairs_have_reliable_correspondence(pairs) is True


_SHUFFLED_CORRESPONDENCE_WORDS = [
    _word("w1", "Good", 0.0, 0.2), _word("w2", "morning", 0.2, 0.4), _word("w3", "to", 0.4, 0.5),
    _word("w4", "everyone", 0.5, 0.7), _word("w5", "here", 0.7, 0.85), _word("w6", "today.", 0.85, 1.0),
    _word("w7", "Why", 1.4, 1.55), _word("w8", "was", 1.55, 1.7), _word("w9", "this", 1.7, 1.85),
    _word("w10", "matter", 1.85, 2.1), _word("w11", "never", 2.1, 2.3), _word("w12", "investigated?", 2.3, 2.7),
    _word("w13", "He", 3.1, 3.2), _word("w14", "laid", 3.2, 3.4), _word("w15", "several", 3.4, 3.65),
    _word("w16", "charges", 3.65, 3.9), _word("w17", "against", 3.9, 4.1), _word("w18", "the", 4.1, 4.2),
    _word("w19", "officer.", 4.2, 4.5),
    _word("w20", "Sinister", 4.9, 5.1), _word("w21", "things", 5.1, 5.3),
    _word("w22", "are", 5.3, 5.4), _word("w23", "happening.", 5.4, 5.7),
]
_SHUFFLED_CORRESPONDENCE_WORDS_BY_ID = {w["word_id"]: w for w in _SHUFFLED_CORRESPONDENCE_WORDS}
_SHUFFLED_CORRESPONDENCE_SEGMENTS_BY_ID = {
    "seg-1": _segment("seg-1", [w["word_id"] for w in _SHUFFLED_CORRESPONDENCE_WORDS]),
}
_SHUFFLED_CORRESPONDENCE_GROUP = {
    "group_id": "native_phrase_0039",
    "speaker_id": "SPEAKER_00",
    "segment_ids": ["seg-1"],
    "start_ms": 0,
    "source_end_ms": 5700,
    "source_span_ms": 5700,
    "source_text": "Good morning to everyone today. Why was this matter never investigated? He laid several charges against the officer. Sinister things are happening.",
    "parts": [{
        "type": "text", "segment_id": "seg-1",
        "text": (
            "Sawubona kubantu bonke abakhona namuhla. Kungani lolu daba lwalungaphenywanga? "
            "Wafaka amacala amaningana ngokumelene nesikhulu. "
            "Wafaka amacala amaningana ngokumelene nesikhulu futhi kwaba nzima kakhulu ukuqonda konke okwenzekile ngempela lapha."
        ),
    }],
}


def test_build_island_phrase_groups_falls_back_end_to_end_when_a_sentence_looks_shuffled():
    # End-to-end version of the two unit tests above: a real derive_english_islands
    # + split_zulu_islands pipeline run over a 4-sentence group where the split
    # sentence counts match (4 == 4, so the existing count check alone would pass
    # this through) but the last Zulu sentence is a near-duplicate of the third's
    # much longer content -- build_island_phrase_groups must reject the whole
    # split so the caller falls back to safe whole-group rendering.
    assert build_island_phrase_groups(
        _SHUFFLED_CORRESPONDENCE_GROUP,
        _SHUFFLED_CORRESPONDENCE_WORDS_BY_ID,
        _SHUFFLED_CORRESPONDENCE_SEGMENTS_BY_ID,
    ) is None


def _write_wav(path: Path, *, seconds: float, framerate: int = 16000) -> None:
    nframes = int(seconds * framerate)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


def test_stitch_islands_to_group_wav_concatenates_in_order(tmp_path):
    a, b, c = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
    _write_wav(a, seconds=1.0)
    _write_wav(b, seconds=0.5)
    _write_wav(c, seconds=2.0)
    output = tmp_path / "stitched.wav"

    result = stitch_islands_to_group_wav([a, b, c], output, inter_island_silence_ms=[100, 0])

    assert result["final_duration_ms"] == 1000 + 100 + 500 + 2000
    assert result["island_boundaries_ms"] == [1000, 1600, 3600]
    with wave.open(str(output), "rb") as handle:
        params = handle.getparams()
        assert params.nchannels == 1 and params.sampwidth == 2 and params.framerate == 16000
        assert params.nframes == int(round(result["final_duration_ms"] * 16000 / 1000.0))


def test_stitch_islands_to_group_wav_rejects_format_mismatch(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a, seconds=1.0, framerate=16000)
    _write_wav(b, seconds=1.0, framerate=8000)
    with pytest.raises(ValueError, match="format mismatch"):
        stitch_islands_to_group_wav([a, b], tmp_path / "out.wav", inter_island_silence_ms=[0])


def test_stitch_islands_to_group_wav_requires_at_least_one_path(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        stitch_islands_to_group_wav([], tmp_path / "out.wav")
