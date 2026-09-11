"""Sentences, not multi-sentence blocks, are the atomic native-dub translation/timing
unit. This reuses derive_english_islands (speech_islands.py) -- previously a post-Pass-2
retrofit -- to split each Pass-1 restored segment into sentence-level records with real
timing BEFORE Pass 2 translates anything, so a block-level rush average can no longer
hide a badly-rushed individual sentence behind comfortably-paced neighbors (confirmed
real case: native_phrase_0001 averaged "+13.1% rush" while two of its six sentences were
individually at +43.9%/+27.0% and the other four were at 0%).
"""

from pathlib import Path

from mathula_tv.native_dub import (
    _build_pass1_sentence_records,
    _build_temporal_mask_source_groups,
    _merge_fragmented_pass1_segments,
    _split_pass1_segment_into_sentences,
)
from mathula_tv.atomic_io import atomic_write_json


def _word(word_id: str, text: str, start: float, end: float) -> dict:
    return {"word_id": word_id, "text": text, "start": start, "end": end}


def test_single_sentence_segment_is_not_split():
    segment = {
        "segment_id": "seg-1",
        "speaker_id": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 2000,
        "restored_text": "This is one sentence.",
    }
    records = _split_pass1_segment_into_sentences(segment, None)
    assert len(records) == 1
    assert records[0]["segment_id"] == "seg-1__s00"
    assert records[0]["parent_segment_id"] == "seg-1"
    assert records[0]["start_ms"] == 0
    assert records[0]["end_ms"] == 2000
    assert records[0]["restored_text"] == "This is one sentence."


def test_multi_sentence_segment_splits_with_real_word_alignment():
    words = [
        _word("w1", "My", 0.0, 0.2),
        _word("w2", "colleague", 0.2, 0.6),
        _word("w3", "Nyati.", 0.6, 1.0),
        _word("w4", "Why", 2.0, 2.2),
        _word("w5", "was", 2.2, 2.4),
        _word("w6", "it", 2.4, 2.5),
        _word("w7", "urgent?", 2.5, 3.0),
    ]
    words_by_id = {w["word_id"]: w for w in words}
    segments_by_id = {"seg-1": {"segment_id": "seg-1", "source_word_ids": [w["word_id"] for w in words]}}
    segment = {
        "segment_id": "seg-1",
        "speaker_id": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 3000,
        "restored_text": "My colleague Nyati. Why was it urgent?",
    }
    records = _split_pass1_segment_into_sentences(segment, (words_by_id, segments_by_id))
    assert [r["segment_id"] for r in records] == ["seg-1__s00", "seg-1__s01"]
    assert [r["restored_text"] for r in records] == ["My colleague Nyati.", "Why was it urgent?"]
    assert [r["parent_segment_id"] for r in records] == ["seg-1", "seg-1"]
    # Real word-anchored timestamps, not a proportional character-length guess.
    assert records[0]["start_ms"] == 0
    assert records[0]["end_ms"] == 1000
    assert records[1]["start_ms"] == 2000
    assert records[1]["end_ms"] == 3000
    assert records[0]["alignment_method"] == "word_exact"
    assert records[1]["alignment_method"] == "word_exact"


def test_multi_sentence_segment_without_raw_word_index_still_splits_proportionally():
    # No raw ASR transcript for this job (word_index=None) -- must still split by
    # sentence using derive_english_islands' own proportional-by-character-length
    # fallback, not silently keep the whole multi-sentence segment as one unit.
    segment = {
        "segment_id": "seg-1",
        "speaker_id": "SPEAKER_00",
        "start_ms": 1000,
        "end_ms": 3000,
        "restored_text": "Short one. This second sentence is much longer than the first.",
    }
    records = _split_pass1_segment_into_sentences(segment, None)
    assert len(records) == 2
    assert records[0]["restored_text"] == "Short one."
    assert records[1]["restored_text"] == "This second sentence is much longer than the first."
    assert records[0]["alignment_method"] == "proportional_fallback"
    assert records[1]["alignment_method"] == "proportional_fallback"
    # Chronological and contiguous, spanning the segment's own real window exactly.
    assert records[0]["start_ms"] == 1000
    assert records[0]["end_ms"] == records[1]["start_ms"]
    assert records[1]["end_ms"] == 2000 + 1000  # segment's own end_ms


def test_build_pass1_sentence_records_reads_raw_transcript_from_job_root(tmp_path: Path):
    words = [_word("w1", "Hi.", 0.0, 0.5), _word("w2", "Bye.", 1.0, 1.5)]
    atomic_write_json(
        tmp_path / "analysis" / "transcript_en_raw.json",
        {
            "words": words,
            "segments": [{"segment_id": "seg-1", "source_word_ids": ["w1", "w2"]}],
        },
    )
    pass1_segments = [{
        "segment_id": "seg-1",
        "speaker_id": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 1500,
        "restored_text": "Hi. Bye.",
    }]
    records = _build_pass1_sentence_records(pass1_segments, job_root=tmp_path)
    assert [r["restored_text"] for r in records] == ["Hi.", "Bye."]
    assert records[0]["alignment_method"] == "word_exact"


def test_build_temporal_mask_source_groups_produces_one_group_per_sentence_record():
    records = [
        {
            "segment_id": "seg-1__s00", "speaker_id": "SPEAKER_00",
            "start_ms": 0, "end_ms": 1000, "restored_text": "First sentence.",
        },
        {
            "segment_id": "seg-1__s01", "speaker_id": "SPEAKER_00",
            "start_ms": 1000, "end_ms": 2500, "restored_text": "Second sentence.",
        },
    ]
    groups = _build_temporal_mask_source_groups(records)
    assert [g["group_id"] for g in groups] == ["native_phrase_0001", "native_phrase_0002"]
    assert groups[0]["segment_ids"] == ["seg-1__s00"]
    assert groups[0]["source_text"] == "First sentence."
    assert groups[0]["source_span_ms"] == 1000
    assert groups[1]["source_span_ms"] == 1500


# --- re-joining ASR/diarization fragments split mid-utterance -----------------------
# Confirmed real case: "More" and "specifically, we know that when Fadiel Adams opened
# these cases..." are two adjacent segments, same speaker, 0ms gap -- genuinely one
# sentence ("More specifically, ...") that the ASR/diarization stage happened to cut at
# a micro-pause. "More" carries no terminal punctuation, the tell that it isn't a
# complete sentence on its own. Left unmerged, "More" becomes an independent
# 200ms-window sentence with no possible natural-paced Zulu rendering (+286% rush in
# production). Sentence splitting only works within one segment's own text, so this has
# to be fixed before splitting, not after.

def test_fragment_with_no_terminal_punctuation_merges_into_the_next_segment():
    segments = [
        {
            "segment_id": "seg-00005", "speaker_id": "SPEAKER_01",
            "start_ms": 112190, "end_ms": 112390, "restored_text": "More",
        },
        {
            "segment_id": "seg-00006", "speaker_id": "SPEAKER_01",
            "start_ms": 112390, "end_ms": 128670,
            "restored_text": "specifically, we know that when Fadiel Adams opened these cases.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 1
    assert merged[0]["segment_id"] == "seg-00005"
    assert merged[0]["segment_ids"] == ["seg-00005", "seg-00006"]
    assert merged[0]["restored_text"] == (
        "More specifically, we know that when Fadiel Adams opened these cases."
    )
    assert merged[0]["start_ms"] == 112190
    assert merged[0]["end_ms"] == 128670


def test_title_abbreviation_period_is_not_mistaken_for_a_sentence_end():
    # Confirmed real case: "You were at Mr." (ends 233630ms) and "Lincoln's house."
    # (starts 233630ms, 0ms gap, same speaker) are one sentence ("You were at Mr.
    # Lincoln's house.") split exactly at the abbreviation's own period -- naive
    # terminal-punctuation matching sees the trailing "." and wrongly concludes
    # "You were at Mr." is already a complete sentence.
    segments = [
        {
            "segment_id": "seg-00023", "speaker_id": "SPEAKER_02",
            "start_ms": 232830, "end_ms": 233630, "restored_text": "You were at Mr.",
        },
        {
            "segment_id": "seg-00024", "speaker_id": "SPEAKER_02",
            "start_ms": 233630, "end_ms": 234190, "restored_text": "Lincoln's house.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 1
    assert merged[0]["restored_text"] == "You were at Mr. Lincoln's house."


def test_two_genuinely_separate_sentences_are_not_merged():
    # Each segment already ends in terminal punctuation -- both are complete
    # sentences on their own, so merging them would reintroduce the multi-sentence
    # block problem this whole redesign exists to fix.
    segments = [
        {
            "segment_id": "seg-1", "speaker_id": "SPEAKER_00",
            "start_ms": 0, "end_ms": 1000, "restored_text": "First sentence.",
        },
        {
            "segment_id": "seg-2", "speaker_id": "SPEAKER_00",
            "start_ms": 1000, "end_ms": 2000, "restored_text": "Second sentence.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 2
    assert [item["segment_id"] for item in merged] == ["seg-1", "seg-2"]


def test_fragment_from_a_different_speaker_is_not_merged():
    segments = [
        {
            "segment_id": "seg-1", "speaker_id": "SPEAKER_00",
            "start_ms": 0, "end_ms": 1000, "restored_text": "More",
        },
        {
            "segment_id": "seg-2", "speaker_id": "SPEAKER_01",
            "start_ms": 1000, "end_ms": 2000, "restored_text": "specifically about that.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 2


def test_small_positive_gap_still_merges_as_a_breath_pause():
    # Confirmed real case: "And" (ends 330190ms) and "there was a bit of an
    # interaction with one of the journalists a short while ago." (starts 330350ms)
    # -- a 160ms gap, an ordinary breath pause mid-sentence, not a sentence boundary.
    # A gap-<=0-only threshold misses this (real speech has small positive gaps
    # between words even within one continuous utterance); left unmerged, the lone
    # word "And" against its own ~120ms window measured +287.5% rush in production
    # -- structurally impossible regardless of how many candidates exist for it.
    segments = [
        {
            "segment_id": "seg-00048", "speaker_id": "SPEAKER_01",
            "start_ms": 330070, "end_ms": 330190, "restored_text": "And",
        },
        {
            "segment_id": "seg-00049", "speaker_id": "SPEAKER_01",
            "start_ms": 330350, "end_ms": 332830,
            "restored_text": "there was a bit of an interaction with one of the journalists a short while ago.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 1
    assert merged[0]["restored_text"] == (
        "And there was a bit of an interaction with one of the journalists a short while ago."
    )


def test_fragment_with_a_real_pause_before_the_next_segment_is_not_merged():
    # A genuine gap (a real pause, not a diarization cut mid-phrase) must not trigger
    # the merge even if the earlier segment lacks terminal punctuation -- the ceiling
    # (900ms) matches this codebase's own established "same articulatory phrase"
    # threshold (build_phrase_groups' max_join_gap_ms default), not an arbitrarily
    # small one.
    segments = [
        {
            "segment_id": "seg-1", "speaker_id": "SPEAKER_00",
            "start_ms": 0, "end_ms": 1000, "restored_text": "More",
        },
        {
            "segment_id": "seg-2", "speaker_id": "SPEAKER_00",
            "start_ms": 2500, "end_ms": 3500, "restored_text": "specifically about that.",
        },
    ]
    merged = _merge_fragmented_pass1_segments(segments)
    assert len(merged) == 2


def test_build_pass1_sentence_records_merges_fragments_before_splitting(tmp_path: Path):
    # End-to-end: no raw ASR word index for this job (word_index=None), so timing
    # falls back to proportional -- confirms the merge happens regardless of whether
    # real word alignment is available.
    pass1_segments = [
        {
            "segment_id": "seg-00005", "speaker_id": "SPEAKER_01",
            "start_ms": 112190, "end_ms": 112390, "restored_text": "More",
        },
        {
            "segment_id": "seg-00006", "speaker_id": "SPEAKER_01",
            "start_ms": 112390, "end_ms": 128670,
            "restored_text": "specifically, we know that when Fadiel Adams opened these cases.",
        },
    ]
    records = _build_pass1_sentence_records(pass1_segments, job_root=tmp_path)
    assert len(records) == 1
    assert records[0]["restored_text"] == (
        "More specifically, we know that when Fadiel Adams opened these cases."
    )
    assert records[0]["start_ms"] == 112190
    assert records[0]["end_ms"] == 128670
