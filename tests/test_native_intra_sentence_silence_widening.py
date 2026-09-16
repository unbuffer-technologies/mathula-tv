"""Tests for native_dub._attempt_intra_sentence_silence_widening (Phase 27): a
render-time-only repair of last resort for a genuinely single-sentence group
whose natural Zulu underfills its window well beyond the small, unconditional
MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS pad -- splits the sentence at its own
single largest real ASR-detected internal pause (derive_intra_sentence_pause_
islands) and widens that real gap to close (or shrink) the shortfall.

Modeled on tests/test_native_timing_regions.py's own _FakeSplitTts fixture
pattern for a _synthesize_group-calling function (real WAV files, duration
derived deterministically from word count so tests don't need to guess
_reverse_engineer_zulu_chunk_split's exact split point -- only that it never
loses or duplicates a word, so the two pieces' word counts always sum to the
original text's own word count).
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.native_dub import (
    INTRA_SENTENCE_SILENCE_GAP_CAP_MS,
    _attempt_intra_sentence_silence_widening,
)


def _write_real_wav(path: Path, *, ms: int, framerate: int = 16000) -> None:
    nframes = int(ms * framerate / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


class _FakeSplitTts:
    default_voice = "zu-ZA-ThandoNeural"

    def __init__(self, ms_per_word: int = 250, fail_on: set[str] | None = None):
        self._ms_per_word = ms_per_word
        self._fail_on = fail_on or set()
        self.calls: list[str] = []

    def synthesize(self, request, output_path, *, force=False):
        self.calls.append(request.text)
        if request.turn_id in self._fail_on:
            raise RuntimeError("simulated synthesis failure")
        duration_ms = len(request.text.split()) * self._ms_per_word
        _write_real_wav(Path(output_path), ms=duration_ms)
        return SimpleNamespace(duration_ms=duration_ms, voice=request.voice)


def _word(word_id: str, text: str, start: float, end: float) -> dict:
    return {"word_id": word_id, "text": text, "start": start, "end": end}


def _segment(segment_id: str, word_ids: list[str]) -> dict:
    return {"segment_id": segment_id, "source_word_ids": word_ids}


# 10 English words with one real, qualifying 700ms pause before word index 5
# ("six") -- 5 words on each side, comfortably clearing derive_intra_sentence_
# pause_islands's own INTRA_SENTENCE_MIN_WORDS_PER_SIDE=3 floor.
_WORDS = [
    _word("w1", "Alpha", 0.0, 0.2),
    _word("w2", "bravo", 0.2, 0.4),
    _word("w3", "charlie", 0.4, 0.6),
    _word("w4", "delta", 0.6, 0.8),
    _word("w5", "echo", 0.8, 1.0),
    _word("w6", "six", 1.7, 1.9),
    _word("w7", "seven", 1.9, 2.1),
    _word("w8", "eight", 2.1, 2.3),
    _word("w9", "nine", 2.3, 2.5),
    _word("w10", "ten.", 2.5, 2.7),
]
_WORDS_BY_ID = {w["word_id"]: w for w in _WORDS}
_SEGMENTS_BY_ID = {"seg-1": _segment("seg-1", [w["word_id"] for w in _WORDS])}
_WORD_INDEX = (_WORDS_BY_ID, _SEGMENTS_BY_ID)

# 8 real (placeholder) Zulu words -- _reverse_engineer_zulu_chunk_split only ever
# cuts at whitespace and never loses/duplicates a word, so however it splits
# these, the two pieces' word counts always sum to exactly 8.
_ZULU_TEXT = "wordone wordtwo wordthree wordfour wordfive wordsix wordseven wordeight"
_GROUP = {
    "group_id": "native_phrase_0099",
    "speaker_id": "S0",
    "segment_ids": ["seg-1"],
    "source_text": "Alpha bravo charlie delta echo six seven eight nine ten.",
    "parts": [{"type": "text", "text": _ZULU_TEXT}],
}
_VOICE_INFO = {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}}
_TOTAL_ZULU_WORDS = len(_ZULU_TEXT.split())


def test_widening_closes_the_shortfall_when_the_real_gap_is_generous(tmp_path):
    ms_per_word = 250
    total_measured_ms = _TOTAL_ZULU_WORDS * ms_per_word  # deterministic regardless of split point
    source_window_ms = total_measured_ms + 800  # needed_gap_ms=800, well under the cap
    tts = _FakeSplitTts(ms_per_word=ms_per_word)

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=_GROUP, source_window_ms=source_window_ms, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is not None
    assert result["method"] == "intra_sentence_silence_widening"
    assert len(tts.calls) == 2  # exactly one real synthesis call per island
    assert result["final_duration_ms"] == source_window_ms
    assert result["end_error_ms"] == 0
    assert result["widened_gap_ms"] == 800
    assert Path(result["output_path"]).is_file()
    # Real production crash caught here: a real render wrote this whole dict
    # into a JSON artifact and failed with "Object of type PosixPath is not
    # JSON serializable" -- every path-shaped field, including each nested
    # island's own "path", must be a plain string, never a raw Path object.
    json.dumps(result)


def test_widening_caps_at_the_gap_cap_and_leaves_an_honest_residual(tmp_path):
    ms_per_word = 250
    total_measured_ms = _TOTAL_ZULU_WORDS * ms_per_word
    needed_gap_ms = INTRA_SENTENCE_SILENCE_GAP_CAP_MS + 1500  # far exceeds the cap
    source_window_ms = total_measured_ms + needed_gap_ms
    tts = _FakeSplitTts(ms_per_word=ms_per_word)

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=_GROUP, source_window_ms=source_window_ms, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is not None
    assert result["widened_gap_ms"] == INTRA_SENTENCE_SILENCE_GAP_CAP_MS
    assert result["final_duration_ms"] == total_measured_ms + INTRA_SENTENCE_SILENCE_GAP_CAP_MS
    # The residual is honestly reflected, never fabricated away.
    assert result["final_duration_ms"] < source_window_ms
    assert result["end_error_ms"] == result["final_duration_ms"] - source_window_ms


def test_widening_is_skipped_when_the_split_does_not_actually_help(tmp_path):
    ms_per_word = 250
    total_measured_ms = _TOTAL_ZULU_WORDS * ms_per_word
    source_window_ms = total_measured_ms - 200  # already meets/exceeds the window on its own
    tts = _FakeSplitTts(ms_per_word=ms_per_word)

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=_GROUP, source_window_ms=source_window_ms, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is None


def test_widening_falls_back_cleanly_on_a_degenerate_zulu_split(tmp_path):
    # A single-word Zulu text can never be split into two non-empty pieces.
    group = {**_GROUP, "parts": [{"type": "text", "text": "wordone"}]}
    tts = _FakeSplitTts()

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=group, source_window_ms=10_000, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is None
    assert tts.calls == []  # never even attempted a synthesis call


def test_widening_falls_back_cleanly_on_an_island_synthesis_failure(tmp_path):
    # The first island's own synthetic id -- confirmed to match _synthesize_
    # group's own turn_id=str(group["group_id"]) convention.
    tts = _FakeSplitTts(fail_on={"native_phrase_0099__pause00"})

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=_GROUP, source_window_ms=10_000, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is None  # never raises -- the caller falls through to today's exact warning


def test_widening_never_attempted_when_word_index_is_none(tmp_path):
    tts = _FakeSplitTts()

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=_GROUP, source_window_ms=10_000, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=None, force=True,
        job_root=None, progress=None,
    )

    assert result is None
    assert tts.calls == []


def test_widening_never_attempted_for_a_multi_sentence_group(tmp_path):
    group = {**_GROUP, "source_text": "Alpha bravo. Charlie delta echo six seven eight nine ten."}
    tts = _FakeSplitTts()

    result = _attempt_intra_sentence_silence_widening(
        chosen_group=group, source_window_ms=10_000, voice_info=_VOICE_INFO,
        tts=tts, audio_dir=tmp_path, word_index=_WORD_INDEX, force=True,
        job_root=None, progress=None,
    )

    assert result is None
    assert tts.calls == []
