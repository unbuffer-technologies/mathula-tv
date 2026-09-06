"""Sentence-level "speech island" derivation and stitching for native-dub timing.

Native-dub's syllable/timing budget is computed per phrase GROUP (one or more
diarization segments spoken back-to-back by one speaker). A group can span several
sentences, and real per-sentence analysis of production jobs showed the group-level
aggregate can mask a distribution problem: one sentence in a group can run 40-60%
over its own proportional time budget while another runs comfortably under -- and a
single whole-clip speed-up cannot fix that, because it can only rush (or not) the
clip as a whole.

This module derives sentence-level "islands" with REAL timing (from word-level ASR
timestamps, not an estimate), so native-dub's existing measure-and-repair controller
can operate at that finer granularity and rush only the islands that actually need
it. Island derivation is deliberately deterministic (no AI call): sentence boundaries
come from punctuation plus a small title-abbreviation exception list (see
split_into_sentences), not from any model judgment, and the real timing signal
comes from data the pipeline already has.
"""

from __future__ import annotations

import difflib
import io
import re
import statistics
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from .azure_tts import canonicalize_pcm_wav

# Mirrors the sentence-split convention already used for TTS chunking elsewhere in
# this codebase (see synthesis.py's plan_synthesis_units) -- not a new heuristic.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# A period after a title is never a real sentence boundary -- a title is always
# followed by a name, never left dangling at the end of a thought. Confirmed in
# real production data: naive period-splitting shredded one clean ASR segment,
# "Mr. Adams, there was a lot of activity ... and a lot of it involved Mr. Brown
# Mogotsi assisting people with opening cases in Soweto. Did you by any chance
# speak to Mr. Brown Mogotsi ...", into 5 garbage fragments (one of them a bare
# "Mr."). The translation model then padded those fragments with invented content
# to make each feel grammatically complete -- a real "do not invent information"
# violation whose actual root cause was never the translation step at all. 9 of
# these false split points were confirmed across one real 76-segment transcript.
# isiZulu prefixes many of these with its own noun-class agreement (u-/i-), so it
# gets its own list rather than reusing the English one.
ENGLISH_TITLE_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "adv", "prof", "sgt", "capt", "gen", "lt", "rev",
    "hon", "sen", "rep", "gov", "maj", "col", "cmdr", "supt", "insp", "const",
    "brig", "adm", "jr", "sr",
})
ZULU_TITLE_ABBREVIATIONS = frozenset({
    "mnu", "nkz", "nks", "dkt", "adv", "solwazi", "jen",
})


def _ends_with_title_abbreviation(fragment: str, abbreviations: frozenset[str]) -> bool:
    words = fragment.strip().split()
    if not words:
        return False
    last_word = words[-1].strip(".").casefold()
    candidates = {last_word}
    if last_word.startswith("u-") or last_word.startswith("i-"):
        candidates.add(last_word[2:])
    elif last_word.startswith("u"):
        candidates.add(last_word[1:])
    return bool(candidates & abbreviations)


def split_into_sentences(text: str, abbreviations: frozenset[str]) -> list[str]:
    """Split on sentence-ending punctuation, refusing to split after a title
    abbreviation (see the module-level comment above for the real production case
    this fixes). A naive punctuation split runs first; any piece that turns out to
    end in a title gets glued back onto the fragment that follows it.
    """
    raw_parts = [part for part in SENTENCE_SPLIT_RE.split(text) if part]
    merged: list[str] = []
    for part in raw_parts:
        if merged and _ends_with_title_abbreviation(merged[-1], abbreviations):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [s.strip() for s in merged if s.strip()]

# A degenerate island (near-zero duration) cannot carry a meaningful timing budget
# regardless of content. Below this floor, treat the whole group as unsplittable
# rather than let a downstream budget/repair calculation operate on noise.
MIN_ISLAND_DURATION_MS = 300

# A short island's word-count ratio is naturally noisy (Zulu morphology can pack a
# preposition into one word, or split one English word across two), so the
# correspondence check below only trusts pairs with at least this many English words.
MIN_WORDS_FOR_LENGTH_RATIO_CHECK = 4

# How many multiples of a group's OWN median Zulu/English word-count ratio a single
# pair may deviate before it's treated as a correspondence failure rather than normal
# translation variance. Calibrated against a real production job spanning 57 groups/
# 134 islands: the worst genuine bug (see _pairs_have_reliable_correspondence's
# docstring) sat at 8.5x deviation from its own group's median; the worst pair
# anywhere else in that job (a legitimate short-island outlier) sat at 2.3x. 3.0x
# sits cleanly in the gap between those two.
MAX_ISLAND_LENGTH_RATIO_DEVIATION = 3.0


def load_raw_word_index(
    raw_transcript: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Build {word_id: word} and {segment_id: segment} lookups from a raw ASR transcript.

    ``raw_transcript`` is the parsed contents of a job's
    ``analysis/transcript_en_raw.json`` -- the only place word-level timestamps
    survive; Pass 1's normalized segments strip them.
    """
    words_by_id = {
        str(word["word_id"]): word
        for word in raw_transcript.get("words") or []
        if word.get("word_id") is not None
    }
    segments_by_id = {
        str(segment["segment_id"]): segment
        for segment in raw_transcript.get("segments") or []
        if segment.get("segment_id") is not None
    }
    return words_by_id, segments_by_id


def _word_ms(word: Mapping[str, Any], key: str) -> int:
    return int(round(float(word[key]) * 1000.0))


def _group_spoken_zulu_text(group: Mapping[str, Any]) -> str:
    return " ".join(
        str(part.get("text") or "").strip()
        for part in group.get("parts", [])
        if part.get("type") == "text" and str(part.get("text") or "").strip()
    )


def _normalize_word_for_alignment(text: str) -> str:
    return re.sub(r"[^\w]", "", str(text)).lower()


def _align_words_to_timestamps(
    restored_words: Sequence[str],
    raw_words: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, bool]] | None:
    """Return one (start_ms, end_ms, exact) per restored word.

    Real ASR word timestamps are used wherever a restored word matches a raw
    word (``exact=True``); the interior of a local mismatch (Pass 1's STT
    correction changed wording somewhere -- "British Rabanda Justice College"
    -> "the Bar and a Justice College", "24" <-> "twenty four") is interpolated
    between the nearest real anchors on either side (``exact=False``), rather
    than falling back to proportional estimation across the WHOLE group. A
    correction confined to one sentence therefore only costs precision on that
    one sentence -- every other sentence in the same group still gets exact
    real timestamps, instead of the previous all-or-nothing behavior where one
    mismatch anywhere in an 8-sentence group degraded every sentence's timing
    to a character-length guess (confirmed in production: this was inflating
    "required rush" warnings on sentences that would have aligned perfectly on
    their own). The ``exact`` flag is returned per word -- not inferred after
    the fact by checking whether a word's text merely APPEARS somewhere in the
    raw sequence, which would false-positive on any common word repeated
    elsewhere in the transcript -- so callers can tell a genuinely word-exact
    sentence from one that only partially aligned.
    """
    if not raw_words or not restored_words:
        return None
    raw_norm = [_normalize_word_for_alignment(word["text"]) for word in raw_words]
    restored_norm = [_normalize_word_for_alignment(word) for word in restored_words]
    matcher = difflib.SequenceMatcher(None, raw_norm, restored_norm, autojunk=False)

    timestamps: list[tuple[int, int] | None] = [None] * len(restored_words)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            word = raw_words[block.a + offset]
            timestamps[block.b + offset] = (_word_ms(word, "start"), _word_ms(word, "end"))
    exact_flags = [value is not None for value in timestamps]

    fallback_start_ms = _word_ms(raw_words[0], "start")
    fallback_end_ms = _word_ms(raw_words[-1], "end")
    index = 0
    total = len(timestamps)
    while index < total:
        if timestamps[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < total and timestamps[index] is None:
            index += 1
        gap_end = index
        before = timestamps[gap_start - 1] if gap_start > 0 else None
        after = timestamps[gap_end] if gap_end < total else None
        span_start_ms = before[1] if before else (after[0] if after else fallback_start_ms)
        span_end_ms = after[0] if after else (before[1] if before else fallback_end_ms)
        span_end_ms = max(span_end_ms, span_start_ms + (gap_end - gap_start))
        count = gap_end - gap_start
        for offset in range(count):
            piece_start = span_start_ms + int(round((span_end_ms - span_start_ms) * offset / count))
            piece_end = span_start_ms + int(round((span_end_ms - span_start_ms) * (offset + 1) / count))
            timestamps[gap_start + offset] = (piece_start, max(piece_end, piece_start + 1))

    # Every index was either a matched anchor or filled by the interpolation pass
    # above -- there is no remaining gap for the caller to (mis)handle by index.
    return [
        (t[0], t[1], exact) if t is not None else (fallback_start_ms, fallback_end_ms, False)
        for t, exact in zip(timestamps, exact_flags)
    ]


def _proportional_split_islands(
    sentences: Sequence[str], *, group_id: str, start_ms: int, end_ms: int,
) -> list[dict[str, Any]] | None:
    """Allocate each sentence a slice of the group's window proportional to its length.

    Used whenever exact word-level alignment isn't possible (Pass 1's STT
    restoration frequently changes word count somewhere in a block -- "British
    Rabanda" -> "the Bar and a", "24" <-> "twenty four" -- which breaks a strict
    word-count match even though only a small part of the sentence changed). This
    mirrors the same technique already used for TTS chunking in
    synthesis.py's plan_synthesis_units: real total duration, apportioned by
    character length. Timing is approximate rather than word-exact, but still far
    finer-grained than treating the whole multi-sentence group as one unit.
    """
    total_chars = sum(len(sentence) for sentence in sentences)
    if total_chars <= 0:
        return None
    window_ms = end_ms - start_ms
    if window_ms <= 0:
        return None
    islands: list[dict[str, Any]] = []
    cursor_ms = start_ms
    for index, sentence in enumerate(sentences):
        is_last = index == len(sentences) - 1
        if is_last:
            island_end_ms = end_ms
        else:
            share = len(sentence) / total_chars
            island_end_ms = min(end_ms, cursor_ms + max(1, int(round(window_ms * share))))
        islands.append({
            "island_id": f"{group_id}__isl{index:02d}",
            "index": index,
            "source_text": sentence,
            "start_ms": cursor_ms,
            "end_ms": island_end_ms,
            "alignment_method": "proportional_fallback",
        })
        cursor_ms = island_end_ms
    return islands


def derive_english_islands(
    group: Mapping[str, Any],
    words_by_id: Mapping[str, Mapping[str, Any]],
    segments_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Split one phrase group's English source into sentence-level islands with real timing.

    Aligns the restored text's words against the raw ASR word sequence (see
    ``_align_words_to_timestamps``): a sentence composed entirely of words that
    match the raw transcript gets exact real timestamps, and only the specific
    word(s) around an actual STT correction fall back to local interpolation --
    not the whole group. A group-wide proportional character-length split is
    used only as a last resort, when there is no raw word data to anchor to at
    all (e.g. a missing segment lookup). Returns None only when there is
    nothing to split (a single-sentence group) or the group's own window is
    degenerate. A too-short individual island (e.g. a bare trailing acronym
    repeated as its own "sentence") is NOT rejected here --
    ``build_island_phrase_groups`` merges it into a neighbor instead, since
    that decision must stay in lockstep with the paired Zulu sentence on the
    other side of the same split point.
    """
    source_text = str(group.get("source_text") or "").strip()
    if not source_text:
        return None
    sentences = split_into_sentences(source_text, ENGLISH_TITLE_ABBREVIATIONS)
    if len(sentences) <= 1:
        return None

    segment_ids = [str(value) for value in group.get("segment_ids") or []]
    group_id = str(group.get("group_id") or "")
    word_sequence: list[Mapping[str, Any]] | None = []
    for segment_id in segment_ids:
        segment = segments_by_id.get(segment_id)
        if segment is None:
            word_sequence = None
            break
        for word_id in segment.get("source_word_ids") or []:
            word = words_by_id.get(str(word_id))
            if word is None:
                word_sequence = None
                break
            word_sequence.append(word)
        if word_sequence is None:
            break

    restored_words = source_text.split()
    aligned = _align_words_to_timestamps(restored_words, word_sequence) if word_sequence else None
    if aligned is not None:
        islands: list[dict[str, Any]] = []
        cursor = 0
        for index, sentence in enumerate(sentences):
            count = len(sentence.split())
            span = aligned[cursor:cursor + count]
            cursor += count
            start_ms = span[0][0]
            end_ms = span[-1][1]
            all_exact = all(exact for _start, _end, exact in span)
            islands.append({
                "island_id": f"{group_id}__isl{index:02d}",
                "index": index,
                "source_text": sentence,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "alignment_method": "word_exact" if all_exact else "word_aligned_partial",
            })
        return islands

    start_ms = int(group.get("start_ms") or 0)
    end_ms = int(group.get("source_end_ms") or 0)
    return _proportional_split_islands(sentences, group_id=group_id, start_ms=start_ms, end_ms=end_ms)


def split_zulu_islands(spoken_text: str) -> list[str]:
    return split_into_sentences(str(spoken_text or ""), ZULU_TITLE_ABBREVIATIONS)


def _merge_short_islands(
    pairs: Sequence[Mapping[str, Any]], *, min_duration_ms: int,
) -> list[dict[str, Any]] | None:
    """Merge any English/Zulu island pair shorter than ``min_duration_ms`` into a neighbor.

    A trailing fragment (a bare repeated acronym like "IDAC. IDAC") or a leading
    one-word fragment would otherwise force the whole group to give up on
    splitting entirely. Merging into the previous pair when one exists (a
    trailing fragment reads naturally as a continuation of what came before it),
    or the next pair otherwise, salvages the rest of the split instead of
    discarding it. English and Zulu text are combined together in the same merge
    so the two sides never drift out of positional alignment. Returns None only
    when merging collapses everything down to a single pair (nothing left worth
    splitting) or a pair still can't clear the floor afterward.
    """
    merged: list[dict[str, Any]] = [dict(pair) for pair in pairs]
    while len(merged) > 1:
        short_index = next(
            (i for i, pair in enumerate(merged) if pair["end_ms"] - pair["start_ms"] < min_duration_ms),
            None,
        )
        if short_index is None:
            break
        has_previous = short_index > 0
        target_index = short_index - 1 if has_previous else short_index + 1
        short_pair = merged.pop(short_index)
        if target_index > short_index:
            target_index -= 1
        target = merged[target_index]
        short_comes_first = short_pair["start_ms"] < target["start_ms"]
        merged[target_index] = {
            "start_ms": min(target["start_ms"], short_pair["start_ms"]),
            "end_ms": max(target["end_ms"], short_pair["end_ms"]),
            "en_text": (
                f"{short_pair['en_text']} {target['en_text']}" if short_comes_first
                else f"{target['en_text']} {short_pair['en_text']}"
            ),
            "zu_text": (
                f"{short_pair['zu_text']} {target['zu_text']}" if short_comes_first
                else f"{target['zu_text']} {short_pair['zu_text']}"
            ),
            "alignment_method": "merged",
        }
    if len(merged) <= 1:
        return None
    if any(pair["end_ms"] - pair["start_ms"] < min_duration_ms for pair in merged):
        return None
    return merged


def _pairs_have_reliable_correspondence(pairs: Sequence[Mapping[str, Any]]) -> bool:
    """Reject a split whose per-sentence Zulu/English length ratios look shuffled.

    Sentence-count parity (already checked before this runs) is necessary but not
    sufficient: Pass 2's translation is only asked to render a group's MEANING
    fluently, never to preserve English sentence-order correspondence, so a
    same-count Zulu paragraph can still merge or reorder content across sentence
    boundaries. Real production case: an 18-sentence group where Zulu sentence 12
    was a near-duplicate of sentence 11's much longer content (64 Zulu words
    standing in for a 12-word English sentence -- 8.5x this group's own ~0.6 median
    ratio) got squeezed via ~11x atempo into an unintelligible 3.6s slot, with the
    same shift cascading through several neighboring sentences too. One pair
    failing this check is enough to distrust the WHOLE split for this group, so the
    caller falls back to the existing, already-safe whole-group rendering, which
    never assumed per-sentence correspondence in the first place.
    """
    ratios = [
        len(str(pair["zu_text"]).split()) / len(str(pair["en_text"]).split())
        for pair in pairs
        if len(str(pair["en_text"]).split()) >= MIN_WORDS_FOR_LENGTH_RATIO_CHECK
    ]
    if len(ratios) < 2:
        return True
    median_ratio = statistics.median(ratios)
    if median_ratio <= 0:
        return True
    return all(
        max(ratio / median_ratio, median_ratio / ratio) <= MAX_ISLAND_LENGTH_RATIO_DEVIATION
        for ratio in ratios
        if ratio > 0
    )


def build_island_phrase_groups(
    group: Mapping[str, Any],
    words_by_id: Mapping[str, Mapping[str, Any]],
    segments_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Build one synthetic phrase-group dict per sentence-level island of ``group``.

    Each returned dict has the same shape ``build_phrase_groups``/temporal-mask
    reconstruction already produce (group_id, speaker_id, segment_ids, start_ms,
    source_end_ms, source_span_ms, source_text, source_segments, parts), so it can
    be handed directly to the existing measure-and-repair controller unmodified.
    Extra ``parent_*`` fields identify the original group for re-stitching after
    measurement. Returns None when islands cannot be safely derived, the English
    and Zulu sentence counts disagree, the per-sentence correspondence looks
    unreliable (see ``_pairs_have_reliable_correspondence``), or merging short
    islands collapses the split entirely -- the caller must fall back to the whole
    group in that case.
    """
    english_islands = derive_english_islands(group, words_by_id, segments_by_id)
    if english_islands is None:
        return None
    zulu_sentences = split_zulu_islands(_group_spoken_zulu_text(group))
    if len(zulu_sentences) != len(english_islands):
        return None

    pairs = [
        {
            "start_ms": int(english["start_ms"]),
            "end_ms": int(english["end_ms"]),
            "en_text": str(english["source_text"]),
            "zu_text": zulu_text,
            "alignment_method": english.get("alignment_method"),
        }
        for english, zulu_text in zip(english_islands, zulu_sentences)
    ]
    if not _pairs_have_reliable_correspondence(pairs):
        return None
    merged_pairs = _merge_short_islands(pairs, min_duration_ms=MIN_ISLAND_DURATION_MS)
    if merged_pairs is None:
        return None

    group_id = str(group.get("group_id") or "")
    speaker_id = str(group.get("speaker_id") or "")
    parent_segment_ids = [str(value) for value in group.get("segment_ids") or []]
    parent_start_ms = int(group.get("start_ms") or 0)
    parent_source_end_ms = int(group.get("source_end_ms") or 0)
    parent_source_span_ms = int(group.get("source_span_ms") or 0)

    pseudo_groups: list[dict[str, Any]] = []
    for index, pair in enumerate(merged_pairs):
        island_id = f"{group_id}__isl{index:02d}"
        pseudo_groups.append({
            "group_id": island_id,
            "speaker_id": speaker_id,
            "segment_ids": [island_id],
            "start_ms": int(pair["start_ms"]),
            "source_end_ms": int(pair["end_ms"]),
            "source_span_ms": int(pair["end_ms"]) - int(pair["start_ms"]),
            "source_text": pair["en_text"],
            "source_segments": [{"segment_id": island_id, "source_text": pair["en_text"]}],
            "parts": [{"type": "text", "segment_id": island_id, "text": pair["zu_text"]}],
            "island_index": index,
            "island_count": len(merged_pairs),
            "alignment_method": pair.get("alignment_method"),
            "parent_group_id": group_id,
            "parent_speaker_id": speaker_id,
            "parent_segment_ids": parent_segment_ids,
            "parent_start_ms": parent_start_ms,
            "parent_source_end_ms": parent_source_end_ms,
            "parent_source_span_ms": parent_source_span_ms,
        })
    return pseudo_groups


def stitch_islands_to_group_wav(
    island_paths_in_order: Sequence[Path],
    output_path: Path,
    *,
    inter_island_silence_ms: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Concatenate island WAV files (in chronological order) into one continuous WAV.

    Azure TTS output is always canonicalized to one fixed mono/PCM16/sample-rate
    format, so this is pure frame concatenation -- a format mismatch between islands
    indicates a real upstream bug and is raised rather than silently papered over.
    """
    paths = list(island_paths_in_order)
    if not paths:
        raise ValueError("stitch_islands_to_group_wav requires at least one island path")
    silences = list(inter_island_silence_ms) if inter_island_silence_ms is not None else [0] * (len(paths) - 1)
    if len(silences) != len(paths) - 1:
        raise ValueError("inter_island_silence_ms must have exactly one entry per gap between islands")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_params: tuple[int, int, int] | None = None
    boundaries_ms: list[int] = []
    cursor_ms = 0
    with wave.open(str(output_path), "wb") as dst:
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as src:
                params = src.getparams()
                frames = src.readframes(params.nframes)
            shape = (params.nchannels, params.sampwidth, params.framerate)
            if reference_params is None:
                reference_params = shape
                dst.setparams((params.nchannels, params.sampwidth, params.framerate, 0, "NONE", "not compressed"))
            elif shape != reference_params:
                raise ValueError(f"Island WAV format mismatch at {path}: {shape} != {reference_params}")
            dst.writeframes(frames)
            cursor_ms += int(round(params.nframes * 1000.0 / params.framerate))
            boundaries_ms.append(cursor_ms)
            if index < len(silences) and silences[index] > 0:
                nchannels, sampwidth, framerate = reference_params
                silence_frames = int(round(silences[index] * framerate / 1000.0))
                dst.writeframes(b"\0" * silence_frames * nchannels * sampwidth)
                cursor_ms += silences[index]
    return {"final_duration_ms": cursor_ms, "island_boundaries_ms": boundaries_ms}


def slice_wav_by_bookmarks(
    source_path: Path,
    island_ids_in_order: Sequence[str],
    bookmark_offsets_ms: Mapping[str, int],
    output_dir: Path,
    *,
    output_suffix: str = ".sdk-slice.wav",
    trim_digital_silence_ms: int = 2000,
) -> dict[str, dict[str, Any]]:
    """Split one whole-group SDK synthesis WAV into per-island WAVs at bookmark offsets.

    Symmetric to stitch_islands_to_group_wav but doing frame-range extraction instead
    of concatenation: a single bookmarked Speech SDK call synthesizes a whole group in
    one pass (paying Azure's per-utterance overhead once instead of once per island --
    see azure_tts_sdk.synthesize_group_with_bookmarks), and this recovers the real
    per-island boundaries the downstream measure-and-repair controller needs by
    slicing that one WAV at the BookmarkReached offsets.

    Internal boundaries were originally assumed to carry no extra engine-inserted
    silence beyond what sits at the SSML text position, on the theory that real Azure
    silence only accumulates at the very start of the first island and the very end of
    the last one. Real-job validation disproved this: Azure's neural voice inserts a
    genuine multi-hundred-millisecond digital-silence pause after each sentence inside
    the SAME continuous synthesis, and because a bookmark only fires once the NEXT
    island's own speech begins, that whole pause lands inside the EARLIER island's
    slice. Confirmed real case: a 2-island group's isl00 slice carried 680ms of exact
    digital silence between its own speech and isl01's bookmark. Trim each slice the
    same way canonicalize_pcm_wav already trims whole-clip edges -- exact-zero-byte
    only, so it can never remove real speech, only the pause Azure itself inserted.
    """
    ids = list(island_ids_in_order)
    if not ids:
        raise ValueError("slice_wav_by_bookmarks requires at least one island id")
    missing = [island_id for island_id in ids if island_id not in bookmark_offsets_ms]
    if missing:
        raise ValueError(f"Missing bookmark offsets for island(s): {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(params.nframes)
    nchannels, sampwidth, framerate = params.nchannels, params.sampwidth, params.framerate
    total_duration_ms = int(round(params.nframes * 1000.0 / framerate))
    frame_size = nchannels * sampwidth

    starts_ms = [int(bookmark_offsets_ms[island_id]) for island_id in ids]
    ends_ms = starts_ms[1:] + [total_duration_ms]

    result: dict[str, dict[str, Any]] = {}
    for island_id, start_ms, end_ms in zip(ids, starts_ms, ends_ms):
        if end_ms <= start_ms:
            raise ValueError(
                f"Bookmark offsets for island {island_id!r} are inverted or empty: "
                f"start_ms={start_ms}, end_ms={end_ms}"
            )
        start_frame = int(round(start_ms * framerate / 1000.0))
        end_frame = int(round(end_ms * framerate / 1000.0))
        slice_frames = frames[start_frame * frame_size : end_frame * frame_size]

        raw_buffer = io.BytesIO()
        with wave.open(raw_buffer, "wb") as raw:
            raw.setparams((nchannels, sampwidth, framerate, 0, "NONE", "not compressed"))
            raw.writeframes(slice_frames)
        trimmed_audio, trim_metadata = canonicalize_pcm_wav(
            raw_buffer.getvalue(),
            sample_rate=framerate,
            channels=nchannels,
            sample_width=sampwidth,
            trim_digital_silence_ms=trim_digital_silence_ms,
        )

        output_path = output_dir / f"{island_id}{output_suffix}"
        output_path.write_bytes(trimmed_audio)
        result[island_id] = {
            "path": output_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": trim_metadata["duration_ms"],
            "trimmed_leading_ms": trim_metadata["trimmed_leading_ms"],
            "trimmed_trailing_ms": trim_metadata["trimmed_trailing_ms"],
        }
    return result
