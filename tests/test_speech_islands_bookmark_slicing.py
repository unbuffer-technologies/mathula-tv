"""Tests for slice_wav_by_bookmarks, the counterpart to stitch_islands_to_group_wav
used by the Speech SDK grouped-synthesis path: it splits ONE whole-group WAV (produced
by a single bookmarked SDK call) back into per-island WAVs at bookmark offsets.
"""
import wave

import pytest

from mathula_tv.speech_islands import slice_wav_by_bookmarks


def _write_tone_wav(path, *, seconds: float, framerate: int = 16000, sample_value: int = 10000) -> None:
    nframes = int(round(seconds * framerate))
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(sample_value.to_bytes(2, "little", signed=False) * nframes)


def test_slices_three_islands_at_bookmark_offsets(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=3.0)  # 3000ms total
    offsets = {"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}

    result = slice_wav_by_bookmarks(source, ["g1__isl00", "g1__isl01", "g1__isl02"], offsets, tmp_path)

    assert result["g1__isl00"]["start_ms"] == 0
    assert result["g1__isl00"]["end_ms"] == 1000
    assert result["g1__isl00"]["duration_ms"] == 1000
    assert result["g1__isl01"]["start_ms"] == 1000
    assert result["g1__isl01"]["end_ms"] == 1800
    assert result["g1__isl02"]["start_ms"] == 1800
    assert result["g1__isl02"]["end_ms"] == 3000
    assert result["g1__isl02"]["duration_ms"] == 1200

    for island_id, info in result.items():
        assert info["path"].exists()
        with wave.open(str(info["path"]), "rb") as handle:
            params = handle.getparams()
            assert params.nchannels == 1 and params.sampwidth == 2 and params.framerate == 16000
            expected_frames = int(round(info["duration_ms"] * 16000 / 1000.0))
            assert params.nframes == expected_frames


def test_slice_filenames_use_the_configured_suffix(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=1.0)
    result = slice_wav_by_bookmarks(source, ["g1__isl00"], {"g1__isl00": 0}, tmp_path)
    assert result["g1__isl00"]["path"].name == "g1__isl00.sdk-slice.wav"


def test_single_island_covers_whole_source(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=2.5)
    result = slice_wav_by_bookmarks(source, ["only"], {"only": 0}, tmp_path)
    assert result["only"]["start_ms"] == 0
    assert result["only"]["end_ms"] == 2500


def test_rejects_missing_bookmark_offset(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=1.0)
    with pytest.raises(ValueError, match="Missing bookmark offsets"):
        slice_wav_by_bookmarks(source, ["a", "b"], {"a": 0}, tmp_path)


def test_rejects_inverted_offsets(tmp_path):
    # A bookmark-reporting bug (offsets out of order) must surface loudly rather than
    # silently produce a broken (zero-or-negative-length) slice.
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=2.0)
    with pytest.raises(ValueError, match="inverted or empty"):
        slice_wav_by_bookmarks(source, ["a", "b"], {"a": 1000, "b": 500}, tmp_path)


def test_rejects_duplicate_offsets_producing_empty_slice(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=2.0)
    with pytest.raises(ValueError, match="inverted or empty"):
        slice_wav_by_bookmarks(source, ["a", "b"], {"a": 500, "b": 500}, tmp_path)


def test_requires_at_least_one_island(tmp_path):
    source = tmp_path / "group.sdk-group.wav"
    _write_tone_wav(source, seconds=1.0)
    with pytest.raises(ValueError, match="at least one island"):
        slice_wav_by_bookmarks(source, [], {}, tmp_path)


# --- internal-boundary silence trimming ---------------------------------------------
# Real-job validation found the original assumption behind this module (that internal
# bookmark boundaries carry no extra engine-inserted silence) was wrong: Azure's neural
# voice inserts a genuine digital-silence pause after each sentence inside the SAME
# continuous synthesis, and because a bookmark only fires once the NEXT island's own
# speech begins, that whole pause lands inside the EARLIER island's slice. Confirmed
# real case: a 2-island group's isl00 slice carried 680ms of exact digital silence
# between its own speech and isl01's bookmark. Slices are now trimmed the same
# amplitude-threshold way canonicalize_pcm_wav trims whole-clip edges (real Azure
# "silence" is usually low-level dither noise, not literal 0x0000).

def test_trailing_digital_silence_before_the_next_bookmark_is_trimmed(tmp_path):
    framerate = 16000
    speech_frames = int(0.3 * framerate)  # 300ms of real "speech"
    silence_frames = int(0.68 * framerate)  # 680ms of digital silence, matching the real case
    source = tmp_path / "group.sdk-group.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes((10000).to_bytes(2, "little") * speech_frames)  # island a: speech
        handle.writeframes((0).to_bytes(2, "little", signed=True) * silence_frames)  # island a: trailing silence
        handle.writeframes((10000).to_bytes(2, "little") * speech_frames)  # island b: speech

    total_ms = int(round((speech_frames + silence_frames + speech_frames) * 1000 / framerate))
    bookmark_b_ms = int(round((speech_frames + silence_frames) * 1000 / framerate))
    result = slice_wav_by_bookmarks(source, ["a", "b"], {"a": 0, "b": bookmark_b_ms}, tmp_path)

    # island a's RAW bookmark-computed span still includes the trailing silence...
    assert result["a"]["end_ms"] == bookmark_b_ms
    # ...but its measured duration_ms (and the WAV actually written) is trimmed down
    # to just the real speech, not the raw bookmark-to-bookmark span.
    assert result["a"]["duration_ms"] < bookmark_b_ms - 50
    assert result["a"]["trimmed_trailing_ms"] > 600
    with wave.open(str(result["a"]["path"]), "rb") as handle:
        assert handle.getnframes() < speech_frames + silence_frames - int(0.5 * framerate)

    # island b (no trailing silence in this fixture) is essentially untouched.
    assert result["b"]["end_ms"] == total_ms
    assert abs(result["b"]["duration_ms"] - (total_ms - bookmark_b_ms)) <= 1


def test_slices_preserve_frame_content_boundaries(tmp_path):
    # Write a WAV whose first half and second half have distinguishable sample
    # values, then confirm slicing actually cuts at the right frame, not just
    # produces the right nframes count.
    framerate = 16000
    first_half_frames = framerate  # 1s of value 10000
    second_half_frames = framerate  # 1s of value 20000
    source = tmp_path / "group.sdk-group.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes((10000).to_bytes(2, "little") * first_half_frames)
        handle.writeframes((20000).to_bytes(2, "little") * second_half_frames)

    result = slice_wav_by_bookmarks(source, ["a", "b"], {"a": 0, "b": 1000}, tmp_path)

    with wave.open(str(result["a"]["path"]), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        assert frames == (10000).to_bytes(2, "little") * first_half_frames
    with wave.open(str(result["b"]["path"]), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        assert frames == (20000).to_bytes(2, "little") * second_half_frames
