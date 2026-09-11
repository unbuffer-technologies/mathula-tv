"""Tests for native_dub._synthesize_speaker_turns_with_bookmarks (Phase 14): the
pre-pass that synthesizes a whole multi-member speaker turn in ONE bookmarked
Speech SDK call, makes ONE speed-fit decision against the turn's own aggregate
window, and slices the result back into real per-sentence WAVs -- instead of
fitting every sentence in the turn independently, which has a confirmed real
defect (the turn's last member drifts audibly past its true mouth-close because
its own "next thing" is a distant different turn -- see the plan's Phase 14
section for the full diagnosis).

Modeled directly on tests/test_native_sdk_presynthesis.py's fixture pattern
(FakeBoundary/pcm_wav/completed-response), the direct precedent this new
mechanism is built from at turn instead of island granularity. The "compress"
branch invokes real ffmpeg (via the unmodified _speed_fit_overflow_wav),
matching the real-audio, tolerance-based assertion style already used in
tests/test_native_hard_sync_speed.py -- not mocked, since atempo needs real
audio content to process meaningfully.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.azure_tts_sdk import SDKBookmarkEvent, SDKGroupSynthesisResponse
from mathula_tv.native_dub import _synthesize_speaker_turns_with_bookmarks
from mathula_tv.tts_ssml import SSMLBounds

VOICE = "zu-ZA-ThandoNeural"


def _pcm_wav(*, seconds: float, rate: int = 24_000) -> bytes:
    """Constant-value (non-silent, non-zero) tone -- canonicalize_pcm_wav's
    exact-zero-byte-only trimming never removes any of it, so slice durations
    come out as exact arithmetic on the bookmark offsets given.
    """
    frames = (1200).to_bytes(2, "little", signed=True) * int(rate * seconds)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    return output.getvalue()


def _sine_wav(path: Path, *, seconds: float, rate: int = 24_000) -> None:
    """A real sine tone -- used only for the compress-branch test, where real
    ffmpeg atempo processing needs real audio content, matching
    test_native_hard_sync_speed.py's own _write_tone pattern.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        payload = bytearray()
        for n in range(frames):
            sample = int(7000 * math.sin(2 * math.pi * 220 * n / rate))
            payload += struct.pack("<h", sample)
        handle.writeframes(payload)


def _joint_wav_with_inter_sentence_pauses(segments: list[tuple[int, int]], *, rate: int = 24_000) -> bytes:
    """Build one continuous WAV from an ordered sequence of (tone_ms, pause_ms)
    segments concatenated directly -- simulating one real Azure bookmarked
    synthesis call where each sentence's real speech is immediately followed
    by a real, genuine digital silence pause before the next sentence's own
    bookmark fires (exactly the documented real-world case that motivates
    slice_wav_by_bookmarks's own trimming feature -- and the case this
    mechanism must NOT silently strip, per the confirmed real regression on
    job 33cd7b46...).
    """
    frames = bytearray()
    for tone_ms, pause_ms in segments:
        frames += (1200).to_bytes(2, "little", signed=True) * int(rate * tone_ms / 1000)
        frames += (0).to_bytes(2, "little", signed=True) * int(rate * pause_ms / 1000)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(bytes(frames))
    return output.getvalue()


class FakeBoundary:
    def __init__(self, *, responses=()):
        self.responses = list(responses)
        self.calls: list[str] = []

    def synthesize_group(self, ssml, *, timeout_seconds):
        self.calls.append(ssml)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _completed_response(bookmarks_ms: dict[str, int], *, audio: bytes):
    marks = sorted(bookmarks_ms, key=lambda mark: bookmarks_ms[mark])
    return SDKGroupSynthesisResponse(
        audio_data=audio,
        bookmarks=tuple(
            SDKBookmarkEvent(mark=mark, audio_offset_ticks=bookmarks_ms[mark] * 10_000) for mark in marks
        ),
    )


def _fake_tts_backend():
    return SimpleNamespace(
        configured_voices=(VOICE,), sample_rate=24_000, channels=1, sample_width=2, ssml_bounds=SSMLBounds(),
    )


def _turn_group(group_id: str, *, speaker_id="S0", start_ms, end_ms, text="Normal sentence right here now."):
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": start_ms,
        "source_end_ms": end_ms, "source_span_ms": end_ms - start_ms, "source_text": text,
        "parts": [{"type": "text", "segment_id": f"{group_id}_seg", "text": text}],
    }


def _voice_assignments(*speaker_ids):
    speaker_ids = speaker_ids or ("S0",)
    return {
        speaker_id: {"selected_voice": VOICE, "base_prosody": {"pitch_percent": 0, "volume_percent": 0}}
        for speaker_id in speaker_ids
    }


def _run(*, phrase_groups, sdk_boundary, tts=None, source_duration_ms=20_000, progress=None, voice_assignments=None):
    return _synthesize_speaker_turns_with_bookmarks(
        phrase_groups=phrase_groups, source_duration_ms=source_duration_ms,
        voice_assignments=voice_assignments or _voice_assignments(), tts=tts or _fake_tts_backend(),
        sdk_boundary=sdk_boundary, audio_dir=Path("."), preferred_raw_speed_percent=6,
        max_natural_speed_percent=12, mouth_close_lag_tolerance_ms=40,
        protected_gap_overflow=True, max_protected_gap_overflow_ms=400, min_protected_pause_ms=120,
        force=True, progress=progress,
    )


def test_single_member_turns_are_never_eligible(tmp_path):
    boundary = FakeBoundary()
    groups = [_turn_group("g1", start_ms=0, end_ms=3000)]
    result = _run(phrase_groups=groups, sdk_boundary=boundary)
    assert result == {}
    assert boundary.calls == []


def test_a_turn_containing_a_bridge_member_is_synthesized_with_it(tmp_path):
    # A bridge member no longer disqualifies its turn: once timing is decided
    # at the aggregate-turn level, a bridge's own tiny native window is never
    # independently judged at all, so it's treated exactly like any other
    # member. A 2-member turn (one normal, one bridge) is eligible.
    audio = _pcm_wav(seconds=3.9)
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 3000}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence right here now."),
        _turn_group("g2", start_ms=3000, end_ms=3900, text="All right."),  # bridge: <=4 words, <=1500ms
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert set(result) == {"g1", "g2"}
    assert len(boundary.calls) == 1  # g1+g2 synthesized together, one real call


def test_a_bridge_at_the_start_of_a_turn_no_longer_excludes_the_rest_of_it(tmp_path):
    # Real production regression this fixes: "That's right, Mfundo." (a
    # bridge) sat at the very start of an 86-second, 10-member turn, and the
    # OLD whole-turn exclusion rule silently reverted every other member back
    # to the independent per-sentence path. Now the bridge is simply included
    # in the same one-shot synthesis as the rest of its turn.
    audio = _pcm_wav(seconds=6.9)
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 900, "g3": 3900}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=900, text="All right."),  # bridge: <=4 words, <=1500ms
        _turn_group("g2", start_ms=900, end_ms=3900, text="Normal sentence one right here now."),
        _turn_group("g3", start_ms=3900, end_ms=6900, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert set(result) == {"g1", "g2", "g3"}
    assert result["g1"]["final_duration_ms"] == 900
    assert result["g2"]["final_duration_ms"] == 3000
    assert result["g3"]["final_duration_ms"] == 3000
    assert len(boundary.calls) == 1  # all three synthesized together, one real call


def test_exact_fit_turn_uses_natural_slices_directly(tmp_path):
    # Turn span 6000ms. Natural total lands EXACTLY at the window (6000ms) --
    # not a byte over -- so no fit is needed at all. Any real overflow, however
    # small, now demands tight compression to the true window (see the
    # "longer speech has real compression headroom" branch below), so this
    # exact-fit case is the only natural-pace scenario left to test here for a
    # non-bridge turn -- gap-overflow leniency stays a bridge-only concept.
    audio = _pcm_wav(seconds=6.0)
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 3000}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert set(result) == {"g1", "g2"}
    assert result["g1"]["natural_duration_ms"] == result["g1"]["final_duration_ms"] == 3000
    assert result["g2"]["natural_duration_ms"] == result["g2"]["final_duration_ms"] == 3000
    assert result["g1"]["turn_speed_fit"]["method"] == "none"
    assert result["g1"]["voice"] == VOICE
    for gid in ("g1", "g2"):
        assert Path(result[gid]["path"]).exists()
    assert len(boundary.calls) == 1


def test_natural_slices_preserve_real_inter_sentence_pauses(tmp_path):
    """Confirmed real regression (job 33cd7b46...): the default slicing helper
    trims trailing digital silence (correct for the unrelated island-
    presynthesis use case) -- reused unmodified here, it silently deleted a
    real, deliberate Azure inter-sentence pause from an interior member's
    slice, shrinking a real 6-member turn's total by ~3.2s below its own
    already-fitted target. g1's real content (1000ms) is immediately followed
    by a real 500ms digital-silence pause before g2's bookmark fires -- exactly
    Azure's own documented real behavior -- and the combined slice durations
    must equal the full natural total (2500ms), not 2000ms.
    """
    audio = _joint_wav_with_inter_sentence_pauses([(1000, 500), (1000, 0)])
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 1500}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=1250, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=1250, end_ms=2500, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert result["g1"]["turn_speed_fit"]["method"] == "none"  # confirms the "fits" branch ran
    assert result["g1"]["final_duration_ms"] + result["g2"]["final_duration_ms"] == 2500
    assert result["g1"]["final_duration_ms"] == 1500  # 1000ms real speech + its own real pause
    assert result["g2"]["final_duration_ms"] == 1000


def test_interior_boundaries_reflow_to_the_real_audio_not_the_stale_estimate(tmp_path):
    """Confirmed real defect (job 33cd7b46...): placing each member's real,
    naturally-paced slice at its OLD, independent-per-sentence-ESTIMATE start_ms
    (from Phase 13's decision-time proportional-share reallocation) produced
    audible silence gaps INSIDE a turn that used to play as one continuous
    utterance, because the real one-shot synthesis paces differently than that
    estimate assumed. Only the turn's OUTER boundaries -- the first member's
    real source start (true lip movement start) and wherever the real audio
    actually ends when played from there (true lip movement stop) -- are real
    visual events worth pinning; interior boundaries must reflow to the real
    per-member slice durations instead, so the turn keeps playing gap-free.
    """
    # Bookmark split is asymmetric (900ms / 5100ms) and deliberately does NOT
    # match the groups' own committed 3000ms/3000ms windows -- if placement used
    # the stale per-sentence start_ms (3000) instead of the real slice boundary
    # (900), g2 would start 2100ms later than its own real audio, opening a gap.
    # Total natural duration (6000ms) lands EXACTLY at the turn's own window,
    # so no compression fires here -- this test isolates interior placement
    # from the (separately tested) compression-target behavior.
    audio = _pcm_wav(seconds=6.0)
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 900}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=1000, end_ms=4000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=4000, end_ms=7000, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    # g1's real slice (900ms) is placed starting at the turn's TRUE outer start
    # (1000, the first member's own real committed start -- the lip-movement
    # anchor), not wherever Phase 13 estimated internally.
    assert result["g1"]["start_ms"] == 1000
    assert result["g1"]["source_end_ms"] == 1000 + 900
    # g2 starts EXACTLY where g1's real audio ends -- zero internal gap --
    # rather than at its own stale committed start_ms (4000).
    assert result["g2"]["start_ms"] == result["g1"]["source_end_ms"] == 1900
    assert result["g2"]["source_end_ms"] == 1900 + 5100
    # The turn's real total lands exactly at the true outer end (1000 + 6000 =
    # 7000), which also happens to equal the OLD raw committed end here since
    # this fixture's natural total was chosen to land exactly on the window.
    assert result["g2"]["source_end_ms"] == 7000


def test_undersized_turn_pads_only_the_last_members_slice(tmp_path):
    # Turn span 6000ms; natural total only 5000ms (well short even with early
    # tolerance) -- but free_gap_ms is huge (next turn far away), so the small-
    # shortfall pad path can't apply (shortfall 1000ms > MAX_UNRESOLVED_EARLY_
    # SILENCE_PAD_MS=150) -- accepted at natural pace instead. Use a shortfall
    # within the pad ceiling to exercise the real padding branch.
    audio = _pcm_wav(seconds=5.9)  # shortfall vs 6000ms window is 100ms <= 150ms cap
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 2950}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert result["g1"]["final_duration_ms"] == result["g1"]["natural_duration_ms"] == 2950
    # Only the LAST member's slice grows (padding only ever extends the tail).
    assert result["g2"]["final_duration_ms"] > result["g2"]["natural_duration_ms"] == 2950
    assert result["g1"]["turn_speed_fit"]["method"] == "trailing_silence_pad"


def test_overflow_turn_applies_one_shared_speed_factor_to_both_members(tmp_path):
    # Turn span 6000ms -- the TRUE window, not window+late-tolerance, is the
    # compression target (a non-bridge turn always demands tight mouth-close
    # sync once it has real overflow to compress -- see the module-level note
    # on the compress branch). Real sine audio deliberately much longer
    # (9000ms) to force real ffmpeg atempo compression -- both members must
    # show the SAME speed_factor (one shared decision), and both real slices
    # must exist.
    source_wav = tmp_path / "turn_source.wav"
    _sine_wav(source_wav, seconds=9.0)
    audio = source_wav.read_bytes()
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 4500}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    tts = _fake_tts_backend()
    result = _synthesize_speaker_turns_with_bookmarks(
        phrase_groups=groups, source_duration_ms=20_000, voice_assignments=_voice_assignments(),
        tts=tts, sdk_boundary=boundary, audio_dir=tmp_path, preferred_raw_speed_percent=6,
        max_natural_speed_percent=12, mouth_close_lag_tolerance_ms=40, protected_gap_overflow=True,
        max_protected_gap_overflow_ms=400, min_protected_pause_ms=120, force=True, progress=None,
    )
    assert result["g1"]["turn_speed_fit"]["method"] == "ffmpeg_atempo"
    assert result["g1"]["turn_speed_fit"]["speed_factor"] == result["g2"]["turn_speed_fit"]["speed_factor"]
    assert result["g1"]["final_duration_ms"] < result["g1"]["natural_duration_ms"]
    assert result["g2"]["final_duration_ms"] < result["g2"]["natural_duration_ms"]
    for gid in ("g1", "g2"):
        assert Path(result[gid]["path"]).exists()
    # Sped-up total should land close to the TRUE 6000ms window (real ffmpeg
    # quantization tolerance, matching test_native_hard_sync_speed.py's own
    # range-based assertions rather than exact equality) -- not the old,
    # looser window+late-tolerance target.
    total_final = result["g1"]["final_duration_ms"] + result["g2"]["final_duration_ms"]
    assert 5900 <= total_final <= 6100


def test_small_overflow_still_gets_hard_synced_not_left_to_drift(tmp_path):
    """The user's own compression-policy insight, directly tested: a non-bridge
    turn that overflows its window by only a small amount (well within what the
    OLD window+late-tolerance target would have silently accepted at natural
    pace) must still be hard-synced to the TRUE window, not allowed to drift
    into the following silence -- because a real, multi-sentence turn always
    has genuine compression headroom, unlike a short bridge utterance (which
    this mechanism excludes from eligibility entirely). Real sine audio is
    used (not the constant-tone fixture) since this exercises real ffmpeg
    atempo, not just slice-duration arithmetic.
    """
    source_wav = tmp_path / "small-overflow-source.wav"
    _sine_wav(source_wav, seconds=6.3)  # 300ms over the 6000ms window
    audio = source_wav.read_bytes()
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 3150}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    result = _run(
        phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000,
    )
    assert result["g1"]["turn_speed_fit"]["method"] == "ffmpeg_atempo"
    total_final = result["g1"]["final_duration_ms"] + result["g2"]["final_duration_ms"]
    # Landed at the TRUE window, not drifted out to the old ~6400ms tolerance.
    assert total_final <= 6100



def test_a_failing_turn_falls_back_with_no_members_finalized(tmp_path):
    boundary = FakeBoundary(responses=[RuntimeError("Azure Speech SDK is not installed")])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    warnings: list[str] = []
    result = _run(phrase_groups=groups, sdk_boundary=boundary, progress=warnings.append)
    assert result == {}
    assert any("FAILED for g1..g2" in message for message in warnings)


def test_only_the_failing_turn_falls_back_others_still_finalize(tmp_path):
    audio = _pcm_wav(seconds=6.2)
    boundary = FakeBoundary(
        responses=[RuntimeError("boom"), _completed_response({"g3": 0, "g4": 3100}, audio=audio)]
    )
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text="Normal sentence one right here now."),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
        _turn_group("g3", speaker_id="S1", start_ms=10_000, end_ms=13_000, text="Another sentence one right here."),
        _turn_group("g4", speaker_id="S1", start_ms=13_000, end_ms=16_000, text="Another sentence two right here."),
    ]
    result = _run(
        phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000,
        voice_assignments=_voice_assignments("S0", "S1"),
    )
    assert set(result) == {"g3", "g4"}


# --- wiring: _synthesize_speaker_turns_with_bookmarks uses one flat prosody wrapper ----


def test_turn_synthesis_sends_one_flat_prosody_wrapper_for_the_whole_turn(tmp_path):
    # A per-clause variable-rate SSML mechanism (multiple sibling <prosody>
    # elements, one per punctuation-bounded clause) was built, then reverted
    # (2026-09-03) after a real A/B listening test found it sounded audibly
    # worse than one flat <prosody> wrapper around the whole turn -- splitting
    # a continuous utterance into multiple prosody-scoped elements disrupts
    # Azure's own natural inter-clause prosodic continuity. This proves the
    # SSML sent for a turn is exactly one <prosody> element, not several.
    dense_text = (
        "Sentence with a normal opening, then a genuinely much longer and far more "
        "information dense closing clause packed full of many extra words right here."
    )
    audio = _pcm_wav(seconds=6.0)
    boundary = FakeBoundary(responses=[_completed_response({"g1": 0, "g2": 3000}, audio=audio)])
    groups = [
        _turn_group("g1", start_ms=0, end_ms=3000, text=dense_text),
        _turn_group("g2", start_ms=3000, end_ms=6000, text="Normal sentence two right here now."),
    ]
    result = _run(phrase_groups=groups, sdk_boundary=boundary, source_duration_ms=20_000)
    assert set(result) == {"g1", "g2"}
    sent_ssml = boundary.calls[0]
    assert sent_ssml.count("<prosody") == 1  # exactly one wrapper for the whole turn
    assert 'rate="-' not in sent_ssml  # no per-clause slowdown mechanism left
