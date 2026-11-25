import numpy as np

from mathula_tv import voice_takes as vt

RATE = 24000


def _word(freqs, seconds):
    t = np.arange(int(seconds * RATE)) / RATE
    wave = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate(freqs))
    return (0.2 * wave * np.hanning(len(t)) ** 0.3).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


LEAD_WORDS = [(200, 900, 2400), (330, 1500, 3000), (250, 700, 2100), (400, 1800, 3400), (180, 1100, 2700)]
NEW_WORDS = [(300, 2000, 3600), (220, 600, 1900)]


def _speech(words, pace):
    parts = []
    for f in words:
        parts += [_word(f, 0.30 * pace), _silence(0.06 * pace)]
    return np.concatenate(parts)


def test_the_end_of_the_repeated_sentence_is_found_by_alignment_even_when_the_generation_is_slower():
    alone = _speech(LEAD_WORDS, 1.0)
    lead = _speech(LEAD_WORDS, 1.2)                          # the same sentence, 20% slower inside the longer generation
    generation = np.concatenate([_silence(0.15), lead, _silence(0.1), _speech(NEW_WORDS, 1.0)])
    expected = 0.15 + len(lead) / RATE
    end = vt.align_end(generation, alone, RATE)
    assert end is not None and abs(end - expected) < 0.15


def test_the_cut_lands_in_the_quiet_after_the_last_repeated_word_and_not_the_quiet_between_words_before_it():
    alone = _speech(LEAD_WORDS, 1.0)
    lead = _speech(LEAD_WORDS, 1.2)
    generation = np.concatenate([_silence(0.15), lead, _silence(0.1), _speech(NEW_WORDS, 1.0)])
    end = vt.align_end(generation, alone, RATE)
    cut, found = vt.find_cut(generation, RATE, end, before_s=vt.CUT_WINDOW_ALIGNED_S, after_s=vt.CUT_WINDOW_ALIGNED_S)
    last_word_end = 0.15 + len(lead) / RATE - 0.06 * 1.2
    assert found and last_word_end - 0.05 <= cut <= last_word_end + 0.06 * 1.2 + 0.1 + 0.05
    assert cut > 0.15 + 4 * 0.36 * 1.2                       # not in the gap before the last repeated word


def test_too_little_audio_to_align_gives_none():
    assert vt.align_end(np.zeros(100, dtype=np.float32), np.zeros(50, dtype=np.float32), RATE) is None


def _speechlike(seconds: float, rate: int = 24000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    return (0.1 * np.sin(2 * np.pi * 150 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_a_digital_zero_hole_inside_speech_costs_the_take_and_is_filled():
    rate = 24000
    clean = np.concatenate([_speechlike(0.5), _speechlike(0.5, seed=1)])
    holed = np.concatenate([_speechlike(0.5), np.zeros(int(0.096 * rate), dtype=np.float32), _speechlike(0.5, seed=1)])
    assert vt.dropouts(clean, rate) == []
    assert len(vt.dropouts(holed, rate)) == 1
    assert vt.score_span(holed, rate)["score"] >= vt.score_span(clean, rate)["score"] + vt.DROPOUT_COST - 0.01
    tone = _speechlike(1.0, seed=2) * 0.05
    filled = vt.fill_dropouts(holed, rate, tone, 40.0, -20.0)
    assert vt.dropouts(filled, rate) == [] and len(filled) == len(holed)
    assert np.array_equal(filled[: int(0.5 * rate)], holed[: int(0.5 * rate)])        # speech untouched


def test_the_models_own_padding_at_the_ends_is_not_a_dropout():
    rate = 24000
    padded = np.concatenate([np.zeros(int(0.1 * rate), dtype=np.float32), _speechlike(0.5), np.zeros(int(0.1 * rate), dtype=np.float32)])
    assert vt.dropouts(padded, rate) == []
    cores = [(_speechlike(0.5), 2400, 2400), (_speechlike(0.5, seed=3), 0, 2400)]
    audio, _ = vt.assemble(cores, [0], rate, _speechlike(1.0, seed=2) * 0.05, 40.0)
    assert vt.dropouts(audio, rate) == []


def test_a_click_left_after_a_dropout_at_the_end_of_a_span_is_stripped_and_the_timeline_kept():
    rate = 24000
    speech = _speechlike(0.6)
    click = np.zeros(int(0.04 * rate), dtype=np.float32)
    click[10:60] = 0.5
    span = np.concatenate([speech, np.zeros(int(0.096 * rate), dtype=np.float32), click])
    core, head, tail = vt.strip_stray_ends(span, rate)
    assert head == 0 and tail == len(span) - len(speech)
    assert np.array_equal(core, speech)
    assert vt.dropouts(core, rate) == []


def test_a_dropout_in_the_middle_of_speech_is_not_stripped_and_a_fragment_at_the_head_is():
    rate = 24000
    middle = np.concatenate([_speechlike(0.5), np.zeros(int(0.1 * rate), dtype=np.float32), _speechlike(0.5, seed=1)])
    assert vt.strip_stray_ends(middle, rate)[1:] == (0, 0)
    fragment = np.concatenate([_speechlike(0.05), np.zeros(int(0.1 * rate), dtype=np.float32), _speechlike(0.5, seed=1)])
    core, head, tail = vt.strip_stray_ends(fragment, rate)
    assert head == int(0.15 * rate) and tail == 0 and len(core) == int(0.5 * rate)


def test_a_join_keeps_the_voices_own_spacing_unless_a_longer_pause_was_planned_and_keeps_only_edges_when_short():
    import pytest

    assert vt.join_target_s(-200, 0.36) == 0.36                       # the speaker ran on, the voice left 0.36 s: the voice's own spacing stands
    assert vt.join_target_s(700, 0.36) == pytest.approx(0.9)          # a held pause (0.2 + 0.7) is longer: it wins
    assert vt.join_target_s(-200, None) == 0.0 and vt.join_target_s(150, None) == pytest.approx(0.35)
    assert vt.edge_keep_s(0.0, share=0.6, most_s=0.12, least_s=0.03) == 0.03
    assert vt.edge_keep_s(0.9, share=0.6, most_s=0.12, least_s=0.03) == 0.12 and vt.edge_keep_s(0.9, share=0.4, most_s=0.08, least_s=0.03) == 0.08
    assert vt.join_fill_s(0.36, 0.12, 0.08) == pytest.approx(0.16)
    assert vt.join_fill_s(0.1, 0.12, 0.08) == 0.0


def _speech_block(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.1 * np.sin(2 * np.pi * 150 * t) + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


def test_voiced_sound_before_the_first_word_of_a_cold_span_is_cut_and_a_quiet_lead_in_is_left():
    quiet = (0.0004 * np.random.default_rng(1).standard_normal(int(0.5 * RATE))).astype(np.float32)
    stray = np.concatenate([_speech_block(0.4, 5), quiet[: int(0.1 * RATE)]])                       # 0.4 s of speech-level sound, then a pause: "out"
    core = np.concatenate([stray, _speech_block(3.0, 2)])
    first_word = len(stray) / RATE
    cut = vt.stray_before(core, RATE, first_word)
    assert cut is not None and abs(cut - (first_word - vt.HEAD_KEEP_S)) < 1e-6                      # kept HEAD_KEEP_S in front of the word
    calm = np.concatenate([quiet, _speech_block(3.0, 2)])
    assert vt.stray_before(calm, RATE, len(quiet) / RATE) is None                                    # a quiet half second before the first word is silence, not a sound
    assert vt.stray_before(np.concatenate([_speech_block(0.1, 3), _speech_block(3.0, 2)]), RATE, 0.1) is None     # too little before the word to be anything


def test_voiced_sound_after_the_last_word_of_a_span_nothing_follows_is_cut():
    body = _speech_block(3.0, 2)
    tail = _speech_block(0.5, 7)
    core = np.concatenate([body, np.zeros(int(0.05 * RATE), dtype=np.float32) + 0.0003, tail])
    cut = vt.stray_after(core, RATE, 3.0)
    assert cut is not None and abs(cut - (3.0 + vt.TAIL_KEEP_S)) < 1e-6
    assert vt.stray_after(np.concatenate([body, _speech_block(0.1, 3)]), RATE, 3.0) is None        # a word's own decay is not a stray sound


def test_a_pause_the_speaker_makes_themself_is_not_held_against_a_take():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate = 24000
    speech = (np.sin(np.linspace(0, 2000, rate)) * 0.3).astype(np.float32)
    take = np.concatenate([speech, np.zeros(int(0.35 * rate), dtype=np.float32), speech])          # one 0.35 s pause in the middle
    assert vt.quiet_seconds(take, rate) > 0.3                           # counted by default: the take with fewer pauses wins, which is the breathless one
    assert vt.quiet_seconds(take, rate, 0.5) == 0.0                     # the speaker pauses that long themself: phrasing, not hesitation
    assert vt.score_span(take, rate, 0.5)["score"] < vt.score_span(take, rate)["score"]
