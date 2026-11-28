import numpy as np

from mathula_tv import cut_audit as ca

RATE = 24000


def _speech(rng, seconds, level=0.08):
    return rng.normal(0, level, int(seconds * RATE)).astype(np.float32)


def _quiet(rng, seconds):
    return rng.normal(0, 0.0004, int(seconds * RATE)).astype(np.float32)


def _dub(*parts):
    return np.concatenate(parts)


def test_a_sliver_of_sound_alone_in_a_pause_is_flagged_with_its_time_and_unit():
    rng = np.random.default_rng(3)
    clean = _dub(_speech(rng, 3.0), _quiet(rng, 0.9), _speech(rng, 3.0))
    assert ca.find_seam_flags(clean, RATE, [0.0]) == []
    leaked = _dub(_speech(rng, 3.0), _quiet(rng, 0.45), _speech(rng, 0.04), _quiet(rng, 0.45), _speech(rng, 3.0))          # prod-y1 0:56: 40 ms of the next word in the middle of the pause
    flags = ca.find_seam_flags(leaked, RATE, [0.0, 2.0])
    assert [f["kind"] for f in flags] == ["burst_in_pause"] and abs(flags[0]["t"] - 3.45) < 0.03
    assert flags[0]["unit"] == 1 and flags[0]["length_ms"] in range(30, 60) and flags[0]["quiet_before_ms"] >= 400 and flags[0]["time"] == "0:03.5"


def test_a_word_with_speech_close_to_it_is_not_a_burst_and_a_long_sound_is_not_either():
    rng = np.random.default_rng(4)
    close = _dub(_speech(rng, 2.0), _quiet(rng, 0.06), _speech(rng, 0.2), _quiet(rng, 0.06), _speech(rng, 2.0))          # a short word between words at a normal pace
    assert ca.find_seam_flags(close, RATE) == []
    word = _dub(_speech(rng, 2.0), _quiet(rng, 0.5), _speech(rng, 0.6), _quiet(rng, 0.5), _speech(rng, 2.0))              # 0.6 s is a word
    assert ca.find_seam_flags(word, RATE) == []


def test_speech_that_jumps_in_level_across_a_pause_and_a_stall_inside_a_unit_are_flagged():
    rng = np.random.default_rng(5)
    jump = _dub(_speech(rng, 3.0, 0.02), _quiet(rng, 0.4), _speech(rng, 3.0, 0.12))
    flags = ca.find_seam_flags(jump, RATE)
    assert [f["kind"] for f in flags] == ["level_step"] and flags[0]["step_db"] > 7.0
    stall = _dub(_speech(rng, 3.0), _quiet(rng, 2.6), _speech(rng, 3.0))
    assert [f["kind"] for f in ca.find_seam_flags(stall, RATE, [0.0])] == ["long_pause"]
    assert ca.find_seam_flags(stall, RATE, [0.0, 4.0]) == []                                      # a long silence between two units is the source's own pause


def test_the_report_lists_each_flag_with_its_time():
    flags = [{"t": 56.66, "time": "0:56.7", "unit": 2, "kind": "burst_in_pause", "length_ms": 40}]
    assert ca.report(flags)[0].strip().startswith("0:56.7") and "unit 2" in ca.report(flags)[0] and "burst_in_pause" in ca.report(flags)[0]


def test_a_faint_breath_or_the_rooms_ripple_in_a_pause_is_not_a_burst():
    rng = np.random.default_rng(6)
    breath = _dub(_speech(rng, 3.0), _quiet(rng, 0.4), _speech(rng, 0.1, 0.006), _quiet(rng, 0.4), _speech(rng, 3.0))          # about 22 dB under the speech
    assert ca.find_seam_flags(breath, RATE) == []
