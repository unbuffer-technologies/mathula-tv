import numpy as np

from mathula_tv import phone_acoustics as pa

SR = pa.SAMPLE_RATE


def vowel(seconds: float, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return level * (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t))


def plosive_b() -> tuple[np.ndarray, tuple[str, float, float]]:
    """Silence (closure), a sharp burst, then the vowel: an English-style b.  The b spans 0.05-0.17 s."""
    rng = np.random.default_rng(0)
    signal = np.concatenate([vowel(0.05), 1e-4 * rng.standard_normal(int(0.12 * SR)), 0.4 * rng.standard_normal(int(0.004 * SR)), vowel(0.3)])
    return signal, ("b", 0.05, 0.17)


def implosive_b() -> tuple[np.ndarray, tuple[str, float, float]]:
    """Voicing through the closure at a low level, then a gradual rise over 40 ms into the vowel: no burst.  The b spans 0.05-0.17 s."""
    closure = vowel(0.12, 0.008)
    rise = int(0.04 * SR)
    onset = vowel(0.3) * np.concatenate([np.linspace(0.15, 1.0, rise), np.ones(int(0.3 * SR) - rise)])
    return np.concatenate([vowel(0.05), closure, onset]), ("b", 0.05, 0.17)


def test_a_plosive_has_a_sharp_release_after_a_silent_closure_and_an_implosive_does_not():
    plosive_signal, plosive_phone = plosive_b()
    implosive_signal, implosive_phone = implosive_b()
    plosive = pa.phone_features(plosive_signal, *plosive_phone)
    implosive = pa.phone_features(implosive_signal, *implosive_phone)
    assert plosive["release_rise_db"] > implosive["release_rise_db"] + 6
    assert plosive["voiced_closure"] < 0.3 and implosive["voiced_closure"] > 0.5


def test_only_stops_get_stop_measures_and_only_vowels_get_vowel_measures():
    features = pa.phone_features(vowel(0.3), "n", 0.05, 0.15)                     # a nasal: neither
    assert features == {"token": "n", "duration_ms": 100}
    vowel_measures = pa.phone_features(vowel(0.3), "a", 0.05, 0.15)
    assert "release_rise_db" not in vowel_measures and vowel_measures["token"] == "a"           # a vowel has no release; its mouth opening is measured


def test_differences_name_what_is_made_differently_even_when_the_labels_differ():
    plosive_signal, plosive_phone = plosive_b()
    implosive_signal, implosive_phone = implosive_b()
    speaker = [pa.timed_features(plosive_signal, [("p", *plosive_phone[1:])])] * 3        # the recogniser called the speaker's b "p"
    voice = pa.timed_features(implosive_signal, [implosive_phone])
    notes = pa.differences(speaker, voice)
    assert len(notes) == 1 and "softer" in notes[0] and "vocal folds vibrate" in notes[0] and "3 of the speaker's" in notes[0]
    same = pa.timed_features(plosive_signal, [plosive_phone])
    assert pa.differences(speaker, same) == []                                             # the same kind of b: nothing to report


def test_one_odd_occurrence_does_not_decide():
    plosive_signal, plosive_phone = plosive_b()
    odd_signal, odd_phone = implosive_b()
    good = pa.timed_features(plosive_signal, [plosive_phone])
    odd = pa.timed_features(odd_signal, [odd_phone])
    voice = pa.timed_features(plosive_signal, [plosive_phone])
    assert pa.differences([good, good, odd], voice) == []                                 # the median is the plosive


def _stop(token, voiced, delay=40, rise=15.0, closure=30, hiss=0.3):
    return {"token": token, "duration_ms": 40, "release_rise_db": rise, "closure_ms": closure, "voiced_closure": voiced, "voicing_delay_ms": delay, "hiss_after_release": hiss}


def test_affricates_are_paired_by_where_they_are_made_so_their_voicing_is_compared_whatever_the_label():
    speaker = [[{"token": "o", "duration_ms": 80}, _stop("dʒ", 1.0, delay=0)]]
    voice = [{"token": "o", "duration_ms": 80}, _stop("tʃ", 0.0, delay=60)]                  # the recogniser called the voice's sound ch
    notes = pa.differences(speaker, voice)
    assert notes and "vocal folds vibrate through 0% of the closure against the speaker's 100%" in notes[0] and "voicing starts 60 ms" in notes[0]


def test_measures_from_several_cuts_of_the_same_audio_are_compared_by_their_median():
    speaker = [[{"token": "o", "duration_ms": 80}, _stop("dʒ", 1.0, delay=0)]]
    cuts = [[{"token": "o", "duration_ms": 80}, _stop("dʒ", v, delay=60)] for v in (0.0, 0.0, 0.33, 0.67, 0.0)] + [[{"token": "o", "duration_ms": 80}]]
    notes = pa.differences_over_cuts(speaker, cuts)
    assert notes and "through 0% of the closure against the speaker's 100%" in notes[0]       # median of 0, 0, .33, .67, 0 (the cut with no stop is not counted)
    near = [[{"token": "o", "duration_ms": 80}, _stop("tʃ", v, delay=0)] for v in (1.0, 0.9, 1.0)]
    assert pa.differences_over_cuts(speaker, near) == []                                      # tʃ or dʒ, the voicing is the speaker's: nothing to report
    assert pa.differences_over_cuts(speaker, [[{"token": "o", "duration_ms": 80}]]) == []      # no stop found in the cut: no claim
    assert pa.differences_over_cuts([], near) == [] and pa.differences_over_cuts(speaker, []) == []


def _vowel(f1, f2, seconds=0.2, pitch=110.0):
    """A voiced sound whose resonances sit at f1 and f2: a pulse train through two resonators."""
    n = int(seconds * pa.SAMPLE_RATE)
    source = np.zeros(n)
    source[::int(pa.SAMPLE_RATE / pitch)] = 1.0
    out = source
    for centre, width in ((f1, 80.0), (f2, 120.0)):
        radius = np.exp(-np.pi * width / pa.SAMPLE_RATE)
        a1, a2 = -2 * radius * np.cos(2 * np.pi * centre / pa.SAMPLE_RATE), radius * radius
        y = np.zeros(n)
        for i in range(n):
            y[i] = out[i] - a1 * (y[i - 1] if i > 0 else 0) - a2 * (y[i - 2] if i > 1 else 0)
        out = y
    return out / (np.abs(out).max() + 1e-9) * 0.5


def test_how_open_the_mouth_is_comes_from_the_first_formant():
    open_a = pa.vowel_features(_vowel(750, 1300), 0.0, 0.2)
    closed_i = pa.vowel_features(_vowel(380, 2200), 0.0, 0.2)
    assert abs(open_a["onset_f1"] - 750) < 120 and abs(closed_i["onset_f1"] - 380) < 90
    assert open_a["onset_f1"] - closed_i["onset_f1"] > 250
    assert pa.vowel_features(np.zeros(3200), 0.0, 0.2) == {} and pa.vowel_features(_vowel(750, 1300), 0.0, 0.03) == {}       # silence or a sliver: no claim


def test_a_vowel_that_starts_more_closed_than_the_speakers_is_named_in_plain_words():
    speaker = [[{"token": "s", "duration_ms": 80}, {"token": "aɪ", "duration_ms": 160, "onset_f1": 880, "offset_f1": 480}, {"token": "n", "duration_ms": 60}]]
    shut = [{"token": "s", "duration_ms": 80}, {"token": "eɪ", "duration_ms": 160, "onset_f1": 394, "offset_f1": 311}, {"token": "n", "duration_ms": 60}]
    notes = pa.differences(speaker, shut)
    assert len(notes) == 1 and "vowel number 2 (eɪ)" in notes[0] and "starts more closed" in notes[0] and "394 Hz against the speaker's 880 Hz" in notes[0]
    assert "closes less" in notes[0]                                                                      # it hardly moves, the speaker's falls 400 Hz
    near = [{"token": "s", "duration_ms": 80}, {"token": "aɪ", "duration_ms": 160, "onset_f1": 780, "offset_f1": 420}, {"token": "n", "duration_ms": 60}]
    assert pa.differences(speaker, near) == []                                                            # a different vocal tract, not a different vowel
    cuts = [[{"token": "s"}, {"token": "eɪ", "onset_f1": value, "offset_f1": 320}, {"token": "n"}] for value in (390, 400, 700)]
    assert "starts more closed" in pa.differences_over_cuts(speaker, cuts)[0]                            # the median of the cuts, one odd cut does not decide


def test_the_vowel_note_says_what_the_voices_own_open_a_reaches():
    speaker = [[{"token": "s"}, {"token": "aɪ", "onset_f1": 940, "offset_f1": 500}, {"token": "n"}]]
    voice = [{"token": "s"}, {"token": "aɪ", "onset_f1": 560, "offset_f1": 330}, {"token": "n"}]
    assert "this voice's own plain 'a' reaches about 1200 Hz" in pa.differences(speaker, voice, voice_open_f1=1200)[0]
    assert "plain 'a'" not in pa.differences(speaker, voice)[0]
