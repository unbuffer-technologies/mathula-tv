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


def test_only_stops_get_stop_measures():
    features = pa.phone_features(vowel(0.3), "a", 0.05, 0.15)
    assert features == {"token": "a", "duration_ms": 100}


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
