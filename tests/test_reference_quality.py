import numpy as np

from mathula_tv import reference_quality as rq

RATE = 24000


def _speech(rng, s, a=0.08):
    return rng.normal(0, a, int(s * RATE)).astype(np.float32)


def _quiet(rng, s):
    return rng.normal(0, 0.0004, int(s * RATE)).astype(np.float32)


def test_pauses_inside_a_clip_are_counted_and_its_ends_are_not():
    rng = np.random.default_rng(1)
    steady = np.concatenate([_quiet(rng, 0.1), _speech(rng, 6.0), _quiet(rng, 0.1)])
    assert rq.inner_pauses(steady, RATE) == (0.0, 0)                                              # the quiet at its ends is the cut, not hesitation
    halting = np.concatenate([_speech(rng, 2.0), _quiet(rng, 0.5), _speech(rng, 2.0), _quiet(rng, 0.3), _speech(rng, 2.0)])
    seconds, count = rq.inner_pauses(halting, RATE)
    assert count == 2 and abs(seconds - 0.8) < 0.05
    assert rq.inner_pauses(np.concatenate([_speech(rng, 2.0), _quiet(rng, 0.06), _speech(rng, 2.0)]), RATE)[1] == 0          # a closure is not a pause


def test_a_steady_clip_scores_above_a_halting_one():
    rng = np.random.default_rng(2)
    words = lambda rates: [{"text": "a" * 8, "start": i, "end": i + 8 / r} for i, r in enumerate(rates)]
    steady = rq.measure(np.concatenate([_speech(rng, 6.0)]), RATE, words([12, 12.5, 11.5, 12, 12.2, 11.8]))
    halting = rq.measure(np.concatenate([_speech(rng, 2.0), _quiet(rng, 0.6), _speech(rng, 2.0), _quiet(rng, 0.4), _speech(rng, 2.0)]), RATE, words([12, 5, 18, 6, 20, 9]))
    assert steady["fluency"] > 0.95 and halting["fluency"] < 0.75 and steady["pauses"] == 0 and halting["pauses"] == 2
    assert halting["rate_cv"] > steady["rate_cv"] and rq.rate_spread(words([12, 12])) is None


def test_noise_clipping_and_fidelity_are_read_from_the_audio_and_the_aligner():
    rng = np.random.default_rng(3)
    clean = np.concatenate([_speech(rng, 2.0), _quiet(rng, 0.3), _speech(rng, 2.0)])
    noisy = clean + rng.normal(0, 0.02, len(clean)).astype(np.float32)
    assert rq.noise_ratio_db(clean, RATE) > rq.noise_ratio_db(noisy, RATE) + 10
    assert rq.clipped_share(np.clip(clean * 20, -1, 1)) > 0.2 and rq.clipped_share(clean) == 0.0
    aligned = [{"letters": [["a", 0.1, 0.9], ["b", 0.2, 0.7]]}, None, {"letters": [["c", 0.3, 0.8]]}]
    assert rq.fidelity(aligned) == 0.8 and rq.fidelity([None]) is None
