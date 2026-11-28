import numpy as np

from mathula_tv import room_tone as rt


def test_the_room_tone_goes_under_every_stretch_of_exact_silence_and_nothing_else():
    rate = rt.RATE
    rng = np.random.default_rng(2)
    tone = (rng.normal(0, 0.002, rate)).astype(np.float32)
    speech = lambda s: rng.normal(0, 0.1, int(s * rate)).astype(np.float32)
    track = np.concatenate([np.zeros(int(0.5 * rate), np.float32), speech(1.0), np.zeros(int(1.6 * rate), np.float32), speech(1.0), np.zeros(int(0.005 * rate), np.float32), speech(0.5)])
    out, seconds = rt.bed(track, rate, tone, 40.0)
    assert abs(seconds - 2.1) < 0.01                                                        # the 0.5 s before, and the 1.6 s between; the 5 ms hole is left alone
    gap = out[int(1.5 * rate): int(3.1 * rate)]
    assert np.abs(gap).max() > 0 and 20 * np.log10(np.sqrt((gap[int(0.1 * rate): -int(0.1 * rate)] ** 2).mean())) < -45 and 20 * np.log10(np.sqrt((gap[int(0.1 * rate): -int(0.1 * rate)] ** 2).mean())) > -80
    assert np.array_equal(out[int(0.5 * rate): int(1.5 * rate)], track[int(0.5 * rate): int(1.5 * rate)])          # the speech is untouched
    silent, none = rt.bed(np.zeros(1000, np.float32), rate, tone, 40.0)
    assert none == 0.0 and not silent.any()                                                 # nothing to match the level to: left as it was
