import unittest

import numpy as np

from mathula_tv import audio_glitches as ag

SR = ag.SAMPLE_RATE


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    """A steady voiced-like signal at about -20 dBFS."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wave = 0.10 * np.sin(2 * np.pi * 140 * t) + 0.05 * np.sin(2 * np.pi * 280 * t) + 0.01 * rng.standard_normal(len(t))
    fade = np.minimum(1.0, np.minimum(np.arange(len(t)), np.arange(len(t))[::-1]) / (0.008 * SR))      # real audio is faded, not cut
    return (wave * fade * 32767).astype(np.int16)


def silence(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


def kinds(samples: np.ndarray) -> list[str]:
    return [event["type"] for event in ag.scan(samples)["events"]]


class AudioGlitchTest(unittest.TestCase):
    def test_clean_speech_has_no_events(self):
        self.assertEqual(kinds(speech(3.0)), [])

    def test_sentence_pause_is_counted_not_reported(self):
        result = ag.scan(np.concatenate([speech(1.0), silence(400), speech(1.0)]))
        self.assertEqual(result["events"], [])
        self.assertEqual(result["sentence_pauses"], 1)

    def test_dropout_inside_speech(self):
        result = ag.scan(np.concatenate([speech(1.0), silence(90), speech(1.0, 1)]))
        self.assertEqual([event["type"] for event in result["events"]], ["dropout"])
        self.assertAlmostEqual(result["events"][0]["start_s"], 1.0, delta=0.03)

    def test_a_very_short_stop_is_not_a_dropout(self):
        self.assertEqual(kinds(np.concatenate([speech(1.0), silence(30), speech(1.0, 1)])), [])

    def test_unexpected_long_zero_stretch_is_a_dead_gap_not_a_dropout(self):
        events = ag.scan(np.concatenate([speech(1.0), silence(900), speech(1.0, 1)]))["events"]
        self.assertEqual([event["type"] for event in events], ["dead_gap"])
        self.assertEqual(events[0]["ms"], 900)

    def test_stutter_is_blocks_of_silence_in_a_row(self):
        pieces = [speech(1.0)]
        for index in range(6):
            pieces += [silence(25), speech(0.025, index)]
        pieces.append(speech(1.0, 9))
        self.assertIn("stutter", kinds(np.concatenate(pieces)))

    def test_click_is_an_isolated_high_frequency_burst(self):
        samples = speech(2.0)
        burst = (np.random.default_rng(3).standard_normal(int(SR * 0.003)) * 9000).astype(np.int16)
        samples[SR:SR + len(burst)] = burst
        events = [event for event in ag.scan(samples)["events"] if event["type"] == "click"]
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["start_s"], 1.0, delta=0.02)

    def test_locate_names_the_unit(self):
        events = [{"type": "dropout", "start_s": 15.5}]
        ag.locate(events, [{"group_id": "u03", "start_ms": 27310, "tts_duration_ms": 99440}], 59.39)
        self.assertEqual((events[0]["unit"], events[0]["unit_s"]), ("u03", 47.58))


if __name__ == "__main__":
    unittest.main()
