import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from mathula_tv import delivery

RATE = 16000


def _tone(hz, seconds, level=0.3):
    t = np.arange(int(RATE * seconds)) / RATE
    # a few harmonics, like a voice
    return level * sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3))


def _write(path, samples):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def _words(texts, start, gap=0.05, length=0.3):
    out, t = [], start
    for text in texts:
        out.append({"text": text, "start": t, "end": t + length})
        t += length + gap
    return out


class TrackTest(unittest.TestCase):
    def test_pitch_of_a_steady_tone_is_found_in_semitones_above_100_hz(self):
        samples = np.concatenate([_tone(120, 1.0), np.zeros(RATE // 2, dtype=np.float32) + 1e-4, _tone(200, 1.0)]).astype(np.float32)
        semitones, loudness = delivery.track(samples, RATE)
        low = float(np.nanmedian(semitones[20:90]))
        high = float(np.nanmedian(semitones[170:240]))
        self.assertAlmostEqual(low, 12 * np.log2(1.2), delta=0.3)
        self.assertAlmostEqual(high, 12 * np.log2(2.0), delta=0.3)
        self.assertGreater(loudness[50], loudness[120] + 20)          # the quiet gap is far below the tone

    def test_noise_has_no_pitch(self):
        noise = (np.random.default_rng(1).normal(0, 0.1, RATE * 2)).astype(np.float32)
        semitones, _ = delivery.track(noise, RATE)
        self.assertLess(float(np.mean(~np.isnan(semitones))), 0.15)


class MeasurementTest(unittest.TestCase):
    def test_pauses_pace_and_repeats(self):
        words = _words(["we", "will", "not", "stop"], 0.0) + _words(["we", "will", "not", "stop"], 3.0)
        pauses = delivery._pauses(words)
        self.assertEqual([p["after"] for p in pauses], ["stop"])
        self.assertGreater(pauses[0]["ms"], 1000)
        self.assertGreater(delivery._pace(words), 2.0)
        self.assertEqual(delivery.repeated_phrases("We will not stop. We will not stop. Never."), ["we will not stop"])
        self.assertEqual(delivery.repeated_phrases("of the people"), [])

    def test_a_faster_louder_higher_segment_reads_that_way_against_the_speakers_own_usual(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "a.wav"
            parts = [_tone(110, 3.0, 0.1), np.zeros(8000), _tone(110, 3.0, 0.1), np.zeros(8000), _tone(150, 3.0, 0.4)]
            _write(audio, np.concatenate(parts).astype(np.float32))
            segments = [{"segment_id": "s1", "speaker_id": "A", "start_ms": 0, "end_ms": 3000, "restored_text": "one two three"},
                        {"segment_id": "s2", "speaker_id": "A", "start_ms": 3500, "end_ms": 6500, "restored_text": "four five six"},
                        {"segment_id": "s3", "speaker_id": "A", "start_ms": 7000, "end_ms": 10000, "restored_text": "seven eight nine"}]
            words = (_words(["one", "two", "three", "four", "five", "six"], 0.0, 0.2, 0.3)[:3]
                     + _words(["four", "five", "six"], 3.5, 0.2, 0.3)
                     + _words(["seven", "eight", "nine", "ten", "elev"], 7.0, 0.02, 0.2))
            sheet = delivery.build_sheet(audio, segments, words)
        fast = sheet["segments"]["s3"]
        self.assertGreater(fast["pitch_vs_usual_st"], 3)
        self.assertGreater(fast["loudness_vs_usual_db"], 6)
        self.assertGreater(fast["pace_vs_usual_pct"], 10)
        self.assertAlmostEqual(sheet["segments"]["s1"]["pitch_vs_usual_st"], 0, delta=0.5)
        self.assertIn("A", sheet["speakers"])


class ForTheAgentAndRendererTest(unittest.TestCase):
    SHEET = {
        "schema_version": delivery.SCHEMA_VERSION, "speakers": {"A": {"pace_wps": {"usual": 3.0}}},
        "segments": {
            "s1": {"speaker_id": "A", "start_ms": 0, "end_ms": 4000, "words": 10, "pitch_vs_usual_st": 5.0, "loudness_vs_usual_db": 9.0,
                   "pauses": [{"ms": 800, "after": "money", "before": "and"}], "stressed_words": ["grant"], "repeats": []},
            "s2": {"speaker_id": "A", "start_ms": 4500, "end_ms": 6500, "words": 5, "pitch_vs_usual_st": -1.0, "loudness_vs_usual_db": -1.0,
                   "pauses": [], "stressed_words": [], "repeats": []},
            "s3": {"speaker_id": "A", "start_ms": 7000, "end_ms": 8000, "words": 3, "pitch_vs_usual_st": None, "loudness_vs_usual_db": None},
        },
    }

    def test_agent_view_is_compact_and_scoped(self):
        view = delivery.for_agent(self.SHEET, ["s1", "s3"])
        self.assertEqual([r["segment_id"] for r in view["segments"]], ["s1", "s3"])
        self.assertEqual(view["segments"][0]["pauses"], [{"ms": 800, "after": "money"}])
        self.assertNotIn("repeats", view["segments"][0])
        self.assertIn("recognise", view["note"])

    def test_unit_prosody_is_a_capped_nudge_weighted_by_segment_length(self):
        units = [{"item_id": "u1", "source_segment_ids": ["s1"]}, {"item_id": "u2", "source_segment_ids": ["s2"]},
                 {"item_id": "u3", "source_segment_ids": ["s3"]}, {"item_id": "u4", "source_segment_ids": ["s1", "s2"]}]
        result = delivery.unit_prosody(self.SHEET, units)
        self.assertEqual(result["u1"]["pitch_percent"], delivery.MAX_PITCH_PERCENT)         # +5 st would be +10 %: capped
        self.assertEqual(result["u1"]["volume_percent"], delivery.MAX_VOLUME_PERCENT)
        self.assertEqual(result["u2"]["pitch_percent"], -2)
        self.assertEqual(result["u2"]["volume_percent"], 0)                                    # -1 dB * 0.5 rounds to 0 (banker's rounding)
        self.assertEqual(result["u3"]["pitch_percent"], 0)                                     # nothing measured: no change
        self.assertTrue(-delivery.MAX_PITCH_PERCENT <= result["u4"]["pitch_percent"] <= delivery.MAX_PITCH_PERCENT)

    def test_write_and_read_back_and_off_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "native_dub"
            native.mkdir()
            units = [{"item_id": "u1", "source_segment_ids": ["s1"]}]
            delivery.write_unit_prosody(native, self.SHEET, units)
            self.assertEqual(delivery.read_unit_prosody(Path(directory), "u1"), (4, 2))
            self.assertEqual(delivery.read_unit_prosody(Path(directory), "missing"), (0, 0))
            self.assertEqual(delivery.read_unit_prosody(None, "u1"), (0, 0))
            delivery.write_unit_prosody(native, self.SHEET, units, enabled=False)
            self.assertEqual(delivery.read_unit_prosody(Path(directory), "u1"), (0, 0))
            self.assertFalse((native / delivery.PROSODY_NAME).exists())

    def test_load_sheet_rejects_foreign_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.json"
            path.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")
            self.assertIsNone(delivery.load_sheet(path))
            self.assertIsNone(delivery.load_sheet(Path(directory) / "nope.json"))


if __name__ == "__main__":
    unittest.main()
