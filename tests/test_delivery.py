import os
import json
import tempfile
import unittest
from unittest import mock
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
            room = (np.random.default_rng(2).normal(0, 0.001, 24000)).astype(np.float32)      # a real recording is never digitally silent
            parts = [_tone(110, 3.0, 0.1), room, _tone(110, 3.0, 0.1), room, _tone(150, 3.0, 0.4)]
            _write(audio, np.concatenate(parts).astype(np.float32))
            segments = [{"segment_id": "s1", "speaker_id": "A", "start_ms": 0, "end_ms": 3000, "restored_text": "one two three"},
                        {"segment_id": "s2", "speaker_id": "A", "start_ms": 4500, "end_ms": 7500, "restored_text": "four five six"},
                        {"segment_id": "s3", "speaker_id": "A", "start_ms": 9000, "end_ms": 12000, "restored_text": "seven eight nine"}]
            words = (_words(["one", "two", "three"], 0.0, 0.2, 0.3)
                     + _words(["four", "five", "six"], 4.5, 0.2, 0.3)
                     + _words(["seven", "eight", "nine", "ten", "elev"], 9.0, 0.02, 0.2))
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
        self.assertEqual(view["segments"][0]["pauses_inside"], ["800 ms after 'money'"])
        self.assertEqual(view["segments"][0]["loudness_vs_usual_db"], 9.0)
        self.assertNotIn("repeats", view["segments"][0])
        self.assertNotIn("loudness_vs_usual_db", delivery.for_agent(self.SHEET, ["s2"])["segments"][0])
        self.assertIn("recognise", view["note"])

    def test_the_listener_view_has_the_breaths_and_whether_the_speaker_stopped_after_each_figure(self):
        sheet = {"schema_version": delivery.SCHEMA_VERSION, "speakers": {}, "segments": {"s1": {
            "speaker_id": "A", "start_ms": 0, "end_ms": 4000, "words": 10, "pauses": [],
            "breaths": [{"ms": 240, "after": "elections", "before": "has"}, {"ms": 180, "after": "margin", "before": "has"}],
            "figures": [{"figure": "31%", "stop_after_ms": 0, "next": "in"}, {"figure": "11", "stop_after_ms": None, "next": None}]}}}
        row = delivery.for_agent(sheet)["segments"][0]
        self.assertEqual(row["breaths_inside"], ["240 ms after 'elections'", "180 ms after 'margin'"])
        self.assertEqual(row["figures_spoken"], ["31%: 0 ms stop, then 'in'", "11: ends the segment"])
        self.assertIn("LISTENER", delivery.AGENT_NOTE)

    def test_short_stops_are_breaths_and_the_time_a_speaker_takes_to_say_a_number_is_not_a_pause(self):
        words = [{"text": "from", "start": 0.0, "end": 0.2}, {"text": "75.8%", "start": 1.8, "end": 1.84},      # 1.6 s of speech the recogniser put in the gap
                 {"text": "in", "start": 1.9, "end": 2.0}, {"text": "2006", "start": 2.2, "end": 2.5},
                 {"text": "margin", "start": 2.7, "end": 3.0}, {"text": "has", "start": 3.2, "end": 3.4}, {"text": "fallen", "start": 4.0, "end": 4.3}]
        self.assertEqual([p["after"] for p in delivery._pauses(words)], ["has"])                                  # 0.6 s: a pause; the gap before 75.8% is a number being said
        self.assertEqual([b["after"] for b in delivery._breaths(words)], ["2006", "margin"])                    # 0.2 s stops: breaths (the 0.2 s before 2006 is a figure being said)
        self.assertEqual([f["figure"] for f in delivery._figures(words)], ["75.8%", "2006"])
        self.assertEqual(delivery._figures(words)[0]["stop_after_ms"], 60)

    def test_pause_marker_becomes_a_break_between_the_pieces_of_speech(self):
        # ~pauseN is the TOTAL silence at the spot: the 0.92 s a full stop leaves (and the 0.11 s every break measures over) counts toward it
        self.assertEqual(delivery.split_pauses("Imali yethu. ~pause1500 Uma uya"), ["Imali yethu.", 1500 - 920 - 110, "Uma uya"])
        self.assertEqual(delivery.split_pauses("Imali yethu. ~pause900 Uma uya"), ["Imali yethu. Uma uya"])        # the stop alone is already longer than 0.9 s: no break, one text
        self.assertEqual(delivery.split_pauses("Sawubona"), ["Sawubona"])
        self.assertEqual(delivery.split_pauses("A ~pause5000 B"), ["A", 2000 - 110, "B"])            # clamped to the longest allowed pause
        self.assertEqual(delivery.split_pauses("~pause800 A"), ["A"])                                # none before the first word
        self.assertEqual(delivery.split_pauses("A ~pause800"), ["A"])                                # none after the last: it would be trimmed
        self.assertEqual(delivery.split_pauses("A ~pause800 ~pause800 B"), ["A", 800 - 110, "B"])    # none doubled
        long = delivery.split_pauses(" ".join(f"w{i} ~pause2000" for i in range(6)) + " end")
        self.assertLessEqual(sum(p for p in long if isinstance(p, int)), delivery.MAX_MARKED_PAUSE_TOTAL_MS)
        self.assertEqual(delivery.strip_pauses("Imali yethu. ~pause900 Uma uya"), "Imali yethu. Uma uya")

    def test_load_sheet_rejects_foreign_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.json"
            path.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")
            self.assertIsNone(delivery.load_sheet(path))
            self.assertIsNone(delivery.load_sheet(Path(directory) / "nope.json"))


if __name__ == "__main__":
    unittest.main()


class VoicePitchTest(unittest.TestCase):
    def test_persona_pitch_is_on_unless_switched_off_and_bounded(self):
        from mathula_tv.native_dub import _voice_pitch_percent

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATHULA_PERSONA_PITCH", None)
            self.assertEqual(_voice_pitch_percent({"pitch_percent": -9}), -9)
            self.assertEqual(_voice_pitch_percent({"pitch_percent": 40}), 10)
        with mock.patch.dict(os.environ, {"MATHULA_PERSONA_PITCH": "0"}):
            self.assertEqual(_voice_pitch_percent({"pitch_percent": -6}), 0)
