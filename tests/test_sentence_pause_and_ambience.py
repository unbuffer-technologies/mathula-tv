import os
import unittest
from pathlib import Path
from unittest import mock

from mathula_tv.mixing import ambience_gaps
from mathula_tv.tts_ssml import (
    SafeSSMLBuilder,
    SSMLValidationError,
    build_group_bookmark_ssml,
    sentence_pause_ms,
    validate_ssml,
    SSMLBounds,
)

VOICE = "zu-ZA-ThembaNeural"


def _builder():
    return SafeSSMLBuilder(allowed_voices=[VOICE])


class SentencePauseTest(unittest.TestCase):
    def test_default_sets_the_sentence_pause_in_both_builders(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATHULA_SENTENCE_PAUSE_MS", None)
            document = _builder().build_plain_text("Sawubona. Unjani?", voice=VOICE, pitch_percent=-6)
            xml, _ = build_group_bookmark_ssml([("u1", "Sawubona. Unjani?")], voice=VOICE, allowed_voices=[VOICE])
        for text in (document.xml, xml):
            self.assertIn('<mstts:silence type="Sentenceboundary-exact" value="400ms"/>', text)
            self.assertLess(text.index("mstts:silence"), text.index("<prosody"))

    def test_zero_keeps_the_voices_own_pause_and_the_old_document(self):
        with mock.patch.dict(os.environ, {"MATHULA_SENTENCE_PAUSE_MS": "0"}):
            document = _builder().build_plain_text("Sawubona.", voice=VOICE)
        self.assertNotIn("mstts", document.xml)

    def test_value_is_clamped_and_bad_values_fall_back(self):
        with mock.patch.dict(os.environ, {"MATHULA_SENTENCE_PAUSE_MS": "99999"}):
            self.assertEqual(sentence_pause_ms(), 1000)
        with mock.patch.dict(os.environ, {"MATHULA_SENTENCE_PAUSE_MS": "abc"}):
            self.assertEqual(sentence_pause_ms(), 400)

    def test_validator_rejects_other_silence_types_and_misplaced_silence(self):
        bounds = SSMLBounds()
        good = _builder().build_plain_text("Sawubona.", voice=VOICE).xml
        validate_ssml(good, bounds=bounds, allowed_voices=[VOICE])
        for broken in (
            good.replace("Sentenceboundary-exact", "Comma-exact"),
            good.replace('value="400ms"', 'value="5000ms"'),
            good.replace("Sawubona.", 'Sawubona.<mstts:silence type="Sentenceboundary-exact" value="400ms"/>'),
        ):
            with self.assertRaises(SSMLValidationError):
                validate_ssml(broken, bounds=bounds, allowed_voices=[VOICE])


class AmbienceGapsTest(unittest.TestCase):
    def test_only_long_gaps_between_units_with_a_margin_each_side(self):
        windows = [(0, 10_000), (10_800, 20_000), (23_450, 30_000), (30_000, 40_000)]
        self.assertEqual(ambience_gaps(windows), [(20_250, 23_200)])

    def test_gap_after_an_overrunning_unit_is_measured_from_where_it_ends(self):
        windows = [(0, 10_000), (10_500, 24_000), (25_000, 30_000), (30_500, 31_000)]
        self.assertEqual(ambience_gaps(windows, min_gap_ms=1500), [])

    def test_before_first_and_after_last_unit_is_left_alone(self):
        self.assertEqual(ambience_gaps([(8_000, 9_000)]), [])
        self.assertEqual(ambience_gaps([]), [])


class GapSpeechTest(unittest.TestCase):
    def _run(self, answers):
        from mathula_tv import native_dub

        class Backend:
            def __init__(self):
                self.calls = 0

            def transcribe(self, path, locale):
                result = answers[self.calls]
                self.calls += 1
                if isinstance(result, Exception):
                    raise result
                return result

        messages = []
        with mock.patch("mathula_tv.cli.create_speech_backend", return_value=Backend()), \
                mock.patch("mathula_tv.config.load_settings", return_value=object()), \
                mock.patch("mathula_tv.azure_stt.normalize", side_effect=lambda raw: raw), \
                mock.patch.object(native_dub.subprocess, "run"):
            kept = native_dub._gaps_without_speech(
                Path("source.mp4"), [(1000, 3000), (5000, 8000), (10000, 12000)], messages.append)
        return kept, messages

    def test_a_gap_with_speech_or_an_unreadable_check_gets_no_source_sound(self):
        kept, messages = self._run([
            {"words": []},                                            # applause: kept
            {"words": [{"text": "Police"}, {"text": "station"}]},     # English speech: left silent
            RuntimeError("network down"),                             # could not check: left silent
        ])
        self.assertEqual(kept, [(1000, 3000)])
        self.assertTrue(any("Police station" in m for m in messages))

    def test_azures_empty_transcription_means_nothing_was_heard(self):
        kept, _ = self._run([RuntimeError("Azure Speech returned an empty transcription")] * 3)
        self.assertEqual(kept, [(1000, 3000), (5000, 8000), (10000, 12000)])


if __name__ == "__main__":
    unittest.main()
