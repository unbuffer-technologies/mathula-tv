from types import SimpleNamespace

from mathula_tv import omnivoice_duration as od
from mathula_tv.dub_workbench import Workbench
from mathula_tv.submission_rank import BestSubmission, score_key


# -- the measurement says what it means ------------------------------------------------------------------------------------------------------------------------------

def _present(result):
    bench = object.__new__(Workbench)
    bench._episode = lambda: {"speed_policy": "natural"}
    return bench._present(dict(result))


def test_a_unit_past_its_window_but_inside_the_free_silence_is_told_it_is_allowed_and_why():
    """prod-114 turns 9-13: 2440 ms in a 2100 ms window reported fits_naturally true and overrun_pct 0, and the agent could not see why."""
    out = _present({"measured_ms": 2440, "window_ms": 2100, "free_gap_after_ms": 1280, "late_tolerance_ms": 1280, "required_speed_percent": 0.0, "direction": None})
    assert out["fits_naturally"] is True and out["overrun_pct"] == 0.0
    assert "340 ms past its window end into the 1280 ms of free silence" in out["verdict"] and "collides with nothing" in out["verdict"]


def test_a_unit_past_what_the_free_silence_allows_is_told_by_how_much_it_would_collide():
    out = _present({"measured_ms": 3000, "window_ms": 2100, "free_gap_after_ms": 400, "late_tolerance_ms": 400, "required_speed_percent": 14.3, "direction": "compress"})
    assert "runs 900 ms past its window end" in out["verdict"] and "only 400 ms of that is allowed" in out["verdict"] and "overruns by 500 ms" in out["verdict"] and "would collide" in out["verdict"]


def test_a_short_unit_and_a_fitting_unit_are_described_too():
    short = _present({"measured_ms": 1500, "window_ms": 2100, "free_gap_after_ms": 500, "late_tolerance_ms": 500, "required_speed_percent": 0.0, "direction": "expand", "residual_shortfall_pct": 28.6})
    assert "ends 600 ms before its window ends" in short["verdict"] and "28.6% of the window is left as dead air" in short["verdict"]
    fits = _present({"measured_ms": 2000, "window_ms": 2100, "free_gap_after_ms": 500, "late_tolerance_ms": 500, "required_speed_percent": 0.0, "direction": None})
    assert fits["verdict"] == "fits: ends 100 ms inside its window"
    assert "verdict" not in _present({"note": "nothing measured"})


# -- the first-draft length guide follows the voice's real pace -------------------------------------------------------------------------------------------------------

# The natural-pace whole-job run of 2026-11-18 (speed 0.85): unit -> the seconds it was actually spoken in (units of 10 s or more, where the padding is negligible).
SPOKEN_S = {"u01": 39.0, "u04": 33.4, "u05": 21.6, "u06": 69.0, "u07": 15.4, "u08": 50.7, "u09": 26.6, "u10": 22.0, "u12": 33.1, "u13": 109.0, "u16": 90.8, "u17": 105.2, "u18": 33.7,
            "u19": 60.1, "u20": 22.6, "u21": 157.2}


def test_the_length_guide_for_omnivoice_matches_the_characters_that_really_filled_real_windows():
    """Validated on the 2026-11-18 run: the guide (the free estimate inverted) is within 1.3% median of what filled each window; the generic Azure guide of 12.4 characters a second was 18.8% off."""
    import json
    import re
    import statistics
    from pathlib import Path

    service = Path(__file__).resolve().parents[1] / "gcp" / "omnivoice_service"
    units = {u["id"]: u for u in json.loads((service / "units.json").read_text(encoding="utf-8"))["units"]}
    refs = json.loads((service / "refs" / "unit" / "refs_unit.json").read_text(encoding="utf-8"))
    errors, azure = [], []
    for uid, spoken in SPOKEN_S.items():
        unit = units[uid]
        held = sum(int(n) for n in re.findall(r"~pause(\d+)", unit["text_pauses"])) / 1000
        actual = len(unit["text"]) * (unit["window_s"] - held) / (spoken - held)            # characters that would fill the window's speech time at the pace it was spoken
        guide = od.chars_to_fill(unit["window_s"] - held, refs[uid]["text"], refs[uid]["end_s"] - refs[uid]["start_s"], 0.85)
        guide *= od.SPEECH_FACTOR                                                          # that run spoke whole units in one generation: the formula as it was, before the fitted factor of the cut-and-joined dub
        errors.append(abs(guide - actual) / actual * 100)
        azure.append(abs((unit["window_s"] - held) * 12.4 - actual) / actual * 100)
    assert statistics.median(errors) < 3.0 and max(errors) < 10.0
    assert statistics.median(azure) > 3 * statistics.median(errors)                           # and far better than the generic guide it replaces


def test_the_length_guide_carries_the_fitted_share_of_the_speech_of_the_cut_and_joined_dub():
    text, seconds = "So people that are within that same standpoint as ourselves, that's the angle that we're taking.", 5.28
    pace = od.seconds_per_weight(text, seconds)
    assert abs(od.chars_to_fill(30.0, text, seconds, 1.0) - (29.8 / (pace * od.TYPICAL_WEIGHT_PER_CHAR * od.SPEECH_FACTOR))) <= 1          # a window of 30 s, less the 200 ms of padding, at the fitted share


def test_a_slower_reference_means_fewer_characters_fill_a_window_and_a_short_window_loses_the_padding():
    text, seconds = "So people that are within that same standpoint as ourselves, that's the angle that we're taking.", 5.28
    assert od.chars_to_fill(34.0, text, 7.0, 0.85) < od.chars_to_fill(34.0, text, seconds, 0.85)          # more seconds for the same words: fewer characters fit
    assert od.chars_to_fill(1.6, text, seconds, 0.85) < 1.6 / 34.0 * od.chars_to_fill(34.0, text, seconds, 0.85)    # 200 ms of a 1.6 s window is padding, not speech


def test_open_clip_gives_each_segment_its_own_omnivoice_length_and_its_share_of_the_clip():
    bench = object.__new__(Workbench)
    reference = SimpleNamespace(text="So people that are within that same standpoint as ourselves, that's the angle that we're taking.", seconds=5.28)
    bench._tts = SimpleNamespace(speed=0.85, reference_for=lambda group, root: reference)
    bench.pronunciation_root = None
    guide = bench._omnivoice_guide([{"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 10000}, {"segment_id": "s2", "speaker_id": "SPEAKER_01", "start_ms": 10500, "end_ms": 15500}])
    assert set(guide) == {"s1", "s2"} and guide["s1"] > guide["s2"] and all(v % 5 == 0 for v in guide.values())
    assert guide["s1"] == od.chars_to_fill(10.0, reference.text, reference.seconds, 0.85) // 5 * 5 or abs(guide["s1"] - od.chars_to_fill(10.0, reference.text, reference.seconds, 0.85)) <= 2
    azure = object.__new__(Workbench)
    azure._tts = SimpleNamespace()                                                      # a voice with no reference clips: no OmniVoice guide
    assert azure._omnivoice_guide([{"segment_id": "s1", "start_ms": 0, "end_ms": 1000}]) == {}

    def broken():
        raise ValueError("AZURE_SPEECH_KEY is missing")

    unbuilt = object.__new__(Workbench)
    unbuilt._tts, unbuilt._tts_factory = None, broken
    assert unbuilt._omnivoice_guide([{"segment_id": "s1", "start_ms": 0, "end_ms": 1000}]) == {}          # open_clip must not fail because the voice cannot be built yet


# -- timeline_fit works in the agent's own units ------------------------------------------------------------------------------------------------------------------------

def _bench(lengths):
    bench = object.__new__(Workbench)
    bench.segments = [{"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 4000}, {"segment_id": "s2", "speaker_id": "SPEAKER_00", "start_ms": 4500, "end_ms": 8000},
                      {"segment_id": "s3", "speaker_id": "SPEAKER_01", "start_ms": 12200, "end_ms": 15000}]
    bench.voices = {}
    bench._text_ms = lambda text, speaker, start, end: (lengths[text], 10.0)
    return bench


def test_timeline_fit_accepts_a_unit_that_spans_several_segments():
    bench = _bench({"unit one": 9000, "unit two": 2500})
    out = bench.timeline_fit([{"item_id": "u1", "start_ms": 0, "end_ms": 8000, "text": "unit one"}, {"item_id": "u2", "start_ms": 12200, "end_ms": 15000, "text": "unit two", "speaker_id": "SPEAKER_01"}])
    first, second = out["sentences"]
    assert first["segment_id"] == "u1" and first["window_ms"] == 8000 and first["over_window_by_ms"] == 1000 and first["free_gap_after_ms"] == 4200
    assert second["segment_id"] == "u2" and second["starts_ms"] == 12200 and second["late_by_ms"] == 0
    assert out["late_sentences"] == []                                                  # the long unit's overrun lands in free silence before the next speaker


def test_a_unit_that_runs_into_the_next_speakers_start_is_named():
    bench = _bench({"unit one": 13500})
    out = bench.timeline_fit([{"start_ms": 0, "end_ms": 8000, "text": "unit one"}])
    assert out["runs_into_another_speaker"] == ["0-8000"] and out["sentences"][0]["runs_into_next_speaker_ms"] == 1300


def test_timeline_fit_refuses_an_item_with_no_window_and_still_takes_segment_ids():
    import pytest

    bench = _bench({"x": 1000})
    with pytest.raises(ValueError, match="needs a segment_id, or a start_ms and an end_ms"):
        bench.timeline_fit([{"text": "x"}])
    assert bench.timeline_fit([{"segment_id": "s1", "text": "x"}])["sentences"][0]["segment_id"] == "s1"


# -- the run's best submission is recorded, never lost ----------------------------------------------------------------------------------------------------------------------

def _result(gates, cross, collisions, rushed, dead_air):
    return {"accepted": True, "headline": {"gate_failures": gates, "cross_speaker_collisions": cross, "collisions": collisions, "groups_rushed": rushed, "residual_dead_air_pct": dead_air}}


def test_the_ranking_puts_gates_and_collisions_before_dead_air():
    first, second, third = _result(6, 9, 15, 16, 2.8), _result(4, 5, 7, 12, 4.3), _result(1, 1, 2, 6, 6.74)           # prod-114's three submissions
    assert score_key(first) > score_key(second) > score_key(third)                                                      # each is better than the last, although dead air rose
    assert score_key(_result(0, 0, 0, 0, 9.0)) < score_key(_result(1, 0, 0, 0, 0.0))                                   # no gate failure beats less dead air
    assert score_key(None) > score_key(first)                                                                           # no score is the worst


def test_the_best_submission_is_kept_and_the_report_says_whether_the_last_one_is_it():
    best = BestSubmission()
    assert best.observe(5, _result(2, 0, 0, 0, 1.0), [{"item_id": "a"}]) is True
    assert best.observe(8, _result(0, 0, 1, 2, 3.0), [{"item_id": "b"}]) is True                                        # fewer gate failures: better
    assert best.observe(11, _result(0, 0, 1, 2, 3.0), [{"item_id": "c"}]) is True                                       # a tie goes to the later one
    assert best.report()["final_is_best"] is True and best.best["turn"] == 11
    assert best.observe(14, _result(3, 0, 0, 0, 0.5), [{"item_id": "d"}]) is False                                      # a regression: not the best
    report = best.report()
    assert report["final_is_best"] is False and report["best_turn"] == 11 and report["final_turn"] == 14 and report["submissions"] == 4
    assert best.best["items"] == [{"item_id": "c"}]                                                                      # the best one's units are still there to restore
    again = BestSubmission()
    again.load(best.to_dict())
    assert again.report() == report
    assert BestSubmission().observe(1, {"accepted": False}, []) is False and BestSubmission().report() == {"submissions": 0}
