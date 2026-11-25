import json
from types import SimpleNamespace

from mathula_tv import duration_model as dm

VOICE = "zu-ZA-ThembaNeural"


def _text(i: int) -> str:
    words = ["ngiyabonga", "abantu", "ukuthi", "kuhle", "sizobona", "umbuso", "namuhla"]
    return " ".join(words[(i + k) % len(words)] for k in range(2 + i % 9)) + f" sa{i}" + ("." if i % 3 else ", ~pause400")      # unique: the ledger keeps one row per text


def _true_ms(text: str) -> float:
    f = dm.features(text)
    return 70.0 * f[0] + 45.0 * f[1] + 20.0 * f[2] + f[7] + 150.0                           # a voice whose duration the features can explain exactly


def _ledger(work, count: int = 120, voice: str = VOICE) -> None:
    for i in range(count):
        dm.record(work, voice=voice, text=_text(i), measured_ms=_true_ms(_text(i)))


def test_pause_markers_are_counted_as_silence_apart_from_the_speech():
    with_pause = dm.features("Sawubona ~pause800 baba.")
    plain = dm.features("Sawubona baba.")
    assert with_pause[dm.FEATURES.index("pause_ms")] == 800 and plain[dm.FEATURES.index("pause_ms")] == 0
    assert with_pause[dm.FEATURES.index("vowels")] == plain[dm.FEATURES.index("vowels")]          # the marker's letters are not speech


def test_the_model_learns_a_voices_timing_from_the_ledger_and_says_how_far_it_can_be_trusted(tmp_path):
    _ledger(tmp_path)
    model = dm.refit(tmp_path)
    assert VOICE in model["voices"] and model["voices"][VOICE]["rows"] == 120
    text = "ngiyabonga abantu ukuthi kuhle."
    found = dm.predict(model, voice=VOICE, text=text)
    assert abs(found["predicted_ms"] - _true_ms(text)) / _true_ms(text) < 0.02
    assert found["basis"] == VOICE and found["band_pct"] < 3 and found["fitted_on"] == 120


def test_a_voice_with_too_little_data_uses_the_pooled_model_and_a_rate_setting_is_taken_out(tmp_path):
    _ledger(tmp_path, 120, VOICE)
    _ledger(tmp_path, 5, "zu-ZA-ThandoNeural")
    model = dm.refit(tmp_path)
    assert "zu-ZA-ThandoNeural" not in model["voices"]
    found = dm.predict(model, voice="zu-ZA-ThandoNeural", text="ngiyabonga abantu.")
    assert found["basis"] == "pooled"
    base = dm.predict(model, voice=VOICE, text="ngiyabonga abantu.")["predicted_ms"]
    faster = dm.predict(model, voice=VOICE, text="ngiyabonga abantu.", rate_pct=10)["predicted_ms"]
    assert abs(faster - base / 1.10) <= 1
    assert dm.predict({}, voice=VOICE, text="x") is None


def test_the_ledger_keeps_the_latest_duration_of_a_text_and_the_model_refits_when_it_has_grown(tmp_path):
    dm.record(tmp_path, voice=VOICE, text="a", measured_ms=500)
    dm.record(tmp_path, voice=VOICE, text="a", measured_ms=700)
    assert [r["measured_ms"] for r in dm._read_ledger(tmp_path)] == [700]
    _ledger(tmp_path, 80)
    first = dm.load(tmp_path)
    assert first["ledger_rows"] >= 80
    (tmp_path / dm.MODEL).write_text(json.dumps({**first, "marker": 1}), encoding="utf-8")
    assert dm.load(tmp_path).get("marker") == 1                                                    # nothing new: the saved model is read as it is
    for i in range(500, 500 + dm.REFIT_AFTER_NEW_ROWS):
        dm.record(tmp_path, voice=VOICE, text=_text(i) + str(i), measured_ms=_true_ms(_text(i)))
    assert "marker" not in dm.load(tmp_path)                                                       # the ledger grew enough: refitted


def test_bootstrap_takes_the_accepted_dubs_with_their_speakers_voice(tmp_path):
    clip = tmp_path / "experience" / "corpus" / "c1"
    clip.mkdir(parents=True)
    (clip / "voice_resolution.json").write_text(json.dumps({"assignments": {"SPEAKER_00": {"selected_voice": VOICE, "base_prosody": {"rate_percent": 0}}}}), encoding="utf-8")
    run = tmp_path / "experience" / "runs" / "prod-1" / "c1" / "r1"
    run.mkdir(parents=True)
    groups = [{"speaker_id": "SPEAKER_00", "spoken_text": _text(i), "measured_ms": int(_true_ms(_text(i)))} for i in range(40)]
    groups.append({"speaker_id": "SPEAKER_00", "spoken_text": "x", "measured_ms": 900, "collides_with": ["u2"]})        # a unit that collides was cut short: not a natural duration
    (run / "scorecard.json").write_text(json.dumps({"groups": groups}), encoding="utf-8")
    assert dm.bootstrap(tmp_path) == 40
    assert all(r["voice"] == VOICE for r in dm._read_ledger(tmp_path))


def _bench(tmp_path):
    from mathula_tv.dub_workbench import Workbench

    _ledger(tmp_path)
    bench = object.__new__(Workbench)
    bench.store = SimpleNamespace(root=tmp_path / "experience")
    bench._tts = SimpleNamespace(default_voice=VOICE)
    bench.voices = {"SPEAKER_00": {"selected_voice": VOICE, "base_prosody": {"rate_percent": 0}}}
    bench.segments = []
    return bench


def test_the_agent_estimates_candidates_for_free_and_is_told_which_are_far_from_the_window(tmp_path):
    bench = _bench(tmp_path)
    near = "ngiyabonga abantu ukuthi kuhle sizobona umbuso namuhla."
    window = int(_true_ms(near))
    out = bench.estimate_fit([{"text": near, "start_ms": 0, "end_ms": window, "speaker_id": "SPEAKER_00"},
                              {"text": near * 3, "start_ms": 0, "end_ms": window, "speaker_id": "SPEAKER_00"}])
    first, second = out["results"]
    assert abs(first["estimated_gap_pct"]) < 3 and "worth a measurement" in first["reading"]
    assert second["estimated_gap_pct"] > 100 and "over" in second["reading"]
    assert "not a measurement" in out["note"]


def test_a_real_measurement_is_recorded_and_the_agent_sees_how_its_estimate_did(tmp_path):
    bench = _bench(tmp_path)
    text = "ngiyabonga abantu ukuthi kuhle."
    result = {"measured_ms": int(_true_ms(text) * 1.10)}                                          # the voice took 10% longer than the model said
    bench._learn_duration(text, bench.voices["SPEAKER_00"], result)
    assert result["estimate_off_pct"] < -8 and result["estimated_ms"] > 0
    assert any(r["text"] == text and r["source"] == "measure" for r in dm._read_ledger(tmp_path))
    for i in range(3):                                                                              # three measurements in this run: its estimates lean on them
        bench._learn_duration(_text(i + 200), bench.voices["SPEAKER_00"], {"measured_ms": int(_true_ms(_text(i + 200)) * 1.10)})
    corrected = bench._estimate_ms(bench.voices["SPEAKER_00"], text)
    assert corrected["run_correction"] > 1.05 and corrected["predicted_ms"] > int(_true_ms(text) * 1.05)


def test_the_agent_has_the_tool_and_is_told_to_use_it_before_it_measures():
    from mathula_tv import dub_agent

    names = {t["function"]["name"] for t in dub_agent.TOOL_SPECS}
    assert "estimate_fit" in names
    assert "estimate_fit (free)" in dub_agent.SYSTEM_PROMPT
    dispatch = dub_agent.build_dispatch(SimpleNamespace(**{n: (lambda *a, **k: None) for n in ("open_clip", "recall_lessons", "measure_tts", "measure_tts_batch", "check_text",
                                                                                                   "preview_turn_padding", "submit_dub", "get_submission", "estimate_fit")}),
                                         clip_id="clipa", run_id="r", tts_budget=5)
    assert "estimate_fit" in dispatch
