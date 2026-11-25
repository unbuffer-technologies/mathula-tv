import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from mathula_tv import omnivoice_tts as ot


def _words():
    """Two people under two raw labels: A speaks 0-10 s (two sentences), B speaks 12-20 s."""
    words = []
    for i in range(10):                                                   # A: "w0 w1 w2 w3 w4." (0-5 s)  then "w5 ... w9." (5-10 s)
        words.append({"text": f"w{i}" + ("." if i in (4, 9) else ""), "start": i * 1.0, "end": i * 1.0 + 0.9, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False})
    for i in range(8):                                                    # B: 12-20 s, one sentence of 8 words
        words.append({"text": f"b{i}" + ("." if i == 7 else ""), "start": 12.0 + i, "end": 12.0 + i + 0.9, "raw_speaker": "AZURE_2", "confidence": 0.8, "overlap": False})
    return words


def _audio(tmp_path):
    rate = 48000
    t = np.arange(rate * 25) / rate
    left = 0.2 * np.sin(2 * np.pi * 200 * t)
    path = tmp_path / "mix.wav"
    sf.write(str(path), np.stack([left, left], axis=1).astype("float32"), rate, subtype="PCM_16")
    return path


def _bank(tmp_path):
    return ot.ReferenceBank(_words(), {"AZURE_1": "SPEAKER_00", "AZURE_2": "SPEAKER_01"}, _audio(tmp_path), cache_dir=tmp_path / "cache")


def test_reference_clips_are_sentence_bounded_runs_of_one_speaker(tmp_path):
    bank = _bank(tmp_path)
    spans = [(round(c["start"]), round(c["end"]), c["speaker"]) for c in bank._candidates]
    assert (0, 5, "SPEAKER_00") in spans and (5, 10, "SPEAKER_00") in spans and (12, 20, "SPEAKER_01") in spans


def test_the_clip_comes_from_inside_the_units_own_window_so_a_merged_label_cannot_give_the_wrong_voice(tmp_path):
    bank = _bank(tmp_path)
    inside_b = bank.choose("SPEAKER_00", 11500, 21000)            # the unit's label says SPEAKER_00 but the person talking in this window is the other one
    assert inside_b["speaker"] == "SPEAKER_01"
    assert bank.choose("SPEAKER_00", 0, 5200)["start"] == 0.0


def test_a_window_too_short_for_a_clip_borrows_the_closest_clip_of_the_same_speaker(tmp_path):
    bank = _bank(tmp_path)
    near = bank.choose("SPEAKER_00", 10500, 11800)                # nothing inside 10.5-11.8 s: the closest SPEAKER_00 clip is the second one (5-10 s)
    assert near["speaker"] == "SPEAKER_00" and near["start"] == 5.0
    assert bank.choose("SPEAKER_01", 10500, 11800)["start"] == 12.0


def test_a_picked_reference_is_a_24k_mono_wav_of_the_clip_and_is_cached(tmp_path):
    bank = _bank(tmp_path)
    ref = bank.pick("SPEAKER_01", 12000, 20000)
    data, rate = sf.read(io.BytesIO(ref.wav))
    assert rate == 24000 and data.ndim == 1 and abs(len(data) / rate - ref.seconds) < 0.05 and ref.text.startswith("b0 b1")
    assert any((tmp_path / "cache").glob("*.wav")) and bank.pick("SPEAKER_01", 12000, 20000).wav == ref.wav


def test_dictionary_parts_become_text_and_silence():
    from mathula_tv.tts_ssml import BreakPart, CharacterPart, EmphasisPart, SubstitutionPart, TextPart

    flat = ot.flatten_parts([TextPart("Sawubona"), BreakPart(400), SubstitutionPart("MK", "emm kay"), EmphasisPart("kakhulu"), CharacterPart("ANC")])
    assert flat == ["Sawubona", 400, "emm kay", "kakhulu", "ANC"]


def test_each_sentence_is_its_own_span_and_a_pause_cuts_as_well():
    spans, gaps = ot.to_spans(["Ngiyabonga kakhulu baba. Sawubona baba wami.", 800, "Siyakwemukela sonke."])
    assert spans == ["Ngiyabonga kakhulu baba.", "Sawubona baba wami.", "Siyakwemukela sonke."] and gaps == [ot.SPAN_GAP_MS, 800]          # spoken apart: the voice speeds up when it says two together
    spans, gaps = ot.to_spans(["Yebo. Siyakwemukela sonke lapha namhlanje."])
    assert spans == ["Yebo. Siyakwemukela sonke lapha namhlanje."] and gaps == []                      # a sentence of a word or two is kept with the next
    long_text = " ".join(f"Isivakashi sesibili {i}." for i in range(40))
    spans, gaps = ot.to_spans([long_text])
    assert len(spans) == 40 and gaps == [ot.SPAN_GAP_MS] * 39
    assert ot.to_spans([500, "Sawubona."]) == (["Sawubona."], [])                        # a silence before any speech is dropped
    assert ot.to_spans([]) == ([], [])


class FakeClient:
    def __init__(self):
        self.calls = []

    def synthesize(self, *, spans, gaps_ms, reference, durations_s=None, speed=1.0, lora=True, **_ignored):
        self.calls.append({"spans": list(spans), "gaps": list(gaps_ms), "ref": reference, "speed": speed})
        wav = io.BytesIO()
        sf.write(wav, np.zeros(int(24000 * (1.5 * len(spans)))), 24000, format="WAV", subtype="PCM_16")
        return {"audio_b64": base64.b64encode(wav.getvalue()).decode(), "span_starts_s": [0.0], "span_ms": [1500], "gpu_s": 0.5}


def _backend(tmp_path):
    return ot.OmniVoiceTTSBackend(FakeClient(), speed=0.85, banks={"job": _bank(tmp_path)})


def test_a_group_is_spoken_through_the_service_with_its_own_reference_and_the_result_reads_like_azures(tmp_path):
    backend = _backend(tmp_path)
    group = {"group_id": "measure", "speaker_id": "SPEAKER_00", "start_ms": 0, "source_end_ms": 5200, "source_span_ms": 5200,
             "parts": [{"type": "text", "segment_id": "s", "text": "Sawubona baba. Siyabonga kakhulu."}]}
    out = tmp_path / "out" / "r0_measure_abc.wav"
    result = backend.synthesize_group(group=group, voice_info={"selected_voice": "zu-ZA-ThembaNeural"}, output_path=out, preferred_ms=5000, maximum_ms=5500, force=False, job_root="job")
    call = backend.client.calls[0]
    assert call["ref"].start_s == 0.0 and call["speed"] == 0.85 and call["spans"] == ["Sawubona baba. Siyabonga kakhulu."]
    assert out.is_file() and result.duration_ms == 1500 and result.backend == "omnivoice" and result.voice == "zu-ZA-ThembaNeural" and not result.idempotent_reuse
    again = backend.synthesize_group(group=group, voice_info={"selected_voice": "zu-ZA-ThembaNeural"}, output_path=out, preferred_ms=5000, maximum_ms=5500, force=False, job_root="job")
    assert again.idempotent_reuse and len(backend.client.calls) == 1                 # the audio that was measured is reused, not generated again
    backend.synthesize_group(group={**group, "parts": [{"type": "text", "segment_id": "s", "text": "Kunye."}]}, voice_info={"selected_voice": "zu-ZA-ThembaNeural"}, output_path=out,
                             preferred_ms=5000, maximum_ms=5500, force=False, job_root="job")
    assert len(backend.client.calls) == 2                                            # a changed text at the same path is regenerated


def test_a_held_pause_in_the_text_becomes_silence_between_two_spans_and_the_free_estimate_needs_no_service(tmp_path):
    backend = _backend(tmp_path)
    group = {"group_id": "m", "speaker_id": "SPEAKER_01", "start_ms": 12000, "source_end_ms": 20000, "source_span_ms": 8000,
             "parts": [{"type": "text", "segment_id": "s", "text": "Sawubona. ~pause900 Siyabonga."}]}
    spans, gaps = backend.plan(group, {"selected_voice": "x"}, "job")
    assert gaps == [700] and len(spans) == 2                                         # ~pause900 is 900 ms in all; the model keeps 200 ms of it at the two edges
    assert ot.split_marked_pauses("a ~pause150 b") == ["a", "b"]                      # shorter than the least held pause: left to the model's own phrasing
    capped = ot.split_marked_pauses("Uma ~pause2000 kakhulu ~pause2000 futhi ~pause2000 kanjalo ~pause2000 ngoba")
    assert [p for p in capped if isinstance(p, int)] == [1800, 1800, 700]            # 4500 ms in all: the fourth marker has nothing left
    seconds = backend.estimate_seconds(group, {"selected_voice": "x"}, "job")
    assert seconds > 0.9 and backend.client.calls == []


def test_the_agent_estimates_with_the_omnivoice_formula_and_sees_it_beside_a_real_measurement():
    from types import SimpleNamespace

    from mathula_tv.dub_workbench import Workbench

    bench = object.__new__(Workbench)
    bench._tts = SimpleNamespace(default_voice="v", estimate_seconds=lambda group, info, root: 4.0)
    bench.voices = {"SPEAKER_00": {"selected_voice": "zu-ZA-ThembaNeural", "base_prosody": {}}}
    bench.segments = []
    bench.pronunciation_root = None
    out = bench.estimate_fit([{"text": "Sawubona", "start_ms": 0, "end_ms": 4100, "speaker_id": "SPEAKER_00"},
                              {"text": "Sawubona", "start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_00"}])
    near, far = out["results"]
    assert near["estimated_ms"] == 4000 and near["band_pct"] == Workbench.OMNIVOICE_BAND_PCT and "worth a measurement" in near["reading"]
    assert far["estimated_gap_pct"] < -30 and "short of" in far["reading"]
    measured = {"measured_ms": 4200}
    bench._show_omnivoice_estimate("Sawubona", "SPEAKER_00", 0, 4100, measured)
    assert measured["estimated_ms"] == 4000 and measured["estimate_off_pct"] == -4.8
    assert bench._learn_duration("Sawubona", None, {"measured_ms": 4200}) is None               # the Azure-fitted model is left alone when the voice is OmniVoice


def test_the_estimate_band_is_learned_from_real_pairs_and_the_constant_is_only_the_prior(tmp_path):
    from mathula_tv.dub_workbench import Workbench

    bench = object.__new__(Workbench)
    bench.store = SimpleNamespace(root=tmp_path / "experience")
    prior = Workbench.OMNIVOICE_BAND_PCT
    assert bench._omnivoice_band_pct() == prior                                            # no pairs recorded yet
    for i in range(10):
        bench._record_estimate_pair(1000 + i, 1000)
    assert bench._omnivoice_band_pct() == prior                                            # ten pairs are not enough to trust
    for i in range(20):
        bench._record_estimate_pair(1000 + (i % 10) * 5, 1000)                             # errors of 0 to 4.5 %
    assert bench._omnivoice_band_pct() == 4.0                                              # the 90th percentile of what was seen, not the prior
    for _ in range(10):
        bench._record_estimate_pair(1200, 1000)                                            # a run of 20 % misses: the band follows the estimator
    assert bench._omnivoice_band_pct() == 20.0


def test_every_real_omnivoice_measurement_is_recorded_beside_the_estimate_that_was_made_for_it(tmp_path):
    from mathula_tv.dub_workbench import Workbench

    bench = object.__new__(Workbench)
    bench.store = SimpleNamespace(root=tmp_path / "experience")
    bench._omnivoice_ms = lambda *a: 4000
    result = {"measured_ms": 4200}
    bench._show_omnivoice_estimate("Sawubona", "SPEAKER_00", 0, 4100, result)
    pairs = [json.loads(line) for line in (tmp_path / "omnivoice_estimates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert pairs == [{"estimated_ms": 4000, "measured_ms": 4200}] and result["estimate_off_pct"] == -4.8


def test_the_backend_needs_a_url_and_the_provider_switch_picks_it(monkeypatch):
    with pytest.raises(RuntimeError, match="OMNIVOICE_URL"):
        ot.OmniVoiceTTSBackend.from_environment({})
    backend = ot.OmniVoiceTTSBackend.from_environment({"OMNIVOICE_URL": "https://x.run.app/", "OMNIVOICE_SPEED": "1.0", "OMNIVOICE_LORA": "0"})
    assert backend.client.url == "https://x.run.app" and backend.speed == 1.0 and backend.lora is False
    from mathula_tv import cli

    monkeypatch.setenv("MATHULA_TTS_PROVIDER", "omnivoice")
    monkeypatch.setenv("OMNIVOICE_URL", "https://x.run.app")
    assert isinstance(cli.create_tts_backend(object()), ot.OmniVoiceTTSBackend)


def test_every_request_to_the_voice_carries_64_steps_unless_the_caller_sets_its_own(monkeypatch):
    from mathula_tv import dub_agent
    from mathula_tv import omnivoice_tts as ot

    sent = []
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: sent.append(body) or {"audio_b64": ""})
    client = ot.OmniVoiceClient("https://voice.example")
    monkeypatch.setattr(client, "_id_token", lambda: "t")
    client.synthesize(spans=["Sawubona."], gaps_ms=[], reference=None)
    client.synthesize(spans=["Sawubona."], gaps_ms=[], reference=None, generation={"position_temperature": 1.0})
    client.synthesize(spans=["Sawubona."], gaps_ms=[], reference=None, generation={"num_step": 16})
    assert ot.DEFAULT_NUM_STEP == 64
    assert sent[0]["generation"] == {"num_step": 64}                                                         # the first measurement is made as the final audio will be
    assert sent[1]["generation"] == {"num_step": 64, "position_temperature": 1.0}                            # a run's own setting is added to it
    assert sent[2]["generation"] == {"num_step": 16}                                                         # and may replace it
    environment = ot.OmniVoiceTTSBackend.from_environment({"OMNIVOICE_URL": "https://voice.example", "OMNIVOICE_NUM_STEP": "32"})
    assert environment.client.default_generation == {"num_step": 32}


def test_a_clip_gets_one_copy_of_the_voice_for_each_cluster_of_speech_between_one_and_four():
    from mathula_tv import omnivoice_tts as ot

    assert [ot.copies_for(s) for s in (0, 15, 20, 21, 41, 64, 66, 80, 5000)] == [1, 1, 1, 2, 3, 4, 4, 4, 4]


def test_warming_wakes_the_voice_once_and_reports_the_copies_that_answered_and_does_nothing_for_another_voice(monkeypatch):
    from mathula_tv import dub_agent
    from mathula_tv import omnivoice_tts as ot

    asked = []
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: asked.append((url, body)) or {"instance": "only"})
    client = ot.OmniVoiceClient("https://voice.example", sleep=lambda s: None)
    monkeypatch.setattr(client, "_id_token", lambda: "t")
    said = []
    assert client.warm(1, progress=said.append) == ["only"] and len(asked) == 1 and asked[0][0].endswith("/v1/warm") and asked[0][1] == {"hold_s": 0.0} and "1 copy of the voice answered" in said[0]
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    assert client.warm(2) == []                                                                       # a service that does not answer is not a reason to stop the job
    assert ot.warm_voice(600, environ={}) is None                                                     # not the OmniVoice voice: nothing to warm
