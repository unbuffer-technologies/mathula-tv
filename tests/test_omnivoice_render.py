import base64
import io

import numpy as np
import soundfile as sf

from mathula_tv import omnivoice_render as render
from mathula_tv import omnivoice_tts as ot
from test_omnivoice_tts import _bank


class SecondsClient:
    """A service that speaks every span for a fixed time (a second and a half unless a duration is pinned)."""

    def __init__(self, seconds=1.5):
        self.calls, self.seconds = [], seconds

    def synthesize(self, *, spans, gaps_ms, reference, durations_s=None, speed=1.0, lora=True, **_ignored):
        self.calls.append({"spans": list(spans), "gaps": list(gaps_ms), "durations": durations_s, "ref": reference})
        lengths = [(d if d else self.seconds) + 0.2 for d in (durations_s or [None] * len(spans))]          # each span speaks for its time plus the model's 200 ms of padding
        pieces, starts, position = [], [], 0.0
        for i, length in enumerate(lengths):
            starts.append(round(position, 3))
            pieces.append(0.1 * np.ones(int(24000 * length)))
            position += length
            if i < len(gaps_ms):
                pieces.append(np.zeros(int(24000 * gaps_ms[i] / 1000.0)))
                position += gaps_ms[i] / 1000.0
        wav = io.BytesIO()
        sf.write(wav, np.concatenate(pieces), 24000, format="WAV", subtype="PCM_16")
        return {"audio_b64": base64.b64encode(wav.getvalue()).decode(), "span_starts_s": starts, "span_ms": [int(x * 1000) for x in lengths], "gpu_s": 0.1}


def _backend(tmp_path, seconds=1.5):
    return ot.OmniVoiceTTSBackend(SecondsClient(seconds), speed=0.85, banks={"job": _bank(tmp_path)})


ITEMS = [{"item_id": "u1", "start_ms": 0, "end_ms": 4000, "speaker_id": "SPEAKER_00", "zulu_text": "Sawubona baba."},
         {"item_id": "u2", "start_ms": 12200, "end_ms": 15000, "speaker_id": "SPEAKER_01", "zulu_text": "Siyabonga kakhulu."}]


def test_every_unit_is_laid_at_its_source_start_and_nothing_is_pushed(tmp_path):
    backend = _backend(tmp_path)
    out = render.render_submission(ITEMS, "job", backend, total_s=20)
    rows = {r["unit"]: r for r in out["rows"]}
    assert rows["u1"]["spoken_s"] == 1.7 and rows["u1"]["window_s"] == 4.0 and rows["u1"]["runs_into_next_s"] < 0          # 1.5 s + 200 ms padding; it ends well before u2 starts
    track = out["track"]
    assert len(track) == 20 * 24000 and track[int(0.5 * 24000)] > 0 and track[int(11 * 24000)] == 0 and track[int(12.5 * 24000)] > 0         # speech where the source speaks, silence between


def test_a_unit_that_runs_into_the_next_speaker_is_reported_as_the_scorer_counts_it(tmp_path):
    long_unit = [{**ITEMS[0], "end_ms": 12000}, ITEMS[1]]
    backend = _backend(tmp_path, seconds=13.0)                                                   # speaks 13.2 s from 0 s: past the 12.2 s start of the next speaker
    rows = render.render_submission(long_unit, "job", backend)["rows"]
    summary = render.summarise(rows)
    assert rows[0]["runs_into_next_s"] == 1.0 and rows[0]["next_is_other_speaker"] is True
    assert summary["collisions"] == 1 and summary["cross_speaker_collisions"] == 1 and summary["worst_collision_s"] == 1.0


def test_exact_mode_pins_each_unit_to_its_window_with_one_bounded_correction(tmp_path):
    backend = _backend(tmp_path)
    out = render.render_submission([ITEMS[0]], "job", backend, exact=True)
    row = out["rows"][0]
    call = backend.client.calls[0]
    assert call["durations"] and abs(sum(call["durations"]) + 0.2 * len(call["spans"]) - 4.0) < 1e-6              # the pinned lengths plus the model's padding fill the window
    assert abs(row["spoken_s"] - 4.0) < 0.01 and row["attempts"] == 1 and len(backend.client.calls) == 1            # it landed: no second try


def test_a_unit_with_no_text_is_reported_and_does_not_stop_the_others(tmp_path):
    backend = _backend(tmp_path)
    out = render.render_submission([{**ITEMS[0], "zulu_text": "   "}, ITEMS[1]], "job", backend)
    assert out["rows"][0]["error"] == "no text" and out["rows"][1]["spoken_s"] > 0 and render.summarise(out["rows"])["units"] == 1
