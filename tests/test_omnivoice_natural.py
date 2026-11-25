import re
import base64
import io

import numpy as np
import soundfile as sf

from mathula_tv import omnivoice_tts as ot
from mathula_tv.internal_silence import internal_silence_s, silent_runs

RATE = 24000


def _speech(seconds, seed=0):
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _quiet(seconds):
    return np.full(int(seconds * RATE), 0.0004, dtype=np.float32)


def _sentence(start, count, per_word, *, raw="AZURE_1"):
    words = []
    at = start
    for i in range(count):
        words.append({"text": f"w{i}" + ("." if i == count - 1 else ""), "start": at, "end": at + per_word * 0.9, "raw_speaker": raw, "confidence": 0.95, "overlap": False})
        at += per_word
    return words


def _bank(tmp_path):
    rate = 48000
    path = tmp_path / "mix.wav"
    sf.write(str(path), np.zeros((rate * 60, 2), dtype=np.float32), rate, subtype="PCM_16")
    slow = _sentence(0.0, 18, 0.45)                                      # 0-8.7 s: about 2.1 words a second
    for w in slow[9:]:
        w["start"] += 0.4                                                # a hesitation in the middle (a gap, but not a new run)
        w["end"] += 0.4
    words = slow + _sentence(10.0, 26, 0.32) + _sentence(20.0, 26, 0.31) + _sentence(30.0, 26, 0.33)      # the rest: the speaker's usual 3 words a second, fluent
    return ot.ReferenceBank(words, {"AZURE_1": "SPEAKER_00"}, path, cache_dir=tmp_path / "cache")


def test_a_reference_is_chosen_for_fluency_and_the_speakers_usual_pace_not_for_the_slowest_clip(tmp_path):
    bank = _bank(tmp_path)
    slow = min(bank._candidates, key=lambda c: c["wps"])
    assert slow["start"] < 1.0 and slow["gaps"] >= 1
    order = bank.ranked("SPEAKER_00", 0, 40000)
    assert order[-1]["start"] == slow["start"]                                                # last, not first
    assert order[0]["gaps"] == 0 and abs(order[0]["wps"] / bank._pace["SPEAKER_00"] - 1.0) < 0.2


def _tone(freq, seconds, seed=0):
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * freq * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def _span_audio(*, spike=False, gap_s=0.0, seconds=3.0, pad_s=0.1):
    head = np.concatenate([_tone(340, 0.2), _tone(180, seconds - 0.2, 1)]) if spike else _tone(180, seconds)
    half = int(seconds * RATE / 2)
    pieces = [head[:half]] + ([_quiet(gap_s)] if gap_s else []) + [head[half:]]
    pad = np.zeros(int(pad_s * RATE), dtype=np.float32)
    return np.concatenate([pad, np.concatenate(pieces), pad])


def _answer(spans_audio, gaps_ms):
    """What the service returns: the spans one after another with exact-zero gaps between them."""
    pieces, starts, lengths, position = [], [], [], 0
    for i, span in enumerate(spans_audio):
        starts.append(round(position / RATE, 3))
        lengths.append(int(len(span) / RATE * 1000))
        pieces.append(span)
        position += len(span)
        if i < len(gaps_ms):
            gap = np.zeros(int(gaps_ms[i] / 1000.0 * RATE), dtype=np.float32)
            pieces.append(gap)
            position += len(gap)
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(pieces), RATE, format="WAV", subtype="PCM_16")
    return {"audio_b64": base64.b64encode(buffer.getvalue()).decode(), "sample_rate": RATE, "span_starts_s": starts, "span_ms": lengths, "gpu_s": 1.0}


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _spread(count, start, end):
    step = (end - start) / max(count, 1)
    return [{"start": round(start + i * step, 3), "end": round(start + (i + 1) * step * 0.95, 3), "score": 0.9} for i in range(count)]


class ScriptedClient:
    """A service that answers each call from a script: ``make(call number, reference)`` gives the audio of each unit span.  A span that follows another is asked for with the last sentence of the
    one before in front of it; this fake speaks that lead for a second, a quarter second of quiet, then the span.  ``aligns``: whether it also returns the word times (a service that predates alignment
    does not, and is then asked for each lead alone as well, which it appends at the end of the answer)."""

    def __init__(self, make, aligns=True, lead_texts=()):
        self.make, self.calls, self.aligns, self.requests, self.refs, self.lead_texts, self.pins = make, [], aligns, [], [], list(lead_texts), []

    def synthesize(self, *, spans, gaps_ms, reference, durations_s=None, speed=1.0, lora=True, align=False, seeds=None, **_ignored):
        self.calls.append(reference.start_s)
        self.requests.append({"spans": list(spans), "align": align})
        self.refs.append(reference)
        self.pins.append(durations_s)
        mains = self.make(len(self.calls) - 1, reference)[:len(spans)]
        out, words = [], []
        for index, audio in enumerate(mains):
            before = " ".join(spans[index - 1].split()[:-1]) if index > 0 and align and self.aligns else (spans[index - 1] if index > 0 else "")      # the span before, without the word the service speaks after it
            led = index > 0 and bool(_sentences(spans[index])) and bool(_sentences(before)) and _sentences(spans[index])[0] == _sentences(before)[-1]
            lead_here = next((t for t in self.lead_texts if spans[index].startswith(t + " ")), None)
            led = led or lead_here is not None
            if led:
                audio = np.concatenate([np.zeros(int(0.1 * RATE), dtype=np.float32), _tone(180, 1.0, 7), _quiet(0.25), audio[int(0.1 * RATE):]])
            count = len(spans[index].split())
            n_lead = (len(lead_here.split()) if lead_here else len(_sentences(spans[index])[0].split())) if led else 0
            follow = align and self.aligns and (index + 1 < len(mains) or spans[index].split()[-1:] == [ot.TRAILING_FOLLOW])                     # the service speaks the first word of the next span after this one (the last span of a unit gets a plain word): a quarter second of quiet, then 0.4 s
            seconds = len(audio) / RATE
            spread = _spread(n_lead, 0.1, 1.1) + _spread(count - n_lead - (1 if follow else 0), 1.35 if led else 0.1, seconds - 0.1)
            if follow:
                audio = np.concatenate([audio[: len(audio) - int(0.1 * RATE)], _quiet(0.25), _tone(200, 0.4, 9), np.zeros(int(0.1 * RATE), dtype=np.float32)])
                spread += [{"start": seconds - 0.1 + 0.25, "end": seconds - 0.1 + 0.65, "score": 0.9}]
            out.append(audio)
            words.append(spread)
        out += [_span_audio(seconds=1.0) for _ in spans[len(mains):]]                          # the leads, each alone
        answer = _answer(out, [0] * (len(out) - 1))
        if align and self.aligns:
            answer["words"] = words + [None] * (len(out) - len(words))
            answer["aligned"] = True
        return answer


GROUP = {"group_id": "u", "speaker_id": "SPEAKER_00", "start_ms": 0, "source_end_ms": 40000, "source_span_ms": 40000, "parts": [{"type": "text", "segment_id": "s", "text": "Sawubona baba."}]}


def _backend(tmp_path, client):
    return ot.OmniVoiceTTSBackend(client, banks={"job": _bank(tmp_path)})


def test_the_voice_speaks_at_the_reference_pace_with_nothing_pinned_stretched_or_cut(tmp_path):
    assert ot.DEFAULT_SPEED == 1.0
    client = ScriptedClient(lambda call, ref: [_span_audio()])
    audio, rate, answer, report, reference = _backend(tmp_path, client).speak(GROUP, ["Sawubona baba."], [], "job")
    assert len(client.calls) == 1 and reference.start_s >= 1.0 and report["takes"] == 1          # a clean first take is the only one needed
    assert abs(len(audio) / rate - 3.2) < 0.1                                                     # what the model produced, pads included (the unit end now keeps the word's decay up to its own cut)


def test_each_span_is_the_best_of_the_takes_not_the_best_whole_take(tmp_path):
    # take 0 spikes in span 0, take 1 spikes in span 1, take 2 spikes in both: the best of each span comes from takes 1 and 0
    plan = {0: (True, False), 1: (False, True), 2: (True, True)}
    client = ScriptedClient(lambda call, ref: [_span_audio(spike=plan[call][0]), _span_audio(spike=plan[call][1])])
    backend = _backend(tmp_path, client)
    backend.nameless_draws = 3                                                                                              # the Studio makes one draw where there are no names; three here, to test the choice between them
    audio, rate, answer, report, reference = backend.speak(GROUP, ["One.", "Two."], [150], "job", all_takes=True)
    assert len(client.calls) == 3 and report["picked"] == [1, 0]
    assert all(s["onset_ratio"] < 1.6 for s in report["spans"])
    assert len(answer["span_starts_s"]) == 2 and answer["span_starts_s"][0] == 0.0 and answer["span_starts_s"][1] > 3.0


def _words_for():
    return _sentence(0.0, 26, 0.32) + _sentence(10.0, 26, 0.31) + _sentence(20.0, 26, 0.33)


def test_the_join_between_two_spans_is_room_tone_not_digital_zero_and_far_under_the_speech(tmp_path):
    rng = np.random.default_rng(3)
    source = np.concatenate([0.2 * np.sin(2 * np.pi * 150 * np.arange(RATE * 20) / RATE) * 0.8, 0.002 * rng.standard_normal(RATE * 5)]).astype(np.float32)
    path = tmp_path / "src.wav"
    sf.write(str(path), source, RATE, subtype="PCM_16")
    bank = ot.ReferenceBank(_words_for(), {"AZURE_1": "SPEAKER_00"}, path, cache_dir=tmp_path / "cache")
    backend = ot.OmniVoiceTTSBackend(ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()]), banks={"job": bank})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["One.", "Two."], [150], "job")
    next_start = answer["span_starts_s"][1]
    between = audio[int(2.5 * rate): int(next_start * rate)]                                          # from inside the first span's speech to the start of the second
    assert np.abs(between).max() > 0 and (np.abs(between[int(0.5 * rate):]) < 1e-7).mean() < 0.05    # no exact zeros left in the join
    speech_rms = np.sqrt((audio[int(1.0 * rate): int(2.0 * rate)] ** 2).mean())
    window = int(0.05 * rate)
    floor = min(np.sqrt((between[i:i + window] ** 2).mean()) for i in range(0, len(between) - window, window))
    assert 20 * np.log10(floor / speech_rms) < -25                                                    # and the join is a floor, far under the speech
    assert (tmp_path / "cache" / "room_tone.npz").is_file()


def test_a_run_makes_one_take_and_never_a_second_one_on_its_own_the_studio_makes_the_draws_by_hand(tmp_path):
    bank = _bank(tmp_path)
    slow = min(bank._candidates, key=lambda c: c["wps"])
    bank._score = lambda c: 10.0 if c is slow else 0.0                                                 # the slow clip ranks first
    client = ScriptedClient(lambda call, ref: [_span_audio(gap_s=1.0)] if ref.start_s < 1.0 else [_span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": bank})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Sawubona baba."], [], "job")
    assert len(client.calls) == 1 and report["takes"] == 1 and "fell_back_to_next_reference" not in report            # a gappy take is not redone, with another take or another reference
    taken: list = []
    client.calls.clear()
    backend.speak(GROUP, ["Sawubona baba."], [], "job", takes_out=taken, all_takes=True)
    assert len(client.calls) == 1 and len(taken[0]) == 1                                               # the Studio makes one draw where a unit has no names: the draws are of the names
    client.calls.clear()
    backend.nameless_draws = 3
    taken.clear()
    backend.speak(GROUP, ["Sawubona baba."], [], "job", takes_out=taken, all_takes=True)
    assert len(client.calls) == 3 and len(taken[0]) == 3


def test_a_pitch_spike_at_the_start_and_quiet_stretches_are_what_a_span_is_scored_on():
    from mathula_tv import voice_takes as vt

    clean, spiked = _span_audio(), _span_audio(spike=True)
    assert vt.onset_ratio(vt.trim_zero_pads(clean)[0], RATE) < 1.3 and vt.onset_ratio(vt.trim_zero_pads(spiked)[0], RATE) > 1.6
    assert vt.score_span(vt.trim_zero_pads(spiked)[0], RATE)["score"] > vt.score_span(vt.trim_zero_pads(clean)[0], RATE)["score"] + 0.5
    assert vt.quiet_seconds(vt.trim_zero_pads(_span_audio(gap_s=0.8))[0], RATE) > 0.6 and vt.quiet_seconds(vt.trim_zero_pads(clean)[0], RATE) == 0.0
    core, lead, tail = vt.trim_zero_pads(clean)
    assert lead == int(0.1 * RATE) and tail == int(0.1 * RATE) and len(core) == len(clean) - lead - tail


def test_room_tone_is_the_sources_quietest_non_silent_stretch_and_is_filled_under_the_speech_level(tmp_path):
    from mathula_tv import room_tone as rt

    rng = np.random.default_rng(1)
    speech = 0.2 * np.sin(2 * np.pi * 150 * np.arange(RATE * 10) / RATE)
    source = np.concatenate([speech, 0.003 * rng.standard_normal(RATE * 3), speech]).astype(np.float32)
    path = tmp_path / "s.wav"
    sf.write(str(path), source, RATE, subtype="PCM_16")
    tone, delta = rt.find_room_tone(path)
    assert len(tone) == RATE and 30.0 <= delta <= 60.0 and np.sqrt((tone ** 2).mean()) < 0.01
    filled = rt.fill(RATE * 2, tone, delta, speech_db=-14.0)
    assert len(filled) == RATE * 2 and abs(20 * np.log10(np.sqrt((filled[2000:-2000] ** 2).mean())) - (-14.0 - delta)) < 1.5
    assert abs(filled[0]) < 1e-3 and np.abs(rt.fill(RATE, np.zeros(RATE, dtype=np.float32), 40.0, -14.0)).max() == 0.0      # faded in; a source with no floor gives silence, not noise


def test_internal_silence_is_measured_and_the_lead_in_and_tail_are_not_counted():
    audio = np.concatenate([_quiet(0.5), _speech(1.0), _quiet(0.9), _speech(1.0, 1), _quiet(0.5)])
    assert abs(internal_silence_s(audio, RATE) - 0.9) < 0.05 and len(silent_runs(audio, RATE)) == 1
    assert internal_silence_s(_speech(3.0), RATE) == 0.0


def test_a_reference_clip_is_cut_in_the_quiet_between_words_and_not_through_the_speech(tmp_path):
    rate = 24000
    t = np.arange(rate * 12) / rate
    tone = 0.2 * np.sin(2 * np.pi * 180 * t)
    for a, b in ((3.0, 3.35), (9.0, 9.4)):                                    # the real silences between words
        tone[int(a * rate): int(b * rate)] = 0.0005
    path = tmp_path / "s.wav"
    sf.write(str(path), tone.astype(np.float32), rate, subtype="PCM_16")
    words = [{"text": f"w{i}.", "start": 3.4 + i * 0.9, "end": 3.4 + i * 0.9 + 0.8, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False} for i in range(7)]
    bank = ot.ReferenceBank(words, {"AZURE_1": "SPEAKER_00"}, path, cache_dir=tmp_path / "cache")
    # the recogniser put the clip's start 0.12 s too late (inside the speech) and its end 0.1 s too early
    start, end = bank.snap(3.5, 8.9, 3.0, 9.4)
    assert 3.0 <= start <= 3.4 and 9.0 <= end <= 9.4                          # both edges now sit in a silence
    assert bank.snap(5.0, 6.0, 4.8, 6.2) == (5.0, 6.0)                         # no quieter place beside it: left where it was


def test_the_last_sentence_of_the_span_before_leads_the_next_one_and_is_cut_off_at_the_quiet_after_it():
    from mathula_tv import voice_takes as vt

    assert vt.lead_sentence("Okokuqala kuhle. Sanibonani ekuseni kuwe Ayanda.") == "Sanibonani ekuseni kuwe Ayanda."
    assert vt.lead_sentence("x" * 300 + ".") is None and vt.lead_sentence("") is None
    # a generation that says the lead for 1 s, is quiet for 0.25 s, then says the new text for 3 s
    core = np.concatenate([_tone(180, 1.0, 1), _quiet(0.25), _tone(190, 3.0, 2)])
    start_s, end_s = vt.active_extent(core, RATE)
    assert start_s < 0.05 and abs(end_s - 4.25) < 0.05
    cut, found = vt.find_cut(core, RATE, 1.1)                                                   # the lead took about a second on its own
    assert found and 1.0 <= cut <= 1.25
    no_valley = _tone(180, 4.0, 3)
    cut2, found2 = vt.find_cut(no_valley, RATE, 1.0)
    assert not found2


def test_a_service_that_does_not_align_is_asked_for_each_lead_alone_and_the_repeat_is_cut_by_warping(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()], aligns=False)
    backend = _backend(tmp_path, client)
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."], [150], "job")
    assert client.requests[0]["align"] is True and len(client.requests[0]["spans"]) == 2                  # first it asks the new way
    assert client.requests[1]["spans"] == ["Okokuqala kuhle. Siyabonga baba.", "Siyabonga baba. Kunjalo nje.", "Siyabonga baba."] and client.requests[1]["align"] is False
    assert backend.service_aligns is False and all(not r["align"] for r in client.requests[2:])           # and does not try again
    note = report["overlap"][0]
    assert note["by"] == "warp" and note["found"] is True and note["lead_s"] > 0.8 and 1.0 < note["cut_s"] < 1.5
    assert 3.0 < answer["span_ms"][1] / 1000.0 < 3.5                                                   # what is left of span 1 is its own text (about 3.2 s), the repeated second and a half is gone
    assert report["spans"][1]["uncertain_cut"] is False


def test_with_word_times_one_request_carries_every_span_and_the_cut_is_between_the_last_lead_word_and_the_first_new_one(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()])
    backend = _backend(tmp_path, client)
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."], [150], "job")
    assert client.requests[0] == {"spans": ["Okokuqala kuhle. Siyabonga baba. Kunjalo", "Siyabonga baba. Kunjalo nje. " + ot.TRAILING_FOLLOW], "align": True}          # no lead spoken alone: one request, two spans, the first followed by the first word of the second, the last by a plain word
    assert all(r["align"] for r in client.requests) and backend.service_aligns is True
    note = report["overlap"][0]
    assert note["by"] == "words" and 0.9 < note["lead_end_s"] < 1.0 and 1.2 < note["new_start_s"] < 1.3 and 0.95 < note["cut_s"] < 1.3          # seconds into the span without the model's own 0.1 s of padding
    assert 3.0 < answer["span_ms"][1] / 1000.0 < 3.5 and report["spans"][1]["uncertain_cut"] is False


def test_the_first_word_of_the_next_span_is_spoken_after_a_span_and_cut_off_so_its_last_word_is_not_the_end_of_the_generation(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()])
    audio, rate, answer, report, reference = _backend(tmp_path, client).speak(GROUP, ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."], [150], "job")
    tail = report["tails"][0]
    assert tail["tail_by"] == "words" and tail["tail_word_end_s"] < tail["tail_next_start_s"] and tail["tail_cut_s"] <= tail["tail_word_end_s"] + 0.12 + 1e-6
    assert tail["tail_cut_s"] >= tail["tail_word_end_s"] + 0.03 - 1e-6                      # the word keeps its own decay
    assert answer["span_ms"][0] / 1000.0 < 3.4                                              # the 0.65 s the service spoke for the next span's first word is not in the audio
    last = client.requests[0]["spans"]
    assert last[0].split()[-1] == "Kunjalo" and last[1].split()[-1] == ot.TRAILING_FOLLOW              # the last span of the unit is followed by a plain word: the model does not end on its last word


def test_a_span_whose_words_are_missing_is_asked_again_the_old_way_for_that_take_only(tmp_path):
    class Flaky(ScriptedClient):
        def synthesize(self, **kw):
            reply = super().synthesize(**kw)
            if kw.get("align") and len(self.requests) == 1:
                reply["words"][1] = None                                                                # the aligner failed on span 1 of the first request
            return reply

    client = Flaky(lambda call, ref: [_span_audio(), _span_audio()])
    backend = _backend(tmp_path, client)
    backend.speak(GROUP, ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."], [150], "job")
    assert client.requests[1]["align"] is False and len(client.requests[1]["spans"]) == 3
    assert backend.service_aligns is True                                                                   # one span without words does not make the service a non-aligning one


def test_the_silence_between_two_cut_spans_is_the_voices_own_gap_or_the_longer_planned_pause(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()])
    for gap in (700, 150, -200):
        audio, rate, answer, report, reference = _backend(tmp_path, client).speak(GROUP, ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."], [gap], "job")
        note = report["silences"][0]
        planned = max(0.0, 0.2 + gap / 1000.0)
        assert 0.2 < note["natural_gap_s"] < 0.5                                              # the fake service puts the next span's first word about a quarter second after the last
        assert abs(note["target_silence_s"] - max(planned, note["natural_gap_s"])) < 1e-6
        assert abs(note["silence_s"] - note["target_silence_s"]) < 0.002                       # the cuts' own quiet and the room tone make up exactly that
        assert note["kept_tail_s"] >= 0.03 - 1e-6 and note["kept_head_s"] >= 0.03 - 1e-6      # neither word loses its edge


def test_a_zulu_prefix_is_written_apart_from_a_name_the_source_says_and_nothing_else_is_touched():
    names = {"phineas", "manyama"}
    assert ot.separate_name_prefixes("ngowayengu-Superintendent we-JMPD uPhineas Manyama, osolwa", names) == "ngowayengu-Superintendent we-JMPD u-Phineas Manyama, osolwa"
    assert ot.separate_name_prefixes("Ngithe kaManyama uyaphuma", names) == "Ngithe ka-Manyama uyaphuma"
    assert ot.separate_name_prefixes("yoMnyango wamaPhoyisa noSolwa", names) == "yoMnyango wamaPhoyisa noSolwa"          # not names the source says: untouched
    assert ot.separate_name_prefixes("uPhineas", set()) == "uPhineas" and ot.separate_name_prefixes("u-Phineas Phineas", names) == "u-Phineas Phineas"


def _bank_with_a_name(tmp_path):
    bank = _bank(tmp_path)
    for w in bank.words:
        if 10.0 <= float(w["start"]) < 20.0 and w["text"] in ("w5", "w6"):
            w["text"] = "Phineas" if w["text"] == "w5" else "Manyama,"
    bank._candidates = bank._build(bank.words, {"AZURE_1": "SPEAKER_00"})
    return bank


def test_the_speakers_own_names_are_cut_from_the_source_and_put_in_front_of_the_reference_three_times(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    primer = bank.primer_for("SPEAKER_00", 0.0, 40.0, "Siyabonga uPhineas Manyama ukhulumile.")
    assert primer["names"] == ["manyama", "phineas"] and primer["reps"] == 3
    assert [t for t, _, _ in primer["snippets"]] == ["Phineas Manyama"]                       # the two names are one run of the source, cut once, written as the dub writes them
    base = bank.as_reference(bank.ranked("SPEAKER_00", 0, 40000)[0])
    primed = bank.primed(base, primer)
    assert primed.text == "Phineas Manyama " * 3 + base.text                                  # names first, then the fluent clip, the transcript matching
    assert primed.seconds > base.seconds + 3 * primer["snippets"][0][2] - 0.01
    assert bank.primer_for("SPEAKER_00", 0.0, 40.0, "Sawubona baba.") is None and bank.primed(base, None) is base
    assert bank.primer_for("SPEAKER_01", 0.0, 40.0, "uPhineas") is None                       # another speaker's names are not this voice's


def test_the_primer_gives_way_to_the_budget_when_there_are_many_names(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    import mathula_tv.omnivoice_tts as module
    old = module.PRIMER_MAX_S
    module.PRIMER_MAX_S = 1.0                                                                    # the snippet is about 0.6 s: three copies do not fit, two do not, one does
    try:
        primer = bank.primer_for("SPEAKER_00", 0.0, 40.0, "uPhineas Manyama")
    finally:
        module.PRIMER_MAX_S = old
    assert primer["reps"] == 1 and primer["seconds"] <= 1.0 + 1e-9


def test_a_span_with_names_is_spoken_at_the_fluent_clips_pace_and_again_with_the_names_at_the_pace_that_reference_gives(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    lead = "Siyabonga uPhineas Manyama."
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()], lead_texts=[lead])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": bank})
    spans = ["Okokuqala kuhle. " + lead, "Kunjalo nje."]
    audio, rate, answer, report, reference = backend.speak(GROUP, spans, [150], "job")
    base_text = bank.as_reference(bank.ranked("SPEAKER_00", 0, 40000)[0]).text
    assert client.refs[0].text == base_text and len(client.requests[0]["spans"]) == 2 and client.pins[0] is None          # first: every span at the fluent clip's pace
    assert client.refs[1].text.startswith("Phineas Manyama Phineas Manyama Phineas Manyama ") and client.refs[1].text.endswith(base_text)       # then the span with names, names first
    assert len(client.requests[1]["spans"]) == 1 and client.requests[1]["spans"][0].startswith("Okokuqala kuhle. ") and client.pins[1] is None          # not pinned: pinning to the fluent length squeezed the short syllable in front of a name (prod-125 1:01)
    assert report["primers"][0]["names"] == ["manyama", "phineas"] and report["primers"][1] is None
    assert len(answer["span_starts_s"]) == 2 and answer["span_starts_s"][1] > answer["span_starts_s"][0] + 2.0       # both spans, in order
    assert report["overlap"][0]["by"] == "words"                                                # the cut still found the lead by the service's word times


def test_the_free_estimate_is_the_fluent_clips_pace_whatever_names_a_span_carries(tmp_path):
    from mathula_tv import omnivoice_duration as od

    spans = ["Okokuqala uPhineas Manyama kuhle ngempela.", "Kunjalo nje kuhle ngempela."]
    bank = _bank_with_a_name(tmp_path)
    backend = ot.OmniVoiceTTSBackend(ScriptedClient(lambda call, ref: [_span_audio()]), banks={"job": bank})
    backend.plan = lambda group, voice, root: (spans, [150])
    reference = bank.pick("SPEAKER_00", 0, 40000)
    assert abs(backend.estimate_seconds(GROUP, {}, "job") - od.estimate_seconds(spans, [150], reference.text, reference.seconds, backend.speed)) < 1e-9


def test_a_name_said_again_later_is_cut_once_and_names_are_dropped_before_copies_are(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    for w in bank.words:
        if 20.0 <= float(w["start"]) < 30.0 and w["text"] in ("w5", "w9", "w12"):
            w["text"] = {"w5": "manyama", "w9": "Solwa", "w12": "Madlanga"}[w["text"]]
    primer = bank.primer_for("SPEAKER_00", 0.0, 40.0, "uPhineas Manyama noSolwa noMadlanga")
    texts = [t for t, _, _ in primer["snippets"]]
    assert sum(1 for t in texts if "manyama" in t.lower()) == 1                                    # not once per mention
    import mathula_tv.omnivoice_tts as module
    old = module.PRIMER_MAX_S
    module.PRIMER_MAX_S = 3.2                                                                      # room for three copies of one short run, not of everything
    try:
        tight = bank.primer_for("SPEAKER_00", 0.0, 40.0, "uPhineas Manyama noSolwa noMadlanga")
    finally:
        module.PRIMER_MAX_S = old
    assert tight["reps"] == 3 and len(tight["snippets"]) < len(texts)                              # the copies stay at three, a name went


def test_a_name_is_found_by_how_it_sounds_when_the_recogniser_spelt_it_differently(tmp_path):
    bank = _bank(tmp_path)
    for w in bank.words:                                                                          # the source says "Kanima Panga"; the dub writes Canny Maphanga
        if 10.0 <= float(w["start"]) < 20.0 and w["text"] in ("w5", "w6"):
            w["text"] = "Kanima" if w["text"] == "w5" else "Panga,"
    bank._candidates = bank._build(bank.words, {"AZURE_1": "SPEAKER_00"})
    matches = bank.name_matches("SPEAKER_00", 0.0, 40.0, "nentatheli ephezulu uCanny Maphanga, bobabili")
    assert [m["text"] for m in matches] == ["Canny Maphanga"] and matches[0]["source_text"] == "Kanima Panga," and matches[0]["similarity"] >= 0.8
    primer = bank.primer_for("SPEAKER_00", 0.0, 40.0, "uCanny Maphanga uyintatheli")
    assert [t for t, _, _ in primer["snippets"]] == ["Canny Maphanga"]                           # the audio of Kanima Panga, written as the dub writes the name
    assert bank.name_matches("SPEAKER_00", 0.0, 40.0, "Sawubona Mayelana ngalokhu") == []         # words the speaker did not say, however capitalised
    assert bank.name_matches("SPEAKER_01", 0.0, 40.0, "uCanny Maphanga") == []                    # not this voice's
    bank.known_names = [["Canny", "Maphanga"]]
    assert [m["text"] for m in bank.name_matches("SPEAKER_00", 0.0, 40.0, "Canny, sisalinde")] == ["Canny Maphanga"]     # her first name alone is her whole name
    assert ot.separate_name_prefixes("uCanny Maphanga uyintatheli", {"canny", "maphanga"}) == "u-Canny Maphanga uyintatheli"


def test_a_run_of_words_stretched_over_half_a_minute_is_not_a_reference_clip(tmp_path):
    stretched = [{"text": "South", "start": 0.2, "end": 10.0, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False},
                 {"text": "African", "start": 10.1, "end": 20.0, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False},
                 {"text": "Bureau.", "start": 20.1, "end": 29.1, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False}]
    assert ot.ReferenceBank._build(stretched, {"AZURE_1": "SPEAKER_00"}) == []                  # three words over 28.9 s: longer than any clip may be, and trimming stops at three words
    fine = [{"text": f"w{i}", "start": 40.0 + i * 0.4, "end": 40.0 + i * 0.4 + 0.35, "raw_speaker": "AZURE_1", "confidence": 0.9, "overlap": False} for i in range(15)]
    fine[-1]["text"] = "w14."
    clips = ot.ReferenceBank._build(stretched + fine, {"AZURE_1": "SPEAKER_00"})
    assert len(clips) == 1 and clips[0]["start"] == 40.0 and clips[0]["end"] - clips[0]["start"] <= ot.CLIP_MAX_S


def test_a_name_the_job_already_restored_is_found_by_its_restored_spelling_not_the_raw_guess(tmp_path):
    bank = _bank(tmp_path)
    for w in bank.words:                                                                          # the recogniser heard "kenny" alone; the job restored it to Canny in pass 1
        if 20.0 <= float(w["start"]) < 30.0 and w["text"] == "w5":
            w["text"] = "kenny"
    bank._candidates = bank._build(bank.words, {"AZURE_1": "SPEAKER_00"})
    assert bank.name_matches("SPEAKER_00", 0.0, 40.0, "Canny, sisalinde") == []                  # raw "kenny" is 0.8 like "Canny": under the bar for one word
    inside = sorted((w for w in bank.words if 20.0 <= float(w["start"]) < 30.0), key=lambda w: float(w["start"]))
    heard = " ".join(str(w["text"]) for w in inside)
    seg = {"start_ms": 20000, "end_ms": 29900, "changed": True, "source_text": heard.lower(), "restored_text": heard.replace("kenny", "Canny") + "."}
    assert bank.restore([seg]) == 1
    found = bank.name_matches("SPEAKER_00", 0.0, 40.0, "Canny, sisalinde")
    assert [m["text"] for m in found] == ["Canny"] and found[0]["source_text"] == "kenny" and found[0]["restored_text"] == "Canny"
    other = {"start_ms": 20000, "end_ms": 29900, "changed": True, "source_text": "not the same words at all", "restored_text": "Not the same words at all"}
    assert bank.restore([other]) == 0                                                              # words that do not line up one for one are left as heard


def test_every_generation_is_seeded_from_what_it_says_so_the_same_unit_gives_the_same_audio_again(tmp_path):
    seen = []

    class Recording(ScriptedClient):
        def synthesize(self, **kw):
            seen.append((tuple(kw["spans"]), tuple(kw.get("seeds") or ()), kw.get("durations_s")))
            return super().synthesize(**kw)

    spans = ["Okokuqala kuhle. Siyabonga baba.", "Kunjalo nje."]
    for _ in range(2):
        client = Recording(lambda call, ref: [_span_audio(), _span_audio()])
        _backend(tmp_path, client).speak(GROUP, spans, [150], "job")
    half = len(seen) // 2
    assert half >= 1 and seen[:half] == seen[half:]                                            # two runs, the same requests with the same seeds
    first = seen[0]
    assert len(first[1]) == len(first[0]) and len(set(first[1])) == len(first[1])             # one seed for each span, and different ones
    assert ot.stable_seed("a", 0, "first", "1.00-2.00") == ot.stable_seed("a", 0, "first", "1.00-2.00") != ot.stable_seed("a", 1, "first", "1.00-2.00")     # another take is another draw


def test_a_spans_sentences_are_timed_from_the_word_times_without_the_lead_and_the_trailing_word():
    words = [{"start": 0.1, "end": 0.5}, {"start": 0.6, "end": 1.0},                  # the lead sentence, two words
             {"start": 1.5, "end": 1.9}, {"start": 2.0, "end": 2.6}, {"start": 3.1, "end": 3.5}, {"start": 3.6, "end": 4.2},      # the span: two sentences of two words
             {"start": 4.8, "end": 5.2}]                                              # the first word of the next span
    marks = ot.sentence_marks("Sawubona baba. Siyabonga kakhulu.", words, 2, 1.5)
    assert [m["start"] for m in marks] == [0.0, 1.6] and [m["end"] for m in marks] == [1.1, 2.7]       # seconds from the start of the audio kept, the lead's 1.5 s gone
    assert [m["chars"] for m in marks] == [len("Sawubona baba."), len("Siyabonga kakhulu.")]
    assert ot.sentence_marks("Sawubona.", None, 0, 0.0) == [] and ot.sentence_marks("", words, 0, 0.0) == []


def test_the_sources_sentences_are_its_timed_words_and_each_dub_sentence_is_matched_by_its_share_of_the_speech():
    source_words = [{"text": "Good", "start": 10.0, "end": 10.4, "raw_speaker": "A"}, {"text": "morning.", "start": 10.5, "end": 11.0, "raw_speaker": "A"},
                    {"text": "Tonight", "start": 12.0, "end": 12.6, "raw_speaker": "A"}, {"text": "we", "start": 12.7, "end": 12.9, "raw_speaker": "A"}, {"text": "go.", "start": 13.0, "end": 14.0, "raw_speaker": "A"},
                    {"text": "other", "start": 12.0, "end": 12.5, "raw_speaker": "B"}]
    theirs = ot.source_sentences(source_words, {"A": "SPEAKER_00", "B": "SPEAKER_01"}, "SPEAKER_00", 10.0, 15.0)
    assert theirs == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 4.0}]                      # the other speaker's words are not in it
    dub = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 8.0}]
    assert ot.match_source(dub, theirs) == [0, 1] and ot.match_source(dub, []) == [None, None]


def test_the_measurement_carries_the_sentences_the_voice_wrote_beside_its_audio(tmp_path):
    import json

    from mathula_tv.dub_workbench import Workbench

    audio = tmp_path / "unit.wav"
    (tmp_path / "unit.omnivoice.json").write_text(json.dumps({"natural": {"sentences": [{"n": 1, "text": "Sawubona baba.", "chars": 14, "dub_s": 1.1, "start_s": 0.1, "source_s": 0.9, "source_start_s": 0.0}]}}), encoding="utf-8")
    shown = Workbench._measured_sentences(audio)
    assert shown["sentences"] == [{"n": 1, "chars": 14, "dub_s": 1.1, "at_s": 0.1, "source_s": 0.9, "source_at_s": 0.0, "starts": "Sawubona baba."}] and "source_s" in shown["note"]
    assert Workbench._measured_sentences(tmp_path / "missing.wav") is None and Workbench._measured_sentences(None) is None


def test_a_name_the_speaker_says_is_protected_from_the_dictionarys_respelling_and_put_back_as_written():
    text = "ngowayengu-Superintendent we-JMPD uCanny Maphanga, ye-JMPD. Canny, JMPDs no Eric kuEric"
    protected, mapping = ot.protect_names(text, ["JMPD", "Canny", "Maphanga"])
    assert "JMPD" not in protected.replace("JMPDs", "") and "Canny" not in protected and "Maphanga" not in protected
    assert protected.startswith("ngowayengu-Superintendent we-Zqprot") and "uZqprot" in protected          # a Zulu prefix stays in front
    assert "JMPDs" in protected and "Eric" in protected                                                       # a longer word and a name that is not covered are left alone
    assert ot.restore_names(protected, mapping) == text
    assert ot.protect_names("nothing here", []) == ("nothing here", {})
    assert ot.ReferenceBank.dub_name_runs("wayengaphansi kwamaHawks, owanikwa") == [["Hawks"]]          # a name glued to a long Zulu prefix is still a name
    glued, glued_map = ot.protect_names("wayengaphansi kwamaHawks, ama-Hawks, Hawksbill", ["Hawks"])
    assert glued.count("Zqprot") == 2 and "Hawksbill" in glued          # a prefix of any length may be glued to the name; a longer word is not the name
    assert ot.restore_names(glued, glued_map) == "wayengaphansi kwamaHawks, ama-Hawks, Hawksbill"
    assert ot.decouple_names(ot.separate_name_prefixes("wayengaphansi kwamaHawks, owanikwa", {"hawks"}), {"hawks"}) == "wayengaphansi kwa-ma Hawks, owanikwa"          # glued, then apart: the name is the speaker's word


def test_plan_gives_the_dictionary_only_what_the_speaker_does_not_say(tmp_path):
    import mathula_tv.native_dub as native

    seen = []

    def respell(piece, *, job_root, voice):                       # the dictionary: every JMPD and every Canny respelled, as the job's research does
        seen.append(piece)
        return piece.replace("JMPD", "Jey Em Pii Dii").replace("Canny", "Kheni").replace("Bridjit", "Brijit")

    old = native._native_dub_tts_ready_parts
    native._native_dub_tts_ready_parts = respell
    try:
        bank = _bank(tmp_path)
        for w in bank.words:
            if 10.0 <= float(w["start"]) < 20.0 and w["text"] in ("w5", "w6"):
                w["text"] = "Kanima" if w["text"] == "w5" else "JMPD"
        bank._candidates = bank._build(bank.words, {"AZURE_1": "SPEAKER_00"})
        backend = ot.OmniVoiceTTSBackend(ScriptedClient(lambda call, ref: [_span_audio()]), banks={"job": bank})
        group = {"group_id": "u", "speaker_id": "SPEAKER_00", "start_ms": 0, "source_end_ms": 40000, "source_span_ms": 40000,
                 "parts": [{"type": "text", "segment_id": "s", "text": "Sawubona ye-JMPD. Sibonga abase-Bridjit uCanny."}]}
        spans, _ = backend.plan(group, {"selected_voice": "x"}, "job")
    finally:
        native._native_dub_tts_ready_parts = old
    text = " ".join(spans)
    assert "ye-JMPD" in text and "Jey Em" not in text                          # the speaker says JMPD in the source: no respelling
    assert "Brijit" in text                                                     # Bridjit is not said by the speaker here: the dictionary reads it as before
    assert "Zqprot" not in text and all("Zqprot" not in piece for piece in seen[:0]) and any("Zqprot" in piece for piece in seen)


def test_a_name_after_a_long_prefix_is_its_own_word_and_the_prefixs_last_syllable_is_set_apart():
    names = {"hawks", "superintendent", "brigitte", "tmpd", "chief"}
    assert ot.decouple_names("ozokhumbula ukuthi wayekwi-Hawks, owaphathiswa", names) == "ozokhumbula ukuthi waye-kwi Hawks, owaphathiswa"
    assert ot.decouple_names("ngowayengu-Superintendent we-JMPD abase-Brigitte nakwi-TMPD owayengu-Chief", names | {"jmpd"}) == "ngowaye-ngu Superintendent we-JMPD aba-se Brigitte na-kwi TMPD owaye-ngu Chief"
    assert ot.decouple_names("wayekwi-Hawks u-Canny ka-Manyama", {"canny", "manyama"}) == "wayekwi-Hawks u-Canny ka-Manyama"          # not a name the speaker says, and short prefixes stay
    assert ot.decouple_names("wayekwi-Hawks", set()) == "wayekwi-Hawks" and ot.decouple_names("kwi-Hawks", {"hawks"}) == "kwi-Hawks"


def test_the_silence_between_the_name_copies_makes_the_reference_the_pace_of_the_fluent_clip(tmp_path):
    import numpy as np

    bank = _bank_with_a_name(tmp_path)
    rate = ot.REFERENCE_RATE
    clip = np.zeros(int(8 * rate), dtype=np.float32)
    base = ot.Reference(wav=ot.wav_bytes(clip, rate), text="x" * 99 + " ", seconds=8.0, start_s=0.0, end_s=8.0, speaker="SPEAKER_00")          # 12.5 characters per second

    def seconds_with(name_text: str, name_seconds: float) -> float:
        piece = ot.wav_bytes(np.zeros(int(name_seconds * rate), dtype=np.float32), rate)
        primer = {"names": ["n"], "snippets": [(name_text, piece, name_seconds)], "reps": 3, "seconds": 3 * name_seconds}
        return bank.primed(base, primer).seconds

    fast = seconds_with("Phineas Manyama", 0.9)                   # 16 characters in 0.9 s: 18 per second, faster than the clip
    assert abs(fast - (8.0 + 3 * (16 / 12.5))) < 0.05              # each copy now takes 16 / 12.5 = 1.28 s, as the clip's pace says
    slow = seconds_with("Eric", 1.5)                                # 5 characters in 1.5 s: slower than the clip; it cannot be sped up
    assert abs(slow - (8.0 + 3 * (1.5 + ot.PRIMER_GAP_S))) < 0.05


def _names_unit(tmp_path, primed_seconds):
    bank = _bank_with_a_name(tmp_path)
    lead = "Siyabonga uPhineas Manyama."
    lengths = {0: [_span_audio(seconds=4.0), _span_audio(seconds=3.0)], 1: [_span_audio(seconds=primed_seconds)], 2: [_span_audio(seconds=4.0)]}
    client = ScriptedClient(lambda call, ref: lengths.get(call, [_span_audio(seconds=3.0)]), lead_texts=[lead])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": bank})
    backend.speak(GROUP, ["Okokuqala kuhle. " + lead, "Kunjalo nje."], [150], "job")
    return client


def test_a_span_the_names_made_faster_than_the_speaker_is_stretched_to_the_speakers_length(tmp_path):
    client = _names_unit(tmp_path, primed_seconds=3.0)                       # the fluent clip took 4.0 s; with the names the voice read it in 3.0
    assert client.pins[1] is None and client.pins[2] is not None and len(client.pins[2]) == 1
    assert client.pins[2][0] > 4.5                                             # asked again at the fluent length (4.0 s and the 1.25 s lead), longer than the 4.25 s the primed pass took


def test_a_span_the_names_made_slower_than_the_speaker_is_left_alone(tmp_path):
    client = _names_unit(tmp_path, primed_seconds=5.0)                       # slower than the 4.0 s of the fluent clip: pinning it down squeezed the syllable before a name
    assert len(client.calls) == 2 and client.pins[1] is None                   # the first pass and the one primed pass: nothing pinned


class _SeedClient(ScriptedClient):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seed_log = []

    def synthesize(self, *, spans, seeds=None, **kw):
        self.seed_log.append(list(zip(spans, seeds or [])))
        return super().synthesize(spans=spans, seeds=seeds, **kw)


def _first_pass_seeds(tmp_path, marks=None, retakes=None):
    from mathula_tv import draws

    tmp_path.mkdir(parents=True, exist_ok=True)
    bank = _bank(tmp_path)
    client = _SeedClient(lambda call, ref: [_span_audio(), _span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={str(tmp_path): bank})
    spans = ["Okokuqala kuhle impela.", "Kunjalo nje yena."]
    ref = backend.reference_for(GROUP, tmp_path)
    for text, number, rs, re_ in (marks or []):
        draws.commit(tmp_path, text, rs if rs is not None else ref.start_s, re_ if re_ is not None else ref.end_s, number)
    backend.speak(GROUP, spans, [150], tmp_path, retakes=retakes)
    return [seed for _, seed in client.seed_log[0]][:2], spans                    # the first pass: one seed per span, in order


def test_an_approved_retake_changes_the_seed_of_that_sentence_only_and_a_mark_for_another_clip_is_ignored(tmp_path):
    plain, spans = _first_pass_seeds(tmp_path / "a")
    marked, _ = _first_pass_seeds(tmp_path / "b", marks=[(spans[0], 1, None, None)])
    assert marked[1] == plain[1] and marked[0] != plain[0]                      # only the marked sentence has another draw
    other_clip, _ = _first_pass_seeds(tmp_path / "c", marks=[(spans[0], 1, 1.0, 2.0)])
    assert other_clip == plain                                                  # made with another reference clip: no longer applies
    again, _ = _first_pass_seeds(tmp_path / "d", marks=[(spans[0], 1, None, None)])
    assert again == marked                                                      # the same mark gives the same draw every time
    tried, _ = _first_pass_seeds(tmp_path / "e", retakes={0: 1})
    assert tried == marked                                                      # trying retake 1 is the draw committing it keeps


def test_the_speakers_natural_pause_is_read_from_the_pauses_in_their_own_clips(tmp_path):
    rate = ot.REFERENCE_RATE
    speech = (np.sin(np.linspace(0, 3000, rate)) * 0.3).astype(np.float32)
    clip = np.concatenate([speech, np.zeros(int(0.4 * rate), dtype=np.float32), speech, np.zeros(int(0.4 * rate), dtype=np.float32), speech])          # two 0.4 s pauses
    bank = ot.ReferenceBank.__new__(ot.ReferenceBank)
    bank._candidates = [{"speaker": "S", "start": float(i * 10), "end": float(i * 10 + 5)} for i in range(6)] + [{"speaker": "T", "start": 100.0, "end": 105.0}]
    bank._clip = lambda start, end: ot.wav_bytes(clip, rate)
    pause = bank.natural_pause_s("S")
    assert pause is not None and 0.3 < pause < 0.5                      # twelve pauses of about 0.4 s
    assert bank.natural_pause_s("S", 75.0) == pause                      # all the pauses are the same length here
    assert bank.natural_pause_s("nobody") is None                       # no clips of that speaker


def test_a_full_stop_is_given_the_speakers_breath_even_when_the_voice_ran_the_next_sentence_straight_on(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bank = _bank(tmp_path)
    bank.natural_pause_s = lambda speaker, percentile=95.0: 0.5 if percentile == 75.0 else None          # the speaker leaves about half a second
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={str(tmp_path): bank})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Okokuqala kuhle impela.", "Kunjalo nje yena."], [-200], tmp_path)
    assert report["silences"] and report["silences"][0]["target_silence_s"] >= 0.5          # planned tight (-200 ms): the breath is the floor


def test_every_take_is_handed_out_and_a_committed_take_is_the_one_kept(tmp_path):
    from mathula_tv import draws

    tmp_path.mkdir(parents=True, exist_ok=True)
    bank = _bank(tmp_path)
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={str(tmp_path): bank})
    backend.nameless_draws = 3
    spans = ["Okokuqala kuhle impela.", "Kunjalo nje yena."]
    ref = backend.reference_for(GROUP, tmp_path)
    plain = []
    backend.speak(GROUP, spans, [150], tmp_path, takes_out=plain, all_takes=True)
    assert len(plain) == 2 and all(len(takes) == ot.vt.MAX_TAKES if hasattr(ot, "vt") else len(takes) == 3 for takes in plain)          # three draws of each sentence, whatever their scores
    draws.commit(tmp_path, spans[1], ref.start_s, ref.end_s, 0, take=2)
    _, _, _, report, _ = backend.speak(GROUP, spans, [150], tmp_path)
    assert report["picked"][1] == 2 and report["takes"] == 3                      # the committed take is kept, and it was made even though the clean first take would have been enough
    assert draws.take_for(draws.load(tmp_path), spans[1], ref.start_s + 5, ref.end_s) is None          # a mark made with another reference clip does not apply



def test_the_usual_pace_is_how_the_speaker_speaks_overall_not_the_median_of_the_few_clips_found():
    clips = [{"speaker": "SPEAKER_00", "wps": w} for w in (2.87, 2.84, 3.2, 2.51, 1.95)]                       # five short fluent sentences: their median is 2.84
    words = []
    for run in range(2):                                                                                       # 150 words at 2.5 a second, in two runs a second apart
        base = run * 31.0
        words += [{"text": f"w{i}", "start": base + i * 0.4, "end": base + i * 0.4 + 0.3, "raw_speaker": "AZURE_1", "overlap": False} for i in range(75)]
    pace = ot.ReferenceBank._speaker_pace(clips, words, {"AZURE_1": "SPEAKER_00"})
    assert 2.4 < pace["SPEAKER_00"] < 2.65 and pace[""] == pace["SPEAKER_00"]                                  # the speaker's own pace, under the clips' median of 2.84
    few = ot.ReferenceBank._speaker_pace(clips, words[:10], {"AZURE_1": "SPEAKER_00"})
    assert few["SPEAKER_00"] == 2.84                                                                           # too little speech to know: the clips' median stands


def test_sentences_can_be_spoken_together_in_one_generation_for_an_experiment_and_the_gap_after_the_last_is_kept():
    spans = ["Konke sekulungisiwe manje.", "Sesilungele ukuqhubeka.", "Sesihlangene namaqembu ezepolitiki.", "Imikhandlu isisunguliwe."]
    merged, gaps = ot.merge_spans(spans, [150, 300, 450], 2)
    assert merged == ["Konke sekulungisiwe manje. Sesilungele ukuqhubeka.", "Sesihlangene namaqembu ezepolitiki. Imikhandlu isisunguliwe."] and gaps == [300]
    three, gaps3 = ot.merge_spans(spans, [150, 300, 450], 3)
    assert len(three) == 2 and three[0].count(".") == 3 and gaps3 == [450]
    assert ot.merge_spans(spans, [150, 300, 450], 4) == (["Konke sekulungisiwe manje. Sesilungele ukuqhubeka. Sesihlangene namaqembu ezepolitiki. Imikhandlu isisunguliwe."], [])
    long = ["a" * 200, "b" * 200]
    assert ot.merge_spans(long, [150], 3)[0] == long                                                                   # never past 320 characters: the model would chunk it


def test_room_tone_is_a_steady_floor_never_a_breath_that_every_join_would_then_repeat(tmp_path):
    from mathula_tv import room_tone as rt

    rng = np.random.default_rng(2)
    speech = 0.2 * np.sin(2 * np.pi * 150 * np.arange(RATE * 10) / RATE)
    breath = 0.02 * rng.standard_normal(int(0.15 * RATE))                 # a short loud burst of hiss, as a breath at the end of a pause
    floor = 0.002 * rng.standard_normal(RATE * 2)
    quiet_with_breath = np.concatenate([breath, floor])                       # the quietest second on average, but it starts with the breath
    steady_floor = 0.0035 * rng.standard_normal(RATE * 3)                      # a little louder, perfectly steady
    source = np.concatenate([speech, quiet_with_breath, speech, steady_floor, speech]).astype(np.float32)
    path = tmp_path / "s.wav"
    sf.write(str(path), source, RATE, subtype="PCM_16")
    tone, _delta = rt.find_room_tone(path)
    frames = [20 * np.log10(np.sqrt((tone[i:i + 1200] ** 2).mean()) + 1e-9) for i in range(0, len(tone) - 1200, 1200)]
    assert max(frames) - min(frames) <= rt.STEADY_DB                           # no burst anywhere in the tone that is laid in the joins


def _spoken_words(count, per_word, gap=0.05):
    """(audio, times): count words of tone, each per_word long, with a short quiet between them."""
    pieces, times, at = [], [], 0.0
    for i in range(count):
        pieces += [_speech(per_word, seed=i), _quiet(gap)]
        times.append({"text": f"w{i}", "start": at, "end": at + per_word})
        at += per_word + gap
    return np.concatenate(pieces), times


def _mid(audio, t0, t1):
    return (t0 + t1) / 2.0


def test_a_sentence_spoken_again_replaces_itself_in_the_cluster_and_the_words_after_it_move_by_the_difference():
    part, times = _spoken_words(8, 0.3, gap=0.3)          # w0..w7, a real pause between the words
    small, small_times = _spoken_words(3, 0.5, gap=0.3)   # the sentence w3..w5 spoken again: 3 words of 0.5 s (0.6 s longer in all)
    joined, moved = ot.swap_word_run(part, RATE, times, 3, 5, small, small_times, 0, 2, _mid, keep_s=0.05)
    grown = (3 * 0.5 + 2 * 0.3) - (3 * 0.3 + 2 * 0.3)
    assert abs(len(joined) / RATE - (len(part) / RATE + grown)) < 0.1                    # less the crossfades, and the quiet this synthetic snippet has no room for before its first word
    assert moved[:3] == times[:3]                                                         # before the sentence: untouched
    assert abs(moved[6]["start"] - (times[6]["start"] + grown)) < 0.1                   # after it: moved by the difference
    assert abs((moved[5]["end"] - moved[3]["start"]) - (small_times[2]["end"] - small_times[0]["start"])) < 0.03
    assert moved[6]["start"] - moved[5]["end"] > 0.25                                    # the pause after the sentence is still whole


def test_a_swap_that_cannot_be_made_is_refused_and_name_words_are_found_by_how_they_are_said():
    part, times = _spoken_words(3, 0.3)
    small, small_times = _spoken_words(3, 0.3)
    assert ot.swap_word_run(part, RATE, [times[0], None, times[2]], 1, 1, small, small_times, 1, 1, _mid) is None
    assert ot.name_runs("Kanti futhi i-NEC iwamukele umbiko".split(), {"nec"}) == [(2, 2)]
    assert ot.name_runs("Canny Maphanga, uthe Canny Maphanga".split(), {"canny", "maphanga"}) == [(0, 1), (3, 4)]


def test_a_name_with_a_zulu_prefix_is_swapped_together_with_the_vowel_before_it_that_it_runs_into():
    assert ot.joined_with_the_word_before("Kanti futhi i-NEC iwamukele".split(), 2) == 1                 # futhi i-NEC: two i's in a row
    assert ot.joined_with_the_word_before("Kanti futhi NEC iwamukele".split(), 2) == 2                   # no prefix: the name alone
    assert ot.joined_with_the_word_before("Kanti ngenxa yi-ANC".split(), 2) == 1                           # 'ngenxa' ends in a vowel: taken with it
    assert ot.joined_with_the_word_before("Kanti umsebenz yi-ANC".split(), 2) == 2                         # a word ending in a consonant: the name alone
    assert ot.joined_with_the_word_before("I-NEC iyasebenza".split(), 0) == 0                              # at the start of a sentence there is no word before


def test_the_draw_with_the_names_said_longest_is_the_one_used_and_a_span_without_names_keeps_the_audio_score():
    assert ot.pick_take([{"score": 0.0, "name_s": 0.38}, {"score": 0.4, "name_s": 0.301}, {"score": 0.2, "name_s": 0.401}]) == 2          # i-NEC as heard
    assert ot.pick_take([{"score": 0.5, "name_s": 0.42}, {"score": 0.9, "name_s": 0.641}, {"score": 0.1, "name_s": 0.54}]) == 1          # yi-ANC as heard
    assert ot.pick_take([{"score": 0.3, "name_s": 0.0}, {"score": 0.1, "name_s": 0.0}, {"score": 0.2, "name_s": 0.0}]) == 1              # no names: the least wrong audio
    assert ot.pick_take([{"score": 0.4, "name_s": 0.3}, {"score": 0.1, "name_s": 0.3}]) == 1                                        # equal names: the cleaner audio


def test_sentences_are_packed_for_as_long_as_they_fit_and_a_sentence_that_never_fits_stands_alone():
    spans = ["a" * 60, "b" * 50, "c" * 40, "d" * 70, "e" * 200, "f" * 30]
    fits = lambda text: len(text) <= 190                                                 # the budget, as characters here
    merged, gaps = ot.merge_spans(spans, [10, 20, 30, 40, 50], ot.FILL_SENTENCES, fits=fits)
    assert [len(m) for m in merged] == [152, 70, 200, 30] and gaps == [30, 40, 50]                 # the gap after the last sentence of each cluster is the one kept
    assert len(ot.merge_spans(spans, [10] * 5, 3, fits=lambda text: True)[0]) == 2                       # the sentence cap still holds when one is given


def test_a_render_draws_a_span_with_names_as_often_as_the_studio_does_and_a_run_draws_it_once(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    lead = "Siyabonga uPhineas Manyama."
    spans = ["Okokuqala kuhle. " + lead, "Kunjalo nje."]
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio()], lead_texts=[lead])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": bank})
    assert backend.speak(GROUP, spans, [150], "job")[3]["takes"] == 1                  # a run: one
    backend.name_draws = True
    assert backend.speak(GROUP, spans, [150], "job")[3]["takes"] == 3                  # the render: the draws of the names


def test_the_primed_generations_are_asked_for_with_the_first_pass_when_copies_are_held_and_one_after_another_when_one_is(tmp_path):
    import threading
    import time

    class Counting(ScriptedClient):
        def __init__(self, parallel):
            super().__init__(lambda call, ref: [_span_audio(), _span_audio()])
            self.parallel, self.lock, self.now, self.peak = parallel, threading.Lock(), 0, 0

        def synthesize(self, **kw):
            with self.lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
            time.sleep(0.1)
            try:
                return super().synthesize(**kw)
            finally:
                with self.lock:
                    self.now -= 1

    spans = ["Okokuqala uPhineas Manyama kuhle.", "Kunjalo uPhineas Manyama nje."]
    peaks = {}
    for copies in (1, 3):
        client = Counting(copies)
        folder = tmp_path / f"c{copies}"
        folder.mkdir()
        backend = ot.OmniVoiceTTSBackend(client, banks={"job": _bank_with_a_name(folder)})
        audio, rate, answer, report, reference = backend.speak(GROUP, spans, [150], "job")
        peaks[copies] = (client.peak, len(client.requests), [p["names"] for p in report["primers"]])
    assert peaks[1][0] == 1 and peaks[3][0] == 3                                                 # first pass + two primed generations in the air together, only with three copies
    assert peaks[1][1] == peaks[3][1] >= 3 and all({"manyama", "phineas"} <= set(names) for names in peaks[3][2])         # the same requests either way: the audio does not depend on it


def test_a_title_before_a_name_is_not_part_of_the_name():
    runs = ot.ReferenceBank.dub_name_runs
    assert runs("Hlobo luni, i-ActionSA efisa ukulubona, Mnu Mashaba, ngemuva kokuthola") == [["ActionSA"], ["Mashaba"]] or runs("Mnu Mashaba, ngemuva")[0] == ["Mashaba"]
    assert runs("noMnu. Phineas Manyama, siyabonga") == [["Phineas", "Manyama"]]
    assert runs("uMnumzane Khumalo uzoba yisekela") == [["Khumalo"]]
    assert runs("Dkt Matebula uthe") == [["Matebula"]] and runs("Mnumzana Cele") == [["Cele"]]
    assert runs("Mnu") == []                                                                       # a title alone is no name


def test_an_abbreviated_title_is_spelled_out_for_the_voice_even_when_the_name_after_it_is_one_the_speaker_says(tmp_path):
    bank = _bank_with_a_name(tmp_path)
    backend = ot.OmniVoiceTTSBackend(ScriptedClient(lambda call, ref: [_span_audio()]), banks={"job": bank})
    group = {**GROUP, "parts": [{"type": "text", "segment_id": "s", "text": "Sawubona Mnu Phineas Manyama, siyabonga kakhulu."}]}
    spans, _gaps = backend.plan(group, {"selected_voice": "x"}, "job")
    spoken = " ".join(spans)
    assert "Mnumzana" in spoken and "Mnu " not in spoken                                            # the voice says the word, not the abbreviation
    assert "Phineas Manyama" in spoken                                                              # and the name the speaker says keeps the spelling it was heard with
    group = {**GROUP, "parts": [{"type": "text", "segment_id": "s", "text": "Sikhulume noMnu. Adams, nokho."}]}
    assert "noMnumzana Adams" in " ".join(backend.plan(group, {"selected_voice": "x"}, "job")[0])


def test_a_unit_of_one_sentence_gets_the_times_of_its_words_for_the_studio_strip(tmp_path):
    bank = _bank(tmp_path)
    backend = ot.OmniVoiceTTSBackend(ScriptedClient(lambda call, ref: [_span_audio()]), banks={"job": bank})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Sawubona baba wami nonke."], [], "job")
    assert len(report["words"]) == 4 and [w["text"] for w in report["words"]] == ["Sawubona", "baba", "wami", "nonke."]
    assert all(w["end_s"] > w["start_s"] for w in report["words"])


def _draws(*durations, text="wase-Ekurhuleni", extra=None):
    return [{"score": 0.1 * i, "name_s": s, "names": [{"text": text, "s": s}] + ([extra[i]] if extra else [])} for i, s in enumerate(durations)]


def test_a_draw_that_said_a_name_far_longer_than_the_others_is_set_aside_even_though_it_is_the_longest():
    echo = _draws(0.98, 1.72, 0.94)                                                     # prod-x1 "wase-Ekurhuleni": the longest is a word said twice
    assert ot.stretched_draws(echo) == [False, True, False] and ot.pick_take(echo) == 0  # draws 1 and 3 are the usual ones, and of them the longer (0.98) is used
    assert ot.pick_take(_draws(0.98, 1.72, 1.05)) == 2


def test_the_draws_the_ear_chose_are_not_set_aside_by_the_guard():
    assert ot.stretched_draws(_draws(0.38, 0.30, 0.40, text="i-NEC")) == [False, False, False] and ot.pick_take(_draws(0.38, 0.30, 0.40, text="i-NEC")) == 2
    assert ot.stretched_draws(_draws(0.42, 0.641, 0.54, text="yi-ANC")) == [False, False, False] and ot.pick_take(_draws(0.42, 0.641, 0.54, text="yi-ANC")) == 1


def test_with_fewer_than_three_draws_the_guard_says_nothing_and_the_longest_name_stands():
    assert ot.stretched_draws(_draws(0.98, 1.72)) == [False, False] and ot.pick_take(_draws(0.98, 1.72)) == 1
    assert ot.stretched_draws([{"score": 0.0}] * 3) == [False] * 3                       # no names: nothing to compare


def test_each_name_of_a_span_is_compared_with_its_own_draws():
    other = [{"text": "Mashaba", "s": 0.23}, {"text": "Mashaba", "s": 0.25}, {"text": "Mashaba", "s": 0.9}]
    flags = ot.stretched_draws(_draws(0.98, 1.0, 1.05, extra=other))
    assert flags == [False, False, True]                                                # the third said "Mashaba" at four times the usual length


def test_a_name_is_given_its_pace_against_the_rest_of_its_sentence():
    words = "Hlobo luni lokwakheka komkhandlu wase-Ekurhuleni efisa ukulubona".split()
    times = [{"start": 0.0, "end": 0.22}, {"start": 0.22, "end": 0.40}, {"start": 0.40, "end": 0.80}, {"start": 0.80, "end": 1.30}, {"start": 1.30, "end": 3.03}, {"start": 3.03, "end": 3.31}, {"start": 3.31, "end": 3.89}]
    [name] = ot.name_lengths(words, times, {"ekurhuleni"})
    assert name["text"] == "wase-Ekurhuleni" and abs(name["s"] - 1.73) < 0.001 and name["pace"] < 0.5          # 8.7 letters a second against about 20 for the rest
    assert ot.name_lengths(words, [None] * 7, {"ekurhuleni"}) == []


def test_the_guard_looks_at_each_word_of_a_run_of_names_so_a_second_name_does_not_hide_an_echo():
    # prod-x1 row 0: the run "wase-Ekurhuleni i-ActionSA" took 1.86, 2.37 and 1.94 s (1.2x the median), but the first word alone 0.98, 1.72 and 0.94 s
    draws = [{"score": 0.0, "name_s": run, "names": [{"text": "wase-Ekurhuleni i-ActionSA", "s": run, "words": [{"text": "wase-Ekurhuleni", "s": a}, {"text": "i-ActionSA", "s": run - a}]}]}
             for run, a in ((1.864, 0.98), (2.366, 1.72), (1.944, 0.94))]
    assert ot.stretched_draws(draws) == [False, True, False] and ot.pick_take(draws) == 2                    # draw 2 is the longest run, and it is set aside: of the rest 3 is the longer
    [name] = ot.name_lengths("Hlobo luni lokwakheka komkhandlu wase-Ekurhuleni i-ActionSA efisa".split(),
                             [{"start": 0.0, "end": 0.2}, {"start": 0.2, "end": 0.4}, {"start": 0.4, "end": 0.8}, {"start": 0.8, "end": 1.3}, {"start": 1.3, "end": 3.03}, {"start": 3.03, "end": 3.65}, {"start": 3.65, "end": 3.9}],
                             {"ekurhuleni", "actionsa"})
    assert [w["text"] for w in name["words"]] == ["wase-Ekurhuleni", "i-ActionSA"] and abs(name["words"][0]["s"] - 1.73) < 0.001


def test_an_english_word_kept_in_english_after_a_zulu_prefix_is_a_candidate_for_the_speakers_own_audio():
    runs = ot.ReferenceBank.dub_name_runs
    assert runs("angikwazi ukukunika ama-details amanye, kodwa") == [["details"]]
    assert runs("izihlalo ezingu-forty kulo mkhandlu?") == [["forty"]]
    assert runs("uPhineas Manyama, ama-details okugcina") == [["Phineas", "Manyama"], ["details"]]               # a name run and a word, apart
    assert runs("Ngakho-ke yilokho, ngoba-ke nje, ama-photos kuphela") == [["photos"]]                         # short stems and Zulu words with a hyphen are not taken
    assert runs("i-ANC iyavuma, no-1 kuphela, kuka-230") == [["ANC"]]


def test_a_snap_that_may_not_go_into_the_word_never_cuts_its_edges_off(tmp_path):
    bank = _bank(tmp_path)
    for start, end in ((2.0, 2.34), (5.1, 5.5), (12.0, 12.3)):
        inside_start, inside_end = bank.snap(start, end, start - 0.3, end + 0.3, inward=0.0)
        assert inside_start <= start + 1e-9 and inside_end >= end - 1e-9                 # outward only: the word is whole
        assert inside_end - inside_start >= end - start - 1e-9



def _bank_where_the_speaker_says(tmp_path, word):
    bank = _bank(tmp_path)
    for w in bank.words:
        if 10.0 <= float(w["start"]) < 20.0 and w["text"] in ("w5", "w6"):
            w["text"] = word if w["text"] == "w5" else "again,"
    bank._candidates = bank._build(bank.words, {"AZURE_1": "SPEAKER_00"})
    return bank


def test_a_word_after_a_glued_prefix_is_spoken_as_two_words_and_its_times_come_back_as_the_words_of_the_text(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": _bank(tmp_path)})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Kunamaqembu ama-details okugcina silinde."], [], "job")
    sent = [t for r in client.requests for t in r["spans"]]
    assert sent and all("ama details" in t and "ama-details" not in t for t in sent)               # the speaker did not say it: the voice must say all of it
    assert [w["text"] for w in report["words"]] == ["Kunamaqembu", "ama-details", "okugcina", "silinde."]      # the words of the dub text, not of what the voice was sent


def test_a_word_the_speaker_says_stays_glued_in_the_sentence_it_is_cut_out_of_and_is_spoken_in_full_in_the_primed_one(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio(), _span_audio(), _span_audio()])
    backend = ot.OmniVoiceTTSBackend(client, banks={"job": _bank_where_the_speaker_says(tmp_path, "details")})
    audio, rate, answer, report, reference = backend.speak(GROUP, ["Kunamaqembu ama-details okugcina silinde."], [], "job")
    sent = [t for r in client.requests for t in r["spans"]]
    assert any("ama-details" in t and "ama details" not in t for t in sent)                        # the cluster: its own "details" is replaced, and the word after it stays whole
    assert any("ama details" in t for t in sent)                                                    # the primed sentence the word is taken from, or the primed span itself, says it in full
    assert [w["text"] for w in report["words"]] == ["Kunamaqembu", "ama-details", "okugcina", "silinde."]


def test_two_names_with_a_word_or_so_between_them_are_taken_as_one_stretch_from_the_primed_sentence():
    words = "Kodwa ngicabanga ukuthi yebo, ama-details okugcina, silinde i-ANC ukuthi iwabeke phansi,".split()
    names = ["anc", "details"]
    assert ot.swap_regions(words, names) == [(3, 7)]                          # yebo ... i-ANC: "okugcina" is not left between two splices
    far = "Kodwa ama-details okugcina kakhulu silinde i-ANC ukuthi".split()
    assert ot.swap_regions(far, names) == [(0, 1), (4, 5)]                       # two words between: two stretches, as before
    assert ot.swap_regions("Sawubona baba wami.".split(), names) == []
    assert ot.swap_regions("Kanti futhi i-NEC iwamukele".split(), ["nec"]) == [(1, 2)]          # one name: the name with the vowel before it, as before


def test_the_last_span_of_a_unit_is_followed_by_a_plain_word_that_is_cut_away_and_it_fades_out(tmp_path):
    client = ScriptedClient(lambda call, ref: [_span_audio()])
    audio, rate, answer, report, reference = _backend(tmp_path, client).speak(GROUP, ["Sawubona baba wami."], [], "job")
    assert client.requests[0]["spans"][0].split()[-1] == ot.TRAILING_FOLLOW                       # the model does not end on the last word
    assert [w["text"] for w in report["words"]] == ["Sawubona", "baba", "wami."]                  # the plain word is not in the words
    assert len(audio) / rate < 3.4                                                                 # and not in the audio: the 0.65 s the service spoke for it is cut away
    assert abs(float(audio[-1])) < 1e-3                                                            # and the end fades out, it does not stop
