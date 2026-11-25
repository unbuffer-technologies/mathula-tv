import base64
import hashlib
import io

import numpy as np
import soundfile as sf

from mathula_tv import dub_agent
from mathula_tv import omnivoice_cache as oc
from mathula_tv import omnivoice_tts as ot

RATE = 24000


def _span_audio(text: str, seed: int) -> np.ndarray:
    """What the fake service says for a span: it depends on the text and the seed and on nothing else, like the real one."""
    digest = int(hashlib.sha256(f"{text}|{seed}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(digest)
    length = 3000 + 900 * len(text)
    pad = np.zeros(1200, dtype=np.int16)                                  # the model pads every span with exact zeros at both ends (50 ms)
    return np.concatenate([pad, (rng.standard_normal(length) * 3000).astype(np.int16), pad])


class FakeService:
    """The voice service's /v1/synthesize, as far as the client can tell: spans one after another with the gaps between them, word times from the span's own audio."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, **kwargs):
        self.requests.append(body)
        spans, seeds = body["spans"], body.get("seeds") or [None] * len(body["spans"])
        gaps = list(body.get("gaps_ms") or [])[: len(spans) - 1]
        gaps += [0] * (len(spans) - 1 - len(gaps))
        chunks, starts, lengths, words, elapsed = [], [], [], [], 0
        for i, text in enumerate(spans):
            audio = _span_audio(text, seeds[i] if seeds[i] is not None else 7)
            starts.append(round(elapsed / RATE, 3)); lengths.append(int(len(audio) / RATE * 1000))
            chunks.append(audio); elapsed += len(audio)
            n = max(1, len(text.split()))
            words.append([{"start": round(k * len(audio) / RATE / n, 3), "end": round((k + 0.9) * len(audio) / RATE / n, 3), "score": 0.9} for k in range(n)])
            if i < len(gaps):
                silence = np.zeros(int(gaps[i] / 1000.0 * RATE), dtype=np.int16)
                chunks.append(silence); elapsed += len(silence)
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), RATE, format="WAV", subtype="PCM_16")
        answer = {"audio_b64": base64.b64encode(buffer.getvalue()).decode("ascii"), "sample_rate": RATE, "span_starts_s": starts, "span_ms": lengths, "gpu_s": 1.5 * len(spans)}
        if body.get("align"):
            answer["words"], answer["aligned"] = words, True
        return answer


def _client(tmp_path, monkeypatch, *, max_bytes=oc.DEFAULT_MAX_BYTES):
    service = FakeService()
    monkeypatch.setattr(dub_agent, "post_with_retry", service)
    client = ot.OmniVoiceClient("http://voice", cache=oc.SpanCache(tmp_path / "cache", max_bytes=max_bytes))
    monkeypatch.setattr(client, "_id_token", lambda: "t")
    return client, service


REF = ot.Reference(wav=b"reference-audio", text="a clip", seconds=8.0, start_s=1.0, end_s=9.0, speaker="SPEAKER_00")


def _ask(client, spans, seeds, *, gaps=None, align=True, reference=REF, **kw):
    return client.synthesize(spans=spans, gaps_ms=gaps if gaps is not None else [100 * (i + 1) for i in range(len(spans) - 1)], reference=reference, align=align, seeds=seeds, **kw)


def _segments(reply):
    audio, rate = oc.decode_wav(reply["audio_b64"])
    return [audio[int(float(s) * rate): int((float(s) + ms / 1000.0) * rate)] for s, ms in zip(reply["span_starts_s"], reply["span_ms"])]


def _same_up_to_a_shift(a, b, most=30):
    """The service reports where a span starts to the millisecond, so a span cut out of its reply moves by up to half a millisecond (12 samples): the same audio when it matches at some shift that small."""
    return any(np.array_equal(a[max(0, k): max(0, k) + 2000], b[max(0, -k): max(0, -k) + 2000]) for k in range(-most, most + 1))


def test_a_request_said_before_is_answered_from_the_cache_with_the_same_audio_times_and_words(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    first = _ask(client, ["Sawubona baba wami.", "Kunjalo nje yena."], [11, 12])
    again = _ask(client, ["Sawubona baba wami.", "Kunjalo nje yena."], [11, 12])
    assert len(service.requests) == 1 and again["cache"] == {"hits": 2, "spans": 2} and again["gpu_s"] == 0.0
    for a, b in zip(_segments(first), _segments(again)):
        assert abs(len(a) - len(b)) <= 24 and _same_up_to_a_shift(a, b)
    assert again["words"] == first["words"] and again["aligned"] is True
    assert abs(again["span_starts_s"][1] - first["span_starts_s"][1]) < 0.002 and again["span_ms"] == first["span_ms"]


def test_only_the_spans_that_changed_are_asked_for_and_the_reply_is_put_together_like_the_services(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    spans = ["Sawubona baba wami.", "Kunjalo nje yena.", "Siyabonga kakhulu."]
    first = _ask(client, spans, [1, 2, 3], gaps=[150, 250])
    changed = _ask(client, [spans[0], "Kunjalo nje futhi yena.", spans[2]], [1, 2, 3], gaps=[150, 250])
    assert len(service.requests) == 2 and service.requests[1]["spans"] == ["Kunjalo nje futhi yena."] and service.requests[1]["seeds"] == [2]
    assert changed["cache"] == {"hits": 2, "spans": 3}
    direct = FakeService()(None, None, {"spans": [spans[0], "Kunjalo nje futhi yena.", spans[2]], "seeds": [1, 2, 3], "gaps_ms": [150, 250], "align": True})
    for a, b in zip(_segments(changed), _segments(direct)):
        assert abs(len(a) - len(b)) <= 24 and _same_up_to_a_shift(a, b)
    assert all(abs(x - y) < 0.003 for x, y in zip(changed["span_starts_s"], direct["span_starts_s"]))      # the gaps are where the service puts them, to the millisecond
    assert changed["words"][0] == first["words"][0] and changed["words"][2] == first["words"][2] and changed["words"][1] == direct["words"][1]


def test_a_span_with_no_seed_is_never_cached_and_a_changed_reference_or_setting_is_not_the_same_span(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    _ask(client, ["Sawubona baba wami."], [None])
    _ask(client, ["Sawubona baba wami."], [None])
    assert len(service.requests) == 2                                                                  # unseeded: asked twice
    _ask(client, ["Sawubona baba wami."], [5]); _ask(client, ["Sawubona baba wami."], [5])
    assert len(service.requests) == 3                                                                  # seeded: once
    other = ot.Reference(wav=b"another-clip", text="a clip", seconds=8.0, start_s=2.0, end_s=10.0, speaker="SPEAKER_00")
    _ask(client, ["Sawubona baba wami."], [5], reference=other)
    _ask(client, ["Sawubona baba wami."], [5], generation={"position_temperature": 1.0})
    _ask(client, ["Sawubona baba wami."], [5], speed=1.1)
    assert len(service.requests) == 6                                                                  # another voice, another setting, another speed: another span


def test_words_that_were_not_kept_are_asked_for_again_and_a_service_that_cannot_align_gets_no_words_invented(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    _ask(client, ["Sawubona baba wami."], [4], align=False)
    reply = _ask(client, ["Sawubona baba wami."], [4], align=True)
    assert len(service.requests) == 2 and "words" in reply                                             # audio without words does not answer a request for words
    assert len(_ask(client, ["Sawubona baba wami."], [4], align=True)["words"]) == 1 and len(service.requests) == 2
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: {k: v for k, v in service(url, headers, body).items() if k not in ("words", "aligned")})
    mixed = _ask(client, ["Sawubona baba wami.", "Kunjalo nje yena."], [4, 9])
    assert "words" not in mixed                                                                        # the missing span's words cannot be had: the caller is told the way it already is


def test_the_cache_is_kept_under_its_size_by_dropping_the_span_used_longest_ago(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, max_bytes=60_000)
    for seed in range(1, 8):
        _ask(client, ["Sawubona baba wami nonke."], [seed], align=False)
    sizes = sum(p.stat().st_size for p in (tmp_path / "cache").glob("*.npz"))
    assert sizes <= 60_000
    before = len(service.requests)
    _ask(client, ["Sawubona baba wami nonke."], [7], align=False)                                       # the latest is still there
    assert len(service.requests) == before


def test_a_client_without_a_cache_asks_the_service_every_time_and_the_environment_switches_it_on_and_off(tmp_path, monkeypatch):
    service = FakeService()
    monkeypatch.setattr(dub_agent, "post_with_retry", service)
    plain = ot.OmniVoiceClient("http://voice")
    monkeypatch.setattr(plain, "_id_token", lambda: "t")
    _ask(plain, ["Sawubona."], [1]); _ask(plain, ["Sawubona."], [1])
    assert len(service.requests) == 2
    on = ot.OmniVoiceTTSBackend.from_environment({"OMNIVOICE_URL": "http://v", "OMNIVOICE_CACHE_DIR": str(tmp_path / "c"), "OMNIVOICE_CACHE_TAG": "x"})
    assert on.client.cache is not None and on.client.cache_tag == "x"
    off = ot.OmniVoiceTTSBackend.from_environment({"OMNIVOICE_URL": "http://v", "OMNIVOICE_CACHE_DIR": str(tmp_path / "c"), "OMNIVOICE_CACHE": "0"})
    assert off.client.cache is None
    assert ot.OmniVoiceTTSBackend.from_environment({"OMNIVOICE_URL": "http://v"}).client.cache is None      # no folder given and no real environment: no cache
    other_tag = oc.span_key(text="a", language="zu", reference="r", duration=None, speed=1.0, lora=True, generation={}, seed=1, tag="y")
    assert other_tag != oc.span_key(text="a", language="zu", reference="r", duration=None, speed=1.0, lora=True, generation={}, seed=1, tag="x")      # a new tag is a new voice


def test_a_units_spans_are_spoken_side_by_side_on_as_many_copies_as_are_held_and_the_reply_is_the_same_as_from_one(tmp_path, monkeypatch):
    import threading
    import time

    service = FakeService()
    in_flight, peak, lock = [0], [0], threading.Lock()

    def slow(url, headers, body, **kw):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        try:
            return service(url, headers, body)
        finally:
            with lock:
                in_flight[0] -= 1

    monkeypatch.setattr(dub_agent, "post_with_retry", slow)
    spans = ["Sawubona baba wami.", "Kunjalo nje yena.", "Siyabonga kakhulu bo.", "Hamba kahle."]
    seeds = [1, 2, 3, 4]
    one = ot.OmniVoiceClient("http://voice")
    monkeypatch.setattr(one, "_id_token", lambda: "t")
    many = ot.OmniVoiceClient("http://voice")
    monkeypatch.setattr(many, "_id_token", lambda: "t")
    many.set_parallel(3)
    alone = _ask(one, spans, seeds, gaps=[120, 240, 360])
    peak[0] = 0
    together = _ask(many, spans, seeds, gaps=[120, 240, 360])
    assert peak[0] == 3 and together["copies"] == 3                                                   # three requests in the air, never more than the copies
    assert len([r for r in service.requests]) == 1 + 3                                                # one request alone, three shares together
    for a, b in zip(_segments(alone), _segments(together)):
        assert abs(len(a) - len(b)) <= 24 and _same_up_to_a_shift(a, b)                              # every span is the span it would have been
    assert together["words"] == alone["words"] and all(abs(x - y) < 0.003 for x, y in zip(together["span_starts_s"], alone["span_starts_s"]))
    assert sorted(i for share in [r["spans"] for r in service.requests[1:]] for i in range(len(share))) and sum(len(r["spans"]) for r in service.requests[1:]) == 4


def test_the_cache_and_the_copies_work_together_only_the_missing_spans_are_shared_out(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    client.set_parallel(4)
    spans = ["Sawubona baba wami.", "Kunjalo nje yena.", "Siyabonga kakhulu bo.", "Hamba kahle."]
    _ask(client, spans, [1, 2, 3, 4])
    n = len(service.requests)
    _ask(client, spans, [1, 2, 3, 4])
    assert len(service.requests) == n                                                                  # all cached: nothing asked
    _ask(client, [spans[0], "Kunjalo nje futhi yena.", spans[2], "Hamba kahle impela."], [1, 2, 3, 4])
    assert sum(len(r["spans"]) for r in service.requests[n:]) == 2                                      # two spans changed: two spans asked, on two copies


def test_copies_are_held_by_leases_the_minimum_is_the_largest_live_one_and_a_dead_lease_is_dropped(tmp_path):
    from mathula_tv import omnivoice_scale as sc

    calls = []
    held = [0]

    def fake(method, url, body=None, **kw):
        calls.append((method, body))
        if method == "PATCH":
            held[0] = body["scaling"]["minInstanceCount"]
        return {"scaling": {"minInstanceCount": held[0]}}

    say = lambda m: None
    assert sc.acquire(3, tmp_path, call=fake, progress=say) == 3 and held[0] == 3
    leases = sc._read(tmp_path / sc.LEASE_FILE)
    leases["999999"] = {"pid": 999999, "host": sc.socket.gethostname(), "min": 4, "at": sc.time.time()}          # a run that died holding four
    sc._write(tmp_path / sc.LEASE_FILE, leases)
    assert "999999" not in sc.live_leases(tmp_path)                                                    # its process is gone: not honoured
    assert sc.release(tmp_path, call=fake, progress=say) == 0 and held[0] == 0                          # nothing live holds it any more
    held[0] = 2                                                                                         # left held by a run that died
    assert sc.release_if_stale(tmp_path, call=fake, progress=say) is True and held[0] == 0
    held[0] = 0
    assert sc.release_if_stale(tmp_path, call=fake, progress=say) is False
    old = {"1": {"pid": sc.os.getpid(), "host": sc.socket.gethostname(), "min": 4, "at": sc.time.time() - sc.MAX_AGE_S - 5}}
    sc._write(tmp_path / sc.LEASE_FILE, old)
    assert sc.live_leases(tmp_path) == {}                                                              # too old
    assert sc.set_min(9, call=fake) == sc.MAX_COPIES and held[0] == sc.MAX_COPIES                       # never more than the service's own cap

