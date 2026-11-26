import json

import pytest

from mathula_tv import zulu_rules
from mathula_tv.console import Console

JOB = "abcd1234" + "0" * 24
CLIP = JOB[:8]
THANDO, THEMBA = "zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"
PASSWORD = "correct horse battery"


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv(zulu_rules.RULES_CONFIG_ENV, str(tmp_path / "rules_config.json"))
    work = tmp_path / "work"
    (work / "resources" / "cmudict").mkdir(parents=True)
    (work / "resources" / "cmudict" / "cmudict.dict").write_text("jacob JH EY1 K AH0 B\nheidi HH AY1 D IY0\ncourt K AO1 R T\n", encoding="utf-8")
    (work / "experience" / "runs" / "prod-1" / CLIP / "r1").mkdir(parents=True)
    (work / "experience" / "corpus.json").write_text(json.dumps({"schema_version": "mathula-experience-corpus-v1", "clips": {CLIP: {"job_id": JOB}}}), encoding="utf-8")
    items = [{"item_id": "u1", "speaker_id": "S0", "zulu_text": "Sawubona Heidi. UJacob Zuma ukhulumile ne-SABC namuhla."},
             {"item_id": "u2", "speaker_id": "S1", "zulu_text": "Jacob uthe akazukuphendula."}]
    (work / "experience" / "runs" / "prod-1" / CLIP / "r1" / "submission.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    (work / "jobs" / JOB / "direct_dub").mkdir(parents=True)
    (work / "jobs" / JOB / "manifest.json").write_text(json.dumps({"source_filename": "madlanga.mp4", "created_at": "2026-11-02T10:00:00+00:00"}), encoding="utf-8")
    (work / "jobs" / JOB / "direct_dub" / "voice_resolution.json").write_text(json.dumps({"assignments": {"S0": {"selected_voice": THANDO}, "S1": {"selected_voice": THEMBA}}}), encoding="utf-8")
    spoken = []

    def fake_synth(*, job_root, speaker, voice, text, target):
        spoken.append((voice, text))
        target.write_bytes(b"RIFFfake")

    console = Console(work, synthesize=fake_synth)
    console.set_password(PASSWORD)
    console.spoken = spoken
    return console


def call(console, method, path, body=None, cookie=None, csrf=True):
    headers = {}
    if cookie:
        headers["Cookie"] = f"session={cookie}"
    if csrf and method != "GET":
        headers["X-Requested-With"] = "console"
    status, response_headers, payload = console.handle(method, path, headers, json.dumps(body).encode() if body is not None else b"")
    return status, response_headers, (json.loads(payload) if response_headers.get("Content-Type", "").startswith("application/json") else payload)


def signed_in(console):
    status, headers, _ = call(console, "POST", "/api/login", {"password": PASSWORD})
    assert status == 200 and "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"] and "Secure" in headers["Set-Cookie"]
    return headers["Set-Cookie"].split(";")[0].split("=", 1)[1]


def test_nothing_but_the_page_answers_before_sign_in_and_the_page_carries_a_strict_policy(world):
    status, headers, body = call(world, "GET", "/")
    assert status == 200 and b"Pronunciation Console" in body
    nonce = headers["Content-Security-Policy"].split("nonce-")[1].split("'")[0]
    assert f'nonce="{nonce}"'.encode() in body and "default-src 'none'" in headers["Content-Security-Policy"] and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    for path in ("/api/jobs", "/api/book", "/api/rules", "/api/session", "/audio/" + "a" * 32 + ".wav"):
        assert call(world, "GET", path)[0] == 401
    assert call(world, "POST", "/api/correct", {"word": "x"}, csrf=False)[0] == 403          # no X-Requested-With: not from the page


def test_a_wrong_password_is_refused_and_five_wrong_ones_lock_the_door_for_ten_minutes(world, monkeypatch):
    monkeypatch.setattr("mathula_tv.console.time.sleep", lambda _s: None)
    for _ in range(5):
        assert call(world, "POST", "/api/login", {"password": "nope-nope-nope"})[0] == 401
    status, _, body = call(world, "POST", "/api/login", {"password": PASSWORD})
    assert status == 429 and "ten minutes" in body["error"]


def test_a_session_cookie_expires_and_cannot_be_forged(world):
    cookie = signed_in(world)
    assert call(world, "GET", "/api/session", cookie=cookie)[0] == 200
    assert call(world, "GET", "/api/session", cookie=cookie[:-4] + "0000")[0] == 401
    assert call(world, "GET", "/api/session", cookie="9999999999.abc.def")[0] == 401
    world.clock = lambda: 4_000_000_000                                                  # long after it was issued
    assert call(world, "GET", "/api/session", cookie=cookie)[0] == 401


def test_the_password_is_stored_only_as_a_hash_and_must_be_long_enough(world):
    stored = (world.dir / "auth.json").read_text(encoding="utf-8")
    assert PASSWORD not in stored and "hash" in stored
    with pytest.raises(ValueError):
        world.set_password("short")


def test_the_review_lists_the_dubs_and_the_words_the_rules_found_with_what_they_say(world):
    cookie = signed_in(world)
    jobs = call(world, "GET", "/api/jobs", cookie=cookie)[2]
    assert jobs[0]["job"] == JOB and jobs[0]["name"] == "madlanga.mp4" and jobs[0]["runs"][0]["run"] == "prod-1"
    words = {w["word"]: w for w in call(world, "GET", f"/api/words?job={JOB}&run=prod-1", cookie=cookie)[2]}
    assert {"Jacob", "Heidi", "SABC"} <= set(words) and words["Jacob"]["status"] == "unreviewed" and words["Jacob"]["spelling"] == "jeykob" and words["Jacob"]["times"] == 2
    detail = call(world, "GET", f"/api/word?job={JOB}&run=prod-1&word=Jacob", cookie=cookie)[2]
    assert detail["rules"] == "jeykob" and detail["derivation"] and detail["alternatives"][0]["spelling"] == "jeykob" and detail["alternatives"][0]["differences"] == []
    assert any(a["spelling"] == "jeykab" and a["differences"][0]["rule"] == "letter_vowel" for a in detail["alternatives"])
    assert {s["voice"] for s in detail["samples"]} == {THANDO, THEMBA}
    assert call(world, "GET", f"/api/word?job={JOB}&run=prod-1&word=Nobody", cookie=cookie)[0] == 404
    assert call(world, "GET", "/api/words?job=../../etc&run=prod-1", cookie=cookie)[0] == 400


def test_hearing_a_spelling_says_the_real_sentence_with_it_in_the_right_voice_and_a_clip_already_made_is_free(world):
    cookie = signed_in(world)
    body = {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykab", "voice": THEMBA, "sample": 0}
    status, _, first = call(world, "POST", "/api/audio", body, cookie)
    assert status == 200 and first["url"].startswith("/audio/") and "jeykab" in first["text"] and "Jacob" not in first["text"] and world.spoken[0][0] == THEMBA
    again = call(world, "POST", "/api/audio", body, cookie)[2]
    assert again["url"] == first["url"] and len(world.spoken) == 1
    clip = call(world, "GET", first["url"], cookie=cookie)
    assert clip[0] == 200 and clip[2] == b"RIFFfake" and clip[1]["Content-Type"] == "audio/wav"
    assert call(world, "GET", "/audio/../../auth.json", cookie=cookie)[0] == 404
    assert call(world, "POST", "/api/audio", {**body, "voice": "en-US-Someone"}, cookie)[0] == 400
    assert call(world, "POST", "/api/audio", {**body, "spelling": "<script>"}, cookie)[0] == 400


def test_a_clip_is_served_in_byte_ranges_because_iphone_safari_asks_for_them(world):
    clip = bytes(range(100))
    reply = world._audio_reply
    assert reply({}, clip, None)[0] == 200 and reply({}, clip, None)[1]["Accept-Ranges"] == "bytes"
    status, headers, body = reply({}, clip, "bytes=0-1")                      # Safari's first probe
    assert (status, body, headers["Content-Range"]) == (206, clip[:2], "bytes 0-1/100")
    assert reply({}, clip, "bytes=90-")[2] == clip[90:] and reply({}, clip, "bytes=-10")[2] == clip[90:]
    assert reply({}, clip, "bytes=10-999")[1]["Content-Range"] == "bytes 10-99/100"
    assert reply({}, clip, "bytes=200-")[0] == 416 and reply({}, clip, "garbage")[0] == 200
    page = call(world, "GET", "/")[1]["Content-Security-Policy"]
    assert "media-src 'self' data:" in page                                  # the silent clip that unlocks the player on a phone


def test_there_is_a_ceiling_on_new_clips_a_day(world):
    world.max_audio_per_day = 2
    cookie = signed_in(world)
    for spelling in ("jeykob", "jeykab"):
        assert call(world, "POST", "/api/audio", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": spelling, "voice": THANDO}, cookie)[0] == 200
    status, _, body = call(world, "POST", "/api/audio", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jakobe", "voice": THANDO}, cookie)
    assert status == 429 and "limit" in body["error"]
    assert call(world, "POST", "/api/audio", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykob", "voice": THANDO}, cookie)[0] == 200          # again: free


def test_correct_commits_the_spelling_and_the_next_job_uses_it(world):
    cookie = signed_in(world)
    call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykob"}, cookie)
    entry = call(world, "GET", "/api/book", cookie=cookie)[2]["entries"][0]
    assert entry["word"] == "Jacob" and entry["kind"] == "approved" and entry["spelling"] == {"*": "jeykob"}
    words = {w["word"]: w for w in call(world, "GET", f"/api/words?job={JOB}&run=prod-1", cookie=cookie)[2]}
    assert words["Jacob"]["status"] == "approved"
    from mathula_tv import job_pronunciations as jp
    from mathula_tv import pronunciation_book as book

    done = jp.with_committed_words([], book.load(world.book_path)["entries"], ["UJacob uthe"])
    assert done[0]["by_voice"] == {THANDO: "jeykob", THEMBA: "jeykob"}
    assert call(world, "POST", "/api/book/revoke", {"word": "Jacob"}, cookie)[2] == {"revoked": True}


def test_a_spelling_of_the_listeners_own_is_committed_and_kept_as_a_lesson_for_the_rules(world):
    cookie = signed_in(world)
    status, _, entry = call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jakobe", "note": "Zulu say it with e"}, cookie)
    assert status == 200 and entry["kind"] == "custom" and entry["spelling"] == {"*": "jakobe"} and entry["rule_spelling"] == "jeykob"
    book_view = call(world, "GET", "/api/book", cookie=cookie)[2]
    lesson = book_view["feedback"][0]
    assert lesson["word"] == "Jacob" and lesson["chosen"] == "jakobe" and lesson["rule_spelling"] == "jeykob" and lesson["note"] == "Zulu say it with e"
    assert lesson["english"] and lesson["derivation"] and lesson["reading"] and lesson["differences"] == []          # the sounds, how the rules built it; not one of the rules' alternatives
    assert lesson["diff"] == [{"from": "jeykob", "to": "jakobe", "shape": "jV1ykV1b → jV1kV1bV1", "where": "only syllable", "pattern": "ey→a, ∅→e (inside)"}]
    assert book_view["tally"]["custom"] == 1 and "exceptions" not in book_view["tally"]
    assert call(world, "POST", "/api/not-correct", {}, cookie)[0] == 404          # no separate exception path any more
    assert call(world, "GET", "/api/rules", cookie=cookie)[2]["rules"][0]["default"] is None          # committing does not change a rule by itself
    again = call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jakobe", "note": "still"}, cookie)[2]
    assert len(call(world, "GET", "/api/book", cookie=cookie)[2]["feedback"]) == 1 and again["kind"] == "custom"          # the same lesson is updated, not repeated


def test_an_alternative_of_the_rules_records_which_open_choice_it_took_and_the_rules_page_can_make_it_the_default(world):
    cookie = signed_in(world)
    status, _, entry = call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykab", "note": "o is wrong"}, cookie)
    assert status == 200 and entry["kind"] == "custom"
    row = call(world, "GET", "/api/book", cookie=cookie)[2]["feedback"][0]
    assert [d["rule"] for d in row["differences"]] == ["letter_vowel"]
    rules_view = call(world, "GET", "/api/rules", cookie=cookie)[2]
    letter = next(r for r in rules_view["rules"] if r["rule"] == "letter_vowel")
    assert letter["wanted"] == {"0": 1} and letter["words"] == {"0": ["Jacob"]} and letter["as_written"] == 1 and letter["default"] is None
    assert rules_view["lessons"] == 1 and rules_view["analysis"] is None and rules_view["patterns"][0]["words"] == ["Jacob"]
    changed = call(world, "POST", "/api/rules", {"rule": "letter_vowel", "pick": 0, "note": "Jacob and Cyril both"}, cookie)[2]
    assert next(r for r in changed["rules"] if r["rule"] == "letter_vowel")["default"] == 0 and changed["history"][0]["note"] == "Jacob and Cyril both"
    assert zulu_rules.rule_defaults() == {"letter_vowel": 0}
    assert call(world, "POST", "/api/rules", {"rule": "letter_vowel", "pick": 9}, cookie)[0] == 400
    assert call(world, "POST", "/api/rules", {"rule": "letter_vowel", "pick": None}, cookie)[2]["rules"][0] is not None and zulu_rules.rule_defaults() == {}


def test_the_same_change_in_different_words_shows_as_one_pattern_and_the_model_is_asked_once_with_the_lessons(world):
    cookie = signed_in(world)
    assert call(world, "POST", "/api/learn", {}, cookie)[0] == 400          # nothing to learn from yet
    asked = []

    def analyst(payload):
        asked.append(payload)
        return {"summary": "the last letter is written longer", "findings": [{"pattern": "V3r -> V2rr", "why": "x", "words": ["Jacob"], "proposed_change": "y", "open_choice": None, "confidence": "low"}]}

    world._analyse = analyst
    call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykobb", "note": "final b is soft"}, cookie)
    call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Heidi", "spelling": "haydii"}, cookie)
    rules_view = call(world, "GET", "/api/rules", cookie=cookie)[2]
    assert rules_view["lessons"] == 2 and rules_view["patterns"] and rules_view["patterns"][0]["words"]
    record = call(world, "POST", "/api/learn", {}, cookie)[2]
    assert record["lessons"] == 2 and record["result"]["summary"] and len(asked) == 1
    payload = asked[0]
    assert {l["word"] for l in payload["lessons"]} == {"Jacob", "Heidi"} and payload["lessons"][0]["english_sounds"] and payload["the_rules"] and "letter_vowel" in payload["open_choices"]
    assert next(l for l in payload["lessons"] if l["word"] == "Jacob")["listener_note"] == "final b is soft"
    assert call(world, "GET", "/api/rules", cookie=cookie)[2]["analysis"]["result"]["findings"][0]["confidence"] == "low"          # kept for the page
    world.MAX_ANALYSES_PER_DAY = 1
    assert call(world, "POST", "/api/learn", {}, cookie)[0] == 429 and len(asked) == 1


def test_a_server_error_is_never_shown_to_the_browser(world, monkeypatch):
    cookie = signed_in(world)
    monkeypatch.setattr(world, "jobs", lambda: 1 / 0)
    status, _, body = call(world, "GET", "/api/jobs", cookie=cookie)
    assert status == 500 and body == {"error": "something went wrong on the server"}


def test_the_clips_of_the_next_words_are_made_ahead_in_both_voices_and_a_clip_made_ahead_is_free_to_play(world):
    cookie = signed_in(world)
    status, _, body = call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob", "Heidi"]}, cookie)
    assert status == 200 and body == {"queued": 4}                                    # two words, two voices each
    assert world.wait_idle()
    made = sorted((voice, text) for voice, text in world.spoken)
    assert {v for v, _ in made} == {THANDO, THEMBA} and sum("jeykob" in t for _, t in made) == 2 and sum("haydi" in t for _, t in made) == 2
    before = len(world.spoken)
    assert call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob", "Heidi"]}, cookie)[2] == {"queued": 0}          # nothing to do again
    play = call(world, "POST", "/api/audio", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykob", "voice": THEMBA}, cookie)
    assert play[0] == 200 and len(world.spoken) == before                              # the clip was there: no new synthesis


def test_the_ready_check_says_which_clips_are_made_and_pressing_not_correct_makes_every_alternative(world):
    cookie = signed_in(world)
    items = [{"spelling": "jeykob", "voice": THANDO}, {"spelling": "jeykab", "voice": THANDO}]
    assert call(world, "POST", "/api/ready", {"job": JOB, "run": "prod-1", "word": "Jacob", "items": items}, cookie)[2] == {"ready": [False, False]}
    queued = call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob"], "alternatives": True}, cookie)[2]["queued"]
    assert queued == 16                                                                # the rules' choice and seven alternatives, in two voices
    assert world.wait_idle(10)
    assert call(world, "POST", "/api/ready", {"job": JOB, "run": "prod-1", "word": "Jacob", "items": items}, cookie)[2] == {"ready": [True, True]}
    assert call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Nobody"]}, cookie)[2] == {"queued": 0}


def test_making_clips_ahead_stops_quietly_at_the_days_ceiling_and_never_needs_the_listener_to_wait(world):
    world.max_audio_per_day = 3
    cookie = signed_in(world)
    call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob", "Heidi", "SABC"]}, cookie)
    assert world.wait_idle(10) and len(world.spoken) == 3
    status, _, body = call(world, "POST", "/api/audio", {"job": JOB, "run": "prod-1", "word": "SABC", "spelling": "es-eyi-bi-siii", "voice": THANDO}, cookie)
    assert status == 429 and "limit" in body["error"]
    assert call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob"]})[0] == 401          # not signed in


def test_a_clip_that_cannot_be_made_does_not_stop_the_ones_after_it(world):
    original = world._synthesize
    calls = []

    def flaky(*, job_root, speaker, voice, text, target):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("the voice service said no")
        original(job_root=job_root, speaker=speaker, voice=voice, text=text, target=target)

    world._synthesize = flaky
    cookie = signed_in(world)
    call(world, "POST", "/api/prefetch", {"job": JOB, "run": "prod-1", "words": ["Jacob", "Heidi"]}, cookie)
    assert world.wait_idle(10) and len(calls) == 4 and len(world.spoken) == 3


def test_wordsmith_gives_a_typed_word_in_both_voices_spelled_by_the_rules_with_the_alternatives(world):
    cookie = signed_in(world)
    result = call(world, "POST", "/api/smith", {"text": "Jacob"}, cookie)[2]
    assert result["source"] == "rules" and result["spelling"] == {THANDO: "jeykob", THEMBA: "jeykob"} and result["voices"] == [THANDO, THEMBA]
    assert result["pieces"][0]["derivation"] and result["alternatives"][0]["spelling"] == "jeykob" and any(a["spelling"] == "jeykab" for a in result["alternatives"])
    heard = [call(world, "POST", "/api/smith-audio", {"spelling": result["spelling"][v], "voice": v}, cookie) for v in (THANDO, THEMBA)]
    assert all(status == 200 for status, _, _ in heard) and heard[0][2]["url"] != heard[1][2]["url"]
    assert world.spoken == [(THANDO, "jeykob"), (THEMBA, "jeykob")]
    assert call(world, "POST", "/api/smith-audio", {"spelling": "jeykob", "voice": THANDO}, cookie)[2]["url"] == heard[0][2]["url"] and len(world.spoken) == 2          # again: free
    assert call(world, "GET", heard[0][2]["url"], cookie=cookie)[2] == b"RIFFfake"


def test_wordsmith_uses_the_book_leaves_unknown_words_and_typed_words_alone_and_spells_a_phrase_word_by_word(world):
    cookie = signed_in(world)
    call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jakobe"}, cookie)
    booked = call(world, "POST", "/api/smith", {"text": "jacob"}, cookie)[2]
    assert booked["source"] == "book" and booked["spelling"][THANDO] == "jakobe"
    assert call(world, "POST", "/api/smith", {"text": "Jacob", "as_typed": True}, cookie)[2]["spelling"][THANDO] == "Jacob"
    zulu = call(world, "POST", "/api/smith", {"text": "umengameli"}, cookie)[2]
    assert zulu["source"] == "as typed" and zulu["spelling"][THEMBA] == "umengameli" and zulu["alternatives"] == []
    phrase = call(world, "POST", "/api/smith", {"text": "Heidi, i-SABC umengameli."}, cookie)[2]
    assert phrase["source"] == "mixed" and len(phrase["pieces"]) == 3 and phrase["alternatives"] == []
    assert phrase["spelling"][THANDO].startswith("haydi,") and phrase["spelling"][THANDO].endswith("umengameli.") and "i-" in phrase["spelling"][THANDO]


def test_wordsmith_needs_a_session_refuses_bad_input_and_counts_against_the_daily_ceiling(world):
    assert call(world, "POST", "/api/smith", {"text": "Jacob"})[0] == 401
    assert call(world, "POST", "/api/smith-audio", {"spelling": "jeykob", "voice": THANDO})[0] == 401
    cookie = signed_in(world)
    assert call(world, "POST", "/api/smith", {"text": "  "}, cookie)[0] == 400
    assert call(world, "POST", "/api/smith", {"text": "x" * 81}, cookie)[0] == 400
    assert call(world, "POST", "/api/smith-audio", {"spelling": "<script>", "voice": THANDO}, cookie)[0] == 400
    assert call(world, "POST", "/api/smith-audio", {"spelling": "jeykob", "voice": "en-US-Someone"}, cookie)[0] == 400
    world.max_audio_per_day = 1
    assert call(world, "POST", "/api/smith-audio", {"spelling": "jeykob", "voice": THANDO}, cookie)[0] == 200
    assert call(world, "POST", "/api/smith-audio", {"spelling": "jeykab", "voice": THANDO}, cookie)[0] == 429


def test_wordsmith_commits_what_was_heard_as_approved_or_an_exception_for_one_voice_or_both_and_takes_it_back(world):
    cookie = signed_in(world)
    smith = call(world, "POST", "/api/smith", {"text": "Jacob"}, cookie)[2]
    assert smith["committable"] is True
    entry = call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "jeykob", THEMBA: "jeykob"}}, cookie)[2]
    assert entry["kind"] == "approved" and entry["spelling"] == {"*": "jeykob"}
    after = call(world, "POST", "/api/smith", {"text": "jacob"}, cookie)[2]
    assert after["source"] == "book" and after["spelling"][THEMBA] == "jeykob"
    entry = call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "jeykob", THEMBA: "jakobe"}}, cookie)[2]          # the voices differ: each its own
    assert entry["spelling"] == {THANDO: "jeykob", THEMBA: "jakobe"} and entry["rule_spelling"] == "jeykob"
    assert call(world, "POST", "/api/smith", {"text": "Jacob"}, cookie)[2]["spelling"] == {THANDO: "jeykob", THEMBA: "jakobe"}
    entry = call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "jakobe", THEMBA: "jakobe"}}, cookie)[2]          # one spelling again: the per-voice ones go
    assert entry["kind"] == "custom" and entry["spelling"] == {"*": "jakobe"}
    assert "Jacob" in {e["word"] for e in call(world, "GET", "/api/book", cookie=cookie)[2]["entries"]}
    assert call(world, "POST", "/api/book/revoke", {"word": "Jacob"}, cookie)[2]["revoked"] is True
    assert call(world, "POST", "/api/smith", {"text": "Jacob"}, cookie)[2]["source"] == "rules"


def test_wordsmith_commits_only_a_single_bare_word_and_only_with_a_session(world):
    assert call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "jeykob", THEMBA: "jeykob"}})[0] == 401
    cookie = signed_in(world)
    for text in ("i-SABC", "Heidi, Jacob", "Jacob,"):
        assert call(world, "POST", "/api/smith", {"text": text}, cookie)[2]["committable"] is False
    spellings = {THANDO: "jeykob", THEMBA: "jeykob"}
    assert call(world, "POST", "/api/smith-commit", {"word": "Heidi Jacob", "spelling": spellings}, cookie)[0] == 400
    assert call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "jeykob"}}, cookie)[0] == 400          # a spelling for each voice
    assert call(world, "POST", "/api/smith-commit", {"word": "Jacob", "spelling": {THANDO: "<b>", THEMBA: "jeykob"}}, cookie)[0] == 400
    assert call(world, "GET", "/api/book", cookie=cookie)[2]["entries"] == []


def test_the_models_answer_contract_is_one_the_shared_client_can_send():
    from mathula_tv.foundry_grok import _compact_schema_contract

    contract = _compact_schema_contract(Console.ANALYSIS_SCHEMA)
    assert "findings" in contract and "summary" in contract


def _source_job(world, words, audio=True):
    analysis = world.work_dir / "jobs" / JOB / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "corrected_transcript.json").write_text(json.dumps({"words": [
        {"text": t, "start": s, "end": e, "confidence": c, "speaker": "AZURE_1"} for t, s, e, c in words]}), encoding="utf-8")
    if audio:
        (world.work_dir / "jobs" / JOB / "audio").mkdir(parents=True, exist_ok=True)
        (world.work_dir / "jobs" / JOB / "audio" / "analysis_mono.wav").write_bytes(b"RIFFsource")
    cuts = []
    world._cut = lambda audio_path, start, duration, target: (cuts.append((round(start, 2), round(duration, 2))), target.write_bytes(b"RIFFcut"))
    return cuts


SAID = [("Good", 1.0, 1.3, .9), ("evening,", 1.3, 1.9, .9), ("Jacob", 2.0, 2.5, .6), ("Zuma", 2.5, 3.0, .9), ("said", 3.0, 3.3, .9), ("that", 3.3, 3.5, .9),
        ("Jacob's", 20.0, 20.5, .95), ("friends", 20.5, 21.0, .9), ("and", 21.0, 21.2, .9), ("then", 21.2, 21.5, .9),
        ("Jacob.", 22.0, 22.4, .99), ("Later", 40.0, 40.4, .9), ("JACOB", 60.0, 60.5, .8)]


def test_the_source_audio_finds_where_a_word_is_said_best_recognised_first_and_not_twice_in_one_breath(world):
    _source_job(world, SAID)
    cookie = signed_in(world)
    found = call(world, "GET", f"/api/source?job={JOB}&word=jacob", cookie=cookie)[2]["occurrences"]
    assert [o["index"] for o in found] == [2, 6, 12] and [o["at"] for o in found] == ["0:02", "0:20", "1:00"]          # in the order said; the 22.0 "Jacob." is within 8 s of the 20.0 one, so it is left out
    assert all("confidence" not in o for o in found)          # nothing is scored
    first = found[0]
    assert first["word"] == "Jacob" and first["pre"].endswith("evening,") and first["post"].startswith("Zuma") and first["job"] == JOB and first["job_name"] == "madlanga.mp4"
    assert call(world, "GET", f"/api/source?job={JOB}&word=Nobody", cookie=cookie)[2]["occurrences"] == []
    across = call(world, "GET", "/api/source?word=Zuma", cookie=cookie)[2]["occurrences"]          # no job: every job's transcript
    assert [o["index"] for o in across] == [3] and across[0]["job_name"] == "madlanga.mp4"
    assert call(world, "GET", "/api/source?job=../../x&word=Jacob", cookie=cookie)[0] == 400


def test_the_source_audio_is_cut_around_the_word_or_the_phrase_once_and_is_free(world):
    cuts = _source_job(world, SAID)
    cookie = signed_in(world)
    word = call(world, "POST", "/api/source-audio", {"job": JOB, "index": 2, "mode": "word"}, cookie)[2]
    assert word["url"].startswith("/audio/") and (word["start"], word["end"]) == (1.88, 2.62) and cuts == [(1.88, 0.74)]
    assert call(world, "GET", word["url"], cookie=cookie)[2] == b"RIFFcut"
    assert call(world, "POST", "/api/source-audio", {"job": JOB, "index": 2, "mode": "word"}, cookie)[2]["url"] == word["url"] and len(cuts) == 1
    phrase = call(world, "POST", "/api/source-audio", {"job": JOB, "index": 6, "mode": "phrase"}, cookie)[2]
    assert (phrase["start"], phrase["end"]) == (19.9, 22.5) and phrase["url"] != word["url"]          # Jacob's friends and then Jacob., which follow without a pause; not back across the silence to 3.5, nor on to "Later" after one
    spent = world._budget()[1]["count"]
    assert spent == 0                                                  # cutting is local: it never counts against the voice budget
    for body in ({"job": JOB, "index": 99, "mode": "word"}, {"job": JOB, "index": -1, "mode": "word"}, {"job": JOB, "index": 2, "mode": "all"}, {"job": JOB, "index": "x", "mode": "word"}):
        assert call(world, "POST", "/api/source-audio", body, cookie)[0] == 400


def test_the_source_audio_needs_a_session_and_a_job_with_audio(world):
    _source_job(world, SAID, audio=False)
    assert call(world, "GET", f"/api/source?job={JOB}&word=Jacob")[0] == 401
    assert call(world, "POST", "/api/source-audio", {"job": JOB, "index": 2, "mode": "word"})[0] == 401
    cookie = signed_in(world)
    assert call(world, "GET", f"/api/source?job={JOB}&word=Jacob", cookie=cookie)[2]["occurrences"] == []          # no audio: nothing to offer
    assert call(world, "POST", "/api/source-audio", {"job": JOB, "index": 2, "mode": "word"}, cookie)[0] == 404


def test_ffmpeg_cuts_the_real_window(tmp_path):
    import shutil
    import struct
    import wave

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    source = tmp_path / "s.wav"
    with wave.open(str(source), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", 3000) for _ in range(16000 * 3)))
    target = tmp_path / "t.wav"
    Console._ffmpeg_cut(source, 1.0, 0.5, target)
    with wave.open(str(target), "rb") as w:
        assert w.getframerate() == 24000 and abs(w.getnframes() / w.getframerate() - 0.5) < 0.05


def test_a_word_timed_very_short_still_gets_a_clip_long_enough_to_hear(world):
    cuts = _source_job(world, [("Ralphs,", 1117.97, 1118.13, .16), ("Ralphs,", 1118.13, 1118.33, .16)])
    cookie = signed_in(world)
    clip = call(world, "POST", "/api/source-audio", {"job": JOB, "index": 0, "mode": "word"}, cookie)[2]
    assert round(clip["end"] - clip["start"], 2) == world.SOURCE_MIN_WORD and cuts[0][1] == world.SOURCE_MIN_WORD and clip["start"] < 1117.97 and clip["end"] > 1118.13


def test_a_recogniser_only_shows_what_it_hears_in_a_source_clip_once_and_nothing_else_uses_it(world):
    _source_job(world, SAID)
    heard = []

    def recognise(wav):
        heard.append(wav.name)
        return {"phones": "dʒ eɪ k ə b", "reads_as": "jacob"}

    world._recognise = recognise
    assert call(world, "POST", "/api/source-hear", {"job": JOB, "index": 2, "mode": "word"})[0] == 401
    cookie = signed_in(world)
    first = call(world, "POST", "/api/source-hear", {"job": JOB, "index": 2, "mode": "word"}, cookie)[2]
    assert first["phones"] == "dʒ eɪ k ə b" and first["reads_as"] == "jacob" and first["mode"] == "word" and first["url"].startswith("/audio/") and "confidence" not in first and "score" not in first
    assert call(world, "POST", "/api/source-hear", {"job": JOB, "index": 2, "mode": "word"}, cookie)[2] == first and len(heard) == 1          # once per clip
    call(world, "POST", "/api/source-hear", {"job": JOB, "index": 2, "mode": "phrase"}, cookie)
    assert len(heard) == 2
    assert call(world, "POST", "/api/source-hear", {"job": JOB, "index": 99, "mode": "word"}, cookie)[0] == 400
    assert call(world, "POST", "/api/source-hear", {"job": JOB, "index": 2, "mode": "x"}, cookie)[0] == 400
    assert world._budget()[1]["count"] == 0          # local: no voice is asked


def test_no_recogniser_is_reachable_from_the_agent_or_the_dub_pipeline_through_the_console():
    import inspect

    from mathula_tv import dub_agent

    source = inspect.getsource(dub_agent)
    for name in ("english_listener", "LocalGop", "phoneme_recognizer", "allosaurus", "pronunciation_assess"):
        assert name not in source


def test_the_recogniser_models_are_let_go_when_idle_and_load_again_when_asked(world, monkeypatch):
    loaded = []

    class Models:
        def __init__(self):
            loaded.append(1)

        def decode(self, wav):
            return ["a"]

        def transcribe(self, wav):
            return "a"

    import mathula_tv.english_listener as el
    import mathula_tv.pronunciation_assess as pa

    monkeypatch.setattr(pa, "LocalGop", Models)
    monkeypatch.setattr(el, "EnglishListener", Models)
    world.RECOGNISER_IDLE_SECONDS = 3600
    wav = world.work_dir / "x.wav"
    wav.write_bytes(b"x")
    assert world._recognise_local(wav) == {"phones": "a", "reads_as": "a"} and len(loaded) == 2 and world._recognisers is not None
    world._recognise_local(wav)
    assert len(loaded) == 2                                         # still loaded: not read twice
    assert world.unload_recognisers() is True and world._recognisers is None and world.unload_recognisers() is False
    world._recognise_local(wav)
    assert len(loaded) == 4                                         # asked again: loaded again
    world._unload_timer.cancel()


def _asked_job(world):
    """A dub whose Zulu holds a word the dictionary has but nobody said (Wehle), and a restored transcript with a stretch the captions do not support."""
    (world.work_dir / "resources" / "cmudict" / "cmudict.dict").write_text("jacob JH EY1 K AH0 B\nheidi HH AY1 D IY0\nwehle W EH1 L IY0\n", encoding="utf-8")
    path = world.work_dir / "experience" / "runs" / "prod-1" / CLIP / "r1" / "submission.json"
    items = json.loads(path.read_text())["items"]
    items[0].update({"source_segment_ids": ["seg-1"], "start_ms": 1000, "end_ms": 5000, "source_text": "Good evening Heidi and Jacob Zuma said that thing",
                     "zulu_text": "Sawubona Heidi. ~pause800 UJacob Zuma ukhulumile ne-SABC namuhla."})
    items.append({"item_id": "u3", "speaker_id": "S0", "zulu_text": "Wehle ngesenti elilodwa kuleli sonto.", "source_text": "It only went down by one cent this week.", "start_ms": 6000, "end_ms": 9000})
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    root = world.work_dir / "jobs" / JOB
    (root / "native_dub").mkdir(parents=True, exist_ok=True)
    segments = [{"segment_id": "seg-1", "start_ms": 1000, "end_ms": 5000, "speaker_id": "AZURE_1", "source_text": "Good evening Heidi and Jacob Zuma said that thing",
                 "restored_text": "Good evening Heidi and Jacob Zuma said that thing", "changed": False},
                {"segment_id": "seg-3", "start_ms": 6000, "end_ms": 9000, "speaker_id": "AZURE_1", "source_text": "It only went down by one cent this week",
                 "restored_text": "It only went down by one cent this week", "changed": False}]
    (root / "native_dub" / "pass1_restored_transcript.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    folder = root / "input" / "youtube_captions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "v.en-orig.json3").write_text(json.dumps({"events": [{"tStartMs": 900, "dDurationMs": 4300, "segs": [{"utf8": "sawubona bantu bakithi sengiyaphuma ngiyabonga kakhulu"}]},
                                                                  {"tStartMs": 6000, "dDurationMs": 3000, "segs": [{"utf8": "It only went down by one cent this week"}]}]}), encoding="utf-8")


def test_a_word_of_the_dub_nobody_said_is_asked_about_not_respelled_and_the_answer_decides(world):
    _asked_job(world)
    cookie = signed_in(world)
    asked = call(world, "GET", f"/api/questions?job={JOB}&run=prod-1", cookie=cookie)[2]
    assert [w["word"] for w in asked["words"]] == ["Wehle"] and asked["words"][0]["source"].startswith("It only went down") and "Wehle ngesenti" in asked["words"][0]["dub"]
    assert [s["id"] for s in asked["stretches"]] == ["seg-1"] and asked["open"] == 2 and "zulu" in asked["languages"]
    assert "Wehle" not in [w["word"] for w in call(world, "GET", f"/api/words?job={JOB}&run=prod-1", cookie=cookie)[2]]               # left as written until the listener says
    status, _, after = call(world, "POST", "/api/answer", {"job": JOB, "run": "prod-1", "kind": "word", "word": "Wehle", "language": "zulu"}, cookie=cookie)
    assert status == 200 and after["words"] == [] and [a["word"] for a in after["answered"]] == ["Wehle"] and after["answered"][0]["answer"]["language"] == "zulu" and after["open"] == 1
    assert "Wehle" not in [w["word"] for w in call(world, "GET", f"/api/words?job={JOB}&run=prod-1", cookie=cookie)[2]]               # Zulu: still left as written
    call(world, "POST", "/api/answer", {"job": JOB, "run": "prod-1", "kind": "word", "word": "Wehle", "language": "english"}, cookie=cookie)
    assert "Wehle" in [w["word"] for w in call(world, "GET", f"/api/words?job={JOB}&run=prod-1", cookie=cookie)[2]]                      # English: the rules spell it
    back = call(world, "POST", "/api/answer-revoke", {"job": JOB, "run": "prod-1", "word": "Wehle"}, cookie=cookie)[2]
    assert [w["word"] for w in back["words"]] == ["Wehle"]


def test_a_stretch_answer_is_kept_for_the_job_and_as_a_lesson_and_bad_answers_are_refused(world):
    _asked_job(world)
    cookie = signed_in(world)
    body = {"job": JOB, "run": "prod-1", "kind": "stretch", "id": "seg-1", "choice": "typed", "text": "Sawubona bantu bakithi"}
    assert call(world, "POST", "/api/answer", body, cookie=cookie)[0] == 200
    kept = json.loads((world.work_dir / "jobs" / JOB / "native_dub" / "language_answers.json").read_text())
    assert kept["stretches"]["seg-1"]["text"] == "Sawubona bantu bakithi"
    assert json.loads((world.work_dir / "language_lessons.json").read_text())["stretches"][0]["final"] == "Sawubona bantu bakithi"
    for bad in ({**body, "choice": "guess"}, {**body, "id": "seg-404"}, {**body, "id": "../x"}, {**body, "kind": "other"}, {**body, "text": ""}):
        assert call(world, "POST", "/api/answer", bad, cookie=cookie)[0] == 400
    assert call(world, "POST", "/api/answer", body, csrf=False, cookie=cookie)[0] == 403
    assert call(world, "GET", f"/api/questions?job={JOB}&run=prod-1")[0] == 401


def test_a_stretch_can_be_heard_from_the_job_source_audio_once_and_free(world):
    _asked_job(world)
    cuts = _source_job(world, SAID)
    cookie = signed_in(world)
    clip = call(world, "POST", "/api/segment-audio", {"job": JOB, "id": "seg-1"}, cookie=cookie)[2]
    assert clip["url"].startswith("/audio/") and clip["start"] == 0.85 and clip["end"] == 5.15 and cuts == [(0.85, 4.3)]
    assert call(world, "POST", "/api/segment-audio", {"job": JOB, "id": "seg-1"}, cookie=cookie)[2] == clip and len(cuts) == 1
    assert call(world, "POST", "/api/segment-audio", {"job": JOB, "id": "seg-404"}, cookie=cookie)[0] == 404


def test_every_question_brings_the_source_audio_the_youtube_caption_and_the_dub_to_read_raw(world):
    _asked_job(world)
    cookie = signed_in(world)
    q = call(world, "GET", f"/api/questions?job={JOB}&run=prod-1", cookie=cookie)[2]
    word = q["words"][0]
    assert word["word"] == "Wehle" and word["start_ms"] == 6000 and word["end_ms"] == 9000 and word["caption"] == "It only went down by one cent this week" and word["track"] == "en-orig"
    assert word["item"] == 2 and word["voice"] == THANDO and word["tts_text"] == "Wehle ngesenti elilodwa kuleli sonto."
    stretch = q["stretches"][0]
    assert stretch["id"] == "seg-1" and stretch["start_ms"] == 1000 and stretch["end_ms"] == 5000 and stretch["caption"].startswith("sawubona") and stretch["track"] == "en-orig"
    assert stretch["item"] == 0 and stretch["voice"] == THANDO and stretch["dub"].startswith("Sawubona Heidi.") and "~pause" in stretch["dub"] and "~pause" not in stretch["tts_text"]


def test_the_raw_tts_of_a_dub_item_is_its_zulu_as_written_in_its_speakers_voice_with_no_lexicon_and_no_book(world):
    _asked_job(world)
    roots = []
    original = world._synthesize
    world._synthesize = lambda **kw: (roots.append(kw["job_root"]), original(**kw))[1]
    cookie = signed_in(world)
    clip = call(world, "POST", "/api/raw-audio", {"job": JOB, "run": "prod-1", "item": 0}, cookie=cookie)[2]
    assert clip["voice"] == THANDO and clip["text"] == "Sawubona Heidi. UJacob Zuma ukhulumile ne-SABC namuhla." and world.spoken[-1] == (THANDO, clip["text"])
    assert roots[-1].name == world.SMITH_JOB                                                  # a job root with nothing in it: this job's lexicon is not applied
    assert call(world, "POST", "/api/raw-audio", {"job": JOB, "run": "prod-1", "item": 1}, cookie=cookie)[2]["voice"] == THEMBA
    assert call(world, "POST", "/api/raw-audio", {"job": JOB, "run": "prod-1", "item": 0}, cookie=cookie)[2] == clip and len(world.spoken) == 2          # made once, free after
    for bad in (7, -1, "0", None):
        assert call(world, "POST", "/api/raw-audio", {"job": JOB, "run": "prod-1", "item": bad}, cookie=cookie)[0] == 400
    assert call(world, "POST", "/api/raw-audio", {"job": JOB, "run": "prod-1", "item": 0}, csrf=False, cookie=cookie)[0] == 403


def test_the_source_audio_of_a_word_question_is_its_dub_items_window(world):
    _asked_job(world)
    cuts = _source_job(world, SAID)
    cookie = signed_in(world)
    clip = call(world, "POST", "/api/segment-audio", {"job": JOB, "start_ms": 6000, "end_ms": 9000}, cookie=cookie)[2]
    assert clip["start"] == 5.85 and clip["end"] == 9.15 and cuts == [(5.85, 3.3)]
    for bad in ({"start_ms": 9000, "end_ms": 6000}, {"start_ms": -1, "end_ms": 5}, {"start_ms": "a", "end_ms": 5}, {}):
        assert call(world, "POST", "/api/segment-audio", {"job": JOB, **bad}, cookie=cookie)[0] == 400


def test_a_comment_to_the_model_is_kept_with_or_without_an_answer_and_a_comment_alone_decides_nothing(world):
    _asked_job(world)
    cookie = signed_in(world)
    sent = call(world, "POST", "/api/answer", {"job": JOB, "run": "prod-1", "kind": "word", "word": "Wehle", "comment": "wehle is the verb ehla, it went down"}, cookie=cookie)[2]
    assert [w["word"] for w in sent["words"]] == ["Wehle"] and sent["words"][0]["comment"].startswith("wehle is the verb") and sent["words"][0]["answer"] is None and sent["open"] == 2
    again = call(world, "POST", "/api/answer", {"job": JOB, "run": "prod-1", "kind": "word", "word": "Wehle", "language": "zulu"}, cookie=cookie)[2]
    assert again["words"] == [] and again["answered"][0]["comment"].startswith("wehle is the verb")                                 # the comment stays when the answer comes
    body = {"job": JOB, "run": "prod-1", "kind": "stretch", "id": "seg-1"}
    asked = call(world, "POST", "/api/answer", {**body, "comment": "this is Zulu, he thanks the audience"}, cookie=cookie)[2]
    row = asked["stretches"][0]
    assert row["answer"] is None and row["comment"] == "this is Zulu, he thanks the audience" and asked["open"] == 1                  # still open: nothing was decided
    assert not (world.work_dir / "language_lessons.json").read_text().count('"stretches": [\n  {')                                  # and it is no lesson for other videos yet
    decided = call(world, "POST", "/api/answer", {**body, "choice": "transcript"}, cookie=cookie)[2]["stretches"][0]
    assert decided["answer"]["choice"] == "transcript" and decided["comment"] == "this is Zulu, he thanks the audience"
    lesson = json.loads((world.work_dir / "language_lessons.json").read_text())["stretches"][0]
    assert lesson["comment"] == "this is Zulu, he thanks the audience" and lesson["choice"] == "transcript"
    assert call(world, "POST", "/api/answer", {**body, "choice": "guess"}, cookie=cookie)[0] == 400
    assert call(world, "POST", "/api/answer", {"job": JOB, "run": "prod-1", "kind": "word", "word": "Nothing"}, cookie=cookie)[0] == 400          # neither a language nor a comment


def test_a_stretch_can_be_heard_from_a_job_that_has_audio_but_no_word_timings(world):
    _asked_job(world)
    (world.work_dir / "jobs" / JOB / "audio").mkdir(parents=True, exist_ok=True)
    (world.work_dir / "jobs" / JOB / "audio" / "mix_source_stereo.wav").write_bytes(b"RIFFsource")          # the native pipeline's jobs have no corrected_transcript.json
    cuts = []
    world._cut = lambda audio_path, start, duration, target: (cuts.append((audio_path.name, round(start, 2))), target.write_bytes(b"RIFFcut"))
    cookie = signed_in(world)
    assert call(world, "POST", "/api/segment-audio", {"job": JOB, "id": "seg-1"}, cookie=cookie)[0] == 200 and cuts == [("mix_source_stereo.wav", 0.85)]
    (world.work_dir / "jobs" / JOB / "audio" / "mix_source_stereo.wav").unlink()
    assert call(world, "POST", "/api/segment-audio", {"job": JOB, "id": "seg-9"}, cookie=cookie)[0] == 404


def test_a_commit_and_a_revoke_reach_the_jobs_lexicon_at_once(world):
    lexicon = world.work_dir / "jobs" / JOB / "native_dub" / "pronunciation_names.json"
    lexicon.parent.mkdir(parents=True, exist_ok=True)
    lexicon.write_text(json.dumps({"entries": [{"display_text": "Jacob", "tts_text": "Jacob", "tts_text_by_voice": {THANDO: "jeykhab", THEMBA: "jeykhab"}, "kind": "english_code_switch", "confidence": 0.9,
                                                "source": "rules", "reads_as": "word", "english": "x", "derivation": []}]}), encoding="utf-8")
    cookie = signed_in(world)
    spelled = lambda: json.loads(lexicon.read_text())["entries"][0]["tts_text_by_voice"]
    call(world, "POST", "/api/correct", {"job": JOB, "run": "prod-1", "word": "Jacob", "spelling": "jeykob"}, cookie)
    assert spelled() == {THANDO: "jeykob", THEMBA: "jeykob"}                                    # no production run in between
    assert call(world, "POST", "/api/book/revoke", {"word": "Jacob"}, cookie)[2] == {"revoked": True}
    assert spelled() == {THANDO: "jeykhab", THEMBA: "jeykhab"}



def test_a_pinned_word_is_offered_in_every_jobs_list_before_any_text_holds_it(tmp_path):
    import json as _json
    from mathula_tv import console

    work = tmp_path
    (work / "pinned_words.json").write_text(_json.dumps(["details"]), encoding="utf-8")
    c = console.Console.__new__(console.Console)
    c.work_dir = work
    assert c._pinned_words() == ["details"]
    (work / "pinned_words.json").write_text("not json", encoding="utf-8")
    assert c._pinned_words() == []


class _FakeOmni:
    """The voice backend as the console meets it: it plans a dub text into sentences, picks a reference clip and speaks sentences."""
    default_voice = "omnivoice-zu"

    def __init__(self):
        self.calls = []

    def plan(self, group, voice, corpus):
        import re

        spans = [s for s in re.split(r"(?<=[.]) ", group["parts"][0]["text"]) if s]
        return spans, [150] * (len(spans) - 1)

    def reference_for(self, group, corpus):
        from types import SimpleNamespace

        return SimpleNamespace(start_s=10.0, end_s=18.0)

    def speak(self, group, spans, gaps, corpus, retakes=None, takes_out=None, all_takes=False, raw_out=None):
        import numpy as np

        self.calls.append((list(spans), dict(retakes or {})))
        if takes_out is not None:
            takes_out[:] = [[np.full(2400, 0.1 * (t + 1), dtype=np.float32) for t in range(3)] for _ in spans]
        words = [{"span": i, "text": f"w{i}", "start_s": i + 0.1, "end_s": i + 0.6} for i in range(len(spans))]
        return np.zeros(24000 * len(spans), dtype=np.float32), 24000, {"span_starts_s": [float(i) for i in range(len(spans))]}, {"words": words, "picked": [0] * len(spans)}, self.reference_for(group, corpus)


@pytest.fixture()
def drawn(world, tmp_path):
    omni = _FakeOmni()
    world._omni_factory = lambda: omni
    world.omni = omni
    return world


def test_the_draws_tab_lists_each_sentence_with_its_retake_and_a_committed_draw_is_kept_as_a_number(drawn):
    cookie = signed_in(drawn)
    status, _, view = call(drawn, "GET", f"/api/draws?job={JOB}&run=prod-1", cookie=cookie)
    assert status == 200 and [u["id"] for u in view["units"]] == ["u1", "u2"]
    assert [s["text"] for s in view["units"][0]["spans"]] == ["Sawubona Heidi.", "UJacob Zuma ukhulumile ne-SABC namuhla."] and all(s["retake"] == 0 and not s["marked"] for u in view["units"] for s in u["spans"])
    status, _, done = call(drawn, "POST", "/api/draw-commit", {"job": JOB, "run": "prod-1", "unit": 0, "span": 1, "retake": 2, "note": "Jacob"}, cookie=cookie)
    assert status == 200 and done["mark"]["retake"] == 2 and done["mark"]["ref"] == [10.0, 18.0]
    _, _, view = call(drawn, "GET", f"/api/draws?job={JOB}&run=prod-1", cookie=cookie)
    second = view["units"][0]["spans"][1]
    assert second["marked"] and second["retake"] == 2 and not view["units"][0]["spans"][0]["marked"]
    saved = json.loads((drawn.work_dir / "experience" / "corpus" / CLIP / "retakes.json").read_text(encoding="utf-8"))
    assert len(saved["marks"]) == 1 and "ref" in next(iter(saved["marks"].values()))             # a number and a reference clip, no audio
    _, _, back = call(drawn, "POST", "/api/draw-revoke", {"job": JOB, "run": "prod-1", "unit": 0, "span": 1}, cookie=cookie)
    assert back["revoked"] is True
    _, _, view = call(drawn, "GET", f"/api/draws?job={JOB}&run=prod-1", cookie=cookie)
    assert not view["units"][0]["spans"][1]["marked"]


def test_hearing_a_draw_makes_that_sentence_with_that_retake_once_and_a_draw_heard_before_is_free(drawn):
    import time

    cookie = signed_in(drawn)
    body = {"job": JOB, "run": "prod-1", "unit": 0, "span": 1, "retake": 3}
    for _ in range(100):
        status, _, answer = call(drawn, "POST", "/api/draw-hear", body, cookie=cookie)
        assert status == 200
        if answer.get("url"):
            break
        time.sleep(0.05)
    assert answer["url"].startswith("/audio/") and answer["retake"] == 3 and (drawn.audio_dir / answer["url"].split("/")[-1]).is_file()
    spans, retakes = drawn.omni.calls[0]
    assert spans == ["Sawubona Heidi.", "UJacob Zuma ukhulumile ne-SABC namuhla."] and retakes == {1: 3}          # the sentence before it comes along; only the heard one is retaken
    call(drawn, "POST", "/api/draw-hear", body, cookie=cookie)
    assert len(drawn.omni.calls) == 1                                                            # the same draw again costs nothing


def test_a_draw_must_be_a_small_whole_number_of_a_sentence_that_exists(drawn):
    cookie = signed_in(drawn)
    for body in ({"unit": 0, "span": 1, "retake": 10}, {"unit": 0, "span": 9, "retake": 1}, {"unit": 7, "span": 0, "retake": 1}, {"unit": 0, "span": 0, "retake": "1"}):
        status, _, _ = call(drawn, "POST", "/api/draw-commit", {"job": JOB, "run": "prod-1", **body}, cookie=cookie)
        assert status == 400
    status, _, _ = call(drawn, "GET", f"/api/draws?job={JOB}&run=prod-1")
    assert status == 401                                                                          # signed in only


def test_the_studio_makes_every_unit_once_and_lays_them_on_one_timeline_with_the_time_of_every_word(drawn):
    import time

    path = drawn.work_dir / "experience" / "runs" / "prod-1" / CLIP / "r1" / "submission.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    items[0].update(start_ms=0, end_ms=4000)
    items[1].update(start_ms=10000, end_ms=12000)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    cookie = signed_in(drawn)
    answer = {}
    for _ in range(200):
        status, _, answer = call(drawn, "POST", "/api/studio-run", {"job": JOB, "run": "prod-1"}, cookie=cookie)
        assert status == 200
        if answer.get("ready"):
            break
        assert answer["pending"] and answer["total"] == 2
        time.sleep(0.05)
    assert answer.get("ready"), answer
    assert answer["url"].startswith("/audio/") and [u["start_s"] for u in answer["units"]] == [0.0, 10.0] and answer["seconds"] >= 11.0
    second = [s for s in answer["sentences"] if s["unit"] == 1]
    assert second and second[0]["words"][0]["start"] == pytest.approx(10.1)                  # the unit's words are shifted to where the unit is laid
    assert len(drawn.omni.calls) == 2                                                          # one synthesis per unit
    call(drawn, "POST", "/api/studio-run", {"job": JOB, "run": "prod-1"}, cookie=cookie)
    assert len(drawn.omni.calls) == 2                                                          # asked again: nothing is made again
    first = answer["sentences"][0]
    assert len(first["draws"]) == 3 and first["committed"] is None and first["picked"] == 0                  # three draws of every sentence, ready to play
    status, _, done = call(drawn, "POST", "/api/studio-commit", {"job": JOB, "run": "prod-1", "unit": 1, "span": 0, "take": 2, "note": "kamuva"}, cookie=cookie)
    assert status == 200 and done["mark"]["take"] == 2
    for _ in range(200):
        _, _, answer = call(drawn, "POST", "/api/studio-run", {"job": JOB, "run": "prod-1"}, cookie=cookie)
        if answer.get("ready"):
            break
        time.sleep(0.05)
    assert len(drawn.omni.calls) == 3 and next(s for s in answer["sentences"] if s["unit"] == 1 and s["index"] == 0)["committed"] == 2       # a committed draw remakes only that unit
    for body in ({"take": 3}, {"take": "1"}, {"take": 1, "span": 9}):
        status, _, _ = call(drawn, "POST", "/api/studio-commit", {"job": JOB, "run": "prod-1", "unit": 0, "span": 0, **body}, cookie=cookie)
        assert status == 400


class _LlmClient:
    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def complete(self, messages, **_):
        self.sent.append(json.loads(messages[1]["content"]))
        body = {"items": [{"id": x["id"], **self.answer} for x in self.sent[-1]]}
        return {"choices": [{"message": {"content": json.dumps(body)}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _studio_ready(console, cookie):
    import time

    for _ in range(200):
        _, _, answer = call(console, "POST", "/api/studio-run", {"job": JOB, "run": "prod-1"}, cookie=cookie)
        if answer.get("ready"):
            return answer
        time.sleep(0.05)
    raise AssertionError("the studio was not made")


def test_repair_sends_the_word_its_span_and_the_next_one_for_investigation_and_keeps_nothing(drawn):
    from mathula_tv import ear_lessons

    cookie = signed_in(drawn)
    _studio_ready(drawn, cookie)
    llm = _LlmClient({"cause": "a run of names fused into a conjunction", "rule": "When names run straight into a conjunction, give them room.", "rewrite": "Sawubona, Heidi.", "next_note": "start the next sentence with a short word"})
    drawn._llm_factory = lambda: llm
    status, _, found = call(drawn, "POST", "/api/studio-investigate", {"job": JOB, "run": "prod-1", "unit": 0, "span": 0, "word": "w0"}, cookie=cookie)
    assert status == 200 and found["cause"].startswith("a run of names") and found["rewrite"] == "Sawubona, Heidi." and "kept_as_lesson" not in found
    sent = llm.sent[0][0]
    assert sent["sentence"] == "Sawubona Heidi." and sent["next_sentence"] == "UJacob Zuma ukhulumile ne-SABC namuhla." and sent["word"] == "w0"          # the span and the next one go together
    assert "x_median" not in sent and "syllables_per_second" not in sent                              # pace is not what the model is told: it gave false flags
    assert found["word_clip_url"].startswith("/audio/") and "heard_phones" not in found                 # the word alone to play; no recogniser reading (it counted syllables wrongly)
    assert (drawn.audio_dir / found["word_clip_url"].split("/")[-1]).is_file()
    assert not any("give them room" in c["fix"] for c in ear_lessons.for_agent(drawn.work_dir)["concepts"])                  # a model's guess is not stored as a lesson: the ear decides


def test_repair_needs_the_studio_and_refuses_bad_requests(drawn):
    cookie = signed_in(drawn)
    status, _, _ = call(drawn, "POST", "/api/studio-investigate", {"job": JOB, "run": "prod-1", "unit": 0, "span": 0, "word": "w0"}, cookie=cookie)
    assert status == 409                                                                  # the studio of the run is not made yet
    _studio_ready(drawn, cookie)
    drawn._llm_factory = lambda: _LlmClient({"cause": "unclear", "rule": "", "rewrite": "", "next_note": ""})
    status, _, found = call(drawn, "POST", "/api/studio-investigate", {"job": JOB, "run": "prod-1", "unit": 0, "span": 0, "word": "w0"}, cookie=cookie)
    assert status == 200 and found["cause"] == "unclear"
    for body in ({"unit": 0, "span": 9, "word": "w0"}, {"unit": 0, "span": 0, "word": "nope"}, {"unit": 0, "span": 0, "word": ""}):
        assert call(drawn, "POST", "/api/studio-investigate", {"job": JOB, "run": "prod-1", **body}, cookie=cookie)[0] == 400


def test_a_word_the_ear_flags_is_recorded_as_heard_and_shows_in_the_studio_and_can_be_taken_back(drawn):
    cookie = signed_in(drawn)
    _studio_ready(drawn, cookie)
    body = {"job": JOB, "run": "prod-1", "unit": 0, "span": 0, "word": "w0"}
    status, _, done = call(drawn, "POST", "/api/studio-flag", {**body, "flagged": True, "note": "clipped"}, cookie=cookie)
    assert status == 200 and done["flagged"] is True and done["flags_in_this_run"] == 1
    saved = json.loads((drawn.work_dir / "ear_flags.json").read_text(encoding="utf-8"))["flags"]
    assert len(saved) == 1 and saved[0]["word"] == "w0" and saved[0]["sentence"] == "Sawubona Heidi." and saved[0]["next_sentence"].startswith("UJacob") and saved[0]["note"] == "clipped"
    assert set(saved[0]) >= {"in_audio_draw", "committed_draw", "word_start_s", "word_end_s", "at"} and "pace" not in json.dumps(saved)          # what was heard and where, nothing counted
    again = _studio_ready(drawn, cookie)
    assert next(s for s in again["sentences"] if s["unit"] == 0 and s["index"] == 0)["flagged"] == ["w0"] and next(s for s in again["sentences"] if s["unit"] == 0 and s["index"] == 1)["flagged"] == []
    call(drawn, "POST", "/api/studio-flag", {**body, "flagged": True}, cookie=cookie)
    assert len(json.loads((drawn.work_dir / "ear_flags.json").read_text(encoding="utf-8"))["flags"]) == 1                    # the same word is flagged once
    status, _, undone = call(drawn, "POST", "/api/studio-flag", {**body, "flagged": False}, cookie=cookie)
    assert status == 200 and undone["flags_in_this_run"] == 0 and json.loads((drawn.work_dir / "ear_flags.json").read_text(encoding="utf-8"))["flags"] == []
    for bad in ({"word": "nope", "flagged": True}, {"span": 9, "flagged": True}, {"flagged": "yes"}):
        assert call(drawn, "POST", "/api/studio-flag", {**body, **bad}, cookie=cookie)[0] == 400


def test_a_sign_in_lasts_twelve_months_and_still_expires_after_them(world):
    from mathula_tv import console as c

    assert c.SESSION_SECONDS == 365 * 24 * 3600                                          # 31,536,000 s: the cookie's Max-Age, under the 400 days browsers allow
    cookie = signed_in(world)
    issued = world.clock()
    world.clock = lambda: issued + 364 * 24 * 3600
    assert call(world, "GET", "/api/session", cookie=cookie)[0] == 200                  # still in, almost a year later
    world.clock = lambda: issued + 366 * 24 * 3600
    assert call(world, "GET", "/api/session", cookie=cookie)[0] == 401                  # and out after the twelve months


def test_a_run_can_ask_for_its_own_decoding_settings_and_its_studio_is_a_different_audio(drawn):
    runs = drawn.work_dir / "experience" / "runs"
    (runs / "prod-1-pt1" / CLIP / "r1").mkdir(parents=True)
    (runs / "prod-1-pt1" / CLIP / "r1" / "submission.json").write_text((runs / "prod-1" / CLIP / "r1" / "submission.json").read_text(encoding="utf-8"), encoding="utf-8")
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"generation": {"position_temperature": 1, "bogus": "x"}}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1") is None and drawn._run_voice(JOB, "prod-1-pt1") == {"generation": {"position_temperature": 1.0}}          # only numbers are kept
    drawn._omni_backend = None
    drawn._omni_factory = lambda: _FakeOmni()                                                                                                    # a copy is a second backend, not the shared one
    base = drawn._omni_with(None)
    other = drawn._omni_with({"generation": {"position_temperature": 1.0}, "span_sentences": 2})
    assert base is drawn._omni() and other is not base and other.generation == {"position_temperature": 1.0} and other.span_sentences == 2
    assert drawn._omni_with({"generation": {"position_temperature": 1.0}, "span_sentences": 2}) is other
    first = drawn.studio_run(JOB, "prod-1", make=False)
    second = drawn.studio_run(JOB, "prod-1-pt1", make=False)
    assert first["pending"] and second["pending"]                                                                                               # nothing made, nothing spent, only read
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"span_sentences": "fill", "splice": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"span_sentences": 99, "splice_names": False}                                                  # a run can ask for no splicing
    assert drawn._omni_with({"span_sentences": 99, "splice_names": False}).splice_names is False and drawn._omni_with({"span_sentences": 99}).splice_names is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"primer_syllables": 1.5}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"primer_syllables": 1.5} and drawn._omni_with({"primer_syllables": 1.5}).match_primer_syllables == 1.5
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"pin_to_reference": True}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"pin_to_reference": True} and drawn._omni_with({"pin_to_reference": True}).pin_to_reference is True and drawn._omni_with({"span_sentences": 99}).pin_to_reference is False
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"priming": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"priming": False} and drawn._omni_with({"priming": False}).priming is False and drawn._omni_with({"span_sentences": 99}).priming is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"host_per_draw": True, "splice_refine": True, "splice_phrases": True}), encoding="utf-8")
    voiced = drawn._run_voice(JOB, "prod-1-pt1")
    assert voiced == {"host_per_draw": True, "splice_refine": True, "splice_phrases": True}
    made = drawn._omni_with(voiced)
    assert made.host_per_draw is True and made.splice_refine is True and made.splice_phrases is True
    plain = drawn._omni_with({"span_sentences": 99})
    assert plain.host_per_draw is False and plain.splice_refine is False and plain.splice_phrases is False
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"primer_align": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"primer_align": False} and drawn._omni_with({"primer_align": False}).primer_align is False and plain.primer_align is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"splice_sync": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"splice_sync": False} and drawn._omni_with({"splice_sync": False}).splice_sync is False and plain.splice_sync is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"splice_mix": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"splice_mix": False} and drawn._omni_with({"splice_mix": False}).splice_mix is False and plain.splice_mix is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"splice_inside_prefix": True}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"splice_inside_prefix": True} and drawn._omni_with({"splice_inside_prefix": True}).splice_inside_prefix is True and plain.splice_inside_prefix is False
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"primer_usual_pace": False}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"primer_usual_pace": False} and drawn._omni_with({"primer_usual_pace": False}).primer_usual_pace is False and plain.primer_usual_pace is True
    (runs / "prod-1-pt1" / CLIP / "voice.json").write_text(json.dumps({"host_spaced": True}), encoding="utf-8")
    assert drawn._run_voice(JOB, "prod-1-pt1") == {"host_spaced": True} and drawn._omni_with({"host_spaced": True}).host_spaced is True and plain.host_spaced is False


def test_no_function_of_the_console_page_is_defined_twice():
    """A second ``function name(`` silently replaces the first in JavaScript: the Studio's run picker once took the name of the function that opens a word's draws, and pressing any word loaded nothing."""
    import re
    from collections import Counter

    from mathula_tv import console_ui

    names = Counter(re.findall(r"^(?:async )?function (\w+)\(", console_ui.PAGE, flags=re.M))
    assert [n for n, c in names.items() if c > 1] == []
