import json

from mathula_tv import ear_clips

ITEMS = [
    {"item_id": "u01", "speaker_id": "SPEAKER_00", "zulu_text": "Sawubona. UJacob Zuma ukhulumile ne-SABC ~pause500 namuhla ngomsebenzi omude kakhulu wokuhlola."},
    {"item_id": "u02", "speaker_id": "SPEAKER_01", "zulu_text": "Ngiyabonga. Jacob uthe akazukuphendula."},
    {"item_id": "u03", "speaker_id": "SPEAKER_01", "zulu_text": "Akukho okunye ngoJacobson."},
]
VOICES = {"SPEAKER_00": "zu-ZA-ThandoNeural", "SPEAKER_01": "zu-ZA-ThembaNeural"}


def test_sentences_drop_the_held_pause_marks_and_split_at_the_full_stops():
    assert ear_clips.sentences("Sawubona. UJacob ukhulumile ~pause500 namuhla!") == ["Sawubona.", "UJacob ukhulumile", "namuhla!"]


def test_the_word_is_found_bare_after_a_hyphen_or_with_a_prefix_glued_on_but_not_inside_another_word():
    pattern = ear_clips.word_pattern("Jacob")
    assert all(pattern.search(t) for t in ("Jacob uthe", "uJacob Zuma", "ku-Jacob", "kuJacob"))
    assert not pattern.search("ngoJacobson") and not pattern.search("Jacobs")


def test_each_voice_is_heard_on_a_word_before_a_voice_is_heard_twice():
    rows = ear_clips.pick(ITEMS, ["Jacob"], VOICES)
    assert [(r["voice"], r["unit"]) for r in rows] == [("zu-ZA-ThembaNeural", "u02"), ("zu-ZA-ThandoNeural", "u01")] or {r["voice"] for r in rows} == set(VOICES.values())
    assert len(rows) == 2 and all("Jacob" in r["text"] for r in rows)
    assert ear_clips.pick(ITEMS, ["Nobody"], VOICES) == []


def test_the_clips_are_spoken_through_the_jobs_own_dictionary_and_voice(tmp_path):
    from mathula_tv.native_dub import _native_dub_tts_ready_text

    job = tmp_path / "job"
    (job / "direct_dub").mkdir(parents=True)
    (job / "native_dub").mkdir()
    (job / "direct_dub" / "voice_resolution.json").write_text(json.dumps({"assignments": {s: {"selected_voice": v, "base_prosody": {}} for s, v in VOICES.items()}}), encoding="utf-8")
    (job / "native_dub" / "pronunciation_names.json").write_text(json.dumps({"schema": "x", "voice": "v", "entries": [
        {"display_text": "Jacob", "tts_text": "Jacob", "tts_text_by_voice": {v: "jeykob" for v in VOICES.values()}, "kind": "english_code_switch", "source": "rules"}]}), encoding="utf-8")
    spoken = []

    def fake(*, tts, group, voice_info, output_path, preferred_ms, maximum_ms, force, job_root):
        text = group["parts"][0]["text"]
        spoken.append((voice_info["selected_voice"], _native_dub_tts_ready_text(text, job_root=job_root, voice=voice_info["selected_voice"])))
        output_path.write_bytes(b"wav")

    made = ear_clips.make_clips(job, ITEMS, ["Jacob"], tmp_path / "out", tts=object(), synthesize=fake)
    assert len(made) == 2 and all((tmp_path / "out").glob("*.wav"))
    assert all("jeykob" in text and "Jacob" not in text for _, text in spoken)          # the render's spelling, not the word as written
    assert {voice for voice, _ in spoken} == set(VOICES.values())
    assert made[0]["path"].endswith(".wav") and made[0]["word"] == "Jacob"


def test_a_spelling_replaces_the_word_in_its_sentence_and_the_zulu_prefix_stays():
    assert ear_clips.substitute("UJacob Zuma ukhulumile ne-SABC", "Jacob", "jeykob") == "Ujeykob Zuma ukhulumile ne-SABC"
    assert ear_clips.substitute("ngivela e-MK Party", "MK", "em-kheeyi") == "ngivela e-em-kheeyi Party"
    assert ear_clips.substitute("Jacobson", "Jacob", "x") == "Jacobson"
