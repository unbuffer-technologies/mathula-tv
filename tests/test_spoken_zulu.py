from mathula_tv.spoken_zulu import SpokenZulu, skeleton_english, skeleton_zulu

ROWS = [
    {"domain": "Telecommunication", "transcript": "Izinto ezifana nekhwalithi ye-microphone, amandla nokuhlanganisa.", "duration": 30.0},
    {"domain": "Agriculture", "transcript": "Ngathenga i-microphone ukuze ngicule nabangani bami.", "duration": 20.0},
    {"domain": "Telecommunication", "transcript": "Amathuluzi ahamba nama-microphone nama-earphones.", "duration": 20.0},
    {"domain": "Telecommunication", "transcript": "Kuhlangene amayikhrofoni amaningi esiteji.", "duration": 20.0},
    {"domain": "Transport", "transcript": "Kukhona umbhobho ozomemezela ukuthi ibhasi izofika.", "duration": 20.0},
    {"domain": "Health", "transcript": "Umbhobho wamanzi uyavuza endlini yami.", "duration": 20.0},
    {"domain": "Finance", "transcript": "Ngivule i-akhawunti entsha ebhange namhlanje.", "duration": 20.0},
    {"domain": "Finance", "transcript": "Wangena ezikweletini ebhange ngenxa yemali.", "duration": 20.0},
]


def test_skeletons_match_an_english_word_and_its_zulu_respellings_but_not_a_native_word():
    assert skeleton_english("microphone") == "mkrfn"
    assert skeleton_zulu("mayikhrofoni") == "mkrfn" and skeleton_zulu("makrofoni") == "mkrfn" and skeleton_zulu("njengemayikhrofoni").endswith("mkrfn")
    assert skeleton_zulu("umbhobho") != skeleton_english("microphone")


def test_the_english_word_kept_behind_particles_is_found_and_respellings_are_discovered():
    found = SpokenZulu(ROWS).lookup("microphone")
    kept = {f["form"]: f["count"] for f in found["english_word_kept"]["written_as"]}
    assert kept == {"i-microphone": 1, "ye-microphone": 1, "nama-microphone": 1} and found["english_word_kept"]["total"] == 3
    assert [f["form"] for f in found["respelled_in_zulu"]["written_as"]] == ["amayikhrofoni"]
    assert "speakers keep the English word" in found["reading"] and "respell it in Zulu" in found["reading"]
    assert {e["domain"] for e in found["examples"]} >= {"Telecommunication", "Agriculture"}          # spread over domains


def test_a_form_the_agent_is_thinking_of_is_counted_with_examples_that_show_its_meaning():
    found = SpokenZulu(ROWS).lookup("microphone", ["umbhobho", "umnyango"])
    by_form = {c["form"]: c for c in found["checked_forms"]}
    assert by_form["umbhobho"]["spoken_count"] == 2 and {e["domain"] for e in by_form["umbhobho"]["examples"]} == {"Transport", "Health"}
    assert by_form["umnyango"]["spoken_count"] == 0 and "examples" not in by_form["umnyango"]


def test_a_word_speakers_never_use_says_so_and_a_short_skeleton_matches_nothing():
    sp = SpokenZulu(ROWS)
    none = sp.lookup("coalition")
    assert none["english_word_kept"]["total"] == 0 and "neither keep nor respell" in none["reading"]
    assert sp.lookup("account")["respelled_in_zulu"]["written_as"] == []          # skeleton 'knt' is under the length floor: no guessing
    assert sp.lookup("coalition")["respelled_in_zulu"]["written_as"] == []        # 'kltn' would match ezikweletini (in the debts): four consonants prove nothing
    assert sp.lookup("")["source"].startswith("spoken isiZulu") and "english_word_kept" not in sp.lookup("")


def test_a_missing_file_gives_none(tmp_path):
    assert SpokenZulu.from_file(tmp_path / "nothing.jsonl") is None


def test_a_folder_of_spoken_sources_is_read_together(tmp_path):
    import json

    (tmp_path / "anv.jsonl").write_text(json.dumps({"domain": "Telecommunication", "transcript": "Ngathenga i-microphone entsha.", "duration": 10}) + chr(10), encoding="utf-8")
    (tmp_path / "soap.jsonl").write_text(json.dumps({"domain": "Soap opera dialogue", "transcript": "uma nje ye-microphone sure", "duration": 5}) + chr(10), encoding="utf-8")
    both = SpokenZulu.from_directory(tmp_path)
    assert both.lookup("microphone")["english_word_kept"]["total"] == 2
    assert {e["domain"] for e in both.lookup("microphone")["examples"]} == {"Telecommunication", "Soap opera dialogue"}
    assert SpokenZulu.from_directory(tmp_path / "missing") is None
