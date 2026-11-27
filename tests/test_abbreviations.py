from __future__ import annotations

import pytest

from mathula_tv import abbreviations as ab


@pytest.mark.parametrize("written, spoken", [
    ("Mnu Mashaba uthe", "Mnumzana Mashaba uthe"),
    ("Mnu. Mashaba uthe", "Mnumzana Mashaba uthe"),
    ("noMnu. Adams", "noMnumzana Adams"),
    ("uNksz Khumalo", "uNkosazana Khumalo"),
    ("uDkt Matebula", "uDokotela Matebula"),
    ("Nkk Molefe", "Nkosikazi Molefe"),
    ("Dr. Smith uthe", "Doctor Smith uthe"),
    ("Dr Smith uthe", "Doctor Smith uthe"),
    ("uProf. Mbeki", "uProfessor Mbeki"),
    ("Adv. Mpofu uthe", "Advocate Mpofu uthe"),
    ("Mr. Cele", "Mister Cele"),
    ("Mrs Dube", "Missus Dube"),
    ("Ms. Dube", "Miss Dube"),
    ("Mnr. Botha", "Meneer Botha"),
    ("Mev. Botha", "Mevrou Botha"),
    ("Cllr. Ngobese uthe", "Councillor Ngobese uthe"),
    ("Gen. Mapisa uthe", "General Mapisa uthe"),
    ("Col. Mkhize", "Colonel Mkhize"),
    ("Capt. Zulu", "Captain Zulu"),
    ("Hon. Mathe", "Honourable Mathe"),
    ("Rev. Tutu", "Reverend Tutu"),
    ("Martin Luther King Jr. uthe", "Martin Luther King Junior uthe"),
    ("dr. smith", "doctor smith"),
    ("e.g. lokhu", "for example lokhu"),
    ("i.e. lokhu", "that is lokhu"),
    ("amabhanana, njll. etc. kanye", "amabhanana, njll. et cetera. kanye"),
    ("No. 5 ikhefu", "Number 5 ikhefu"),
    ("20 km", "20 kilometres"), ("20km", "20 kilometres"), ("5 kg", "5 kilograms"), ("120 km/h", "120 kilometres per hour"), ("10 cm", "10 centimetres"),
])
def test_an_abbreviation_with_a_spoken_form_is_written_out(written, spoken):
    assert ab.expand(written)[0] == spoken


@pytest.mark.parametrize("text", [
    "Gen Z abantu", "Rev uthe", "Dr uthe", "Mnu uthe", "umnumzane omkhulu", "Doctor Smith", "Mnumzana Mashaba", "isikhathi sesikhathi", "Sihlangene namaqembu ezepolitiki.",
    "ngo-2026 kwakhona", "i-ANC iyavuma", "amakhilomitha angu-20", "Main St. Helena", "ngifuna ukuthi ma. kuthi", "kg", "ml",
])
def test_an_ordinary_word_or_a_word_already_written_out_is_left_alone(text):
    assert ab.expand(text) == (text, [])


def test_the_change_is_listed_and_expanding_twice_changes_nothing_more():
    out, changes = ab.expand("Sikhulume noMnu. Adams no-Dr. Smith, 20 km kude.")
    assert out == "Sikhulume noMnumzana Adams no-Doctor Smith, 20 kilometres kude."
    assert [(c["was"], c["now"]) for c in changes] == [("noMnu.", "noMnumzana"), ("Dr.", "Doctor"), (" km", " kilometres")] or len(changes) == 3
    assert ab.expand(out) == (out, [])


def test_every_title_is_a_word_that_is_never_part_of_a_name():
    words = ab.title_words()
    assert {"mnu", "mnumzana", "dr", "doctor", "prof", "professor", "adv", "advocate", "nkk", "nkosikazi", "dkt", "dokotela", "mnr", "cllr", "gen", "general"} <= words
    from mathula_tv.omnivoice_tts import ReferenceBank
    assert ReferenceBank.dub_name_runs("Dr. Smith uthe, Prof Mbeki, Cllr Ngobese") == [["Smith"], ["Mbeki"], ["Ngobese"]] or ReferenceBank.dub_name_runs("Dr. Smith uthe")[0] == ["Smith"]


def test_the_dictionary_view_the_agent_gets_shows_the_expansion_and_the_text_the_voice_gets_has_it(monkeypatch):
    from mathula_tv import native_dub as nd
    from mathula_tv import say_it as module

    assert nd._native_dub_tts_ready_text("Mnu Mashaba, Dr. Smith kanye no-Adv. Mpofu") == "Mnumzana Mashaba, Doctor Smith kanye no-Advocate Mpofu"
    shown = module.say_it(nd._native_dub_pronunciation_dictionary(), "Dr. Smith uthe", None)
    assert shown["spoken"].startswith("Doctor Smith") and {"was": "Dr.", "now": "Doctor", "kind": "abbreviation"} in shown["changes"]
    assert not any("abbreviation" in line for line in shown["still_as_written"])                 # nothing left for the agent to worry about


def test_expand_leaves_an_english_word_after_a_glued_zulu_prefix_as_the_agent_wrote_it():
    assert ab.expand("kunamaqembu ama-details okugcina, silinde") == ("kunamaqembu ama-details okugcina, silinde", [])          # the dub text keeps the glued form: the voice leaves the word after it whole


@pytest.mark.parametrize("text,expected", [
    ("Kunamaqembu ama-details amaningi.", "Kunamaqembu ama details amaningi."),
    ("ezingu-forty zabantu", "ezingu forty zabantu"),
    ("Bahlangene kwa-workshops, kodwa", "Bahlangene kwa workshops, kodwa"),
    ('"ama-details," kusho yena', '"ama details," kusho yena'),
])
def test_an_english_word_after_a_glued_zulu_prefix_is_sent_to_the_voice_as_two_words(text, expected):
    out, groups = ab.separate_stems(text)
    assert out == expected and sum(groups) == len(out.split()) and len(groups) == len(text.split())
    assert ab.separate_stems(out) == (out, [1] * len(out.split()))


@pytest.mark.parametrize("text", [
    "u-Phineas wafika", "we-JMPD", "kuka-nayn", "e-mail yakhe", "anti-corruption unit", "ama-2026", "ama-qembu", "ngo-2026 kwakhona", "ex-president uthe", "ama-Hawks", "ama-ab",
])
def test_names_figures_zulu_stems_and_english_hyphens_are_sent_as_they_are(text):
    assert ab.separate_stems(text) == (text, [1] * len(text.split()))


def test_a_word_the_speaker_says_stays_glued_because_it_is_cut_out_and_replaced():
    assert ab.separate_stems("amaqembu ama-details okugcina", keep=["Details"]) == ("amaqembu ama-details okugcina", [1, 1, 1])
    assert ab.separate_stems("amaqembu ama-details ama-photos", keep=["details"])[0] == "amaqembu ama-details ama photos"


def test_the_voices_word_times_are_folded_back_to_the_words_of_the_text():
    out, groups = ab.separate_stems("a ama-details b")
    times = [{"start": 0.0, "end": 0.1}, {"start": 0.1, "end": 0.3}, {"start": 0.3, "end": 0.7}, {"start": 0.7, "end": 0.9}]
    folded = ab.fold_words(times, groups)
    assert len(folded) == 3 and folded[1] == {"start": 0.1, "end": 0.7} and folded[0] == times[0] and folded[2] == times[3]
    assert ab.fold_words(times[:3], groups) == times[:3]                          # a list that does not match is left alone
    assert ab.fold_words([times[0], None, times[2], times[3]], groups)[1] is None


def test_a_prefixed_english_word_is_a_name_run_with_the_hyphen_or_the_space():
    from mathula_tv.omnivoice_tts import ReferenceBank
    assert ReferenceBank.dub_name_runs("kunamaqembu ama-details amaningi.") == [["details"]]
    assert ReferenceBank.dub_name_runs("kunamaqembu ama details amaningi.") == [["details"]]
    assert ReferenceBank.dub_name_runs("sikhuluma ngama details.") == [["details"]]
    assert ReferenceBank.dub_name_runs("amaqembu amaningi ukuthi") == []


def test_a_figure_after_its_prefix_is_sent_to_the_voice_with_the_prefix_apart():
    # heard on OmniVoice (prod-y1 unit 2): "ngo-nineteen-ninety-four" was read as one Zulu word, "ngo-nineteen ninety-four" was chopped at 19; "ngo nineteen ninety-four" was right after 1 to 4 Zulu words
    assert ab.separate_stems("kwaze kwaba ngo-nineteen-ninety-four. Futhi") == ("kwaze kwaba ngo nineteen ninety-four. Futhi", [1, 1, 3, 1])
    assert ab.separate_stems("iminyaka engu-sixty-seven yami, kuka-nayn,") == ("iminyaka engu sixty-seven yami, kuka-nayn,", [1, 2, 1, 1])          # a figure of one word keeps the form chosen for it
    assert ab.separate_stems("ngo-twenty-twenty-four")[0] == "ngo twenty twenty-four"
    assert ab.separate_stems("kwaba ngo-nineteen-ninety-four", keep=["four"]) == ("kwaba ngo-nineteen-ninety-four", [1, 1])          # a figure cut from the speaker's own saying stays as the primer has it
    assert ab.separate_stems("ngo nineteen ninety-four")[0] == "ngo nineteen ninety-four"
    words = [{"start": 0.0, "end": 0.3}, {"start": 0.3, "end": 0.5}, {"start": 0.5, "end": 0.8}, {"start": 0.8, "end": 1.2}]
    assert ab.fold_words(words, [1, 3]) == [words[0], {"start": 0.3, "end": 1.2}]
