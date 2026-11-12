from collections import Counter

from mathula_tv.grammar_test import CATEGORIES, build_items, correct, normalise


def test_the_item_set_is_complete_unique_and_balanced():
    items = build_items()
    assert len({i.id for i in items}) == len(items)
    counts = Counter(i.category for i in items)
    assert set(counts) == set(CATEGORIES)
    assert all(n >= 15 for n in counts.values()), counts
    assert 140 <= len(items) <= 200, len(items)
    for i in items:
        assert i.accept and i.accept[0].strip() and i.prompt.strip(), i.id
        assert "None" not in i.prompt and "{" not in i.prompt, i.id


def test_gold_answers_are_pinned_to_attested_forms():
    by_prompt = {i.prompt: i for i in build_items()}

    def gold(fragment: str) -> str:
        found = [i for p, i in by_prompt.items() if fragment in p]
        assert len(found) == 1, (fragment, len(found))
        return found[0].accept[0]

    assert gold("prefix 'kwa' (of) and the noun 'ukuba'") == "kokuba"
    assert gold("prefix 'na' (with / and) and the noun 'izinto'") == "nezinto"
    assert gold("Write the number 25 in isiZulu words so that it agrees") == "abangamashumi amabili nanhlanu"
    assert gold("Write the number 4500 in isiZulu words, counting on its own") == "izinkulungwane ezine namakhulu amahlanu"
    assert gold("Verb: ukubona (to see). Subject: I. Object: you (one person). Form: future") == "ngizokubona"


def test_every_extension_item_has_the_softened_passive_where_the_corpus_has_it():
    ext = {i.prompt: i.accept[0] for i in build_items() if i.category == "extension"}
    answers = set(ext.values())
    assert {"banjwa", "khulunywa", "boshwa", "gwetshwa", "thunyelwa", "vunyelwa", "hanjiswa", "khonjiswa", "enziwa", "akhiwa", "xhunyaniswa"} <= answers


def test_the_relative_items_accept_the_forms_with_and_without_yo_and_with_an_object_concord():
    rel = [i for i in build_items() if i.category == "relative"]
    present = next(i for i in rel if "'who/which does' that describes abantu" in i.prompt)
    assert correct(present, "abasebenza") and correct(present, "abasebenzayo") and not correct(present, "abasebenzile")
    person = next(i for i in rel if "izincwadi" in i.prompt and "that I" in i.prompt)
    assert all(correct(person, a) for a in ("engifuna", "engifunayo", "engizifuna", "engizifunayo"))


def test_scoring_ignores_case_punctuation_and_hyphens_but_nothing_else():
    items = build_items()
    num = next(i for i in items if i.category == "numeral" and "25" in i.prompt)
    assert normalise("  Abangamashumi  amabili nanhlanu. ") == "abangamashumi amabili nanhlanu"
    assert correct(num, "Abangamashumi amabili nanhlanu.") and correct(num, "aba-ngama-shumi amabili nanhlanu")      # a hyphen inside a word is a morpheme break
    assert not correct(num, "abangamashumi-amabili nanhlanu")                                                       # a hyphen is not a substitute for the space between words
    assert not correct(num, "abangamashumi amabili nesihlanu") and not correct(num, "") and not correct(num, None)
