from pathlib import Path

from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms
from mathula_tv.pronunciation_probe import best_per_number, numbers_in, probe_numbers


def test_numbers_are_read_from_digits_and_english_words():
    assert numbers_in("Angaba owesi seven osusiwe") == [7]
    assert numbers_in("on twenty two October") == [22]
    assert numbers_in("ngo twenty-one") == [21]
    assert numbers_in("owesi 12") == [12]
    assert numbers_in("twenty twenty six") == [20, 26] and numbers_in("owesi free") == []


def _fakes(heard_for):
    last = {}

    def synthesize(text, path):
        last["text"] = text

    def transcribe(path):
        return heard_for(last["text"])

    return synthesize, transcribe


def test_a_shipped_word_that_is_heard_every_time_needs_no_alternative(tmp_path):
    synthesize, transcribe = _fakes(lambda text: "ngowesi seven osusiwe" if "seven" in text else "x")
    rows = probe_numbers(synthesize, transcribe, numbers=[7], word_for=lambda n: "seven", alternatives={7: ["sevn"]},
                         contexts=("ordinal",), audio_dir=tmp_path, repeats=2)
    assert [r["candidate"] for r in rows] == ["seven"] and rows[0]["score"] == 1.0


def test_a_failing_shipped_word_tries_alternatives_and_the_better_one_wins(tmp_path):
    # the recogniser hears "free" for "three" but "three" for the tuned respelling
    synthesize, transcribe = _fakes(lambda text: "ngowesi three" if "triiy" in text else "ngowesi free")
    rows = probe_numbers(synthesize, transcribe, numbers=[3], word_for=lambda n: "three", alternatives={3: ["triiy"]},
                         contexts=("ordinal", "date"), audio_dir=tmp_path, repeats=2)
    assert {r["role"] for r in rows} == {"shipped", "alternative"}
    best = best_per_number(rows)[3]
    assert best["candidate"] == "triiy" and best["score"] == 1.0 and best["shipped"] is False


def test_on_a_tie_the_shipped_word_is_kept(tmp_path):
    synthesize, transcribe = _fakes(lambda text: "ngowesi ten")
    rows = probe_numbers(synthesize, transcribe, numbers=[10], word_for=lambda n: "ten", alternatives={10: ["tenn"]},
                         contexts=("ordinal",), audio_dir=tmp_path, repeats=1, try_alternatives_below=1.1)
    assert best_per_number(rows)[10]["candidate"] == "ten"


def test_every_day_of_the_month_is_code_switched_like_an_ordinal():
    dictionary = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))
    expected = {1: "one", 10: "ten", 12: "twelve", 22: "twenty-two", 31: "thirty-one"}
    for day in range(1, 32):
        tts = dictionary.apply(f"kusukela ngo-{day} October 2026, ngokwesitatimende").tts_text
        assert tts.startswith("kusukela ngo-") and " October 2026, ngokwesitatimende" in tts
        assert "first" not in tts and "second" not in tts and "mhla" not in tts
        if day in expected:
            assert f"ngo-{expected[day]} October" in tts
    assert dictionary.apply("ngomhla ka-22 Okthoba 2026").tts_text.startswith("mhla zingamashumi")   # explicit isiZulu construction kept


def test_a_recogniser_that_hears_nothing_scores_zero_instead_of_crashing(tmp_path):
    from mathula_tv.pronunciation_probe import probe_candidate

    def silent(path):
        raise RuntimeError("Azure Speech returned an empty transcription")

    result = probe_candidate(lambda text, path: None, silent, number=3, candidate="three", carrier="x {x}",
                             audio_dir=tmp_path, repeats=2)
    assert result["score"] == 0.0 and "nothing recognised" in result["samples"][0]["heard"]
