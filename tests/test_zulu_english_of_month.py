"""The English "of" the translator leaves between a day and its month is dropped and the month is read by its everyday isiZulu name ("ngo-fooo Novemba")."""
from __future__ import annotations

import pytest

from mathula_tv.pronunciation import PronunciationDictionary, PronunciationEntry, with_default_organisation_initialisms


def _dictionary(*entries: PronunciationEntry) -> PronunciationDictionary:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-of-month", entries=entries))


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("Kusukela ngo-4th of November, kuningi osekuqalisiwe.", "Kusukela ngo-four Novemba, kuningi osekuqalisiwe."),
        ("Kusukela ngo-22nd of October kuningi.", "Kusukela ngo-twenty-two Okthoba kuningi."),
        ("Ngo-1st of august.", "Ngo-wan Agasti."),
        ("Kusukela ngo-4 of November.", "Kusukela ngo-four Novemba."),          # the day is a figure like any other: said in English
    ),
)
def test_the_english_of_between_a_day_and_its_month_is_said_as_the_everyday_month_name_alone(source: str, expected: str) -> None:
    result = _dictionary().apply(source)
    assert result.spoken_text == source and result.tts_text == expected


def test_a_day_the_job_spells_keeps_its_spelling_and_only_the_connector_and_month_change() -> None:
    fooo = PronunciationEntry(display_text="4th", spoken_text="4th", tts_text="fooo", language="zu-ZA", source="job_rules", confidence=0.9, kind="english_code_switch")
    result = _dictionary(fooo).apply("Kusukela ngo-4th of November, kuningi.")
    assert result.tts_text == "Kusukela ngo-fooo Novemba, kuningi."


def test_other_uses_of_of_are_left_alone() -> None:
    for text in ("Ngicela umhlangano of the ANC ngoNovemba.", "Izinto zonke of November zasala.", "ngo-4th of nothing"):
        assert not [s for s in _dictionary().apply(text).substitutions if s.kind == "date"]


def test_a_day_written_after_its_month_is_said_as_the_job_spells_the_number_not_as_an_ordinal() -> None:
    fooo = PronunciationEntry(display_text="4th", spoken_text="4th", tts_text="fooo", language="zu-ZA", source="job_rules", confidence=0.9, kind="english_code_switch")
    assert _dictionary(fooo).apply("Kusukela ngoNovemba 4, kuningi.").tts_text == "Kusukela ngoNovember fooo, kuningi."
    assert _dictionary().apply("Kusukela ngoNovemba 4, kuningi.").tts_text == "Kusukela ngoNovember fourth, kuningi."          # a job with no spelling for the day keeps the plain English ordinal
    twenty_two = PronunciationEntry(display_text="22nd", spoken_text="22nd", tts_text="tweenti-tuu", language="zu-ZA", source="job_rules", confidence=0.9, kind="english_code_switch")
    assert _dictionary(fooo, twenty_two).apply("uDisemba 22 no-ngoNovemba 4").tts_text == "uDecember tweenti-tuu no-ngoNovember fooo"


def test_the_ordinal_of_a_day_is_how_it_is_written_in_english() -> None:
    from mathula_tv.pronunciation import written_ordinal
    assert [written_ordinal(d) for d in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "31st"]


def test_a_year_after_a_hyphenated_preposition_is_said_in_english_with_a_wan_for_one() -> None:
    for source, expected in (
        ("amametro lapho naniseneningi eliphelele ngo-2021.", "amametro lapho naniseneningi eliphelele ngo-twenty-twenty-wan."),
        ("njengoba u-2029 usuyinhloso", "njengoba u-twenty-twenty-nine usuyinhloso"),
        ("kusukela ka-2021 kuya ngo-2024", "kusukela ka-twenty-twenty-wan kuya ngo-twenty-twenty-four"),
        ("ngo-2001", "ngo-twenty-oh-wan"),
    ):
        assert _dictionary().apply(source).tts_text == expected
    for untouched in ("ngo-2021 abantu", ):
        assert "twenty" in _dictionary().apply(untouched).tts_text
    assert _dictionary().apply("izinkulungwane ezingu-2021").tts_text == "izinkulungwane ezingu-2021"            # a quantity, not a year: no date preposition before the hyphen

