from __future__ import annotations

import pytest

from mathula_tv.direct_azure_dub import load_approved_segments
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    ZU_NATIVE_DATE_PRONUNCIATION_VERSION,
    normalise_dates_for_tts,
    with_default_organisation_initialisms,
)
from mathula_tv.target_text import build_target_text


def _dictionary(locale: str = "zu-ZA") -> PronunciationDictionary:
    return with_default_organisation_initialisms(
        PronunciationDictionary("v1", language=locale, job_id="job-date")
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "uMnu. Adams angeza ngomhla ka-19 Agasti.",
            "uMnumzane Adams angeza mhla ziyishumi nesishiyagalolunye kuNcwaba.",
        ),
        (
            "Impendulo yayidingeka ngomhla ka-25 Okthoba 2025.",
            "Impendulo yayidingeka mhla zingamashumi amabili nesihlanu kuMfumfu 2025.",
        ),
        (
            "Isikhala esilandelayo ngumhla ka-19 August.",
            "Isikhala esilandelayo ngumhla weshumi nesishiyagalolunye kuNcwaba.",
        ),
        (
            "Sihlehlisele umhla ka-3 Mashi.",
            "Sihlehlisele umhla wesithathu kuNdasa.",
        ),
        (
            "19 Agasti. Icala liyaqhubeka.",
            "mhla ziyishumi nesishiyagalolunye kuNcwaba. Icala liyaqhubeka.",
        ),
        (
            "Mhla ziyi-1 kuNhlangulana.",
            "mhla lulunye kuNhlangulana.",
        ),
        (
            "mhla zingama-31 Ncwaba 2026",
            "mhla zingamashumi amathathu nanye kuNcwaba 2026",
        ),
    ),
)
def test_zulu_calendar_dates_receive_native_hidden_tts(
    source: str,
    expected: str,
) -> None:
    result = _dictionary().apply(source)

    assert result.spoken_text == source
    assert result.tts_text == expected
    date_substitutions = [
        item for item in result.substitutions if item.kind == "date"
    ]
    assert len(date_substitutions) == 1


def test_target_text_audits_date_change_without_changing_caption() -> None:
    source = "Ubufakazi bumiselwe umhla ka-19 Agasti 2026."
    target = build_target_text(source, dictionary=_dictionary())

    assert target.spoken_text == source
    assert target.subtitle_text == source
    assert target.tts_text == (
        "Ubufakazi bumiselwe umhla weshumi nesishiyagalolunye kuNcwaba 2026."
    )
    assert target.changes[0].change_type == "date_normalisation"
    assert target.pronunciation_substitutions[0].source == "application_default"


def test_direct_dub_rebuilds_every_variant_from_native_date_policy() -> None:
    segments = load_approved_segments(
        {
            "units": [
                {
                    "unit_id": "unit_0001",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 0,
                    "end_ms": 5000,
                    "source_text": "The hearing is postponed to 19 August.",
                    "variants": [
                        {
                            "variant_id": variant_id,
                            "spoken_text": spoken,
                            "tts_text": spoken,
                            "meaning_preserved": True,
                        }
                        for variant_id, spoken in (
                            ("natural", "Icala lihlehliselwe umhla ka-19 Agasti."),
                            ("concise", "Lihlehliselwe ngomhla ka-19 Agasti."),
                            ("compact", "19 Agasti."),
                        )
                    ],
                }
            ]
        },
        source_duration_ms=5000,
        pronunciation_dictionary=_dictionary(),
    )

    assert [variant.tts_text for variant in segments[0].variants] == [
        "Icala lihlehliselwe umhla weshumi nesishiyagalolunye kuNcwaba.",
        "Lihlehliselwe mhla ziyishumi nesishiyagalolunye kuNcwaba.",
        "mhla ziyishumi nesishiyagalolunye kuNcwaba.",
    ]
    assert [variant.spoken_text for variant in segments[0].variants] == [
        "Icala lihlehliselwe umhla ka-19 Agasti.",
        "Lihlehliselwe ngomhla ka-19 Agasti.",
        "19 Agasti.",
    ]


def test_explicit_reviewed_date_entry_remains_more_authoritative() -> None:
    from mathula_tv.pronunciation import PronunciationEntry

    dictionary = PronunciationDictionary(
        "v1",
        language="zu-ZA",
        entries=(
            PronunciationEntry(
                display_text="19 Agasti",
                spoken_text="19 Agasti",
                tts_text="reviewed exact date",
                language="zu-ZA",
                kind="date",
            ),
        ),
    )

    assert dictionary.apply("19 Agasti").tts_text == "reviewed exact date"


def test_date_normalizer_is_isolated_from_other_locales_and_month_mentions() -> None:
    assert _dictionary("xh-ZA").apply("19 August").tts_text == "19 August"
    assert _dictionary().apply("Sahlangana ngo-Okthoba.").tts_text == (
        "Sahlangana ngo-Okthoba."
    )
    result = normalise_dates_for_tts("19 Agasti", _dictionary())
    assert "kuNcwaba" in result.tts_text
    assert ZU_NATIVE_DATE_PRONUNCIATION_VERSION.startswith(
        "mathula-zu-native-calendar-date"
    )


# Real user-reported production defect (job 8372120960474ef6b1d75af7de51a605):
# a bare year with no day attached ("Month YYYY", or the Zulu-possessive
# "ka-YYYY" glue) fell through the day-requiring _ZU_CALENDAR_DATE mechanism
# entirely and was read either digit-by-digit or as a raw cardinal "amount"
# instead of a natural year. Calibrated live via real Azure zu-ZA-ThandoNeural
# synthesis + en-ZA STT round-trip; the user picked plain English spelling
# ("twenty twenty-four") over a phonetic respelling by ear.
@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "kuningi okwakwenzeka ngoDisemba 2024, futhi okuningi",
            "kuningi okwakwenzeka ngoDisemba twenty twenty-four, futhi okuningi",
        ),
        (
            "kwakunguNovemba ka-2024, uDisemba ka-2024 noma uJanuwari ka-2025",
            "kwakunguNovemba ka-twenty twenty-four, uDisemba ka-twenty twenty-four "
            "noma uJanuwari ka-twenty twenty-faif",
        ),
        ("ngonyaka ka-1999", "ngonyaka ka-nineteen ninety-nine"),
        ("ngonyaka ka-2000", "ngonyaka ka-twenty hundred"),
        ("ngonyaka ka-2005", "ngonyaka ka-twenty oh faif"),
        (
            "ngasekupheleni kuka-2024, mhlawumbe ekuqaleni kuka-2025",
            "ngasekupheleni kuka-twenty twenty-four, mhlawumbe ekuqaleni "
            "kuka-twenty twenty-faif",
        ),
        ("ngonyaka ka-2001", "ngonyaka ka-twenty oh wani"),
    ),
)
def test_bare_year_with_no_day_gets_a_natural_english_year_reading(
    source: str, expected: str,
) -> None:
    result = _dictionary().apply(source)
    assert result.spoken_text == source
    assert result.tts_text == expected


def test_a_year_shaped_number_with_no_month_or_ka_context_keeps_its_reference_reading() -> None:
    # Deliberately scoped to real date-context evidence (a month name, or the
    # "ka-" glue) -- a first, broader attempt at this fix (any bare 19xx/20xx
    # token) was caught by the existing suite regressing a genuinely unrelated
    # docket/reference number that happens to be year-shaped. "elingu-" is not
    # the date-reference "ka-" glue, so this stays untouched by either
    # mechanism (same as before this fix, out of its scope).
    result = _dictionary().apply("Icala elingu-2024 alikaqedwa.")
    assert "twenty twenty-four" not in result.tts_text


# Real user-reported production defect (job fb3d08b63fed4d90922b08f7e325b906):
# a bare day-of-month number with no ordinal suffix, in English "Month <day>"
# word order (as opposed to the day-before-month "mhla ka-3 Mashi" order
# _ZU_CALENDAR_DATE already handles), was left as a raw digit and Azure's own
# number reading came out sounding like "on" rather than "1". Calibrated live
# via real Azure zu-ZA-ThembaNeural synthesis + zu-ZA STT round-trip -- "wani"
# recovered a clean "1", matching the same fix already applied to
# _ZU_YEAR_ONES[1] for the bare-year case.
#
# Redirected by real user feedback, same session: "we should be code
# switching to 'November first'... most native speakers in South Africa
# code switch dates to english instead of the native pronunciation." Live
# re-verification confirmed the month's own spelling matters, not just the
# digit -- "November first" (English month + English ordinal) round-trips
# cleanly, while "Novemba first" (Zulu month + English ordinal) mispronounces
# "first" as "fast". Only "November"/"1" was directly re-verified after this
# redirect (the real reported case); other months follow the same code-
# switch direction without individual live re-verification.
@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("ahead of Novemba 1", "ahead of November first"),
        # The real committed text: a Zulu class-11 relative concord ("lwa-")
        # glued directly onto the already-recognized "ngo-" month alias with
        # no space -- confirmed this only works because _ZU_MONTH_BARE_DAY
        # reuses _GLUED_PREFIX_LOOKBEHIND rather than a fixed prefix list.
        # The concord prefix itself stays in Zulu (sentence grammar, not
        # part of the date) -- only the month/day lexical content switches.
        (
            "lohulumeni basekhaya lwangoNovemba 1.",
            "lohulumeni basekhaya lwangoNovember first.",
        ),
        ("kwenzeka ngoDisemba 21", "kwenzeka ngoDecember twenty-first"),
        ("kwenzeka ngoDisemba 15", "kwenzeka ngoDecember fifteenth"),
    ),
)
def test_bare_day_of_month_with_no_ordinal_suffix_gets_a_natural_english_code_switch_reading(
    source: str, expected: str,
) -> None:
    result = _dictionary().apply(source)
    assert result.spoken_text == source
    assert result.tts_text == expected


def test_a_full_calendar_date_is_unaffected_by_the_bare_day_mechanism() -> None:
    # The day-before-month, day+month+year case (_ZU_CALENDAR_DATE) must not
    # also get a second, overlapping bare-day candidate.
    result = _dictionary().apply("Sihlehlisele umhla ka-3 Mashi 2026.")
    assert result.tts_text == "Sihlehlisele umhla wesithathu kuNdasa 2026."


def test_an_ordinal_suffixed_day_is_not_touched_by_the_bare_day_mechanism() -> None:
    # "1st"/"4th" has a trailing word character right after the digit, so the
    # (?!\w) boundary excludes it -- this is a different, already-solved
    # concern (the turn-block literal-preservation check), not TTS reading.
    result = _dictionary().apply("ngesikhathi sikaNovemba 1st.")
    assert "first" not in result.tts_text


def test_a_full_day_month_year_date_still_leaves_its_own_year_digits_raw() -> None:
    # The day-attached case (_ZU_CALENDAR_DATE) is untouched by this fix --
    # confirmed deliberately scoped to the day-less case only, matching the
    # existing established behavior asserted elsewhere in this file (e.g.
    # test_target_text_audits_date_change_without_changing_caption).
    result = _dictionary().apply("Sihlehlisele umhla ka-3 Mashi 2026.")
    assert result.tts_text == "Sihlehlisele umhla wesithathu kuNdasa 2026."


def test_a_five_digit_reference_number_is_unaffected_by_the_year_exclusion() -> None:
    result = _dictionary().apply("Icala elingu-30245 alikaqedwa.")
    assert "twenty twenty-four" not in result.tts_text
