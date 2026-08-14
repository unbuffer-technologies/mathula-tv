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
