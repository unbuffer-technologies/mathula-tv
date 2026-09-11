from pathlib import Path

from mathula_tv.azure_stt import AzureSpeechError
from mathula_tv.entity_pronunciation_audit import (
    EntityPronunciationAuditOptions,
    audit_entity_pronunciations,
    entity_surface_recognized,
)


def test_entity_surface_recognized_accepts_fused_zulu_prefixes_and_boundaries():
    assert entity_surface_recognized(
        "SA First Forum",
        "lokho kuyingxenye ye-s a first forum umgomo wethu",
    )
    assert entity_surface_recognized(
        "South African Police Service",
        "ibizwa ngokuthi South African Police Service namuhla",
    )


def test_entity_surface_recognized_rejects_wrong_english_vowel_identity():
    assert not entity_surface_recognized(
        "SA First Forum",
        "lokho kuyingxenye ye-s a fast forum umgomo wethu",
    )
    assert not entity_surface_recognized(
        "The Big Five cartel",
        "lokho yi Big fever cartel",
    )
    assert not entity_surface_recognized(
        "Independent Police Investigative Directorate",
        "Independent Policy Investigative Directorate",
    )
    assert not entity_surface_recognized(
        "Investigating Directorate Against Corruption",
        "Investigating Directorate Against Correction",
    )


def test_entity_surface_recognized_accepts_number_word_and_accent_spelling():
    assert entity_surface_recognized(
        "The Big Five cartel",
        "lokho kubizwa ngokuthi de big 5 cartel",
    )


def test_entity_surface_recognized_accepts_en_za_boundary_artifacts():
    assert entity_surface_recognized(
        "Port Shepstone",
        "lokho kubizwa ngokuthi Port Shapestone ngiyaphinda Port Shapestone",
    )
    assert entity_surface_recognized(
        "Yellow jersey truck",
        "lokho kubizwa ngokuthi yellow jazzy truck",
    )
    assert entity_surface_recognized(
        "Kgosi Mampuru II Correctional Centre",
        "kosi Mampuru the second correctional centre",
    )
    assert entity_surface_recognized(
        "Durban Pier 2",
        "dor bin piah to",
    )
    assert entity_surface_recognized(
        "The Hawks",
        "inhlangano ngokwe-dha hawks izophenya",
    )
    assert entity_surface_recognized(
        "The Hawks",
        "inkantolo ithi ngokwe dia hawks umbiko uzocacisa udaba",
    )
    assert not entity_surface_recognized(
        "The Hawks",
        "inhlangano ngokwe-te hawks izophenya",
    )
    assert not entity_surface_recognized(
        "The Hawks",
        "inhlangano itehoks ithe izophenya",
    )


def test_entity_surface_recognized_rejects_meaning_changing_surface_matches():
    assert not entity_surface_recognized("blue lights", "blue liquids")
    assert not entity_surface_recognized("yellow jersey truck", "yellow chassis truck")
    assert not entity_surface_recognized("Medicare Tshwane District", "Medicare Tshwane Street")


def test_entity_surface_recognized_requires_literal_single_letter_designation():
    assert entity_surface_recognized("Witness A", "namuhla u Witness A ukhuluma")
    assert not entity_surface_recognized("Witness A", "namuhla u Witness E ukhuluma")


def test_inventory_audit_records_empty_stt_as_failure_without_aborting(
    tmp_path, monkeypatch
):
    registry = tmp_path / "registry.json"
    registry.write_text(
        '{"entities":[{"entity_id":"phrase_one","entity_type":"phrase",'
        '"canonical_text":"First","active":true,'
        '"translation_policy":"protected_token","spoken_forms":{}}]}',
        encoding="utf-8",
    )

    class TTS:
        sample_rate = 24_000

        def synthesize(self, request, output, *, force=False):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"raw")
            return type("Result", (), {"duration_ms": 100})()

    class STT:
        def transcribe(self, audio: Path, locale: str):
            raise AzureSpeechError("Azure Speech returned an empty transcription")

    def timing(source, output, **kwargs):
        output.write_bytes(b"production")

    monkeypatch.setattr(
        "mathula_tv.entity_pronunciation_audit._timing_adjust", timing
    )
    report = audit_entity_pronunciations(
        registry_path=registry,
        output_dir=tmp_path / "out",
        tts_backend=TTS(),
        stt_backend=STT(),
        options=EntityPronunciationAuditOptions(voices=("zu-ZA-TestNeural",)),
    )

    assert report["observation_count"] == 1
    assert report["failed_observation_count"] == 1
    assert report["observations"][0]["recognized_text"] == ""
