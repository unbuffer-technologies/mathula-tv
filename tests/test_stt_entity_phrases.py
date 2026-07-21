from __future__ import annotations

import json
from pathlib import Path

from mathula_tv.azure_stt import request_definition
from mathula_tv.stt_phrases import configure_source_stt_backend, load_stt_phrase_bundle


def test_request_definition_adds_azure_fast_phrase_list():
    definition = request_definition(
        "en-ZA",
        max_speakers=10,
        phrases=("Madlanga", "Thabo Bester"),
        biasing_weight=1.6,
    )
    assert definition["phraseList"] == {
        "phrases": ["Madlanga", "Thabo Bester"],
        "biasingWeight": 1.6,
    }


def test_phrase_bundle_prioritizes_overrides_and_excludes_unsafe_alias(tmp_path: Path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "mathula-entity-registry-v1",
                "entities": [
                    {
                        "active": True,
                        "canonical_text": "Vusimuzi Cat Matlala",
                        "display_text": 'Vusimuzi "Cat" Matlala',
                        "stt_phrases": ["Cat Matlala"],
                        "aliases": ["Cat", "Matlala"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps({"phrases": ["Madlanga supermarket", "hoshkhari"]}),
        encoding="utf-8",
    )

    bundle = load_stt_phrase_bundle(
        registry_path=registry,
        overrides_path=overrides,
        max_phrases=20,
        biasing_weight=1.6,
    )

    assert bundle.phrases[:2] == ("Madlanga supermarket", "hoshkhari")
    assert "Cat Matlala" in bundle.phrases
    assert "Cat" not in bundle.phrases
    assert "Cat" in bundle.excluded_unsafe_aliases


def test_configure_source_backend_supports_router_without_affecting_other_backends(
    tmp_path: Path,
    monkeypatch,
):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "mathula-entity-registry-v1",
                "entities": [
                    {
                        "active": True,
                        "canonical_text": "Thabo Bester",
                        "display_text": "Thabo Bester",
                        "stt_phrases": ["Thabo Bester"],
                        "aliases": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"phrases": ["hoshkhari"]}), encoding="utf-8")

    monkeypatch.setenv("MATHULA_TV_STT_ENTITY_REGISTRY", str(registry))
    monkeypatch.setenv("MATHULA_TV_STT_PHRASE_OVERRIDES", str(overrides))

    class Fast:
        phrase_list = ()
        phrase_biasing_weight = None

    class Router:
        fast = Fast()

    router = Router()
    bundle = configure_source_stt_backend(router)

    assert bundle is not None
    assert router.fast.phrase_list == ("hoshkhari", "Thabo Bester")
    assert router.fast.phrase_biasing_weight == 1.6

    class BlindAuditBackend:
        pass

    assert configure_source_stt_backend(BlindAuditBackend()) is None
