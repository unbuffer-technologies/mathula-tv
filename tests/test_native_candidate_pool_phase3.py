"""The Zulu terminology glossary (replaces rolling continuity_context) used by
the --candidate-pool architecture. The English length ladder this file used to
also test was removed in Phase 9 (see the approved plan at
C:\\Users\\mokgethwa\\.claude\\plans\\calm-wiggling-parasol.md) -- each sentence
now gets exactly ONE natural Grok translation, and timing is instead handled at
the discourse-segment level (see test_native_timing_regions.py).

Fake providers here follow the same shape used throughout this file's other
provider-facing tests (test_native_qa_concurrency.py, test_referent_audit.py):
plain objects exposing .config and .complete_json(...), not a real
FoundryGrokProvider -- _ensure_zulu_glossary does not type-check the provider,
it just calls the protocol.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from mathula_tv.native_dub import (
    _ensure_zulu_glossary,
    _glossary_conformance,
    _normalize_zulu_glossary,
    native_dub_paths,
)
from mathula_tv.atomic_io import atomic_write_json


def _valid_glossary() -> dict:
    return {
        "terms": [{"english": "load shedding", "zulu": "ukucinywa kukagesi", "kind": "other"}],
        "do_not_translate": ["Eskom"],
        "register": {"formality": "formal", "address_form": "neutral", "tense_default": "present"},
    }


# --- _normalize_zulu_glossary -------------------------------------------------------


def test_normalize_zulu_glossary_accepts_a_well_formed_response():
    normalized = _normalize_zulu_glossary(_valid_glossary())
    assert normalized["terms"][0]["zulu"] == "ukucinywa kukagesi"
    assert normalized["do_not_translate"] == ["Eskom"]
    assert normalized["register"]["formality"] == "formal"


def test_normalize_zulu_glossary_falls_back_to_empty_for_a_non_mapping():
    # No legacy fallback exists for a brand-new artifact -- an empty glossary
    # (no terms/do_not_translate, no register opinion) is the correct degenerate
    # case, not an error.
    normalized = _normalize_zulu_glossary(None)
    assert normalized == {"terms": [], "do_not_translate": [], "register": {}}


def test_normalize_zulu_glossary_rejects_a_malformed_register():
    with pytest.raises(jsonschema.ValidationError):
        _normalize_zulu_glossary({"terms": [], "do_not_translate": [], "register": {"formality": "formal"}})


def test_normalize_zulu_glossary_truncates_oversized_lists():
    bloated = {
        "terms": [{"english": f"e{i}", "zulu": f"z{i}", "kind": "other"} for i in range(60)],
        "do_not_translate": [f"dnt{i}" for i in range(60)],
        "register": {"formality": "formal", "address_form": "neutral", "tense_default": "present"},
    }
    normalized = _normalize_zulu_glossary(bloated)
    assert len(normalized["terms"]) == 40
    assert len(normalized["do_not_translate"]) == 30


# --- _glossary_conformance -----------------------------------------------------------


def test_glossary_conformance_counts_matching_terms():
    glossary = _valid_glossary()
    source = "There was load shedding again last night."
    matching = "Bekukhona ukucinywa kukagesi phezu kwalokho."
    assert _glossary_conformance(glossary, source, matching) == 1


def test_glossary_conformance_is_zero_when_zulu_form_absent():
    glossary = _valid_glossary()
    source = "There was load shedding again last night."
    non_matching = "Kwakukhona okuthile phezu kwalokho."
    assert _glossary_conformance(glossary, source, non_matching) == 0


# --- _ensure_zulu_glossary: generation + caching -------------------------------------


class _FakeGlossaryProvider:
    def __init__(self, glossary: dict):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._glossary = glossary
        self.calls = 0

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls += 1
        assert operation == "native_zulu_glossary"
        return SimpleNamespace(
            data={"zulu_terminology_glossary": self._glossary},
            model="grok-test", input_tokens=100, output_tokens=50, attempts=1,
        )


def _job_root_with_pass1(tmp_path: Path) -> Path:
    job_root = tmp_path / "job"
    paths = native_dub_paths(job_root)
    paths.pass1.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.pass1, {
        "corrections": [], "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
    })
    return job_root


def test_ensure_zulu_glossary_generates_and_writes_the_artifact(tmp_path):
    job_root = _job_root_with_pass1(tmp_path)
    provider = _FakeGlossaryProvider(_valid_glossary())
    segments = [{"segment_id": "s1", "speaker_id": "S", "source_text": "There was load shedding."}]

    glossary, usage = _ensure_zulu_glossary(
        job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None,
    )

    assert provider.calls == 1
    assert glossary["terms"][0]["english"] == "load shedding"
    assert usage == {"input_tokens": 100, "output_tokens": 50, "attempts": 1}
    assert native_dub_paths(job_root).zulu_glossary.is_file()


def test_ensure_zulu_glossary_reuses_cache_without_a_second_call(tmp_path):
    job_root = _job_root_with_pass1(tmp_path)
    provider = _FakeGlossaryProvider(_valid_glossary())
    segments = [{"segment_id": "s1", "speaker_id": "S", "source_text": "There was load shedding."}]

    _ensure_zulu_glossary(job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None)
    assert provider.calls == 1

    glossary, usage = _ensure_zulu_glossary(
        job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None,
    )
    assert provider.calls == 1  # not called again
    assert glossary["terms"][0]["english"] == "load shedding"
    assert usage == {"input_tokens": 100, "output_tokens": 50, "attempts": 1}


def test_ensure_zulu_glossary_regenerates_when_pass1_changes(tmp_path):
    job_root = _job_root_with_pass1(tmp_path)
    provider = _FakeGlossaryProvider(_valid_glossary())
    segments = [{"segment_id": "s1", "speaker_id": "S", "source_text": "There was load shedding."}]
    _ensure_zulu_glossary(job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None)
    assert provider.calls == 1

    paths = native_dub_paths(job_root)
    atomic_write_json(paths.pass1, {
        "corrections": [{"segment_id": "s1", "corrected_text": "x", "reason_code": "asr_substitution"}],
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
    })

    _ensure_zulu_glossary(job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None)
    assert provider.calls == 2


def test_ensure_zulu_glossary_force_regenerates_even_with_a_fresh_cache(tmp_path):
    job_root = _job_root_with_pass1(tmp_path)
    provider = _FakeGlossaryProvider(_valid_glossary())
    segments = [{"segment_id": "s1", "speaker_id": "S", "source_text": "There was load shedding."}]
    _ensure_zulu_glossary(job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, progress=None)

    _ensure_zulu_glossary(
        job_root=job_root, provider=provider, pass1={}, segments=segments, context_ledger={}, force=True, progress=None,
    )
    assert provider.calls == 2


