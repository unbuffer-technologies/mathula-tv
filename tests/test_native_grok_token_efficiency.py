from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.ai_provider import StructuredAIRequest
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.foundry_grok import FoundryGrokConfig, FoundryGrokProvider
from mathula_tv.native_dub import run_native_pass1, run_native_pass2


class _HTTPResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _SequenceSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _HTTPResponse(self.payloads.pop(0))


def _provider(session, *, repairs=0):
    return FoundryGrokProvider(
        FoundryGrokConfig(
            api_key="secret",
            endpoint="https://example.services.ai.azure.com",
            deployment="grok-test",
            max_retries=0,
            max_schema_repairs=repairs,
        ),
        session=session,
        sleep=lambda _: None,
    )


def test_grok_sends_compact_contract_not_full_json_schema():
    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    ])
    provider = _provider(session)
    provider.complete_json(
        operation="compact-contract",
        system_prompt="Return JSON.",
        payload={"sentinel": "payload"},
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
    )
    body = session.calls[0][1]["json"]
    wire = json.loads(body["messages"][1]["content"])
    assert "response_json_schema" not in wire
    assert wire["output_contract"] == {"ok": "boolean"}


def test_schema_repair_does_not_resend_original_large_payload():
    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":"wrong"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4},
        },
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        },
    ])
    provider = _provider(session, repairs=1)
    provider.complete_json(
        operation="repair",
        system_prompt="Use HUGE_SENTINEL only as input evidence.",
        payload={"text": "HUGE_SENTINEL" * 1000},
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
    )
    second_body = session.calls[1][1]["json"]
    second_text = json.dumps(second_body["messages"], ensure_ascii=False)
    assert "HUGE_SENTINEL" not in second_text
    assert "invalid_output" in second_text


def test_schema_repair_logs_the_concise_validation_reason_not_a_full_dump():
    # Regression test for a real diagnostic gap: the actual jsonschema error was only
    # ever sent to Grok in the repair request, never surfaced to the console -- so a
    # systematic failure (every call needing a repair round for the same recurring
    # reason) looked identical in the log to an occasional, unremarkable one.
    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"spoken_text": "hi"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true, "spoken_text": "hi"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        },
    ])
    messages: list[str] = []
    provider = FoundryGrokProvider(
        FoundryGrokConfig(
            api_key="secret",
            endpoint="https://example.services.ai.azure.com",
            deployment="grok-test",
            max_retries=0,
            max_schema_repairs=1,
        ),
        session=session,
        sleep=lambda _: None,
        progress=messages.append,
    )
    provider.complete_json(
        operation="test_op",
        system_prompt="Return JSON.",
        payload={"sentinel": "payload"},
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok", "spoken_text"],
            "properties": {"ok": {"type": "boolean"}, "spoken_text": {"type": "string"}},
        },
    )
    failure_lines = [m for m in messages if "schema validation failed" in m]
    assert len(failure_lines) == 1
    # The concise, one-line jsonschema message ("'ok' is a required property"), not a
    # multi-line dump of the whole failing instance and schema.
    assert "'ok' is a required property" in failure_lines[0]
    assert failure_lines[0].count("\n") == 0


def test_max_schema_repairs_default_is_two_not_one():
    # Regression test for a real production crash: a schema-invalid first attempt
    # (wrong top-level JSON shape) survived one repair round with a DIFFERENT, still-
    # invalid shape, exhausting the old default of 1 and crashing the whole run.
    config = FoundryGrokConfig(
        api_key="secret", endpoint="https://example.services.ai.azure.com", deployment="grok-test",
    )
    assert config.max_schema_repairs == 2


def test_max_schema_repairs_ceiling_allows_three():
    config = FoundryGrokConfig(
        api_key="secret", endpoint="https://example.services.ai.azure.com", deployment="grok-test",
        max_schema_repairs=3,
    )
    assert config.max_schema_repairs == 3
    try:
        FoundryGrokConfig(
            api_key="secret", endpoint="https://example.services.ai.azure.com", deployment="grok-test",
            max_schema_repairs=4,
        )
    except ValueError as exc:
        assert "MATHULA_TV_GROK_SCHEMA_REPAIRS" in str(exc)
    else:
        raise AssertionError("expected ValueError for max_schema_repairs=4")


def test_two_repair_rounds_recovers_from_two_consecutive_wrong_shapes():
    # Mirrors the exact real failure sequence: initial attempt wraps results in the
    # wrong top-level shape (dict keyed by id), the FIRST repair also gets it wrong
    # (bare array, no wrapper), and only the SECOND repair round actually succeeds --
    # this must not crash with the new default of 2 repair rounds.
    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"g1": {"ok": true}}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '[{"ok": true}]'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"masks": [{"ok": true}]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    ])
    provider = _provider(session, repairs=2)
    response = provider.complete_json(
        operation="test_op",
        system_prompt="Return JSON.",
        payload={"sentinel": "payload"},
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["masks"],
            "properties": {"masks": {"type": "array"}},
        },
    )
    assert response.data == {"masks": [{"ok": True}]}
    assert response.attempts == 3


class _CapturePassProvider:
    provider = "azure-foundry-grok"

    def __init__(self, max_output_tokens=8192):
        self.config = SimpleNamespace(deployment="grok-test", max_output_tokens=max_output_tokens)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        op = kwargs["operation"]
        if op == "native_stt_restoration":
            data = {"corrections": []}
        elif op.startswith("native_natural_translation"):
            data = {
                "translations": [
                    {"segment_id": item["id"], "spoken_text": f"Zulu {item['id']}"}
                    for item in kwargs["payload"]["segments"]
                ]
            }
        else:
            raise AssertionError(op)
        return SimpleNamespace(
            data=data,
            model="grok-test",
            input_tokens=10,
            output_tokens=5,
            attempts=1,
        )


def _write_small_transcript(job_root: Path):
    atomic_write_json(
        job_root / "analysis" / "transcript_en.json",
        {
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "A simple sentence.",
                }
            ],
        },
    )


def test_pass1_wire_drops_timing_and_source_echo(tmp_path: Path):
    job_root = tmp_path / "job"
    _write_small_transcript(job_root)
    provider = _CapturePassProvider()
    result = run_native_pass1(job_root=job_root, provider=provider)
    call = provider.calls[0]
    assert set(call["payload"]["segments"][0]) == {"id", "spk", "text"}
    assert call["max_output_tokens"] == 4096
    assert result["token_strategy"]["source_text_echo_disabled"] is True
    assert result["token_strategy"]["full_json_schema_sent_to_model"] is False


def test_pass2_prebatches_without_repeating_whole_transcript(tmp_path: Path):
    job_root = tmp_path / "job"
    segments = []
    for i in range(1, 21):
        segments.append(
            {
                "segment_id": f"seg-{i:02d}",
                "speaker_id": "SPEAKER_00",
                "start_ms": (i - 1) * 1000,
                "end_ms": i * 1000,
                "source_text": "source " + ("word " * 120),
                "restored_text": "source " + ("word " * 120),
                "changed": False,
            }
        )
    atomic_write_json(job_root / "native_dub" / "pass1_restored_transcript.json", {"segments": segments})
    provider = _CapturePassProvider(max_output_tokens=1000)
    result = run_native_pass2(job_root=job_root, provider=provider)
    translation_calls = [c for c in provider.calls if c["operation"].startswith("native_natural_translation")]
    assert len(translation_calls) > 1
    assert result["prebatched_for_token_budget"] is True
    assert result["token_strategy"]["whole_transcript_repeated_per_batch"] is False
    for call in translation_calls:
        payload = call["payload"]
        assert "whole_transcript_context" not in payload
        assert "context_ledger" in payload
        assert set(payload["segments"][0]) == {"id", "spk", "text", "src_ms", "target_raw_ms"}
        assert payload["segments"][0]["target_raw_ms"] == round(payload["segments"][0]["src_ms"] * 1.06)


def test_editorial_grok_replaces_legacy_huge_prompt_and_duplicate_transcripts():
    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
        }
    ])
    provider = _provider(session)
    request = StructuredAIRequest(
        operation="editorial_package",
        payload={
            "job_id": "j",
            "target_language": "zu-ZA",
            "source_duration_seconds": 60.0,
            "maximum_intro_cut_seconds": 15.0,
            "complete_english_transcript": {
                "segments": [
                    {"segment_id": "s1", "speaker_id": "S0", "source_text": "English evidence"}
                ]
            },
            "candidate_boundaries": [
                {"block_id": "s1", "start_seconds": 0, "end_seconds": 2, "source_text": "English evidence", "approved_isiZulu_text": "Zulu"}
            ],
            "full_rendered_transcript": [
                {"block_id": "s1", "source_text": "English evidence", "approved_isiZulu_text": "Zulu duplicate"}
            ],
            "grounded_context": {"native_context_ledger": {"summary": "story", "entities": [], "speaker_roles": [], "story_beats": [{"segment_ids": ["s1"], "note": "beat"}]}},
            "requirements": {"very_verbose_policy": True},
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="legacy-large",
        system_prompt="LEGACY_HUGE_PROMPT " * 1000,
        response_schema_version="v1",
    )
    provider.complete_structured(request)
    body = session.calls[0][1]["json"]
    joined = json.dumps(body["messages"], ensure_ascii=False)
    assert "LEGACY_HUGE_PROMPT" not in joined
    assert "full_rendered_transcript" not in joined
    assert "very_verbose_policy" not in joined
    assert "evidence_segments" in joined
    assert body["max_tokens"] <= 3500


def test_editorial_compact_prompt_explicitly_forbids_a_bare_candidates_array():
    """Regression test for a real production failure, twice in a row.

    Grok returned a bare top-level array of 3 title/hook candidates instead of the
    required wrapper object (which also needs primary_story_angle, caption,
    topic_hashtags, etc. alongside candidates). It finished normally (no truncation
    warning), so it genuinely believed the array alone was the complete answer -- the
    old compact prompt described 5 deliverables as separate prose bullets without
    ever stating they must all be fields of one JSON object, and described
    "footer-hook candidates" and "opening boundary" as if they were separate lists
    rather than one unified `candidates` structure. The fix must say so explicitly.
    """

    session = _SequenceSession([
        {
            "model": "grok-test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
        }
    ])
    provider = _provider(session)
    request = StructuredAIRequest(
        operation="editorial_package",
        payload={
            "job_id": "j",
            "target_language": "zu-ZA",
            "source_duration_seconds": 60.0,
            "maximum_intro_cut_seconds": 15.0,
            "complete_english_transcript": {
                "segments": [{"segment_id": "s1", "speaker_id": "S0", "source_text": "English evidence"}]
            },
            "candidate_boundaries": [
                {"block_id": "s1", "start_seconds": 0, "end_seconds": 2, "source_text": "English evidence", "approved_isiZulu_text": "Zulu"}
            ],
            "full_rendered_transcript": [
                {"block_id": "s1", "source_text": "English evidence", "approved_isiZulu_text": "Zulu duplicate"}
            ],
            "grounded_context": {"native_context_ledger": {"summary": "story", "entities": [], "speaker_roles": [], "story_beats": []}},
            "requirements": {},
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="v1",
        system_prompt="unused for editorial_package -- the compact prompt replaces it",
        response_schema_version="v1",
    )
    provider.complete_structured(request)
    sent_system_prompt = session.calls[0][1]["json"]["messages"][0]["content"]
    assert "never return a bare array" in sent_system_prompt
    assert "ONE unified object" in sent_system_prompt
    assert "opening_boundary_block_id" in sent_system_prompt
    assert "candidates" in sent_system_prompt
    assert "primary_story_angle" in sent_system_prompt
    assert "topic_hashtags" in sent_system_prompt
