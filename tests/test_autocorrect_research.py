from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv.autocorrect_research import (
    apply_researched_corrections,
    build_research_profile,
    display_name_research_corrections,
    extract_name_candidates,
    run_name_research_autocorrection,
    validate_researched_corrections,
)


def transcript() -> dict:
    return {
        "schema_version": "transcript-v1",
        "complete_text": "Kanyaku said the charges against Khumalo were withdrawn.",
        "segments": [
            {
                "segment_id": "seg-1",
                "start": 0.0,
                "end": 4.0,
                "source_text": "Kanyaku said the charges against Khumalo were withdrawn.",
                "words": [
                    {"text": "Kanyaku", "start": 0.0, "end": 0.6},
                    {"text": "said", "start": 0.6, "end": 0.9},
                ],
            }
        ],
        "words": [
            {"text": "Kanyaku", "start": 0.0, "end": 0.6},
            {"text": "said", "start": 0.6, "end": 0.9},
        ],
        "autocorrect": {"schema_version": "autocorrect-v1"},
    }


def correction() -> dict:
    return {
        "segment_id": "seg-1",
        "raw_text": "Kanyaku",
        "canonical_text": "Kganyago",
        "entity_type": "person",
        "confidence": 0.99,
        "source_urls": [
            "https://www.npa.gov.za/media/kganyago-khumalo",
            "https://www.sabcnews.com/sabcnews/kganyago-khumalo/",
        ],
        "story_match_terms": ["Khumalo", "charges withdrawn", "NPA"],
        "reason": "The same NPA spokesperson is named Kaizer Kganyago.",
    }


class Metadata:
    server_tool_evidence = (
        {"url": "https://www.npa.gov.za/media/kganyago-khumalo", "title": "NPA"},
        {"url": "https://www.sabcnews.com/sabcnews/kganyago-khumalo/", "title": "SABC"},
    )

    def to_dict(self):
        return {
            "provider": "test",
            "server_tool_evidence": list(self.server_tool_evidence),
        }


def research_response(request) -> dict:
    """Return one entity-level correction for Kanyaku and unresolved peers."""

    corrections = []
    unresolved = []
    for candidate in request.payload["candidate_entities"]:
        aliases = {
            str(value).casefold()
            for value in [
                candidate.get("representative_text"),
                *candidate.get("aliases", []),
            ]
            if value
        }
        if "kanyaku" in aliases:
            item = correction()
            item.pop("segment_id", None)
            item.pop("raw_text", None)
            item["entity_id"] = candidate["entity_id"]
            corrections.append(item)
        else:
            unresolved.append(
                {
                    "entity_id": candidate["entity_id"],
                    "reason": "already correctly spelled or insufficient evidence",
                }
            )
    return {
        "schema_version": "mathula-autocorrect-name-research-v2",
        "corrections": corrections,
        "unresolved_candidates": unresolved,
    }


class Provider:
    def __init__(self):
        self.calls = 0
        self.last_request = None

    def complete_structured(self, request):
        self.calls += 1
        self.last_request = request
        return SimpleNamespace(data=research_response(request), metadata=Metadata())


class EntityProvider:
    """Deterministic test double for the transcript-wide AI inventory phase."""

    def __init__(self, entities=None, error: Exception | None = None):
        self.entities = entities
        self.error = error
        self.calls = 0
        self.last_request = None

    def complete_structured(self, request):
        self.calls += 1
        self.last_request = request
        if self.error is not None:
            raise self.error
        assert request.operation == "autocorrect_entity_inventory"
        if self.entities is None:
            candidates = extract_name_candidates(request.payload["transcript"])
            entities = [
                {
                    "entity_id": f"entity-{index}",
                    "canonical_name": "",
                    "entity_type": "person",
                    "confidence": 0.5,
                    "needs_research": True,
                    "reason": "test fixture marks heuristic name as uncertain",
                    "surface_forms": [item["raw_text"]],
                    "mentions": [
                        {
                            "segment_id": item["segment_id"],
                            "raw_text": item["raw_text"],
                        }
                    ],
                }
                for index, item in enumerate(candidates, start=1)
            ]
        else:
            entities = self.entities(request) if callable(self.entities) else self.entities
        return SimpleNamespace(
            data={
                "schema_version": "mathula-autocorrect-entity-inventory-v1",
                "entities": entities,
            },
            metadata=Metadata(),
        )


def test_extracts_sentence_initial_name_candidate() -> None:
    candidates = extract_name_candidates(transcript())
    assert any(item["raw_text"] == "Kanyaku" for item in candidates)


def test_accepts_two_source_same_story_correction() -> None:
    accepted, rejected = validate_researched_corrections(
        {"corrections": [correction()]},
        candidates=extract_name_candidates(transcript()),
        actual_evidence_urls=[item["url"] for item in Metadata.server_tool_evidence],
    )
    assert len(accepted) == 1
    assert rejected == []
    assert accepted[0]["canonical_text"] == "Kganyago"


def test_rejects_unsupported_name_replacement() -> None:
    item = correction()
    item["source_urls"] = ["https://example.com/claim"]
    accepted, rejected = validate_researched_corrections(
        {"corrections": [item]},
        candidates=extract_name_candidates(transcript()),
        actual_evidence_urls=["https://example.com/claim"],
    )
    assert accepted == []
    assert rejected[0]["validation_reason"] == (
        "insufficient_independent_or_authoritative_sources"
    )


def test_applies_name_to_authoritative_transcript_and_words() -> None:
    corrected, applied = apply_researched_corrections(transcript(), [correction()])
    assert len(applied) == 1
    assert corrected["segments"][0]["source_text"].startswith("Kganyago said")
    assert corrected["segments"][0]["words"][0]["text"] == "Kganyago"
    assert corrected["words"][0]["text"] == "Kganyago"
    assert corrected["complete_text"].startswith("Kganyago said")
    assert corrected["autocorrect"]["name_research"]["applied_count"] == 1


def test_research_call_is_cached_by_authoritative_input(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-1"
    (job_root / "analysis").mkdir(parents=True)
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-1",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    first = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=provider,

        entity_provider=EntityProvider(),
    )
    second = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=provider,

        entity_provider=EntityProvider(),
    )

    assert provider.calls == 1
    assert first["transcript"]["segments"][0]["source_text"].startswith("Kganyago")
    assert second["transcript"]["segments"][0]["source_text"].startswith("Kganyago")
    assert second["artifact"]["cache_reused"] is True


def test_displays_applied_correction_on_stderr(capsys, monkeypatch) -> None:
    monkeypatch.setenv("MATHULA_TV_AUTOCORRECT_SHOW_CORRECTIONS", "1")
    artifact = {
        "status": "completed",
        "candidate_count": 2,
        "applied_count": 1,
        "unresolved_count": 1,
        "rejected_count": 0,
        "applied_corrections": [correction()],
    }

    display_name_research_corrections(artifact)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'seg-1: "Kanyaku" -> "Kganyago"' in captured.err
    assert "99%" in captured.err
    assert "npa.gov.za" in captured.err
    assert "sabcnews.com" in captured.err
    assert "applied 1 correction(s); unresolved 1; rejected 0" in captured.err


def test_console_corrections_can_be_disabled(capsys, monkeypatch) -> None:
    monkeypatch.setenv("MATHULA_TV_AUTOCORRECT_SHOW_CORRECTIONS", "0")
    display_name_research_corrections(
        {
            "status": "completed",
            "candidate_count": 1,
            "applied_count": 1,
            "applied_corrections": [correction()],
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_anonymous_witness_designations_are_not_research_candidates() -> None:
    source = transcript()
    source["segments"][0]["source_text"] = (
        "Witness M appeared before the Madlanga Commission with Andrea Johnson."
    )
    candidates = extract_name_candidates(source)
    raw = {item["raw_text"] for item in candidates}
    assert "Witness M" not in raw
    assert any("Andrea Johnson" in item for item in raw)


def test_builds_madlanga_commission_research_profile() -> None:
    profile = build_research_profile(
        transcript={
            "complete_text": "The Madlanga Commission heard Advocate Andrea Johnson."
        },
        source_metadata={
            "youtube_source": {
                "title": "Madlanga Commission Day 146 | Andrea Johnson",
                "channel": "SABC News",
            }
        },
        local_context={
            "entity_bindings": {
                "entities_found_global": ["case_or_inquiry_madlanga_commission"]
            },
            "context_providers": [],
        },
    )
    assert profile["name"] == "madlanga_commission"
    assert "criminaljusticecommission.org.za" in profile["official_domains"]
    assert "madlangacommission.co.za" in profile["trusted_index_domains"]
    assert profile["hearing_clues"]["day_numbers"] == ["146"]
    assert "never research or reveal" in profile["anonymous_witness_policy"]


def test_accepts_one_official_madlanga_commission_source() -> None:
    item = {
        **correction(),
        "raw_text": "Padayashi",
        "canonical_text": "Padayachee",
        "source_urls": [
            "https://criminaljusticecommission.org.za/hearings/2026/07/14"
        ],
        "story_match_terms": ["IDAC", "Khumalo", "Madlanga Commission"],
    }
    source = transcript()
    source["segments"][0]["source_text"] = (
        "Padayashi testified about IDAC and Khumalo at the Madlanga Commission."
    )
    accepted, rejected = validate_researched_corrections(
        {"corrections": [item]},
        candidates=extract_name_candidates(source),
        actual_evidence_urls=item["source_urls"],
        authoritative_domains=["criminaljusticecommission.org.za"],
    )
    assert len(accepted) == 1
    assert rejected == []


def test_madlanga_profile_and_local_context_reach_research_agent(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-madlanga"
    (job_root / "analysis").mkdir(parents=True)
    (job_root / "input").mkdir(parents=True)
    (job_root / "input" / "youtube_source.json").write_text(
        '{"title":"Madlanga Commission Day 146 | Andrea Johnson",'
        '"channel":"SABC News","video_id":"abc"}',
        encoding="utf-8",
    )
    (job_root / "analysis" / "domain_classification.json").write_text(
        '{"primary_domain":"commission","evidence":["Madlanga Commission"]}',
        encoding="utf-8",
    )
    (job_root / "analysis" / "context.json").write_text(
        '{"providers":[{"provider_name":"commission-master-case",'
        '"confidence":0.9,"matched_entities":["Andrea Johnson"],'
        '"relevant_facts":[{"title":"IDAC evidence",'
        '"people":["Andrea Johnson","Dumisani Khumalo"],'
        '"supporting_documents":["DAY 146.pdf"]}]}]}',
        encoding="utf-8",
    )
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-madlanga",
        source_filename="Madlanga Commission Day 146.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=provider,

        entity_provider=EntityProvider(),
    )

    payload = provider.last_request.payload
    assert payload["research_profile"]["name"] == "madlanga_commission"
    assert payload["research_profile"]["hearing_clues"]["day_numbers"] == ["146"]
    assert payload["local_context"]["context_providers"][0][
        "provider_name"
    ] == "commission-master-case"
    assert payload["requirements"]["preserve_anonymous_witness_designations"] is True


def test_new_job_reads_candidate_matched_master_case_before_context_exists(
    tmp_path: Path, monkeypatch
) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-madlanga-new"
    (job_root / "analysis").mkdir(parents=True)
    (job_root / "input").mkdir(parents=True)
    (job_root / "input" / "youtube_source.json").write_text(
        '{"title":"Madlanga Commission | Khumalo charges",'
        '"channel":"SABC News","video_id":"abc"}',
        encoding="utf-8",
    )
    master_case = tmp_path / "master_case_state.json"
    master_case.write_text(
        '{"people":[{"name":"Kaizer Kganyago",'
        '"role":"NPA spokesperson",'
        '"matter":"charges against Khumalo were withdrawn"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("COMMISSION_MASTER_CASE_PATH", str(master_case))
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-madlanga-new",
        source_filename="Madlanga Commission Khumalo charges.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=provider,

        entity_provider=EntityProvider(),
    )

    master_context = provider.last_request.payload["local_context"][
        "commission_master_case"
    ]
    assert master_context["status"] == "loaded"
    assert any(
        "Kganyago" in item["name"] or "Kganyago" in item["excerpt"]
        for item in master_context["candidate_matches"]
    )


def test_exact_registry_name_is_not_sent_for_web_research(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-known"
    (job_root / "analysis").mkdir(parents=True)
    registry = tmp_path / "entity_registry.json"
    registry.write_text(
        '{"entities":[{"entity_id":"person_kganyago",'
        '"entity_type":"person","canonical_text":"Kaizer Kganyago",'
        '"display_text":"Kaizer Kganyago","aliases":["Kganyago"]}]}',
        encoding="utf-8",
    )
    source = transcript()
    source["complete_text"] = "Kganyago spoke about the matter."
    source["segments"][0]["source_text"] = "Kganyago spoke about the matter."
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-known",
        source_filename="report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        registry_path=registry,
        provider=provider,

        entity_provider=EntityProvider(),
    )

    assert provider.calls == 0
    assert result["artifact"]["status"] == "no_research_candidates"
    assert result["artifact"]["extracted_candidate_count"] >= 1


def test_latest_configured_confusion_guard_is_preserved(tmp_path: Path) -> None:
    import pytest

    from mathula_tv.autocorrect_stage import assert_no_configured_autocorrect_confusions

    config = tmp_path / "config"
    config.mkdir()
    (config / "autocorrect_hints.json").write_text(
        '{"rules":[{"pattern":"\\\\bKanyaku\\\\b",'
        '"replacement":"Kganyago","flags":["IGNORECASE"],'
        '"auto_apply":true}]}',
        encoding="utf-8",
    )
    source = transcript()

    with pytest.raises(ValueError, match="Kanyaku"):
        assert_no_configured_autocorrect_confusions(
            source,
            project_root=tmp_path,
        )


def test_default_research_backend_uses_azure_responses_web_search(
    tmp_path: Path, monkeypatch
) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-azure"
    (job_root / "analysis").mkdir(parents=True)
    job = SimpleNamespace(
        job_id="job-azure",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    class AzureBackend:
        calls = 0

        def research_json(self, *, system_prompt, payload, output_schema):
            self.calls += 1
            assert "candidate_entities" in payload
            request = SimpleNamespace(payload=payload)
            return SimpleNamespace(data=research_response(request), metadata=Metadata())

    backend = AzureBackend()
    monkeypatch.setattr(
        "mathula_tv.autocorrect_research.AzureResponsesWebResearchProvider.from_environment",
        lambda: backend,
    )

    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=None,

        entity_provider=EntityProvider(),
        force=True,
    )

    assert backend.calls == 1
    assert result["transcript"]["segments"][0]["source_text"].startswith(
        "Kganyago"
    )


def test_failed_research_artifact_is_not_reused_as_cache(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-retry"
    (job_root / "analysis").mkdir(parents=True)
    job = SimpleNamespace(
        job_id="job-retry",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    class FailingProvider:
        calls = 0

        def complete_structured(self, request):
            self.calls += 1
            raise RuntimeError("temporary web-search failure")

    failing = FailingProvider()
    first = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=failing,

        entity_provider=EntityProvider(),
    )
    assert first["artifact"]["status"] == "research_failed_advisory"

    succeeding = Provider()
    second = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=succeeding,

        entity_provider=EntityProvider(),
    )

    assert failing.calls == 1
    assert succeeding.calls == 1
    assert second["artifact"]["status"] == "completed"
    assert second["transcript"]["segments"][0]["source_text"].startswith(
        "Kganyago"
    )


def test_large_transcript_is_compacted_and_batched(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-large"
    (job_root / "analysis").mkdir(parents=True)
    source = {
        "schema_version": "transcript-v1",
        "complete_text": "Madlanga Commission " + ("background " * 10000),
        "segments": [],
        "words": [{"text": "noise"}] * 50000,
        "autocorrect": {"schema_version": "autocorrect-v1"},
    }
    for index in range(18):
        name = f"Kanyaku{chr(65 + index)}"
        source["segments"].append(
            {
                "segment_id": f"seg-{index}",
                "start": index * 2.0,
                "end": index * 2.0 + 1.5,
                "source_text": (
                    f"NPA spokesperson {name} discussed Khumalo and the charges."
                ),
                "words": [{"text": name}] * 1000,
            }
        )

    class CompactProvider:
        def __init__(self):
            self.calls = []

        def complete_structured(self, request):
            self.calls.append(request.payload)
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [],
                    "unresolved_candidates": [
                        {
                            "entity_id": item["entity_id"],
                            "reason": "not enough evidence",
                        }
                        for item in request.payload["candidate_entities"]
                    ],
                },
                metadata=Metadata(),
            )

    monkeypatch.setenv("MATHULA_TV_AUTOCORRECT_RESEARCH_BATCH_SIZE", "5")
    provider = CompactProvider()
    job = SimpleNamespace(
        job_id="job-large",
        source_filename="Madlanga Commission report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=provider,

        entity_provider=EntityProvider(),
        force=True,
    )

    assert len(provider.calls) >= 4
    assert all("transcript" not in payload for payload in provider.calls)
    assert all("transcript_context" in payload for payload in provider.calls)
    assert all(len(str(payload)) < 100_000 for payload in provider.calls)
    assert result["artifact"]["request_compaction"][
        "full_transcript_json_chars"
    ] > result["artifact"]["request_compaction"][
        "largest_batch_request_json_chars"
    ] * 5


def test_one_failed_batch_does_not_discard_successful_kganyago_correction(
    tmp_path: Path, monkeypatch
) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-partial"
    (job_root / "analysis").mkdir(parents=True)
    source = transcript()
    source["segments"].append(
        {
            "segment_id": "seg-2",
            "start": 4.0,
            "end": 8.0,
            "source_text": "Padayashi discussed another matter.",
        }
    )
    source["complete_text"] += " Padayashi discussed another matter."

    class PartialProvider:
        def __init__(self):
            self.calls = 0

        def complete_structured(self, request):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("temporary batch failure")
            return SimpleNamespace(data=research_response(request), metadata=Metadata())

    monkeypatch.setenv("MATHULA_TV_AUTOCORRECT_RESEARCH_BATCH_SIZE", "1")
    provider = PartialProvider()
    job = SimpleNamespace(
        job_id="job-partial",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=provider,

        entity_provider=EntityProvider(),
        force=True,
    )

    assert result["artifact"]["status"] == "completed_with_batch_failures"
    assert result["artifact"]["applied_count"] == 1
    assert result["transcript"]["segments"][0]["source_text"].startswith(
        "Kganyago"
    )
    assert result["artifact"]["batch_failures"]


def test_job_cache_ignores_volatile_autocorrect_timestamps(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-volatile"
    (job_root / "analysis").mkdir(parents=True)
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-volatile",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    first_transcript = transcript()
    first_transcript["autocorrect"]["generated_at"] = "2026-07-28T10:00:00Z"
    second_transcript = transcript()
    second_transcript["autocorrect"]["generated_at"] = "2026-07-28T11:00:00Z"

    run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=first_transcript,
        provider=provider,

        entity_provider=EntityProvider(),
    )
    second = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=second_transcript,
        provider=provider,

        entity_provider=EntityProvider(),
    )

    assert provider.calls == 1
    assert second["artifact"]["cache_reused"] is True


def test_accepted_correction_cache_reused_across_matching_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "shared-name-cache"
    monkeypatch.setenv(
        "MATHULA_TV_AUTOCORRECT_RESEARCH_CACHE_DIR", str(cache_dir)
    )
    first_root = tmp_path / "working" / "jobs" / "job-a"
    second_root = tmp_path / "working" / "jobs" / "job-b"
    (first_root / "analysis").mkdir(parents=True)
    (second_root / "analysis").mkdir(parents=True)

    first_job = SimpleNamespace(
        job_id="job-a",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/a.mp4",
        source_language="en-ZA",
    )
    first_provider = Provider()
    first = run_name_research_autocorrection(
        job_root=first_root,
        job=first_job,
        transcript=transcript(),
        provider=first_provider,

        entity_provider=EntityProvider(),
    )
    assert first_provider.calls == 1
    assert first["artifact"]["applied_count"] == 1

    second_source = transcript()
    second_source["segments"][0]["segment_id"] = "different-segment-id"

    class RemainingOnlyProvider:
        calls = 0

        def complete_structured(self, request):
            self.calls += 1
            assert [
                item["representative_text"]
                for item in request.payload["candidate_entities"]
            ] == ["Khumalo"]
            candidate = request.payload["candidate_entities"][0]
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [],
                    "unresolved_candidates": [
                        {
                            "entity_id": candidate["entity_id"],
                            "reason": "already correctly spelled",
                        }
                    ],
                },
                metadata=Metadata(),
            )

    second_provider = RemainingOnlyProvider()
    second_job = SimpleNamespace(
        job_id="job-b",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/b.mp4",
        source_language="en-ZA",
    )
    second = run_name_research_autocorrection(
        job_root=second_root,
        job=second_job,
        transcript=second_source,
        provider=second_provider,

        entity_provider=EntityProvider(),
    )

    assert second_provider.calls == 1
    assert second["transcript"]["segments"][0]["source_text"].startswith(
        "Kganyago"
    )
    assert second["artifact"]["persistent_cache"][
        "accepted_correction_hits"
    ] == 1
    assert second["artifact"]["provider_call_count"] == 1


def test_progress_reports_batch_start_and_completion(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("MATHULA_TV_AUTOCORRECT_SHOW_PROGRESS", "1")
    job_root = tmp_path / "working" / "jobs" / "job-progress"
    (job_root / "analysis").mkdir(parents=True)
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-progress",
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=provider,

        entity_provider=EntityProvider(),
        force=True,
    )

    captured = capsys.readouterr()
    assert "identified" in captured.err
    assert "batch 1/" in captured.err
    assert "researching" in captured.err
    assert "completed in" in captured.err


def test_discourse_words_are_not_research_candidates() -> None:
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": (
                    "Let's continue. Therefore, this matters. Good morning. "
                    "Yeah, I agree. Appreciate your time. Kaiser Kanyaku responded."
                ),
            }
        ]
    }
    raw = {item["raw_text"] for item in extract_name_candidates(source)}
    assert "Let's" not in raw
    assert "Therefore" not in raw
    assert "Good" not in raw
    assert "Yeah" not in raw
    assert "Appreciate" not in raw
    assert "Kaiser Kanyaku" in raw


def test_candidate_extraction_does_not_cross_sentence_boundaries() -> None:
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": (
                    "The Malanga Commission. For example, it heard evidence in "
                    "South Africa. The next witness was Mr. Kanyaku."
                ),
            }
        ]
    }
    raw = {item["raw_text"] for item in extract_name_candidates(source)}
    assert "Malanga Commission" in raw
    assert "Malanga Commission. For" not in raw
    assert "South Africa" not in raw  # stable, correctly spelled geography
    assert "South Africa. The" not in raw
    assert "Mr. Kanyaku" in raw


def test_candidate_extraction_strips_possessive_punctuation() -> None:
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": (
                    "Brigadier Dineo Mokwele's statement mentioned the NDPP's review."
                ),
            }
        ]
    }
    raw = {item["raw_text"] for item in extract_name_candidates(source)}
    assert "Brigadier Dineo Mokwele" in raw
    assert "Brigadier Dineo Mokwele's" not in raw
    assert "NDPP" not in raw  # stable exact institutional acronym


def test_registry_display_and_stt_forms_skip_web_research(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-registry"
    (job_root / "analysis").mkdir(parents=True)
    registry = tmp_path / "entity_registry.json"
    registry.write_text(
        '''{
          "entities": [
            {
              "canonical_text": "Judicial Commission of Inquiry into Allegations of State Capture",
              "display_text": "Zondo Commission",
              "aliases": [],
              "stt_phrases": ["State Capture Commission"]
            }
          ]
        }''',
        encoding="utf-8",
    )
    provider = Provider()
    job = SimpleNamespace(
        job_id="job-registry",
        source_filename="public affairs.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": "The Zondo Commission met Kanyaku after the hearing.",
            }
        ]
    }

    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        registry_path=registry,
        provider=provider,

        entity_provider=EntityProvider(),
    )

    sent = {
        alias
        for item in provider.last_request.payload["candidate_entities"]
        for alias in item.get("research_aliases", item.get("aliases", []))
    }
    assert "Zondo Commission" not in sent
    assert "Kanyaku" in sent
    assert result["artifact"]["known_registry_hit_count"] >= 1


def test_actual_news_snippet_filters_normal_words_before_research() -> None:
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": (
                    "Let's circle back to our breaking news story. Good morning. "
                    "Kaiser Kanyaku spoke for the NPA. Therefore, the matter continues."
                ),
            },
            {
                "segment_id": "seg-2",
                "source_text": (
                    "Malanga Commission. For example, Advocate Mhutibi referred to "
                    "Mr. Kaiser Kanyahude."
                ),
            },
        ]
    }
    raw = [item["raw_text"] for item in extract_name_candidates(source)]
    assert raw == [
        "Kaiser Kanyaku",
        "NPA",
        "Malanga Commission",
        "Advocate Mhutibi",
        "Mr. Kaiser Kanyahude",
    ]



def test_web_search_receives_only_ai_confirmed_uncertain_names(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-ai-gate"
    (job_root / "analysis").mkdir(parents=True)
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": (
                    "Firstly, Parliament and the Constitutional Court were mentioned. "
                    "An EFF member then referred to Mr. Blosser. Doesn't that matter?"
                ),
            }
        ]
    }
    entities = [
        {
            "entity_id": "parliament",
            "canonical_name": "Parliament",
            "entity_type": "organisation",
            "confidence": 0.99,
            "needs_research": False,
            "reason": "Stable and internally consistent institution name",
            "surface_forms": ["Parliament"],
            "mentions": [{"segment_id": "seg-1", "raw_text": "Parliament"}],
        },
        {
            "entity_id": "constitutional-court",
            "canonical_name": "Constitutional Court",
            "entity_type": "court_or_case",
            "confidence": 0.99,
            "needs_research": False,
            "reason": "Stable and internally consistent court name",
            "surface_forms": ["Constitutional Court"],
            "mentions": [
                {"segment_id": "seg-1", "raw_text": "Constitutional Court"}
            ],
        },
        {
            "entity_id": "eff",
            "canonical_name": "EFF",
            "entity_type": "organisation",
            "confidence": 0.99,
            "needs_research": False,
            "reason": "Stable acronym",
            "surface_forms": ["EFF"],
            "mentions": [{"segment_id": "seg-1", "raw_text": "EFF"}],
        },
        {
            "entity_id": "person-blosser",
            "canonical_name": "",
            "entity_type": "person",
            "confidence": 0.38,
            "needs_research": True,
            "reason": "Surname may be an STT spelling error",
            "surface_forms": ["Mr. Blosser"],
            "mentions": [{"segment_id": "seg-1", "raw_text": "Mr. Blosser"}],
        },
    ]

    class WebProvider:
        calls = 0
        last_request = None

        def complete_structured(self, request):
            self.calls += 1
            self.last_request = request
            assert [
                item["representative_text"]
                for item in request.payload["candidate_entities"]
            ] == ["Mr. Blosser"]
            candidate = request.payload["candidate_entities"][0]
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [],
                    "unresolved_candidates": [
                        {
                            "entity_id": candidate["entity_id"],
                            "reason": "not enough evidence",
                        }
                    ],
                },
                metadata=Metadata(),
            )

    web = WebProvider()
    job = SimpleNamespace(
        job_id="job-ai-gate",
        source_filename="simple sabc clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=web,
        entity_provider=EntityProvider(entities),
        force=True,
    )

    assert web.calls == 1
    assert result["artifact"]["candidate_source"] == "ai_entity_inventory_clusters"
    assert result["artifact"]["entity_count"] == 4
    assert result["artifact"]["research_entity_count"] == 1
    assert result["artifact"]["candidate_count"] == 1
    assert result["artifact"]["candidate_name_spans"][0]["raw_text"] == "Mr. Blosser"


def test_inventory_groups_aliases_without_researching_certain_entity(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-aliases"
    (job_root / "analysis").mkdir(parents=True)
    source = {
        "segments": [
            {
                "segment_id": "seg-1",
                "source_text": "The Constitutional Court issued its ruling.",
            },
            {
                "segment_id": "seg-2",
                "source_text": "The Con Court judgment was discussed later.",
            },
        ]
    }
    entities = [
        {
            "entity_id": "constitutional-court",
            "canonical_name": "Constitutional Court",
            "entity_type": "court_or_case",
            "confidence": 0.99,
            "needs_research": False,
            "reason": "Both aliases are clear in context",
            "surface_forms": ["Constitutional Court", "Con Court"],
            "mentions": [
                {"segment_id": "seg-1", "raw_text": "Constitutional Court"},
                {"segment_id": "seg-2", "raw_text": "Con Court"},
            ],
        }
    ]
    web = Provider()
    job = SimpleNamespace(
        job_id="job-aliases",
        source_filename="court clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=web,
        entity_provider=EntityProvider(entities),
        force=True,
    )

    assert web.calls == 0
    assert result["artifact"]["status"] == "no_research_candidates"
    assert result["artifact"]["entity_count"] == 1
    assert result["artifact"]["inventory_mention_count"] == 2


def test_inventory_failure_skips_web_and_leaves_transcript_unchanged(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-inventory-failure"
    (job_root / "analysis").mkdir(parents=True)
    web = Provider()
    source = transcript()
    job = SimpleNamespace(
        job_id="job-inventory-failure",
        source_filename="clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )

    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=web,
        entity_provider=EntityProvider(error=RuntimeError("inventory unavailable")),
        force=True,
    )

    assert web.calls == 0
    assert result["artifact"]["status"] == "entity_inventory_failed_advisory"
    assert result["transcript"] == source


def test_incomplete_web_json_is_rejected_locally_without_schema_repair(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-incomplete-web"
    (job_root / "analysis").mkdir(parents=True)

    class IncompleteProvider:
        calls = 0

        def complete_structured(self, request):
            self.calls += 1
            candidate = request.payload["candidate_entities"][0]
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [
                        {
                            "entity_id": candidate["entity_id"],
                            "canonical_text": "Kganyago",
                        }
                    ],
                    "unresolved_candidates": [],
                },
                metadata=Metadata(),
            )

    web = IncompleteProvider()
    job = SimpleNamespace(
        job_id="job-incomplete-web",
        source_filename="clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=transcript(),
        provider=web,
        entity_provider=EntityProvider(),
        force=True,
    )

    assert web.calls >= 1
    assert result["artifact"]["status"] == "completed"
    assert result["artifact"]["applied_count"] == 0
    assert any(
        item["validation_reason"] == "confidence_below_0.95"
        for item in result["artifact"]["rejected_corrections"]
    )


def test_uncertain_alias_cluster_is_researched_once_and_propagated(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-nkosi-cluster"
    (job_root / "analysis").mkdir(parents=True)
    aliases = [
        "Sergeant Fine Gossi",
        "Mr. Fannie Nkosi",
        "Sergeant F e Ngosi",
        "Ngosi",
        "Nkosi",
        "Sergeant Amkosi",
    ]
    source = {
        "schema_version": "transcript-v1",
        "segments": [
            {
                "segment_id": f"seg-{index}",
                "source_text": f"{alias} gave evidence about the same matter.",
            }
            for index, alias in enumerate(aliases, start=1)
        ],
    }
    entity = {
        "entity_id": "person-fannie-nkosi",
        "canonical_name": "",
        "entity_type": "person",
        "confidence": 0.31,
        "needs_research": True,
        "reason": "Conflicting phonetic spellings across repeated references",
        "surface_forms": aliases,
        "mentions": [
            {"segment_id": f"seg-{index}", "raw_text": alias}
            for index, alias in enumerate(aliases, start=1)
        ],
    }

    class ClusterProvider:
        calls = 0
        last_request = None

        def complete_structured(self, request):
            self.calls += 1
            self.last_request = request
            candidates = request.payload["candidate_entities"]
            assert len(candidates) == 1
            assert candidates[0]["entity_id"] == "person-fannie-nkosi"
            assert len(candidates[0]["mentions"]) == 6
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [
                        {
                            "entity_id": "person-fannie-nkosi",
                            "canonical_text": "Fannie Nkosi",
                            "entity_type": "person",
                            "confidence": 0.99,
                            "source_urls": correction()["source_urls"],
                            "story_match_terms": ["Nkosi", "sergeant", "evidence"],
                            "reason": "The sources identify the same sergeant as Fannie Nkosi.",
                        }
                    ],
                    "unresolved_candidates": [],
                },
                metadata=Metadata(),
            )

    web = ClusterProvider()
    job = SimpleNamespace(
        job_id="job-nkosi-cluster",
        source_filename="public affairs clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=web,
        entity_provider=EntityProvider([entity]),
        force=True,
    )

    corrected = [item["source_text"] for item in result["transcript"]["segments"]]
    assert web.calls == 1
    assert result["artifact"]["candidate_count"] == 1
    assert result["artifact"]["candidate_mention_count"] == 6
    assert result["artifact"]["accepted_count"] == 1
    assert result["artifact"]["expanded_correction_count"] == 4
    assert result["artifact"]["applied_count"] == 4
    assert corrected[0].startswith("Sergeant Fannie Nkosi")
    assert corrected[1].startswith("Mr. Fannie Nkosi")
    assert corrected[2].startswith("Sergeant Fannie Nkosi")
    assert corrected[3].startswith("Nkosi")
    assert corrected[4].startswith("Nkosi")
    assert corrected[5].startswith("Sergeant Nkosi")


def test_v13_10_1_flat_inventory_cache_is_reclustered(tmp_path: Path) -> None:
    import json

    job_root = tmp_path / "working" / "jobs" / "job-legacy-inventory"
    (job_root / "analysis").mkdir(parents=True)
    source = {
        "schema_version": "transcript-v1",
        "segments": [
            {"segment_id": "seg-1", "source_text": "General Kumalo testified."},
            {"segment_id": "seg-2", "source_text": "Lieutenant-General Comalo replied."},
        ],
    }
    entity = {
        "entity_id": "person-kumalo",
        "canonical_name": "",
        "entity_type": "person",
        "confidence": 0.35,
        "needs_research": True,
        "reason": "Conflicting spellings",
        "surface_forms": ["General Kumalo", "Lieutenant-General Comalo"],
        "mentions": [
            {"segment_id": "seg-1", "raw_text": "General Kumalo"},
            {"segment_id": "seg-2", "raw_text": "Lieutenant-General Comalo"},
        ],
    }

    class UnresolvedProvider:
        calls = 0

        def complete_structured(self, request):
            self.calls += 1
            candidates = request.payload["candidate_entities"]
            return SimpleNamespace(
                data={
                    "schema_version": "mathula-autocorrect-name-research-v2",
                    "corrections": [],
                    "unresolved_candidates": [
                        {
                            "entity_id": item["entity_id"],
                            "reason": "not enough evidence",
                        }
                        for item in candidates
                    ],
                },
                metadata=Metadata(),
            )

    job = SimpleNamespace(
        job_id="job-legacy-inventory",
        source_filename="clip.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )
    first_web = UnresolvedProvider()
    run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=first_web,
        entity_provider=EntityProvider([entity]),
    )

    inventory_path = job_root / "analysis" / "autocorrection_name_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["schema_version"] = "mathula-autocorrect-entity-inventory-artifact-v1"
    inventory["candidate_entities"] = [
        {
            "segment_id": mention["segment_id"],
            "raw_text": mention["raw_text"],
            "inventory_entity_id": "person-kumalo",
            "inventory_entity_type": "person",
        }
        for mention in entity["mentions"]
    ]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    (job_root / "analysis" / "autocorrection_name_research.json").unlink()

    class ReclusteringProvider(UnresolvedProvider):
        def complete_structured(self, request):
            candidates = request.payload["candidate_entities"]
            assert len(candidates) == 1
            assert candidates[0]["entity_id"] == "person-kumalo"
            assert len(candidates[0]["mentions"]) == 2
            return super().complete_structured(request)

    second_web = ReclusteringProvider()
    result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=source,
        provider=second_web,
        entity_provider=EntityProvider(error=AssertionError("inventory reran")),
    )

    assert second_web.calls == 1
    assert result["artifact"]["candidate_count"] == 1
    assert result["artifact"]["candidate_mention_count"] == 2
    assert result["artifact"]["entity_inventory_cache_reused"] is True
