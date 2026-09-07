"""Phase 17: web-researched pronunciation for the native pipeline. Reuses the
EXISTING AzureResponsesWebResearchProvider (azure_web_research.py, real Azure
OpenAI Responses API web_search tool, already production-used by the legacy
autocorrect_research.py pipeline) as a narrow, additive side-channel for
entity/pronunciation research only -- never translation, never
FoundryGrokProvider.

Fake `research_json` follows the exact pattern already proven in
test_autocorrect_research.py::test_default_research_backend_uses_azure_responses_web_search
(a plain object exposing `research_json(*, system_prompt, payload, output_schema)`,
monkeypatched onto `AzureResponsesWebResearchProvider.from_environment`) -- just
patched onto `mathula_tv.native_dub`'s own imported name instead.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.native_dub import (
    _apply_native_pronunciation_research_to_job_cache,
    _ensure_native_pronunciation_research,
    _native_dub_tts_ready_text,
    _native_pronunciation_candidate_id,
    _native_pronunciation_research_candidates,
    native_dub_paths,
)


def _job_root(tmp_path: Path) -> Path:
    # Two levels deep (working/jobs/<id>), matching production exactly --
    # the project-level pronunciation cache derives its path as
    # job_root.parent.parent (see _native_pronunciation_cache_path), so a
    # shallower fixture here would leak that cache into pytest's SHARED
    # tmp root across tests instead of this test's own isolated tmp_path.
    job_root = tmp_path / "working" / "jobs" / "job"
    native_dub_paths(job_root).root.mkdir(parents=True, exist_ok=True)
    return job_root


class _Metadata:
    def __init__(self, *, evidence_url: str = "https://example.test/evidence", usage=None, attempts: int = 1):
        self.server_tool_evidence = ({"url": evidence_url, "title": "Evidence"},)
        self.usage = usage or {"input_tokens": 40, "output_tokens": 20}
        self.request_attempts = attempts


class _FakeResearchBackend:
    """Echoes canonical_text unchanged with a grounded pronunciation for every
    candidate, citing the SAME evidence url the fake Metadata reports -- the
    real, minimal shape validate_researched_corrections requires to accept.
    """

    def __init__(self, *, evidence_url: str = "https://example.test/evidence", unresolved: set[str] = frozenset()):
        self.calls = 0
        self.evidence_url = evidence_url
        self.unresolved = unresolved

    def research_json(self, *, system_prompt, payload, output_schema):
        self.calls += 1
        corrections = []
        unresolved_candidates = []
        for candidate in payload["candidate_entities"]:
            entity_id = candidate["entity_id"]
            if entity_id in self.unresolved:
                unresolved_candidates.append({"entity_id": entity_id, "reason": "no direct evidence"})
                continue
            corrections.append({
                "entity_id": entity_id,
                "canonical_text": candidate["representative_text"],
                "entity_type": candidate["entity_type"],
                "pronunciation_mode": "word_name",
                "pronunciation_evidence_urls": [self.evidence_url],
                "pronunciation_tts_text": f"{candidate['representative_text']}-tts",
                "pronunciation_confidence": 0.95,
                "pronunciation_reason": "test evidence",
            })
        return SimpleNamespace(
            data={"schema_version": "mathula-autocorrect-name-research-v2",
                  "corrections": corrections, "unresolved_candidates": unresolved_candidates},
            metadata=_Metadata(evidence_url=self.evidence_url),
        )


class _IdentityChangingBackend:
    """Returns a DIFFERENT canonical_text than the candidate's own name --
    proves the needs_identity_research=False gate rejects it outright.
    """

    def research_json(self, *, system_prompt, payload, output_schema):
        candidate = payload["candidate_entities"][0]
        return SimpleNamespace(
            data={"schema_version": "mathula-autocorrect-name-research-v2", "corrections": [{
                "entity_id": candidate["entity_id"], "canonical_text": "A Totally Different Name",
                "entity_type": candidate["entity_type"], "pronunciation_mode": "word_name",
                "pronunciation_evidence_urls": ["https://example.test/evidence"],
                "pronunciation_tts_text": "respelling", "pronunciation_confidence": 0.95,
            }], "unresolved_candidates": []},
            metadata=_Metadata(),
        )


class _UngroundedBackend:
    """Cites a url NOT present in server_tool_evidence -- proves the grounding
    check (not just presence of a url) is what gates acceptance.
    """

    def research_json(self, *, system_prompt, payload, output_schema):
        candidate = payload["candidate_entities"][0]
        return SimpleNamespace(
            data={"schema_version": "mathula-autocorrect-name-research-v2", "corrections": [{
                "entity_id": candidate["entity_id"], "canonical_text": candidate["representative_text"],
                "entity_type": candidate["entity_type"], "pronunciation_mode": "word_name",
                "pronunciation_evidence_urls": ["https://not-actually-cited.test/page"],
                "pronunciation_tts_text": "respelling", "pronunciation_confidence": 0.95,
            }], "unresolved_candidates": []},
            metadata=_Metadata(),
        )


class _ExplodingBackend:
    def research_json(self, *, system_prompt, payload, output_schema):
        raise RuntimeError("network exploded")


def _context_ledger(*names: str) -> dict:
    return {"entities": [{"name": name, "role": "witness"} for name in names]}


def _glossary(*, do_not_translate=(), terms=()) -> dict:
    return {"terms": list(terms), "do_not_translate": list(do_not_translate), "register": {}}


# --- _native_pronunciation_research_candidates -------------------------------------


def test_candidates_dedup_across_context_ledger_and_glossary():
    context_ledger = _context_ledger("Ayanda Nyati", "Ayanda Nyati")  # exact duplicate
    glossary = _glossary(
        do_not_translate=["Justice College"],
        terms=[
            {"english": "Justice College", "zulu": "Justice College", "kind": "organisation"},  # same name, dedup
            {"english": "the Hawks", "zulu": "amaphoyisa", "kind": "organisation"},  # translated -- excluded
            {"english": "load shedding", "zulu": "ukucinywa kukagesi", "kind": "other"},  # wrong kind -- excluded
        ],
    )
    candidates = _native_pronunciation_research_candidates(context_ledger=context_ledger, glossary=glossary)
    names = sorted(c["representative_text"] for c in candidates)
    assert names == ["Ayanda Nyati", "Justice College"]
    for candidate in candidates:
        assert candidate["needs_identity_research"] is False
        assert candidate["research_goals"] == ["target_locale_pronunciation"]


def test_candidates_cap_at_forty():
    context_ledger = _context_ledger(*[f"Person {i}" for i in range(50)])
    candidates = _native_pronunciation_research_candidates(context_ledger=context_ledger, glossary=_glossary())
    assert len(candidates) == 40


def test_candidate_id_is_stable_and_case_insensitive():
    first = _native_pronunciation_candidate_id("Justice College")
    second = _native_pronunciation_candidate_id("  justice college  ")
    assert first == second
    assert first != _native_pronunciation_candidate_id("Different Name")


# --- _ensure_native_pronunciation_research -----------------------------------------


def test_ensure_research_writes_artifact_and_accepts_a_grounded_entry(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    # "Not-Yet-Hardcoded Institute" deliberately avoids anything already in
    # pronunciation.py's own _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS (e.g.
    # "Justice College") -- a hardcoded name is now skipped before research
    # entirely (see the dedicated skip tests below), so using one here would
    # make this general smoke test assert on the wrong mechanism.
    context_ledger = _context_ledger("Ayanda Nyati")
    glossary = _glossary(do_not_translate=["Not-Yet-Hardcoded Institute"])

    artifact, usage = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=glossary, progress=None,
    )

    assert backend.calls == 1
    assert artifact["status"] == "completed"
    assert len(artifact["accepted_corrections"]) == 2
    assert usage["input_tokens"] == 40
    assert native_dub_paths(job_root).pronunciation_research.is_file()


def test_ensure_research_skips_the_network_call_with_no_candidates(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)

    class _ShouldNeverBeCalled:
        def research_json(self, **kwargs):
            raise AssertionError("must not be called for an empty candidate list")

    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment",
        lambda: _ShouldNeverBeCalled(),
    )
    artifact, usage = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger(), glossary=_glossary(), progress=None,
    )
    assert artifact["status"] == "empty"
    assert artifact["accepted_corrections"] == []
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


def test_ensure_research_reuses_cache_without_a_second_call(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Ayanda Nyati")
    glossary = _glossary()

    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=glossary, progress=None,
    )
    assert backend.calls == 1

    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=glossary, progress=None,
    )
    assert backend.calls == 1  # not called again


def test_ensure_research_regenerates_when_glossary_changes(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Ayanda Nyati")
    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )
    assert backend.calls == 1

    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger,
        glossary=_glossary(do_not_translate=["A New Name"]), progress=None,
    )
    assert backend.calls == 2


def test_ensure_research_force_regenerates_even_with_a_fresh_cache(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Ayanda Nyati")
    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )
    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None, force=True,
    )
    assert backend.calls == 2


def test_ensure_research_skips_a_name_already_hardcoded_in_pronunciation_py(tmp_path, monkeypatch):
    # Real production incident (2026-09-02): fresh research spent real
    # tokens re-researching "Lincoln", which pronunciation.py's own
    # _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS already hand-fixes -- and the
    # fresh research produced a WORSE candidate that round-trip
    # verification correctly rejected anyway, after paying for it.
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Lincoln", "Genuinely New Name")

    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )

    assert backend.calls == 1
    assert artifact["candidate_count"] == 2  # both counted, only one actually researched
    accepted_names = {c["representative_text"] for c in artifact["accepted_corrections"]}
    assert accepted_names == {"Genuinely New Name"}  # "Lincoln" never went through research at all


def test_ensure_research_reuses_a_prior_jobs_cross_job_cache(tmp_path, monkeypatch):
    # Real production incident (2026-09-02): a smaller re-cut of an
    # already-processed video re-researched the exact same entities from
    # scratch. Two DIFFERENT job_roots sharing the same project root
    # (tmp_path/"working") must share this cache.
    project_root = tmp_path / "working"
    job_root_a = project_root / "jobs" / "job-a"
    job_root_b = project_root / "jobs" / "job-b"
    native_dub_paths(job_root_a).root.mkdir(parents=True, exist_ok=True)
    native_dub_paths(job_root_b).root.mkdir(parents=True, exist_ok=True)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Ayanda Nyati")

    _ensure_native_pronunciation_research(
        job_root=job_root_a, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )
    assert backend.calls == 1

    artifact_b, _ = _ensure_native_pronunciation_research(
        job_root=job_root_b, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )
    assert backend.calls == 1  # job B never called the backend at all
    accepted_names = {c["representative_text"] for c in artifact_b["accepted_corrections"]}
    assert accepted_names == {"Ayanda Nyati"}


def test_ensure_research_force_bypasses_the_cross_job_cache_too(tmp_path, monkeypatch):
    project_root = tmp_path / "working"
    job_root_a = project_root / "jobs" / "job-a"
    job_root_b = project_root / "jobs" / "job-b"
    native_dub_paths(job_root_a).root.mkdir(parents=True, exist_ok=True)
    native_dub_paths(job_root_b).root.mkdir(parents=True, exist_ok=True)
    backend = _FakeResearchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    context_ledger = _context_ledger("Ayanda Nyati")

    _ensure_native_pronunciation_research(
        job_root=job_root_a, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )
    _ensure_native_pronunciation_research(
        job_root=job_root_b, context_ledger=context_ledger, glossary=_glossary(), progress=None, force=True,
    )
    assert backend.calls == 2  # force=True on job B re-researched instead of reusing job A's cache hit


def test_ensure_research_rejects_a_changed_canonical_text(tmp_path, monkeypatch):
    """Load-bearing: needs_identity_research=False routes into the branch that
    rejects outright when canonical_text differs from the candidate's own
    name -- proves this phase can never silently change a name.
    """
    job_root = _job_root(tmp_path)
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment",
        lambda: _IdentityChangingBackend(),
    )
    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Ayanda Nyati"), glossary=_glossary(), progress=None,
    )
    assert artifact["accepted_corrections"] == []
    assert len(artifact["rejected_corrections"]) == 1


def test_ensure_research_rejects_an_ungrounded_evidence_url(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment",
        lambda: _UngroundedBackend(),
    )
    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Ayanda Nyati"), glossary=_glossary(), progress=None,
    )
    assert artifact["accepted_corrections"] == []
    assert len(artifact["rejected_corrections"]) == 1


def test_ensure_research_degrades_non_fatally_on_provider_error(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment",
        lambda: _ExplodingBackend(),
    )
    artifact, usage = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Ayanda Nyati"), glossary=_glossary(), progress=None,
    )
    assert artifact["status"] == "degraded"
    assert artifact["accepted_corrections"] == []
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    assert native_dub_paths(job_root).pronunciation_research.is_file()


# --- batching (real production issue: a single request covering all candidates ----
# --- repeatedly hit Azure's per-minute token quota during heavy session usage) -----


class _CountingBatchBackend:
    """Records the size of every research_json call -- proves candidates are
    actually split into multiple, smaller requests rather than one big one.
    """

    def __init__(self):
        self.batch_sizes: list[int] = []

    def research_json(self, *, system_prompt, payload, output_schema):
        entities = payload["candidate_entities"]
        self.batch_sizes.append(len(entities))
        corrections = [{
            "entity_id": candidate["entity_id"],
            "canonical_text": candidate["representative_text"],
            "entity_type": candidate["entity_type"],
            "pronunciation_mode": "word_name",
            "pronunciation_evidence_urls": ["https://example.test/evidence"],
            "pronunciation_tts_text": f"{candidate['representative_text']}-tts",
            "pronunciation_confidence": 0.95,
        } for candidate in entities]
        return SimpleNamespace(
            data={"schema_version": "mathula-autocorrect-name-research-v2",
                  "corrections": corrections, "unresolved_candidates": []},
            metadata=_Metadata(usage={"input_tokens": 10, "output_tokens": 5}),
        )


def _many_names(count: int) -> list[str]:
    return [f"Person Number {i}" for i in range(count)]


def test_research_splits_candidates_into_smaller_batches(tmp_path, monkeypatch):
    import math

    from mathula_tv.native_dub import _PRONUNCIATION_RESEARCH_BATCH_SIZE

    backend = _CountingBatchBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    job_root = _job_root(tmp_path)
    total_candidates = 15
    context_ledger = _context_ledger(*_many_names(total_candidates))

    artifact, usage = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )

    expected_batch_count = math.ceil(total_candidates / _PRONUNCIATION_RESEARCH_BATCH_SIZE)
    assert len(backend.batch_sizes) == expected_batch_count
    assert sum(backend.batch_sizes) == total_candidates
    assert all(size <= _PRONUNCIATION_RESEARCH_BATCH_SIZE for size in backend.batch_sizes)
    assert len(artifact["accepted_corrections"]) == total_candidates
    assert artifact["status"] == "completed"
    # Usage sums across every batch, not just the last one.
    assert usage["input_tokens"] == 10 * expected_batch_count
    assert usage["output_tokens"] == 5 * expected_batch_count
    assert usage["attempts"] == expected_batch_count


class _SecondBatchFailsBackend:
    """The first batch succeeds; the second raises -- proves one batch's
    failure never discards an earlier batch's already-accepted corrections.
    """

    def __init__(self):
        self.calls = 0

    def research_json(self, *, system_prompt, payload, output_schema):
        self.calls += 1
        entities = payload["candidate_entities"]
        if self.calls == 2:
            raise RuntimeError("HTTP 429: rate_limit_exceeded")
        corrections = [{
            "entity_id": candidate["entity_id"],
            "canonical_text": candidate["representative_text"],
            "entity_type": candidate["entity_type"],
            "pronunciation_mode": "word_name",
            "pronunciation_evidence_urls": ["https://example.test/evidence"],
            "pronunciation_tts_text": f"{candidate['representative_text']}-tts",
            "pronunciation_confidence": 0.95,
        } for candidate in entities]
        return SimpleNamespace(
            data={"schema_version": "mathula-autocorrect-name-research-v2",
                  "corrections": corrections, "unresolved_candidates": []},
            metadata=_Metadata(),
        )


def test_a_failed_batch_does_not_discard_an_earlier_successful_batch(tmp_path, monkeypatch):
    from mathula_tv.native_dub import _PRONUNCIATION_RESEARCH_BATCH_SIZE

    backend = _SecondBatchFailsBackend()
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    job_root = _job_root(tmp_path)
    # Exactly two batches, regardless of the configured batch size.
    context_ledger = _context_ledger(*_many_names(_PRONUNCIATION_RESEARCH_BATCH_SIZE * 2))

    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=context_ledger, glossary=_glossary(), progress=None,
    )

    assert backend.calls == 2
    assert artifact["status"] == "degraded"
    assert len(artifact["accepted_corrections"]) == _PRONUNCIATION_RESEARCH_BATCH_SIZE  # first batch survives


# --- job-scoped dictionary application ----------------------------------------------


def test_apply_research_to_job_cache_produces_a_working_substitution(tmp_path):
    job_root = _job_root(tmp_path)
    research = {
        "accepted_corrections": [{
            "entity_id": "x", "canonical_text": "Justice College", "entity_type": "organisation",
            "pronunciation_tts_text": "Jastis Koleji", "pronunciation_confidence": 0.95,
            "pronunciation_mode": "word_name",
            "grounded_pronunciation_evidence_urls": ["https://example.test/evidence"],
        }],
    }
    _apply_native_pronunciation_research_to_job_cache(job_root, research)
    result = _native_dub_tts_ready_text(
        "waseBritish Rabanda Justice College usejoyina manje.", job_root=job_root,
    )
    assert "Jastis Koleji" in result
    assert "Justice College" not in result


def test_tts_ready_text_falls_back_to_default_when_job_root_unregistered(tmp_path):
    unregistered_job_root = tmp_path / "never-registered"
    text = "waseBritish Rabanda Justice College usejoyina manje."
    assert _native_dub_tts_ready_text(text, job_root=unregistered_job_root) == _native_dub_tts_ready_text(text)


def test_tts_ready_text_omitting_job_root_matches_todays_default_exactly(tmp_path):
    # A name NOT already in the hand-curated dictionary -- isolates "did
    # job-scoped research leak into the job-agnostic default" from any
    # unrelated, already-shipped hand-curated entry.
    job_root = _job_root(tmp_path)
    _apply_native_pronunciation_research_to_job_cache(job_root, {"accepted_corrections": [{
        "entity_id": "x", "canonical_text": "Nabanda Academy", "entity_type": "organisation",
        "pronunciation_tts_text": "Nabanda Akhademi", "pronunciation_confidence": 0.95,
        "pronunciation_mode": "word_name",
        "grounded_pronunciation_evidence_urls": ["https://example.test/evidence"],
    }]})
    text = "wase Nabanda Academy usejoyina manje."
    # Omitting job_root entirely must never pick up ANY job's research.
    assert "Nabanda Akhademi" not in _native_dub_tts_ready_text(text)


def test_render_reloads_research_from_disk(tmp_path):
    """Simulates render_native_dub running as a separate process from
    build_candidate_pool: the in-memory job cache is gone, only the artifact
    on disk survives -- confirms the exact read-and-reapply path works.
    """
    job_root = _job_root(tmp_path)
    paths = native_dub_paths(job_root)
    atomic_write_json(paths.pronunciation_research, {
        "accepted_corrections": [{
            "entity_id": "x", "canonical_text": "Justice College", "entity_type": "organisation",
            "pronunciation_tts_text": "Jastis Koleji", "pronunciation_confidence": 0.95,
            "pronunciation_mode": "word_name",
            "grounded_pronunciation_evidence_urls": ["https://example.test/evidence"],
        }],
    })
    # Mirrors render_native_dub's own disk-read step exactly.
    _apply_native_pronunciation_research_to_job_cache(job_root, read_json(paths.pronunciation_research))
    result = _native_dub_tts_ready_text("Justice College usejoyina.", job_root=job_root)
    assert "Jastis Koleji" in result


# --- Phase 24: self-supervised pronunciation fallback ------------------------------
# Real user-reported production defect (job fb3d08b63fed4d90922b08f7e325b906):
# "Godfrey Gidi", a local EFF mayoral candidate, came back with no accepted
# correction because web research found no indexed audio of him speaking his
# own name -- a real limit of that mechanism, confirmed by direct inspection
# of context_ledger.json (the entity WAS correctly sent to research). This
# generalizes the fix: for any entity with no accepted correction, generate a
# couple of respelling guesses via a cheap non-web-search Grok call and
# empirically verify each one with the SAME real TTS+STT round-trip judge
# Phase 18 already built (_measure_pronunciation_round_trip), adopting only a
# candidate that measurably beats the raw spelling.


class _FakeRoundTripTts:
    default_voice = "zu-ZA-ThembaNeural"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.configured_voices = ("zu-ZA-ThembaNeural",)

    def synthesize(self, request, output_path, *, force=False):
        self.calls.append((request.text, request.voice))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-wav")
        return SimpleNamespace(duration_ms=1000)


class _FakeRoundTripStt:
    def __init__(self, transcript_by_text: dict[str, str]):
        self._transcript_by_text = transcript_by_text
        self._last_text: str | None = None

    def transcribe(self, path, *, locale):
        return {"__transcript__": self._transcript_by_text.get(self._last_text, "")}

    def synthesize_hook(self, text: str) -> None:
        self._last_text = text


def _wire_tts_stt_pair(tts: _FakeRoundTripTts, stt: _FakeRoundTripStt):
    real_synthesize = tts.synthesize

    def synthesize(request, output_path, *, force=False):
        result = real_synthesize(request, output_path, force=force)
        stt.synthesize_hook(request.text)
        return result

    tts.synthesize = synthesize  # type: ignore[method-assign]


class _FakeGrokProvider:
    """Fake FoundryGrokProvider.complete_json for self-supervised respelling
    requests -- returns a fixed respellings-by-name map, tracks calls.
    """

    def __init__(self, respellings_by_name: dict[str, list[str]]):
        self.respellings_by_name = respellings_by_name
        self.calls = 0
        self.last_payload = None

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls += 1
        self.last_payload = payload
        candidates = []
        for item in payload["candidates"]:
            respellings = self.respellings_by_name.get(item["name"], [])
            candidates.append({"entity_id": item["entity_id"], "respellings": respellings})
        return SimpleNamespace(
            data={"schema_version": "mathula-pronunciation-self-supervised-v1", "candidates": candidates},
            input_tokens=15, output_tokens=8, attempts=1,
        )


@pytest.fixture(autouse=False)
def _fake_normalize_for_self_supervised(monkeypatch):
    import mathula_tv.native_dub as nd

    def fake_normalize(raw, **kwargs):
        text = raw.get("__transcript__", "")
        return {"words": [{"text": w} for w in text.split()]}

    monkeypatch.setattr(nd, "normalize_stt_response", fake_normalize)


def test_self_supervised_fallback_adopts_a_verified_respelling_for_an_unresolved_entity(
    tmp_path, monkeypatch, _fake_normalize_for_self_supervised,
):
    # NOTE: uses a fictional entity name ("Thabo Ndlovu"), not the real
    # "Godfrey Gidi" incident this mechanism was built to fix -- "Godfrey Gidi"
    # is now itself a hardcoded, reviewed dictionary entry, so using it here
    # would make `_ensure_native_pronunciation_research`'s own "already
    # hardcoded, skip research" check correctly bypass this test's whole
    # exercised code path before it ever ran.
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    grok = _FakeGrokProvider({"Thabo Ndlovu": ["Thabo Ndloovu", "Thabo Ndlavu"]})
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoThabo Ndlovu kulesi sigaba.": "kukhulunywa ngo tabo ndlobu kulesi sigaba",
        "Kukhulunywa ngoThabo Ndloovu kulesi sigaba.": "kukhulunywa ngo thabo ndlovu kulesi sigaba",
        "Kukhulunywa ngoThabo Ndlavu kulesi sigaba.": "kukhulunywa ngo thabo ndlela kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    artifact, usage = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, tts=tts, stt_backend=stt, grok_provider=grok,
    )

    assert grok.calls == 1
    assert len(artifact["self_supervised_corrections"]) == 1
    record = artifact["self_supervised_corrections"][0]
    assert record["canonical_text"] == "Thabo Ndlovu"
    assert record["pronunciation_tts_text"] == "Thabo Ndloovu"
    assert record["correction_mode"] == "self_supervised_no_evidence"
    assert record["round_trip_verified"] is True
    assert usage["input_tokens"] >= 15

    # Flows into a working substitution via the existing consumption path.
    _apply_native_pronunciation_research_to_job_cache(job_root, artifact)
    result = _native_dub_tts_ready_text("Thabo Ndlovu wathi.", job_root=job_root)
    assert "Thabo Ndloovu" in result


def test_self_supervised_fallback_skips_when_no_respelling_beats_the_raw_spelling(
    tmp_path, monkeypatch, _fake_normalize_for_self_supervised,
):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    grok = _FakeGrokProvider({"Thabo Ndlovu": ["Thabo Ndlavu"]})
    tts = _FakeRoundTripTts()
    # Both raw and candidate recover equally poorly -- no real improvement.
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoThabo Ndlovu kulesi sigaba.": "kukhulunywa ngo x y kulesi sigaba",
        "Kukhulunywa ngoThabo Ndlavu kulesi sigaba.": "kukhulunywa ngo a b kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, tts=tts, stt_backend=stt, grok_provider=grok,
    )

    assert artifact["self_supervised_corrections"] == []


def test_self_supervised_fallback_never_runs_without_a_grok_provider(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({})

    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, tts=tts, stt_backend=stt,  # no grok_provider
    )

    assert artifact["self_supervised_corrections"] == []
    assert tts.calls == []  # never even attempted a round trip


def test_self_supervised_fallback_never_runs_without_tts_or_stt(tmp_path, monkeypatch):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )

    class _ShouldNeverBeCalled:
        def complete_json(self, **kwargs):
            raise AssertionError("must not be called without tts/stt_backend to verify against")

    artifact, _ = _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, grok_provider=_ShouldNeverBeCalled(),  # no tts/stt_backend
    )

    assert artifact["self_supervised_corrections"] == []


def test_self_supervised_request_drops_an_identity_mismatched_entity_id():
    from mathula_tv.native_dub import _request_self_supervised_pronunciation_candidates

    class _MismatchedGrokProvider:
        def complete_json(self, **kwargs):
            return SimpleNamespace(
                data={"schema_version": "mathula-pronunciation-self-supervised-v1", "candidates": [
                    {"entity_id": "not-requested", "respellings": ["Whatever"]},
                ]},
                input_tokens=5, output_tokens=3, attempts=1,
            )

    by_entity, usage = _request_self_supervised_pronunciation_candidates(
        provider=_MismatchedGrokProvider(),
        entities=[{"entity_id": "real-id", "representative_text": "Real Name", "entity_type": "person"}],
    )
    assert by_entity == {}
    assert usage["input_tokens"] == 5


def test_self_supervised_request_with_no_entities_never_calls_the_provider():
    from mathula_tv.native_dub import _request_self_supervised_pronunciation_candidates

    class _ShouldNeverBeCalled:
        def complete_json(self, **kwargs):
            raise AssertionError("must not be called with an empty entity list")

    by_entity, usage = _request_self_supervised_pronunciation_candidates(
        provider=_ShouldNeverBeCalled(), entities=[],
    )
    assert by_entity == {}
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}


# --- raw ASR hint: real audio-grounded evidence beats a spelling-only guess --------
# Real finding, same session: the raw (pre-correction) Pass-1 ASR transcript for
# job fb3d08b63fed4d90922b08f7e325b906 literally read "Godrej Gade" where the
# corrected transcript reads "Godfrey Gidi" -- direct user feedback: "the English
# audio has correct pronunciation of 'Gardee', can't learn phonetics from
# [spelling] there." Pass 1's own corrections record already carries this for
# free; it just wasn't being read.


def _pass1_correction(original: str, corrected: str, *, reason_code: str = "name_or_entity") -> dict:
    return {
        "segment_id": "seg-00001", "reason_code": reason_code,
        "original_text": original, "corrected_text": corrected,
    }


def test_find_raw_asr_hint_locates_the_real_pre_correction_wording():
    from mathula_tv.native_dub import _find_raw_asr_hint_for_entity

    corrections = [_pass1_correction(
        "the party's mayoral candidate, Godrej Gade, to address.",
        "the party's mayoral candidate, Godfrey Gidi, to address.",
    )]
    assert _find_raw_asr_hint_for_entity("Godfrey Gidi", corrections) == (
        "the party's mayoral candidate, Godrej Gade, to address."
    )


def test_find_raw_asr_hint_ignores_a_non_name_correction():
    from mathula_tv.native_dub import _find_raw_asr_hint_for_entity

    corrections = [_pass1_correction(
        "Godfrey Gidi was their too.", "Godfrey Gidi was there too.",
        reason_code="grammar",
    )]
    assert _find_raw_asr_hint_for_entity("Godfrey Gidi", corrections) is None


def test_find_raw_asr_hint_returns_none_when_no_correction_mentions_the_entity():
    from mathula_tv.native_dub import _find_raw_asr_hint_for_entity

    corrections = [_pass1_correction("Cyril Ramaphosa spoke.", "Cyril Ramaphosa spoke.")]
    assert _find_raw_asr_hint_for_entity("Godfrey Gidi", corrections) is None


def test_self_supervised_fallback_passes_the_real_raw_asr_hint_to_the_request(
    tmp_path, monkeypatch, _fake_normalize_for_self_supervised,
):
    # Fictional entity ("Thabo Ndlovu") for the same reason as the tests
    # above -- "Godfrey Gidi" is now a hardcoded dictionary entry and would
    # be skipped before reaching this code path.
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    grok = _FakeGrokProvider({"Thabo Ndlovu": ["Thabo Ndloovu"]})
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoThabo Ndlovu kulesi sigaba.": "kukhulunywa ngo tabo ndlobu kulesi sigaba",
        "Kukhulunywa ngoThabo Ndloovu kulesi sigaba.": "kukhulunywa ngo thabo ndlovu kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)
    pass1_corrections = [_pass1_correction(
        "the mayoral candidate, Tabo Ndlobu, to address.",
        "the mayoral candidate, Thabo Ndlovu, to address.",
    )]

    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, tts=tts, stt_backend=stt, grok_provider=grok,
        pass1_corrections=pass1_corrections,
    )

    assert grok.calls == 1
    sent = grok.last_payload["candidates"][0]
    assert sent["name"] == "Thabo Ndlovu"
    assert "Tabo Ndlobu" in sent["raw_asr_hint"]


def test_self_supervised_fallback_omits_raw_asr_hint_when_none_found(
    tmp_path, monkeypatch, _fake_normalize_for_self_supervised,
):
    job_root = _job_root(tmp_path)
    backend = _FakeResearchBackend(unresolved={_native_pronunciation_candidate_id("Thabo Ndlovu")})
    monkeypatch.setattr(
        "mathula_tv.native_dub.AzureResponsesWebResearchProvider.from_environment", lambda: backend,
    )
    grok = _FakeGrokProvider({"Thabo Ndlovu": ["Thabo Ndloovu"]})
    tts = _FakeRoundTripTts()
    stt = _FakeRoundTripStt({
        "Kukhulunywa ngoThabo Ndlovu kulesi sigaba.": "kukhulunywa ngo tabo ndlobu kulesi sigaba",
        "Kukhulunywa ngoThabo Ndloovu kulesi sigaba.": "kukhulunywa ngo thabo ndlovu kulesi sigaba",
    })
    _wire_tts_stt_pair(tts, stt)

    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=_context_ledger("Thabo Ndlovu"), glossary=_glossary(),
        progress=None, tts=tts, stt_backend=stt, grok_provider=grok,
        pass1_corrections=(),  # no corrections recorded for this job
    )

    sent = grok.last_payload["candidates"][0]
    assert "raw_asr_hint" not in sent
