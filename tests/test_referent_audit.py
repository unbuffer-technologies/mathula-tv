from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.entity_registry import EntityRegistry
from mathula_tv.foundry_grok import GrokOutputTruncated
from mathula_tv.referent_audit import (
    _estimate_output_tokens,
    _group_records,
    audit_and_repair_pass2,
    deterministic_entity_findings,
)


def _write_registry(path: Path, entities: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": "mathula-entity-registry-v1", "entities": entities}, indent=2),
        encoding="utf-8",
    )


def _fadiel_entity() -> dict:
    return {
        "entity_id": "fadiel_adams",
        "entity_type": "person",
        "canonical_text": "Fadiel Adams",
        "display_text": "Fadiel Adams",
        "aliases": ["Adams"],
        "domains": ["madlanga_commission"],
        "roles": [],
        "source_status": "reviewed",
        "active": True,
        "must_preserve": True,
        "translation_policy": "protected_token",
        "stt_phrases": [],
        "spoken_forms": {},
        "do_not_confuse_with": [],
        "sources": [],
    }


def _commission_entity() -> dict:
    return {
        "entity_id": "organisation_madlanga_commission_of_inquiry",
        "entity_type": "organisation",
        "canonical_text": "Madlanga Commission of Inquiry",
        "display_text": "Madlanga Commission of Inquiry",
        "aliases": ["Madlanga Commission", "the Commission", "Judicial Commission of Inquiry"],
        "domains": ["madlanga_commission"],
        "roles": [],
        "source_status": "reviewed",
        "active": True,
        "must_preserve": True,
        "translation_policy": "protected_token",
        "stt_phrases": [],
        "spoken_forms": {},
        "do_not_confuse_with": [],
        "sources": [],
    }


class _FakeProvider:
    def __init__(self, responses: list[tuple[str, dict]]):
        self._responses = list(responses)
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.calls: list[dict] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append({"operation": operation, "payload": payload})
        assert self._responses, f"no scripted response left for operation={operation}"
        expected_operation, data = self._responses.pop(0)
        assert operation == expected_operation, f"expected {expected_operation}, got {operation}"
        return SimpleNamespace(data=data, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)


def test_deterministic_entity_findings_flags_dropped_entity(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_fadiel_entity()])
    registry = EntityRegistry(registry_path=registry_path)

    findings = deterministic_entity_findings(
        "Fadiel Adams opened the case in Soweto.",
        "Kwavulwa icala eSoweto.",
        registry,
    )
    assert len(findings) == 1
    assert findings[0]["problem"] == "missing"
    assert findings[0]["entity_id"] == "fadiel_adams"


def test_deterministic_entity_findings_accepts_present_entity(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_fadiel_entity()])
    registry = EntityRegistry(registry_path=registry_path)

    findings = deterministic_entity_findings(
        "Fadiel Adams opened the case in Soweto.",
        "UFadiel Adams wavula icala eSoweto.",
        registry,
    )
    assert findings == []


def test_deterministic_check_does_not_flag_translated_organisation_names(tmp_path):
    """Regression test for a real false positive found in production.

    "the Commission" is a registered alias, but organisations are properly
    translated into a natural Zulu equivalent ("iKhomishini"), unlike this
    corpus's people, who keep their literal English name. A live run flagged
    5 groups as "missing the Commission" purely because none of them contained
    the literal English alias text -- none were real translation problems.
    """

    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_commission_entity()])
    registry = EntityRegistry(registry_path=registry_path)

    findings = deterministic_entity_findings(
        "What the Commission has heard thus far.",
        "Lokhu iKhomishini esikuzwile kuze kube manje.",
        registry,
    )
    assert findings == []


def test_group_records_from_temporal_mask_groups():
    pass2 = {
        "temporal_mask_groups": [
            {"group_id": "native_phrase_0006", "segment_ids": ["seg-00008"], "source_text": "EN", "spoken_text": "ZU"},
        ],
        "translations": [{"segment_id": "seg-00008", "spoken_text": "ZU"}],
    }
    records = _group_records(pass2)
    assert records == [
        {"group_id": "native_phrase_0006", "segment_ids": ["seg-00008"], "english": "EN", "zulu": "ZU"}
    ]


def test_group_records_from_flat_translations_when_no_temporal_groups():
    pass2 = {
        "translations": [
            {"segment_id": "seg-1", "source_text": "Hello.", "spoken_text": "Sawubona."},
        ]
    }
    records = _group_records(pass2)
    assert records == [
        {"group_id": "seg-1", "segment_ids": ["seg-1"], "english": "Hello.", "zulu": "Sawubona."}
    ]


def test_audit_and_repair_fixes_minister_mistranslated_as_god():
    pass2 = {
        "translations": [
            {"segment_id": "seg-00008", "spoken_text": "UJenerali Lincoln ... osondele kunkulunkulu ..."},
        ],
        "temporal_mask_groups": [
            {
                "group_id": "native_phrase_0006",
                "segment_ids": ["seg-00008"],
                "source_text": "... someone that was close to the minister ...",
                "spoken_text": "UJenerali Lincoln ... osondele kunkulunkulu ...",
            }
        ],
    }
    provider = _FakeProvider(
        [
            (
                "native_referent_audit_batch",
                {
                    "findings": [
                        {
                            "group_id": "native_phrase_0006",
                            "status": "issue",
                            "checked_terms": [
                                {
                                    "term": "the minister",
                                    "category": "role_or_title",
                                    "verification": "substituted",
                                    "found_instead": "God",
                                }
                            ],
                        }
                    ]
                },
            ),
            (
                "native_referent_repair_batch",
                {
                    "repairs": [
                        {
                            "group_id": "native_phrase_0006",
                            "corrected_zulu": "UJenerali Lincoln ... osondele kungqongqoshe ...",
                        }
                    ]
                },
            ),
        ]
    )

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None)

    assert summary["groups_checked"] == 1
    assert len(summary["ai_findings"]) == 1
    assert len(summary["repairs"]) == 1
    assert summary["unresolved_group_ids"] == []
    assert pass2["temporal_mask_groups"][0]["spoken_text"] == "UJenerali Lincoln ... osondele kungqongqoshe ..."
    assert pass2["translations"][0]["spoken_text"] == "UJenerali Lincoln ... osondele kungqongqoshe ..."
    assert len(provider.calls) == 2


def test_audit_skips_repair_call_when_everything_is_ok():
    pass2 = {
        "translations": [{"segment_id": "seg-1", "source_text": "Yes.", "spoken_text": "Yebo."}],
    }
    provider = _FakeProvider(
        [
            (
                "native_referent_audit_batch",
                {"findings": [{"group_id": "seg-1", "status": "ok"}]},
            )
        ]
    )

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None)

    assert summary["repairs"] == []
    assert summary["ai_findings"] == []
    assert len(provider.calls) == 1  # only the audit call, no repair call
    assert pass2["translations"][0]["spoken_text"] == "Yebo."


def test_audit_leaves_group_unresolved_when_max_repair_rounds_is_zero():
    pass2 = {
        "translations": [{"segment_id": "seg-1", "source_text": "No.", "spoken_text": "Yebo."}],
    }
    provider = _FakeProvider(
        [
            (
                "native_referent_audit_batch",
                {
                    "findings": [
                        {
                            "group_id": "seg-1",
                            "status": "issue",
                            "checked_terms": [
                                {
                                    "term": "No",
                                    "category": "negation",
                                    "verification": "substituted",
                                    "found_instead": "Yebo",
                                }
                            ],
                        }
                    ]
                },
            )
        ]
    )

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None, max_repair_rounds=0)

    assert summary["repairs"] == []
    assert summary["unresolved_group_ids"] == ["seg-1"]
    assert pass2["translations"][0]["spoken_text"] == "Yebo."  # left untouched, not silently accepted as fine
    assert len(provider.calls) == 1


def test_deterministic_finding_merged_into_repair_payload_even_when_ai_says_ok(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_fadiel_entity()])
    registry = EntityRegistry(registry_path=registry_path)

    pass2 = {
        "translations": [
            {
                "segment_id": "seg-1",
                "source_text": "Fadiel Adams opened the case.",
                "spoken_text": "Kwavulwa icala.",
            }
        ],
    }
    provider = _FakeProvider(
        [
            (
                "native_referent_audit_batch",
                {"findings": [{"group_id": "seg-1", "status": "ok"}]},
            ),
            (
                "native_referent_repair_batch",
                {
                    "repairs": [
                        {"group_id": "seg-1", "corrected_zulu": "UFadiel Adams wavula icala."},
                    ]
                },
            ),
        ]
    )

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=registry)

    assert len(summary["deterministic_findings"]) == 1
    repair_payload = provider.calls[1]["payload"]["groups"][0]
    assert any(issue["source_term"] == "Fadiel Adams" for issue in repair_payload["issues"])
    assert pass2["translations"][0]["spoken_text"] == "UFadiel Adams wavula icala."


def test_estimate_output_tokens_scales_with_content_length():
    small = _estimate_output_tokens(["Yebo."], per_item_overhead=80)
    large = _estimate_output_tokens(["Yebo. " * 400], per_item_overhead=80)
    assert large > small * 10


class _TruncatingRepairProvider:
    """Raises GrokOutputTruncated for any repair batch larger than max_batch_size.

    Models a real Grok deployment whose flat per-item token budget undershoots
    once a batch contains one or more unusually long groups: the fix is to
    retry with a smaller batch, not to fail the whole audit.
    """

    def __init__(self, max_batch_size: int, *, audit_findings: dict):
        self.max_batch_size = max_batch_size
        self._audit_findings = audit_findings
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.repair_call_sizes: list[int] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        groups = payload["groups"]
        if operation == "native_referent_audit_batch":
            findings = [
                {"group_id": item["group_id"], **self._audit_findings[item["group_id"]]}
                for item in groups
            ]
            return SimpleNamespace(data={"findings": findings}, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)

        assert operation == "native_referent_repair_batch"
        self.repair_call_sizes.append(len(groups))
        if len(groups) > self.max_batch_size:
            raise GrokOutputTruncated(
                "hit max_tokens", model="grok-test", input_tokens=10, output_tokens=1500, attempts=1
            )
        repairs = [{"group_id": item["group_id"], "corrected_zulu": f"FIXED:{item['group_id']}"} for item in groups]
        return SimpleNamespace(data={"repairs": repairs}, model="grok-test", input_tokens=10, output_tokens=50, attempts=1)


def test_repair_batch_retries_as_smaller_batches_on_truncation():
    group_ids = [f"g{i}" for i in range(5)]
    pass2 = {
        "translations": [{"segment_id": gid, "source_text": f"EN {gid}", "spoken_text": f"ZU {gid}"} for gid in group_ids],
    }
    audit_findings = {
        gid: {
            "status": "issue",
            "checked_terms": [
                {"term": "term", "category": "name", "verification": "substituted", "found_instead": "wrong"}
            ],
        }
        for gid in group_ids
    }
    provider = _TruncatingRepairProvider(max_batch_size=1, audit_findings=audit_findings)

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None, batch_size=12)

    assert summary["unresolved_group_ids"] == []
    assert len(summary["repairs"]) == 5
    for gid in group_ids:
        translation = next(item for item in pass2["translations"] if item["segment_id"] == gid)
        assert translation["spoken_text"] == f"FIXED:{gid}"
    # First attempt is the full batch (and truncates, per max_batch_size=1); the rest of the
    # calls are the recursive split down to single items that then succeed.
    assert provider.repair_call_sizes[0] == 5
    assert len(provider.repair_call_sizes) > 1
    assert provider.repair_call_sizes.count(1) == 5


def test_repair_leaves_single_group_unresolved_when_it_alone_still_truncates():
    pass2 = {
        "translations": [{"segment_id": "g0", "source_text": "EN", "spoken_text": "ZU"}],
    }
    audit_findings = {
        "g0": {
            "status": "issue",
            "checked_terms": [{"term": "term", "category": "name", "verification": "missing"}],
        },
    }
    # max_batch_size=0 means even a single-item batch is "too big" and truncates.
    provider = _TruncatingRepairProvider(max_batch_size=0, audit_findings=audit_findings)

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None, batch_size=12)

    assert summary["unresolved_group_ids"] == ["g0"]
    assert summary["repairs"] == []
    # Left untouched, not silently accepted or crashed.
    assert pass2["translations"][0]["spoken_text"] == "ZU"


def test_checked_terms_from_extract_then_verify_survive_into_all_ai_verdicts():
    """The per-term extract-then-verify checklist must reach the persisted record.

    Without this, there is no way to see *what the model actually checked* for a
    group it did not flag -- exactly the visibility gap that made the minister/God
    miss hard to diagnose from the first live run.
    """

    pass2 = {
        "translations": [{"segment_id": "seg-1", "source_text": "Speak to the minister.", "spoken_text": "Khuluma nomkhulunkulu."}],
    }
    provider = _FakeProvider(
        [
            (
                "native_referent_audit_batch",
                {
                    "findings": [
                        {
                            "group_id": "seg-1",
                            "status": "ok",
                            "checked_terms": [
                                {
                                    "term": "the minister",
                                    "category": "role_or_title",
                                    "verification": "present",
                                }
                            ],
                        }
                    ]
                },
            )
        ]
    )

    summary = audit_and_repair_pass2(provider=provider, pass2=pass2, registry=None)

    verdict = next(v for v in summary["all_ai_verdicts"] if v["group_id"] == "seg-1")
    assert verdict["checked_terms"] == [
        {"term": "the minister", "category": "role_or_title", "verification": "present"}
    ]
