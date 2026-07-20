from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def _base(name: str, path: Path | None = None) -> dict:
    return {"provider_name": name, "source_identifier": str(path) if path else None, "source_checksum": None, "source_modification_time": None, "matched_entities": [], "relevant_facts": [], "confidence": 0.0, "warnings": [], "enrichment_applied": False}


class ContextProvider(ABC):
    @abstractmethod
    def enrich(self, classification: dict, transcript: str) -> dict: ...


class JsonContextProvider(ContextProvider):
    name = "json"
    relevant_domains: set[str] = set()

    def __init__(self, path: Path): self.path = path

    def enrich(self, classification: dict, transcript: str) -> dict:
        result = _base(self.name, self.path)
        domains = {classification["primary_domain"], *classification.get("secondary_domains", [])}
        if not domains.intersection(self.relevant_domains) and classification["primary_domain"] != "mixed":
            result["warnings"].append("Provider excluded because clip is unrelated")
            return result
        try:
            raw = self.path.read_bytes()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            result["warnings"].append(f"Context unavailable: {type(exc).__name__}")
            return result
        result["source_checksum"] = hashlib.sha256(raw).hexdigest()
        result["source_modification_time"] = self.path.stat().st_mtime
        terms = {word.lower() for word in transcript.split() if len(word) > 4}
        facts = []
        for item in _walk_records(data):
            rendered = json.dumps(item, ensure_ascii=False)
            if any(term in rendered.lower() for term in terms):
                facts.append(item)
            if len(facts) >= 12:
                break
        result.update(relevant_facts=facts, confidence=0.75 if facts else 0.0, enrichment_applied=bool(facts))
        return result


def _walk_records(value: Any):
    if isinstance(value, dict):
        if any(k in value for k in ("fact", "name", "title", "summary", "allegation")):
            yield value
        for child in value.values(): yield from _walk_records(child)
    elif isinstance(value, list):
        for child in value: yield from _walk_records(child)


class CommissionContextProvider(JsonContextProvider):
    name = "commission-master-case"
    relevant_domains = {"commission"}


class PoliticsContextProvider(JsonContextProvider):
    name = "politics-context"
    relevant_domains = {"politics", "local_elections"}


class NoContextProvider(ContextProvider):
    def enrich(self, classification: dict, transcript: str) -> dict:
        return _base("none")


def providers_for(classification: dict, commission_path: Path, politics_path: Path) -> list[ContextProvider]:
    domains = {classification["primary_domain"], *classification.get("secondary_domains", [])}
    providers: list[ContextProvider] = []
    if "commission" in domains or classification["primary_domain"] == "mixed": providers.append(CommissionContextProvider(commission_path))
    if domains.intersection({"politics", "local_elections"}) or classification["primary_domain"] == "mixed": providers.append(PoliticsContextProvider(politics_path))
    return providers or [NoContextProvider()]

