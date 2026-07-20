from __future__ import annotations

import re

PATTERNS = {
    "commission": r"\b(madlanga|commission|testif(?:y|ied)|witness|hearing)\b",
    "local_elections": r"\b(municipal election|local election|ward candidate|ward \d+|councillor|ballot)\b",
    "politics": r"\b(campaign|coalition|parliament|political party|anc|da|eff|mk party|cabinet|minister)\b",
    "general_news": r"\b(weather|sport|traffic|accident|fire|health|school|business|community)\b",
}


def classify(text: str) -> dict:
    lowered = text.lower()
    matches = {domain: [m.group(0) for m in re.finditer(pattern, lowered, re.I)] for domain, pattern in PATTERNS.items()}
    active = [d for d in ("commission", "local_elections", "politics", "general_news") if matches[d]]
    # A title alone does not make the story political; require an issue/action signal.
    if "politics" in active and not re.search(r"\b(campaign|coalition|parliament|policy|election|vote|cabinet|government|party)\b", lowered):
        active.remove("politics")
    primary = "general_news" if not active else ("mixed" if len(active) > 1 else active[0])
    secondary = active if primary == "mixed" else [d for d in active if d != primary]
    evidence = [f"Transcript contains '{term}'" for domain in active for term in matches[domain][:2]]
    return {"schema_version": "domain-classification-v1", "primary_domain": primary, "secondary_domains": secondary, "confidence": min(0.98, 0.55 + 0.08 * len(evidence)), "evidence": evidence or ["No specialist-domain evidence; defaulted to general news"], "entities": {"people": [], "political_parties": _parties(text), "municipalities": [], "provinces": [], "wards": re.findall(r"\bward\s+\d+\b", text, re.I), "government_institutions": [], "commission_entities": []}, "election_relevance": "local_elections" in active, "warnings": []}


def _parties(text: str) -> list[str]:
    return [name for name in ("ANC", "DA", "EFF", "MK Party", "IEC") if re.search(rf"\b{re.escape(name)}\b", text, re.I)]

