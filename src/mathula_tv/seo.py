from __future__ import annotations

import json
from pathlib import Path
import jsonschema

from .translation import JsonClient

SEO_PROMPT = """Create a truthful natural isiZulu YouTube metadata package from the supplied grounded material. Choose the strongest supported angle and preserve genuine sensational energy with explicit attribution. Never fabricate quotations, claims, entities, Commission relevance or election relevance; do not force Madlanga Commission into the title. Return strict JSON with title, description, tags_csv, tags, thumbnail_hook, category, primary_domain, secondary_domains, entities, main_entities, political_parties, municipalities, provinces, election_relevance, claim_attributions, context_providers_used, search_intent_summary, factual_grounding_notes, uncertainty_warnings and human_review_flags. entities and main_entities must both be arrays and tags_csv must exactly join tags with comma-space separators."""
REQUIRED = {"title", "description", "tags_csv", "tags", "thumbnail_hook", "category", "primary_domain", "secondary_domains", "entities", "main_entities", "claim_attributions", "context_providers_used", "factual_grounding_notes", "uncertainty_warnings", "human_review_flags"}


def validate_seo(data: dict) -> dict:
    missing = REQUIRED - data.keys()
    if missing: raise ValueError(f"SEO fields missing: {sorted(missing)}")
    if not 1 <= len(data["title"]) <= 100: raise ValueError("YouTube title must be 1-100 characters")
    if not isinstance(data["tags"], list) or sum(len(str(t)) + 1 for t in data["tags"]) > 500: raise ValueError("Invalid YouTube tags")
    if data["tags_csv"] != ", ".join(data["tags"]): raise ValueError("tags_csv must match the tags array")
    if not isinstance(data["claim_attributions"], list): raise ValueError("claim_attributions must be a list")
    schema=json.loads((Path(__file__).resolve().parents[2]/"schemas/seo_package.schema.json").read_text())
    jsonschema.validate(data,schema)
    return data


def generate_seo(client: JsonClient, transcript: dict, translated: dict, classification: dict, context: list[dict], repair_retries: int = 2) -> dict:
    payload = {"english_transcript": transcript, "isizulu_transcript": translated, "classification": classification, "context": context}
    error = None
    for _ in range(repair_retries + 1):
        result = client.complete_json(SEO_PROMPT + (f" Repair this validation error: {error}" if error else ""), payload)
        try: return validate_seo(result)
        except (ValueError, jsonschema.ValidationError) as exc: error = str(exc); payload["invalid_response"] = result
    raise ValueError(f"SEO validation failed after repair: {error}")
